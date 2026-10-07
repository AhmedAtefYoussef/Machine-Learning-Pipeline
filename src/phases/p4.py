"""Phase 4: regularization. Ridge, lasso and elastic net (our own solvers, src/regularization.py) on the Phase 3
target design plus the six candidate columns we had kept out, followed by a verdict for every original column.

Conventions
  * The solvers take X WITHOUT the bias column (they fit the intercept themselves); `n_features` in the artifact
    counts the bias column, like p1..p3, while `feature_names` and `weights` do not contain it.
  * Objective (sklearn scaling): (1/2n)||z - b - Xw||^2 + alpha*rho*||w||_1 + (alpha/2)(1-rho)||w||^2, z = log1p(cnt).
  * All R2 / RMSE are on the bike scale, with the back-transform method of Phase 1 recomputed from the fitting rows.
  * Each lasso / elastic-net solution that exhausts MAX_SWEEPS is counted in `not_converged`, never hidden.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import verdicts as vd
from src.common import (back_factor, bootstrap_r2, chrono_split, config_seed, day_block_folds, from_target,
                        load_config, load_train, paired_bootstrap_delta_r2, r2, read_artifact, rmse, seeded_split,
                        set_threads, to_target, write_artifact)
from src.features import Design, DesignSpec, sources
from src.phases.p3 import ladder_specs
from src.regularization import (alpha_grid, alpha_max, enet_cd, enet_path, group_bootstrap_rows,
                                ridge_closed_form, ridge_path)
from src.validation import ETA_HEADROOM

SOLVER_TOL = 1.0e-7        # coordinate descent stops when no coefficient moves more than this (fixed by S-4-01)
MAX_SWEEPS = 20000
NONZERO_TOL = 1.0e-10      # |w| above this counts as non-zero (same rule as stability selection)
UNREGULARISED_ALPHA = 1.0e-8
PATH_NAMES = ("temp", "atemp", "hum", "windspeed", "workingday", "holiday", "trend", "yr", "instant", "ws_3")
METHODS = ("l2", "l1", "enet")


# ----------------------------------------------------------------------------- data containers

@dataclass
class Split:
    """Design matrices WITHOUT the bias column, fitted on the fitting rows only, plus the targets."""
    names: list[str]
    X_fit: np.ndarray
    z_fit: np.ndarray
    cnt_fit: np.ndarray
    X_eval: np.ndarray
    cnt_eval: np.ndarray

    def keep(self, columns: list[int]) -> "Split":
        """Same rows, only these columns (no re-standardising)."""
        return Split([self.names[j] for j in columns], self.X_fit[:, columns], self.z_fit, self.cnt_fit,
                     self.X_eval[:, columns], self.cnt_eval)


@dataclass
class Curve:
    """One regularisation path: alphas (descending), intercepts, coefficients and the per-alpha scores."""
    rho: float
    alphas: np.ndarray
    B: np.ndarray
    W: np.ndarray
    rows: list[dict]


@dataclass
class Fit:
    """One model at its chosen hyper-parameters."""
    method: str
    alpha: float
    rho: float
    index: int            # position of alpha on the grid of its curve
    b: float
    w: np.ndarray


@dataclass
class SolverLog:
    """Counts solutions that used up all their sweeps, and where."""
    not_converged: int = 0
    where: dict = field(default_factory=dict)
    slowest_sweeps: int = 0

    def record(self, where: str, sweeps) -> None:
        sweeps = np.atleast_1d(np.asarray(sweeps))
        hit = int(np.sum(sweeps >= MAX_SWEEPS))
        self.slowest_sweeps = max(self.slowest_sweeps, int(sweeps.max()))
        if hit:
            self.not_converged += hit
            self.where[where] = self.where.get(where, 0) + hit


@dataclass
class Context:
    """What every step shares."""
    cfg: dict
    p4cfg: dict
    seed: int
    B: int
    back_method: str
    all_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    spec: DesignSpec
    split: Split
    log: SolverLog
    design_from: dict


TIMINGS: dict[str, float] = {}


@contextmanager
def timed(step: str):
    """Record and print the wall-clock time of one step."""
    start = time.perf_counter()
    yield
    TIMINGS[step] = time.perf_counter() - start
    print(f"[p4] {step}: {TIMINGS[step]:.1f} s")


# ----------------------------------------------------------------------------- designs

def add_candidates(spec: DesignSpec, candidate_blocks: list[str], train_df: pd.DataFrame):
    """Append each candidate block whose columns are not already in the design; returns (spec, added, skipped)."""
    added, skipped = [], []
    for block in candidate_blocks:
        present = set(Design(spec).names)
        bigger = DesignSpec(spec.base, spec.power_cols, spec.degree, spec.blocks + (block,))
        if set(Design(bigger).names) <= present:
            skipped.append(block)
        else:
            spec, added = bigger, added + [block]
    return spec, added, skipped


def pick_design(cfg: dict, p2: dict, p3: dict, train_df: pd.DataFrame) -> tuple[DesignSpec, dict]:
    """Base spec (Phase 3 target or a named ladder level) plus candidates, and the `design_from` record."""
    level = cfg["p4"].get("design_level", "target")
    if level == "target":
        base_spec = DesignSpec.from_dict(p3["target_complexity"]["design_spec"])
        level, index = p3["target_complexity"]["level"], p3["target_complexity"]["index"]
    else:
        anchor_spec = DesignSpec.from_dict(p2["design_spec"])
        chosen = next(lv for lv in ladder_specs(cfg, anchor_spec) if lv["name"] == level)
        base_spec, index = chosen["spec"], chosen["index"]
    spec, added, skipped = add_candidates(base_spec, cfg["p4"]["candidate_blocks"], train_df)
    design_from = {"level": level, "index": index, "n_features_before_candidates": len(Design(base_spec).names),
                   "candidates_added": added, "candidates_skipped": skipped}
    return spec, design_from


def build_split(spec: DesignSpec, fit_df: pd.DataFrame, eval_df: pd.DataFrame) -> Split:
    """Fit the design on fit_df only, transform both frames, drop the bias column."""
    design = Design(spec).fit(fit_df)
    X_fit, X_eval = design.transform(fit_df)[:, 1:], design.transform(eval_df)[:, 1:]
    cnt_fit = fit_df["cnt"].to_numpy(dtype=np.float64)
    return Split(design.names[1:], X_fit, to_target(cnt_fit), cnt_fit, X_eval,
                 eval_df["cnt"].to_numpy(dtype=np.float64))


def fold_splits(ctx: Context, spec: DesignSpec | None = None) -> list[Split]:
    """Day-block folds inside train_df; each fold gets its own design fitted on that fold's fitting rows."""
    spec = spec or ctx.spec
    folds = day_block_folds(ctx.train_df, ctx.cfg["day_block_folds"])
    return [build_split(spec, ctx.train_df[folds != f], ctx.train_df[folds == f])
            for f in range(ctx.cfg["day_block_folds"])]


# ----------------------------------------------------------------------------- scoring

def predict_bikes(split: Split, b: float, w: np.ndarray, back_method: str) -> tuple[np.ndarray, np.ndarray]:
    """Bike-scale predictions (fitting rows, evaluation rows); eta on evaluation rows capped like validation.fit_predict."""
    eta_fit = b + split.X_fit @ w
    s = back_factor(back_method, split.z_fit, eta_fit, split.cnt_fit)
    eta_eval = np.minimum(b + split.X_eval @ w, split.z_fit.max() + ETA_HEADROOM)
    return from_target(eta_fit, s), from_target(eta_eval, s)


def score(split: Split, b: float, w: np.ndarray, back_method: str) -> tuple[dict, np.ndarray]:
    """(train/eval R2 and RMSE, evaluation predictions) of the model (b, w)."""
    pred_fit, pred_eval = predict_bikes(split, b, w, back_method)
    scores = {"train_r2": r2(split.cnt_fit, pred_fit), "train_rmse": rmse(split.cnt_fit, pred_fit),
              "r2": r2(split.cnt_eval, pred_eval), "rmse": rmse(split.cnt_eval, pred_eval)}
    return scores, pred_eval


def count_nonzero(w: np.ndarray) -> int:
    return int(np.sum(np.abs(w) > NONZERO_TOL))


def best_index(alphas: np.ndarray, scores: np.ndarray) -> int:
    """Position of the highest score; among exact ties the larger alpha."""
    tied = np.flatnonzero(scores == scores.max())
    return int(tied[np.argmax(alphas[tied])])


# ----------------------------------------------------------------------------- grids and paths

def make_grid(split: Split, rho: float, p4cfg: dict) -> np.ndarray:
    """Descending alphas: ridge over its fixed range, lasso / elastic net from alpha_max down by the min ratio."""
    n = p4cfg["n_alphas"]
    if rho == 0.0:
        lo, hi = p4cfg["ridge_alpha_range"]
        return np.logspace(np.log10(hi), np.log10(lo), n)
    return alpha_grid(alpha_max(split.X_fit, split.z_fit, rho), p4cfg["l1_alpha_min_ratio"], n)


def solve_path(split: Split, alphas: np.ndarray, rho: float, log: SolverLog, where: str):
    """Intercepts and coefficients for descending alphas (ridge: eigendecomposition; otherwise warm-started CD)."""
    if rho == 0.0:
        return ridge_path(split.X_fit, split.z_fit, alphas)
    B, W, sweeps = enet_path(split.X_fit, split.z_fit, alphas, rho, SOLVER_TOL, MAX_SWEEPS)
    log.record(where, sweeps)
    return B, W


def fit_at(split: Split, alpha: float, rho: float, grid: np.ndarray, log: SolverLog, where: str):
    """One model at `alpha`; CD is warm-started down the part of `grid` that is >= alpha."""
    if rho == 0.0:
        return ridge_closed_form(split.X_fit, split.z_fit, alpha)
    B, W = solve_path(split, grid[grid >= alpha], rho, log, where)
    return float(B[-1]), W[-1]


def build_curve(split: Split, rho: float, ctx: Context, where: str) -> Curve:
    """Whole path on the fitting rows, scored on the evaluation rows of `split`."""
    alphas = make_grid(split, rho, ctx.p4cfg)
    B, W = solve_path(split, alphas, rho, ctx.log, where)
    rows = []
    for a, b, w in zip(alphas, B, W):
        s, _ = score(split, b, w, ctx.back_method)
        rows.append({"alpha": float(a), "val_r2": s["r2"], "val_rmse": s["rmse"], "train_r2": s["train_r2"],
                     "n_nonzero": count_nonzero(w)})
    return Curve(rho, alphas, B, W, rows)


def curve_scores(curve: Curve) -> np.ndarray:
    return np.array([r["val_r2"] for r in curve.rows])


def validation_curves(ctx: Context) -> dict:
    """{'l2': Curve, 'l1': Curve, 'enet': {rho: Curve}}."""
    curves = {"l2": build_curve(ctx.split, 0.0, ctx, "curve l2"), "l1": build_curve(ctx.split, 1.0, ctx, "curve l1")}
    curves["enet"] = {rho: build_curve(ctx.split, rho, ctx, f"curve enet {rho}") for rho in ctx.p4cfg["l1_ratios"]}
    return curves


def choose_fits(curves: dict) -> dict[str, Fit]:
    """Per method the alpha (and for elastic net the rho) with the best validation R2."""
    def at_best(method: str, curve: Curve) -> Fit:
        i = best_index(curve.alphas, curve_scores(curve))
        return Fit(method, float(curve.alphas[i]), curve.rho, i, float(curve.B[i]), curve.W[i].copy())

    fits = {"l2": at_best("l2", curves["l2"]), "l1": at_best("l1", curves["l1"])}
    enet_fits = [at_best("enet", c) for c in curves["enet"].values()]
    best = max(enet_fits, key=lambda f: (curves["enet"][f.rho].rows[f.index]["val_r2"]))
    fits["enet"] = best
    return fits


# ----------------------------------------------------------------------------- day-block cross-validation

def cv_fold_r2(folds: list[Split], alphas: np.ndarray, rho: float, ctx: Context, where: str) -> np.ndarray:
    """(k, len(alphas)) held-out-day R2 on the bike scale; the design of each fold is its own."""
    out = np.zeros((len(folds), len(alphas)))
    for f, split in enumerate(folds):
        B, W = solve_path(split, alphas, rho, ctx.log, f"{where} fold {f}")
        out[f] = [score(split, b, w, ctx.back_method)[0]["r2"] for b, w in zip(B, W)]
    return out


def cv_summary(alphas: np.ndarray, fold_r2: np.ndarray, val_rows: list[dict]) -> dict:
    """Mean and standard error per alpha, best alpha and the one-standard-error alpha, with validation R2 at both."""
    k = fold_r2.shape[0]
    mean = fold_r2.mean(axis=0)
    se = fold_r2.std(axis=0, ddof=1) / np.sqrt(k)
    best = best_index(alphas, mean)
    qualifies = mean >= mean[best] - se[best]
    one_se = int(np.flatnonzero(alphas == alphas[qualifies].max())[0])
    return {"alphas": [float(a) for a in alphas], "mean": mean.tolist(), "se": se.tolist(),
            "cv_best_alpha": float(alphas[best]), "cv_1se_alpha": float(alphas[one_se]),
            "val_r2_at_cv_best": val_rows[best]["val_r2"], "val_r2_at_cv_1se": val_rows[one_se]["val_r2"]}


def cross_validation(curves: dict, fits: dict[str, Fit], ctx: Context) -> dict:
    """cv.{l2,l1,enet}; elastic net only at its chosen rho. Grids are the absolute alphas of the full-train curves."""
    folds = fold_splits(ctx)
    chosen = {"l2": curves["l2"], "l1": curves["l1"], "enet": curves["enet"][fits["enet"].rho]}
    return {m: cv_summary(c.alphas, cv_fold_r2(folds, c.alphas, c.rho, ctx, f"cv {m}"), c.rows)
            for m, c in chosen.items()}


# ----------------------------------------------------------------------------- methods

def chrono_result(fit: Fit, grid: np.ndarray, chrono: Split, ctx: Context) -> dict:
    """Design and model fitted on the early part, scored on the late part, same alpha and rho."""
    b, w = fit_at(chrono, fit.alpha, fit.rho, grid, ctx.log, f"chrono {fit.method}")
    s, _ = score(chrono, b, w, ctx.back_method)
    return {"chrono_r2": s["r2"], "chrono_rmse": s["rmse"]}


def describe_methods(curves: dict, fits: dict[str, Fit], cv: dict, ctx: Context):
    """methods.{l2,l1,enet} and the validation predictions of each (kept for the paired comparisons)."""
    early_df, late_df = chrono_split(ctx.all_df, ctx.cfg["chrono"]["cut_date"])
    chrono = build_split(ctx.spec, early_df, late_df)
    y_val = ctx.split.cnt_eval
    methods, preds = {}, {}
    for m, fit in fits.items():
        curve = curves["enet"][fit.rho] if m == "enet" else curves[m]
        s, pred = score(ctx.split, fit.b, fit.w, ctx.back_method)
        lo, hi, _ = bootstrap_r2(y_val, pred, ctx.seed, ctx.B)
        methods[m] = {"lambda": fit.alpha, "l1_ratio": fit.rho, "val_r2": s["r2"], "val_rmse": s["rmse"],
                      "val_lo": lo, "val_hi": hi, "train_r2": s["train_r2"], "train_rmse": s["train_rmse"],
                      "n_nonzero": count_nonzero(fit.w), "n_features": ctx.split.X_fit.shape[1] + 1,
                      "day_block_r2": cv[m]["mean"][fit.index], "day_block_se": cv[m]["se"][fit.index],
                      **chrono_result(fit, curve.alphas, chrono, ctx)}
        preds[m] = pred
    return methods, preds


def unregularised_model(ctx: Context):
    """Ridge with a negligible penalty: the closed-form least-squares fit (scores, validation predictions)."""
    b, w = ridge_closed_form(ctx.split.X_fit, ctx.split.z_fit, UNREGULARISED_ALPHA)
    s, pred = score(ctx.split, b, w, ctx.back_method)
    return {"alpha": UNREGULARISED_ALPHA, "train_r2": s["train_r2"], "train_rmse": s["train_rmse"],
            "val_r2": s["r2"], "val_rmse": s["rmse"]}, pred


def paired(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, ctx: Context) -> dict:
    delta, lo, hi = paired_bootstrap_delta_r2(y, pred_a, pred_b, ctx.seed, ctx.B)
    return {"delta": delta, "lo": lo, "hi": hi}


def comparisons(preds: dict, pred_unreg: np.ndarray, ctx: Context) -> dict:
    """Paired bootstrap on validation: method differences and each method minus the unregularised fit."""
    y = ctx.split.cnt_eval
    out = {f"{a}_minus_{b}": paired(y, preds[a], preds[b], ctx) for a, b in (("l1", "l2"), ("enet", "l2"), ("enet", "l1"))}
    out.update({f"{m}_minus_unregularised": paired(y, preds[m], pred_unreg, ctx) for m in ("l1", "l2", "enet")})
    return out


def recommend(methods: dict, preds: dict, fits: dict[str, Fit], feature_names: list[str], ctx: Context) -> dict:
    """Within noise of the best on validation, then the best on held-out days."""
    best = max(METHODS, key=lambda m: methods[m]["val_r2"])
    within = [m for m in METHODS if m == best or _interval_has_zero(
        paired(ctx.split.cnt_eval, preds[m], preds[best], ctx))]
    pick = max(within, key=lambda m: methods[m]["day_block_r2"])
    m, fit = methods[pick], fits[pick]
    return {"method": pick, "candidates": within, "best_on_validation": best, "lambda": m["lambda"],
            "l1_ratio": m["l1_ratio"], "val_r2": m["val_r2"], "val_rmse": m["val_rmse"],
            "day_block_r2": m["day_block_r2"], "chrono_r2": m["chrono_r2"],
            "rule": "within-noise on validation, then best on held-out days",
            "intercept": fit.b, "weights": fit.w.tolist(), "feature_names": feature_names}


def _interval_has_zero(d: dict) -> bool:
    return d["lo"] <= 0.0 <= d["hi"]


# ----------------------------------------------------------------------------- paths and stability

def coefficient_paths(curves: dict, names: list[str]) -> tuple[dict, dict]:
    """paths.{l1,l2} for the plotted names, and entry_alpha (largest lasso alpha with a non-zero weight) per feature."""
    shown = [n for n in PATH_NAMES if n in names]
    paths = {m: {"alphas": [float(a) for a in curves[m].alphas],
                 "coef": {n: curves[m].W[:, names.index(n)].tolist() for n in shown}} for m in ("l1", "l2")}
    lasso = curves["l1"]
    entry = {}
    for j, name in enumerate(names):
        active = np.abs(lasso.W[:, j]) > NONZERO_TOL
        entry[name] = float(lasso.alphas[active].max()) if active.any() else None
    return paths, entry


def stability_frequency(split: Split, fit: Fit, groups: np.ndarray, ctx: Context, where: str) -> np.ndarray:
    """Share of B date-bootstraps in which each coefficient is non-zero (like regularization.stability_selection,
    but with our sweep budget so that non-convergence is counted)."""
    B = ctx.p4cfg["stability"]["B"]
    selected = np.zeros(split.X_fit.shape[1])
    for b in range(B):
        rows = group_bootstrap_rows(groups, np.random.default_rng(ctx.seed + b))
        _, w, sweeps = enet_cd(split.X_fit[rows], split.z_fit[rows], fit.alpha, fit.rho, None, SOLVER_TOL, MAX_SWEEPS)
        ctx.log.record(where, sweeps)
        selected += np.abs(w) > NONZERO_TOL
    return selected / B


def stability(fits: dict[str, Fit], ctx: Context) -> dict:
    """stability.{l1,enet}.freq per feature, resampling whole training dates."""
    groups = ctx.train_df["dteday"].to_numpy()
    names = ctx.split.names
    return {m: {"freq": dict(zip(names, stability_frequency(ctx.split, fits[m], groups, ctx, f"stability {m}").tolist()))}
            for m in ("l1", "enet")}


# ----------------------------------------------------------------------------- does a penalty rescue the top of the ladder?

def rich_check(p2: dict, fits: dict[str, Fit], rec_pred: np.ndarray, ctx: Context) -> dict:
    """Last ladder level without candidates: unregularised, ridge and lasso on validation (informational only)."""
    top = ladder_specs(ctx.cfg, DesignSpec.from_dict(p2["design_spec"]))[-1]
    split = build_split(top["spec"], ctx.train_df, ctx.val_df)
    b, w = ridge_closed_form(split.X_fit, split.z_fit, UNREGULARISED_ALPHA)
    unreg = score(split, b, w, ctx.back_method)[0]["r2"]
    folds = fold_splits(ctx, top["spec"])
    result = {"level": top["name"], "n_features": split.X_fit.shape[1] + 1, "unregularised_val_r2": unreg}
    best_pred, best_r2 = None, -np.inf
    for m, rho in (("l2", 0.0), ("l1", 1.0)):
        curve = build_curve(split, rho, ctx, f"rich {m}")
        i = best_index(curve.alphas, curve_scores(curve))
        # day-block score only at the chosen alpha: the path of each fold stops there
        fold_r2 = cv_fold_r2(folds, curve.alphas[: i + 1], rho, ctx, f"rich cv {m}")[:, -1]
        entry = {"lambda": float(curve.alphas[i]), "val_r2": curve.rows[i]["val_r2"],
                 "day_block_r2": float(fold_r2.mean())}
        if m == "l1":
            entry["n_nonzero"] = curve.rows[i]["n_nonzero"]
        result[m] = entry
        if entry["val_r2"] > best_r2:
            best_r2, best_pred = entry["val_r2"], score(split, curve.B[i], curve.W[i], ctx.back_method)[1]
    result["paired_best_vs_recommended"] = paired(ctx.split.cnt_eval, best_pred, rec_pred, ctx)
    return result


# ----------------------------------------------------------------------------- verdicts

def ridge_predictions(split: Split, lam: float, back_method: str) -> np.ndarray:
    """Evaluation predictions of ridge at lam on this split's columns."""
    b, w = ridge_closed_form(split.X_fit, split.z_fit, lam)
    return score(split, b, w, back_method)[1]


def columns_without(split: Split, dropped: set[str]) -> Split:
    return split.keep([j for j, n in enumerate(split.names) if n not in dropped])


def drop_cost(split: Split, reference_pred: np.ndarray, dropped: set[str], lam: float, ctx: Context) -> dict:
    """R2(full) - R2(without `dropped`) on the evaluation rows, paired bootstrap interval (positive = dropping hurts)."""
    pred = ridge_predictions(columns_without(split, dropped), lam, ctx.back_method)
    return paired(split.cnt_eval, reference_pred, pred, ctx)


def drop_cost_day_block(folds: list[Split], reference_r2: list[float], dropped: set[str], lam: float, ctx: Context) -> float:
    """Mean over folds of R2(full) - R2(without `dropped`); every fold refits its design."""
    costs = []
    for split, full_r2 in zip(folds, reference_r2):
        pred = ridge_predictions(columns_without(split, dropped), lam, ctx.back_method)
        costs.append(full_r2 - r2(split.cnt_eval, pred))
    return float(np.mean(costs))


def solo_r2(split: Split, column: str, lam: float, ctx: Context) -> float | None:
    """Validation R2 of ridge (same lambda) on the own features of `column` only."""
    mine = vd.own(column, split.names)
    if not mine:
        return None
    only = split.keep([split.names.index(n) for n in mine])
    return r2(split.cnt_eval, ridge_predictions(only, lam, ctx.back_method))


def lasso_numbers(column: str, names: list[str], w: np.ndarray, freq: dict, ranks: dict) -> dict:
    """What the lasso (chosen alpha) and its stability selection say about one column."""
    mine, touching = vd.own(column, names), vd.involving(column, names)
    nonzero = {n for n, v in zip(names, w) if abs(v) > NONZERO_TOL}
    return {"n_own": len(mine), "n_own_nonzero": len([n for n in mine if n in nonzero]),
            "n_involving": len(touching), "n_involving_nonzero": len([n for n in touching if n in nonzero]),
            "max_freq_own": max((freq[n] for n in mine), default=None),
            "mean_freq_involving": float(np.mean([freq[n] for n in touching])) if touching else None,
            "entry_rank": ranks[column]}


def column_numbers(column: str, ctx: Context, lam: float, ref_pred: np.ndarray, folds: list[Split],
                   ref_fold_r2: list[float], lasso_w: np.ndarray, freq: dict, ranks: dict) -> dict:
    """Every number the verdict rule and its evidence sentence use, for one original column."""
    names = ctx.split.names
    alone = set(vd.involving(column, names))
    others = vd.partners(column)
    group_set = alone.union(*(vd.involving(p, names) for p in others)) if others else None
    group = None
    if others:
        group = {**drop_cost(ctx.split, ref_pred, group_set, lam, ctx), "members": [column] + others}
    return {"drop_alone": drop_cost(ctx.split, ref_pred, alone, lam, ctx),
            "drop_alone_day_block": drop_cost_day_block(folds, ref_fold_r2, alone, lam, ctx),
            "drop_group": group,
            "solo_r2": solo_r2(ctx.split, column, lam, ctx),
            "lasso": lasso_numbers(column, names, lasso_w, freq, ranks),
            "max_abs_corr": vd.max_abs_corr(ctx.split.X_fit, names, column),
            "vif_max": vd.vif_max(ctx.split.X_fit, names, column)}


def column_verdicts(fits: dict[str, Fit], entry_alpha: dict, stab: dict, ctx: Context) -> dict:
    """The verdict, evidence and numbers of all 14 original columns (reference = ridge at its chosen lambda)."""
    lam = fits["l2"].alpha
    thr = ctx.p4cfg["verdict"]
    ref_pred = ridge_predictions(ctx.split, lam, ctx.back_method)
    folds = fold_splits(ctx)
    ref_fold_r2 = [r2(s.cnt_eval, ridge_predictions(s, lam, ctx.back_method)) for s in folds]
    ranks = vd.entry_ranks(entry_alpha, ctx.split.names)
    numbers = {c: column_numbers(c, ctx, lam, ref_pred, folds, ref_fold_r2, fits["l1"].w,
                                 stab["l1"]["freq"], ranks) for c in vd.ORIGINAL}
    verdict = {c: vd.basic_verdict(numbers[c], thr["min_delta"], thr["solo_min"]) for c in vd.ORIGINAL}
    representative = vd.promote_representatives(verdict, numbers, thr["min_delta"])
    out = {}
    for c in vd.ORIGINAL:
        carriers = vd.carried_by(c, verdict, numbers[c])
        out[c] = {"verdict": verdict[c], "evidence": vd.evidence_sentence(c, numbers[c], carriers),
                  "carried_by": carriers, "representative": representative[c], "numbers": numbers[c]}
    return out


# ----------------------------------------------------------------------------- survivors

def survivors(verdict: dict, fits: dict[str, Fit], stab: dict, ctx: Context) -> dict:
    """Lasso non-zero AND stable AND every source column judged useful; then a ridge refit on the survivors."""
    names, lam = ctx.split.names, fits["l2"].alpha
    threshold = ctx.p4cfg["stability"]["threshold"]
    nonzero = [n for n, v in zip(names, fits["l1"].w) if abs(v) > NONZERO_TOL]
    stable = [n for n in nonzero if stab["l1"]["freq"][n] >= threshold]
    kept = [n for n in stable if all(verdict[c]["verdict"] == "useful" for c in sources(n))]
    original = [c for c in vd.ORIGINAL if verdict[c]["verdict"] == "useful" and any(c in sources(n) for n in kept)]
    columns = [names.index(n) for n in kept]
    split = ctx.split.keep(columns)
    scores, _ = score(split, *ridge_closed_form(split.X_fit, split.z_fit, lam), ctx.back_method)
    fold_r2 = [r2(f.cnt_eval, ridge_predictions(f.keep(columns), lam, ctx.back_method)) for f in fold_splits(ctx)]
    return {"survivors_expanded": kept, "survivors_original": original,
            "survivor_counts": {"lasso_nonzero": len(nonzero), "stable": len(stable), "after_verdicts": len(kept)},
            "survivor_refit": {"n_features": len(kept) + 1, "val_r2": scores["r2"], "val_rmse": scores["rmse"],
                               "day_block_r2": float(np.mean(fold_r2))}}


# ----------------------------------------------------------------------------- run

def build_context(cfg: dict, p1: dict, p2: dict, p3: dict) -> Context:
    seed = config_seed(cfg)
    all_df = load_train(cfg)
    train_df, val_df = seeded_split(all_df, seed, cfg["split"]["test_size"])
    spec, design_from = pick_design(cfg, p2, p3, train_df)
    return Context(cfg, cfg["p4"], seed, cfg["bootstrap"]["B"], p1["backtransform"]["method"], all_df, train_df,
                  val_df, spec, build_split(spec, train_df, val_df), SolverLog(), design_from)


def grid_summary(curves: dict) -> dict:
    def span(c: Curve) -> dict:
        return {"min": float(c.alphas.min()), "max": float(c.alphas.max())}
    return {"l2": span(curves["l2"]), "l1": span(curves["l1"]),
            "enet": {str(rho): span(c) for rho, c in curves["enet"].items()}}


def run(cfg: dict | None = None) -> dict:
    """Execute Phase 4 and write artifacts/p4.json."""
    cfg = cfg if cfg is not None else load_config()
    set_threads(cfg.get("threads", 1))
    TIMINGS.clear()
    p1, p2, p3 = read_artifact("p1"), read_artifact("p2"), read_artifact("p3")
    with timed("design and baseline"):
        ctx = build_context(cfg, p1, p2, p3)
        unreg, pred_unreg = unregularised_model(ctx)
    with timed("validation curves"):
        curves = validation_curves(ctx)
        fits = choose_fits(curves)
    with timed("day-block cross-validation"):
        cv = cross_validation(curves, fits, ctx)
    with timed("methods, comparisons, recommendation"):
        methods, preds = describe_methods(curves, fits, cv, ctx)
        compare = comparisons(preds, pred_unreg, ctx)
        recommended = recommend(methods, preds, fits, ctx.split.names, ctx)
    with timed("paths and stability"):
        paths, entry_alpha = coefficient_paths(curves, ctx.split.names)
        stab = stability(fits, ctx)
    with timed("rich check"):
        rich = rich_check(p2, fits, preds[recommended["method"]], ctx)
    with timed("column verdicts"):
        verdict = column_verdicts(fits, entry_alpha, stab, ctx)
    with timed("survivors"):
        surv = survivors(verdict, fits, stab, ctx)

    payload = {
        "design_from": ctx.design_from, "design_spec": ctx.spec.to_dict(),
        "feature_names": ctx.split.names, "n_features": ctx.split.X_fit.shape[1] + 1,
        "back_method": ctx.back_method, "unregularised": unreg,
        "curves": {"l2": curves["l2"].rows, "l1": curves["l1"].rows,
                   "enet": {str(rho): c.rows for rho, c in curves["enet"].items()}},
        "cv": cv, "methods": methods, "comparisons": compare, "recommended": recommended,
        "paths": paths, "entry_alpha": entry_alpha, "stability": stab,
        "stability_threshold": ctx.p4cfg["stability"]["threshold"], "verdict_thresholds": ctx.p4cfg["verdict"],
        "rich_check": rich, "column_verdicts": verdict, **surv,
        "not_converged": {"count": ctx.log.not_converged, "where": ctx.log.where},
        "solver": {"tol": SOLVER_TOL, "max_sweeps": MAX_SWEEPS}, "grids": grid_summary(curves)}
    write_artifact("p4", payload, upstream="p3", cfg=cfg)
    print(f"[p4] total {sum(TIMINGS.values()):.1f} s | not converged: {ctx.log.not_converged} | "
          f"most sweeps used: {ctx.log.slowest_sweeps}")
    return payload


if __name__ == "__main__":
    run()
