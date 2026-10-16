"""Shared phase library: the seeded split, bike-scale scoring, closed-form fits and the three validators
(ARCHITECTURE section 8). Phases 1-4 import their common helpers from here, never from each other.

Every fit here is ordinary least squares on z = ((cnt + 1)^power - 1) / power (log1p for power 0) with a Design
learned on the FITTING frame only. Scores are on the bike scale: cnt_hat = clip(s * q(eta) - 1, 0, None), where the
back-transform factor s is recomputed from the fitting frame's own residuals (the Target comes from Phase 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common import Target, chrono_split, config_seed, day_block_folds, load_train, r2, rmse, seeded_split
from src.features import BASE, HR, Design, DesignSpec
from src.gd import mse_loss

N_RESIDUAL_BINS = 10      # equal-count bins of temp / hum in the residual profile (a report choice, not a tuning knob)


# ----------------------------------------------------------------------------- the split and bike-scale scores

@dataclass
class Split:
    """Seeded train/validation frames with the target on both scales."""
    seed: int
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    cnt_tr: np.ndarray   # bikes, train
    cnt_va: np.ndarray   # bikes, validation
    z_tr: np.ndarray     # target.forward(bikes), train
    z_va: np.ndarray     # target.forward(bikes), validation


def config_target(cfg: dict) -> Target:
    """The target named in config.yaml; the back-transform method is not needed to form z, so it is left at 'none'."""
    return Target(float(cfg["p1"]["target_power"]), "none")


def make_split(cfg: dict, target: Target | None = None) -> Split:
    """The single seeded split of the repo, with the target on the bike scale and on the z scale of `target`."""
    target = config_target(cfg) if target is None else target
    seed = config_seed(cfg)
    train_df, val_df = seeded_split(load_train(cfg), seed, cfg["split"]["test_size"])
    cnt_tr = train_df["cnt"].to_numpy(dtype=np.float64)
    cnt_va = val_df["cnt"].to_numpy(dtype=np.float64)
    return Split(seed, train_df, val_df, cnt_tr, cnt_va, target.forward(cnt_tr), target.forward(cnt_va))


def base_design(split: Split) -> tuple[Design, np.ndarray, np.ndarray]:
    """The 35-column base design (plus bias) fitted on train only; returns (design, X_train, X_validation)."""
    design = Design(DesignSpec(base=BASE)).fit(split.train_df)
    return design, design.transform(split.train_df), design.transform(split.val_df)


def bike_predictions(split: Split, X_tr, X_va, w, target: Target) -> tuple[float, np.ndarray, np.ndarray]:
    """(factor, train bikes, validation bikes) of the weights w, with s from the train residuals only."""
    return target.predict(X_tr @ w, split.cnt_tr, X_va @ w)


def model_scores(split: Split, X_tr, X_va, w, target: Target) -> dict:
    """Train/validation R2 and RMSE on the bike scale, R2 on the z scale, and the factor used."""
    factor, pred_tr, pred_va = bike_predictions(split, X_tr, X_va, w, target)
    return {"factor": factor,
            "train_r2": r2(split.cnt_tr, pred_tr), "train_rmse": rmse(split.cnt_tr, pred_tr),
            "val_r2": r2(split.cnt_va, pred_va), "val_rmse": rmse(split.cnt_va, pred_va),
            "train_r2_log": r2(split.z_tr, X_tr @ w), "val_r2_log": r2(split.z_va, X_va @ w)}


def oracle_gap(split: Split, X_tr, X_va, w, target: Target) -> dict:
    """Closed-form least squares (test oracle only, never used for fitting) against our GD weights."""
    w_exact = np.linalg.lstsq(X_tr, split.z_tr, rcond=None)[0]
    scores_gd = model_scores(split, X_tr, X_va, w, target)
    scores_exact = model_scores(split, X_tr, X_va, w_exact, target)
    return {"label": "oracle: np.linalg.lstsq, used only to check GD",
            "max_abs_weight_diff": float(np.max(np.abs(w - w_exact))),
            "loss_gap": mse_loss(X_tr, split.z_tr, w) - mse_loss(X_tr, split.z_tr, w_exact),
            "val_r2_diff": scores_gd["val_r2"] - scores_exact["val_r2"]}


def curve_points(history, max_points: int) -> list:
    """[[iteration, loss], ...] with at most max_points points; cut after the first non-finite loss (stored as null)."""
    losses = np.asarray(history, dtype=np.float64)
    finite = np.isfinite(losses)
    if not finite.all():
        losses = losses[: int(np.argmin(finite)) + 1]
    picks = np.unique(np.linspace(0, len(losses) - 1, min(max_points, len(losses))).round().astype(int))
    return [[int(i), float(losses[i]) if np.isfinite(losses[i]) else None] for i in picks]


def _equal_count_bins(values: np.ndarray, resid: np.ndarray, n_bins: int) -> list:
    """Equal-count bins of `values` (sorted, split into n_bins runs) with the mean residual of each."""
    parts = np.array_split(np.argsort(values, kind="stable"), n_bins)
    return [{"lo": float(values[p].min()), "hi": float(values[p].max()),
             "mean_resid": float(resid[p].mean()), "n": int(len(p))} for p in parts]


def residual_profile(df: pd.DataFrame, resid: np.ndarray) -> dict:
    """Mean z-scale residual by (workingday, hour) cell and by 10 equal-count temp and hum bins."""
    table = pd.DataFrame({"workingday": df["workingday"].to_numpy(), "hr": df["hr"].to_numpy(), "r": resid})
    cells = table.groupby(["workingday", "hr"])["r"].agg(["mean", "size"]).reset_index()
    by_wd_hr = [{"workingday": int(row.workingday), "hr": int(row.hr), "mean_resid": float(row["mean"]),
                 "n": int(row["size"])} for _, row in cells.iterrows()]
    return {"by_wd_hr": by_wd_hr,
            "by_temp_bin": _equal_count_bins(df["temp"].to_numpy(dtype=np.float64), resid, N_RESIDUAL_BINS),
            "by_hum_bin": _equal_count_bins(df["hum"].to_numpy(dtype=np.float64), resid, N_RESIDUAL_BINS)}


# ----------------------------------------------------------------------------- the complexity ladder

def ladder_specs(cfg: dict, anchor_spec: DesignSpec) -> list[dict]:
    """Cumulative ladder: levels before `phase2` are degree 1; the phase2 level is the anchor; later levels extend it."""
    levels, accumulated, seen_anchor = [], [], False
    for index, level in enumerate(cfg["p3"]["ladder"]):
        accumulated += [b for b in level["add"] if b not in accumulated]
        if level.get("phase2"):
            assert set(accumulated) <= set(anchor_spec.blocks), "Phase 2 design lacks blocks of earlier levels"
            spec, seen_anchor = anchor_spec, True
        elif not seen_anchor:
            base = [c for c in BASE if c not in HR] if level.get("drop_hour") else list(BASE)
            spec = DesignSpec(base=tuple(base), blocks=tuple(accumulated))
        else:
            extra = [b for b in accumulated if b not in anchor_spec.blocks]
            spec = DesignSpec(base=anchor_spec.base, power_cols=anchor_spec.power_cols, degree=anchor_spec.degree,
                              blocks=anchor_spec.blocks + tuple(extra))
        levels.append({"name": level["name"], "index": index, "spec": spec, "is_anchor": bool(level.get("phase2"))})
    return levels


# ----------------------------------------------------------------------------- closed-form fits


def fit_linear(X: np.ndarray, y_z: np.ndarray, alpha: float = 1e-8) -> np.ndarray:
    """Normal equations (X'X/n + alpha*D) w = X'y/n, D = identity with a 0 for the bias column (column 0)."""
    n = X.shape[0]
    gram = X.T @ X / n
    penalty = np.full(X.shape[1], alpha)
    penalty[0] = 0.0  # the bias is never penalised
    gram[np.diag_indices_from(gram)] += penalty
    rhs = X.T @ y_z / n
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:  # exactly singular even after the ridge term: minimum-norm solution
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def fit_predict(spec: DesignSpec, fit_df: pd.DataFrame, eval_df: pd.DataFrame, target: Target,
                alpha: float = 1e-8) -> tuple[dict, np.ndarray]:
    """Fit `spec` on fit_df; return (scores dict, bike-scale predictions for eval_df)."""
    design = Design(spec).fit(fit_df)
    X_fit, X_eval = design.transform(fit_df), design.transform(eval_df)
    cnt_fit, cnt_eval = fit_df["cnt"].to_numpy(dtype=np.float64), eval_df["cnt"].to_numpy(dtype=np.float64)
    z_fit = target.forward(cnt_fit)
    w = fit_linear(X_fit, z_fit, alpha)
    eta_fit = X_fit @ w
    eta_eval = target.capped(X_eval @ w, cnt_fit)   # capping again inside predict() changes nothing
    _, pred_fit, pred_eval = target.predict(eta_fit, cnt_fit, eta_eval)
    scores = {"p": int(X_fit.shape[1]),
              "train_r2": r2(cnt_fit, pred_fit), "train_rmse": rmse(cnt_fit, pred_fit),
              "r2": r2(cnt_eval, pred_eval), "rmse": rmse(cnt_eval, pred_eval),
              "train_r2_log": r2(z_fit, eta_fit), "r2_log": r2(target.forward(cnt_eval), eta_eval)}
    return scores, pred_eval


def evaluate_spec(spec: DesignSpec, fit_df: pd.DataFrame, eval_df: pd.DataFrame, target: Target,
                  alpha: float = 1e-8) -> dict:
    """Scores of one spec fitted on fit_df and evaluated on eval_df (p, train_r2, train_rmse, r2, rmse, *_log)."""
    return fit_predict(spec, fit_df, eval_df, target, alpha)[0]


def day_block_scores(spec: DesignSpec, train_df: pd.DataFrame, target: Target, k: int, alpha: float) -> dict:
    """k-fold CV where each fold is a set of whole days (rank of date mod k); folds live INSIDE train_df."""
    folds = day_block_folds(train_df, k)
    fold_scores = []
    for f in range(k):
        held = folds == f
        fold_scores.append(evaluate_spec(spec, train_df[~held], train_df[held], target, alpha))
    r2s = [s["r2"] for s in fold_scores]
    return {"r2": float(np.mean(r2s)), "sd": float(np.std(r2s, ddof=1)), "folds": r2s,
            "train_r2": float(np.mean([s["train_r2"] for s in fold_scores]))}


def chrono_scores(spec: DesignSpec, train_all: pd.DataFrame, cut_date: str, target: Target, alpha: float) -> dict:
    """Fit on dates before cut_date, evaluate on dates from cut_date on."""
    early_df, late_df = chrono_split(train_all, cut_date)
    s = evaluate_spec(spec, early_df, late_df, target, alpha)
    return {"train_r2": s["train_r2"], "r2": s["r2"], "rmse": s["rmse"]}


def three_validators(spec: DesignSpec, train_df: pd.DataFrame, val_df: pd.DataFrame, train_all: pd.DataFrame,
                     cut_date: str, target: Target, k: int = 5, alpha: float = 1e-8) -> dict:
    """The seeded split, day-block CV inside train_df, and the chronological split, each fitted on its own training part."""
    seeded = evaluate_spec(spec, train_df, val_df, target, alpha)
    return {"seeded": {key: seeded[key] for key in ("p", "train_r2", "train_rmse", "r2", "rmse")},
            "day_block": day_block_scores(spec, train_df, target, k, alpha),
            "chrono": chrono_scores(spec, train_all, cut_date, target, alpha)}


def nested_day_subsets(train_df: pd.DataFrame, fractions: list[float], seed: int) -> list[tuple[int, pd.DataFrame]]:
    """(n_days, rows) for each fraction: nested prefixes of ONE permutation of the training dates."""
    dates = pd.to_datetime(train_df["dteday"])
    unique_dates = np.sort(dates.unique())
    order = np.random.default_rng(seed).permutation(len(unique_dates))
    subsets = []
    for fraction in fractions:
        n_days = max(1, int(round(fraction * len(unique_dates))))
        keep = unique_dates[order[:n_days]]
        subsets.append((n_days, train_df[dates.isin(keep).to_numpy()]))
    return subsets


def learning_curve(spec: DesignSpec, train_df: pd.DataFrame, val_df: pd.DataFrame, fractions: list[float],
                   seed: int, target: Target, alpha: float = 1e-8) -> list[dict]:
    """Train and validation R2 as the number of training DAYS grows (subsets are nested)."""
    rows = []
    for fraction, (n_days, subset) in zip(fractions, nested_day_subsets(train_df, fractions, seed)):
        s = evaluate_spec(spec, subset, val_df, target, alpha)
        rows.append({"fraction": fraction, "n_days": n_days, "n_rows": int(len(subset)),
                     "train_r2": s["train_r2"], "val_r2": s["r2"]})
    return rows


def noise_floor(train_df: pd.DataFrame) -> dict:
    """Share of cnt variance explained by cell means over (yr, mnth, workingday, hr, weathersit<=3).

    This is an optimistic ceiling for any model that only sees those factors: with many cells holding one or two
    rows the within-cell sum of squares is biased downwards, so we also report a degrees-of-freedom adjusted value.
    """
    cells = train_df[["yr", "mnth", "workingday", "hr"]].copy()
    cells["ws"] = np.minimum(train_df["weathersit"].to_numpy(), 3)
    y = train_df["cnt"].to_numpy(dtype=np.float64)
    cell_id = cells.groupby(list(cells.columns)).ngroup().to_numpy()
    sizes = np.bincount(cell_id)
    cell_means = np.bincount(cell_id, weights=y) / sizes
    within_ss = float(np.sum((y - cell_means[cell_id]) ** 2))
    total_ss = float(np.sum((y - y.mean()) ** 2))
    n, n_cells = len(y), len(sizes)
    # every row its own cell: no degrees of freedom are left, the adjusted value is defined as 0
    adjusted = 0.0 if n == n_cells else 1.0 - (within_ss / (n - n_cells)) / (total_ss / (n - 1))
    return {"noise_floor_r2": 1.0 - within_ss / total_ss,
            "noise_floor_r2_adjusted": adjusted,
            "n_cells": int(n_cells), "singleton_cell_share": float(np.mean(sizes == 1)),
            "singleton_row_share": float(np.sum(sizes == 1) / n)}
