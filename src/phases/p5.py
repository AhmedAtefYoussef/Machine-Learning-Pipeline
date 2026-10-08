"""Phase 5: logistic regression for "is this a high-demand hour?" (our own gradient descent, src/logistic.py).

Pipeline: surviving columns of Phase 4 -> label rule fitted on the training rows -> ridge strength by validation
ROC-AUC -> operating threshold from the 3:1 miss / false-alarm cost -> metrics, calibration, coefficients ->
label-rule foils -> retrospective over Phases 1-5 -> artifacts/p5.json.

    p = sigmoid(x . w),   L(w) = mean cross-entropy + (l2 / 2) ||w[1:]||^2,   bias column 0 never penalised.
    Predict "high" when p >= t.  Cost of a threshold on n rows: (c * FN + FP) / n with c = miss / false-alarm ratio.

scikit-learn is not imported here; everything comes from our own modules.
"""
from __future__ import annotations

import re
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from src.common import config_seed, load_config, load_train, read_artifact, seeded_split, set_threads, write_artifact
from src.features import BASE, Design, DesignSpec, sources
from src.gd import gradient_check, lambda_max
from src.labels import apply_thresholds, fit_thresholds
from src.logistic import (calibration_table, confusion, fit_logistic, logloss, logloss_grad, pr_auc, prf, roc_auc,
                          sigmoid)

THRESHOLD_GRID = [round(0.05 * k, 2) for k in range(1, 20)]   # 0.05 ... 0.95 for the cost / F1 curves
CALIBRATION_BINS = 10
CURVE_POINTS = 200            # points kept of the ROC / PR curves
CALIBRATION_MIN_N = 30        # a bin with fewer rows is too noisy to enter the max gap
TOP_COEFFICIENTS = 15
AUC_TIE = 1.0e-4              # l2 values within this ROC-AUC of the best count as tied; the larger l2 wins
BOOTSTRAP_RNG_OFFSET = 5      # rng = default_rng(seed + 5): the Phase 5 validation bootstrap
HOUR_COLUMN = re.compile(r"^hr_\d+$")
FOIL_RULES = ([], ["workingday", "hr"])   # global threshold, and the (day type, hour) rule without the year
TIMINGS: dict[str, float] = {}


@contextmanager
def timed(step: str):
    """Record and print the wall-clock time of one step."""
    start = time.perf_counter()
    yield
    TIMINGS[step] = time.perf_counter() - start
    print(f"[p5] {step}: {TIMINGS[step]:.1f} s")


# ----------------------------------------------------------------------------- step 1: the surviving columns

def survivor_design(p4: dict, train_df: pd.DataFrame) -> tuple[Design, list[str]]:
    """The Phase 4 design fitted on the training rows, and the survivor names (all must exist in it)."""
    design = Design(DesignSpec.from_dict(p4["design_spec"])).fit(train_df)
    names = list(p4["survivors_expanded"])
    missing = [n for n in names if n not in design.names]
    assert not missing, f"survivors not in the design: {missing[:5]}"
    return design, names


def design_matrix(design: Design, names: list[str], df: pd.DataFrame) -> np.ndarray:
    """Bias column followed by the named columns (training statistics, nothing refitted)."""
    columns = design.transform(df)[:, design.subset(names)]
    return np.column_stack([np.ones(len(df)), columns])


# ----------------------------------------------------------------------------- step 2: labels

def label_rule(train_df: pd.DataFrame, val_df: pd.DataFrame, group_by: list[str], quantile: float):
    """(threshold table, y_train, y_val): thresholds fitted on the training rows only."""
    table = fit_thresholds(train_df, group_by, quantile)
    return table, apply_thresholds(train_df, table, group_by), apply_thresholds(val_df, table, group_by)


def describe_rule(table: pd.DataFrame, group_by: list[str], quantile: float) -> dict:
    """The chain_check key `threshold_rule`."""
    return {"quantile": quantile, "group_by": list(group_by), "fitted_on": "train", "n_cells": len(table),
            "min_cell_n": int(table["n"].min()), "table": table.to_dict("records")}


def majority_accuracy(y: np.ndarray) -> float:
    """Accuracy of always predicting the more frequent class."""
    share = float(np.mean(y))
    return max(share, 1.0 - share)


def class_balance(y_tr: np.ndarray, y_va: np.ndarray, val_df: pd.DataFrame) -> dict:
    """Positive share in train, validation and each validation year, plus the majority-class accuracy."""
    year = val_df["yr"].to_numpy()
    return {"train": float(np.mean(y_tr)), "val": float(np.mean(y_va)),
            "val_by_year": {str(y): float(np.mean(y_va[year == y])) for y in (0, 1)},
            "majority_accuracy": majority_accuracy(y_va)}


# ----------------------------------------------------------------------------- step 3: model and l2

def default_lr(X: np.ndarray, l2: float) -> float:
    """1 / L for the smoothness L = lambda_max(X'X/n) / 4 + l2 (the step fit_logistic uses by default)."""
    return 1.0 / (lambda_max(X) / 4.0 + l2)


def fit_candidate(X_tr, y_tr, X_va, y_va, l2: float, max_iter: int):
    """(GDResult, row) for one l2: iterations, stop reason, train / validation log-loss, validation ROC-AUC."""
    result = fit_logistic(X_tr, y_tr, l2=l2, lr=default_lr(X_tr, l2), max_iter=max_iter)
    w = result.weights
    row = {"l2": l2, "iterations": result.iterations, "stop_reason": result.stop_reason,
           "train_logloss": logloss(X_tr, y_tr, w), "val_logloss": logloss(X_va, y_va, w),
           "val_roc_auc": roc_auc(y_va, sigmoid(X_va @ w))}
    return result, row


def choose_l2(rows: list[dict]) -> int:
    """Index of the best validation ROC-AUC; any l2 within AUC_TIE of it ties and the largest l2 wins."""
    best = max(r["val_roc_auc"] for r in rows)
    tied = [i for i, r in enumerate(rows) if r["val_roc_auc"] >= best - AUC_TIE]
    return max(tied, key=lambda i: rows[i]["l2"])


def check_gradient(X, y, w: np.ndarray, l2: float) -> float:
    """Max relative error of the analytic log-loss gradient against central differences at w."""
    return gradient_check(lambda v: logloss(X, y, v, l2), lambda v: logloss_grad(X, y, v, l2), w)


# ----------------------------------------------------------------------------- steps 4-5: threshold and metrics

def cost_at(y, p, thr: float, cost_ratio: float) -> float:
    """(c * FN + FP) / n for the rule p >= thr."""
    c = confusion(y, p, thr)
    return (cost_ratio * c["fn"] + c["fp"]) / len(y)


def metrics_at(y, p, thr: float, cost_ratio: float) -> dict:
    """Everything reported for one operating threshold."""
    return {"threshold": thr, **prf(y, p, thr), "roc_auc": roc_auc(y, p), "pr_auc": pr_auc(y, p),
            "confusion": confusion(y, p, thr), "cost": cost_at(y, p, thr, cost_ratio)}


def threshold_curve(y, p, cost_ratio: float) -> list[dict]:
    """Accuracy, precision, recall, F1 and cost on the validation rows for each grid threshold."""
    return [{"thr": t, **prf(y, p, t), "cost": cost_at(y, p, t, cost_ratio)} for t in THRESHOLD_GRID]


def best_threshold(curve: list[dict], key: str, largest: bool) -> float:
    """Grid threshold with the largest (or smallest) value of `key`; ties go to the first (lowest) threshold."""
    pick = max if largest else min
    return pick(curve, key=lambda r: r[key])["thr"]


def curve_points(y, p, n_points: int = CURVE_POINTS) -> list[dict]:
    """ROC / PR points (fpr, recall, precision) at about n_points thresholds spread over the sorted scores."""
    order = np.argsort(-p, kind="mergesort")
    tp, fp = np.cumsum(y[order] == 1), np.cumsum(y[order] != 1)
    keep = np.unique(np.linspace(0, len(y) - 1, n_points).astype(int))
    return [{"thr": float(p[order][i]), "fpr": float(fp[i] / fp[-1]), "recall": float(tp[i] / tp[-1]),
             "precision": float(tp[i] / (tp[i] + fp[i]))} for i in keep]


def bootstrap_auc(y, p, seed: int, B: int) -> dict:
    """95% percentile interval of the validation ROC-AUC over B resamples of the validation rows."""
    rng = np.random.default_rng(seed + BOOTSTRAP_RNG_OFFSET)
    idx = rng.integers(0, len(y), size=(B, len(y)))
    aucs = [roc_auc(y[i], p[i]) for i in idx]
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi)}


def calibration(y, p) -> tuple[list[dict], float]:
    """Calibration table and the largest |mean predicted - observed share| over bins with enough rows."""
    table = calibration_table(y, p, CALIBRATION_BINS)
    gaps = [abs(r["mean_p"] - r["frac_pos"]) for r in table if r["n"] >= CALIBRATION_MIN_N]
    return table, float(max(gaps))


# ----------------------------------------------------------------------------- step 6: coefficients

def top_coefficients(names: list[str], w: np.ndarray, k: int = TOP_COEFFICIENTS) -> list[dict]:
    """The k largest |weight| (columns are standardised, so weights are comparable)."""
    order = np.argsort(-np.abs(w), kind="mergesort")[:k]
    return [{"name": names[i], "weight": float(w[i])} for i in order]


def abs_share_by_column(names: list[str], w: np.ndarray) -> dict:
    """Share of sum|w| per original column; a product splits its |w| equally among its source columns."""
    share: dict[str, float] = {}
    for name, weight in zip(names, w):
        origin = sources(name)
        for column in origin:
            share[column] = share.get(column, 0.0) + abs(float(weight)) / len(origin)
    total = sum(share.values())
    return {c: v / total for c, v in sorted(share.items(), key=lambda kv: -kv[1])}


# ----------------------------------------------------------------------------- step 7: label-rule foils

def foil_matrices(design: Design, names: list[str], train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """{'hour': (X_tr, X_va), 'trend': (X_tr, X_va)}: bias + the hour dummies / the trend column.

    They come from the surviving columns; a column that did not survive is taken from the BASE design."""
    base = Design(DesignSpec(BASE, (), 1, ())).fit(train_df)

    def build(wanted: list[str]) -> tuple:
        own = design if set(wanted) <= set(names) else base
        return design_matrix(own, wanted, train_df), design_matrix(own, wanted, val_df)

    hours = [n for n in names if HOUR_COLUMN.match(n)]
    return {"hour": build(hours or [n for n in base.names if HOUR_COLUMN.match(n)]), "trend": build(["trend"])}


def foil_fit(X_tr, y_tr, X_va, y_va, l2: float, max_iter: int, where: str, unconverged: list):
    """Validation probabilities of one foil model; a fit that runs out of iterations is recorded."""
    result = fit_logistic(X_tr, y_tr, l2=l2, lr=default_lr(X_tr, l2), max_iter=max_iter)
    if result.stop_reason != "converged":
        unconverged.append({"fit": where, "stop_reason": result.stop_reason, "iterations": result.iterations})
    return sigmoid(X_va @ result.weights)


def label_variant(group_by: list[str], p5cfg: dict, train_df, val_df, mats: dict, full: tuple, l2: float,
                  unconverged: list, p_full=None) -> dict:
    """Class balance and the three AUCs (hour only / trend only / full survivor model) under one label rule.

    `p_full` is the validation probability of the full model when it has already been fitted for this rule."""
    table, y_tr, y_va = label_rule(train_df, val_df, group_by, p5cfg["quantile"])
    name, year = "+".join(group_by) or "global", val_df["yr"].to_numpy()
    if p_full is None:
        p_full = foil_fit(full[0], y_tr, full[1], y_va, l2, p5cfg["max_iter"], f"{name} full", unconverged)
    p_hour = foil_fit(mats["hour"][0], y_tr, mats["hour"][1], y_va, l2, p5cfg["max_iter"], f"{name} hour",
                      unconverged)
    p_trend = foil_fit(mats["trend"][0], y_tr, mats["trend"][1], y_va, l2, p5cfg["max_iter"], f"{name} trend",
                       unconverged)
    return {"group_by": list(group_by), "pos_train": float(np.mean(y_tr)), "pos_val": float(np.mean(y_va)),
            "pos_val_2011": float(np.mean(y_va[year == 0])), "pos_val_2012": float(np.mean(y_va[year == 1])),
            "auc_hour_only": roc_auc(y_va, p_hour), "auc_trend_only": roc_auc(y_va, p_trend),
            "auc_full": roc_auc(y_va, p_full), "acc_full_at_0_5": prf(y_va, p_full, 0.5)["accuracy"],
            "majority_accuracy": majority_accuracy(y_va)}


# ----------------------------------------------------------------------------- step 8: retrospective

def target_text(p1: dict) -> str:
    """Name of the Phase 1 target, built from the artifact (transform family and exponent)."""
    return f"{p1['target_transform']}-family target (exponent {p1['target_power']:g})"


def retrospective(p1: dict, p2: dict, p3: dict, p4: dict, p5_row: dict) -> list[dict]:
    """One row per phase built from the stored artifacts: what it consumed, its settings and its scores."""
    est, est_anchor = p3["estimates_target"], p3["estimates"]
    methods = p4["methods"]
    rec = p4["recommended"]
    method_text = "; ".join(f"{m} lambda={methods[m]['lambda']:.3g} val_r2={methods[m]['val_r2']:.4f}"
                            for m in ("l2", "l1", "enet"))
    rows = [
        {"phase": "P1 gradient descent", "consumed": f"35 base columns + bias, {target_text(p1)}",
         "hyperparameters": f"lr={p1['lr']:.3f}, {p1['iterations']} iterations", "n_features": p1["n_features"],
         "train_score": p1["train_r2"], "val_score": p1["val_r2"], "val_rmse": p1["val_rmse"], "extra": ""},
        {"phase": "P2 polynomial", "consumed": "P1 weights as the starting point",
         "hyperparameters": f"degree={p2['degree']} on {','.join(p2['power_cols'])}, blocks={','.join(p2['blocks'])}",
         "n_features": len(p2["feature_names"]), "train_score": p2["train_r2"], "val_score": p2["val_r2"],
         "val_rmse": p2["val_rmse"], "extra": f"gain over P1 = {p2['gain_over_p1']:.4f} R2"},
        {"phase": "P3 bias-variance", "consumed": f"P2 design as anchor of a {len(p3['ladder'])}-level ladder",
         "hyperparameters": f"target {p3['target_level']} ({p3['target_complexity']['n_features']} weights)",
         "n_features": p3["target_complexity"]["n_features"], "train_score": est["seeded"]["train_r2"],
         "val_score": est["seeded"]["r2"], "val_rmse": est["seeded"]["rmse"],
         "extra": (f"target seeded/day-block/chrono = {est['seeded']['r2']:.3f}/{est['day_holdout']['r2']:.3f}/"
                   f"{est['chrono']['r2']:.3f}; anchor = {est_anchor['seeded']['r2']:.3f}/"
                   f"{est_anchor['day_holdout']['r2']:.3f}/{est_anchor['chrono']['r2']:.3f}")},
        {"phase": "P4 regularization", "consumed": "P3 target design + 6 candidate columns",
         "hyperparameters": f"{method_text}; recommended {rec['method']} lambda={rec['lambda']:.3g}",
         "n_features": len(p4["survivors_expanded"]) + 1, "train_score": None, "val_score": rec["val_r2"],
         "val_rmse": rec["val_rmse"], "extra": f"{len(p4['survivors_expanded'])} surviving columns from "
                                               f"{len(p4['survivors_original'])} original columns"},
        p5_row,
    ]
    return rows


# ----------------------------------------------------------------------------- run

def run(cfg: dict | None = None) -> dict:
    """Execute Phase 5 and write artifacts/p5.json."""
    cfg = cfg if cfg is not None else load_config()
    set_threads(cfg.get("threads", 1))
    TIMINGS.clear()
    p5cfg, seed = cfg["p5"], config_seed(cfg)
    p1, p2, p3, p4 = (read_artifact(n) for n in ("p1", "p2", "p3", "p4"))
    train_df, val_df = seeded_split(load_train(cfg), seed, cfg["split"]["test_size"])
    cost_ratio = p5cfg["cost_ratio_miss_to_false_alarm"]

    with timed("design and labels"):
        design, names = survivor_design(p4, train_df)
        X_tr, X_va = (design_matrix(design, names, d) for d in (train_df, val_df))
        table, y_tr, y_va = label_rule(train_df, val_df, p5cfg["group_by"], p5cfg["quantile"])
        balance = class_balance(y_tr, y_va, val_df)

    with timed("l2 sweep"):
        fits = [fit_candidate(X_tr, y_tr, X_va, y_va, l2, p5cfg["max_iter"]) for l2 in p5cfg["l2_candidates"]]
        rows = [row for _, row in fits]
        chosen = choose_l2(rows)
        result, l2 = fits[chosen][0], rows[chosen]["l2"]
        w = result.weights
        # at the optimum the gradient is ~1e-6, so central differences are noise-limited there; the halfway point
        # (0.5 w, gradient well away from zero) shows the formula itself is right
        grad_checks = {name: check_gradient(X_tr, y_tr, v, l2)
                       for name, v in (("at_zeros", np.zeros(len(w))), ("at_half", 0.5 * w))}

    with timed("threshold and metrics"):
        p_va, p_tr = sigmoid(X_va @ w), sigmoid(X_tr @ w)
        t_cost = 1.0 / (1.0 + cost_ratio)
        curve = threshold_curve(y_va, p_va, cost_ratio)
        t_f1, t_cost_empirical = best_threshold(curve, "f1", True), best_threshold(curve, "cost", False)
        metrics, at_half, at_f1 = (metrics_at(y_va, p_va, t, cost_ratio) for t in (t_cost, 0.5, t_f1))
        train_metrics = {k: v for k, v in metrics_at(y_tr, p_tr, t_cost, cost_ratio).items()
                         if k in ("accuracy", "f1", "roc_auc")}
        cal_table, cal_gap = calibration(y_va, p_va)
        boot = bootstrap_auc(y_va, p_va, seed, cfg["bootstrap"]["B"])

    weight_names = ["bias"] + names
    with timed("foils"):
        mats, unconverged = foil_matrices(design, names, train_df, val_df), []
        rules = [list(r) for r in FOIL_RULES] + [list(p5cfg["group_by"])]
        variants = [label_variant(r, p5cfg, train_df, val_df, mats, (X_tr, X_va), l2, unconverged,
                                  p_va if r == list(p5cfg["group_by"]) else None) for r in rules]

    p5_row = {"phase": "P5 logistic", "consumed": "P4 surviving columns, label = above the "
              f"{p5cfg['quantile'] * 100:g}th percentile of " + "+".join(p5cfg["group_by"]), "hyperparameters": f"l2={l2:g}, threshold {t_cost:.2f}, "
              f"{result.iterations} iterations", "n_features": X_tr.shape[1], "train_score": train_metrics["roc_auc"],
              "val_score": metrics["roc_auc"], "val_rmse": None,
              "extra": f"accuracy {metrics['accuracy']:.3f}, F1 {metrics['f1']:.3f}, ROC-AUC {metrics['roc_auc']:.3f}"}
    payload = {
        "features": names, "threshold_rule": describe_rule(table, p5cfg["group_by"], p5cfg["quantile"]),
        "class_balance": balance, "metrics": metrics, "metrics_at_0_5": at_half, "metrics_at_f1_opt": at_f1,
        "train_metrics": train_metrics, "val_auc_bootstrap": boot, "calibration": cal_table,
        "calibration_max_gap": cal_gap, "threshold_curve": curve, "curve_points": curve_points(y_va, p_va), "cost_ratio": cost_ratio, "t_cost": t_cost,
        "t_f1": t_f1, "t_cost_empirical": t_cost_empirical, "n_features": X_tr.shape[1], "l2": l2,
        "l2_sweep": rows, "iterations": result.iterations, "stop_reason": result.stop_reason,
        "lr": default_lr(X_tr, l2), "gradient_check": max(grad_checks["at_zeros"], grad_checks["at_half"]),
        "gradient_check_at_zeros": grad_checks["at_zeros"], "gradient_check_at_half": grad_checks["at_half"],
        "weights": w.tolist(), "weight_names": weight_names,
        "top_coefficients": top_coefficients(weight_names[1:], w[1:]),
        "coef_abs_share_by_column": abs_share_by_column(weight_names[1:], w[1:]),
        "label_variants": variants, "foil_not_converged": unconverged,
        "retrospective": retrospective(p1, p2, p3, p4, p5_row),
        "n_train": len(train_df), "n_val": len(val_df),
    }
    write_artifact("p5", payload, upstream="p4", cfg=cfg)
    return payload


if __name__ == "__main__":
    run()
