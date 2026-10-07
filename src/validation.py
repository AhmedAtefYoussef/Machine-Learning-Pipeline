"""Closed-form fits and the three validators used by Phase 3 (ARCHITECTURE section 8).

Every fit here is ordinary least squares on z = log1p(cnt) with a Design learned on the FITTING frame only.
Scores are on the bike scale: cnt_hat = clip(exp(eta) * s - 1, 0, None), where the back-transform factor s is
recomputed from the fitting frame's own residuals (method name comes from Phase 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import (back_factor, chrono_split, day_block_folds, from_target, r2, rmse, to_target)
from src.features import Design, DesignSpec

ETA_HEADROOM = 1.0  # predictions on log scale are capped this far above the largest fitted target (guards exp overflow)


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


def fit_predict(spec: DesignSpec, fit_df: pd.DataFrame, eval_df: pd.DataFrame, back_method: str,
                alpha: float = 1e-8) -> tuple[dict, np.ndarray]:
    """Fit `spec` on fit_df; return (scores dict, bike-scale predictions for eval_df)."""
    design = Design(spec).fit(fit_df)
    X_fit, X_eval = design.transform(fit_df), design.transform(eval_df)
    z_fit = to_target(fit_df["cnt"].to_numpy())
    w = fit_linear(X_fit, z_fit, alpha)
    eta_fit = X_fit @ w
    # An over-flexible fit on very few rows can extrapolate to absurd eta; cap it so exp() stays finite.
    eta_eval = np.minimum(X_eval @ w, z_fit.max() + ETA_HEADROOM)
    s = back_factor(back_method, z_fit, eta_fit, fit_df["cnt"].to_numpy())
    cnt_fit, cnt_eval = fit_df["cnt"].to_numpy(dtype=np.float64), eval_df["cnt"].to_numpy(dtype=np.float64)
    pred_fit, pred_eval = from_target(eta_fit, s), from_target(eta_eval, s)
    scores = {"p": int(X_fit.shape[1]),
              "train_r2": r2(cnt_fit, pred_fit), "train_rmse": rmse(cnt_fit, pred_fit),
              "r2": r2(cnt_eval, pred_eval), "rmse": rmse(cnt_eval, pred_eval),
              "train_r2_log": r2(z_fit, eta_fit), "r2_log": r2(to_target(cnt_eval), eta_eval)}
    return scores, pred_eval


def evaluate_spec(spec: DesignSpec, fit_df: pd.DataFrame, eval_df: pd.DataFrame, back_method: str,
                  alpha: float = 1e-8) -> dict:
    """Scores of one spec fitted on fit_df and evaluated on eval_df (p, train_r2, train_rmse, r2, rmse, *_log)."""
    return fit_predict(spec, fit_df, eval_df, back_method, alpha)[0]


def day_block_scores(spec: DesignSpec, train_df: pd.DataFrame, back_method: str, k: int, alpha: float) -> dict:
    """k-fold CV where each fold is a set of whole days (rank of date mod k); folds live INSIDE train_df."""
    folds = day_block_folds(train_df, k)
    fold_scores = []
    for f in range(k):
        held = folds == f
        fold_scores.append(evaluate_spec(spec, train_df[~held], train_df[held], back_method, alpha))
    r2s = [s["r2"] for s in fold_scores]
    return {"r2": float(np.mean(r2s)), "sd": float(np.std(r2s, ddof=1)), "folds": r2s,
            "train_r2": float(np.mean([s["train_r2"] for s in fold_scores]))}


def chrono_scores(spec: DesignSpec, train_all: pd.DataFrame, cut_date: str, back_method: str, alpha: float) -> dict:
    """Fit on dates before cut_date, evaluate on dates from cut_date on."""
    early_df, late_df = chrono_split(train_all, cut_date)
    s = evaluate_spec(spec, early_df, late_df, back_method, alpha)
    return {"train_r2": s["train_r2"], "r2": s["r2"], "rmse": s["rmse"]}


def three_validators(spec: DesignSpec, train_df: pd.DataFrame, val_df: pd.DataFrame, train_all: pd.DataFrame,
                     cut_date: str, back_method: str, k: int = 5, alpha: float = 1e-8) -> dict:
    """The seeded split, day-block CV inside train_df, and the chronological split, each fitted on its own training part."""
    seeded = evaluate_spec(spec, train_df, val_df, back_method, alpha)
    return {"seeded": {key: seeded[key] for key in ("p", "train_r2", "train_rmse", "r2", "rmse")},
            "day_block": day_block_scores(spec, train_df, back_method, k, alpha),
            "chrono": chrono_scores(spec, train_all, cut_date, back_method, alpha)}


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
                   seed: int, back_method: str, alpha: float = 1e-8) -> list[dict]:
    """Train and validation R2 as the number of training DAYS grows (subsets are nested)."""
    rows = []
    for fraction, (n_days, subset) in zip(fractions, nested_day_subsets(train_df, fractions, seed)):
        s = evaluate_spec(spec, subset, val_df, back_method, alpha)
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
