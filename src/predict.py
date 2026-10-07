"""Phase 6: final model and submission. The ONLY module that reads the hidden test frame.

The regression model recommended by Phase 4 (fitted on the surviving columns) is
  A  refitted on the training rows exactly as Phase 4 did: it must reproduce the stored weights (a self-check);
  B  refitted with the same hyper-parameters on training + validation rows: this one is submitted.
The design (means, standard deviations, humidity fill) is fitted on the training rows only and never refitted.

    cnt_hat = clip(exp(min(b + x . w, z_max + headroom)) * s - 1, 0)      s = back-transform factor of Phase 1's
                                                                         method, recomputed from the fitting rows.

Writes sample_submission.csv (repo root: columns instant, cnt in the order of the hidden file) and artifacts/p6.json.
Run it with `python -m src.predict` or `python run.py submission`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import (back_factor, config_seed, load_config, load_test, load_train, read_artifact,
                        seeded_split, set_threads, to_target, write_artifact)
from src.features import Design, DesignSpec
from src.phases.p4 import MAX_SWEEPS, SOLVER_TOL
from src.regularization import alpha_grid, alpha_max, enet_path, ridge_closed_form
from src.validation import ETA_HEADROOM

SUBMISSION_PATH = "sample_submission.csv"
REPRODUCTION_TOL = 1.0e-6     # model A must match the Phase 4 weights this closely
PRED_DECIMALS = 2
PROFILE_BAND = (0.4, 2.5)     # same plausibility band as tools/submission_check.py


# ----------------------------------------------------------------------------- the penalised fit

def fit_penalised(X: np.ndarray, z: np.ndarray, rec: dict, p4cfg: dict) -> tuple[float, np.ndarray]:
    """(intercept, weights) of the recommended method at its lambda / l1_ratio, solved as in Phase 4.

    Ridge is closed form. Lasso / elastic net run the warm-started path of Phase 4 down to lambda (the grid starts at
    alpha_max of THESE rows), so model A lands on exactly the stored solution."""
    alpha, rho = rec["lambda"], rec["l1_ratio"]
    if rho == 0.0:
        return ridge_closed_form(X, z, alpha)
    grid = alpha_grid(alpha_max(X, z, rho), p4cfg["l1_alpha_min_ratio"], p4cfg["n_alphas"])
    path = np.append(grid[grid > alpha], alpha)
    B, W, sweeps = enet_path(X, z, path, rho, SOLVER_TOL, MAX_SWEEPS)
    assert sweeps[-1] < MAX_SWEEPS, "coordinate descent did not converge at the final lambda"
    return float(B[-1]), W[-1]


def unclipped_bikes(b: float, w: np.ndarray, X_fit, cnt_fit, X_new, back_method: str) -> np.ndarray:
    """exp(eta) * s - 1 for the rows X_new (may be below 0); s comes from the fitting rows."""
    z_fit = to_target(cnt_fit)
    s = back_factor(back_method, z_fit, b + X_fit @ w, cnt_fit)
    eta_new = np.minimum(b + X_new @ w, z_fit.max() + ETA_HEADROOM)   # same cap as validation.fit_predict
    return np.exp(eta_new) * s - 1.0


# ----------------------------------------------------------------------------- sanity summaries

def clip_at_zero(raw: np.ndarray) -> np.ndarray:
    """Clip negative bike counts to 0."""
    return np.clip(raw, 0.0, None)


def pred_summary(n_clipped: int, pred: np.ndarray) -> dict:
    return {"mean": float(pred.mean()), "median": float(np.median(pred)), "min": float(pred.min()),
            "max": float(pred.max()), "n_below_1": int(np.sum(pred < 1.0)), "n_clipped": n_clipped}


def profile_ratio(test_df: pd.DataFrame, pred: np.ndarray, labelled_df: pd.DataFrame) -> dict:
    """Mean prediction / mean labelled cnt in each (workingday, hr) cell."""
    got = test_df.assign(pred=pred).groupby(["workingday", "hr"])["pred"].mean()
    ratio = (got / labelled_df.groupby(["workingday", "hr"])["cnt"].mean().reindex(got.index)).dropna()
    lo, hi = PROFILE_BAND
    return {"min": float(ratio.min()), "max": float(ratio.max()),
            "cells_outside_0_4_2_5": int(np.sum((ratio < lo) | (ratio > hi)))}


def daily_total_ratio(test_df: pd.DataFrame, pred: np.ndarray, labelled_df: pd.DataFrame) -> dict:
    """Predicted total of each test day / mean daily total of labelled days in the same year-month."""
    def month(d: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(d["dteday"]).dt.to_period("M")

    daily = labelled_df.groupby([month(labelled_df), "dteday"])["cnt"].sum().groupby(level=0).mean()
    predicted = test_df.assign(pred=pred).groupby([month(test_df), "dteday"])["pred"].sum().groupby(level=0).first()
    ratio = (predicted / daily.reindex(predicted.index)).dropna()
    return {"min": float(ratio.min()), "max": float(ratio.max())}


def agreement(pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """How far the validated model A and the submitted model B are apart on the hidden rows."""
    diff = np.abs(pred_a - pred_b)
    return {"corr": float(np.corrcoef(pred_a, pred_b)[0, 1]), "mean_ratio": float(pred_b.mean() / pred_a.mean()),
            "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}


# ----------------------------------------------------------------------------- build and run

def build(cfg: dict, p4: dict, p1: dict) -> tuple[pd.DataFrame, dict]:
    """(submission frame, p6 payload without the provenance fields)."""
    rec = p4["recommended"]
    assert rec["fitted_on"] == "survivors", "the recommended model must have been fitted on the surviving columns"
    labelled_df = load_train(cfg)
    train_df, val_df = seeded_split(labelled_df, config_seed(cfg), cfg["split"]["test_size"])
    test_df = load_test(cfg)

    design = Design(DesignSpec.from_dict(p4["design_spec"])).fit(train_df)   # training statistics only
    columns = design.subset(rec["feature_names"])
    X_tr, X_va, X_te = (design.transform(d)[:, columns] for d in (train_df, val_df, test_df))
    cnt_tr, cnt_va = (d["cnt"].to_numpy(dtype=np.float64) for d in (train_df, val_df))
    back_method = p1["backtransform"]["method"]

    b_a, w_a = fit_penalised(X_tr, to_target(cnt_tr), rec, cfg["p4"])
    gap = max(abs(b_a - rec["intercept"]), float(np.max(np.abs(w_a - np.asarray(rec["weights"])))))
    assert gap < REPRODUCTION_TOL, f"model A does not reproduce the Phase 4 weights (max gap {gap:.2e})"

    X_all, cnt_all = np.vstack([X_tr, X_va]), np.concatenate([cnt_tr, cnt_va])
    b_b, w_b = fit_penalised(X_all, to_target(cnt_all), rec, cfg["p4"])
    pred_a = clip_at_zero(unclipped_bikes(b_a, w_a, X_tr, cnt_tr, X_te, back_method))
    raw_b = unclipped_bikes(b_b, w_b, X_all, cnt_all, X_te, back_method)
    pred = np.round(clip_at_zero(raw_b), PRED_DECIMALS)

    sub = pd.DataFrame({"instant": test_df["instant"].to_numpy(), "cnt": pred})
    year = test_df["yr"].to_numpy()
    payload = {
        "model": {"method": rec["method"], "lambda": rec["lambda"], "l1_ratio": rec["l1_ratio"]},
        "refit_on": "train+validation", "n_fit_rows": len(cnt_all), "n_test_rows": len(test_df),
        "model_a_gap_to_p4": gap, "pred": pred_summary(int(np.sum(raw_b < 0.0)), pred),
        "train_cnt_mean": float(cnt_tr.mean()), "labelled_cnt_mean": float(cnt_all.mean()),
        "a_vs_b_on_test": agreement(pred_a, pred), "profile_ratio": profile_ratio(test_df, pred, labelled_df),
        "daily_total_ratio": daily_total_ratio(test_df, pred, labelled_df),
        "by_year_mean": {str(y): float(pred[year == y].mean()) for y in (0, 1)},
    }
    return sub, payload


def run(cfg: dict | None = None) -> dict:
    """Write sample_submission.csv and artifacts/p6.json; returns the p6 payload."""
    cfg = cfg if cfg is not None else load_config()
    set_threads(cfg.get("threads", 1))
    p4, p1 = read_artifact("p4"), read_artifact("p1")
    read_artifact("p5")   # must exist: p6 is chained to it
    sub, payload = build(cfg, p4, p1)
    sub.to_csv(SUBMISSION_PATH, index=False, lineterminator="\n", encoding="utf-8")
    write_artifact("p6", payload, upstream="p5", cfg=cfg)
    return payload


if __name__ == "__main__":
    run()
