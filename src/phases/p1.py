"""Phase 1: linear model for the power-transformed target z = ((cnt + 1)^power - 1) / power (power 0 = log1p), fitted with
our own gradient descent (ARCHITECTURE sections 1, 3, 6; ADR-017).

Pipeline: seeded split -> 36-column design -> learning rate from lambda_max -> lr sweep -> main GD run ->
oracle check (least squares) -> back-transform choice -> scores -> residual profile -> hour-encoding ablation
-> exponent table -> asymmetric-cost bonus -> artifacts/p1.json.

Only numpy, pandas and our own src modules are used here (no sklearn in this file).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.common import (Target, bootstrap_r2, load_config, r2, read_artifact, rmse, set_threads, valid_back_methods,
                        write_artifact)
from src.features import Design
from src.gd import gradient_check, lambda_max, mse_grad, mse_loss, run_gd
from src.gd_asym import asym_grad, asym_loss, fit_asymmetric, operator_costs
from src.validation import (Split, base_design, bike_predictions, config_target, curve_points, make_split,
                            model_scores, oracle_gap, residual_profile)

HOUR_HARMONICS = 6        # sin/cos pairs k = 1..6 for the harmonic hour encoding (6 pairs span all 24 hours)
HOURS_PER_DAY = 24
SWEEP_CURVE_POINTS = 200  # points kept per lr-sweep curve
MAIN_CURVE_POINTS = 300   # points kept of the main loss curve
CONSTANT_STD = 1e-12      # a column with a smaller train std is treated as constant (std set to 1)


# ----------------------------------------------------------------------------- gradient descent steps

def learning_rate(X: np.ndarray, fraction: float) -> tuple[float, float, float]:
    """(lambda_max, bound, lr): stable GD needs lr < bound = 2 / lambda_max(X'X/n); we use lr = fraction * bound."""
    lam = lambda_max(X)
    bound = 2.0 / lam
    return lam, bound, fraction * bound


def _finite_or_none(value: float):
    return float(value) if np.isfinite(value) else None


def lr_sweep(X, z, lam: float, cfg: dict) -> list:
    """One GD run from zeros per fraction of the stability bound; shows too-small, good and too-large learning rates."""
    p1 = cfg["p1"]
    rows = []
    for fraction in p1["sweep_fractions"]:
        lr = fraction * 2.0 / lam
        with np.errstate(all="ignore"):  # diverging runs overflow on purpose
            res = run_gd(X, z, np.zeros(X.shape[1]), lr, p1["tol_loss"], p1["tol_grad"], p1["sweep_max_iter"])
        rows.append({"fraction": fraction, "lr": lr, "stop_reason": res.stop_reason, "iterations": res.iterations,
                     "loss_final": _finite_or_none(res.loss_final),
                     "loss_curve": curve_points(res.loss_history, SWEEP_CURVE_POINTS)})
    return rows


def gradient_check_errors(X, z, seed: int) -> dict:
    """Finite-difference gradient error at zeros and at one random point.

    We do not check at the final weights: the gradient there is about 1e-6, so the relative error
    measures round-off, not the formula.
    """
    def error_at(w):
        return gradient_check(lambda v: mse_loss(X, z, v), lambda v: mse_grad(X, z, v), w)
    w_random = np.random.default_rng(seed).normal(size=X.shape[1])
    at_zeros, at_random = error_at(np.zeros(X.shape[1])), error_at(w_random)
    return {"max": max(at_zeros, at_random), "at_zeros": at_zeros, "at_random": at_random}


# ----------------------------------------------------------------------------- back-transform and scores

def backtransform_table(split: Split, X_tr, X_va, w, target: Target, methods) -> dict:
    """Validation score of every back-transform candidate valid for this exponent; the best val R2 wins
    (ties go to the first in the list)."""
    methods = valid_back_methods(target.power, methods)
    candidates = {}
    for method in methods:
        factor, _, pred_va = bike_predictions(split, X_tr, X_va, w, replace(target, back_method=method))
        candidates[method] = {"factor": factor, "val_r2": r2(split.cnt_va, pred_va),
                              "val_rmse": rmse(split.cnt_va, pred_va),
                              "val_mean_ratio": float(pred_va.mean() / split.cnt_va.mean())}
    best = methods[0]
    for method in methods:
        if candidates[method]["val_r2"] > candidates[best]["val_r2"]:
            best = method
    return {"method": best, "factor": candidates[best]["factor"], "candidates": candidates}


# ----------------------------------------------------------------------------- hour-encoding ablation

def _hour_block(df: pd.DataFrame, kind: str) -> np.ndarray:
    """Unscaled replacement for the 23 hour dummies: 'numeric' = hr, 'harmonics6' = sin/cos(2 pi k hr / 24)."""
    hr = df["hr"].to_numpy(dtype=np.float64)
    if kind == "numeric":
        return hr[:, None]
    angles = [2.0 * np.pi * k * hr / HOURS_PER_DAY for k in range(1, HOUR_HARMONICS + 1)]
    return np.column_stack([f(a) for a in angles for f in (np.sin, np.cos)])


def _standardise(block_tr: np.ndarray, block_va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise columns with train mean/std (ddof=0). A numerically constant column (sin(pi*hr) = 0) keeps std 1."""
    mean, std = block_tr.mean(axis=0), block_tr.std(axis=0)
    std = np.where(std < CONSTANT_STD, 1.0, std)
    return (block_tr - mean) / std, (block_va - mean) / std


def hour_ablation(split: Split, design: Design, X_tr, X_va, w_main, scores_main, cfg: dict, target: Target) -> dict:
    """Same pipeline with the hour dummies replaced by numeric hr or six harmonics; own GD from zeros, same lr rule."""
    p1 = cfg["p1"]
    keep = [j for j, name in enumerate(design.names) if not name.startswith("hr_")]
    out = {"one_hot": {"n_features": X_tr.shape[1], "val_r2": scores_main["val_r2"],
                       "iterations": None, "stop_reason": None}}
    for kind in ("numeric", "harmonics6"):
        block_tr, block_va = _standardise(_hour_block(split.train_df, kind), _hour_block(split.val_df, kind))
        A_tr, A_va = np.hstack([X_tr[:, keep], block_tr]), np.hstack([X_va[:, keep], block_va])
        _, _, lr = learning_rate(A_tr, p1["lr_fraction_of_bound"])
        res = run_gd(A_tr, split.z_tr, np.zeros(A_tr.shape[1]), lr, p1["tol_loss"], p1["tol_grad"], p1["max_iter"])
        scores = model_scores(split, A_tr, A_va, res.weights, target)
        out[kind] = {"n_features": A_tr.shape[1], "val_r2": scores["val_r2"], "iterations": res.iterations,
                     "stop_reason": res.stop_reason}
    return out


# ----------------------------------------------------------------------------- exponent table

TABLE_BACK_METHOD = "ls"   # every exponent in the table is scored with the least-squares factor


def target_power_table(split: Split, X_tr, X_va, cfg: dict) -> list:
    """The same base design fitted by our own GD from zeros under each candidate exponent (same lr, same stopping rule).

    Shows how the exponent changes the Phase 1 score; the exponent actually used comes from config.yaml (ADR-017)."""
    p1 = cfg["p1"]
    _, _, lr = learning_rate(X_tr, p1["lr_fraction_of_bound"])
    rows = []
    for power in p1["target_power_candidates"]:
        target = Target(float(power), TABLE_BACK_METHOD)
        z_split = replace(split, z_tr=target.forward(split.cnt_tr), z_va=target.forward(split.cnt_va))
        res = run_gd(X_tr, z_split.z_tr, np.zeros(X_tr.shape[1]), lr, p1["tol_loss"], p1["tol_grad"], p1["max_iter"])
        scores = model_scores(z_split, X_tr, X_va, res.weights, target)
        rows.append({"power": float(power), "iterations": res.iterations, "stop_reason": res.stop_reason,
                     "val_r2": scores["val_r2"], "val_rmse": scores["val_rmse"], "train_r2": scores["train_r2"]})
    return rows


# ----------------------------------------------------------------------------- bonus: asymmetric cost

def _bike_scale_scores(cnt, pred, k: float) -> dict:
    """Operator costs plus R2, RMSE and mean prediction of a bike-scale prediction vector."""
    return {**operator_costs(cnt, pred, k), "r2": r2(cnt, pred), "rmse": rmse(cnt, pred),
            "mean_pred": float(pred.mean())}


def _shift_by_cell(df: pd.DataFrame, pred_mse: np.ndarray, pred_asym: np.ndarray) -> list:
    """Mean MSE-model and asymmetric-model prediction per (workingday, hour) cell."""
    table = pd.DataFrame({"workingday": df["workingday"].to_numpy(), "hr": df["hr"].to_numpy(),
                          "mse": pred_mse, "asym": pred_asym})
    cells = table.groupby(["workingday", "hr"])[["mse", "asym"]].mean().reset_index()
    return [{"workingday": int(r.workingday), "hr": int(r.hr), "mean_pred_mse": float(r.mse),
             "mean_pred_asym": float(r.asym)} for r in cells.itertuples()]


def asymmetric_bonus(split: Split, X_tr, X_va, w_mse, target: Target, cfg: dict) -> dict:
    """Refit with an under-prediction penalty k on the bike scale, starting from the MSE weights."""
    bonus_cfg = cfg["p1"]["bonus"]
    k = bonus_cfg["k_under"]
    check = gradient_check(lambda w: asym_loss(X_tr, split.cnt_tr, w, target, k),
                           lambda w: asym_grad(X_tr, split.cnt_tr, w, target, k), w_mse)
    res = fit_asymmetric(X_tr, split.cnt_tr, w_mse, target, k=k, max_iter=bonus_cfg["max_iter"],
                         tol_loss=bonus_cfg["tol_loss"])
    pred_mse = target.to_bikes(X_va @ w_mse, 1.0)   # both models: q(Xw) - 1, no factor, clipped at 0
    pred_asym = target.to_bikes(X_va @ res.weights, 1.0)
    return {"k": k, "stop_reason": res.stop_reason, "iterations": res.iterations, "gradient_check": check,
            "grad_norm_final": res.grad_norm_final,
            "train_loss_at_mse_weights": asym_loss(X_tr, split.cnt_tr, w_mse, target, k),
            "train_loss_final": res.loss_final,
            "val": {"mse_model": _bike_scale_scores(split.cnt_va, pred_mse, k),
                    "asym_model": _bike_scale_scores(split.cnt_va, pred_asym, k)},
            "mean_shift_ratio": float(pred_asym.mean() / pred_mse.mean()),
            "by_wd_hr_shift": _shift_by_cell(split.val_df, pred_mse, pred_asym),
            "weights": res.weights}


# ----------------------------------------------------------------------------- orchestration

def run(cfg: dict | None = None, write: bool = True) -> dict:
    """Run Phase 1. With write=True store artifacts/p1.json and return it as read back from disk; with
    write=False touch no file and return the payload (the notebook's live mode)."""
    cfg = cfg if cfg is not None else load_config()
    set_threads(cfg["threads"])
    p1 = cfg["p1"]
    split = make_split(cfg)
    design, X_tr, X_va = base_design(split)

    lam, bound, lr = learning_rate(X_tr, p1["lr_fraction_of_bound"])
    condition = float(np.linalg.cond(X_tr.T @ X_tr / len(X_tr)))
    sweep = lr_sweep(X_tr, split.z_tr, lam, cfg)

    res = run_gd(X_tr, split.z_tr, np.zeros(X_tr.shape[1]), lr, p1["tol_loss"], p1["tol_grad"], p1["max_iter"])
    w = res.weights
    check = gradient_check_errors(X_tr, split.z_tr, split.seed)

    backtransform = backtransform_table(split, X_tr, X_va, w, config_target(cfg), p1["backtransform_candidates"])
    target = replace(config_target(cfg), back_method=backtransform["method"])
    scores = model_scores(split, X_tr, X_va, w, target)
    _, _, pred_va = bike_predictions(split, X_tr, X_va, w, target)
    lo, hi, se = bootstrap_r2(split.cnt_va, pred_va, split.seed, cfg["bootstrap"]["B"])

    power_table = target_power_table(split, X_tr, X_va, cfg)
    bonus = asymmetric_bonus(split, X_tr, X_va, w, target, cfg)
    bonus.pop("weights")  # the refit weights are not part of the artifact

    payload = {
        "weights": w, "feature_names": design.names, "scaler": design.scaler_dict(), "lr": lr,
        "iterations": res.iterations, "stop_reason": res.stop_reason,
        "train_loss_final": mse_loss(X_tr, split.z_tr, w),
        "val_r2": scores["val_r2"], "val_rmse": scores["val_rmse"], "target_transform": "power",
        "target_power": target.power,
        "design_spec": design.spec.to_dict(), "lambda_max": lam, "lr_bound": bound,
        "lr_fraction": p1["lr_fraction_of_bound"], "condition_number": condition,
        "tol_loss": p1["tol_loss"], "tol_grad": p1["tol_grad"], "max_iter": p1["max_iter"],
        "grad_norm_final": res.grad_norm_final, "gradient_check": check["max"],
        "gradient_check_at_zeros": check["at_zeros"], "gradient_check_at_random": check["at_random"],
        "loss_curve": curve_points(res.loss_history, MAIN_CURVE_POINTS), "lr_sweep": sweep,
        "oracle": oracle_gap(split, X_tr, X_va, w, target),
        "backtransform": backtransform,
        "train_r2": scores["train_r2"], "train_rmse": scores["train_rmse"],
        "train_r2_log": scores["train_r2_log"], "val_r2_log": scores["val_r2_log"],
        "val_bootstrap": {"lo": lo, "hi": hi, "se": se},
        "residual_profile": residual_profile(split.train_df, split.z_tr - X_tr @ w),
        "hour_encoding_ablation": hour_ablation(split, design, X_tr, X_va, w, scores, cfg, target),
        "target_power_table": power_table,
        "target_power_best_here": max(power_table, key=lambda row: row["val_r2"])["power"],
        "bonus": bonus,
        "n_train": len(split.train_df), "n_val": len(split.val_df), "n_features": X_tr.shape[1],
    }
    if not write:
        return payload
    write_artifact("p1", payload, upstream=None, cfg=cfg)
    return read_artifact("p1")


if __name__ == "__main__":
    run()
