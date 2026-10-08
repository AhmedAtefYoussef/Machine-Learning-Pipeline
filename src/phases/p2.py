"""Phase 2: polynomial and interaction expansion, continuing our own gradient descent from the Phase 1 weights.

Pipeline: load p1 -> expanded design (base columns keep the p1 scaler) -> lift p1 weights (new columns start at 0, so
the first loss equals the last p1 loss) -> degree sweep -> power-column sweep -> block ablation -> final model
-> oracle, bootstrap, paired comparison with p1, residual profile -> artifacts/p2.json.

Only numpy, pandas and our own src modules are used here (no sklearn in this file).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common import (Target, bootstrap_r2, load_config, paired_bootstrap_delta_r2, read_artifact,
                        set_threads, write_artifact)
from src.features import BASE, Design, DesignSpec
from src.gd import lambda_max, mse_loss
from src.phases.p1 import (Split, bike_predictions, curve_points, make_split, model_scores, oracle_gap,
                           residual_profile, run_gd)
from src.poly import lift_weights

CHAIN_TOL = 1e-9          # P2 initial loss must equal P1 final loss to this tolerance (ARCHITECTURE section 1)
MAIN_CURVE_POINTS = 300   # points kept of the final loss curve


@dataclass
class Fit:
    """One expanded-design GD run: summary row plus everything needed to reuse it."""
    row: dict            # JSON-ready summary (no weights), as stored in the sweep tables
    weights: np.ndarray
    design: Design
    X_tr: np.ndarray
    X_va: np.ndarray
    loss_history: list


def _spec(degree: int, power_cols, blocks) -> DesignSpec:
    return DesignSpec(base=BASE, power_cols=tuple(power_cols), degree=degree, blocks=tuple(blocks))


def fit_expanded(spec: DesignSpec, split: Split, p1: dict, cfg: dict) -> Fit:
    """Build the design with the p1 scaler on the base columns, lift p1 weights, run GD from there."""
    p2 = cfg["p2"]
    design = Design(spec).fit(split.train_df, base_scaler=p1["scaler"])
    assert design.names[: len(p1["feature_names"])] == p1["feature_names"], "base columns must come first, unchanged"
    X_tr, X_va = design.transform(split.train_df), design.transform(split.val_df)
    w0 = lift_weights(np.array(p1["weights"]), p1["feature_names"], design.names)
    init_loss = mse_loss(X_tr, split.z_tr, w0)
    assert abs(init_loss - p1["train_loss_final"]) <= CHAIN_TOL, \
        f"chain broken: init loss {init_loss!r} vs p1 final loss {p1['train_loss_final']!r}"
    lam = lambda_max(X_tr)
    lr = p2["lr_fraction_of_bound"] * 2.0 / lam
    res = run_gd(X_tr, split.z_tr, w0, lr, p2["tol_loss"], p2["tol_grad"], p2["max_iter"])
    scores = model_scores(split, X_tr, X_va, res.weights, Target.from_artifact(p1))
    row = {"spec": spec.to_dict(), "degree": spec.degree, "power_cols": list(spec.power_cols),
           "n_features": X_tr.shape[1], "lambda_max": lam, "lr": lr, "iterations": res.iterations,
           "stop_reason": res.stop_reason, "init_loss": init_loss, "train_loss_final": mse_loss(X_tr, split.z_tr, res.weights),
           "train_r2": scores["train_r2"], "train_rmse": scores["train_rmse"],
           "val_r2": scores["val_r2"], "val_rmse": scores["val_rmse"]}
    return Fit(row, res.weights, design, X_tr, X_va, res.loss_history)


def smallest_within_tol(items: list, tol: float, size) -> object:
    """The item with the smallest size whose val_r2 is within tol of the best val_r2 (first in the list on ties)."""
    best = max(item["val_r2"] for item in items)
    eligible = [item for item in items if item["val_r2"] >= best - tol]
    return min(eligible, key=size)


def degree_sweep(split, p1, cfg, blocks, widest_cols) -> list:
    """One run per candidate degree, with the widest power-column list."""
    return [fit_expanded(_spec(d, widest_cols if d > 1 else (), blocks), split, p1, cfg)
            for d in cfg["p2"]["degree_candidates"]]


def power_col_sweep(split, p1, cfg, blocks, degree) -> list:
    """One run per candidate power-column list at the chosen degree."""
    return [fit_expanded(_spec(degree, cols, blocks), split, p1, cfg) for cols in cfg["p2"]["power_col_candidates"]]


def run(cfg: dict | None = None) -> dict:
    """Run Phase 2, write artifacts/p2.json and return the artifact as read back from disk."""
    cfg = cfg if cfg is not None else load_config()
    set_threads(cfg["threads"])
    p2 = cfg["p2"]
    p1 = read_artifact("p1")
    target = Target.from_artifact(p1)
    split = make_split(cfg, target)
    blocks = tuple(p2["blocks"])
    widest = max(p2["power_col_candidates"], key=len)

    degree_fits = degree_sweep(split, p1, cfg, blocks, widest)
    degree_rows = [f.row for f in degree_fits]
    chosen_degree = smallest_within_tol(degree_rows, p2["plateau_tol"], lambda r: r["degree"])["degree"]

    if chosen_degree == 1:  # no powers to choose between
        col_fits = []
        chosen = next(f for f in degree_fits if f.row["degree"] == 1)
    else:
        col_fits = power_col_sweep(split, p1, cfg, blocks, chosen_degree)
        rows = [f.row for f in col_fits]
        pick = smallest_within_tol(rows, p2["plateau_tol"], lambda r: len(r["power_cols"]))
        chosen = col_fits[rows.index(pick)]
    spec, final = chosen.design.spec, chosen

    ablation = fit_expanded(_spec(spec.degree, spec.power_cols, [b for b in blocks if b != "wd_x_hr"]),
                            split, p1, cfg).row

    w, X_tr, X_va = final.weights, final.X_tr, final.X_va
    scores = model_scores(split, X_tr, X_va, w, target)
    _, _, pred_va = bike_predictions(split, X_tr, X_va, w, target)
    n_base = len(p1["feature_names"])
    _, _, pred_va_p1 = bike_predictions(split, X_tr[:, :n_base], X_va[:, :n_base], np.array(p1["weights"]), target)
    lo, hi, se = bootstrap_r2(split.cnt_va, pred_va, split.seed, cfg["bootstrap"]["B"])
    delta, d_lo, d_hi = paired_bootstrap_delta_r2(split.cnt_va, pred_va, pred_va_p1, split.seed, cfg["bootstrap"]["B"])
    names = final.design.names
    payload = {
        "init_loss": final.row["init_loss"], "init_weights_source": "p1", "degree": spec.degree,
        "feature_names": names, "weights": w, "val_r2": scores["val_r2"], "val_rmse": scores["val_rmse"],
        "design_spec": spec.to_dict(), "power_cols": list(spec.power_cols), "blocks": list(spec.blocks),
        "expanded_feature_names": [n for n in names if n not in set(p1["feature_names"])],
        "scaler": final.design.scaler_dict(), "lr": final.row["lr"], "lambda_max": final.row["lambda_max"],
        "condition_number": float(np.linalg.cond(X_tr.T @ X_tr / len(X_tr))),
        "iterations": final.row["iterations"], "stop_reason": final.row["stop_reason"],
        "train_loss_final": final.row["train_loss_final"],
        "grad_norm_final": float(np.linalg.norm(X_tr.T @ (X_tr @ w - split.z_tr) / len(split.z_tr))),
        "train_r2": scores["train_r2"], "train_rmse": scores["train_rmse"],
        "train_r2_log": scores["train_r2_log"], "val_r2_log": scores["val_r2_log"],
        "p1_val_r2": p1["val_r2"], "gain_over_p1": scores["val_r2"] - p1["val_r2"],
        "degree_sweep": degree_rows, "power_col_sweep": [f.row for f in col_fits],
        "block_ablation": {"without_wd_x_hr": {k: ablation[k] for k in ("n_features", "val_r2", "train_r2", "iterations")}},
        "plateau_tol": p2["plateau_tol"], "oracle": oracle_gap(split, X_tr, X_va, w, target),
        "loss_curve": curve_points(final.loss_history, MAIN_CURVE_POINTS),
        "val_bootstrap": {"lo": lo, "hi": hi, "se": se},
        "paired_vs_p1": {"delta": delta, "lo": d_lo, "hi": d_hi},
        "residual_profile": residual_profile(split.train_df, split.z_tr - X_tr @ w),
        "backtransform": {"method": target.back_method, "factor": scores["factor"]},
    }
    write_artifact("p2", payload, upstream="p1", cfg=cfg)
    return read_artifact("p2")


if __name__ == "__main__":
    run()
