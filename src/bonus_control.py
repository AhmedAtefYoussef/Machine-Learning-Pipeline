"""Control for the asymmetric-cost bonus: how much of the upward shift comes from the asymmetry itself?

The Phase 1 model minimises squared error on log-bikes; the bonus model minimises a 3:1 weighted squared error
on bikes. Those two differ in two ways at once (the scale of the loss and the asymmetry). This control fits the
same bike-scale loss with k = 1 (no asymmetry), starting from the same Phase 1 weights, so the shift can be split
into "fitting on bikes" (k = 1 versus Phase 1) and "asymmetry" (k = 3 versus k = 1).

Side study only: reads artifacts/p1.json, writes artifacts/bonus_control.json, feeds no phase.
Run: python -m src.bonus_control
"""
from __future__ import annotations

import numpy as np

from src.common import from_target, load_config, r2, read_artifact, rmse, set_threads, write_artifact
from src.gd_asym import fit_asymmetric, operator_costs
from src.phases.p1 import base_design, make_split


def describe(cnt: np.ndarray, pred: np.ndarray, k: float) -> dict:
    """Operator costs (judged with the 3:1 ratio), R2, RMSE and mean prediction on the given rows."""
    return {**operator_costs(cnt, pred, k), "r2": r2(cnt, pred), "rmse": rmse(cnt, pred),
            "mean_pred": float(np.mean(pred))}


def run(cfg: dict | None = None) -> dict:
    """Fit the k = 1 control and store its validation numbers next to the shift decomposition."""
    cfg = cfg if cfg is not None else load_config()
    set_threads(cfg["threads"])
    p1 = read_artifact("p1")
    bonus_cfg = cfg["p1"]["bonus"]
    k = bonus_cfg["k_under"]
    split = make_split(cfg)
    _, X_tr, X_va = base_design(split)
    w_mse = np.array(p1["weights"], dtype=np.float64)

    res = fit_asymmetric(X_tr, split.cnt_tr, w_mse, k=1.0, max_iter=bonus_cfg["max_iter"],
                         tol_loss=bonus_cfg["tol_loss"])
    pred_k1 = from_target(X_va @ res.weights, 1.0)   # exp(Xw) - 1, clipped at 0, as in the Phase 1 bonus table
    mean_mse = p1["bonus"]["val"]["mse_model"]["mean_pred"]
    mean_k3 = p1["bonus"]["val"]["asym_model"]["mean_pred"]
    k1 = describe(split.cnt_va, pred_k1, k)
    payload = {
        "k_control": 1.0, "k_cost": k, "iterations": res.iterations, "stop_reason": res.stop_reason,
        "val": {"k1_model": k1},
        "shift_from_fitting_on_bikes": k1["mean_pred"] / mean_mse,
        "shift_from_asymmetry": mean_k3 / k1["mean_pred"],
        "total_shift": mean_k3 / mean_mse,
    }
    write_artifact("bonus_control", payload, upstream="p1", cfg=cfg)
    return read_artifact("bonus_control")


if __name__ == "__main__":
    out = run()
    print("bonus_control:", {key: (round(val, 4) if isinstance(val, float) else val)
                             for key, val in out.items() if key != "val"})
    print("k = 1 model on validation:", {key: round(val, 4) for key, val in out["val"]["k1_model"].items()})
