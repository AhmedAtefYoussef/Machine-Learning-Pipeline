"""Phase 3: bias-variance. A nested complexity ladder around the Phase 2 design, a polynomial-degree axis,
learning curves, and three validators (seeded, held-out days, chronological). Writes artifacts/p3.json.

All fits are closed-form least squares (src/validation.py); the Phase 2 design is the anchor and the ladder level
labelled `phase2` in config.yaml IS that design.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common import (bootstrap_r2, chrono_split, config_seed, load_config, load_train,
                        paired_bootstrap_delta_r2, r2, read_artifact, rmse, seeded_split, set_threads,
                        write_artifact)
from src.features import BASE, HR, Design, DesignSpec
from src.validation import fit_predict, learning_curve, noise_floor, three_validators

ANCHOR_CHECK_TOL = 2e-3      # closed form here vs gradient descent in Phase 2
OVERFIT_MARGIN = 0.002       # a level counts as worse than the target when its validation R2 is this much lower


@dataclass
class Context:
    """Everything every fit in this phase shares."""
    all_df: pd.DataFrame      # all labelled rows (used only by the chronological validator)
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    cut_date: str
    k: int
    alpha: float
    back_method: str
    seed: int


# ----------------------------------------------------------------------------- set-up

def build_context(cfg: dict, p1: dict) -> Context:
    """Seeded split (the one split of the repo) plus the phase settings."""
    seed = config_seed(cfg)
    all_df = load_train(cfg)
    train_df, val_df = seeded_split(all_df, seed, cfg["split"]["test_size"])
    return Context(all_df, train_df, val_df, cfg["chrono"]["cut_date"], cfg["day_block_folds"],
                   cfg["p3"]["fit_alpha"], p1["backtransform"]["method"], seed)


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


# ----------------------------------------------------------------------------- scoring rows

def score_row(spec: DesignSpec, ctx: Context) -> dict:
    """The three validators on one spec, in the shape stored in p3.json."""
    v = three_validators(spec, ctx.train_df, ctx.val_df, ctx.all_df, ctx.cut_date, ctx.back_method, ctx.k, ctx.alpha)
    seeded = {key: v["seeded"][key] for key in ("train_r2", "train_rmse", "r2", "rmse")}
    return {"blocks": list(spec.blocks), "degree": spec.degree, "power_cols": list(spec.power_cols),
            "n_features": v["seeded"]["p"], "seeded": seeded, "day_block": v["day_block"], "chrono": v["chrono"],
            "gap_seeded": seeded["train_r2"] - seeded["r2"]}


def score_ladder(levels: list[dict], ctx: Context) -> list[dict]:
    """One row per ladder level."""
    return [{"name": lv["name"], "index": lv["index"], **score_row(lv["spec"], ctx)} for lv in levels]


def score_degree_axis(anchor_spec: DesignSpec, degrees: list[int], ctx: Context) -> list[dict]:
    """Anchor blocks and power columns, only the polynomial degree changes."""
    rows = []
    for degree in degrees:
        spec = DesignSpec(base=anchor_spec.base, power_cols=anchor_spec.power_cols, degree=degree,
                          blocks=anchor_spec.blocks)
        rows.append({"degree": degree, **{k: v for k, v in score_row(spec, ctx).items() if k != "degree"}})
    return rows


def check_anchor(rows: list[dict], anchor_index: int, p2: dict, anchor_spec: DesignSpec, ctx: Context) -> float:
    """Closed form must reproduce the Phase 2 gradient-descent validation R2 and feature list."""
    names = Design(anchor_spec).fit(ctx.train_df).names
    assert names == p2["feature_names"], "anchor feature names differ from p2.json"
    diff = rows[anchor_index]["seeded"]["r2"] - p2["val_r2"]
    assert abs(diff) <= ANCHOR_CHECK_TOL, f"anchor closed-form R2 differs from p2 by {diff}"
    return float(diff)


# ----------------------------------------------------------------------------- target level and diagnosis

def pick_target(rows: list[dict], plateau_tol: float) -> int:
    """Simplest ladder level whose seeded validation R2 is within `plateau_tol` of the best level (ADR-013)."""
    best_r2 = max(row["seeded"]["r2"] for row in rows)
    for i, row in enumerate(rows):
        if row["seeded"]["r2"] >= best_r2 - plateau_tol:
            return i
    return len(rows) - 1


def first_overfit_level(rows: list[dict], target: int) -> int | None:
    """First level above the target that is clearly worse on validation while fitting training data better."""
    t = rows[target]["seeded"]
    for row in rows[target + 1:]:
        if row["seeded"]["r2"] <= t["r2"] - OVERFIT_MARGIN and row["seeded"]["train_r2"] > t["train_r2"]:
            return row["index"]
    return None


def seeded_predictions(spec: DesignSpec, ctx: Context) -> np.ndarray:
    """Bike-scale predictions of `spec` on the seeded validation rows."""
    return fit_predict(spec, ctx.train_df, ctx.val_df, ctx.back_method, ctx.alpha)[1]


def chrono_detail(spec: DesignSpec, ctx: Context) -> dict:
    """What the chronological split does to one spec: scores, level drift, and a level-corrected diagnostic."""
    early_df, late_df = chrono_split(ctx.all_df, ctx.cut_date)
    _, pred = fit_predict(spec, early_df, late_df, ctx.back_method, ctx.alpha)
    cnt_late = late_df["cnt"].to_numpy(dtype=np.float64)
    corrected = pred * cnt_late.mean() / pred.mean()  # diagnostic only: uses the late period's own mean
    return {"cut_date": ctx.cut_date, "n_early": int(len(early_df)), "n_late": int(len(late_df)),
            "mean_cnt_early": float(early_df["cnt"].mean()), "mean_cnt_late": float(cnt_late.mean()),
            "r2": r2(cnt_late, pred), "rmse": rmse(cnt_late, pred), "mean_ratio": float(pred.mean() / cnt_late.mean()),
            "r2_level_corrected": r2(cnt_late, corrected)}


def classify(anchor: dict, target: dict, gain_lo: float, gain_hi: float, day_block_gain: float) -> str:
    """under-fit / over-fit / reasonably fit, by the rule fixed in the spec."""
    if target["index"] > anchor["index"] and gain_lo > 0 and day_block_gain > 0:
        return "under-fit"
    if target["index"] < anchor["index"] and gain_hi < 0 and day_block_gain < 0:
        return "over-fit"
    return "reasonably fit"


def estimates_block(row: dict, boot: tuple[float, float, float]) -> dict:
    """The three estimates of one design in the chain_check shape."""
    return {"seeded": {"r2": row["seeded"]["r2"], "rmse": row["seeded"]["rmse"],
                       "train_r2": row["seeded"]["train_r2"], "lo": boot[0], "hi": boot[1]},
            "day_holdout": {"r2": row["day_block"]["r2"], "sd": row["day_block"]["sd"],
                            "train_r2": row["day_block"]["train_r2"]},
            "chrono": dict(row["chrono"])}


def gaps_block(est: dict) -> dict:
    """leakage = seeded - day_holdout, drift = day_holdout - chrono."""
    return {"leakage": est["seeded"]["r2"] - est["day_holdout"]["r2"],
            "drift": est["day_holdout"]["r2"] - est["chrono"]["r2"]}


# ----------------------------------------------------------------------------- run

def run(cfg: dict | None = None) -> dict:
    """Execute Phase 3 and write artifacts/p3.json."""
    cfg = cfg if cfg is not None else load_config()
    set_threads(cfg.get("threads", 1))
    p1, p2 = read_artifact("p1"), read_artifact("p2")
    ctx = build_context(cfg, p1)
    anchor_spec = DesignSpec.from_dict(p2["design_spec"])
    levels = ladder_specs(cfg, anchor_spec)
    anchor_index = next(lv["index"] for lv in levels if lv["is_anchor"])

    rows = score_ladder(levels, ctx)
    anchor_diff = check_anchor(rows, anchor_index, p2, anchor_spec, ctx)
    degree_rows = score_degree_axis(anchor_spec, cfg["p3"]["degree_axis"], ctx)
    target_index = pick_target(rows, float(cfg["p3"]["plateau_tol"]))
    anchor_row, target_row = rows[anchor_index], rows[target_index]
    target_spec = levels[target_index]["spec"]

    y_val = ctx.val_df["cnt"].to_numpy(dtype=np.float64)
    pred_anchor, pred_target = seeded_predictions(anchor_spec, ctx), seeded_predictions(target_spec, ctx)
    gain, gain_lo, gain_hi = paired_bootstrap_delta_r2(y_val, pred_target, pred_anchor, ctx.seed, cfg["bootstrap"]["B"])
    boot_anchor = bootstrap_r2(y_val, pred_anchor, ctx.seed, cfg["bootstrap"]["B"])
    boot_target = bootstrap_r2(y_val, pred_target, ctx.seed, cfg["bootstrap"]["B"])

    curve_specs = {"anchor": anchor_spec, "target": target_spec, "top": levels[-1]["spec"]}
    curves = {key: learning_curve(spec, ctx.train_df, ctx.val_df, cfg["p3"]["learning_fractions"], ctx.seed,
                                  ctx.back_method, ctx.alpha) for key, spec in curve_specs.items()}

    est_anchor, est_target = estimates_block(anchor_row, boot_anchor), estimates_block(target_row, boot_target)
    day_block_gain = target_row["day_block"]["r2"] - anchor_row["day_block"]["r2"]
    degree_r2 = [r["seeded"]["r2"] for r in degree_rows if r["degree"] >= 2]
    diagnosis = {
        "label": classify(anchor_row, target_row, gain_lo, gain_hi, day_block_gain),
        "anchor_gap_train_val": anchor_row["gap_seeded"],
        "gain_target_over_anchor": gain, "gain_ci_lo": gain_lo, "gain_ci_hi": gain_hi,
        "day_block_gain": day_block_gain,
        "chrono_gain": target_row["chrono"]["r2"] - anchor_row["chrono"]["r2"],
        "degree_axis_range": float(max(degree_r2) - min(degree_r2)) if degree_r2 else 0.0,
        "overfit_from_level": first_overfit_level(rows, target_index)}

    payload = {
        "ladder": rows, "degree_axis": degree_rows, "anchor_check": anchor_diff,
        "target_level": target_row["name"], "target_index": target_index,
        "best_level_day_block": max(rows, key=lambda r: r["day_block"]["r2"])["name"],
        "best_level_chrono": max(rows, key=lambda r: r["chrono"]["r2"])["name"],
        "overfit_from_level": diagnosis["overfit_from_level"],
        "paired_target_vs_anchor": {"delta": gain, "lo": gain_lo, "hi": gain_hi},
        "val_bootstrap_anchor": dict(zip(("lo", "hi", "se"), boot_anchor)),
        "val_bootstrap_target": dict(zip(("lo", "hi", "se"), boot_target)),
        "learning_curves": curves,
        "chrono_detail": {"anchor": chrono_detail(anchor_spec, ctx), "target": chrono_detail(target_spec, ctx)},
        "noise_floor": noise_floor(ctx.train_df),
        "estimates": est_anchor, "estimates_target": est_target,
        "gaps": {"anchor": gaps_block(est_anchor), "target": gaps_block(est_target)},
        "diagnosis": diagnosis,
        "target_complexity": {"level": target_row["name"], "index": target_index,
                              "n_features": target_row["n_features"], "degree": target_spec.degree,
                              "power_cols": list(target_spec.power_cols), "blocks": list(target_spec.blocks),
                              "design_spec": target_spec.to_dict()},
        "anchor": {"degree": anchor_spec.degree, "power_cols": list(anchor_spec.power_cols),
                   "blocks": list(anchor_spec.blocks), "n_features": anchor_row["n_features"],
                   "anchor_index": anchor_index,
                   "feature_names_sha256": hashlib.sha256("\n".join(p2["feature_names"]).encode()).hexdigest(),
                   "p2_val_r2": p2["val_r2"]},
        "cut_date": ctx.cut_date, "k_folds": ctx.k, "fit_alpha": ctx.alpha, "back_method": ctx.back_method,
        "plateau_tol": float(cfg["p3"]["plateau_tol"])}
    write_artifact("p3", payload, upstream="p2", cfg=cfg)
    return payload


if __name__ == "__main__":
    run()
