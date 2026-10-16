"""Copy the few diagnostic numbers the report quotes from exp/v2/*.json into artifacts/diagnostics.json, so the
report's number check can trace them. These are diagnostics (H13): no phase reads this file.
Usage: python -X utf8 -m exp.v2.export_diagnostics
"""
import json
import pathlib

import numpy as np
import pandas as pd

from exp.v2.lib import ALL, SEED, TRAIN, VAL, cfg
from src.common import Target, bootstrap_r2, read_artifact
from src.features import DesignSpec
from src.validation import fit_predict

E = pathlib.Path("exp/v2")
e2 = json.loads((E / "e2_features_0.1.json").read_text(encoding="utf-8"))
e3 = json.loads((E / "e3_ceiling.json").read_text(encoding="utf-8"))
e4 = json.loads((E / "e4_weights.json").read_text(encoding="utf-8"))
e5 = json.loads((E / "e5_classifier.json").read_text(encoding="utf-8"))
base = e2["base"]


def block(name: str) -> dict:
    r = e2["single"][name]
    return {"held_out_days_delta": r["days"] - base["days"], "chrono_delta": r["chrono"] - base["chrono"],
            "seeded_delta": r["seeded"] - base["seeded"]}


weather = [n for n in e2["single"] if n not in ("year_month levels", "trend hinges (quarterly)", "trend hinges (half-year)",
                                                "year_x_hr", "year_x_wd_x_daypart", "atemp gap")]
tree = e3["ceiling|HGB poisson 1500 x 0.03"]


def test_coverage() -> dict:
    """Which test days exist (dates only, no labels) and whether the day before each is a training day."""
    te = pd.read_csv(cfg["paths"]["test"], usecols=["dteday"], parse_dates=["dteday"])["dteday"].drop_duplicates()
    prev_in_train = (te - pd.Timedelta(days=1)).isin(ALL["dteday"])      # one flag per test day
    return {"test_days_of_month": sorted(te.dt.day.unique().tolist()), "n_test_days": int(len(te)),
            "n_test_days_with_previous_day_in_train": int(prev_in_train.sum()),
            "share_test_days_with_previous_day_in_train": round(float(prev_in_train.mean()), 4)}


def share_val_rows_sharing_a_day_with_train() -> float:
    """Fraction of seeded-validation rows whose calendar day also has rows in the training part (the split's optimism)."""
    return round(float(VAL["dteday"].isin(TRAIN["dteday"]).mean()), 4)


def recommended_val_predictions() -> np.ndarray:
    """Seeded-validation predictions of the Phase 3 target model (Phase 4's recommended model scores within 3e-5 R2)."""
    p3 = read_artifact("p3")
    target = Target(p3["target_power_used"], p3["back_method"])
    spec = DesignSpec.from_dict(p3["target_complexity"]["design_spec"])
    return fit_predict(spec, TRAIN, VAL, target, p3["fit_alpha"])[1]


def day_block_bootstrap_r2(y: np.ndarray, yhat: np.ndarray, days: np.ndarray, seed: int, B: int = 1000) -> np.ndarray:
    """R2 of B resamples of whole days (with replacement), exact from per-day sums, so no loop over rows."""
    _, g = np.unique(days, return_inverse=True)
    per_day = lambda v: np.bincount(g, weights=v)                           # noqa: E731  sum of v within each day
    n, sy, syy, sse = per_day(np.ones_like(y)), per_day(y), per_day(y * y), per_day((y - yhat) ** 2)
    pick = np.random.default_rng(seed).integers(0, len(n), size=(B, len(n)))
    N, SY, SYY, SSE = n[pick].sum(1), sy[pick].sum(1), syy[pick].sum(1), sse[pick].sum(1)
    return 1.0 - SSE / (SYY - SY ** 2 / N)


def interval_comparison() -> dict:
    """Row-bootstrap vs day-block-bootstrap 95% interval of the recommended model's seeded-validation R2."""
    y, yhat = VAL["cnt"].to_numpy(float), recommended_val_predictions()
    lo, hi, _ = bootstrap_r2(y, yhat, SEED)
    d_lo, d_hi = np.percentile(day_block_bootstrap_r2(y, yhat, VAL["dteday"].to_numpy(), SEED), [2.5, 97.5])
    return {"row_bootstrap_interval": [round(lo, 4), round(hi, 4)], "day_block_bootstrap_interval": [round(float(d_lo), 4), round(float(d_hi), 4)],
            "day_block_to_row_width_ratio": round(float((d_hi - d_lo) / (hi - lo)), 4)}


out = {
    "note": "Diagnostic numbers from exp/v2 (second exploration round). Not part of the chain; nothing reads this file.",
    "feature_blocks_tested": len(e2["single"]),
    "weather_blocks_max_abs_days_delta": max(abs(e2["single"][n]["days"] - base["days"]) for n in weather),
    "year_month_levels": block("year_month levels"),
    "quarterly_trend_hinges": block("trend hinges (quarterly)"),
    "observation_weights": {"held_out_days_delta": e4["0.1|1.0"]["days_delta"], "paired_se": e4["0.1|1.0"]["days_se"],
                            "seeded_delta": e4["0.1|1.0"]["seeded_delta"], "seeded_lo": e4["0.1|1.0"]["seeded_lo"],
                            "seeded_hi": e4["0.1|1.0"]["seeded_hi"]},
    "tree_model": {"seeded": tree["seeded"], "held_out_days": tree["days"], "chrono": tree["chrono"],
                   "seeded_minus_days": tree["seeded"] - tree["days"]},
    "tree_classifier_roc_auc": e5["HGB 600 x 0.05"]["roc_auc"],
    "test_coverage": test_coverage(),
    "share_val_rows_sharing_a_day_with_train": share_val_rows_sharing_a_day_with_train(),
    "val_r2_interval_row_vs_day_block": interval_comparison(),
}
pathlib.Path("artifacts/diagnostics.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
print(json.dumps(out, indent=1))
