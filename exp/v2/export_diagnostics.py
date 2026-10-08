"""Copy the few diagnostic numbers the report quotes from exp/v2/*.json into artifacts/diagnostics.json, so the
report's number check can trace them. These are diagnostics (H13): no phase reads this file.
Usage: python -X utf8 -m exp.v2.export_diagnostics
"""
import json
import pathlib

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
}
pathlib.Path("artifacts/diagnostics.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
print(json.dumps(out, indent=1))
