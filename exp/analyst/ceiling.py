"""EXP-A2 diagnostic ceiling (H13: never enters the chain, the submission or a graded claim).

HistGradientBoostingRegressor on log1p(cnt) with the raw input columns, scored with the same three validators as
the Phase 3 target model (C5). No tuning on validation: two fixed configurations, both reported.
Output: exp/analyst/ceiling.json.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from exp.analyst.lib import fit_predict_x, paired_boot, save, scores_only, setup, three
from src.common import read_artifact

RAW = ["hr", "weekday", "workingday", "holiday", "weathersit", "temp", "atemp", "hum", "windspeed", "yr", "mnth",
       "season", "days"]
CONTEXT = ["ws_lag1", "hum_lag1", "wet3", "day_temp", "day_hum", "day_wet", "day_rain"]
CONFIG = {"exp": "A2_ceiling", "raw": RAW, "context": CONTEXT,
          "hgb_default": {"max_iter": 100, "learning_rate": 0.1},
          "hgb_long": {"max_iter": 600, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20},
          "target": "log1p(cnt), back-transform expm1, clip at 0", "early_stopping": False}


def hgb_fitter(columns, params, seed):
    def fitter(fit_df, eval_df):
        model = HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **params)
        model.fit(fit_df[columns].to_numpy(float), np.log1p(fit_df["cnt"].to_numpy(float)))
        return np.clip(np.expm1(model.predict(eval_df[columns].to_numpy(float))), 0.0, None)
    return fitter


def stacked_fitter(spec, back, columns, params, seed):
    """C5 linear model, then HGB on its log residuals (how much structure the linear design still leaves)."""
    def fitter(fit_df, eval_df):
        lin = fit_predict_x(spec, fit_df, eval_df, back)
        resid = np.log1p(fit_df["cnt"].to_numpy(float)) - lin["eta_fit"]
        model = HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **params)
        model.fit(fit_df[columns].to_numpy(float), resid)
        eta = lin["eta"] + model.predict(eval_df[columns].to_numpy(float))
        return np.clip(np.exp(eta) * lin["s"] - 1.0, 0.0, None)
    return fitter


def main() -> None:
    cfg, seed, all_df, train_df, val_df, spec, back, cut, k = setup()
    args = (spec, train_df, val_df, all_df, cut, back, k)
    runs = {"C5_linear": three(*args)}
    runs["hgb_default_raw"] = three(*args, fitter=hgb_fitter(RAW, CONFIG["hgb_default"], seed))
    runs["hgb_long_raw"] = three(*args, fitter=hgb_fitter(RAW, CONFIG["hgb_long"], seed))
    runs["hgb_long_raw_plus_context"] = three(*args, fitter=hgb_fitter(RAW + CONTEXT, CONFIG["hgb_long"], seed))
    runs["C5_plus_hgb_on_residual"] = three(*args, fitter=stacked_fitter(spec, back, RAW, CONFIG["hgb_default"], seed))
    y_val = val_df["cnt"].to_numpy(float)
    y_tr = train_df["cnt"].to_numpy(float)
    base = runs["C5_linear"]
    table = {}
    for name, res in runs.items():
        row = scores_only(res)
        if name != "C5_linear":
            row["seeded_minus_C5"] = paired_boot(y_val, res["seeded_pred"], base["seeded_pred"], seed)
            row["oof_minus_C5"] = paired_boot(y_tr, res["oof_pred"], base["oof_pred"], seed)
            row["fold_deltas"] = [a - b for a, b in zip(res["folds"], base["folds"])]
        table[name] = row
        print(f"{name:28s} seeded {res['seeded']:.4f} | held-out days {res['day_block']:.4f} (sd {res['day_block_sd']:.4f})"
              f" | chrono {res['chrono']:.4f} (mean ratio {res['chrono_mean_ratio']:.3f})"
              f" | leakage gap {res['seeded'] - res['day_block']:+.4f}")
    floor = read_artifact("p3")["noise_floor"]
    sha = save("ceiling", {"table": table, "noise_floor_p3": floor}, CONFIG)
    print("config_sha", sha, "| noise floor", {a: round(b, 4) for a, b in floor.items()})
    for name, row in table.items():
        if "oof_minus_C5" in row:
            print(f"  {name}: seeded delta {row['seeded_minus_C5']}, OOF delta {row['oof_minus_C5']}, "
                  f"fold deltas {[round(d, 4) for d in row['fold_deltas']]}")


if __name__ == "__main__":
    main()
