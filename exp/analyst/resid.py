"""EXP-A1 residual analysis of the Phase 3 target model (C5), seeded validation rows and held-out days (OOF).

Diagnostic only (H13). Output: exp/analyst/resid.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from exp.analyst.lib import cluster_boot_r2, fit_predict_x, save, setup, three
from src.common import back_factor, bootstrap_r2, day_block_folds, from_target, r2, rmse, to_target

CONFIG = {"exp": "A1_residuals", "model": "p3 target (C5)", "alpha": 1e-8, "top_share": 0.05, "low_cnt": 5}


def table(frame: pd.DataFrame, key) -> list[dict]:
    """Per group: n, mean cnt, bias and RMSE on bikes, share of total squared error, bias and RMSE on log scale."""
    total = float((frame["e"] ** 2).sum())
    rows = []
    for name, g in frame.groupby(key, observed=True):
        rows.append({"group": name if not isinstance(name, tuple) else "|".join(str(x) for x in name),
                     "n": int(len(g)), "row_share": round(len(g) / len(frame), 4),
                     "mean_cnt": round(float(g["y"].mean()), 1),
                     "bias": round(float(g["e"].mean()), 2), "rmse": round(float(np.sqrt((g["e"] ** 2).mean())), 2),
                     "sse_share": round(float((g["e"] ** 2).sum()) / total, 4),
                     "log_bias": round(float(g["le"].mean()), 4),
                     "log_rmse": round(float(np.sqrt((g["le"] ** 2).mean())), 4)})
    return rows


def analyse(df: pd.DataFrame, pred: np.ndarray, eta: np.ndarray) -> dict:
    y = df["cnt"].to_numpy(float)
    f = pd.DataFrame({"y": y, "pred": pred, "e": pred - y, "le": eta - to_target(y), "hr": df["hr"].to_numpy(),
                      "daytype": df["daytype"].to_numpy(), "mnth": df["mnth"].to_numpy(), "yr": df["yr"].to_numpy(),
                      "ws": df["ws_c"].to_numpy().astype(int), "holiday": df["holiday"].to_numpy(),
                      "weekday": df["weekday"].to_numpy(), "date": df["dteday"].to_numpy()})
    top_cut = float(np.quantile(y, 1 - CONFIG["top_share"]))
    f["band"] = np.where(y <= CONFIG["low_cnt"], "cnt<=5", np.where(y >= top_cut, "top5%", "middle"))
    f["pred_decile"] = pd.qcut(f["pred"], 10, labels=False, duplicates="drop")
    hour_day = table(f, ["daytype", "hr"])
    out = {"n": int(len(f)), "r2": r2(y, pred), "rmse": rmse(y, pred), "r2_log": r2(to_target(y), eta),
           "mean_ratio": float(pred.mean() / y.mean()), "top_cut": top_cut,
           "by_hour_daytype_top10_sse": sorted(hour_day, key=lambda r: -r["sse_share"])[:10],
           "by_hour_daytype_top6_abs_bias": sorted(hour_day, key=lambda r: -abs(r["bias"]))[:6],
           "by_hour_daytype_top6_log_rmse": sorted(hour_day, key=lambda r: -r["log_rmse"])[:6],
           "by_daytype": table(f, "daytype"), "by_month": table(f, "mnth"), "by_year": table(f, "yr"),
           "by_weathersit": table(f, "ws"), "by_holiday": table(f, "holiday"), "by_weekday": table(f, "weekday"),
           "by_band": table(f, "band"), "by_pred_decile": table(f, "pred_decile")}
    rush = f["hr"].isin([7, 8, 9, 16, 17, 18, 19]) & (f["daytype"] == "work")
    night = f["hr"].isin([0, 1, 2, 3, 4, 5])
    out["rush_work"] = {"row_share": float(rush.mean()), "sse_share": float((f.loc[rush, "e"] ** 2).sum() / (f["e"] ** 2).sum())}
    out["night_0_5"] = {"row_share": float(night.mean()), "sse_share": float((f.loc[night, "e"] ** 2).sum() / (f["e"] ** 2).sum()),
                        "log_sse_share": float((f.loc[night, "le"] ** 2).sum() / (f["le"] ** 2).sum())}
    # heteroscedasticity: spread of the residual against the prediction
    out["hetero"] = {"corr_abs_e_pred": float(np.corrcoef(np.abs(f["e"]), f["pred"])[0, 1]),
                     "corr_abs_log_e_eta": float(np.corrcoef(np.abs(f["le"]), eta)[0, 1]),
                     "sd_e_low_decile": float(f.loc[f["pred_decile"] == 0, "e"].std()),
                     "sd_e_high_decile": float(f.loc[f["pred_decile"] == f["pred_decile"].max(), "e"].std()),
                     "sd_log_e_low_decile": float(f.loc[f["pred_decile"] == 0, "le"].std()),
                     "sd_log_e_high_decile": float(f.loc[f["pred_decile"] == f["pred_decile"].max(), "le"].std())}
    # day effect: how much of the residual is shared by the whole day (unobserved events, weather not in the columns)
    day_mean_log = f.groupby("date")["le"].transform("mean")
    day_mean = f.groupby("date")["e"].transform("mean")
    sizes = f.groupby("date")["e"].transform("size")
    multi = sizes >= 3
    out["day_effect"] = {"log_var_share_of_day_mean": float((day_mean_log[multi] ** 2).sum() / (f.loc[multi, "le"] ** 2).sum()),
                         "bike_var_share_of_day_mean": float((day_mean[multi] ** 2).sum() / (f.loc[multi, "e"] ** 2).sum()),
                         "rows_used": int(multi.sum()), "median_rows_per_day": float(sizes.groupby(f["date"]).first().median())}
    worst = f.assign(se=f["e"] ** 2).groupby("date").agg(sse=("se", "sum"), n=("e", "size"), bias=("e", "mean"),
                                                         hol=("holiday", "max"), ws=("ws", "mean"))
    worst["sse_share"] = worst["sse"] / worst["sse"].sum()
    top = worst.sort_values("sse_share", ascending=False).head(10)
    out["worst_days"] = [{"date": str(pd.Timestamp(d).date()), "n": int(r.n), "sse_share": round(float(r.sse_share), 4),
                          "bias": round(float(r.bias), 1), "holiday": int(r.hol), "mean_ws": round(float(r.ws), 2)}
                         for d, r in top.iterrows()]
    out["worst_days_top10_sse_share"] = float(top["sse_share"].sum())
    out["worst_days_top5pct_sse_share"] = float(worst["sse_share"].sort_values(ascending=False)
                                                .head(max(1, len(worst) // 20)).sum())
    return out


def backtransform_and_clip(spec, train_df, val_df) -> dict:
    """Validation R2 for the three back-transform factors, with and without the 0 clip and the eta cap."""
    base = fit_predict_x(spec, train_df, val_df, "none", cap=False)
    y_fit, y = train_df["cnt"].to_numpy(float), val_df["cnt"].to_numpy(float)
    out = {}
    for method in ("none", "ls", "duan"):
        s = back_factor(method, to_target(y_fit), base["eta_fit"], y_fit)
        unclipped = np.exp(base["eta"]) * s - 1.0
        pred = from_target(base["eta"], s)
        out[method] = {"factor": s, "val_r2": r2(y, pred), "val_r2_unclipped": r2(y, unclipped),
                       "mean_ratio": float(pred.mean() / y.mean()), "n_clipped_at_0": int(np.sum(unclipped < 0)),
                       "min_unclipped": float(unclipped.min())}
    capped = fit_predict_x(spec, train_df, val_df, "ls", cap=True)
    out["eta_cap"] = {"n_capped_val": capped["n_capped"], "max_eta_val": float(base["eta"].max()),
                      "cap_value": float(to_target(y_fit).max() + 1.0), "max_pred_val": float(capped["pred"].max()),
                      "max_cnt_train": float(y_fit.max())}
    # best possible single factor on the validation rows themselves (oracle, shows how much a factor can matter)
    e = np.exp(base["eta"])
    s_oracle = float(np.sum((y + 1.0) * e) / np.sum(e * e))
    out["oracle_factor_on_val"] = {"factor": s_oracle, "val_r2": r2(y, from_target(base["eta"], s_oracle))}
    return out


def main() -> None:
    cfg, seed, all_df, train_df, val_df, spec, back, cut, k = setup()
    res = three(spec, train_df, val_df, all_df, cut, back, k)
    full = fit_predict_x(spec, train_df, val_df, back)
    folds = day_block_folds(train_df, k)
    eta_oof = np.empty(len(train_df))
    capped = 0
    for f in range(k):
        held = folds == f
        r = fit_predict_x(spec, train_df[~held], train_df[held], back)
        eta_oof[held] = r["eta"] + np.log(r["s"])  # log of the back-transformed prediction + 1 (before the clip)
        capped += r["n_capped"]
    y_val = val_df["cnt"].to_numpy(float)
    row_lo, row_hi, _ = bootstrap_r2(y_val, res["seeded_pred"], seed, 1000)
    day_lo, day_hi = cluster_boot_r2(y_val, res["seeded_pred"], val_df["dteday"].to_numpy(), seed, 1000)
    oof_lo, oof_hi = cluster_boot_r2(train_df["cnt"].to_numpy(float), res["oof_pred"], train_df["dteday"].to_numpy(),
                                     seed, 1000)
    payload = {
        "scores": {"seeded": res["seeded"], "day_block": res["day_block"], "day_block_sd": res["day_block_sd"],
                   "folds": res["folds"], "chrono": res["chrono"], "chrono_mean_ratio": res["chrono_mean_ratio"],
                   "oof_pooled_r2": r2(train_df["cnt"].to_numpy(float), res["oof_pred"]), "n_features": full["p"]},
        "intervals": {"seeded_row_bootstrap": [row_lo, row_hi], "seeded_day_cluster_bootstrap": [day_lo, day_hi],
                      "oof_day_cluster_bootstrap": [oof_lo, oof_hi]},
        "seeded": analyse(val_df, res["seeded_pred"], full["eta"] + np.log(full["s"])),
        "held_out_days": analyse(train_df, res["oof_pred"], eta_oof),
        "backtransform_clip": backtransform_and_clip(spec, train_df, val_df),
        "oof_eta_capped": capped}
    sha = save("resid", payload, CONFIG)
    s, h = payload["seeded"], payload["held_out_days"]
    print("config_sha", sha, "| scores", {a: round(b, 4) for a, b in payload["scores"].items() if isinstance(b, float)})
    print("intervals", {a: [round(x, 4) for x in b] for a, b in payload["intervals"].items()})
    for label, part in (("seeded", s), ("held-out days", h)):
        print(f"--- {label}: r2 {part['r2']:.4f} rmse {part['rmse']:.2f} r2_log {part['r2_log']:.4f} mean ratio {part['mean_ratio']:.4f}")
        print(" rush_work", part["rush_work"], "night", part["night_0_5"])
        print(" hetero", {a: round(b, 3) for a, b in part["hetero"].items()})
        print(" day_effect", part["day_effect"])
        for key in ("by_daytype", "by_year", "by_weathersit", "by_holiday", "by_band", "by_month", "by_weekday",
                    "by_pred_decile", "by_hour_daytype_top10_sse", "by_hour_daytype_top6_abs_bias",
                    "by_hour_daytype_top6_log_rmse"):
            print(" ", key)
            for row in part[key]:
                print("    ", " ".join(f"{a}={b}" for a, b in row.items()))
        print("  worst days", part["worst_days_top10_sse_share"], part["worst_days_top5pct_sse_share"])
        for row in part["worst_days"]:
            print("    ", row)
    print("backtransform/clip", payload["backtransform_clip"], "oof capped", capped)


if __name__ == "__main__":
    main()
