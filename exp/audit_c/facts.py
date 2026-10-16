"""Audit C (diagnostic, read-only): dataset facts for the team. Inputs: data/train.csv (labels) and data/test.csv
(inputs only: date coverage and feature ranges; no label exists and nothing is tuned on it).
Output: exp/audit_c/facts.json. Nothing here enters the chain or the submission.
"""
import json

import numpy as np
import pandas as pd

from src.common import config_seed, load_config, seeded_split

cfg = load_config()
tr = pd.read_csv(cfg["paths"]["train"], parse_dates=["dteday"])
te = pd.read_csv(cfg["paths"]["test"], parse_dates=["dteday"])
out: dict = {}


def ts(df):
    return df["dteday"] + pd.to_timedelta(df["hr"], unit="h")


# ---- A coverage
out["coverage"] = {
    "train_dom": sorted(tr["dteday"].dt.day.unique().tolist()),
    "test_dom": sorted(te["dteday"].dt.day.unique().tolist()),
    "train_range": [str(tr["dteday"].min().date()), str(tr["dteday"].max().date())],
    "test_range": [str(te["dteday"].min().date()), str(te["dteday"].max().date())],
    "train_days": int(tr["dteday"].nunique()), "test_days": int(te["dteday"].nunique()),
    "test_weekday_counts": te.groupby("dteday")["weekday"].first().value_counts().sort_index().to_dict(),
    "test_workingday_days": int(te.groupby("dteday")["workingday"].first().sum()),
    "test_day_overlap_train": int(te["dteday"].isin(tr["dteday"]).sum()),
    "test_prev_day_in_train": float((te["dteday"] - pd.Timedelta(days=1)).isin(tr["dteday"]).mean()),
    "test_next_day_in_train": float((te["dteday"] + pd.Timedelta(days=1)).isin(tr["dteday"]).mean()),
}

# ---- B missing hours
full = []
for d in tr["dteday"].unique():
    full += [(pd.Timestamp(d), h) for h in range(24)]
have = set(zip(tr["dteday"], tr["hr"]))
miss = pd.DataFrame([x for x in full if x not in have], columns=["dteday", "hr"])
per_day = miss.groupby("dteday").size().sort_values(ascending=False)
have_te = set(zip(te["dteday"], te["hr"]))
miss_te = [(str(pd.Timestamp(d).date()), h) for d in te["dteday"].unique() for h in range(24) if (pd.Timestamp(d), h) not in have_te]
# runs of consecutive missing hours (on the within-day timeline)
runs = []
for d, g in miss.groupby("dteday"):
    hs = sorted(g["hr"]); start = prev = hs[0]
    for h in hs[1:] + [None]:
        if h is not None and h == prev + 1:
            prev = h; continue
        runs.append((str(pd.Timestamp(d).date()), int(start), int(prev)))
        if h is not None:
            start = prev = h
long_runs = sorted(runs, key=lambda r: r[1] - r[2])[:8]
out["missing"] = {
    "train_missing_hours": int(len(miss)), "by_hour": miss["hr"].value_counts().sort_index().to_dict(),
    "share_missing_in_0_6h": float(miss["hr"].between(0, 6).mean()),
    "days_most_missing": {str(k.date()): int(v) for k, v in per_day.head(8).items()},
    "longest_runs": long_runs,
    "test_missing": miss_te,
}

# ---- C low counts and outage-like days
out["low"] = {"cnt_le_5_share": float((tr["cnt"] <= 5).mean()), "cnt_eq_1": int((tr["cnt"] == 1).sum()),
              "cnt_le_5_by_hour": tr[tr["cnt"] <= 5]["hr"].value_counts().sort_index().to_dict()}
day = tr.groupby("dteday").agg(n=("cnt", "size"), tot=("cnt", "sum"), wd=("workingday", "first"), ws=("weathersit", "max"),
                                hum=("hum", "mean"), temp=("temp", "mean"), yr=("yr", "first"), mn=("mnth", "first"))
day["ref"] = day.groupby(["yr", "mn", "wd"])["tot"].transform("median")
day["ratio"] = day["tot"] / day["ref"]
low_days = day.sort_values("ratio").head(10)
out["low"]["lowest_days_vs_month_daytype_median"] = [
    {"date": str(i.date()), "rows": int(r.n), "total": int(r.tot), "ratio": round(float(r.ratio), 3), "max_ws": int(r.ws),
     "mean_hum": round(float(r.hum), 2)} for i, r in low_days.iterrows()]
# do short days have low totals? (outage signal)
short = day[day["n"] < 22]
out["low"]["short_days(<22 rows)_median_ratio"] = float(short["ratio"].median()) if len(short) else None
out["low"]["n_short_days"] = int(len(short))
# hours next to a gap inside the day
tr_ts = ts(tr)
s = set(tr_ts)
adj = np.array([((t - pd.Timedelta(hours=1)) not in s and t.hour > 0) or ((t + pd.Timedelta(hours=1)) not in s and t.hour < 23)
                for t in tr_ts])
out["low"]["rows_adjacent_to_gap"] = int(adj.sum())

# ---- D quirks
w4 = tr[tr["weathersit"] == 4]
h0 = tr[tr["hum"] == 0]
out["quirks"] = {
    "weathersit4_rows": w4[["dteday", "hr", "cnt"]].astype(str).to_dict("records"),
    "hum0_date": sorted(h0["dteday"].dt.date.astype(str).unique().tolist()),
    "hum0_weathersit": h0["weathersit"].value_counts().to_dict(),
    "hum0_day_neighbours_hum": float(tr[tr["dteday"].isin(h0["dteday"] + pd.Timedelta(days=1)) |
                                        tr["dteday"].isin(h0["dteday"] - pd.Timedelta(days=1))]["hum"].mean()),
    "hum_fill_train_median_pos": float(tr.loc[tr["hum"] > 0, "hum"].median()),
    "hum_median_ws3": float(tr.loc[(tr["hum"] > 0) & (tr["weathersit"] == 3), "hum"].median()),
    "hum_min_positive": float(tr.loc[tr["hum"] > 0, "hum"].min()),
}
wz = tr["windspeed"] == 0
levels = np.sort(tr["windspeed"].unique())
out["quirks"]["wind_levels_lowest"] = levels[:4].round(4).tolist()
out["quirks"]["wind0_share_train_test"] = [float(wz.mean()), float((te["windspeed"] == 0).mean())]
out["quirks"]["wind0_share_by_hour_0to5_vs_12to17"] = [float(wz[tr["hr"].between(0, 5)].mean()), float(wz[tr["hr"].between(12, 17)].mean())]
# cnt at wind 0 vs the next level, inside hour x workingday cells (log ratio)
lo = tr[(tr["windspeed"] == 0) | (tr["windspeed"] == levels[1])].copy()
lo["z"] = np.log1p(lo["cnt"])
g = lo.groupby(["hr", "workingday", lo["windspeed"] == 0])["z"].mean().unstack()
out["quirks"]["wind0_minus_next_level_logcnt_mean_over_cells"] = float((g[True] - g[False]).mean())
# run lengths of wind 0
o = tr.sort_values(["dteday", "hr"])
r = (o["windspeed"] == 0).astype(int).to_numpy()
lens, c = [], 0
for v in r:
    if v: c += 1
    elif c: lens.append(c); c = 0
out["quirks"]["wind0_run_lengths"] = {"n_runs": len(lens), "median": float(np.median(lens)), "max": int(max(lens)),
                                      "share_rows_in_runs_ge3": float(sum(l for l in lens if l >= 3) / max(1, sum(lens)))}
a = tr[tr["dteday"] == "2012-08-17"]
out["quirks"]["atemp_2012_08_17"] = {"atemp_unique": a["atemp"].round(4).unique().tolist(), "temp_range": [float(a["temp"].min()), float(a["temp"].max())]}
fit = np.polyfit(tr.loc[tr["dteday"] != "2012-08-17", "temp"], tr.loc[tr["dteday"] != "2012-08-17", "atemp"], 1)
res_t = te["atemp"] - np.polyval(fit, te["temp"])
out["quirks"]["test_atemp_max_abs_resid_vs_temp"] = float(res_t.abs().max())
out["quirks"]["train_atemp_max_abs_resid_excl_0817"] = float((tr.loc[tr["dteday"] != "2012-08-17", "atemp"] -
                                                              np.polyval(fit, tr.loc[tr["dteday"] != "2012-08-17", "temp"])).abs().max())

# ---- E redundancy
both = pd.concat([tr, te])
wk_from_date = (both["dteday"].dt.dayofweek + 1) % 7  # Mon=1 ... Sun=0
out["redundancy"] = {
    "weekday_eq_date_weekday": float((wk_from_date == both["weekday"]).mean()),
    "yr_eq_date_year": float(((both["dteday"].dt.year - 2011) == both["yr"]).mean()),
    "mnth_eq_date_month": float((both["dteday"].dt.month == both["mnth"]).mean()),
    "season_eq_quarter": float((((both["mnth"] - 1) // 3 + 1) == both["season"]).mean()),
    "workingday_rule_exceptions": int(((both["weekday"].between(1, 5) & (both["holiday"] == 0)).astype(int) != both["workingday"]).sum()),
    "holiday_on_weekend": int(((both["holiday"] == 1) & ~both["weekday"].between(1, 5)).sum()),
    "train_holiday_dates": sorted(tr.loc[tr["holiday"] == 1, "dteday"].dt.date.astype(str).unique().tolist()),
    "test_holiday_dates": sorted(te.loc[te["holiday"] == 1, "dteday"].dt.date.astype(str).unique().tolist()),
    "instant_monotone_in_time": bool(both.sort_values("instant")[["dteday", "hr"]].apply(tuple, axis=1).is_monotonic_increasing),
}
num = tr[["temp", "atemp", "hum", "windspeed"]].copy()
num["doy_cos"] = np.cos(2 * np.pi * tr["dteday"].dt.dayofyear / 365.25)
num["trend"] = (tr["dteday"] - pd.Timestamp("2011-01-01")).dt.days
num["hr"] = tr["hr"]
C = num.corr()
out["collinearity"] = {"corr": C.round(3).to_dict()}
Xn = (num - num.mean()) / num.std()
inv = np.linalg.inv(np.corrcoef(Xn.to_numpy().T))
out["collinearity"]["vif"] = dict(zip(num.columns, np.diag(inv).round(2).tolist()))

# ---- F nonlinear: hr x workingday, temp curve after hour x day-type adjustment
z = np.log1p(tr["cnt"])
cell = tr.groupby(["hr", "workingday"])["cnt"].mean().unstack()
out["nonlinear"] = {"hr_profile_wd1": cell[1].round(0).astype(int).tolist(), "hr_profile_wd0": cell[0].round(0).astype(int).tolist(),
                    "peak_ratio_wd1_over_wd0_at_8h_17h_13h": [round(float(cell.loc[h, 1] / cell.loc[h, 0]), 2) for h in (8, 17, 13)]}
resid = z - z.groupby([tr["hr"], tr["workingday"], tr["yr"]]).transform("mean")
tb = pd.cut(tr["temp"], np.arange(0, 1.05, 0.1))
out["nonlinear"]["temp_bin_partial_logcnt"] = {str(k): [round(float(v), 3), int(n)] for (k, v), n in
                                               zip(resid.groupby(tb, observed=True).mean().items(), tb.value_counts(sort=False)[lambda x: x > 0])}
hb = pd.cut(tr["hum"], np.arange(0, 1.05, 0.1))
out["nonlinear"]["hum_bin_partial_logcnt"] = {str(k): round(float(v), 3) for k, v in resid.groupby(hb, observed=True).mean().items()}
# temp effect by hour group (does the temp slope differ by hour?)
slopes = {}
for name, hrs in (("night0-5", range(0, 6)), ("am_peak7-9", range(7, 10)), ("midday10-15", range(10, 16)), ("pm_peak16-19", range(16, 20)), ("eve20-23", range(20, 24))):
    m = tr["hr"].isin(hrs)
    slopes[name] = round(float(np.polyfit(tr.loc[m, "temp"], resid[m], 1)[0]), 3)
out["nonlinear"]["temp_slope_partial_logcnt_by_daypart"] = slopes

# ---- G target transform: mean-variance law in cells
cells = tr.groupby(["yr", "workingday", "hr", "season"])["cnt"].agg(["mean", "var", "size"])
cells = cells[cells["size"] >= 20]
b = float(np.polyfit(np.log(cells["mean"]), np.log(cells["var"]), 1)[0])
y = tr["cnt"].to_numpy(float)


def skew(v):
    v = np.asarray(v, float); return float(((v - v.mean()) ** 3).mean() / v.std() ** 3)


out["target"] = {"var_mean_slope_b": round(b, 3), "implied_power_1_minus_b_over_2": round(1 - b / 2, 3), "n_cells": int(len(cells)),
                 "skew": {"raw": skew(y), "sqrt": skew(np.sqrt(y + 1)), "pow0.1": skew(((y + 1) ** 0.1 - 1) / 0.1), "log1p": skew(np.log1p(y))},
                 "share_top10pct_rows_in_squared_dev": float(np.sort((y - y.mean()) ** 2)[-len(y) // 10:].sum() / ((y - y.mean()) ** 2).sum())}

# ---- H trend and growth
mm = tr.groupby(["yr", "mnth"])["cnt"].mean().unstack(0)
out["trend"] = {"monthly_mean_2011": mm[0].round(0).astype(int).tolist(), "monthly_mean_2012": mm[1].round(0).astype(int).tolist(),
                "ratio_2012_over_2011_by_month": (mm[1] / mm[0]).round(2).tolist()}
lm = tr.groupby(["yr", "mnth"])["cnt"].apply(lambda v: np.log(v).mean()).unstack(0)
out["trend"]["log_ratio_by_month"] = (lm[1] - lm[0]).round(3).tolist()

# ---- I random-split optimism
seed = config_seed(cfg)
T, V = seeded_split(tr, seed, cfg["split"]["test_size"])
tts = set(ts(T))
vts = ts(V)
out["optimism"] = {
    "val_rows_sharing_day_with_train": float(V["dteday"].isin(T["dteday"]).mean()),
    "val_rows_with_prev_or_next_hour_in_train": float(np.mean([(t - pd.Timedelta(hours=1)) in tts or (t + pd.Timedelta(hours=1)) in tts for t in vts])),
    "val_rows_with_both_neighbours_in_train": float(np.mean([(t - pd.Timedelta(hours=1)) in tts and (t + pd.Timedelta(hours=1)) in tts for t in vts])),
    "mean_train_rows_same_day_per_val_row": float(V["dteday"].map(T.groupby("dteday").size()).mean()),
}
# share of log-cnt variance at day level after removing hour x day type x year means (what day-sharing leaks)
r = resid.copy()
day_mean = r.groupby(tr["dteday"]).transform("mean")
out["optimism"]["day_level_share_of_residual_var"] = float(day_mean.var() / r.var())

# ---- J test feature ranges vs train
rng = {}
for c in ["temp", "atemp", "hum", "windspeed"]:
    rng[c] = {"train": [float(tr[c].min()), float(tr[c].max())], "test": [float(te[c].min()), float(te[c].max())]}
out["test_ranges"] = rng
out["test_ranges"]["weathersit"] = te["weathersit"].value_counts().sort_index().to_dict()
out["test_ranges"]["ws3_by_day"] = {str(k.date()): int(v) for k, v in te[te["weathersit"] == 3].groupby("dteday").size().items()}
out["test_ranges"]["holiday_rows"] = int(te["holiday"].sum())

json.dump(out, open("exp/audit_c/facts.json", "w"), indent=1, default=str)
print("ok", len(json.dumps(out, default=str)))
