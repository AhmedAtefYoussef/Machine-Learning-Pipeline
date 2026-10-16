"""Audit C probes (DIAGNOSTIC ONLY; nothing enters the chain or the submission; test.csv not opened).

On the Phase 3 target design (power target 0.1, closed-form ridge alpha 1e-4, exp.v2.lib):
 P1 out-of-fold residual structure on the 5 held-out-day folds: day-level share, by year-month, holiday, day of month.
 P2 test-mimicking validator: hold out day 19 of every month (the hidden test is day 20, next to day 19), and days 17-19.
 P3 lead weather (situation of the next hour, worst of the next 3 hours; inputs only, same lookup rule as the memory).
 P4 year-month levels re-scored on the day-19 validator (they gain on held-out days and lose on chrono, ADR-018).
Output: exp/audit_c/probe.json
"""
import hashlib
import json

import numpy as np
import pandas as pd

from exp.v2.lib import ALL, TRAIN, FOLDS, base_matrices, delta, evaluate, fit_predict, forward, standardise
from src.common import r2, read_artifact
from src.features import Design, DesignSpec

LAM = 0.1
SPEC = DesignSpec.from_dict(read_artifact("p3")["target_complexity"]["design_spec"])
CONFIG = {"lam": LAM, "spec": SPEC.to_dict(), "alpha": 1e-4, "probes": ["oof_resid", "dom19", "lead_ws", "ym_dom19"]}
SHA = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]
df = ALL
out = {"config_sha": SHA}

# ---- P1 out-of-fold residuals (transformed scale and bikes) on held-out day folds of the training portion
oof = pd.Series(np.nan, index=TRAIN.index)
for name, X, Xe, cnt, cnt_e, fi, ei in base_matrices(SPEC):
    if name.startswith("day"):
        pe, _ = fit_predict(X, Xe, cnt, LAM)
        oof.loc[ei] = pe
T = TRAIN.assign(pred=oof)
T["rz"] = forward(T["cnt"].to_numpy(float), LAM) - forward(T["pred"].to_numpy(float), LAM)
T["rb"] = T["cnt"] - T["pred"]
dm = T.groupby("dteday")["rz"].transform("mean")
out["oof"] = {
    "r2_bikes": r2(T["cnt"].to_numpy(float), T["pred"].to_numpy(float)),
    "day_level_share_resid_var_z": float(dm.var() / T["rz"].var()),
    "day_level_share_sq_err_bikes": float(((T.groupby("dteday")["rb"].transform("mean")) ** 2).sum() / (T["rb"] ** 2).sum()),
    "ym_mean_rz": {f"{y + 2011}-{m:02d}": round(float(v), 3) for (y, m), v in T.groupby(["yr", "mnth"])["rz"].mean().items()},
    "ym_mean_ratio_bikes": {f"{y + 2011}-{m:02d}": round(float(g["cnt"].sum() / g["pred"].sum()), 3) for (y, m), g in T.groupby(["yr", "mnth"])},
    "holiday_ratio_bikes_by_date": {str(d.date()): round(float(g["cnt"].sum() / g["pred"].sum()), 3) for d, g in T[T["holiday"] == 1].groupby("dteday")},
    "dom_ratio_bikes": {int(d): round(float(g["cnt"].sum() / g["pred"].sum()), 3) for d, g in T.groupby(T["dteday"].dt.day)},
    "worst_days_sq_err_share": [(str(d.date()), round(float(v), 4)) for d, v in
                                ((T["rb"] ** 2).groupby(T["dteday"]).sum() / (T["rb"] ** 2).sum()).sort_values(ascending=False).head(8).items()],
    "rush_wd_rows_share_sq_err": [float(((T["workingday"] == 1) & T["hr"].isin([7, 8, 17, 18, 19])).mean()),
                                  float((T["rb"][(T["workingday"] == 1) & T["hr"].isin([7, 8, 17, 18, 19])] ** 2).sum() / (T["rb"] ** 2).sum())],
}
# heteroscedasticity on the transformed scale: residual sd by predicted-count decile
dec = pd.qcut(T["pred"], 10, labels=False)
out["oof"]["rz_sd_by_pred_decile"] = T.groupby(dec)["rz"].std().round(3).tolist()
out["oof"]["rb_sd_by_pred_decile"] = T.groupby(dec)["rb"].std().round(1).tolist()

# ---- P2 test-mimic validators on the whole train file: fit days != D, evaluate D
dom = ALL["dteday"].dt.day


def score_split(mask_eval, extra=None):
    fit_df, ev_df = ALL[~mask_eval], ALL[mask_eval]
    d = Design(SPEC).fit(fit_df)
    X, Xe = d.transform(fit_df), d.transform(ev_df)
    if extra is not None:
        A, B = standardise(extra, fit_df.index, ev_df.index)
        X, Xe = np.hstack([X, A]), np.hstack([Xe, B])
    pe, _ = fit_predict(X, Xe, fit_df["cnt"].to_numpy(float), LAM)
    return r2(ev_df["cnt"].to_numpy(float), pe)


out["mimic"] = {"day19": score_split(dom == 19), "days17_19": score_split(dom >= 17), "day1": score_split(dom == 1),
                "day10": score_split(dom == 10)}

# ---- P3 lead weather; P4 year-month levels
ts = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
ws = np.minimum(df["weathersit"].to_numpy(), 3).astype(float)
s = pd.Series(ws, index=ts.to_numpy()); s = s[~s.index.duplicated()]


def ahead(h):
    v = s.reindex(ts.to_numpy() + np.timedelta64(h, "h")).to_numpy(float)
    return np.where(np.isnan(v), ws, v)


lead1 = ahead(1)
next3 = np.maximum.reduce([ahead(1), ahead(2), ahead(3)])
lead = pd.DataFrame({"l1_2": (lead1 == 2) * 1.0, "l1_3": (lead1 == 3) * 1.0, "n3_2": (next3 == 2) * 1.0, "n3_3": (next3 == 3) * 1.0}, index=df.index)
ym = df["yr"].to_numpy() * 12 + df["mnth"].to_numpy() - 1
ymf = pd.DataFrame(np.stack([(ym == k).astype(float) for k in range(1, 24)], axis=1), index=df.index, columns=[f"ym{k}" for k in range(1, 24)])
base = evaluate(SPEC, LAM)
out["base"] = {k: v for k, v in base.items() if k != "val_pred"}
for name, extra in (("lead_ws", lead), ("lead1_only", lead[["l1_2", "l1_3"]])):
    r = evaluate(SPEC, LAM, extra)
    out[name] = {"delta": delta(r, base), "day19": score_split(dom == 19, extra), **{k: v for k, v in r.items() if k != "val_pred"}}
out["ym_levels"] = {"day19": score_split(dom == 19, ymf), "days17_19": score_split(dom >= 17, ymf)}
out["lead_ws"]["days17_19"] = score_split(dom >= 17, lead)
json.dump(out, open("exp/audit_c/probe.json", "w"), indent=1, default=str)
print(SHA, "base days", round(base["days"], 4), "chrono", round(base["chrono"], 4))
print("lead", out["lead_ws"]["delta"], "| day19", round(out["lead_ws"]["day19"], 4), "vs", round(out["mimic"]["day19"], 4))
print("lead1", out["lead1_only"]["delta"])
print("mimic", {k: round(v, 4) for k, v in out["mimic"].items()}, "ym", {k: round(v, 4) for k, v in out["ym_levels"].items()})
