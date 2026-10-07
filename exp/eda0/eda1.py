"""EDA1 (chief, diagnostic only): time representation, back-transform factor, complexity ladder preview.
lstsq/ridge(1e-6) oracle; log1p target; three validators. Output exp/eda0/eda1.json.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
seed = int(hashlib.sha256("_".join(sorted(cfg["team_ids"])).encode()).hexdigest(), 16) % 100000
tr_all = pd.read_csv("data/train.csv", parse_dates=["dteday"])
tr, va = train_test_split(tr_all, test_size=0.20, random_state=seed)
T0 = pd.Timestamp("2011-01-01")
CUT = pd.Timestamp("2012-07-01")


def raw_feats(df, ref):
    X = {}
    for h in range(1, 24):
        X[f"hr_{h}"] = (df.hr == h).astype(float)
    X["workingday"] = df.workingday.astype(float); X["holiday"] = df.holiday.astype(float)
    ws = df.weathersit.clip(upper=3)
    X["ws_2"] = (ws == 2).astype(float); X["ws_3"] = (ws == 3).astype(float)
    hum = df.hum.astype(float).copy(); hum[df.hum == 0] = ref.hum[ref.hum > 0].median()
    X["temp"] = df.temp; X["hum"] = hum; X["windspeed"] = df.windspeed
    X["trend"] = (df.dteday - T0).dt.days.astype(float); X["yr"] = df.yr.astype(float)
    doy = df.dteday.dt.dayofyear
    for k in (1, 2, 3):
        X[f"doy_s{k}"] = np.sin(2 * np.pi * k * doy / 365.25); X[f"doy_c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    for d in range(1, 7):
        X[f"wk_{d}"] = (df.weekday == d).astype(float)
    for m in range(2, 13):
        X[f"mn_{m}"] = (df.mnth == m).astype(float)
    return pd.DataFrame(X, index=df.index)


HR = [f"hr_{h}" for h in range(1, 24)]


def design(Z, spec):
    """Z = standardized raw feats (train stats). spec: dict of options. Returns DataFrame (unstandardized new columns)."""
    cols = list(HR) + ["workingday", "holiday", "ws_2", "ws_3", "temp", "hum", "windspeed"]
    cols += {"trend": ["trend"], "yr": ["yr"], "both": ["trend", "yr"]}[spec.get("time", "trend")]
    for k in range(1, spec.get("doy", 1) + 1):
        cols += [f"doy_s{k}", f"doy_c{k}"]
    if spec.get("nohour"):
        cols = [c for c in cols if c not in HR]
    D = {c: Z[c] for c in cols}
    for c in ("temp", "hum", "windspeed"):
        for p in range(2, spec.get("deg", 1) + 1):
            D[f"{c}^{p}"] = Z[c] ** p
    def inter(a_list, b_list):
        for a in a_list:
            for b in b_list:
                D[f"{a}*{b}"] = Z[a] * Z[b]
    blocks = spec.get("blocks", [])
    if "wdhr" in blocks: inter(["workingday"], HR)
    if "hr_temp" in blocks: inter(HR, ["temp"])
    if "hr_hum" in blocks: inter(HR, ["hum"])
    if "hr_ws" in blocks: inter(HR, ["ws_2", "ws_3"])
    if "hr_yr" in blocks: inter(HR, ["yr" if spec.get("time") == "yr" else "trend"])
    if "hr_doy" in blocks: inter(HR, ["doy_s1", "doy_c1"])
    if "wd_wx" in blocks: inter(["workingday"], ["temp", "hum", "ws_2", "ws_3"])
    if "wk" in blocks:
        for d in range(1, 7): D[f"wk_{d}"] = Z[f"wk_{d}"]
    if "wk_hr" in blocks: inter([f"wk_{d}" for d in range(1, 7)], HR)
    if "wdhr_temp" in blocks:
        for h in HR: D[f"wd*{h}*temp"] = Z["workingday"] * Z[h] * Z["temp"]
    if "mn_hr" in blocks: inter([f"mn_{m}" for m in range(2, 13)], HR)
    if "mn_wd_hr" in blocks:
        for m in range(2, 13):
            for h in HR: D[f"mn_{m}*wd*{h}"] = Z[f"mn_{m}"] * Z["workingday"] * Z[h]
    return pd.DataFrame(D, index=Z.index)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fit_eval(trn, val, spec, ridge=1e-6, back="none", ret=False):
    Rt, Rv = raw_feats(trn, trn), raw_feats(val, trn)
    mu, sd = Rt.mean(), Rt.std(ddof=0).replace(0, 1)
    Dt, Dv = design((Rt - mu) / sd, spec), design((Rv - mu) / sd, spec)
    m2, s2 = Dt.mean(), Dt.std(ddof=0).replace(0, 1)
    Xt, Xv = np.c_[np.ones(len(Dt)), ((Dt - m2) / s2).values], np.c_[np.ones(len(Dv)), ((Dv - m2) / s2).values]
    y = np.log1p(trn.cnt.values)
    A = Xt.T @ Xt / len(Xt); A[np.diag_indices_from(A)] += ridge; A[0, 0] -= ridge
    w = np.linalg.solve(A, Xt.T @ y / len(Xt))
    et, ev = Xt @ w, Xv @ w
    if back == "none": pt, pv = np.expm1(et), np.expm1(ev)
    elif back == "duan":
        s = np.mean(np.exp(y - et)); pt, pv = np.exp(et) * s - 1, np.exp(ev) * s - 1
    elif back == "ls":
        s = ((trn.cnt.values + 1) * np.exp(et)).sum() / (np.exp(2 * et)).sum(); pt, pv = np.exp(et) * s - 1, np.exp(ev) * s - 1
    elif back == "ratio":
        s = (trn.cnt.values + 1).sum() / np.exp(et).sum(); pt, pv = np.exp(et) * s - 1, np.exp(ev) * s - 1
    pt, pv = np.clip(pt, 0, None), np.clip(pv, 0, None)
    out = (float(r2(trn.cnt.values, pt)), float(r2(val.cnt.values, pv)), Xt.shape[1])
    if ret: return out + (pv, float(pv.mean() / val.cnt.mean()))
    return out


def day_folds(trn, k=5):
    days = np.sort(trn.dteday.unique()); m = {d: i % k for i, d in enumerate(days)}
    return trn.dteday.map(m).values


FO = day_folds(tr)


def three(spec, back="none"):
    a, b, p = fit_eval(tr, va, spec, back=back)
    cvs = [fit_eval(tr[FO != i], tr[FO == i], spec, back=back)[:2] for i in range(5)]
    c_tr, c_va, _ = fit_eval(tr_all[tr_all.dteday < CUT], tr_all[tr_all.dteday >= CUT], spec, back=back)
    return dict(p=p, train=round(a, 4), seeded=round(b, 4), cv_tr=round(float(np.mean([c[0] for c in cvs])), 4),
                daycv=round(float(np.mean([c[1] for c in cvs])), 4), chrono_tr=round(c_tr, 4), chrono=round(c_va, 4))


out = {}
print("== time representation (wdhr, deg3) ==")
for time in ("trend", "yr", "both"):
    for doy in (1, 2, 3):
        k = f"time={time},doy={doy}"; out[k] = three({"time": time, "doy": doy, "deg": 3, "blocks": ["wdhr"]}); print(k, out[k])
print("== P1-level time rep (additive) ==")
for time in ("trend", "yr", "both"):
    for doy in (1, 2):
        k = f"P1 time={time},doy={doy}"; out[k] = three({"time": time, "doy": doy}); print(k, out[k])
print("== back-transform (yr? trend, doy2, wdhr, deg3) ==")
for back in ("none", "duan", "ls", "ratio"):
    for time in ("trend", "yr"):
        sp = {"time": time, "doy": 2, "deg": 3, "blocks": ["wdhr"]}
        r = three(sp, back); mr = fit_eval(tr, va, sp, back=back, ret=True)[4]
        k = f"back={back},time={time}"; out[k] = dict(r, val_mean_ratio=round(mr, 4)); print(k, out[k])
print("== ladder ==")
for time in ("trend", "yr"):
    base = {"time": time, "doy": 2}
    ladder = [("L0 nohour", dict(base, nohour=True)), ("L1 additive", base), ("L2 +wdhr", dict(base, blocks=["wdhr"])),
              ("L3 +deg3", dict(base, deg=3, blocks=["wdhr"]))]
    acc = ["wdhr"]
    for b in ["hr_temp", "hr_hum", "hr_ws", "wd_wx", "hr_yr", "hr_doy", "wk", "wk_hr", "wdhr_temp", "mn_hr", "mn_wd_hr"]:
        acc = acc + [b]; ladder.append((f"L+{b}", dict(base, deg=3, blocks=list(acc))))
    for name, sp in ladder:
        k = f"{time} {name}"; out[k] = three(sp); print(k, out[k])
json.dump(out, open("exp/eda0/eda1.json", "w"), indent=1)
