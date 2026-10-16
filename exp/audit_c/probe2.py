"""Audit C probe 2 (DIAGNOSTIC ONLY; nothing enters the chain or the submission; test.csv not opened).

The hidden test is day 20 of each of the 24 observed months, so its time structure is interpolation inside known
months. Out-of-fold residuals of the target design show year-month level errors (2011 Jan-Apr over-predicted by
11-18 %, 2012 Feb-Apr under by 10-13 %): the linear trend + annual harmonics forces one year-on-year ratio, the data has
2.0-2.4x growth in Jan-Apr and 1.3-1.7x afterwards. Candidates scored on 19 day-of-month hold-outs (fit on the other 18
days of every month, evaluate one day of every month: the closest copy of the hidden test), the 5 held-out-day folds
and the chronological split:
  ym      23 year-month levels
  yr_doy  year x the four day-of-year harmonics (4 columns; the season shape may differ between the years)
Output: exp/audit_c/probe2.json
"""
import hashlib
import json

import numpy as np
import pandas as pd

from exp.v2.lib import ALL, delta, evaluate, fit_predict, standardise
from src.common import r2, read_artifact
from src.features import Design, DesignSpec

LAM = 0.1
SPEC = DesignSpec.from_dict(read_artifact("p3")["target_complexity"]["design_spec"])
CONFIG = {"lam": LAM, "spec": SPEC.to_dict(), "alpha": 1e-4, "cands": ["ym", "yr_doy"], "validator": "dom1..19 + days + chrono"}
SHA = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]
df = ALL
dom = df["dteday"].dt.day.to_numpy()
doy = df["dteday"].dt.dayofyear.to_numpy(float)
yr = df["yr"].to_numpy(float)
ym = df["yr"].to_numpy() * 12 + df["mnth"].to_numpy() - 1
H = np.stack([np.sin(2 * np.pi * k * doy / 365.25) if t == "s" else np.cos(2 * np.pi * k * doy / 365.25)
              for k in (1, 2) for t in ("s", "c")], axis=1)
CANDS = {
    "ym": pd.DataFrame(np.stack([(ym == k).astype(float) for k in range(1, 24)], axis=1), index=df.index),
    "yr_doy": pd.DataFrame(H * yr[:, None], index=df.index),
}
cache = {}


def dom_scores(extra):
    res = []
    for D in range(1, 20):
        m = dom == D
        fit_df, ev_df = df[~m], df[m]
        if D not in cache:
            d = Design(SPEC).fit(fit_df)
            cache[D] = (d.transform(fit_df), d.transform(ev_df))
        X, Xe = cache[D]
        if extra is not None:
            A, B = standardise(extra, fit_df.index, ev_df.index)
            X, Xe = np.hstack([X, A]), np.hstack([Xe, B])
        pe, _ = fit_predict(X, Xe, fit_df["cnt"].to_numpy(float), LAM)
        res.append(r2(ev_df["cnt"].to_numpy(float), pe))
    return np.array(res)


base_dom = dom_scores(None)
base = evaluate(SPEC, LAM)
out = {"config_sha": SHA, "base_dom": base_dom.round(4).tolist(), "base_dom_mean": float(base_dom.mean()),
       "base_dom_sd": float(base_dom.std(ddof=1)), "base_dom_range": [float(base_dom.min()), float(base_dom.max())]}
print(SHA, "base dom mean %.4f sd %.4f range %.4f-%.4f" % (base_dom.mean(), base_dom.std(ddof=1), base_dom.min(), base_dom.max()))
for name, extra in CANDS.items():
    s = dom_scores(extra)
    d = s - base_dom
    r = evaluate(SPEC, LAM, extra)
    out[name] = {"dom_delta_mean": float(d.mean()), "dom_delta_se": float(d.std(ddof=1) / np.sqrt(len(d))), "dom_wins": int((d > 0).sum()),
                 "days_chrono": delta(r, base), "days": r["days"], "chrono": r["chrono"]}
    print(f"{name:7s} dom {d.mean():+.4f} (se {d.std(ddof=1) / np.sqrt(len(d)):.4f}, {(d > 0).sum()}/19) | {delta(r, base)}")
json.dump(out, open("exp/audit_c/probe2.json", "w"), indent=1)
