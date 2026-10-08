"""E2 (diagnostic): which further input-only feature blocks help on top of the current target design?

Each candidate block is added alone to the Phase 3 target design (power target lam from argv, default 0.1) and scored
on the three validators; then a greedy forward pass adds blocks while held-out days improve by more than two paired
standard errors and the chronological score does not fall by more than 0.003.
All blocks use inputs only (calendar, weather, and the weather of neighbouring hours); none uses cnt.
Output: exp/v2/e2_features_<lam>.json
"""
import json
import sys

import numpy as np
import pandas as pd

from exp.v2.lib import ALL, delta, evaluate, row
from src.common import read_artifact
from src.features import DesignSpec

LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
TARGET = DesignSpec.from_dict(read_artifact("p3")["target_complexity"]["design_spec"])
df = ALL
ts = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
hr, wd = df["hr"].to_numpy(), df["workingday"].to_numpy(float)
temp, hum, wind = df["temp"].to_numpy(float), df["hum"].to_numpy(float), df["windspeed"].to_numpy(float)
ws = np.minimum(df["weathersit"].to_numpy(), 3)
days = (pd.to_datetime(df["dteday"]) - pd.Timestamp("2011-01-01")).dt.days.to_numpy(float)
doy = pd.to_datetime(df["dteday"]).dt.dayofyear.to_numpy(float)
H = np.stack([(hr == h).astype(float) for h in range(24)], axis=1)          # all 24 hour indicators


def lagged(values, hours):
    """Value `hours` earlier by timestamp; the row's own value when that hour is missing."""
    s = pd.Series(values, index=ts.to_numpy())
    s = s[~s.index.duplicated()]
    prev = s.reindex(ts.to_numpy() - np.timedelta64(hours, "h")).to_numpy(float)
    return np.where(np.isnan(prev), values, prev)


def frame(mat, prefix):
    mat = np.asarray(mat, float)
    mat = mat[:, None] if mat.ndim == 1 else mat
    return pd.DataFrame(mat, index=df.index, columns=[f"{prefix}{i}" for i in range(mat.shape[1])])


s1, c1 = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
ym = (df["yr"].to_numpy() * 12 + df["mnth"].to_numpy() - 1)
rain_recent = np.maximum.reduce([lagged((ws == 3).astype(float), k) for k in range(1, 7)])
dry_run = np.minimum.reduce([lagged((ws == 1).astype(float), k) for k in range(1, 4)])       # clear for the last 3 hours
knots = [91, 182, 273, 365, 456, 547, 638]
daypart = np.select([hr < 6, hr < 10, hr < 16, hr < 20], [0, 1, 2, 3], 4)
P = np.stack([(daypart == k).astype(float) for k in range(5)], axis=1)

BLOCKS = {
    "hr_x_doy (daylight)": frame(np.hstack([H * s1[:, None], H * c1[:, None]]), "hd"),
    "daypart_x_doy": frame(np.hstack([P * s1[:, None], P * c1[:, None], P * wd[:, None] * s1[:, None], P * wd[:, None] * c1[:, None]]), "pd"),
    "year_month levels": frame(np.stack([(ym == k).astype(float) for k in range(1, 24)], axis=1), "ym"),
    "trend hinges (quarterly)": frame(np.stack([np.maximum(days - k, 0) for k in knots], axis=1), "th"),
    "trend hinges (half-year)": frame(np.stack([np.maximum(days - k, 0) for k in (182, 365, 547)], axis=1), "t2"),
    "lag temp/hum + change": frame(np.stack([lagged(temp, 1), lagged(hum, 1), temp - lagged(temp, 3), hum - lagged(hum, 3)], axis=1), "lg"),
    "rain in last 6h, dry 3h": frame(np.stack([rain_recent, dry_run, rain_recent * wd], axis=1), "rr"),
    "wd_x_hr_x_hum": frame(H[:, 1:] * (wd * hum)[:, None], "whh"),
    "wd_x_hr_x_rain": frame(H * (wd * (ws == 3))[:, None], "whr"),
    "hr_x_rain": frame(H * (ws == 3)[:, None].astype(float), "hrr"),
    "hr_x_mist": frame(H * (ws == 2)[:, None].astype(float), "hrm"),
    "hr_x_wind": frame(H * wind[:, None], "hw"),
    "temp_x_hum, temp_x_season": frame(np.stack([temp * hum, temp * s1, temp * c1, hum * s1, hum * c1], axis=1), "tx"),
    "wd_x_temp2, wd_x_hum2": frame(np.stack([wd * temp ** 2, wd * hum ** 2, wd * temp * hum], axis=1), "wt"),
    "hr_x_temp2": frame(H * (temp ** 2)[:, None], "ht2"),
    "atemp gap": frame(np.stack([df["atemp"].to_numpy(float) - temp, (df["atemp"].to_numpy(float) - temp) ** 2], axis=1), "ag"),
    "year_x_hr": frame(H * df["yr"].to_numpy(float)[:, None], "yh"),
    "year_x_wd_x_daypart": frame(np.hstack([P * df["yr"].to_numpy(float)[:, None], P * (wd * df["yr"].to_numpy(float))[:, None]]), "ywp"),
    "holiday_x_daypart": frame(P * df["holiday"].to_numpy(float)[:, None], "hp"),
    "hr_x_wet3": frame(H * (np.maximum.reduce([lagged(ws.astype(float), k) for k in (1, 2, 3)]) == 3)[:, None].astype(float), "hw3"),
}

base = evaluate(TARGET, LAM)
print(row(f"target, lam={LAM}", base))
out = {"lam": LAM, "base": {k: v for k, v in base.items() if k != "val_pred"}, "single": {}, "greedy": []}
singles = {}
for name, block in BLOCKS.items():
    r = evaluate(TARGET, LAM, block)
    singles[name] = r
    out["single"][name] = {k: v for k, v in r.items() if k != "val_pred"}
    print(f"+ {name:28s} (+{block.shape[1]:3d})", delta(r, base))


def gain(a, b):
    d = np.array(a["days_folds"]) - np.array(b["days_folds"])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


print("--- greedy forward (accept: days gain > 2 se and chrono not below -0.003) ---")
chosen, current, cur_extra = [], base, None
remaining = dict(BLOCKS)
while remaining:
    best = None
    for name, block in remaining.items():
        extra = block if cur_extra is None else pd.concat([cur_extra, block], axis=1)
        r = evaluate(TARGET, LAM, extra)
        g, se = gain(r, current)
        if g > 2 * se and g > 0.0005 and r["chrono"] - current["chrono"] > -0.003 and (best is None or g > best[1]):
            best = (name, g, se, r, extra)
    if best is None:
        break
    name, g, se, current, cur_extra = best
    chosen.append(name); remaining.pop(name)
    out["greedy"].append({"added": name, "days_gain": g, "se": se, **{k: v for k, v in current.items() if k != "val_pred"}})
    print(row(f"+ {name}", current), f"| days +{g:.4f} (se {se:.4f})")
print("chosen:", chosen)
print("total vs base:", delta(current, base))
json.dump(out, open(f"exp/v2/e2_features_{LAM}.json", "w"), indent=1)
