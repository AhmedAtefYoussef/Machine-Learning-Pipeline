"""Audit C probe 3 (DIAGNOSTIC ONLY): interval width of R2 for a 24-day test-sized sample, row bootstrap vs day-cluster
bootstrap, on out-of-fold held-out-day predictions of the target design (no seeded-validation look).
Output: exp/audit_c/probe3.json
"""
import json

import numpy as np
import pandas as pd

from exp.v2.lib import ALL, TRAIN, base_matrices, fit_predict
from src.common import r2, read_artifact
from src.features import DesignSpec

SPEC = DesignSpec.from_dict(read_artifact("p3")["target_complexity"]["design_spec"])
oof = pd.Series(np.nan, index=TRAIN.index)
for name, X, Xe, cnt, cnt_e, fi, ei in base_matrices(SPEC):
    if name.startswith("day"):
        oof.loc[ei] = fit_predict(X, Xe, cnt, 0.1)[0]
y, p = TRAIN["cnt"].to_numpy(float), oof.to_numpy()
days = TRAIN["dteday"].to_numpy()
ud = np.unique(days)
idx_by_day = {d: np.flatnonzero(days == d) for d in ud}
rng = np.random.default_rng(44615)
B = 1000
row_r2, day_r2, test_like = [], [], []
n = len(y)
for _ in range(B):
    i = rng.integers(0, n, n); row_r2.append(r2(y[i], p[i]))
    dd = rng.choice(ud, len(ud)); j = np.concatenate([idx_by_day[d] for d in dd]); day_r2.append(r2(y[j], p[j]))
    dd = rng.choice(ud, 24, replace=False); j = np.concatenate([idx_by_day[d] for d in dd]); test_like.append(r2(y[j], p[j]))


def ci(v):
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


out = {"n_days_lt24_rows": int((ALL.groupby("dteday").size() < 24).sum()),
       "oof_r2": r2(y, p), "row_boot_ci": ci(row_r2), "day_boot_ci": ci(day_r2), "random_24_days_ci": ci(test_like),
       "random_24_days_sd": float(np.std(test_like, ddof=1))}
json.dump(out, open("exp/audit_c/probe3.json", "w"), indent=1)
print(out)
