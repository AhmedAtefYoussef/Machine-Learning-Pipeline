"""EDA-label (chief, diagnostic only): compare Phase 5 label rules before the phase is specified.
sklearn LogisticRegression is used here as a quick probe (H13: exp/ only). Output exp/eda0/eda_label.json.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.common import load_config, load_train, seeded_split, team_seed
from src.features import BASE, Design, DesignSpec

cfg = load_config()
seed = team_seed(cfg["team_ids"])
tr, va = seeded_split(load_train(cfg), seed)
Q = 0.75


def label(fit_df, df, keys):
    if not keys:
        thr = fit_df.cnt.quantile(Q)
        return (df.cnt > thr).astype(int).values
    q = fit_df.groupby(keys).cnt.quantile(Q).rename("thr")
    return (df.cnt.values > df.join(q, on=keys).thr.values).astype(int)


def auc_of(cols_spec, ytr, yva, keep=None):
    d = Design(cols_spec).fit(tr)
    Xtr, Xva = d.transform(tr)[:, 1:], d.transform(va)[:, 1:]
    if keep is not None:
        idx = [i for i, n in enumerate(d.names[1:]) if keep(n)]
        Xtr, Xva = Xtr[:, idx], Xva[:, idx]
    m = LogisticRegression(C=100.0, max_iter=5000).fit(Xtr, ytr)
    p = m.predict_proba(Xva)[:, 1]
    return float(roc_auc_score(yva, p)), float(((p > 0.5) == yva).mean())


full = DesignSpec(base=BASE, power_cols=("temp", "hum"), degree=3, blocks=("wd_x_hr", "hr_x_temp", "hr_x_hum"))
base = DesignSpec(base=BASE)
out = {}
for name, keys in [("global", []), ("wd_hr", ["workingday", "hr"]), ("yr_wd_hr", ["yr", "workingday", "hr"])]:
    ytr, yva = label(tr, tr, keys), label(tr, va, keys)
    r = {"pos_train": float(ytr.mean()), "pos_val": float(yva.mean()),
         "pos_val_2011": float(yva[va.yr.values == 0].mean()), "pos_val_2012": float(yva[va.yr.values == 1].mean())}
    r["auc_hour_only"], _ = auc_of(base, ytr, yva, keep=lambda n: n.startswith("hr_"))
    r["auc_hour_wd"], _ = auc_of(DesignSpec(base=BASE, blocks=("wd_x_hr",)), ytr, yva, keep=lambda n: "hr_" in n or n == "workingday")
    r["auc_trend_only"], _ = auc_of(base, ytr, yva, keep=lambda n: n == "trend")
    r["auc_base"], r["acc_base"] = auc_of(base, ytr, yva)
    r["auc_full"], r["acc_full"] = auc_of(full, ytr, yva)
    r["majority_acc"] = float(max(yva.mean(), 1 - yva.mean()))
    if keys:
        r["min_cell_n"] = int(tr.groupby(keys).size().min())
    out[name] = r
    print(name, {k: round(v, 4) for k, v in r.items()})
json.dump(out, open("exp/eda0/eda_label.json", "w"), indent=1)
