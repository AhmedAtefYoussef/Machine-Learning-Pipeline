#!/usr/bin/env python3
"""Validate sample_submission.csv against test.csv and the training demand profile. Exit 1 on hard failures.
Hard: columns == [instant,cnt]; rows/order == test.csv; no NaN/inf; no negatives; numeric.
Soft (WARN): per-(workingday,hour) mean prediction outside [0.4x, 2.5x] of the train-mean for that cell;
             total predicted demand per test day outside [0.4x, 2.5x] of the mean daily total of train days in the same month.
Usage: submission_check.py --sub sample_submission.csv --test data/test.csv --train data/train.csv
"""
import argparse, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--sub", default="sample_submission.csv")
ap.add_argument("--test", default="data/test.csv")
ap.add_argument("--train", default="data/train.csv")
a = ap.parse_args()
sub, te, tr = pd.read_csv(a.sub), pd.read_csv(a.test), pd.read_csv(a.train)
fails, warns = [], []
if list(sub.columns) != ["instant", "cnt"]:
    fails.append(f"columns {list(sub.columns)} != ['instant','cnt']")
else:
    if len(sub) != len(te):
        fails.append(f"rows {len(sub)} != {len(te)}")
    elif not (sub.instant.values == te.instant.values).all():
        fails.append("instant order differs from test.csv")
    if not np.isfinite(sub.cnt.astype(float)).all():
        fails.append("NaN/inf in cnt")
    elif (sub.cnt < 0).any():
        fails.append(f"{(sub.cnt<0).sum()} negative predictions")
    if (sub.cnt == 0).all():
        fails.append("all predictions are 0 (template not filled)")
if not fails:
    m = te.merge(sub, on="instant")
    prof = tr.groupby(["workingday", "hr"]).cnt.mean()
    got = m.groupby(["workingday", "hr"]).cnt.mean()
    r = (got / prof.reindex(got.index)).dropna()
    bad = r[(r < 0.4) | (r > 2.5)]
    if len(bad):
        warns.append(f"{len(bad)} (workingday,hour) cells far from train profile: {bad.round(2).to_dict()}")
    tr["month"] = pd.to_datetime(tr.dteday).dt.to_period("M")
    daily = tr.groupby(["month", "dteday"]).cnt.sum().groupby("month").mean()
    m["month"] = pd.to_datetime(m.dteday).dt.to_period("M")
    pt = m.groupby(["month", "dteday"]).cnt.sum().groupby("month").first()
    rr = (pt / daily.reindex(pt.index)).dropna()
    badd = rr[(rr < 0.4) | (rr > 2.5)]
    if len(badd):
        warns.append(f"{len(badd)} test days with implausible daily totals vs same-month train mean: {badd.round(2).to_dict()}")
for w in warns:
    print("WARN", w)
for f in fails:
    print("FAIL", f)
print(f"submission_check: {len(fails)} failures, {len(warns)} warnings")
sys.exit(1 if fails else 0)
