"""E3 (diagnostic): fine grid for the target exponent, the ladder under the new target, what a bike-scale loss would
still add, and a stronger tree-model ceiling. Nothing here enters the chain (H13). Output: exp/v2/e3_ceiling.json
"""
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from exp.v2.lib import ALPHA, PAIRS, base_matrices, cfg, delta, evaluate, fit_predict, row
from src.common import r2, read_artifact
from src.features import BASE, DesignSpec

p3 = read_artifact("p3")
TARGET = DesignSpec.from_dict(p3["target_complexity"]["design_spec"])
PILOT = DesignSpec(base=BASE, blocks=("wd_x_hr",))
out = {}

print("== fine grid of the exponent ==")
for label, spec in (("pilot (base + wd x hr)", PILOT), ("target", TARGET)):
    ref = evaluate(spec, 0.0)
    for lam in (0.0, 0.05, 0.1, 0.15, 0.2):
        r = evaluate(spec, lam)
        out[f"{label}|{lam}"] = {k: v for k, v in r.items() if k != "val_pred"}
        print(row(f"{label} lam={lam}", r), "|", delta(r, ref))

print("== ladder under lam = 0.1 ==")
anchor = DesignSpec.from_dict(read_artifact("p2")["design_spec"])
acc = []
for level in cfg["p3"]["ladder"]:
    acc += level["add"]
    if level.get("drop_hour"):
        spec = DesignSpec(base=[c for c in BASE if not c.startswith("hr_")])
    elif level.get("phase2"):
        spec = anchor
    elif level["name"] in ("C1_additive", "C2_wd_x_hr"):
        spec = DesignSpec(base=BASE, blocks=tuple(acc))
    else:
        spec = DesignSpec(base=BASE, power_cols=anchor.power_cols, degree=anchor.degree, blocks=tuple(dict.fromkeys(acc)))
    r = evaluate(spec, 0.1)
    out[f"ladder|{level['name']}"] = {k: v for k, v in r.items() if k != "val_pred"}
    print(row(level["name"], r))


def nls(X, cnt, w0, iters=200):
    """Least squares on bikes with log link (Levenberg-Marquardt), small ridge."""
    n = len(cnt); R = ALPHA * np.eye(X.shape[1]); R[0, 0] = 0
    w, lam = w0.copy(), 1e-2
    obj = lambda v: float(0.5 * np.mean((np.exp(np.clip(X @ v, -30, 12)) - cnt) ** 2) + 0.5 * ALPHA * v[1:] @ v[1:])
    cur = obj(w)
    for _ in range(iters):
        mu = np.exp(np.clip(X @ w, -30, 12)); J = X * mu[:, None]
        g = J.T @ (mu - cnt) / n + R @ w; A = J.T @ J / n + R
        ok = False
        for _ in range(30):
            d = np.linalg.solve(A + lam * np.diag(np.diag(A)), g); new = obj(w - d)
            if new < cur:
                w, lam, ok = w - d, max(lam / 3, 1e-9), True
                break
            lam *= 4
        if not ok or abs(cur - new) < 1e-11 * max(1, abs(cur)):
            break
        cur = new
    return w


print("== what a bike-scale loss still adds on the target design ==")
scores = {"power 0.1": {}, "bike-scale NLS": {}}
for name, X, Xe, cnt, cnt_e, _, _ in base_matrices(TARGET):
    pe, _ = fit_predict(X, Xe, cnt, 0.1)
    z = np.log1p(cnt); R = ALPHA * np.eye(X.shape[1]); R[0, 0] = 0
    w0 = np.linalg.solve(X.T @ X / len(z) + R, X.T @ z / len(z))
    w = nls(X, cnt, w0)
    scores["power 0.1"][name] = r2(cnt_e, pe)
    scores["bike-scale NLS"][name] = r2(cnt_e, np.exp(np.minimum(X @ w if False else Xe @ w, np.log(cnt.max()) + 1)))
for k, v in scores.items():
    days = [v[f"day{f}"] for f in range(5)]
    out[f"loss|{k}"] = {"seeded": v["seeded"], "days": float(np.mean(days)), "days_folds": days, "chrono": v["chrono"]}
    print(f"{k:16s} seeded={v['seeded']:.4f} days={np.mean(days):.4f} chrono={v['chrono']:.4f}")
d = np.array(out["loss|bike-scale NLS"]["days_folds"]) - np.array(out["loss|power 0.1"]["days_folds"])
print(f"NLS - power 0.1 on days: {d.mean():+.4f} (se {d.std(ddof=1) / np.sqrt(5):.4f}, {int((d > 0).sum())}/5)")

print("== tree-model ceiling (diagnostic) ==")
COLS = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
        "instant", "ws_lag1", "wet3"]
for label, kw in (("HGB poisson 1500 x 0.03", dict(loss="poisson", max_iter=1500, learning_rate=0.03, max_leaf_nodes=31,
                                                    min_samples_leaf=10, l2_regularization=1.0, random_state=0)),
                  ("HGB poisson 3000 x 0.02, 63 leaves", dict(loss="poisson", max_iter=3000, learning_rate=0.02, max_leaf_nodes=63,
                                                             min_samples_leaf=10, l2_regularization=1.0, random_state=0))):
    v = {}
    for name, fit_df, eval_df in PAIRS:
        m = HistGradientBoostingRegressor(**kw).fit(fit_df[COLS], fit_df["cnt"])
        v[name] = r2(eval_df["cnt"].to_numpy(float), m.predict(eval_df[COLS]))
    days = [v[f"day{f}"] for f in range(5)]
    out[f"ceiling|{label}"] = {"seeded": v["seeded"], "days": float(np.mean(days)), "chrono": v["chrono"]}
    print(f"{label:36s} seeded={v['seeded']:.4f} days={np.mean(days):.4f} chrono={v['chrono']:.4f}")
json.dump(out, open("exp/v2/e3_ceiling.json", "w"), indent=1)
