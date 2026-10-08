"""DIAGNOSTIC ONLY (never enters the chain or the submission): does the scale of the loss cost us R2 on bikes?

Same design for every row of the table: the Phase 4 surviving columns (read from artifacts/p4.json, not modified).
Three fits, same small ridge term, compared on the three validators:
  log    least squares on log1p(cnt)  (what the pipeline does; back-transform factor 'ls')
  pois   Poisson regression with log link (IRLS), prediction = exp(Xw)
  nls    least squares on bikes with log link, prediction = exp(Xw) (damped Gauss-Newton, started from the log fit)
Output: exp/loss_scale/diag.json and a printed table.
"""
import json

import numpy as np

from src.common import (back_factor, chrono_split, config_seed, day_block_folds, from_target, load_config, load_train,
                        paired_bootstrap_delta_r2, r2, read_artifact, seeded_split, set_threads, to_target)
from src.features import Design, DesignSpec

ALPHA = 1e-4   # ridge on the non-bias weights, objective scaled by 1/n, same for all three fits
cfg = load_config()
set_threads(1)
seed = config_seed(cfg)
p4 = read_artifact("p4")
SPEC = DesignSpec.from_dict(p4["design_spec"])
KEEP = p4["survivors_expanded"]


def matrices(fit_df, eval_df):
    """Design fitted on fit_df only; bias + surviving columns for both frames."""
    design = Design(SPEC).fit(fit_df)
    cols = [0] + [design.names.index(name) for name in KEEP]
    return design.transform(fit_df)[:, cols], design.transform(eval_df)[:, cols]


def ridge_matrix(p):
    R = ALPHA * np.eye(p)
    R[0, 0] = 0.0
    return R


def fit_log(X, cnt):
    z = to_target(cnt)
    w = np.linalg.solve(X.T @ X / len(z) + ridge_matrix(X.shape[1]), X.T @ z / len(z))
    return w, back_factor("ls", z, X @ w, cnt)


def fit_poisson(X, cnt, w0, iters=60):
    """Minimise (1/n) sum(mu - y*eta) + (alpha/2)|w|^2 by Newton steps with step halving."""
    n, R = len(cnt), ridge_matrix(X.shape[1])
    w = w0.copy()

    def obj(v):
        eta = np.clip(X @ v, -30, 12)
        return float(np.mean(np.exp(eta) - cnt * eta) + 0.5 * v[1:] @ v[1:] * ALPHA)
    cur = obj(w)
    for _ in range(iters):
        mu = np.exp(np.clip(X @ w, -30, 12))
        g = X.T @ (mu - cnt) / n + R @ w
        H = (X * mu[:, None]).T @ X / n + R
        d = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-6 and obj(w - t * d) > cur - 1e-12:
            t /= 2
        w = w - t * d
        new = obj(w)
        if abs(cur - new) < 1e-12 * max(1.0, abs(cur)):
            break
        cur = new
    return w


def fit_nls(X, cnt, w0, iters=200):
    """Minimise (1/2n) sum(exp(eta) - y)^2 + (alpha/2)|w|^2 by Levenberg-Marquardt."""
    n, R = len(cnt), ridge_matrix(X.shape[1])
    w, lam = w0.copy(), 1e-2

    def obj(v):
        mu = np.exp(np.clip(X @ v, -30, 12))
        return float(0.5 * np.mean((mu - cnt) ** 2) + 0.5 * v[1:] @ v[1:] * ALPHA)
    cur = obj(w)
    for _ in range(iters):
        mu = np.exp(np.clip(X @ w, -30, 12))
        J = X * mu[:, None]
        g = J.T @ (mu - cnt) / n + R @ w
        A = J.T @ J / n + R
        improved = False
        for _ in range(30):
            d = np.linalg.solve(A + lam * np.diag(np.diag(A)), g)
            new = obj(w - d)
            if new < cur:
                w, lam, improved = w - d, max(lam / 3, 1e-9), True
                break
            lam *= 4
        if not improved or abs(cur - new) < 1e-11 * max(1.0, abs(cur)):
            break
        cur = new
    return w


def predictions(fit_df, eval_df):
    X, Xe = matrices(fit_df, eval_df)
    cnt = fit_df["cnt"].to_numpy(dtype=float)
    w_log, s = fit_log(X, cnt)
    start = w_log.copy()
    start[0] += np.log(s)                       # exp(start) ~ cnt + 1 on average
    w_p = fit_poisson(X, cnt, start)
    w_n = fit_nls(X, cnt, w_p)
    cap = np.log(cnt.max()) + 1.0               # same safety cap idea as validation.fit_predict
    out = {"log": from_target(Xe @ w_log, s), "pois": np.exp(np.minimum(Xe @ w_p, cap)), "nls": np.exp(np.minimum(Xe @ w_n, cap))}
    train = {"log": from_target(X @ w_log, s), "pois": np.exp(X @ w_p), "nls": np.exp(X @ w_n)}
    return out, train, cnt


all_df = load_train(cfg)
train_df, val_df = seeded_split(all_df, seed, cfg["split"]["test_size"])
res = {"alpha": ALPHA, "n_columns": len(KEEP) + 1}

pred, train_pred, cnt_tr = predictions(train_df, val_df)
y = val_df["cnt"].to_numpy(dtype=float)
res["seeded"] = {k: r2(y, v) for k, v in pred.items()}
res["seeded_train"] = {k: r2(cnt_tr, v) for k, v in train_pred.items()}
res["seeded_mean_ratio"] = {k: float(v.mean() / y.mean()) for k, v in pred.items()}
for k in ("pois", "nls"):
    d, lo, hi = paired_bootstrap_delta_r2(y, pred[k], pred["log"], seed, cfg["bootstrap"]["B"])
    res[f"seeded_delta_{k}_minus_log"] = {"delta": d, "lo": lo, "hi": hi}

folds = day_block_folds(train_df, cfg["day_block_folds"])
per_fold = {k: [] for k in ("log", "pois", "nls")}
for f in range(cfg["day_block_folds"]):
    fit_part, held = train_df[folds != f], train_df[folds == f]
    pf, _, _ = predictions(fit_part, held)
    for k, v in pf.items():
        per_fold[k].append(r2(held["cnt"].to_numpy(dtype=float), v))
res["day_block"] = {k: float(np.mean(v)) for k, v in per_fold.items()}
res["day_block_folds"] = per_fold
for k in ("pois", "nls"):
    diff = np.array(per_fold[k]) - np.array(per_fold["log"])
    res[f"day_block_delta_{k}_minus_log"] = {"mean": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                                             "folds_better": int((diff > 0).sum())}

early, late = chrono_split(all_df, cfg["chrono"]["cut_date"])
pc, _, _ = predictions(early, late)
yl = late["cnt"].to_numpy(dtype=float)
res["chrono"] = {k: r2(yl, v) for k, v in pc.items()}
res["chrono_mean_ratio"] = {k: float(v.mean() / yl.mean()) for k, v in pc.items()}

json.dump(res, open("exp/loss_scale/diag.json", "w", encoding="utf-8"), indent=1)
print(f"design: {res['n_columns']} columns (Phase 4 survivors), ridge {ALPHA}")
print(f"{'fit':6s} {'train':>8s} {'seeded':>8s} {'days':>8s} {'chrono':>8s} {'val mean ratio':>15s} {'chrono mean ratio':>18s}")
for k in ("log", "pois", "nls"):
    print(f"{k:6s} {res['seeded_train'][k]:8.4f} {res['seeded'][k]:8.4f} {res['day_block'][k]:8.4f} {res['chrono'][k]:8.4f} "
          f"{res['seeded_mean_ratio'][k]:15.3f} {res['chrono_mean_ratio'][k]:18.3f}")
for k in ("pois", "nls"):
    a, b = res[f"seeded_delta_{k}_minus_log"], res[f"day_block_delta_{k}_minus_log"]
    print(f"{k} - log: seeded {a['delta']:+.4f} [{a['lo']:+.4f}, {a['hi']:+.4f}] | held-out days {b['mean']:+.4f} "
          f"(se {b['se']:.4f}, better in {b['folds_better']}/5 folds) | chrono {res['chrono'][k] - res['chrono']['log']:+.4f}")
