"""Shared helpers for the v2 exploration (DIAGNOSTIC ONLY: nothing here enters the chain or the submission).

Everything is fitted on the fit rows only. Scores are R2 on bikes on three validators:
  seeded   the team's 80/20 split                    days   5 folds of whole held-out days inside the training portion
  chrono   train before the cut date, validate after
"""
import numpy as np
import pandas as pd

from src.common import chrono_split, config_seed, day_block_folds, load_config, load_train, r2, seeded_split, set_threads
from src.features import Design, DesignSpec

cfg = load_config()
set_threads(1)
SEED = config_seed(cfg)
ALL = load_train(cfg)
TRAIN, VAL = seeded_split(ALL, SEED, cfg["split"]["test_size"])
FOLDS = day_block_folds(TRAIN, cfg["day_block_folds"])
EARLY, LATE = chrono_split(ALL, cfg["chrono"]["cut_date"])
PAIRS = [("seeded", TRAIN, VAL)] + [(f"day{f}", TRAIN[FOLDS != f], TRAIN[FOLDS == f]) for f in range(cfg["day_block_folds"])] \
    + [("chrono", EARLY, LATE)]
ALPHA = 1e-4
_BASE_CACHE: dict = {}


def base_matrices(spec: DesignSpec):
    """[(name, X_fit, X_eval, cnt_fit, cnt_eval, fit_index, eval_index)] for the seven fit/eval pairs."""
    key = repr(spec.to_dict())
    if key not in _BASE_CACHE:
        out = []
        for name, fit_df, eval_df in PAIRS:
            d = Design(spec).fit(fit_df)
            out.append((name, d.transform(fit_df), d.transform(eval_df), fit_df["cnt"].to_numpy(float),
                        eval_df["cnt"].to_numpy(float), fit_df.index, eval_df.index))
        _BASE_CACHE[key] = out
    return _BASE_CACHE[key]


def forward(cnt, lam):
    """Power-family target on cnt + 1: log for lam = 0, ((cnt+1)^lam - 1)/lam otherwise."""
    return np.log1p(cnt) if lam == 0 else ((cnt + 1.0) ** lam - 1.0) / lam


def inverse(eta, lam):
    """q = cnt + 1 on the bike scale (before the least-squares factor)."""
    return np.exp(eta) if lam == 0 else np.maximum(lam * eta + 1.0, 1e-9) ** (1.0 / lam)


def fit_predict(X, Xe, cnt, lam, alpha=ALPHA, weights=None):
    """Ridge on the transformed target; returns bike-scale predictions for the eval rows and the fit rows."""
    z = forward(cnt, lam)
    R = alpha * np.eye(X.shape[1]); R[0, 0] = 0.0
    if weights is None:
        w = np.linalg.solve(X.T @ X / len(z) + R, X.T @ z / len(z))
    else:
        Xw = X * weights[:, None]
        w = np.linalg.solve(Xw.T @ X / weights.sum() + R, Xw.T @ z / weights.sum())
    q = inverse(X @ w, lam)
    s = float(((cnt + 1.0) * q).sum() / (q * q).sum())
    cap = (cnt.max() + 1.0) * np.e                       # same idea as the eta cap of validation.fit_predict
    return np.clip(s * np.minimum(inverse(Xe @ w, lam), cap) - 1.0, 0, None), np.clip(s * q - 1.0, 0, None)


def standardise(extra: pd.DataFrame, fit_index, eval_index):
    """Extra columns standardised with the fit rows' mean/std (std 0 -> column dropped)."""
    A = extra.loc[fit_index].to_numpy(float); B = extra.loc[eval_index].to_numpy(float)
    mu, sd = A.mean(0), A.std(0)
    keep = sd > 1e-12
    return (A[:, keep] - mu[keep]) / sd[keep], (B[:, keep] - mu[keep]) / sd[keep]


def evaluate(spec: DesignSpec, lam: float, extra: pd.DataFrame | None = None, alpha=ALPHA, drop=None):
    """dict(seeded, train, days, days_folds, chrono, p) for one configuration."""
    res, folds = {}, []
    for name, X, Xe, cnt, cnt_e, fi, ei in base_matrices(spec):
        if drop is not None:
            X, Xe = X[:, drop], Xe[:, drop]
        if extra is not None and extra.shape[1]:
            A, B = standardise(extra, fi, ei)
            X, Xe = np.hstack([X, A]), np.hstack([Xe, B])
        pe, pf = fit_predict(X, Xe, cnt, lam, alpha)
        score = r2(cnt_e, pe)
        if name == "seeded":
            res["seeded"], res["train"], res["p"] = score, r2(cnt, pf), X.shape[1]
            res["val_pred"] = pe
        elif name == "chrono":
            res["chrono"] = score
            res["chrono_ratio"] = float(pe.mean() / cnt_e.mean())
        else:
            folds.append(score)
    res["days"], res["days_folds"] = float(np.mean(folds)), folds
    return res


def delta(a: dict, b: dict) -> str:
    """a minus b on the three validators, with the paired fold standard error for days."""
    d = np.array(a["days_folds"]) - np.array(b["days_folds"])
    return (f"seeded {a['seeded'] - b['seeded']:+.4f} | days {d.mean():+.4f} (se {d.std(ddof=1) / np.sqrt(len(d)):.4f}, "
            f"{int((d > 0).sum())}/5) | chrono {a['chrono'] - b['chrono']:+.4f}")


def row(name: str, r: dict) -> str:
    return f"{name:34s} p={r['p']:4d} train={r['train']:.4f} seeded={r['seeded']:.4f} days={r['days']:.4f} chrono={r['chrono']:.4f}"
