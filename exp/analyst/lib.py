"""Analyst helpers (diagnostic only, H13: nothing here enters the chain).

Reads the labelled training file through src.common.load_train only; the hidden-test file is never opened.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from src.common import (back_factor, chrono_split, config_seed, day_block_folds, from_target, load_config,
                        load_train, r2, read_artifact, rmse, seeded_split, set_threads, to_target)
from src.features import Design, DesignSpec
from src.validation import ETA_HEADROOM, fit_linear

OUT_DIR = "exp/analyst"


def setup():
    """(cfg, seed, all_df with helper columns, train_df, val_df, C5 spec, back method, cut date, k)."""
    cfg = load_config()
    set_threads(1)
    seed = config_seed(cfg)
    all_df = add_context_columns(load_train(cfg))
    train_df, val_df = seeded_split(all_df, seed, cfg["split"]["test_size"])
    p1, p3 = read_artifact("p1"), read_artifact("p3")
    spec = DesignSpec.from_dict(p3["target_complexity"]["design_spec"])
    return cfg, seed, all_df, train_df, val_df, spec, p1["backtransform"]["method"], cfg["chrono"]["cut_date"], \
        cfg["day_block_folds"]


def add_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Input-only helper columns (no target): previous-hour weather, same-day weather aggregates, day type."""
    df = df.copy()
    ts = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    ws = np.minimum(df["weathersit"].to_numpy(), 3).astype(float)
    look = pd.DataFrame({"ws": ws, "hum": df["hum"].to_numpy(float), "temp": df["temp"].to_numpy(float)},
                        index=ts.to_numpy())
    for lag in (1, 2, 3):
        prev = look.reindex((ts - pd.Timedelta(hours=lag)).to_numpy())
        df[f"ws_lag{lag}"] = np.where(np.isnan(prev["ws"].to_numpy()), ws, prev["ws"].to_numpy())
        if lag == 1:
            df["hum_lag1"] = np.where(np.isnan(prev["hum"].to_numpy()), df["hum"].to_numpy(float),
                                      prev["hum"].to_numpy())
    df["wet3"] = np.maximum.reduce([df["ws_lag1"].to_numpy(), df["ws_lag2"].to_numpy(), df["ws_lag3"].to_numpy()])
    df["ws_c"] = ws
    by_day = df.groupby("dteday")
    df["day_temp"] = by_day["temp"].transform("mean")
    df["day_hum"] = by_day["hum"].transform("mean")
    df["day_wet"] = by_day["ws_c"].transform(lambda s: float(np.mean(s >= 2)))
    df["day_rain"] = by_day["ws_c"].transform(lambda s: float(np.mean(s >= 3)))
    df["days"] = (pd.to_datetime(df["dteday"]) - pd.Timestamp("2011-01-01")).dt.days.astype(float)
    df["daytype"] = np.where(df["workingday"] == 1, "work", "off")
    return df


def hr_onehot(df: pd.DataFrame) -> np.ndarray:
    hr = df["hr"].to_numpy()
    return np.stack([(hr == h).astype(float) for h in range(24)], axis=1)


def fit_predict_x(spec: DesignSpec, fit_df, eval_df, back_method: str, extra_fn=None, alpha: float = 1e-8,
                  poisson_iters: int = 0, cap: bool = True, wls_power: float = 0.0) -> dict:
    """Same fit as src.validation.fit_predict, plus optional extra columns (statistics from fit_df only)
    and an optional Poisson-style IRLS refinement (diagnostic). Returns predictions and log-scale values."""
    design = Design(spec).fit(fit_df)
    X_fit, X_eval = design.transform(fit_df), design.transform(eval_df)
    if extra_fn is not None:
        E_fit, E_eval = np.asarray(extra_fn(fit_df), float), np.asarray(extra_fn(eval_df), float)
        mean, std = E_fit.mean(axis=0), E_fit.std(axis=0)
        keep = std > 0
        X_fit = np.hstack([X_fit, (E_fit[:, keep] - mean[keep]) / std[keep]])
        X_eval = np.hstack([X_eval, (E_eval[:, keep] - mean[keep]) / std[keep]])
    cnt_fit = fit_df["cnt"].to_numpy(float)
    z_fit = to_target(cnt_fit)
    w = fit_linear(X_fit, z_fit, alpha)
    for _ in range(poisson_iters):  # IRLS for a log-link Poisson fit of cnt + 1
        eta = np.minimum(X_fit @ w, z_fit.max() + ETA_HEADROOM)
        mu = np.exp(eta)
        work = eta + (cnt_fit + 1.0 - mu) / mu
        sw = np.sqrt(mu / mu.mean())
        w = fit_linear(X_fit * sw[:, None], work * sw, alpha)
    if wls_power > 0:  # two-stage weighted least squares on z: weights = (first-stage fitted bikes) ** power
        sw = np.exp(0.5 * wls_power * np.minimum(X_fit @ w, z_fit.max() + ETA_HEADROOM))
        sw = sw / sw.mean()
        w = fit_linear(X_fit * sw[:, None], z_fit * sw, alpha)
    eta_fit = X_fit @ w
    raw_eval = X_eval @ w
    eta_eval = np.minimum(raw_eval, z_fit.max() + ETA_HEADROOM) if cap else raw_eval
    s = back_factor(back_method, z_fit, eta_fit, cnt_fit)
    return {"pred": from_target(eta_eval, s), "eta": eta_eval, "s": s, "p": int(X_fit.shape[1]),
            "pred_fit": from_target(eta_fit, s), "eta_fit": eta_fit,
            "n_capped": int(np.sum(raw_eval > z_fit.max() + ETA_HEADROOM)),
            "n_clipped": int(np.sum(np.exp(eta_eval) * s - 1.0 < 0.0))}


def three(spec, train_df, val_df, all_df, cut, back, k, extra_fn=None, poisson_iters=0, fitter=None,
          wls_power=0.0) -> dict:
    """Seeded / held-out-day / chronological R2 with predictions kept for paired comparisons.

    `fitter(fit_df, eval_df) -> bike-scale predictions` replaces the linear fit (used by the ceiling probe)."""
    def run(fit_df, eval_df):
        if fitter is not None:
            return fitter(fit_df, eval_df)
        return fit_predict_x(spec, fit_df, eval_df, back, extra_fn, poisson_iters=poisson_iters,
                             wls_power=wls_power)["pred"]

    out = {}
    pred_val = run(train_df, val_df)
    out["seeded_pred"] = pred_val
    out["seeded"] = r2(val_df["cnt"].to_numpy(float), pred_val)
    out["seeded_rmse"] = rmse(val_df["cnt"].to_numpy(float), pred_val)
    folds = day_block_folds(train_df, k)
    oof = np.empty(len(train_df))
    fold_r2 = []
    for f in range(k):
        held = folds == f
        oof[held] = run(train_df[~held], train_df[held])
        fold_r2.append(r2(train_df["cnt"].to_numpy(float)[held], oof[held]))
    out["oof_pred"], out["folds"] = oof, fold_r2
    out["day_block"], out["day_block_sd"] = float(np.mean(fold_r2)), float(np.std(fold_r2, ddof=1))
    early, late = chrono_split(all_df, cut)
    pred_late = run(early, late)
    out["chrono_pred"] = pred_late
    out["chrono"] = r2(late["cnt"].to_numpy(float), pred_late)
    out["chrono_mean_ratio"] = float(pred_late.mean() / late["cnt"].mean())
    return out


def scores_only(res: dict) -> dict:
    return {k: (v if not isinstance(v, np.ndarray) else None) for k, v in res.items()
            if not isinstance(v, np.ndarray)}


def paired_boot(y, a, b, seed, B=1000) -> tuple[float, float, float]:
    """R2(a) - R2(b) with a paired row-bootstrap 95% interval."""
    y, a, b = (np.asarray(v, float) for v in (y, a, b))
    idx = np.random.default_rng(seed).integers(0, len(y), size=(B, len(y)))
    ys = y[idx]
    sst = np.sum((ys - ys.mean(axis=1, keepdims=True)) ** 2, axis=1)
    d = (np.sum((ys - b[idx]) ** 2, axis=1) - np.sum((ys - a[idx]) ** 2, axis=1)) / sst
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(r2(y, a) - r2(y, b)), float(lo), float(hi)


def cluster_boot_r2(y, pred, groups, seed, B=1000) -> tuple[float, float]:
    """95% interval of R2 resampling whole groups (dates) instead of rows."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    codes = pd.factorize(groups)[0]
    n_g = codes.max() + 1
    sy, syy, sse, cnt = (np.bincount(codes, weights=v, minlength=n_g) for v in
                         (y, y * y, (y - pred) ** 2, np.ones_like(y)))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        g = rng.integers(0, n_g, n_g)
        n = cnt[g].sum()
        sst = syy[g].sum() - sy[g].sum() ** 2 / n
        vals.append(1.0 - sse[g].sum() / sst)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def save(name: str, payload: dict, config: dict) -> str:
    """Write exp/analyst/<name>.json with the sha of its config; returns the config sha."""
    sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    body = {"config": config, "config_sha": sha, **payload}

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o).__name__)

    with open(f"{OUT_DIR}/{name}.json", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(body, indent=1, sort_keys=True, default=default))
    return sha
