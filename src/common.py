"""Data, seed, split, metric and artifact layer shared by every phase (ARCHITECTURE sections 2, 3, 5).

Everything here is float64 and deterministic: the only randomness is `np.random.default_rng(seed)`
inside the bootstrap helpers, and the only seeded data split is `seeded_split`.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

CONFIG_PATH = "config.yaml"
ARTIFACT_DIR = "artifacts"
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def load_config(path: str = CONFIG_PATH) -> dict:
    """Read the YAML config into a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def team_seed(ids: Sequence[str]) -> int:
    """Team seed from student ids: sha256 of the sorted ids joined by '_', mod 100000 (brief section 1.8)."""
    def derive(id_list: Sequence[str]) -> int:
        joined = "_".join(sorted(str(i) for i in id_list))
        return int(hashlib.sha256(joined.encode()).hexdigest(), 16) % 100000

    # self-test against the value published in the brief
    assert derive(["34521", "40218", "41190"]) == 41698, "team_seed self-test failed"
    return derive(ids)


def config_seed(cfg: dict) -> int:
    """Seed derived from cfg['team_ids']."""
    return team_seed(cfg["team_ids"])


def set_threads(n: int = 1) -> None:
    """Pin BLAS/OpenMP to `n` threads so sums are added in the same order on every run and machine.

    The environment variables cover libraries loaded later; `threadpool_limits` covers numpy's BLAS
    when it is already loaded (as in a notebook kernel).
    """
    for var in _THREAD_VARS:
        os.environ[var] = str(n)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n)
    except ImportError:  # the environment variables above are then the only pin
        pass


def load_train(cfg: dict) -> pd.DataFrame:
    """Labelled training table (n, 15) with `dteday` parsed to datetime."""
    return pd.read_csv(cfg["paths"]["train"], parse_dates=["dteday"])


def load_test(cfg: dict) -> pd.DataFrame:
    """Unlabelled hidden-test table; call only from the final prediction step."""
    return pd.read_csv(cfg["paths"]["test"], parse_dates=["dteday"])


def seeded_split(df: pd.DataFrame, seed: int, test_size: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ONE seeded split of the repo: (train_df, val_df) from a single train_test_split call."""
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed)
    return train_df, val_df


def chrono_split(df: pd.DataFrame, cut_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(early_df, late_df): rows with dteday < cut_date and rows with dteday >= cut_date."""
    cut = pd.Timestamp(cut_date)
    dates = pd.to_datetime(df["dteday"])
    return df[dates < cut], df[dates >= cut]


def day_block_folds(df: pd.DataFrame, k: int = 5) -> np.ndarray:
    """Fold id (n,) = rank of the row's date among the sorted unique dates, mod k. No RNG."""
    dates = pd.to_datetime(df["dteday"]).to_numpy()
    unique_dates = np.unique(dates)  # sorted
    rank = np.searchsorted(unique_dates, dates)
    return (rank % k).astype(int)


# ----------------------------------------------------------------------------- target

def to_target(cnt: np.ndarray) -> np.ndarray:
    """z = log1p(cnt)."""
    return np.log1p(np.asarray(cnt, dtype=np.float64))


def back_factor(method: str, z_train: np.ndarray, eta_train: np.ndarray, cnt_train: np.ndarray) -> float:
    """Back-transform factor s for cnt_hat = exp(eta) * s - 1 ('none', 'duan' or 'ls'), from train residuals."""
    z_train = np.asarray(z_train, dtype=np.float64)
    eta_train = np.asarray(eta_train, dtype=np.float64)
    if method == "none":
        return 1.0
    if method == "duan":  # smearing estimate: mean of exp(residual)
        return float(np.mean(np.exp(z_train - eta_train)))
    if method == "ls":  # least-squares scale of exp(eta) onto cnt + 1
        e = np.exp(eta_train)
        return float(np.sum((np.asarray(cnt_train, dtype=np.float64) + 1.0) * e) / np.sum(e * e))
    raise ValueError(f"unknown back-transform method: {method!r}")


def from_target(eta: np.ndarray, s: float) -> np.ndarray:
    """cnt_hat = clip(exp(eta) * s - 1, 0, None) on the bike scale."""
    return np.clip(np.exp(np.asarray(eta, dtype=np.float64)) * s - 1.0, 0.0, None)


# ----------------------------------------------------------------------------- metrics

def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Coefficient of determination 1 - SSE/SST."""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - yhat) ** 2) / sst)


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root mean squared error."""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _bootstrap_r2_values(y: np.ndarray, yhat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """R2 of each bootstrap resample; idx is (B, n) row indices."""
    ys, ps = y[idx], yhat[idx]
    sst = np.sum((ys - ys.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - np.sum((ys - ps) ** 2, axis=1) / sst


def _bootstrap_indices(n: int, seed: int, B: int) -> np.ndarray:
    """(B, n) resampling indices from default_rng(seed)."""
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def bootstrap_r2(y: np.ndarray, yhat: np.ndarray, seed: int, B: int = 1000) -> tuple[float, float, float]:
    """(lo, hi, se): percentile 95% interval and standard error of R2 over B row bootstraps."""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    vals = _bootstrap_r2_values(y, yhat, _bootstrap_indices(len(y), seed, B))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), float(vals.std(ddof=1))


def paired_bootstrap_delta_r2(y: np.ndarray, yhat_a: np.ndarray, yhat_b: np.ndarray, seed: int,
                              B: int = 1000) -> tuple[float, float, float]:
    """(delta, lo, hi): R2(a) - R2(b) on the data and its paired-bootstrap 95% interval (same resamples)."""
    y = np.asarray(y, dtype=np.float64)
    a = np.asarray(yhat_a, dtype=np.float64)
    b = np.asarray(yhat_b, dtype=np.float64)
    idx = _bootstrap_indices(len(y), seed, B)
    deltas = _bootstrap_r2_values(y, a, idx) - _bootstrap_r2_values(y, b, idx)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(r2(y, a) - r2(y, b)), float(lo), float(hi)


# ----------------------------------------------------------------------------- artifacts

def sha256_file(path: str) -> str:
    """Hex sha256 of a file's bytes."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _artifact_path(name: str) -> str:
    return f"{ARTIFACT_DIR}/{name}.json"


def _json_default(obj):
    """Convert numpy types for json.dumps; anything else is an error."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_artifact(name: str, payload: dict, upstream: str | None = None, cfg: dict | None = None) -> str:
    """Write artifacts/<name>.json (sorted keys, indent 1) adding seed, config_sha256, upstream_sha256; returns the path."""
    cfg = cfg if cfg is not None else load_config()
    body = dict(payload)
    body["seed"] = config_seed(cfg)
    body["config_sha256"] = sha256_file(CONFIG_PATH)
    body["upstream_sha256"] = sha256_file(_artifact_path(upstream)) if upstream else None
    # allow_nan=False makes NaN/inf raise ValueError instead of writing invalid JSON
    text = json.dumps(body, sort_keys=True, indent=1, default=_json_default, allow_nan=False)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    path = _artifact_path(name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def read_artifact(name: str) -> dict:
    """Load artifacts/<name>.json."""
    with open(_artifact_path(name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def cached_or_run(name: str, run_fn, cfg: dict, upstream: str | None = None) -> dict:
    """Return artifacts/<name>.json if it was built from this exact config and upstream artifact, else run the phase.

    Used by the notebook for the slow phases only. A config edited in memory (the live re-run case) differs
    from config.yaml on disk, so the phase is recomputed.
    """
    path = _artifact_path(name)
    if os.path.exists(path) and cfg == load_config():
        art = read_artifact(name)
        same_config = art.get("config_sha256") == sha256_file(CONFIG_PATH)
        same_upstream = upstream is None or art.get("upstream_sha256") == sha256_file(_artifact_path(upstream))
        if same_config and same_upstream:
            print(f"loaded cached artifacts/{name}.json (same config and upstream artifact); "
                  f"run `python run.py {name}` or change CFG to recompute")
            return art
    return run_fn(cfg)

