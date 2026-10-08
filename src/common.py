"""Data, seed, split, metric and artifact layer shared by every phase (ARCHITECTURE sections 2, 3, 5).

Everything here is float64 and deterministic: the only randomness is `np.random.default_rng(seed)`
inside the bootstrap helpers, and the only seeded data split is `seeded_split`.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
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


MEMORY_INPUT_COLUMNS = ["dteday", "hr", "weathersit"]   # the only columns the weather memory reads (no label)


def _timestamps(df: pd.DataFrame) -> pd.Series:
    """Hour of each row: dteday + hr hours."""
    return pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")


def _situation(df: pd.DataFrame) -> np.ndarray:
    """Weather situation with category 4 merged into 3 (as in the features)."""
    return np.minimum(df["weathersit"].to_numpy(), 3)


def add_weather_memory(df: pd.DataFrame, lookup: pd.DataFrame | None = None) -> pd.DataFrame:
    """Copy of `df` with two integer columns built from the weather of the hours before each row (inputs only).

    ws_lag1 = weather situation one hour earlier; wet3 = the largest situation 1, 2 and 3 hours earlier.
    Hours are found by timestamp (dteday + hr hours) in `lookup`, the frame whose rows form the timeline (default: df
    itself). An earlier hour that is missing from the timeline counts as the row's own situation. Only dteday, hr and
    weathersit are read; the label is never touched.
    """
    timeline = df if lookup is None else lookup
    known = pd.Series(_situation(timeline), index=_timestamps(timeline))
    known = known[~known.index.duplicated()]
    own = _situation(df)
    ts = _timestamps(df)

    def hours_earlier(h: int) -> np.ndarray:
        before = known.reindex(ts - pd.Timedelta(hours=h)).to_numpy(dtype=np.float64)
        return np.where(np.isnan(before), own, before).astype(int)

    out = df.copy()
    out["ws_lag1"] = hours_earlier(1)
    out["wet3"] = np.maximum.reduce([hours_earlier(1), hours_earlier(2), hours_earlier(3)])
    return out


def load_train(cfg: dict) -> pd.DataFrame:
    """Labelled training table with `dteday` parsed to datetime and the weather memory columns added.

    The memory columns are computed from the training file alone (its own rows are the timeline)."""
    return add_weather_memory(pd.read_csv(cfg["paths"]["train"], parse_dates=["dteday"]))


def load_test(cfg: dict) -> pd.DataFrame:
    """Unlabelled hidden-test table with the weather memory columns; call only from the final prediction step.

    The timeline is the input columns (dteday, hr, weathersit) of the training file AND of this file, so the first
    hour of a hidden day sees the last hour of the day before. No label is read for this."""
    test_df = pd.read_csv(cfg["paths"]["test"], parse_dates=["dteday"])
    train_inputs = pd.read_csv(cfg["paths"]["train"], parse_dates=["dteday"], usecols=MEMORY_INPUT_COLUMNS)
    timeline = pd.concat([train_inputs, test_df[MEMORY_INPUT_COLUMNS]], ignore_index=True)
    return add_weather_memory(test_df, lookup=timeline)


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

BACK_METHODS = ("none", "duan", "ls")
Q_FLOOR = 1e-9   # q(eta) = (power*eta + 1)^(1/power) needs a positive base; below this the base is held here


@dataclass(frozen=True)
class Target:
    """The power-family target of the whole chain (ADR-017) and the way predictions are turned back into bikes.

    forward   z = ((cnt + 1)^power - 1) / power   (power = 0 is the limit z = log(cnt + 1) = log1p(cnt))
    q(eta)    = cnt + 1 on the bike scale = (power*eta + 1)^(1/power)   (power = 0: exp(eta)); the inverse of forward
    to_bikes  cnt_hat = clip(s * q(eta) - 1, 0, None), s = back-transform factor ('none' = 1, 'ls', 'duan')
    """
    power: float
    back_method: str

    def forward(self, cnt: np.ndarray) -> np.ndarray:
        """z = ((cnt + 1)^power - 1) / power, or log1p(cnt) when power == 0."""
        cnt = np.asarray(cnt, dtype=np.float64)
        if self.power == 0:
            return np.log1p(cnt)
        return ((cnt + 1.0) ** self.power - 1.0) / self.power

    def q(self, eta: np.ndarray) -> np.ndarray:
        """cnt + 1 implied by the linear predictor eta: exp(eta), or (power*eta + 1)^(1/power) for power > 0."""
        eta = np.asarray(eta, dtype=np.float64)
        if self.power == 0:
            return np.exp(eta)
        return np.maximum(self.power * eta + 1.0, Q_FLOOR) ** (1.0 / self.power)

    def dq_deta(self, eta: np.ndarray) -> np.ndarray:
        """d q / d eta = q^(1 - power) (power = 0: q itself); the chain-rule factor of the bonus gradient."""
        return self.q(eta) ** (1.0 - self.power)

    def factor(self, eta_train: np.ndarray, cnt_train: np.ndarray) -> float:
        """Back-transform factor s from the training rows only.

        'none' -> 1; 'ls' -> argmin_s sum (cnt + 1 - s q)^2 = sum (cnt + 1) q / sum q^2;
        'duan' -> mean(exp(z - eta)), the smearing estimate, which is defined for the log (power 0) only."""
        if self.back_method == "none":
            return 1.0
        if self.back_method == "ls":
            q = self.q(eta_train)
            return float(np.sum((np.asarray(cnt_train, dtype=np.float64) + 1.0) * q) / np.sum(q * q))
        if self.back_method == "duan":
            if self.power != 0:
                raise ValueError("Duan's smearing factor is defined for the log target (power 0) only")
            return float(np.mean(np.exp(self.forward(cnt_train) - np.asarray(eta_train, dtype=np.float64))))
        raise ValueError(f"unknown back-transform method: {self.back_method!r}")

    def bikes_unclipped(self, eta: np.ndarray, s: float) -> np.ndarray:
        """s * q(eta) - 1 (may be below 0)."""
        return s * self.q(eta) - 1.0

    def to_bikes(self, eta: np.ndarray, s: float) -> np.ndarray:
        """cnt_hat = clip(s * q(eta) - 1, 0, None)."""
        return np.clip(self.bikes_unclipped(eta, s), 0.0, None)

    def eta_cap(self, cnt_train: np.ndarray) -> float:
        """The eta at which q = e * (max(cnt_train) + 1): predictions are capped there so q stays finite.

        forward(e*(max + 1) - 1); for power 0 this is log1p(max) + 1 (the largest fitted z plus one)."""
        top = float(np.max(cnt_train))
        if self.power == 0:
            return float(np.log1p(top) + 1.0)
        return float(self.forward(np.e * (top + 1.0) - 1.0))

    def to_dict(self) -> dict:
        """The keys of the Phase 1 artifact that describe the target (read back by `from_artifact`)."""
        return {"target_transform": "power", "target_power": self.power,
                "backtransform": {"method": self.back_method}}

    @staticmethod
    def from_artifact(p1: dict) -> "Target":
        """The target of the chain, as stored by Phase 1."""
        return Target(float(p1["target_power"]), p1["backtransform"]["method"])


def valid_back_methods(power: float, methods: Sequence[str]) -> list[str]:
    """The candidate back-transform methods that exist for this exponent (Duan's factor needs power 0)."""
    return [m for m in methods if m != "duan" or power == 0]


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
    # the hash must describe the config the phase ran with: a drill with an edited CFG is not the file on disk
    if cfg == load_config():
        body["config_sha256"] = sha256_file(CONFIG_PATH)
    else:
        body["config_sha256"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()
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

