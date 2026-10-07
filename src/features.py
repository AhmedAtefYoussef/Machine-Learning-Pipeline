"""Feature layer: named raw columns, the `DesignSpec` / `Design` pair that builds the standardised design matrix
(ARCHITECTURE section 4).

Derived columns (powers, interaction blocks) are products/powers of the STANDARDISED first-order columns and are
then standardised again with their own train mean/std. Every statistic is learned on the fitting frame only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

HR = tuple(f"hr_{h}" for h in range(1, 24))
SE = tuple(f"se_{k}" for k in range(2, 5))
MN = tuple(f"mn_{m}" for m in range(2, 13))
WK = tuple(f"wk_{d}" for d in range(1, 7))
DOY = ("doy_s1", "doy_c1", "doy_s2", "doy_c2")
MEMORY = ("wslag1_2", "wslag1_3", "wet3_2", "wet3_3")   # weather one hour earlier / worst of the three hours before

BASE: list[str] = list(HR + ("workingday", "holiday", "ws_2", "ws_3", "temp", "hum", "windspeed", "trend") + DOY)
CANDIDATES: tuple[str, ...] = ("atemp", "yr", "instant") + SE + MN + WK + MEMORY
REGISTRY: tuple[str, ...] = tuple(BASE) + CANDIDATES  # every first-order name, in column order

_EPOCH = pd.Timestamp("2011-01-01")


def raw_columns(df: pd.DataFrame, hum_fill: float) -> pd.DataFrame:
    """Unscaled named columns (n, len(REGISTRY)) for every candidate; zero humidity is replaced by hum_fill."""
    dates = pd.to_datetime(df["dteday"])
    hr = df["hr"].to_numpy()
    ws = np.minimum(df["weathersit"].to_numpy(), 3)  # weathersit 4 merged into 3
    hum = df["hum"].to_numpy(dtype=np.float64)
    doy = dates.dt.dayofyear.to_numpy(dtype=np.float64)
    cols: dict[str, np.ndarray] = {}
    for h in range(1, 24):
        cols[f"hr_{h}"] = (hr == h).astype(np.float64)
    cols["workingday"] = df["workingday"].to_numpy(dtype=np.float64)
    cols["holiday"] = df["holiday"].to_numpy(dtype=np.float64)
    cols["ws_2"] = (ws == 2).astype(np.float64)
    cols["ws_3"] = (ws == 3).astype(np.float64)
    cols["temp"] = df["temp"].to_numpy(dtype=np.float64)
    cols["hum"] = np.where(hum == 0.0, hum_fill, hum)
    cols["windspeed"] = df["windspeed"].to_numpy(dtype=np.float64)
    cols["trend"] = (dates - _EPOCH).dt.days.to_numpy(dtype=np.float64)
    for k in (1, 2):
        angle = 2.0 * np.pi * k * doy / 365.25
        cols[f"doy_s{k}"] = np.sin(angle)
        cols[f"doy_c{k}"] = np.cos(angle)
    cols["atemp"] = df["atemp"].to_numpy(dtype=np.float64)
    cols["yr"] = df["yr"].to_numpy(dtype=np.float64)
    cols["instant"] = df["instant"].to_numpy(dtype=np.float64)
    for k in range(2, 5):
        cols[f"se_{k}"] = (df["season"].to_numpy() == k).astype(np.float64)
    for m in range(2, 13):
        cols[f"mn_{m}"] = (df["mnth"].to_numpy() == m).astype(np.float64)
    for d in range(1, 7):
        cols[f"wk_{d}"] = (df["weekday"].to_numpy() == d).astype(np.float64)
    for k in (2, 3):  # weather memory: columns made by common.add_weather_memory
        cols[f"wslag1_{k}"] = (df["ws_lag1"].to_numpy() == k).astype(np.float64)
        cols[f"wet3_{k}"] = (df["wet3"].to_numpy() == k).astype(np.float64)
    return pd.DataFrame(cols, index=df.index)


def sources(feature_name: str) -> frozenset[str]:
    """Original data column(s) behind an expanded feature name ('bias' -> empty set)."""
    if feature_name == "bias":
        return frozenset()
    if "*" in feature_name:
        return frozenset().union(*(sources(part) for part in feature_name.split("*")))
    name = feature_name.split("^")[0]
    prefix_map = (("hr_", "hr"), ("ws_", "weathersit"), ("doy_", "dteday"), ("se_", "season"),
                  ("mn_", "mnth"), ("wk_", "weekday"), ("wslag1_", "weathersit"), ("wet3_", "weathersit"))
    if name == "trend":
        return frozenset({"dteday"})
    for prefix, original in prefix_map:
        if name.startswith(prefix):
            return frozenset({original})
    return frozenset({name})


# ----------------------------------------------------------------------------- blocks

def _block_table() -> dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]:
    """block name -> (extra first-order columns it needs, product factor tuples in column order)."""
    wd = "workingday"
    return {
        "wd_x_hr": ((), [(wd, h) for h in HR]),
        "hr_x_temp": ((), [(h, "temp") for h in HR]),
        "hr_x_hum": ((), [(h, "hum") for h in HR]),
        "wk": (WK, []),
        "wk_x_hr": (WK, [(w, h) for w in WK for h in HR]),
        "wd_x_hr_x_temp": ((), [(wd, h, "temp") for h in HR]),
        "hr_x_ws": ((), [(h, w) for h in HR for w in ("ws_2", "ws_3")]),
        "wd_x_wx": ((), [(wd, v) for v in ("temp", "hum", "ws_2", "ws_3")]),
        "hr_x_doy": ((), [(h, d) for h in HR for d in ("doy_s1", "doy_c1")]),
        "hr_x_trend": ((), [(h, "trend") for h in HR]),
        "mn": (MN, []),
        "mn_x_hr": (MN, [(m, h) for m in MN for h in HR]),
        "mn_x_wd_x_hr": (MN, [(m, wd, h) for m in MN for h in HR]),
        "c_atemp": (("atemp",), []),
        "c_yr": (("yr",), []),
        "c_instant": (("instant",), []),
        "c_season": (SE, []),
        "c_mnth": (MN, []),
        "c_weekday": (WK, []),
        "wx_detail": ((), [("hum^2",), ("hum^3",), ("windspeed^2",), ("temp", "windspeed")]),
        "ws_memory": (MEMORY, []),
    }


BLOCK_NAMES: tuple[str, ...] = tuple(_block_table())


@dataclass(frozen=True)
class DesignSpec:
    """What to build: first-order columns, columns raised to powers, max power, and interaction blocks."""
    base: tuple[str, ...]
    power_cols: tuple[str, ...] = ()
    degree: int = 1
    blocks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("base", "power_cols", "blocks"):  # accept lists, store tuples (hashable, frozen)
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if self.degree < 1:
            raise ValueError("degree must be >= 1")
        for name in self.base + self.power_cols:
            if name not in REGISTRY:
                raise ValueError(f"unknown column {name!r}")
        for block in self.blocks:
            if block not in BLOCK_NAMES:
                raise ValueError(f"unknown block {block!r}")

    def to_dict(self) -> dict:
        """JSON-ready dict (tuples become lists)."""
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "DesignSpec":
        """Inverse of to_dict."""
        return cls(base=tuple(d["base"]), power_cols=tuple(d["power_cols"]), degree=int(d["degree"]),
                   blocks=tuple(d["blocks"]))


def _first_order_names(spec: DesignSpec) -> list[str]:
    """spec.base, then any column a power or block needs, in registry order."""
    table = _block_table()
    needed = set(spec.base) | set(spec.power_cols)
    for block in spec.blocks:
        extra, factors = table[block]
        needed.update(extra)
        for factor in factors:
            needed.update(f.split("^")[0] for f in factor)
    added = [name for name in REGISTRY if name in needed and name not in spec.base]
    return list(spec.base) + added


def _derived_recipes(spec: DesignSpec) -> list[tuple[str, tuple[str, int] | tuple[str, ...]]]:
    """(name, recipe) for powers then block products; recipe is ('pow', k) with the column in the name, or factors."""
    recipes: list = []
    for col in spec.power_cols:
        for k in range(2, spec.degree + 1):
            recipes.append((f"{col}^{k}", (col, k)))
    table = _block_table()
    for block in spec.blocks:
        for factors in table[block][1]:
            name = "*".join(factors)
            if name in {n for n, _ in recipes}:   # already made by power_cols (or another block): never twice
                continue
            if len(factors) == 1 and "^" in factors[0]:   # a power inside a block, e.g. hum^2
                column, power = factors[0].split("^")
                recipes.append((name, (column, int(power))))
            else:
                recipes.append((name, factors))
    return recipes


class Design:
    """Standardised design matrix builder: fit statistics on one frame, then transform any frame."""

    def __init__(self, spec: DesignSpec) -> None:
        self.spec = spec
        self._first = _first_order_names(spec)
        self._recipes = _derived_recipes(spec)
        self._names = ["bias"] + self._first + [name for name, _ in self._recipes]
        self._mean: dict[str, float] = {}
        self._std: dict[str, float] = {}
        self._constants: set[str] = set()  # columns whose train std was 0
        self._hum_fill: float | None = None

    # -- construction helpers
    def _first_order_z(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Standardised first-order columns using the stored statistics."""
        raw = raw_columns(df, self._hum_fill)
        return {n: (raw[n].to_numpy() - self._mean[n]) / self._std[n] for n in self._first}

    @staticmethod
    def _derive(recipe, z: dict[str, np.ndarray]) -> np.ndarray:
        """One unstandardised derived column from standardised first-order columns."""
        if len(recipe) == 2 and isinstance(recipe[1], int):
            return z[recipe[0]] ** recipe[1]
        out = z[recipe[0]].copy()
        for factor in recipe[1:]:
            out = out * z[factor]
        return out

    # -- public API
    def fit(self, train_df: pd.DataFrame, base_scaler: dict | None = None) -> "Design":
        """Learn hum_fill and column mean/std (ddof=0, std 0 -> 1) on train_df; names in base_scaler reuse its statistics."""
        given = {}
        if base_scaler is not None:
            given = {n: (m, s) for n, m, s in zip(base_scaler["names"], base_scaler["mean"], base_scaler["std"])}
            self._hum_fill = float(base_scaler["hum_fill"])
        else:
            hum = train_df["hum"].to_numpy(dtype=np.float64)
            self._hum_fill = float(np.median(hum[hum > 0]))
        raw = raw_columns(train_df, self._hum_fill)
        self._mean, self._std, self._constants = {}, {}, set()

        def learn(name: str, col: np.ndarray) -> np.ndarray:
            if name in given:
                mean, std = float(given[name][0]), float(given[name][1])
            else:
                mean = float(col.mean())
                std = float(col.std())  # ddof=0
                if std == 0.0:
                    self._constants.add(name)
                    std = 1.0
            self._mean[name], self._std[name] = mean, std
            return (col - mean) / std

        z = {n: learn(n, raw[n].to_numpy()) for n in self._first}
        for name, recipe in self._recipes:
            learn(name, self._derive(recipe, z))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Design matrix (n, 1+p) with a ones column first."""
        if self._hum_fill is None:
            raise RuntimeError("Design.transform called before fit")
        z = self._first_order_z(df)
        out = np.empty((len(df), len(self._names)), dtype=np.float64)
        out[:, 0] = 1.0
        for j, name in enumerate(self._first, start=1):
            out[:, j] = z[name]
        for j, (name, recipe) in enumerate(self._recipes, start=1 + len(self._first)):
            out[:, j] = (self._derive(recipe, z) - self._mean[name]) / self._std[name]
        return out

    def fit_transform(self, train_df: pd.DataFrame, base_scaler: dict | None = None) -> np.ndarray:
        """fit(train_df, base_scaler) then transform(train_df)."""
        return self.fit(train_df, base_scaler).transform(train_df)

    @property
    def names(self) -> list[str]:
        """Column names, 'bias' first."""
        return list(self._names)

    @property
    def constant_columns(self) -> list[str]:
        """Columns with zero train variance (std kept at 1, so they are all zeros on train)."""
        return [n for n in self._names[1:] if n in self._constants]

    def scaler_dict(self) -> dict:
        """{'names','mean','std','hum_fill'} over all non-bias columns (JSON floats)."""
        names = self._names[1:]
        return {"names": list(names), "mean": [self._mean[n] for n in names],
                "std": [self._std[n] for n in names], "hum_fill": self._hum_fill}

    def subset(self, names: list[str]) -> list[int]:
        """Column indices (into the matrix from transform) of the given names."""
        index = {n: i for i, n in enumerate(self._names)}
        return [index[n] for n in names]
