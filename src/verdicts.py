"""Column verdicts for Phase 4: pure functions, no data access (the refits that produce the numbers live in p4.py).

Every ORIGINAL data column gets one of three verdicts:
    useful        removing it alone costs validation R2
    redundant     removing it alone costs nothing, but its information is carried by partner columns
                  (removing it together with its partners costs R2, or it explains a visible share of R2 by itself)
    uninformative removing it costs nothing and it explains nothing by itself
"""
from __future__ import annotations

import numpy as np

from src.features import sources

ORIGINAL = ("season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum",
            "windspeed", "dteday", "instant")

# Columns that describe (part of) the same thing; judged together when one alone is not enough.
GROUPS = {"temperature": ("temp", "atemp"),
          "season": ("season", "mnth", "dteday"),
          "level": ("yr", "instant", "dteday"),
          "daytype": ("weekday", "workingday", "holiday")}

VIF_CAP = 1.0e6


# ----------------------------------------------------------------------------- which features belong to a column

def involving(column: str, feature_names: list[str]) -> list[str]:
    """Features built (at least partly) from `column`, e.g. hr -> hr_1.., hr_1*temp, ..."""
    return [n for n in feature_names if column in sources(n)]


def own(column: str, feature_names: list[str]) -> list[str]:
    """Features built from `column` alone (no interaction with another original column)."""
    return [n for n in feature_names if sources(n) == frozenset({column})]


def partners(column: str) -> list[str]:
    """Columns sharing a GROUP with `column` (in ORIGINAL order); empty for hr, weathersit, hum, windspeed."""
    together = {m for members in GROUPS.values() if column in members for m in members}
    return [c for c in ORIGINAL if c in together and c != column]


# ----------------------------------------------------------------------------- numbers that need only the design matrix

def _correlation(X: np.ndarray) -> np.ndarray:
    """Correlation matrix of the columns of X; constant columns get correlation 0 with everything."""
    Xc = X - X.mean(axis=0)
    norm = np.sqrt((Xc ** 2).sum(axis=0))
    norm[norm == 0.0] = np.inf
    Z = Xc / norm
    return Z.T @ Z


def max_abs_corr(X: np.ndarray, feature_names: list[str], column: str) -> dict:
    """Largest |correlation| between an own feature of `column` and an own feature of any other column."""
    corr = np.abs(_correlation(X))
    mine = [feature_names.index(n) for n in own(column, feature_names)]
    best = {"value": 0.0, "with": None}
    for j, name in enumerate(feature_names):
        other = sources(name)
        if len(other) != 1 or column in other or not mine:
            continue
        value = float(corr[mine, j].max())
        if value > best["value"]:
            best = {"value": value, "with": next(iter(other))}
    return best


def vif_max(X: np.ndarray, feature_names: list[str], column: str) -> float | None:
    """Largest variance inflation factor over the first-order own features of `column` (None if it has none).

    VIF_j = 1 / (1 - R2_j) with R2_j from regressing column j on all other first-order columns; the regression
    uses the pseudo-inverse so exactly collinear sets give R2 = 1, reported as VIF_CAP.
    """
    first = [i for i, n in enumerate(feature_names) if "*" not in n and "^" not in n]
    mine = [i for i in first if feature_names[i] in own(column, feature_names)]
    if not mine:
        return None
    corr = _correlation(X[:, first])
    position = {i: k for k, i in enumerate(first)}
    best = 0.0
    for i in mine:
        j = position[i]
        others = [k for k in range(len(first)) if k != j]
        coef = np.linalg.pinv(corr[np.ix_(others, others)]) @ corr[others, j]
        r2_j = float(corr[j, others] @ coef)
        best = max(best, min(1.0 / max(1.0 - r2_j, 1.0 / VIF_CAP), VIF_CAP))
    return best


def entry_ranks(entry_alpha: dict[str, float | None], feature_names: list[str]) -> dict[str, int | None]:
    """Rank of each original column by the largest lasso entry alpha among its own features (1 = enters first).

    Columns none of whose own features ever enters the path get None; ties share the better rank.
    """
    best = {}
    for column in ORIGINAL:
        alphas = [entry_alpha[n] for n in own(column, feature_names) if entry_alpha.get(n) is not None]
        best[column] = max(alphas) if alphas else None
    ranks = {}
    for column, value in best.items():
        ranks[column] = None if value is None else 1 + sum(1 for v in best.values() if v is not None and v > value)
    return ranks


# ----------------------------------------------------------------------------- the rule

def alone_hurts(numbers: dict, min_delta: float) -> bool:
    """Dropping the column alone costs R2: lower interval end above 0 and cost at least min_delta."""
    d = numbers["drop_alone"]
    return d["lo"] > 0 and d["delta"] >= min_delta


def group_hurts(numbers: dict, min_delta: float) -> bool:
    """Dropping the column together with all its partners costs R2 (False when it has no partners)."""
    d = numbers["drop_group"]
    return d is not None and d["lo"] > 0 and d["delta"] >= min_delta


def basic_verdict(numbers: dict, min_delta: float, solo_min: float) -> str:
    """useful / redundant / uninformative for one column, before the representative fix."""
    if alone_hurts(numbers, min_delta):
        return "useful"
    solo = numbers["solo_r2"]
    if group_hurts(numbers, min_delta) or (solo is not None and solo >= solo_min):
        return "redundant"
    return "uninformative"


def promote_representatives(verdicts: dict[str, str], numbers: dict[str, dict], min_delta: float) -> dict[str, bool]:
    """Representative fix, applied to `verdicts` in place (GROUPS order); returns {column: is_representative}.

    A group whose members jointly matter (dropping a member with its partners hurts) but where no member is useful
    alone would otherwise lose all its information. We keep the member with the highest solo R2 as `useful`.
    """
    representative = {c: False for c in ORIGINAL}
    for members in GROUPS.values():
        hurts = any(group_hurts(numbers[c], min_delta) for c in members)
        if hurts and not any(verdicts[c] == "useful" for c in members):
            keeper = max(members, key=lambda c: -np.inf if numbers[c]["solo_r2"] is None else numbers[c]["solo_r2"])
            verdicts[keeper] = "useful"
            representative[keeper] = True
    return representative


def carried_by(column: str, verdicts: dict[str, str], numbers: dict) -> list[str]:
    """For a redundant column, the useful partners; if there are none, the most correlated other column."""
    if verdicts[column] != "redundant":
        return []
    useful = [p for p in partners(column) if verdicts[p] == "useful"]
    if useful:
        return useful
    other = numbers["max_abs_corr"]["with"]
    return [other] if other else []


# ----------------------------------------------------------------------------- the sentence

def _signed_interval(d: dict) -> str:
    return f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"


def evidence_sentence(column: str, numbers: dict, carriers: list[str]) -> str:
    """One sentence built only from the stored numbers (a positive cost means validation R2 gets worse)."""
    parts = [f"dropping it alone costs validation R2 {_signed_interval(numbers['drop_alone'])}"]
    group = numbers["drop_group"]
    if group is not None:
        names = ", ".join(m for m in group["members"] if m != column)
        parts.append(f"dropping it with {names} costs {group['delta']:+.4f}")
    if numbers["solo_r2"] is not None:
        parts.append(f"its own columns alone reach validation R2 {numbers['solo_r2']:.4f}")
    lasso = numbers["lasso"]
    if lasso["n_own"]:
        parts.append(f"lasso keeps {lasso['n_own_nonzero']} of {lasso['n_own']} own columns "
                     f"(selected in up to {100 * lasso['max_freq_own']:.0f}% of resamples)")
    corr = numbers["max_abs_corr"]
    if corr["with"]:
        parts.append(f"|r| = {corr['value']:.3f} with {corr['with']}")
    if carriers:
        parts.append("carried by " + ", ".join(carriers))
    return "; ".join(parts) + "."
