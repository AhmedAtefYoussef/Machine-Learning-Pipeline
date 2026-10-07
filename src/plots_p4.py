"""Phase 4 figures. Each function takes the p4 payload (dict from artifacts/p4.json) and returns a Figure.

One question per figure, labelled axes, a legend, colour-blind-safe (Okabe-Ito) colours, matplotlib only.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=False)  # headless-safe; a notebook backend already chosen is kept
import matplotlib.pyplot as plt
import numpy as np

from src.features import sources
from src.verdicts import ORIGINAL

COLORS = {"l2": "#0072B2", "l1": "#D55E00", "enet": "#009E73", "unregularised": "#000000",
          "cv": "#CC79A7", "alone": "#56B4E9", "group": "#E69F00"}
LABELS = {"l2": "ridge (L2)", "l1": "lasso (L1)", "enet": "elastic net"}
VERDICT_COLORS = {"useful": "#009E73", "redundant": "#E69F00", "uninformative": "#999999"}


def _enet_rows(p4: dict) -> list[dict]:
    """Validation curve of the elastic net at its chosen l1_ratio."""
    return p4["curves"]["enet"][str(p4["methods"]["enet"]["l1_ratio"])]


def _method_rows(p4: dict, method: str) -> list[dict]:
    return _enet_rows(p4) if method == "enet" else p4["curves"][method]


def plot_validation_curves(p4: dict):
    """Question: how much penalty does each method want, and does any penalty beat the unregularised fit?"""
    best = p4["methods"]
    floor = min(m["val_r2"] for m in best.values()) - 0.15
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ("l2", "l1", "enet"):
        rows = _method_rows(p4, m)
        name = LABELS[m] + (f", rho = {best['enet']['l1_ratio']}" if m == "enet" else "")
        ax.plot([r["alpha"] for r in rows], [max(r["val_r2"], floor) for r in rows], "o-", ms=3,
                color=COLORS[m], label=name)
        ax.plot(best[m]["lambda"], best[m]["val_r2"], "*", ms=16, color=COLORS[m], mec="k", label=f"{m} chosen")
    ax.axhline(p4["unregularised"]["val_r2"], color=COLORS["unregularised"], ls="--", label="unregularised")
    ax.set_xscale("log")
    ax.set_ylim(floor, max(m["val_r2"] for m in best.values()) + 0.01)
    ax.set_xlabel("penalty alpha (log scale)")
    ax.set_ylabel("validation R$^2$ on bikes (cnt)")
    ax.set_title("Validation R$^2$ along each regularisation path")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cv_vs_validation(p4: dict, method: str):
    """Question: does the held-out-day CV pick the same penalty as the validation split?"""
    cv, rows = p4["cv"][method], _method_rows(p4, method)
    alphas = np.array(cv["alphas"])
    mean, se = np.array(cv["mean"]), np.array(cv["se"])
    floor = max(mean.max(), max(r["val_r2"] for r in rows)) - 0.15
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(alphas, np.maximum(mean, floor), yerr=se, fmt="s--", ms=3, color=COLORS["cv"], capsize=2,
                label="held-out days: mean $\\pm$ s.e. (inside training)")
    ax.plot([r["alpha"] for r in rows], [max(r["val_r2"], floor) for r in rows], "o-", ms=3, color=COLORS[method],
            label="seeded validation split")
    ax.axvline(cv["cv_best_alpha"], color=COLORS["cv"], lw=1, label="CV best")
    ax.axvline(cv["cv_1se_alpha"], color=COLORS["cv"], lw=1, ls=":", label="CV 1-s.e. rule")
    ax.axvline(p4["methods"][method]["lambda"], color=COLORS[method], lw=1, ls="--", label="validation best")
    ax.set_xscale("log")
    ax.set_ylim(floor, max(mean.max(), max(r["val_r2"] for r in rows)) + 0.01)
    ax.set_xlabel("penalty alpha (log scale)")
    ax.set_ylabel("R$^2$ on bikes (cnt)")
    ax.set_title(f"{LABELS[method]}: two ways to choose the penalty")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_paths(p4: dict, method: str):
    """Question: which coefficients shrink first, and do correlated columns trade weight (temp vs atemp)?"""
    path = p4["paths"][method]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, coef in path["coef"].items():
        ax.plot(path["alphas"], coef, lw=1.5, label=name)
    ax.axvline(p4["methods"][method]["lambda"], color="k", ls="--", lw=1, label="chosen alpha")
    ax.set_xscale("log")
    ax.set_xlabel("penalty alpha (log scale)")
    ax.set_ylabel("coefficient (standardised column, log1p(cnt) scale)")
    ax.set_title(f"Coefficient paths, {LABELS[method]}")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_verdicts(p4: dict):
    """Question: what does validation R$^2$ lose when a column (or the column with its partners) is removed?"""
    cols = list(ORIGINAL)
    verdicts = p4["column_verdicts"]
    y = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for offset, key, color, label in ((-0.2, "drop_alone", COLORS["alone"], "drop the column alone"),
                                      (0.2, "drop_group", COLORS["group"], "drop it with its partners")):
        nums = [verdicts[c]["numbers"][key] for c in cols]
        has = [n is not None for n in nums]
        delta = np.array([n["delta"] if n else 0.0 for n in nums])
        lo = np.array([n["lo"] if n else 0.0 for n in nums])
        hi = np.array([n["hi"] if n else 0.0 for n in nums])
        ax.barh(y[has] + offset, delta[has], 0.38, color=color, label=label,
                xerr=[(delta - lo)[has], (hi - delta)[has]], capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} ({verdicts[c]['verdict']})" for c in cols])
    for tick, c in zip(ax.get_yticklabels(), cols):
        tick.set_color(VERDICT_COLORS[verdicts[c]["verdict"]])
    ax.axvline(0.0, color="k", lw=0.8)
    ax.axvline(p4["verdict_thresholds"]["min_delta"], color="k", ls=":", lw=1, label="min_delta")
    ax.invert_yaxis()
    ax.set_xlabel("validation R$^2$ lost when removed (full minus reduced; 95% paired bootstrap)")
    ax.set_title("Drop tests per original column (ridge at its chosen lambda)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def _max_freq_by_column(freq: dict) -> list[float]:
    """Highest selection frequency among each original column's own features."""
    out = []
    for c in ORIGINAL:
        mine = [f for name, f in freq.items() if sources(name) == frozenset({c})]
        out.append(max(mine) if mine else 0.0)
    return out


def plot_stability(p4: dict):
    """Question: which original columns does the lasso select consistently when whole days are resampled?"""
    cols = list(ORIGINAL)
    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 0.2, _max_freq_by_column(p4["stability"]["l1"]["freq"]), 0.4, color=COLORS["l1"], label="lasso")
    ax.bar(x + 0.2, _max_freq_by_column(p4["stability"]["enet"]["freq"]), 0.4, color=COLORS["enet"],
           label="elastic net")
    ax.axhline(p4["stability_threshold"], color="k", ls="--", lw=1, label="selection threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("selection frequency (best own feature)")
    ax.set_title("Stability selection over resampled training days")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
