"""Phase 4 figures. Each function takes the p4 payload (dict from artifacts/p4.json) and returns a Figure.

One question per figure; the three methods and the two validators keep their colour from `plotstyle.SERIES`.
"""
from __future__ import annotations

import numpy as np

from src import plotstyle as style
from src.features import sources
from src.plotstyle import SERIES
from src.verdicts import ORIGINAL

style.apply()

LABELS = {"l2": "ridge (L2)", "l1": "lasso (L1)", "enet": "elastic net"}
COLOUR = {"l2": SERIES["ridge"], "l1": SERIES["lasso"], "enet": SERIES["enet"]}
TALL = (7.2, 4.4)  # room for the legend under the axes
# the columns the path figure is about get a colour, the others stay grey
PATH_COLUMNS = ("temp", "atemp", "instant", "yr", "trend", "workingday")


def _enet_rows(p4: dict) -> list[dict]:
    """Validation curve of the elastic net at its chosen l1_ratio."""
    return p4["curves"]["enet"][str(p4["methods"]["enet"]["l1_ratio"])]


def _method_rows(p4: dict, method: str) -> list[dict]:
    return _enet_rows(p4) if method == "enet" else p4["curves"][method]


def _stage_inputs(p4: dict, stage: str) -> tuple[dict, dict, dict, str]:
    """(methods, curves, unregularised, title) of stage A (full design) or B (surviving columns)."""
    if stage == "full":
        return p4["methods"], p4["curves"], p4["unregularised"], "full design"
    if stage == "final":
        final = p4["final"]
        return final["methods"], final["curves"], final["unregularised"], "surviving columns"
    raise ValueError("stage must be 'full' or 'final'")


def plot_validation_curves(p4: dict, stage: str = "full"):
    """Question: how much penalty does each method want, and does any penalty beat the unregularised fit?"""
    best, curves, unreg, which = _stage_inputs(p4, stage)
    floor = min(m["val_r2"] for m in best.values()) - 0.15
    fig, ax = style.new_axes(f"How much penalty does each method want? ({which})", "penalty alpha (log scale)",
                             "validation R² on bikes", figsize=TALL)
    ax.set_xscale("log")
    ax.set_ylim(floor, max(m["val_r2"] for m in best.values()) + 0.01)
    for m in ("l2", "l1", "enet"):
        rows = curves["enet"][str(best["enet"]["l1_ratio"])] if m == "enet" else curves[m]
        name = LABELS[m] + (f", rho = {best['enet']['l1_ratio']}" if m == "enet" else "")
        ax.plot([r["alpha"] for r in rows], [max(r["val_r2"], floor) for r in rows], "o-", ms=3,
                color=COLOUR[m], label=name)
        ax.plot(best[m]["lambda"], best[m]["val_r2"], "*", ms=14, color=COLOUR[m], mec=style.INK)
    ax.plot([], [], "*", ms=12, color=style.MUTED, label="chosen penalty")
    style.baseline(ax, unreg["val_r2"], "no penalty")
    style.legend(ax, ncol=2)
    return fig


def plot_cv_vs_validation(p4: dict, method: str):
    """Question: does the held-out-day CV pick the same penalty as the validation split?"""
    cv, rows = p4["cv"][method], _method_rows(p4, method)
    mean, se = np.array(cv["mean"]), np.array(cv["se"])
    top = max(mean.max(), max(r["val_r2"] for r in rows))
    floor = top - 0.15
    fig, ax = style.new_axes(f"{LABELS[method]}: do two ways of choosing the penalty agree?",
                             "penalty alpha (log scale)", "R² on bikes", figsize=TALL)
    ax.set_xscale("log")
    ax.set_ylim(floor, top + 0.01)
    ax.errorbar(cv["alphas"], np.maximum(mean, floor), yerr=se, fmt="s--", ms=3, capsize=2,
                          color=SERIES["held_out_days"], label="held-out days: mean ± s.e.")
    ax.plot([r["alpha"] for r in rows], [max(r["val_r2"], floor) for r in rows], "o-", ms=3,
                       color=SERIES["validation"], label="validation split")
    style.mark(ax, cv["cv_best_alpha"], "CV best")
    style.mark(ax, cv["cv_1se_alpha"], "CV 1-s.e. rule", level=1)
    style.mark(ax, p4["methods"][method]["lambda"], "validation best", level=2)
    style.legend(ax, ncol=2)
    return fig


def plot_paths(p4: dict, method: str):
    """Question: which coefficients shrink first, and do correlated columns trade weight (temp vs atemp)?"""
    path = p4["paths"][method]
    fig, ax = style.new_axes(f"Which coefficients shrink first? ({LABELS[method]})", "penalty alpha (log scale)",
                             "coefficient (standardised, target scale)", figsize=TALL)
    others = 0
    for name, coef in path["coef"].items():
        if name in PATH_COLUMNS:
            ax.plot(path["alphas"], coef, color=style.PALETTE[PATH_COLUMNS.index(name)], label=name)
        else:
            ax.plot(path["alphas"], coef, color=style.LIGHT_GREY, lw=1.2, label="other columns" if not others else "_nolegend_")
            others += 1
    ax.set_xscale("log")
    style.mark(ax, p4["methods"][method]["lambda"], "chosen alpha")
    style.legend(ax, ncol=4)
    return fig


def plot_verdicts(p4: dict):
    """Question: what does validation R2 lose when a column (or the column with its partners) is removed?"""
    cols = list(ORIGINAL)
    verdicts = p4["column_verdicts"]
    y = np.arange(len(cols))
    fig, ax = style.new_axes("What does validation R² lose when a column is removed?",
                             "validation R² lost when removed (full minus reduced, 95% paired bootstrap)", "",
                             figsize=(7.2, 6.0))
    for offset, key, series, label in ((-0.2, "drop_alone", "drop_alone", "drop the column alone"),
                                       (0.2, "drop_group", "drop_group", "drop it with its partners")):
        nums = [verdicts[c]["numbers"][key] for c in cols]
        has = [n is not None for n in nums]
        delta = np.array([n["delta"] if n else 0.0 for n in nums])
        lo = np.array([n["lo"] if n else 0.0 for n in nums])
        hi = np.array([n["hi"] if n else 0.0 for n in nums])
        ax.barh(y[has] + offset, delta[has], 0.38, color=SERIES[series], label=label,
                xerr=[(delta - lo)[has], (hi - delta)[has]], capsize=2, error_kw={"ecolor": style.INK_SECONDARY})
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} ({verdicts[c]['verdict']})" for c in cols])
    ax.axvline(0.0, color=style.MUTED, lw=0.8)
    style.mark(ax, p4["verdict_thresholds"]["min_delta"], "threshold for 'useful'")
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.legend(ax, ncol=2)
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
    fig, ax = style.new_axes("Which columns does the lasso select consistently?", "",
                             "selection frequency (best own feature)", figsize=TALL)
    ax.bar(x - 0.2, _max_freq_by_column(p4["stability"]["l1"]["freq"]), 0.4, color=SERIES["lasso"], label="lasso")
    ax.bar(x + 0.2, _max_freq_by_column(p4["stability"]["enet"]["freq"]), 0.4, color=SERIES["enet"],
           label="elastic net")
    ax.axhline(p4["stability_threshold"], color=style.MUTED, ls=":", lw=1)
    style.note(ax, (len(cols) - 1.4, p4["stability_threshold"]), "selection threshold", offset=(0, 4))
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    style.legend(ax, ncol=2)
    return fig
