"""Phase 3 figures. Each function takes the p3 payload (dict from artifacts/p3.json) and returns a Figure.

One question per figure; colours come from `plotstyle.SERIES` (the three validators keep their colour everywhere).
"""
from __future__ import annotations

import numpy as np

from src import plotstyle as style
from src.plotstyle import SERIES

style.apply()

R2_FLOOR = -0.2  # curves below this are clipped in the plots (tiny-sample, huge-model fits explode)
TALL = (7.2, 4.4)  # room for the legend under the axes
# (artifact path, legend label, series colour, line style) of the four scores drawn on the ladder and the degree axis
SCORES = (("seeded.train_r2", "train", "train", "o-"), ("seeded.r2", "validation", "validation", "o-"),
          ("day_block.r2", "held-out days", "held_out_days", "s--"), ("chrono.r2", "chronological", "chronological", "^:"))


def _clip(values: list[float]) -> list[float]:
    return [max(v, R2_FLOOR) for v in values]


def _score(row: dict, path: str) -> float:
    outer, inner = path.split(".")
    return row[outer][inner]


def _draw_scores(ax, x, rows: list[dict]) -> None:
    """The four scores of each row against x, with a legend."""
    for path, label, series, marker in SCORES:
        ax.plot(x, _clip([_score(r, path) for r in rows]), marker, color=SERIES[series], label=label)
    style.legend(ax, ncol=4)


def plot_ladder(p3: dict):
    """Question: where on the complexity ladder does validation stop improving, and which estimate says so?"""
    rows = p3["ladder"]
    x = [r["n_features"] for r in rows]
    fig, ax = style.new_axes("Where on the complexity ladder does validation stop improving?",
                             "number of weights (log scale)", "R² on bikes", figsize=TALL)
    ax.set_xscale("log")
    ax.set_ylim(R2_FLOOR, 1.0)
    _draw_scores(ax, x, rows)
    style.mark(ax, x[p3["anchor"]["anchor_index"]], "anchor (Phase 2)")
    style.mark(ax, x[p3["target_index"]], "target (best validation)", level=1)
    return fig


def plot_degree_axis(p3: dict):
    """Question: does raising only the polynomial degree change the fit?"""
    rows = p3["degree_axis"]
    fig, ax = style.new_axes("Does raising only the polynomial degree change the fit?",
                             "degree of the power columns (anchor blocks fixed)", "R² on bikes", figsize=TALL)
    _draw_scores(ax, [r["degree"] for r in rows], rows)
    return fig


def plot_learning_curves(p3: dict):
    """Question: is the remaining error a bias problem (curves meet) or a variance problem (gap closes with data)?"""
    labels = {"anchor": "anchor", "target": "target", "top": "top of ladder"}
    fig, ax = style.new_axes("Is the remaining error bias or variance? (dashed = training)",
                             "training rows (nested sets of whole days)", "R² on bikes", figsize=TALL)
    for key, curve in p3["learning_curves"].items():
        n = [c["n_rows"] for c in curve]
        ax.plot(n, _clip([c["train_r2"] for c in curve]), "o--", color=SERIES[key], alpha=0.7,
                label=f"{labels[key]}: train")
        ax.plot(n, _clip([c["val_r2"] for c in curve]), "o-", color=SERIES[key], label=f"{labels[key]}: validation")
    ax.set_ylim(R2_FLOOR, 1.0)
    style.legend(ax, ncol=3)
    return fig


def plot_three_estimates(p3: dict):
    """Question: how much does the answer depend on how we validate, for the anchor and the target?"""
    names = ["seeded", "day_holdout", "chrono"]
    anchor, target = p3["estimates"], p3["estimates_target"]
    pos = np.arange(len(names))
    fig, ax = style.new_axes("How much does the answer depend on how we validate?", "", "R² on bikes")
    a_vals = [anchor[n]["r2"] for n in names]
    t_vals = [target[n]["r2"] for n in names]
    ax.bar(pos - 0.2, a_vals, 0.4, color=SERIES["anchor"], label=f"anchor ({p3['anchor']['n_features']} weights)")
    ax.bar(pos + 0.2, t_vals, 0.4, color=SERIES["target"],
           label=f"target {p3['target_level']} ({p3['target_complexity']['n_features']} weights)")
    for x, v in zip(np.concatenate([pos - 0.2, pos + 0.2]), a_vals + t_vals):
        ax.annotate(f"{v:.3f}", (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                    color=style.INK_SECONDARY)
    ax.set_xticks(pos)
    ax.set_xticklabels(["validation split", "held-out days", "chronological"])
    ax.set_ylim(min(a_vals + t_vals) - 0.05, 1.0)
    style.legend(ax, ncol=2)
    return fig
