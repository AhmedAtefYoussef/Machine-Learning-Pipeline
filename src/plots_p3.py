"""Phase 3 figures. Each function takes the p3 payload (dict from artifacts/p3.json) and returns a Figure.

One question per figure, labelled axes, a legend, colour-blind-safe (Okabe-Ito) colours, matplotlib only.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=False)  # headless-safe; a notebook backend already chosen is kept
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"train": "#0072B2", "seeded": "#D55E00", "day_block": "#009E73", "chrono": "#CC79A7",
          "anchor": "#E69F00", "target": "#56B4E9", "top": "#000000"}
R2_FLOOR = -0.2  # curves below this are clipped in the plots (tiny-sample, huge-model fits explode)


def _clip(values: list[float]) -> list[float]:
    return [max(v, R2_FLOOR) for v in values]


def plot_ladder(p3: dict):
    """Question: where on the complexity ladder does validation stop improving, and which estimate says so?"""
    rows = p3["ladder"]
    x = [r["n_features"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, _clip([r["seeded"]["train_r2"] for r in rows]), "o-", color=COLORS["train"], label="seeded train")
    ax.plot(x, _clip([r["seeded"]["r2"] for r in rows]), "o-", color=COLORS["seeded"], label="seeded validation")
    ax.plot(x, _clip([r["day_block"]["r2"] for r in rows]), "s--", color=COLORS["day_block"], label="held-out days (CV)")
    ax.plot(x, _clip([r["chrono"]["r2"] for r in rows]), "^:", color=COLORS["chrono"], label="chronological")
    anchor, target = p3["anchor"]["anchor_index"], p3["target_index"]
    ax.axvline(x[anchor], color=COLORS["anchor"], lw=1.5, alpha=0.8, label="anchor (Phase 2)")
    ax.axvline(x[target], color=COLORS["target"], lw=1.5, alpha=0.8, label="target (best seeded)")
    ax.set_xscale("log")
    ax.set_ylim(R2_FLOOR, 1.0)
    ax.set_xlabel("number of weights (log scale)")
    ax.set_ylabel("R$^2$ on bikes (cnt)")
    ax.set_title("Complexity ladder: training vs three validation estimates")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_degree_axis(p3: dict):
    """Question: does raising only the polynomial degree change the fit?"""
    rows = p3["degree_axis"]
    x = [r["degree"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, [r["seeded"]["train_r2"] for r in rows], "o-", color=COLORS["train"], label="seeded train")
    ax.plot(x, [r["seeded"]["r2"] for r in rows], "o-", color=COLORS["seeded"], label="seeded validation")
    ax.plot(x, [r["day_block"]["r2"] for r in rows], "s--", color=COLORS["day_block"], label="held-out days (CV)")
    ax.plot(x, [r["chrono"]["r2"] for r in rows], "^:", color=COLORS["chrono"], label="chronological")
    ax.set_xlabel("polynomial degree of the power columns")
    ax.set_ylabel("R$^2$ on bikes (cnt)")
    ax.set_title("Degree axis (anchor blocks and power columns fixed)")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_learning_curves(p3: dict):
    """Question: is the remaining error a bias problem (curves meet) or a variance problem (gap closes with data)?"""
    curves = p3["learning_curves"]
    labels = {"anchor": "anchor", "target": "target", "top": "top of ladder"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, curve in curves.items():
        n = [c["n_rows"] for c in curve]
        ax.plot(n, _clip([c["train_r2"] for c in curve]), "o--", color=COLORS[key], alpha=0.7,
                label=f"{labels[key]}: train")
        ax.plot(n, _clip([c["val_r2"] for c in curve]), "o-", color=COLORS[key], label=f"{labels[key]}: validation")
    ax.set_xlabel("number of training rows (nested sets of whole days)")
    ax.set_ylabel("R$^2$ on bikes (cnt)")
    ax.set_ylim(R2_FLOOR, 1.0)
    ax.set_title("Learning curves (dashed = training, solid = validation)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_three_estimates(p3: dict):
    """Question: how much does the answer depend on how we validate, for the anchor and the target?"""
    names = ["seeded", "day_holdout", "chrono"]
    pretty = ["seeded split", "held-out days", "chronological"]
    anchor, target = p3["estimates"], p3["estimates_target"]
    pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    a_vals = [anchor[n]["r2"] for n in names]
    t_vals = [target[n]["r2"] for n in names]
    ax.bar(pos - 0.2, a_vals, 0.4, color=COLORS["anchor"], label=f"anchor ({p3['anchor']['n_features']} weights)")
    ax.bar(pos + 0.2, t_vals, 0.4, color=COLORS["target"],
           label=f"target {p3['target_level']} ({p3['target_complexity']['n_features']} weights)")
    for x, v in zip(np.concatenate([pos - 0.2, pos + 0.2]), a_vals + t_vals):
        ax.text(x, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(pos)
    ax.set_xticklabels(pretty)
    ax.set_ylim(min(a_vals + t_vals) - 0.05, 1.0)
    ax.set_ylabel("R$^2$ on bikes (cnt)")
    ax.set_title("Same designs, three validators")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
