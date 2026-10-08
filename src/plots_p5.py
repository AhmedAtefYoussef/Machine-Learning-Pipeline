"""Phase 5 and 6 figures. The p5 functions take the payload of artifacts/p5.json; one question per figure.

Figures are built with `matplotlib.figure.Figure` directly; colours come from `plotstyle.SERIES`. A figure never has two
y-axes: when two scales are needed it has two panels side by side.
"""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from src import plotstyle as style
from src.plots import DAY_TYPES
from src.plotstyle import SERIES

style.apply()

RULE_LABELS = {"global": "one global threshold", "workingday+hr": "per (day type, hour)",
               "yr+workingday+hr": "per (year, day type, hour)"}
TALL = (7.2, 4.4)  # room for the legend under the axes


def _two_panels(figsize=(9.0, 4.0)):
    fig = Figure(figsize=figsize, layout="constrained")
    return fig, fig.subplots(1, 2)


def plot_roc_pr(p5: dict) -> Figure:
    """Question: how well does the classifier rank hours, and what do precision and recall look like along the way?"""
    fig, (roc_ax, pr_ax) = _two_panels()
    points = p5["curve_points"]
    style.label_axes(roc_ax, f"Can it rank hours? ROC-AUC {p5['metrics']['roc_auc']:.3f}", "false positive rate", "recall")
    style.label_axes(pr_ax, f"What does it cost in precision? PR-AUC {p5['metrics']['pr_auc']:.3f}", "recall", "precision")
    roc_ax.plot([0, 1], [0, 1], ":", color=style.MUTED)
    style.note(roc_ax, (0.6, 0.6), "chance")
    roc_ax.plot([q["fpr"] for q in points], [q["recall"] for q in points], color=SERIES["validation"])
    pr_ax.plot([q["recall"] for q in points], [q["precision"] for q in points], color=SERIES["validation"])
    style.baseline(pr_ax, p5["class_balance"]["val"], "share of positives (chance)")
    for ax in (roc_ax, pr_ax):
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
    return fig


def plot_threshold_curve(p5: dict) -> Figure:
    """Question: where should the alarm threshold sit when a miss costs more than a false alarm?"""
    fig, (score_ax, cost_ax) = _two_panels()
    style.label_axes(score_ax, "What do we catch at each threshold?", "threshold on the predicted probability",
                     "precision, recall, F1")
    style.label_axes(cost_ax, "What does each threshold cost?", "threshold on the predicted probability",
                     f"cost per row (miss = {p5['cost_ratio']:g} x false alarm)")
    rows = p5["threshold_curve"]
    thr = [r["thr"] for r in rows]
    for key, label in (("precision", "precision"), ("recall", "recall"), ("f1", "F1")):
        score_ax.plot(thr, [r[key] for r in rows], "-o", ms=3, color=SERIES[key], label=label)
    cost_ax.plot(thr, [r["cost"] for r in rows], "-s", ms=3, color=SERIES["cost"])
    for ax in (score_ax, cost_ax):
        style.mark(ax, p5["t_cost"], f"1/(1+c) = {p5['t_cost']:.2f}")
        style.mark(ax, 0.5, "0.5", level=1)
    style.legend(score_ax)
    return fig


def plot_calibration(p5: dict) -> Figure:
    """Question: when the model says p, is the share of high-demand hours about p?"""
    fig, ax = style.new_axes(f"When the model says p, is the share about p? (largest gap {p5['calibration_max_gap']:.3f})",
                             "mean predicted probability in the bin", "observed share of high-demand hours",
                             figsize=(6.0, 4.4))
    rows = [r for r in p5["calibration"] if r["n"] > 0]
    ax.plot([0, 1], [0, 1], ":", color=style.MUTED)
    style.note(ax, (0.8, 0.8), "perfect calibration", offset=(4, -14))
    ax.plot([r["mean_p"] for r in rows], [r["frac_pos"] for r in rows], "-o", color=SERIES["validation"])
    for r in rows:
        ax.annotate(str(r["n"]), (r["mean_p"], r["frac_pos"]), textcoords="offset points", xytext=(4, -10), fontsize=7,
                    color=style.MUTED)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    return fig


def plot_label_variants(p5: dict) -> Figure:
    """Question: how much of a high ROC-AUC is just the clock or the calendar, for each label rule?"""
    fig, ax = style.new_axes("How much of a high ROC-AUC is just the clock or the calendar?", "label rule", "ROC-AUC",
                             figsize=TALL)
    variants = p5["label_variants"]
    x = np.arange(len(variants))
    for k, (key, series, label) in enumerate((("auc_hour_only", "hour_only", "hour of day only"),
                                              ("auc_trend_only", "trend_only", "trend only"),
                                              ("auc_full", "all_columns", "all surviving columns"))):
        values = [v[key] for v in variants]
        ax.bar(x + (k - 1) * 0.27, values, 0.27, color=SERIES[series], label=label)
        for xi, v in zip(x + (k - 1) * 0.27, values):
            ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                        color=style.INK_SECONDARY)
    style.baseline(ax, 0.5, "chance", side="right")
    ax.set_xticks(x)
    ax.set_xticklabels([RULE_LABELS.get("+".join(v["group_by"]) or "global", "+".join(v["group_by"]))
                        for v in variants])
    ax.set_ylim(0.4, 1.05)
    style.legend(ax)
    return fig


def plot_test_profile(train_df, test_df, sub) -> Figure:
    """Question: does the predicted demand on the hidden days follow the hourly shape of the training days?"""
    fig, ax = style.new_axes("Do the predicted hidden days follow the hourly shape of the training days?", "hour of day",
                             "mean bikes per hour", figsize=TALL)
    predicted = test_df.merge(sub, on="instant")
    for flag, label, colour in DAY_TYPES:
        seen = train_df[train_df["workingday"] == flag].groupby("hr")["cnt"].mean()
        ahead = predicted[predicted["workingday"] == flag].groupby("hr")["cnt"].mean()
        ax.plot(seen.index, seen.values, "-o", ms=4, color=colour, label=f"{label}, training")
        ax.plot(ahead.index, ahead.values, "--s", ms=4, color=colour, label=f"{label}, predicted")
    style.legend(ax, ncol=2)
    return fig
