"""Phase 5 and 6 figures. The p5 functions take the payload of artifacts/p5.json; one question per figure.

Figures are built with `matplotlib.figure.Figure` directly; colours come from `plotstyle.SERIES`. A figure never has two
y-axes: when two scales are needed it has two panels side by side.
"""
from __future__ import annotations

from matplotlib.figure import Figure

from src import plotstyle as style
from src.plots import DAY_TYPES
from src.plotstyle import SERIES

style.apply()

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
