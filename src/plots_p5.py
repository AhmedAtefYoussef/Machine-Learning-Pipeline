"""Phase 5 and 6 figures. The p5 functions take the payload of artifacts/p5.json; one question per figure.

Figures are built with `matplotlib.figure.Figure` directly (no pyplot), colour-blind-safe Okabe-Ito colours.
"""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from src.plots import BLACK, BLUE, DAY_TYPES, GREEN, ORANGE, PURPLE, SKY, VERMILLION, _new_axes

RULE_LABELS = {"global": "one global threshold", "workingday+hr": "per (day type, hour)",
               "yr+workingday+hr": "per (year, day type, hour)"}


def plot_roc_pr(p5: dict) -> Figure:
    """Question: how well does the classifier rank hours, and what do precision and recall look like along the way?"""
    fig = Figure(figsize=(9.0, 4.2), layout="constrained")
    roc_ax, pr_ax = fig.subplots(1, 2)
    points = p5["curve_points"]
    roc_ax.plot([0, 1], [0, 1], ":", color=BLACK, label="chance")
    roc_ax.plot([q["fpr"] for q in points], [q["recall"] for q in points], color=BLUE,
                label=f"ROC-AUC {p5['metrics']['roc_auc']:.3f}")
    pr_ax.axhline(p5["class_balance"]["val"], ls=":", color=BLACK, label="share of positives")
    pr_ax.plot([q["recall"] for q in points], [q["precision"] for q in points], color=VERMILLION,
               label=f"PR-AUC {p5['metrics']['pr_auc']:.3f}")
    for ax, (x, y, title) in ((roc_ax, ("false positive rate", "recall", "ROC curve (validation)")),
                              (pr_ax, ("recall", "precision", "Precision-recall curve (validation)"))):
        ax.set(xlabel=x, ylabel=y, title=title, xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right" if ax is roc_ax else "upper right", fontsize=8)
    return fig


def plot_threshold_curve(p5: dict) -> Figure:
    """Question: where should the alarm threshold sit when a miss costs more than a false alarm?"""
    fig, ax = _new_axes("Validation scores against the alarm threshold", "threshold on the predicted probability",
                        "precision, recall, F1")
    rows = p5["threshold_curve"]
    thr = [r["thr"] for r in rows]
    for key, colour, label in (("precision", BLUE, "precision"), ("recall", GREEN, "recall"), ("f1", PURPLE, "F1")):
        ax.plot(thr, [r[key] for r in rows], "-o", ms=3, color=colour, label=label)
    cost_ax = ax.twinx()
    cost_ax.plot(thr, [r["cost"] for r in rows], "--s", ms=3, color=VERMILLION,
                 label=f"cost per row (miss = {p5['cost_ratio']:g} x false alarm)")
    cost_ax.set_ylabel("cost per row")
    ax.axvline(p5["t_cost"], color=BLACK, ls=":", label=f"1/(1+c) = {p5['t_cost']:.2f}")
    ax.axvline(0.5, color=SKY, ls=":", label="0.5")
    handles = ax.get_legend_handles_labels()[0] + cost_ax.get_legend_handles_labels()[0]
    ax.legend(handles=handles, fontsize=7, loc="lower center", ncols=2)
    return fig


def plot_calibration(p5: dict) -> Figure:
    """Question: when the model says p, is the share of high-demand hours about p?"""
    fig, ax = _new_axes(f"Calibration on validation (largest gap {p5['calibration_max_gap']:.3f})",
                        "mean predicted probability in the bin", "observed share of high-demand hours")
    rows = [r for r in p5["calibration"] if r["n"] > 0]
    ax.plot([0, 1], [0, 1], ":", color=BLACK, label="perfect calibration")
    ax.plot([r["mean_p"] for r in rows], [r["frac_pos"] for r in rows], "-o", color=BLUE, label="validation bins")
    for r in rows:
        ax.annotate(str(r["n"]), (r["mean_p"], r["frac_pos"]), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    return fig


def plot_label_variants(p5: dict) -> Figure:
    """Question: how much of a high ROC-AUC is just the clock or the calendar, for each label rule?"""
    fig, ax = _new_axes("Validation ROC-AUC of three models under each label rule", "label rule", "ROC-AUC")
    variants = p5["label_variants"]
    x = np.arange(len(variants))
    for k, (key, colour, label) in enumerate((("auc_hour_only", ORANGE, "hour of day only"),
                                              ("auc_trend_only", SKY, "trend only"),
                                              ("auc_full", BLUE, "all surviving columns"))):
        values = [v[key] for v in variants]
        ax.bar(x + (k - 1) * 0.27, values, 0.27, color=colour, label=label)
        for xi, v in zip(x + (k - 1) * 0.27, values):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(0.5, color=BLACK, ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([RULE_LABELS.get("+".join(v["group_by"]) or "global", "+".join(v["group_by"]))
                        for v in variants], fontsize=8)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    return fig


def plot_test_profile(train_df, test_df, sub) -> Figure:
    """Question: does the predicted demand on the hidden days follow the hourly shape of the training days?"""
    fig, ax = _new_axes("Mean bikes per hour: training days and predicted hidden days", "hour of day",
                        "mean bikes per hour")
    predicted = test_df.merge(sub, on="instant")
    for flag, label, colour in DAY_TYPES:
        seen = train_df[train_df["workingday"] == flag].groupby("hr")["cnt"].mean()
        ahead = predicted[predicted["workingday"] == flag].groupby("hr")["cnt"].mean()
        ax.plot(seen.index, seen.values, "-o", ms=3, color=colour, label=f"{label}, training")
        ax.plot(ahead.index, ahead.values, "--s", ms=3, color=colour, label=f"{label}, predicted")
    ax.legend(fontsize=8)
    return fig
