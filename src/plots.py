"""Figures for Phases 1 and 2 and the closing summary (matplotlib only; one question per figure).

Figures are built with `matplotlib.figure.Figure` directly (no pyplot), so they work headless and display inline in a
notebook through `show`. Every plot function returns the Figure. Colours come from `plotstyle.SERIES`.
"""
from __future__ import annotations

import io

import numpy as np
from matplotlib.figure import Figure

from src import plotstyle as style
from src.plotstyle import SERIES

style.apply()

DAY_TYPES = ((1, "working day", SERIES["working"]), (0, "non-working day", SERIES["non_working"]))


def plot_demand_profile(train_df) -> Figure:
    """Question: how does mean hourly demand differ between working and non-working days?"""
    fig, ax = style.new_axes("Mean hourly demand by day type", "hour of day", "bikes per hour")
    lines = []
    for flag, label, colour in DAY_TYPES:
        profile = train_df[train_df["workingday"] == flag].groupby("hr")["cnt"].mean()
        lines.append(ax.plot(profile.index, profile.values, marker="o", ms=4, color=colour, label=label)[0])
    style.end_labels(ax, lines, [label for _, label, _ in DAY_TYPES])
    return fig


def plot_lr_sweep(p1: dict) -> Figure:
    """Question: which learning rates converge, crawl or diverge? Diverged runs are dashed."""
    fig, ax = style.new_axes("Which learning rates converge, crawl or diverge?", "iteration (symlog scale)",
                             "training loss (log scale)")
    for run, colour in zip(p1["lr_sweep"], style.PALETTE):
        points = [(i, v) for i, v in run["loss_curve"] if v is not None and v > 0]
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, "--" if run["stop_reason"] == "diverged" else "-", color=colour,
                label=f"{run['fraction']:g} x bound ({run['stop_reason']})")
    ax.set_yscale("log")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(left=0)
    style.legend(ax, ncol=2)
    return fig


def plot_loss_curve(art: dict) -> Figure:
    """Question: does the loss of the final run fall smoothly, and where does it stop?"""
    fig, ax = style.new_axes(f"Training loss, stopped '{art['stop_reason']}' after {art['iterations']} iterations",
                             "iteration", "training loss (log scale)")
    xs, ys = zip(*[(i, v) for i, v in art["loss_curve"] if v is not None])
    ax.plot(xs, ys, color=SERIES["train"])
    style.baseline(ax, art["train_loss_final"], "final loss", side="left")
    ax.set_yscale("log")
    return fig


def _by_hour(profile: dict, flag: int) -> tuple[list, list]:
    cells = sorted((c for c in profile["by_wd_hr"] if c["workingday"] == flag), key=lambda c: c["hr"])
    return [c["hr"] for c in cells], [c["mean_resid"] for c in cells]


def plot_residual_profile(art_before: dict, art_after: dict | None = None) -> Figure:
    """Question: does the residual on the target scale z show a pattern by hour and day type? Dashed lines = after (if given)."""
    fig, ax = style.new_axes("Does the residual of the target z follow a pattern by hour and day type?", "hour of day",
                             "mean residual of z (training rows)")
    ax.axhline(0.0, color=style.MUTED, lw=0.8)
    for flag, label, colour in DAY_TYPES:
        tag = " (before)" if art_after else ""
        hours, means = _by_hour(art_before["residual_profile"], flag)
        ax.plot(hours, means, "-o", ms=4, color=colour, label=label + tag)
        if art_after is not None:
            hours, means = _by_hour(art_after["residual_profile"], flag)
            ax.plot(hours, means, "--s", ms=4, color=colour, label=f"{label} (after)")
    style.legend(ax, ncol=2)
    return fig


def plot_degree_sweep(p2: dict) -> Figure:
    """Question: does a higher polynomial degree still improve train and validation R2?"""
    fig, ax = style.new_axes("Does a higher degree still improve the fit?", "polynomial degree", "R² on bikes")
    rows = sorted(p2["degree_sweep"], key=lambda r: r["degree"])
    degrees = [r["degree"] for r in rows]
    lines = [ax.plot(degrees, [r["train_r2"] for r in rows], "-o", color=SERIES["train"], label="train")[0],
             ax.plot(degrees, [r["val_r2"] for r in rows], "-s", color=SERIES["validation"], label="validation")[0]]
    style.mark(ax, p2["degree"], f"chosen degree {p2['degree']}")
    ax.set_xticks(degrees)
    style.end_labels(ax, lines, ["train", "validation"])
    return fig


def plot_bonus_shift(p1: dict) -> Figure:
    """Question: where does the asymmetric cost move the predictions? Working days, mean validation prediction by hour."""
    fig, ax = style.new_axes("Where does penalising under-prediction move the forecast? (working days)", "hour of day",
                             "mean predicted bikes per hour")
    cells = sorted((c for c in p1["bonus"]["by_wd_hr_shift"] if c["workingday"] == 1), key=lambda c: c["hr"])
    hours = np.array([c["hr"] for c in cells])
    lines = [ax.plot(hours, [c["mean_pred_mse"] for c in cells], "-o", ms=4, color=SERIES["mse"],
                     label="squared error (MSE)")[0],
             ax.plot(hours, [c["mean_pred_asym"] for c in cells], "-s", ms=4, color=SERIES["asymmetric"],
                     label=f"asymmetric, k = {p1['bonus']['k']:g}")[0]]
    style.end_labels(ax, lines, ["MSE", "asymmetric"])
    return fig


def plot_target_power(p1: dict, p3: dict | None = None) -> Figure:
    """Question: which target exponent scores best, for the Phase 1 design and (if p3 is given) for the target design?"""
    fig, ax = style.new_axes("Which exponent of the target transform scores best?",
                             "exponent of the target transform (0 = log)", "R² on bikes")
    table = p1["target_power_table"]
    lines = [ax.plot([r["power"] for r in table], [r["val_r2"] for r in table], "-o", ms=5, color=SERIES["first"],
                     label="Phase 1 design (validation)")[0]]
    labels = ["Phase 1 design"]
    if p3 is not None:
        check = p3["target_power_check"]
        for key, label, series, marker in (("seeded_r2", "target design, validation", "validation", "s"),
                                           ("day_block_r2", "target design, held-out days", "held_out_days", "^"),
                                           ("chrono_r2", "target design, chronological", "chronological", "d")):
            lines.append(ax.plot([r["power"] for r in check], [r[key] for r in check], "-" + marker, ms=5,
                                 color=SERIES[series], label=label)[0])
            labels.append(label.replace("target design, ", ""))
    style.mark(ax, p1["target_power"], f"exponent used {p1['target_power']:g}")
    style.end_labels(ax, lines, labels)
    return fig


def plot_pipeline_summary(p1: dict, p2: dict, p3: dict, p4: dict) -> Figure:
    """Question: how good is the model each phase ends with? Validation R2 on bikes, one bar per phase."""
    fig, ax = style.new_axes("How good is the model each phase ends with?", "validation R² on bikes", "",
                             figsize=(7.2, 2.8))
    names = ["Phase 1: gradient descent", "Phase 2: polynomial", "Phase 3: bias-variance", "Phase 4: regularization"]
    scores = [p1["val_r2"], p2["val_r2"], p3["estimates_target"]["seeded"]["r2"], p4["recommended"]["val_r2"]]
    ax.barh(names, scores, color=SERIES["validation"], height=0.6)
    for y, score in enumerate(scores):
        ax.annotate(f"{score:.3f}", (score, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9,
                    color=style.INK_SECONDARY)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.grid(False)
    return fig


def show(fig):
    """Display a matplotlib figure as a PNG in the notebook (works with any backend) and return nothing."""
    from IPython.display import Image, display
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    display(Image(buffer.getvalue()))
