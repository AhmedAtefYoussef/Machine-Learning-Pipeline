"""Figures for Phases 1 and 2 and the closing summary (matplotlib only; one question per figure).

Figures are built with `matplotlib.figure.Figure` directly (no pyplot), so they work headless and display inline in a
notebook through `show`. Every plot function returns the Figure. Colours come from `plotstyle.SERIES`.
"""
from __future__ import annotations

import io

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


def show(fig):
    """Display a matplotlib figure as a PNG in the notebook (works with any backend) and return nothing."""
    from IPython.display import Image, display
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    display(Image(buffer.getvalue()))
