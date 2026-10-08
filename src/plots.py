"""Figures for the notebook and report (matplotlib only, no seaborn; one question per figure).

Figures are built with `matplotlib.figure.Figure` directly (no pyplot), so they work headless (Agg) and display
inline in a notebook when returned from a cell. Every function returns the Figure.
"""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERMILLION, PURPLE, SKY, BLACK = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
                                                      "#56B4E9", "#000000")
SWEEP_COLOURS = (BLUE, SKY, GREEN, ORANGE, VERMILLION, PURPLE)
DAY_TYPES = ((1, "working day", BLUE), (0, "non-working day", ORANGE))
FIG_SIZE = (7.0, 4.2)


def _new_axes(title: str, xlabel: str, ylabel: str):
    """A fresh figure with one labelled axes."""
    fig = Figure(figsize=FIG_SIZE, layout="constrained")
    ax = fig.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return fig, ax


def plot_demand_profile(train_df) -> Figure:
    """Question: how does mean hourly demand differ between working and non-working days?"""
    fig, ax = _new_axes("Mean hourly demand by day type", "hour of day", "mean bikes per hour")
    for flag, label, colour in DAY_TYPES:
        profile = train_df[train_df["workingday"] == flag].groupby("hr")["cnt"].mean()
        ax.plot(profile.index, profile.values, marker="o", ms=3, color=colour, label=label)
    ax.legend()
    return fig


def plot_lr_sweep(p1: dict) -> Figure:
    """Question: which learning rates converge, crawl or diverge? Diverged runs are dashed."""
    fig, ax = _new_axes("Training loss for different learning rates", "iteration", "training loss (log scale)")
    for run, colour in zip(p1["lr_sweep"], SWEEP_COLOURS):
        points = [(i, v) for i, v in run["loss_curve"] if v is not None and v > 0]
        if not points:
            continue
        xs, ys = zip(*points)
        style = "--" if run["stop_reason"] == "diverged" else "-"
        ax.plot(xs, ys, style, color=colour, label=f"{run['fraction']:g} x bound ({run['stop_reason']})")
    ax.set_yscale("log")
    ax.set_xscale("symlog", linthresh=10)
    ax.legend(fontsize=8)
    return fig


def plot_loss_curve(art: dict) -> Figure:
    """Question: does the loss of the final run fall smoothly, and where does it stop?"""
    fig, ax = _new_axes(f"Training loss, stopped '{art['stop_reason']}' after {art['iterations']} iterations",
                        "iteration", "training loss (log scale)")
    xs, ys = zip(*[(i, v) for i, v in art["loss_curve"] if v is not None])
    ax.plot(xs, ys, color=BLUE, label="training loss")
    ax.axhline(art["train_loss_final"], color=VERMILLION, ls=":", label="final loss")
    ax.set_yscale("log")
    ax.legend()
    return fig


def _by_hour(profile: dict, flag: int) -> tuple[list, list]:
    cells = sorted((c for c in profile["by_wd_hr"] if c["workingday"] == flag), key=lambda c: c["hr"])
    return [c["hr"] for c in cells], [c["mean_resid"] for c in cells]


def plot_residual_profile(art_before: dict, art_after: dict | None = None) -> Figure:
    """Question: does the residual on the target scale z show a pattern by hour and day type? Dashed lines = after (if given)."""
    fig, ax = _new_axes("Mean residual of the target z by hour (train rows)", "hour of day", "mean residual of z")
    ax.axhline(0.0, color=BLACK, lw=0.8)
    for flag, label, colour in DAY_TYPES:
        hours, means = _by_hour(art_before["residual_profile"], flag)
        ax.plot(hours, means, "-o", ms=3, color=colour, label=f"{label}" + (" (before)" if art_after else ""))
        if art_after is not None:
            hours, means = _by_hour(art_after["residual_profile"], flag)
            ax.plot(hours, means, "--s", ms=3, color=colour, label=f"{label} (after)")
    ax.legend(fontsize=8)
    return fig


def plot_degree_sweep(p2: dict) -> Figure:
    """Question: does a higher polynomial degree still improve train and validation R2?"""
    fig, ax = _new_axes("R2 against polynomial degree", "degree", "R2 on bikes")
    rows = sorted(p2["degree_sweep"], key=lambda r: r["degree"])
    degrees = [r["degree"] for r in rows]
    ax.plot(degrees, [r["train_r2"] for r in rows], "-o", color=BLUE, label="train")
    ax.plot(degrees, [r["val_r2"] for r in rows], "-s", color=VERMILLION, label="validation")
    ax.axvline(p2["degree"], color=BLACK, ls=":", label=f"chosen degree {p2['degree']}")
    ax.set_xticks(degrees)
    ax.legend()
    return fig


def plot_bonus_shift(p1: dict) -> Figure:
    """Question: where does the asymmetric cost move the predictions? Working days, mean validation prediction by hour."""
    fig, ax = _new_axes("Effect of penalising under-prediction (working days)", "hour of day",
                        "mean predicted bikes per hour")
    cells = sorted((c for c in p1["bonus"]["by_wd_hr_shift"] if c["workingday"] == 1), key=lambda c: c["hr"])
    hours = np.array([c["hr"] for c in cells])
    ax.plot(hours, [c["mean_pred_mse"] for c in cells], "-o", ms=3, color=BLUE, label="squared error (MSE)")
    ax.plot(hours, [c["mean_pred_asym"] for c in cells], "-s", ms=3, color=VERMILLION,
            label=f"asymmetric, k = {p1['bonus']['k']:g}")
    ax.legend()
    return fig


def plot_target_power(p1: dict, p3: dict | None = None) -> Figure:
    """Question: which target exponent scores best, for the Phase 1 design and (if p3 is given) for the target design?"""
    fig, ax = _new_axes("Validation R2 against the target exponent", "exponent of the target transform (0 = log)",
                        "R2 on bikes")
    table = p1["target_power_table"]
    ax.plot([r["power"] for r in table], [r["val_r2"] for r in table], "-o", ms=4, color=BLUE,
            label="Phase 1 design (validation)")
    if p3 is not None:
        check = p3["target_power_check"]
        for key, label, colour, marker in (("seeded_r2", "target design, seeded", VERMILLION, "s"),
                                           ("day_block_r2", "target design, held-out days", GREEN, "^"),
                                           ("chrono_r2", "target design, chronological", ORANGE, "d")):
            ax.plot([r["power"] for r in check], [r[key] for r in check], "-" + marker, ms=4, color=colour, label=label)
    ax.axvline(p1["target_power"], color=BLACK, ls=":", label=f"exponent used {p1['target_power']:g}")
    ax.legend(fontsize=8)
    return fig


def show(fig):
    """Display a matplotlib figure as a PNG in the notebook (works with any backend) and return nothing."""
    import io

    from IPython.display import Image, display
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    display(Image(buffer.getvalue()))

