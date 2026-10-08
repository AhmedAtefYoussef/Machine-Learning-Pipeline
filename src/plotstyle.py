"""One chart style for the whole notebook: a fixed palette, one colour per entity, and a few small helpers.

Importing a plot module calls `apply()`, so every figure gets the same look.

Labelling rule: a figure with at most four series whose line ends are well separated names them with `end_labels`
and has no legend box; every other multi-series figure has a `legend` under the axes and no end labels. Dotted
reference lines are named with `baseline`, `mark` or `note`, never with a legend entry. Figures are built with
`matplotlib.figure.Figure` (no pyplot), which keeps them headless-safe.
"""
from __future__ import annotations

import matplotlib as mpl
from cycler import cycler
from matplotlib.figure import Figure
from matplotlib.transforms import blended_transform_factory

# Validated for colour-blind separation; used in this order, never cycled or reordered.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK_SECONDARY, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#ffffff"
LIGHT_GREY = "#d9d8d4"

# Colour follows the entity: the same name has the same colour in every figure.
SERIES = {
    "train": PALETTE[0], "validation": PALETTE[1], "held_out_days": PALETTE[2], "chronological": PALETTE[3],
    "working": PALETTE[0], "non_working": PALETTE[1],
    "ridge": PALETTE[0], "lasso": PALETTE[1], "enet": PALETTE[2],
    "first": PALETTE[0], "second": PALETTE[1],
    "mse": PALETTE[0], "asymmetric": PALETTE[1],
    "anchor": PALETTE[4], "target": PALETTE[5], "top": PALETTE[3],
    "drop_alone": PALETTE[0], "drop_group": PALETTE[3],
    "precision": PALETTE[0], "recall": PALETTE[2], "f1": PALETTE[4], "cost": PALETTE[3],
    "hour_only": PALETTE[3], "trend_only": PALETTE[4], "all_columns": PALETTE[0],
}
DEFAULT_SIZE = (7.2, 4.0)


def apply() -> None:
    """Set the matplotlib defaults of the notebook (white background, quiet grid, no frame, readable sizes)."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.figsize": DEFAULT_SIZE, "figure.dpi": 110, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": 600, "axes.titlelocation": "left",
        "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": LIGHT_GREY, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED, "text.color": INK,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#ecebe7", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 7,
        "legend.frameon": False, "axes.prop_cycle": cycler(color=PALETTE),
    })


def label_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Title (the question the figure answers) and axis labels with units."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def new_axes(title: str, xlabel: str, ylabel: str, figsize=None):
    """A fresh figure with one labelled axes."""
    fig = Figure(figsize=figsize or DEFAULT_SIZE, layout="constrained")
    ax = fig.subplots()
    label_axes(ax, title, xlabel, ylabel)
    return fig, ax


def legend(ax, ncol: int = 3) -> None:
    """Legend below the axes (the constrained layout makes room), so that it never covers a line."""
    ax.figure.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=ncol, fontsize=9)


def end_labels(ax, lines, labels, min_gap: float = 0.055) -> None:
    """Write each label just right of the last point of its line, pushed apart where they would touch."""
    ax.autoscale_view()   # the limits must be final before positions are read from them
    to_axes = ax.transData + ax.transAxes.inverted()
    ends = [(to_axes.transform((line.get_xdata()[-1], line.get_ydata()[-1]))[1], line, label)
            for line, label in zip(lines, labels)]
    ends.sort(key=lambda end: end[0])
    placed = []
    for y, line, label in ends:
        y = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y)
        ax.annotate(label, (line.get_xdata()[-1], y), xycoords=blended_transform_factory(ax.transData, ax.transAxes),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK_SECONDARY,
                    annotation_clip=False)


def mark(ax, x, text: str, level: int = 0) -> None:
    """A thin dotted vertical marker with a small grey label (level staggers labels of nearby markers)."""
    ax.axvline(x, color=MUTED, ls=":", lw=1)
    ax.annotate(text, (x, 0.03 + 0.07 * level), xycoords=blended_transform_factory(ax.transData, ax.transAxes),
                xytext=(4, 0), textcoords="offset points", va="bottom", fontsize=8, color=MUTED)


def note(ax, xy, text: str, offset=(6, -12)) -> None:
    """A small grey note next to a point, to name a dotted reference line."""
    ax.annotate(text, xy, xytext=offset, textcoords="offset points", fontsize=8, color=MUTED)


def baseline(ax, y, text: str, side: str = "right") -> None:
    """A thin dotted horizontal reference line with a small grey label at its left or right end."""
    left = side == "left"
    ax.axhline(y, color=MUTED, ls=":", lw=1)
    ax.annotate(text, (0.01 if left else 0.99, y), xycoords=blended_transform_factory(ax.transAxes, ax.transData),
                xytext=(0, -3 if left else 3), textcoords="offset points", ha="left" if left else "right",
                va="top" if left else "bottom", fontsize=8,
                color=MUTED)
