"""Draw the report figures from the artifacts (no computation here). Usage: python -X utf8 report/make_figs.py"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
FIGS = HERE / "figs"
COLORS = {"train": "#0072B2", "val": "#D55E00", "days": "#009E73", "chrono": "#CC79A7", "grey": "#666666"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def load(name: str) -> dict:
    return json.loads((ART / f"{name}.json").read_text(encoding="utf-8"))


def fig_lr_sweep(p1: dict) -> None:
    """Loss against iteration for each learning rate of the Phase 1 sweep."""
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for run in p1["lr_sweep"]:
        pts = [(i, v) for i, v in run["loss_curve"] if v is not None and v > 0]
        style = "--" if run["stop_reason"] == "diverged" else "-"
        ax.plot([p[0] + 1 for p in pts], [p[1] for p in pts], style, lw=1.1,
                label=f"{run['fraction']:g} x bound ({run['stop_reason']})")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.1, 1e3)
    ax.set_xlabel("iteration"); ax.set_ylabel("training loss (MSE/2, log scale)")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "lr_sweep.png", dpi=200); plt.close(fig)


def fig_ladder(p3: dict) -> None:
    """Train and the three held-out R2 estimates along the complexity ladder."""
    rows = [r for r in p3["ladder"] if r["index"] >= 1]
    x = [r["n_features"] for r in rows]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.plot(x, [r["seeded"]["train_r2"] for r in rows], "o-", ms=3, lw=1.1, color=COLORS["train"], label="train")
    ax.plot(x, [r["seeded"]["r2"] for r in rows], "o-", ms=3, lw=1.1, color=COLORS["val"], label="validation (seeded)")
    ax.plot(x, [r["day_block"]["r2"] for r in rows], "s-", ms=3, lw=1.1, color=COLORS["days"], label="held-out days")
    ax.plot(x, [r["chrono"]["r2"] for r in rows], "^-", ms=3, lw=1.1, color=COLORS["chrono"], label="chronological")
    for key, text in (("anchor", "Phase 2"), ("target", "target")):
        idx = p3["anchor"]["anchor_index"] if key == "anchor" else p3["target_complexity"]["index"]
        n = p3["ladder"][idx]["n_features"]
        ax.axvline(n, color=COLORS["grey"], lw=0.6, ls=":")
        ax.text(n, 0.958, text, rotation=90, va="top", ha="right", fontsize=6, color=COLORS["grey"])
    ax.set_xscale("log"); ax.set_ylim(0.70, 0.96)
    ax.set_xlabel("number of weights (log scale)"); ax.set_ylabel("R² on bikes")
    ax.legend(fontsize=6, frameon=False, loc="lower center")
    fig.tight_layout(); fig.savefig(FIGS / "ladder.png", dpi=200); plt.close(fig)


def fig_verdicts(p4: dict) -> None:
    """Validation R2 lost when a column is dropped alone and with its copies."""
    cols = list(p4["column_verdicts"])
    cols.sort(key=lambda c: p4["column_verdicts"][c]["numbers"]["drop_alone"]["delta"])
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for i, c in enumerate(cols):
        n = p4["column_verdicts"][c]["numbers"]
        group = n["drop_group"]["delta"] if n["drop_group"] else None
        if group is not None:
            ax.barh(i + 0.2, max(group, 1e-5), height=0.38, color=COLORS["days"], label="with its copies" if i == 0 else None)
        ax.barh(i - 0.2, max(n["drop_alone"]["delta"], 1e-5), height=0.38, color=COLORS["val"],
                label="alone" if i == 0 else None)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([f"{c}: {p4['column_verdicts'][c]['verdict']}" for c in cols], fontsize=6)
    ax.set_xscale("log"); ax.set_xlim(1e-5, 1)
    ax.axvline(p4["verdict_thresholds"]["min_delta"], color=COLORS["grey"], lw=0.6, ls=":")
    ax.set_xlabel("validation R² lost when dropped (log scale)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGS / "verdicts.png", dpi=200); plt.close(fig)


def fig_threshold(p5: dict) -> None:
    """Precision, recall, F1 and operator cost against the probability cut-off."""
    curve = p5["threshold_curve"]
    x = [r["thr"] for r in curve]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for key, color in (("recall", COLORS["val"]), ("precision", COLORS["train"]), ("f1", COLORS["days"])):
        ax.plot(x, [r[key] for r in curve], lw=1.1, color=color, label=key)
    ax.plot(x, [r["cost"] for r in curve], lw=1.1, color=COLORS["chrono"], ls="--", label="cost per hour (3 x miss + false alarm)")
    ax.axvline(p5["t_cost"], color=COLORS["grey"], lw=0.6, ls=":"); ax.axvline(0.5, color=COLORS["grey"], lw=0.6, ls=":")
    ax.set_xlabel("probability cut-off"); ax.set_ylabel("value on validation rows"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(FIGS / "threshold.png", dpi=200); plt.close(fig)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    fig_lr_sweep(load("p1")); fig_ladder(load("p3")); fig_verdicts(load("p4")); fig_threshold(load("p5"))
    print("make_figs: 4 figures in report/figs")


if __name__ == "__main__":
    main()
