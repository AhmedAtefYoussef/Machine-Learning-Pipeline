# %% [markdown]
# ## Expectation — Phase 3: Bias–Variance Tradeoff
#
# *Written before running the phase (see the git history of this file).*
#
# **What we do.** We take the Phase 2 design (its degree and feature list, loaded from `artifacts/p2.json`) as the anchor and move complexity both below and above it on a nested ladder of ten designs, from "no hour information" (13 weights) up to "month × working day × hour" (about 900 weights). We also vary the pure polynomial degree, and the amount of training data. Every design is scored three ways: on our seeded validation set, on whole held-out days inside the training portion, and on one chronological split (train before 1 July 2012, validate after).
#
# **What we expect, and why.**
#
# 1. *Phase 2 is not over-fit.* Its training and validation R² should differ by less than 0.01. If anything we expect it to be slightly under-fit: letting the weather effect depend on the hour should still raise validation R² by 0.01–0.02.
# 2. *Degree alone will look flat.* Raising the power of temperature and humidity from 3 to 8 should move neither curve by more than 0.002. A flat line is not evidence of a good fit, which is why we need the wider ladder.
# 3. *Variance appears late.* On the ladder we expect validation R² to peak somewhere around 250–450 weights and then fall while training R² keeps rising. The first clearly harmful levels should be the month × hour blocks, where many cells hold only a handful of training rows.
# 4. *Learning curves.* At Phase 2 complexity, validation R² should stop improving after roughly half the training days (a bias signature). At the top of the ladder the train–validation gap should be visibly wider with little data and shrink as data is added (a variance signature).
# 5. *The time question.* We expect the seeded estimate and the held-out-day estimate to be almost equal (within 0.005). Our model has no way to memorise a particular day, so sharing days between training and validation should hardly help it. The chronological estimate should be clearly lower, by 0.04–0.08, because the model has to extrapolate a growth trend into six months it has never seen, having observed July–December only once.
# 6. *Which estimate to trust.* For the hidden test set, which is the 20th of each month and so lies inside the observed period, the held-out-day estimate is the right one. For genuinely future months the chronological estimate is the honest one. We expect the diagnosis to stay "bias first, variance only at the top of the ladder" under all three, but the chronological split should prefer a simpler model than the seeded split does, especially disliking the blocks that let the trend vary by hour.

# %% [markdown]
# ### Justification: what we varied
#
# TODO(chief)

# %%
# Phase 3 runs everything from artifacts/p1.json and artifacts/p2.json; CFG comes from the setup notebook.
import pandas as pd

from src.common import load_config, read_artifact
from src.phases import p3 as p3mod
from src.plots_p3 import plot_degree_axis, plot_ladder, plot_learning_curves, plot_three_estimates

CFG = globals().get("CFG") or load_config()
p3 = p3mod.run(CFG)  # writes artifacts/p3.json; about 10 seconds

# %%
# The chain: the anchor is exactly the Phase 2 design, and our closed-form fit must reproduce its validation R2.
p2 = read_artifact("p2")
print("Phase 2 degree:", p2["degree"], "| features (with bias):", len(p2["feature_names"]))
print("Phase 2 power columns:", p2["power_cols"], "| blocks:", p2["blocks"])
print("Phase 2 validation R2 (gradient descent):", round(p2["val_r2"], 5))
print("anchor validation R2 (closed form here): ", round(p3["ladder"][p3["anchor"]["anchor_index"]]["seeded"]["r2"], 5))
print("anchor_check (closed form - gradient descent):", f"{p3['anchor_check']:.2e}")
print("back-transform method reused from Phase 1:", p3["back_method"])

# %%
# Complexity ladder: three validators per level.
ladder = pd.DataFrame([{"level": r["name"], "weights": r["n_features"], "train_r2": r["seeded"]["train_r2"],
                        "seeded_r2": r["seeded"]["r2"], "gap": r["gap_seeded"], "day_block_r2": r["day_block"]["r2"],
                        "day_block_sd": r["day_block"]["sd"], "chrono_r2": r["chrono"]["r2"]} for r in p3["ladder"]])
print(ladder.round(4).to_string(index=False))
print("target level:", p3["target_level"], "| best on held-out days:", p3["best_level_day_block"],
      "| best chronologically:", p3["best_level_chrono"], "| over-fit from level:", p3["overfit_from_level"])

# %%
fig = plot_ladder(p3)

# %%
# Degree axis: anchor blocks and power columns fixed, only the degree changes.
degrees = pd.DataFrame([{"degree": r["degree"], "weights": r["n_features"], "train_r2": r["seeded"]["train_r2"],
                         "seeded_r2": r["seeded"]["r2"], "day_block_r2": r["day_block"]["r2"],
                         "chrono_r2": r["chrono"]["r2"]} for r in p3["degree_axis"]])
print(degrees.round(4).to_string(index=False))
print("range of seeded R2 over degrees >= 2:", round(p3["diagnosis"]["degree_axis_range"], 5))

# %%
fig = plot_degree_axis(p3)

# %%
# Learning curves for the anchor, the target and the top of the ladder (clipped at R2 = -0.2 in the plot).
curves = pd.DataFrame([{"design": key, **c} for key, curve in p3["learning_curves"].items() for c in curve])
print(curves.round(4).to_string(index=False))

# %%
fig = plot_learning_curves(p3)

# %% [markdown]
# ### Justification: the chronological cut
#
# TODO(chief)

# %%
# Three estimates for the anchor and the target, their gaps, and the bootstrap interval of the seeded estimate.
rows = []
for label, est, gaps in (("anchor", p3["estimates"], p3["gaps"]["anchor"]),
                         ("target", p3["estimates_target"], p3["gaps"]["target"])):
    rows.append({"design": label, "seeded": est["seeded"]["r2"], "ci_lo": est["seeded"]["lo"],
                 "ci_hi": est["seeded"]["hi"], "day_holdout": est["day_holdout"]["r2"],
                 "day_sd": est["day_holdout"]["sd"], "chrono": est["chrono"]["r2"],
                 "leakage_gap": gaps["leakage"], "drift_gap": gaps["drift"]})
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("paired gain target - anchor (seeded):", {k: round(v, 4) for k, v in p3["paired_target_vs_anchor"].items()})

# %%
fig = plot_three_estimates(p3)

# %% [markdown]
# ### Justification: which estimate to trust
#
# TODO(chief)

# %%
# Chronological detail: the late period has a much higher level of demand (growth) than the early one.
detail = pd.DataFrame(p3["chrono_detail"]).T
print(detail.to_string())
print("r2_level_corrected rescales late predictions by the late period's own mean: a diagnostic of level drift only, "
      "not a usable estimate.")

# %%
# How much variance is left once the design cells are fully known? (an optimistic ceiling)
for key, value in p3["noise_floor"].items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

# %%
# Diagnosis and target complexity.
for key, value in p3["diagnosis"].items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
target = p3["target_complexity"]
print("target:", target["level"], "| weights:", target["n_features"], "| degree:", target["degree"],
      "| power columns:", target["power_cols"])
print("blocks:", target["blocks"])

# %% [markdown]
# ### Justification: target complexity
#
# TODO(chief)

# %% [markdown]
# ## Outcome — Phase 3
#
# TODO(chief)
