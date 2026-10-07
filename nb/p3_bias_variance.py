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
# - **A ladder of nested designs, anchored at Phase 2.** Level C3 is exactly the Phase 2 design, read from `artifacts/p2.json`. Below it we remove what Phase 2 added (C2: no powers; C1: no interactions; C0: no hour at all). Above it we add blocks: weather effects that depend on the hour (C4), a separate hour profile for each weekday (C5), weather detail and weather memory (C6, see the note below), finer weather interactions (C7), hour-specific season and trend (C8), and finally month × hour and month × day type × hour (C9, C10), where many cells contain only a few training rows. Bias shows as "training and validation both low, both rise when we add a block"; variance shows as "training keeps rising, validation falls".
# - **The degree alone**, from 1 to 8, with everything else as in Phase 2, because the task asks about the degree; we expected this axis to be flat.
# - **The amount of training data** (10% to 100% of the training days) for the anchor, the target and the top level: more data cures variance but not bias.
# - **Three ways of scoring** each design: the seeded validation set; five folds of whole held-out days inside the training portion (folds are fixed by the calendar, not by a random seed); and one chronological split.
# - **How the sweeps are fitted.** These several hundred fits use the closed-form least-squares solution, which our gradient descent reaches to within 1e-5 in Phases 1 and 2 (the chain cell below checks it again on the anchor); the largest levels would need tens of thousands of iterations each. A tiny ridge term (1e-8) keeps the solution defined where columns are exact copies of each other (`holiday` is determined by the weekday dummies and `workingday`).
# - **Note on level C6.** This level was not in our first ladder. After the first complete run we looked at where the C5 model's errors were (working-day rush hours and rainy hours carried about half and a fifth of the squared error) and added the terms that address them: squares and cubes of humidity, wind terms, a temperature effect per working-day hour, and the weather situation one hour earlier and the worst of the previous three hours (wet roads keep riders away after the rain has stopped). These last columns use the weather of neighbouring rows, never their counts. Because this level was designed after looking at validation errors, we judge it mainly by the two estimates that were not used to design it.

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
# We train on every date before {{p3.cut_date}} ({{p3.chrono_detail.anchor.n_early}} rows) and validate on every date from then on ({{p3.chrono_detail.anchor.n_late}} rows, about a quarter of the file). We put the cut there for three reasons: the held-out part is about the same size as our seeded validation set, so the two estimates are comparably precise; it covers half a seasonal cycle (summer to winter) instead of one season; and the training part still contains one full year plus the first half of the second, so the model has seen each month at least once. A later cut would test only autumn and winter; an earlier one would leave too little of 2012 to learn the growth from.

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
# - **The gap has two possible causes, and we measured them separately.** Seeded versus held-out days isolates the effect of sharing days between training and validation: {{p3.gaps.anchor.leakage:.4f}} for the Phase 2 model and {{p3.gaps.target.leakage:.4f}} for the target, both far smaller than the spread between day folds ({{p3.estimates.day_holdout.sd:.3f}}). Held-out days versus chronological isolates what changes when the validation period lies in the future: {{p3.gaps.anchor.drift:.3f}} and {{p3.gaps.target.drift:.3f}}.
# - **So the gap is drift, not leakage.** In the late period the mean demand is {{p3.chrono_detail.anchor.mean_cnt_late:.0f}} bikes against {{p3.chrono_detail.anchor.mean_cnt_early:.0f}} before the cut, and the Phase 2 model's predictions there are {{p3.chrono_detail.anchor.mean_ratio:.2f}} times the true mean: the straight trend fitted in log space extrapolates growth too steeply. If we only correct that level (a diagnostic, not a usable score) R² returns to {{p3.chrono_detail.anchor.r2_level_corrected:.3f}}, so most of the gap is the level and the hourly shape still fits.
# - **Which one to trust.** For data from new time periods, the chronological estimate ({{p3.estimates.chrono.r2:.3f}} for Phase 2, {{p3.estimates_target.chrono.r2:.3f}} for the target) is the honest one, and it is the number we would quote to an operator planning next year. The seeded number is the most optimistic of the three. For our hidden test set specifically, which is the 20th of each month inside the two observed years, the held-out-day estimate is the closest match, and it agrees with the seeded one.
# - **Does it change the diagnosis?** No in kind, yes in degree: see the Outcome.

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
# Rule: the simplest ladder level whose seeded validation R² is within {{p3.plateau_tol}} of the best level. It selects {{p3.target_complexity.level}} ({{p3.target_complexity.n_features}} weights): validation R² {{p3.ladder.6.seeded.r2:.4f}} against {{p3.ladder.5.seeded.r2:.4f}} one level below and {{p3.ladder.7.seeded.r2:.4f}} one level above. The other two estimates agree: held-out days {{p3.ladder.5.day_block.r2:.4f}} → {{p3.ladder.6.day_block.r2:.4f}} → {{p3.ladder.7.day_block.r2:.4f}}, chronological {{p3.ladder.5.chrono.r2:.3f}} → {{p3.ladder.6.chrono.r2:.3f}} → {{p3.ladder.7.chrono.r2:.3f}}.
#
# We have to be open about two things. First, our original rule was "the level with the highest validation R²". On the first ladder it picked a level 140 weights larger than its neighbour on a difference of 0.0005, which is not a rule we could defend, so we replaced it with the same "smallest within 0.001 of the best" rule we had already fixed for the degree in Phase 2. Second, level C6 itself was added after that first run (see the note above). Both changes are in the git history, and the rule still looks only at the seeded validation set.

# %% [markdown]
# ## Outcome — Phase 3
#
# **Diagnosis: the Phase 2 model is {{p3.diagnosis.label}} (high bias, no measurable variance).**
# - Its training and validation R² differ by {{p3.diagnosis.anchor_gap_train_val:.4f}}. A model that scores the same on rows it has and has not seen is not over-fit.
# - Adding structure raises validation R² from {{p3.estimates.seeded.r2:.4f}} to {{p3.estimates_target.seeded.r2:.4f}}: a gain of {{p3.diagnosis.gain_target_over_anchor:.3f}} with a paired 95% interval of {{p3.diagnosis.gain_ci_lo:.3f}}–{{p3.diagnosis.gain_ci_hi:.3f}}, confirmed on held-out days (+{{p3.diagnosis.day_block_gain:.3f}}) and on the chronological split (+{{p3.diagnosis.chrono_gain:.3f}}). Something real was missing.
# - The Phase 2 learning curve is flat: with 10% of the training days validation R² is already {{p3.learning_curves.anchor.0.val_r2:.3f}}, with all of them {{p3.learning_curves.anchor.5.val_r2:.3f}}. More data does not help a model that is too simple.
# - Variance does exist, but only at the top of the ladder: from level {{p3.diagnosis.overfit_from_level}} on, training R² keeps rising ({{p3.ladder.10.seeded.train_r2:.4f}} at C10) while validation falls ({{p3.ladder.10.seeded.r2:.4f}}) and the chronological score collapses ({{p3.ladder.10.chrono.r2:.3f}}); with 10% of the data the top level fails completely and the gap closes as data is added.
# - The target itself now shows a small gap (training {{p3.estimates_target.seeded.train_r2:.4f}}, validation {{p3.estimates_target.seeded.r2:.4f}}) and its learning curve is still rising at full size ({{p3.learning_curves.target.4.val_r2:.3f}} with 80% of the days, {{p3.learning_curves.target.5.val_r2:.3f}} with all). That is where a little variance begins, and it is what Phase 4 has to watch.
#
# **The time question.** Seeded {{p3.estimates.seeded.r2:.3f}}, held-out days {{p3.estimates.day_holdout.r2:.3f}}, chronological {{p3.estimates.chrono.r2:.3f}} for Phase 2; {{p3.estimates_target.seeded.r2:.3f}} / {{p3.estimates_target.day_holdout.r2:.3f}} / {{p3.estimates_target.chrono.r2:.3f}} for the target. The gap is drift (the future has a higher level of demand than a straight log-trend predicts well), not leakage between hours of the same day. We trust the chronological number for future periods. It does not change the diagnosis, the simpler models are worse on every split, but it does change how far we go: it is the split that punishes the levels above the target most clearly, so it supports stopping there.
#
# **What surprised us.**
# - We expected sharing days between training and validation to matter at least a little. It does not ({{p3.gaps.anchor.leakage:.4f}} for Phase 2; for the target the held-out days even score slightly higher than the seeded set). The warning in the task statement is right in general, but a linear model with a few hundred weights has no way to recognise an individual day, so it cannot profit from its neighbours.
# - The chronological split was kinder to the richer target than to Phase 2 (gap {{p3.gaps.target.drift:.3f}} against {{p3.gaps.anchor.drift:.3f}}). We had expected the simpler model to travel better through time. Both overshoot the level of the later months; the richer model gets the hourly shape more right.
# - The degree axis moved validation R² by only {{p3.diagnosis.degree_axis_range:.4f}} between degree 2 and 8. Had we varied only the degree we would have concluded "reasonably fit" and missed {{p3.diagnosis.gain_target_over_anchor:.3f}} of R².
# - Our own first selection rule failed on a near-tie, and our first ladder stopped one useful level short; the residuals of the model told us more about what was missing than the ladder we had planned in advance.
#
# **How much is left.** Averaging `cnt` over cells of year, month, day type, hour and weather situation explains {{p3.noise_floor.noise_floor_r2_adjusted:.3f}} of the training variance after adjusting for the number of cells. That is a benchmark, not a ceiling: our target already matches it on unseen rows. A diagnostic tree model that we keep outside the pipeline (`exp/analyst/`, never used for any prediction we submit) scored about 0.944 on held-out days, so the target ({{p3.estimates_target.day_holdout.r2:.3f}}) is within about 0.002 of what a far more flexible model gets from the same columns. We stop adding structure here.
#
# **Handed to Phase 4:** diagnosis "{{p3.diagnosis.label}}", target complexity {{p3.target_complexity.level}} with {{p3.target_complexity.n_features}} weights (degree {{p3.target_complexity.degree}}). The model we carry forward is not over-fit, but it is the first one with a visible train–validation gap, so we expect regularization to act as insurance with at most a small gain; Phase 4 tests that, including on the over-fit top level.
