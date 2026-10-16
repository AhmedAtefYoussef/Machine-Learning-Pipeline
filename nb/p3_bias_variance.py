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
# ## Expectation — Phase 3, second pass
#
# *Written before re-running the chain with the power target (λ = 0.1). The cell above is unchanged; the ladder now has
# eleven levels because level C6 was added during the first pass.*
#
# 1. *Diagnosis.* Unchanged in kind: Phase 2 under-fit, target at level C6, variance only at the top of the ladder.
# 2. *The time question.* This is where we expect the real change. With the log target the chronological estimate of the
#    target was about 0.91 against 0.94 on held-out days. A straight trend on the log scale is exponential growth; on the
#    λ = 0.1 scale it grows more slowly, so the overshoot in the later months should shrink and the chronological
#    estimate should rise to about 0.93 to 0.94, almost closing the gap.
# 3. *Held-out days.* A smaller gain, about +0.003 to +0.004 for the target.
# 4. *Exponent check.* A new table scores the target design for several exponents. We expect a flat optimum around 0.1
#    on validation and held-out days, with 0.15 to 0.2 slightly better chronologically. If the best exponent were far
#    from the one Phase 1 used, our chain would be inconsistent and we would have to start again.

# %% [markdown]
# ### Justification: what we varied
#
# - **A nested ladder anchored at Phase 2.** Level C3 is exactly the Phase 2 design. Below it we remove what Phase 2 added (C2 has no powers, C1 no products, C0 no hour). Above it we add hour-dependent weather (C4), an hour profile per weekday (C5), weather detail and weather memory (C6), finer weather terms (C7), hour-specific season and trend (C8), and month × hour blocks (C9, C10). Bias shows as training and validation rising together when a block is added; variance as training rising while validation falls.
# - **The degree alone**, from 1 to 8, because the brief asks about it, and **the amount of training data** for the anchor, the target and the top level.
# - **Three scores per design:** the seeded validation set, five folds of held-out days fixed by the calendar, and the chronological split.
# - **Closed form.** These several hundred fits use exact least squares, which our gradient descent matches on the anchor (asserted below). A tiny ridge term keeps the solution defined where columns are exact copies, such as `holiday`.
# - **Level C6 came late.** We added it after the first complete run, from a residual analysis of C5 whose errors sat mostly in working-day rush hours and in rain: humidity powers, wind terms, a temperature effect per working-day hour, and the weather of the previous one and three hours (inputs only, never counts). It was designed after we had looked at all three validators, so none of its estimates is untouched.

# %%
# Phase 3 builds a ladder of designs on top of the Phase 2 artifact and scores each one three ways.
from src import plots_p3
from src.phases import p3 as p3mod

p3 = phase("p3", p3mod.run, CFG, RECOMPUTE_ALL, upstream="p2", live=True)   # about 15 seconds

# %%
# The chain: the anchor is exactly the Phase 2 design, and our closed-form fit reproduces its validation R².
anchor = p3["anchor"]
assert (anchor["degree"], anchor["blocks"], anchor["power_cols"]) == (p2["degree"], p2["blocks"], p2["power_cols"])
assert anchor["n_features"] == len(p2["feature_names"]), "the anchor is not the Phase 2 design"
assert abs(anchor["p2_val_r2"] - p2["val_r2"]) < 1e-12 and abs(p3["anchor_check"]) < 1e-6, "the anchor does not reproduce Phase 2"
print(f"anchor = Phase 2 design ({anchor['n_features']} weights); closed form minus gradient descent: {p3['anchor_check']:.1e}")

# %%
# The complexity ladder with the three validators for every level (highlighted: the target level), and as a plot.
ladder = pd.json_normalize(p3["ladder"])
table(ladder[["name", "n_features", "seeded.train_r2", "seeded.r2", "day_block.r2", "chrono.r2"]],
      f"Complexity ladder; over-fit from level {p3['overfit_from_level']}", ladder["name"] == p3["target_level"])
show(plots_p3.plot_ladder(p3))

# %%
# The degree axis alone: anchor blocks and power columns fixed, only the degree changes.
axis = pd.json_normalize(p3["degree_axis"])
table(axis[["degree", "n_features", "seeded.r2", "day_block.r2", "chrono.r2"]], "Degree axis (validation R² by validator)")

# %%
# Learning curves for the anchor, the target and the top of the ladder.
show(plots_p3.plot_learning_curves(p3))

# %% [markdown]
# ### Justification: the chronological cut
#
# We train on dates before {{p3.cut_date}} ({{p3.chrono_detail.anchor.n_early}} rows) and validate on the rest ({{p3.chrono_detail.anchor.n_late}} rows). The held-out part is about as large as our seeded validation set, so the two estimates are comparably precise. It covers half a seasonal cycle, from summer to winter, and the training part still holds one full year plus half of the next, so every month is seen at least once. A later cut would test only autumn and winter; an earlier one would leave too little of 2012 to learn the growth from.

# %%
# The anchor and the target scored by the seeded split, held-out days and the chronological split, with their gaps.
gaps = p3["gaps"]
designs = [{"design": "anchor", **p3["estimates"], "gaps": gaps["anchor"]}, {"design": "target", **p3["estimates_target"], "gaps": gaps["target"]}]
est = pd.json_normalize(designs)
table(est[["design", "seeded.r2", "seeded.lo", "seeded.hi", "day_holdout.r2", "day_holdout.sd", "chrono.r2", "gaps.leakage", "gaps.drift"]],
      "Three estimates (highlighted: the target)", est["design"] == "target")

# %% [markdown]
# ### Justification: which estimate to trust
#
# - **Two causes, measured separately.** Seeded minus held-out days measures what sharing days between training and validation is worth: {{p3.gaps.anchor.leakage:.4f}} for Phase 2 and {{p3.gaps.target.leakage:.4f}} for the target, both smaller than the spread between day folds ({{p3.estimates.day_holdout.sd:.3f}}). Held-out days minus chronological measures drift: {{p3.gaps.anchor.drift:.3f}} and {{p3.gaps.target.drift:.3f}}.
# - **So the gap is drift.** After the cut mean demand is {{p3.chrono_detail.anchor.mean_cnt_late:.0f}} bikes against {{p3.chrono_detail.anchor.mean_cnt_early:.0f}} before, and Phase 2 predicts {{p3.chrono_detail.anchor.mean_ratio:.2f}} times the true mean there. Correcting only that level, as a diagnostic, gives R² {{p3.chrono_detail.anchor.r2_level_corrected:.3f}}: the hourly shape still fits.
# - **Which to trust.** The chronological estimate for future periods. For our hidden test, which lies inside the two observed years, held-out days are the closest match. The seeded number is the easiest to flatter.

# %%
# The diagnosis, and the complexity level we carry forward to Phase 4.
level = p3["target_complexity"]
table({**p3["diagnosis"], "target": level["level"], "target weights": level["n_features"], "target degree": level["degree"],
       "target power columns": level["power_cols"], "target blocks": level["blocks"]}, "Diagnosis and target complexity")

# %% [markdown]
# ### Justification: exponent check
#
# Phase 1 fixed λ = {{p3.target_power_used}} for the whole chain; here we test it on the target design. Validation prefers {{p3.target_power_best_seeded}}: {{p3.target_power_check.1.seeded_r2:.4f}} at {{p3.target_power_check.1.power}}, {{p3.target_power_check.2.seeded_r2:.4f}} at {{p3.target_power_check.2.power}}, {{p3.target_power_check.3.seeded_r2:.4f}} at {{p3.target_power_check.3.power}}, and {{p3.target_power_check.0.seeded_r2:.4f}} for the log. Held-out days agree. The chronological split would like a slightly larger exponent ({{p3.target_power_check.3.chrono_r2:.3f}} at {{p3.target_power_check.3.power}} against {{p3.target_power_check.2.chrono_r2:.3f}}). The exponent in use is within {{p3.plateau_tol}} of the best on validation ({{p3.target_power_consistent}}), so the chain stands. We do not move to the larger exponent, because every choice is tuned on the validation set.

# %%
# The target design in closed form under each candidate exponent (the chain itself uses the exponent from Phase 1).
check = pd.DataFrame(p3["target_power_check"])
table(check[["power", "seeded_r2", "day_block_r2", "chrono_r2"]], f"Target exponent on the target design (in use: {p3['target_power_used']}; "
      f"within the plateau tolerance: {p3['target_power_consistent']})", np.isclose(check["power"], p3["target_power_used"]))

# %% [markdown]
# ### Justification: target complexity
#
# Rule: the simplest ladder level whose validation R² is within {{p3.plateau_tol}} of the best level. It selects {{p3.target_complexity.level}} ({{p3.target_complexity.n_features}} weights): {{p3.ladder.6.seeded.r2:.4f}} against {{p3.ladder.5.seeded.r2:.4f}} one level below and {{p3.ladder.7.seeded.r2:.4f}} one level above. Held-out days ({{p3.ladder.5.day_block.r2:.4f}} → {{p3.ladder.6.day_block.r2:.4f}} → {{p3.ladder.7.day_block.r2:.4f}}) and the chronological split ({{p3.ladder.5.chrono.r2:.3f}} → {{p3.ladder.6.chrono.r2:.3f}} → {{p3.ladder.7.chrono.r2:.3f}}) agree. Our first rule was "the highest validation R²". On the first ladder it picked a much larger level for a difference far inside the noise, so we switched to the plateau rule we had already fixed for the degree in Phase 2. That change and level C6 are both in the git history.

# %% [markdown]
# ## Outcome — Phase 3
#
# **Diagnosis: the Phase 2 model is {{p3.diagnosis.label}}, with high bias and no measurable variance.**
# - Its training and validation R² differ by {{p3.diagnosis.anchor_gap_train_val:.4f}}.
# - More structure raises validation R² from {{p3.estimates.seeded.r2:.4f}} to {{p3.estimates_target.seeded.r2:.4f}} (gain {{p3.diagnosis.gain_target_over_anchor:.3f}}, paired 95% interval {{p3.diagnosis.gain_ci_lo:.3f}}–{{p3.diagnosis.gain_ci_hi:.3f}}), confirmed on held-out days (+{{p3.diagnosis.day_block_gain:.3f}}) and chronologically (+{{p3.diagnosis.chrono_gain:.3f}}).
# - The Phase 2 learning curve is flat: {{p3.learning_curves.anchor.0.val_r2:.3f}} with the smallest share of training days, {{p3.learning_curves.anchor.5.val_r2:.3f}} with all of them.
# - Variance appears only at the top. From level {{p3.diagnosis.overfit_from_level}} on, training R² keeps rising ({{p3.ladder.10.seeded.train_r2:.4f}} at C10) while validation falls ({{p3.ladder.10.seeded.r2:.4f}}). The chronological score falls right after the target ({{p3.ladder.6.chrono.r2:.3f}} at C6, {{p3.ladder.8.chrono.r2:.3f}} at C8, {{p3.ladder.10.chrono.r2:.3f}} at C10), so it is the split that tells us to stop there.
# - The target shows a small gap (training {{p3.estimates_target.seeded.train_r2:.4f}}, validation {{p3.estimates_target.seeded.r2:.4f}}) and its learning curve still rises at full size ({{p3.learning_curves.target.4.val_r2:.3f}} → {{p3.learning_curves.target.5.val_r2:.3f}}). That is the first sign of variance, for Phase 4 to watch.
#
# **The time question.** Seeded, held-out days and chronological: {{p3.estimates.seeded.r2:.3f}}, {{p3.estimates.day_holdout.r2:.3f}} and {{p3.estimates.chrono.r2:.3f}} for Phase 2; {{p3.estimates_target.seeded.r2:.3f}}, {{p3.estimates_target.day_holdout.r2:.3f}} and {{p3.estimates_target.chrono.r2:.3f}} for the target. The diagnosis is the same on every split.
#
# **What surprised us.**
# - *First pass:* sharing days does not help our model. The warning in the brief is still real: a tree model we tried in `exp/v2/` scores {{diagnostics.tree_model.seeded:.3f}} on the seeded split and {{diagnostics.tree_model.held_out_days:.3f}} on held-out days. A linear model is too stiff to recognise an individual day.
# - *First pass:* the degree axis moved validation R² by only {{p3.diagnosis.degree_axis_range:.4f}}. Varying the degree alone would have missed {{p3.diagnosis.gain_target_over_anchor:.3f}} of R².
# - *Second pass, better than expected:* the chronological estimate of the target rose from {{first_pass.p3.target.chrono:.3f}} to {{p3.estimates_target.chrono.r2:.3f}}, and the drift gap shrank from {{first_pass.p3.drift_gap_target:.3f}} to {{p3.gaps.target.drift:.3f}}. The log over-shot the later months by a factor {{first_pass.p3.chrono_mean_ratio_target:.2f}}, our target by {{p3.chrono_detail.target.mean_ratio:.2f}}. Most of what we had called drift was our choice of target: a straight line on the log scale is exponential growth. The diagnosis, the target level and the exponent did not change.
#
# **How much is left.** Cell means over year, month, day type, hour and weather explain {{p3.noise_floor.noise_floor_r2_adjusted:.3f}} of the training variance after adjusting for the number of cells, and our target scores above that on unseen rows. The tree model reaches {{diagnostics.tree_model.held_out_days:.3f}} on held-out days and {{diagnostics.tree_model.chrono:.3f}} chronologically, against {{p3.estimates_target.day_holdout.r2:.3f}} and {{p3.estimates_target.chrono.r2:.3f}} for our target. We also tried {{diagnostics.feature_blocks_tested}} more groups of input-only features; no weather or hour group moved held-out days by more than {{diagnostics.weather_blocks_max_abs_days_delta:.4f}}, and separate levels per calendar month gained {{diagnostics.year_month_levels.held_out_days_delta:.3f}} on held-out days but changed the chronological score by {{diagnostics.year_month_levels.chrono_delta:.3f}}. We stop adding structure here.
#
# **Handed to Phase 4:** the diagnosis "{{p3.diagnosis.label}}" and target level {{p3.target_complexity.level}} with {{p3.target_complexity.n_features}} weights.
