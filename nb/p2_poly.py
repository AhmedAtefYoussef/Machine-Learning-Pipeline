# %% [markdown]
# ## Expectation — Phase 2: Polynomial Regression
#
# *Written before running the phase (see the git history of this file).*
#
# **What we do.** We load the Phase 1 weight vector from `artifacts/p1.json`, expand the design, and continue our own gradient descent from those weights (new columns start at zero).
#
# **What we expect, and why.**
#
# 1. *The chain.* Because the 35 base columns keep the Phase 1 scaler and every new weight starts at zero, the very first loss of Phase 2 must equal the last loss of Phase 1 to about 1e-9. If it does not, the hand-over is broken.
# 2. *Which terms matter.* The degree-2 products `workingday × hour` (23 columns) should give by far the largest gain, because they let the hour profile differ between working and non-working days. We expect validation R² to rise from Phase 1's level to about 0.89 from this block alone.
# 3. *Powers.* Demand rises with temperature and then flattens, and falls with humidity, so squares and cubes of `temp` and `hum` should add a little more (about +0.02). We expect powers of `windspeed` to add nothing measurable. We do not expand `trend`: a power of time would extrapolate badly.
# 4. *Degree.* We expect the curve of validation R² against degree to flatten after degree 2 or 3; degree 4 should not beat degree 3 by more than 0.001. Our rule, fixed in advance, is to take the smallest degree within 0.001 of the best.
# 5. *Optimisation.* Powers of a variable are correlated with each other and the interaction columns are correlated with the hour dummies, so the expanded design is worse conditioned. We expect roughly ten times more iterations than Phase 1 (a few thousand) at the same "half the bound" learning rate, still with a monotone loss.
# 6. *Fit quality.* Training and validation R² should stay within about 0.01 of each other: 60–70 weights are still few for 8,708 rows. If so, this model is more likely under-fit than over-fit, which is the question for Phase 3.

# %% [markdown]
# ## Expectation — Phase 2, second pass
#
# *Written before re-running the chain with the power target (λ = 0.1). The cell above is unchanged.*
#
# 1. *The chain* must hold exactly as before: first Phase 2 loss equal to the last Phase 1 loss.
# 2. *Choices.* We expect the same answers as in the first run: the working-day × hour block carries almost all of the
#    gain, degree 3, powers of `temp` only. If the degree or the columns change, it will be by a margin near our 0.001
#    tolerance.
# 3. *Accuracy.* About 0.918 on validation (0.910 with the log target).

# %% [markdown]
# ### Justification: initialisation from Phase 1
# The expanded design keeps the 35 Phase 1 columns in the same positions and with the Phase 1 scaler (read from `artifacts/p1.json`, not refitted). Each new column gets its own training mean and standard deviation. The longer weight vector is the Phase 1 vector with zeros in the new positions. A zero weight switches a column off, so the expanded model starts as exactly the Phase 1 model: its first loss must equal Phase 1's last loss. The cell below checks this to 1e-9. Starting there instead of at random also means gradient descent only has to learn the correction that the new columns allow.

# %%
# Phase 2 starts from the Phase 1 weights, so its first loss must equal Phase 1's last loss (the hand-over).
import src.phases.p2 as p2mod
from src.poly import lift_weights

p2 = phase("p2", p2mod.run, CFG, upstream="p1", live=True, recompute_all=RECOMPUTE_ALL)   # reads artifacts/p1.json, fits Phase 2 here and now
assert abs(p2["init_loss"] - p1["train_loss_final"]) <= 1e-9, "the hand-over from Phase 1 to Phase 2 is broken"
view.p2_handover(p1, p2)

# %%
# How the Phase 1 weights are placed into the longer Phase 2 vector.
show_source(lift_weights)

# %% [markdown]
# ### Justification: what we expanded
# We expand only where Phase 1's residuals showed structure.
# - **`workingday × hour` (23 degree-2 products).** The Phase 1 residuals differ by day type at almost every hour. Without this block the expanded model reaches only {{p2.block_ablation.without_wd_x_hr.val_r2:.4f}} on validation; with it {{p2.val_r2:.4f}} (ablation below). This is the one expansion that matters.
# - **Powers of `temp`.** Demand rises with temperature and flattens when it is hot. At the chosen degree, powers of `temp` alone give {{p2.power_col_sweep.0.val_r2:.4f}}, adding `hum` gives {{p2.power_col_sweep.1.val_r2:.4f}} and adding `windspeed` as well gives {{p2.power_col_sweep.2.val_r2:.4f}}. Differences below {{p2.plateau_tol}} are far inside the noise of our validation set (standard error {{p2.val_bootstrap.se:.4f}}), so by our rule we keep the smallest set: `temp` only.
# - **Not expanded.** Hour dummies, weather dummies and `holiday` are 0/1, so their powers are themselves. `trend` is not raised to a power, because a polynomial in time bends away sharply outside the observed dates. The day-of-year terms are already non-linear.

# %%
# The expanded design: which blocks and power columns, how many new columns, and how hard it is to optimise.
view.p2_design(p2)

# %% [markdown]
# ### Justification: degree
# "Degree" is the highest total degree of any term: the interaction products are degree 2 and the powers go up to the chosen degree. With every candidate fitted by our own gradient descent from the Phase 1 weights, validation R² is {{p2.degree_sweep.0.val_r2:.4f}} at degree 1 (interactions only), {{p2.degree_sweep.1.val_r2:.4f}} at degree 2, {{p2.degree_sweep.2.val_r2:.4f}} at degree 3 and {{p2.degree_sweep.3.val_r2:.4f}} at degree 4. Our rule, fixed before the run, is the smallest degree within {{p2.plateau_tol}} of the best: degree {{p2.degree}}. The square captures "warmer is better", the cube captures the flattening at high temperature; a fourth power adds nothing. The learning rate is again half the stability bound, recomputed for this design (λ_max {{p2.lambda_max:.3f}}, lr {{p2.lr:.4f}}), with the same stopping rule as Phase 1.

# %%
# Validation R² for each candidate degree (the highlighted row is our choice).
view.p2_degree_sweep(p2)

# %%
# The same sweep as a plot.
show(plots.plot_degree_sweep(p2))

# %%
# Which variables get powers: temperature alone, or humidity and wind as well.
view.p2_power_cols(p2)

# %%
# What the working-day x hour block adds.
view.p2_ablation(p2)

# %%
# Phase 2 against Phase 1 on the validation rows, and how the run ended.
display(view.p2_scores(p1, p2))
display(view.p2_fit(p2))

# %%
# The residuals by hour and day type before (Phase 1) and after (Phase 2) the expansion.
show(plots.plot_residual_profile(p1, p2))

# %% [markdown]
# ## Outcome — Phase 2
#
# **What happened.**
# - The hand-over held exactly: the first Phase 2 loss equals the last Phase 1 loss ({{p2.init_loss:.6f}}).
# - Chosen model: degree {{p2.degree}}, powers of `temp`, plus `workingday × hour`: {{p2.power_col_sweep.0.n_features}} weights. Validation R² rose from {{p2.p1_val_r2:.4f}} to {{p2.val_r2:.4f}} (gain {{p2.paired_vs_p1.delta:.3f}}, paired 95% interval {{p2.paired_vs_p1.lo:.3f}}–{{p2.paired_vs_p1.hi:.3f}}); RMSE is {{p2.val_rmse:.1f}} bikes. Training R² is {{p2.train_r2:.4f}}: still no gap.
# - As expected, almost all of the gain is the interaction block (degree 1 with interactions already gives {{p2.degree_sweep.0.val_r2:.4f}}); the temperature powers add about 0.02, and the curve is flat after degree 3.
# - The residual profile by hour and day type is now flat.
#
# **What surprised us.**
# - Optimisation was much easier than we feared: {{p2.iterations}} iterations, not thousands. We had expected the new columns to be strongly correlated with the old ones, but because we build powers and products from standardised columns the condition number only rose to {{p2.condition_number:.0f}}.
# - Humidity powers were not worth keeping ({{p2.power_col_sweep.0.val_r2:.4f}} → {{p2.power_col_sweep.1.val_r2:.4f}}); we had expected humidity to need a curve as much as temperature does.
# - Training and validation R² are almost identical. A model that fits unseen rows as well as its own training rows is not over-fit; the open question for Phase 3 is whether it is still too simple.
#
# **Second pass.** With the power target the choices are the ones we expected: the same block, the same degree, powers of `temp` only, and validation R² {{p2.val_r2:.4f}} against {{first_pass.p2.val_r2:.4f}} with the log target. One detail differs from the first pass: the degree curve is not quite as flat before degree 3 ({{p2.degree_sweep.1.val_r2:.4f}} at degree 2, {{p2.degree_sweep.2.val_r2:.4f}} at degree 3), so the cube of temperature earns its place more clearly now.
#
# **Handed to Phase 3:** degree {{p2.degree}} and the expanded feature list in `artifacts/p2.json`.
