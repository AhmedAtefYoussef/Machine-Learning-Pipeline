# %% [markdown]
# ## Expectation — Phase 1: Gradient Descent
#
# *Written before running the phase (see the git history of this file).*
#
# **What we fit.** A linear model for `log1p(cnt)` on 35 standardised columns: 23 hour dummies, `workingday`, `holiday`, two weather-situation dummies, `temp`, `hum`, `windspeed`, a linear trend in days, and two day-of-year harmonics. We fit it with our own gradient descent, starting from zero weights.
#
# **What we expect, and why.**
#
# 1. *Convergence.* The MSE loss is a quadratic bowl, so gradient descent is stable only when the learning rate is below 2 / λ_max of XᵀX/n. Our columns are standardised and hour dummies are almost uncorrelated with each other, so we expect λ_max close to 2 and a bound close to 1. With half the bound we expect a smooth, monotone loss curve that stops in a few hundred iterations (not thousands). A learning rate 5% above the bound should blow up; one at 0.1% of the bound should still be far from the minimum after 5,000 iterations.
# 2. *Correctness.* Our final weights should match the closed-form least-squares weights to better than 1e-3, and the finite-difference gradient check should agree to about 1e-8.
# 3. *Accuracy.* We expect a validation R² (on bikes) of roughly 0.68–0.73, clearly below what the data allows. The reason is structural: this model adds one hour effect and one working-day effect, but the data's rush-hour peaks (8h, 17–18h) exist only on working days. An additive model has to average the two day types, so we expect its residuals, grouped by hour and day type, to show a clear pattern. That pattern is what Phase 2 should fix.
# 4. *Train vs validation.* With 36 weights and 8,708 rows we expect almost no gap (under 0.02).
# 5. *Back-transform.* We will try no correction, Duan's smearing factor and a least-squares factor. The textbook says Duan should help; we suspect it will not, because the large log-residuals sit in quiet night hours while the factor inflates every prediction, including the peaks that dominate R².
# 6. *Bonus (asymmetric cost).* When under-prediction costs three times as much, the fitted predictions should move up. We expect the share of under-predicted validation hours to fall from about one half to about one third or less, at the price of a higher plain RMSE.

# %% [markdown]
# ## Expectation — Phase 1, second pass
#
# *Written before re-running the chain with a new target (see the git history). The cell above is our original
# expectation and is unchanged.*
#
# **What changes.** Our first complete run used `log1p(cnt)` as the target. Its Phase 3 analysis, and the exploration we
# did afterwards (`exp/v2/`), showed that the log is too strong: it makes an error of a few bikes at 3 am count as much
# as an error of a hundred bikes at 5 pm, while R² on bikes is decided by the busy hours. We therefore re-run everything
# with a milder transform of the same family, z = ((cnt + 1)^λ − 1) / λ with λ = 0.1 (λ → 0 is the log, λ = 1 is the
# raw count). Nothing else in Phase 1 changes: same 35 columns, same gradient descent, same learning-rate rule.
#
# **What we expect.**
# 1. *Accuracy.* Validation R² should rise from about 0.72 to about 0.735. For this small model alone an even larger
#    exponent (0.25 to 0.3) would score a little higher; we knowingly keep 0.1 because the exponent has to be fixed once
#    for the whole chain and the richer models of the later phases prefer about 0.1. The cell that compares exponents
#    should show exactly this.
# 2. *Optimisation.* Unchanged. The design matrix is the same, so λ_max, the stability bound and the learning rate are
#    identical, and the iteration count should again be a few hundred.
# 3. *Back-transform.* The least-squares factor should move closer to 1 than the 1.04 of the log model, because a milder
#    transform distorts the mean less. Duan's factor is defined for the log only, so we now compare "no factor" and the
#    least-squares factor.
# 4. *Bonus.* The asymmetric model should still move predictions up and cut the share of under-predicted hours to about
#    0.3. The plain model's own share of under-predictions should be closer to one half than the 0.52 of the log model.

# %% [markdown]
# ### Justification: representation
#
# - **Target.** We model a transformed count, not the count itself: counts are right-skewed, their spread grows with their level, and effects multiply (a rainy rush hour loses a share of its riders, not a fixed number). Which transform, and why, is the next justification. R² is always computed on bikes after transforming back.
# - **Hour.** One-hot, 23 dummies with hour 0 as reference. As a single number the hour gives a validation R² of only {{p1.hour_encoding_ablation.numeric.val_r2:.3f}}, because demand does not rise or fall steadily through the day. Six sine/cosine pairs reach {{p1.hour_encoding_ablation.harmonics6.val_r2:.3f}} and the dummies {{p1.hour_encoding_ablation.one_hot.val_r2:.3f}} (table below). We kept the dummies: slightly better, and each weight reads directly as "the effect of this hour".
# - **Weather situation.** Dummies, with category 4 merged into 3 because it has a single training row.
# - **`dteday`.** A trend in days since 1 January 2011 (demand grew strongly) and two day-of-year sine/cosine pairs (the season, without a jump between December and January).
# - **Cleaning.** Humidity 0 occurs on one day only (a sensor failure); those rows get the training median. Wind speed is left as measured.
# - **Left out until Phase 4.** `atemp`, `season`, `mnth`, `yr`, `instant` and `weekday` each repeat information that is already in the design almost exactly (for example `workingday` is an exact function of `weekday` and `holiday`). Duplicates make the weights non-unique and slow gradient descent down. Without them the condition number is {{p1.condition_number:.1f}}. Phase 4 puts them back and lets the models judge them.
# - **Scaling.** Every column is standardised with the training mean and standard deviation only.

# %%
# Phase 1 is fitted from scratch here and now; the table shows the design and the learning rate it was given.
import src.phases.p1 as p1mod
from src import bonus_control
from src.common import Target
from src.gd import gradient_descent, mse_grad, mse_loss
from src.gd_asym import asym_grad, asym_loss
from src.nbtools import p1_live_run

p1 = phase("p1", p1mod.run, CFG, live=True, recompute_all=RECOMPUTE_ALL)   # fits Phase 1 from scratch, here and now
view.p1_design(p1)

# %%
# The target transform of the whole chain, as implemented: forward, inverse, factor, back to bikes.
show_source(Target)

# %% [markdown]
# **Live check on this machine.** The cell above ran the whole of Phase 1 and compared it with the stored artifact.
# The next cell shows the core of it step by step: our gradient descent from zero weights with the same settings, and
# the comparison of the result with the stored weight vector. Different machines add floating-point numbers in slightly different order, so we expect
# agreement to many decimal places rather than bit for bit.

# %%
# The core of Phase 1 once more: our gradient descent from zero weights, compared with the stored weights.
live, live_lr = p1_live_run(CFG)
display(view.p1_live_check(p1, live, live_lr))
assert np.allclose(live.weights, p1["weights"], atol=1e-8), "the live Phase 1 run does not reproduce the stored weights"

# %%
# Our gradient descent, exactly as it runs above: the loss, its gradient, and the loop.
show_source(mse_loss, mse_grad, gradient_descent)

# %% [markdown]
# ### Justification: learning rate and stopping rule
# - **Learning rate.** For a quadratic loss, gradient descent is stable only below 2 / λ_max of XᵀX/n. We compute λ_max = {{p1.lambda_max:.4f}} from our design, so the bound is {{p1.lr_bound:.4f}}, and we use {{p1.lr_fraction}} of it: lr = {{p1.lr:.4f}}. At half the bound every direction of the loss surface shrinks without overshooting, so the loss can only go down, and we keep a factor-two safety margin.
# - **Evidence that it converges rather than stalls or diverges.** The sweep below uses the same data and start. At 0.001 and 0.01 of the bound the run is still far from finished after {{p1.lr_sweep.0.iterations}} iterations (stall). At 0.1 it converges but needs {{p1.lr_sweep.2.iterations}} iterations. Our setting needs {{p1.lr_sweep.3.iterations}}. At 0.9 it is faster ({{p1.lr_sweep.4.iterations}}) but leaves almost no margin. At 1.05 the loss explodes and the run is stopped after {{p1.lr_sweep.5.iterations}} iterations (diverge).
# - **Stopping rule.** We stop when the relative change of the loss is at most {{p1.tol_loss}} and the gradient norm is below {{p1.tol_grad}}, with a cap of {{p1.max_iter}} iterations. The loss test alone can fire on a flat stretch far from the minimum; the gradient test alone depends on the scale of the problem; together they mean "flat because we are at the bottom".
# - **Start.** Zero weights, so anyone who re-runs this cell gets exactly the same weight vector.

# %%
# Six learning rates, from far too small to too large, and what each one does.
view.p1_lr_sweep(p1)

# %%
# The same sweep as loss curves.
show(plots.plot_lr_sweep(p1))

# %%
# How the final run ended, and its loss curve.
display(view.p1_fit(p1))
show(plots.plot_loss_curve(p1))

# %%
# Correctness: the exact least-squares solution is the reference that gradient descent should reach.
view.p1_oracle(p1)

# %% [markdown]
# ### Justification: back-transform
#
# The inverse transform of an average is not the average count: without a correction the model's predictions are {{p1.backtransform.candidates.none.val_mean_ratio:.3f}} of the true mean on validation. We compared two options, each fitted on the training rows and judged on validation: no factor (R² {{p1.backtransform.candidates.none.val_r2:.4f}}) and a least-squares factor s = Σ(cnt+1)·q / Σq² = {{p1.backtransform.candidates.ls.factor:.3f}}, the single multiplier of q = (λz + 1)^(1/λ) that minimises squared error on bikes (R² {{p1.backtransform.candidates.ls.val_r2:.4f}}, mean ratio {{p1.backtransform.candidates.ls.val_mean_ratio:.3f}}). We keep the least-squares factor. In our first pass, with the log target, we also tried Duan's smearing factor, the textbook correction for a log model: it lowered validation R² from {{first_pass.p1.none_val_r2:.3f}} to {{first_pass.p1.duan_val_r2:.3f}}, because the large log-errors sit in quiet night hours while the factor inflates the peaks. Duan's factor is defined for the log only, so it is not in this table. The method is fixed for every later phase; each phase recomputes the factor from its own training residuals.

# %%
# The two back-transform options, each fitted on the training rows and judged on the validation rows.
view.p1_backtransform(p1)

# %% [markdown]
# ### Justification: target transform
#
# We fit z = ((cnt + 1)^λ − 1) / λ with λ = {{p1.target_power}}. This family runs from the logarithm (λ → 0) to the raw count (λ = 1); predictions come back to bikes through the inverse, (λz + 1)^(1/λ) − 1.
#
# - **Why not the log.** Our first complete run used `log1p(cnt)` (it is kept in the git history under the tag `v1-submitted`). On the log scale an error of four bikes at 3 am weighs as much as an error of two hundred at 5 pm, but R² on bikes is decided by the busy hours. The log fit spent its effort in the wrong place.
# - **Why not the raw count.** At λ = 1 the model is additive in bikes, and the multiplicative structure of the data (a busy hour on a growing system in good weather) is lost: validation R² falls to {{p1.target_power_table.7.val_r2:.3f}}.
# - **What the table below shows for this phase.** With the Phase 1 design, validation R² is {{p1.target_power_table.0.val_r2:.4f}} for the log, {{p1.target_power_table.2.val_r2:.4f}} at λ = 0.1, {{p1.target_power_table.4.val_r2:.4f}} at 0.2, {{p1.target_power_table.5.val_r2:.4f}} at 0.3 and {{p1.target_power_table.6.val_r2:.4f}} at 0.5. Taken alone, this phase would choose λ = {{p1.target_power_best_here}}.
# - **Why we use 0.1 anyway.** The transform has to be the same in every phase, otherwise Phase 2 could not start from Phase 1's weights. The best exponent shrinks as the model gains structure: a simple model needs a milder transform to make up for what it cannot express, a richer one does not. Our first pass and the pilot fits after it showed that the model the chain ends with prefers about 0.1, so that is the value we fix here, giving up about 0.005 of R² in this phase. Phase 3 checks the exponent again on its target design; if that check disagreed with 0.1 we would have to restart the chain.
# - **Honesty note.** This choice was made from a full earlier run, not from this phase alone. We say so because the brief asks for our expectations to be written first: the second-pass Expectation cell above was committed before this run.

# %%
# The same design under each candidate exponent of the target transform (0 = log, 1 = raw counts).
display(view.p1_target_power(p1))
show(plots.plot_target_power(p1))

# %%
# Scores of the Phase 1 model on the training and validation rows.
view.p1_scores(p1)

# %%
# Three ways to encode the hour of day.
view.p1_hour_encoding(p1)

# %%
# What is left over: the residuals by hour and day type.
show(plots.plot_residual_profile(p1))

# %% [markdown]
# ### Justification: bonus
#
# **The operator's cost.** Under-predicting by some amount costs three times as much as over-predicting by the same amount. We read this as a weighted squared error on bikes (the cost is paid in bikes, not on the transformed scale):
#
# L(w) = (1/n) Σ cᵢ (ŷᵢ − yᵢ)², with ŷᵢ = q(xᵢ·w) − 1, q(η) = (λη + 1)^(1/λ), and cᵢ = 3 if ŷᵢ < yᵢ, else 1.
#
# **Gradient.** By the chain rule, ∂ŷᵢ/∂w = q′(ηᵢ)·xᵢ with q′(η) = (λη + 1)^(1/λ − 1) = q(η)^(1−λ), so ∇L = (2/n) Σ cᵢ (ŷᵢ − yᵢ) q(ηᵢ)^(1−λ) xᵢ. (For the log, λ → 0, this is the familiar exp(η)·xᵢ.) The weight cᵢ jumps only where the residual is zero, so it adds nothing to the gradient. Our finite-difference check agrees to {{p1.bonus.gradient_check:.1e}}.
#
# **Optimiser.** This loss is no longer a quadratic, so the fixed-step bound of the MSE model does not apply. We keep plain gradient descent but choose each step by backtracking (halve the step until the loss decreases enough), starting from the Phase 1 MSE weights. It stopped as "{{p1.bonus.stop_reason}}" after {{p1.bonus.iterations}} iterations. The absolute-error version of the same cost is shown in the table as a sensitivity check. This model is a side study: Phase 2 receives the MSE weights.

# %%
# The bonus: a model that fears under-prediction three times as much, against the plain model.
view.p1_bonus(p1)

# %%
# Where the asymmetric cost moves the predictions.
show(plots.plot_bonus_shift(p1))

# %%
# The asymmetric loss and its gradient as implemented.
show_source(asym_loss, asym_grad)

# %% [markdown]
# **A control for the bonus.** The comparison above changes two things at once: the bonus model is fitted on bikes
# instead of on the transformed scale, *and* it weights under-predictions three times. To separate them we fit the same bike-scale
# loss with k = 1 (no asymmetry), from the same starting weights. Fitting on bikes alone already moves the mean
# prediction up by a factor {{bonus_control.shift_from_fitting_on_bikes:.3f}} (a fit on a transformed scale aims near the median, a
# bike-scale fit at the mean) and lowers the share of under-predicted hours from
# {{p1.bonus.val.mse_model.under_share:.2f}} to {{bonus_control.val.k1_model.under_share:.2f}}. The asymmetry itself
# adds a further factor {{bonus_control.shift_from_asymmetry:.3f}} and brings the share down to
# {{p1.bonus.val.asym_model.under_share:.2f}}. So of the total shift of {{bonus_control.total_shift:.2f}}, roughly two
# thirds is the operator's asymmetry and one third is the change of scale. (The k = 1 model also has the best plain
# RMSE of the three, {{bonus_control.val.k1_model.rmse:.1f}}, for this additive design: squared error on bikes is what
# RMSE measures. The table's "MSE model" is the Phase 1 model without a back-transform factor, RMSE
# {{p1.bonus.val.mse_model.rmse:.1f}}; with the factor it is {{p1.val_rmse:.1f}}.)

# %%
# The control for the bonus: the same bike-scale loss without the asymmetry (k = 1).
control = bonus_control.run(CFG)   # side study: reads artifacts/p1.json, writes artifacts/bonus_control.json
view.p1_bonus_control(p1, control)

# %% [markdown]
# ## Outcome — Phase 1
#
# **What happened.**
# - Gradient descent converged in {{p1.iterations}} iterations at lr = {{p1.lr:.4f}} (λ_max {{p1.lambda_max:.3f}}, as expected close to 2). The weights differ from the closed-form solution by at most {{p1.oracle.max_abs_weight_diff:.1e}} and the gradient check agrees to {{p1.gradient_check:.1e}}.
# - Validation R² is {{p1.val_r2:.4f}} (95% interval {{p1.val_bootstrap.lo:.3f}}–{{p1.val_bootstrap.hi:.3f}}), RMSE {{p1.val_rmse:.1f}} bikes; training R² is {{p1.train_r2:.4f}}, so there is essentially no gap, as expected for 36 weights. With the log target of our first pass the same model scored {{first_pass.p1.val_r2:.4f}}.
# - The residual plot shows the pattern we predicted in both passes: on non-working days the model is far too low at night and too high at commute hours, because it has one hour profile for both day types.
# - Bonus: with the 3:1 cost the mean prediction rises by a factor of {{p1.bonus.mean_shift_ratio:.2f}}; the share of under-predicted validation hours falls from {{p1.bonus.val.mse_model.under_share:.2f}} to {{p1.bonus.val.asym_model.under_share:.2f}} and the asymmetric cost from {{p1.bonus.val.mse_model.sq_cost:.0f}} to {{p1.bonus.val.asym_model.sq_cost:.0f}}, while plain RMSE worsens from {{p1.bonus.val.mse_model.rmse:.1f}} to {{p1.bonus.val.asym_model.rmse:.1f}}.
#
# **What surprised us.**
# - *First pass:* Duan's factor, the textbook correction, was the worst option, and the log target itself turned out to be the main weakness of our whole first chain. We only saw that after Phase 3, which is why there is a second pass.
# - *Second pass, expectation met:* validation R² rose to about 0.735 as predicted, the learning rate is unchanged because the design matrix is the same, and the exponent table peaks at {{p1.target_power_best_here}} for this simple model, above the 0.1 we use.
# - *Second pass, expectation wrong:* we predicted that the back-transform factor would move closer to 1 with the milder transform. It moved away, to {{p1.backtransform.candidates.ls.factor:.3f}} from {{first_pass.p1.backtransform_factor:.3f}}. Without a factor the predictions are still only {{p1.backtransform.candidates.none.val_mean_ratio:.2f}} of the true mean, about the same as with the log, so the milder transform did not reduce the under-shoot of the mean as we had assumed; we do not have a cleaner explanation than that for this additive model.
# - The hour as a plain number is still hopeless (R² {{p1.hour_encoding_ablation.numeric.val_r2:.2f}}): no transform makes a straight line through 0–23 follow a day with two peaks.
# - The plain model, read without its factor, under-predicts about half of the hours ({{p1.bonus.val.mse_model.under_share:.2f}}) but is {{p1.bonus.val.mse_model.mean_error:.0f}} bikes low on average: a fit on a transformed scale aims near the median, and the mean of a skewed count is higher. For an operator who fears empty docks the "neutral" model is already biased the wrong way.
#
# **Handed to Phase 2:** the weight vector, the scaler, the target exponent, lr rule and stopping rule in `artifacts/p1.json`.
