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
# - **Hour.** 23 dummies, with hour 0 as the reference. The hour as one number reaches a validation R² of only {{p1.hour_encoding_ablation.numeric.val_r2:.3f}}, six sine/cosine pairs reach {{p1.hour_encoding_ablation.harmonics6.val_r2:.3f}} and the dummies {{p1.hour_encoding_ablation.one_hot.val_r2:.3f}}. We keep the dummies: they score best and each weight reads as "the effect of this hour".
# - **Weather situation.** Dummies, with category 4 merged into 3 because it has a single training row.
# - **`dteday`.** A trend in days since 1 January 2011 and two day-of-year sine/cosine pairs, so the season has no jump at New Year.
# - **Cleaning and scaling.** The one day with humidity 0 gets the training median. Every column is standardised with training rows only.
# - **Held back until Phase 4.** `atemp`, `season`, `mnth`, `yr`, `instant` and `weekday` repeat information that is already in the design. Without them the condition number is {{p1.condition_number:.1f}}, which keeps gradient descent fast. Phase 4 puts them back and judges them.

# %%
# Phase 1 is fitted from scratch here and checked against the stored artifact (Phases 1 to 3 are quick).
import src.phases.p1 as p1mod
from src.gd import gradient_descent, mse_grad, run_gd
from src.validation import base_design, make_split

p1 = phase("p1", p1mod.run, CFG, RECOMPUTE_ALL, live=True)

# %%
# The design Phase 1 was given and the learning rate it follows from.
table({"training rows": p1["n_train"], "columns (with bias)": p1["n_features"], "target exponent": p1["target_power"],
       "λ_max of XᵀX/n": p1["lambda_max"], "stability bound 2/λ_max": p1["lr_bound"], "learning rate": p1["lr"],
       "condition number": p1["condition_number"]}, "Phase 1: the design and the learning rate")

# %%
# Spot-check: our gradient descent from zero weights, run again here, must reproduce the stored weights.
split = make_split(CFG)
X_train = base_design(split)[1]
knobs = CFG["p1"]
live = run_gd(X_train, split.z_tr, np.zeros(X_train.shape[1]), p1["lr"], knobs["tol_loss"], knobs["tol_grad"], knobs["max_iter"])
gap = np.abs(live.weights - np.array(p1["weights"])).max()
assert gap < 1e-8, f"the live run does not reproduce the stored weights ({gap:.1e})"
print(f"{live.stop_reason} after {live.iterations} iterations; largest weight difference to the stored run {gap:.1e}")
show_source(gradient_descent, mse_grad)

# %% [markdown]
# ### Justification: learning rate and stopping rule
#
# - **Learning rate.** Gradient descent on a quadratic loss is stable below 2 / λ_max of XᵀX/n. Our design gives λ_max = {{p1.lambda_max:.4f}} and a bound of {{p1.lr_bound:.4f}}. We use {{p1.lr_fraction}} of the bound (lr = {{p1.lr:.4f}}), which keeps a factor-two safety margin.
# - **Evidence.** In the sweep below, {{p1.lr_sweep.0.fraction}} and {{p1.lr_sweep.1.fraction}} of the bound are still far from the minimum after {{p1.lr_sweep.0.iterations}} iterations. {{p1.lr_sweep.2.fraction}} needs {{p1.lr_sweep.2.iterations}} iterations and our setting {{p1.lr_sweep.3.iterations}}. {{p1.lr_sweep.4.fraction}} is faster ({{p1.lr_sweep.4.iterations}}) but leaves almost no margin, and {{p1.lr_sweep.5.fraction}} diverges and is stopped after {{p1.lr_sweep.5.iterations}}.
# - **Stopping rule.** We stop when the relative change of the loss is at most {{p1.tol_loss}} and the gradient norm is below {{p1.tol_grad}}, with a cap of {{p1.max_iter}} iterations. Together the two tests mean "flat because we are at the bottom", not a flat stretch on the way down.
# - **Start.** Zero weights, so every re-run gives the same weight vector.

# %%
# Six learning rates from far too small to too large: what each one does, and the loss curves.
sweep = pd.DataFrame(p1["lr_sweep"])
table(sweep[["fraction", "lr", "stop_reason", "iterations", "loss_final"]], "Learning-rate sweep (highlighted: the one we use)",
      np.isclose(sweep["fraction"], p1["lr_fraction"]))
show(plots.plot_lr_sweep(p1))

# %%
# Correctness: weights against the exact least-squares solution, gradient checks, and the scores with their interval.
oracle, boot = p1["oracle"], p1["val_bootstrap"]
table({"largest weight difference to the closed form": oracle["max_abs_weight_diff"], "loss gap (GD minus exact)": oracle["loss_gap"],
       "gradient check at zeros": p1["gradient_check_at_zeros"], "gradient check at a random point": p1["gradient_check_at_random"],
       "train R², RMSE": [p1["train_r2"], p1["train_rmse"]], "validation R², RMSE": [p1["val_r2"], p1["val_rmse"]],
       "validation R² 95% interval": [boot["lo"], boot["hi"]]}, "Phase 1 against the closed-form solution")

# %% [markdown]
# ### Justification: back-transform
#
# Without a correction, the predictions average only {{p1.backtransform.candidates.none.val_mean_ratio:.3f}} of the true mean on validation. We fit one least-squares factor on the training rows, s = Σ(cnt+1)·q / Σq² = {{p1.backtransform.candidates.ls.factor:.3f}}, which raises validation R² from {{p1.backtransform.candidates.none.val_r2:.4f}} to {{p1.backtransform.candidates.ls.val_r2:.4f}}, so we keep it. Every later phase refits the factor on its own training residuals. Duan's smearing factor exists for the log only; in our first pass it lowered R² from {{first_pass.p1.none_val_r2:.3f}} to {{first_pass.p1.duan_val_r2:.3f}}, because it inflates the peaks to repair errors that sit in quiet night hours.

# %%
# The two back-transform options, each fitted on the training rows and judged on the validation rows.
options = pd.DataFrame(p1["backtransform"]["candidates"]).T.rename_axis("method").reset_index()
table(options, "Back-transform (highlighted: our choice)", options["method"] == p1["backtransform"]["method"])

# %% [markdown]
# ### Justification: target transform
#
# We fit z = ((cnt + 1)^λ − 1) / λ with λ = {{p1.target_power}}. λ → 0 gives the log and λ = 1 the raw count; predictions come back to bikes through the inverse.
#
# - **Not the log.** Our first pass (git tag `v1-submitted`) used `log1p(cnt)`. On that scale an error of four bikes at 3 am weighs as much as an error of two hundred at 5 pm, while R² on bikes is decided by the busy hours.
# - **Not the raw count.** At λ = 1 the effects become additive in bikes, and validation R² falls to {{p1.target_power_table.7.val_r2:.3f}}.
# - **Why not this phase's own best.** In the table below Phase 1 alone prefers λ = {{p1.target_power_best_here}} ({{p1.target_power_table.5.val_r2:.4f}} against {{p1.target_power_table.2.val_r2:.4f}} at {{p1.target_power}}). The exponent must be the same in every phase, or Phase 2 could not start from these weights. We fixed it from pilot fits of the final design, made after we had looked at all three validators of Phase 3, so this is not a Phase 1 decision. Phase 3 checks it again on its target design.

# %%
# The same design under each candidate exponent of the target transform (0 = log, 1 = raw counts).
power = pd.DataFrame(p1["target_power_table"])
table(power[["power", "iterations", "train_r2", "val_r2", "val_rmse"]], f"Target exponent (highlighted: in use, {p1['target_power']})",
      np.isclose(power["power"], p1["target_power"]))

# %%
# What is left over: the residuals by hour and day type. They motivate Phase 2.
show(plots.plot_residual_profile(p1))

# %% [markdown]
# ### Justification: bonus
#
# **Loss.** Under-predicting costs {{p1.bonus.k:g}} times as much as over-predicting by the same amount. The cost is paid in bikes, so we use a weighted squared error on bikes:
#
# L(w) = (1/n) Σ cᵢ (ŷᵢ − yᵢ)², with ŷᵢ = q(xᵢ·w) − 1, q(η) = (λη + 1)^(1/λ), and cᵢ = k if ŷᵢ < yᵢ, else 1.
#
# **Gradient.** ∂ŷᵢ/∂w = q(ηᵢ)^(1−λ) xᵢ, so ∇L = (2/n) Σ cᵢ (ŷᵢ − yᵢ) q(ηᵢ)^(1−λ) xᵢ. The weight cᵢ jumps only where the residual is zero, so it adds no term. Our finite-difference check differs by {{p1.bonus.gradient_check:.9f}}.
#
# **Optimiser.** The loss is not quadratic, so the fixed-step bound does not apply. We use gradient descent with backtracking (halve the step until the loss falls enough), started from the Phase 1 weights; it stopped as "{{p1.bonus.stop_reason}}" after {{p1.bonus.iterations}} iterations. Phase 2 receives the MSE weights, not these.
#
# **A control.** The bonus model changes two things at once: it is fitted on bikes, and it weights misses {{p1.bonus.k:g}} times. The "k = 1" row of the table fits on bikes without the asymmetry. Fitting on bikes alone raises the mean prediction by a factor {{bonus_control.shift_from_fitting_on_bikes:.3f}}, and the asymmetry adds a factor {{bonus_control.shift_from_asymmetry:.3f}}, for a total of {{bonus_control.total_shift:.2f}}.
#
# **Which baseline.** The table's "MSE model" is the Phase 1 model without its back-transform factor (validation R² {{p1.bonus.val.mse_model.r2:.4f}}, against {{p1.val_r2:.4f}} with the factor). Measured against the reported Phase 1 model, which already multiplies its predictions by {{p1.backtransform.candidates.ls.factor:.3f}}, the upward shift is smaller than {{bonus_control.total_shift:.2f}}.

# %%
# The bonus: the plain model, the same bike-scale loss without asymmetry (k = 1, a control), and the asymmetric model.
import src.bonus_control as bonus_control

control = bonus_control.run(CFG) if RECOMPUTE_ALL else read_artifact("bonus_control")
val = p1["bonus"]["val"]
models = {"MSE model": val["mse_model"], "k = 1 control": control["val"]["k1_model"], f"asymmetric, k = {p1['bonus']['k']:g}": val["asym_model"]}
frame = pd.DataFrame(models).T.rename_axis("model").reset_index()
table(frame[["model", "under_share", "mean_error", "mean_pred", "rmse", "sq_cost", "abs_cost"]], "The bonus on the validation rows")

# %% [markdown]
# ## Outcome — Phase 1
#
# **What happened.**
# - Gradient descent converged in {{p1.iterations}} iterations at lr = {{p1.lr:.4f}}, with λ_max = {{p1.lambda_max:.3f}} close to 2 as expected. The weights match the closed-form solution and both gradient checks pass (correctness table above).
# - Validation R² is {{p1.val_r2:.4f}} (95% interval {{p1.val_bootstrap.lo:.3f}}–{{p1.val_bootstrap.hi:.3f}}) and RMSE {{p1.val_rmse:.1f}} bikes; training R² is {{p1.train_r2:.4f}}, so there is no gap. With the log target of our first pass the same model scored {{first_pass.p1.val_r2:.4f}}.
# - The residual plot shows the pattern we predicted: with one hour profile for both day types, non-working days are predicted too low at night and too high at commute hours.
# - Bonus: the share of under-predicted validation hours falls from {{p1.bonus.val.mse_model.under_share:.2f}} to {{p1.bonus.val.asym_model.under_share:.2f}} and the asymmetric cost from {{p1.bonus.val.mse_model.sq_cost:.0f}} to {{p1.bonus.val.asym_model.sq_cost:.0f}}, while plain RMSE rises from {{p1.bonus.val.mse_model.rmse:.1f}} to {{p1.bonus.val.asym_model.rmse:.1f}}.
#
# **What surprised us.**
# - *First pass:* Duan's factor was the worst option, and the log target was the main weakness of the whole first chain. We only saw this after Phase 3, which is why there is a second pass.
# - *Second pass:* validation R² rose as predicted and the learning rate did not change. We were wrong about the back-transform: the factor moved away from 1 ({{first_pass.p1.backtransform_factor:.3f}} → {{p1.backtransform.candidates.ls.factor:.3f}}), not towards it, and we have no clean explanation for this additive model.
# - The plain model under-predicts about half of the hours, yet its mean error is {{p1.bonus.val.mse_model.mean_error:.0f}} bikes. A fit on a transformed scale aims near the median, which is below the mean of a skewed count. For an operator who fears empty docks, the "neutral" model is already biased the wrong way.
#
# **Handed to Phase 2:** the weights, the scaler, the target exponent and the learning-rate and stopping rules, in `artifacts/p1.json`.
