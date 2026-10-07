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
# ### Justification: representation
# - **Target.** We model `log1p(cnt)`. Counts are right-skewed and their spread grows with their level, and effects multiply (a rainy rush hour loses a share of its riders, not a fixed number). R² is still computed on bikes after transforming back.
# - **Hour.** One-hot, 23 dummies with hour 0 as reference. As a single number the hour gives a validation R² of only {{p1.hour_encoding_ablation.numeric.val_r2:.3f}}, because demand does not rise or fall steadily through the day. Six sine/cosine pairs reach {{p1.hour_encoding_ablation.harmonics6.val_r2:.3f}} and the dummies {{p1.hour_encoding_ablation.one_hot.val_r2:.3f}} (table below). We kept the dummies: slightly better, and each weight reads directly as "the effect of this hour".
# - **Weather situation.** Dummies, with category 4 merged into 3 because it has a single training row.
# - **`dteday`.** A trend in days since 1 January 2011 (demand grew strongly) and two day-of-year sine/cosine pairs (the season, without a jump between December and January).
# - **Cleaning.** Humidity 0 occurs on one day only (a sensor failure); those rows get the training median. Wind speed is left as measured.
# - **Left out until Phase 4.** `atemp`, `season`, `mnth`, `yr`, `instant` and `weekday` each repeat information that is already in the design almost exactly (for example `workingday` is an exact function of `weekday` and `holiday`). Duplicates make the weights non-unique and slow gradient descent down. Without them the condition number is {{p1.condition_number:.1f}}. Phase 4 puts them back and lets the models judge them.
# - **Scaling.** Every column is standardised with the training mean and standard deviation only.

# %%
p1 = p1mod.run(CFG)   # fits Phase 1 from scratch and writes artifacts/p1.json
print("training rows        :", p1["n_train"], "| validation rows:", p1["n_val"])
print("columns (with bias)  :", p1["n_features"])
print("target               :", p1["target_transform"])
print("first columns        :", ", ".join(p1["feature_names"][:4]), "...")
print("lambda_max of X'X/n  :", round(p1["lambda_max"], 4))
print("stability bound 2/lambda_max:", round(p1["lr_bound"], 4))
print("learning rate used   :", round(p1["lr"], 4), "=", p1["lr_fraction"], "x bound")
print("condition number     :", round(p1["condition_number"], 1))

# %% [markdown]
# ### Justification: learning rate and stopping rule
# - **Learning rate.** For a quadratic loss, gradient descent is stable only below 2 / λ_max of XᵀX/n. We compute λ_max = {{p1.lambda_max:.4f}} from our design, so the bound is {{p1.lr_bound:.4f}}, and we use {{p1.lr_fraction}} of it: lr = {{p1.lr:.4f}}. At half the bound every direction of the loss surface shrinks without overshooting, so the loss can only go down, and we keep a factor-two safety margin.
# - **Evidence that it converges rather than stalls or diverges.** The sweep below uses the same data and start. At 0.001 and 0.01 of the bound the run is still far from finished after {{p1.lr_sweep.0.iterations}} iterations (stall). At 0.1 it converges but needs {{p1.lr_sweep.2.iterations}} iterations. Our setting needs {{p1.lr_sweep.3.iterations}}. At 0.9 it is faster ({{p1.lr_sweep.4.iterations}}) but leaves almost no margin. At 1.05 the loss explodes and the run is stopped after {{p1.lr_sweep.5.iterations}} iterations (diverge).
# - **Stopping rule.** We stop when the relative change of the loss is at most {{p1.tol_loss}} and the gradient norm is below {{p1.tol_grad}}, with a cap of {{p1.max_iter}} iterations. The loss test alone can fire on a flat stretch far from the minimum; the gradient test alone depends on the scale of the problem; together they mean "flat because we are at the bottom".
# - **Start.** Zero weights, so anyone who re-runs this cell gets exactly the same weight vector.

# %%
sweep = pd.DataFrame([{"fraction of bound": r["fraction"], "lr": r["lr"], "stop": r["stop_reason"],
                       "iterations": r["iterations"], "final loss": r["loss_final"]} for r in p1["lr_sweep"]])
print(sweep.round(4).to_string(index=False))

# %%
plots.plot_lr_sweep(p1)

# %%
print("stop reason :", p1["stop_reason"])
print("iterations  :", p1["iterations"])
print("final loss  :", round(p1["train_loss_final"], 6))
print("gradient norm at the end:", p1["grad_norm_final"], "(tolerance", p1["tol_grad"], ")")
plots.plot_loss_curve(p1)

# %%
oracle = p1["oracle"]
print(oracle["label"])
print("largest weight difference:", oracle["max_abs_weight_diff"])
print("loss gap (GD minus exact):", oracle["loss_gap"])
print("validation R2 difference :", oracle["val_r2_diff"])
print("gradient check (max relative error): at zeros", p1["gradient_check_at_zeros"],
      "| at a random point", p1["gradient_check_at_random"])

# %% [markdown]
# ### Justification: back-transform
# The model predicts log-bikes; `exp(prediction) − 1` systematically under-shoots the mean of `cnt`. We compared three corrections, each fitted on the training rows and judged on validation: no factor (R² {{p1.backtransform.candidates.none.val_r2:.4f}}, mean prediction {{p1.backtransform.candidates.none.val_mean_ratio:.3f}} of the true mean), Duan's smearing factor {{p1.backtransform.candidates.duan.factor:.3f}} (R² {{p1.backtransform.candidates.duan.val_r2:.4f}}, mean ratio {{p1.backtransform.candidates.duan.val_mean_ratio:.3f}}), and a least-squares factor {{p1.backtransform.candidates.ls.factor:.3f}} (R² {{p1.backtransform.candidates.ls.val_r2:.4f}}, mean ratio {{p1.backtransform.candidates.ls.val_mean_ratio:.3f}}). Duan over-corrects: the big log-errors are in quiet night hours, but the factor multiplies the peaks too. We keep the least-squares factor, s = Σ(cnt+1)·e^η / Σ e^{2η}, which is the multiplier that minimises squared error on bikes. The method is now fixed for every later phase; each phase recomputes the factor from its own training residuals.

# %%
candidates = pd.DataFrame(p1["backtransform"]["candidates"]).T
candidates.index.name = "method"
print(candidates.round(4).to_string())
print("chosen:", p1["backtransform"]["method"], "with factor", round(p1["backtransform"]["factor"], 4))

# %%
boot = p1["val_bootstrap"]
scores = pd.DataFrame({"R2": [p1["train_r2"], p1["val_r2"]], "RMSE": [p1["train_rmse"], p1["val_rmse"]],
                       "R2 on log1p(cnt)": [p1["train_r2_log"], p1["val_r2_log"]]}, index=["train", "validation"])
print(scores.round(4).to_string())
print(f"validation R2 95% interval: [{boot['lo']:.4f}, {boot['hi']:.4f}] (standard error {boot['se']:.4f})")

# %%
ablation = pd.DataFrame(p1["hour_encoding_ablation"]).T[["n_features", "val_r2", "iterations", "stop_reason"]]
ablation.index.name = "hour encoding"
print(ablation.to_string())

# %%
plots.plot_residual_profile(p1)

# %% [markdown]
# ### Justification: bonus
# **The operator's cost.** Under-predicting by some amount costs three times as much as over-predicting by the same amount. We read this as a weighted squared error on bikes (the cost is paid in bikes, not log-bikes):
#
# L(w) = (1/n) Σ cᵢ (ŷᵢ − yᵢ)², with ŷᵢ = exp(xᵢ·w) − 1 and cᵢ = 3 if ŷᵢ < yᵢ, else 1.
#
# **Gradient.** By the chain rule, ∂ŷᵢ/∂w = exp(xᵢ·w)·xᵢ, so ∇L = (2/n) Σ cᵢ (ŷᵢ − yᵢ) exp(xᵢ·w) xᵢ. The weight cᵢ jumps only where the residual is zero, so it adds nothing to the gradient. Our finite-difference check agrees to {{p1.bonus.gradient_check:.1e}}.
#
# **Optimiser.** This loss is no longer a quadratic, so the fixed-step bound of the MSE model does not apply. We keep plain gradient descent but choose each step by backtracking (halve the step until the loss decreases enough), starting from the Phase 1 MSE weights. It stopped as "{{p1.bonus.stop_reason}}" after {{p1.bonus.iterations}} iterations. The absolute-error version of the same cost is shown in the table as a sensitivity check. This model is a side study: Phase 2 receives the MSE weights.

# %%
bonus = p1["bonus"]
compare = pd.DataFrame(bonus["val"]).T[["sq_cost", "abs_cost", "under_share", "mean_error", "r2", "rmse", "mean_pred"]]
print(f"asymmetric cost with k = {bonus['k']:g}: {bonus['iterations']} iterations, stop reason {bonus['stop_reason']}")
print("gradient check of the asymmetric loss:", bonus["gradient_check"])
print(compare.round(4).to_string())
print("mean prediction ratio asymmetric / MSE:", round(bonus["mean_shift_ratio"], 4))

# %%
plots.plot_bonus_shift(p1)

# %% [markdown]
# ## Outcome — Phase 1
#
# **What happened.**
# - Gradient descent converged in {{p1.iterations}} iterations at lr = {{p1.lr:.4f}} (λ_max {{p1.lambda_max:.3f}}, as expected close to 2). The weights differ from the closed-form solution by at most {{p1.oracle.max_abs_weight_diff:.1e}} and the gradient check agrees to {{p1.gradient_check:.1e}}.
# - Validation R² is {{p1.val_r2:.4f}} (95% interval {{p1.val_bootstrap.lo:.3f}}–{{p1.val_bootstrap.hi:.3f}}), RMSE {{p1.val_rmse:.1f}} bikes; training R² is {{p1.train_r2:.4f}}, so there is essentially no gap, as expected for 36 weights.
# - The residual plot shows the pattern we predicted: on non-working days the model is far too low at night and too high at commute hours, because it has one hour profile for both day types.
# - Bonus: with the 3:1 cost the mean prediction rises by a factor of {{p1.bonus.mean_shift_ratio:.2f}}; the share of under-predicted validation hours falls from {{p1.bonus.val.mse_model.under_share:.2f}} to {{p1.bonus.val.asym_model.under_share:.2f}} and the asymmetric cost from {{p1.bonus.val.mse_model.sq_cost:.0f}} to {{p1.bonus.val.asym_model.sq_cost:.0f}}, while plain RMSE worsens from {{p1.bonus.val.mse_model.rmse:.1f}} to {{p1.bonus.val.asym_model.rmse:.1f}}.
#
# **What surprised us.**
# - Duan's factor was not just unhelpful, it cost {{p1.backtransform.candidates.none.val_r2:.3f}} → {{p1.backtransform.candidates.duan.val_r2:.3f}} in R². A small least-squares factor was the best of the three, which we had not expected to differ from "no factor".
# - The hour as a plain number was even worse than we thought (R² {{p1.hour_encoding_ablation.numeric.val_r2:.2f}}): in log space the night trough dominates and a straight line through 0–23 cannot follow it.
# - The plain MSE model under-predicts more than half of the hours ({{p1.bonus.val.mse_model.under_share:.2f}}) and is {{p1.bonus.val.mse_model.mean_error:.0f}} bikes low on average. A log-scale fit aims at the median, not the mean, so for an operator who fears empty docks the "neutral" model is already biased the wrong way.
#
# **Handed to Phase 2:** the weight vector, the scaler, lr rule and stopping rule in `artifacts/p1.json`.
