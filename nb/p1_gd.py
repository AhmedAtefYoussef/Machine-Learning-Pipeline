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
# TODO(chief)

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
# TODO(chief)

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
# TODO(chief)

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
# TODO(chief)

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
# TODO(chief)
