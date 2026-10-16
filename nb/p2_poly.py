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
#
# The Phase 1 columns keep their positions and the Phase 1 scaler, read from `artifacts/p1.json`; each new column is standardised on the training rows. The weight vector is the Phase 1 vector with zeros for the new columns, so the expanded model starts as exactly the Phase 1 model, and its first loss must equal Phase 1's last loss (asserted below). Gradient descent then only has to learn the correction that the new columns allow.

# %%
# Phase 2 starts from the Phase 1 weights, so its first loss must equal Phase 1's last loss (the hand-over).
import src.phases.p2 as p2mod
from src.poly import lift_weights

p2 = phase("p2", p2mod.run, CFG, RECOMPUTE_ALL, upstream="p1", live=True)
assert abs(p2["init_loss"] - p1["train_loss_final"]) <= 1e-9, "the hand-over from Phase 1 to Phase 2 is broken"
print(f"first Phase 2 loss {p2['init_loss']:.9f} = last Phase 1 loss {p1['train_loss_final']:.9f}; "
      f"{len(p2['feature_names'])} weights, {p2['iterations']} iterations")
show_source(lift_weights)

# %% [markdown]
# ### Justification: what we expanded, and the degree
#
# - **`workingday × hour`.** Phase 1's residuals differ by day type at almost every hour. Without these products the model reaches {{p2.block_ablation.without_wd_x_hr.val_r2:.4f}} on validation, with them {{p2.val_r2:.4f}} (table below).
# - **Powers of `temp` only.** At the chosen degree, powers of `temp` give {{p2.power_col_sweep.0.val_r2:.4f}}, adding `hum` gives {{p2.power_col_sweep.1.val_r2:.4f}} and adding `windspeed` too gives {{p2.power_col_sweep.2.val_r2:.4f}}. These differences are below our tolerance of {{p2.plateau_tol}} and inside the noise (validation standard error {{p2.val_bootstrap.se:.4f}}), so we keep the smallest set.
# - **Not expanded.** A 0/1 column equals its own powers, a power of the trend would bend away outside the observed dates, and the day-of-year terms are already curved.

# %% [markdown]
# ### Justification: degree
#
# The degree is the highest total degree of any term: the products are degree 2 and the powers go up to the chosen degree. Validation R² is {{p2.degree_sweep.0.val_r2:.4f}} at degree 1 (products only), {{p2.degree_sweep.1.val_r2:.4f}} at 2, {{p2.degree_sweep.2.val_r2:.4f}} at 3 and {{p2.degree_sweep.3.val_r2:.4f}} at 4. Our rule, fixed before the run, takes the smallest degree within {{p2.plateau_tol}} of the best: degree {{p2.degree}}. The learning rate is again half the stability bound of this design (lr {{p2.lr:.4f}}), with the Phase 1 stopping rule.

# %%
# Validation R² for each degree, for each set of power columns, and without the workingday x hour block.
runs = [(f"degree {r['degree']}", r) for r in p2["degree_sweep"]]
runs += [("powers of " + ", ".join(r["power_cols"]), r) for r in p2["power_col_sweep"]]
runs += [("without the workingday x hour block", p2["block_ablation"]["without_wd_x_hr"])]
sweep = pd.DataFrame([{"setting": name, "weights": r["n_features"], "train_r2": r["train_r2"], "val_r2": r["val_r2"]} for name, r in runs])
chosen = [f"degree {p2['degree']}", "powers of " + ", ".join(p2["power_cols"])]
table(sweep, "What we varied in Phase 2 (highlighted: the choices we kept)", sweep["setting"].isin(chosen))

# %%
# The residuals by hour and day type before (Phase 1) and after (Phase 2) the expansion.
show(plots.plot_residual_profile(p1, p2))

# %% [markdown]
# ## Outcome — Phase 2
#
# **What happened.**
# - The hand-over held: the first Phase 2 loss equals the last Phase 1 loss ({{p2.init_loss:.6f}}).
# - The chosen model has degree {{p2.degree}}, powers of `temp` and the `workingday × hour` block: {{p2.power_col_sweep.0.n_features}} weights. Validation R² rose from {{p2.p1_val_r2:.4f}} to {{p2.val_r2:.4f}} (gain {{p2.paired_vs_p1.delta:.3f}}, paired 95% interval {{p2.paired_vs_p1.lo:.3f}}–{{p2.paired_vs_p1.hi:.3f}}) and RMSE is {{p2.val_rmse:.1f}} bikes. Training R² is {{p2.train_r2:.4f}}, so there is still no gap.
# - Almost all of the gain is the interaction block, and the residual profile by hour and day type is now flat. With the log target the same choices scored {{first_pass.p2.val_r2:.4f}}.
#
# **What surprised us.**
# - Optimisation was easy: {{p2.iterations}} iterations, not thousands. Powers and products of standardised columns kept the condition number at {{p2.condition_number:.0f}}.
# - Humidity did not need a curve, although we expected it to.
# - In the second pass the cube of temperature earns its place more clearly than before ({{p2.degree_sweep.1.val_r2:.4f}} at degree 2, {{p2.degree_sweep.2.val_r2:.4f}} at degree 3).
# - Training and validation scores are almost equal, so the model is not over-fit. Whether it is too simple is the question for Phase 3.
#
# **Handed to Phase 3:** the degree and the expanded feature list, in `artifacts/p2.json`.
