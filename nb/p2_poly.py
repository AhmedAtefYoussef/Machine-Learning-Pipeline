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
# ### Justification: initialisation from Phase 1
# TODO(chief)

# %%
p2 = p2mod.run(CFG)   # reads artifacts/p1.json, fits Phase 2 and writes artifacts/p2.json
difference = p2["init_loss"] - p1["train_loss_final"]
print("Phase 1 final training loss :", repr(p1["train_loss_final"]))
print("Phase 2 initial loss        :", repr(p2["init_loss"]))
print("difference                  :", difference)
assert abs(difference) <= 1e-9, "the hand-over from Phase 1 to Phase 2 is broken"
print("hand-over check passed (|difference| <= 1e-9)")

# %% [markdown]
# ### Justification: what we expanded
# TODO(chief)

# %%
print("blocks      :", p2["blocks"])
print("power cols  :", p2["power_cols"])
print("columns     :", len(p2["feature_names"]), "(", len(p2["expanded_feature_names"]), "new )")
print("new columns :", ", ".join(p2["expanded_feature_names"][:6]), "...")
print("lambda_max  :", round(p2["lambda_max"], 4), "| lr:", round(p2["lr"], 4),
      "| condition number:", round(p2["condition_number"], 1))

# %% [markdown]
# ### Justification: degree
# TODO(chief)

# %%
rows = pd.DataFrame(p2["degree_sweep"])[["degree", "n_features", "iterations", "stop_reason", "train_r2", "val_r2"]]
print(rows.round(4).to_string(index=False))
print("chosen degree:", p2["degree"], "(smallest within", p2["plateau_tol"], "of the best validation R2)")

# %%
plots.plot_degree_sweep(p2)

# %%
cols = pd.DataFrame(p2["power_col_sweep"])[["power_cols", "n_features", "iterations", "stop_reason", "train_r2", "val_r2"]]
cols["power_cols"] = cols["power_cols"].apply(lambda c: ", ".join(c))
print(cols.round(4).to_string(index=False))
print("chosen power columns:", p2["power_cols"])

# %%
ablation = p2["block_ablation"]["without_wd_x_hr"]
print("without the workingday x hour block:", ablation["n_features"], "columns, validation R2",
      round(ablation["val_r2"], 4), "(train", round(ablation["train_r2"], 4), ")")
print("with the block                     :", len(p2["feature_names"]), "columns, validation R2",
      round(p2["val_r2"], 4), "(train", round(p2["train_r2"], 4), ")")

# %%
paired = p2["paired_vs_p1"]
boot = p2["val_bootstrap"]
final = pd.DataFrame({"R2": [p1["val_r2"], p2["val_r2"]], "RMSE": [p1["val_rmse"], p2["val_rmse"]]},
                     index=["Phase 1", "Phase 2"])
print(final.round(4).to_string())
print(f"Phase 2 validation R2 95% interval: [{boot['lo']:.4f}, {boot['hi']:.4f}]")
print(f"gain over Phase 1: {paired['delta']:.4f}, paired 95% interval [{paired['lo']:.4f}, {paired['hi']:.4f}]")
print("stop reason:", p2["stop_reason"], "after", p2["iterations"], "iterations; oracle max weight difference:",
      p2["oracle"]["max_abs_weight_diff"])

# %%
plots.plot_residual_profile(p1, p2)

# %% [markdown]
# ## Outcome — Phase 2
# TODO(chief)
