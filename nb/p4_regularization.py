# %% [markdown]
# ## Expectation — Phase 4: Regularization (L1, L2, Elastic Net)
#
# *Written before running the phase (see the git history of this file).*
#
# **What we do.** We load the target complexity from `artifacts/p3.json`, add back the six columns we had kept out for conditioning reasons (`atemp`, `yr`, `instant`, `season`, `mnth`, `weekday`), and fit Ridge, Lasso and Elastic Net with our own solvers on exactly the same standardised design, the same training rows and the same grid density. For each method we pick the penalty by validation R² and then check that choice on held-out days and on the chronological split. Finally we judge every original column with drop tests, lasso selection frequency and correlations.
#
# **What we expect, and why.**
#
# 1. *Regularization will be a robustness choice, not an accuracy one.* Phase 3 should hand us a model that is not over-fit, so we expect the best penalties to be small and all three methods to land within about ±0.002 validation R² of the unpenalised fit, which is inside the bootstrap noise. We do not expect any method to "win" clearly.
# 2. *Lasso will not be very sparse.* The signal here is spread over many hour-specific columns. We expect Lasso at its best penalty to keep well over half of the coefficients.
# 3. *Lasso will be unreliable on duplicates.* `temp` and `atemp` are correlated at 0.985. We expect Lasso to keep both at small penalties, or to pick one of them arbitrarily, and Elastic Net to share the weight between them. This is why a zero from Lasso is only one piece of evidence for us, next to the drop tests and the selection frequency over resampled days.
# 4. *Where the penalty should matter.* On the most complex ladder level, which Phase 3 should show to be over-fit, we expect a penalty to recover most of the lost validation R², but not to beat the target design by more than the noise.
# 5. *Column verdicts we predict.* Useful: `hr`, `temp`, `hum`, `weathersit`, and the day-type information. Redundant: `atemp` (carried by `temp`), `season` and `mnth` (carried by the day-of-year terms from `dteday`), `yr` and `instant` (carried by the trend from `dteday`), and at least one of `weekday` / `workingday` / `holiday`, since `workingday` is an exact function of the other two. We are least sure about `windspeed` and `holiday`: their effect may be too small to separate from noise, in which case the honest verdict is "uninformative".
# 6. *Surprise we are watching for.* If dropping a column we believe to be useful costs nothing, another column is carrying its information and we have the labels the wrong way round.

# %% [markdown]
# ## Expectation — Phase 4, second pass
#
# *Written before re-running the chain with the power target (λ = 0.1). The cell above is unchanged.*
#
# 1. *Regularization* should again be insurance: three methods tied within noise, tiny penalties, Lasso not sparse.
# 2. *Verdicts* should be the same as in the first pass, with one candidate for a flip: `windspeed` was useful by a
#    small margin and could fall back to uninformative.
# 3. *Final model.* About 0.941 on validation, 0.946 on held-out days and 0.93 to 0.94 chronologically.

# %% [markdown]
# ### Justification: search ranges and strategy
#
# - **Phase 3 shapes the search.** The model we carry forward is under-fit, so we expect small penalties. The grids reach practically zero, and the unpenalised fit is a reference row. If a best penalty is the smallest value on its grid we read it as "no penalty wanted", not as a finding about that value, and we do not extend the grid again.
# - **Design.** The Phase 3 target design plus the held-back columns `atemp`, `yr`, `instant`, `season` and `mnth`. The Expectation also lists `weekday`, but that column is already in the target design. Columns are standardised on training rows and the intercept is not penalised.
# - **Objective.** (1/2n)‖y − b − Xw‖² + α·ρ‖w‖₁ + (α/2)(1 − ρ)‖w‖², with ρ = 0 for Ridge, ρ = 1 for Lasso and a grid of ρ values for Elastic Net. Ridge is solved in closed form; Lasso and Elastic Net by our own cyclic coordinate descent with soft-thresholding, warm-started along the path.
# - **Grids.** 40 log-spaced values. Ridge runs from {{p4.grids.l2.min:.6f}} to {{p4.grids.l2.max:.0f}}; Lasso from the smallest penalty that zeroes every weight ({{p4.grids.l1.max:.3f}}) down to {{p4.grids.l1.min:.6f}}. Elastic Net uses the same 40 penalties for each mixing ratio ρ in {0.1, 0.3, 0.5, 0.7, 0.9, 0.95}; we stop at 0.95 because ρ = 1 is the Lasso itself.
# - **Fits that did not converge.** {{p4.not_converged.count}} Lasso fits at one large grid value used all {{p4.solver.max_sweeps}} sweeps, because exact copies slow coordinate descent there. None of them is a chosen model.

# %%
# Phase 4 adds the held-back columns to the Phase 3 target design; the slowest phase, so a valid stored artifact is loaded.
from src import plots_p4
from src.common import sha256_file
from src.phases import p4 as p4mod
from src.regularization import cd_sweep, soft_threshold

p4 = phase("p4", p4mod.run, CFG, RECOMPUTE_ALL, upstream="p3")   # python run.py p4 recomputes it

# %%
# The chain, asserted: built on the Phase 3 artifact on disk, from its target level and width.
origin, level = p4["design_from"], p3["target_complexity"]
assert p4["upstream_sha256"] == sha256_file("artifacts/p3.json"), "Phase 4 was not built from this Phase 3 artifact"
assert origin["level"] == level["level"] and origin["n_features_before_candidates"] == level["n_features"], "wrong Phase 3 level"
print(f"level {origin['level']}: {origin['n_features_before_candidates']} weights + {len(origin['candidates_added'])} candidate columns "
      f"= {p4['n_features']}; solver runs that used all sweeps: {p4['not_converged']['count']}")
show_source(soft_threshold, cd_sweep)

# %%
# How the validation R² of each method moves with the penalty, and the penalty each one chose (highlighted: best).
show(plots_p4.plot_validation_curves(p4))
none = p4["unregularised"]
methods = pd.DataFrame({**p4["methods"], "none": {**none, "lambda": none["alpha"]}}).T.rename_axis("method").reset_index()
table(methods[["method", "lambda", "l1_ratio", "val_r2", "day_block_r2", "chrono_r2", "n_nonzero", "n_features"]],
      "Chosen penalty per method; 'none' is the same design with a negligible penalty", methods["method"] == p4["full_design_best"]["method"])

# %% [markdown]
# ### Justification: fair comparison of the three methods
#
# All three methods get the same design, rows, day folds, grid density and selection rule. Elastic Net has a second knob and so more candidates, which is a small advantage on validation; that is one reason we also report held-out days and the chronological split. For each pair of methods we give a paired bootstrap interval of the difference on the same validation rows. An interval that contains zero is a tie.

# %%
# Fair comparison: paired bootstrap differences between the methods (an interval that contains zero is a tie).
diffs = pd.DataFrame(p4["comparisons"]).T.rename_axis("comparison").reset_index()
table(diffs, "Differences in validation R² between methods, with their intervals")

# %% [markdown]
# ### Justification: choice of lambda per method
#
# For each method we take the penalty with the best validation R² on bikes, with ties going to the larger penalty, because tuning uses the validation set. The table below also shows the penalty that is best on held-out days and the "one standard error" penalty. If the validation choice were an accident of this split, these would disagree strongly with it.

# %%
# Two ways to pick the penalty: best validation R² against held-out-day cross-validation (best and 1-s.e. rule).
picks = pd.DataFrame(p4["cv"]).T.rename_axis("method").reset_index()
picks["validation_best"] = [p4["methods"][m]["lambda"] for m in picks["method"]]
picks["r2_at_validation_best"] = [p4["methods"][m]["val_r2"] for m in picks["method"]]
table(picks[["method", "validation_best", "cv_best_alpha", "cv_1se_alpha", "r2_at_validation_best", "val_r2_at_cv_best", "val_r2_at_cv_1se"]],
      "Penalty by validation and by held-out-day CV")

# %%
# How the lasso coefficients shrink as the penalty grows.
show(plots_p4.plot_paths(p4, "l1"))

# %% [markdown]
# ### Justification: the verdict rule
#
# We fixed the rule before the run. The reference is Ridge at its chosen penalty on the full design. Dropping a column removes every feature built from it, then we refit.
#
# - **Useful:** dropping it alone costs at least {{p4.verdict_thresholds.min_delta}} validation R², with a paired interval above zero.
# - **Redundant:** dropping it alone costs nothing, but dropping it with its partners hurts, or on its own it explains at least {{p4.verdict_thresholds.solo_min}} of the variance. Partners are groups we measured to be copies: {`temp`, `atemp`}, {`season`, `mnth`, `dteday`}, {`yr`, `instant`, `dteday`}, {`weekday`, `workingday`, `holiday`}.
# - **Uninformative:** neither.
# - **One representative per group.** If a group matters but no single member is missed, we keep the member that explains most on its own and call it useful.
#
# Lasso selection frequency is shown next to the drop tests but never decides a verdict alone: with copies in the design, which copy Lasso keeps is partly arbitrary.

# %%
# The verdict for every original column, from the drop tests, and the drop tests as a plot.
verdicts = [{"column": c, "verdict": v["verdict"], "drop_alone": v["numbers"]["drop_alone"]["delta"], "solo_r2": v["numbers"]["solo_r2"],
             "lasso_max_freq": v["numbers"]["lasso"]["max_freq_own"], "carried_by": v["carried_by"]} for c, v in p4["column_verdicts"].items()]
table(pd.DataFrame(verdicts), "Verdict for every original column (drop_alone = validation R² lost when the column is removed)")
show(plots_p4.plot_verdicts(p4))

# %% [markdown]
# ### Justification: the survivor rule
#
# We first remove every feature of a column that is not useful ({{p4.survivor_counts.useful_design}} features remain). On that design we run Lasso again, with the penalty chosen by validation R² ({{p4.selection.lambda:.6f}}). A feature survives if its weight is non-zero ({{p4.survivor_counts.lasso_nonzero}} features) and Lasso selects it in at least {{p4.stability_threshold}} of 50 fits on resampled training days ({{p4.survivor_counts.stable}} features).
#
# To be plain about it: the chosen penalty is the smallest on the grid ({{p4.grids.l1.min:.6f}}), so the optimum is not bracketed, and Lasso keeps {{p4.selection.n_nonzero}} of {{p4.selection.n_columns}} features. The real selection is done by the verdicts, which are drop tests scored on the validation rows. A truly sparse Lasso, at the one-standard-error penalty of the day folds, would score {{p4.final.cv.l1.val_r2_at_cv_1se:.4f}} on the survivors instead of {{p4.recommended.val_r2:.4f}}.
#
# Our first version took the Lasso zeros before removing the copies. Lasso then let `yr`, `instant` and the month dummies stand in for trend terms, the verdicts removed the stand-ins, and the chronological score fell. Lasso should choose only when its choices are no longer between copies.

# %%
# The survivors: Lasso on the design of useful columns only, non-zero at its chosen penalty and stable over resampled days.
pick = p4["selection"]
table({**{f"survivors, {k}": v for k, v in p4["survivor_counts"].items()}, "selection lasso: penalty": pick["lambda"],
       "selection lasso: validation R²": pick["val_r2"], "selection lasso: non-zero": f"{pick['n_nonzero']} of {pick['n_columns']}",
       "original columns kept": p4["survivors_original"], "expanded survivor columns": len(p4["survivors_expanded"])},
      "The survivors handed to Phase 5")

# %% [markdown]
# ### Justification: final model on the surviving columns
#
# Stage A tells us which columns matter, and for that it needs every column. The redundant ones add nothing to a shipped model: dropping `mnth` alone changes validation R² by {{p4.column_verdicts.mnth.numbers.drop_alone.delta:.4f}} (interval {{p4.column_verdicts.mnth.numbers.drop_alone.lo:.4f}} to {{p4.column_verdicts.mnth.numbers.drop_alone.hi:.4f}}; negative means better without it), and several of them are extra descriptions of time, where the chronological split shows the model to be fragile. So we repeat the search on the survivors (stage B) and recommend from there. The rule for choosing among methods was fixed before the first Phase 4 run: among methods within paired noise of the best on validation, take the best on held-out days. Stage B itself was added after that first run; the earlier runs are in the git history.

# %%
# Stage B: the same search on the surviving columns only (highlighted: the method we recommend).
final, rec = p4["final"], p4["recommended"]
none = final["unregularised"]
stage_b = pd.DataFrame({**final["methods"], "none": {**none, "lambda": none["alpha"]}}).T.rename_axis("method").reset_index()
table(stage_b[["method", "lambda", "l1_ratio", "val_r2", "day_block_r2", "chrono_r2", "n_nonzero", "n_features"]],
      f"Stage B on {final['n_columns']} columns; within noise of the best: {rec['candidates']}; solver runs that used all sweeps: "
      f"{p4['not_converged_final']['count']}", stage_b["method"] == rec["method"])

# %% [markdown]
# ## Outcome — Phase 4
#
# **What happened.**
# - **Where there is variance, the penalty works.** On the over-fit top ladder level, Lasso raises validation R² from {{p4.rich_check.unregularised_val_r2:.4f}} to {{p4.rich_check.l1.val_r2:.4f}} and keeps {{p4.rich_check.l1.n_nonzero}} of {{p4.rich_check.n_features}} weights. That is still not better than our recommended model (difference {{p4.rich_check.paired_best_vs_recommended.delta:.4f}}, interval {{p4.rich_check.paired_best_vs_recommended.lo:.4f}} to {{p4.rich_check.paired_best_vs_recommended.hi:.4f}}).
# - **On the full target design the three methods tie.** Ridge {{p4.methods.l2.val_r2:.4f}}, Lasso {{p4.methods.l1.val_r2:.4f}} and Elastic Net {{p4.methods.enet.val_r2:.4f}}, against {{p4.unregularised.val_r2:.4f}} without a penalty; every paired interval between methods contains zero. The penalties are small (Ridge {{p4.methods.l2.lambda:.3f}}, Lasso {{p4.methods.l1.lambda:.6f}}), and Elastic Net chose l1_ratio {{p4.methods.enet.l1_ratio}}, which is almost Lasso. Lasso is not sparse: it keeps {{p4.methods.l1.n_nonzero}} weights.
# - **On the survivors the penalty changes nothing.** Ridge's best penalty is the grid floor ({{p4.final.methods.l2.lambda:.6f}}), and every method is within {{p4.final.comparisons.l1_minus_unregularised.hi:.4f}} of the unpenalised fit ({{p4.final.unregularised.val_r2:.4f}}). Once the copies are gone there is nothing left for a penalty to fix.
# - **Verdicts** (table above). Useful: `hr`, `temp`, `workingday`, `weathersit`, `weekday`, `hum`, `windspeed`, and `dteday` as the representative of time. Redundant: `atemp`, `yr`, `instant`, `season`, `mnth` and `holiday`. No column is uninformative.
# - **Final model:** {{p4.recommended.method}} on {{p4.survivor_counts.stable}} surviving features, λ = {{p4.recommended.lambda:.6f}}: validation R² {{p4.recommended.val_r2:.4f}}, RMSE {{p4.recommended.val_rmse:.1f}} bikes, held-out days {{p4.recommended.day_block_r2:.4f}}, chronological {{p4.recommended.chrono_r2:.3f}}. Our first pass ended at {{first_pass.p4.val_r2:.4f}}, {{first_pass.p4.val_rmse:.1f}} bikes, {{first_pass.p4.day_block_r2:.4f}} and {{first_pass.p4.chrono_r2:.3f}}. Our intervals resample rows, but days are the real sampling unit: resampling whole days gives {{diagnostics.val_r2_interval_row_vs_day_block.day_block_bootstrap_interval.0:.3f}}–{{diagnostics.val_r2_interval_row_vs_day_block.day_block_bootstrap_interval.1:.3f}}, {{diagnostics.val_r2_interval_row_vs_day_block.day_block_to_row_width_ratio:.2f}} times as wide.
#
# **What surprised us.**
# - Lasso's choices between copies were as arbitrary as we feared. It keeps `atemp` in {{p4.column_verdicts.atemp.numbers.lasso.max_freq_own:.2f}} of the resampled fits although dropping `atemp` costs nothing, and never keeps `instant` (frequency {{p4.column_verdicts.instant.numbers.lasso.max_freq_own:.2f}}) although `instant` alone explains {{p4.column_verdicts.instant.numbers.solo_r2:.3f}} of the variance.
# - `dteday` is useful only as a group. Dropping it alone costs {{p4.column_verdicts.dteday.numbers.drop_alone.delta:.4f}} because its copies step in; dropping it with them costs {{p4.column_verdicts.dteday.numbers.drop_group.delta:.3f}}. It is called useful through the representative rule.
# - `mnth` is slightly harmful: month dummies add steps to a season the day-of-year terms already describe.
# - `windspeed` was uninformative in our very first run, as one linear column. With the terms Phase 3 added, dropping it costs {{p4.column_verdicts.windspeed.numbers.drop_alone.delta:.4f}}. A verdict is about a column as we represent it.
# - *Second pass:* every verdict is unchanged. The recommended method changed from Elastic Net to {{p4.recommended.method}}; with tied methods that is the held-out-day tie-break, not a finding.
#
# **Handed on:** the surviving features to Phase 5, and the stage-B model to the submission.
