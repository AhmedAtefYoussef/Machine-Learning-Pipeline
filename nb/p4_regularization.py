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
# ### Justification: search ranges and strategy
#
# - **How Phase 3 shapes the search.** The diagnosis was "under-fit, no measurable variance" for the model we carry forward. So we do not expect a large penalty to help, and the grid has to reach down to practically zero to show that; we also add the unpenalised fit as a reference line. At the other end the grid must go far enough to show the model being destroyed, so that the optimum is visibly inside the range and not at an edge.
# - **Design.** The Phase 3 target design plus the columns we had held back (`atemp`, `yr`, `instant`, `season`, `mnth`; `weekday` is already in the target), so that every original column is in front of the three methods. All columns are standardised on the training rows; the intercept is never penalised.
# - **Objective.** (1/2n)‖y − b − Xw‖² + α·ρ‖w‖₁ + (α/2)(1 − ρ)‖w‖², with ρ = 0 for Ridge, ρ = 1 for Lasso and ρ in {0.1, 0.3, 0.5, 0.7, 0.9, 0.95} for Elastic Net. Ridge is solved in closed form; Lasso and Elastic Net by our own cyclic coordinate descent with soft-thresholding, warm-started along the path. Our tests check all three against a reference library to better than 1e-3.
# - **Grids.** 40 log-spaced values per path. Ridge: {{p4.grids.l2.min:.0e}} to {{p4.grids.l2.max:.0f}}. Lasso and Elastic Net: from the smallest penalty that sets every weight to zero (computed from the data: {{p4.grids.l1.max:.3f}} for Lasso) down to one ten-thousandth of it. Log spacing because the effect of a penalty is multiplicative.
# - **Honesty note.** {{p4.not_converged.count}} Lasso fits, all at one large grid value far from the chosen one, used the full {{p4.solver.max_sweeps}} sweeps without meeting our tolerance; the exactly duplicated columns in this design make coordinate descent crawl there. None of them is a chosen model.

# %%
# Phase 4 loads artifacts/p3.json for the target design, adds the candidate columns and runs everything with our own
# solvers. This is the slowest phase of the notebook: a full run takes about 15 minutes, so the cell below loads the
# stored artifact when the config and the upstream artifact are unchanged (python run.py p4 recomputes it).
import pandas as pd

from src.common import cached_or_run, load_config, read_artifact
from src.phases import p4 as p4mod
from src.plots_p4 import (plot_cv_vs_validation, plot_paths, plot_stability, plot_validation_curves,
                          plot_verdicts)

p4 = phase("p4", p4mod.run, upstream="p3")

# %%
# The chain: target level from Phase 3, candidate columns added, resulting width (n_features counts the bias column).
p3 = read_artifact("p3")
origin = p4["design_from"]
print("Phase 3 target level:", p3["target_level"], "| weights (with bias):", p3["target_complexity"]["n_features"])
print("design used here    :", origin["level"], "| weights before candidates:", origin["n_features_before_candidates"])
print("candidates added    :", origin["candidates_added"], "| skipped (already present):", origin["candidates_skipped"])
print("weights after candidates (with bias):", p4["n_features"], "| back-transform method:", p4["back_method"])
print("solver tolerance:", p4["solver"]["tol"], "| max sweeps:", p4["solver"]["max_sweeps"],
      "| solutions that used all sweeps:", p4["not_converged"]["count"], p4["not_converged"]["where"])

# %%
# Baseline: the same design with a negligible penalty (closed-form least squares).
print({k: round(v, 5) for k, v in p4["unregularised"].items()})

# %%
from src.plots import show  # displays a figure as a PNG in the notebook
show(plot_validation_curves(p4))

# %%
# Chosen hyper-parameters (best validation R2 on each path; ties go to the larger alpha).
chosen = pd.DataFrame([{"method": m, "lambda": v["lambda"], "l1_ratio": v["l1_ratio"], "val_r2": v["val_r2"],
                        "non_zero": v["n_nonzero"], "of": v["n_features"] - 1} for m, v in p4["methods"].items()])
print(chosen.round(5).to_string(index=False))
enet = pd.DataFrame([{"l1_ratio": float(rho), "best_val_r2": max(r["val_r2"] for r in rows),
                      "best_alpha": max(rows, key=lambda r: (r["val_r2"], r["alpha"]))["alpha"]}
                     for rho, rows in p4["curves"]["enet"].items()])
print(enet.round(5).to_string(index=False))

# %%
# Our solvers (src/regularization.py): closed-form Ridge, and one sweep of coordinate descent with soft-thresholding,
# which is the whole of Lasso (l1_ratio = 1) and Elastic Net.
from src.regularization import cd_sweep, enet_cd_gram, ridge_closed_form, soft_threshold

show_source(soft_threshold, ridge_closed_form, cd_sweep, enet_cd_gram)

# %%
# The chain, asserted: this phase was built on the Phase 3 artifact that is on disk now, and on its target level.
from src.common import sha256_file

assert p4["upstream_sha256"] == sha256_file("artifacts/p3.json"), "Phase 4 was not built from this Phase 3 artifact"
assert p4["design_from"]["level"] == p3["target_complexity"]["level"], "Phase 4 did not use the Phase 3 target level"
assert p4["design_from"]["n_features_before_candidates"] == p3["target_complexity"]["n_features"]
print("Phase 4 consumed Phase 3: diagnosis", repr(p3["diagnosis"]["label"]), "-> target", p3["target_complexity"]["level"],
      "(hash and level asserted)")

# %% [markdown]
# ### Justification: fair comparison of the three methods
#
# The three methods get the same design matrix, the same training rows, the same validation rows, the same day folds, the same number of grid points and the same selection rule. Elastic Net has a second knob and therefore six times as many candidates, which gives it a small advantage on the validation set; that is one reason we do not compare on validation alone. For each method at its chosen setting we report validation R² with a bootstrap interval, the held-out-day R² and the chronological R², and for each pair of methods a paired bootstrap interval of the difference on the same validation rows. A difference whose interval contains zero is a tie.

# %%
# Fair comparison: same design, same rows, same grid density; three validators and the paired intervals.
table = pd.DataFrame([{"method": m, "val_r2": v["val_r2"], "val_lo": v["val_lo"], "val_hi": v["val_hi"],
                       "day_block_r2": v["day_block_r2"], "day_block_se": v["day_block_se"],
                       "chrono_r2": v["chrono_r2"], "non_zero": v["n_nonzero"]} for m, v in p4["methods"].items()])
print(table.round(4).to_string(index=False))
diffs = pd.DataFrame([{"comparison": k, **v} for k, v in p4["comparisons"].items()])
print(diffs.round(5).to_string(index=False))
best = p4["full_design_best"]
print("full design (with candidate columns): best on validation:", best["method"], round(best["val_r2"], 5))

# %% [markdown]
# ### Justification: choice of lambda per method
#
# For each method we take the penalty with the highest validation R² on bikes (ties go to the larger penalty), because the task says tuning uses the validation set and R² on bikes is the metric we are scored on. Two checks sit beside it in the table below: the penalty that is best on held-out days inside the training portion, and the "one standard error" penalty, the largest one whose held-out-day score is within one standard error of that best. If the validation choice were an accident of this particular split, these would disagree strongly with it. Because we select on the validation rows, the validation scores of this phase are slightly optimistic; the held-out-day and chronological columns are not affected by that.

# %%
# Two ways to pick the penalty: best validation R2 versus day-block CV (best and one-standard-error rule).
picks = pd.DataFrame([{"method": m, "validation_best": p4["methods"][m]["lambda"], "cv_best": c["cv_best_alpha"],
                       "cv_1se": c["cv_1se_alpha"], "val_r2_at_cv_best": c["val_r2_at_cv_best"],
                       "val_r2_at_cv_1se": c["val_r2_at_cv_1se"], "val_r2_at_validation_best": p4["methods"][m]["val_r2"]}
                      for m, c in p4["cv"].items()])
print(picks.round(5).to_string(index=False))

# %%
show(plot_cv_vs_validation(p4, "l1"))

# %%
show(plot_paths(p4, "l1"))

# %%
show(plot_paths(p4, "l2"))

# %%
# Where does each correlated pair enter the lasso path? (largest alpha with a non-zero weight; None = never)
for name in ("temp", "atemp", "yr", "instant", "trend", "workingday", "holiday"):
    print(f"{name:12s} enters at alpha = {p4['entry_alpha'][name]}")

# %% [markdown]
# ### Justification: the verdict rule
#
# We fixed the rule before running the phase. The reference model is Ridge at its chosen penalty on the full design; "dropping" a column means removing every feature built from it (for `hr` that includes all its interactions) and refitting.
#
# - **Useful:** dropping the column alone lowers validation R² by at least {{p4.verdict_thresholds.min_delta}}, with a paired bootstrap interval that stays above zero.
# - **Redundant:** dropping it alone costs nothing, but either dropping it together with its partner columns hurts, or the column on its own explains at least {{p4.verdict_thresholds.solo_min}} of the variance. Its information is real but another column already carries it. Partners are the groups we measured to be copies of each other: {`temp`, `atemp`}, {`season`, `mnth`, `dteday`}, {`yr`, `instant`, `dteday`}, {`weekday`, `workingday`, `holiday`}.
# - **Uninformative:** neither.
# - **One representative per group.** If a group matters as a whole but no single member is missed when dropped, we keep the member that explains most on its own and call it useful; the others are redundant "carried by" it.
#
# Lasso evidence is reported next to the drop tests (how many of the column's own features Lasso keeps, how often over 50 resamples of training days, and where it enters the path), but a Lasso zero alone does not decide a verdict: with copies in the design, which copy Lasso keeps is partly arbitrary.

# %%
# Verdict for every original column (cost = validation R2 lost when removed; interval = paired bootstrap).
rows = []
for col, v in p4["column_verdicts"].items():
    n = v["numbers"]
    rows.append({"column": col, "verdict": v["verdict"],
                 "alone": f"{n['drop_alone']['delta']:+.4f} [{n['drop_alone']['lo']:+.4f}, {n['drop_alone']['hi']:+.4f}]",
                 "with_partners": None if n["drop_group"] is None else round(n["drop_group"]["delta"], 4),
                 "solo_r2": None if n["solo_r2"] is None else round(n["solo_r2"], 4),
                 "lasso_kept": f"{n['lasso']['n_own_nonzero']}/{n['lasso']['n_own']}",
                 "max_freq": None if n["lasso"]["max_freq_own"] is None else round(n["lasso"]["max_freq_own"], 2),
                 "carried_by": ", ".join(v["carried_by"]), "representative": v["representative"]})
print(pd.DataFrame(rows).to_string(index=False))

# %%
show(plot_verdicts(p4))

# %%
show(plot_stability(p4))

# %% [markdown]
# ### Justification: the survivor rule
#
# The feature subset handed to Phase 5 is what L1 leaves non-zero, made reproducible. The order matters. We first remove every feature built from a column that is not "useful"; there are {{p4.survivor_counts.useful_design}} features left. On that reduced design we run Lasso again (penalty chosen by validation R²: {{p4.selection.lambda:.1e}}), and a feature survives if its weight is non-zero ({{p4.survivor_counts.lasso_nonzero}} features) and it is selected in at least {{p4.stability_threshold}} of 50 Lasso fits on resampled training days ({{p4.survivor_counts.stable}} remain). To be plain about what this means: on this reduced design Lasso removes almost nothing, so the surviving subset is decided almost entirely by the column verdicts, and those verdicts were decided by drop tests scored on the validation set. The subset is a tuning decision made on validation rows, like the degree and the penalties; no feature statistic was fitted on them.
#
# Our first version did it the other way round: take the Lasso zeros from the full design, then remove the copies. That turned out to be wrong for a reason worth stating. With month dummies, `yr` and `instant` in the design, Lasso let them stand in for some of the trend and day-of-year terms and zeroed those; the verdict step then removed the stand-ins as redundant, and the survivors had lost part of the time description twice. Validation R² hardly noticed, but the chronological score fell by about 0.013. Lasso should choose among features only after its choices are no longer between copies.

# %%
# Survivors: Lasso on the design of useful columns only; non-zero at its chosen lambda and stable over resampled days.
sel = p4["selection"]
print("counts (useful design -> lasso non-zero -> stable):", p4["survivor_counts"])
print("selection lasso: lambda =", round(sel["lambda"], 6), "| val R2 =", round(sel["val_r2"], 4),
      "| non-zero:", sel["n_nonzero"], "of", sel["n_columns"], "| sweeps exhausted:", sel["not_converged"]["count"])
print("original columns kept:", p4["survivors_original"])
print("survivor columns (expanded features):", len(p4["survivors_expanded"]))

# %% [markdown]
# ### Justification: final model on the surviving columns
#
# Stage A above answers "which columns matter" and needs every column in the design. It is not the model we want to ship: the columns judged redundant add nothing (dropping `mnth` alone changes validation R² by {{p4.column_verdicts.mnth.numbers.drop_alone.delta:.4f}}, interval {{p4.column_verdicts.mnth.numbers.drop_alone.lo:.4f}} to {{p4.column_verdicts.mnth.numbers.drop_alone.hi:.4f}}; a negative number means the model is better without it), and several of them are extra descriptions of time, which is exactly where our chronological split shows the model to be fragile. So we repeat the same search for the three methods on the surviving columns only (stage B) and recommend among those. The rule for choosing among the three methods was fixed before the first Phase 4 run: among the methods whose validation difference to the best is within the paired bootstrap noise, take the one with the best held-out-day R². What we added after seeing the first run is this second stage itself, that is, applying the rule to models fitted on the survivors; the earlier runs are in the git history.

# %%
# Stage B: the same search (ridge, lasso, elastic net) on the surviving columns only.
final = p4["final"]
print("columns:", final["n_columns"], "| solutions that used all sweeps:", p4["not_converged_final"])
table = pd.DataFrame([{"method": m, "lambda": v["lambda"], "l1_ratio": v["l1_ratio"], "val_r2": v["val_r2"],
                       "day_block_r2": v["day_block_r2"], "day_block_se": v["day_block_se"],
                       "chrono_r2": v["chrono_r2"], "non_zero": v["n_nonzero"]} for m, v in final["methods"].items()])
print(table.round(5).to_string(index=False))
print("unregularised on the survivors:", {k: round(v, 5) for k, v in final["unregularised"].items()})
diffs = pd.DataFrame([{"comparison": k, **v} for k, v in final["comparisons"].items()])
print(diffs.round(5).to_string(index=False))
rec = p4["recommended"]
print("within noise of the best:", rec["candidates"], "| recommended:", rec["method"], "| fitted on:", rec["fitted_on"],
      "| lambda:", rec["lambda"], "| l1_ratio:", rec["l1_ratio"])
print("recommended: val R2", round(rec["val_r2"], 5), "| day-block", round(rec["day_block_r2"], 5),
      "| chrono", round(rec["chrono_r2"], 5))

# %%
show(plot_validation_curves(p4, stage="final"))

# %%
# Does a penalty rescue the over-fit top of the ladder? (informational)
rich = p4["rich_check"]
print(f"level {rich['level']} | weights (with bias): {rich['n_features']} | unregularised val R2: "
      f"{rich['unregularised_val_r2']:.4f}")
for m in ("l2", "l1"):
    print(m, {k: round(v, 5) if isinstance(v, float) else v for k, v in rich[m].items()})
print("best penalised top level minus the recommended (surviving-column) model:",
      {k: round(v, 5) for k, v in rich["paired_best_vs_recommended"].items()})

# %% [markdown]
# ## Outcome — Phase 4
#
# **What happened.**
# - **The three methods are tied, and none clearly beats the unpenalised fit.** On the full design: Ridge {{p4.methods.l2.val_r2:.4f}}, Lasso {{p4.methods.l1.val_r2:.4f}}, Elastic Net {{p4.methods.enet.val_r2:.4f}} validation R², against {{p4.unregularised.val_r2:.4f}} without a penalty; every paired interval between methods contains zero, and the gain over no penalty is about {{p4.comparisons.l2_minus_unregularised.delta:.4f}}. The chosen penalties are tiny (Ridge {{p4.methods.l2.lambda:.1e}}, Lasso {{p4.methods.l1.lambda:.1e}}, Elastic Net {{p4.methods.enet.lambda:.1e}} with l1_ratio {{p4.methods.enet.l1_ratio}}). As expected after an "under-fit" diagnosis, regularization here is insurance.
# - **Where there is variance, the penalty works.** On the over-fit top ladder level Lasso raises validation R² from {{p4.rich_check.unregularised_val_r2:.4f}} to {{p4.rich_check.l1.val_r2:.4f}} while keeping {{p4.rich_check.l1.n_nonzero}} of {{p4.rich_check.n_features}} weights. That is still not better than our recommended model (difference {{p4.rich_check.paired_best_vs_recommended.delta:.4f}}, interval {{p4.rich_check.paired_best_vs_recommended.lo:.4f}} to {{p4.rich_check.paired_best_vs_recommended.hi:.4f}}).
# - **Lasso is not sparse:** it keeps {{p4.methods.l1.n_nonzero}} weights. The signal is spread over many hour-specific columns.
# - **Verdicts** (table above). Useful: `hr` (dropping it costs {{p4.column_verdicts.hr.numbers.drop_alone.delta:.3f}}), `temp` ({{p4.column_verdicts.temp.numbers.drop_alone.delta:.4f}}), `workingday` ({{p4.column_verdicts.workingday.numbers.drop_alone.delta:.4f}}), `weathersit` ({{p4.column_verdicts.weathersit.numbers.drop_alone.delta:.4f}}), `weekday` ({{p4.column_verdicts.weekday.numbers.drop_alone.delta:.4f}}), `hum` ({{p4.column_verdicts.hum.numbers.drop_alone.delta:.4f}}), `windspeed` ({{p4.column_verdicts.windspeed.numbers.drop_alone.delta:.4f}}), and `dteday` as the kept representative of time. Redundant: `atemp` (dropping it costs {{p4.column_verdicts.atemp.numbers.drop_alone.delta:.4f}}, dropping it together with `temp` costs {{p4.column_verdicts.atemp.numbers.drop_group.delta:.4f}}), `yr` and `instant` (together with the date terms {{p4.column_verdicts.yr.numbers.drop_group.delta:.4f}}, alone nothing), `season` and `mnth`, and `holiday` (fully determined by `weekday` and `workingday`). No column came out uninformative.
# - **Survivors:** {{p4.survivor_counts.stable}} features from the eight useful original columns.
# - **Final model (stage B):** all three methods again within noise; recommended {{p4.recommended.method}} with λ = {{p4.recommended.lambda:.1e}}: validation R² {{p4.recommended.val_r2:.4f}}, RMSE {{p4.recommended.val_rmse:.1f}} bikes, held-out days {{p4.recommended.day_block_r2:.4f}}, chronological {{p4.recommended.chrono_r2:.3f}}.
#
# **What surprised us.**
# - **Lasso's choices between copies were as arbitrary as we feared, and in one case backwards.** It keeps `atemp` in {{p4.column_verdicts.atemp.numbers.lasso.max_freq_own:.2f}} of the resampled fits although dropping `atemp` costs nothing, and it never keeps `instant` (frequency {{p4.column_verdicts.instant.numbers.lasso.max_freq_own:.2f}}) although `instant` alone explains {{p4.column_verdicts.instant.numbers.solo_r2:.3f}} of the variance. "Lasso set it to zero" was evidence for us, but it would have misled us without the drop tests.
# - **`dteday` is useful only as a group.** Dropping the trend and day-of-year terms alone costs {{p4.column_verdicts.dteday.numbers.drop_alone.delta:.4f}}, because `yr`, `instant`, `season` and `mnth` then step in; dropping them all costs {{p4.column_verdicts.dteday.numbers.drop_group.delta:.3f}}. We had predicted the copies would be redundant; we had not expected them to be able to replace the original so completely.
# - **`mnth` is slightly harmful.** Removing it improves validation R² (see above). Month dummies add steps to a season that the smooth day-of-year terms already describe.
# - **`windspeed` is useful after all**, but only because Phase 3's weather-detail level gave it a square and an interaction with temperature. In our first Phase 4 run, with wind as a single linear column, the same rule called it uninformative. A verdict about a column is a verdict about the column *as we represented it*.
# - **Our own survivor rule was wrong at first** (see the justification above): taking Lasso zeros before removing the copies quietly cost robustness on later months.
# - Some Lasso fits at one large penalty value on the full design ran out of sweeps ({{p4.not_converged.count}} fits; none is a chosen model). The exactly duplicated columns are the cause; on the survivors every fit converged.
#
# **Handed to Phase 5:** the surviving feature list. Handed to the submission: the stage-B recommended model.
