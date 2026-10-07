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
# TODO(chief)

# %%
# Phase 4 loads artifacts/p3.json for the target design, adds the candidate columns and runs everything with our own
# solvers. This is the slowest phase of the notebook: a full run takes about 15 minutes, so the cell below loads the
# stored artifact when the config and the upstream artifact are unchanged (python run.py p4 recomputes it).
import pandas as pd

from src.common import cached_or_run, load_config, read_artifact
from src.phases import p4 as p4mod
from src.plots_p4 import (plot_cv_vs_validation, plot_paths, plot_stability, plot_validation_curves,
                          plot_verdicts)

CFG = globals().get("CFG") or load_config()
p4 = cached_or_run("p4", p4mod.run, CFG, upstream="p3")

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
fig = plot_validation_curves(p4)

# %%
# Chosen hyper-parameters (best validation R2 on each path; ties go to the larger alpha).
chosen = pd.DataFrame([{"method": m, "lambda": v["lambda"], "l1_ratio": v["l1_ratio"], "val_r2": v["val_r2"],
                        "non_zero": v["n_nonzero"], "of": v["n_features"] - 1} for m, v in p4["methods"].items()])
print(chosen.round(5).to_string(index=False))
enet = pd.DataFrame([{"l1_ratio": float(rho), "best_val_r2": max(r["val_r2"] for r in rows),
                      "best_alpha": max(rows, key=lambda r: (r["val_r2"], r["alpha"]))["alpha"]}
                     for rho, rows in p4["curves"]["enet"].items()])
print(enet.round(5).to_string(index=False))

# %% [markdown]
# ### Justification: fair comparison of the three methods
#
# TODO(chief)

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
# TODO(chief)

# %%
# Two ways to pick the penalty: best validation R2 versus day-block CV (best and one-standard-error rule).
picks = pd.DataFrame([{"method": m, "validation_best": p4["methods"][m]["lambda"], "cv_best": c["cv_best_alpha"],
                       "cv_1se": c["cv_1se_alpha"], "val_r2_at_cv_best": c["val_r2_at_cv_best"],
                       "val_r2_at_cv_1se": c["val_r2_at_cv_1se"], "val_r2_at_validation_best": p4["methods"][m]["val_r2"]}
                      for m, c in p4["cv"].items()])
print(picks.round(5).to_string(index=False))

# %%
fig = plot_cv_vs_validation(p4, "l1")

# %%
fig = plot_paths(p4, "l1")

# %%
fig = plot_paths(p4, "l2")

# %%
# Where does each correlated pair enter the lasso path? (largest alpha with a non-zero weight; None = never)
for name in ("temp", "atemp", "yr", "instant", "trend", "workingday", "holiday"):
    print(f"{name:12s} enters at alpha = {p4['entry_alpha'][name]}")

# %% [markdown]
# ### Justification: the verdict rule
#
# TODO(chief)

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
fig = plot_verdicts(p4)

# %%
fig = plot_stability(p4)

# %% [markdown]
# ### Justification: the survivor rule
#
# TODO(chief)

# %%
# Survivors: lasso non-zero, stable over resampled days, and every source column judged useful.
print("counts:", p4["survivor_counts"])
print("original columns kept:", p4["survivors_original"])
print("survivor columns (expanded features):", len(p4["survivors_expanded"]))

# %% [markdown]
# ### Justification: final model on the surviving columns
#
# TODO(chief)

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
fig = plot_validation_curves(p4, stage="final")

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
# TODO(chief)
