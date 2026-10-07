# %% [markdown]
# ## Expectation — Phase 5: Logistic Regression
#
# *Written before running the phase (see the git history of this file).*
#
# **What we do.** We turn `cnt` into a yes/no target, "is this a high-demand hour?", and train our own logistic regression (gradient descent on the log-loss) using only the feature columns that survived Phase 4, loaded from `artifacts/p4.json`.
#
# **The flaw of a single global threshold, as we see it.** If "high" means "more than the 75th percentile of all hours", then high demand is simply 8 am and 5–6 pm on working days and midday at weekends. A classifier for that label only has to read the clock; it tells the operator nothing they do not already know from the timetable. A threshold per (working day, hour) fixes the clock problem but creates a calendar one: demand grew about 65% from 2011 to 2012, so nearly every "high" hour would be a 2012 hour.
#
# **Our label.** An hour is high-demand when its `cnt` is above the 75th percentile of training hours with the same year, day type and hour of day. The 96 thresholds are computed on the training portion only and looked up for validation rows. "High" then means "busier than three quarters of comparable hours", which is the case where the usual allocation of bikes is not enough.
#
# **What we expect, and why.**
#
# 1. *Class balance.* About 25% positives in training and validation, and about the same in each year. The majority-class accuracy is therefore about 0.75, so accuracy alone will flatter any model; we will read F1, precision, recall, ROC-AUC and PR-AUC together.
# 2. *The foils.* Under the global threshold, hour dummies alone should reach an AUC near 0.85 and the full model near 0.98: impressive and useless. Under the (working day, hour) rule without year, the trend alone should reach about 0.8. Under our rule, hour alone should sit at chance (0.50) and trend alone close to it.
# 3. *Our classifier.* With the easy cues removed, the model has to rely on weather and season. We expect ROC-AUC between 0.82 and 0.88, well below the foil's number, and we will treat that lower figure as the honest one. Accuracy at a 0.5 cut-off should be only a few points above 0.75.
# 4. *Operating threshold.* Missing a high-demand hour (empty docks) is worse than a false alarm (a few idle bikes). With the same 3:1 cost ratio as the Phase 1 bonus, the cost-minimising cut-off for a calibrated model is 1/(1+3) = 0.25, not 0.5. We expect recall to rise sharply and precision to fall at 0.25, and the validation cost curve to have its minimum near 0.25 if our probabilities are well calibrated.
# 5. *Features.* Hour dummies should matter little on their own here, because the label is already relative to the hour; temperature, humidity, weather situation and the day-of-year terms should carry the model.

# %% [markdown]
# ### Justification: label rule
#
# TODO(chief)

# %%
# Phase 5 loads artifacts/p4.json for the surviving columns, builds the label with thresholds fitted on the training
# rows, and fits our own logistic regression. A full run takes about three minutes (the l2 sweep and the label-rule
# foils), so the cell loads the stored artifact when config and upstream are unchanged.
import pandas as pd

from src.common import cached_or_run, load_config, read_artifact
from src.phases import p5 as p5mod
from src.plots_p5 import plot_calibration, plot_label_variants, plot_roc_pr, plot_threshold_curve

CFG = globals().get("CFG") or load_config()
p5 = cached_or_run("p5", p5mod.run, CFG, upstream="p4")  # writes artifacts/p5.json when it has to run

# %%
# The chain: every feature of this phase must be a survivor of Phase 4.
p4 = read_artifact("p4")
survivors = set(p4["survivors_expanded"])
assert set(p5["features"]) <= survivors, "Phase 5 uses a column that Phase 4 dropped"
print("Phase 4 surviving columns :", len(survivors), "from original columns", p4["survivors_original"])
print("Phase 5 features (no bias):", len(p5["features"]), "| all in the Phase 4 survivors: True")
print("columns incl. bias        :", p5["n_features"], "| training rows:", p5["n_train"], "| validation rows:", p5["n_val"])

# %%
# The label: high demand = cnt above the quantile of training hours with the same year, day type and hour.
rule = p5["threshold_rule"]
print("quantile:", rule["quantile"], "| cells:", rule["group_by"], "| fitted on:", rule["fitted_on"])
print("number of cells:", rule["n_cells"], "| fewest training hours in a cell:", rule["min_cell_n"])
thresholds = pd.DataFrame(rule["table"])
peaks = thresholds[thresholds["hr"].isin([3, 8, 12, 17])].pivot_table(index=["yr", "workingday"], columns="hr",
                                                                      values="threshold")
print("\nthreshold (bikes) for four hours of the day:")
print(peaks.round(1).to_string())
balance = p5["class_balance"]
print(pd.Series({"positives, train": balance["train"], "positives, validation": balance["val"],
                 "positives, validation 2011": balance["val_by_year"]["0"],
                 "positives, validation 2012": balance["val_by_year"]["1"],
                 "majority-class accuracy": balance["majority_accuracy"]}).round(4).to_string())

# %%
# The foils: the same three models under three label rules. Only the last rule keeps the hour and the trend at chance.
foils = pd.DataFrame(p5["label_variants"])
foils["rule"] = foils["group_by"].map(lambda g: " + ".join(g) or "global")
print(foils[["rule", "pos_train", "pos_val", "pos_val_2011", "pos_val_2012", "auc_hour_only", "auc_trend_only",
             "auc_full", "acc_full_at_0_5", "majority_accuracy"]].round(3).to_string(index=False))
print("foil fits that did not converge:", p5["foil_not_converged"])

# %%
plot_label_variants(p5)

# %% [markdown]
# ### Justification: regularisation of the classifier
#
# TODO(chief)

# %%
# Ridge strength by validation ROC-AUC (ties within 1e-4 go to the larger l2); the cost of each fit is its iterations.
sweep = pd.DataFrame(p5["l2_sweep"])
print(sweep.round(5).to_string(index=False))
print("\nchosen l2:", p5["l2"], "| iterations:", p5["iterations"], "| stop:", p5["stop_reason"],
      "| learning rate:", round(p5["lr"], 4))
print("gradient check (max relative error): at zeros", f"{p5['gradient_check_at_zeros']:.1e}",
      "| halfway", f"{p5['gradient_check_at_half']:.1e}", "| at the final weights", f"{p5['gradient_check_at_final']:.1e}",
      "(the gradient is ~1e-6 there, so the differences are noise-limited)")

# %% [markdown]
# ### Justification: metrics
#
# TODO(chief)

# %%
# Metrics on the validation rows at the cost threshold 1/(1+c), at 0.5 and at the F1-optimal grid threshold.
columns = ["threshold", "accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc", "cost"]
table = pd.DataFrame({"at 1/(1+c)": p5["metrics"], "at 0.5": p5["metrics_at_0_5"],
                      "at F1-optimal": p5["metrics_at_f1_opt"]}).T[columns]
print(table.round(4).to_string())
print("\nmajority-class accuracy:", round(p5["class_balance"]["majority_accuracy"], 4))
print("validation ROC-AUC 95% interval:", round(p5["val_auc_bootstrap"]["lo"], 4), "to",
      round(p5["val_auc_bootstrap"]["hi"], 4))
print("training rows at 1/(1+c):", {k: round(v, 4) for k, v in p5["train_metrics"].items()})
conf = p5["metrics"]["confusion"]
print("\nconfusion matrix at 1/(1+c) (rows = truth):")
print(pd.DataFrame([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]], index=["truly normal", "truly high"],
                   columns=["predicted normal", "predicted high"]).to_string())

# %% [markdown]
# ### Justification: operating threshold
#
# TODO(chief)

# %%
print("cost ratio (miss : false alarm):", p5["cost_ratio"], "| 1/(1+c) =", p5["t_cost"])
print("grid threshold with the lowest validation cost:", p5["t_cost_empirical"],
      "| with the highest F1:", p5["t_f1"])
plot_threshold_curve(p5)

# %%
plot_roc_pr(p5)

# %%
print("largest gap between predicted and observed share (bins with >= 30 rows):", round(p5["calibration_max_gap"], 4))
plot_calibration(p5)

# %%
# What the classifier relies on: the largest standardised weights, and the share of sum|w| by original column.
print(pd.DataFrame(p5["top_coefficients"]).round(3).to_string(index=False))
print("\nshare of sum|w| by original column:")
print(pd.Series(p5["coef_abs_share_by_column"]).round(3).to_string())

# %% [markdown]
# ### Pipeline retrospective
#
# TODO(chief)

# %%
retro = pd.DataFrame(p5["retrospective"])
print(retro[["phase", "n_features", "train_score", "val_score", "val_rmse"]].round(4).to_string(index=False))
print()
for _, row in retro.iterrows():
    print(f"{row['phase']}: consumed {row['consumed']}; settings {row['hyperparameters']}" +
          (f"; {row['extra']}" if row["extra"] else ""))

# %% [markdown]
# ## Outcome — Phase 5
#
# TODO(chief)
