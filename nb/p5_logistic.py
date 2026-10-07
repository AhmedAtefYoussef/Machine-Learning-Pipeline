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
# - **The flaw of one global threshold.** If "high" means "above the 75th percentile of all hours", the label is the timetable: 8h and 17–18h on working days, midday at weekends. In the table below, hour dummies alone then reach a ROC-AUC of {{p5.label_variants.0.auc_hour_only:.3f}} and the full model {{p5.label_variants.0.auc_full:.3f}}. That looks excellent and tells the operator nothing new.
# - **Fixing the clock creates a calendar problem.** With one threshold per (day type, hour), hour alone drops to {{p5.label_variants.1.auc_hour_only:.3f}}, but because demand grew so much, only {{p5.label_variants.1.pos_val_2011:.3f}} of 2011 validation hours are "high" against {{p5.label_variants.1.pos_val_2012:.3f}} of 2012 hours, and the trend alone reaches {{p5.label_variants.1.auc_trend_only:.3f}}. Now the label is the year.
# - **Our rule.** An hour is high-demand when `cnt` is above the {{p5.threshold_rule.quantile}} quantile of training hours with the same year, day type and hour: {{p5.threshold_rule.n_cells}} thresholds, each from at least {{p5.threshold_rule.min_cell_n}} training hours, looked up for validation rows. Hour alone is at chance ({{p5.label_variants.2.auc_hour_only:.3f}}), trend alone nearly so ({{p5.label_variants.2.auc_trend_only:.3f}}), and the positive share is the same in both years ({{p5.label_variants.2.pos_val_2011:.3f}} and {{p5.label_variants.2.pos_val_2012:.3f}}).
# - **Why this is meaningful for the operator.** The operator already plans for rush hours and for growth. What they cannot read off a timetable is whether *this* 8 am will be busier than a normal 8 am this year. The top quarter of comparable hours is where a normal allocation of bikes runs short.
# - **Why the 75th percentile.** It keeps enough positives to learn from (about a quarter) while still meaning "clearly above normal"; a 90th percentile would leave about five positives in the smallest cells.
# - **Limit.** A new year has no cell of its own; in use, the thresholds would have to be rolled forward, for example last year's cell scaled by the fitted trend.

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
# The classifier is our own logistic regression: gradient descent on the mean log-loss, with the step 1/(λ_max/4 + l2), which is the safe step for this loss because its curvature is at most a quarter of that of the squared loss. It uses only the Phase 4 survivors (the assert above). We add a small ridge term because, without any penalty, some weights of rarely active columns keep growing and gradient descent does not settle (the l2 = 0 row below stops at the iteration cap). The strength is chosen by validation ROC-AUC among {{p5.l2_sweep.0.l2}}, {{p5.l2_sweep.1.l2}}, {{p5.l2_sweep.2.l2}} and {{p5.l2_sweep.3.l2}}; ties within 0.0001 go to the larger value. Chosen: {{p5.l2}}.

# %%
# Ridge strength by validation ROC-AUC (ties within 1e-4 go to the larger l2); the cost of each fit is its iterations.
sweep = pd.DataFrame(p5["l2_sweep"])
print(sweep.round(5).to_string(index=False))
print("\nchosen l2:", p5["l2"], "| iterations:", p5["iterations"], "| stop:", p5["stop_reason"],
      "| learning rate:", round(p5["lr"], 4))
print("gradient check (max relative error): at zeros", f"{p5['gradient_check_at_zeros']:.1e}",
      "| halfway to the final weights", f"{p5['gradient_check_at_half']:.1e}")

# %% [markdown]
# ### Justification: metrics
#
# About one hour in four is positive, so a model that always answers "normal" has an accuracy of {{p5.class_balance.majority_accuracy:.3f}}. Accuracy alone would flatter us, so we read it together with: **recall** (how many high-demand hours we catch, the operator's main concern), **precision** (how many alarms are real), **F1** (their balance), **ROC-AUC** (ranking quality, independent of the cut-off and of the class balance) and **PR-AUC** (ranking quality on the positive class, more demanding when positives are the minority). The task's three required numbers, accuracy, F1 and ROC-AUC, are reported at our operating cut-off; the others are shown beside them.

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
# A missed high-demand hour means empty docks and lost customers; a false alarm means a few idle bikes. We use the same 3:1 cost ratio as in the Phase 1 bonus. For a model whose probabilities are calibrated, raising an alarm is worth it when p × 3 > (1 − p) × 1, that is when p > 1/(1+3) = {{p5.t_cost}}. So our cut-off is {{p5.t_cost}}, not the habitual 0.5. The plot below checks the argument on the validation rows: the cost is lowest at {{p5.t_cost_empirical}} on our grid, and the calibration plot further down shows how far the probabilities can be trusted (largest gap {{p5.calibration_max_gap:.3f}}).

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
# Each phase used what the previous one produced, and the table below is built from the five artifact files.
#
# - **Phase 1** gave a working optimiser and an honest baseline: validation R² {{p1.val_r2:.3f}} with {{p1.n_features}} weights. Its residuals showed the missing structure.
# - **Phase 2** started from Phase 1's weights (first loss equal to Phase 1's last) and added what those residuals asked for: R² {{p2.val_r2:.3f}}.
# - **Phase 3** showed that this model was still too simple, not too flexible, and that the gap to a chronological split is drift rather than leakage. It moved the target to {{p3.target_complexity.n_features}} weights: {{p3.estimates_target.seeded.r2:.3f}} on validation, {{p3.estimates_target.day_holdout.r2:.3f}} on held-out days, {{p3.estimates_target.chrono.r2:.3f}} chronologically.
# - **Phase 4** found, as that diagnosis implies, that penalties change little (all three methods within noise of each other), used them to sort the 14 columns into useful, redundant and uninformative, and recommended {{p4.recommended.method}} on the {{p4.survivor_counts.after_verdicts}} surviving features: R² {{p4.recommended.val_r2:.3f}}, RMSE {{p4.recommended.val_rmse:.1f}}.
# - **Phase 5** reused exactly those features for a classifier: ROC-AUC {{p5.metrics.roc_auc:.3f}}, F1 {{p5.metrics.f1:.3f}}, accuracy {{p5.metrics.accuracy:.3f}} at the cost-based cut-off.
#
# The thread through all five: on this data the errors come from bias. Every gain came from giving the linear model structure the data really has; the classical variance cures (higher degree, stronger penalty) did nothing measurable. **The regression model our pipeline recommends, and the one behind our submission, is the Phase 4 stage-B model.**

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
# **What happened.**
# - Class balance is as planned: {{p5.class_balance.train:.3f}} positives in training, {{p5.class_balance.val:.3f}} in validation, {{p5.class_balance.val_by_year.0:.3f}} and {{p5.class_balance.val_by_year.1:.3f}} by year.
# - The foils behaved as predicted: global threshold, hour-only AUC {{p5.label_variants.0.auc_hour_only:.3f}} and full model {{p5.label_variants.0.auc_full:.3f}}; our rule, hour-only {{p5.label_variants.2.auc_hour_only:.3f}}.
# - Our classifier: ROC-AUC {{p5.metrics.roc_auc:.3f}} (95% interval {{p5.val_auc_bootstrap.lo:.3f}}–{{p5.val_auc_bootstrap.hi:.3f}}), PR-AUC {{p5.metrics.pr_auc:.3f}}. At the cut-off {{p5.metrics.threshold}}: accuracy {{p5.metrics.accuracy:.3f}}, F1 {{p5.metrics.f1:.3f}}, precision {{p5.metrics.precision:.3f}}, recall {{p5.metrics.recall:.3f}}. At 0.5: accuracy {{p5.metrics_at_0_5.accuracy:.3f}}, F1 {{p5.metrics_at_0_5.f1:.3f}}, recall {{p5.metrics_at_0_5.recall:.3f}}.
# - The cost argument holds up: at {{p5.metrics.threshold}} we miss {{p5.metrics.confusion.fn}} high-demand hours and raise {{p5.metrics.confusion.fp}} false alarms (cost {{p5.metrics.cost:.3f}} per hour); at 0.5 we would miss {{p5.metrics_at_0_5.confusion.fn}} (cost {{p5.metrics_at_0_5.cost:.3f}}).
#
# **What surprised us.**
# - Accuracy is *lower* at our chosen cut-off ({{p5.metrics.accuracy:.3f}}) than at 0.5 ({{p5.metrics_at_0_5.accuracy:.3f}}), and barely above the do-nothing baseline of {{p5.class_balance.majority_accuracy:.3f}}. Judged by accuracy alone our operating point looks like a mistake; judged by the operator's cost it is clearly the better one. This is the clearest case in the project of a metric pointing the wrong way.
# - The lowest validation cost was at {{p5.t_cost_empirical}} rather than exactly {{p5.t_cost}}, and the largest calibration gap is {{p5.calibration_max_gap:.3f}}: the probabilities are usable but not perfectly calibrated. We kept the theoretical cut-off instead of tuning it on the validation rows.
# - The honest task is much harder than the trivial one (AUC {{p5.label_variants.2.auc_full:.3f}} against {{p5.label_variants.0.auc_full:.3f}}). Whether an hour is unusually busy for its slot depends on things we only partly observe: the weather explains some of it, events and school holidays are not in the data.
# - Even with the label made relative to the hour, hour-related columns still carry a large share of the weights (table above): the weather effect differs by hour, so the survivors' hour × weather interactions matter even though the hour alone predicts nothing.
