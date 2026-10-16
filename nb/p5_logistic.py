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
# ## Expectation — Phase 5, second pass
#
# *Written before re-running the chain with the power target (λ = 0.1). The cell above is unchanged.*
#
# The label and the classifier do not use the regression target, so Phase 5 changes only if the surviving feature list
# changes. We expect the same class balance and a ROC-AUC within 0.005 of the first pass (0.886). A tree model that we
# tried outside the pipeline reaches about 0.90 on the same label, so we do not expect the logistic model to go higher
# than that.

# %% [markdown]
# ### Justification: label rule
#
# The Expectation explains why one global threshold labels the timetable and a threshold per (day type, hour) labels the year. The table below measures both foils.
#
# - **Global threshold.** Hour dummies alone reach a ROC-AUC of {{p5.label_variants.0.auc_hour_only:.3f}} and the full model {{p5.label_variants.0.auc_full:.3f}}: impressive, and nothing the operator does not already know.
# - **Per (day type, hour).** Hour alone falls to {{p5.label_variants.1.auc_hour_only:.3f}}, but {{p5.label_variants.1.pos_val_2011:.3f}} of 2011 validation hours are positive against {{p5.label_variants.1.pos_val_2012:.3f}} of 2012 hours, and the trend alone reaches {{p5.label_variants.1.auc_trend_only:.3f}}.
# - **Ours.** {{p5.threshold_rule.n_cells}} thresholds by year, day type and hour, each from at least {{p5.threshold_rule.min_cell_n}} training hours. Hour alone scores {{p5.label_variants.2.auc_hour_only:.3f}}, trend alone {{p5.label_variants.2.auc_trend_only:.3f}}, and the positive share is the same in both years ({{p5.label_variants.2.pos_val_2011:.3f}} and {{p5.label_variants.2.pos_val_2012:.3f}}).
# - **Why the {{p5.threshold_rule.quantile}} quantile.** It leaves about a quarter of positives to learn from while still meaning "clearly busier than normal"; a higher quantile would leave very few positives in the smallest cells.
# - **Limit.** A new year has no cell of its own. In use, the thresholds would have to be rolled forward with the trend.

# %%
# Phase 5 labels hours as high demand from training-row thresholds and fits our own logistic regression on the Phase 4 survivors.
from src import plots_p5
from src.logistic import fit_logistic, logloss_grad
from src.phases import p5 as p5mod

p5 = phase("p5", p5mod.run, CFG, RECOMPUTE_ALL, upstream="p4")   # a full run takes about three minutes
assert set(p5["features"]) <= set(p4["survivors_expanded"]), "Phase 5 uses a column that Phase 4 dropped"
print(f"Phase 5 uses {len(p5['features'])} features, all among the {len(p4['survivors_expanded'])} Phase 4 survivors")

# %%
# The label: high demand means above a quantile of training hours with the same year, day type and hour.
rule, balance = p5["threshold_rule"], p5["class_balance"]
cells = pd.DataFrame(rule["table"])
peaks = cells[cells["hr"].isin([3, 8, 12, 17])].pivot_table(index=["yr", "workingday"], columns="hr", values="threshold")
table(peaks.reset_index(), f"Threshold in bikes at hours 3, 8, 12, 17 ({rule['quantile']} quantile of training hours, {rule['n_cells']} cells)")
table({"positives, train": balance["train"], "positives, validation": balance["val"], "positives, validation 2011": balance["val_by_year"]["0"],
       "positives, validation 2012": balance["val_by_year"]["1"], "majority-class accuracy": balance["majority_accuracy"]}, "Class balance")

# %%
# The foils: the same three models under three label rules. Only our rule keeps the hour and the trend near chance (AUC 0.5).
foils = pd.DataFrame(p5["label_variants"])
foils["rule"] = foils["group_by"].map(lambda g: " + ".join(g) or "global")
ours = foils["group_by"].map(lambda g: list(g) == list(p5["threshold_rule"]["group_by"]))
table(foils[["rule", "pos_train", "pos_val", "auc_hour_only", "auc_trend_only", "auc_full", "majority_accuracy"]],
      "Three label rules (highlighted: ours)", ours)

# %% [markdown]
# ### Justification: regularisation of the classifier
#
# Our logistic regression is gradient descent on the mean log-loss with the step 1/(λ_max/4 + l2), which is safe because the curvature of this loss is at most a quarter of that of the squared loss. Without a penalty, some weights of rarely active columns keep growing and the fit reaches the iteration cap (the l2 = 0 row below). So we add a small ridge term, chosen by validation ROC-AUC among {{p5.l2_sweep.0.l2}}, {{p5.l2_sweep.1.l2}}, {{p5.l2_sweep.2.l2}} and {{p5.l2_sweep.3.l2}}, with near-ties going to the larger value. Chosen: {{p5.l2}}.

# %%
# The ridge strength of the classifier, chosen by validation ROC-AUC, and our logistic regression.
sweep = pd.DataFrame(p5["l2_sweep"])
table(sweep[["l2", "iterations", "stop_reason", "train_logloss", "val_logloss", "val_roc_auc"]], "Ridge strength (highlighted: chosen)",
      np.isclose(sweep["l2"], p5["l2"]))
print(f"chosen fit: {p5['stop_reason']} after {p5['iterations']} iterations; gradient check {p5['gradient_check_at_zeros']:.1e}")
show_source(logloss_grad, fit_logistic)

# %% [markdown]
# ### Justification: metrics
#
# About one hour in four is positive, so always answering "normal" already gives an accuracy of {{p5.class_balance.majority_accuracy:.3f}}. We therefore read accuracy together with recall (how many busy hours we catch, the operator's main concern), precision (how many alarms are real), F1, ROC-AUC (ranking quality, independent of the cut-off) and PR-AUC (ranking quality on the smaller class). The three required metrics are reported at our operating cut-off.

# %%
# Validation metrics at the cost-based cut-off, at 0.5 and at the F1-optimal cut-off, with the confusion counts.
cuts = {"cost-based": p5["metrics"], "0.5": p5["metrics_at_0_5"], "F1-optimal": p5["metrics_at_f1_opt"]}
rows = [{"cut-off": name, **{k: v for k, v in m.items() if k != "confusion"}, **m["confusion"]} for name, m in cuts.items()]
metrics = pd.DataFrame(rows)
table(metrics[["cut-off", "threshold", "accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc", "cost", "tn", "fp", "fn", "tp"]],
      "Validation metrics (highlighted: our operating cut-off)", [True, False, False])

# %% [markdown]
# ### Justification: operating threshold
#
# A missed busy hour means empty docks; a false alarm means a few idle bikes. With the cost ratio of the Phase 1 bonus, a calibrated model should raise an alarm when {{p1.bonus.k:g}}·p > 1 − p, that is when p > {{p5.t_cost}}. On the validation rows the cost curve below is lowest at {{p5.t_cost_empirical}} on our grid, next to the theory, and the largest calibration gap is {{p5.calibration_max_gap:.3f}}. We keep the theoretical cut-off instead of tuning it on the validation rows.

# %%
# Where the alarm threshold should sit when a miss costs more than a false alarm, and how well the classifier ranks hours.
show(plots_p5.plot_threshold_curve(p5))
show(plots_p5.plot_roc_pr(p5))

# %%
# What the classifier relies on: the largest standardised weights, and the share of each original column.
table(pd.DataFrame(p5["top_coefficients"]), "The largest standardised weights")
share = pd.Series(p5["coef_abs_share_by_column"], name="share of sum |w|").rename_axis("original column").reset_index()
table(share, "Share of sum |w| by original column")

# %% [markdown]
# ### Pipeline retrospective
#
# Each phase used what the one before it produced; the table below is built from the five artifact files.
#
# - **Phase 1** gave a working optimiser and an honest baseline: validation R² {{p1.val_r2:.3f}} with {{p1.n_features}} weights. Its residuals showed what was missing.
# - **Phase 2** started from Phase 1's weights and added what those residuals asked for: R² {{p2.val_r2:.3f}}.
# - **Phase 3** showed that this model was still too simple rather than too flexible, and that the gap to the chronological split is drift rather than leakage. It moved the target to {{p3.target_complexity.n_features}} weights: {{p3.estimates_target.seeded.r2:.3f}} on validation, {{p3.estimates_target.day_holdout.r2:.3f}} on held-out days, {{p3.estimates_target.chrono.r2:.3f}} chronologically.
# - **Phase 4** found, as that diagnosis implies, that penalties change little, used drop tests to sort the original columns, and recommended {{p4.recommended.method}} on {{p4.survivor_counts.stable}} surviving features: R² {{p4.recommended.val_r2:.3f}}, RMSE {{p4.recommended.val_rmse:.1f}}.
# - **Phase 5** reused exactly those features for a classifier: ROC-AUC {{p5.metrics.roc_auc:.3f}}, F1 {{p5.metrics.f1:.3f}}, accuracy {{p5.metrics.accuracy:.3f}} at the cost-based cut-off.
#
# The lesson of the whole chain: on this data the error comes from bias. Every gain came from giving the linear model structure the data really has, and the usual cures for variance did nothing measurable. **The regression model our pipeline recommends, and the one behind our submission, is the Phase 4 stage-B model.**

# %%
# The five phases side by side, built from the five artifact files (highlighted: the regression model we recommend).
retro = pd.DataFrame(p5["retrospective"])
table(retro[["phase", "consumed", "hyperparameters", "extra", "n_features", "train_score", "val_score", "val_rmse"]],
      "Pipeline retrospective", retro["phase"].str.startswith("P4"))

# %% [markdown]
# ## Outcome — Phase 5
#
# **What happened.**
# - Class balance is as planned: {{p5.class_balance.train:.3f}} positives in training, {{p5.class_balance.val:.3f}} in validation, and {{p5.class_balance.val_by_year.0:.3f}} and {{p5.class_balance.val_by_year.1:.3f}} by year.
# - Our classifier: ROC-AUC {{p5.metrics.roc_auc:.3f}} (95% interval {{p5.val_auc_bootstrap.lo:.3f}}–{{p5.val_auc_bootstrap.hi:.3f}}) and PR-AUC {{p5.metrics.pr_auc:.3f}}. At the cut-off {{p5.metrics.threshold}}: accuracy {{p5.metrics.accuracy:.3f}}, F1 {{p5.metrics.f1:.3f}}, precision {{p5.metrics.precision:.3f}}, recall {{p5.metrics.recall:.3f}}. At 0.5: accuracy {{p5.metrics_at_0_5.accuracy:.3f}}, F1 {{p5.metrics_at_0_5.f1:.3f}}, recall {{p5.metrics_at_0_5.recall:.3f}}.
# - At {{p5.metrics.threshold}} we miss {{p5.metrics.confusion.fn}} busy hours and raise {{p5.metrics.confusion.fp}} false alarms (cost {{p5.metrics.cost:.3f}} per hour); at 0.5 we would miss {{p5.metrics_at_0_5.confusion.fn}} (cost {{p5.metrics_at_0_5.cost:.3f}}).
#
# **What surprised us.**
# - Accuracy is lower at our cut-off ({{p5.metrics.accuracy:.3f}}) than at 0.5 ({{p5.metrics_at_0_5.accuracy:.3f}}) and barely above the do-nothing {{p5.class_balance.majority_accuracy:.3f}}. Judged by accuracy our operating point looks like a mistake; judged by the operator's cost it is clearly better.
# - The honest task is much harder than the trivial one (AUC {{p5.label_variants.2.auc_full:.3f}} against {{p5.label_variants.0.auc_full:.3f}}). Whether an hour is unusually busy for its slot depends partly on events and school holidays that are not in the data.
# - Hour × weather products still carry a large share of the weights, although the hour alone predicts nothing: the weather effect differs by hour.
# - *Second pass:* the classifier hardly moved (ROC-AUC {{p5.metrics.roc_auc:.3f}} against {{first_pass.p5.roc_auc:.3f}}), as expected, and the chosen ridge strength changed to {{p5.l2}} within the noise of the AUC. A tree model outside the pipeline reaches {{diagnostics.tree_classifier_roc_auc:.3f}} on the same label.
