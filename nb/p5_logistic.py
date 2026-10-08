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
# - **The flaw of one global threshold.** If "high" means "above the 75th percentile of all hours", the label is the timetable: 8h and 17–18h on working days, midday at weekends. In the table below, hour dummies alone then reach a ROC-AUC of {{p5.label_variants.0.auc_hour_only:.3f}} and the full model {{p5.label_variants.0.auc_full:.3f}}. That looks excellent and tells the operator nothing new.
# - **Fixing the clock creates a calendar problem.** With one threshold per (day type, hour), hour alone drops to {{p5.label_variants.1.auc_hour_only:.3f}}, but because demand grew so much, only {{p5.label_variants.1.pos_val_2011:.3f}} of 2011 validation hours are "high" against {{p5.label_variants.1.pos_val_2012:.3f}} of 2012 hours, and the trend alone reaches {{p5.label_variants.1.auc_trend_only:.3f}}. Now the label is the year.
# - **Our rule.** An hour is high-demand when `cnt` is above the {{p5.threshold_rule.quantile}} quantile of training hours with the same year, day type and hour: {{p5.threshold_rule.n_cells}} thresholds, each from at least {{p5.threshold_rule.min_cell_n}} training hours, looked up for validation rows. Hour alone is at chance ({{p5.label_variants.2.auc_hour_only:.3f}}), trend alone nearly so ({{p5.label_variants.2.auc_trend_only:.3f}}), and the positive share is the same in both years ({{p5.label_variants.2.pos_val_2011:.3f}} and {{p5.label_variants.2.pos_val_2012:.3f}}).
# - **Why this is meaningful for the operator.** The operator already plans for rush hours and for growth. What they cannot read off a timetable is whether *this* 8 am will be busier than a normal 8 am this year. The top quarter of comparable hours is where a normal allocation of bikes runs short.
# - **Why the 75th percentile.** It keeps enough positives to learn from (about a quarter) while still meaning "clearly above normal"; a 90th percentile would leave about five positives in the smallest cells.
# - **Limit.** A new year has no cell of its own; in use, the thresholds would have to be rolled forward, for example last year's cell scaled by the fitted trend.

# %%
# Phase 5 builds the high-demand label from training-row thresholds and fits our own logistic regression on the Phase 4 survivors; a full run takes about three minutes, so the stored artifact is loaded when nothing upstream changed.
from src import plots_p5
from src.logistic import fit_logistic, logloss, logloss_grad, sigmoid
from src.phases import p5 as p5mod

p5 = phase("p5", p5mod.run, CFG, upstream="p4", recompute_all=RECOMPUTE_ALL)   # writes artifacts/p5.json when it has to run

# %%
# The chain: every feature of this phase must be a survivor of Phase 4.
rows = view.p5_chain_rows(p4, p5)
assert all(passed for *_, passed in rows), "Phase 5 uses a column that Phase 4 dropped"
display(view.checks(rows, "Phase 5 consumed Phase 4"))
view.p5_design(p5)

# %%
# The label: high demand means above a quantile of training hours with the same year, day type and hour.
display(view.p5_thresholds(p5))
view.p5_balance(p5)

# %%
# The foils: the same three models under three label rules. Only the last rule keeps the hour and the trend at chance.
view.p5_label_variants(p5)

# %%
# The foils as a plot.
show(plots_p5.plot_label_variants(p5))

# %% [markdown]
# ### Justification: regularisation of the classifier
#
# The classifier is our own logistic regression: gradient descent on the mean log-loss, with the step 1/(λ_max/4 + l2), which is the safe step for this loss because its curvature is at most a quarter of that of the squared loss. It uses only the Phase 4 survivors (the assert above). We add a small ridge term because, without any penalty, some weights of rarely active columns keep growing and gradient descent does not settle (the l2 = 0 row below stops at the iteration cap). The strength is chosen by validation ROC-AUC among {{p5.l2_sweep.0.l2}}, {{p5.l2_sweep.1.l2}}, {{p5.l2_sweep.2.l2}} and {{p5.l2_sweep.3.l2}}; ties within 0.0001 go to the larger value. Chosen: {{p5.l2}}.

# %%
# The ridge strength of the classifier, chosen by validation ROC-AUC, and how the chosen fit ended.
display(view.p5_l2_sweep(p5))
display(view.p5_fit(p5))

# %%
# Our logistic regression: the log-loss, its gradient, and the fit, which reuses the Phase 1 gradient-descent loop.
show_source(sigmoid, logloss, logloss_grad, fit_logistic)

# %% [markdown]
# ### Justification: metrics
#
# About one hour in four is positive, so a model that always answers "normal" has an accuracy of {{p5.class_balance.majority_accuracy:.3f}}. Accuracy alone would flatter us, so we read it together with: **recall** (how many high-demand hours we catch, the operator's main concern), **precision** (how many alarms are real), **F1** (their balance), **ROC-AUC** (ranking quality, independent of the cut-off and of the class balance) and **PR-AUC** (ranking quality on the positive class, more demanding when positives are the minority). The task's three required numbers, accuracy, F1 and ROC-AUC, are reported at our operating cut-off; the others are shown beside them.

# %%
# Validation metrics at the cost-based cut-off, at 0.5 and at the F1-optimal cut-off, and the confusion matrix.
display(view.p5_metrics(p5))
display(view.p5_confusion(p5))

# %% [markdown]
# ### Justification: operating threshold
#
# A missed high-demand hour means empty docks and lost customers; a false alarm means a few idle bikes. We use the same 3:1 cost ratio as in the Phase 1 bonus. For a model whose probabilities are calibrated, raising an alarm is worth it when p × 3 > (1 − p) × 1, that is when p > 1/(1+3) = {{p5.t_cost}}. So our cut-off is {{p5.t_cost}}, not the habitual 0.5. The plot below checks the argument on the validation rows: the cost is lowest at {{p5.t_cost_empirical}} on our grid, and the calibration plot further down shows how far the probabilities can be trusted (largest gap {{p5.calibration_max_gap:.3f}}).

# %%
# Where the alarm threshold should sit when a miss costs more than a false alarm.
display(view.p5_threshold(p5))
show(plots_p5.plot_threshold_curve(p5))

# %%
# How well the classifier ranks hours.
show(plots_p5.plot_roc_pr(p5))

# %%
# Whether the predicted probabilities can be trusted.
display(view.p5_calibration(p5))
show(plots_p5.plot_calibration(p5))

# %%
# What the classifier relies on: the largest standardised weights, and the share of each original column.
display(view.p5_top_coefficients(p5))
view.p5_column_share(p5)

# %% [markdown]
# ### Pipeline retrospective
#
# Each phase used what the previous one produced, and the table below is built from the five artifact files.
#
# - **Phase 1** gave a working optimiser and an honest baseline: validation R² {{p1.val_r2:.3f}} with {{p1.n_features}} weights. Its residuals showed the missing structure.
# - **Phase 2** started from Phase 1's weights (first loss equal to Phase 1's last) and added what those residuals asked for: R² {{p2.val_r2:.3f}}.
# - **Phase 3** showed that this model was still too simple, not too flexible, and that the gap to a chronological split is drift rather than leakage. It moved the target to {{p3.target_complexity.n_features}} weights: {{p3.estimates_target.seeded.r2:.3f}} on validation, {{p3.estimates_target.day_holdout.r2:.3f}} on held-out days, {{p3.estimates_target.chrono.r2:.3f}} chronologically.
# - **Phase 4** found, as that diagnosis implies, that penalties change little (all three methods within noise of each other), used them to sort the 14 columns into useful, redundant and uninformative, and recommended {{p4.recommended.method}} on the {{p4.survivor_counts.stable}} surviving features: R² {{p4.recommended.val_r2:.3f}}, RMSE {{p4.recommended.val_rmse:.1f}}.
# - **Phase 5** reused exactly those features for a classifier: ROC-AUC {{p5.metrics.roc_auc:.3f}}, F1 {{p5.metrics.f1:.3f}}, accuracy {{p5.metrics.accuracy:.3f}} at the cost-based cut-off.
#
# The thread through all five: on this data the errors come from bias. Every gain came from giving the linear model structure the data really has; the classical variance cures (higher degree, stronger penalty) did nothing measurable. **The regression model our pipeline recommends, and the one behind our submission, is the Phase 4 stage-B model.**

# %%
# The five phases side by side, built from the five artifact files.
view.p5_retrospective(p5)

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
# - The cost argument holds up in practice: on our grid of cut-offs (steps of 0.05) the validation cost is lowest at {{p5.t_cost_empirical}}, next to the theoretical {{p5.t_cost}}, and the probabilities are usable but not perfectly calibrated (largest gap {{p5.calibration_max_gap:.3f}}). We kept the theoretical cut-off instead of tuning it on the validation rows.
# - *Second pass:* as expected the classifier hardly moved (ROC-AUC {{p5.metrics.roc_auc:.3f}} against {{first_pass.p5.roc_auc:.3f}}), because neither the label nor the features depend on the regression target; only the survivor list changed slightly. The chosen ridge strength changed to {{p5.l2}}, by a margin inside the noise of the validation AUC.
# - The honest task is much harder than the trivial one (AUC {{p5.label_variants.2.auc_full:.3f}} against {{p5.label_variants.0.auc_full:.3f}}). Whether an hour is unusually busy for its slot depends on things we only partly observe: the weather explains some of it, events and school holidays are not in the data.
# - Even with the label made relative to the hour, hour-related columns still carry a large share of the weights (table above): the weather effect differs by hour, so the survivors' hour × weather interactions matter even though the hour alone predicts nothing.
