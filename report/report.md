# Rush Hour: predicting hourly bike demand with a five-phase chained pipeline

Team IDs 16006283, 16007032, 16009837 · team seed 44615 · Machine Learning, Winter 2026, Project 1. Every number below is read from our own `artifacts/p1.json … p6.json`; the notebook reproduces them.

## 1 What the data forced us to do

`train.csv` holds days 1–19 of every month of 2011–2012 and `test.csv` the 20th of every month, so the hidden test is whole unseen days inside the same two years. Three findings shaped every later choice. **(a) The signal is hour × day type:** working days have two commute peaks, other days one midday hump; a model with one hour profile cannot fit both. **(b) Demand grew strongly** from 2011 to 2012 and is right-skewed with spread growing with level, so we model `log1p(cnt)` throughout and always report R² and RMSE on bikes after transforming back. **(c) Several columns are copies:** `atemp` of `temp` (r = 0.985), `season` of `mnth`, `workingday` of `weekday` and `holiday` (exactly), and `yr`, `instant` and the date all measure time. Quirks: humidity is 0 on one day (sensor failure, replaced by the training median), weather situation 4 occurs once (merged into 3), `cnt` is never 0 and night hours are the ones missing, so our models never see a zero-demand hour.

**Splits.** One seeded 80/20 split (8708 / 2178 rows) gives every reported score and every tuning decision. Scalers, the humidity fill, expansions, selection and class thresholds are fitted on the training rows only. Two further estimates are shown next to the validation score but never used to report "validation": five folds of whole held-out days inside the training portion (fixed by the calendar, no seed), and the one chronological split of Phase 3. `test.csv` is read once, at the end.

## 2 Phase 1: gradient descent

**Representation.** 35 standardised columns: 23 hour dummies, `workingday`, `holiday`, two weather dummies, `temp`, `hum`, `windspeed`, a trend in days and two day-of-year sine/cosine pairs from `dteday`. Hour as one number gives validation R² 0.193, six harmonics 0.715, dummies 0.721; we kept the dummies. The copies listed above stay out until Phase 4 so that the weights are unique (condition number 45.5).

**Learning rate and stopping.** For the MSE loss gradient descent is stable only below 2/λ_max of XᵀX/n. We compute λ_max = 1.975 and use half the bound, lr = 0.5064, starting from zeros. Sweep at fixed data and start: at 0.001 and 0.01 of the bound the run has not finished after 5000 iterations (stall); at 0.1 it needs 2388; ours 474; at 1.05 the loss explodes after 94 (Figure 1a). We stop when the relative loss change is below 1e-10 and the gradient norm below 1e-6: either test alone can fire on a plateau. The final weights match the closed-form solution to within 1e-5 (used only as a check).

**Back-transform.** Chosen on validation among no factor (R² 0.7181), Duan's smearing (0.6956) and a least-squares factor (0.7212). Duan over-corrects because the large log-errors sit in quiet night hours while the factor inflates the peaks. We keep the least-squares factor for all phases.

**Result.** Validation R² 0.7212 (95% bootstrap interval 0.694–0.748), RMSE 97.3; train R² 0.7293. The residuals by hour differ sharply between day types, which is what Phase 2 addresses. *Surprise:* Duan's correction, the textbook choice, was the worst option.

**Bonus: asymmetric cost.** With under-prediction three times as costly, we minimise L(w) = (1/n) Σ cᵢ(ŷᵢ − yᵢ)², ŷᵢ = exp(xᵢ·w) − 1, cᵢ = 3 if ŷᵢ < yᵢ and 1 otherwise, on bikes because the cost is paid in bikes. Since ∂ŷᵢ/∂w = exp(xᵢ·w)xᵢ, the gradient is ∇L = (2/n) Σ cᵢ(ŷᵢ − yᵢ)exp(xᵢ·w)xᵢ; cᵢ jumps only where the residual is zero and adds no term. The loss is not quadratic, so we use gradient descent with a backtracking step from the Phase 1 weights (4376 iterations, "converged"). Predictions shift up by a factor 1.28; the share of under-predicted validation hours falls from 0.52 to 0.30, the asymmetric cost from 22183 to 14056, and plain RMSE rises from 97.9 to 99.9. The MSE model itself under-predicts 0.52 of hours: a log-scale fit targets the median.

## 3 Phase 2: polynomial regression

**Initialisation.** The 35 base columns keep the Phase 1 scaler from `p1.json`; new columns get their own; the weight vector is Phase 1's with zeros in the new slots. The first Phase 2 loss therefore equals the last Phase 1 loss (0.166405, difference below 1e-9, asserted in the notebook).

**What we expanded.** Only where Phase 1's residuals showed structure: the 23 degree-2 products `workingday × hour` (without them the expanded model reaches 0.7404) and powers of `temp`. Adding powers of `hum` or `windspeed` changes validation R² by less than our tolerance (0.9105 → 0.9111 → 0.9108). `trend` is never raised to a power: a polynomial in time bends away outside the observed dates.

**Degree.** Rule fixed before the run: the smallest degree within 0.001 of the best. Validation R² by degree 1–4: 0.8928, 0.9083, 0.9108, 0.9107 → degree 3.

**Result.** 61 weights, 611 iterations at half the bound (lr 0.3650), validation R² 0.9105, RMSE 55.2; gain over Phase 1 0.189 (paired interval 0.165–0.216); train R² 0.9121. *Surprise:* optimisation stayed easy (condition number 64), because powers and products are built from standardised columns.

## 4 Phase 3: bias–variance and the time question

**What we varied.** A nested ladder anchored at the Phase 2 design (level C3, loaded from `p2.json`): below it we remove what Phase 2 added, above it we add blocks until cells hold only a few rows. We also varied the degree alone (2 to 8: validation R² moves by 0.0026, a flat line that proves nothing) and the amount of training data. Sweeps use the closed-form solution, which our gradient descent reproduces to 1e-5.

| level (weights) | train | validation | held-out days | chronological |
|---|---|---|---|---|
| C1 additive (36) | 0.729 | 0.721 | 0.727 | 0.567 |
| C3 = Phase 2 (61) | 0.912 | 0.910 | 0.909 | 0.858 |
| C4 + hour × temp, hour × hum (107) | 0.922 | 0.923 | 0.918 | 0.885 |
| C5 + weekday × hour (251) | 0.932 | 0.932 | 0.926 | 0.893 |
| C6 + weather detail and memory (282) | 0.947 | 0.939 | 0.943 | 0.914 |
| C8 + hour × season, hour × trend (401) | 0.948 | 0.938 | 0.941 | 0.877 |
| C10 + month × day type × hour (918) | 0.952 | 0.933 | 0.938 | 0.841 |

**Diagnosis: Phase 2 is under-fit (bias), not over-fit.** Its train–validation gap is 0.0016; its learning curve is flat (0.903 with a tenth of the training days, 0.910 with all); and adding blocks raises validation R² by 0.029 (paired interval 0.021–0.037), confirmed on held-out days (+0.033) and chronologically (+0.056). Variance appears only at the top: from level 9 on, train R² keeps rising while all three held-out scores fall (Figure 1b).

**The time question.** We train before 2012-07-01 and validate after (a quarter of the file, half a seasonal cycle, each month seen at least once in training). For Phase 2 the three estimates are 0.910 (seeded), 0.909 (held-out days) and 0.858 (chronological); for the target 0.939, 0.943 and 0.914. We separated the two possible causes. Sharing days between training and validation is worth only 0.0012 (far below the fold spread 0.013): a linear model with a few hundred weights cannot recognise an individual day. The rest of the gap, 0.052, is drift: after the cut mean demand is 262 against 168 bikes and our log-linear trend overshoots it (predictions average 1.11 times the truth). **We trust the chronological estimate for future periods**; for our hidden test, which lies inside the observed period, held-out days are the closest match. The diagnosis does not change in kind, the simpler models are worse on every split, but the chronological split is the one that punishes the upper levels, so it limits how far up we go.

**Target complexity.** Rule: the simplest level within 0.001 validation R² of the best → C6_weather_detail, 282 weights. Two disclosures. Our first rule was plain "highest validation R²"; on its first run it preferred a level 140 weights larger on a difference of 0.0005, so we replaced it with the tolerance rule we already used for the degree. And level C6 was added after a residual analysis of C5 (error concentrated in working-day rush hours and in rain): powers of humidity, wind terms, `workingday × hour × temp`, and the weather situation of the previous one and three hours (inputs only). It was accepted because it gained on held-out days in every fold and on the chronological split; having been designed after looking at validation errors, its validation score is the most optimistic number in this report.

## 5 Phase 4: regularization and column verdicts

**Search, informed by Phase 3.** The model we carry forward is not over-fit, so the grid must reach practically zero and include the unpenalised fit as reference. All three methods use the same standardised design (the target plus the held-back copies `atemp`, `yr`, `instant`, `season`, `mnth`, so every original column is present), the same rows, 40 log-spaced penalties each (Ridge 1e-06–1000; Lasso and Elastic Net from the all-zero penalty down to a ten-thousandth of it; l1_ratio in six steps from 0.1 to 0.95), the same rule (highest validation R² on bikes) and the same checks. Ridge is closed-form; Lasso and Elastic Net are our own coordinate descent, agreeing with a reference library to better than 1e-3.

| stage A: full design (299 weights) | λ | l1_ratio | non-zero | validation R² | RMSE | held-out days | chrono |
|---|---|---|---|---|---|---|---|
| unpenalised | – | – | – | 0.9372 | 46.2 | – | – |
| Ridge (L2) | 4.9e-06 | – | 298 | 0.9382 | 45.8 | 0.9421 | 0.913 |
| Lasso (L1) | 1.8e-04 | – | 288 | 0.9382 | 45.8 | 0.9416 | 0.898 |
| Elastic Net | 6.1e-04 | 0.3 | 290 | 0.9382 | 45.8 | 0.9416 | 0.898 |

**Comparison.** The three are tied: the paired bootstrap intervals of every pairwise difference contain zero (for example Lasso − Ridge -0.0000, interval -0.0006 to 0.0006), and so does each method's gain over the unpenalised fit. Regularization here is insurance, not a cure, exactly what an "under-fit" diagnosis implies. Where the model is over-fit it does help: on the top ladder level Lasso lifts validation R² from 0.9334 to 0.9362 keeping 532 weights, still not above our recommended model (difference -0.0024, interval -0.0054 to 0.0006). Penalties chosen on held-out days instead would have changed validation R² by at most 0.001. Lasso is not sparse (288 weights survive): the signal is spread over many hour-specific columns.

**Verdict for every column.** Rule fixed before the run; reference = Ridge at its chosen λ. *Useful:* dropping every feature built from the column lowers validation R² by at least 0.001 with a paired interval above zero. *Redundant:* dropping it alone costs nothing, but dropping it with its measured copies does, or it explains at least 0.01 on its own. *Uninformative:* neither. Where a group of copies matters but no member is missed alone, the member that explains most on its own is kept as the useful representative (marked *).

| column | verdict | Δ R² if dropped alone [95% interval] | Δ R² if dropped with copies | R² alone | Lasso keeps (freq.) | carried by |
|---|---|---|---|---|---|---|
| hr | useful | 0.6816 [0.6254, 0.7372] | – | 0.490 | 23 of 23 (1.00) | |
| temp | useful | 0.0292 [0.0224, 0.0376] | 0.0293 | 0.072 | 3 of 3 (1.00) | |
| atemp | redundant | 0.0000 [-0.0000, 0.0001] | 0.0293 | 0.105 | 1 of 1 (0.74) | temp |
| hum | useful | 0.0049 [0.0017, 0.0079] | – | 0.095 | 3 of 3 (1.00) | |
| windspeed | useful | 0.0016 [0.0001, 0.0033] | – | 0.013 | 2 of 2 (1.00) | |
| weathersit | useful | 0.0092 [0.0027, 0.0160] | – | 0.019 | 6 of 6 (1.00) | |
| workingday | useful | 0.0103 [0.0040, 0.0188] | 0.1879 | -0.000 | 1 of 1 (0.90) | |
| weekday | useful | 0.0089 [0.0057, 0.0121] | 0.1879 | 0.002 | 6 of 6 (1.00) | |
| holiday | redundant | 0.0000 [-0.0000, 0.0000] | 0.1879 | 0.000 | 0 of 1 (0.72) | weekday, workingday |
| dteday (trend, season terms) | useful* | 0.0004 [-0.0005, 0.0013] | 0.1203 | 0.122 | 3 of 5 (1.00) | |
| yr | redundant | -0.0001 [-0.0002, 0.0001] | 0.1044 | 0.066 | 0 of 1 (0.56) | dteday |
| instant | redundant | 0.0002 [-0.0003, 0.0006] | 0.1044 | 0.086 | 0 of 1 (0.00) | dteday |
| season | redundant | -0.0000 [-0.0000, 0.0000] | 0.0034 | 0.052 | 1 of 3 (0.98) | dteday |
| mnth | redundant | -0.0012 [-0.0022, -0.0003] | 0.0034 | 0.061 | 9 of 11 (0.96) | dteday |

Three things in this table surprised us. Lasso's choice between copies is arbitrary and once backwards: it keeps `atemp`, which costs nothing to drop, in most resampled fits and never keeps `instant`, which alone explains 0.086; so "Lasso set it to zero" was evidence for us but never the verdict. `dteday` is useful only as a group: its copies can replace it completely (cost alone 0.0004, with them 0.120). And `windspeed` is useful only since Phase 3 gave it a square and an interaction with temperature; as a single linear column the same rule had called it uninformative. A verdict judges a column as we represented it (Figure 2a).

**Survivors and final model (stage B).** We first remove every feature built from a column that is not useful ({{p4.survivor_counts.useful_design}} features remain), then run Lasso on that reduced design; a feature survives if its weight is non-zero at the chosen λ and it is selected in at least 0.6 of 50 Lasso fits on resampled training days: 288 → 288 features. (Our first version took the Lasso zeros before removing the copies; Lasso had let month dummies and `instant` stand in for date terms, the verdict step then removed the stand-ins, and the chronological score fell by about 0.013. We corrected the order.) The redundant columns add nothing: dropping `mnth` alone changes validation R² by -0.0012, so the model is slightly better without it. We therefore repeated the same search on the survivors only and recommend among those: Ridge 0.9385 / 0.9423 / 0.901 (validation / held-out days / chronological), Lasso 0.9386 / 0.9423 / 0.901, Elastic Net 0.9386 / 0.9424 / 0.901. They are again tied on validation; our rule (within noise on validation, then best on held-out days) recommends **enet** with λ = 7.7e-04, l1_ratio 0.3: validation R² 0.9386, RMSE 45.7. This second stage was added after the first Phase 4 run; all runs are in the git history.

## 6 Phase 5: logistic regression

**Label.** A single global threshold has an obvious flaw: "high" then just means rush hour. Under the global 75th percentile, hour dummies alone reach ROC-AUC 0.855 and the full model 0.985: impressive and useless, the operator already has a clock. A threshold per (day type, hour) fixes the clock but not the calendar: 0.054 of 2011 validation hours are positive against 0.421 of 2012, and the trend alone reaches 0.810. **Our label:** `cnt` above the 75th percentile of training hours with the same year, day type and hour (96 thresholds from the training rows, at least 50 hours each): "busier than three quarters of comparable hours", the case where the normal allocation runs short. Positives are 0.246 of training and 0.250 of validation hours (0.239 / 0.262 by year); hour alone is at chance (0.502) and trend alone at 0.577.

**Model.** Our own logistic regression (gradient descent on the log-loss, lr = 1/(λ_max/4 + l2), gradient-checked) on the 288 Phase 4 survivors only (asserted). A small ridge term is chosen by validation ROC-AUC: l2 = 0.0001, 15307 iterations.

**Metrics for a 1:3 class balance.** Always answering "normal" is right 0.750 of the time, so accuracy flatters; we read it with F1, precision, recall, ROC-AUC and PR-AUC. **Threshold.** A missed high-demand hour (empty docks) costs more than a false alarm; with the 3:1 ratio of the bonus the cost-minimising cut-off of a calibrated model is 1/(1+3) = 0.25.

| cut-off | accuracy | F1 | precision | recall | ROC-AUC | PR-AUC | cost per hour |
|---|---|---|---|---|---|---|---|
| 0.25 (ours) | 0.790 | 0.661 | 0.554 | 0.820 | 0.882 | 0.725 | 0.300 |
| 0.5 | 0.823 | 0.619 | 0.670 | 0.574 | 0.882 | 0.725 | 0.390 |

ROC-AUC has a 95% interval of 0.866–0.897. At 0.25 we catch 0.82 of the high-demand hours at a precision of 0.55; at 0.5 accuracy looks better but 232 high-demand hours are missed instead of 98. On validation the cost is lowest at a cut-off of 0.25, close to the theoretical 0.25, and the largest calibration gap is 0.136. *Surprise:* accuracy is lower at our cut-off than at 0.5 and barely above the do-nothing baseline; judged by accuracy our operating point looks like a mistake, judged by the operator's cost (0.300 against 0.390 per hour) it is clearly better (Figure 2b). The honest label is also far harder than the trivial one (AUC 0.882 against 0.985): whether an hour is unusually busy for its slot depends on things we only partly observe.

## 7 Retrospective, final model and what we do not claim

| phase | consumes | key choices | weights | validation |
|---|---|---|---|---|
| 1 GD | seeded split | lr 0.506 = half the bound, 474 iterations | 36 | R² 0.721, RMSE 97.3 |
| 2 Polynomial | P1 weights as start | degree 3 (temp), workingday × hour | 61 | R² 0.910, RMSE 55.2 |
| 3 Bias–variance | P2 degree and features | under-fit; target C6_weather_detail | 282 | R² 0.939; days 0.943; chrono 0.914 |
| 4 Regularization | P3 target and diagnosis | λ per method (table), 288 survivors, enet | 277 | R² 0.939, RMSE 45.7 |
| 5 Logistic | P4 survivors | year × day type × hour label, cut-off 0.25 | 278 | acc 0.790, F1 0.661, AUC 0.882 |

The chain is checked in code: each artifact stores the hash of the one it consumed; Phase 2's first loss equals Phase 1's last; Phase 3's anchor is Phase 2's design; Phase 4's design is Phase 3's target; Phase 5's features are a subset of Phase 4's survivors. The story the numbers tell is one of bias: each step that added structure the data really has (day-type hour profiles, then weather by hour, then weekday profiles, then weather detail) raised validation R² from 0.72 to 0.94, while penalties, the classical cure for variance, changed nothing measurable.

**Submitted model.** enet on the Phase 4 survivors with the validated hyper-parameters, weights refitted on training + validation rows (10886 rows; the Phase 3 learning curve of the target is still rising at full size) with scaler and humidity fill kept as fitted on the training rows. Refit and validated model agree on the test rows (correlation 0.9997, mean ratio 1.001). Sanity of the 574 predictions: none negative, mean 197.8 bikes (training mean 191.6), per (day type, hour) cell between 0.77 and 1.09 times the training profile.

**What we do not claim.** (1) Validation scores are slightly optimistic: back-transform, degree, target level and penalties were all chosen on the same 2178 rows, and two rules (target level, stage B) and one ladder level were changed after first results; held-out days and the chronological split are our guard, and for unseen future months the honest figure is the chronological 0.901, not 0.939. (2) The trend is a straight line in log space: fine inside 2011–2012, too steep beyond. (3) The model has never seen a zero-demand hour, a new weather regime, or a holiday type absent from training; the Phase 5 thresholds exist only for the two observed years. (4) A diagnostic tree model kept outside the pipeline suggests a little accuracy is still left for a more flexible model class; within linear models we found no further change that beat the noise on held-out days.

<div class="figrow"><div><img src="figs/lr_sweep.png"><p class="cap"><b>Figure 1a.</b> Phase 1 loss for six learning rates given as fractions of the stability bound: stall, converge, diverge.</p></div><div><img src="figs/ladder.png"><p class="cap"><b>Figure 1b.</b> Phase 3 ladder: training R² keeps rising with the number of weights; the three held-out estimates peak at the target and then fall, the chronological one most.</p></div></div>
<div class="figrow"><div><img src="figs/verdicts.png"><p class="cap"><b>Figure 2a.</b> Phase 4: validation R² lost when a column is dropped alone and together with its copies (dotted line: the 0.001 threshold).</p></div><div><img src="figs/threshold.png"><p class="cap"><b>Figure 2b.</b> Phase 5: precision, recall, F1 and operator cost against the probability cut-off (dotted lines: 0.25 and 0.5).</p></div></div>
