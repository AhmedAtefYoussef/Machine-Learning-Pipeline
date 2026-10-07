# Analyst audit after Phase 4 (first run) — 2026-10-07

Scope: `src/common.py`, `features.py`, `validation.py`, `phases/p1..p4`, `verdicts.py`; artifacts p1-p3 (config sha `e21f737e`) and the **stale** `p4.json` of 15:29 (config sha `18c7f596`, no `final` key; stage B not seen).
Numbers below come from `exp/analyst/{resid,ceiling,hyp,hyp_combo,hyp_wls}.json` (scripts next to them, diagnostic only, H13). The hidden-test file was not opened. Seeded-validation looks spent by this audit: 40 (35 linear variants, 4 ceiling probes, 1 residual run).

## 1 Correctness and leakage

No leak found: every `Design.fit`, `hum_fill`, back-transform factor and alpha grid is computed on the fitting frame; day-block folds live inside the training portion; the hidden-test file is not read by p1-p4. Concrete findings:

| # | sev | where | finding | action |
|---|---|---|---|---|
| D1 | high (transient) | `artifacts/p4.json`, `p5.json` | At 15:39 `p4.upstream_sha256` (`f73b59c5`) != sha of current `p3.json` (`10809717`), `p4.config_sha256` != `config.yaml`, no `final`; `p5.json` (15:37) is newer than `p4.json` (15:29), so it was built on the stale Phase 4. | After the Phase 4 re-run: rebuild p5, run chain check. |
| D2 | medium | `src/features.py:20` + `:87-88`, `src/validation.py:18-29` | In C5 `holiday` = (wk_1+..+wk_5) - workingday exactly (DATA_CARD: 0 exceptions), so the design is rank deficient and only `fit_alpha = 1e-8` makes it solvable. Proof: adding holiday x hour changes all three validators by 0.0000 (it is already in the span). Weights of holiday / workingday / wk_* are not unique; "251 weights" is rank 250. | Say it in Phase 3; do not interpret those coefficients. Dropping `holiday` (and windspeed) costs nothing: seeded -0.0002, days +0.0005, chrono +0.0038. |
| D3 | medium | `src/validation.py:110-128`, `p3.json.noise_floor` | The "noise floor" (0.955, adjusted 0.935) is an in-sample cell-mean score with 30% singleton cells. It is not a limit: the diagnostic model reaches 0.944 on held-out days and the linear model with 33 more columns reaches 0.943, both above the "adjusted" 0.935. | Rename to "cell-mean benchmark"; do not present it as the best achievable. |
| D4 | medium | `src/phases/p4.py:349-357` and stale `p4.json` | The full candidate design (268 columns) scores chrono 0.870 for all three methods against 0.893 for C5 unregularised, with no gain on seeded (0.9313 vs 0.9317) or held-out days (0.9248 vs 0.9262): the re-admitted yr / instant / mnth / season terms cost 0.023 on unseen months. `recommend` never looks at the chronological score. | Check that the stage-B recommendation has chrono >= 0.89; report this loss as the reason those columns are "redundant". |
| D5 | low | `src/validation.py:41`, `src/phases/p4.py:182` (`ETA_HEADROOM`, `validation.py:15`) | Cap = max train log-target + 1.0 (about 2640 bikes). Not a leak (train target only). Inert for C5: 0 of 2178 validation rows, 0 of 8708 held-out-day rows, max validation eta 6.94 vs cap 7.88. But the count of capped rows is stored nowhere, so for C8/C9 and the 10%-of-days learning-curve fits nobody can say whether the reported score was cut. | Store `n_capped` per fit in p3/p4; one sentence in the notebook. |
| D6 | low | `src/common.py:141-164` | Row bootstrap ignores that validation rows share days (23-39% of the squared residual is a whole-day offset). Seeded interval: rows [0.9226, 0.9407], whole days [0.9200, 0.9420]: 22% wider. The paired intervals of the verdict rule (`p4.py:443-446`) are also row-based. | State it; verdicts with a lower end just above 0 are not safe. |
| D7 | low | `src/validation.py:70-74`, `src/phases/p3.py:28,45` | The chronological validator fits on all labelled rows before the cut, which includes the seeded validation rows of that period. Not a leak into the seeded score, but the split is "over all of train.csv", not "inside the training portion". | Say so where the number is reported. |
| D8 | low | `src/phases/p4.py:255-259, 443-446, 553-568` | Alpha choice, column verdicts, survivor selection, the stage-B re-tuning and the reported validation R2 all use the same 2178 rows (double use in stage B). All differences involved are < 0.001, so the bias is small, but no score is untouched: held-out days are consulted by `recommend`, chrono by ADR-013. | One sentence in the Phase 4 Outcome. |
| D9 | low | stale `p4.json.not_converged` | 6 lasso solutions hit `max_sweeps` (validation curve, chrono, 4 CV folds). If the chosen lasso alpha (1.8e-4, near the low end of its grid) is one of them, the lasso row is an unconverged solution. | Verify after the re-run; report the count. |
| D10 | low | `src/common.py:214-229` | `cached_or_run` accepts an artifact when config and upstream hashes match; a change in `src/` does not invalidate it, and the notebook then shows numbers it did not compute. | Say in the notebook that Phase 4 is loaded from cache and how to force a recompute. |
| D11 | doc | `STATE.md`, `docs/adr/002` | STATE still lists Phases 1-4 as "Expectation written / todo". ADR-002 expects method `none`; the artifact chose `ls` (factor 1.041 in Phase 1, 0.9955 at C5: it now shrinks predictions). | Update both. |

## 2 Residuals of the Phase 3 target model (C5, 251 columns)

Scores reproduced: seeded 0.9317 (RMSE 48.2), held-out days 0.9262 (fold sd 0.0112, pooled 0.9265, day-cluster interval [0.9175, 0.9339]), chronological 0.8929 (mean prediction 9.4% too high).

Held-out days (8708 out-of-fold rows) / seeded validation (2178 rows). Bias = prediction - actual, in bikes.

| slice | row share | bias | RMSE | share of squared error | log RMSE |
|---|---|---|---|---|---|
| working-day rush hours 7-9, 16-19 | 20% / 19% | | | **48% / 40%** | |
| working day 18h / 17h / 8h | 2.9% each | +0.6 / -1.3 / -6.3 | 100 / 98 / 90 | 12.1% / 11.4% / 9.4% | 0.24 / 0.23 / 0.27 |
| night 0-5h | 25% / 24% | | | 1.6% / 1.4% (but 43% / 37% of the log-scale error) | |
| non-working days | 32% / 32% | -7.1 / -5.2 | 52.9 / 57.0 | 37% / 45% | 0.35 / 0.38 |
| working days | 68% / 68% | -3.0 / -2.1 | 46.9 / 43.4 | 63% / 55% | 0.31 / 0.31 |
| year 2011 | 50% | -5.1 / -2.6 | 36.5 / 38.3 | 28% / 31% | 0.35 / 0.36 |
| year 2012 | 50% | -3.5 / -3.5 | 58.7 / 56.1 | 72% / 69% | 0.30 / 0.31 |
| weathersit 1 | 66% / 67% | -1.9 / -1.2 | 43.6 / 43.6 | 52% / 55% | 0.27 / 0.28 |
| weathersit 2 | 26% / 24% | -7.5 / -6.1 | 48.8 / 48.1 | 26% / 24% | 0.32 / 0.35 |
| weathersit 3 | **7.8% / 8.4%** | -13.1 / -9.6 | **81.3 / 75.9** | **21.5% / 20.7%** | 0.59 / 0.58 |
| holidays | 2.8% / 3.0% | -12.1 / -16.1 | 81.1 / 82.9 | 7.8% / 8.8% | 0.52 / 0.50 |
| cnt <= 5 | 6.4% / 6.5% | +1.7 / +1.8 | 3.6 / 3.3 | 0.03% | 0.54 / 0.57 (log bias +0.29 / +0.30) |
| top 5% of cnt | 5.0% / 5.1% | -12.1 / -19.9 | 105.4 / 89.1 | 23.3% / 17.5% | 0.18 / 0.15 |
| worst months (held-out days) | | Mar -14.5, Feb -11.3, Jun +8.6 | Jul 61.7, Oct 57.4, May 55.8 | Jul 13.1%, Oct 11.8%, May 11.0% | |

- **Heteroscedastic on both scales.** Bike residual sd grows from 3.8 (lowest prediction decile) to 90.7 (highest); log residual sd falls from 0.49 to 0.18. Least squares on the log scale therefore spends its effort on night hours that carry 1.6% of the scored error.
- **Calibration by prediction decile (held-out days):** deciles 6-9 under-predict by 10-14 bikes, the top decile over-predicts by 18. The top decile alone is 36% of the squared error.
- **Whole-day effects:** 23% of the squared residual (39% on the seeded rows) is an offset shared by all hours of a day; the worst 5% of days hold 26% of the error (2012-03-17: -104 bikes per hour, 2012-05-14: +91, 2012-04-01: -92; three of the ten worst are holidays). No column in the data explains these days.
- **Back-transform:** validation R2 none 0.93164, ls 0.93167 (factor 0.9955), Duan 0.9261 (factor 1.048); the best single factor fitted on validation itself gives 0.93167. Nothing to gain. Mean prediction / mean cnt = 0.984 seeded, 0.978 held-out days.
- **Clipping and cap:** no prediction is clipped at 0 (minimum 0.32 bikes) and no eta is capped, on validation or on held-out days.

Where the error is: high-demand hours (rush hours, 2012, top decile), rainy hours (weathersit 3) and a few atypical days. Not in the quiet hours, the back-transform or the clip.

## 3 Diagnostic ceiling (H13: `exp/analyst/ceiling.py`, never in the chain or the submission)

HistGradientBoostingRegressor on log1p(cnt), 13 raw columns (hr, weekday, workingday, holiday, weathersit, temp, atemp, hum, windspeed, yr, mnth, season, days), two fixed configurations, no tuning.

| model | seeded | held-out days (fold sd) | chronological | seeded - days |
|---|---|---|---|---|
| C5 linear (chain) | 0.9317 | 0.9262 (0.011) | 0.8929 | 0.0055 |
| HGB default (100 trees) | 0.9409 | 0.9392 (0.009) | 0.8968 | 0.0017 |
| HGB 600 trees, lr 0.05 | 0.9546 | **0.9444** (0.009) | 0.9062 | 0.0101 |
| same + previous-hour and day weather | 0.9586 | 0.9476 (0.008) | 0.9080 | 0.0110 |
| C5 + HGB on its log residuals | 0.9563 | 0.9450 (0.010) | 0.9295 | 0.0113 |

Gap of the chain model to the ceiling: **0.018 on held-out days** (paired out-of-fold interval [+0.014, +0.023]), 0.023 seeded, 0.013 chronological (0.037 against the stacked probe). The flexible model's seeded score is 0.010 above its held-out-day score: the seeded split flatters flexible models, as the leakage-audit skill predicts; for the linear model the difference is 0.0055.
`p3.json` cell-mean benchmark: 0.9547 in-sample, 0.9350 degrees-of-freedom adjusted, 2636 cells, 30% singletons. It is exceeded on unseen days by the probe, so it is not an upper limit (finding D3). The realistic limit on unseen days with these columns is about 0.945-0.95.

## 4 Hypotheses (one block added to C5, closed form, same three validators; paired differences against C5)

Noise: fold sd 0.011 and bootstrap half-width 0.009 for a single score; the paired standard error of a difference over the 5 folds is given in brackets.

| # | hypothesis | seeded delta [95% paired] | held-out days delta (paired se, folds up) | chrono delta | cost | risk | compliant | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | Weather detail, same row: hum^2, hum^3, windspeed^2, temp x windspeed, + block `wd_x_hr_x_temp` (`wx_smooth+wd_x_hr_x_temp`) | +0.0073 [+0.0036, +0.0109] | +0.0101 (0.0011, 5/5) | +0.0203 | low: 27-32 columns, one block exists | low | yes | adopt with 2 |
| 2 | Weather of the previous hours: weathersit one hour before, worst weathersit of the last 3 hours (4 columns, inputs only, past only) | +0.0041 [-0.0010, +0.0093] | +0.0107 (0.0019, 5/5) | +0.0107 | medium: `raw_columns` must see neighbouring rows | medium: explain that no statistic is fitted and that the 20th's first hour looks at the 19th | yes | adopt with 1 |
| 1+2 | hum powers + wind terms + lagged weathersit + `wd_x_hr_x_temp` (33 columns, 284 in total) | **+0.0080 [+0.0031, +0.0136]** -> 0.9397 | **+0.0169 (0.0025, 5/5)** -> 0.9431 | **+0.0256** -> 0.9185 | medium: new ladder level, re-run P3-P5 | medium (post-hoc level, ADR needed) | yes | **recommend** |
| 3 | Whole-day weather means (uses later hours of the day) | +0.0058 | +0.0075 (0.0012); only +0.002 on top of 1+2 | +0.0026; -0.002 on top of 1+2 | low | high: looks ahead in the inputs | doubtful | reject |
| 4 | Weight the log-scale fit by the fitted demand (two-stage WLS) / Poisson IRLS | +0.0068 / +0.0084 | +0.0081 (0.0011) / +0.0104 (0.0011); +0.005 on top of 1+2 | +0.0168 / +0.0238 | high: changes the loss of P1-P2, breaks freeze F1 | high | no (new loss / model class) | diagnostic only |
| 5 | Concave trend (sqrt of days) | +0.0022 | +0.0031 (0.0009) | **-0.0112** | low | | yes | reject (loses on chrono) |
| 6 | yr step instead of trend | -0.0063 | -0.0057 | +0.0071 | low | | yes | reject |
| 7 | Hour profile per year | -0.0002 | -0.0008 | -0.0158 | low | | yes | reject |
| 8 | Day type x season / trend / temp | +0.0002 | +0.0001 | -0.0289 | low | | yes | reject |
| 9 | Free hour profile for holidays | -0.0029 | -0.0061 (0.0062) | -0.0045 | low | | yes | reject (13 holiday dates are too few) |
| 10 | Rain x day part x day type; hour x weathersit; day part x season | -0.0002 / -0.0010 / +0.0011 | +0.0003 / -0.0005 / +0.0010 | -0.0021 / -0.0020 / +0.0004 | low | | yes | no gain above noise |
| 11 | Drop holiday and windspeed (Phase 4 verdicts) | -0.0002 | +0.0005 | +0.0038 | none | | yes | confirms the verdicts |

Parts of hypothesis 1 alone (held-out days / chrono): hum powers +0.0069 / +0.0118; wind terms +0.0022 / +0.0086; `wd_x_hr_x_temp` +0.0019 / +0.0023; temp x hum 0.0000; weather x weathersit -0.0012 / -0.0015 (seeded +0.0029: a seeded-only gain, left out). Phase 2 rejected hum powers because they gained 0.0006 < 0.001 there; with hour x hum in the design they gain 0.0036 seeded.

**Recommendation.** One more round: a ladder level "C5 + weather detail" (rows 1+2). Expected on held-out days +0.017 (2 paired se: +0.012 to +0.022), above the 0.011 fold sd and the 0.009 bootstrap half-width, 5 of 5 folds up; chronological +0.026, not a loss. It must pass the existing Phase 3 rule on the seeded validation set (it does: +0.0080, interval above 0, above `plateau_tol`), and needs a superseding ADR because the ladder is extended after the residuals were seen.

**Stop condition.** Today none fires: the gap to the ceiling on held-out days is 0.018 and a compliant hypothesis beats the noise. After adopting rows 1+2 the held-out-day score is 0.943 against 0.944 for the ceiling (0.948 with the same lagged inputs): **(b) fires**, and no remaining compliant hypothesis is above noise (**(a)**). The remaining distance is the loss function (row 4) and unexplained whole days, neither of which is open to us.

## 5 What an examiner will attack

1. **Rule changed after seeing data** (ADR-013: argmax -> "simplest within 0.001"). Documented, but 0.001 is nine times smaller than the bootstrap half-width, and the ADR argues with the chronological score, which the rule itself does not use. A new ladder level would be a second post-hoc change: log it with the number of validation looks.
2. **Everything is tuned on the same 2178 rows**: back-transform (3 options), degree (4), power columns (3), ladder (10), alphas (8 paths x 40), verdicts, stage B. No estimate is untouched. Say "optimistic by construction" and always show held-out days and chrono next to it.
3. **Regularisation does nothing** (all three within 0.0005 of the unregularised fit; ridge picked the lowest alpha of its old grid). Present it as the finding it is: 8708 rows for 251-268 standardised columns leave little variance to remove; the penalty only matters on the 910-column level.
4. **The eta cap** is not in any ADR and its effect is not recorded (D5).
5. **Phase 4 runtime and the cache** (D10): can every member re-run it live, and does the notebook compute what it shows? The stale artifact has no `timings` key, so the runtime cannot be checked from it.
6. **"Noise floor" 0.935 below our own diagnostic 0.944** (D3).
7. **windspeed "uninformative"** holds for the linear term only; windspeed^2 and temp x windspeed gain +0.002 on held-out days and +0.009 chrono. Word the verdict as "uninformative as represented".
8. **Rank-deficient design** (D2) and the meaning of "251 weights".
9. **Seed roster unconfirmed** (AMBIGUITIES A2) and the row bootstrap that treats hours of one day as independent (D6).
