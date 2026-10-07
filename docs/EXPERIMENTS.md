# EXPERIMENTS (append-only). Diagnostic work lives in exp/ and never enters the chain. Looks at the seeded validation set are counted.

# EXP-001 Phase-0 design options (exp/eda0/eda.py → eda.json)
hypothesis: target, hour encoding, cleaning flags and polynomial degree matter by more than the fold spread      category: modeling
compliance: yes (train-only fits; closed-form oracle; test file not read for modelling)      looks at seeded validation: 1
result: log1p beats raw from the workingday x hour block on (0.899 vs 0.839); one-hot hour 0.655 vs 6 harmonics 0.651 vs numeric < 0.4; hum fix and wind flag within ±0.001; degree plateau after 2-3.
decision: adopt ADR-001, 003, 004, 007.

# EXP-002 Time representation, back-transform, ladder preview (exp/eda0/eda1.py → eda1.json)
hypothesis: trend vs yr and the smearing factor change the day-block and chronological estimates; a block ladder shows a real over-fit region      category: validation
compliance: yes      looks at seeded validation: 1
result: trend + 2 harmonics 0.911 / 0.911 / 0.858 (seeded / day-block / chrono) vs yr 0.905 / 0.904 / 0.876; Duan −0.007 seeded and −0.06 chrono; ladder peaks near 400 columns (day-block 0.933) and falls to 0.929 at 900 while train keeps rising; hr x trend hurts chrono by 0.04.
decision: adopt ADR-002, 005, 006; ladder order fixed in config.yaml before Phase 3 runs.

# EXP-A1 Residual analysis of the Phase 3 target model C5 (exp/analyst/resid.py → resid.json; analyst)
hypothesis: the remaining error is concentrated in a few slices that name the next representation change      category: validation
config_sha: 4d739b5fba1f               compliance: yes (diagnostic, train-only fits, hidden-test file not opened)      looks at seeded validation: 1
expected_gain: none (diagnosis)   cost: 4 s   risk: low
result: held-out days: working-day rush hours 20% of rows / 48% of squared error; weathersit 3 7.8% / 21.5% (RMSE 81 vs 44); holidays 2.8% / 7.8%; night hours 25% / 1.6% (43% of the log-scale error); top prediction decile 36%; 23% of the squared residual is a whole-day offset. Back-transform none 0.93164 / ls 0.93167 / Duan 0.9261; 0 rows clipped at 0, 0 rows hit the eta cap. Seeded interval by rows [0.9226, 0.9407], by whole days [0.9200, 0.9420].
decision: adopt (as diagnosis; report: reports/analysis/audit.md)     adr: none

# EXP-A2 Diagnostic ceiling: gradient-boosted trees on the raw columns (exp/analyst/ceiling.py → ceiling.json; analyst)
hypothesis: a flexible model on the same columns shows how far C5 is from what the data allow      category: validation
config_sha: e6020426f213               compliance: yes ONLY as a diagnostic (H13): never in the chain, the submission or a graded claim      looks at seeded validation: 4
expected_gain: not applicable   cost: 25 s   risk: low
result: seeded / held-out days / chrono: C5 0.9317 / 0.9262 / 0.8929; HGB default 0.9409 / 0.9392 / 0.8968; HGB 600 trees 0.9546 / 0.9444 / 0.9062; with lagged and day weather 0.9586 / 0.9476 / 0.9080; C5 + HGB on residuals 0.9563 / 0.9450 / 0.9295. Gap on held-out days 0.018 (paired out-of-fold interval +0.014 to +0.023). The p3 cell-mean benchmark (0.935 adjusted) is exceeded on unseen days, so it is not an upper limit.
decision: adopt (as the stop-condition reference)     adr: none

# EXP-A3 Cheapest tests of improvement hypotheses on C5 (exp/analyst/hyp.py → hyp.json, hyp_combo.json, hyp_wls.json; analyst)
hypothesis: weather detail (hum powers, wind terms, workingday x hour x temp) and the weather of the previous hours remove the weathersit-3 and rush-hour error      category: modeling
config_sha: c897b1d5e3ec (single blocks), c2c159424b88 (combinations), fe4170b89a93 (weighted fits)               compliance: rows 1+2 yes (representation only, inputs only, no fitted statistic); whole-day weather doubtful (looks ahead); weighted / Poisson fit no (new loss)      looks at seeded validation: 35
expected_gain: +0.010 to +0.020 on held-out days, >= 0 on chrono; noise: fold sd 0.011, bootstrap half-width 0.009   cost: 40 s   risk: medium (ladder extended after seeing residuals)
result: seeded / held-out days (paired se) / chrono deltas against C5: weather detail + wd_x_hr_x_temp +0.0073 / +0.0101 (0.0011) / +0.0203; lagged weathersit +0.0041 / +0.0107 (0.0019) / +0.0107; both (33 columns) +0.0080 [+0.0031, +0.0136] / +0.0169 (0.0025, 5 of 5 folds) / +0.0256, giving 0.9397 / 0.9431 / 0.9185. Rejected: sqrt trend (chrono -0.011), yr for trend (days -0.006), hour x year (chrono -0.016), day type x time (chrono -0.029), free holiday profile (days -0.006), rain x day part, hour x weathersit (within noise). holiday x hour changes nothing (already in the span: the C5 design is rank deficient). Weighted log fit +0.0081 and Poisson IRLS +0.0104 on held-out days (diagnostic, not compliant).
decision: defer to chief: adopt "C5 + weather detail + lagged weathersit" as one new ladder level if the P3-P5 re-run is accepted; after it the held-out-day score is within 0.002 of the ceiling (stop condition b)     adr: needs a superseding ADR (ladder extended after Phase 3)
