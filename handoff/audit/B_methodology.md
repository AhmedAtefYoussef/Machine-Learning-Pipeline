# Audit B: ML methodology, correctness, leakage, requirement fit (analyst, read-only)
scope: Project_1_description.md vs src/, config.yaml, artifacts/p1..p6.json, docs/adr/001-018, handoff/AMBIGUITIES.md, report/report.template.md.
method: code read + artifact key read. No experiment run, so no EXPERIMENTS.md block.
verdict: no leakage found, no H1-H13 violation, no correctness bug that changes a graded number. Findings are about defensibility, plus one inconsistent baseline (B5).
Expected metric gain of every fix below: 0 within noise. **No further gain above noise.**

## Verified OK (evidence)
- H1/1.8: seed 44615 from config.team_ids; single `train_test_split(df, test_size=0.20, random_state=seed)` in src/common.py:118. It runs on raw CSV row order (the added memory columns do not reorder rows), so it equals the reference split. n_train 8708 / n_val 2178.
- H2: Design.fit on the fitting frame only (features.py:209-236); hum_fill = train median of hum>0; P2 reuses P1 scaler (p2.py:45); day-block folds and the chrono split refit the design on their own fitting rows (validation.py:196, p4.py:158). The back-transform factor comes from fitting rows (common.py:172). Label thresholds come from train (labels.py:14).
- H11: test.csv is read only in common.load_test, called only from predict.py:103; never split or shuffled; submission has 574 rows in test order (predict.py:121).
- H4: P1/P2 use own GD (gd.py). lstsq appears only as an oracle (p1.oracle.max_abs_weight_diff 4.8e-6, p2 likewise). lr sweep shows stall/converge/diverge (p1.lr_sweep fractions 0.001/0.01 max_iter, 0.1-0.9 converged, 1.05 diverged).
- H5: p2.init_loss == p1.train_loss_final (assert p2.py:50); P2 starts from lifted P1 weights; p3 anchor names == p2.feature_names (p3.py:97); p4.design_from = C6 = p3.target; P5 features = p4.survivors_expanded; p6.model == p4.recommended (l1).
- A10: every R2/RMSE is on cnt after the inverse power transform. Clipping: p6.pred.n_clipped 0, min 0.94.
- H6: three own solvers on the same design/rows/folds; oracle tests exist. H7: survivors only; label from cnt.
- Phase 3 evidence is sound: anchor gap 0.0028, flat learning curve, ladder gain +0.0235 [0.0177, 0.0298] confirmed on days (+0.028) and chrono (+0.044); variance from C10 (train up, all held-out scores down). Verdict: under-fit, which is defensible.
- Phase 5 fix for the global-threshold flaw is shown by foils (p5.label_variants: global hour-only AUC 0.855; (wd,hr) 2011/2012 positives 5%/42%; (yr,wd,hr) hour-only 0.50, trend-only 0.58).

## Findings
### B1 Medium: target exponent and ladder level C6 were chosen after looking at all three validators
evidence: ADR-017 picks lam=0.1 from exp/v2 pilots on seeded, held-out days and chrono. Phase 1's own table prefers 0.3 (p1.target_power_best_here=0.3: 0.7416 vs 0.7361 at 0.1). ADR-015 inserted C6 after a residual audit, accepted on held-out days and chrono. So a decision used in Phase 1 was informed by Phase 3-level results, and no untouched estimate remains.
why (grading): "Expectation before result" / "obviously after-the-fact" (1.7). A TA will ask "why 0.1 in Phase 1 when your table says 0.3?" and "which estimate is clean?".
fix: docs only. (a) The P1 justification cell should state the exponent was fixed for the end model (ADR-017), P1's own best is 0.3 (delta 0.0055), and P3 re-checks it (p3.target_power_consistent true). (b) The report "What we do not claim" (1) already discloses that held-out days and chrono were looked at; keep it, and name lam and C6 explicitly there. (c) WALKTHROUGH: a one-line answer for each.
artifacts change: no.

### B2 Medium: the L1 selection sits on the lower edge of its grid, so the "subset L1 leaves non-zero" is near-trivial
evidence: p4.selection.lambda 8.35e-5 = alpha_max x 1e-4 (grid minimum, best index 39 of 40). It keeps 277 of 280 (only hr_12*temp, hr_9*hum, wk_6*hr_16 are zeroed). Stability freq is at least 0.84 everywhere, so the 0.6 threshold never binds (survivor_counts lasso_nonzero = stable = 277). The optimum is not bracketed. The real selection is done by the verdict step (it removes holiday, atemp, yr, instant, season, mnth: 299 -> 280).
why: the brief asks for "the feature subset that L1-type regularization leaves non-zero". Here it is met formally, but a TA may say the selection is decorative or that the grid did not bracket the optimum.
fix: (a) docs: say plainly that the optimum is at the grid edge and that L1 removes 3 features, which the report already half-says ("Lasso removes almost nothing"), and give the cv_1se subset as a sensitivity (p4.final.cv.l1.cv_1se_alpha 0.00365, val 0.9381, i.e. -0.003 for real sparsity). (b) optional: set l1_alpha_min_ratio 1e-5 so the optimum is interior; expected delta 0 +- 2e-4.
artifacts change: (a) no; (b) yes, P4->P6 rerun of about 30 min. Recommend (a) only.

### B3 Low-Med: stage-B ridge lambda is on the grid edge, which contradicts the extension rule in ADR-010
evidence: p4.final.methods.l2.lambda = 1e-6 = ridge_alpha_range min (index 39 of 40; val 0.94125177 vs 0.94125178 at the next point). ADR-010 says the range was extended once for exactly this reason. Stage-A ridge 0.001 is interior.
why: inconsistent with the team's own rule; "is regularization doing anything?" Every stage-B method minus unregularised is within +-2.5e-4 (p4.final.comparisons.*_minus_unregularised).
fix: docs: amend ADR-010 to say that an edge at the unpenalised end is reported as "no penalty wanted" (the report already says "i.e. none"), rather than extending again. The evidence that regularization works where variance exists is p4.rich_check: C10 unreg 0.9384 -> lasso 0.9394. Lead with that sentence in Phase 4.
artifacts change: no.

### B4 Low-Med: the bonus shift is measured against a different baseline than the reported Phase 1 model
evidence: p1.py:270-271 scores both bonus models with factor s=1. p1.bonus.val.mse_model.r2 0.7278 = p1.backtransform.candidates.none, while the headline P1 model uses ls factor 1.0693 (val 0.7361). The report says "predictions shift up by a factor 1.28" (p1.bonus.mean_shift_ratio). Against the reported model the shift is about 1.20 (225.5 / (175.9 x 1.069)). The under-prediction share 0.50 also refers to the factor-free model.
why: the bonus asks to "show how the fitted predictions shift compared to your MSE model". A TA cross-checking 0.7361 against 0.7278 will spot the mismatch.
fix: cheapest: one report sentence saying "both without the back-transform factor; against the reported P1 model the shift is x1.20". Code fix: score mse_model with the ls factor in asymmetric_bonus. That changes p1.json bytes, so the whole chain rehashes (P4 about 30 min, numbers downstream unchanged).
artifacts change: doc fix no; code fix yes (full cascade). Recommend the doc fix.

### B5 Low: verdict evidence is selected and scored on the same validation rows; dteday is "useful" only by promotion
evidence: the verdict reference is ridge at the val-chosen lambda, with drop tests scored on the same 2178 val rows (p4.py:498-506). The day-block drop costs agree in sign for all 14 columns, which is good and worth saying. dteday: drop_alone +0.0002 [-0.0004, +0.0008] (not significant), useful only through promote_representatives (representative=true; drop_group 0.120). evidence_sentence (verdicts.py:165) does not mention the promotion.
why: "show the numbers that support each verdict". A TA reading "dteday useful, dropping it costs 0.0002 (CI spans 0)" sees a contradiction.
fix: report/notebook prose: "useful as the representative of the time group {season, mnth, yr, instant, dteday}: dropping the group costs 0.120; its copies make the alone-drop zero". Optional code: add the representative clause to evidence_sentence (P4 rerun).
artifacts change: prose no; code yes (P4 about 30 min).

### B6 Low: the comparison is fair, but elastic net gets six times more looks at the validation set
evidence: l1/l2 have 40 candidates each, enet has 6 x 40 = 240 (config p4.l1_ratios). It is immaterial here (enet-l1 -4e-7 [-1.1e-6, 2.9e-7]), and enet picked 0.95, i.e. lasso.
fix: one sentence in the fairness paragraph. artifacts change: no.

### B7 Low: interval widths are not all reported; the chrono estimate has no interval
evidence: seeded target CI [0.9332, 0.9489] (width 0.016); day-block fold sd 0.0077 (se about 0.0035); paired method deltas +-2e-4; chrono 0.938 is a single number. Seeded vs held-out days: gaps.target.leakage -0.0049, so seeded is NOT optimistic relative to days for this linear model (the report says so).
fix: optional: a day-block bootstrap of the late rows in exp/ (diagnostic), cited in the report as an approximate +-. In-chain it would change p3 and cascade, so do not.
artifacts change: exp-only no.

### B8 Low: Phase 5 operating point and the l2 sweep need one sentence each
evidence: t=0.25 assumes calibration, but p5.calibration_max_gap is 0.091 and the empirical cost-min on val is 0.20 (p5.t_cost_empirical). Accuracy at 0.25 is 0.784 vs majority 0.750 (0.827 at 0.5). The l2=0 candidate hit max_iter 50000 (p5.l2_sweep[0]); it was not chosen. The label groups by yr, a non-survivor (label, not a feature: H7-compliant, but a predictable question).
fix: prose only. artifacts change: no.

### B9 Low: rebuild dependencies are incomplete; the chain hash covers neither code nor data
evidence: run.py:20-27 PHASE_DEPS and Makefile:12-20 omit src/gd_asym.py (p1), src/phases/p1.py (p2 imports it), src/phases/p3.py (p4 imports ladder_specs), src/verdicts.py (p4), src/labels.py (p5), src/predict.py (p6) and data/*.csv. Staleness is mtime-based (git checkout and OneDrive touch mtimes). tools/chain_check.py order stops at p5; p6 is checked only by run.py p6_chain. nb has RECOMPUTE_ALL and a data-file check (nb/p0_setup.py:110), which mitigates this.
why: "make all rebuilds phase N if anything upstream changed" (CLAUDE.md) is not fully true, so a stale artifact is possible after an edit to verdicts.py or labels.py.
fix: add the missing files to PHASE_DEPS and the Makefile prerequisites; optionally add p6 to the chain_check order. artifacts change: no.

### B10 Low: weather memory is a joint, not row-wise, transform of test.csv
evidence: common.py:105-113 builds test lags from the train-input and test-input timeline (no labels, no fitted statistic). Train rows right after a test day (the 21st, hours 0-2) fall back to their own value, because test hours are absent from the train timeline, while test rows see the full timeline. This is a small construction asymmetry.
why: H2/H11 say "test only transformed". This is compliant (a deterministic lookup, nothing estimated), and ADR-015 records it, but expect the question.
fix: WALKTHROUGH answer; optional: build train lags from the same union timeline (inputs only). Expected delta well under 1e-4. artifacts change: optional fix yes (full cascade). Do not.

### B11 Low: a report trace key is fragile
evidence: report.template.md:33 "{{p2.power_col_sweep.0.n_features}} weights". It is correct (61) only because the chosen power-column list is index 0.
fix: use the count of p2.feature_names, or add p2.n_features. artifacts change: no for the template; adding a key needs a P2 rerun, so prefer the template change.

## Ranked improvement table (analyst format)
| # | hypothesis | exp. delta held-out days (noise +-0.0035 se) | exp. delta chrono | cost | risk | compliant | cheapest test |
|---|---|---|---|---|---|---|---|
| B2(b) | bracket the selection-lasso optimum (min ratio 1e-5) | 0 +- 0.0002 | 0 +- 0.001 | P4 rerun 30 min | low | yes | rerun P4 in a worktree, diff p4.final |
| B4 (code) | score the bonus MSE baseline with the ls factor | n/a (bonus) | n/a | full cascade | low | yes | 3-line change, run P1 only in exp/ |
| B10 | union timeline for train lags | < 1e-4 | < 1e-4 | full cascade | low | yes | exp/ closed-form C6 fit |
| B3 | ridge grid to 1e-8 or alpha=0 | 0 (3e-8 measured) | 0 | P4 rerun | none | yes | read p4.final.curves.l2 |
None exceeds noise. Recommend only the zero-cost doc fixes: B1, B2(a), B3, B4 (doc sentence), B5 prose, B6, B8, B9 deps, B11.
