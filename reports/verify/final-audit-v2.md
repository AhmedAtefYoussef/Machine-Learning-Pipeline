# Final audit, second pass (branch `v2/ceiling`, HEAD 6940856)

Independent audit against `Project_1_description.md` only. Read: the description, the executed `rush_hour.ipynb` (cell sources and outputs), `report/report.md`, `artifacts/*.json`, `config.yaml`, `src/`, `nb/`, git history and tags. Not read: `reports/`, `docs/adr/`, `docs/WALKTHROUGH.md`, `handoff/`, `exp/`. Cell numbers are 0-based indices into `rush_hour.ipynb`.

## Overall: PASS WITH FIXES

No rule of the description is broken outright. Four points are open to a strict reading (R1 to R4), and the report contains one false statement (R5). All five can be fixed without changing a single model number; only R4 needs the notebook re-executed.

---

## A. Description compliance (own checklist)

| # | Requirement (section) | Verdict | Evidence |
|---|---|---|---|
| 1 | Five algorithms in order GD, polynomial, bias-variance, regularization, logistic (1.7) | pass | Notebook sections at cells 8-34, 35-49, 50-71, 72-100, 101-121; `src/phases/p1..p5.py` |
| 2 | P1 gradient descent from scratch, no sklearn fit/predict (1.7) | pass | `src/gd.py:47-81`; the only sklearn import in `src/` is `train_test_split` (`src/common.py:17`) |
| 3 | Configurable learning rate, convergence reported (1.7, 1.9) | pass | `lr` argument and `config.yaml p1.lr_fraction_of_bound`; sweep in `p1.lr_sweep`: 0.001/0.01 max_iter at 5000, 0.1 converged 2499, 0.5 converged 496, 0.9 converged 273, 1.05 diverged 93 (cell 17) |
| 4 | P1 produces weight vector and hyper-parameters (1.9, 1.10) | pass | `p1.json`: `weights`, `lr`, `iterations`, `tol_*`, `scaler` |
| 5 | P2 fitted with the team's own P1 GD code (1.7) | pass | `src/phases/p2.py:55` calls `p1.run_gd` -> `gd.gradient_descent`; no library optimiser |
| 6 | P2 starts from the P1 weight vector (1.9) | pass | `lift_weights` (`src/poly.py`); `p2.init_loss` = `p1.train_loss_final` = 0.3735811984758046, difference 0.0, asserted in `p2.py:50` and cell 38; every sweep fit starts there too |
| 7 | P2 self-chosen degree, selective expansion, produces degree and feature list (1.9) | pass | degree 3, `power_cols=[temp]`, block `wd_x_hr`, 61 names in `p2.feature_names`; justification cells 37, 40, 42 |
| 8 | P3 empirical diagnosis by varying complexity, train vs held-out (1.7, 1.9) | pass | 11-level ladder, degree axis, learning curves in `p3.json`; label `under-fit`, anchor gap 0.0028, gain 0.0235 [0.0177, 0.0298] |
| 9 | P3 consumes P2 degree and expansion (1.9) | pass | anchor spec from `p2.design_spec`; `check_anchor` asserts feature names and R2 (diff -3.9e-08); cell 54 |
| 10 | Exactly one chronological split, not seeded (1.8, 1.9) | pass | single `chrono.cut_date: 2012-07-01` (`common.chrono_split`); 8151 / 2735 rows; no RNG |
| 11 | Compare, explain the gap, which to trust, does it change the diagnosis (1.9) | pass | cells 62-64, 71; anchor 0.918 / 0.918 / 0.894, target 0.941 / 0.946 / 0.938; gap attributed to drift (mean ratio 1.08 / 1.06) |
| 12 | P3 produces diagnosis and target complexity (1.9, 1.10) | pass | `p3.target_complexity`: C6_weather_detail, 282 weights, degree 3 |
| 13 | P4 all three of L1, L2, Elastic Net implemented and compared (1.7) | pass | own solvers `src/regularization.py` (closed-form ridge, coordinate descent); `p4.methods`, `p4.comparisons`, stage B `p4.final` |
| 14 | lambda and l1_ratio search, informed by P3 (1.9) | pass | 40 log-spaced alphas per path, six l1_ratios; justification cell 74 |
| 15 | lambda per method, surviving L1 subset (1.9, 1.10) | pass, see W2 | `p4.methods.*.lambda`, `p4.final.methods.*.lambda`, `p4.survivors_expanded` (277) |
| 16 | Verdict for every input column with numbers (1.9) | pass | all 14 columns in `p4.column_verdicts`, each with drop-alone delta and interval, group drop, solo R2, Lasso counts and frequency; cell 91, report table |
| 17 | P5 target derived from cnt, fixes the global-threshold flaw (1.9) | pass | 75th percentile per (yr, workingday, hr), 96 cells from train rows; foils show hour-only AUC 0.855 -> 0.501, trend-only 0.810 -> 0.577 |
| 18 | P5 only Phase 4 survivors (1.7) | pass | `p5.features` subset of `p4.survivors_expanded` (277 = 277), asserted cell 105 and cell 127 |
| 19 | Accuracy, F1, ROC-AUC on validation (1.6) | pass | 0.784 / 0.659 / 0.885 at cut-off 0.25 (`p5.metrics`) |
| 20 | Pipeline retrospective (1.9) | pass | cells 119-120, report section 7 |
| 21 | Chaining of numeric artifacts (1.7) | pass | see D |
| 22 | Team seed formula (1.8) | pass | sorted ids `16006283_16007032_16009837` -> 44615; example -> 41698 (recomputed) |
| 23 | Single seeded `train_test_split(..., test_size=0.20, random_state=seed)` (1.8) | pass | `common.seeded_split` is the only call in `src/`; 8708 / 2178 |
| 24 | "No third split ... all evaluation and tuning uses the validation set" (1.8) | **at risk (R1)** | see below |
| 25 | test.csv never split, shuffled, tuned on or fitted on (1.8) | pass | read only in `common.load_test`, called from `src/predict.py` and cell 125; only `transform` is applied |
| 26 | Scalers, encoders, expansions fitted on train only (1.8) | pass | `Design.fit(train_df)` everywhere; fold and chronological fits refit their own design on their own fitting rows |
| 27 | Feature selection fitted on train only (1.8) | **at risk (R2)** | see below |
| 28 | Refit on train + validation justified (1.8) | pass | cell 122, report section 7; scaler and humidity fill stay as fitted on train |
| 29 | Expectation before, Outcome after, wrong expectations admitted (1.7) | pass, **R3** for the second pass | see B |
| 30 | Written justification for every open choice (1.7) | pass | 24 justification cells covering representation, target, learning rate, stopping, back-transform, expansion, degree, ladder, cut date, trust, lambda, fairness, verdict rule, survivor rule, label, metrics, threshold, refit |
| 31 | R2 and RMSE on validation for Phases 1-4 (1.6) | pass | notebook cell 120 gives all four; the report gives RMSE for P1, P2, P4 but not P3 (W6) |
| 32 | No external data (1.12) | pass | only `data/train.csv`, `data/test.csv`; nothing downloaded |
| 33 | Submission: columns, rows, order (1.11c) | pass | 574 rows, `instant,cnt`, order equals `test.csv` and the template, no NaN, no negatives |
| 34 | Submission from the recommended regression model (1.7) | pass | `p6.model` = `p4.recommended` (l1, lambda 0.00016960834781995135, l1_ratio 1.0) |
| 35 | Notebook runs top to bottom with all five phases (1.11a) | pass, **at risk (R4)** | 84 code cells, execution counts 1..84 in order, 0 error outputs |
| 36 | Report at most 6 pages, own numbers (1.11b) | pass | 5 pages; `report.pdf` and `report/report.pdf` identical |
| 37 | Bonus: asymmetric loss, gradient derived in the report, shift shown (1.9) | pass | `src/gd_asym.py`, report section 2, shift 1.28, k = 1 control |

### R1. Date folds inside the training portion (section 1.8)

- Letter: the folds are not seeded (`common.day_block_folds`: rank of date mod 5), so "the only seeded split" holds. The chronological split is a single cut. Neither breaks the letter.
- Spirit: "all your evaluation and tuning uses the validation set". Two uses go beyond a check:
  1. `src/phases/p4.py:348-355` (`recommend`): among methods within validation noise, the method is picked by held-out-day R2. This decides which model is submitted.
  2. Level C6 and the second-pass rejections were accepted on held-out days and the chronological split (cell 52, report section 7).
- Mitigation that is already true: `p4.recommended.best_on_validation` = `l1` = `p4.recommended.method`. A validation-only rule gives the same model, the same exponent (0.1 is the seeded best, `p3.target_power_best_seeded`), the same level and the same penalties. Nothing delivered depends on the folds.
- Inconsistency: cell 7 says the folds "are used only as a check", while cell 96 and cell 100 state the tie-break.

### R2. Feature selection decided on validation rows (section 1.8)

The surviving subset is set almost entirely by the column verdicts, which are validation drop tests; Lasso removes 3 of 280 features. Stage-B scores and Phase 5 scores are then reported on the same validation rows. The team reads this as tuning and says so plainly (cell 94, report section 5 and section 7). No statistic is fitted on validation rows. Defensible, but an examiner who reads "fit feature selection on your training portion only" strictly can object.

### R4. The notebook loads stored results by default (sections 1.11a, 1.7 Evaluation)

- Cells 11, 38, 53, 75, 104 print "loaded cached artifacts/pN.json". The phase code is a base64 zip in cell 1 (238 kB) and is shown through `inspect.getsource`.
- Only the Phase 1 gradient descent (cell 14) and the submission (cell 123) run live.
- Measured on this machine: Phase 1 runs in 6 s, Phase 2 in 3 s, Phase 3 in 15 s, the submission in 12 s, and all four rewrite their artifacts byte-identically (`git status --short artifacts` empty). There is no need to cache Phases 1-3.
- Risk: an examiner sees five phases whose outputs are read from files and whose code is in a blob. This is disclosed in cell 0, and `RECOMPUTE_ALL` exists, so it is a grade risk, not a rule break.

---

## B. The second pass

| Question | Verdict | Evidence |
|---|---|---|
| Is choosing the exponent from an earlier complete run within the rules? | pass | The exponent is a hyper-parameter chosen on validation; `test.csv` is not involved in any `src/` path before `predict.py`. Not verified by me: that the scripts in `exp/v2/` never read `test.csv` (I was not allowed to read `exp/`). |
| Disclosed plainly? | pass | Cells 9, 23 ("Honesty note"), 68, 34, 71; report line 3, section 2 ("This choice comes from the first pass"), section 7 item (1) |
| Second-pass Expectation cells before second-pass artifacts? | pass | Cells added in 3f64bce (2026-10-08 16:41); artifacts p1-p6 regenerated in 2d988eb (17:25) |
| Original Expectation cells unchanged since first commit? | pass | Byte-identical between first commit (2ef385f, 133ca8a, b5114d4) and HEAD for all five files; second-pass cells identical between 3f64bce and HEAD; all ten appear verbatim in the executed notebook; none contains a `{{placeholder}}` |
| Originals before first artifacts? | pass, narrowly | Expectation commits 14:33, 14:34, 14:38; first artifacts 165baa8 at 14:41 |
| Anything obviously after-the-fact? | **at risk (R3)** | see below |

### R3. Second-pass expectations match outcomes to the third decimal

Commit 3f64bce contains both the second-pass Expectation cells and the `exp/v2` exploration that had already fitted the power target. Predicted against delivered: Phase 1 "about 0.735" -> 0.7361; Phase 2 "about 0.918" -> 0.9178; Phase 4 "about 0.941 ... 0.946" -> 0.9413 / 0.9463; Phase 3 chronological "0.93 to 0.94" -> 0.938. Cell 9 names the exploration, but no cell says in one plain sentence that these numbers had already been seen. The one wrong prediction (back-transform factor, 1.041 -> 1.069) is admitted in cell 34. Cell 7 makes the equivalent disclosure for the first pass (pilot fits) but not for the second.

---

## C. Leakage

| Item | Verdict | Evidence |
|---|---|---|
| Scaler, humidity fill | pass | `Design.fit` on the fitting frame only; Phase 2 reuses the Phase 1 scaler from `p1.json` |
| Weather-memory columns (`ws_lag1`, `wet3`) | pass, disclosed | `common.add_weather_memory` reads only `dteday`, `hr`, `weathersit`; lags look backward only; built on the whole training file before the split, so a training row can carry the weather input of a validation row, never its count. For `test.csv` the timeline is train inputs plus test inputs. Disclosed in cell 52 and report section 7 |
| Date folds | pass | each fold refits its design and back-transform factor on its own fitting rows (`p4.build_split`, `validation.day_block_scores`) |
| Chronological split | pass | design and factor fitted on the early rows only |
| Refit on train + validation | pass | weights and factor from the 10886 labelled rows; scaler from train; lambda as validated; model A reproduces Phase 4 weights with gap 0.0 |
| Phase 5 thresholds | pass | `labels.fit_thresholds(train_df, ...)`; validation labels are looked up; a missing cell raises |
| Back-transform factor | pass | `Target.factor` from fitting rows; method chosen on validation (tuning) |
| Hyper-parameters on validation | pass | exponent, back-transform, degree, level, penalties, l2 of the classifier; the classification cut-off is not tuned (0.25 from the cost ratio; empirical optimum 0.2 is reported and not used) |
| Bundle in cell 1 | pass | 32 members: `config.yaml`, `src/**`, `artifacts/p1..p6.json`, `artifacts/bonus_control.json`, the empty `data/sample_submission.csv`. No data file. Every member equals the file on disk. `p6.json` holds summary statistics of the test predictions only |

---

## D. Reproducibility and chain

| Check | Result |
|---|---|
| `python -m src.phases.p1` | 6 s, `git status --short artifacts` empty |
| `python -m src.phases.p2`, `p3`, `src.predict` | 3 s, 15 s, 12 s; artifacts and `sample_submission.csv` unchanged |
| Seed | 44615 from the config ids; 41698 for the example ids |
| `upstream_sha256` | p2<-p1, p3<-p2, p4<-p3, p5<-p4, p6<-p5 all equal the sha256 of the upstream file; p1 has none |
| `config_sha256`, `seed` | identical in p1..p6, equal to the current `config.yaml` and 44615 |
| Chain assertions executed in the notebook | cell 38 (loss hand-over), 54 (anchor), 81 (hash and level), 105 (subset), 127 (13 assertions, all True, followed by an `assert`) |
| Live Phase 1 check | cell 14: 496 iterations, largest weight difference 0.0 |
| Submission against `p6.json` | mean 199.3179, min 0.94, max 884.93, 574 rows: equal |
| Submission against `p4.json` | method, lambda, l1_ratio equal to `p4.recommended` |
| `.gitattributes` | `* -text`, so the config hash does not depend on the checkout |

Note: during this audit `docs/WALKTHROUGH.md` became modified in the working tree (mtime 17:36:41, +155 / -86). No command I ran references that file (`grep WALKTHROUGH src run.py tools` is empty), and I did not open or restore it. Something else was writing to the repository at the same time.

---

## E. Numbers in `report/report.md`

About 150 numbers checked against `p1..p6.json`, `first_pass.json`, `diagnostics.json`, `bonus_control.json`, across every section: section 2 (lr, lambda_max, sweep iterations, exponent table, back-transform, interval, bonus and control), section 3 (loss, ablation, degree and column sweeps, iterations, gain and interval), section 4 (all 28 ladder cells, gaps, learning curve, time question, exponent check), section 5 (both tables in full: 4 method rows, 14 verdict rows; comparisons; survivors; stage B), section 6 (class balance, foils, both metric rows, interval, confusion counts), section 7 (retrospective, submission checks, exploration numbers).

Mismatches:

1. **Wrong.** Line 104: "the cost is lowest at a cut-off of 0.2, the theoretical value". The theoretical value is 0.25 (stated two lines above); 0.2 is `p5.t_cost_empirical`. Source: `report/report.template.md`, "`{{p5.t_cost_empirical}}`, the theoretical value". The notebook has it right (cell 121: "lowest at 0.2, next to the theoretical 0.25").
2. Loose. Line 53: "The chronological split would prefer 0.15 (0.941 against 0.938)". The chronological best is 0.2 (0.94116) ahead of 0.15 (0.94110). Cell 68 lists both.
3. Loose. Line 49: "from level 10 on, train R2 keeps rising while all three held-out scores fall". The chronological score already falls at C7 (0.928) and C8 (0.904); line 51 says so.

Everything else agrees to the printed precision.

---

## F. Over-claims and inconsistencies

| # | Where | Issue |
|---|---|---|
| F1 | cell 7 against cells 96, 100 and `p4.py:348` | Folds "used only as a check" against the held-out-day tie-break (R1) |
| F2 | report line 104 | "0.2, the theoretical value" (E1) |
| F3 | report section 7 table, cell 100, `p4.final.methods.l1.n_nonzero` | The recommended Lasso has 270 non-zero weights; the report gives 278 weights and 277 survivors, and Phase 5 uses 277. Nowhere stated |
| F4 | report line 3 | "Every number below is read from our own artifacts; the notebook reproduces them". `first_pass.json` and `diagnostics.json` are hand-assembled summaries of the tagged first run and of `exp/v2` (the file's own note: "nothing reads this file"); the notebook does not recompute them |
| F5 | cell 7 | "`test.csv` is read once, at the very end", in the cell that also states that `test.csv` holds the 20th of every month. The dates were looked at during data inspection |
| F6 | cell 120 output, `src/phases/p5.py:265` | "P3 target design + 6 candidate columns"; five were added, `c_weekday` was skipped (cell 76) |
| F7 | cell 97 output | "unregularised ... 'alpha': 0.0" is 1e-08 rounded |
| F8 | artifact keys | `train_r2_log`, `val_r2_log` now hold R2 on the power scale; cell 25 labels them correctly |

Stale log-target mentions presented as current: none found. Every remaining mention of the log is either the original Expectation cell (kept unchanged on purpose), a first-pass comparison, or the lambda -> 0 limit.

Candour worth keeping: cells 23, 52, 70, 94, 96 and report section 7 state every rule or design that was changed after seeing results.

---

## G. Judgment and ordered fixes

The work implements every requirement of the description and breaks no rule outright. The engineering is unusually well checked (hash chain, byte-identical re-runs, admitted wrong expectations). The risks are about how a strict examiner reads section 1.8 and the Expectation rule, and about a notebook that shows stored results.

### Breaks a rule of the description

None.

### At risk under a strict reading (smallest fix first)

1. **R5 / E1, report line 104.** In `report/report.template.md` replace ", the theoretical value" with " on our grid, next to the theoretical 0.25". Rebuild the PDF. No number changes.
2. **R1, cell 7 and report section 1.** Replace "used only as a check" with the truth: the folds also break ties between methods that are within noise on validation, and add "the method picked is also the best on validation (Lasso), so no choice behind the submission depends on the folds". Prose only; `p4.recommended.best_on_validation` backs it.
3. **R3, cell 7 (not the Expectation cells, which must stay unedited).** Add one sentence: the second-pass expectations were written after an exploration (`exp/v2`) that had already fitted the power target, so their numbers are forecasts from results we had seen; the commit order is 3f64bce then 2d988eb.
4. **R2, report section 5.** One sentence mapping to the rule: Lasso weights and selection frequencies are fitted on training rows only; the keep-or-drop threshold is tuned on validation; stage-B validation scores are therefore selected and evaluated on the same rows. Most of this is already in cell 94.
5. **R4, cell 2 `phase()`.** Run Phases 1-3 live always (24 s in total, byte-identical artifacts, so the hash chain and every number stay valid) and keep the cache for Phases 4 and 5 only. Update the "Runtime" paragraph of cell 0 and re-execute the notebook. Larger alternative: replace the base64 blob by visible `%%writefile` cells.

### Weakens the grade

- W1 (F3). State that the recommended Lasso keeps 270 of the 277 surviving features.
- W2. The selection Lasso sits at the lower edge of its grid (`p4.selection.lambda` 8.35e-05 = alpha_max x 1e-4) and stage-B Ridge at its grid minimum (1e-06). Already read as "no penalty wanted" in cells 74 and 100; say the same for the selection Lasso in the report.
- W3 (F4). Soften report line 3: first-pass and exploration numbers come from `first_pass.json` and `diagnostics.json`, which the notebook does not recompute.
- W4 (F5). Reword to "the pipeline reads `test.csv` once, at the end; we looked at its dates only when inspecting the data".
- W5. Confirm with `grep -rn "load_test\|test.csv" exp/` that no exploration script reads the test file (not checked here).
- W6. Add the Phase 3 validation RMSE (44.7) to the report's retrospective row.

### Cosmetic

- E2, E3: "would prefer 0.15 to 0.2"; "the chronological score falls from C7 on".
- F6: the "6 candidate columns" string in `src/phases/p5.py:265` (changing it needs Phase 5 and the submission step re-run, about 3 minutes; leaving it is acceptable).
- F7, F8.
- The Expectation cells are long for "a short markdown cell".
