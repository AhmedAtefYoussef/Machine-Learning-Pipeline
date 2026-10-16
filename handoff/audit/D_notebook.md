# Audit D: notebook simplicity and deliverable consistency (read-only; 2026-10-16)

Scope: nb/p0..p6, tools/build_nb|polish_nb|embed_bundle, src/nbview|nbtools|plots*, report template, README, WALKTHROUGH, STATE, brief 1.7/1.11.

## 1. Measured state (rush_hour.ipynb at HEAD 66b80ac)
- 129 cells = 84 code + 45 md; file 2.4 MB; 0 errors; 0 unrendered `{{`.
- Code cells: median 2 lines; only the setup cell is big (2,202 lines, 260 KB of base64). Code is short, but the logic is hidden.
- Calls: 66 `view.*` tables (63 HTML outputs), 26 `show(plots.*)` PNGs (~60-90 KB each), 6 `show_source` dumps.
- Markdown: 11.8k words (about 25 pages). Expectation 2,885 (10 cells) · Justification 4,580 (23) · Outcome 2,555 (5) · other 1,803 (7).
- Output weight per phase (KB): P0 70 · P1 333 · P2 153 · P3 353 · P4 544 · P5 310 · P6 144.
- show_source outputs break the skill's 30-line rule: Target 72, Ridge/CD 46, GD 46, logistic 34 lines.
- H9 check (git script over every commit of nb/p1..p5): all 10 Expectation cells are byte-identical to their first commit, and every one was committed before its phase's first artifact (P3 14:34 < p3.json 14:41; P4/P5 14:38 < 15:31/15:55). PASS.
- Brief 1.11 (submission): root `sample_submission.csv` has 574 rows, columns `instant,cnt`, instant order equal to data/test.csv, min 0.94, no NaN. PASS. `data/sample_submission.csv` is the untouched all-zero template (config `paths.sample_submission`). Report: 5 pages (≤6), and root report.pdf is byte-identical to report/report.pdf. PASS.

## 2. Simplicity findings (ranked by how much each hurts H12 "explain and re-run any cell")
S1 Logic hidden behind 71 `nbview` functions (697 lines). A cell like `view.p4_verdicts(p4)` cannot be explained without opening src/. Each function is only a column pick, a rename and a highlight over `styled()` (29 lines of Styler code).
S2 Chain assertions sit inside `view.p4_chain_rows / p5_chain_rows / p6_chain_rows` (6-21 lines each), so the most-graded checks (H5) are not visible in the cell.
S3 `nbtools.phase()` is opaque. It covers live/cached/recompute, rewrites the artifact and restores the stored bytes in a `finally` block, then compares the two with a recursive `largest_difference`. It takes too long to explain orally. Its WALKTHROUGH description alone is 150 words.
S4 Duplicated work: P1 GD runs twice per notebook run, once in `phase("p1", live=True)` and again in `p1_live_run` (which reaches into the private helpers of p1mod). `bonus_control.run(CFG)` rewrites artifacts/bonus_control.json on every run.
S5 Too many outputs: tables and plots duplicate each other (the same data shown both ways) for the lr sweep, degree sweep, ladder, degree axis, learning curves, three estimates, label variants, verdicts and target power (8+ pairs). The skill asks for one plot per question. Other repeats: the loss curve repeats the lr sweep; the P1-only residual plot is repeated by the P1-vs-P2 plot; ridge paths, stability, cv-vs-val, stage-B curves, calibration plot and the pipeline-summary plot add little.
S6 Duplicated prose. The Phase 5 retrospective table is shown again in Phase 6 ("the same table", as the notebook itself says). The label-rule justification restates the P5 Expectation almost word for word. The "what we varied" and "time question" text is repeated across the P3 Justifications and the P3 Outcome. The honesty notes about pilot fits are in P0 and again in several Justifications.
S7 Markdown volume: Justifications average 200 words. The brief asks for a "short justification" and a "short" Expectation. Expectations are frozen (H9) and must stay verbatim, so all the savings have to come from Justification, Outcome and P0 prose.
S8 The setup cell (about 100 visible lines of upload/sha logic plus 260 KB base64) is the only long code cell and the hardest one to explain. It is needed for Colab, but its upload logic (Colab rename matching, sha check) can be cut to about 25 lines.
S9 The P1 code cell imports 6 names, some only for `show_source`. Imports are scattered across phases (fine), but some are unused in the cell that imports them (`bonus_control`, `asym_*`).

## 3. Consistency findings (stale numbers and claims)
C1 README "tests/ 256 tests" vs `pytest --collect-only`: 313 collected (163 test functions). Stale.
C2 Phase 4 runtime: STATE says "≈ 20 min", while README, WALKTHROUGH, p0 intro and p4 comment say "about 30". Notebook runtime: p0 says "about two minutes", README says "about 1 min", WALKTHROUGH says "a few minutes". Pick one value, measured.
C3 WALKTHROUGH l.54 says "the five phase cells print 'loaded cached artifacts/...'". Since 3388160, P1-P3 print "computed live just now". Stale.
C4 WALKTHROUGH l.47 says "run.py decides what to rebuild by file time". `run.py stale()` uses the config/upstream content hash plus source mtime. Half stale.
C5 STATE "129 cells" is true now but will go stale with any restructure. Drop hard-coded cell counts from STATE/README.
C6 The P3 Outcome hard-types diagnostic numbers (0.966, 0.950, about 0.90, 0.004, 0.003, 0.05, 0.001, "twenty"). The values agree with artifacts/diagnostics.json but are not `{{diagnostics.*}}` placeholders, so number_trace cannot catch drift. Same in P3 Justification ("about half and a fifth"). The report quotes diagnostics through placeholders, so the notebook is the weaker of the two.
C7 The P4 Expectation lists `weekday` among the six held-back columns, but the Justification says `weekday` is already in the target. The Expectation is frozen, so this is fine; the Outcome should state the discrepancy in one line (it does not).
C8 Notebook vs report scope. The report "describes the finalized pipeline only", while the notebook carries both passes. Consistent by design; README and STATE say so. OK.
C9 README results table (0.736/94.7, 0.918/52.8, 282 w, 0.941/0.946/0.938, 0.941/44.7, 0.784/0.659/0.885) and the first-pass numbers match the artifacts. OK, but the values are typed by hand. Prefer "see report/artifacts" or render README from a template.
C10 The WALKTHROUGH cites cells by position ("the cell after the validation curves", "last cell of Phase 6"), and TRACE.csv has 6 notebook references. Both must be re-pointed after any restructure.
C11 Root vs data/ submission: correct, but README l.96 "data/ ... sample_submission.csv (as provided)" sits next to the root file. Add one line: "upload the ROOT sample_submission.csv; data/ holds the empty template".

## 4. Target notebook outline (about 70 cells / about 42 code / about 13 plots / about 20 tables / about 6.5k md words)
Principles: Expectation cells stay byte-identical (both passes). Code cells are 3-10 lines with a 1-line comment saying why. Tables are built inline from the artifact (`pd.DataFrame(p1["lr_sweep"])[cols]`) through one `table()` helper. Chain checks are plain `assert`s in the cell. There is one plot per question, and show_source is kept for 4 core algorithms only, each ≤30 lines.
Format: [type] purpose (approx lines | md words)

P0 Setup (7 cells)
- [md] title, how to run on Colab and locally, runtime, how to change a knob (≤150 w; move the drill text to WALKTHROUGH)
- [code] setup: unpack bundle, place/upload data, sha check (collapsed; about 25 visible lines + base64)
- [code] config: load CFG, RECOMPUTE_ALL, SEED = config_seed, brief self-test assert, print the key knobs (about 10)
- [code] load train, seeded split, print sizes (4)
- [code] plot: demand profile by hour × day type (1)
- [code] table: data quirks (hum=0 day, ws=0 share, weathersit 4, atemp/temp r) (3)
- [md] what we saw + splits + honesty note about the pilot fits, said once (≤220 w)

P1 GD (14)
- [md] Expectation (orig, verbatim) · [md] Expectation 2nd pass (verbatim)
- [md] J representation + target exponent (≤160 w; the hour-encoding numbers go inline)
- [code] p1 = phase("p1", ...) and print design size, λ_max, bound, lr, iterations (5)
- [code] spot-check: GD from zeros with the stored settings, assert allclose to p1 weights, show_source(gradient_descent) (6). This replaces p1_live_run and the duplicate live run.
- [md] J learning rate + stopping rule (≤120 w)
- [code] lr sweep: table (fraction, iterations, stop) + one loss-curve plot (4)
- [code] correctness: oracle diff, gradient check, train/val R², RMSE, bootstrap CI in one table (4)
- [md] J back-transform (≤100 w)
- [code] back-transform + exponent table (one table, no plot) (4)
- [code] plot: residuals by hour × day type (motivates P2) (1)
- [md] J bonus: loss, gradient, optimiser, k=1 control explained (≤170 w)
- [code] bonus: one table with MSE model, k=1 control, asymmetric model (under-share, mean shift, RMSE, cost) (5)
- [md] Outcome P1 (≤250 w, both passes)

P2 Polynomial (7)
- [md] Expectation ×2 (verbatim)
- [md] J init + what we expanded + degree rule (≤180 w)
- [code] p2 = phase("p2", ...); assert init_loss == p1 final loss (1e-9); print n features and iterations; show_source(lift_weights) (6)
- [code] one table: degree sweep + power-column sweep + wd×hr ablation (val R²) (5)
- [code] plot: residual profile P1 vs P2 (1)
- [md] Outcome P2 (≤200 w)

P3 Bias-variance (12)
- [md] Expectation ×2 (verbatim)
- [md] J ladder, degree axis, learning curves, fitting in closed form, the C6 note, chrono cut (≤230 w)
- [code] p3 = phase("p3", ...); assert anchor == p2 design and anchor R² reproduces p2 (4)
- [code] ladder: table (level, weights, train, val, days, chrono) + plot (3)
- [code] table: degree axis (2)
- [code] plot: learning curves (anchor, target, top) (1)
- [code] table: three estimates + leakage/drift gaps for anchor and target (3)
- [md] J which estimate to trust + target-complexity rule (ADR-013 disclosure) (≤180 w)
- [code] table: exponent check (seeded/days/chrono) + consistency flag (3)
- [code] table: diagnosis + target handed to P4 (2)
- [md] Outcome P3 (≤300 w; diagnostics through `{{diagnostics.*}}` placeholders)

P4 Regularization (13)
- [md] Expectation ×2 (verbatim)
- [md] J design, objective, grids, fair comparison, λ rule, non-converged note (≤200 w)
- [code] p4 = phase("p4", ...); assert p4 design_from == p3 target + upstream hash; show_source(soft_threshold, cd_sweep) (6)
- [code] plot: validation curves stage A + table: methods incl. unpenalised row (3)
- [code] table: paired differences + λ picks (val vs days vs 1-SE) (3)
- [code] plot: lasso coefficient path (1)
- [md] J verdict rule (≤130 w)
- [code] verdicts: table + drop-test plot (3)
- [md] J survivor rule + stage B + tie-break (ADR-014/016) (≤170 w)
- [code] one table: survivor counts + stage-B methods + recommended row (4)
- [md] Outcome P4 (≤300 w; mention the rich-level check in one line instead of its own cell)

P5 Logistic (13)
- [md] Expectation ×2 (verbatim)
- [md] J label rule (≤120 w; do not restate the Expectation)
- [code] p5 = phase("p5", ...); assert set(features) <= set(p4 survivors); thresholds and class balance in one table (5)
- [code] table: label variants (global / wd×hr / ours: AUC hour-only, trend-only, full) (2)
- [md] J l2, step size, metrics, cost cut-off (≤170 w)
- [code] table: l2 sweep + fit stop; then metrics at 0.25 / 0.5 / F1-opt with confusion counts (4)
- [code] plots: threshold-cost curve + ROC/PR (2)
- [code] table: top coefficients + column share (2)
- [md] Pipeline retrospective (≤150 w; required by brief 1.10)
- [code] table: retrospective, built from p1..p5 (2)
- [md] Outcome P5 (≤220 w)

P6 Submission (5) (no Expectation/Outcome needed; build_nb checks p1-p5 only)
- [md] model, refit rationale, what we check without labels, limits (≤150 w)
- [code] p6 = predict.run(CFG); print model, n_fit_rows, A-vs-B corr (3)
- [code] submission asserts inline (columns, 574 rows, instant order, finite, ≥0) (7)
- [code] chain asserts inline: hash chain p1→p6, one seed, init_loss, subset, model A gap (10)
- [code] plot: test-day hourly profile vs training (2). Drop pipeline_summary table and plot (duplicate of the P5 retrospective).

Removed or merged cells: p1_design+live_check; p1_fit+loss_curve; p1 target-power plot; p1_scores; hour-encoding table (now inline in J); bonus-shift plot; asym show_source; Target show_source; p2_design; p2_degree plot; p2_power_cols; p2_ablation; p2_fit; p3_chain; degree-axis plot; estimates plot; chrono_detail; noise_floor (into Outcome); p3 exponent plot; p4_design; p4_unregularised; enet_ratios; ridge show_source; cv-vs-val plot; ridge path; p4_entry; stability plot; stage-B 4 tables → 1; stage-B curves; p4_rich; p5_design; label-variants plot; p5_threshold table; calibration table + plot (into Outcome: max gap); p6_summary; pipeline_summary table + plot.

## 5. Helpers: inline vs keep
Keep in src/ (the pipeline; tested; the chain needs them): common, features, gd, gd_asym, poly, validation, regularization, verdicts, labels, logistic, phases/p1..p5, predict, bonus_control.
Keep, smaller: plots*.py + plotstyle, down to the about 13 functions used (delete about 15 unused: loss_curve, target_power, bonus_shift, degree_axis, three_estimates, cv_vs_validation, stability, label_variants, calibration, pipeline_summary, ...). Matplotlib code inline would be bulky.
Replace: nbview.py (71 fns, 697 lines) → one `table(df, caption, highlight=None)` in nbtools (about 15 lines) plus inline column picks in the cells. *_chain_rows → inline asserts.
Simplify: `phase()` to about 12 lines: `if cached and not RECOMPUTE_ALL: load; else run`. For live P1-P3, give run(cfg, write=False) a mode that returns the payload without touching disk. Then the restore-in-finally and `largest_difference` can go (or `largest_difference` shrinks to a numeric max over the flattened float leaves, ≤8 lines). Note: the live P1-P3 runs came from the 2nd-audit fix (3388160); keep the live behaviour and drop only the mechanism.
Inline: p1_live_run (6 visible lines using public p1mod functions; make `base_design`, `learning_rate`, `run_gd` public names). Drop: `_same_inputs`, `_read_json`.
show_source: keep for gradient_descent+mse_grad, lift_weights, soft_threshold+cd_sweep, logloss_grad+fit_logistic. Drop Target, asym_*, ridge_closed_form, enet_cd_gram from the notebook and point to them in WALKTHROUGH.

## 6. Required follow-ups if the outline is adopted
- Keep the Expectation text byte-identical. Add a check to verify-fast: extract the Expectation blocks and compare them with their first git commit (the script used in this audit is about 20 lines).
- Re-point WALKTHROUGH cell references, docs/TRACE.csv notebook locations, the STATE deliverables line and the README runtimes. Fix C1-C4 and C6 regardless of the restructure.
- Re-run `run.py verify` (nb errors=0, number_trace, standalone_check on Colab-like empty dir) and `run.py fingerprint` (artifacts must be unchanged: the restructure is presentation only). Then a verifier gate.
- Risk: medium (prose rewrite of Justification and Outcome cells, about 4k words). No metric changes; the chain and the submission are untouched.
