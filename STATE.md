# STATE (chief; resume from here, not from a transcript)

seed 44615 (IDs 16007032, 16009837, 16006283; confirmed by the user as the registered roster on 2026-10-07) · deadline 18 Oct 2026
env: Windows, `python` 3.14, no make/pandoc → `python run.py <target>` (all | p1..p5 | submission | nb | report | verify-fast | verify | fingerprint). Phase 4 ≈ 20 min; the notebook loads the stored p4/p5 artifacts when config and upstream are unchanged.

## Deliverables (repo root)
`rush_hour.ipynb` (129 cells, executed, 0 errors; self-contained: runs on Colab with only train.csv and test.csv; cells call `src/nbview.py` tables and `src/plots*.py` figures styled by `src/plotstyle.py`, helpers in `src/nbtools.py`, collapsed setup cell via `tools/polish_nb.py`) · `report.pdf` = `report/report.pdf` (5 pages; describes the finalized pipeline only, at the user's request; source report/report.template.md, numbers traced) · `sample_submission.csv` (574 rows) · `docs/WALKTHROUGH.md`

## Two passes
- First pass (tag `v1-submitted`): log1p target. Key numbers kept in `artifacts/first_pass.json`.
- Second pass (branch `v2/ceiling`, ADR-017): target = power transform with exponent 0.1, set in `config.yaml` (`p1.target_power`), carried by `common.Target`; everything else unchanged. Exploration E1-E5 in `exp/v2/` (ADR-018 lists what was measured and rejected); diagnostic numbers quoted by the report are in `artifacts/diagnostics.json`.

## Phase status (numbers live in artifacts/pN.json; do not copy them here by hand)
| phase | status |
|---|---|
| 1 GD | second pass done; exponent table in p1.json |
| 2 Polynomial | second pass done |
| 3 Bias-variance | second pass done; target C6_weather_detail; exponent check consistent |
| 4 Regularization | second pass done; stage A (verdicts) -> selection (ADR-016) -> stage B (ADR-014); recommended Lasso |
| 5 Logistic | second pass done |
| 6 Submission, notebook, report | rebuilt; `python run.py verify` (strict) passes |
| independent audit, second pass | PASS WITH FIXES, no rule of the description broken; fixes applied (reports/verify/final-audit-v2.md) |
| verifier gate, second pass | 39 pass, 1 fail fixed (notebook message), 15/15 requirements verified (reports/verify/gate-final-v2.json) |

## Rule and design changes made after first results (all disclosed in notebook and report)
ADR-013 (P3 target: plateau rule instead of argmax) · ADR-014 (final model on survivors, stage B) · ADR-015 (one new ladder level after residual analysis) · ADR-016 (L1 selection after removing redundant columns) · ADR-017 (power target instead of log, second pass)

## Known limits, disclosed
validation scores optimistic (tuned on the same rows) · some Lasso grid points on the full design hit the sweep cap (none chosen) · eta cap in validation.fit_predict (inert for all reported fits) · trend extrapolation · no zero-demand hours in the data · stage-B Ridge optimum at the grid floor (no penalty wanted)

## Definition of done
- [x] D1 · [x] D2 (leak_scan 0; tests) · [x] D3 · [x] D4 (chain_check 5/5; p6 upstream ok) · [x] D5 · [x] D6 · [x] D7 · [x] D8 (0 TODO stubs; Expectation commits precede artifacts) · [x] D9 (run.py nb errors=0) · [x] D10 (5 pages; number_trace 0 untraced) · [x] D11 (submission_check 0 failures, 0 warnings) · [x] D12 bonus (p1.bonus) · [x] D13 WALKTHROUGH · [x] D14 · [x] D15 TRACE verified + `run.py verify` strict · [x] D16 reports/analysis/audit.md (stop condition b after ADR-015) · [x] D17 verifier H1–H13 · [x] D18 TOKENS (conclusion from the ledger; explain-usage skill not run) · [x] D19 final audit (its fix 1, roster confirmation, done 2026-10-07)

## Open for the user
1. Roster IDs confirmed on 2026-10-07.
2. The notebook is self-contained (setup cell); on Colab upload the notebook, train.csv and test.csv.
3. Rehearse with docs/WALKTHROUGH.md; the git history shows how the work was produced, including both passes.
