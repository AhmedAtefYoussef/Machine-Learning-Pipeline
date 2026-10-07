# SPEC (chief). Requirement → acceptance criterion. Authority: Project_1_description.md. Checks run through `python run.py verify-fast|verify`.

| R | Requirement | Acceptance criterion (test / tool / verifier question) |
|---|---|---|
| R1 | Five chained phases in order | AC-1 `artifacts/p1..p5.json` exist; notebook has nb/p1..p5 sections in order (build_nb). |
| R2 | Each phase consumes the previous artifact | AC-2 chain_check: hash chain, P2 init_loss == P1 train_loss_final (1e-9), init_weights_source == "p1", P5 features ⊆ P4 survivors; p3.anchor equals p2 degree/features; p4.design_from equals p3.target_complexity. |
| R3 | Expectation before / Outcome after / justification per open choice | AC-3 build_nb structure check; git log shows each nb/pN Expectation committed before artifacts/pN.json; verifier: every open choice has a Justification cell. |
| R4 | Team seed, one seeded split, one chronological split | AC-4 tests/test_common.py seed + split; `rg "train_test_split\(" src nb` → exactly one call (src/common.py); chain_check seed recompute. |
| R5 | Fits on train only; test only transformed at the end | AC-5 leak_scan clean; tests/test_features.py leakage tests; test file read only in src/common.load_test, used only by src/predict.py. |
| R6 | No external data | AC-6 verifier: only data/*.csv read; no network calls. |
| R7 | P1 GD from scratch, configurable lr, convergence shown | AC-7 leak_scan L3 clean; tests/test_gd.py gradient check < 1e-6, oracle < 1e-3; p1.json has lr, iterations, stop_reason == "converged", lr_sweep with ≥ 1 diverged and ≥ 1 max_iter entry. |
| R8 | P2 expansion fitted with own GD from P1 weights | AC-8 AC-2 init-loss equality; p2.json degree, feature_names, ablation table; val_r2(P2) ≥ val_r2(P1). |
| R9 | P3 empirical diagnosis + chronological comparison | AC-9 p3.json: ladder with train and held-out scores on three validators, degree axis, learning curve, bootstrap interval, diagnosis, target_complexity, trust statement. |
| R10 | L1, L2, ENet implemented, compared; verdict per column | AC-10 tests/test_regularization.py oracle 1e-3; p4.json methods {l1,l2,enet} with lambda (and l1_ratio); verdict + numeric evidence for all 14 columns (chain_check). |
| R11 | P5 label, logistic on survivors, acc/F1/AUC, retrospective | AC-11 chain_check subset; p5.json threshold_rule from train, metrics, foil comparison, retrospective table. |
| R12 | Bonus asymmetric loss | AC-12 tests/test_gd_asym.py gradient check; p1.json.bonus with costs and prediction shift; derivation in report. |
| R13 | R2 + RMSE on validation (P1-P4); acc/F1/AUC (P5) | AC-13 keys val_r2, val_rmse in p1, p2, p4 (per method), p3 estimates; p5 metrics. |
| R14 | Submission from the recommended regression model | AC-14 submission_check passes; p6.json.model == p4.recommended. |
| R15 | Notebook runs top to bottom; report ≤ 6 pages; explainable | AC-15 `run.py nb` errors=0; PDF page count ≤ 6; number_trace check passes; docs/WALKTHROUGH.md exists. |

Hard constraints H1-H13: see CLAUDE.md. Decisions: docs/adr/. Interfaces: docs/ARCHITECTURE.md. Readings of unclear points: handoff/AMBIGUITIES.md.
