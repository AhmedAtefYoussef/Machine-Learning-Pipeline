# Audit A: architecture, modularity, engineering principles, code efficiency (read-only pass, HEAD 66b80ac)

Scope: run.py, Makefile, src/**, tools/*, exp/* (imports only). Nothing in src/ nb/ tests/ tools/ edited.
Measurements: scratchpad scripts only (no artifact written). BP = behavior-preserving (artifact bytes unchanged).
Rebuild cost rule (run.py `stale`): editing ANY dep file by mtime marks that phase and everything after it stale.
common.py / features.py edit -> P1..P5 rebuild (~25-30 min). regularization.py or phases/p4.py -> P4+P5 (~22 min).
nb-layer modules (nbview, nbtools, plots*, plotstyle) are NOT run.py deps: editing them costs one `run.py nb` only.

## Findings (ranked)

A1 HIGH. Staleness is mtime-based, so every fresh clone/worktree rebuilds the whole chain.
- run.py:66-81 `stale()`: hashes for config/upstream, but `mtime(src) > mtime(artifact)` for sources.
- Measured in this worktree: config and upstream hashes all match, yet `stale(n)` is True for n=1..5 only because
  git checkout wrote src/phases/p1..p4.py, logistic.py after artifacts/*.json.
- Impact: teammate/grader runs `python run.py all` after cloning and it rebuilds about 30 min of work for nothing.
  Artifacts would come out identical (deterministic), but it costs a lot of time, and a phase built on a different
  machine can differ in the last digit, which breaks the "stored file kept" logic in the notebook.
- Fix: sidecar `artifacts/.build_manifest.json` {phase: {path: sha256}} written by build_phase after a successful
  build; `stale()` compares content hashes, not mtimes. Seed it once from HEAD (artifacts already verified against
  HEAD). Not p?.json, so chain_check/fingerprint unaffected. +20 lines run.py.
- BP: yes (no artifact touched). H5: strengthens it. Do before deadline.

A2 HIGH. run.py dependency lists are incomplete (silent staleness the other way).
- run.py:20-27 PHASE_DEPS vs real imports:
  p1 imports src/gd_asym.py (bonus in p1.json) -> missing.
  p2 imports src/phases/p1.py helpers (p2.py:19) and gd.py (p2.py:18) -> missing.
  p4 imports src/verdicts.py (p4.py:21) and src/phases/p3.py `ladder_specs` (p4.py:25) -> missing.
  p5 imports src/labels.py, src/gd.py (p5.py:23-24) -> missing.
  p6/predict imports phases/p4.py constants + regularization.py (predict.py:23-24): no staleness check at all
  (only p6_chain hash catches it after the fact).
- Hash chaining does not catch it: if p3.py changes `ladder_specs` but p3.json bytes stay the same, p4 is not
  rebuilt and the chain still checks out.
- Fix: derive deps as the transitive closure of `src.*` imports (ast, ~15 lines) instead of hand lists; feed it
  into A1's manifest. Makefile has the same gap (frozen kit: leave it, see A12).
- BP: yes. Combined with A1 there is no spurious rebuild. If done WITHOUT A1, adding p1.py/p3.py marks p2/p4 stale
  by mtime -> a 22-min rebuild. Run the `stale()` dry-run first.

A3 MED. `run.py nb` never builds phases first, and two other cache checks ignore source changes.
- run.py:153-162 renders `{{...}}` from whatever artifacts are on disk, then executes.
- common.py:322 `cached_or_run` and nbtools.py:46 `_same_inputs` check only config + upstream hash, so after a
  p4.py edit the notebook loads the old p4.json and prints "loaded cached" (leaky abstraction: three staleness
  definitions, run.stale / cached_or_run / _same_inputs, all slightly different).
- bonus_control.json is in no target graph: it is built only by the notebook cell (nb/p1_gd.py:185), but its
  placeholders (nb/p1_gd.py:173-179) are rendered BEFORE execution -> stale prose after a P1 change until a 2nd build.
- Fix: nb(): `build_phase(5)`, then `python -m src.bonus_control` if stale, then render (+6 lines). Keep one
  `is_current(name, cfg, upstream)` in common.py used by cached_or_run and nbtools (-10 lines, later).
- BP: yes for the run.py part (no-op when up to date). The common.py part triggers a full rebuild -> defer.

A4 MED. exp/ libraries are broken at HEAD and duplicate src.
- exp/analyst/lib.py:13-16 and exp/loss_scale/diag.py:14 import `back_factor, from_target, to_target`
  (removed from common.py) and `validation.ETA_HEADROOM` (gone). Verified: the attributes do not exist ->
  ImportError. ceiling.py, hyp.py, resid.py, diag.py cannot be re-run at HEAD (H12: "re-run any cell").
- exp/v2/lib.py:38-60 re-implements Target.forward / q / ls factor / eta cap (cap applied to q, not eta: same
  result) and fit_linear. It is diagnostic only and works, but it is a second copy of the target maths.
- exp/eda0/eda.py:51, eda1.py:15 call train_test_split directly (same seed, so the same split, but it is a second
  seeded-split call site; leak_scan does not scan exp/).
- Fix: README line in exp/analyst, exp/loss_scale: "first pass (log1p API, tag v1-submitted); not runnable at
  HEAD; results kept in *.json" (or `git checkout v1-submitted` to re-run). exp/v2/lib: use common.Target (-15).
  Do not re-run any logged experiment.
- BP: yes (exp/ is outside the chain). H13 OK.

A5 MED. ARCHITECTURE.md describes APIs that no longer exist (defensibility in the live walkthrough).
- docs/ARCHITECTURE.md:43 lists to_target / back_factor / from_target (replaced by `common.Target`, common.py:143).
- :59 `evaluate_spec(..., back_method, ...)`: the real signature takes `target: Target` (validation.py:50).
- :78 F1 freeze says "target = log1p": superseded by ADR-017 (power 0.1).
- :6 says tools/ is frozen kit, yet tools/build_nb, embed_bundle, polish_nb, standalone_check were added later.
- Fix: doc-only edit by the chief. BP: yes. H9/H12 defensibility.

A6 MED. Bike-scale prediction is implemented 5 times, and the copies are not identical.
- p1.py:124 `bike_predictions` (NO eta cap) · validation.py:30-47 `fit_predict` (cap) · p4.py:177 `predict_bikes`
  (cap) · predict.py:177 `unclipped_bikes` (cap) · exp/v2/lib.py:47 (cap on q) · gd_asym.py:12 `_eta` (cap).
- STATE says the cap is inert for every reported fit, so P1 is unaffected today. It is still a latent
  inconsistency that a reviewer can point at.
- Fix: `Target.predict(eta_fit, cnt_fit, eta_new) -> (s, fit_bikes, new_bikes)` in common.py; callers become
  one line each. About -25 lines.
- BP: yes iff the cap stays inert in P1 (verify with fingerprint). It touches common.py -> full rebuild -> post-deadline.

A7 MED. Phase modules double as libraries: hidden coupling (the root cause of A2).
- p2 <- p1 (Split, bike_predictions, curve_points, make_split, model_scores, oracle_gap: p2.py:19);
  p4 <- p3.ladder_specs; predict <- p4.SOLVER_TOL/MAX_SWEEPS (predict.py:23); nbtools, bonus_control <- p1.
- Fix: move the shared helpers down: scoring -> validation.py (with A6), ladder_specs -> validation.py, solver
  constants -> regularization.py. Phase modules then import only libraries. About ±0 lines, much clearer graph.
- BP: yes. It touches p1-p4 -> full rebuild -> post-deadline.

A8 MED. Duplicate stability selection; the library version is dead in production.
- regularization.py:136 `stability_selection` is used only by tests (test_regularization.py:191-210).
  p4.py:382 `stability_frequency` is a copy with tol 1e-7 / 20000 sweeps and sweep logging.
- regularization.py:128 `group_bootstrap_rows` rebuilds the {date: rows} map in every call (584 dates x n
  compares, x150 calls).
- Fix: add tol, max_sweeps, return sweeps to stability_selection, have p4 call it, and precompute the row map
  once (-12 lines).
- BP: yes (same rng streams default_rng(seed+b), same solver arguments). It touches regularization.py -> P4 rebuild -> defer.

A9 LOW. P4 runtime (~20 min). Measured; the cheap micro-optimisations do not help.
- Real stage-A design: X_fit 8708x298; one 40-alpha lasso path = 52.9 s, 75,710 sweeps, 0.70 ms/sweep.
  centred_gram 0.02 s; fold_splits 0.2 s, so the repeated fold builds (p4.py:293,501,558,411) cost < 1 s
  and are not worth caching.
- Row-access plus Python-float `cd_sweep` variant: bit-identical W/B/sweeps, but SLOWER in the dense regime
  (1.02 vs 0.70 ms/sweep; 1.27x faster only in the sparse head). Rejected.
- Options left: (a) run the independent paths in parallel processes (6 enet rhos, 5 folds x 3 methods, 2x50
  stability fits; 12 cores): bit-identical, estimated 20 -> ~5 min, +30 lines, costs readability (H12).
  (b) Active-set CD (glmnet-style): estimated 3-10x, NOT BP (tol-level drift can flip survivors/verdicts).
  (c) Larger SOLVER_TOL: NOT BP, and S-4-01 fixed the tolerance.
- Recommendation: none before the deadline. p4.json is final and no planned change needs a P4 rerun.

A10 LOW. gd.gradient_descent computes X@w twice per iteration (gd.py:63-65: loss_fn and grad_fn both form the
  residual): 3 mat-vecs per iteration instead of 2 for P1, P2 and P5 (P5: 12 GD runs up to 50k iterations).
- Fix: an optional `loss_and_grad(w)` closure; bit-identical arithmetic, about 1.3-1.5x faster.
- BP: yes, but it touches gd.py (H4 code shown in the notebook) and rebuilds P1->P5. Post-deadline only.

A11 LOW. Naming and interface collisions (explainability in the walkthrough).
- `Split`: p1.py:34 (frames + targets) vs p4.py:40 (matrices). `Fit`: p2.py:28 vs p4.py:66. `Context`:
  p3.py:26 vs p4.py:93.
- `curve_points`: p1.py:81 (loss curve) vs p5.py:148 (ROC/PR points).
- `METHODS`: tuple in p4.py:34 vs label dict in nbview.py:14, plus the same dict as `LABELS` in plots_p4.py:16.
- `timed` duplicated in p4.py:114 and p5.py:41. `TALL=(7.2,4.4)` defined 3 times (plots_p3:15, p4:18, p5:19).
- `PROFILE_BAND` duplicated in predict.py:29 and tools/submission_check.py.
- `evaluate_spec` is a one-line wrapper of `fit_predict` (validation.py:50).
- Fix now (nb layer, cheap): one METHOD_LABELS and TALL in plotstyle.py (-6 lines). Phase renames go
  post-deadline (they rebuild the chain; keep artifact key names unchanged).

A12 LOW. Tooling redundancy and drift.
- The Makefile (frozen kit) has drifted from run.py: python3, deps missing features.py/gd_asym/verdicts/labels,
  leak_scan without the `--test-ok` list (run.py:29-33), no p6 chain, nb without render/embed. On this machine
  run.py is the only working path. Fix: one README line saying "Makefile is non-authoritative; use run.py".
- tools/fingerprint.py:20 hashes p?.json only (p1..p6). It skips sample_submission.csv, bonus_control.json and
  diagnostics.json. Proof procedure for any refactor: fingerprint, plus sha256 of sample_submission.csv and
  bonus_control.json, before and after.
- nbview.p6_chain_rows (nbview.py:661) duplicates tools/chain_check.py on purpose (the notebook must assert the
  chain on Colab without tools/). Keep it.

A13 LOW. Config handling.
- config_sha256 hashes the whole file (common.py:303), so editing any knob or comment invalidates P1..P5.
  Per-section hashes would change the artifact contract -> no.
- Hard-coded knobs live as module constants (p4 SOLVER_TOL/MAX_SWEEPS/UNREGULARISED_ALPHA/PATH_NAMES, p5
  AUC_TIE/BOOTSTRAP_RNG_OFFSET, p3 OVERFIT_MARGIN): justified per spec, but split between config.yaml and code.
  Moving them to config.yaml changes the config hash -> full rebuild -> no.
- Rule until the deadline: do not touch config.yaml.
- RNG offsets: bootstrap_r2 uses seed+0 (p3, p4), stability uses seed+0..49, p5 uses seed+5, so the streams
  overlap across purposes. Harmless (independent uses), but ARCHITECTURE section 2 asks for a documented k per
  stream. Doc-only fix.

A14 INFO (no action). nbview.py (697 lines) is about 60 single-use table builders of 5-10 lines each, with no
  duplicated logic. It is long, not over-engineered. p4.py (668 lines) has a clear step structure (stage A
  curves -> CV -> verdicts -> survivors -> stage B -> rich check) and no dead functions. The dead-code scan of
  src/ (no production caller) found only `Design.constant_columns` (tests only) and `stability_selection` (A8).
  gd.py vs gd_asym.py is justified (Armijo backtracking vs fixed step; GDResult is shared).
  validation.py vs predict.py: the overlap is only A6.

## Do now (2 days, all BP, no phase rebuild)
1. A1 + A2: manifest-hash staleness with an import-closure dep graph in run.py. Test: `run.py all` on a fresh
   checkout prints "up to date" x5; touching labels.py marks only p5.
2. A3 (run.py part): nb builds phases and bonus_control before rendering.
3. A4 README notes in exp/; A5 ARCHITECTURE doc fixes; A12 README line; A11 nb-layer label/TALL dedupe.
After each: `python run.py verify-fast` and fingerprint equal (p1..p6 hashes unchanged).

## Defer (each forces a P1..P5 or P4..P5 rebuild; batch into ONE rebuild if ever done)
A6, A7, A8, A10, A11 phase renames, A3 common.is_current. Procedure: fingerprint + submission sha before,
`run.py all && run.py submission`, then fingerprint again and byte-diff p1..p6.json and sample_submission.csv.
Expect identical bytes. A9(a) only if a P4 rerun becomes unavoidable.

## Hard constraints touched
None of the proposed fixes changes H1-H13 behavior. A1/A2/A3 strengthen H5 and H10. A4/A5 support H12.
A10 touches H4 code (own GD) but keeps it own-GD. A9(b) and (c) are rejected: they change results after freeze F2.

## Target layout (minimum that still does the job; post-deadline)
```
run.py                 targets + hash-manifest staleness (import-closure deps)
config.yaml
src/
  common.py            config, seed, splits, Target (+ predict helper, A6), metrics, bootstrap, artifact I/O, timed
  features.py          raw_columns, DesignSpec, Design
  gd.py                own GD (+ optional fused loss/grad)        gd_asym.py   bonus loss + Armijo GD
  poly.py              lift_weights
  validation.py        fit_linear, scoring, three validators, ladder_specs, learning curve, noise floor
  regularization.py    ridge/enet solvers, paths, stability_selection, SOLVER_TOL/MAX_SWEEPS
  verdicts.py          pure verdict rules
  logistic.py          logistic GD + metrics (+ labels.py: 40 lines, could merge)
  phases/p1..p5.py     orchestration only: import libraries, never another phase
  predict.py           P6 (name kept: leak_scan whitelist "*predict*")
  bonus_control.py     side study, a run.py target
  nb/                  nbview.py, nbtools.py, plots.py (p1-p2), plots_p3/p4/p5.py, plotstyle.py  (not chain deps)
tools/                 unchanged (frozen kit + checkers)
exp/v2/                live diagnostics on src APIs;  exp/{analyst,loss_scale,eda0}: archived first pass
```
(src/nb/ is an optional move: it changes nb imports and the embed bundle, so skip it before the deadline.)
