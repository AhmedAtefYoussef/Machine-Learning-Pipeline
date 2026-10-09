# Walkthrough: how to explain and re-run every part of the Rush Hour pipeline

This is our rehearsal guide for the live evaluation, where each of us must be able to explain any cell and change and re-run it on the spot. It is written so that any one of us can answer alone. We do not quote results here: every score, iteration count, penalty value and feature count lives in `artifacts/pN.json` and in the Outcome cells of the notebook, and this guide says where to look. The only numbers typed below are settings from `config.yaml` or constants in the code.

The project was run twice. The first complete run (git tag `v1-submitted`) fitted the logarithm of the count. The second pass fits a milder power transform of the count with exponent 0.1 and carries it through all five phases; everything in the repository, the notebook and the report describes the second pass, with the first pass kept as a reference (section 4, questions 1 to 3, explains the two passes). Wherever a formula below has a logarithmic special case, it is the first pass.

## 1. How to run

### Commands

There is no `make` on our machine, so `run.py` plays that role. It sets the environment itself (UTF-8, one BLAS thread, `PYTHONPATH=.`), so the commands below work from the project root in Git Bash without further setup. The interpreter is `python`, not `python3`.

| What we want | Command | What it does |
|---|---|---|
| The whole chain | `python run.py all` | Builds Phases 1 to 5 in order, skipping the ones that are up to date. |
| One phase | `python run.py p3` | Builds Phases 1 to 3, again only the stale ones. |
| The submission | `python run.py submission` | Runs `src/predict.py`: writes `sample_submission.csv` and `artifacts/p6.json`. |
| The notebook | `python run.py nb` | Fills the `{{...}}` placeholders of `nb/p*.py` into `build/nb/`, embeds the project bundle into the setup cell (`tools/embed_bundle.py`), assembles and executes `rush_hour.ipynb`. |
| The notebook in an empty folder | `python -X utf8 tools/standalone_check.py` | Copies only `train.csv` and `test.csv` into a temporary folder, runs a copy of `rush_hour.ipynb` there with a fresh kernel (as on Colab), and reports errors, the files created and whether the submission it wrote is identical to the repository's. Nothing in the repository changes. |
| The report | `python run.py report` | Fills `report/report.template.md` into `report/report.md` and builds the PDF. |
| Quick checks | `python run.py verify-fast` | Lint, tests, leak scan, chain check, requirement trace. |
| Full checks | `python run.py verify` | The quick checks plus notebook run, submission check and the report number trace. |
| Artifact hashes | `python run.py fingerprint` | Stable hashes of the artifacts, to prove that a refactor changed no number. |
| The bonus control | `python -m src.bonus_control` | Fits the bike-scale loss with k = 1 from the Phase 1 weights and writes `artifacts/bonus_control.json`. A side study that feeds no phase. |
| Start over | `python run.py clean` | Deletes the artifacts, the notebook and the rendered report. |

To run exactly one phase without the staleness logic, call its module directly: `PYTHONPATH=. python -X utf8 -m src.phases.p5` (each module calls `set_threads` itself, so the result is the same as through `run.py`).

### The notebook is self-contained

The first code cell of `rush_hour.ipynb` carries our project files (`src/`, `config.yaml`, the stored `artifacts/` and the empty submission template) as a compressed block of text, unpacks every member that does not exist yet and never overwrites a file that does. On our machines everything is already there, so nothing is unpacked. On Google Colab only the notebook, `train.csv` and `test.csv` are uploaded; the cell asks for the data files if they are missing and prints, for each, whether it is the file our stored results were computed from (a hash comparison). If a data file differs, the stored artifacts do not belong to it and `RECOMPUTE_ALL = True` is the way out.

Every phase cell goes through the helper `phase(name, run_fn, cfg, upstream, live, recompute_all)` in `src/nbtools.py`. With `RECOMPUTE_ALL = True` (set in the configuration cell and passed on as `recompute_all`) it simply runs the phase. Otherwise there are two behaviours. Phases 1 to 3 are called with `live=True`: the phase is computed here and now (seconds), the result is compared with the stored artifact by `largest_difference`, the cell prints "computed live just now; largest relative difference to the stored artifact …", and the stored file is put back so that the later phases, which are tied to it by its hash, stay valid (another machine can differ in the last digits). Phases 4 and 5 are called without `live`: `common.cached_or_run` loads the stored artifact if the configuration in memory equals `config.yaml` on disk, the artifact's `config_sha256` equals the hash of `config.yaml`, and its `upstream_sha256` equals the hash of the previous artifact; the cell then prints "loaded cached artifacts/...". Any other case runs the phase. Two further cells always compute: the step-by-step Phase 1 cell, which re-runs our gradient descent from zero weights and asserts that it reproduces the stored weight vector, and the bonus-control cell, which calls `bonus_control.run(CFG)` and rewrites `artifacts/bonus_control.json`.

The notebook cells themselves are short: each starts with one comment line and then calls into `src`. Tables come from `src/nbview.py` (one small function per table, all built on `styled`, `facts` and `checks`; the row that records a choice is highlighted), figures from `src/plots*.py` with the shared style of `src/plotstyle.py` (the same entity has the same colour in every figure), and `nbtools.show_source` prints the source of our own functions (for example `Target`, `mse_loss`, `mse_grad`, `gradient_descent`, `asym_loss`, `asym_grad`), so the algorithm is visible where it is used. None of these modules computes a result: they only read the artifacts and format them.

### Runtimes

Phases 1 and 2 together take well under a minute. Phase 3 takes about ten seconds, because its several hundred fits use the closed form (the exponent check adds one more set of fits per candidate exponent). Phase 4 is the slow one, about 30 minutes: three methods, six elastic-net mixing values, five day folds, 50 stability resamples, and all of it twice (stage A on the full candidate design, stage B on the survivors). Phase 5 takes about three minutes. With the stored artifacts the whole notebook runs in a few minutes; with `RECOMPUTE_ALL = True` it recomputes the chain and takes well over half an hour.

### Where things live and how the chain works

Every phase writes one JSON file, `artifacts/p1.json` to `artifacts/p5.json`, and the submission step writes `artifacts/p6.json`. Each phase loads the previous phase's file from disk, never from memory. `common.write_artifact` adds three fields to every file: `seed` (recomputed from the team IDs), `config_sha256` (hash of `config.yaml`, or of the edited configuration when a drill changed `CFG` in memory) and `upstream_sha256` (hash of the previous artifact file, empty for Phase 1). So if anyone edits or rebuilds Phase 2, the hash stored inside `p3.json` no longer matches and `tools/chain_check.py` fails until Phases 3 to 6 are rebuilt. The last notebook cell prints the same assertions in one table: the hash chain, one seed everywhere, the first Phase 2 loss equal to the last Phase 1 loss, Phase 5 features inside the Phase 4 survivors, and the submitted model reproducing the Phase 4 weights.

Three more files sit beside the chain and feed nothing. `artifacts/first_pass.json` holds the key numbers of the first pass (log target) that the notebook and report quote for comparison. `artifacts/diagnostics.json` holds the numbers of the exploration outside the chain (`exp/v2/`) that the report quotes. `artifacts/bonus_control.json` is the k = 1 control of the bonus.

`run.py` decides what to rebuild by file time: a phase is stale when its artifact is missing or older than `config.yaml`, `src/features.py`, `src/common.py`, its own source files or the upstream artifact. Two consequences matter in the room. First, a rebuild cascades: rebuilding Phase 2 makes 3, 4 and 5 stale. Second, saving `config.yaml` makes every phase stale, whatever we changed in it, so `python run.py p5` after a config edit rebuilds Phase 4 as well. For a live drill we therefore prefer the notebook: change the `CFG` dictionary in the configuration cell and run only the cell of the phase in question. Do not "run all below" after an edit: because `CFG` differs from `config.yaml`, `cached_or_run` recomputes every later phase cell it reaches, Phase 4 included.

One thing to know before drilling: a phase run with a `CFG` changed in memory still overwrites its artifact file (stamped with the hash of the edited configuration, so `cached_or_run` will not trust it afterwards). Copy the `artifacts/` folder before a rehearsal and copy it back afterwards (or `git checkout -- artifacts sample_submission.csv`), and finish with `python run.py verify-fast`.

### Before we walk into the room

1. Run `python run.py verify-fast` and confirm that the chain check passes for all artifacts.
2. Open `rush_hour.ipynb` and confirm that the five phase cells print "loaded cached artifacts/...", so nobody waits half an hour by accident. If one does not, `RECOMPUTE_ALL` or the configuration has been touched.
3. Copy the `artifacts/` folder somewhere safe, so that a drill can be undone in seconds. Check that `git tag` lists `v1-submitted`.
4. Check that `team_ids` in `config.yaml` are our registered IDs and that the setup cell prints the seed we expect.
5. Each of us opens one artifact file in an editor, so that a number can be looked up without scrolling the notebook. `artifacts/first_pass.json` should be open too, for the "why twice" questions.
6. Agree who answers first on which phase, but be ready for any phase: the examiner chooses.

### The knobs in `config.yaml`

- `team_ids`: the roster; the seed is derived from it by `common.team_seed` and never typed.
- `split.test_size` (0.20), `chrono.cut_date` (2012-07-01), `day_block_folds` (5), `bootstrap.B` (1000), `threads` (1).
- `p1`: `target_power` (0.1, the exponent of the target transform; 0 is the log), `target_power_candidates` (the exponents compared in the Phase 1 and Phase 3 tables; the exponent in use must be in this list), `lr_fraction_of_bound` (0.5), `tol_loss` (1e-10), `tol_grad` (1e-6), `max_iter` (50000), `sweep_fractions`, `sweep_max_iter`, `backtransform_candidates` (`none`, `ls`), and for the bonus `bonus.k_under` (3.0).
- `p2`: `blocks` (the `wd_x_hr` interaction block), `degree_candidates` (1 to 4), `power_col_candidates`, `plateau_tol` (0.001), and the same learning-rate and stopping settings as Phase 1.
- `p3`: the `ladder` (names and blocks of each level), `degree_axis`, `learning_fractions`, `fit_alpha` (1e-8), `plateau_tol` (0.001).
- `p4`: `candidate_blocks`, `n_alphas` (40), `ridge_alpha_range`, `l1_alpha_min_ratio` (1e-4), `l1_ratios`, `stability.B` (50) and `stability.threshold` (0.6), `verdict.min_delta` (0.001) and `verdict.solo_min` (0.01).
- `p5`: `quantile` (0.75), `group_by` (year, working day, hour), `l2_candidates`, `cost_ratio_miss_to_false_alarm` (3.0), `max_iter`.

## 2. Phase by phase

Each phase section of the notebook has the original Expectation cell, a "second pass" Expectation cell, the justifications, and one Outcome cell. The original Expectation cells describe the first pass and are unchanged; the second-pass cells were committed before the chain was re-run; the Outcome says for each expectation whether it was met.

### Phase 0: setup, seed, split and features (`nb/p0_setup.py`, `src/common.py`, `src/features.py`)

This part unpacks the project bundle, loads the configuration, derives the seed, makes the one seeded split and shows what we saw in the data before modelling. It hands on `train_df` and `val_df`; every later phase recreates exactly the same split from the same seed.

**`team_seed` and `config_seed`.** The IDs are converted to strings, sorted, joined with underscores, hashed with SHA-256, and the seed is that hash as an integer modulo 100000. The function first checks itself against the worked example of the brief and fails if it does not reproduce it. Changing `team_ids` changes the seed, the split and therefore every number in the project.

**`seeded_split`.** This is the single call to `train_test_split` in the whole repository, with `test_size` 0.20 and the team seed. Nothing else is split with a seed.

**`chrono_split` and `day_block_folds`.** The chronological split puts rows before `chrono.cut_date` in the early part and the rest in the late part. The day folds assign each training row the rank of its date among the sorted training dates, modulo the number of folds, so whole days are held out and no random number is involved.

**`Target` (the whole target transform lives in one frozen dataclass).** The chain fits z = ((cnt + 1)^λ − 1)/λ with λ = `p1.target_power` = 0.1; the limit λ → 0 is z = log1p(cnt), the first pass, and λ = 1 is the raw count shifted by one. `Target(power, back_method)` is built from the configuration in Phase 1 (`p1.config_target`) and by every later phase from the stored Phase 1 artifact (`Target.from_artifact`), so nobody can use a different transform downstream. Its methods:

- `forward(cnt)` gives z.
- `q(eta)` is the inverse: cnt + 1 = (λη + 1)^(1/λ), and exp(η) for λ = 0. A floor (`Q_FLOOR`) keeps the base positive.
- `dq_deta(eta)` = q(η)^(1−λ), the derivative of q; it is what the bonus gradient needs.
- `factor(eta_train, cnt_train)` is the back-transform factor s, always computed on the fitting rows. `none` gives s = 1; `ls` gives the least-squares factor s = Σ(cnt + 1)·q(η) / Σ q(η)², the one multiplier that minimises squared error on bikes; `duan` (the mean of exp(z − η)) exists for λ = 0 only and raises an error otherwise. `valid_back_methods` drops it from the candidate list for any other exponent.
- `to_bikes(eta, s)` = clip(s·q(η) − 1, 0), with `bikes_unclipped` the same without the clip.
- `eta_cap(cnt_train)` is the η at which q reaches e times (largest training count + 1), that is forward(e·(max + 1) − 1); for λ = 0 this is the largest training target plus 1.0. Evaluation predictions are capped there so that q stays finite.
- `to_dict` and `from_artifact` write and read the target keys of `p1.json` (`target_power`, `backtransform.method`).

Phase 1 picks the back-transform method by validation R² among `none` and `ls`; later phases reuse the method and recompute s from their own training residuals.

**`r2`, `rmse`, `bootstrap_r2`, `paired_bootstrap_delta_r2`.** R² and RMSE are always computed on bikes. The bootstrap resamples validation rows (`bootstrap.B` times) and returns a 95% percentile interval; the paired version uses the same resamples for two models and returns the interval of their R² difference, which is how we decide whether two models really differ.

**`add_weather_memory`, `load_train`, `load_test`.** At load time two columns are added from the weather situation of earlier hours: the situation one hour earlier and the worst situation of the three hours before. They are looked up by timestamp; if an earlier hour is missing from the file, the row's own situation is used. Only `dteday`, `hr` and `weathersit` are read. `load_test` is the only reader of the hidden file and builds the timeline from the input columns of both files, so the first hour of a hidden day can see the last hour of the day before.

**`raw_columns`, `DesignSpec`, `Design`.** `raw_columns` turns a frame into named, unscaled columns: hour dummies with hour 0 as reference, `workingday`, `holiday`, weather-situation dummies with category 4 merged into 3, `temp`, `hum` (zeros replaced by the training median of positive humidity), `windspeed`, `trend` (days since 1 January 2011), two day-of-year sine and cosine pairs, and the candidates we hold back until Phase 4. A `DesignSpec` says which first-order columns, which powers up to which degree, and which interaction blocks to build. `Design.fit` learns the humidity fill and every mean and standard deviation on the frame it is given, and `Design.transform` applies them to any frame. Powers and products are built from the standardised columns and then standardised again, which keeps the design well conditioned. `sources` maps an expanded feature name back to the original column or columns behind it, and Phase 4 uses it for the verdicts.

**`write_artifact`, `read_artifact`, `cached_or_run`.** Described in section 1. `write_artifact` refuses to store NaN or infinity, so a broken run fails loudly instead of leaving a bad file.

### Phase 1: gradient descent (`nb/p1_gd.py`, `src/gd.py`, `src/gd_asym.py`, `src/phases/p1.py`)

Phase 1 fits a linear model for z = ((cnt + 1)^λ − 1)/λ on the base design with our own gradient descent, started from zero weights. It hands on the weight vector, the scaler, the learning rate, the iteration count, the target exponent and the chosen back-transform method in `artifacts/p1.json`.

**`mse_loss` and `mse_grad`.** L(w) = (1/2n)·‖Xw − z‖² and ∇L(w) = (1/n)·Xᵀ(Xw − z). X already contains the bias column. Nothing in the configuration changes these; they are the definition of the phase. The exponent only changes the vector z that they are fitted to.

**`lambda_max` and `learning_rate`.** `lambda_max` is the largest eigenvalue of XᵀX/n. For a quadratic loss, the error along each eigen-direction is multiplied by (1 − lr·λ) at every step, so gradient descent is stable exactly when lr < 2/λ_max. We use lr = `lr_fraction_of_bound` × 2/λ_max with the fraction 0.5, that is lr = 1/λ_max: every factor then lies between 0 and 1, so no direction overshoots. Raising the fraction towards 1 speeds up the slow directions but makes the steepest one oscillate; above 1 the steepest direction grows at every step and the loss explodes. The bound depends on X only, so changing the exponent leaves λ_max, the bound and the learning rate unchanged.

**`gradient_descent`.** The update is w ← w − lr·∇L(w). It stops as "converged" when |L_new − L_prev| ≤ `tol_loss` × L_prev and ‖∇L‖₂ < `tol_grad`, as "diverged" when the loss is not finite or exceeds one million times the starting loss, and otherwise as "max_iter". Both convergence tests must hold together. Loosening the tolerances stops earlier and leaves the weights further from the exact solution; tightening them costs iterations and gains nothing visible in R².

**`lr_sweep`.** One run from zeros for each value in `sweep_fractions` (0.001 to 1.05 of the bound), capped at `sweep_max_iter`. It is our evidence for "stall, converge, diverge": see `p1.json → lr_sweep` and the plot under the learning-rate justification.

**`gradient_check` and `gradient_check_errors`.** The analytic gradient is compared with central differences, (L(w + εe_j) − L(w − εe_j))/(2ε), on 20 random coordinates. We check at zeros and at a random point, not at the final weights, because the gradient there is almost zero and a relative error would only measure round-off.

**`oracle_gap`.** The closed-form least-squares solution is computed once and compared with our weights (`p1.json → oracle`). It is a check only and never used for fitting.

**The live check.** One notebook cell recomputes the main gradient descent from zeros with `make_split`, `base_design`, `learning_rate` and `run_gd`, prints iterations and stop reason next to the stored ones, and asserts that the weights agree with `p1.json → weights` to 1e-8. Different machines add floating-point numbers in a different order, so we expect agreement to many decimals, not bit for bit. This is the cell to point at when someone asks "did you really run it?".

**`backtransform_table`, `bike_predictions`, `model_scores`.** Each method in `backtransform_candidates` that exists for the exponent (`none` and `ls`) is scored on validation and the best one is kept (`p1.json → backtransform`, with each candidate's factor, R², RMSE and the ratio of the mean prediction to the mean demand). Removing a candidate from the list removes it from the comparison; the method chosen here is reused by every later phase. Duan's factor is not in this table: it is defined for the log only, and its first-pass result is quoted from `first_pass.json`.

**`target_power_table` and `target_power_best_here`.** The same base design is fitted by our own gradient descent, from zeros, with the same learning rate and stopping rule, once for each exponent in `target_power_candidates`, scored on bikes with the least-squares factor (`p1.json → target_power_table`, one row per exponent; `target_power_best_here` is the exponent with the best validation R²). The exponent actually used comes from `config.yaml`, not from this table. The plot `plot_target_power` draws the table.

**`hour_ablation` and `residual_profile`.** The ablation refits the same model with the hour as a single number and as six sine and cosine pairs instead of dummies (`p1.json → hour_encoding_ablation`). The residual profile averages the residual on the z scale per day type and hour; its pattern is what motivates the Phase 2 interaction block.

**`asym_loss`, `asym_grad`, `fit_asymmetric`, `asymmetric_bonus` (bonus).** The operator's cost is L(w) = (1/n)·Σ cᵢ·(ŷᵢ − yᵢ)², with ŷᵢ = q(xᵢ·w) − 1, q(η) = (λη + 1)^(1/λ), and cᵢ = k when ŷᵢ < yᵢ and 1 otherwise; k is `p1.bonus.k_under` = 3. For λ = 0 this is ŷ = exp(xw) − 1. By the chain rule ∂ŷᵢ/∂w = q′(ηᵢ)·xᵢ with q′(η) = q(η)^(1−λ) (`Target.dq_deta`), so ∇L = (2/n)·Σ cᵢ·(ŷᵢ − yᵢ)·q(ηᵢ)^(1−λ)·xᵢ; for λ = 0 the factor is the familiar exp(ηᵢ). The weight c contributes no extra term because it only jumps where the residual is zero. The linear predictor is held at or below `Target.eta_cap` so that q stays finite. This loss is not quadratic, so the 2/λ_max bound does not apply. `fit_asymmetric` chooses each step by backtracking: it halves the step until L(w − t·g) ≤ L(w) − 1e-4·t·‖g‖², then starts the next iteration from twice the accepted step, and stops when the relative loss change stays below the tolerance for ten iterations in a row. It starts from the Phase 1 MSE weights and its result never enters the chain. Raising k pushes predictions further up. Both models in the bonus table are compared as q(Xw) − 1 without a back-transform factor.

**`bonus_control.run`.** The bonus changes two things at once: the fit is on bikes instead of the z scale, and under-prediction is weighted. The control fits the same bike-scale loss with k = 1 from the same start, so that the upward shift splits into "fitting on bikes" and "asymmetry" (`bonus_control.json → shift_from_fitting_on_bikes`, `shift_from_asymmetry`, `total_shift`).

### Phase 2: polynomial regression (`nb/p2_poly.py`, `src/poly.py`, `src/phases/p2.py`)

Phase 2 loads the Phase 1 weights, expands the design with the working day × hour products and powers of temperature, and continues our own gradient descent from the lifted weights, on the same target (built with `Target.from_artifact`). It hands on the chosen degree, the expanded feature list and its scores in `artifacts/p2.json`.

**`lift_weights`.** The longer weight vector has the Phase 1 weight in the slot of every Phase 1 column and zero everywhere else. It asserts that every Phase 1 name exists in the bigger design.

**`fit_expanded`.** It builds the expanded design with `Design.fit(train_df, base_scaler=p1["scaler"])`, so the Phase 1 columns keep exactly the Phase 1 means and standard deviations, lifts the weights, and asserts that the first loss equals `p1.train_loss_final` to 1e-9. Then it recomputes λ_max for the new design, sets lr = `p2.lr_fraction_of_bound` × 2/λ_max and runs the same `gradient_descent` with the same stopping rule. The equality of the two losses is only possible because both phases use the same exponent.

**`degree_sweep`, `power_col_sweep`, `smallest_within_tol`.** One gradient-descent fit per value in `degree_candidates` (with the widest list of power columns), then one per list in `power_col_candidates` at the chosen degree. `smallest_within_tol` picks the smallest option whose validation R² is within `plateau_tol` of the best. A larger tolerance favours simpler models; a tolerance of 0 is a plain "take the best". "Degree" is the highest total degree of any term, so the interaction products count as degree 2.

**The block ablation in `run`.** The chosen model is refitted without the `wd_x_hr` block and stored as `p2.json → block_ablation.without_wd_x_hr`, next to the paired bootstrap gain over Phase 1 (`paired_vs_p1`).

### Phase 3: bias–variance (`nb/p3_bias_variance.py`, `src/validation.py`, `src/phases/p3.py`)

Phase 3 takes the Phase 2 design as its anchor, moves complexity below and above it on a nested ladder, varies the degree alone and the amount of training data, scores everything with three validators, and re-checks the target exponent on the design it selects. It hands on a diagnosis and a target complexity in `artifacts/p3.json`. The ladder has eleven levels (C0 to C10).

**`fit_linear`, `fit_predict`, `evaluate_spec`.** These fits solve the normal equations (XᵀX/n + α·D)w = Xᵀz/n, where D is the identity with a zero for the bias and α = `p3.fit_alpha` = 1e-8. The tiny ridge term only keeps the solution defined where columns are exact copies of each other. `fit_predict` fits the design on the fitting frame only, takes the back-transform factor from the fitting rows' own residuals, and caps η on evaluation rows at `target.eta_cap(cnt_fit)` (q at most e times the largest training count plus one), so that an over-flexible fit on very few rows cannot make q explode.

**`three_validators` (with `day_block_scores` and `chrono_scores`).** The three validators are: (1) the seeded split, fit on `train_df`, score on `val_df`; (2) held-out days, `day_block_folds` folds of whole days inside `train_df`, each fold with its own design fitted on the other folds, reported as mean and standard deviation; (3) the chronological split, fit on all labelled rows before `chrono.cut_date`, score on the rows from that date on. `gaps_block` then defines leakage = seeded − held-out days and drift = held-out days − chronological.

**`ladder_specs` and `score_ladder`.** The ladder in `config.yaml` is cumulative. Levels before the one marked `phase2` are degree 1; the `phase2` level is exactly the Phase 2 design read from `p2.json`; later levels add their blocks to it. Adding, removing or reordering levels in `p3.ladder` changes what the phase compares.

**`check_anchor`.** The closed-form fit of the anchor must have the same feature names as `p2.json` and reproduce Phase 2's validation R² (`p3.json → anchor_check`). This is the chain link from Phase 2.

**`score_degree_axis` and `learning_curve`.** The degree axis keeps the anchor's blocks and power columns and changes only the degree (`p3.degree_axis`). The learning curves refit the anchor, the target and the top level on nested subsets of the training days (`p3.learning_fractions`): more data cures variance but not bias.

**`pick_target`.** The target is the lowest ladder level whose seeded validation R² is within `p3.plateau_tol` (0.001) of the best level. `first_overfit_level` reports the first level above the target that is worse on validation by more than 0.002 while fitting the training rows better.

**`classify`.** The label is "under-fit" when the target lies above the anchor, the paired bootstrap interval of the gain is above zero and the held-out-day gain is positive; "over-fit" in the mirror case; otherwise "reasonably fit".

**`target_power_check` and `exponent_summary`.** The target design is refitted in closed form once for every exponent in `target_power_candidates`, each time with the three validators (`p3.json → target_power_check`: seeded, held-out-day and chronological R² per exponent). `exponent_summary` stores the exponent in use (`target_power_used`), the exponent with the best seeded R² (`target_power_best_seeded`) and a flag, `target_power_consistent`, that is true when the exponent in use scores within `p3.plateau_tol` of the best seeded score. If it were false, the exponent fixed in Phase 1 would disagree with the model the chain ends with, and the whole chain would have to be restarted with another exponent. The exponent in use must be one of the candidates, because the summary looks its row up.

**`chrono_detail` and `noise_floor`.** `chrono_detail` stores the mean demand before and after the cut, the ratio of mean prediction to mean demand in the late part, and an R² after rescaling the predictions to the late mean. That last number is a diagnostic of level drift only. `noise_floor` is the share of variance explained by cell means over year, month, day type, hour and weather situation; it is a benchmark, not a ceiling.

### Phase 4: regularization (`nb/p4_regularization.py`, `src/regularization.py`, `src/verdicts.py`, `src/phases/p4.py`)

Phase 4 loads the target design from `p3.json`, adds back the held-back columns, fits Ridge, Lasso and Elastic Net with our own solvers, judges all 14 input columns and selects the survivors (stage A). It then repeats the same search on the survivors only and recommends one model (stage B). It hands on λ per method, the survivors, the verdicts and the recommended model in `artifacts/p4.json`. The penalised fits are on the z scale; the scores are on bikes through the same `Target`. (Here λ is the penalty; the exponent of the target is always called the exponent.)

**The objective.** J(b, w) = (1/2n)·‖z − b − Xw‖² + α·ρ·‖w‖₁ + (α/2)·(1 − ρ)·‖w‖². ρ is the `l1_ratio`: 0 is Ridge, 1 is Lasso, values between are Elastic Net. The intercept b is never penalised; the solvers centre X and z and recover b = z̄ − x̄·w.

**`ridge_closed_form` and `ridge_path`.** With G = XcᵀXc/n and c = Xcᵀzc/n on centred data, Ridge solves (G + αI)w = c. `ridge_path` does this for the whole grid from one eigendecomposition of G.

**`soft_threshold`, `cd_sweep`, `enet_cd`, `enet_path`.** Coordinate descent updates one weight at a time: w_j ← S(z_j, α·ρ) / (G_jj + α·(1 − ρ)), with z_j = c_j − (Gw)_j + G_jj·w_j and the soft-threshold S(z, t) = sign(z)·max(|z| − t, 0). The soft-threshold is what produces exact zeros. A sweep is one pass over all coordinates; the solver stops when no weight moved by more than 1e-7 in a sweep, with at most 20000 sweeps. `enet_path` runs down a descending grid and starts each solution from the previous one.

**`alpha_max`, `alpha_grid`, `make_grid`.** `alpha_max` = max_j |c_j| / ρ is the smallest penalty at which every weight is zero. The Lasso and Elastic Net grids are `n_alphas` = 40 log-spaced values from there down to `l1_alpha_min_ratio` times it; the Ridge grid is 40 log-spaced values over `ridge_alpha_range` (1e-6 to 1e3). More grid points give a finer curve and a proportionally longer run.

**`add_candidates` and `pick_design`.** The design is the Phase 3 target plus each block in `p4.candidate_blocks` whose columns are not already present (`p4.json → design_from`). All three methods get this same matrix.

**`validation_curves` and `choose_fits`.** Every path is scored on the validation rows. Per method we take the penalty with the highest validation R² on bikes, ties to the larger penalty; for Elastic Net also the best `l1_ratio` among `p4.l1_ratios`.

**`cross_validation` and `cv_summary`.** The same grids are refitted on the five day folds. We report the penalty that is best on held-out days and the "one standard error" penalty, the largest one whose mean fold score is within one standard error of that best. They sit beside the validation choice as a check.

**`describe_methods`, `comparisons`, `unregularised_model`.** For each method at its chosen setting we store validation R² with a bootstrap interval, RMSE, the held-out-day score, the chronological score and the number of non-zero weights, plus paired bootstrap differences between the methods and against the fit with a negligible penalty.

**`stability`.** Lasso at its chosen penalty is refitted on `stability.B` = 50 resamples of whole training days; the result is the share of resamples in which each weight is non-zero (`p4.json → stability.l1.freq`).

**`column_verdicts` with `drop_cost`, `solo_r2` and the rule in `src/verdicts.py`.** The reference model is Ridge at its chosen penalty. For each original column we measure the validation R² lost when every feature built from it is removed (with a paired bootstrap interval), the loss when it is removed together with its partner columns, and the R² of its own features alone. `basic_verdict` then says: **useful** if the drop-alone interval lies above zero and the loss is at least `verdict.min_delta`; **redundant** if not useful but the group drop hurts or the solo R² is at least `verdict.solo_min`; **uninformative** otherwise. The partner groups are {temp, atemp}, {season, mnth, dteday}, {yr, instant, dteday} and {weekday, workingday, holiday}. `promote_representatives` handles a group that matters as a whole while no member is missed alone: the member with the highest solo R² becomes useful and the others are redundant, "carried by" it. Lasso counts, selection frequency, entry order, correlations and VIF are stored as supporting evidence.

**`survivors`.** A feature survives when its Lasso weight is non-zero at the chosen penalty, its selection frequency is at least `stability.threshold` = 0.6, and every original column it is built from has the verdict useful (`p4.json → survivor_counts`, `survivors_expanded`, `survivors_original`).

**`final_stage` and `recommend`.** Stage B repeats curves, choice, cross-validation and comparisons on the surviving columns. `recommend` takes the method with the best validation R², keeps every method whose paired interval against it contains zero, and among those picks the best held-out-day R² (`p4.json → recommended`, with `candidates`, `best_on_validation`, intercept, weights and feature names). When the methods are tied within noise, this is a tie-break on held-out days, not a finding. In the first pass it picked Elastic Net; in the second pass it picks the method named in `p4.json → recommended.method` (Lasso), and `first_pass.json → p4.method` records the old choice.

**`rich_check`.** The top ladder level, which Phase 3 shows to be over-fit, is fitted unpenalised, with Ridge and with Lasso, to see whether a penalty rescues it. It is informational and feeds nothing.

### Phase 5: logistic regression (`nb/p5_logistic.py`, `src/labels.py`, `src/logistic.py`, `src/phases/p5.py`)

Phase 5 builds a yes/no label from `cnt`, fits our own logistic regression on the Phase 4 survivors only, chooses an operating cut-off from the operator's costs, and writes the retrospective. It hands on `artifacts/p5.json`. The label does not depend on the target transform.

**`fit_thresholds`, `apply_thresholds`, `label_rule`.** label_i = 1 when cnt_i is above the `quantile` (0.75) of training `cnt` in the row's cell, where a cell is one combination of `group_by` (year, working day, hour). The thresholds come from the training rows only and are looked up for validation rows; a row whose cell is missing raises an error instead of receiving a silent default.

**`survivor_design` and `design_matrix`.** The Phase 4 design is fitted on the training rows and only the columns in `p4.survivors_expanded` are kept. The notebook asserts that every Phase 5 feature is a Phase 4 survivor.

**`sigmoid`, `logloss`, `logloss_grad`.** p = sigmoid(x·w); L(w) = −(1/n)·Σ[y·log p + (1 − y)·log(1 − p)] + (l2/2)·‖w[1:]‖²; ∇L = (1/n)·Xᵀ(p − y) + l2·[0, w[1:]]. The bias is not penalised. The code uses `logaddexp` and a sigmoid that never exponentiates a positive number, so nothing overflows.

**`fit_logistic` and `default_lr`.** This is the same `gradient_descent` engine as Phase 1, from zero weights, with step 1/(λ_max/4 + l2). The curvature of the log-loss is at most a quarter of that of the squared loss, plus the ridge term, so this is the safe step.

**`fit_candidate` and `choose_l2`.** One fit per value in `l2_candidates`; the best validation ROC-AUC wins, and values within 0.0001 of it count as tied, with the larger penalty winning the tie (`p5.json → l2_sweep`, `l2`).

**`cost_at`, `metrics_at`, `threshold_curve`, `best_threshold`.** We predict "high" when p ≥ t. With a miss costing c = `cost_ratio_miss_to_false_alarm` = 3 times a false alarm, an alarm is worth raising when p·c > (1 − p), that is when p > 1/(1 + c) = 0.25. The cost of a cut-off on n rows is (c·FN + FP)/n. We report accuracy, precision, recall, F1, ROC-AUC and PR-AUC at 0.25, at 0.5 and at the cut-off with the best F1 on a grid.

**`roc_auc`, `pr_auc`, `calibration`, `bootstrap_auc`.** ROC-AUC uses the rank formula with averaged ties, PR-AUC is average precision. The calibration table compares predicted and observed shares in ten bins, and the bootstrap gives a 95% interval for the validation AUC.

**`label_variant` and `foil_matrices`.** The same models are fitted under two foil rules, one global threshold and (working day, hour) without the year, with hour dummies only, trend only, and the full feature set. This is our evidence for the label rule (`p5.json → label_variants`).

**`retrospective`.** One row per phase, built from the five artifact files: what the phase consumed, its settings and its scores.

### Phase 6: final model and submission (`nb/p6_submission.py`, `src/predict.py`)

Phase 6 takes the recommended model from `p4.json`, refits it and predicts the hidden days. Nothing is tuned here. It writes `sample_submission.csv` and `artifacts/p6.json`. The regression model is the one the pipeline recommends, and its exponent and back-transform method come from `p1.json`.

**`fit_penalised`.** It solves the recommended method at its stored λ and `l1_ratio` in the same way as Phase 4: closed form for Ridge, otherwise the warm-started path down to λ. It asserts that coordinate descent converged at the final λ.

**`build`.** The design is fitted on the training rows only. Model A is refitted on the training rows and must reproduce the stored Phase 4 weights to 1e-6 (`p6.json → model_a_gap_to_p4`). Model B uses the same columns, λ and `l1_ratio` but is fitted on training plus validation rows, and it is the one submitted. The scaler and the humidity fill are not refitted.

**`unclipped_bikes` and `clip_at_zero`.** cnt_hat = clip(s·q(min(b + x·w, η_cap)) − 1, 0), with q the inverse of the target transform, s recomputed from the fitting rows by the Phase 1 method, and the same η cap as in `validation.fit_predict`.

**`profile_ratio`, `daily_total_ratio`, `agreement`.** These are sanity checks without labels: the predicted profile by day type and hour against the training profile, the predicted daily totals against the same month's training days, and how far models A and B are apart on the hidden rows.

### Where to read each number when asked

We never quote a result from memory in the room; we open the artifact or scroll to the cell. These are the keys we are most likely to need.

| Question | Where the answer is |
|---|---|
| Which exponent the chain uses | `config.yaml → p1.target_power`; stored as `p1.json → target_power` |
| Exponent table on the Phase 1 design; which exponent Phase 1 alone prefers | `p1.json → target_power_table` (one row per exponent), `target_power_best_here` |
| Exponent table on the target design; is the exponent consistent | `p3.json → target_power_check` (seeded, day-block, chrono per exponent), `target_power_used`, `target_power_best_seeded`, `target_power_consistent` |
| Numbers of the first pass (log target) | `artifacts/first_pass.json` (`p1`, `p2`, `p3`, `p4`, `p5`, `target_transform`); git tag `v1-submitted` |
| Diagnostic numbers quoted in the report (tree model, observation weights, month levels, weather blocks) | `artifacts/diagnostics.json`; raw results in `exp/v2/*.json` |
| Learning rate, bound, λ_max, iterations, stop reason of Phase 1 | `p1.json → lr`, `lr_bound`, `lambda_max`, `iterations`, `stop_reason` |
| How close gradient descent is to the exact solution | `p1.json → oracle`, `p2.json → oracle`; the live Phase 1 cell |
| Stall, converge, diverge | `p1.json → lr_sweep` (one row per fraction) |
| Back-transform comparison | `p1.json → backtransform.candidates` (`none`, `ls`) |
| Phase 1 and Phase 2 scores and intervals | `val_r2`, `val_rmse`, `train_r2`, `val_bootstrap` in `p1.json` and `p2.json` |
| Hand-over from Phase 1 to Phase 2 | `p2.json → init_loss` against `p1.json → train_loss_final` |
| Degree, power columns, interaction ablation | `p2.json → degree`, `degree_sweep`, `power_col_sweep`, `block_ablation` |
| Bonus loss, gradient check, shift | `p1.json → bonus`; split into fitting on bikes and asymmetry in `bonus_control.json` |
| Ladder table with the three validators | `p3.json → ladder` (one row per level) |
| Target level and why | `p3.json → target_level`, `target_complexity`, `plateau_tol` |
| Diagnosis and its evidence | `p3.json → diagnosis`, `learning_curves`, `paired_target_vs_anchor` |
| Seeded, held-out-day and chronological estimates and their gaps | `p3.json → estimates`, `estimates_target`, `gaps`, `chrono_detail` (`mean_ratio` is the overshoot of the late months) |
| Penalty and `l1_ratio` per method, with all three scores | `p4.json → methods` (stage A), `final.methods` (stage B) |
| Do the methods differ? Does a penalty help at all? | `p4.json → comparisons`, `final.comparisons`, `unregularised`, `rich_check` |
| Why this method is recommended | `p4.json → recommended` (`candidates`, `best_on_validation`, `rule`) |
| Verdict and evidence for one column | `p4.json → column_verdicts.<column>` (`verdict`, `evidence`, `numbers`, `carried_by`) |
| Survivor counts at each step of the rule | `p4.json → survivor_counts`, `survivors_original` |
| The model behind the submission | `p4.json → recommended`, `p6.json → model` |
| Label thresholds and class balance | `p5.json → threshold_rule`, `class_balance` |
| Classifier metrics at the three cut-offs | `p5.json → metrics`, `metrics_at_0_5`, `metrics_at_f1_opt` |
| Foil label rules | `p5.json → label_variants` |
| Submission sanity checks | `p6.json → pred`, `a_vs_b_on_test`, `profile_ratio`, `daily_total_ratio` |

## 3. Change and re-run drills

For each drill: the change, what to re-run, and what should happen. Unless said otherwise, make the change on `CFG` in the configuration cell and run only the cell of the named phase (`p1 = phase("p1", p1mod.run)`, `p3 = phase("p3", p3mod.run, upstream="p2")`, and so on), which is the fastest route. Because `CFG` then differs from `config.yaml`, `phase()` recomputes instead of loading. Editing `config.yaml` and running `python run.py pN` gives the same numbers but rebuilds every phase, Phase 4 included.

1. **Target exponent 0.0, the first pass.** Set `CFG["p1"]["target_power"] = 0.0` and run the Phase 1, 2 and 3 cells in order (about a minute and a few seconds). The target is the log again. Phases 1 to 3 should reproduce the numbers of the first pass: compare `p1.val_r2`, the Phase 2 degree and the Phase 3 target estimates with `artifacts/first_pass.json`. The back-transform candidates are `none` and `ls`; to see Duan's factor back, add `duan` to `CFG["p1"]["backtransform_candidates"]`, which works for exponent 0 only. The Phase 3 consistency flag should now be false, because on the target design the log is not within the tolerance of the best exponent. Everything downstream (Phases 4 and 5, the submission) is now stale and would take the 30 minutes of Phase 4 to rebuild; for the demonstration stop after Phase 3. Restore the saved `artifacts/` afterwards.

2. **Target exponent 0.3.** Set `CFG["p1"]["target_power"] = 0.3` and run the Phase 1 cell. This is the exponent that suits the simple Phase 1 design better, so its validation R² should be higher than at 0.1 (compare with the 0.3 row of `target_power_table`; the table uses the least-squares factor, so the two can differ if `none` wins in the main run). The learning rate and bound are unchanged, because the design matrix is. Then run Phases 2 and 3: the exponent check in Phase 3 should now report `target_power_consistent` false, because on the target design 0.3 is clearly worse than the best exponent; that is the check telling us the chain would have to be restarted with another exponent. This drill is the live answer to "why not 0.3 if Phase 1 prefers it?". An exponent that is not in `target_power_candidates` makes Phase 3 fail when it looks up the used exponent; add it to the list first.

3. **One exponent on the Phase 1 design, touching no artifact.** In a scratch cell: `split = p1mod.make_split(CFG); design, X_tr, X_va = p1mod.base_design(split); print(pd.DataFrame(p1mod.target_power_table(split, X_tr, X_va, CFG)).round(4).to_string())` shows the whole Phase 1 table recomputed (eight gradient-descent fits, seconds). For a single exponent, set `CFG["p1"]["target_power_candidates"] = [0.2]` first, or build it by hand: `from src.common import Target; t = Target(0.2, "ls"); s2 = p1mod.make_split(CFG, t); _, _, lr = p1mod.learning_rate(X_tr, CFG["p1"]["lr_fraction_of_bound"]); res = p1mod.run_gd(X_tr, s2.z_tr, np.zeros(X_tr.shape[1]), lr, 1e-10, 1e-6, 50000); print(p1mod.model_scores(s2, X_tr, X_va, res.weights, t)["val_r2"])`. `write_artifact` is never called, so the chain is untouched.

4. **`RECOMPUTE_ALL`.** Set `RECOMPUTE_ALL = True` in the configuration cell and run the phase cells. Every `phase(...)` call now runs its phase from scratch (the live Phase 1 cell and the bonus control run anyway). The whole chain takes well over half an hour, mostly Phase 4, and rewrites the artifacts. On our machine the result should equal what was stored (`python run.py fingerprint` before and after); on another machine the last digits can differ. The setup cell suggests this switch when the data files do not match the stored hashes.

5. **Learning-rate fraction 0.9.** Set `CFG["p1"]["lr_fraction_of_bound"] = 0.9` and run the Phase 1 cell (seconds). It still converges, in fewer iterations (compare with the 0.9 row of `lr_sweep`), and reaches the same weights within the tolerance, so R² does not move. The steepest direction now flips sign at every step, but the loss still falls because every factor |1 − lr·λ| is below 1. The weights differ in their last digits, so `p1.json` changes and Phases 2 to 6 are stale; running the Phase 2 cell shows that the hand-over check still passes. The live check cell uses the same `CFG`, so it still agrees with the freshly written artifact.

6. **Learning-rate fraction 1.05.** The steepest direction is now multiplied by −1.1 at every step, so the loss grows until the "diverged" rule fires. We show it without touching the artifact, in a scratch cell: `split = p1mod.make_split(CFG); design, X_tr, X_va = p1mod.base_design(split); lam, bound, lr = p1mod.learning_rate(X_tr, 1.05); res = p1mod.run_gd(X_tr, split.z_tr, np.zeros(X_tr.shape[1]), lr, 1e-10, 1e-6, 5000); print(res.stop_reason, res.iterations)`. Running the whole phase at 1.05 is not a good demonstration, because the scores of a diverged model are not finite and `write_artifact` refuses to store them. The stored sweep already contains this case in its last row.

7. **A fraction of exactly 1.0 in the sweep.** Add `1.0` to `CFG["p1"]["sweep_fractions"]` and run the Phase 1 cell. At the bound the steepest direction is multiplied by −1: it neither shrinks nor grows, so the run ends as "max_iter" with a loss that has stopped falling. This is the boundary between the two regimes.

8. **Looser tolerances.** Set `tol_loss` to 1e-6 and `tol_grad` to 1e-3 in `CFG["p1"]` and run the Phase 1 cell. It stops earlier, `oracle.max_abs_weight_diff` gets larger, and the validation R² changes only in late decimals. The live check cell uses the same `CFG`, so it stops at the same place. Then run the Phase 2 cell: the first Phase 2 loss still equals the last Phase 1 loss, because Phase 2 starts from whatever Phase 1 stored. Changing only `tol_loss` may change nothing, since both tests must hold.

9. **More degree candidates.** Set `CFG["p2"]["degree_candidates"] = [1, 2, 3, 4, 5, 6]` and run the Phase 2 cell (under a minute). Higher degrees are worse conditioned, so λ_max rises, the learning rate falls and the iteration counts in the degree table rise. The curve is flat after the chosen degree, so the rule should pick the same degree; the Phase 3 degree axis shows the same flatness up to degree 8. If the degree did change, the anchor of Phase 3 would change and everything downstream would rebuild.

10. **Remove the working day × hour block.** Set `CFG["p2"]["blocks"] = []` and run the Phase 2 cell. Validation R² falls to the value already stored in `block_ablation.without_wd_x_hr`, because the model again has a single hour profile for both day types. Do not continue to Phase 3 in this state: `ladder_specs` asserts that the Phase 2 design contains the blocks of the earlier ladder levels, and it will stop. The quick way to make the point is to read the stored ablation.

11. **Move the chronological cut.** Set `CFG["chrono"]["cut_date"] = "2012-10-01"` and run the Phase 3 cell (about ten seconds, plus the exponent check). The seeded and held-out-day columns do not move. The chronological score changes: a later cut leaves less far to extrapolate the trend and more of 2012 to learn from, so we expect a smaller drift gap, on a smaller late part that covers only autumn and winter. An earlier cut (try "2012-04-01") should widen the gap. The target level cannot change, because `pick_target` looks only at the seeded column. Phase 4 also reports chronological scores, so a real change of the cut means the long rebuild; for the demonstration Phase 3 alone is enough.

12. **Plateau tolerance of Phase 3.** Set `CFG["p3"]["plateau_tol"] = 0.0` and run the Phase 3 cell: the rule becomes "highest validation R²", our original rule, and the target may move to a higher level. Set it to 0.01 and the target moves to a simpler level. The tolerance is also the one used by `target_power_consistent`, so a tolerance of 0 can flip that flag; say so if it does. Read `target_level` and the three scores of the levels around it. A different target changes the Phase 4 design, so the full rebuild takes the long Phase 4; showing the Phase 3 table is the quick version.

13. **Remove the level added after the residual analysis.** Delete the `C6_weather_detail` entry from `CFG["p3"]["ladder"]` and run the Phase 3 cell. This shows what ADR-015 bought: compare the target's held-out-day and chronological scores with and without the level. The markdown under "Justification: target complexity" quotes ladder rows by position, so its sentence is only right for the full ladder.

14. **Ten day folds instead of five.** Set `CFG["day_block_folds"] = 10` and run the Phase 3 cell. The held-out-day means move a little and the spread between folds changes; the seeded and chronological columns stay. It shows that the folds are fixed by the calendar and not by a seed.

15. **The `l1_ratio` list.** Changing `CFG["p4"]["l1_ratios"]` means re-running Phase 4 (about 30 minutes), then Phases 5 and 6. The quick way is one path in a scratch cell: `ctx = p4mod.build_context(CFG, read_artifact("p1"), read_artifact("p2"), read_artifact("p3")); curve = p4mod.build_curve(ctx.split, 0.5, ctx, "demo"); print(max(r["val_r2"] for r in curve.rows))`. This writes no artifact. A ratio near 1 behaves like Lasso and one near 0 like Ridge; the stored table of best scores per ratio (the cell after the validation curves) shows how little the ratio matters on this design.

16. **Stability threshold 0.9.** A real change needs Phase 4 again, because the survivors feed stage B. To show the effect at once, filter the stored frequencies: `freq = p4["stability"]["l1"]["freq"]; len([n for n in p4["survivors_expanded"] if freq[n] >= 0.9])`. Fewer features survive; the ones that go are those Lasso keeps in only some resamples of days.

17. **Verdict thresholds.** The verdict functions are pure, so they can be re-applied to the stored numbers without refitting: `from src import verdicts as vd; {c: vd.basic_verdict(v["numbers"], 0.005, 0.01) for c, v in p4["column_verdicts"].items()}`. This is the verdict before the representative step. With a larger `min_delta`, columns with a small but real drop cost move from useful to redundant or uninformative.

18. **Label quantile 0.9.** Set `CFG["p5"]["quantile"] = 0.9` and run the Phase 5 cell; `cached_or_run` sees the changed configuration and recomputes (about three minutes). Only about one hour in ten is now positive, so the majority-class accuracy rises to about 0.9 and accuracy flatters the model even more, each cell has fewer positives so the AUC interval widens, and F1 should fall. The cut-off stays at 0.25, because it depends only on the cost ratio.

19. **Label without the year.** Set `CFG["p5"]["group_by"] = ["workingday", "hr"]` and run the Phase 5 cell. The positive share now differs strongly between the two years and the trend alone predicts the label well above chance: the label has become the calendar. The same case is already stored as the second row of `label_variants`, which is the instant way to show it.

20. **Cost ratio 1.** Set `CFG["p5"]["cost_ratio_miss_to_false_alarm"] = 1.0` and run the Phase 5 cell. The cut-off becomes 1/(1 + 1) = 0.5, so the main metrics equal the "at 0.5" row: higher accuracy and precision, lower recall. The weights and the AUC do not change, because the cost ratio enters only after the fit. The Phase 1 bonus has its own knob, `p1.bonus.k_under`, and is not affected.

21. **Bonus with k = 1.** Set `CFG["p1"]["bonus"]["k_under"] = 1.0` and run the Phase 1 cell, then the bonus-control cell. The loss becomes plain squared error on bikes, which is exactly what the control already fits, so the bonus model and `bonus_control.json → val.k1_model` should agree. The share of under-predicted hours moves back towards balance, but the predictions need not return to the MSE model's, because a least-squares fit on bikes aims at the mean while our fit on the transformed scale aims lower.

22. **No ridge term in the classifier.** Set `CFG["p5"]["l2_candidates"] = [0.0]` and run the Phase 5 cell. Gradient descent is expected to stop at the iteration cap instead of converging (see the first row of `l2_sweep`), and the chain table in the last notebook cell then reports that the chosen Phase 5 model did not converge. It shows why we add a small penalty.

After any drill that wrote an artifact, restore the saved `artifacts/` folder (or `git checkout -- artifacts sample_submission.csv`), reset `RECOMPUTE_ALL` and `CFG` by restarting the kernel, and run `python run.py verify-fast`.

## 4. Likely oral questions

**1. Why did you run the pipeline twice, is that allowed, and did you tune on the test set?** The first complete run (git tag `v1-submitted`) fitted the logarithm of the count. Its Phase 3 analysis and a diagnostic outside the chain showed that the log weights quiet night hours too heavily for a score that is computed on bikes. The exponent of a power transform is a tuning decision, like the degree or the penalty, and the brief lets us tune on the validation set; we made it with the seeded validation set, the held-out days and the chronological split, and `test.csv` was never involved (it is read once, at the very end, and only transformed). The second-pass Expectation cells were committed before the chain was re-run, and the original Expectation cells and the whole first pass stay visible in git. The key numbers of the first pass are in `artifacts/first_pass.json`, so the notebook can state for every expectation whether it was met. We also say plainly that this looks at the validation data once more, so the validation scores are slightly optimistic; that is why the held-out-day and chronological scores sit beside them.

**2. Why not the log?** On the log scale an error of a few bikes at 3 am counts as much as an error of a hundred bikes at 5 pm, but R² on bikes is decided by the busy hours; the log fit spent its effort in the wrong place. A straight trend on the log scale is also exponential growth, which overshoots the later months (see question 8). The power family has the log as its λ → 0 limit, so the first pass is a special case of the second and the drill with `target_power` 0.0 reproduces it.

**3. Why not raw counts?** At exponent 1 the model is additive in bikes, and the multiplicative structure of the data (a busy hour on a growing system in good weather: effects scale with the level) is lost; the exponent tables show it (`p1.json → target_power_table` and `p3.json → target_power_check`, last rows). The family lets the data choose a point between the two. The counts are also right-skewed and their spread grows with their level, which a milder transform tames.

**4. Why exponent 0.1 when Phase 1 alone prefers a larger one?** The exponent has to be the same in every phase, otherwise Phase 2 could not start from Phase 1's weights (the first-loss equality) and Phase 3 and 4 would not refine the same model. The best exponent falls as the model gains structure: the simple Phase 1 design prefers a larger exponent (`p1.json → target_power_best_here`, the table `target_power_table`), because a milder transform partly compensates for structure the model cannot express, while the richer designs prefer about 0.1 (`p3.json → target_power_check`). We fix the exponent for the model the chain ends with. Phase 1 pays a little R² for this, and its Outcome cell says so openly.

**5. Is fixing the exponent from an earlier run cheating?** It is a tuning decision made on validation data, which the brief permits, and we do not hide that it came from a full earlier run rather than from Phase 1 alone: the Phase 1 justification says so. What would be cheating is using `test.csv`, and we did not. The safeguard against a bad choice is the consistency check in Phase 3: it refits the target design for every candidate exponent and compares the one in use with the best (`target_power_consistent`: within `p3.plateau_tol` of the best seeded score). If it were false, we would have to restart the chain with another exponent. We do not move to the exponent that the chronological split likes slightly better, because the exponent is tuned on the validation set like every other choice.

**6. What is the back-transform factor now, and why is Duan gone?** The model predicts z; the inverse of an average is not the average count, so the predictions come back to bikes as cnt_hat = clip(s·q(η) − 1, 0) with q(η) = (λη + 1)^(1/λ). The candidates are `none` (s = 1) and the least-squares factor s = Σ(cnt + 1)·q / Σq², computed on the fitting rows, which is the single multiplier that minimises squared error on bikes (`p1.json → backtransform.candidates`, and we reuse the winner in every phase, recomputing s from each phase's own residuals). Duan's smearing factor, the mean of exp(z − η), is derived for a log model; it is defined for λ = 0 only (`Target.factor` raises an error otherwise), so it belongs to the first pass, where it lowered validation R² because it inflated the peaks (`first_pass.json → p1.duan_val_r2` against `none_val_r2`).

**7. What exactly did the first pass do differently?** The first pass used z = log1p(cnt), the λ → 0 limit of the same formula: predictions cnt_hat = clip(exp(η)·s − 1, 0), Duan, none or least-squares factor, and an η cap of the largest training target plus 1.0. The same code runs it with `target_power` 0.0. We kept the first pass in git and in `first_pass.json` so that every improvement can be attributed.

**8. Why did the chronological estimate improve so much?** The seeded and held-out-day scores moved only a little, but the chronological score rose and the drift gap shrank (`p3.json → estimates_target.chrono` and `gaps.target.drift` against `first_pass.json → p3`). A straight trend on the log scale is exponential growth, so the first pass over-shot the later months (`p3.json → chrono_detail.target.mean_ratio` against `first_pass.json → p3.chrono_mean_ratio_target`); on the milder scale the same linear trend grows more slowly. So most of what we called drift in the first pass was our own choice of target, not a property of the data. We predicted this in the second-pass Expectation cell and the Outcome says that it was met.

**9. What did you try in the second exploration round, and why was nothing adopted?** On top of the target design we tried twenty groups of input-only features (daylight by hour, lagged temperature and humidity, rain in the last six hours, more hour × weather products and others), month-level time effects (year-month levels, quarterly trend hinges, year × hour) and observation weights that approximate a bike-scale loss (ADR-018; numbers in `artifacts/diagnostics.json`). The acceptance rule was fixed beforehand (as in ADR-015): a change must gain on held-out days by more than two paired standard errors and must not lose more than 0.003 on the chronological split. None of the weather and hour blocks gained more than 0.001 on held-out days; the month-level effects gain on shuffled splits and lose heavily on later months, because they describe the months we have rather than time; the observation weights gain less than two standard errors and would put a loss other than the phase's own into the final model. So we adopted nothing and stopped there. The remaining error is mostly day-level (events we do not observe).

**10. Why did the recommended method change from Elastic Net to Lasso?** It did not change because Lasso became better. The three methods are tied within the paired bootstrap noise, and `recommend` keeps every method whose interval against the best contains zero and picks the best held-out-day score among them. With the new target that tie-break lands on Lasso instead of Elastic Net (`p4.json → recommended.candidates`, `best_on_validation`). It is a tie-break among methods tied within noise, not a finding about Lasso; the prediction quality is the same.

**11. Why is Ridge's best penalty on the survivors the smallest grid value?** After the column verdicts and the survivor rule, the model that is left is small compared with the number of training rows and not over-fit, so there is almost no variance to remove and the best validation score is reached at the weakest penalty: the smallest value of `ridge_alpha_range` (compare the penalty in `p4.json → final.methods.l2` with the grid). That is the expected outcome of Phase 3's diagnosis, not a bug; it also means Ridge on the survivors is close to unpenalised least squares. A best value at the edge of the grid means a still smaller penalty might score marginally higher, but the differences between penalties that small are far inside the bootstrap noise, so we did not extend the grid.

**12. Why is the hour one-hot?** Demand does not rise or fall steadily through the day, so a single number for the hour fits badly. Sine and cosine pairs come close to the dummies but each dummy weight reads directly as "the effect of this hour" (`p1.json → hour_encoding_ablation`). The dummies also make the working day × hour block of Phase 2 simple to build.

**13. Why were some columns held back until Phase 4?** `atemp`, `yr`, `instant`, `season`, `mnth` and `weekday` each repeat, exactly or almost exactly, something already in the base design. Copies make the Gram matrix singular or nearly so: the weights are then not unique and plain gradient descent slows down. Phase 4 puts them all back, where the penalised solvers can cope with copies and the verdicts are made by models (ADR-003).

**14. Why half the stability bound?** The bound 2/λ_max is where the steepest direction stops shrinking. At half of it every direction's error is multiplied by a factor between 0 and 1, so nothing overshoots, the loss can only fall, and a factor of two is left as margin. The sweep shows the alternatives: tiny fractions stall, 0.9 is faster with almost no margin, 1.05 diverges. The exponent does not enter: the bound depends on the design only.

**15. Why does the first Phase 2 loss equal the last Phase 1 loss?** The expanded design keeps the Phase 1 columns with the Phase 1 scaler, every new weight starts at zero, and both phases fit the same z (the exponent is read from `p1.json`). A zero weight switches its column off, so the expanded model starts as exactly the Phase 1 model. The code asserts this to 1e-9; if someone built Phase 2 independently, or with another exponent, the assertion and the chain check would fail.

**16. Why did the degree alone look flat?** Raising the power of temperature beyond the chosen degree adds almost nothing, on any validator (`p3.json → diagnosis.degree_axis_range`). A flat line is not proof of a good fit: the missing structure was in interactions, not in powers. That is why we built the wider ladder, and had we varied only the degree we would have wrongly concluded "reasonably fit".

**17. Why do the seeded and held-out-day estimates agree while the chronological one is lower, and which do we trust?** Sharing days between training and validation hardly helps a linear model with a few hundred weights, because it cannot recognise an individual day; so the leakage gap is small (`p3.json → gaps`). The chronological gap is drift: a straight trend on the transformed scale extrapolates growth too steeply into months the model has not seen (`chrono_detail.mean_ratio`), and the steeper the transform's growth, the worse (question 8). For genuinely future periods we trust the chronological estimate. For the hidden test, which is the 20th of each month inside the observed two years, the held-out-day estimate is the closest match.

**18. Does the chronological result change the diagnosis?** Not in kind: the simpler models are worse on all three validators, so the Phase 2 model is under-fit either way. It does change how far we go, because it is the split that punishes the levels above the target most clearly, especially blocks that let the trend vary by hour.

**19. Why did we change rules after first runs, and add a ladder level after a residual analysis? Is that still legitimate?** ADR-013: plain "highest validation R²" picked a much larger level on a difference far inside the bootstrap noise, so we replaced it with the "simplest within 0.001" rule Phase 2 already used. ADR-014: the first Phase 4 run showed that the full candidate design is not the one our own verdicts endorse, so the recommended model is now fitted on the survivors. ADR-015: residuals of the first target were concentrated in working-day rush hours and rainy hours, so we added one level with weather detail and weather memory. It is legitimate because each change is written down with its evidence, is visible in the git history, never touched `test.csv`, and the new level was required to win on held-out days and not lose chronologically, the two estimates not used to design it. ADR-017 and ADR-018 follow the same pattern for the second pass. We say openly that these looks make the seeded validation score more optimistic.

**20. Why did regularisation "do nothing", and why is that expected?** Phase 3 handed over a model that is not over-fit: thousands of training rows for a few hundred standardised columns leave little variance to remove. So the best penalties are small and all three methods lie within the paired bootstrap noise of the unpenalised fit (`p4.json → comparisons`). We predicted this in the Expectation cell, in both passes. The penalty matters where there is variance, on the over-fit top level (`rich_check`), and its real use for us was to rank and select columns.

**21. Why do Lasso zeros alone not decide a verdict?** When two columns carry the same information, which copy Lasso keeps is partly arbitrary and can change with the resample or the penalty. So a zero is one piece of evidence. The verdict comes from drop tests with paired intervals, alone and with the partner group, and from the solo R²; Lasso counts and selection frequencies are reported beside them. Every verdict was the same in the second pass as in the first.

**22. Why is `dteday` the kept representative, and why are `yr`, `instant`, `season` and `mnth` redundant?** These columns all describe time: `yr` and `instant` the level, `season` and `mnth` the time of year. Dropping any one alone costs nothing because the others cover for it, while dropping a whole group hurts. The rule then keeps the member that explains most on its own, which is `dteday` through its trend and day-of-year terms; the others are "carried by" it (`p4.json → column_verdicts`, fields `representative` and `carried_by`). ADR-014 adds that carrying the copies along made the model worse on later months.

**23. What are the "weather memory" columns, and why are they not leakage?** They are the weather situation one hour earlier and the worst situation of the three hours before, on the idea that wet roads keep riders away after the rain has stopped. They read only `dteday`, `hr` and `weathersit`, never `cnt`, and only hours in the past. No statistic is fitted on validation or test rows: the dummies are standardised with training means like every other column. For the hidden file the timeline joins the input columns of both files so that the first hour of a hidden day can look back; no label is involved.

**24. Why is the label defined per year × day type × hour?** A single global threshold makes "high demand" the timetable: hour dummies alone then predict it very well and the classifier tells the operator nothing new. A threshold per day type and hour fixes that but, because demand grew strongly, makes nearly all positives fall in the second year, so the label becomes the calendar. With the year in the cell, hour alone and trend alone are near chance and the positive share is the same in both years (`p5.json → label_variants`). The limitation is that a new year has no cell of its own; the thresholds would have to be rolled forward.

**25. Why is the cut-off 0.25, and why did accuracy go down?** Missing a high-demand hour costs three times a false alarm, so an alarm pays when p·3 > 1 − p, that is p > 1/(1 + 3) = 0.25. A lower cut-off raises more alarms: recall goes up, precision and accuracy go down. Accuracy counts both errors equally, which is exactly what the operator does not do, so judged by cost the 0.25 point is better (`p5.json → metrics.cost` against `metrics_at_0_5.cost`). We kept the theoretical cut-off and did not tune it on the validation rows.

**26. Why do we refit on train + validation, and what stays fixed?** The learning curve of the target was still rising at full size, so more rows should help a little, and the validation rows are days from the same months as the hidden ones. Fixed: the feature subset, λ, `l1_ratio`, the target exponent, the back-transform method, the scaler and the humidity fill. Only the weights, the intercept and the back-transform factor are recomputed. Before that, the code refits on the training rows alone and asserts that it reproduces the Phase 4 weights.

**27. What can the model not do?** It extrapolates growth as a straight line on the transformed scale, which is milder than the log but still a trend that Phase 3 showed to overshoot somewhat beyond the observed period. It has never seen a zero-demand hour, because quiet hours are missing rather than recorded as zero, nor weather outside the observed range, nor holidays other than those in the training days. Whole-day effects such as events are not in the data; the remaining error is mostly day-level (ADR-018). For a later period we would expect something nearer the chronological estimate than the validation one.

**28. What is the eta cap, and has it changed?** In the first pass it was the largest training target plus 1.0, which protected `exp`. Now it is `Target.eta_cap`: the η at which q reaches e times (the largest training count + 1), that is forward(e·(max + 1) − 1); for λ = 0 this is the old value. Evaluation predictions in `validation.fit_predict`, the bonus fit (`gd_asym`) and the final prediction (`predict.unclipped_bikes`) are capped there, so q stays finite when an over-flexible design is fitted on very few rows, as in the smallest learning-curve subsets of the top ladder level. It uses the training counts only, so it is not a leak. We do not store how many rows it touches, which is a weakness we should admit if asked.

**29. Why did some Lasso fits not converge?** A few solutions used all 20000 sweeps without meeting the 1e-7 tolerance; the count and the places are in `p4.json → not_converged` and `not_converged_final`. The design contains exactly duplicated information, and along such directions coordinate descent can only crawl. We count these solutions instead of hiding them and check that none is a chosen model; `predict.fit_penalised` additionally asserts convergence at the submitted λ.

**30. Does tuning on the validation set make our scores optimistic?** Yes, slightly, and we say so. The exponent, the back-transform, the degree, the power columns, the ladder level, the penalties, the verdicts and stage B were all chosen on the same validation rows, as the brief prescribes. That is why every validation score is shown with the held-out-day and chronological scores beside it, and why the recommendation rule uses held-out days to break ties.

**31. Are the day folds a forbidden third split?** We read the brief as forbidding a second seeded split. The day folds use no seed, lie entirely inside the training portion, and never touch a validation row. Every reported score and every tuning decision uses the seeded validation set; the folds only confirm (AMBIGUITIES A3).

**32. Why closed-form fits in Phase 3 when gradient descent is required?** Gradient descent is required for Phases 1 and 2, and there we use it for every reported model. Phase 3 needs several hundred fits (more now, with the exponent check) and the largest levels would need tens of thousands of iterations each. The closed form reaches the same optimum, which the oracle checks in Phases 1 and 2 and `anchor_check` in Phase 3 demonstrate (ADR-009).

**33. Is a closed-form Ridge "implemented"?** Yes: we wrote the normal equations with the penalty ourselves, and Lasso and Elastic Net are our own coordinate descent. scikit-learn appears only inside the tests as an oracle and for the single `train_test_split`.

**34. Some columns are exact combinations of others. What does that do to the weights?** `holiday` is determined by the weekday dummies and `workingday`, so from the weekday level on the design is rank-deficient and only the 1e-8 ridge term makes the solution unique. Predictions are unaffected, but the individual weights of those columns should not be interpreted.

**35. Is the "noise floor" an upper limit?** No. It is an in-sample cell-mean score with many cells holding a single row, so we call it a benchmark. Our target matches it on unseen rows. A diagnostic tree model kept outside the pipeline (`exp/v2/`, numbers in `artifacts/diagnostics.json`) scores higher on seeded and held-out days, and lower chronologically; the gap between its seeded and held-out-day scores shows the within-day leakage that a flexible model uses and ours cannot. That model is never used for any prediction we submit (H13).

**36. How wide is the uncertainty of our R²?** The bootstrap interval of each validation score is stored next to it (for example `p3.json → estimates_target.seeded.lo` and `hi`). It resamples rows as if they were independent, although hours of one day share a common offset, so the true interval is somewhat wider. Differences between models are judged with the paired bootstrap, which is far tighter than comparing two separate intervals.

**37. Why is the bonus loss squared and on bikes, not absolute or on the transformed scale?** The cost is paid in bikes. The weighted squared error is the direct modification of MSE and has a gradient everywhere; the absolute version has only a sub-gradient at zero. The model is ŷ = q(xw) − 1, so the gradient picks up the factor q′(η) = q(η)^(1−λ) from the chain rule (exp(η) for the log). We report the absolute cost for both models as a sensitivity (`p1.json → bonus.val`), and the k = 1 control (`bonus_control.json`) separates the effect of fitting on bikes from the effect of the asymmetry.

**38. How do we know our gradients are right?** Finite-difference checks for the MSE gradient, the asymmetric gradient (now including the power-target chain-rule factor) and the log-loss gradient are stored in the artifacts (`gradient_check*` keys in `p1.json` and `p5.json`; `p1.json → bonus.gradient_check`). The tests compare our solvers with a reference library, and the Phase 1 and 2 weights are compared with the exact least-squares solution.

**39. Did you really run the gradient descent, or did you just load the stored weights?** The Phase 1 cell may load the stored artifact, but the next cell (the live check) runs our gradient descent again from zero weights with the same settings on the machine in front of us and asserts agreement with the stored weights to 1e-8. `RECOMPUTE_ALL = True` recomputes the whole chain. The notebook also runs in an empty folder: `python -X utf8 tools/standalone_check.py`.

**40. What if the examiner recomputes our seed?** `common.team_seed` reproduces the worked example of the brief, and the seed is stored in every artifact. It is derived from `team_ids` in `config.yaml`, so we must be certain those are our registered IDs; a different roster gives a different split and every number changes.

## 5. Where each requirement lives

| Req. | Requirement | Notebook section | Module | Artifact key |
|---|---|---|---|---|
| R1 | Five chained phases in order | Sections Phase 1 to Phase 5 | `src/phases/p1.py` to `p5.py` | `artifacts/p1.json` to `p5.json` exist |
| R2 | Each phase consumes the previous artifact | Hand-over cell of Phase 2; "The chain" cells of Phases 3 to 5; last cell of Phase 6 | `common.write_artifact`, `common.Target.from_artifact`, `poly.lift_weights`, `p3.check_anchor`, `p4.pick_design`, `p5.survivor_design` | `upstream_sha256` in every file; `p1.target_power`; `p2.init_loss`, `p2.init_weights_source`; `p3.anchor`, `p3.anchor_check`; `p4.design_from`; `p5.features` |
| R3 | Expectation, Outcome and justification per open choice | Original and second-pass Expectation cells, Justification and Outcome cells of every phase | `nb/p1_gd.py` to `nb/p5_logistic.py` | git history of the `nb/` files; `artifacts/first_pass.json` for the first-pass comparison |
| R4 | Team seed, one seeded split, one chronological split | Setup cells; "Justification: the chronological cut" | `common.team_seed`, `common.seeded_split`, `common.chrono_split` | `seed` in every file; `p3.cut_date`, `p3.chrono_detail` |
| R5 | Fits on train only; test only transformed at the end | Phase 0 "Splits"; Phase 6 | `features.Design.fit`, `labels.fit_thresholds`, `common.load_test`, `predict.build` | `p1.scaler`; `p5.threshold_rule.fitted_on`; `p6.refit_on` |
| R6 | No external data | Setup cells | `common.load_train`, `common.load_test` | `paths` in `config.yaml` |
| R7 | Own gradient descent, configurable learning rate, convergence shown | Phase 1, "Justification: learning rate and stopping rule"; the live check cell | `gd.gradient_descent`, `p1.learning_rate`, `p1.lr_sweep`, `p1.run_gd` | `p1.lr`, `p1.iterations`, `p1.stop_reason`, `p1.lr_sweep`, `p1.oracle` |
| R8 | Expansion fitted with own GD from the Phase 1 weights | Phase 2, "initialisation", "what we expanded", "degree" | `poly.lift_weights`, `p2.fit_expanded`, `p2.degree_sweep` | `p2.degree`, `p2.feature_names`, `p2.degree_sweep`, `p2.block_ablation` |
| R9 | Empirical diagnosis and chronological comparison | Phase 3, all Justification cells (including "exponent check") and Outcome | `validation.three_validators`, `p3.pick_target`, `p3.classify`, `p3.target_power_check` | `p3.ladder`, `p3.degree_axis`, `p3.learning_curves`, `p3.estimates`, `p3.gaps`, `p3.diagnosis`, `p3.target_complexity`, `p3.target_power_consistent` |
| R10 | L1, L2, Elastic Net compared; verdict per column | Phase 4, "search ranges", "fair comparison", "choice of lambda", "verdict rule", "survivor rule" | `regularization.py`, `verdicts.py`, `p4.column_verdicts`, `p4.survivors` | `p4.methods`, `p4.comparisons`, `p4.cv`, `p4.column_verdicts`, `p4.survivors_expanded` |
| R11 | Label, logistic on survivors, metrics, retrospective | Phase 5, "label rule", "metrics", "operating threshold", "Pipeline retrospective" | `labels.py`, `logistic.fit_logistic`, `p5.label_variant`, `p5.retrospective` | `p5.threshold_rule`, `p5.metrics`, `p5.label_variants`, `p5.retrospective` |
| R12 | Bonus: asymmetric loss | Phase 1, "Justification: bonus" and the bonus control | `gd_asym.asym_loss`, `asym_grad`, `fit_asymmetric`, `common.Target.dq_deta`, `bonus_control.run` | `p1.bonus`, `bonus_control.json` |
| R13 | R² and RMSE on validation; accuracy, F1 and ROC-AUC | Score cells of every phase | `common.r2`, `common.rmse`, `logistic.prf`, `logistic.roc_auc` | `val_r2`, `val_rmse` in `p1`, `p2`, `p4.methods`; `p3.estimates`; `p5.metrics` |
| R14 | Submission from the recommended regression model | Phase 6 | `predict.build`, `predict.run` | `p4.recommended`, `p6.model`, `p6.model_a_gap_to_p4`, `sample_submission.csv` |
| R15 | Notebook runs top to bottom; report at most 6 pages; explainable | Whole notebook; setup cell and `phase()` helper | `run.py` targets `nb`, `report`, `verify`; `tools/embed_bundle.py`, `tools/standalone_check.py` | `rush_hour.ipynb`, the report PDF, this file |
