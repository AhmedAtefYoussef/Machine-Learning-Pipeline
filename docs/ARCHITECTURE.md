# ARCHITECTURE (chief). Interface contract: specs refer to this file by section. Signatures are binding; internals are the coder's.

## 0 Environment (this machine)
Windows, Git Bash, Python 3.14 as `python` (NOT `python3`), no `make`, no pandoc. `run.py` replaces the Makefile (same target names).
Every python process: `-X utf8`; env `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, `PYTHONPATH=.`. Open every text file with `encoding="utf-8"`, write with `newline="\n"`.
Kit files (`CLAUDE.md`, `.claude/`, `tools/`, `Makefile`) are frozen: never edit. The string `test.csv` may appear only in `config.yaml`, `src/common.py`, `src/predict.py`, `nb/p6_submission.py` (leak_scan L2 flags it elsewhere, docstrings included).
leak_scan L3 flags, in P1/P2 files (`src/gd*.py`, `src/poly.py`, `src/phases/p1.py`, `src/phases/p2.py`, `nb/p1*`, `nb/p2*`): any attribute call `.predict(...)`, any `.fit(...)` on a receiver named model/clf/reg/lr/est, any sklearn.linear_model import. L1 flags `.fit*(...)` whose argument names contain val/valid/test/te. Name things accordingly (`design.fit(train_df)`).

## 1 Data flow
config.yaml → seed → `seeded_split` (the ONE seeded call) → train_df / val_df
P1 `artifacts/p1.json` (weights, scaler, lr, iterations) → P2 lifts weights, GD on expanded design → `p2.json` (degree, feature list)
→ P3 ladder + validators → `p3.json` (diagnosis, target_complexity) → P4 ridge/lasso/enet on the target design + candidate columns → `p4.json` (λ per method, survivors, verdicts, recommended model)
→ P5 logistic on survivors → `p5.json` → `src/predict.py` → `sample_submission.csv` + `artifacts/p6.json`.
Each phase module: `run(cfg: dict | None = None) -> dict` loads the upstream artifact FROM DISK, computes, writes its artifact via `common.write_artifact`, returns the payload. `python -m src.phases.pN` calls `run()`.

## 2 Determinism
float64 everywhere; no RNG except `np.random.default_rng(seed + k)` with a documented k; GD starts from zeros (P1) or lifted P1 weights (P2); JSON written with `sort_keys=True, indent=1`; floats stored at full precision (`repr` round-trip). Same config → byte-identical artifact (no timestamps inside `pN.json`).

## 3 Target (ADR-001, ADR-002)
`z = log1p(cnt)`. Models are linear in z. Prediction on the bike scale: `cnt_hat = clip(exp(eta) * s - 1, 0, None)` where `s` is the back-transform factor. P1 evaluates methods `none` (s=1), `duan` (s = mean(exp(z - eta)) on train), `ls` (s = Σ(cnt+1)·e^eta / Σ e^(2 eta) on train) on validation, stores the chosen method name in `p1.json["backtransform"]["method"]`; later phases reuse that METHOD (s recomputed from their own train residuals). All R²/RMSE are on the bike scale (`cnt`), plus `*_log` variants on z.

## 4 Features (`src/features.py`)
`raw_columns(df, hum_fill) -> DataFrame` unscaled named columns (ALL candidates, same index as df):
- `hr_1..hr_23` (one-hot, hr 0 = reference) · `workingday` · `holiday` · `ws_2`, `ws_3` (weathersit with 4 merged into 3; 1 = reference)
- `temp` · `hum` (rows with hum == 0 replaced by `hum_fill` = median of train hum > 0) · `windspeed`
- `trend` = days since 2011-01-01 (float) · `doy_s1, doy_c1, doy_s2, doy_c2` = sin/cos(2πk·dayofyear/365.25)
- candidates, not in the P1 base: `atemp` · `yr` · `instant` · `se_2..se_4` (season) · `mn_2..mn_12` (mnth) · `wk_1..wk_6` (weekday, 0 = Sunday reference)
`BASE = hr_1..hr_23, workingday, holiday, ws_2, ws_3, temp, hum, windspeed, trend, doy_s1, doy_c1, doy_s2, doy_c2` (35 columns).
- weather memory (ADR-015; frame columns `ws_lag1`, `wet3` made by `common.add_weather_memory` at load time, inputs only): `wslag1_2`, `wslag1_3`, `wet3_2`, `wet3_3`; blocks `wx_detail` = hum^2, hum^3, windspeed^2, temp*windspeed (4) and `ws_memory` = the four memory columns (4). `load_test` builds the timeline from both files' input columns.
`sources(feature_name) -> frozenset[str]` original column(s) behind an expanded feature: hr_* → hr; ws_* → weathersit; trend, doy_* → dteday; se_* → season; mn_* → mnth; wk_* → weekday; `a^k` → sources(a); `a*b[*c]` → union. `bias` → ∅.

`DesignSpec` (frozen dataclass; JSON-serialisable via `to_dict`/`from_dict`): `base: tuple[str]` (first-order columns), `power_cols: tuple[str]`, `degree: int` (1 = no powers), `blocks: tuple[str]`.
Blocks (products of STANDARDISED first-order columns; any first-order column a block needs is added to the first-order set automatically, after `base`, in registry order):
`wd_x_hr` workingday*hr_h (23) · `hr_x_temp` (23) · `hr_x_hum` (23) · `wk` wk_1..6 first-order (6) · `wk_x_hr` wk_d*hr_h (138) · `wd_x_hr_x_temp` workingday*hr_h*temp (23) · `hr_x_ws` hr_h*ws_2, hr_h*ws_3 (46) · `wd_x_wx` workingday*{temp,hum,ws_2,ws_3} (4) · `hr_x_doy` hr_h*{doy_s1,doy_c1} (46) · `hr_x_trend` (23) · `mn` mn_2..12 first-order (11) · `mn_x_hr` (253) · `mn_x_wd_x_hr` mn_m*workingday*hr_h (253) · candidates first-order: `c_atemp`, `c_yr`, `c_instant`, `c_season` (se_2..4), `c_mnth` (= `mn`), `c_weekday` (= `wk`).
Names: powers `temp^2`, products `workingday*hr_7`, `mn_3*workingday*hr_7`. Column order: bias, first-order (base then added), powers (col-major: temp^2,temp^3,hum^2,…), blocks in spec order.

`class Design(spec)`:
- `fit(train_df, base_scaler: dict | None = None) -> Design` learns `hum_fill`, first-order mean/std (ddof=0; std 0 → 1), then derived columns computed from standardised first-order z, then their own mean/std. If `base_scaler` (`{"names","mean","std","hum_fill"}`, as stored in p1.json) is given, those names reuse exactly those statistics (P1→P2 lift) and only the rest is fitted.
- `transform(df) -> np.ndarray` shape (n, 1+p), column 0 = ones. · `names -> list[str]` (starts with `bias`) · `scaler_dict() -> dict` (`names, mean, std, hum_fill` for all non-bias columns; JSON floats) · `fit_transform(train_df, ...)`.
- `subset(names) -> column indices` helper for P4 drops / P5 survivors.

## 5 `src/common.py`
`load_config(path="config.yaml") -> dict` · `team_seed(ids) -> int` (asserts the 41698 self-test) · `set_threads()` · `load_train(cfg) -> DataFrame` (dteday parsed) · `load_test(cfg)` · `seeded_split(df, seed, test_size=0.20) -> (train_df, val_df)` (single sklearn `train_test_split`, nothing else) · `chrono_split(df, cut_date) -> (early_df, late_df)` · `day_block_folds(df, k=5) -> np.ndarray[int]` fold = rank of the row's date among the sorted unique dates of `df`, mod k (date-determined, no seed) · `to_target(cnt) = log1p` · `back_factor(method, z_train, eta_train, cnt_train) -> float` · `from_target(eta, s) -> cnt_hat` (clip ≥ 0) · `r2(y, yhat)`, `rmse(y, yhat)` · `bootstrap_r2(y, yhat, seed, B=1000) -> (lo, hi, se)` percentile 95% · `paired_bootstrap_delta_r2(y, yhat_a, yhat_b, seed, B=1000) -> (delta, lo, hi)` · `write_artifact(name, payload, upstream=None, cfg=None) -> path` adds `seed`, `config_sha256` (sha256 of config.yaml bytes), `upstream_sha256` (sha256 of `artifacts/<upstream>.json` bytes or null) · `read_artifact(name) -> dict` · `sha256_file(path)`.

## 6 `src/gd.py` (own optimiser; numpy only)
`mse_loss(X, y, w) -> float` = (1/2n)‖Xw − y‖² · `mse_grad(X, y, w)` = (1/n)Xᵀ(Xw − y) · `lambda_max(X) -> float` largest eigenvalue of XᵀX/n (`np.linalg.eigvalsh`)
`gradient_descent(loss_fn, grad_fn, w0, lr, tol_loss=1e-10, tol_grad=1e-6, max_iter=50000, record_every=1) -> GDResult(weights, iterations, stop_reason, loss_history, grad_norm_final, loss_final)`; `loss_fn(w)`, `grad_fn(w)` closures. Stop `converged` when |ΔL| ≤ tol_loss·max(L_prev, 1e-300) AND ‖∇‖₂ < tol_grad; `diverged` when loss is non-finite or > 1e6 × initial loss; else `max_iter`. `loss_history[0]` = loss at w0.
`gradient_check(loss_fn, grad_fn, w, n_coords=20, eps=1e-6, seed=0) -> float` max relative error, central differences.
`src/gd_asym.py` bonus (ADR-008): raw-scale asymmetric loss for the log-link model, its gradient, `fit_asymmetric(...)` (backtracking GD), see S-0-02.
`src/poly.py`: `lift_weights(w_small, names_small, names_big) -> np.ndarray` (P1 weights in matching slots, zeros elsewhere; asserts every small name exists in big).

## 7 `src/regularization.py` (own solvers; X has NO bias column here, standardised; y centred by the caller via `fit_intercept` helper)
Objective (sklearn scaling): (1/2n)‖y − b − Xw‖² + α·ρ‖w‖₁ + (α/2)(1 − ρ)‖w‖², intercept b unpenalised. ρ=0 ridge, ρ=1 lasso.
`ridge_closed_form(X, y, alpha) -> (b, w)` solves (XᵀX/n + αI)w = Xᵀ(y−ȳ)/n on centred columns · `enet_cd(X, y, alpha, l1_ratio, w0=None, tol=1e-9, max_sweeps=5000) -> (b, w, sweeps)` cyclic coordinate descent using the Gram matrix (covariance updates), stop when max|Δw_j| < tol · `enet_path(X, y, alphas_desc, l1_ratio) -> W (len(alphas), p), b` warm starts · `alpha_max(X, y, l1_ratio)` smallest α with all-zero w · `stability_selection(X, y, alpha, l1_ratio, groups, seed, B=50) -> freq (p,)` bootstrap over `groups` (dates).
`src/logistic.py`: `sigmoid` (stable) · `logloss(X, y, w, l2=0)` · `logloss_grad` (bias column 0 unpenalised) · `fit_logistic(X, y, l2=0, lr=None, tol_loss=1e-10, tol_grad=1e-6, max_iter=50000) -> GDResult` using `gd.gradient_descent`, default lr = 1/(λ_max/4 + l2) · metrics: `roc_auc(y, p)` (rank formula, ties averaged), `pr_auc`, `confusion(y, p, thr)`, `prf(y, p, thr)` (accuracy, precision, recall, f1), `calibration_table(y, p, bins=10)`.

## 8 `src/validation.py`
`fit_linear(X, y_z, alpha=1e-8) -> w` closed-form normal equations with bias unpenalised (P3 sweeps: hundreds of fits; same optimum GD reaches, shown in P1/P2 oracle gap).
`evaluate_spec(spec, fit_df, eval_df, back_method, alpha=1e-8) -> dict(p, train_r2, train_rmse, r2, rmse, train_r2_log, r2_log)` (Design fitted on fit_df only).
`three_validators(spec, train_df, val_df, train_all, cut_date, back_method, k=5) -> dict(seeded{…}, day_block{mean r2, sd, folds[], train_r2}, chrono{…})`. day-block = folds INSIDE train_df.
`learning_curve(spec, train_df, val_df, fractions, seed, back_method)` nested subsets of a fixed permutation of train DATES.
`noise_floor(train_df) -> dict` 1 − within-cell SS / total SS of cnt for cells (yr, mnth, workingday, hr, weathersit≤3), plus count of cells and singleton share.

## 9 Artifact keys (superset of `tools/chain_check.py` header; every number quoted in notebook/report must exist here)
Defined per phase in its spec. chain_check-required keys must be present verbatim. `p1.scaler = {names, mean, std, hum_fill}`.

## 10 Ownership
src/common.py, src/features.py, run.py: coder-A · src/gd.py, gd_asym.py, poly.py: coder-B · src/regularization.py, logistic.py: coder-C · tests/: qa · src/validation.py, src/phases/*, src/predict.py, nb code cells: phase coders per TASKS.yaml · docs/, STATE.md, handoff/specs, nb markdown: chief.
Agents work in the main checkout on disjoint `paths_owned` (no worktrees on this OneDrive path); nobody but the chief runs `git commit`.

## 11 Design freezes
F1 (after P1 gate): BASE columns, cleaning (hum fill, weathersit merge), target = log1p, back-transform method, standardisation. F2 (after P3 gate): target complexity, validators, chronological cut.
