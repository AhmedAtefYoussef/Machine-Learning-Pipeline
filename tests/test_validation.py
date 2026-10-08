"""Tests for src.validation (S-3-01 contract, ARCHITECTURE section 8).

Oracles: np.linalg.lstsq on the standardised design (Design from src.features is the shared, separately
tested transformer), the back-transform formulas of ARCHITECTURE section 3, and hand-computed closed forms.
Small designs (BASE + wd_x_hr) on a subsample of days keep every test fast.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src import common, validation
from src.common import Target
from src.features import BASE, Design, DesignSpec

SPEC = DesignSpec(base=tuple(BASE), power_cols=(), degree=1, blocks=("wd_x_hr",))
T_DUAN = Target(0.0, "duan")   # the log target with Duan's smearing factor (the v1 setting)
TOL = 1e-5          # GD-free closed forms: ridge alpha = 1e-8 versus plain lstsq


# ---------------------------------------------------------------- oracle
def _r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def _oracle(spec, fit_df, eval_df, method):
    d = Design(spec).fit(fit_df)
    Xf, Xe = d.transform(fit_df), d.transform(eval_df)
    zf = np.log1p(fit_df["cnt"].to_numpy(float))
    ze = np.log1p(eval_df["cnt"].to_numpy(float))
    w = np.linalg.lstsq(Xf, zf, rcond=None)[0]
    ef, ee = Xf @ w, Xe @ w
    cf = fit_df["cnt"].to_numpy(float)
    ce = eval_df["cnt"].to_numpy(float)
    if method == "none":
        s = 1.0
    elif method == "duan":
        s = float(np.mean(np.exp(zf - ef)))
    elif method == "ls":
        s = float(np.sum((cf + 1) * np.exp(ef)) / np.sum(np.exp(2 * ef)))
    else:
        raise ValueError(method)
    pf = np.clip(np.exp(ef) * s - 1, 0, None)
    pe = np.clip(np.exp(ee) * s - 1, 0, None)
    return {"p": Xf.shape[1], "train_r2": _r2(cf, pf), "train_rmse": float(np.sqrt(np.mean((cf - pf) ** 2))),
            "r2": _r2(ce, pe), "rmse": float(np.sqrt(np.mean((ce - pe) ** 2))),
            "train_r2_log": _r2(zf, ef), "r2_log": _r2(ze, ee)}


def _oracle_power(spec, fit_df, eval_df, power):
    """Independent power-target oracle: lstsq on z, least-squares factor on q = (power*eta + 1)^(1/power)."""
    d = Design(spec).fit(fit_df)
    Xf, Xe = d.transform(fit_df), d.transform(eval_df)
    cf, ce = fit_df["cnt"].to_numpy(float), eval_df["cnt"].to_numpy(float)
    zf = ((cf + 1) ** power - 1) / power
    w = np.linalg.lstsq(Xf, zf, rcond=None)[0]
    qf, qe = (power * Xf @ w + 1) ** (1 / power), (power * Xe @ w + 1) ** (1 / power)
    s = float(np.sum((cf + 1) * qf) / np.sum(qf * qf))
    return {"r2": _r2(ce, np.clip(s * qe - 1, 0, None)), "train_r2": _r2(cf, np.clip(s * qf - 1, 0, None))}


# ---------------------------------------------------------------- data (every 4th date of the training file)
@pytest.fixture(scope="module")
def sub_all(train_all):
    dates = np.sort(train_all["dteday"].unique())
    keep = dates[::4]
    return train_all[train_all["dteday"].isin(keep)].reset_index(drop=True)


@pytest.fixture(scope="module")
def sub_split(sub_all, seed):
    return common.seeded_split(sub_all, seed)


@pytest.fixture(scope="module")
def cut(cfg):
    return cfg["chrono"]["cut_date"]


# ---------------------------------------------------------------- fit_linear
def _problem(rng, n=400, p=5, offset=50.0, center=False):
    F = rng.normal(size=(n, p)) * rng.uniform(0.5, 2.0, size=p) + rng.uniform(-1, 1, size=p)
    if center:
        F = F - F.mean(axis=0)
    X = np.column_stack([np.ones(n), F])
    w_true = np.concatenate([[offset], rng.normal(size=p)])
    y = X @ w_true + 0.1 * rng.normal(size=n)
    return X, y


def test_fit_linear_equals_lstsq():
    X, y = _problem(np.random.default_rng(0))
    w = validation.fit_linear(X, y)
    ref = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.max(np.abs(w - ref)) < 1e-6


def test_fit_linear_matches_default_alpha_signature_and_shape():
    X, y = _problem(np.random.default_rng(1), n=200, p=3)
    w = validation.fit_linear(X, y, alpha=1e-8)
    assert w.shape == (4,) and np.isfinite(w).all()
    np.testing.assert_array_equal(w, validation.fit_linear(X, y))


def test_fit_linear_does_not_penalise_the_bias():
    # centred features: the intercept of any ridge with a free bias is exactly mean(y), at any alpha
    X, y = _problem(np.random.default_rng(2), center=True)
    ref = np.linalg.lstsq(X, y, rcond=None)[0]
    w = validation.fit_linear(X, y, alpha=1e4)
    assert abs(w[0] - y.mean()) < 1e-8
    assert np.linalg.norm(w[1:]) < np.linalg.norm(ref[1:])             # slopes are shrunk
    # closed form with I' = diag(0, 1, ..., 1); the scaling of alpha (XtX + a I' or XtX + n a I') is not fixed
    X, y = _problem(np.random.default_rng(3), center=False)
    a = 0.5
    Ip = np.eye(X.shape[1])
    Ip[0, 0] = 0.0
    w = validation.fit_linear(X, y, alpha=a)
    cands = [np.linalg.solve(X.T @ X + a * Ip, X.T @ y), np.linalg.solve(X.T @ X + len(y) * a * Ip, X.T @ y)]
    assert min(np.max(np.abs(w - c)) for c in cands) < 1e-8
    # a penalised bias would pull the intercept (about 50) visibly toward zero
    assert w[0] > 40


# ---------------------------------------------------------------- evaluate_spec
@pytest.mark.parametrize("method", ["none", "duan", "ls"])
def test_evaluate_spec_equals_oracle_on_bike_scale(sub_split, method):
    fit_df, eval_df = sub_split
    res = validation.evaluate_spec(SPEC, fit_df, eval_df, Target(0.0, method))
    ref = _oracle(SPEC, fit_df, eval_df, method)
    for k in ("train_r2", "train_rmse", "r2", "rmse", "train_r2_log", "r2_log"):
        assert res[k] == pytest.approx(ref[k], abs=TOL, rel=TOL), k
    assert res["p"] in (ref["p"], ref["p"] - 1)                        # with or without the bias column
    assert res["rmse"] > 5.0                                           # bike scale (cnt sd ~ 180), not log scale


def test_evaluate_spec_power_target_equals_oracle(sub_split):
    fit_df, eval_df = sub_split
    res = validation.evaluate_spec(SPEC, fit_df, eval_df, Target(0.1, "ls"))
    ref = _oracle_power(SPEC, fit_df, eval_df, 0.1)
    for k in ("train_r2", "r2"):
        assert res[k] == pytest.approx(ref[k], abs=TOL, rel=TOL), k


def test_evaluate_spec_fits_on_fit_frame_only(sub_split):
    fit_df, eval_df = sub_split
    base = validation.evaluate_spec(SPEC, fit_df, eval_df, T_DUAN)
    wild = eval_df.assign(temp=eval_df["temp"] * 5 + 3, hum=1.0, windspeed=eval_df["windspeed"] + 10)
    other = validation.evaluate_spec(SPEC, fit_df, wild, T_DUAN)
    for k in ("p", "train_r2", "train_rmse", "train_r2_log"):
        assert other[k] == base[k], k                                  # training scores cannot see the evaluation frame
    assert other["r2"] != base["r2"]                                   # power: the evaluation did change
    shrunk = eval_df.iloc[: len(eval_df) // 3]
    again = validation.evaluate_spec(SPEC, fit_df, shrunk, T_DUAN)
    for k in ("train_r2", "train_rmse", "train_r2_log"):
        assert again[k] == base[k], k
    # the evaluation labels play no part in the fit either
    relab = eval_df.assign(cnt=eval_df["cnt"].iloc[::-1].to_numpy())
    assert validation.evaluate_spec(SPEC, fit_df, relab, T_DUAN)["train_r2"] == base["train_r2"]


def test_evaluate_spec_deterministic(sub_split):
    fit_df, eval_df = sub_split
    np.random.seed(1)
    a = validation.evaluate_spec(SPEC, fit_df, eval_df, T_DUAN)
    np.random.seed(2)
    b = validation.evaluate_spec(SPEC, fit_df, eval_df, T_DUAN)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---------------------------------------------------------------- day-block folds
def test_day_block_folds_hold_out_whole_dates(sub_split):
    tr = sub_split[0]
    k = 5
    folds = common.day_block_folds(tr, k)
    assert len(folds) == len(tr)
    per_date = pd.DataFrame({"d": tr["dteday"].to_numpy(), "f": folds}).groupby("d")["f"].nunique()
    assert (per_date == 1).all()                                       # no date sits in two folds
    dates = np.sort(tr["dteday"].unique())
    rank = {d: i for i, d in enumerate(dates)}
    exp = np.array([rank[d] % k for d in tr["dteday"]])
    np.testing.assert_array_equal(folds, exp)                          # fold = rank of the date mod k, no RNG
    np.testing.assert_array_equal(folds, common.day_block_folds(tr, k))


def _fold_value(f):
    return f["r2"] if isinstance(f, dict) else f


def test_three_validators_day_block_matches_whole_date_oracle(sub_all, sub_split, cut):
    tr, va = sub_split
    res = validation.three_validators(SPEC, tr, va, sub_all, cut, T_DUAN, k=5)
    folds = common.day_block_folds(tr, 5)
    r2s, trs = [], []
    for f in range(5):
        o = _oracle(SPEC, tr[folds != f], tr[folds == f], "duan")
        r2s.append(o["r2"])
        trs.append(o["train_r2"])
    db = res["day_block"]
    assert len(db["folds"]) == 5
    np.testing.assert_allclose([_fold_value(x) for x in db["folds"]], r2s, atol=TOL)
    assert db["r2"] == pytest.approx(float(np.mean(r2s)), abs=TOL)
    assert db["sd"] == pytest.approx(float(np.std(r2s)), abs=TOL) or db["sd"] == pytest.approx(float(np.std(r2s, ddof=1)), abs=TOL)
    assert db["train_r2"] == pytest.approx(float(np.mean(trs)), abs=TOL)


def test_three_validators_day_block_is_deterministic_and_rng_free(sub_all, sub_split, cut):
    tr, va = sub_split
    np.random.seed(11)
    a = validation.three_validators(SPEC, tr, va, sub_all, cut, T_DUAN, k=5)
    np.random.seed(12)
    b = validation.three_validators(SPEC, tr, va, sub_all, cut, T_DUAN, k=5)
    assert json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float)


def test_day_block_calls_never_share_a_date(sub_all, sub_split, cut, monkeypatch):
    """If three_validators goes through validation.evaluate_spec, every call must have disjoint date sets
    unless it is the seeded call (random rows, shares dates by design)."""
    tr, va = sub_split
    calls = []
    real = validation.evaluate_spec

    def spy(spec, fit_df, eval_df, *a, **k):
        calls.append((set(fit_df["dteday"]), set(eval_df["dteday"]), len(fit_df), len(eval_df)))
        return real(spec, fit_df, eval_df, *a, **k)

    monkeypatch.setattr(validation, "evaluate_spec", spy)
    validation.three_validators(SPEC, tr, va, sub_all, cut, T_DUAN, k=5)
    disjoint = [c for c in calls if not (c[0] & c[1])]
    if calls:       # the spy sees nothing when the module uses private helpers; the oracle test above covers that case
        assert len(disjoint) >= 5 + 1                                  # 5 day-block folds + the chronological split


# ---------------------------------------------------------------- seeded + chronological
def test_three_validators_seeded_and_chrono_match_oracle(sub_all, sub_split, cut):
    tr, va = sub_split
    res = validation.three_validators(SPEC, tr, va, sub_all, cut, T_DUAN, k=5)
    seeded = _oracle(SPEC, tr, va, "duan")
    for k in ("train_r2", "train_rmse", "r2", "rmse"):
        assert res["seeded"][k] == pytest.approx(seeded[k], abs=TOL, rel=TOL), k
    cut_ts = pd.Timestamp(cut)
    early = sub_all[sub_all["dteday"] < cut_ts]
    late = sub_all[sub_all["dteday"] >= cut_ts]
    assert len(early) > 100 and len(late) > 100
    chrono = _oracle(SPEC, early, late, "duan")
    for k in ("train_r2", "r2", "rmse"):
        assert res["chrono"][k] == pytest.approx(chrono[k], abs=TOL, rel=TOL), k


def test_chrono_trains_strictly_before_the_cut_date(sub_all, cut):
    cut_ts = pd.Timestamp(cut)
    early, late = common.chrono_split(sub_all, cut)
    assert early["dteday"].max() < cut_ts <= late["dteday"].min()
    assert len(early) + len(late) == len(sub_all)


def test_chrono_score_ignores_rows_on_or_after_the_cut_only_for_training(sub_all, sub_split, cut):
    """Corrupting the training labels of rows after the cut must not change chrono train_r2, because a
    row on or after the cut is never in the chronological training part (it is evaluated, not fitted)."""
    tr, va = sub_split
    cut_ts = pd.Timestamp(cut)
    base = validation.three_validators(SPEC, tr, va, sub_all, cut, T_DUAN, k=5)["chrono"]
    ref = _oracle(SPEC, sub_all[sub_all["dteday"] < cut_ts], sub_all[sub_all["dteday"] >= cut_ts], "duan")
    assert base["train_r2"] == pytest.approx(ref["train_r2"], abs=TOL)
    corrupted = sub_all.copy()
    late_mask = corrupted["dteday"] >= cut_ts
    corrupted.loc[late_mask, "temp"] = 0.0
    out = validation.three_validators(SPEC, tr, va, corrupted, cut, T_DUAN, k=5)["chrono"]
    assert out["train_r2"] == base["train_r2"]                         # training part untouched by late rows
    assert out["r2"] != base["r2"]


# ---------------------------------------------------------------- learning curve
FRACS = [0.25, 0.5, 1.0]


def test_learning_curve_subsets_are_nested_prefixes_of_one_permutation(sub_split, seed):
    tr, va = sub_split
    rows = validation.learning_curve(SPEC, tr, va, FRACS, seed, T_DUAN)
    assert len(rows) == len(FRACS)
    dates = np.sort(tr["dteday"].unique())
    perm = np.random.default_rng(seed).permutation(dates)              # SPEC: rng(seed) permutation of the training dates
    n_days = [r["n_days"] for r in rows]
    assert n_days == sorted(n_days)
    for f, r in zip(FRACS, rows):
        assert r["fraction"] == pytest.approx(f)
        assert abs(r["n_days"] - f * len(dates)) <= 1
        sel = set(perm[: r["n_days"]])
        sub = tr[tr["dteday"].isin(sel)]
        assert r["n_rows"] == len(sub)
        o = _oracle(SPEC, sub, va, "duan")
        assert r["train_r2"] == pytest.approx(o["train_r2"], abs=TOL, rel=TOL)
        assert r["val_r2"] == pytest.approx(o["r2"], abs=TOL, rel=TOL)
    assert rows[-1]["n_days"] == len(dates) and rows[-1]["n_rows"] == len(tr)


def test_learning_curve_deterministic_and_seed_dependent(sub_split, seed):
    tr, va = sub_split
    a = validation.learning_curve(SPEC, tr, va, FRACS, seed, T_DUAN)
    b = validation.learning_curve(SPEC, tr, va, FRACS, seed, T_DUAN)
    assert json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float)
    c = validation.learning_curve(SPEC, tr, va, FRACS, seed + 1, T_DUAN)
    assert [r["train_r2"] for r in a[:2]] != [r["train_r2"] for r in c[:2]]
    assert a[-1]["val_r2"] == pytest.approx(c[-1]["val_r2"], abs=1e-12)     # the full set is seed free


# ---------------------------------------------------------------- noise floor
CELL = ["yr", "mnth", "workingday", "hr", "weathersit"]


def _floor_oracle(fr, merge4):
    w = np.minimum(fr["weathersit"], 3) if merge4 else fr["weathersit"]
    f = fr.assign(weathersit=w)
    if not merge4:
        f = f[f["weathersit"] <= 3]
    y = f["cnt"].to_numpy(float)
    cell_mean = f.groupby(CELL)["cnt"].transform("mean").to_numpy(float)
    return 1.0 - np.sum((y - cell_mean) ** 2) / np.sum((y - y.mean()) ** 2)


def _numeric_values(d, out=None, prefix=""):
    out = {} if out is None else out
    for k, v in d.items():
        if isinstance(v, dict):
            _numeric_values(v, out, prefix + k + ".")
        elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[prefix + k] = float(v)
    return out


def test_noise_floor_in_unit_interval_and_matches_oracle(train_df):
    res = validation.noise_floor(train_df)
    vals = _numeric_values(res)
    assert vals, "noise_floor returned no numbers"
    for k, v in vals.items():
        assert np.isfinite(v), k
        if "floor" in k or "share" in k:
            assert 0.0 <= v <= 1.0, k
        else:
            assert v >= 0.0, k
    targets = [_floor_oracle(train_df, True), _floor_oracle(train_df, False)]
    assert any(abs(v - t) < 1e-9 for v in vals.values() for t in targets), (vals, targets)
    assert vals.get("n_cells", vals.get("cells", 1)) >= 1


def test_noise_floor_one_cell_is_zero(train_df):
    one = train_df.iloc[:300].assign(yr=0, mnth=1, workingday=1, hr=5, weathersit=1)
    v1 = _numeric_values(validation.noise_floor(one))
    floors = [v for k, v in v1.items() if "floor" in k and "adjusted" not in k]
    assert floors and all(abs(v) < 1e-12 for v in floors)              # whole variance is within the single cell


def test_noise_floor_nearly_all_singletons_is_near_one(train_df):
    uniq = train_df.drop_duplicates(subset=CELL).iloc[:200]
    assert len(uniq) == 200 and not uniq.duplicated(subset=CELL).any()
    pair = pd.concat([uniq, uniq.iloc[[0]]])                            # one cell with two rows, 199 singletons
    v = _numeric_values(validation.noise_floor(pair))
    raw = [x for k, x in v.items() if "floor" in k and "adjusted" not in k]
    assert raw and all(0.0 <= x <= 1.0 for x in raw)
    assert all(x > 0.99 for x in raw)                                  # only the duplicated pair has within-cell spread
    shares = [x for k, x in v.items() if "singleton" in k]
    assert shares and all(0.0 <= x <= 1.0 for x in shares)


def test_noise_floor_all_singletons_is_one_without_crashing(train_df):
    uniq = train_df.drop_duplicates(subset=CELL).iloc[:200]
    v = _numeric_values(validation.noise_floor(uniq))                   # every row its own cell
    raw = [x for k, x in v.items() if "floor" in k and "adjusted" not in k]
    assert raw and all(abs(x - 1.0) < 1e-12 for x in raw)
    assert all(abs(x - 1.0) < 1e-12 for k, x in v.items() if "singleton" in k)
