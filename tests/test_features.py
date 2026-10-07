"""Tests for src.features, derived from ARCHITECTURE section 4."""
import json

import numpy as np
import pandas as pd
import pytest

features = pytest.importorskip("src.features")
pytest.importorskip("src.common")

BASE = features.BASE
DesignSpec = features.DesignSpec
Design = features.Design


def base_spec():
    return DesignSpec(base=tuple(BASE), power_cols=(), degree=1, blocks=())


def spec_with(blocks=(), power_cols=(), degree=1):
    return DesignSpec(base=tuple(BASE), power_cols=tuple(power_cols), degree=degree, blocks=tuple(blocks))


@pytest.fixture(scope="module")
def base_design(train_df):
    return Design(base_spec()).fit(train_df)


def col(design, X, name):
    return X[:, design.names.index(name)]


def test_base_definition():
    assert len(BASE) == 35
    assert BASE[:23] == [f"hr_{h}" for h in range(1, 24)] or tuple(BASE[:23]) == tuple(f"hr_{h}" for h in range(1, 24))
    assert list(BASE[23:]) == ["workingday", "holiday", "ws_2", "ws_3", "temp", "hum", "windspeed",
                               "trend", "doy_s1", "doy_c1", "doy_s2", "doy_c2"]


def test_base_design_shape_and_standardised(base_design, train_df):
    X = base_design.transform(train_df)
    assert X.shape == (len(train_df), 36)
    assert base_design.names[0] == "bias" and len(base_design.names) == 36
    assert X.dtype == np.float64
    assert np.all(X[:, 0] == 1.0)
    assert np.abs(X[:, 1:].mean(axis=0)).max() < 1e-12
    assert np.allclose(X[:, 1:].std(axis=0), 1.0, atol=1e-12)


def test_lambda_max_about_1_97(base_design, train_df):
    X = base_design.transform(train_df)
    lam = np.linalg.eigvalsh(X.T @ X / len(X)).max()  # leak-ok: oracle
    assert lam == pytest.approx(1.97, abs=0.01)


def test_scaler_dict_json_and_train_stats(base_design, train_df):
    s = base_design.scaler_dict()
    json.dumps(s)
    assert set(["names", "mean", "std", "hum_fill"]) <= set(s)
    assert s["names"] == base_design.names[1:]
    assert len(s["mean"]) == len(s["std"]) == 35
    i = s["names"].index("temp")
    assert s["mean"][i] == pytest.approx(train_df["temp"].mean(), abs=1e-12)
    assert s["std"][i] == pytest.approx(train_df["temp"].std(ddof=0), abs=1e-12)
    assert s["hum_fill"] == pytest.approx(train_df.loc[train_df["hum"] > 0, "hum"].median(), abs=1e-12)


def test_stats_from_fit_frame_only(base_design, train_df, val_df):
    before = json.dumps(base_design.scaler_dict(), sort_keys=True)
    shifted = val_df.copy()
    shifted["temp"] = shifted["temp"] + 0.4
    shifted["windspeed"] = shifted["windspeed"] * 3
    Xs = base_design.transform(shifted)
    assert json.dumps(base_design.scaler_dict(), sort_keys=True) == before
    s = base_design.scaler_dict()
    i = s["names"].index("temp")
    expect = (shifted["temp"].to_numpy() - s["mean"][i]) / s["std"][i]
    assert np.allclose(col(base_design, Xs, "temp"), expect, atol=1e-12)
    Xv = base_design.transform(val_df)
    expect_v = (val_df["temp"].to_numpy() - train_df["temp"].mean()) / train_df["temp"].std(ddof=0)
    assert np.allclose(col(base_design, Xv, "temp"), expect_v, atol=1e-12)
    assert abs(col(base_design, Xv, "temp").mean()) > 1e-6  # val is not re-centred


def test_transform_does_not_mutate_input(base_design, val_df):
    snap = val_df.copy(deep=True)
    base_design.transform(val_df)
    pd.testing.assert_frame_equal(val_df, snap)


def test_hum_zero_replaced_by_fill(train_df):
    df = train_df.head(40).copy()
    df.loc[df.index[:5], "hum"] = 0.0
    raw = features.raw_columns(df, 0.61)
    assert (raw["hum"].iloc[:5] == 0.61).all()
    assert np.allclose(raw["hum"].iloc[5:].to_numpy(), df["hum"].iloc[5:].to_numpy())
    assert raw.index.equals(df.index)


def test_design_hum_fill_is_train_median_of_positive(train_df, val_df):
    fit_df = train_df.copy()
    fit_df.loc[fit_df.index[:50], "hum"] = 0.0
    d = Design(base_spec()).fit(fit_df)
    pos = fit_df.loc[fit_df["hum"] > 0, "hum"]
    assert d.scaler_dict()["hum_fill"] == pytest.approx(pos.median(), abs=1e-12)
    # transform: a zero-hum row gets the standardised fill value
    s = d.scaler_dict()
    i = s["names"].index("hum")
    row = val_df.head(3).copy()
    row["hum"] = 0.0
    X = d.transform(row)
    exp = (s["hum_fill"] - s["mean"][i]) / s["std"][i]
    assert np.allclose(col(d, X, "hum"), exp, atol=1e-12)


def test_weathersit_4_merged_into_ws3(train_df):
    df = train_df.head(12).copy()
    df["weathersit"] = [1, 2, 3, 4] * 3
    raw = features.raw_columns(df, 0.6)
    assert raw["ws_2"].tolist() == [0, 1, 0, 0] * 3
    assert raw["ws_3"].tolist() == [0, 0, 1, 1] * 3


def test_hr_onehot_reference_hour_zero(train_df):
    df = train_df.head(48).copy()
    df["hr"] = np.arange(48) % 24
    raw = features.raw_columns(df, 0.6)
    hr_cols = [f"hr_{h}" for h in range(1, 24)]
    assert "hr_0" not in raw.columns
    assert (raw.loc[df["hr"].to_numpy() == 0, hr_cols].sum(axis=1) == 0).all()
    assert (raw[hr_cols].sum(axis=1)[df["hr"].to_numpy() > 0] == 1).all()
    for i in (5, 17, 23):
        assert (raw.loc[df["hr"].to_numpy() == i, f"hr_{i}"] == 1).all()


def test_raw_trend_and_fourier(train_df):
    df = train_df.sample(30, random_state=1)
    raw = features.raw_columns(df, 0.6)
    d = pd.to_datetime(df["dteday"])
    assert np.allclose(raw["trend"], (d - pd.Timestamp("2011-01-01")).dt.days.to_numpy(dtype=float), atol=0)
    doy = d.dt.dayofyear.to_numpy(dtype=float)
    assert np.allclose(raw["doy_s1"], np.sin(2 * np.pi * doy / 365.25), atol=1e-12)
    assert np.allclose(raw["doy_c2"], np.cos(4 * np.pi * doy / 365.25), atol=1e-12)


def test_raw_columns_candidates_present(train_df):
    raw = features.raw_columns(train_df.head(20), 0.6)
    need = ["atemp", "yr", "instant"] + [f"se_{k}" for k in (2, 3, 4)] + [f"mn_{m}" for m in range(2, 13)] \
        + [f"wk_{d}" for d in range(1, 7)] + BASE
    assert [c for c in need if c not in raw.columns] == []
    assert set(raw["windspeed"]) <= set(train_df["windspeed"])  # windspeed untouched (zeros kept)


def test_constant_column_keeps_std_one(train_df):
    fit_df = train_df[train_df["holiday"] == 0]
    d = Design(base_spec()).fit(fit_df)
    s = d.scaler_dict()
    i = s["names"].index("holiday")
    assert s["std"][i] == 1.0
    assert np.all(col(d, d.transform(fit_df), "holiday") == 0.0)


# ---- blocks: (name, product columns, auto-added first-order columns)
BLOCKS = [
    ("wd_x_hr", 23, 0),
    ("hr_x_temp", 23, 0),
    ("hr_x_hum", 23, 0),
    ("wk", 0, 6),
    ("wk_x_hr", 138, 6),
    ("wd_x_hr_x_temp", 23, 0),
    ("hr_x_ws", 46, 0),
    ("wd_x_wx", 4, 0),
    ("hr_x_doy", 46, 0),
    ("hr_x_trend", 23, 0),
    ("mn", 0, 11),
    ("mn_x_hr", 253, 11),
    ("mn_x_wd_x_hr", 253, 11),
    ("c_atemp", 0, 1),
    ("c_yr", 0, 1),
    ("c_instant", 0, 1),
    ("c_season", 0, 3),
    ("c_mnth", 0, 11),
    ("c_weekday", 0, 6),
]


@pytest.mark.parametrize("block,n_prod,n_first", BLOCKS)
def test_block_column_counts(block, n_prod, n_first, train_df):
    d = Design(spec_with(blocks=(block,))).fit(train_df)
    X = d.transform(train_df)
    assert X.shape[1] == len(d.names) == 36 + n_first + n_prod
    assert len(set(d.names)) == len(d.names)
    prods = [n for n in d.names if "*" in n]
    assert len(prods) == n_prod
    assert np.isfinite(X).all()
    # derived columns are standardised on train too (unless constant)
    const = set(getattr(d, "constant_columns", []) or [])
    for n in d.names[1:]:
        if n in const:
            continue
        j = d.names.index(n)
        assert abs(X[:, j].mean()) < 1e-9 and abs(X[:, j].std() - 1) < 1e-9, n


def test_product_values_are_products_of_standardised(train_df):
    d = Design(spec_with(blocks=("wd_x_hr",))).fit(train_df)
    X = d.transform(train_df)
    s = d.scaler_dict()
    # recompute from first-order standardised columns then standardise with own train stats
    prod = col(d, X, "workingday") * col(d, X, "hr_7")
    name = "workingday*hr_7"
    i = s["names"].index(name)
    assert s["mean"][i] == pytest.approx(prod.mean(), abs=1e-10)
    assert s["std"][i] == pytest.approx(prod.std(), abs=1e-10)
    assert np.allclose(col(d, X, name), (prod - prod.mean()) / prod.std(), atol=1e-10)


def test_powers_naming_order_and_values(train_df):
    d = Design(spec_with(power_cols=("temp", "hum", "windspeed"), degree=3)).fit(train_df)
    expected_tail = ["temp^2", "temp^3", "hum^2", "hum^3", "windspeed^2", "windspeed^3"]
    assert d.names[36:] == expected_tail
    X = d.transform(train_df)
    p = col(d, X, "temp") ** 2
    assert np.allclose(col(d, X, "temp^2"), (p - p.mean()) / p.std(), atol=1e-10)


def test_expanded_first_36_equal_base_and_base_scaler_reuse(base_design, train_df, val_df):
    spec = spec_with(blocks=("wd_x_hr",), power_cols=("temp", "hum", "windspeed"), degree=3)
    d_plain = Design(spec).fit(train_df)
    assert len(d_plain.names) == 65
    assert d_plain.names[:36] == base_design.names
    Xp, Xb = d_plain.transform(train_df), base_design.transform(train_df)
    assert np.allclose(Xp[:, :36], Xb, atol=0, rtol=0) or np.abs(Xp[:, :36] - Xb).max() < 1e-12
    d_lift = Design(spec).fit(train_df, base_scaler=base_design.scaler_dict())
    assert d_lift.names == d_plain.names
    assert np.abs(d_lift.transform(train_df) - Xp).max() < 1e-12
    # base_scaler wins over the fit frame's own statistics for base columns
    d_other = Design(spec).fit(val_df, base_scaler=base_design.scaler_dict())
    assert np.abs(d_other.transform(val_df)[:, :36] - base_design.transform(val_df)).max() < 1e-12


def test_fit_transform_and_subset(train_df):
    d = Design(base_spec())
    X = d.fit_transform(train_df)
    assert X.shape == (len(train_df), 36)
    idx = d.subset(["bias", "temp", "hr_3"])
    assert list(np.asarray(idx)) == [0, d.names.index("temp"), d.names.index("hr_3")]


def test_spec_dict_roundtrip_and_frozen():
    spec = spec_with(blocks=("wd_x_hr", "mn"), power_cols=("temp",), degree=2)
    d = spec.to_dict()
    json.dumps(d)
    spec2 = DesignSpec.from_dict(d)
    assert spec2 == spec
    with pytest.raises(Exception):
        spec.degree = 5


def test_transform_before_fit_fails(val_df):
    with pytest.raises(Exception):
        Design(base_spec()).transform(val_df)


def test_deterministic_matrix(train_df):
    a = Design(spec_with(blocks=("hr_x_temp",))).fit(train_df).transform(train_df)
    b = Design(spec_with(blocks=("hr_x_temp",))).fit(train_df).transform(train_df)
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("name,expected", [
    ("hr_5", {"hr"}),
    ("trend", {"dteday"}),
    ("temp^3", {"temp"}),
    ("workingday*hr_7", {"workingday", "hr"}),
    ("mn_3*workingday*hr_7", {"mnth", "workingday", "hr"}),
    ("ws_2", {"weathersit"}),
    ("doy_s1", {"dteday"}),
    ("doy_c2", {"dteday"}),
    ("se_3", {"season"}),
    ("mn_12", {"mnth"}),
    ("wk_4", {"weekday"}),
    ("hum*hr_3", {"hum", "hr"}),
    ("bias", set()),
])
def test_sources(name, expected):
    out = features.sources(name)
    assert isinstance(out, frozenset)
    assert out == frozenset(expected)
