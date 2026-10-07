"""Tests for the weather-memory inputs (ADR-015, S-7-01 items 1-2, ARCHITECTURE section 4).

Definition (SPEC): timestamp = dteday + hr hours; situation 4 counts as 3;
ws_lag1 = situation one hour earlier; wet3 = max of the situations 1, 2 and 3 hours earlier;
an earlier hour missing from the lookup counts as the row's own situation.
Expected values come from a brute-force loop over the raw csv, never from the implementation.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split  # leak-ok: oracle

from src import common, features
from src.features import BASE, Design, DesignSpec

HOUR = pd.Timedelta(hours=1)


# ---------------------------------------------------------------- brute-force oracle
def _sit(s):
    return min(int(s), 3)


def _timeline(frame):
    ts = pd.to_datetime(frame["dteday"]) + pd.to_timedelta(frame["hr"].astype(int), unit="h")
    return {t: _sit(s) for t, s in zip(ts, frame["weathersit"])}


def _brute(frame, lookup=None):
    """Return (ws_lag1, wet3) lists computed by a plain loop, row by row."""
    line = _timeline(frame if lookup is None else lookup)
    ts = pd.to_datetime(frame["dteday"]) + pd.to_timedelta(frame["hr"].astype(int), unit="h")
    lag, wet = [], []
    for t, s in zip(ts, frame["weathersit"]):
        own = _sit(s)
        lag.append(line.get(t - HOUR, own))
        wet.append(max(line.get(t - k * HOUR, own) for k in (1, 2, 3)))
    return np.array(lag), np.array(wet)


def _inputs(frame):
    return frame[[c for c in frame.columns if c not in ("cnt", "ws_lag1", "wet3")]].copy()


@pytest.fixture(scope="module")
def raw_csv(cfg):
    return pd.read_csv(cfg["paths"]["train"], parse_dates=["dteday"])


@pytest.fixture(scope="module")
def inputs(train_all):
    return _inputs(train_all)


def _mk(rows, template):
    """Synthetic frame from (date, hr, weathersit) triples; other columns copied from a real row."""
    base = template.iloc[[0]].drop(columns=["ws_lag1", "wet3", "cnt"], errors="ignore")
    out = pd.concat([base] * len(rows), ignore_index=True)
    out["dteday"] = pd.to_datetime([r[0] for r in rows])
    out["hr"] = [r[1] for r in rows]
    out["weathersit"] = [r[2] for r in rows]
    out["instant"] = np.arange(1, len(rows) + 1)
    return out


# ---------------------------------------------------------------- brute force on real rows
def test_columns_equal_brute_force_on_200_random_training_rows(train_all, raw_csv):
    assert {"ws_lag1", "wet3"} <= set(train_all.columns)
    assert len(train_all) == len(raw_csv)
    pick = np.random.default_rng(7).choice(len(train_all), size=200, replace=False)
    # oracle timeline = the training csv itself
    exp_lag, exp_wet = _brute(train_all.iloc[pick], lookup=raw_csv)
    np.testing.assert_array_equal(train_all["ws_lag1"].to_numpy()[pick], exp_lag)
    np.testing.assert_array_equal(train_all["wet3"].to_numpy()[pick], exp_wet)
    assert set(np.unique(train_all["ws_lag1"])) <= {1, 2, 3}
    assert set(np.unique(train_all["wet3"])) <= {1, 2, 3}
    assert np.issubdtype(train_all["ws_lag1"].dtype, np.integer)
    assert np.issubdtype(train_all["wet3"].dtype, np.integer)


def test_all_rows_equal_brute_force(train_all, raw_csv):
    exp_lag, exp_wet = _brute(raw_csv)
    by_instant = train_all.set_index("instant")
    np.testing.assert_array_equal(by_instant.loc[raw_csv["instant"], "ws_lag1"].to_numpy(), exp_lag)
    np.testing.assert_array_equal(by_instant.loc[raw_csv["instant"], "wet3"].to_numpy(), exp_wet)


def test_random_subset_with_default_lookup_uses_only_itself(inputs):
    sub = inputs.sample(200, random_state=11)
    out = common.add_weather_memory(sub)
    exp_lag, exp_wet = _brute(sub)      # timeline = the 200 rows: most earlier hours are missing
    np.testing.assert_array_equal(out["ws_lag1"].to_numpy(), exp_lag)
    np.testing.assert_array_equal(out["wet3"].to_numpy(), exp_wet)
    assert out.index.equals(sub.index)


def test_function_returns_copy_and_keeps_columns(inputs):
    sub = inputs.iloc[:300].copy()
    snap = sub.copy(deep=True)
    out = common.add_weather_memory(sub)
    pd.testing.assert_frame_equal(sub, snap)                         # input not mutated
    assert set(sub.columns) <= set(out.columns)
    assert set(out.columns) - set(sub.columns) == {"ws_lag1", "wet3"}
    assert len(out) == len(sub)
    for c in sub.columns:
        pd.testing.assert_series_equal(out[c], sub[c])


# ---------------------------------------------------------------- never reads cnt
def test_never_reads_cnt(train_all, inputs):
    ref = common.add_weather_memory(inputs)
    assert "cnt" not in inputs.columns
    noisy = inputs.assign(cnt=np.random.default_rng(1).integers(0, 1000, len(inputs)))
    nan = inputs.assign(cnt=np.nan)
    for variant in (noisy, nan):
        got = common.add_weather_memory(variant)
        np.testing.assert_array_equal(got["ws_lag1"].to_numpy(), ref["ws_lag1"].to_numpy())
        np.testing.assert_array_equal(got["wet3"].to_numpy(), ref["wet3"].to_numpy())
    np.testing.assert_array_equal(ref["ws_lag1"].to_numpy(), train_all["ws_lag1"].to_numpy())


# ---------------------------------------------------------------- training frame from the training file alone
def test_training_columns_independent_of_lookup_extras(train_all, inputs):
    own = common.add_weather_memory(inputs, lookup=inputs)
    default = common.add_weather_memory(inputs)
    for c in ("ws_lag1", "wet3"):
        np.testing.assert_array_equal(own[c].to_numpy(), train_all[c].to_numpy())
        np.testing.assert_array_equal(default[c].to_numpy(), train_all[c].to_numpy())
    # emulate a second file: later days (after each month's last training day) with the worst weather
    last = inputs.groupby(["yr", "mnth"])["dteday"].transform("max")
    template = inputs.iloc[[0]]
    extra_rows = []
    for d in sorted(last.unique()):
        for h in range(24):
            extra_rows.append((pd.Timestamp(d) + pd.Timedelta(days=1), h, 3))
    extra = _mk(extra_rows, template)
    extra["instant"] = 10 ** 6 + np.arange(len(extra))
    both = pd.concat([inputs, extra], ignore_index=True)
    got = common.add_weather_memory(inputs, lookup=both)
    for c in ("ws_lag1", "wet3"):
        np.testing.assert_array_equal(got[c].to_numpy(), train_all[c].to_numpy())


def test_lookup_with_extra_earlier_hour_is_used(inputs):
    ts = pd.to_datetime(inputs["dteday"]) + pd.to_timedelta(inputs["hr"], unit="h")
    present = set(ts)
    missing_prev = [i for i, t in enumerate(ts) if (t - HOUR) not in present]
    assert missing_prev, "the training file has gaps in its hours"
    i = missing_prev[len(missing_prev) // 2]
    row = inputs.iloc[[i]]
    own = _sit(row["weathersit"].iloc[0])
    extra_sit = 3 if own != 3 else 1
    ex = row.copy()
    ex["dteday"] = (ts.iloc[i] - HOUR).normalize()
    ex["hr"] = (ts.iloc[i] - HOUR).hour
    ex["weathersit"] = extra_sit
    ex["instant"] = -1
    lookup = pd.concat([inputs, ex], ignore_index=True)
    got = common.add_weather_memory(row, lookup=lookup)
    exp_lag, exp_wet = _brute(row, lookup=lookup)
    assert exp_lag[0] == extra_sit
    assert got["ws_lag1"].iloc[0] == extra_sit
    assert got["wet3"].iloc[0] == exp_wet[0]
    assert len(got) == 1                                              # lookup rows never leak into the output
    # without the extra row the row falls back to its own situation
    plain = common.add_weather_memory(row, lookup=inputs)
    assert plain["ws_lag1"].iloc[0] == own


# ---------------------------------------------------------------- hand-built edge cases
def test_midnight_crossing_and_gaps(inputs):
    d1, d2 = "2011-05-01", "2011-05-02"
    fr = _mk([(d1, 22, 1), (d1, 23, 3), (d2, 0, 2)], inputs)
    out = common.add_weather_memory(fr)
    # 23:00: hour 22 -> 1, hours 21 and 20 missing -> own situation (3), so wet3 = 3
    # 00:00 of the next day sees 23:00 (3), 22:00 (1), 21:00 missing -> own (2), so wet3 = 3
    assert out["ws_lag1"].tolist() == [1, 1, 3]
    assert out["wet3"].tolist() == [1, 3, 3]
    # first row has no earlier hours: own situation for all
    fr2 = _mk([(d1, 5, 2), (d1, 8, 1)], inputs)
    o2 = common.add_weather_memory(fr2)
    assert o2["ws_lag1"].tolist() == [2, 1]          # hour 7 missing -> own (1)
    assert o2["wet3"].tolist() == [2, 2]             # hour 8: 7, 6 missing -> own 1; hour 5 (t-3) -> 2


def test_weathersit_4_counts_as_3(inputs):
    d = "2012-03-03"
    fr = _mk([(d, 10, 4), (d, 11, 1), (d, 12, 4)], inputs)
    out = common.add_weather_memory(fr)
    assert out["ws_lag1"].tolist() == [3, 3, 1]      # own 4 -> 3 when nothing earlier; row 12 sees hour 11
    assert out["wet3"].tolist() == [3, 3, 3]
    assert set(out["ws_lag1"]) <= {1, 2, 3} and set(out["wet3"]) <= {1, 2, 3}


def test_wet3_window_is_three_hours_and_excludes_own(inputs):
    d = "2011-06-06"
    # hour 6 is four hours before hour 10 -> not in the window
    fr = _mk([(d, 6, 3), (d, 10, 1)], inputs)
    out = common.add_weather_memory(fr)
    assert out["wet3"].iloc[1] == 1
    # own situation is excluded when earlier hours exist: 3 with three clear hours before -> 1
    fr = _mk([(d, 7, 1), (d, 8, 1), (d, 9, 1), (d, 10, 3)], inputs)
    out = common.add_weather_memory(fr)
    assert out["wet3"].iloc[3] == 1 and out["ws_lag1"].iloc[3] == 1
    # the worst of hours 1, 2, 3 earlier wins
    fr = _mk([(d, 7, 3), (d, 8, 1), (d, 9, 2), (d, 10, 1)], inputs)
    out = common.add_weather_memory(fr)
    assert out["wet3"].iloc[3] == 3 and out["ws_lag1"].iloc[3] == 2


def test_row_order_does_not_matter(inputs):
    d = "2011-06-06"
    fr = _mk([(d, 7, 3), (d, 8, 1), (d, 9, 2), (d, 10, 1)], inputs)
    ref = common.add_weather_memory(fr)
    perm = [3, 1, 0, 2]
    sh = common.add_weather_memory(fr.iloc[perm])
    assert sh["ws_lag1"].tolist() == ref["ws_lag1"].iloc[perm].tolist()
    assert sh["wet3"].tolist() == ref["wet3"].iloc[perm].tolist()


# ---------------------------------------------------------------- raw feature columns
def test_raw_columns_are_indicators_of_the_memory_columns(train_all):
    hum_fill = float(train_all.loc[train_all["hum"] > 0, "hum"].median())
    raw = features.raw_columns(train_all, hum_fill)
    for name, col, val in [("wslag1_2", "ws_lag1", 2), ("wslag1_3", "ws_lag1", 3),
                           ("wet3_2", "wet3", 2), ("wet3_3", "wet3", 3)]:
        assert name in raw.columns
        np.testing.assert_array_equal(raw[name].to_numpy(float), (train_all[col].to_numpy() == val).astype(float))
    assert raw.index.equals(train_all.index)
    # existing columns untouched
    np.testing.assert_array_equal(raw["ws_3"].to_numpy(float), (np.minimum(train_all["weathersit"], 3) == 3).astype(float))


def test_weathersit4_and_hum0_and_windspeed0_raw_columns(train_all):
    hum_fill = 0.777
    assert (train_all["hum"] == 0).any() and (train_all["windspeed"] == 0).any()
    raw = features.raw_columns(train_all, hum_fill)
    z = train_all["hum"] == 0
    assert (raw.loc[z, "hum"] == hum_fill).all()
    assert (raw.loc[~z, "hum"].to_numpy() == train_all.loc[~z, "hum"].to_numpy()).all()
    w0 = train_all["windspeed"] == 0
    assert (raw.loc[w0, "windspeed"] == 0).all()                      # windspeed 0 is a value, not a gap
    s4 = train_all["weathersit"] == 4
    if s4.any():
        assert (raw.loc[s4, "ws_3"] == 1).all() and (raw.loc[s4, "ws_2"] == 0).all()


# ---------------------------------------------------------------- blocks
def _names(blocks, power_cols=(), degree=1, frame=None):
    spec = DesignSpec(base=tuple(BASE), power_cols=tuple(power_cols), degree=degree, blocks=tuple(blocks))
    return Design(spec).fit(frame).names


def test_blocks_add_four_columns_each_with_spec_names(train_df):
    ref = _names([], frame=train_df)
    wx = _names(["wx_detail"], frame=train_df)
    ms = _names(["ws_memory"], frame=train_df)
    new_wx = [n for n in wx if n not in ref]
    new_ms = [n for n in ms if n not in ref]
    assert len(wx) == len(ref) + 4 and len(ms) == len(ref) + 4
    assert new_wx == ["hum^2", "hum^3", "windspeed^2", "temp*windspeed"]
    assert sorted(new_ms) == sorted(["wslag1_2", "wslag1_3", "wet3_2", "wet3_3"])
    assert [n for n in ref if n not in wx] == []                       # nothing removed
    assert len(set(wx)) == len(wx) and len(set(ms)) == len(ms)


def test_wx_detail_not_duplicated_when_power_cols_already_has_the_column(train_df):
    ref = _names([], power_cols=["hum"], degree=3, frame=train_df)      # contains hum^2, hum^3
    assert "hum^2" in ref and "hum^3" in ref
    got = _names(["wx_detail"], power_cols=["hum"], degree=3, frame=train_df)
    assert len(set(got)) == len(got)
    assert [n for n in got if n not in ref] == ["windspeed^2", "temp*windspeed"]
    ref2 = _names([], power_cols=["hum"], degree=2, frame=train_df)     # only hum^2 present
    got2 = _names(["wx_detail"], power_cols=["hum"], degree=2, frame=train_df)
    assert len(set(got2)) == len(got2)
    assert [n for n in got2 if n not in ref2] == ["hum^3", "windspeed^2", "temp*windspeed"]


def _std(v):
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def test_wx_detail_and_memory_column_values_follow_the_math(train_df):
    spec = DesignSpec(base=tuple(BASE), power_cols=(), degree=1, blocks=("wx_detail", "ws_memory"))
    d = Design(spec).fit(train_df)
    X = d.transform(train_df)
    col = {n: X[:, i] for i, n in enumerate(d.names)}
    hum = train_df["hum"].to_numpy(float)
    fill = np.median(hum[hum > 0])
    hum = np.where(hum == 0, fill, hum)
    t = train_df["temp"].to_numpy(float)
    w = train_df["windspeed"].to_numpy(float)
    zh, zt, zw = _std(hum), _std(t), _std(w)
    np.testing.assert_allclose(col["hum^2"], _std(zh ** 2), atol=1e-9)
    np.testing.assert_allclose(col["hum^3"], _std(zh ** 3), atol=1e-9)
    np.testing.assert_allclose(col["windspeed^2"], _std(zw ** 2), atol=1e-9)
    np.testing.assert_allclose(col["temp*windspeed"], _std(zt * zw), atol=1e-9)
    for name, src, v in [("wslag1_2", "ws_lag1", 2), ("wslag1_3", "ws_lag1", 3),
                         ("wet3_2", "wet3", 2), ("wet3_3", "wet3", 3)]:
        ind = (train_df[src].to_numpy() == v).astype(float)
        np.testing.assert_allclose(col[name], _std(ind), atol=1e-9)


def test_wx_detail_with_windspeed_all_zero_is_finite(train_df):
    fr = train_df.assign(windspeed=0.0)
    spec = DesignSpec(base=tuple(BASE), power_cols=(), degree=1, blocks=("wx_detail",))
    d = Design(spec).fit(fr)
    X = d.transform(fr)
    assert np.isfinite(X).all()                                       # std 0 -> 1, no division by zero
    col = {n: X[:, i] for i, n in enumerate(d.names)}
    np.testing.assert_allclose(col["windspeed^2"], 0.0, atol=1e-12)
    np.testing.assert_allclose(col["temp*windspeed"], 0.0, atol=1e-12)


def test_hum_zero_rows_use_the_train_fill_in_blocks(train_df, val_df):
    spec = DesignSpec(base=tuple(BASE), power_cols=(), degree=1, blocks=("wx_detail",))
    d = Design(spec).fit(train_df)
    z = val_df.assign(hum=0.0)
    X = d.transform(z)
    assert np.isfinite(X).all()
    i = d.names.index("hum^2")
    j = d.names.index("hum")
    assert np.ptp(X[:, j]) < 1e-12 and np.ptp(X[:, i]) < 1e-12        # all rows replaced by one fill value


# ---------------------------------------------------------------- sources
def test_sources_of_memory_and_product_columns():
    for n in ("wslag1_2", "wslag1_3", "wet3_2", "wet3_3"):
        assert features.sources(n) == frozenset({"weathersit"})
    assert features.sources("temp*windspeed") == frozenset({"temp", "windspeed"})
    assert features.sources("hum^2") == frozenset({"hum"})
    assert features.sources("windspeed^2") == frozenset({"windspeed"})
    assert features.sources("bias") == frozenset()


# ---------------------------------------------------------------- split unchanged by the extra columns
def test_seeded_split_selects_same_instants_as_raw_csv(cfg, seed, split, raw_csv):
    tr, va = split
    exp_tr, exp_va = train_test_split(raw_csv, test_size=cfg["split"]["test_size"], random_state=seed)  # leak-ok: oracle
    assert tr["instant"].tolist() == exp_tr["instant"].tolist()
    assert va["instant"].tolist() == exp_va["instant"].tolist()
    # the memory columns of the split frames are the full-file values (not recomputed on the 80% part)
    lag, wet = _brute(tr, lookup=raw_csv)
    np.testing.assert_array_equal(tr["ws_lag1"].to_numpy(), lag)
    np.testing.assert_array_equal(tr["wet3"].to_numpy(), wet)
