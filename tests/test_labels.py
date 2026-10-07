"""Tests for src.labels (S-5-01 contract), written from the SPEC and pandas as the oracle.

Contract: fit_thresholds(train_df, group_by, quantile) -> DataFrame[group_by + ["threshold", "n"]]
(pandas quantile, linear interpolation, group_by = [] gives one global row);
apply_thresholds(df, table, group_by) -> int array, label 1 iff cnt > threshold of the row's cell,
a cell missing from the table raises.
"""
import numpy as np
import pandas as pd
import pytest

from src import labels

GB = ["yr", "workingday", "hr"]


def _oracle_table(frame, group_by, q):
    """Reference thresholds: groupby(...).cnt.quantile(q) on the given frame only."""
    if not group_by:
        return pd.DataFrame({"threshold": [frame["cnt"].quantile(q)], "n": [len(frame)]})
    g = frame.groupby(group_by)["cnt"]
    out = pd.DataFrame({"threshold": g.quantile(q), "n": g.size()})
    return out.reset_index()


def _oracle_labels(frame, oracle_table, group_by):
    if not group_by:
        return (frame["cnt"].to_numpy() > oracle_table["threshold"].iloc[0]).astype(int)
    merged = frame[group_by + ["cnt"]].merge(oracle_table[group_by + ["threshold"]], on=group_by, how="left")
    assert not merged["threshold"].isna().any()
    return (merged["cnt"].to_numpy() > merged["threshold"].to_numpy()).astype(int)


@pytest.mark.parametrize("q", [0.5, 0.75, 0.9])
def test_thresholds_equal_groupby_quantile(train_df, q):
    tab = labels.fit_thresholds(train_df, GB, q)
    assert list(tab.columns) == GB + ["threshold", "n"]
    ref = _oracle_table(train_df, GB, q)
    assert len(tab) == len(ref)
    a = tab.set_index(GB).sort_index()
    b = ref.set_index(GB).sort_index()
    assert a.index.equals(b.index)
    np.testing.assert_allclose(a["threshold"].to_numpy(float), b["threshold"].to_numpy(float), rtol=0, atol=1e-10)
    np.testing.assert_array_equal(a["n"].to_numpy(), b["n"].to_numpy())
    assert int(tab["n"].sum()) == len(train_df)


def test_thresholds_use_only_the_given_frame(train_df, val_df):
    small = train_df.iloc[:1500]
    tab_small = labels.fit_thresholds(small, GB, 0.75)
    ref_small = _oracle_table(small, GB, 0.75)
    a = tab_small.set_index(GB).sort_index()
    b = ref_small.set_index(GB).sort_index()
    np.testing.assert_allclose(a["threshold"].to_numpy(float), b["threshold"].to_numpy(float), atol=1e-10)
    np.testing.assert_array_equal(a["n"].to_numpy(), b["n"].to_numpy())
    # power check: the table of the subset is not the table of the whole frame
    full = labels.fit_thresholds(train_df, GB, 0.75).set_index(GB)
    common_cells = a.index.intersection(full.index)
    assert (a.loc[common_cells, "threshold"].to_numpy() != full.loc[common_cells, "threshold"].to_numpy()).any()


def test_fit_does_not_mutate_input(train_df):
    before_cols = list(train_df.columns)
    snap = train_df[["cnt", "hr", "yr", "workingday"]].copy()
    labels.fit_thresholds(train_df, GB, 0.75)
    assert list(train_df.columns) == before_cols
    pd.testing.assert_frame_equal(train_df[["cnt", "hr", "yr", "workingday"]], snap)


@pytest.mark.parametrize("q", [0.5, 0.75])
def test_apply_matches_oracle_labels(train_df, val_df, q):
    tab = labels.fit_thresholds(train_df, GB, q)
    ref = _oracle_table(train_df, GB, q)
    for frame in (train_df, val_df):
        y = labels.apply_thresholds(frame, tab, GB)
        assert isinstance(y, np.ndarray)
        assert y.shape == (len(frame),)
        assert np.issubdtype(y.dtype, np.integer)
        assert set(np.unique(y)) <= {0, 1}
        np.testing.assert_array_equal(y, _oracle_labels(frame, ref, GB))


def test_apply_is_strictly_greater_and_cell_specific():
    """Hand-built table: label 1 iff cnt > that cell's threshold (strict)."""
    tab = pd.DataFrame({"yr": [0, 1], "workingday": [1, 1], "hr": [7, 7],
                        "threshold": [100.0, 50.0], "n": [10, 10]})
    df = pd.DataFrame({"yr": [0, 0, 0, 1, 1, 1],
                       "workingday": [1] * 6, "hr": [7] * 6,
                       "cnt": [99, 100, 101, 49, 50, 51]})
    y = labels.apply_thresholds(df, tab, ["yr", "workingday", "hr"])
    np.testing.assert_array_equal(y, [0, 0, 1, 0, 0, 1])
    # the same count 100 is positive in the cell whose threshold is 50
    df2 = pd.DataFrame({"yr": [0, 1], "workingday": [1, 1], "hr": [7, 7], "cnt": [100, 100]})
    np.testing.assert_array_equal(labels.apply_thresholds(df2, tab, ["yr", "workingday", "hr"]), [0, 1])


def test_apply_is_row_local_and_order_preserving(train_df, val_df):
    tab = labels.fit_thresholds(train_df, GB, 0.75)
    y = labels.apply_thresholds(val_df, tab, GB)
    perm = np.random.default_rng(3).permutation(len(val_df))
    y_perm = labels.apply_thresholds(val_df.iloc[perm], tab, GB)
    np.testing.assert_array_equal(y_perm, y[perm])
    one = labels.apply_thresholds(val_df.iloc[[17]], tab, GB)
    assert one.shape == (1,) and one[0] == y[17]


def test_cell_missing_from_table_raises(train_df, val_df):
    partial = train_df[train_df["hr"] != 3]
    tab = labels.fit_thresholds(partial, GB, 0.75)
    assert (val_df["hr"] == 3).any()
    with pytest.raises(Exception):
        labels.apply_thresholds(val_df, tab, GB)


def test_hand_built_table_missing_cell_raises():
    tab = pd.DataFrame({"yr": [0], "workingday": [1], "hr": [7], "threshold": [10.0], "n": [5]})
    df = pd.DataFrame({"yr": [0, 0], "workingday": [1, 0], "hr": [7, 7], "cnt": [20, 20]})
    with pytest.raises(Exception):
        labels.apply_thresholds(df, tab, ["yr", "workingday", "hr"])


@pytest.mark.parametrize("q", [0.5, 0.75])
def test_global_rule(train_df, val_df, q):
    tab = labels.fit_thresholds(train_df, [], q)
    assert len(tab) == 1
    assert list(tab.columns) == ["threshold", "n"]
    thr = train_df["cnt"].quantile(q)
    assert tab["threshold"].iloc[0] == pytest.approx(thr, abs=1e-10)
    assert int(tab["n"].iloc[0]) == len(train_df)
    y = labels.apply_thresholds(val_df, tab, [])
    np.testing.assert_array_equal(y, (val_df["cnt"].to_numpy() > thr).astype(int))


def test_thresholds_unchanged_when_validation_rows_change(train_df, val_df):
    tab_a = labels.fit_thresholds(train_df, GB, 0.75)
    snap = tab_a.copy(deep=True)
    variants = [val_df.assign(cnt=val_df["cnt"] * 10), val_df.iloc[: len(val_df) // 2], val_df.assign(cnt=0)]
    outs = [labels.apply_thresholds(v, tab_a, GB) for v in variants]
    pd.testing.assert_frame_equal(tab_a, snap)                       # applying never alters the table
    tab_b = labels.fit_thresholds(train_df, GB, 0.75)                # refit on the same training frame
    pd.testing.assert_frame_equal(tab_a, tab_b)
    assert outs[2].sum() == 0                                        # cnt = 0 is never above a threshold >= 0
    assert outs[0].sum() >= labels.apply_thresholds(val_df, tab_a, GB).sum()
    # power: a table fitted on train + validation would differ somewhere
    both = pd.concat([train_df, val_df])
    tab_c = labels.fit_thresholds(both, GB, 0.75)
    assert not np.allclose(tab_a.sort_values(GB)["threshold"].to_numpy(),
                           tab_c.sort_values(GB)["threshold"].to_numpy(), atol=1e-12)
    # frames are not mutated by apply
    assert "threshold" not in val_df.columns
