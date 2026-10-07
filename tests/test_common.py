"""Tests for src.common, derived from ARCHITECTURE section 3 and 5."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

common = pytest.importorskip("src.common")

ROOT = Path(__file__).resolve().parent.parent


def test_team_seed_selftest_and_team():
    assert common.team_seed(["34521", "40218", "41190"]) == 41698
    cfg = common.load_config()
    assert common.team_seed(cfg["team_ids"]) == 44615


def test_split_sizes_disjoint_deterministic(train_all, seed):
    a_tr, a_va = common.seeded_split(train_all, seed)
    b_tr, b_va = common.seeded_split(train_all, seed)
    assert (len(a_tr), len(a_va)) == (8708, 2178)
    assert set(a_tr.index).isdisjoint(set(a_va.index))
    assert len(set(a_tr.index) | set(a_va.index)) == len(train_all)
    assert a_tr.index.equals(b_tr.index) and a_va.index.equals(b_va.index)


def test_split_equals_direct_sklearn(train_all, seed):
    from sklearn.model_selection import train_test_split  # leak-ok: oracle
    tr, va = common.seeded_split(train_all, seed)
    otr, ova = train_test_split(train_all, test_size=0.20, random_state=seed)  # leak-ok: oracle
    assert tr.index.equals(otr.index)
    assert va.index.equals(ova.index)


def test_day_block_folds_date_determined(train_df):
    f1 = common.day_block_folds(train_df, k=5)
    state = np.random.get_state()
    np.random.seed(999)
    f2 = common.day_block_folds(train_df, k=5)
    np.random.set_state(state)
    assert np.array_equal(f1, f2)
    assert len(f1) == len(train_df)
    dates = pd.to_datetime(train_df["dteday"]).to_numpy()
    df = pd.DataFrame({"d": dates, "f": f1})
    assert (df.groupby("d")["f"].nunique() == 1).all()
    assert set(np.unique(f1)) == {0, 1, 2, 3, 4}
    # fold = rank of date among sorted unique dates, mod k
    uniq = np.sort(df["d"].unique())
    rank = {d: i for i, d in enumerate(uniq)}
    expect = np.array([rank[d] % 5 for d in dates])
    assert np.array_equal(f1, expect)


def test_chrono_split_boundary(train_all):
    cut = "2012-07-01"
    early, late = common.chrono_split(train_all, cut)
    assert len(early) + len(late) == len(train_all)
    cutd = pd.Timestamp(cut)
    assert (pd.to_datetime(early["dteday"]) < cutd).all()
    assert (pd.to_datetime(late["dteday"]) >= cutd).all()
    assert len(early) > 0 and len(late) > 0


def test_r2_rmse_vs_sklearn(rng):
    from sklearn.metrics import mean_squared_error, r2_score  # leak-ok: oracle
    y = rng.normal(size=200) * 5 + 3
    yh = y + rng.normal(size=200)
    assert common.r2(y, yh) == pytest.approx(r2_score(y, yh), abs=1e-12)  # leak-ok: oracle
    assert common.rmse(y, yh) == pytest.approx(np.sqrt(mean_squared_error(y, yh)), abs=1e-12)  # leak-ok: oracle


def test_to_target_is_log1p():
    cnt = np.array([0.0, 1.0, 10.0, 977.0])
    assert np.allclose(common.to_target(cnt), np.log1p(cnt), atol=0, rtol=1e-15)


def _synthetic(rng, n=500):
    eta = rng.normal(size=n) * 0.5 + 3.0
    z = eta + rng.normal(size=n) * 0.4
    cnt = np.expm1(z)
    return z, eta, cnt


def test_back_factor_formulas(rng):
    z, eta, cnt = _synthetic(rng)
    assert common.back_factor("none", z, eta, cnt) == 1.0
    duan = np.mean(np.exp(z - eta))
    assert common.back_factor("duan", z, eta, cnt) == pytest.approx(duan, rel=1e-12)
    ls = np.sum((cnt + 1) * np.exp(eta)) / np.sum(np.exp(2 * eta))
    assert common.back_factor("ls", z, eta, cnt) == pytest.approx(ls, rel=1e-12)


def test_back_factor_unknown_method_raises(rng):
    z, eta, cnt = _synthetic(rng, 20)
    with pytest.raises(Exception):
        common.back_factor("bogus", z, eta, cnt)


def test_from_target_formula_and_clip():
    eta = np.array([-5.0, 0.0, 1.0, 4.0])
    out = common.from_target(eta, 1.1)
    expect = np.clip(np.exp(eta) * 1.1 - 1, 0, None)
    assert np.allclose(out, expect, rtol=1e-12)
    assert out[0] == 0.0 and out[1] == pytest.approx(0.1, abs=1e-12)
    assert (out >= 0).all()


def test_bootstrap_r2_properties(rng):
    y = rng.normal(size=300) * 4
    yh = y + rng.normal(size=300)
    lo, hi, se = common.bootstrap_r2(y, yh, seed=7, B=200)
    r2 = common.r2(y, yh)
    assert lo < r2 < hi and se > 0
    assert (lo, hi, se) == common.bootstrap_r2(y, yh, seed=7, B=200)


def test_paired_bootstrap_delta(rng):
    y = rng.normal(size=300) * 4
    good = y + rng.normal(size=300) * 0.5
    bad = y + rng.normal(size=300) * 2.0
    d, lo, hi = common.paired_bootstrap_delta_r2(y, good, bad, seed=3, B=200)
    assert d == pytest.approx(common.r2(y, good) - common.r2(y, bad), abs=1e-12)
    assert lo > 0 and hi >= lo


@pytest.fixture()
def art_env(tmp_path, monkeypatch):
    """Run artifact writes in the repo root (artifacts/ dir); clean up test names."""
    names = ["_qa_up", "_qa_a", "_qa_b"]
    d = ROOT / "artifacts"
    d.mkdir(exist_ok=True)
    yield names
    for n in names:
        p = d / f"{n}.json"
        if p.exists():
            p.unlink()


def test_write_artifact_bytes_and_hashes(art_env):
    up, a, _ = art_env
    cfg = common.load_config()
    common.write_artifact(up, {"x": 1.5}, cfg=cfg)
    p1 = Path(common.write_artifact(a, {"v": np.float64(0.1), "w": np.arange(3), "n": np.int64(4)}, upstream=up, cfg=cfg))
    b1 = p1.read_bytes()
    p2 = Path(common.write_artifact(a, {"v": np.float64(0.1), "w": np.arange(3), "n": np.int64(4)}, upstream=up, cfg=cfg))
    assert b1 == p2.read_bytes()
    obj = json.loads(b1.decode("utf-8"))
    assert obj["upstream_sha256"] == hashlib.sha256((ROOT / "artifacts" / f"{up}.json").read_bytes()).hexdigest()
    assert obj["config_sha256"] == hashlib.sha256((ROOT / "config.yaml").read_bytes()).hexdigest()
    assert obj["seed"] == 44615
    assert obj["w"] == [0, 1, 2] and obj["v"] == 0.1 and obj["n"] == 4
    assert b"\r" not in b1
    # sorted keys, indent=1
    assert b1.decode("utf-8").startswith('{\n "')
    keys = list(obj.keys())
    assert keys == sorted(keys)
    assert common.read_artifact(a) == obj


def test_write_artifact_no_upstream_is_null(art_env):
    _, _, b = art_env
    p = Path(common.write_artifact(b, {"x": 1}, cfg=common.load_config()))
    assert json.loads(p.read_text(encoding="utf-8"))["upstream_sha256"] is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_write_artifact_rejects_nonfinite(art_env, bad):
    _, a, _ = art_env
    with pytest.raises(Exception):
        common.write_artifact(a, {"x": bad}, cfg=common.load_config())
    with pytest.raises(Exception):
        common.write_artifact(a, {"x": np.array([1.0, bad])}, cfg=common.load_config())


def test_sha256_file(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert common.sha256_file(p) == hashlib.sha256(b"abc").hexdigest()


def test_load_train_columns_and_dates(train_all):
    assert len(train_all) == 8708 + 2178
    assert pd.api.types.is_datetime64_any_dtype(train_all["dteday"])
    for c in ["instant", "hr", "weathersit", "hum", "cnt"]:
        assert c in train_all.columns


def test_load_test_leaves_unmodified_by_split(cfg, train_all):
    test_df = common.load_test(cfg)
    n0, cols0 = len(test_df), list(test_df.columns)
    common.seeded_split(train_all, 1)
    assert len(test_df) == n0 and list(test_df.columns) == cols0
