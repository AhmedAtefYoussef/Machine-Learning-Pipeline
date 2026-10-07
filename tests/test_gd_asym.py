"""Tests for src.gd_asym (S-0-02: raw-scale asymmetric squared cost, log-link model)."""
import numpy as np
import pytest

ga = pytest.importorskip("src.gd_asym")
gd = pytest.importorskip("src.gd")


def log_linear_data(n=500, p=5, seed=0, sigma=0.5):
    r = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), r.normal(size=(n, p - 1)) * 0.5])
    w_true = np.concatenate([[3.0], r.normal(size=p - 1) * 0.3])
    z = X @ w_true + sigma * r.normal(size=n)
    y = np.expm1(z)
    return X, y, w_true


def test_loss_hand_example():
    X = np.ones((3, 1))
    w = np.array([np.log(4.0)])  # yhat = 3
    y = np.array([1.0, 5.0, 3.0])  # r = 2, -2, 0
    assert ga.asym_loss(X, y, w, k=3.0) == pytest.approx(16.0 / 3.0, rel=1e-12)
    assert ga.asym_loss(X, y, w, k=1.0) == pytest.approx(8.0 / 3.0, rel=1e-12)


def test_grad_hand_example():
    X = np.ones((3, 1))
    w = np.array([np.log(4.0)])
    y = np.array([1.0, 5.0, 3.0])
    g = ga.asym_grad(X, y, w, k=3.0)
    assert g.shape == (1,)
    assert g[0] == pytest.approx(-32.0 / 3.0, rel=1e-12)


def test_loss_matches_formula_random():
    X, y, _ = log_linear_data(100, 4, 1)
    w = np.random.default_rng(2).normal(size=4) * 0.3 + np.array([3.0, 0, 0, 0])
    r = np.expm1(X @ w) - y
    c = np.where(r < 0, 3.0, 1.0)
    assert ga.asym_loss(X, y, w, 3.0) == pytest.approx(np.mean(c * r * r), rel=1e-12)


@pytest.mark.parametrize("k", [3.0, 1.0, 5.0])
def test_gradient_check(k):
    X, y, _ = log_linear_data(200, 5, 3)
    r = np.random.default_rng(4)
    for _ in range(3):
        w = r.normal(size=5) * 0.4
        w[0] = r.uniform(1.0, 2.0)
        assert np.abs(X @ w).max() <= 3.0
        resid = np.expm1(X @ w) - y
        assert np.all(resid != 0)
        loss = lambda v: ga.asym_loss(X, y, v, k)
        grad = lambda v: ga.asym_grad(X, y, v, k)
        assert gd.gradient_check(loss, grad, w, n_coords=5, eps=1e-6, seed=0) < 1e-6


def test_operator_costs_hand_example():
    y = np.array([1.0, 5.0, 3.0, 10.0])
    yhat = np.array([3.0, 3.0, 3.0, 7.0])  # r = 2, -2, 0, -3
    out = ga.operator_costs(y, yhat, k=3.0)
    assert out["sq_cost"] == pytest.approx(43.0 / 4.0, rel=1e-12)
    assert out["abs_cost"] == pytest.approx(17.0 / 4.0, rel=1e-12)
    assert out["under_share"] == pytest.approx(0.5, abs=1e-15)
    assert out["mean_error"] == pytest.approx(-0.75, abs=1e-12)
    assert set(["sq_cost", "abs_cost", "under_share", "mean_error"]) <= set(out)


def test_operator_costs_symmetric_when_k1():
    r = np.random.default_rng(0)
    y = r.uniform(1, 100, size=50)
    yhat = y + r.normal(size=50) * 5
    out = ga.operator_costs(y, yhat, k=1.0)
    assert out["sq_cost"] == pytest.approx(np.mean((yhat - y) ** 2), rel=1e-12)
    assert out["abs_cost"] == pytest.approx(np.mean(np.abs(yhat - y)), rel=1e-12)


def test_fit_lowers_loss_and_reduces_under_share():
    X, y, _ = log_linear_data(500, 5, 5)
    z = np.log1p(y)
    w0 = np.linalg.lstsq(X, z, rcond=None)[0]  # leak-ok: oracle
    res = ga.fit_asymmetric(X, y, w0, k=3.0, max_iter=5000)
    assert ga.asym_loss(X, y, res.weights, 3.0) < ga.asym_loss(X, y, w0, 3.0)
    yh0 = np.expm1(X @ w0)
    yh1 = np.expm1(X @ res.weights)
    assert yh1.mean() > yh0.mean()
    u0 = ga.operator_costs(y, yh0, 3.0)["under_share"]
    u1 = ga.operator_costs(y, yh1, 3.0)["under_share"]
    assert u1 < u0 and u1 < 0.5
    h = np.asarray(res.loss_history)
    assert np.all(np.diff(h) <= 1e-12 * np.abs(h[:-1]))  # Armijo => non-increasing
    assert res.stop_reason in ("converged", "max_iter")


def test_fit_recovers_noise_free_k1():
    X, _, w_true = log_linear_data(300, 4, 6)
    y = np.expm1(X @ w_true)
    w0 = w_true + np.array([0.0, 0.2, -0.2, 0.1])
    res = ga.fit_asymmetric(X, y, w0, k=1.0, max_iter=20000)
    assert ga.asym_loss(X, y, res.weights, 1.0) < 1e-3 * ga.asym_loss(X, y, w0, 1.0)


def test_fit_deterministic():
    X, y, _ = log_linear_data(200, 4, 7)
    w0 = np.linalg.lstsq(X, np.log1p(y), rcond=None)[0]  # leak-ok: oracle
    a = ga.fit_asymmetric(X, y, w0, 3.0, max_iter=300)
    b = ga.fit_asymmetric(X, y, w0, 3.0, max_iter=300)
    assert a.weights.tobytes() == b.weights.tobytes()
    assert a.iterations == b.iterations
