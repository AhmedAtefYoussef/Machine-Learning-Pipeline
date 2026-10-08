"""Tests for src.gd and src.poly (ARCHITECTURE section 6, gd-numerics skill)."""
import numpy as np
import pytest

gd = pytest.importorskip("src.gd")
poly = pytest.importorskip("src.poly")


def make_problem(n=200, p=8, seed=0, noise=0.3):
    r = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), r.normal(size=(n, p - 1))])
    w_true = r.normal(size=p)
    y = X @ w_true + noise * r.normal(size=n)
    return X, y, w_true


def closures(X, y):
    return (lambda w: gd.mse_loss(X, y, w)), (lambda w: gd.mse_grad(X, y, w))


def test_mse_loss_and_grad_formula():
    X, y, _ = make_problem(30, 4, 1)
    w = np.array([0.3, -1.0, 2.0, 0.5])
    res = X @ w - y
    assert gd.mse_loss(X, y, w) == pytest.approx(res @ res / (2 * len(y)), rel=1e-13)
    assert np.allclose(gd.mse_grad(X, y, w), X.T @ res / len(y), rtol=1e-13, atol=1e-14)


def test_gradient_check_mse_independent_fd():
    X, y, _ = make_problem(200, 8, 2)
    loss, grad = closures(X, y)
    w = np.random.default_rng(5).normal(size=8)
    g = grad(w)
    eps = 1e-6
    for j in range(8):
        e = np.zeros(8)
        e[j] = eps
        fd = (loss(w + e) - loss(w - e)) / (2 * eps)
        assert abs(fd - g[j]) / max(abs(g[j]), 1e-12) < 1e-6
    assert gd.gradient_check(loss, grad, w, n_coords=8, eps=1e-6, seed=0) < 1e-6


def test_gradient_check_detects_wrong_gradient():
    X, y, _ = make_problem(100, 5, 3)
    loss, grad = closures(X, y)
    w = np.random.default_rng(1).normal(size=5)
    assert gd.gradient_check(loss, lambda v: 1.1 * grad(v), w, n_coords=5, seed=0) > 1e-3


def test_lambda_max_vs_eigvalsh():
    X, _, _ = make_problem(150, 6, 4)
    ref = np.linalg.eigvalsh(X.T @ X / len(X)).max()  # leak-ok: oracle
    assert gd.lambda_max(X) == pytest.approx(ref, rel=1e-12)


def test_converges_to_lstsq():
    X, y, _ = make_problem(200, 8, 6)
    loss, grad = closures(X, y)
    lam = gd.lambda_max(X)
    res = gd.gradient_descent(loss, grad, np.zeros(8), lr=1.0 / lam)
    w_star = np.linalg.lstsq(X, y, rcond=None)[0]  # leak-ok: oracle
    assert res.stop_reason == "converged"
    assert np.abs(res.weights - w_star).max() < 1e-3
    assert res.iterations > 0
    assert res.loss_final == pytest.approx(loss(res.weights), rel=1e-12)
    assert res.grad_norm_final == pytest.approx(np.linalg.norm(grad(res.weights)), rel=1e-9, abs=1e-12)
    assert res.grad_norm_final < 1e-6


def test_loss_history_start_and_monotone():
    X, y, _ = make_problem(200, 8, 7)
    loss, grad = closures(X, y)
    lam = gd.lambda_max(X)
    w0 = np.random.default_rng(0).normal(size=8)
    for frac in (1.0, 0.5, 0.1):
        res = gd.gradient_descent(loss, grad, w0, lr=frac / lam, max_iter=300)
        h = np.asarray(res.loss_history)
        assert h[0] == loss(w0)
        assert np.all(np.diff(h) <= 1e-12 * np.abs(h[:-1]))


def test_record_every_thins_history():
    X, y, _ = make_problem(100, 5, 8)
    loss, grad = closures(X, y)
    lam = gd.lambda_max(X)
    full = gd.gradient_descent(loss, grad, np.zeros(5), lr=0.5 / lam, max_iter=100, tol_loss=0, tol_grad=0)
    thin = gd.gradient_descent(loss, grad, np.zeros(5), lr=0.5 / lam, max_iter=100, tol_loss=0, tol_grad=0, record_every=10)
    assert thin.loss_history[0] == full.loss_history[0]
    assert len(thin.loss_history) < len(full.loss_history)
    assert np.array_equal(thin.weights, full.weights)


def test_diverged_stop():
    X, y, _ = make_problem(200, 8, 9)
    loss, grad = closures(X, y)
    lam = gd.lambda_max(X)
    res = gd.gradient_descent(loss, grad, np.zeros(8), lr=1.05 * 2.0 / lam, max_iter=50000)
    assert res.stop_reason == "diverged"
    assert res.iterations < 50000


def test_max_iter_stop():
    X, y, _ = make_problem(200, 8, 10)
    loss, grad = closures(X, y)
    lam = gd.lambda_max(X)
    res = gd.gradient_descent(loss, grad, np.zeros(8), lr=1e-4 * 2.0 / lam, max_iter=500)
    assert res.stop_reason == "max_iter"
    assert res.iterations == 500
    assert res.loss_final < res.loss_history[0]


def test_update_rule_single_step():
    X, y, _ = make_problem(50, 4, 11)
    loss, grad = closures(X, y)
    w0 = np.array([1.0, -2.0, 0.5, 0.0])
    res = gd.gradient_descent(loss, grad, w0, lr=0.01, max_iter=1, tol_loss=0, tol_grad=0)
    assert np.allclose(res.weights, w0 - 0.01 * grad(w0), rtol=1e-13, atol=1e-15)


def test_does_not_mutate_w0():
    X, y, _ = make_problem(50, 4, 12)
    loss, grad = closures(X, y)
    w0 = np.zeros(4)
    gd.gradient_descent(loss, grad, w0, lr=0.1 / gd.lambda_max(X), max_iter=20)
    assert np.all(w0 == 0)


def test_determinism_bitwise():
    X, y, _ = make_problem(200, 8, 13)
    loss, grad = closures(X, y)
    lr = 0.5 / gd.lambda_max(X)
    a = gd.gradient_descent(loss, grad, np.zeros(8), lr=lr, max_iter=2000)
    b = gd.gradient_descent(loss, grad, np.zeros(8), lr=lr, max_iter=2000)
    assert a.weights.tobytes() == b.weights.tobytes()
    assert np.asarray(a.loss_history).tobytes() == np.asarray(b.loss_history).tobytes()
    assert a.iterations == b.iterations and a.stop_reason == b.stop_reason


# ---- poly.lift_weights
def test_lift_weights_slots_and_zeros():
    small = ["bias", "a", "b"]
    big = ["bias", "a", "a^2", "b", "a*b"]
    out = poly.lift_weights(np.array([1.0, 2.0, 3.0]), small, big)
    assert np.array_equal(out, np.array([1.0, 2.0, 0.0, 3.0, 0.0]))


def test_lift_weights_reordered_and_missing():
    out = poly.lift_weights(np.array([1.0, 2.0, 3.0]), ["bias", "a", "b"], ["b", "x", "bias", "a"])
    assert np.array_equal(out, np.array([3.0, 0.0, 1.0, 2.0]))
    with pytest.raises(Exception):
        poly.lift_weights(np.array([1.0, 2.0]), ["bias", "zzz"], ["bias", "a"])


def test_lift_preserves_loss_on_synthetic_expansion():
    X, y, _ = make_problem(120, 5, 14)
    names = ["bias", "a", "b", "c", "d"]
    extra = np.column_stack([X[:, 1] ** 2, X[:, 1] * X[:, 2]])
    Xb = np.column_stack([X, extra])
    names_big = names + ["a^2", "a*b"]
    w = np.random.default_rng(3).normal(size=5)
    lifted = poly.lift_weights(w, names, names_big)
    assert abs(gd.mse_loss(Xb, y, lifted) - gd.mse_loss(X, y, w)) < 1e-12


def test_lift_preserves_loss_on_real_design(train_df):
    features = pytest.importorskip("src.features")
    common = pytest.importorskip("src.common")
    base = features.Design(features.DesignSpec(base=tuple(features.BASE), power_cols=(), degree=1, blocks=())).fit(train_df)
    big_spec = features.DesignSpec(base=tuple(features.BASE), power_cols=("temp", "hum", "windspeed"), degree=3, blocks=("wd_x_hr",))
    big = features.Design(big_spec).fit(train_df, base_scaler=base.scaler_dict())
    Xb, Xg = base.transform(train_df), big.transform(train_df)
    z = common.Target(0.1, "ls").forward(train_df["cnt"].to_numpy(dtype=float))
    w = np.random.default_rng(0).normal(size=Xb.shape[1]) * 0.3
    lifted = poly.lift_weights(w, base.names, big.names)
    assert abs(gd.mse_loss(Xg, z, lifted) - gd.mse_loss(Xb, z, w)) < 1e-12
    assert np.count_nonzero(lifted[len(w):]) == 0
