"""Tests for src.regularization (ARCHITECTURE section 7, S-0-03).

Objective: (1/2n)||y - b - Xw||^2 + a*rho*||w||_1 + (a/2)(1-rho)||w||^2, intercept free.
"""
import numpy as np
import pytest

reg = pytest.importorskip("src.regularization")


def make_xy(n=300, p=15, seed=0, offset=True, dup=False):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, p))
    if dup:
        X[:, 1] = 0.99 * X[:, 0] + np.sqrt(1 - 0.99 ** 2) * r.normal(size=n)
        X[:, 4] = 0.99 * X[:, 3] + np.sqrt(1 - 0.99 ** 2) * r.normal(size=n)
    X = (X - X.mean(0)) / X.std(0)
    w_true = np.zeros(p)
    w_true[[0, 3, 6, 9]] = [1.5, -1.0, 0.7, 0.4]
    y = X @ w_true + 2.0 + r.normal(size=n)
    if offset:
        X = X + r.uniform(-1, 1, size=p)  # not centred: solver must handle the intercept
    return X, y


def gram_terms(X, y):
    Xc = X - X.mean(0)
    yc = y - y.mean()
    n = len(y)
    return Xc.T @ Xc / n, Xc.T @ yc / n


def objective(X, y, b, w, a, rho):
    n = len(y)
    res = y - b - X @ w
    return res @ res / (2 * n) + a * rho * np.abs(w).sum() + 0.5 * a * (1 - rho) * w @ w


# ---------------- ridge
@pytest.mark.parametrize("alpha", [1e-4, 1e-2, 1.0])
def test_ridge_vs_sklearn(alpha):
    from sklearn.linear_model import Ridge  # leak-ok: oracle
    X, y = make_xy(300, 15, 1)
    b, w = reg.ridge_closed_form(X, y, alpha)
    ref = Ridge(alpha=len(y) * alpha, fit_intercept=True).fit(X, y)  # leak-ok: oracle
    assert np.abs(w - ref.coef_).max() < 1e-6
    assert b == pytest.approx(ref.intercept_, abs=1e-6)


def test_ridge_alpha_zero_limit_is_ols():
    X, y = make_xy(200, 6, 2)
    b, w = reg.ridge_closed_form(X, y, 1e-12)
    A = np.column_stack([np.ones(len(y)), X])
    ref = np.linalg.lstsq(A, y, rcond=None)[0]  # leak-ok: oracle
    assert np.abs(w - ref[1:]).max() < 1e-6 and abs(b - ref[0]) < 1e-6


def test_ridge_path_matches_closed_form():
    X, y = make_xy(250, 10, 3)
    alphas = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])
    B, W = reg.ridge_path(X, y, alphas)
    assert np.asarray(W).shape == (5, 10)
    for i, a in enumerate(alphas):
        b, w = reg.ridge_closed_form(X, y, a)
        assert np.abs(W[i] - w).max() < 1e-8
        assert B[i] == pytest.approx(b, abs=1e-8)


# ---------------- elastic net / lasso
@pytest.mark.parametrize("dup", [False, True])
@pytest.mark.parametrize("rho", [1.0, 0.5, 0.1])
@pytest.mark.parametrize("alpha", [1e-3, 1e-2, 1e-1])
def test_enet_vs_sklearn(alpha, rho, dup):
    from sklearn.linear_model import ElasticNet, Lasso  # leak-ok: oracle
    X, y = make_xy(300, 15, 4, dup=dup)
    b, w, sweeps = reg.enet_cd(X, y, alpha, rho)
    if rho == 1.0:
        ref = Lasso(alpha=alpha, tol=1e-12, max_iter=200000).fit(X, y)  # leak-ok: oracle
    else:
        ref = ElasticNet(alpha=alpha, l1_ratio=rho, tol=1e-12, max_iter=200000).fit(X, y)  # leak-ok: oracle
    assert np.abs(w - ref.coef_).max() < 1e-3
    assert b == pytest.approx(ref.intercept_, abs=1e-3)
    assert sweeps >= 1
    # our objective is not worse than sklearn's by more than float noise
    assert objective(X, y, b, w, alpha, rho) <= objective(X, y, ref.intercept_, ref.coef_, alpha, rho) + 1e-9


@pytest.mark.parametrize("dup", [False, True])
@pytest.mark.parametrize("rho", [1.0, 0.5, 0.1])
@pytest.mark.parametrize("alpha", [1e-3, 1e-2, 1e-1])
def test_enet_kkt(alpha, rho, dup):
    X, y = make_xy(300, 15, 5, dup=dup)
    b, w, _ = reg.enet_cd(X, y, alpha, rho)
    G, c = gram_terms(X, y)
    grad = c - G @ w - alpha * (1 - rho) * w
    zero = w == 0
    assert np.all(np.abs(grad[zero]) <= alpha * rho + 1e-6)
    nz = ~zero
    assert np.all(np.abs(grad[nz] - alpha * rho * np.sign(w[nz])) <= 1e-6)
    # intercept = ybar - xbar.w
    assert b == pytest.approx(y.mean() - X.mean(0) @ w, abs=1e-10)


def test_enet_lasso_is_sparse_at_large_alpha():
    X, y = make_xy(300, 15, 6)
    _, w, _ = reg.enet_cd(X, y, 0.3, 1.0)
    assert np.count_nonzero(w) < 15


def test_enet_warm_start_same_solution():
    X, y = make_xy(300, 15, 7)
    _, w_cold, s_cold = reg.enet_cd(X, y, 0.02, 0.5)
    _, w_warm, s_warm = reg.enet_cd(X, y, 0.02, 0.5, w0=w_cold)
    assert np.abs(w_warm - w_cold).max() < 1e-8
    assert s_warm <= s_cold


def test_enet_constant_column_stays_zero():
    X, y = make_xy(200, 6, 8, offset=False)
    X[:, 2] = 0.0
    b, w, _ = reg.enet_cd(X, y, 0.01, 0.5)
    assert w[2] == 0.0 and np.isfinite(w).all() and np.isfinite(b)


def test_enet_rho_zero_equals_ridge():
    X, y = make_xy(200, 8, 9)
    b1, w1 = reg.ridge_closed_form(X, y, 0.05)
    b2, w2, _ = reg.enet_cd(X, y, 0.05, 0.0, tol=1e-13)
    assert np.abs(w1 - w2).max() < 1e-6


def test_enet_deterministic():
    X, y = make_xy(200, 8, 10)
    a = reg.enet_cd(X, y, 0.02, 0.5)
    b = reg.enet_cd(X, y, 0.02, 0.5)
    assert a[1].tobytes() == b[1].tobytes() and a[0] == b[0] and a[2] == b[2]


# ---------------- alpha_max / grid
@pytest.mark.parametrize("rho", [1.0, 0.5, 0.1])
def test_alpha_max(rho):
    X, y = make_xy(300, 15, 11)
    _, c = gram_terms(X, y)
    a_max = reg.alpha_max(X, y, rho)
    assert a_max == pytest.approx(np.abs(c).max() / rho, rel=1e-12)
    _, w0, _ = reg.enet_cd(X, y, a_max, rho)
    assert np.all(w0 == 0) or np.abs(w0).max() < 1e-12
    _, w1, _ = reg.enet_cd(X, y, 0.99 * a_max, rho)
    assert np.count_nonzero(np.abs(w1) > 1e-12) >= 1


def test_alpha_grid_log_spaced_descending():
    g = np.asarray(reg.alpha_grid(2.0, 1e-4, 40))
    assert len(g) == 40
    assert g[0] == pytest.approx(2.0, rel=1e-12) and g[-1] == pytest.approx(2e-4, rel=1e-9)
    assert np.all(np.diff(g) < 0)
    ratios = g[1:] / g[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-9)


# ---------------- path
@pytest.mark.parametrize("rho", [1.0, 0.5])
def test_path_warm_equals_cold(rho):
    X, y = make_xy(300, 12, 12)
    a_max = reg.alpha_max(X, y, rho)
    alphas = reg.alpha_grid(a_max, 1e-3, 12)
    B, W, sweeps = reg.enet_path(X, y, alphas, rho)
    W = np.asarray(W)
    assert W.shape == (12, 12) and len(B) == 12 and len(sweeps) == 12
    for i, a in enumerate(alphas):
        b, w, _ = reg.enet_cd(X, y, a, rho)
        assert np.abs(W[i] - w).max() < 1e-6, i
        assert B[i] == pytest.approx(b, abs=1e-6)
    assert np.all(W[0] == 0) or np.abs(W[0]).max() < 1e-12


# ---------------- stability selection
def stability_data(seed=13):
    r = np.random.default_rng(seed)
    n, p = 300, 6
    X = r.normal(size=(n, p))
    X = (X - X.mean(0)) / X.std(0)
    y = 3.0 * X[:, 0] + 1.0 * r.normal(size=n)
    groups = np.repeat(np.arange(60), 5)
    return X, y, groups


def test_stability_selection_properties():
    X, y, g = stability_data()
    f = np.asarray(reg.stability_selection(X, y, 0.05, 1.0, g, seed=7, B=50))
    assert f.shape == (6,)
    assert np.all((f >= 0) & (f <= 1))
    assert np.allclose(f * 50, np.round(f * 50), atol=1e-9)  # multiples of 1/B
    assert f[0] >= 0.95
    assert f[5] < f[0]
    assert f[1:].mean() < f[0]


def test_stability_selection_deterministic():
    X, y, g = stability_data()
    a = reg.stability_selection(X, y, 0.05, 1.0, g, seed=7, B=20)
    b = reg.stability_selection(X, y, 0.05, 1.0, g, seed=7, B=20)
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_stability_selection_large_alpha_selects_nothing():
    X, y, g = stability_data()
    a_max = reg.alpha_max(X, y, 1.0)
    f = np.asarray(reg.stability_selection(X, y, 5 * a_max, 1.0, g, seed=1, B=10))
    assert np.all(f == 0)
