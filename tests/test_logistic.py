"""Tests for src.logistic (ARCHITECTURE section 7, S-0-03)."""
import warnings

import numpy as np
import pytest

lg = pytest.importorskip("src.logistic")
gd = pytest.importorskip("src.gd")


def make_data(n=500, p=6, seed=0):
    r = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), r.normal(size=(n, p - 1))])
    w_true = np.concatenate([[-0.3], r.normal(size=p - 1)])
    pr = 1.0 / (1.0 + np.exp(-(X @ w_true)))
    y = (r.uniform(size=n) < pr).astype(float)
    return X, y


def test_sigmoid_values_and_stability():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = lg.sigmoid(np.array([-800.0, -40.0, 0.0, 40.0, 800.0]))
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(0.0, abs=1e-300) and out[-1] == pytest.approx(1.0, abs=1e-15)
    assert out[2] == 0.5
    assert out[3] == pytest.approx(1 / (1 + np.exp(-40.0)), rel=1e-14)
    z = np.linspace(-5, 5, 11)
    assert np.allclose(lg.sigmoid(z) + lg.sigmoid(-z), 1.0, atol=1e-15)


def test_logloss_hand_examples():
    X = np.array([[1.0], [1.0]])
    y = np.array([1.0, 0.0])
    assert lg.logloss(X, y, np.array([0.0])) == pytest.approx(np.log(2), rel=1e-14)
    X2 = np.array([[1.0, 2.0], [1.0, -1.0], [1.0, 0.5]])
    y2 = np.array([1.0, 0.0, 1.0])
    w = np.array([0.2, -0.4])
    p = 1 / (1 + np.exp(-(X2 @ w)))
    ref = -np.mean(y2 * np.log(p) + (1 - y2) * np.log(1 - p))
    assert lg.logloss(X2, y2, w) == pytest.approx(ref, rel=1e-13)
    # l2 penalty excludes the bias (column 0)
    assert lg.logloss(X2, y2, w, l2=0.3) == pytest.approx(ref + 0.15 * 0.4 ** 2, rel=1e-13)


def test_logloss_extreme_scores_finite():
    X = np.array([[1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert lg.logloss(X, np.array([0.0]), np.array([800.0])) == pytest.approx(800.0, rel=1e-12)
        assert lg.logloss(X, np.array([1.0]), np.array([800.0])) == pytest.approx(0.0, abs=1e-12)
        assert lg.logloss(X, np.array([1.0]), np.array([-800.0])) == pytest.approx(800.0, rel=1e-12)


@pytest.mark.parametrize("l2", [0.0, 0.1])
def test_gradient_check(l2):
    X, y = make_data(200, 6, 1)
    w = np.random.default_rng(2).normal(size=6) * 0.5
    loss = lambda v: lg.logloss(X, y, v, l2)
    grad = lambda v: lg.logloss_grad(X, y, v, l2)
    assert gd.gradient_check(loss, grad, w, n_coords=6, eps=1e-6, seed=0) < 1e-6
    # independent finite differences
    g = grad(w)
    for j in range(6):
        e = np.zeros(6)
        e[j] = 1e-6
        fd = (loss(w + e) - loss(w - e)) / 2e-6
        assert abs(fd - g[j]) / max(abs(g[j]), 1e-9) < 1e-5


def test_grad_formula_and_bias_unpenalised():
    X, y = make_data(100, 4, 3)
    w = np.array([5.0, 1.0, -2.0, 0.5])
    p = 1 / (1 + np.exp(-(X @ w)))
    base = X.T @ (p - y) / len(y)
    g = lg.logloss_grad(X, y, w, l2=0.7)
    assert g[0] == pytest.approx(base[0], rel=1e-12)
    assert np.allclose(g[1:], base[1:] + 0.7 * w[1:], rtol=1e-12)


@pytest.mark.parametrize("l2", [1e-2, 1e-1])
def test_fit_vs_sklearn(l2):
    from sklearn.linear_model import LogisticRegression  # leak-ok: oracle
    X, y = make_data(500, 6, 4)
    res = lg.fit_logistic(X, y, l2=l2)
    n = len(y)
    ref = LogisticRegression(C=1.0 / (n * l2), tol=1e-10, max_iter=10000).fit(X[:, 1:], y)  # leak-ok: oracle
    w_ref = np.concatenate([ref.intercept_, ref.coef_.ravel()])
    assert res.stop_reason == "converged"
    assert np.abs(res.weights - w_ref).max() < 1e-2
    auc_ours = lg.roc_auc(y, lg.sigmoid(X @ res.weights))
    from sklearn.metrics import roc_auc_score  # leak-ok: oracle
    auc_ref = roc_auc_score(y, ref.decision_function(X[:, 1:]))  # leak-ok: oracle
    assert abs(auc_ours - auc_ref) < 1e-3


def test_fit_no_penalty_matches_sklearn_large_C():
    from sklearn.linear_model import LogisticRegression  # leak-ok: oracle
    X, y = make_data(500, 5, 5)
    res = lg.fit_logistic(X, y, l2=0.0)
    ref = LogisticRegression(C=1e10, tol=1e-12, max_iter=10000).fit(X[:, 1:], y)  # leak-ok: oracle
    w_ref = np.concatenate([ref.intercept_, ref.coef_.ravel()])
    assert np.abs(res.weights - w_ref).max() < 1e-2


def test_fit_decreases_loss_and_deterministic():
    X, y = make_data(300, 5, 6)
    a = lg.fit_logistic(X, y, l2=1e-3)
    b = lg.fit_logistic(X, y, l2=1e-3)
    assert a.weights.tobytes() == b.weights.tobytes()
    assert a.loss_final < lg.logloss(X, y, np.zeros(5), 1e-3)
    h = np.asarray(a.loss_history)
    assert np.all(np.diff(h) <= 1e-12 * np.abs(h[:-1]))
    assert h[0] == pytest.approx(np.log(2), rel=1e-12)


# ---------------- metrics
def tied_scores(seed, n=400):
    r = np.random.default_rng(seed)
    y = (r.uniform(size=n) < 0.3).astype(int)
    s = np.round(r.normal(size=n) + 1.2 * y, 1)  # heavy ties
    return y, s


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_roc_auc_vs_sklearn_with_ties(seed):
    from sklearn.metrics import roc_auc_score  # leak-ok: oracle
    y, s = tied_scores(seed)
    assert abs(lg.roc_auc(y, s) - roc_auc_score(y, s)) < 1e-10  # leak-ok: oracle


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pr_auc_vs_sklearn_with_ties(seed):
    from sklearn.metrics import average_precision_score  # leak-ok: oracle
    y, s = tied_scores(seed)
    assert abs(lg.pr_auc(y, s) - average_precision_score(y, s)) < 1e-10  # leak-ok: oracle


def test_auc_extremes():
    y = np.array([0, 0, 0, 1, 1])
    assert lg.roc_auc(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == pytest.approx(1.0, abs=1e-15)
    assert lg.roc_auc(y, np.array([0.9, 0.8, 0.7, 0.2, 0.1])) == pytest.approx(0.0, abs=1e-15)
    assert lg.roc_auc(y, np.full(5, 0.5)) == pytest.approx(0.5, abs=1e-15)
    assert lg.pr_auc(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == pytest.approx(1.0, abs=1e-15)


Y_H = np.array([1, 1, 1, 0, 0, 0, 0, 1])
P_H = np.array([0.9, 0.8, 0.3, 0.6, 0.2, 0.1, 0.4, 0.7])


def test_confusion_hand_example():
    c = lg.confusion(Y_H, P_H, 0.5)
    assert (c["tp"], c["fp"], c["tn"], c["fn"]) == (3, 1, 3, 1)
    c2 = lg.confusion(Y_H, P_H, 0.95)  # predicts nothing
    assert (c2["tp"], c2["fp"], c2["tn"], c2["fn"]) == (0, 0, 4, 4)
    c3 = lg.confusion(Y_H, P_H, 0.05)  # predicts everything
    assert (c3["tp"], c3["fp"], c3["tn"], c3["fn"]) == (4, 4, 0, 0)


def test_prf_hand_example():
    fn = getattr(lg, "prf", None) or getattr(lg, "f1_precision_recall")
    out = fn(Y_H, P_H, 0.5)
    if isinstance(out, dict):
        assert out["precision"] == pytest.approx(0.75)
        assert out["recall"] == pytest.approx(0.75)
        assert out["f1"] == pytest.approx(0.75)
        if "accuracy" in out:
            assert out["accuracy"] == pytest.approx(0.75)
    else:  # tuple form (f1, precision, recall)
        assert sorted(round(float(v), 12) for v in out) == [0.75, 0.75, 0.75]


def test_prf_zero_division_safe():
    fn = getattr(lg, "prf", None) or getattr(lg, "f1_precision_recall")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = fn(Y_H, P_H, 0.95)  # no predicted positives
    vals = list(out.values()) if isinstance(out, dict) else list(out)
    assert all(np.isfinite(v) for v in vals)


def test_calibration_table_hand_example():
    p = np.array([0.05, 0.15, 0.15, 0.95])
    y = np.array([0, 1, 0, 1])
    rows = [r for r in lg.calibration_table(y, p, bins=10) if r["n"] > 0]
    assert sum(r["n"] for r in rows) == 4
    assert [r["n"] for r in rows] == [1, 2, 1]
    assert rows[0]["mean_p"] == pytest.approx(0.05) and rows[0]["frac_pos"] == 0.0
    assert rows[1]["mean_p"] == pytest.approx(0.15) and rows[1]["frac_pos"] == pytest.approx(0.5)
    assert rows[2]["mean_p"] == pytest.approx(0.95) and rows[2]["frac_pos"] == 1.0
    assert rows[0]["lo"] == pytest.approx(0.0) and rows[0]["hi"] == pytest.approx(0.1)
    assert rows[2]["lo"] == pytest.approx(0.9) and rows[2]["hi"] == pytest.approx(1.0)


def test_calibration_includes_p_equal_one():
    p = np.array([0.0, 0.5, 1.0])
    y = np.array([0, 1, 1])
    rows = lg.calibration_table(y, p, bins=10)
    assert sum(r["n"] for r in rows) == 3
