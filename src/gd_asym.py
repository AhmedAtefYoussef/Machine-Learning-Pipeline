"""Bonus (ADR-008): asymmetric cost on the bike scale for the power-target model.

Model: yhat = q(X w) - 1, where q is the inverse of the target transform (src.common.Target; exp for the log).
X has the bias column. Residual r = yhat - y (y = cnt). Cost weight c_i = k if r_i < 0 (under-prediction) else 1.
"""
import numpy as np

from src.common import Target
from src.gd import GDResult


def _eta(X, w, y, target: Target, cap=None):
    """eta = X w, held at or below the cap (default target.eta_cap(y)) so that q(eta) stays finite."""
    cap = target.eta_cap(y) if cap is None else cap
    return np.minimum(X @ w, cap)


def _cost_weights(r, k):
    """c_i = k where r_i < 0 else 1."""
    return np.where(r < 0, k, 1.0)


def asym_loss(X, y, w, target: Target, k=3.0, cap=None):
    """L(w) = (1/n) sum_i c_i r_i^2 with r = q(Xw) - 1 - y."""
    r = target.q(_eta(X, w, y, target, cap)) - 1.0 - y
    return float(np.mean(_cost_weights(r, k) * r ** 2))


def asym_grad(X, y, w, target: Target, k=3.0, cap=None):
    """grad L = (2/n) X^T (c * r * dq/deta); shape (p,).

    The chain rule through q gives the dq/deta factor (exp(Xw) for the log). The weight switch has no
    derivative term because r = 0 exactly where c changes.
    """
    eta = _eta(X, w, y, target, cap)
    r = target.q(eta) - 1.0 - y
    return 2.0 / len(y) * (X.T @ (_cost_weights(r, k) * r * target.dq_deta(eta)))


def _backtrack(loss_fn, w, loss, grad, step):
    """Halve the step until the Armijo condition L(w - t g) <= L(w) - 1e-4 t ||g||^2 holds.

    Returns (step, new_w, new_loss), or None if no acceptable step is found.
    """
    grad_sq = float(grad @ grad)
    for _ in range(60):
        w_try = w - step * grad
        loss_try = loss_fn(w_try)
        if np.isfinite(loss_try) and loss_try <= loss - 1e-4 * step * grad_sq:
            return step, w_try, loss_try
        step /= 2.0
    return None


def fit_asymmetric(X, y, w0, target: Target, k=3.0, max_iter=20000, tol_loss=1e-10, t0=1e-3):
    """Gradient descent with Armijo backtracking on asym_loss.

    Each iteration tries step t, halves it until sufficient decrease, accepts, and the
    next iteration starts from 2t. "converged" when the relative loss change is
    <= tol_loss for 10 consecutive iterations; "stalled" if no step decreases the loss;
    otherwise "max_iter".
    """
    cap = target.eta_cap(y)   # computed once per fit

    def loss_fn(w):
        return asym_loss(X, y, w, target, k, cap)

    w = np.array(w0, dtype=np.float64)
    loss = loss_fn(w)
    history = [loss]
    step = t0
    quiet_count = 0
    stop_reason = "max_iter"
    iterations = 0

    for iterations in range(1, max_iter + 1):
        grad = asym_grad(X, y, w, target, k, cap)
        found = _backtrack(loss_fn, w, loss, grad, step)
        if found is None:
            stop_reason = "stalled"
            iterations -= 1
            break
        step, w, loss_new = found
        quiet_count = quiet_count + 1 if abs(loss - loss_new) <= tol_loss * max(loss, 1e-300) else 0
        loss = loss_new
        history.append(loss)
        step *= 2.0
        if quiet_count >= 10:
            stop_reason = "converged"
            break

    grad_norm = float(np.linalg.norm(asym_grad(X, y, w, target, k, cap)))
    return GDResult(weights=w, iterations=iterations, stop_reason=stop_reason,
                    loss_history=[float(v) for v in history],
                    grad_norm_final=grad_norm, loss_final=float(loss))


def operator_costs(y, yhat, k=3.0):
    """Costs of a prediction vector, r = yhat - y.

    sq_cost = mean(c r^2); abs_cost = mean(k max(-r,0) + max(r,0));
    under_share = mean(r < 0); mean_error = mean(r).
    """
    r = np.asarray(yhat, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {
        "sq_cost": float(np.mean(_cost_weights(r, k) * r ** 2)),
        "abs_cost": float(np.mean(k * np.maximum(-r, 0.0) + np.maximum(r, 0.0))),
        "under_share": float(np.mean(r < 0)),
        "mean_error": float(np.mean(r)),
    }
