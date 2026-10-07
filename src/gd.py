"""Own gradient-descent engine (numpy only).

Notation: X is (n, p) and already contains the bias column, y is (n,), w is (p,).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GDResult:
    """Outcome of one gradient-descent run."""
    weights: np.ndarray                 # (p,) final weights
    iterations: int                     # number of updates performed
    stop_reason: str                    # "converged" | "diverged" | "max_iter"
    loss_history: list = field(default_factory=list)  # loss at w0, then every record_every iterations
    grad_norm_final: float = float("nan")             # ||grad L(w_final)||_2
    loss_final: float = float("nan")                  # L(w_final)


def mse_loss(X, y, w):
    """L(w) = (1/2n) * ||X w - y||^2.   X (n,p), y (n,), w (p,) -> float."""
    residual = X @ w - y
    return float(residual @ residual / (2.0 * len(y)))


def mse_grad(X, y, w):
    """grad L(w) = (1/n) * X^T (X w - y).   Returns shape (p,)."""
    residual = X @ w - y
    return X.T @ residual / len(y)


def lambda_max(X):
    """Largest eigenvalue of X^T X / n (symmetric, so eigvalsh). Stable lr < 2 / lambda_max."""
    gram = X.T @ X / X.shape[0]
    return float(np.linalg.eigvalsh(gram)[-1])


def _has_converged(loss_prev, loss_new, grad_norm, tol_loss, tol_grad):
    """Both tests must hold: relative loss change <= tol_loss AND ||grad|| < tol_grad."""
    small_change = abs(loss_new - loss_prev) <= tol_loss * max(loss_prev, 1e-300)
    return small_change and grad_norm < tol_grad


def gradient_descent(loss_fn, grad_fn, w0, lr, tol_loss=1e-10, tol_grad=1e-6,
                     max_iter=50000, record_every=1):
    """Plain gradient descent: w <- w - lr * grad_fn(w).

    loss_fn(w) -> float and grad_fn(w) -> (p,) are closures over the data.
    Stops "converged" when |L_new - L_prev| <= tol_loss * max(L_prev, 1e-300) and
    ||grad||_2 < tol_grad; "diverged" when the loss is non-finite or exceeds
    1e6 * L(w0); otherwise "max_iter". loss_history[0] is L(w0).
    """
    w = np.array(w0, dtype=np.float64)
    loss = loss_fn(w)
    loss_initial = loss
    grad = grad_fn(w)
    history = [loss]
    stop_reason = "max_iter"
    iterations = 0

    for iterations in range(1, max_iter + 1):
        w = w - lr * grad
        loss_new = loss_fn(w)
        grad = grad_fn(w)
        grad_norm = float(np.linalg.norm(grad))
        if iterations % record_every == 0:
            history.append(loss_new)
        if not np.isfinite(loss_new) or loss_new > 1e6 * loss_initial:
            loss, stop_reason = loss_new, "diverged"
            break
        if _has_converged(loss, loss_new, grad_norm, tol_loss, tol_grad):
            loss, stop_reason = loss_new, "converged"
            break
        loss = loss_new

    return GDResult(weights=w, iterations=iterations, stop_reason=stop_reason,
                    loss_history=[float(v) for v in history],
                    grad_norm_final=float(np.linalg.norm(grad)), loss_final=float(loss))


def gradient_check(loss_fn, grad_fn, w, n_coords=20, eps=1e-6, seed=0):
    """Max relative error between grad_fn and central differences.

    numeric_j = (L(w + eps e_j) - L(w - eps e_j)) / (2 eps), on a random subset of coordinates.
    error_j = |numeric_j - analytic_j| / max(|numeric_j| + |analytic_j|, 1e-12).
    """
    w = np.array(w, dtype=np.float64)
    analytic = grad_fn(w)
    rng = np.random.default_rng(seed)
    coords = rng.choice(len(w), size=min(n_coords, len(w)), replace=False)
    worst = 0.0
    for j in coords:
        step = np.zeros_like(w)
        step[j] = eps
        numeric = (loss_fn(w + step) - loss_fn(w - step)) / (2.0 * eps)
        error = abs(numeric - analytic[j]) / max(abs(numeric) + abs(analytic[j]), 1e-12)
        worst = max(worst, float(error))
    return worst
