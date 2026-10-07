"""Ridge, lasso and elastic-net solvers (numpy only).

Objective (sklearn scaling), intercept b not penalised:

    J(b, w) = (1/2n) ||y - b - X w||^2 + alpha*rho*||w||_1 + (alpha/2)(1-rho)||w||^2

rho = l1_ratio: rho = 0 is ridge, rho = 1 is lasso.  X has shape (n, p) and NO bias
column.  Columns of X and y are centred inside each function, and the intercept is
recovered as b = mean(y) - mean(X) . w, so any X is accepted.
"""
import numpy as np


def centred_gram(X, y):
    """Return (G, c, x_mean, y_mean) with G = Xc'Xc/n (p, p) and c = Xc'yc/n (p,)."""
    n = X.shape[0]
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    Xc = X - x_mean
    yc = y - y_mean
    return Xc.T @ Xc / n, Xc.T @ yc / n, x_mean, y_mean


def intercept(x_mean, y_mean, w):
    """b = ybar - xbar . w  (makes the fitted plane pass through the means)."""
    return y_mean - float(x_mean @ w)


def soft_threshold(z, t):
    """S(z, t) = sign(z) * max(|z| - t, 0)."""
    return np.sign(z) * max(abs(z) - t, 0.0)


def ridge_closed_form(X, y, alpha):
    """Solve (G + alpha I) w = c with G = Xc'Xc/n, c = Xc'yc/n.  Returns (b, w).

    Equals sklearn Ridge(alpha = n * alpha).
    """
    G, c, x_mean, y_mean = centred_gram(X, y)
    w = np.linalg.solve(G + alpha * np.eye(G.shape[0]), c)
    return intercept(x_mean, y_mean, w), w


def ridge_path(X, y, alphas):
    """Ridge for many alphas from one eigendecomposition G = V diag(lam) V'.

    w(alpha) = V diag(1/(lam + alpha)) V' c.  Returns (B (len,), W (len, p)).
    """
    G, c, x_mean, y_mean = centred_gram(X, y)
    lam, V = np.linalg.eigh(G)
    Vc = V.T @ c
    W = np.array([V @ (Vc / (lam + a)) for a in alphas])
    B = np.array([intercept(x_mean, y_mean, w) for w in W])
    return B, W


def cd_sweep(G, c, w, q, alpha, l1_ratio):
    """One cyclic pass over the coordinates, updating w and q = G w in place.

    w_j <- S(z_j, alpha*rho) / (G_jj + alpha*(1-rho)),  z_j = c_j - q_j + G_jj w_j.
    Returns the largest coordinate change max_j |dw_j|.
    """
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)
    biggest = 0.0
    for j in range(G.shape[0]):
        gjj = G[j, j]
        if gjj == 0.0:                      # constant column: coefficient stays 0
            continue
        z = c[j] - q[j] + gjj * w[j]
        w_new = soft_threshold(z, l1) / (gjj + l2)
        delta = w_new - w[j]
        if delta != 0.0:
            q += G[:, j] * delta
            w[j] = w_new
            biggest = max(biggest, abs(delta))
    return biggest


def enet_cd_gram(G, c, alpha, l1_ratio, w0=None, tol=1e-9, max_sweeps=5000):
    """Coordinate descent on a precomputed Gram matrix.  Returns (w, sweeps)."""
    w = np.zeros(G.shape[0]) if w0 is None else np.array(w0, dtype=np.float64)
    q = G @ w
    sweeps = 0
    while sweeps < max_sweeps:
        sweeps += 1
        if cd_sweep(G, c, w, q, alpha, l1_ratio) < tol:
            break
    return w, sweeps


def enet_cd(X, y, alpha, l1_ratio, w0=None, tol=1e-9, max_sweeps=5000, G=None, c=None):
    """Elastic net by cyclic coordinate descent with covariance updates.

    Returns (b, w, sweeps).  Pass G, c (from centred_gram) to reuse the Gram matrix.
    """
    x_mean, y_mean = X.mean(axis=0), float(y.mean())
    if G is None or c is None:
        G, c, _, _ = centred_gram(X, y)
    w, sweeps = enet_cd_gram(G, c, alpha, l1_ratio, w0, tol, max_sweeps)
    return intercept(x_mean, y_mean, w), w, sweeps


def alpha_max(X, y, l1_ratio):
    """Smallest alpha giving w = 0:  max_j |c_j| / rho  (needs rho > 0)."""
    _, c, _, _ = centred_gram(X, y)
    return float(np.max(np.abs(c)) / l1_ratio)


def alpha_grid(a_max, min_ratio, n):
    """n log-spaced alphas from a_max down to min_ratio * a_max (descending)."""
    return np.logspace(np.log10(a_max), np.log10(a_max * min_ratio), n)


def enet_path(X, y, alphas_desc, l1_ratio, tol=1e-9, max_sweeps=5000):
    """Warm-started path over descending alphas.  Returns (B (len,), W (len, p), sweeps (len,))."""
    G, c, x_mean, y_mean = centred_gram(X, y)
    w = np.zeros(G.shape[0])
    W, B, S = [], [], []
    for a in alphas_desc:
        w, sweeps = enet_cd_gram(G, c, a, l1_ratio, w, tol, max_sweeps)
        W.append(w.copy())
        B.append(intercept(x_mean, y_mean, w))
        S.append(sweeps)
    return np.array(B), np.array(W), np.array(S)


def group_bootstrap_rows(groups, rng):
    """Row indices of a bootstrap over unique groups (same count, with replacement)."""
    unique = np.unique(groups)
    drawn = rng.choice(unique, size=len(unique), replace=True)
    rows_of = {g: np.flatnonzero(groups == g) for g in unique}
    return np.concatenate([rows_of[g] for g in drawn])


def stability_selection(X, y, alpha, l1_ratio, groups, seed, B=50):
    """Share of B group-bootstrap fits in which |w_j| > 1e-10.  Returns freq (p,).

    Resample b uses default_rng(seed + b); X is not re-standardised.
    """
    groups = np.asarray(groups)
    selected = np.zeros(X.shape[1])
    for b in range(B):
        rows = group_bootstrap_rows(groups, np.random.default_rng(seed + b))
        _, w, _ = enet_cd(X[rows], y[rows], alpha, l1_ratio)
        selected += np.abs(w) > 1e-10
    return selected / B
