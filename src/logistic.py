"""Logistic regression by gradient descent, plus classification metrics (numpy only).

X has shape (n, d) WITH a bias column in column 0 (never penalised); y in {0, 1}.

    p = sigmoid(X w)
    L(w) = -(1/n) sum[ y log p + (1-y) log(1-p) ] + (l2/2) ||w[1:]||^2
    grad = (1/n) X'(p - y) + l2 * [0, w[1:]]
"""
import numpy as np

from src.gd import gradient_descent, lambda_max


def sigmoid(eta):
    """Numerically stable 1/(1+exp(-eta)): never exponentiates a positive number."""
    eta = np.asarray(eta, dtype=np.float64)
    e = np.exp(-np.abs(eta))
    return np.where(eta >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def logloss(X, y, w, l2=0.0):
    """Mean cross-entropy via -log p = logaddexp(0, -eta), -log(1-p) = logaddexp(0, eta)."""
    eta = X @ w
    data_term = np.mean(y * np.logaddexp(0.0, -eta) + (1.0 - y) * np.logaddexp(0.0, eta))
    return float(data_term + 0.5 * l2 * np.sum(w[1:] ** 2))


def logloss_grad(X, y, w, l2=0.0):
    """(1/n) X'(p - y) + l2 * w with the bias entry of the penalty set to 0."""
    grad = X.T @ (sigmoid(X @ w) - y) / X.shape[0]
    grad[1:] += l2 * w[1:]
    return grad


def fit_logistic(X, y, l2=0.0, lr=None, tol_loss=1e-10, tol_grad=1e-6, max_iter=50000):
    """Gradient descent from w = 0.  Default lr = 1/(lambda_max(X'X/n)/4 + l2).

    The Hessian is at most (1/4) X'X/n + l2 I, so this step is 1/L for the smoothness L.
    """
    if lr is None:
        lr = 1.0 / (lambda_max(X) / 4.0 + l2)
    def loss_fn(w):
        return logloss(X, y, w, l2)

    def grad_fn(w):
        return logloss_grad(X, y, w, l2)

    return gradient_descent(loss_fn, grad_fn, np.zeros(X.shape[1]), lr,
                            tol_loss=tol_loss, tol_grad=tol_grad, max_iter=max_iter)


# ----------------------------------------------------------------- metrics

def average_ranks(scores):
    """1-based ranks of scores; tied values share the mean of their ranks."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores))
    start = 0
    while start < len(scores):
        stop = start
        while stop + 1 < len(scores) and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def roc_auc(y, p):
    """Mann-Whitney: AUC = (sum of positive ranks - n1(n1+1)/2) / (n1 n0)."""
    y = np.asarray(y)
    ranks = average_ranks(np.asarray(p, dtype=np.float64))
    n1 = int(np.sum(y == 1))
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pr_auc(y, p):
    """Average precision: sum_k (R_k - R_{k-1}) P_k over distinct descending thresholds."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(-p, kind="mergesort")
    p_sorted, y_sorted = p[order], y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted != 1)
    last_of_tie = np.r_[p_sorted[1:] != p_sorted[:-1], True]   # one threshold per distinct score
    tp, fp = tp[last_of_tie], fp[last_of_tie]
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def confusion(y, p, thr):
    """Counts for the rule predict 1 iff p >= thr."""
    y = np.asarray(y)
    pred = np.asarray(p) >= thr
    return dict(tp=int(np.sum(pred & (y == 1))), fp=int(np.sum(pred & (y != 1))),
                tn=int(np.sum(~pred & (y != 1))), fn=int(np.sum(~pred & (y == 1))))


def prf(y, p, thr):
    """accuracy, precision, recall, f1 at threshold thr (0 when a denominator is 0)."""
    c = confusion(y, p, thr)
    def safe(a, b):
        return a / b if b > 0 else 0.0

    precision = safe(c["tp"], c["tp"] + c["fp"])
    recall = safe(c["tp"], c["tp"] + c["fn"])
    return dict(accuracy=safe(c["tp"] + c["tn"], len(y)), precision=precision,
                recall=recall, f1=safe(2 * precision * recall, precision + recall))


def calibration_table(y, p, bins=10):
    """Equal-width bins on [0, 1] (last bin includes 1.0): n, mean predicted p, fraction positive."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.minimum(np.searchsorted(edges, p, side="right") - 1, bins - 1)
    rows = []
    for b in range(bins):
        in_bin = which == b
        n = int(in_bin.sum())
        rows.append(dict(lo=float(edges[b]), hi=float(edges[b + 1]), n=n,
                         mean_p=float(p[in_bin].mean()) if n else None,
                         frac_pos=float(y[in_bin].mean()) if n else None))
    return rows
