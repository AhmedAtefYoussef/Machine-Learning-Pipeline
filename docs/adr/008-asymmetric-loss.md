# ADR-008 Bonus: asymmetric operator cost = weighted squared error on the bike scale
status: accepted        phase: p1 (bonus R12)        owner: chief
context: under-prediction costs 3 x over-prediction "by the same amount" (AMBIGUITIES A9); our model is linear in log1p(cnt).
options: weighted squared error / weighted absolute (pinball) error; loss on log scale / on bikes
evidence: weighted squared: minimiser is the 0.75 expectile, differentiable everywhere, direct modification of MSE. Pinball: minimiser is the 0.75 quantile, only a subgradient at 0. The cost is paid in bikes, not log-bikes.
decision: L(w) = (1/n) sum c_i (yhat_i - y_i)^2, yhat_i = exp(x_i.w) - 1, c_i = 3 if yhat_i < y_i else 1; gradient (2/n) X^T (c * r * exp(Xw)); GD with backtracking step, started from the Phase 1 MSE weights (the loss is no longer quadratic, so the fixed-step bound of ADR-007 does not apply). The pinball cost is reported for both models as a sensitivity.
consequences: separate module src/gd_asym.py; it never feeds Phase 2 (the chain carries the MSE weights).
