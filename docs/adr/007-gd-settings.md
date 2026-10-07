# ADR-007 Learning rate and stopping rule for the from-scratch GD
status: accepted        phase: p1        owner: chief
context: R7: configurable learning rate, convergence must be demonstrated.
options: hand-picked lr / lr tied to the stability bound 2/lambda_max of the Gram matrix X^T X / n; stop on iterations / on loss change / on loss change AND gradient norm
evidence: for the quadratic MSE loss GD is stable iff lr < 2/lambda_max. exp/eda0/eda.json: base design lambda_max 1.974 (bound 1.013), lr = 1/lambda_max converges in 475 iterations to within 5e-6 of the closed-form weights; expanded designs have condition numbers 380-570 and need 3000-4600 iterations.
decision: lr = 0.5 x bound = 1/lambda_max, computed from the design, never typed: half the bound keeps every eigen-direction contracting without sign flips, so the loss falls monotonically. Stop when the relative loss change <= 1e-10 AND the gradient norm < 1e-6, cap 50000; start from zeros (Phase 1) so the weights are exactly reproducible. A sweep over 0.001, 0.01, 0.1, 0.5, 0.9, 1.05 x bound shows stall / converge / diverge.
consequences: Phase 2 reuses the rule on its own lambda_max; the closed-form solution is used only as a test oracle.
