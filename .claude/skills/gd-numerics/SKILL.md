---
name: gd-numerics
description: "Use when implementing, testing or debugging gradient descent, polynomial expansion with weight lifting, ridge/lasso/elastic-net solvers, asymmetric loss, or logistic regression in the Rush Hour project: formulas, learning-rate bounds, stopping rules, oracle and gradient checks."
paths: src/**, nb/p1*, nb/p2*, nb/p4*, nb/p5*, tests/**
---
# GD numerics (formulas are exact; tolerances are gates)

**MSE**: X̃ = [1, X] (standardized, fit on train). L(w) = (1/2n)‖X̃w − y‖² (or without ½: be consistent, document). ∇L = (1/n) X̃ᵀ(X̃w − y). Stable step: lr < 2/λ_max(X̃ᵀX̃/n); use lr = 0.5–0.9 × that bound (compute λ_max with `np.linalg.eigvalsh`). Stop when relative loss change < tol AND ‖∇‖ < tol, with an iteration cap; log `stop_reason`, `iterations`, loss curve. Oracle: `np.linalg.lstsq` → require max|w − w_oracle| < 1e-3 on standardized weights after convergence and val R² within 1e-4. Show one larger lr diverging and one smaller lr stalling.

**Gradient check** (every loss): central differences, ε = 1e-6, relative error < 1e-6 on a random subset of ≥ 10 coordinates.

**P1→P2 lift**: base columns keep the P1 scaler; new columns (powers, interactions) get their own scaler fit on train; initial weights = P1 weights in their slots, 0 elsewhere. Then initial P2 loss == final P1 loss to 1e-9 (assert; recorded as `init_loss`). Do not expand `trend` beyond degree 2 (extrapolation).

**Asymmetric loss (bonus, k = 3 under-prediction cost)**, r = ŷ − y, w_i = k if r_i < 0 else 1:
- weighted squared: L = (1/n)Σ w_i r_i², ∇ = (2/n)Σ w_i r_i x̃_i (continuous at r = 0); minimizer is the τ = k/(k+1) = 0.75 expectile.
- linear (pinball): L = (1/n)Σ (k·max(−r,0) + max(r,0)), subgradient −k x̃_i if r<0, +x̃_i if r>0; minimizer is the 0.75 quantile.
"3× as much" is ambiguous between the two; choose one, record it in AMBIGUITIES, show the other as a sensitivity. Implement on the raw-cnt scale even if the base model uses log1p.

**Ridge** closed form w = (X̃ᵀX̃/n + αI′)⁻¹ X̃ᵀy/n with I′ not penalizing the intercept (objective (1/2n)‖y−X̃w‖² + (α/2)‖w‖²; sklearn `Ridge(alpha_s)` equals α = alpha_s/n — verified). **Lasso/ENet** coordinate descent for (1/2n)‖y − Xw‖² + α·ρ‖w‖₁ + (α(1−ρ)/2)‖w‖² (sklearn's scaling): with standardized columns, w_j ← S(z_j, αρ)/(1 + α(1−ρ)), z_j = (1/n)X_jᵀ(partial residual), S(z,t) = sign(z)·max(|z|−t, 0). Warm-start along a descending log grid. Oracle: sklearn within 1e-3 (tests only).

**Logistic**: p = σ(X̃w), L = −(1/n)Σ[y log p + (1−y) log(1−p)] (+ (λ/2)‖w‖² without intercept), ∇ = (1/n)X̃ᵀ(p − y) (+ λw). Use a stable sigmoid and log-loss (clip or `log1p(exp(·))`). Oracle: sklearn LogisticRegression with matching penalty, AUC difference < 1e-3.

**Target**: if log1p is used, report R² on the raw scale, apply and test a smearing correction, clip predictions ≥ 0.
