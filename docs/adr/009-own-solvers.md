# ADR-009 Own solvers everywhere; closed form for the Phase 3 sweeps
status: accepted        phase: p0        owner: chief
context: R7/R8 require own GD for Phases 1-2; R10/R11 say "implement" (AMBIGUITIES A11); Phase 3 needs several hundred fits.
options: sklearn for P4/P5 / own code with sklearn as test oracle; GD for every Phase 3 fit / closed-form normal equations
evidence: GD and the closed form agree to 5e-6 on the weights in exp/eda0; a 900-column ladder level would need more than 50000 GD iterations.
decision: ridge by closed form, lasso and elastic net by cyclic coordinate descent with soft-thresholding, logistic regression by our GD engine; sklearn only inside tests/ and for the single train_test_split. Phase 3 sweeps use the closed form with a 1e-8 ridge for numerical safety and say so.
consequences: tests must show agreement with the oracle (ridge 1e-6, lasso/enet 1e-3, logistic AUC 1e-3).
