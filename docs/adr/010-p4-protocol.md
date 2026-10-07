# ADR-010 Phase 4: search, fair comparison and how λ is chosen (fixed before the phase runs)
status: accepted        phase: p4        owner: chief
context: R10 leaves the search range, the strategy and the fairness protocol to us; description 1.8 says tuning uses the validation set.
options: choose λ on seeded validation / by day-block CV minimum / by the one-standard-error rule; compare methods on validation only / on three validators
evidence: exp/eda0/eda1.json: seeded and day-block estimates agree within 0.003 for these models, so the choice of selector should matter little; the signal is dense (ladder gains come from blocks of 23-138 columns).
decision: identical standardised design, rows, folds and grid density (40 log-spaced values) for all three. Ridge grid 1e-4..1e3; lasso and elastic net from alpha_max (all coefficients zero) down to 1e-4 x alpha_max; l1_ratio in {0.1, 0.3, 0.5, 0.7, 0.9, 0.95}. λ per method = argmax validation R2 on bikes (ties to the larger penalty). The day-block CV minimum and one-SE picks are reported beside it. Recommended model = among methods within the paired bootstrap noise of the best on validation, the one with the best held-out-day R2.
consequences: the reported validation scores are slightly optimistic (selected on the same rows); held-out-day and chronological scores are shown next to them.
