# ADR-018 Second exploration round: what we measured and did not adopt; where the ceiling is
status: accepted        phase: p3/p4 (no change)        owner: chief
context: the request was to push the pipeline to its ceiling. Acceptance rule kept from ADR-015: a change must gain on held-out days by more than two paired standard errors and must not lose more than 0.003 on the chronological split.
options: 20 further input-only feature blocks / observation weights / a bike-scale loss / local time levels
evidence: exp/v2/e2_features_0.1.json, e3_ceiling.json, e4_weights.json, e5_classifier.json, on top of the target design with lam 0.1 (0.9413 seeded / 0.9462 held-out days / 0.9378 chronological).
- Weather and hour blocks (hour x season, lagged temperature and humidity, rain in the last six hours, hour x rain, hour x mist, hour x wind, working-day x hour x humidity, working-day x hour x rain, temp x hum, hour x temp^2, feels-like gap, holiday x part of day): every one within +-0.0007 on held-out days. A greedy forward pass accepts none.
- Local time levels gain on shuffled and interleaved splits and are ruinous on later months: year-month levels +0.0030 held-out days, -0.054 chronological; quarterly trend hinges +0.0027, -0.45. Rejected: they describe the months we have, not time.
- Year x hour and year x day type x part of day: +0.0004 held-out days, -0.011 to -0.014 chronological. Rejected.
- Observation weights (weighted least squares approximating a bike-scale loss): +0.0025 on held-out days with paired se 0.0014 (4 of 5 folds), validation +0.0025 with interval [0.0000, 0.0049]. A bike-scale loss with log link gives the same +0.0025. Below our two-standard-error rule, and it would put a loss other than the phase's own into the final model. Not adopted.
- Diagnostic tree model (Poisson loss, raw columns plus weather memory, never in the chain): 0.966 seeded / 0.950 held-out days / 0.902 chronological. Its 0.016 gap between seeded and held-out days shows the within-day leakage a flexible model exploits and ours does not.
- Classifier: a tree model reaches ROC-AUC 0.901 on our label against 0.886 for the logistic model; Phase 5 may only use the Phase 4 survivors, so we leave it.
decision: adopt nothing from this list. After ADR-017 the final model is within about 0.004 of the diagnostic ceiling on held-out days and about 0.03 above it chronologically. Stop condition (b) of the improvement loop fires again; this was the last round.
consequences: the remaining error is mostly day-level (events we do not observe); the report says so.
