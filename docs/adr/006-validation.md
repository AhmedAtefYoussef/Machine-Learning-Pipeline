# ADR-006 Validators, chronological cut, and where tuning happens
status: accepted        phase: p0 (frozen at F2)        owner: chief
context: R4/R5/R9; the seeded split mixes hours of the same day (100% of validation rows have same-day rows in train); the hidden test is whole unseen days interleaved in time.
options: seeded only / + day-block folds inside train / + chronological; cut at 2012-07-01, 2012-09-01 or 2012-10-01
evidence: exp/eda0: the seeded split gives 8708 / 2178 rows; rows from 2012-07-01 on are 25% of train.csv and cover half a seasonal cycle.
decision: three estimates, one seeded call. (1) seeded 80/20: every reported score and every hyperparameter choice (description 1.8). (2) day-block: 5 folds by date rank mod 5 inside the training portion, no seed, whole days held out: the closest proxy for the hidden test; used to confirm choices, never reported as "validation". (3) one chronological split at 2012-07-01: stress test for drift. Uncertainty: 1000-resample bootstrap of the validation rows.
consequences: scores tuned on (1) are slightly optimistic; (2) and (3) are always shown next to them. No second chronological cut (AMBIGUITIES A4).
