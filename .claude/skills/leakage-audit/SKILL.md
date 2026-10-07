---
name: leakage-audit
description: "Use when writing, testing or reviewing any code that splits data, fits a transform, selects hyperparameters or features, defines thresholds, or touches test.csv in the Rush Hour project: leak rules, the three validators, and how to run the scanner."
paths: src/**, nb/**, tests/**, exp/**
---
# Leakage audit

**Rules (H2, H11)**: every `fit` (scaler, encoder, expansion, selector, imputer, smearing factor, class threshold) sees training rows only; `test.csv` is only `transform`ed, once, at the end; refit on train+val is allowed only at the final stage with fixed hyperparameters and a written justification.

**Why the seeded split is optimistic**: train is days 1–19 of each month and consecutive hours of one day are strongly correlated, so a random 20% leaves each validation hour surrounded by training hours of the same day. The hidden test is the 20th of every month.

**Three validators** (all fitted on their own training portion): (1) seeded 80/20 `train_test_split(random_state=seed)` — the mandatory reporting split; (2) day-grouped holdout (hold out whole `dteday`s; closest proxy for the hidden test); (3) chronological (earlier dates train, latest months validate; cut in `config.yaml`; try a second cut). The gap (1)→(2) is within-day leakage; (2)→(3) is temporal drift/extrapolation.

**Silent leaks to check**: group quantile thresholds computed on all rows; standardization before splitting; target-derived encodings; selecting λ/degree on the same rows later reported as unbiased (say "slightly optimistic" and show (2)/(3) beside it); `instant` or trend extrapolation; imputing with global means; early stopping on validation without saying so.

**Run**: `python3 -I tools/leak_scan.py src nb --scratch-glob 'nb/p1*.py' --scratch-glob 'nb/p2*.py' --scratch-glob 'src/gd*.py'`. Suppress a finding only with `# leak-ok: <reason>` on oracle code in tests; the reason is part of the audit.
