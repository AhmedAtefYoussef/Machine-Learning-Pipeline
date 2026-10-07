# RISKS (chief; analyst appends)

| # | Risk | Mitigation | Owner |
|---|---|---|---|
| K1 | Leakage: a statistic fitted on validation/test rows | Design.fit takes the training frame only; leak_scan; leakage tests; test file read in one function | coder / qa |
| K2 | Validation optimism: seeded split shares days with train; scores tuned on it | day-block and chronological estimates shown beside every seeded score; bootstrap interval; say "slightly optimistic" | chief |
| K3 | Drift / extrapolation: linear trend in log space is exponential growth | trend never raised to a power; chronological stress test; hr x trend block only as a ladder level | chief |
| K4 | Collinearity-driven verdicts (temp/atemp, season/mnth, yr/trend/instant, weekday/workingday/holiday) | drop-alone AND drop-group, stability selection, verdict rule fixed before running | chief |
| K5 | Back-transform bias of the log model | method chosen on validation in P1, mean ratio reported, frozen | chief |
| K6 | Test-day edge effects: the 20th is one day past each month's training window | day-block folds mimic it; submission sanity check against the training profile | verifier |
| K7 | Chain break after a late upstream change | hash chain, run.py cascade, design freezes F1/F2 | chief |
| K8 | Quiet hours missing from the data (cnt ≥ 1) | stated as a limitation; clip at 0 only | chief |
| K9 | Environment: no make/pandoc, cp1252 console, OneDrive path | run.py, -X utf8, headless Chrome for the PDF, no worktrees | chief |
| K10 | Team IDs taken from session notes | closed: confirmed by the user on 2026-10-07 | user |
| K11 | Live defence: code nobody can explain | short functions, config cell, WALKTHROUGH drills | steward / scribe |
