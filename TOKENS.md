# TOKENS (chief; one row per agent run; subagent tokens and tool uses from task notifications)

| wave | agent / task | model | tool uses | subagent tokens | outcome |
|---|---|---|---|---|---|
| 0 | coder-A common, features, run.py (S-0-01) | sonnet | 26 | 104k | done |
| 0 | coder-B gd, gd_asym, poly (S-0-02) | sonnet | 13 | 74k | done |
| 0 | coder-C regularization, logistic (S-0-03) | sonnet | 17 | 79k | done, oracle ≤ 5e-8 |
| 0 | qa-engineer tests (S-0-04) | sonnet | 27 | 113k | 176 tests pass |
| 1 | coder P1+P2+notebook (S-1-01) | sonnet | 27 | 139k | done; 1 spec point corrected by chief (gradient check at optimum) |
| 1 | coder validation+P3 (S-3-01) | sonnet | 40 | 127k | done |
| 2 | verifier gate p1–p3 | sonnet | 19 | 119k | 0 failures, 7 warns |
| 2 | coder P4 (S-4-01) | sonnet | 42 | 180k | partial: runtime 807 s, 6 stalled lasso points (accepted, disclosed) |
| 2 | coder P4 stage B (S-4-02, resumed) | sonnet | 19 | 207k (cumulative) | done |
| 2 | coder P5 + predict (S-5-01) | sonnet | 51 | 181k | files done; final runs cut off by the session limit, finished by the chief |
| 2 | analyst audit + ceiling (EXP-A1..A3) | opus | 25 | 195k | no leak; ceiling gap 0.018; one measured improvement adopted (ADR-015) |
| 3 | qa-engineer tests for weather memory, validation, labels | sonnet | 16 | 129k | 55 tests, 1 edge-case defect |
| 3 | scribe WALKTHROUGH | opus | 24 | 264k | 345 lines; 6 inconsistencies reported |
| 3 | coder S-4-03 selection order + final rebuild | sonnet | pending | pending | running |
