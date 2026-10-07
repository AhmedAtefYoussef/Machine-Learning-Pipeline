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
