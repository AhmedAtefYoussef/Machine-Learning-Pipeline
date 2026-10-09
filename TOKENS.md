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
| 3 | coder S-4-03 selection order + final rebuild | sonnet | 44 | 114k | done; Phase 4 runtime 1395 s |
| 4 | verifier final gate | sonnet | 32 | 138k | 0 failures, 3 warns (all addressed) |
| 4 | independent final audit | opus | 39 | 274k | PASS WITH FIXES; fixes 2-7 applied, fix 1 (roster) is the user's |

| 5 | second pass: coder S-8-01 power target + rebuild | sonnet | 71 | 230k | done; exponent 0 reproduces the first pass exactly; Phase 4 runtime 1786 s |
| 5 | second pass: scribe WALKTHROUGH update | sonnet | 34 | 193k | 414 lines |
| 5 | second pass: independent compliance audit | opus | 37 | 276k | PASS WITH FIXES; no rule broken; fixes applied |
| 5 | second pass: verifier gate | sonnet | 26 | 119k | 39 pass, 1 fail (stale Phase 3 message in the notebook; fixed and rebuilt), 2 warns |

## Conclusion (from this ledger; the `explain-usage` skill was not run)
Subagents used about 2.9M tokens in 17 runs. The largest items were not code but reading-heavy Opus passes (final audit 274k, scribe 264k, analyst 195k) and the Phase 4 coder, which ran three times (180k + 27k + 114k) because the Phase 4 design changed twice after first results (ADR-014, ADR-016). Next time: settle the survivor/final-model logic in a cheap closed-form pilot before specifying Phase 4, give Phase 4 a fast mode so a re-run does not cost 20 minutes, and hand the scribe a digest instead of the source tree.

Second pass (2026-10-08): the exploration itself (E1-E5) was done by the chief with closed-form scripts in `exp/v2/` at no subagent cost; the expensive items were again the full rebuild and the fresh-context audit. One avoidable cost: a 30-minute Phase 4 re-run triggered by file times after the auditor re-ran Phases 1-3 to identical bytes; `run.py` now compares content hashes.

Notebook polish (2026-10-08/09, presentation only): coder S-9-01 107 tool uses, 369k tokens (60 view functions, one chart style, review fixes); independent code review 15 tool uses, 176k tokens (no critical issue, 12 fixes applied). No artifact changed.
