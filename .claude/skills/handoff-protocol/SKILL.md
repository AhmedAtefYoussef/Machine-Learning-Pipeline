---
name: handoff-protocol
description: "Use when delegating to or receiving work from another agent in the Rush Hour project: SPEC and RECEIPT formats, file ownership, message style, escalation triggers, and retry caps."
---
# Handoff protocol

**Channel**: files, not pasted text. Specs in `handoff/specs/S-<phase>-<n>.md` (template `docs/templates/SPEC-task.md`, ≤ 1 page). Receipts in `handoff/receipts/R-<phase>-<n>.json` (template `docs/templates/RECEIPT.json`). Each agent appends to its own `handoff/ledger/<agent>.jsonl` (one JSON line per event: task id, status, turns, note); only the chief edits `STATE.md` and `handoff/TASKS.yaml`.

**Ownership**: write only inside `paths_owned` from the task. Shared logs are append-only per agent. Never edit another agent's file; request it in the receipt `questions`.

**Style**: handoff text may be telegraphic (drop articles/filler, arrows, abbreviations). Equations, identifiers, paths, tolerances, error text stay exact. Use full sentences when a misread could change a number, delete data or overwrite an artifact. Notebook prose, report and docstrings are never telegraphic.

**Escalate to the chief when**: a SPEC step is ambiguous; a test fails twice; an oracle mismatches; loss diverges or is NaN; a metric is outside the expected band by > 0.03 R²; a dependency is needed; anything would touch `test.csv` outside the final prediction step. Do not keep retrying.

**Reply discipline**: ≤ 5 lines in the reply, details in the receipt. No code in receipts. Report what was run and its result; if a check could not run, say which and why.
