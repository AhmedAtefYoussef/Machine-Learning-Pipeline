---
name: token-ledger
description: "Use when planning a wave of work, deciding whether to spawn an agent, setting effort, or reviewing token spend in the Rush Hour project: spawn test, budgets, ledger format, stop rules."
disable-model-invocation: true
---
# Token ledger

**Spawn test** — spawn an agent only if at least one holds: (a) it runs in parallel with other work, (b) it needs isolation (worktree, fresh context), (c) a cheaper model suffices. Otherwise do it inline. A task shorter than the spawn overhead (re-reading CLAUDE.md, skills, spec) is done inline.

**Budget per task** (in `handoff/TASKS.yaml`): max_turns, effort, token_note. Defaults: coder 40 turns/medium; qa 30/high; verifier 25/high; analyst 40/high; experimenter 25/medium; steward 25/low; scribe 25/low. A task that hits its cap returns partial output; the chief decides to resume, split or escalate, never silently raises the cap.

**Guideline split of total effort** (guidance, not measured): design and decisions 15%; implementation 30%; experiments 10% (mostly scripts, few tokens); verification 15%; analysis and improvement 15%; prose 10%; audit and packaging 5%.

**Ledger**: `TOKENS.md` (chief) one row per wave: wave, agents, turns, rough tokens (from session usage), outcome. At the end run the `explain-usage` skill once and paste its one-paragraph conclusion; act on the biggest line item next time.

**Stop rules**: two failed attempts → escalate; chief ≤ 3 automatic continuations on one open item without new information; improvement rounds ≤ 3, stop earlier when no hypothesis has expected gain above the noise interval; never re-run an experiment whose config hash is logged; never re-read a file a digest covers (`docs/DATA_CARD.md`, `STATE.md`).

**Cheap-by-default**: deterministic scripts (`make verify-fast`) before any LLM reading; tail/jq/rg instead of whole-file reads; effort changed per message, not at top level; no new instructions or tools mid-run.
