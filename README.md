# Rush Hour kit (drop-in accelerator for the master prompt)

Copy these into the project repo root, then start the session with `claude --agent chief` and paste Part B of `Opus55_Sonnet55_Master_Prompt_v2.md` as the first message.

```
CLAUDE.md                      static project memory (hard constraints H1-H13, ownership, commands)
Makefile                       phase chain (rebuilds cascade) + compact verify targets
config.example.yaml            copy to config.yaml, fill team_ids
.claude/agents/*.md            chief, coder, qa-engineer, verifier, analyst, experimenter, code-steward, scribe
.claude/skills/*/SKILL.md      handoff-protocol, leakage-audit, gd-numerics, chain-integrity, notebook-build, report-writing, token-ledger
tools/*.py                     leak_scan, chain_check, req_check, submission_check, number_trace, build_nb, fingerprint (tested on fixtures)
docs/TRACE.csv                 R1-R15 traceability seed;  docs/templates/{SPEC-task,ADR,EXPERIMENT}.md, RECEIPT.json
handoff/TASKS.yaml             task board seed (wave 0)
```
Python deps for the tools: pyyaml, numpy, pandas (stdlib otherwise). Optional: ruff, pytest, nbclient, pandoc.
The tools were exercised on synthetic fixtures; they do not know your real artifacts yet, so the first `make verify-fast` after Phase 1 is their real test. Expected phase modules: `src/common.py, gd.py, poly.py, validation.py, regularization.py, logistic.py, phases/p1..p5.py` (Makefile dependencies).
If your Claude Code version rejects a frontmatter field, remove that field; the docs-listed fields used here are name, description, tools, model, effort, isolation, maxTurns, skills, disallowedTools, color (agents) and name, description, paths, disable-model-invocation (skills).
