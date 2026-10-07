---
name: report-writing
description: "Use when writing the Rush Hour report (at most 6 pages), notebook justification/Outcome prose, or WALKTHROUGH.md: structure, voice, number placeholders, PDF build and page-count check."
---
# Report writing

**Voice**: the team's own; plain full sentences; most important point first; explain choices, not theory; refer to the team's actual numbers. No telegraphic style here.

**Numbers**: write placeholders, e.g. `{{p1.val_r2:.3f}}`, in `report/report.template.md`; `python3 -I tools/number_trace.py render report/report.template.md report/report.md` fills them from `artifacts/*.json`; `… check report/report.md` fails on any number that is not in an artifact (whitelist file for page numbers/dates). Never type a metric.

**Structure (≤ 6 pages)**: 1 data findings and what they forced (half page); then one block per phase — *choice → rationale → evidence number → what surprised us* — covering every open choice (lr, stopping rule, representation of hr/season/weathersit/dteday, degree and expanded features, validators and chronological cut, λ and l1_ratio per method, verdict table for every column with its evidence, label rule and threshold, metrics); bonus derivation if done; pipeline retrospective table (phase | input | key hyperparameters | val metrics | lesson); final model and test-prediction sanity. Use compact tables and 3–5 figures; each figure answers one question.

**Build**: `make report` (pandoc) or the `pdf` skill for finer layout; then check page count and a rasterized look with the `pdf-reading` skill. Over 6 pages → cut prose, not evidence.

**WALKTHROUGH.md**: for each phase one paragraph per function ("what it does, which knob to change, what to expect") and ≥ 10 likely oral questions with 2–3 line answers, including "change X and re-run" drills.
