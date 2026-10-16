# v3/modular plan (chief, 2026-10-16; deadline 2026-10-18)

Inputs: A_architecture.md, B_methodology.md, C_dataset.md, D_notebook.md (this folder). Baseline: baseline_fingerprint.txt.

## Ground rule
Every change is either (a) behaviour-preserving, proven by byte-identical artifacts/p1..p6.json, sample_submission.csv and
bonus_control.json after a full rebuild, or (b) prose/docs only. No metric-changing model work: B and C measured every
candidate model change at or below noise (held-out-day se about 0.0035); C's +0.004 harmonic gain breaks the
pre-registered chronological rule (ADR-018) and is disclosed instead. Expectation cells stay byte-identical (H9).
config.yaml is not touched.

## Model hierarchy (who does what)
- Opus 5.5 (reasoning): audits, plan, prose (Justification/Outcome, report, WALKTHROUGH), requirement verification, final review.
  Strengths: judgment under constraints, long-context consistency, catching subtle methodology issues. Cost: high, so
  it receives file paths and a tight brief, never whole transcripts.
- Sonnet 5.5 (code literacy): refactors, tooling, notebook code cells, lint, tests, exp scripts. Strengths: fast,
  accurate on well-specified edits. Weakness: drifts on open design questions, so every brief fixes the interfaces,
  the files it owns and a mechanical acceptance test (fingerprint equality, pytest, verify-fast).
- Parallel only on disjoint files; each coder in its own worktree, commits on its branch; chief merges.

## Workstreams
W1 run.py + tools (Sonnet coder)  — A1 hash-manifest staleness, A2 import-closure deps, A3 nb builds phases and
   bonus_control first; tools/expectation_check.py (H9) wired into verify-fast; fingerprint covers submission,
   bonus_control; tools/ lint clean.
W2 src refactor (Sonnet coder) — A6 one Target.predict, A7 phases import only libraries, A8 one stability_selection,
   A10 fused loss/grad, A11 timed dedupe, run(cfg, write=...) for the notebook's live mode, public p1 helpers,
   src/tests lint clean. Full rebuild; artifacts must be byte-identical.
W4 exp + diagnostics (Sonnet experimenter) — A4 archived-exp notes, exp/v2/lib on common.Target, day-block interval
   and test-coverage facts (C) into artifacts/diagnostics.json.
W3a notebook code (Sonnet coder, after W2) — D outline: nbview -> one table() helper, inline asserts, simple phase(),
   trimmed plots, short commented cells; markdown cells untouched.
W3b prose (Opus scribe, after W3a) — shorter Justification/Outcome per D, B1-B8 and C disclosures, placeholders for
   diagnostics, report template, WALKTHROUGH, README/STATE/ARCHITECTURE/DATA_CARD/TRACE consistency.
V  verification (Opus verifier + code reviewer) — H1-H13, every requirement in Project_1_description.md 1.7-1.11,
   run.py verify strict, fingerprint equality, Colab standalone check.

## Skills and plugins used
Used: project skills chain-integrity, leakage-audit, notebook-build, report-writing, handoff-protocol, gd-numerics;
simplify / code-review (post-merge review); pr-review-toolkit silent-failure-hunter for run.py staleness logic;
humanizer for prose review; pdf-viewer to inspect report.pdf; commit-commands for commits.
Not used (no fit): HR, finance, marketing, sales, design/Figma, Vercel, Supabase, Slack/Notion/Asana connectors,
data-warehouse SQL skills (the data is two CSV files), dashboard/artifact builders (deliverable is ipynb + PDF + CSV).
