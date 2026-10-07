---
name: notebook-build
description: "Use when authoring or assembling the Rush Hour notebook: percent-format phase files in nb/, Expectation and Outcome cells, config cell, headless execution, output limits."
---
# Notebook build

Author phases as plain files `nb/p0_setup.py`, `nb/p1_gd.py` … `nb/p6_submission.py` in percent format (`# %%` code, `# %% [markdown]` markdown, markdown lines prefixed with `# `). `python3 -I tools/build_nb.py --execute --out rush_hour.ipynb` assembles and runs them top-to-bottom (nbclient if installed, else a built-in sequential executor).

**Per phase file**: first cell = `## Expectation` (chief; committed to git BEFORE the phase runs; never edited afterwards); then method cells (code by coder; each open choice gets a short `**Justification**` markdown cell in the team's voice); last cell = `## Outcome` (what happened, what surprised us). `build_nb.py` fails if the first/last markdown cells are not Expectation/Outcome.

**Rules**: config cell at the top of `p0_setup.py` (ids, seed recompute and self-test, paths, chronological cut, lr, degree, λ grid) so any member can change and re-run; phase cells call `src/` functions and load/write `artifacts/` (no logic duplicated in the notebook); outputs ≤ 30 lines per cell; one plot per question with labelled axes and legend; fixed seeds and thread counts; no absolute paths; runtime budget stated at the top; cells that take > 60 s load a cached result and say so.

**Never** hand-edit the `.ipynb`; edit the `.py` cells and rebuild.
