# Rush Hour: predicting hourly bike-sharing demand

A five-phase chained machine-learning pipeline for the *Machine Learning, Winter 2026, Project 1* brief
([Project_1_description.md](Project_1_description.md)). It predicts the hourly rental count `cnt` of a bike-sharing
system and ends with a classifier for "high-demand" hours. Each phase implements one course algorithm and consumes
the numeric artifact of the phase before it.

| Phase | Algorithm | Consumes | Produces |
|---|---|---|---|
| 1 | Gradient descent, written from scratch | the seeded train/validation split | weight vector, learning rate, iteration count |
| 2 | Polynomial regression, fitted with the Phase 1 gradient descent | Phase 1 weights as the starting point | degree and expanded feature list |
| 3 | Bias–variance diagnosis, plus one chronological split | Phase 2 degree and features | diagnosis and target complexity |
| 4 | Ridge, Lasso and Elastic Net (own solvers) | Phase 3 target and diagnosis | λ per method, surviving features, a verdict for every input column |
| 5 | Logistic regression (own gradient descent) | Phase 4 surviving features | classifier, validation metrics, retrospective |

## Deliverables

| File | What it is |
|---|---|
| [rush_hour.ipynb](rush_hour.ipynb) | The executed notebook: all five phases, each opening with an **Expectation** cell and closing with an **Outcome** cell. It is self-contained: its setup cell carries `src/`, `config.yaml` and the stored artifacts, so on Google Colab only the notebook, `train.csv` and `test.csv` need to be uploaded |
| [report.pdf](report.pdf) | The report (5 pages), every number read from the artifact files |
| [sample_submission.csv](sample_submission.csv) | Predictions for `data/test.csv` (574 rows, same order) |
| [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) | Rehearsal guide for the live evaluation: formulas, change-and-re-run drills, likely questions |

## Results

Validation scores on the seeded 20% hold-out (team seed 44615). R² and RMSE are on the bike scale.

| Phase | Model | Validation |
|---|---|---|
| 1 | linear model on 36 weights, gradient descent at half the stability bound | R² 0.736, RMSE 94.7 |
| 2 | degree 3 in `temp` plus working-day × hour interactions | R² 0.918, RMSE 52.8 |
| 3 | diagnosis: under-fit; target design with 282 weights | R² 0.941 (held-out days 0.946, chronological split 0.938) |
| 4 | three penalised methods tied within noise; Lasso on the surviving features | R² 0.941, RMSE 44.7 |
| 5 | "busier than three quarters of comparable hours" label, cut-off 0.25 | accuracy 0.784, F1 0.659, ROC-AUC 0.885 |

These are the numbers of the committed artifacts. The source of truth is `artifacts/p1.json … p6.json`; the notebook
and the report read them from there. Validation scores are slightly optimistic because tuning used the same rows;
the report says where and gives the held-out-day and chronological estimates beside them.

The chain was run twice. The first pass (git tag `v1-submitted`) fitted `log1p(cnt)` and ended at R² 0.939 on
validation, 0.943 on held-out days and 0.914 chronologically. The second pass changes one thing, the target
transform (a power transform with exponent 0.1, decision record 017), which mainly improves the estimate for unseen
months. What else was tried and rejected is in decision record 018 and `exp/v2/`.

## Quick start

Requirements: Python (built and tested on 3.14) with `numpy`, `pandas`, `scikit-learn` (used only for the one seeded
`train_test_split` and as a test oracle), `matplotlib`, `pyyaml`, `threadpoolctl`, `nbformat`, `nbclient`,
`ipykernel`, `markdown-it-py`, `pytest`, and optionally `ruff`. The PDF build needs Chrome or Edge.

```bash
pip install numpy pandas scikit-learn matplotlib pyyaml threadpoolctl nbformat nbclient ipykernel markdown-it-py pytest ruff
```

Everything runs through `run.py` from the repository root (it replaces the kit's `Makefile`, so no `make` is needed):

| Command | What it does | Time |
|---|---|---|
| `python run.py all` | rebuild `artifacts/p1.json … p5.json`; a phase rebuilds when anything upstream changed | about 40 min (Phase 4 is about 30) |
| `python run.py p1` … `p5` | rebuild one phase | seconds, except Phase 4 |
| `python -m src.bonus_control` | the k = 1 control for the asymmetric-cost bonus | seconds |
| `python run.py submission` | refit the recommended model and write `sample_submission.csv` | seconds |
| `python run.py nb` | assemble `nb/p*.py` into `rush_hour.ipynb` and execute it | about 1 min with stored Phase 4/5 artifacts |
| `python run.py report` | render the report template and build the PDF | seconds |
| `python run.py verify-fast` | lint, tests, leak scan, chain check, traceability counts | seconds |
| `python run.py verify` | all of the above plus notebook run, submission check, report number trace | about 2 min |

The notebook imports the modules in `src/`. Inside the repository it uses the files that are there; anywhere else
(for example Google Colab) its setup cell unpacks its own copy of `src/`, `config.yaml` and `artifacts/` and asks for
`train.csv` and `test.csv` if they are not next to it. Every phase cell loads the stored artifact when the configuration
and the upstream artifact are unchanged; set `RECOMPUTE_ALL = True` in the configuration cell to recompute the chain.
`python -X utf8 tools/standalone_check.py` re-runs the notebook in an empty folder with only the two data files.

### Team seed

The seed is derived from the team's student IDs in [config.yaml](config.yaml), never typed:
`int(sha256("_".join(sorted(ids))).hexdigest(), 16) % 100000`. Changing `team_ids` changes the split and therefore
every number; rebuild with `python run.py all`, `python -m src.bonus_control`, `python run.py submission`,
`python run.py nb` and `python run.py report`.

## How the chain is enforced

- Every artifact stores the SHA-256 of the artifact it consumed (`upstream_sha256`) and of the configuration.
- Phase 2 starts from the Phase 1 weights, so its first loss must equal Phase 1's last loss; this is asserted.
- Phase 3 is anchored at the Phase 2 design; Phase 4 builds on the Phase 3 target level; Phase 5 uses only the
  Phase 4 survivors. `tools/chain_check.py` and assertions in the notebook check all of it.
- Scalers, imputation values, expansions and class thresholds are fitted on the training rows only.
  `data/test.csv` is read in one function and used only for the final predictions.

## Repository layout

```
config.yaml            team IDs and every re-run knob
run.py                 task runner (all, p1..p5, submission, nb, report, verify-fast, verify)
data/                  train.csv, test.csv, sample_submission.csv (as provided)
src/                   the implementation
  common.py            seed, split, metrics, artifact I/O with hash chaining
  features.py          the feature design (fit on train, transform elsewhere)
  gd.py, gd_asym.py    gradient descent, and the asymmetric-cost bonus
  poly.py              lifting Phase 1 weights into the expanded design
  validation.py        the three validators, learning curves
  regularization.py    Ridge, coordinate-descent Lasso and Elastic Net, stability selection
  verdicts.py          the per-column verdict rule
  logistic.py, labels.py   logistic regression, metrics, the high-demand label
  phases/p1.py … p5.py one module per phase: load upstream artifact, compute, write artifact
  predict.py           the only module that touches the hidden test rows
nb/                    notebook cells as plain percent-format files, one per phase
artifacts/             p1.json … p6.json, the numeric hand-overs between phases
report/                report template with {{artifact.key}} placeholders, figure and PDF builders
tests/                 256 tests: gradient checks, oracle comparisons, leakage and determinism tests
tools/                 checkers: leak_scan, chain_check, submission_check, number_trace, build_nb, …
docs/                  SPEC, ARCHITECTURE, DATA_CARD, decision records (adr/), TRACE.csv, WALKTHROUGH
reports/               verification gates, the analyst's audit, the independent final audit
exp/                   exploratory and diagnostic scripts; nothing here enters the pipeline
handoff/               task specs and receipts from the build process
```

## Documentation map

| If you want to know… | Read |
|---|---|
| what is required and how each requirement is checked | [docs/SPEC.md](docs/SPEC.md), [docs/TRACE.csv](docs/TRACE.csv) |
| why a choice was made, with the numbers behind it | [docs/adr/](docs/adr/) (18 short decision records) |
| how unclear points in the brief were read | [handoff/AMBIGUITIES.md](handoff/AMBIGUITIES.md) |
| what the data looks like | [docs/DATA_CARD.md](docs/DATA_CARD.md) |
| module interfaces and determinism rules | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| what was verified and what an examiner might attack | [reports/verify/](reports/verify/), [reports/analysis/audit.md](reports/analysis/audit.md) |
| current status and open items | [STATE.md](STATE.md) |

## How this was built

The pipeline was produced with Claude Code under the orchestration prompt in
[Opus55_Sonnet55_Master_Prompt_v2.md](Opus55_Sonnet55_Master_Prompt_v2.md): a lead agent made the design decisions
and wrote the prose, and sub-agents wrote code, tests and audits from the specs in `handoff/specs/`. The git history,
`handoff/` and [TOKENS.md](TOKENS.md) record that process. Rules and designs were changed after first results
(decision records 013 to 017); the notebook and the report disclose each of them.

## Known limits

- The trend is a straight line on the transformed scale: sound inside 2011–2012, somewhat too steep beyond it.
- The data contains no zero-demand hours, so the model has never seen one.
- Phase 4 takes about 30 minutes, so changing an early setting is slow to propagate; see the drills in
  [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) for quicker demonstrations.
- About 35 lint warnings remain, mostly in the kit's `tools/`.
