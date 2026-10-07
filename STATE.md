# STATE (chief; resume from here, not from a transcript)

seed 44615 (IDs 16007032, 16009837, 16006283; user to confirm roster, AMBIGUITIES A2) · deadline 18 Oct 2026
env: Windows, `python` 3.14, no make/pandoc → `python run.py <target>`; agents = general-purpose subagents (model sonnet) reading `.claude/agents/<role>.md`; shared checkout, disjoint paths, chief commits.

## Phase status
| phase | status | evidence |
|---|---|---|
| −1 bootstrap | done | kit adopted at repo root, commits 4c53765, 5910381 |
| 0 decisions | done | docs/adr/001–009, docs/DATA_CARD.md, exp/eda0 |
| wave 0 libs + tests | running | specs S-0-01..04; receipts R-0-02, R-0-03 in |
| 1 GD | Expectation written; spec S-1-01 ready | nb/p1_gd.py |
| 2 Polynomial | Expectation written; spec S-1-01 ready | nb/p2_poly.py |
| 3 Bias-variance | design fixed (config p3 ladder); spec todo | |
| 4 Regularization | design in notes below; spec todo | |
| 5 Logistic | spec todo | |
| 6 Submission, report | todo | |

## Design notes not yet in a spec
- P3: ladder C0..C9 (config), degree axis, learning curves, three validators, noise floor; target = ladder level with max seeded validation R², confirmed on day-block; closed-form fits (ADR-009).
- P4: design = target level + candidate blocks; select α per method on seeded validation R² (bike scale); report day-block CV-min and 1-SE picks beside it; stability selection B=50 over days; verdict rule: useful = drop-alone ΔR² CI above 0 and ≥ 0.001; redundant = not useful alone but group drop hurts or solo R² ≥ 0.01; else uninformative; in a group where no single drop hurts, the member with the highest solo R² is the kept "useful" representative. Callers must check enet_cd sweeps < max_sweeps (R-0-03 caveat).
- P5: label = cnt above the train 0.75 quantile of its (yr, workingday, hr) cell; foils: global threshold, (workingday, hr) without year; threshold on probability from the 3:1 cost ratio (0.25) vs 0.5 vs F1-optimal.
- P6: refit weights on train+val with fixed hyperparameters, transforms stay train-fitted (A13).

## Next action
Collect R-0-01, R-0-04 → run `python run.py verify-fast` → commit → spawn S-1-01 coder (P1+P2) and, in parallel, the P3 validation-module coder.

## Definition of done
- [ ] D1 seed/split/chrono · [ ] D2 no leak · [ ] D3 P1–P2 own GD verified · [ ] D4 chain assertions · [ ] D5 P3 evidence · [ ] D6 P4 three methods + verdicts · [ ] D7 P5 label/metrics/retrospective · [ ] D8 Expectation/Outcome/justifications · [ ] D9 notebook headless · [ ] D10 report ≤ 6 pages traced · [ ] D11 submission valid · [ ] D12 bonus · [ ] D13 WALKTHROUGH · [x] D14 spec kit (kit-check after wave 0) · [ ] D15 TRACE verified · [ ] D16 ceiling report · [ ] D17 H1–H13 verifier pass · [ ] D18 TOKENS · [ ] D19 final audit
