# EXPERIMENTS (append-only). Diagnostic work lives in exp/ and never enters the chain. Looks at the seeded validation set are counted.

# EXP-001 Phase-0 design options (exp/eda0/eda.py → eda.json)
hypothesis: target, hour encoding, cleaning flags and polynomial degree matter by more than the fold spread      category: modeling
compliance: yes (train-only fits; closed-form oracle; test file not read for modelling)      looks at seeded validation: 1
result: log1p beats raw from the workingday x hour block on (0.899 vs 0.839); one-hot hour 0.655 vs 6 harmonics 0.651 vs numeric < 0.4; hum fix and wind flag within ±0.001; degree plateau after 2-3.
decision: adopt ADR-001, 003, 004, 007.

# EXP-002 Time representation, back-transform, ladder preview (exp/eda0/eda1.py → eda1.json)
hypothesis: trend vs yr and the smearing factor change the day-block and chronological estimates; a block ladder shows a real over-fit region      category: validation
compliance: yes      looks at seeded validation: 1
result: trend + 2 harmonics 0.911 / 0.911 / 0.858 (seeded / day-block / chrono) vs yr 0.905 / 0.904 / 0.876; Duan −0.007 seeded and −0.06 chrono; ladder peaks near 400 columns (day-block 0.933) and falls to 0.929 at 900 while train keeps rising; hr x trend hurts chrono by 0.04.
decision: adopt ADR-002, 005, 006; ladder order fixed in config.yaml before Phase 3 runs.
