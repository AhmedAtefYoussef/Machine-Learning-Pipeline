# ADR-001 Target = log1p(cnt), fixed from Phase 1 to Phase 5
status: accepted        phase: p0 (frozen at F1)        owner: chief
context: cnt is right-skewed (skew 1.24) with variance growing with the mean; R2 is scored on bikes (R13). The chain breaks if the target changes between phases (R2).
options: raw cnt / log1p(cnt) with back-transform
evidence: exp/eda0/eda.json variants. Additive model: raw 0.686 vs log 0.655 seeded validation R2 (bike scale). With the workingday x hour block and cubic weather terms: raw 0.839 vs log 0.899 seeded, 0.848 vs 0.901 day-block. Day-block fold sd about 0.012.
decision: log1p. The log model loses 0.03 at Phase 1 but wins 0.06 from Phase 2 on, far outside the fold spread; effects in this data are multiplicative (rush hour scales with year and weather).
consequences: every R2/RMSE is computed on bikes after back-transform (ADR-002); the Phase 1 number is knowingly not the best possible linear model; the asymmetric-cost bonus is defined on bikes (ADR-008).
