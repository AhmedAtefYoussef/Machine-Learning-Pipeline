# ADR-005 dteday -> linear trend + two day-of-year harmonics
status: accepted        phase: p0 (frozen at F1)        owner: chief
context: dteday cannot be used as is; 2012 demand is 1.65 x 2011; the hidden test days (the 20th of each month) lie inside the observed date range.
options: trend / yr step / both; 1, 2 or 3 day-of-year harmonics
evidence: exp/eda0/eda1.json (wd x hr + cubic weather): trend+2 harmonics seeded 0.911, day-block 0.911, chrono 0.858; yr+2 harmonics 0.905 / 0.904 / 0.876; both 0.911 / 0.911 / 0.859; a third harmonic changes nothing (0.9105 / 0.9101). One harmonic costs 0.004-0.005 on both random validators and 0.02 on chrono.
decision: linear trend (never raised to a power) + harmonics k = 1, 2. Trend wins by 0.006 on the validators that resemble the hidden test (interleaved days); yr is slightly safer for far-future extrapolation and is kept as a Phase 4 candidate.
consequences: a linear trend in log space is exponential growth: fine inside 2011-2012, not to be trusted beyond; the chronological split in Phase 3 measures exactly that risk.
