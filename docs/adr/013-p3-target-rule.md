# ADR-013 Phase 3 target complexity: simplest level within 0.001 of the best validation R2 (supersedes "plain argmax")
status: accepted        phase: p3 (frozen at F2)        owner: chief
context: the first Phase 3 run (commit "Phase 1-3 modules and first artifacts") used plain argmax of seeded validation R2 and picked C7 (393 weights), which beats C6 by less than 0.0001 and C5 by 0.0005, against a bootstrap interval of about ±0.011 on that score.
options: plain argmax / simplest level within a tolerance of the best (the rule Phase 2 already uses for the degree) / choose on the chronological split
evidence: artifacts/p3.json of that run. Seeded validation R2: C5 0.9317, C6 0.9322, C7 0.9322. Held-out days: 0.9262 / 0.9281 / 0.9271 (fold sd 0.011). Chronological: 0.8929 / 0.8810 / 0.8584. The levels above C5 add nothing measurable on the seeded split and lose up to 0.035 on unseen months (the hour x trend block extrapolates growth per hour).
decision: target = lowest ladder level whose seeded validation R2 is within p3.plateau_tol = 0.001 of the best. It uses the validation set only, is the same parsimony rule as Phase 2, and no longer lets a 0.00004 difference decide a 140-weight increase.
consequences: target becomes C5 (251 weights); Phase 4 regularises that design. The rule change is reported in the Phase 3 Outcome as a correction we made after seeing a tie.
