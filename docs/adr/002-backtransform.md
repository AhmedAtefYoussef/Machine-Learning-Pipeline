# ADR-002 Back-transform: chosen on validation in Phase 1 among none / Duan / least-squares factor
status: accepted        phase: p1 (frozen at F1)        owner: chief
context: exp(mean of log) under-estimates the mean of cnt; the usual fix is Duan's smearing factor.
options: none (s = 1) / Duan s = mean(exp(residual)) / ls s = sum((cnt+1) e^eta) / sum(e^(2 eta)), all fitted on train
evidence: exp/eda0/eda1.json, design wd x hr + cubic weather, trend: seeded validation R2 none 0.9108, ls 0.9108, Duan 0.9040; chronological 0.858 / 0.858 / 0.798. Mean prediction / mean cnt on validation: none 0.978, Duan 1.036. Log residual variance is concentrated in quiet night hours, so one global factor inflates the peaks, where squared error is decided.
decision: Phase 1 computes all three on validation and keeps the best by validation R2 (expected: none). The METHOD is then frozen; later phases recompute s (if any) from their own training residuals.
consequences: predictions run about 2% low on average; this is stated in the report and matters for the bonus discussion. Clip predictions at 0.
outcome (artifacts/p1.json): the least-squares factor won on validation by 0.003 over no factor at Phase 1 complexity, so `ls` is the frozen method; at Phase 2 complexity and above its factor is within 0.002 of 1, i.e. practically no correction.
