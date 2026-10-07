---
name: chain-integrity
description: "Use when producing, consuming or verifying phase artifacts (artifacts/p1.json to p5.json), hash chaining, the P1-to-P2 initial-loss equality, design freezes, or rebuild cascades in the Rush Hour project."
---
# Chain integrity

Each phase writes `artifacts/pN.json` containing `seed`, `config_sha256`, `upstream_sha256` (sha256 of the upstream artifact file bytes; null for P1) plus its phase keys (see header of `tools/chain_check.py`). Phase code loads the upstream artifact from disk, never from memory or retyped numbers.

**Assertions that must run in the notebook and in `make verify-fast`**: hash chain intact; one seed everywhere and equal to the recomputed team seed; P2 `init_loss` == P1 `train_loss_final` (1e-9); P2 starts from P1 weights; P3 target complexity is carried into P4; P4 has exactly L1, L2, ENet entries and a verdict (`useful|redundant|uninformative`) with evidence for every original column; P5 features ⊆ P4 survivors.

**Rebuild cascade**: `make all` rebuilds phase N when anything upstream changes. A change to representation, target treatment, scaling or the split in P0/P1 invalidates all later artifacts, so the chief declares **design freezes**: F1 after the P1 gate (representation, target transform, scaler, cleaning rules), F2 after the P3 gate (target complexity and validators). After a freeze, an improvement that needs an upstream change is only accepted if the analyst shows an expected gain above the noise interval AND the chief accepts the re-run cost; record it as a superseding ADR.

**Never** hand-edit an artifact. If an artifact is wrong, fix the code and rebuild. Artifacts are immutable per config hash; the active chain is the one `make all` produces.
