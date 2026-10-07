# S-<phase>-<n>  <title>      (chief → <agent>; one page maximum; telegraphic style ok, numbers/paths exact)
goal: <one sentence>
inputs: <paths, artifact ids/hashes>            outputs: <paths + JSON keys>
paths_owned: <only these may be written>        do_not_touch: <everything else>
algorithm: <equations / pseudocode; no prose essays>
constraints: H1..H13 relevant ones; forbidden APIs; thread count; determinism
acceptance:
  - <test name>: <numeric tolerance>            (e.g. gradcheck rel err < 1e-6; |w - w_oracle|_inf < 1e-3)
non_goals: <what NOT to add>
escalate_if: ambiguity | test fails twice | oracle mismatch | divergence/NaN | metric outside band
effort_hint: <medium|high>    receipt: handoff/receipts/R-<phase>-<n>.json
