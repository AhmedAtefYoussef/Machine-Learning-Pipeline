# EXP-<nnn>  (append one block per experiment to docs/EXPERIMENTS.md; never rerun an experiment whose config hash already appears)
hypothesis: <what should improve and why>      category: correctness | validation | modeling | numerics | robustness | documentation
config_sha: <hash of exp config>               compliance: passes H1..H13? <yes/no + which>
expected_gain: <delta metric on day-held-out AND chronological, with noise interval>   cost: <tokens/minutes>   risk: <low/med/high>
result: <numbers>   decision: adopt | reject | defer     adr: <ADR id if adopted>
