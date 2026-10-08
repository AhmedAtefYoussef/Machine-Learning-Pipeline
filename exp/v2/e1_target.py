"""E1 (diagnostic): which target transform should the linear model be fitted on, when the score is R2 on bikes?

Power family on cnt + 1 (lam = 0 is the log the pipeline uses now; lam = 0.5 is the square root; lam = 1 is raw counts),
on the current Phase 3 target design and on the Phase 1 base design, ordinary ridge least squares on the transformed
target. Output: exp/v2/e1_target.json
"""
import json

from exp.v2.lib import delta, evaluate, row
from src.common import read_artifact
from src.features import BASE, DesignSpec

p3 = read_artifact("p3")
TARGET = DesignSpec.from_dict(p3["target_complexity"]["design_spec"])
P1 = DesignSpec(base=BASE)
P2 = DesignSpec.from_dict(read_artifact("p2")["design_spec"])
out = {}
for label, spec in (("target", TARGET), ("p2", P2), ("p1", P1)):
    ref = None
    for lam in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0):
        r = evaluate(spec, lam)
        ref = ref or r
        out[f"{label}|{lam}"] = {k: v for k, v in r.items() if k != "val_pred"}
        print(row(f"{label} lam={lam}", r), "|", delta(r, ref))
json.dump(out, open("exp/v2/e1_target.json", "w"), indent=1)
