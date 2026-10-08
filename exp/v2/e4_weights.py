"""E4 (diagnostic): can observation weights inside ordinary (weighted) least squares on the power target recover what a
bike-scale loss adds? Weights come from a first unweighted fit: w_i = (prediction_i + 1)^(g * 2 * (1 - lam)), i.e. the
delta-method factor that turns squared error on the transformed scale into squared error on bikes (g = 1), damped by g.
Output: exp/v2/e4_weights.json
"""
import json

import numpy as np

from exp.v2.lib import base_matrices, fit_predict
from src.common import paired_bootstrap_delta_r2, r2, read_artifact
from exp.v2.lib import SEED, cfg
from src.features import DesignSpec

TARGET = DesignSpec.from_dict(read_artifact("p3")["target_complexity"]["design_spec"])
out = {}
for lam in (0.1, 0.0):
    ref = None
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        v, val_pred, y_val = {}, None, None
        for name, X, Xe, cnt, cnt_e, _, _ in base_matrices(TARGET):
            _, fit_pred = fit_predict(X, Xe, cnt, lam)
            wts = (fit_pred + 1.0) ** (g * 2.0 * (1.0 - lam)) if g > 0 else None
            pe, _ = fit_predict(X, Xe, cnt, lam, weights=wts)
            v[name] = r2(cnt_e, pe)
            if name == "seeded":
                val_pred, y_val = pe, cnt_e
        days = [v[f"day{f}"] for f in range(5)]
        rec = {"seeded": v["seeded"], "days": float(np.mean(days)), "days_folds": days, "chrono": v["chrono"]}
        if ref is None:
            ref, ref_pred = rec, val_pred
            extra = ""
        else:
            d = np.array(days) - np.array(ref["days_folds"])
            dd, lo, hi = paired_bootstrap_delta_r2(y_val, val_pred, ref_pred, SEED, cfg["bootstrap"]["B"])
            rec.update(days_delta=float(d.mean()), days_se=float(d.std(ddof=1) / np.sqrt(5)), seeded_delta=dd, seeded_lo=lo, seeded_hi=hi)
            extra = (f" | vs unweighted: seeded {dd:+.4f} [{lo:+.4f}, {hi:+.4f}] days {d.mean():+.4f} (se {rec['days_se']:.4f}, "
                     f"{int((d > 0).sum())}/5) chrono {rec['chrono'] - ref['chrono']:+.4f}")
        out[f"{lam}|{g}"] = rec
        print(f"lam={lam} g={g:4.2f} seeded={rec['seeded']:.4f} days={rec['days']:.4f} chrono={rec['chrono']:.4f}{extra}")
json.dump(out, open("exp/v2/e4_weights.json", "w"), indent=1)
