"""EXP-A3 cheapest tests of the improvement hypotheses: one extra block on the C5 design, closed-form fit,
three validators, paired differences against C5. Diagnostic only (H13); nothing here enters the chain.

Usage: python -m exp.analyst.hyp            (single blocks -> hyp.json)
       python -m exp.analyst.hyp a+b c+d    (named combinations -> hyp_combo.json)
"""
from __future__ import annotations

import sys

import numpy as np

from exp.analyst.lib import hr_onehot, paired_boot, save, scores_only, setup, three
from src.features import DesignSpec

DAYPARTS = {"night": (0, 1, 2, 3, 4, 5), "am": (6, 7, 8, 9), "mid": (10, 11, 12, 13, 14, 15),
            "pm": (16, 17, 18, 19), "eve": (20, 21, 22, 23)}


def dayparts(df) -> np.ndarray:
    hr = df["hr"].to_numpy()
    return np.stack([np.isin(hr, hours).astype(float) for hours in DAYPARTS.values()], axis=1)


def doy(df) -> np.ndarray:
    d = df["dteday"].dt.dayofyear.to_numpy(float)
    return np.stack([np.sin(2 * np.pi * d / 365.25), np.cos(2 * np.pi * d / 365.25),
                     np.sin(4 * np.pi * d / 365.25), np.cos(4 * np.pi * d / 365.25)], axis=1)


def col(df, name) -> np.ndarray:
    return df[name].to_numpy(float)[:, None]


def ws_dummies(values) -> np.ndarray:
    return np.stack([(values == 2).astype(float), (values == 3).astype(float)], axis=1)


def products(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.hstack([a[:, [i]] * b for i in range(a.shape[1])])


EXTRA = {
    # holiday gets its own hour profile (holidays are mostly Mondays: weekday x hour gives them a working Monday shape)
    "hol_x_hr": lambda d: hr_onehot(d)[:, 1:] * col(d, "holiday"),
    # hour profile allowed to differ between the two years (no extrapolating slope, unlike hr x trend)
    "yr_x_hr": lambda d: hr_onehot(d) * col(d, "yr"),
    # weather of the previous hours (wet roads, a shower that just ended); inputs only
    "lag_wx": lambda d: np.hstack([ws_dummies(d["ws_lag1"].to_numpy()), col(d, "hum_lag1"),
                                   ws_dummies(d["wet3"].to_numpy())]),
    # weather of the whole day (people decide in the morning); inputs only
    "day_wx": lambda d: np.hstack([col(d, "day_temp"), col(d, "day_hum"), col(d, "day_wet"), col(d, "day_rain")]),
    # smooth weather interactions and powers Phase 2 did not take
    "wx_smooth": lambda d: np.hstack([col(d, "temp") * col(d, "hum"), col(d, "temp") * ws_dummies(d["ws_c"].to_numpy()),
                                      col(d, "hum") ** 2, col(d, "hum") ** 3, col(d, "windspeed") ** 2,
                                      col(d, "hum") * ws_dummies(d["ws_c"].to_numpy()),
                                      col(d, "temp") * col(d, "windspeed")]),
    # season and growth allowed to differ by day type (leisure riding is more seasonal)
    "wd_x_time": lambda d: col(d, "workingday") * np.hstack([doy(d), col(d, "days"), col(d, "temp")]),
    # coarse daylight effect: 5 day parts x 4 day-of-year harmonics (20 columns instead of C7's 69)
    "part_x_doy": lambda d: products(dayparts(d), doy(d)),
    # rain by day part and day type (rain costs most at leisure hours)
    "rain_x_part": lambda d: np.hstack([products(dayparts(d), ws_dummies(d["ws_c"].to_numpy())),
                                        products(dayparts(d), ws_dummies(d["ws_c"].to_numpy())) * col(d, "workingday")]),
    # concave growth next to the linear trend
    "sqrt_trend": lambda d: np.sqrt(col(d, "days")),
    "yr_step": lambda d: col(d, "yr"),
}
# Second round (combination runs only): the parts of wx_smooth, and a free hour profile for holidays.
EXTRA_PARTS = {
    "hum_pow": lambda d: np.hstack([col(d, "hum") ** 2, col(d, "hum") ** 3]),
    "temp_x_hum": lambda d: col(d, "temp") * col(d, "hum"),
    "wx_x_ws": lambda d: np.hstack([col(d, "temp") * ws_dummies(d["ws_c"].to_numpy()),
                                    col(d, "hum") * ws_dummies(d["ws_c"].to_numpy())]),
    "wind": lambda d: np.hstack([col(d, "windspeed") ** 2, col(d, "temp") * col(d, "windspeed")]),
    "hol_free": lambda d: np.hstack([hr_onehot(d) * col(d, "holiday") * (d["weekday"].to_numpy() == 1)[:, None],
                                     dayparts(d) * col(d, "holiday") * (d["weekday"].to_numpy() != 1)[:, None]]),
    "lag_ws_only": lambda d: np.hstack([ws_dummies(d["ws_lag1"].to_numpy()), ws_dummies(d["wet3"].to_numpy())]),
}
SPEC_BLOCKS = {"wd_x_wx": ("wd_x_wx",), "wd_x_hr_x_temp": ("wd_x_hr_x_temp",), "hr_x_ws": ("hr_x_ws",)}
CONFIG = {"exp": "A3_hypotheses", "base": "p3 target (C5)", "alpha": 1e-8, "dayparts": {k: list(v) for k, v in DAYPARTS.items()},
          "extra": sorted(EXTRA), "spec_blocks": sorted(SPEC_BLOCKS),
          "variants": ["drop_holiday_windspeed", "yr_instead_of_trend", "poisson_irls_5"]}
LATE_VARIANTS = ("wls_mu_1", "wls_mu_half")  # combination runs only (keeps the first run's config hash stable)


def with_blocks(spec: DesignSpec, blocks) -> DesignSpec:
    return DesignSpec(spec.base, spec.power_cols, spec.degree, spec.blocks + tuple(blocks))


def without(spec: DesignSpec, names) -> DesignSpec:
    return DesignSpec(tuple(c for c in spec.base if c not in names), spec.power_cols, spec.degree, spec.blocks)


def run_one(name: str, args, spec) -> dict:
    """Three validators for one named variant; '+' joins several."""
    parts = name.split("+")
    known = set(EXTRA) | set(EXTRA_PARTS) | set(SPEC_BLOCKS) | set(CONFIG["variants"]) | set(LATE_VARIANTS)
    assert set(parts) <= known, f"unknown part in {name!r}"
    extras = [EXTRA[p] for p in parts if p in EXTRA] + [EXTRA_PARTS[p] for p in parts if p in EXTRA_PARTS]
    blocks = [b for p in parts if p in SPEC_BLOCKS for b in SPEC_BLOCKS[p]]
    use = with_blocks(spec, blocks) if blocks else spec
    poisson = 0
    if "drop_holiday_windspeed" in parts:
        use = without(use, ("holiday", "windspeed"))
    if "yr_instead_of_trend" in parts:
        use = without(use, ("trend",))
        extras.append(EXTRA["yr_step"])
    if "poisson_irls_5" in parts:
        poisson = 5
    extra_fn = (lambda d: np.hstack([fn(d) for fn in extras])) if extras else None
    wls = 1.0 if "wls_mu_1" in parts else 0.5 if "wls_mu_half" in parts else 0.0
    return three(use, *args[1:], extra_fn=extra_fn, poisson_iters=poisson, wls_power=wls)


def main() -> None:
    cfg, seed, all_df, train_df, val_df, spec, back, cut, k = setup()
    args = (spec, train_df, val_df, all_df, cut, back, k)
    combos = [a for a in sys.argv[1:] if not a.startswith("out=")]
    out_name = next((a[4:] for a in sys.argv[1:] if a.startswith("out=")), "hyp_combo")
    names = combos or (sorted(EXTRA) + sorted(SPEC_BLOCKS) + CONFIG["variants"])
    base = three(*args)
    y_val, y_tr = val_df["cnt"].to_numpy(float), train_df["cnt"].to_numpy(float)
    rows = {"C5": scores_only(base)}
    print(f"{'C5':34s} seeded {base['seeded']:.4f} | days {base['day_block']:.4f} | chrono {base['chrono']:.4f}")
    for name in names:
        res = run_one(name, args, spec)
        row = scores_only(res)
        d_seed = paired_boot(y_val, res["seeded_pred"], base["seeded_pred"], seed)
        d_oof = paired_boot(y_tr, res["oof_pred"], base["oof_pred"], seed)
        fold_d = np.array(res["folds"]) - np.array(base["folds"])
        row.update({"d_seeded": d_seed, "d_oof": d_oof, "d_days": float(fold_d.mean()),
                    "d_days_fold_sd": float(fold_d.std(ddof=1)), "d_days_folds": fold_d.tolist(),
                    "d_days_paired_se": float(fold_d.std(ddof=1) / np.sqrt(len(fold_d))),
                    "folds_positive": int(np.sum(fold_d > 0)), "d_chrono": res["chrono"] - base["chrono"]})
        rows[name] = row
        print(f"{name:34s} seeded {d_seed[0]:+.4f} [{d_seed[1]:+.4f},{d_seed[2]:+.4f}] | days {row['d_days']:+.4f} "
              f"(paired se {row['d_days_paired_se']:.4f}, {row['folds_positive']}/5 folds up, OOF [{d_oof[1]:+.4f},{d_oof[2]:+.4f}])"
              f" | chrono {row['d_chrono']:+.4f} (ratio {res['chrono_mean_ratio']:.3f})")
    config = dict(CONFIG, combos=combos)
    sha = save(out_name if combos else "hyp", {"rows": rows}, config)
    print("config_sha", sha)


if __name__ == "__main__":
    main()
