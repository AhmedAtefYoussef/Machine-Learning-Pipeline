"""Every table the notebook shows, as one consistently styled table.

Three generic builders (`styled`, `facts`, `checks`) do all the formatting; the per-table functions below only choose
and rename columns from the artifacts. The functions read artifacts and nothing else: no model is fitted here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import config_seed, sha256_file

INK, HEADER_FILL, ZEBRA_FILL, CHOICE_FILL, BAR_COLOUR = "#0b0b0b", "#f1f0ed", "#f8f8f6", "#e6f0fb", "#b7d3f6"
METHODS = {"l2": "ridge (L2)", "l1": "lasso (L1)", "enet": "elastic net"}
R5, SCI = "{:.5f}", "{:.3e}"   # formats for columns whose differences live in the fifth decimal, or that are tiny

TABLE_STYLES = [
    {"selector": "", "props": f"border-collapse: collapse; font-size: 13px; background: #ffffff; color: {INK};"},
    {"selector": "caption", "props": f"caption-side: top; text-align: left; font-weight: 600; padding: 2px 0 6px 0; color: {INK};"},
    {"selector": "th", "props": f"background: {HEADER_FILL}; border-bottom: 1px solid #d9d8d4; text-align: left; "
                                f"font-weight: 600; padding: 5px 10px; color: {INK};"},
    {"selector": "td", "props": "padding: 4px 10px;"},
]


def _number(value: float) -> str:
    """Four decimals for ordinary numbers, five for small ones, scientific notation for tiny ones."""
    if value == 0 or abs(value) >= 0.1:
        return f"{value:.4f}"
    return f"{value:.5f}" if abs(value) >= 1e-3 else f"{value:.3e}"


def _cell(value) -> str:
    """One value as text: numbers as above, missing values as a dash, lists joined by commas."""
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return "–"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return _number(float(value))
    if isinstance(value, (list, tuple)):
        return ", ".join(_cell(v) for v in value)
    return str(value)


def styled(df, caption=None, formats=None, highlight=None, bars=None):
    """A pandas Styler with the notebook's look.

    formats   {column: python format string}; the other numbers get the default format of `_number`
    highlight boolean mask or list of row positions: the rows that are "the choice we made"
    bars      columns that get a thin in-cell bar
    """
    df = df.reset_index(drop=True)
    numeric = list(df.select_dtypes("number").columns)
    table = df.style.hide(axis="index").format(_cell).set_table_styles(TABLE_STYLES)
    if caption:
        table = table.set_caption(caption)
    for column, pattern in (formats or {}).items():
        table = table.format(pattern, subset=[column])
    text = [c for c in df.columns if c not in numeric]
    if text:
        table = table.set_properties(subset=text, **{"text-align": "left"})   # the notebook theme right-aligns by default
    if numeric:
        table = table.set_properties(subset=numeric, **{"text-align": "right"})
        table = table.set_table_styles({c: [{"selector": "th", "props": "text-align: right;"}] for c in numeric},
                                       overwrite=False)
    table = table.apply(_zebra, axis=None)
    if bars:
        table = table.bar(subset=bars, color=BAR_COLOUR, height=50, width=100)
    if highlight is not None:
        mask = np.asarray(highlight)
        rows = np.flatnonzero(mask) if mask.dtype == bool else mask.astype(int)
        table = table.apply(lambda frame: _fill(frame, rows), axis=None)
    return table


def _zebra(frame):
    css = pd.DataFrame("", index=frame.index, columns=frame.columns)
    css.iloc[1::2, :] = f"background-color: {ZEBRA_FILL}"
    return css


def _fill(frame, rows):
    css = pd.DataFrame("", index=frame.index, columns=frame.columns)
    css.iloc[rows, :] = f"background-color: {CHOICE_FILL}; font-weight: 600"
    return css


def facts(mapping, caption=None):
    """A two-column 'item | value' table for key-value results."""
    return styled(pd.DataFrame({"item": list(mapping), "value": [_cell(v) for v in mapping.values()]}), caption)


def checks(rows, caption=None):
    """rows of (check, value, passed) as a table with 'passed' or 'FAILED' for each."""
    frame = pd.DataFrame([(check, _cell(value), "passed" if ok else "FAILED") for check, value, ok in rows],
                         columns=["check", "value", "result"])
    return styled(frame, caption).map(_result_colour, subset=["result"])


def _result_colour(text):
    return "color: #1a7f4b" if text == "passed" else "color: #b3261e; font-weight: 600"


def _pick(frame, columns: dict):
    """The given columns of a frame, in this order, under their plain-word names."""
    return frame[list(columns)].rename(columns=columns)


def _rows(mapping: dict, key: str, names: dict) -> pd.DataFrame:
    """A dict of dicts as a table: one row per entry, its name in column `key`, readable names for the entries."""
    return pd.DataFrame([{key: names.get(name, name), **values} for name, values in mapping.items()])


# ---- Phase 0: data -------------------------------------------------------------------------------------------------

def p0_config(cfg, seed):
    return facts({"team ids": cfg["team_ids"],
                  "seed": f"{seed} (sha256 of the sorted ids joined by '_', mod 100000)",
                  "self-test": "41698 reproduced from the brief's example ids",
                  "Phase 1 learning rate, as a fraction of the stability bound": cfg["p1"]["lr_fraction_of_bound"],
                  "Phase 2 degree candidates": cfg["p2"]["degree_candidates"],
                  "plateau tolerance": cfg["p2"]["plateau_tol"]}, "Configuration (config.yaml)")


def p0_split(train_all, train_df, val_df):
    return facts({"labelled rows": train_all.shape, "training rows": train_df.shape, "validation rows": val_df.shape,
                  "columns": ", ".join(train_all.columns)}, "The labelled data and the seeded 80/20 split")


def p0_quirks(train_all):
    return facts({
        "rows with hum == 0": int((train_all["hum"] == 0).sum()),
        "share of rows with windspeed == 0": round(float((train_all["windspeed"] == 0).mean()), 4),
        **{f"rows with weathersit == {k}": int((train_all["weathersit"] == k).sum()) for k in (1, 2, 3, 4)},
        "correlation of temp and atemp": round(float(np.corrcoef(train_all["temp"], train_all["atemp"])[0, 1]), 4),
        "rows where workingday != (Mon-Fri and not holiday)": int(
            (train_all["workingday"] != (train_all["weekday"].between(1, 5) & (train_all["holiday"] == 0)).astype(int)).sum()),
        "rows where season != calendar quarter of mnth": int((train_all["season"] != (train_all["mnth"] - 1) // 3 + 1).sum()),
    }, "Quirks of the data")


# ---- Phase 1: gradient descent -------------------------------------------------------------------------------------

def p1_design(p1):
    return facts({"training rows": p1["n_train"], "validation rows": p1["n_val"],
                  "columns (with bias)": p1["n_features"], "target": p1["target_transform"],
                  "first columns": ", ".join(p1["feature_names"][:4]) + " ...",
                  "λ_max of XᵀX/n": p1["lambda_max"], "stability bound 2/λ_max": p1["lr_bound"],
                  "learning rate used": f"{p1['lr']:.4f} = {p1['lr_fraction']} x bound",
                  "condition number": p1["condition_number"]}, "Phase 1: the design and the learning rate")


def p1_live_check(p1, live, live_lr):
    return facts({"live run": f"{live.stop_reason} after {live.iterations} iterations at lr {live_lr:.4f}",
                  "stored": f"{p1['stop_reason']} after {p1['iterations']} iterations at lr {p1['lr']:.4f}",
                  "largest difference between live and stored weights":
                      float(np.abs(live.weights - np.array(p1["weights"])).max())},
                 "Live gradient descent against the stored run")


def p1_lr_sweep(p1):
    frame = _pick(pd.DataFrame(p1["lr_sweep"]), {"fraction": "fraction of bound", "lr": "learning rate",
                                                 "stop_reason": "stop reason", "iterations": "iterations",
                                                 "loss_final": "final loss"})
    return styled(frame, "Learning-rate sweep (highlighted: the one we use)",
                  formats={"learning rate": "{:.4f}"}, highlight=np.isclose(frame["fraction of bound"], p1["lr_fraction"]))


def p1_fit(p1):
    return facts({"stop reason": p1["stop_reason"], "iterations": p1["iterations"],
                  "final loss": f"{p1['train_loss_final']:.6f}",
                  "gradient norm at the end": f"{p1['grad_norm_final']:.3e} (tolerance {p1['tol_grad']})"},
                 "The final gradient-descent run")


def p1_oracle(p1):
    oracle = p1["oracle"]
    return facts({"reference": oracle["label"], "largest weight difference": oracle["max_abs_weight_diff"],
                  "loss gap (GD minus exact)": oracle["loss_gap"], "validation R² difference": oracle["val_r2_diff"],
                  "gradient check at zeros (max relative error)": p1["gradient_check_at_zeros"],
                  "gradient check at a random point": p1["gradient_check_at_random"]},
                 "Gradient descent against the closed-form solution")


def p1_backtransform(p1):
    frame = _rows(p1["backtransform"]["candidates"], "method", {"ls": "least-squares factor", "none": "no factor"})
    frame = _pick(frame, {"method": "method", "factor": "factor", "val_mean_ratio": "mean ratio (validation)",
                          "val_r2": "validation R²", "val_rmse": "validation RMSE"})
    chosen = p1["backtransform"]
    return styled(frame, f"Back-transform; chosen: {chosen['method']} with factor {chosen['factor']:.4f}",
                  highlight=frame["method"] == {"ls": "least-squares factor", "none": "no factor"}[chosen["method"]])


def p1_target_power(p1):
    frame = _pick(pd.DataFrame(p1["target_power_table"]),
                  {"power": "exponent", "iterations": "iterations", "stop_reason": "stop reason",
                   "train_r2": "train R²", "val_r2": "validation R²", "val_rmse": "validation RMSE"})
    return styled(frame, f"Target exponent (0 = log); in use: {p1['target_power']}, best for this design: "
                         f"{p1['target_power_best_here']}", highlight=np.isclose(frame["exponent"], p1["target_power"]))


def p1_scores(p1):
    boot = p1["val_bootstrap"]
    frame = pd.DataFrame({"": ["train", "validation"], "R²": [p1["train_r2"], p1["val_r2"]],
                          "RMSE": [p1["train_rmse"], p1["val_rmse"]],
                          "R² on the target z": [p1["train_r2_log"], p1["val_r2_log"]]})
    return styled(frame, f"Scores. Validation R² 95% interval [{boot['lo']:.4f}, {boot['hi']:.4f}] "
                         f"(standard error {boot['se']:.4f})")


def p1_hour_encoding(p1):
    frame = _rows(p1["hour_encoding_ablation"], "hour encoding",
                  {"one_hot": "one-hot dummies", "harmonics6": "six sine/cosine pairs", "numeric": "hour as a number"})
    frame = _pick(frame, {"hour encoding": "hour encoding", "n_features": "weights", "val_r2": "validation R²",
                          "iterations": "iterations", "stop_reason": "stop reason"})
    frame["iterations"] = frame["iterations"].astype("Int64")   # missing for the closed-form fit
    return styled(frame, "Hour encodings (highlighted: the one we use)", highlight=frame["hour encoding"] == "one-hot dummies")


def p1_bonus(p1):
    bonus = p1["bonus"]
    frame = _rows({m: bonus["val"][m] for m in ("mse_model", "asym_model")}, "model",
                  {"mse_model": "MSE model", "asym_model": "asymmetric model"})
    frame = _pick(frame, {"model": "model", "sq_cost": "squared cost", "abs_cost": "absolute cost",
                          "under_share": "share under-predicted", "mean_error": "mean error", "r2": "R²",
                          "rmse": "RMSE", "mean_pred": "mean prediction"})
    return styled(frame, f"Asymmetric cost with k = {bonus['k']:g}: {bonus['iterations']} iterations, stop reason "
                         f"{bonus['stop_reason']}, gradient check {bonus['gradient_check']:.1e}; mean prediction ratio "
                         f"asymmetric / MSE {bonus['mean_shift_ratio']:.4f}")


def p1_bonus_control(p1, control):
    rows = {"Phase 1 (z-scale MSE)": p1["bonus"]["val"]["mse_model"], "bike-scale, k = 1": control["val"]["k1_model"],
            f"bike-scale, k = {p1['bonus']['k']:g}": p1["bonus"]["val"]["asym_model"]}
    frame = _rows(rows, "model", {})
    frame = _pick(frame, {"model": "model", "sq_cost": "squared cost", "abs_cost": "absolute cost",
                          "under_share": "share under-predicted", "mean_error": "mean error", "rmse": "RMSE",
                          "mean_pred": "mean prediction"})
    return styled(frame, f"Control for the bonus. Shift from fitting on bikes {control['shift_from_fitting_on_bikes']:.3f}, "
                         f"from the asymmetry {control['shift_from_asymmetry']:.3f}, total {control['total_shift']:.3f}",
                  formats={c: "{:.3f}" for c in ("squared cost", "absolute cost", "share under-predicted", "mean error",
                                                 "RMSE", "mean prediction")})


# ---- Phase 2: polynomial regression --------------------------------------------------------------------------------

def p2_handover(p1, p2):
    difference = p2["init_loss"] - p1["train_loss_final"]
    return facts({"Phase 1 final training loss": repr(p1["train_loss_final"]),
                  "Phase 2 initial loss": repr(p2["init_loss"]), "difference": difference,
                  "hand-over check (|difference| <= 1e-9)": "passed" if abs(difference) <= 1e-9 else "FAILED"},
                 "The hand-over from Phase 1 to Phase 2")


def p2_design(p2):
    return facts({"blocks": p2["blocks"], "power columns": p2["power_cols"],
                  "columns": f"{len(p2['feature_names'])} ({len(p2['expanded_feature_names'])} new)",
                  "first new columns": ", ".join(p2["expanded_feature_names"][:6]) + " ...",
                  "λ_max": p2["lambda_max"], "learning rate": p2["lr"], "condition number": p2["condition_number"]},
                 "Phase 2: the expanded design")


def p2_degree_sweep(p2):
    frame = _pick(pd.DataFrame(p2["degree_sweep"]), {"degree": "degree", "n_features": "weights",
                                                     "iterations": "iterations", "stop_reason": "stop reason",
                                                     "train_r2": "train R²", "val_r2": "validation R²"})
    return styled(frame, f"Degree (chosen: {p2['degree']}, the smallest within {p2['plateau_tol']} of the best "
                         "validation R²)", highlight=frame["degree"] == p2["degree"])


def p2_power_cols(p2):
    sweep = pd.DataFrame(p2["power_col_sweep"])
    sweep["power_cols"] = sweep["power_cols"].apply(", ".join)
    frame = _pick(sweep, {"power_cols": "power columns", "n_features": "weights", "iterations": "iterations",
                          "stop_reason": "stop reason", "train_r2": "train R²", "val_r2": "validation R²"})
    return styled(frame, "Which variables get powers (highlighted: the set we keep)",
                  highlight=frame["power columns"] == ", ".join(p2["power_cols"]))


def p2_ablation(p2):
    without = p2["block_ablation"]["without_wd_x_hr"]
    frame = pd.DataFrame({"model": ["without the workingday x hour block", "with the block"],
                          "weights": [without["n_features"], len(p2["feature_names"])],
                          "train R²": [without["train_r2"], p2["train_r2"]],
                          "validation R²": [without["val_r2"], p2["val_r2"]]})
    return styled(frame, "What the workingday x hour block adds", highlight=[1])


def p2_scores(p1, p2):
    boot, paired = p2["val_bootstrap"], p2["paired_vs_p1"]
    frame = pd.DataFrame({"": ["Phase 1", "Phase 2"], "R²": [p1["val_r2"], p2["val_r2"]],
                          "RMSE": [p1["val_rmse"], p2["val_rmse"]]})
    return styled(frame, f"Validation scores. Phase 2 R² 95% interval [{boot['lo']:.4f}, {boot['hi']:.4f}]; gain over "
                         f"Phase 1 {paired['delta']:.4f}, paired 95% interval [{paired['lo']:.4f}, {paired['hi']:.4f}]")


def p2_fit(p2):
    return facts({"stop reason": p2["stop_reason"], "iterations": p2["iterations"],
                  "largest weight difference to the closed-form solution": p2["oracle"]["max_abs_weight_diff"]},
                 "The Phase 2 run")


# ---- Phase 3: bias and variance ------------------------------------------------------------------------------------

def p3_chain(p2, p3):
    anchor = p3["ladder"][p3["anchor"]["anchor_index"]]["seeded"]["r2"]
    return facts({"Phase 2 degree": p2["degree"], "Phase 2 features (with bias)": len(p2["feature_names"]),
                  "Phase 2 power columns": p2["power_cols"], "Phase 2 blocks": p2["blocks"],
                  "Phase 2 validation R² (gradient descent)": f"{p2['val_r2']:.5f}",
                  "anchor validation R² (closed form here)": f"{anchor:.5f}",
                  "anchor check (closed form minus gradient descent)": p3["anchor_check"],
                  "back-transform method reused from Phase 1": p3["back_method"]}, "The anchor is the Phase 2 design")


def p3_ladder(p3):
    rows = [{"level": r["name"], "weights": r["n_features"], "train R²": r["seeded"]["train_r2"],
             "validation R²": r["seeded"]["r2"], "gap": r["gap_seeded"], "held-out days R²": r["day_block"]["r2"],
             "held-out days sd": r["day_block"]["sd"], "chronological R²": r["chrono"]["r2"]} for r in p3["ladder"]]
    frame = pd.DataFrame(rows)
    return styled(frame, f"Complexity ladder (highlighted: target level). Best on held-out days: {p3['best_level_day_block']}"
                         f"; best chronologically: {p3['best_level_chrono']}; over-fit from level {p3['overfit_from_level']}",
                  highlight=frame["level"] == p3["target_level"])


def p3_degree_axis(p3):
    rows = [{"degree": r["degree"], "weights": r["n_features"], "train R²": r["seeded"]["train_r2"],
             "validation R²": r["seeded"]["r2"], "held-out days R²": r["day_block"]["r2"],
             "chronological R²": r["chrono"]["r2"]} for r in p3["degree_axis"]]
    return styled(pd.DataFrame(rows), "Degree axis (anchor blocks and power columns fixed). Range of validation R² over "
                                      f"degrees >= 2: {p3['diagnosis']['degree_axis_range']:.5f}")


def p3_learning_curves(p3):
    frame = pd.DataFrame([{"design": key, **c} for key, curve in p3["learning_curves"].items() for c in curve])
    return styled(_pick(frame, {"design": "design", "fraction": "share of training days", "n_days": "days",
                                "n_rows": "rows", "train_r2": "train R²", "val_r2": "validation R²"}),
                  "Learning curves (the plot clips R² at -0.2)")


def p3_estimates(p3):
    rows = []
    for label, est, gaps in (("anchor", p3["estimates"], p3["gaps"]["anchor"]),
                             ("target", p3["estimates_target"], p3["gaps"]["target"])):
        rows.append({"design": label, "validation R²": est["seeded"]["r2"], "interval low": est["seeded"]["lo"],
                     "interval high": est["seeded"]["hi"], "held-out days R²": est["day_holdout"]["r2"],
                     "held-out days sd": est["day_holdout"]["sd"], "chronological R²": est["chrono"]["r2"],
                     "leakage gap": gaps["leakage"], "drift gap": gaps["drift"]})
    gain = p3["paired_target_vs_anchor"]
    return styled(pd.DataFrame(rows), f"Three estimates. Paired gain target - anchor (validation): {gain['delta']:.4f}, "
                                      f"interval [{gain['lo']:.4f}, {gain['hi']:.4f}]", highlight=[1])


def p3_chrono_detail(p3):
    frame = pd.DataFrame([{"design": name, **detail} for name, detail in p3["chrono_detail"].items()])
    return styled(frame, "The late period against the early one. 'r2_level_corrected' rescales the late predictions by the "
                         "late period's own mean: a diagnostic of level drift only, not a usable estimate")


def p3_noise_floor(p3):
    return facts(p3["noise_floor"], "How much variance is left once the design cells are fully known (an optimistic ceiling)")


def p3_diagnosis(p3):
    target = p3["target_complexity"]
    return facts({**p3["diagnosis"], "target": target["level"], "target weights": target["n_features"],
                  "target degree": target["degree"], "target power columns": target["power_cols"],
                  "target blocks": target["blocks"]}, "Diagnosis and target complexity")


def p3_exponents(p3):
    frame = _pick(pd.DataFrame(p3["target_power_check"]),
                  {"power": "exponent", "seeded_r2": "validation R²", "day_block_r2": "held-out days R²",
                   "chrono_r2": "chronological R²"})
    return styled(frame, f"Target exponent on the target design (in use: {p3['target_power_used']}; best on validation: "
                         f"{p3['target_power_best_seeded']}; within the plateau tolerance: {p3['target_power_consistent']})",
                  highlight=np.isclose(frame["exponent"], p3["target_power_used"]))


# ---- Phase 4: regularization ---------------------------------------------------------------------------------------

def _stage(p4, stage):
    """(methods, comparisons, unregularised, name of the highlighted method) of stage A ('full') or B ('final')."""
    if stage == "full":
        return p4["methods"], p4["comparisons"], p4["unregularised"], p4["full_design_best"]["method"]
    return p4["final"]["methods"], p4["final"]["comparisons"], p4["final"]["unregularised"], p4["recommended"]["method"]


def p4_design(p3, p4):
    origin = p4["design_from"]
    return facts({"Phase 3 target level": p3["target_level"], "Phase 3 weights (with bias)": p3["target_complexity"]["n_features"],
                  "design used here": origin["level"], "weights before candidates": origin["n_features_before_candidates"],
                  "candidates added": origin["candidates_added"], "candidates skipped (already present)": origin["candidates_skipped"],
                  "weights after candidates (with bias)": p4["n_features"], "back-transform method": p4["back_method"],
                  "solver tolerance": p4["solver"]["tol"], "solver sweeps allowed": p4["solver"]["max_sweeps"],
                  "solutions that used all sweeps": p4["not_converged"]["count"]}, "Phase 4: the design")


def p4_unregularised(p4, stage="full"):
    names = {"alpha": "penalty λ", "train_r2": "train R²", "train_rmse": "train RMSE", "val_r2": "validation R²",
             "val_rmse": "validation RMSE", "day_block_r2": "held-out days R²", "chrono_r2": "chronological R²"}
    shown = {names[k]: SCI.format(v) if k == "alpha" else f"{v:.2f}" if "rmse" in k else f"{v:.5f}"
             for k, v in _stage(p4, stage)[2].items()}
    return facts(shown, "The same design with a negligible penalty (closed-form least squares)")


def p4_methods(p4, stage="full"):
    methods, _, _, chosen = _stage(p4, stage)
    frame = _rows(methods, "method", METHODS)
    frame["of"] = frame["n_features"] - 1
    frame = _pick(frame, {"method": "method", "lambda": "penalty λ", "l1_ratio": "l1 ratio", "val_r2": "validation R²",
                          "val_lo": "interval low", "val_hi": "interval high", "day_block_r2": "held-out days R²",
                          "day_block_se": "held-out days s.e.", "chrono_r2": "chronological R²",
                          "n_nonzero": "non-zero weights", "of": "out of"})
    what = "best on validation" if stage == "full" else f"recommended: {chosen}"
    return styled(frame, f"Chosen penalty per method ({what}, highlighted)",
                  formats={"penalty λ": SCI, **{c: R5 for c in frame.columns if "R²" in c or "interval" in c or "s.e." in c}},
                  highlight=frame["method"] == METHODS[chosen])


def p4_enet_ratios(p4):
    rows = [{"l1 ratio": float(rho), "best validation R²": max(r["val_r2"] for r in curve),
             "best penalty": max(curve, key=lambda r: (r["val_r2"], r["alpha"]))["alpha"]}
            for rho, curve in p4["curves"]["enet"].items()]
    frame = pd.DataFrame(rows)
    return styled(frame, "Elastic net: best penalty for each l1 ratio", formats={"best validation R²": R5, "best penalty": SCI},
                  highlight=np.isclose(frame["l1 ratio"], p4["methods"]["enet"]["l1_ratio"]))


def p4_comparisons(p4, stage="full"):
    frame = pd.DataFrame([{"comparison": k.replace("_minus_", " - "), **v} for k, v in _stage(p4, stage)[1].items()])
    frame = _pick(frame, {"comparison": "comparison", "delta": "difference in validation R²", "lo": "interval low",
                          "hi": "interval high"})
    return styled(frame, "Paired bootstrap differences (an interval that contains zero is a tie)",
                  formats={c: R5 for c in frame.columns[1:]})


def p4_lambda_picks(p4):
    rows = [{"method": METHODS[m], "validation best": p4["methods"][m]["lambda"], "CV best": c["cv_best_alpha"],
             "CV 1-s.e.": c["cv_1se_alpha"], "R² at CV best": c["val_r2_at_cv_best"],
             "R² at CV 1-s.e.": c["val_r2_at_cv_1se"], "R² at validation best": p4["methods"][m]["val_r2"]}
            for m, c in p4["cv"].items()]
    return styled(pd.DataFrame(rows), "Two ways to pick the penalty: best validation R² against held-out-day CV",
                  formats={"validation best": SCI, "CV best": SCI, "CV 1-s.e.": SCI, "R² at CV best": R5,
                           "R² at CV 1-s.e.": R5, "R² at validation best": R5})


def p4_entry(p4):
    frame = pd.DataFrame([{"column": name, "enters the lasso path at alpha": p4["entry_alpha"][name]}
                          for name in ("temp", "atemp", "yr", "instant", "trend", "workingday", "holiday")])
    frame["enters the lasso path at alpha"] = frame["enters the lasso path at alpha"].map(
        lambda a: "never" if a is None else f"{a:.3e}")
    return styled(frame, "Where each correlated pair enters the lasso path (the largest alpha with a non-zero weight)")


def p4_verdicts(p4):
    rows = []
    for column, v in p4["column_verdicts"].items():
        n = v["numbers"]
        rows.append({"column": column, "verdict": v["verdict"],
                     "R² lost, column alone [interval]":
                         f"{n['drop_alone']['delta']:+.4f} [{n['drop_alone']['lo']:+.4f}, {n['drop_alone']['hi']:+.4f}]",
                     "with partners": None if n["drop_group"] is None else n["drop_group"]["delta"],
                     "alone explains": n["solo_r2"], "lasso keeps": f"{n['lasso']['n_own_nonzero']}/{n['lasso']['n_own']}",
                     "best selection frequency": n["lasso"]["max_freq_own"],
                     "carried by": ", ".join(v["carried_by"]), "representative": v["representative"]})
    return styled(pd.DataFrame(rows), "Verdict for every original column (cost = validation R² lost when removed)")


def p4_survivors(p4):
    selection = p4["selection"]
    return facts({**{f"survivors, {k}": v for k, v in p4["survivor_counts"].items()},
                  "selection lasso: penalty λ": SCI.format(selection["lambda"]), "selection lasso: validation R²": f"{selection['val_r2']:.4f}",
                  "selection lasso: non-zero": f"{selection['n_nonzero']} of {selection['n_columns']}",
                  "selection lasso: fits that used all sweeps": selection["not_converged"]["count"],
                  "original columns kept": p4["survivors_original"],
                  "survivor columns (expanded features)": len(p4["survivors_expanded"])}, "The survivors handed to Phase 5")


def p4_recommended(p4):
    rec = p4["recommended"]
    return facts({"stage B columns": p4["final"]["n_columns"], "stage B fits that used all sweeps": p4["not_converged_final"],
                  "methods within noise of the best": [METHODS[m] for m in rec["candidates"]],
                  "recommended": METHODS[rec["method"]], "fitted on": rec["fitted_on"], "penalty λ": f"{rec['lambda']:.3e}",
                  "l1 ratio": rec["l1_ratio"], "validation R²": f"{rec['val_r2']:.5f}",
                  "held-out days R²": f"{rec['day_block_r2']:.5f}", "chronological R²": f"{rec['chrono_r2']:.5f}"},
                 "The recommended regression model")


RICH_NAMES = {"lambda": "penalty λ", "day_block_r2": "held-out days R²", "val_r2": "validation R²", "n_nonzero": "non-zero weights"}


def p4_rich(p4):
    rich = p4["rich_check"]
    gap = rich["paired_best_vs_recommended"]
    return facts({"level": rich["level"], "weights (with bias)": rich["n_features"],
                  "unregularised validation R²": f"{rich['unregularised_val_r2']:.4f}",
                  **{f"{METHODS[m]}: {RICH_NAMES[k]}": SCI.format(v) if k == "lambda" else f"{v:.5f}" if isinstance(v, float) else v
                     for m in ("l2", "l1") for k, v in rich[m].items()},
                  "best penalised top level minus the recommended model":
                      f"{gap['delta']:.5f} [{gap['lo']:.5f}, {gap['hi']:.5f}]"},
                 "Does a penalty rescue the over-fit top of the ladder? (informational)")


def p4_chain_rows(p3, p4):
    """(check, value, passed) rows: Phase 4 was built from the Phase 3 artifact on disk and from its target level."""
    origin = p4["design_from"]
    return [
        ("Phase 4 was built from the Phase 3 artifact on disk", "hash equal", p4["upstream_sha256"] == sha256_file("artifacts/p3.json")),
        ("Phase 4 used the Phase 3 target level", f"{p3['target_complexity']['level']} (diagnosis {p3['diagnosis']['label']!r})",
         origin["level"] == p3["target_complexity"]["level"]),
        ("the design had Phase 3's number of weights before the candidates", origin["n_features_before_candidates"],
         origin["n_features_before_candidates"] == p3["target_complexity"]["n_features"]),
    ]


# ---- Phase 5: logistic regression ----------------------------------------------------------------------------------

def p5_design(p5):
    return facts({"features (no bias)": len(p5["features"]), "columns including bias": p5["n_features"],
                  "training rows": p5["n_train"], "validation rows": p5["n_val"]}, "Phase 5: the design")


def p5_chain_rows(p4, p5):
    """(check, value, passed) rows: every Phase 5 feature is a Phase 4 survivor."""
    survivors = set(p4["survivors_expanded"])
    return [("every Phase 5 feature is a Phase 4 survivor",
             f"{len(p5['features'])} features, {len(survivors)} survivors from {p4['survivors_original']}",
             set(p5["features"]) <= survivors)]


def p5_thresholds(p5):
    rule = p5["threshold_rule"]
    table = pd.DataFrame(rule["table"])
    peaks = table[table["hr"].isin([3, 8, 12, 17])].pivot_table(index=["yr", "workingday"], columns="hr",
                                                                values="threshold").reset_index()
    peaks["yr"] = peaks["yr"].map({0: "2011", 1: "2012"})
    peaks["workingday"] = peaks["workingday"].map({0: "non-working day", 1: "working day"})
    peaks.columns = ["year", "day type"] + [f"hour {h}" for h in peaks.columns[2:]]
    return styled(peaks, f"Threshold in bikes: the {rule['quantile']} quantile of training hours with the same "
                         f"{' + '.join(rule['group_by'])} ({rule['n_cells']} cells, at least {rule['min_cell_n']} hours each, "
                         f"fitted on {rule['fitted_on']})", formats={c: "{:.1f}" for c in peaks.columns[2:]})


def p5_balance(p5):
    balance = p5["class_balance"]
    return facts({"positives, train": balance["train"], "positives, validation": balance["val"],
                  "positives, validation 2011": balance["val_by_year"]["0"],
                  "positives, validation 2012": balance["val_by_year"]["1"],
                  "majority-class accuracy": balance["majority_accuracy"]}, "Class balance")


def p5_label_variants(p5):
    frame = pd.DataFrame(p5["label_variants"])
    frame["rule"] = frame["group_by"].map(lambda g: " + ".join(g) or "global")
    shown = _pick(frame, {"rule": "label rule", "pos_train": "positives, train", "pos_val": "positives, validation",
                          "pos_val_2011": "validation 2011", "pos_val_2012": "validation 2012",
                          "auc_hour_only": "AUC, hour only", "auc_trend_only": "AUC, trend only", "auc_full": "AUC, all columns",
                          "acc_full_at_0_5": "accuracy at 0.5", "majority_accuracy": "majority accuracy"})
    return styled(shown, f"The same three models under three label rules (highlighted: ours). Foil fits that did not "
                         f"converge: {p5['foil_not_converged']}", formats={c: "{:.3f}" for c in shown.columns[1:]},
                  highlight=frame["group_by"].map(lambda g: list(g) == list(p5["threshold_rule"]["group_by"])))


def p5_l2_sweep(p5):
    frame = _pick(pd.DataFrame(p5["l2_sweep"]), {"l2": "ridge strength", "iterations": "iterations", "stop_reason": "stop reason",
                                                 "train_logloss": "train log-loss", "val_logloss": "validation log-loss",
                                                 "val_roc_auc": "validation ROC-AUC"})
    return styled(frame, "Ridge strength by validation ROC-AUC (ties within 1e-4 go to the larger value)",
                  formats={"ridge strength": SCI, **{c: "{:.5f}" for c in frame.columns[3:]}},
                  highlight=np.isclose(frame["ridge strength"], p5["l2"]))


def p5_fit(p5):
    return facts({"chosen ridge strength": SCI.format(p5["l2"]), "iterations": p5["iterations"], "stop reason": p5["stop_reason"],
                  "learning rate": p5["lr"],
                  "gradient check at zeros (max relative error)": f"{p5['gradient_check_at_zeros']:.1e}",
                  "gradient check halfway to the final weights": f"{p5['gradient_check_at_half']:.1e}"},
                 "The chosen classifier")


def p5_metrics(p5):
    columns = ["threshold", "accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc", "cost"]
    frame = pd.DataFrame({"at 1/(1+c)": p5["metrics"], "at 0.5": p5["metrics_at_0_5"], "at F1-optimal": p5["metrics_at_f1_opt"],
                          "training rows, at 1/(1+c)": p5["train_metrics"]}).T.reindex(columns=columns).astype(float)
    frame.insert(0, "cut-off", frame.index)
    frame = frame.rename(columns={"f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"})
    boot = p5["val_auc_bootstrap"]
    return styled(frame, f"Validation metrics (highlighted: our operating cut-off). Majority-class accuracy "
                         f"{p5['class_balance']['majority_accuracy']:.4f}; validation ROC-AUC 95% interval "
                         f"{boot['lo']:.4f} to {boot['hi']:.4f}", highlight=[0])


def p5_confusion(p5):
    conf = p5["metrics"]["confusion"]
    frame = pd.DataFrame({"truth": ["truly normal", "truly high"], "predicted normal": [conf["tn"], conf["fn"]],
                          "predicted high": [conf["fp"], conf["tp"]]})
    return styled(frame, "Confusion matrix at the cost-based cut-off (rows = truth)")


def p5_threshold(p5):
    return facts({"cost ratio (miss : false alarm)": p5["cost_ratio"], "cost-based cut-off 1/(1+c)": p5["t_cost"],
                  "grid threshold with the lowest validation cost": p5["t_cost_empirical"],
                  "grid threshold with the highest F1": p5["t_f1"]}, "The operating threshold")


def p5_calibration(p5):
    return facts({"largest gap between predicted and observed share (bins with >= 30 rows)": p5["calibration_max_gap"]},
                 "Calibration")


def p5_top_coefficients(p5):
    return styled(pd.DataFrame(p5["top_coefficients"]).rename(columns={"name": "feature", "weight": "standardised weight"}),
                  "The largest standardised weights", formats={"standardised weight": "{:.3f}"})


def p5_column_share(p5):
    frame = pd.DataFrame({"original column": list(p5["coef_abs_share_by_column"]),
                          "share of sum |w|": list(p5["coef_abs_share_by_column"].values())})
    return styled(frame, "Share of sum |w| by original column", formats={"share of sum |w|": "{:.3f}"}, bars=["share of sum |w|"])


def p5_retrospective(p5):
    frame = _pick(pd.DataFrame(p5["retrospective"]), {"phase": "phase", "consumed": "consumed", "hyperparameters": "settings",
                                                      "extra": "result", "n_features": "weights", "train_score": "train score",
                                                      "val_score": "validation score", "val_rmse": "validation RMSE"})
    return styled(frame, "Pipeline retrospective (highlighted: the regression model our pipeline recommends)",
                  highlight=frame["phase"].str.startswith("P4"))


# ---- Phase 6 and the whole chain -----------------------------------------------------------------------------------

def p6_summary(p6):
    pred, ab = p6["pred"], p6["a_vs_b_on_test"]
    ratio, daily = p6["profile_ratio"], p6["daily_total_ratio"]
    return facts({
        "model (Phase 4 recommended)": METHODS[p6["model"]["method"]], "penalty λ": SCI.format(p6["model"]["lambda"]),
        "l1 ratio": p6["model"]["l1_ratio"], "refit on": p6["refit_on"], "rows used to fit": p6["n_fit_rows"],
        "rows predicted": p6["n_test_rows"], "model A reproduces the Phase 4 weights (max gap)": p6["model_a_gap_to_p4"],
        "prediction mean": pred["mean"], "prediction median": pred["median"], "prediction min": pred["min"],
        "prediction max": pred["max"], "predictions below 1 bike": pred["n_below_1"],
        "predictions clipped at 0": pred["n_clipped"], "mean cnt of the training rows": p6["train_cnt_mean"],
        "A versus B on the hidden rows: correlation": ab["corr"], "A versus B: mean ratio B / A": ab["mean_ratio"],
        "A versus B: mean absolute difference": ab["mean_abs_diff"], "A versus B: largest absolute difference": ab["max_abs_diff"],
        "profile ratio min / max (per day type and hour)": f"{ratio['min']:.3f} / {ratio['max']:.3f}",
        "cells outside 0.4-2.5": ratio["cells_outside_0_4_2_5"],
        "daily-total ratio min / max": f"{daily['min']:.3f} / {daily['max']:.3f}",
        "mean prediction 2011 / 2012": f"{p6['by_year_mean']['0']:.1f} / {p6['by_year_mean']['1']:.1f}",
    }, "The submission and the model behind it")


def p6_submission(sub, p6):
    daily = p6["daily_total_ratio"]
    return facts({"submission shape": sub.shape, "columns": list(sub.columns),
                  "asserted above": "columns, row count and order equal the hidden file; no NaN; no negatives; not all zero",
                  "cells outside the 0.4-2.5 band of the training profile": p6["profile_ratio"]["cells_outside_0_4_2_5"],
                  "daily totals against same-month training days": f"{daily['min']:.2f} to {daily['max']:.2f}"},
                 "The submission file")


def p6_chain_rows(arts, cfg):
    """The chain assertions as (check, value, passed) rows: hash chain p1 -> p6, one seed, the hand-overs."""
    rows = [(f"{name}.upstream_sha256 == sha256(artifacts/{prev}.json)", "hash equal",
             arts[name]["upstream_sha256"] == sha256_file(f"artifacts/{prev}.json"))
            for prev, name in zip(("p1", "p2", "p3", "p4", "p5"), ("p2", "p3", "p4", "p5", "p6"))]
    p1, p2, p4, p5, p6 = (arts[k] for k in ("p1", "p2", "p4", "p5", "p6"))
    loss_gap = abs(p2["init_loss"] - p1["train_loss_final"])
    rows += [("p1 has no upstream", "none", p1["upstream_sha256"] is None),
             ("same seed in p1..p6 and equal to the seed of the config", config_seed(cfg),
              {a["seed"] for a in arts.values()} == {config_seed(cfg)}),
             ("p2 starts from the p1 weights", p2["init_weights_source"], p2["init_weights_source"] == "p1"),
             ("p2.init_loss == p1.train_loss_final (1e-9)", f"difference {loss_gap:.1e}",
              loss_gap <= 1e-9 * max(1.0, p1["train_loss_final"])),
             ("p5.features are a subset of p4.survivors_expanded", f"{len(p5['features'])} of {len(p4['survivors_expanded'])}",
              set(p5["features"]) <= set(p4["survivors_expanded"])),
             ("p4.recommended was fitted on the survivors, p6 uses the same columns", p4["recommended"].get("fitted_on"),
              p4["recommended"].get("fitted_on") == "survivors"),
             ("model A reproduces the p4 weights (< 1e-6)", f"max gap {p6['model_a_gap_to_p4']:.1e}",
              p6["model_a_gap_to_p4"] < 1e-6),
             ("p5 chosen model converged", p5["stop_reason"], p5["stop_reason"] == "converged")]
    return rows


def pipeline_summary(p1, p2, p3, p4, p5):
    """One line per phase: what it consumed, its settings, and the score of the model it ended with."""
    nan = float("nan")
    scores = [(p1["val_r2"], nan, nan),
              (p2["val_r2"], p3["estimates"]["day_holdout"]["r2"], p3["estimates"]["chrono"]["r2"]),
              (p3["estimates_target"]["seeded"]["r2"], p3["estimates_target"]["day_holdout"]["r2"],
               p3["estimates_target"]["chrono"]["r2"]),
              (p4["recommended"]["val_r2"], p4["recommended"]["day_block_r2"], p4["recommended"]["chrono_r2"]),
              (p5["metrics"]["roc_auc"], nan, nan)]
    frame = _pick(pd.DataFrame(p5["retrospective"]), {"phase": "phase", "consumed": "consumed", "hyperparameters": "settings",
                                                      "n_features": "weights"})
    frame["validation score"], frame["held-out days R²"], frame["chronological R²"] = zip(*scores)
    return styled(frame, "The pipeline at a glance (validation score: R² on bikes for Phases 1 to 4, ROC-AUC for Phase 5; "
                         "highlighted: the regression model we submit)", highlight=frame["phase"].str.startswith("P4"))
