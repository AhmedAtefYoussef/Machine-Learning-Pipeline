# %% [markdown]
# ## Phase 6 — Final model and submission
#
# **Which model.** The regression model our pipeline recommends is the Phase 4 stage-B model: {{p6.model.method}} with λ = {{p6.model.lambda:.1e}} and l1_ratio {{p6.model.l1_ratio}} on the surviving features. Nothing is tuned here.
#
# **Refit on training + validation, and why.** The hyper-parameters stay exactly as validated, but the weights are refitted on all {{p6.n_fit_rows}} labelled rows. The Phase 3 learning curve of the target was still rising at full size ({{p3.learning_curves.target.4.val_r2:.3f}} with 80% of the training days, {{p3.learning_curves.target.5.val_r2:.3f}} with all of them), so a quarter more rows should help a little, and the validation rows are days of the same months as the hidden ones. The scaler and the humidity fill stay as fitted on the training portion, so the hidden rows go through exactly the transformations used everywhere else. The cell first refits the validated model on the training rows alone and asserts that it reproduces the Phase 4 weights.
#
# **What we check without labels.** We cannot score the hidden days, and we do not tune anything on them. We only check that the predictions are sane: same rows and order as `test.csv`, no missing or negative values, the refitted model agrees closely with the validated one (correlation {{p6.a_vs_b_on_test.corr:.4f}}), and the predicted hourly profile by day type looks like the training profile (each cell between {{p6.profile_ratio.min:.2f}} and {{p6.profile_ratio.max:.2f}} times the training mean).
#
# **What the model cannot adapt to.** It extrapolates growth as a straight line in log space, which Phase 3 showed to be too steep beyond the observed period; it has never seen a zero-demand hour, weather outside the observed range, or holidays other than those in the training days. For the hidden days, which lie inside the two observed years, the held-out-day estimate of Phase 4 ({{p4.recommended.day_block_r2:.3f}}) is our best guess of the score; for a genuinely later period we would expect something nearer the chronological estimate ({{p4.recommended.chrono_r2:.3f}}).

# %%
# Phase 6 reads artifacts/p4.json (the recommended regression model on the surviving columns), refits it on training
# plus validation rows and predicts the hidden days. It does not re-run any earlier phase.
import numpy as np
import pandas as pd

from src import predict
from src.common import config_seed, load_config, load_test, load_train, read_artifact, sha256_file
from src.plots_p5 import plot_test_profile

CFG = globals().get("CFG") or load_config()
p6 = predict.run(CFG)  # writes sample_submission.csv and artifacts/p6.json

# %%
# What was submitted and how it compares with the model that was validated (model A, training rows only).
summary = {
    "model (Phase 4 recommended)": p6["model"]["method"],
    "lambda": p6["model"]["lambda"],
    "l1_ratio": p6["model"]["l1_ratio"],
    "refit on": p6["refit_on"],
    "rows used to fit": p6["n_fit_rows"],
    "rows predicted": p6["n_test_rows"],
    "model A reproduces the Phase 4 weights (max gap)": p6["model_a_gap_to_p4"],
    "prediction mean": p6["pred"]["mean"],
    "prediction median": p6["pred"]["median"],
    "prediction min": p6["pred"]["min"],
    "prediction max": p6["pred"]["max"],
    "predictions below 1 bike": p6["pred"]["n_below_1"],
    "predictions clipped at 0": p6["pred"]["n_clipped"],
    "mean cnt of the training rows": p6["train_cnt_mean"],
    "A versus B on the hidden rows: correlation": p6["a_vs_b_on_test"]["corr"],
    "A versus B: mean ratio B / A": p6["a_vs_b_on_test"]["mean_ratio"],
    "A versus B: mean absolute difference": p6["a_vs_b_on_test"]["mean_abs_diff"],
    "A versus B: largest absolute difference": p6["a_vs_b_on_test"]["max_abs_diff"],
    "profile ratio min / max (per day type and hour)": f"{p6['profile_ratio']['min']:.3f} / {p6['profile_ratio']['max']:.3f}",
    "cells outside 0.4-2.5": p6["profile_ratio"]["cells_outside_0_4_2_5"],
    "daily-total ratio min / max": f"{p6['daily_total_ratio']['min']:.3f} / {p6['daily_total_ratio']['max']:.3f}",
    "mean prediction 2011 / 2012": f"{p6['by_year_mean']['0']:.1f} / {p6['by_year_mean']['1']:.1f}",
}
print(pd.Series(summary, name="value").to_string())

# %%
labelled_df, test_df = load_train(CFG), load_test(CFG)
sub = pd.read_csv(predict.SUBMISSION_PATH)
from src.plots import show  # displays a figure as a PNG in the notebook
show(plot_test_profile(labelled_df, test_df, sub))

# %%
# The same checks as tools/submission_check.py, inline (hard failures are asserts).
assert list(sub.columns) == ["instant", "cnt"], list(sub.columns)
assert len(sub) == len(test_df), (len(sub), len(test_df))
assert (sub["instant"].to_numpy() == test_df["instant"].to_numpy()).all(), "row order differs from the hidden file"
assert np.isfinite(sub["cnt"].to_numpy(dtype=float)).all(), "NaN or inf in cnt"
assert (sub["cnt"] >= 0).all(), "negative predictions"
assert not (sub["cnt"] == 0).all(), "all predictions are 0"
print("submission:", sub.shape, "| columns", list(sub.columns), "| order equals the hidden file | no NaN | no negatives")
print("cells outside the 0.4-2.5 band of the training profile:", p6["profile_ratio"]["cells_outside_0_4_2_5"],
      "| daily totals vs same-month training days:", round(p6["daily_total_ratio"]["min"], 2), "to",
      round(p6["daily_total_ratio"]["max"], 2))

# %%
# All chain assertions in one table: hash chain p1 -> p6, shared seed, Phase 2 start loss = Phase 1 final loss,
# Phase 5 features among the Phase 4 survivors, model A reproduces Phase 4.
arts = {name: read_artifact(name) for name in ("p1", "p2", "p3", "p4", "p5", "p6")}
checks = []
for prev, name in zip(("p1", "p2", "p3", "p4", "p5"), ("p2", "p3", "p4", "p5", "p6")):
    checks.append((f"{name}.upstream_sha256 == sha256(artifacts/{prev}.json)",
                   arts[name]["upstream_sha256"] == sha256_file(f"artifacts/{prev}.json")))
checks.append(("p1 has no upstream", arts["p1"]["upstream_sha256"] is None))
checks.append(("same seed in p1..p6 and equal to the seed of the config",
               {a["seed"] for a in arts.values()} == {config_seed(CFG)}))
checks.append(("p2 starts from the p1 weights", arts["p2"]["init_weights_source"] == "p1"))
checks.append(("p2.init_loss == p1.train_loss_final (1e-9)",
               abs(arts["p2"]["init_loss"] - arts["p1"]["train_loss_final"]) <= 1e-9 * max(1.0, arts["p1"]["train_loss_final"])))
checks.append(("p5.features are a subset of p4.survivors_expanded",
               set(arts["p5"]["features"]) <= set(arts["p4"]["survivors_expanded"])))
checks.append(("p4.recommended was fitted on the survivors, p6 uses the same columns",
               arts["p4"]["recommended"].get("fitted_on") == "survivors"))
checks.append(("model A reproduces the p4 weights (< 1e-6)", arts["p6"]["model_a_gap_to_p4"] < 1e-6))
checks.append(("p5 chosen model converged", arts["p5"]["stop_reason"] == "converged"))
report = pd.DataFrame(checks, columns=["assertion", "holds"])
print(report.to_string(index=False))
assert report["holds"].all(), "a chain assertion failed"
