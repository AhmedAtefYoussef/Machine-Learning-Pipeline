# %% [markdown]
# ## Phase 6 — Final model and submission
#
# **Which model.** The regression model our pipeline recommends is the Phase 4 stage-B model: {{p6.model.method}} with λ = {{p6.model.lambda:.1e}} and l1_ratio {{p6.model.l1_ratio}} on the surviving features. Nothing is tuned here.
#
# **Refit on training + validation, and why.** The hyper-parameters stay exactly as validated, but the weights are refitted on all {{p6.n_fit_rows}} labelled rows. The Phase 3 learning curve of the target was still rising at full size ({{p3.learning_curves.target.4.val_r2:.3f}} with 80% of the training days, {{p3.learning_curves.target.5.val_r2:.3f}} with all of them), so a quarter more rows should help a little, and the validation rows are days of the same months as the hidden ones. The scaler and the humidity fill stay as fitted on the training portion, so the hidden rows go through exactly the transformations used everywhere else. The cell first refits the validated model on the training rows alone and asserts that it reproduces the Phase 4 weights.
#
# **What we check without labels.** We cannot score the hidden days, and we do not tune anything on them. We only check that the predictions are sane: same rows and order as `test.csv`, no missing or negative values, the refitted model agrees closely with the validated one (correlation {{p6.a_vs_b_on_test.corr:.4f}}), and the predicted hourly profile by day type looks like the training profile (each cell between {{p6.profile_ratio.min:.2f}} and {{p6.profile_ratio.max:.2f}} times the training mean).
#
# **What the model cannot adapt to.** It extrapolates growth as a straight line on the transformed scale, which Phase 3 showed to be somewhat too steep beyond the observed period; it has never seen a zero-demand hour, weather outside the observed range, or holidays other than those in the training days. For the hidden days, which lie inside the two observed years, the held-out-day estimate of Phase 4 ({{p4.recommended.day_block_r2:.3f}}) is our best guess of the score; for a genuinely later period we would expect something nearer the chronological estimate ({{p4.recommended.chrono_r2:.3f}}).

# %%
# Phase 6 refits the recommended Phase 4 model on training plus validation rows and predicts the hidden days.
from src import predict
from src.common import load_test, sha256_file

predict.run(CFG)   # writes sample_submission.csv and artifacts/p6.json; it re-runs no earlier phase
p6 = read_artifact("p6")
ab = p6["a_vs_b_on_test"]
print(f"model {p6['model']['method']}, penalty {p6['model']['lambda']:.3e}, refit on {p6['refit_on']} ({p6['n_fit_rows']} rows); "
      f"models A and B agree on the hidden rows with correlation {ab['corr']:.4f}")

# %%
# The same checks as tools/submission_check.py, inline: any failure stops the notebook.
test_df = load_test(CFG)
sub = pd.read_csv(predict.SUBMISSION_PATH)
assert list(sub.columns) == ["instant", "cnt"], list(sub.columns)
assert len(sub) == len(test_df) and (sub["instant"].to_numpy() == test_df["instant"].to_numpy()).all(), "rows differ from the hidden file"
assert np.isfinite(sub["cnt"].to_numpy(dtype=float)).all() and (sub["cnt"] >= 0).all(), "NaN, inf or negative predictions"
assert not (sub["cnt"] == 0).all(), "all predictions are 0"
print(f"submission passed: {len(sub)} rows, mean {sub['cnt'].mean():.1f}, min {sub['cnt'].min():.1f}, max {sub['cnt'].max():.1f}")

# %%
# The chain, asserted: the hash chain p1 to p6, one seed, and the hand-overs between phases.
chain = [("p1", p1), ("p2", p2), ("p3", p3), ("p4", p4), ("p5", p5), ("p6", p6)]
assert p1["upstream_sha256"] is None
for (before, _), (name, art) in zip(chain, chain[1:]):
    assert art["upstream_sha256"] == sha256_file(f"artifacts/{before}.json"), f"{name} was not built from this {before}"
assert {art["seed"] for _, art in chain} == {SEED}, "the chain uses more than one seed"
assert p2["init_weights_source"] == "p1" and abs(p2["init_loss"] - p1["train_loss_final"]) <= 1e-9
assert set(p5["features"]) <= set(p4["survivors_expanded"]) and p4["recommended"]["fitted_on"] == "survivors"
assert p6["model_a_gap_to_p4"] < 1e-6 and p5["stop_reason"] == "converged"
print("chain assertions passed")

# %%
# The predicted hourly profile of the hidden days against the profile of the training days.
show(plots_p5.plot_test_profile(train_all, test_df, sub))

# %% [markdown]
# ### The pipeline at a glance
#
# One line per phase, built from the five artifact files: what each phase consumed, what it handed on, and the score of the model it ended with. This is the same table as the retrospective in Phase 5, placed here so that the notebook ends with the whole chain in view.
