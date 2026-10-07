# %% [markdown]
# # Rush Hour: predicting hourly bike demand
#
# This notebook runs the whole project from top to bottom. Every number comes from our own modules in `src/` and from the
# JSON files in `artifacts/`; the cells only call them and show the results.
#
# **Runtime.** Phases 1 and 2 together take well under a minute on a laptop (single thread). The whole notebook is
# budgeted at a few minutes; a cell that would take longer loads a cached artifact and says so.
#
# **How to change a setting.** Every knob lives in `config.yaml`. To try a different value, edit the `CFG` dictionary in the
# next cell (for example `CFG["p1"]["lr_fraction_of_bound"] = 0.9`) and re-run the later cells; the phase cells pass `CFG`
# to the phase code.

# %%
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

from src import plots
from src.common import config_seed, load_config, load_train, seeded_split, team_seed
import src.phases.p1 as p1mod
import src.phases.p2 as p2mod

CFG = load_config()
SEED = config_seed(CFG)
assert team_seed(["34521", "40218", "41190"]) == 41698   # the worked example of the brief
print("team ids  :", CFG["team_ids"])
print("seed      :", SEED, "(sha256 of the sorted ids joined by '_', mod 100000)")
print("self-test : 41698 reproduced from the brief's example ids")
print("knobs     : p1 lr fraction =", CFG["p1"]["lr_fraction_of_bound"],
      "| p2 degrees =", CFG["p2"]["degree_candidates"], "| plateau tol =", CFG["p2"]["plateau_tol"])

# %%
train_all = load_train(CFG)
train_df, val_df = seeded_split(train_all, SEED, CFG["split"]["test_size"])
print("labelled rows :", train_all.shape)
print("train / validation :", train_df.shape, val_df.shape)
print("columns       :", ", ".join(train_all.columns))

# %%
plots.plot_demand_profile(train_df)

# %%
quirks = pd.Series({
    "rows with hum == 0": int((train_all["hum"] == 0).sum()),
    "share of rows with windspeed == 0": round(float((train_all["windspeed"] == 0).mean()), 4),
    "rows with weathersit == 1": int((train_all["weathersit"] == 1).sum()),
    "rows with weathersit == 2": int((train_all["weathersit"] == 2).sum()),
    "rows with weathersit == 3": int((train_all["weathersit"] == 3).sum()),
    "rows with weathersit == 4": int((train_all["weathersit"] == 4).sum()),
    "correlation of temp and atemp": round(float(np.corrcoef(train_all["temp"], train_all["atemp"])[0, 1]), 4),
}, name="value")
print(quirks.to_string())
