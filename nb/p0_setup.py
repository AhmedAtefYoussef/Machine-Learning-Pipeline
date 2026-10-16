# %% [markdown]
# # Rush Hour: predicting hourly bike demand
#
# This notebook runs our whole project from top to bottom. The cells call our modules in `src/`, read and write the JSON files in `artifacts/`, and show the results. Four core algorithms are printed where they are used: gradient descent, the lifting of weights into a larger design, coordinate descent for Lasso, and logistic regression.
#
# **On Colab or a fresh machine.** Upload this notebook together with `train.csv` and `test.csv` and choose *Runtime → Run all*. The setup cell unpacks our project files next to the notebook. It downloads nothing, never overwrites an existing file, and asks for a data file if one is missing.
#
# **Runtime.** About two minutes. Phases 1 to 3 are computed live every time. Phase 4 (about 30 minutes) and Phase 5 (about 3) load their stored artifact when their inputs have not changed, and say so. Setting `RECOMPUTE_ALL = True` in the configuration cell recomputes everything in about 40 minutes.
#
# **Changing a setting.** Every knob is in `CFG`, loaded from `config.yaml`. Change it in the configuration cell and re-run; the changed phase and every later one are recomputed. `docs/WALKTHROUGH.md` has short drills that leave the stored chain alone.

# %%
#@title Setup: unpack the project files (run this cell first)
# The long block of letters below is a compressed copy of src/, config.yaml and artifacts/, nothing else.
import base64
import hashlib
import io
import os
import shutil
import zipfile

import yaml

BUNDLE = """
@@PROJECT_BUNDLE@@
"""
DATA_SHA256 = {"train": "@@SHA_TRAIN@@", "test": "@@SHA_TEST@@"}   # the data files our stored results were computed from

# Unpack the project files that are not here yet; an existing file is never overwritten.
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as bundle:
    new_files = [name for name in bundle.namelist() if not os.path.exists(name)]
    bundle.extractall(".", members=new_files)
print(f"project files: {len(new_files)} unpacked now, {len(bundle.namelist()) - len(new_files)} already present")
del BUNDLE

# Put each data file where config.yaml expects it: copy an upload from next to the notebook or from Colab's /content.
paths = yaml.safe_load(open("config.yaml", encoding="utf-8"))["paths"]
data_paths = {key: paths[key] for key in DATA_SHA256}
for target in data_paths.values():
    for folder in (".", "/content"):
        upload = os.path.join(folder, os.path.basename(target))
        if not os.path.isfile(target) and os.path.isfile(upload):
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            shutil.copy(upload, target)

# Still missing: on Colab ask for the files (a second upload may be renamed 'train (1).csv', so match on the stem).
missing = [target for target in data_paths.values() if not os.path.isfile(target)]
if missing:
    try:
        from google.colab import files
    except ImportError:
        raise FileNotFoundError(f"put {missing} next to this notebook (or in data/) and run this cell again") from None
    for name, content in files.upload().items():
        for target in missing:
            if name.startswith(os.path.splitext(os.path.basename(target))[0]):
                os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                open(target, "wb").write(content)
    assert all(os.path.isfile(target) for target in data_paths.values()), "a data file is still missing; upload both"

# Do the data files match the ones our stored artifacts were computed from? (line endings normalised)
for key, target in data_paths.items():
    digest = hashlib.sha256(open(target, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
    print(target, "matches the stored results" if digest == DATA_SHA256[key] else "DIFFERENT: set RECOMPUTE_ALL = True below")

# %%
# The imports for the whole notebook.
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

from src import plots
from src.common import config_seed, load_config, load_train, read_artifact, seeded_split, team_seed
from src.nbtools import phase, show_source, table
from src.plots import show   # displays a figure as a PNG in the notebook

# %%
# The configuration: every knob lives in CFG (from config.yaml); the seed comes from the team ids.
CFG = load_config()
RECOMPUTE_ALL = False   # True: recompute every phase from scratch (about 40 minutes) instead of loading artifacts
SEED = config_seed(CFG)
assert team_seed(["34521", "40218", "41190"]) == 41698   # the worked example of the brief
table({"team ids": CFG["team_ids"], "seed": SEED, "Phase 1 learning rate (fraction of the stability bound)":
       CFG["p1"]["lr_fraction_of_bound"], "Phase 2 degree candidates": CFG["p2"]["degree_candidates"],
       "plateau tolerance": CFG["p2"]["plateau_tol"]}, "Configuration (config.yaml)")

# %%
# The labelled data and the one seeded 80/20 split that every reported score uses.
train_all = load_train(CFG)
train_df, val_df = seeded_split(train_all, SEED, CFG["split"]["test_size"])
print(f"labelled rows {train_all.shape}, training rows {train_df.shape}, validation rows {val_df.shape}")

# %%
# Mean demand by hour for working and non-working days: the shape every model has to learn.
show(plots.plot_demand_profile(train_df))

# %%
# Oddities of the data that we checked before modelling.
d = train_all
table({"rows with hum == 0": int((d.hum == 0).sum()), "share of rows with windspeed == 0": round(float((d.windspeed == 0).mean()), 4),
       "rows with weathersit == 4": int((d.weathersit == 4).sum()), "correlation of temp and atemp": round(float(d.temp.corr(d.atemp)), 4),
       "rows where workingday != (Mon-Fri and not a holiday)": int((d.workingday != (d.weekday.between(1, 5) & (d.holiday == 0))).sum()),
       "rows where season != calendar quarter of mnth": int((d.season != (d.mnth - 1) // 3 + 1).sum())}, "Quirks of the data")

# %% [markdown]
# ### What we saw in the data before modelling
#
# - **Time structure.** `train.csv` holds days 1 to 19 of every month of 2011 and 2012, and `test.csv` holds day 20 of each month: {{diagnostics.test_coverage.n_test_days}} whole days we never see, inside the same two years. Hours of one day resemble each other, and every validation row of our seeded split shares its day with some training rows.
# - **Shape.** Working days have two commute peaks and other days one midday hump (plot above), so one hour profile cannot fit both.
# - **Growth and skew.** Demand is clearly higher in 2012 than in 2011, it is right-skewed, and its spread grows with its level.
# - **Copies.** `atemp` follows `temp`, `season` is the quarter of `mnth`, `workingday` is an exact function of `weekday` and `holiday`, and `yr`, `instant` and the date all measure time.
# - **Quirks** (table above). Humidity is 0 on one day, wind speed is often exactly 0, weather situation 4 occurs once, and `cnt` is never 0 because quiet night hours are missing rather than recorded as zero.
#
# **Splits.** One seeded 80/20 split is used for every reported score and every tuning choice, and Phase 3 adds one chronological split. Folds of whole days inside the training portion check each choice and, once in Phase 4, break a tie. `test.csv` is used only for the final predictions.
#
# **What our Expectation cells are.** Each one was committed before its phase ran and has not been edited since. Before writing them we had already made pilot fits (`exp/eda0/`, and `exp/v2/` for the second pass), so they are informed forecasts rather than blind guesses. We say this once here; the Outcome cells say where the forecasts were wrong.
