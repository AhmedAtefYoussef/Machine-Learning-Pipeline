# %% [markdown]
# # Rush Hour: predicting hourly bike demand
#
# This notebook runs the whole project from top to bottom. Every number comes from our own modules in `src/` and from the
# JSON files in `artifacts/`; the cells only call them and show the results.
#
# **Running it on Google Colab (or any fresh machine).** Upload this notebook and the two data files of the course,
# `train.csv` and `test.csv`, then choose *Runtime → Run all*. Nothing else is needed: the setup cell below carries our
# own project files (the modules in `src/`, `config.yaml`, the stored `artifacts/` and the empty submission template)
# as text and unpacks them next to the notebook. It downloads nothing and never overwrites a file that already exists.
# If the two data files have not been uploaded yet, the setup cell asks for them. On our own machines, where the project
# folder is already there, the setup cell unpacks nothing.
#
# **Runtime.** About two minutes. Phases 1 to 3 are computed live every time the notebook runs (seconds each) and
# compared with the artifacts of our own run. Phase 4 takes about 30 minutes to compute and Phase 5 about 3, so those
# two cells load the stored artifact when the configuration and the upstream artifact are unchanged, and say so.
# To recompute the whole chain from scratch set `RECOMPUTE_ALL = True` in the configuration cell (about 40 minutes).
# Numbers recomputed on another machine can differ from the text in the last digits, because the markdown cells quote
# the stored run.
#
# **Where the code is.** The cells call our modules in `src/`; the core algorithms are printed where they are used
# (`show_source`), and once the setup cell has run every file can be opened from the file browser.
#
# **How to change a setting.** Every knob lives in `config.yaml`. To try a different value, edit the `CFG` dictionary in
# the configuration cell (for example `CFG["p1"]["lr_fraction_of_bound"] = 0.9`) and re-run the later cells; the phase
# cells pass `CFG` to the phase code. A phase whose configuration changed is recomputed, and so is everything after it,
# so a change in Phase 1 or 2 costs the 30 minutes of Phase 4 and overwrites the stored artifacts and the submission
# file (`git checkout -- artifacts sample_submission.csv` restores them in our repository; on Colab, restart and run the
# setup cell again in a clean session). For a quick demonstration that does not touch the chain, call the functions
# directly in a scratch cell (examples are in `docs/WALKTHROUGH.md`).

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
# **About our expectations.** We looked at the data first, and that look included pilot fits, not only tables. Before writing any Expectation cell we had fitted quick closed-form least-squares models (scripts in `exp/eda0/`) to compare raw and log targets, hour encodings, the working-day × hour block, a first version of the complexity ladder and three candidate label rules for Phase 5. So the Expectation cells are informed predictions, not blind guesses, and some of their numbers are close to the outcomes for that reason. What they could not know, and where they turned out wrong, is said in each Outcome cell. The Expectation cells were committed to git before the corresponding phase code was run and have not been edited since. The same holds, even more strongly, for the second-pass Expectation cells: before writing them we had already fitted the new target on these designs in closed form (`exp/v2/`), so their forecasts are close to the outcomes because we had measured something very similar, not because we foresaw it. What the brief asks of an Expectation, to commit to a prediction and a reason before the phase is run, they do; evidence of foresight they are not.
#
# - **Time structure.** `train.csv` holds days 1-19 of every month of 2011 and 2012; `test.csv` holds the 20th of every month. The hidden test is therefore whole days we have never seen, spread through the same two years. Rows are not independent: hours of one day resemble each other, and every validation row of our seeded split has same-day neighbours in the training portion.
# - **Missing hours.** Some hours are absent, mostly between 2 and 5 at night, and `cnt` is never 0. Quiet hours seem to be missing rather than recorded as zero, so our models never see a zero-demand hour.
# - **Growth.** Mean demand in 2012 is about 1.65 times that of 2011.
# - **Shape.** Working days have two commute peaks (around 8h and 17-18h); other days have one broad midday hump (plot above). An additive model cannot express that.
# - **Copies.** `atemp` follows `temp` (r = 0.985); `season` is the calendar quarter of `mnth`; `workingday` is an exact function of `weekday` and `holiday`; `yr`, `instant` and the date all measure time.
# - **Quirks.** Humidity is 0 on a single day (sensor failure); wind speed is exactly 0 in about 12% of rows and takes only 28 distinct values; weather situation 4 occurs once.
#
# **Splits.** One seeded 80/20 split (the cell above) is used for every reported score and every tuning decision. Phase 3 adds one chronological split. Folds of whole days inside the training portion, fixed by the calendar rather than by a seed, are used as a check on every choice and, in one place, for more than a check: in Phase 4, among methods that are indistinguishable on the validation set, they break the tie (in our run the method they pick is also the best on validation, so the tie-break changed nothing). `test.csv` is used only for the final predictions; nothing is fitted or tuned on it. (Our first look at the data did open it, to see which days it holds.)
