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

BUNDLE = """
@@PROJECT_BUNDLE@@
"""
DATA_SHA256 = {"train": "@@SHA_TRAIN@@", "test": "@@SHA_TEST@@"}   # the data files our stored results were computed from

with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as bundle:
    members = bundle.namelist()
    missing_members = [m for m in members if not os.path.exists(m)]
    for member in missing_members:
        bundle.extract(member, ".")
print(f"project files: {len(members)} in the bundle, {len(missing_members)} unpacked now, "
      f"{len(members) - len(missing_members)} already present")
del BUNDLE

import yaml

DATA_PATHS = {key: yaml.safe_load(open("config.yaml", encoding="utf-8"))["paths"][key] for key in DATA_SHA256}


def find_data_file(target):
    """An uploaded copy of a data file: next to the notebook, or in Colab's upload folder."""
    for folder in (".", "/content"):
        candidate = os.path.join(folder, os.path.basename(target))
        if os.path.isfile(candidate):
            return candidate
    return None


def place_data_files():
    """Copy uploaded data files to the paths config.yaml expects; return the ones still missing."""
    still_missing = []
    for target in DATA_PATHS.values():
        if not os.path.isfile(target):
            found = find_data_file(target)
            if found is None:
                still_missing.append(target)
            else:
                os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                shutil.copy(found, target)
    return still_missing


needed = place_data_files()
if needed:
    names = [os.path.basename(target) for target in needed]
    try:
        from google.colab import files as colab_files
    except ImportError:
        raise FileNotFoundError(f"put {names} next to this notebook (or in data/) and run this cell again") from None
    print("please choose these files in the upload box:", names)
    for name, content in colab_files.upload().items():
        for target in needed:   # Colab may rename a second upload to 'train (1).csv', so match on the stem
            if name.startswith(os.path.splitext(os.path.basename(target))[0]):
                os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                with open(target, "wb") as handle:
                    handle.write(content)
    needed = place_data_files()
    if needed:
        raise FileNotFoundError(f"still missing: {needed}; upload them and run this cell again")

DATA_MATCHES = True
for key, target in DATA_PATHS.items():
    with open(target, "rb") as handle:
        digest = hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()
    same = digest == DATA_SHA256[key]
    DATA_MATCHES = DATA_MATCHES and same
    print(f"{target}: {'the file our stored results were computed from' if same else 'DIFFERENT from the file our stored results were computed from'}")
if not DATA_MATCHES:
    print("The stored artifacts do not belong to these data files: set RECOMPUTE_ALL = True in the configuration cell.")

# %%
# The configuration and the imports for the whole notebook: every knob lives in CFG, which is loaded from config.yaml.
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from IPython.display import display

from src import nbview as view, plots
from src.common import config_seed, load_config, load_train, seeded_split, team_seed
from src.nbtools import phase, show_source
from src.plots import show  # displays a figure as a PNG in the notebook

CFG = load_config()
RECOMPUTE_ALL = False   # True: recompute every phase from scratch (about 40 minutes) instead of loading artifacts
SEED = config_seed(CFG)
assert team_seed(["34521", "40218", "41190"]) == 41698   # the worked example of the brief
view.p0_config(CFG, SEED)

# %%
# The labelled data and the one seeded 80/20 split that every reported score uses.
train_all = load_train(CFG)
train_df, val_df = seeded_split(train_all, SEED, CFG["split"]["test_size"])
view.p0_split(train_all, train_df, val_df)

# %%
# Mean demand by hour for working and non-working days: the shape every model has to learn.
show(plots.plot_demand_profile(train_df))

# %%
# Oddities of the data that we checked before modelling.
view.p0_quirks(train_all)

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
