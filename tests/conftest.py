"""Shared fixtures. Data is loaded only through src.common (repo root as cwd)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")


@pytest.fixture(scope="session")
def cfg():
    common = pytest.importorskip("src.common")
    return common.load_config()


@pytest.fixture(scope="session")
def seed(cfg):
    common = pytest.importorskip("src.common")
    return common.team_seed(cfg["team_ids"])


@pytest.fixture(scope="session")
def train_all(cfg):
    common = pytest.importorskip("src.common")
    return common.load_train(cfg)


@pytest.fixture(scope="session")
def split(train_all, seed):
    common = pytest.importorskip("src.common")
    return common.seeded_split(train_all, seed)


@pytest.fixture(scope="session")
def train_df(split):
    return split[0]


@pytest.fixture(scope="session")
def val_df(split):
    return split[1]


@pytest.fixture()
def rng():
    return np.random.default_rng(12345)
