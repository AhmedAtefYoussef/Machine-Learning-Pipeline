"""Notebook helpers that are neither tables nor plots: print our code, run a phase or load its artifact."""
from __future__ import annotations

import inspect
import json
import os

import numpy as np

import src.phases.p1 as p1mod
from src.common import cached_or_run, load_config


def show_source(*functions):
    """Print the source code of our own functions, so the algorithm is visible where the notebook uses it."""
    for function in functions:
        print(inspect.getsource(function))


def largest_difference(a, b):
    """Largest relative difference between the numbers of two artifacts, and how many other entries differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return 0.0, 1
        parts = [largest_difference(a[k], b[k]) for k in a]
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return 0.0, 1
        parts = [largest_difference(x, y) for x, y in zip(a, b)]
    elif isinstance(a, float) and isinstance(b, float):
        return abs(a - b) / max(abs(a), abs(b), 1e-12), 0
    else:
        return 0.0, int(a != b)
    return max((p[0] for p in parts), default=0.0), sum(p[1] for p in parts)


def _read_json(path):
    """The artifact at `path`, or None if it cannot be read (for example after an interrupted run)."""
    try:
        with open(path, "rb") as handle:
            return json.loads(handle.read())
    except (OSError, ValueError):
        return None


def _same_inputs(a, b):
    """True when two artifacts were built from the same configuration and the same upstream artifact."""
    return b is not None and a.get("config_sha256") == b.get("config_sha256") and a.get("upstream_sha256") == b.get("upstream_sha256")


def phase(name, run_fn, cfg, upstream=None, live=False, recompute_all=False):
    """Run a phase, or load its stored artifact.

    live=False (the slow Phases 4 and 5): load the stored artifact if it was built from this configuration and
    this upstream artifact, otherwise run the phase.
    live=True (Phases 1 to 3, seconds each): always run the phase here and now, then compare the result with the
    stored artifact. If the stored artifact belongs to this configuration it is put back afterwards, so that the
    later phases, which are tied to it by its hash, stay valid; another machine can differ in the last digits.
    """
    path = os.path.join("artifacts", name + ".json")
    if recompute_all or not live or not os.path.exists(path) or cfg != load_config():
        return run_fn(cfg) if recompute_all else cached_or_run(name, run_fn, cfg, upstream=upstream)
    with open(path, "rb") as handle:
        stored_bytes = handle.read()
    stored = json.loads(stored_bytes)
    try:
        run_fn(cfg)                                       # the live run; it rewrites artifacts/<name>.json
    finally:                                              # also when the run is interrupted or raises
        fresh = _read_json(path)                          # what the live run wrote (None if it was cut off)
        if fresh is None or _same_inputs(stored, fresh):
            with open(path, "wb") as handle:
                handle.write(stored_bytes)
    if not _same_inputs(stored, fresh):
        print(f"{name}: computed live; the stored artifact belonged to another configuration and was replaced")
        return fresh
    worst, other = largest_difference(stored, fresh)
    print(f"{name}: computed live just now; largest relative difference to the stored artifact {worst:.1e}, "
          f"{other} other entries differ; stored file kept so that the later phases stay valid")
    if worst > 1e-6 or other:
        print(f"WARNING: the live result of {name} differs from the stored artifact; the stored one is kept. "
              "Set RECOMPUTE_ALL = True in the configuration cell to rebuild the chain from live results.")
    return stored


def p1_live_run(cfg):
    """Our gradient descent from zero weights with the Phase 1 settings, run again here: (result, learning rate)."""
    split = p1mod.make_split(cfg)
    _, design, _ = p1mod.base_design(split)
    _, _, lr = p1mod.learning_rate(design, cfg["p1"]["lr_fraction_of_bound"])
    knobs = cfg["p1"]
    live = p1mod.run_gd(design, split.z_tr, np.zeros(design.shape[1]), lr, knobs["tol_loss"], knobs["tol_grad"],
                        knobs["max_iter"])
    return live, lr
