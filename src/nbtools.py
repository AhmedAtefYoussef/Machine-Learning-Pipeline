"""Notebook helpers: show a table, print our code, run a phase or load its stored artifact. They compute no result."""
from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pandas as pd
from IPython.display import display

from src.common import cached_or_run, load_config, read_artifact

CHOICE_FILL = "background-color: #e6f0fb; font-weight: 600"   # the row that records a choice we made
TABLE_STYLES = [
    {"selector": "", "props": "border-collapse: collapse; font-size: 13px; background: #ffffff; color: #0b0b0b;"},
    {"selector": "caption", "props": "caption-side: top; text-align: left; font-weight: 600; padding: 2px 0 6px 0;"},
    {"selector": "th", "props": "background: #f1f0ed; border-bottom: 1px solid #d9d8d4; text-align: left; padding: 5px 10px;"},
    {"selector": "td", "props": "padding: 4px 10px; text-align: left;"},
]


def _text(value):
    """One table entry as text: five significant digits for floats, a dash for missing values, lists joined."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    if isinstance(value, (float, np.floating)):
        return f"{value:.5g}"
    return ", ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)


def table(data, caption=None, highlight=None):
    """Display a DataFrame (or a dict as item | value rows); `highlight` marks the rows we chose (a boolean mask)."""
    frame = pd.DataFrame({"item": list(data), "value": list(data.values())}) if isinstance(data, dict) else data
    styler = frame.reset_index(drop=True).style.hide(axis="index").format(_text).set_table_styles(TABLE_STYLES)
    if caption:
        styler = styler.set_caption(caption)
    if highlight is not None:
        chosen = np.asarray(highlight, dtype=bool)
        fill = lambda f: pd.DataFrame(np.where(chosen[:, None] & np.ones(f.shape, bool), CHOICE_FILL, ""),  # noqa: E731
                                      index=f.index, columns=f.columns)
        styler = styler.apply(fill, axis=None)
    display(styler)


def show_source(*functions):
    """Print the code of our own functions (docstrings and blank lines left out, to keep the output short)."""
    for function in functions:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        body = tree.body[0].body
        first = body[0]
        is_doc = isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
            and isinstance(first.value.value, str)
        lines = inspect.getsource(function).splitlines()
        if is_doc:
            lines = lines[:first.lineno - 1] + lines[first.end_lineno:]
        print("\n".join(line for line in lines if line.strip()))


def max_abs_diff(a, b):
    """Largest absolute difference between the floats of two artifacts, compared key by key and item by item."""
    if isinstance(a, dict):
        return max([max_abs_diff(a[k], b[k]) for k in a if k in b], default=0.0)
    if isinstance(a, (list, tuple)):
        return max([max_abs_diff(x, y) for x, y in zip(a, b)], default=0.0)
    both_floats = isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating))
    return abs(a - b) if both_floats and not (np.isnan(a) and np.isnan(b)) else 0.0


def phase(name, run_fn, cfg, recompute=False, upstream=None, live=False):
    """The artifact of a phase: computed live and checked (Phases 1-3), loaded if still valid, or recomputed."""
    if recompute:
        return run_fn(cfg)
    if live and cfg == load_config():
        fresh, stored = run_fn(cfg, write=False), read_artifact(name)
        gap = max_abs_diff(fresh, stored)
        assert gap < 1e-6, f"{name}: the live run differs from the stored artifact by {gap:.1e}"
        print(f"{name}: computed live just now; largest difference to the stored artifact {gap:.1e}")
        return stored
    return cached_or_run(name, run_fn, cfg, upstream=upstream)
