"""The notebook helpers in src/nbtools.py: the table helper marks the right rows, the live check measures differences,
and a stored artifact is loaded without running the phase."""
import contextlib
import io
import os

import pandas as pd
import pytest

from src import nbtools
from src.common import load_config
from src.gd import gradient_descent, mse_grad


def test_table_marks_exactly_the_highlighted_rows(monkeypatch):
    shown = []
    monkeypatch.setattr(nbtools, "display", shown.append)   # capture the Styler instead of rendering it
    nbtools.table(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4, 5, 6]}), "caption", [False, True, False])
    html = shown[0].to_html()
    assert html.count("#e6f0fb") == 1 and "row1_col1" in html and "caption" in html


def test_max_abs_diff_on_equal_and_unequal_artifacts():
    a = {"x": [1.0, {"y": 2.0}], "name": "p1", "n": 3}
    assert nbtools.max_abs_diff(a, a) == 0.0
    assert nbtools.max_abs_diff(a, {"x": [1.0, {"y": 2.5}], "name": "p1", "n": 3}) == pytest.approx(0.5)
    assert nbtools.max_abs_diff({"v": float("nan")}, {"v": float("nan")}) == 0.0


def test_phase_loads_a_valid_stored_artifact_without_running():
    if not os.path.exists("artifacts/p4.json"):
        pytest.skip("committed artifacts are missing")

    def must_not_run(cfg):
        raise AssertionError("the phase was run although its artifact is valid")

    with contextlib.redirect_stdout(io.StringIO()):
        art = nbtools.phase("p4", must_not_run, load_config(), upstream="p3")
    assert "methods" in art


def test_show_source_prints_code_without_docstring():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        nbtools.show_source(gradient_descent, mse_grad)
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("def gradient_descent") and len(lines) <= 30 and '"""' not in out.getvalue()
