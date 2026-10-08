"""The notebook's view layer: every table builds from the committed artifacts, `styled` marks the right rows, and the
phase helper loads a stored artifact without running anything."""
import inspect
import json
import os

import pandas as pd
import pytest
from pandas.io.formats.style import Styler

from src import nbtools, nbview
from src.common import load_config, read_artifact

NAMES = ("p1", "p2", "p3", "p4", "p5", "p6")
if not all(os.path.exists(f"artifacts/{n}.json") for n in NAMES):
    pytest.skip("committed artifacts are missing", allow_module_level=True)

ARTS = {n: read_artifact(n) for n in NAMES}
P1, P2, P3, P4, P5, P6 = (ARTS[n] for n in NAMES)
CONFIG = load_config()


def _calls():
    """(function name, arguments) for every artifact-only view function."""
    by_phase = {"p1": (P1,), "p2": (P2,), "p3": (P3,), "p4": (P4,), "p5": (P5,), "p6": (P6,)}
    special = {"p1_live_check": None, "p1_bonus_control": None, "p0_config": None, "p0_split": None, "p0_quirks": None,
               "p6_submission": None, "p6_chain_rows": None}
    extra = {"p2_handover": (P1, P2), "p2_scores": (P1, P2), "p3_chain": (P2, P3), "p4_design": (P3, P4),
             "pipeline_summary": (P1, P2, P3, P4, P5)}
    for name, function in inspect.getmembers(nbview, inspect.isfunction):
        if name in special or name.endswith("_rows") or not (name[0] == "p" and name[1].isdigit() or name == "pipeline_summary"):
            continue
        yield name, function, extra.get(name, by_phase.get(name[:2]))


@pytest.mark.parametrize("name,function,args", list(_calls()), ids=lambda v: v if isinstance(v, str) else "")
def test_view_builds_a_styler(name, function, args):
    table = function(*args)
    assert isinstance(table, (Styler, pd.DataFrame))
    _assert_highlighted_when_captioned(table)


def _assert_highlighted_when_captioned(table):
    """A caption that says a row is highlighted must come with the highlight fill (catches an always-False mask)."""
    html = table.to_html()
    if "highlighted" in html.split("</caption>")[0]:
        assert nbview.CHOICE_FILL in html


def test_stage_views_and_closing_tables():
    for stage in ("full", "final"):
        for view in (nbview.p4_methods, nbview.p4_comparisons, nbview.p4_unregularised):
            table = view(P4, stage)
            assert isinstance(table, Styler)
            _assert_highlighted_when_captioned(table)
    for rows in (nbview.p4_chain_rows(P3, P4), nbview.p5_chain_rows(P4, P5)):
        assert all(passed for *_, passed in rows)
    rows = nbview.p6_chain_rows(ARTS, CONFIG)
    assert all(passed for *_, passed in rows)
    assert isinstance(nbview.checks(rows), Styler)


def test_styled_highlights_exactly_the_requested_rows():
    frame = pd.DataFrame({"name": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
    for choice in ([1], [False, True, False]):
        html = nbview.styled(frame, highlight=choice).to_html()
        assert html.count(nbview.CHOICE_FILL) == 2        # the two cells of row 1, once each
        assert "row1_col0" in html.split(nbview.CHOICE_FILL)[0].rsplit("#T_", 1)[-1]


def test_largest_difference_of_nested_structures():
    same = {"a": [1.0, {"b": 2.0}], "c": "x"}
    assert nbtools.largest_difference(same, {"a": [1.0, {"b": 2.0}], "c": "x"}) == (0.0, 0)
    worst, other = nbtools.largest_difference(same, {"a": [1.0, {"b": 2.5}], "c": "y"})
    assert worst == pytest.approx(0.2) and other == 1


def _live_phase_setup(tmp_path, monkeypatch):
    """A scratch artifacts/ folder with a stored p1 that belongs to the current configuration."""
    (tmp_path / "artifacts").mkdir()
    stored = tmp_path / "artifacts" / "p1.json"
    stored.write_bytes(json.dumps({"config_sha256": "c", "upstream_sha256": None, "x": 1.0}).encode())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(nbtools, "load_config", lambda: CONFIG)
    return stored, stored.read_bytes()


def test_live_phase_puts_the_stored_bytes_back_even_if_the_run_raises(tmp_path, monkeypatch):
    stored, original = _live_phase_setup(tmp_path, monkeypatch)

    def failing_run(cfg):
        stored.write_bytes(b'{"config_sha256": "c", "upstream_sha256": null, "x": 2.0}')
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        nbtools.phase("p1", failing_run, CONFIG, live=True)
    assert stored.read_bytes() == original


def test_live_phase_warns_when_the_live_result_differs(tmp_path, monkeypatch, capsys):
    stored, original = _live_phase_setup(tmp_path, monkeypatch)

    def different_run(cfg):
        stored.write_bytes(b'{"config_sha256": "c", "upstream_sha256": null, "x": 1.5}')

    result = nbtools.phase("p1", different_run, CONFIG, live=True)
    out = capsys.readouterr().out
    assert "WARNING:" in out and "RECOMPUTE_ALL = True" in out
    assert result["x"] == 1.0 and stored.read_bytes() == original


def test_phase_loads_the_cached_artifact_without_running():
    def must_not_run(cfg):
        raise AssertionError("the phase function ran")

    assert nbtools.phase("p1", must_not_run, CONFIG, live=False)["val_r2"] == P1["val_r2"]
