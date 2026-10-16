"""run.py staleness: import-closure dependencies and the content-hash manifest (no phase is ever built here)."""
import pytest

import run

PHASES = ["p1", "p2", "p3", "p4", "p5", "p6", "bonus_control"]


@pytest.fixture
def manifest_of_current_tree(monkeypatch):
    """Pretend the manifest was written from the files as they are now."""
    frozen = {name: run.input_hashes(name) for name in PHASES}
    monkeypatch.setattr(run, "read_manifest", lambda: frozen)


def test_closure_of_p5_includes_labels_and_gd():
    deps = run.source_deps("src/phases/p5.py")
    assert "src/labels.py" in deps and "src/gd.py" in deps and "src/phases/p5.py" in deps


def test_closure_follows_phase_to_phase_imports():
    assert "src/phases/p1.py" in run.source_deps("src/phases/p2.py")
    assert "src/phases/p3.py" in run.source_deps("src/phases/p4.py")
    assert "src/regularization.py" in run.source_deps("src/predict.py")


def test_unchanged_tree_is_not_stale(manifest_of_current_tree):
    assert [name for name in PHASES if run.stale(name)] == []


def test_missing_manifest_entry_is_stale(monkeypatch):
    monkeypatch.setattr(run, "read_manifest", dict)
    assert all(run.stale(name) for name in PHASES)


def test_edited_dependency_marks_its_users_stale(manifest_of_current_tree, monkeypatch):
    real = run.content_hash
    monkeypatch.setattr(run, "content_hash", lambda p: "edited" if p == "src/phases/p3.py" else real(p))
    assert [name for name in PHASES if run.stale(name)] == ["p3", "p4", "p6"]
    # p4 imports ladder_specs from p3; p6 reaches it through p4


def test_edited_shared_module_marks_every_stage_stale(manifest_of_current_tree, monkeypatch):
    real = run.content_hash
    monkeypatch.setattr(run, "content_hash", lambda p: "edited" if p == "src/common.py" else real(p))
    assert all(run.stale(name) for name in PHASES)
