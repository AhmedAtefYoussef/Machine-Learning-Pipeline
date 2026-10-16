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


def test_phases_depend_on_libraries_only():
    # p2 and p4 reach shared helpers through library modules, never through another phase
    assert "src/validation.py" in run.source_deps("src/phases/p2.py")
    assert "src/phases/p1.py" not in run.source_deps("src/phases/p2.py")
    assert "src/phases/p3.py" not in run.source_deps("src/phases/p4.py")
    assert "src/regularization.py" in run.source_deps("src/predict.py")


def test_unchanged_tree_is_not_stale(manifest_of_current_tree):
    assert [name for name in PHASES if run.stale(name)] == []


def test_missing_manifest_entry_is_stale(monkeypatch):
    monkeypatch.setattr(run, "read_manifest", dict)
    assert all(run.stale(name) for name in PHASES)


def test_edited_dependency_marks_its_users_stale(manifest_of_current_tree, monkeypatch):
    real = run.content_hash
    monkeypatch.setattr(run, "content_hash", lambda p: "edited" if p == "src/regularization.py" else real(p))
    # regularization.py is used by p4 and by the submission (p6) only; downstream rebuilds cascade in build_phase
    assert [name for name in PHASES if run.stale(name)] == ["p4", "p6"]


def test_edited_shared_module_marks_every_stage_stale(manifest_of_current_tree, monkeypatch):
    real = run.content_hash
    monkeypatch.setattr(run, "content_hash", lambda p: "edited" if p == "src/common.py" else real(p))
    assert all(run.stale(name) for name in PHASES)
