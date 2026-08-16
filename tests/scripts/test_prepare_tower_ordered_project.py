"""V5A3 ordered-project prepare script: safe source-path resolution tests.

The protected LAS filename contains a non-ASCII character that must never be
duplicated in a fragile cmd.exe batch file.  ``resolve_source_path`` resolves
the source safely via Python: an explicit ``--source``, the canonical
source-bound identity path, or a single-file ``input_las/`` scan.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.prepare_tower_ordered_project import resolve_source_path


def test_explicit_source_path_used(tmp_path):
    source = tmp_path / "my_source.las"
    source.write_bytes(b"las")
    resolved = resolve_source_path(source)
    assert resolved is not None
    assert resolved == source.resolve()


def test_missing_explicit_source_returns_none(tmp_path):
    missing = tmp_path / "not-there.las"
    assert resolve_source_path(missing) is None


def test_no_source_and_no_identity_scans_input_las(tmp_path, monkeypatch):
    import scripts.prepare_tower_ordered_project as prep

    fake_input_las = tmp_path / "input_las"
    fake_input_las.mkdir()
    only = fake_input_las / "a.las"
    only.write_bytes(b"las")
    monkeypatch.setattr(prep, "_REPOSITORY_ROOT", tmp_path)
    resolved = resolve_source_path(None, None)
    assert resolved is not None
    assert resolved == only.resolve()


def test_multiple_sources_fail_closed(tmp_path, monkeypatch):
    import scripts.prepare_tower_ordered_project as prep

    fake_input_las = tmp_path / "input_las"
    fake_input_las.mkdir()
    (fake_input_las / "a.las").write_bytes(b"las")
    (fake_input_las / "b.las").write_bytes(b"las")
    monkeypatch.setattr(prep, "_REPOSITORY_ROOT", tmp_path)
    assert resolve_source_path(None, None) is None


def test_canonical_identity_path_used(tmp_path, monkeypatch):
    import scripts.prepare_tower_ordered_project as prep

    fake_input_las = tmp_path / "input_las"
    fake_input_las.mkdir()
    canonical_path = tmp_path / "canonical.las"
    canonical_path.write_bytes(b"las")

    class _FakeIdentity:
        resolved_access_path = str(canonical_path)

    monkeypatch.setattr(prep, "_REPOSITORY_ROOT", tmp_path)
    resolved = resolve_source_path(None, _FakeIdentity())
    assert resolved == canonical_path.resolve()


def test_no_source_nowhere_fails_closed(tmp_path, monkeypatch) -> None:
    """No --source, no canonical identity, and no input_las must fail closed."""
    import scripts.prepare_tower_ordered_project as prep

    monkeypatch.setattr(prep, "_REPOSITORY_ROOT", tmp_path)
    assert resolve_source_path(None, None) is None


def test_prepare_cli_accepts_no_source(tmp_path) -> None:
    """The real prepare CLI accepts an omitted --source (resolves via the project)."""
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    review_root = repo / "cache" / "tower_review_project"
    if not (review_root / "project.json").is_file():
        pytest.skip("canonical review project not present (would trigger a slow source hash)")
    # The CLI runs with no --source; it must resolve the source through the
    # canonical identity (never "source LAS missing" from a required argument).
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "prepare_tower_ordered_project.py"),
            "--project-root",
            str(tmp_path / "ordered_project"),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo),
        timeout=180,
    )
    assert "source LAS missing" not in result.stderr + result.stdout
