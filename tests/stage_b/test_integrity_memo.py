from __future__ import annotations

import hashlib

import pytest
from core import integrity


@pytest.fixture(autouse=True)
def _fresh_cache():
    integrity.clear_validation_cache()
    yield
    integrity.clear_validation_cache()


def test_unchanged_artifact_is_hashed_once(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"authoritative-content")
    calls = 0
    original = hashlib.sha256

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(integrity.hashlib, "sha256", counted)
    expected = original(artifact.read_bytes()).hexdigest()

    with integrity.validation_session():
        assert integrity.sha256_file(artifact, reusable=True) == expected
        assert integrity.sha256_file(artifact, reusable=True) == expected
    assert calls == 1


def test_file_state_change_revalidates_at_the_next_session(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"first")
    with integrity.validation_session():
        first = integrity.sha256_file(artifact, reusable=True)
        original_mtime = artifact.stat().st_mtime_ns
    artifact.write_bytes(b"other")
    assert artifact.stat().st_mtime_ns != original_mtime
    with integrity.validation_session():
        second = integrity.sha256_file(artifact, reusable=True)

    assert first != second


def test_change_during_hash_fails_closed(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"stable")
    original_snapshot = integrity._snapshot
    calls = 0

    def changed(path):
        nonlocal calls
        calls += 1
        snapshot = original_snapshot(path)
        if calls == 2:
            return snapshot._replace(ctime_ns=snapshot.ctime_ns + 1)
        return snapshot

    monkeypatch.setattr(integrity, "_snapshot", changed)

    with pytest.raises(
        integrity.ArtifactChangedDuringValidation,
        match="ARTIFACT_CHANGED_DURING_VALIDATION",
    ):
        with integrity.validation_session():
            integrity.sha256_file(artifact, reusable=True)
