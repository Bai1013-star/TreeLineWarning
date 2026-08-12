from __future__ import annotations

from pathlib import Path

import pytest

from algorithms.io import headers
from algorithms.io import source_provider
from core.project import (
    SourceAvailability,
    SourceIdentityMismatchError,
    SourceMissingError,
    create_project_writable,
    open_project_read_only,
)
from core.project import session as session_module


def _create_project(root: Path, project, metadata):
    writer = create_project_writable(root, project, source_access_metadata=metadata)
    return writer


def test_missing_source_opens_persisted_metadata_in_degraded_state(
    tmp_path, tiny_project, deterministic_source_metadata
):
    writer = _create_project(tmp_path / "project", tiny_project, deterministic_source_metadata)
    try:
        assert writer.source_availability is SourceAvailability.SOURCE_MISSING
        with pytest.raises(SourceMissingError):
            writer.require_strong_source_identity()
    finally:
        writer.close()


def test_present_compatible_source_is_unverified_until_explicit_strong_gate(
    tmp_path, tiny_las_project
):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    writer = _create_project(root, project, metadata)
    try:
        assert writer.source_availability is SourceAvailability.SOURCE_PRESENT_UNVERIFIED
    finally:
        writer.close()


def test_normal_open_does_not_hash_the_full_source(
    tmp_path, tiny_las_project, monkeypatch
):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    writer = _create_project(root, project, metadata)
    writer.close()

    monkeypatch.setattr(
        session_module,
        "_stream_sha256",
        lambda path: (_ for _ in ()).throw(AssertionError("normal open must not hash source")),
    )
    reader = open_project_read_only(root)
    try:
        assert reader.source_availability is SourceAvailability.SOURCE_PRESENT_UNVERIFIED
    finally:
        reader.close()


def test_fast_proven_file_size_mismatch_is_explicit_degraded_state(
    tmp_path, tiny_las_project
):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    writer = _create_project(root, project, metadata)
    writer.close()
    with source_path.open("ab") as handle:
        handle.write(b"changed")

    reader = open_project_read_only(root)
    try:
        assert reader.source_availability is SourceAvailability.SOURCE_IDENTITY_MISMATCH
        with pytest.raises(SourceIdentityMismatchError):
            reader.require_strong_source_identity()
    finally:
        reader.close()


def test_strong_source_identity_verification_succeeds_without_scientific_provider(
    tmp_path, tiny_las_project, monkeypatch
):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    writer = _create_project(root, project, metadata)
    try:
        monkeypatch.setattr(
            source_provider,
            "load_frv_source",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("P2B must not invoke the scientific source provider")
            ),
        )
        writer.require_strong_source_identity()
        assert writer.source_availability is SourceAvailability.SOURCE_VERIFIED
    finally:
        writer.close()


def test_strong_verification_rejects_a_changed_source(
    tmp_path, tiny_las_project, rewrite_tiny_las
):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    writer = _create_project(root, project, metadata)
    writer.close()
    rewrite_tiny_las(source_path)

    reader = open_project_read_only(root)
    try:
        assert reader.source_availability is SourceAvailability.SOURCE_PRESENT_UNVERIFIED
        with pytest.raises(SourceIdentityMismatchError):
            reader.require_strong_source_identity()
        assert reader.source_availability is SourceAvailability.SOURCE_IDENTITY_MISMATCH
    finally:
        reader.close()


def test_source_signature_change_invalidates_session_only_verification_cache(
    tmp_path, tiny_las_project, rewrite_tiny_las
):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    writer = _create_project(root, project, metadata)
    try:
        writer.require_strong_source_identity()
        assert writer.source_availability is SourceAvailability.SOURCE_VERIFIED
        rewrite_tiny_las(source_path)
        with pytest.raises(SourceIdentityMismatchError):
            writer.require_strong_source_identity()
        assert writer.source_availability is SourceAvailability.SOURCE_IDENTITY_MISMATCH
    finally:
        writer.close()


def test_reopen_starts_unverified_and_verification_cache_is_not_persisted(
    tmp_path, tiny_las_project
):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    writer = _create_project(root, project, metadata)
    generation_files_before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "state").rglob("*")
        if path.is_file()
    }
    try:
        writer.require_strong_source_identity()
        assert writer.source_availability is SourceAvailability.SOURCE_VERIFIED
    finally:
        writer.close()
    generation_files_after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "state").rglob("*")
        if path.is_file()
    }
    assert generation_files_after == generation_files_before

    reopened = open_project_read_only(root)
    try:
        assert reopened.source_availability is SourceAvailability.SOURCE_PRESENT_UNVERIFIED
    finally:
        reopened.close()


def test_fast_check_uses_header_only_access_without_point_materialization(
    tmp_path, tiny_las_project, monkeypatch
):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    monkeypatch.setattr(
        headers.laspy,
        "read",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fast availability must not materialize a point cloud")
        ),
    )
    writer = _create_project(root, project, metadata)
    try:
        assert writer.source_availability is SourceAvailability.SOURCE_PRESENT_UNVERIFIED
    finally:
        writer.close()
