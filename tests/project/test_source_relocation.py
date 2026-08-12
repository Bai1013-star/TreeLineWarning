"""P2C-B1 exact-content source relocation tests.

These tests cover the reviewed relocation contract: content identity is
preserved, only the audited source access record changes, historical Tower/Span
bindings and the run index are preserved, counters/domain IDs never change, the
old source may be missing, wrong content fails closed, same-path is a no-op, and
every relocation requires a live writable writer capability.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import core.project as project_package
from core.contracts.frv import frv_digest
from core.contracts.project import (
    SPAN_ANALYSIS_SCHEMA_VERSION,
    AnalysisStage,
    AnalysisStatus,
    SpanAnalysis,
    TowerRevisionReference,
    derive_span_geometry_digest,
)
from core.project import (
    ProjectIntegrityError,
    ProjectSessionClosedError,
    ProjectSessionError,
    ProjectSessionMode,
    SourceAvailability,
    SourceIdentityMismatchError,
    SourceRelocationPublicationUncertainError,
    WriterAuthorityError,
    create_project_writable,
    open_project_read_only,
    open_project_writable,
)
from core.project import persistence
from core.project import session as session_module
from core.project import source_relocation as source_relocation_module
from core.project.locking import ProjectLockAcquisitionError, ProjectWriterLock
from core.project.persistence import (
    PersistenceError,
    PostPointerDurabilityError,
    RUN_INDEX_SCHEMA_VERSION,
    RunIndex,
    RootPointerReplaceError,
    file_sha256,
    published_json_bytes,
)
from core.project.session import WritableProjectSession
from core.project.source_relocation import (
    SourceCandidateError,
    SourceCandidateIdentityError,
    relocate_source,
)
from tests.contracts.test_project_contracts import _span, _tower

_CONTENT_FIELDS = (
    "source_identity_schema_version",
    "source_id",
    "source_file_sha256",
    "canonical_point_content_digest",
    "canonical_point_content_spec_digest",
    "canonical_point_content_spec_version",
    "source_point_count",
    "source_bounds_min_xyz",
    "source_bounds_max_xyz",
    "coordinate_frame",
    "units",
    "origin_context",
)


def _make_candidate(tmp_path: Path, source_path: Path, name: str = "scene.las") -> Path:
    candidate = tmp_path / "external" / name
    candidate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, candidate)
    return candidate


def _build_topo(root: Path, project, metadata) -> None:
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        binding = session.project.source_reference.binding
        session.create_tower(_tower("T001", binding, 0.0))
        session.create_tower(_tower("T002", binding, 10.0))
        session.create_span(
            _span(binding, from_tower_id="T001", to_tower_id="T002")
        )


def _run_entry(binding) -> dict:
    return {
        "run_id": "run-001",
        "span_id": "S001",
        "status": "COMPLETE",
        "artifact_root_reference": None,
        "source_content_identity_digest": binding.source_content_identity_digest,
        "source_identity_record_digest": binding.source_identity_record_digest,
        "span_dependencies": {
            "source_binding": binding.to_dict(),
            "from_tower": {"tower_id": "T001", "revision": 1},
            "to_tower": {"tower_id": "T002", "revision": 1},
            "geometry_digest": "a" * 64,
            "configuration_digest": "a" * 64,
        },
        "created_at": "2026-08-12T00:00:00Z",
        "completed_at": "2026-08-12T00:10:00Z",
    }


def _seed_run_entry(root: Path, project) -> None:
    loaded = persistence.open_project(root)
    run_index = RunIndex(
        RUN_INDEX_SCHEMA_VERSION,
        loaded.generation_id,
        project.project_id,
        entries=(_run_entry(project.source_reference.binding),),
    )
    persistence._publish_generation(
        root,
        loaded.project,
        source_sidecar=loaded.source_sidecar,
        tower_registry=loaded.tower_registry,
        span_registry=loaded.span_registry,
        run_index=run_index,
    )


def _complete_span_analysis(binding, span) -> SpanAnalysis:
    return SpanAnalysis(
        schema_version=SPAN_ANALYSIS_SCHEMA_VERSION,
        state=AnalysisStatus.COMPLETE,
        active_run_id="run-001",
        stage_statuses={stage: AnalysisStatus.COMPLETE for stage in AnalysisStage},
        run_dependencies=span.current_dependencies,
        historical_run_ids=("run-001",),
    )


def test_exact_copy_relocation_succeeds_and_preserves_identity(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        session.require_strong_source_identity()
        assert session.source_availability is SourceAvailability.SOURCE_VERIFIED
        content_before = (
            session.project.source_reference.source_identity.source_content_identity_digest
        )
        record_before = (
            session.project.source_reference.source_identity.source_identity_record_digest
        )
        project_id_before = session.project.project_id
        counters_before = (
            session.project.next_tower_sequence,
            session.project.next_span_sequence,
        )
        candidate = _make_candidate(tmp_path, source_path)
        result = session.relocate_source(str(candidate))
        assert result.performed is True
        assert session.project.source_reference.source_identity.resolved_access_path == str(
            candidate
        )
        assert (
            session.project.source_reference.source_identity.source_content_identity_digest
            == content_before
        )
        assert (
            session.project.source_reference.source_identity.source_identity_record_digest
            != record_before
        )
        assert session.project.project_id == project_id_before
        assert (
            session.project.next_tower_sequence,
            session.project.next_span_sequence,
        ) == counters_before
    # The original source file is never deleted/renamed/moved by relocation.
    assert source_path.is_file()


def test_access_record_digest_follows_the_frozen_frv_contract(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate = _make_candidate(tmp_path, source_path)
        session.relocate_source(str(candidate))
        new_identity = session.project.source_reference.source_identity
        content_before = new_identity.source_content_identity_digest
        # Content identity must be immutable across the path-only move.
        assert new_identity.to_content_digest_dict()["source_file_sha256"] == (
            project.source_reference.source_identity.source_file_sha256
        )
        record = {
            **new_identity.to_content_digest_dict(),
            "recorded_source_path": new_identity.recorded_source_path,
            "resolved_access_path": new_identity.resolved_access_path,
            "relocation_audit_audit_digest": new_identity.relocation_audit.audit_digest,
            "source_equivalence_audit_audit_digest": (
                new_identity.source_equivalence_audit.audit_digest
                if new_identity.source_equivalence_audit is not None
                else None
            ),
            "source_content_identity_digest": content_before,
        }
        assert new_identity.source_identity_record_digest == frv_digest(record)


def test_relocation_succeeds_when_old_source_is_missing(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata):
        pass
    candidate = _make_candidate(tmp_path, source_path)
    source_path.unlink()

    with open_project_writable(root) as session:
        assert session.source_availability is SourceAvailability.SOURCE_MISSING
        content_before = (
            session.project.source_reference.source_identity.source_content_identity_digest
        )
        result = session.relocate_source(str(candidate))
        assert result.performed is True
        assert (
            session.project.source_reference.source_identity.source_content_identity_digest
            == content_before
        )
        assert session.project.source_reference.source_identity.resolved_access_path == str(
            candidate
        )


def test_wrong_content_candidate_is_rejected_fail_closed(
    tmp_path, tiny_las_project, rewrite_tiny_las
):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate = _make_candidate(tmp_path, source_path)
        rewrite_tiny_las(candidate)
        generation_before = session.generation_id
        path_before = session.project.source_reference.source_identity.resolved_access_path
        content_before = (
            session.project.source_reference.source_identity.source_content_identity_digest
        )
        counter_before = session.project.next_tower_sequence
        with pytest.raises(SourceCandidateIdentityError):
            session.relocate_source(str(candidate))
        assert session.generation_id == generation_before
        assert (
            session.project.source_reference.source_identity.resolved_access_path
            == path_before
        )
        assert (
            session.project.source_reference.source_identity.source_content_identity_digest
            == content_before
        )
        assert session.project.next_tower_sequence == counter_before
        assert session.tower_registry.records == ()


def test_invalid_candidate_paths_are_rejected(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        with pytest.raises(SourceCandidateError):
            session.relocate_source(str(tmp_path / "missing.las"))
        with pytest.raises(SourceCandidateError):
            session.relocate_source(str(tmp_path))
        with pytest.raises(SourceCandidateError):
            session.relocate_source("relative/scene.las")
        with pytest.raises(SourceCandidateError):
            session.relocate_source(str(tmp_path / "bad" / "notes.txt"))


def test_same_path_relocation_is_a_noop(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        generation_before = session.generation_id
        record_before = (
            session.project.source_reference.source_identity.source_identity_record_digest
        )
        path_before = session.project.source_reference.source_identity.resolved_access_path
        result = session.relocate_source(str(source_path))
        assert result.performed is False
        assert result.generation_id == generation_before
        assert session.generation_id == generation_before
        assert (
            session.project.source_reference.source_identity.source_identity_record_digest
            == record_before
        )
        assert session.project.source_reference.source_identity.resolved_access_path == path_before


def test_historical_tower_span_and_run_index_preserved(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    _build_topo(root, project, metadata)
    _seed_run_entry(root, project)
    loaded = persistence.open_project(root)
    tower_records_before = loaded.tower_registry.to_dict()["records"]
    span_records_before = loaded.span_registry.to_dict()["records"]
    run_entries_before = loaded.run_index.to_dict()["entries"]
    candidate = _make_candidate(tmp_path, source_path)

    with open_project_writable(root) as session:
        session.relocate_source(str(candidate))

    with open_project_read_only(root) as reopened:
        assert reopened.tower_registry.to_dict()["records"] == tower_records_before
        assert reopened.span_registry.to_dict()["records"] == span_records_before
        assert reopened.run_index.to_dict()["entries"] == run_entries_before
        assert reopened.project.project_id == project.project_id


def test_counters_and_reservations_preserved(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        binding = session.project.source_reference.binding
        session.create_tower(_tower("T001", binding, 0.0))
        reserved = session.reserve_tower_id()
        assert reserved == "T002"
        next_tower_before = session.project.next_tower_sequence
        candidate = _make_candidate(tmp_path, source_path)
        session.relocate_source(str(candidate))
        assert session.project.next_tower_sequence == next_tower_before
        # T001 was confirmed (and therefore reserved) before T002 was reserved.
        assert [item["display_id"] for item in session.tower_registry.reserved_sequences] == [
            "T001",
            "T002",
        ]
        # The durable reservation remains usable with the current post-relocation binding.
        new_binding = session.project.source_reference.binding
        created = session.create_tower(_tower("T999", new_binding, 10.0), reservation=reserved)
        assert created.tower_id == "T002"


def test_relocation_does_not_mark_existing_analysis_stale(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        binding = session.project.source_reference.binding
        session.create_tower(_tower("T001", binding, 0.0))
        session.create_tower(_tower("T002", binding, 10.0))
        span = _span(binding, from_tower_id="T001", to_tower_id="T002")
        session.create_span(replace(span, analysis=_complete_span_analysis(binding, span)))
        candidate = _make_candidate(tmp_path, source_path)
        session.relocate_source(str(candidate))
        relocated = session.span_registry.records[0]
        assert relocated.analysis.state is AnalysisStatus.COMPLETE
        assert relocated.analysis.state is not AnalysisStatus.STALE


def test_read_only_session_relocation_rejected(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata):
        pass
    generation_before = persistence.open_project(root).generation_id
    candidate = _make_candidate(tmp_path, source_path)
    with open_project_read_only(root) as readonly:
        assert not hasattr(readonly, "relocate_source")
        with pytest.raises(WriterAuthorityError):
            relocate_source(readonly, str(candidate))
    assert persistence.open_project(root).generation_id == generation_before


def test_closed_writable_session_relocation_rejected(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    writer = create_project_writable(root, project, source_access_metadata=metadata)
    writer.close()
    candidate = _make_candidate(tmp_path, source_path)
    with pytest.raises(ProjectSessionClosedError):
        writer.relocate_source(str(candidate))
    assert persistence.open_project(root).generation_id is not None


def test_fabricated_writer_lock_and_session_cannot_forge_authority(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata):
        pass
    loaded = persistence.open_project(root)
    with pytest.raises(ProjectLockAcquisitionError):
        ProjectWriterLock(
            project_root=root,
            lock_path=root / "project.lock",
            handle=io.BytesIO(),
            metadata=None,
            process_key=str(root),
        )
    with pytest.raises(ValueError):
        WritableProjectSession(
            loaded,
            mode=ProjectSessionMode.WRITABLE,
            writer_lock=object(),
            application_version="test",
        )
    assert persistence.open_project(root).generation_id == loaded.generation_id


def test_released_writer_capability_relocation_rejected(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    session = create_project_writable(root, project, source_access_metadata=metadata)
    generation_before = persistence.open_project(root).generation_id
    lock = session._writer_lock
    assert lock is not None
    lock.close()
    candidate = _make_candidate(tmp_path, source_path)
    with pytest.raises(ProjectSessionError):
        session.relocate_source(str(candidate))
    session.close()
    assert persistence.open_project(root).generation_id == generation_before


def test_wrong_project_writer_capability_relocation_rejected(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    wrong_root = tmp_path / "wrong"
    wrong_project, wrong_metadata, _ = tiny_las_project(wrong_root)
    wrong_session = create_project_writable(
        wrong_root, wrong_project, source_access_metadata=wrong_metadata
    )
    wrong_lock = wrong_session._writer_lock
    assert wrong_lock is not None
    right_session = create_project_writable(root, project, source_access_metadata=metadata)
    right_lock = right_session._writer_lock
    assert right_lock is not None
    generation_before = right_session.generation_id
    right_session._writer_lock = wrong_lock
    candidate = _make_candidate(tmp_path, source_path)
    try:
        with pytest.raises(ProjectSessionError):
            right_session.relocate_source(str(candidate))
    finally:
        right_session._writer_lock = right_lock
        right_session.close()
        wrong_session.close()
    assert persistence.open_project(root).generation_id == generation_before


def test_successful_relocation_advances_generation_by_one(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        before = session.generation_id
        before_sequence = int(before.split("-")[1])
        candidate = _make_candidate(tmp_path, source_path)
        result = session.relocate_source(str(candidate))
        assert result.generation_id == session.generation_id
        assert int(session.generation_id.split("-")[1]) == before_sequence + 1


def test_session_id_and_writer_lock_preserved_after_relocation(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        session_id_before = session.session_id
        lock_before = session._writer_lock
        candidate = _make_candidate(tmp_path, source_path)
        session.relocate_source(str(candidate))
        assert session.session_id == session_id_before
        assert session._writer_lock is lock_before
        assert not session.closed
        assert lock_before is not None
        lock_before.assert_live_writer_authority(root)


def test_reopen_sees_new_path_and_starts_unverified(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        project_id_before = session.project.project_id
        candidate = _make_candidate(tmp_path, source_path)
        session.relocate_source(str(candidate))
        session.close()
    with open_project_read_only(root) as reopened:
        assert reopened.project.source_reference.source_identity.resolved_access_path == str(
            candidate
        )
        assert reopened.project.project_id == project_id_before
        assert reopened.source_availability is SourceAvailability.SOURCE_PRESENT_UNVERIFIED


def test_candidate_modification_invalidates_session_verification(
    tmp_path, tiny_las_project, rewrite_tiny_las
):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate = _make_candidate(tmp_path, source_path)
        session.relocate_source(str(candidate))
        session.require_strong_source_identity()
        assert session.source_availability is SourceAvailability.SOURCE_VERIFIED
        rewrite_tiny_las(candidate)
        with pytest.raises(SourceIdentityMismatchError):
            session.require_strong_source_identity()
        assert session.source_availability is SourceAvailability.SOURCE_IDENTITY_MISMATCH


def test_pre_pointer_publication_failure_is_safe(tmp_path, tiny_las_project, monkeypatch):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        generation_before = session.generation_id
        path_before = session.project.source_reference.source_identity.resolved_access_path

        def fail_before_pointer(*args, **kwargs):
            raise RootPointerReplaceError(
                "injected pre-pointer failure",
                operation="root-pointer-replace",
                path=root / "project.json",
            )

        monkeypatch.setattr(persistence, "_publish_generation", fail_before_pointer)
        candidate = _make_candidate(tmp_path, source_path)
        with pytest.raises(RootPointerReplaceError):
            session.relocate_source(str(candidate))
        assert session.generation_id == generation_before
        assert (
            session.project.source_reference.source_identity.resolved_access_path == path_before
        )
        assert session.tower_registry.records == ()


def test_post_pointer_publication_uncertainty_is_explicit(tmp_path, tiny_las_project, monkeypatch):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        generation_before = session.generation_id

        def fail_after_pointer(*args, **kwargs):
            raise PostPointerDurabilityError(
                "injected post-pointer failure",
                operation="root-directory-fsync-after-pointer-replace",
                path=root,
                pointer_replaced=True,
            )

        monkeypatch.setattr(persistence, "_publish_generation", fail_after_pointer)
        candidate = _make_candidate(tmp_path, source_path)
        with pytest.raises(SourceRelocationPublicationUncertainError) as excinfo:
            session.relocate_source(str(candidate))
        assert excinfo.value.pointer_replaced is True
        assert session.generation_id == generation_before


def test_post_pointer_root_directory_failure_revalidates_under_writer_lock(
    tmp_path, tiny_las_project, monkeypatch
):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        generation_before = session.generation_id
        original = persistence._fsync_directory

        def fail_root_fsync(path):
            if Path(path) == root:
                raise persistence.DirectoryDurabilityError(
                    "injected post-pointer root directory fsync failure",
                    operation="root-directory-fsync-after-pointer-replace",
                    path=root,
                )
            return original(path)

        monkeypatch.setattr(persistence, "_fsync_directory", fail_root_fsync)
        candidate = _make_candidate(tmp_path, source_path)
        with pytest.raises(SourceRelocationPublicationUncertainError) as excinfo:
            session.relocate_source(str(candidate))
        assert excinfo.value.pointer_replaced is True
        # The pointer was replaced and the session revalidated the new
        # authoritative state while still holding the same writer lock.
        assert session.generation_id != generation_before
        assert session.project.source_reference.source_identity.resolved_access_path == str(
            candidate
        )


def test_relocation_does_not_mutate_candidate_source(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    candidate = _make_candidate(tmp_path, source_path)
    before_bytes = candidate.read_bytes()
    before_size = candidate.stat().st_size
    before_sha256 = hashlib.sha256(before_bytes).hexdigest()
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        session.relocate_source(str(candidate))
    assert candidate.read_bytes() == before_bytes
    assert candidate.stat().st_size == before_size
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == before_sha256


def test_no_public_run_index_mutation_apis_added():
    forbidden = [
        "create_run",
        "update_run",
        "complete_run",
        "fail_run",
        "mark_run_stale",
        "attach_artifact",
        "remove_artifact",
    ]
    for name in forbidden:
        assert name not in project_package.__all__
        assert not hasattr(project_package, name)
        assert not hasattr(session_module.WritableProjectSession, name)
    from core.project import registry as registry_module

    for name in forbidden:
        assert not hasattr(registry_module, name)


def test_no_p3_or_span_run_adapter_surface_added():
    for name in ["ProjectSpanRunAdapter", "PotreeConverter", "COPC", "PredictionExecution"]:
        assert not hasattr(project_package, name)


def test_second_relocation_to_wrong_content_keeps_current_generation(
    tmp_path, tiny_las_project, rewrite_tiny_las
):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_b = _make_candidate(tmp_path, source_path, name="scene_b.las")
        session.relocate_source(str(candidate_b))
        relocated_generation = session.generation_id
        path_after_first = session.project.source_reference.source_identity.resolved_access_path
        candidate_c = _make_candidate(tmp_path, source_path, name="scene_c.las")
        rewrite_tiny_las(candidate_c)
        with pytest.raises(SourceCandidateIdentityError):
            session.relocate_source(str(candidate_c))
        assert session.generation_id == relocated_generation
        assert (
            session.project.source_reference.source_identity.resolved_access_path
            == path_after_first
        )

# ---------------------------------------------------------------------------
# P2C-B1-01 source-history invariants
# ---------------------------------------------------------------------------


def _rewrite_active_file(loaded, relative, mutate):
    """Rewrite one active generation file and refresh manifest + pointer digests."""
    path = loaded.generation_path.joinpath(*relative.split("/"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_bytes(published_json_bytes(payload))
    manifest_path = loaded.generation_path / "generation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["files"] if item["relative_path"] == relative)
    data = path.read_bytes()
    record["byte_size"] = len(data)
    record["sha256"] = file_sha256(data)
    manifest_path.write_bytes(published_json_bytes(manifest))
    pointer_path = loaded.project_root / "project.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["generation_manifest_sha256"] = file_sha256(manifest_path.read_bytes())
    pointer_path.write_bytes(published_json_bytes(pointer))


def _valid_same_content_history_entry(identity, new_path):
    """Build a self-consistent SourceIdentity dict with the same content, new path."""
    entry = dict(identity)
    record = {
        **{key: entry[key] for key in _CONTENT_FIELDS},
        "recorded_source_path": new_path,
        "resolved_access_path": new_path,
        "relocation_audit_audit_digest": entry["relocation_audit"]["audit_digest"],
        "source_equivalence_audit_audit_digest": (
            entry["source_equivalence_audit"]["audit_digest"]
            if entry["source_equivalence_audit"] is not None
            else None
        ),
        "source_content_identity_digest": entry["source_content_identity_digest"],
    }
    entry["recorded_source_path"] = new_path
    entry["resolved_access_path"] = new_path
    entry["source_identity_record_digest"] = frv_digest(record)
    return entry


def _different_content_history_entry(identity):
    """Build a self-consistent SourceIdentity dict describing DIFFERENT content."""
    entry = dict(identity)
    content = {key: entry[key] for key in _CONTENT_FIELDS}
    content["source_file_sha256"] = "f" * 64
    content_digest = frv_digest(content)
    record = {
        **content,
        "recorded_source_path": entry["recorded_source_path"],
        "resolved_access_path": entry["resolved_access_path"],
        "relocation_audit_audit_digest": entry["relocation_audit"]["audit_digest"],
        "source_equivalence_audit_audit_digest": (
            entry["source_equivalence_audit"]["audit_digest"]
            if entry["source_equivalence_audit"] is not None
            else None
        ),
        "source_content_identity_digest": content_digest,
    }
    entry["source_file_sha256"] = "f" * 64
    entry["source_content_identity_digest"] = content_digest
    entry["source_identity_record_digest"] = frv_digest(record)
    return entry


def _corrupt_span_record(span, *, origin=None, record_digest=None):
    """Rebuild a SpanContract with a corrupted context, recomputing its geometry digest."""
    bad_context = span.source_binding.coordinate_context
    if origin is not None:
        bad_context = replace(bad_context, origin_xyz=origin)
    if record_digest is not None:
        bad_context = replace(bad_context, source_identity_record_digest=record_digest)
        bad_binding = replace(
            span.source_binding,
            coordinate_context=bad_context,
            source_identity_record_digest=record_digest,
        )
    else:
        bad_binding = replace(span.source_binding, coordinate_context=bad_context)
    bad_geometry = replace(span.geometry, coordinate_context=bad_context)
    geometry_digest = derive_span_geometry_digest(
        source_binding=bad_binding,
        from_tower=span.from_tower_reference,
        to_tower=span.to_tower_reference,
        geometry=bad_geometry,
    )
    return replace(
        span,
        source_binding=bad_binding,
        geometry=replace(bad_geometry, geometry_digest=geometry_digest),
    )


def _create_project_with_span(root, project, metadata) -> None:
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        binding = session.project.source_reference.binding
        session.create_tower(_tower("T001", binding, 0.0))
        session.create_tower(_tower("T002", binding, 10.0))
        session.create_span(_span(binding, from_tower_id="T001", to_tower_id="T002"))


def _source_history(payload) -> list:
    return payload["source_identity_history"]


def test_history_different_content_rejected(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata):
        pass
    loaded = persistence.open_project(root)

    def mutate(payload):
        identity = payload["project_source_reference"]["source_identity"]
        _source_history(payload).append(_different_content_history_entry(identity))

    _rewrite_active_file(loaded, "source/source-reference.json", mutate)
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


def test_history_duplicate_record_digest_rejected(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata):
        pass
    loaded = persistence.open_project(root)

    def mutate(payload):
        identity = payload["project_source_reference"]["source_identity"]
        prior = _valid_same_content_history_entry(identity, "prior/moved.las")
        _source_history(payload).extend([prior, dict(prior)])

    _rewrite_active_file(loaded, "source/source-reference.json", mutate)
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


def test_history_current_record_digest_rejected(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata):
        pass
    loaded = persistence.open_project(root)

    def mutate(payload):
        identity = payload["project_source_reference"]["source_identity"]
        _source_history(payload).append(dict(identity))

    _rewrite_active_file(loaded, "source/source-reference.json", mutate)
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


def test_history_inconsistent_record_digest_rejected(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata):
        pass
    loaded = persistence.open_project(root)

    def mutate(payload):
        identity = dict(payload["project_source_reference"]["source_identity"])
        identity["source_identity_record_digest"] = "a" * 64
        _source_history(payload).append(identity)

    _rewrite_active_file(loaded, "source/source-reference.json", mutate)
    with pytest.raises((PersistenceError, ValueError)):
        persistence.open_project(root)


def test_history_malformed_entry_rejected(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata):
        pass
    loaded = persistence.open_project(root)

    def mutate(payload):
        _source_history(payload).append({"garbage": True})

    _rewrite_active_file(loaded, "source/source-reference.json", mutate)
    with pytest.raises((PersistenceError, ValueError)):
        persistence.open_project(root)


def test_history_valid_same_content_accepted(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    _create_project_with_span(root, project, metadata)
    loaded = persistence.open_project(root)
    identity = loaded.project.source_reference.source_identity.to_dict()
    prior = _valid_same_content_history_entry(identity, "prior/moved.las")

    def mutate(payload):
        _source_history(payload).append(prior)

    _rewrite_active_file(loaded, "source/source-reference.json", mutate)
    with open_project_read_only(root) as reopened:
        assert len(reopened.source_sidecar.source_identity_history) == 1
        assert (
            reopened.source_sidecar.source_identity_history[0].source_content_identity_digest
            == reopened.project.source_reference.source_identity.source_content_identity_digest
        )


def test_tower_binding_unknown_record_digest_rejected(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    _create_project_with_span(root, project, metadata)
    loaded = persistence.open_project(root)
    unknown = "d" * 64

    def mutate(payload):
        revision = payload["records"][0]["revisions"][0]
        revision["source_binding"]["source_identity_record_digest"] = unknown
        revision["source_binding"]["coordinate_context"]["source_identity_record_digest"] = unknown
        revision["coordinate_context"]["source_identity_record_digest"] = unknown

    _rewrite_active_file(loaded, "towers/towers.json", mutate)
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


def test_span_binding_unknown_record_digest_rejected(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    _create_project_with_span(root, project, metadata)
    loaded = persistence.open_project(root)
    span = loaded.span_registry.records[0]
    bad_span = _corrupt_span_record(span, record_digest="e" * 64)
    _rewrite_active_file(
        loaded, "spans/spans.json", lambda payload: payload.update({"records": [bad_span.to_dict()]})
    )
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


def test_a_to_b_to_c_history_is_unique_and_deterministic(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        content_before = (
            session.project.source_reference.source_identity.source_content_identity_digest
        )
        candidate_b = _make_candidate(tmp_path, source_path, name="b.las")
        session.relocate_source(str(candidate_b))
        candidate_c = _make_candidate(tmp_path, source_path, name="c.las")
        session.relocate_source(str(candidate_c))
        current = session.project.source_reference.source_identity
        history = session.source_sidecar.source_identity_history
        assert current.resolved_access_path == str(candidate_c)
        assert current.source_content_identity_digest == content_before
        assert len(history) == 2
        assert all(entry.source_content_identity_digest == content_before for entry in history)
        digests = [entry.source_identity_record_digest for entry in history]
        assert len(set(digests)) == len(digests)
        assert current.source_identity_record_digest not in digests
    with open_project_read_only(root) as reopened:
        assert reopened.project.project_id == project.project_id


def test_a_to_b_to_a_has_no_current_history_digest_duplication(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        content_before = (
            session.project.source_reference.source_identity.source_content_identity_digest
        )
        candidate_b = _make_candidate(tmp_path, source_path, name="b.las")
        session.relocate_source(str(candidate_b))
        # Relocate back to the original A path, which is still present on disk.
        session.relocate_source(str(source_path))
        current = session.project.source_reference.source_identity
        history = session.source_sidecar.source_identity_history
        assert current.source_content_identity_digest == content_before
        digests = [entry.source_identity_record_digest for entry in history]
        assert len(set(digests)) == len(digests)
        assert current.source_identity_record_digest not in digests
    with open_project_read_only(root) as reopened:
        assert reopened.project.project_id == project.project_id


# ---------------------------------------------------------------------------
# P2C-B1-02 span coordinate-context provenance
# ---------------------------------------------------------------------------


def test_span_current_binding_correct_context_accepted(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    _create_project_with_span(root, project, metadata)
    with open_project_read_only(root) as reopened:
        assert len(reopened.span_registry.records) == 1


def test_span_current_binding_wrong_context_rejected(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    _create_project_with_span(root, project, metadata)
    loaded = persistence.open_project(root)
    span = loaded.span_registry.records[0]
    bad_span = _corrupt_span_record(span, origin=(99.0, 98.0, 97.0))
    _rewrite_active_file(
        loaded, "spans/spans.json", lambda payload: payload.update({"records": [bad_span.to_dict()]})
    )
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


def test_span_historical_binding_correct_context_accepted(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    _create_project_with_span(root, project, metadata)
    candidate = _make_candidate(tmp_path, source_path)
    with open_project_writable(root) as session:
        session.relocate_source(str(candidate))
    with open_project_read_only(root) as reopened:
        span = reopened.span_registry.records[0]
        allowed = {
            reopened.project.source_reference.binding.source_identity_record_digest,
            *(item.source_identity_record_digest for item in reopened.source_sidecar.source_identity_history),
        }
        assert span.source_binding.source_identity_record_digest in allowed


def test_span_historical_binding_wrong_context_rejected(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    _create_project_with_span(root, project, metadata)
    candidate = _make_candidate(tmp_path, source_path)
    with open_project_writable(root) as session:
        session.relocate_source(str(candidate))
    loaded = persistence.open_project(root)
    span = loaded.span_registry.records[0]
    bad_span = _corrupt_span_record(span, origin=(99.0, 98.0, 97.0))
    _rewrite_active_file(
        loaded, "spans/spans.json", lambda payload: payload.update({"records": [bad_span.to_dict()]})
    )
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


def test_span_recomputed_geometry_cannot_hide_context_corruption(tmp_path, tiny_las_project):
    # Changing the span origin AND recomputing the geometry digest must still be
    # rejected by provenance validation.
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    _create_project_with_span(root, project, metadata)
    loaded = persistence.open_project(root)
    span = loaded.span_registry.records[0]
    original_digest = span.geometry.geometry_digest
    bad_span = _corrupt_span_record(span, origin=(99.0, 98.0, 97.0))
    assert bad_span.geometry.geometry_digest != original_digest
    _rewrite_active_file(
        loaded, "spans/spans.json", lambda payload: payload.update({"records": [bad_span.to_dict()]})
    )
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


# ---------------------------------------------------------------------------
# P2C-B1-03 TOCTOU strong verification
# ---------------------------------------------------------------------------


def test_platform_file_identity_participates_in_verification(tmp_path):
    first = tmp_path / "first.las"
    first.write_bytes(b"content-a")
    second = tmp_path / "second.las"
    second.write_bytes(b"content-b")
    first_identity = source_relocation_module._platform_file_identity(first.stat())
    second_identity = source_relocation_module._platform_file_identity(second.stat())
    if first_identity is None or second_identity is None:
        pytest.skip("platform provides no inode-based file identity")
    assert first_identity != second_identity


def test_same_size_mtime_path_replacement_after_hash_rejected(
    tmp_path, tiny_las_project, monkeypatch
):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    candidate = _make_candidate(tmp_path, source_path)
    data = bytearray(candidate.read_bytes())
    data[50] ^= 0xFF
    replacement = tmp_path / "external" / "replacement.las"
    replacement.write_bytes(bytes(data))
    original_stat = candidate.stat()
    assert replacement.stat().st_size == original_stat.st_size

    original_stream = source_relocation_module._stream_handle_sha256

    def hash_then_swap(handle):
        result = original_stream(handle)
        handle.close()
        os.replace(replacement, candidate)
        os.utime(candidate, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        return result

    monkeypatch.setattr(source_relocation_module, "_stream_handle_sha256", hash_then_swap)

    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        generation_before = session.generation_id
        path_before = session.project.source_reference.source_identity.resolved_access_path
        history_before = session.source_sidecar.source_identity_history
        with pytest.raises(SourceCandidateError):
            session.relocate_source(str(candidate))
        assert session.generation_id == generation_before
        assert session.project.source_reference.source_identity.resolved_access_path == path_before
        assert session.source_sidecar.source_identity_history == history_before
        assert session.tower_registry.records == ()
        assert session.span_registry.records == ()
        assert session.run_index.entries == ()


# ---------------------------------------------------------------------------
# P2C-B1-03-RR-01 post-verification / pre-publication TOCTOU window
# ---------------------------------------------------------------------------


def _same_size_different_content(candidate: Path, tmp_path: Path, name: str) -> Path:
    data = bytearray(candidate.read_bytes())
    data[50] ^= 0xFF
    replacement = tmp_path / "external" / name
    replacement.write_bytes(bytes(data))
    assert replacement.stat().st_size == candidate.stat().st_size
    return replacement


def test_post_verify_path_replacement_before_publication_rejected(
    tmp_path, tiny_las_project, monkeypatch
):
    """Reproduce the rereview probe: swap the path AFTER strong verification
    returns but BEFORE authoritative publication.  The final pre-publication
    path-binding check must reject with no authoritative side effects.
    """
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    candidate = _make_candidate(tmp_path, source_path)
    original_stat = candidate.stat()
    replacement = _same_size_different_content(candidate, tmp_path, "replacement.las")

    original_sidecar = source_relocation_module._relocated_sidecar

    def sidecar_then_swap(sidecar, project, new_identity, verified):
        # Strong verification has completed and the verified lease is alive.
        # Release the verified handle (enabling a path swap on Windows) and
        # swap the path with a same-size/same-mtime different-content file.
        verified.close()
        os.replace(replacement, candidate)
        os.utime(candidate, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        return original_sidecar(sidecar, project, new_identity, verified)

    monkeypatch.setattr(
        source_relocation_module, "_relocated_sidecar", sidecar_then_swap
    )

    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        generation_before = session.generation_id
        path_before = session.project.source_reference.source_identity.resolved_access_path
        record_before = (
            session.project.source_reference.source_identity.source_identity_record_digest
        )
        history_before = session.source_sidecar.source_identity_history
        counters_before = (
            session.project.next_tower_sequence,
            session.project.next_span_sequence,
        )
        with pytest.raises(SourceCandidateError):
            session.relocate_source(str(candidate))
        assert session.generation_id == generation_before
        assert (
            session.project.source_reference.source_identity.resolved_access_path
            == path_before
        )
        assert (
            session.project.source_reference.source_identity.source_identity_record_digest
            == record_before
        )
        assert session.source_sidecar.source_identity_history == history_before
        assert (
            session.project.next_tower_sequence,
            session.project.next_span_sequence,
        ) == counters_before
        assert session.tower_registry.records == ()
        assert session.span_registry.records == ()
        assert session.run_index.entries == ()


def test_final_path_identity_check_compares_verified_file_to_current_path(
    tmp_path, tiny_las_project, monkeypatch
):
    """The final gate re-identifies the candidate path against the verified file
    identity.  A swap to a same-size/same-mtime file with a different inode must
    fail closed even when the verified handle was released by the attacker.
    """
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    candidate = _make_candidate(tmp_path, source_path)
    original_stat = candidate.stat()
    replacement = _same_size_different_content(candidate, tmp_path, "replacement.las")

    original_sidecar = source_relocation_module._relocated_sidecar

    def swap_hook(sidecar, project, new_identity, verified):
        verified.close()
        os.replace(replacement, candidate)
        os.utime(candidate, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        return original_sidecar(sidecar, project, new_identity, verified)

    monkeypatch.setattr(source_relocation_module, "_relocated_sidecar", swap_hook)

    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        # The swap preserves size and restores mtime, so only the file-identity
        # comparison can reject it.
        with pytest.raises(SourceCandidateError):
            session.relocate_source(str(candidate))
        assert persistence.open_project(root).generation_id == session.generation_id


def test_verified_handle_released_after_each_outcome(
    tmp_path, tiny_las_project, monkeypatch
):
    """The verified handle lease must be deterministically closed on every
    outcome; a leaked handle would block later file replacement on Windows.
    """

    def _assert_releasable(path: Path) -> None:
        path.unlink()
        path.write_bytes(b"fresh")

    # A. successful relocation.
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    candidate = _make_candidate(tmp_path, source_path, name="ok.las")
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        session.relocate_source(str(candidate))
    _assert_releasable(candidate)

    # B. rejected final path-identity check.
    root_b = tmp_path / "project-b"
    project_b, metadata_b, source_b = tiny_las_project(root_b)
    candidate_b = _make_candidate(tmp_path, source_b, name="swap.las")
    original_stat = candidate_b.stat()
    replacement_b = _same_size_different_content(candidate_b, tmp_path, "swap-repl.las")
    original_sidecar = source_relocation_module._relocated_sidecar

    def swap_hook_b(sidecar, project, new_identity, verified):
        verified.close()
        os.replace(replacement_b, candidate_b)
        os.utime(candidate_b, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        return original_sidecar(sidecar, project, new_identity, verified)

    monkeypatch.setattr(source_relocation_module, "_relocated_sidecar", swap_hook_b)
    with create_project_writable(root_b, project_b, source_access_metadata=metadata_b) as session:
        with pytest.raises(SourceCandidateError):
            session.relocate_source(str(candidate_b))
    _assert_releasable(candidate_b)
    monkeypatch.setattr(source_relocation_module, "_relocated_sidecar", original_sidecar)

    # C. pre-pointer publication failure.  The project is created first so the
    # injected publication failure only affects relocation, not project creation.
    original_publish = persistence._publish_generation
    root_c = tmp_path / "project-c"
    project_c, metadata_c, source_c = tiny_las_project(root_c)
    candidate_c = _make_candidate(tmp_path, source_c, name="prepointer.las")
    with create_project_writable(root_c, project_c, source_access_metadata=metadata_c):
        pass
    monkeypatch.setattr(
        persistence,
        "_publish_generation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RootPointerReplaceError(
                "injected pre-pointer failure",
                operation="root-pointer-replace",
                path=root_c / "project.json",
            )
        ),
    )
    with open_project_writable(root_c) as session:
        with pytest.raises(RootPointerReplaceError):
            session.relocate_source(str(candidate_c))
    _assert_releasable(candidate_c)
    monkeypatch.setattr(persistence, "_publish_generation", original_publish)

    # D. post-pointer publication uncertainty.
    root_d = tmp_path / "project-d"
    project_d, metadata_d, source_d = tiny_las_project(root_d)
    candidate_d = _make_candidate(tmp_path, source_d, name="postpointer.las")
    with create_project_writable(root_d, project_d, source_access_metadata=metadata_d):
        pass
    monkeypatch.setattr(
        persistence,
        "_publish_generation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PostPointerDurabilityError(
                "injected post-pointer failure",
                operation="root-directory-fsync-after-pointer-replace",
                path=root_d,
                pointer_replaced=True,
            )
        ),
    )
    with open_project_writable(root_d) as session:
        with pytest.raises(SourceRelocationPublicationUncertainError):
            session.relocate_source(str(candidate_d))
    _assert_releasable(candidate_d)
