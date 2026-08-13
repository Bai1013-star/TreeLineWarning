"""P2C-B2 authoritative run-index mutation and lifecycle tests."""

from __future__ import annotations

import io
import json
import shutil
from dataclasses import replace

import pytest

import core.project as project_package
from core.contracts.project import (
    AnalysisStage,
    AnalysisStatus,
    SPAN_ANALYSIS_SCHEMA_VERSION,
    SpanAnalysis,
    derive_span_geometry_digest,
)
from core.project import (
    DuplicateRunError,
    InvalidRunCandidate,
    InvalidRunTransitionError,
    ProjectIntegrityError,
    ProjectSessionClosedError,
    ProjectSessionError,
    ProjectSessionMode,
    RunIndexPublicationUncertainError,
    SourceAvailability,
    WriterAuthorityError,
    create_project_writable,
    open_project_read_only,
    open_project_writable,
)
from core.project import persistence
from core.project.locking import ProjectLockAcquisitionError, ProjectWriterLock
from core.project.persistence import (
    PostPointerDurabilityError,
    RootPointerReplaceError,
    file_sha256,
    logical_json_digest,
    published_json_bytes,
)
from core.project.run_index import register_run
from core.project.session import WritableProjectSession
from tests.contracts.test_project_contracts import _span, _tower


def _project(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, source_path = tiny_las_project(root)
    return root, project, metadata, source_path


def _topology(session):
    binding = session.project.source_reference.binding
    session.create_tower(_tower("T001", binding, 0.0))
    session.create_tower(_tower("T002", binding, 10.0))
    return session.create_span(_span(binding, from_tower_id="T001", to_tower_id="T002"))


def _rewrite_active_run_index(loaded, mutate) -> None:
    path = loaded.generation_path / "runs" / "run-index.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_bytes(published_json_bytes(payload))

    manifest_path = loaded.generation_path / "generation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["files"] if item["relative_path"] == "runs/run-index.json")
    data = path.read_bytes()
    record["byte_size"] = len(data)
    record["sha256"] = file_sha256(data)
    manifest_path.write_bytes(published_json_bytes(manifest))

    pointer_path = loaded.project_root / "project.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["generation_manifest_sha256"] = file_sha256(manifest_path.read_bytes())
    pointer_path.write_bytes(published_json_bytes(pointer))


def _rewrite_active_span_registry(loaded, mutate) -> None:
    path = loaded.generation_path / "spans" / "spans.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_bytes(published_json_bytes(payload))

    manifest_path = loaded.generation_path / "generation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["files"] if item["relative_path"] == "spans/spans.json")
    data = path.read_bytes()
    record["byte_size"] = len(data)
    record["sha256"] = file_sha256(data)
    manifest_path.write_bytes(published_json_bytes(manifest))

    pointer_path = loaded.project_root / "project.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["generation_manifest_sha256"] = file_sha256(manifest_path.read_bytes())
    pointer_path.write_bytes(published_json_bytes(pointer))


def _complete(session, run_id: str = "run-a", span_id: str = "S001"):
    session.register_run(run_id, span_id)
    session.attach_run_artifact_root(run_id, f"artifacts/runs/{run_id}")
    session.start_run(run_id)
    return session.complete_run(run_id)


def test_primary_run_lifecycle_reopens_and_refreshes_same_writer_session(tmp_path, tiny_las_project):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _topology(session)
        towers_before = session.tower_registry.to_dict()["records"]
        source_before = session.project.source_reference.to_dict()
        counters_before = (session.project.next_tower_sequence, session.project.next_span_sequence)
        generation_before = session.generation_id
        registered = session.register_run("run-a", "S001")
        assert registered.status is AnalysisStatus.READY
        assert session.generation_id != generation_before
        assert session.session_id is not None
        assert persistence.open_project(root).generation_id == session.generation_id
        started = session.start_run("run-a")
        assert started.status is AnalysisStatus.RUNNING
        assert persistence.open_project(root).generation_id == session.generation_id
        attached = session.attach_run_artifact_root("run-a", "artifacts/runs/run-a")
        assert persistence.open_project(root).generation_id == session.generation_id
        completed = session.complete_run("run-a")
        assert completed.status is AnalysisStatus.COMPLETE
        assert completed.completed_at is not None
        assert persistence.open_project(root).generation_id == session.generation_id
        assert attached.artifact_root_reference == "artifacts/runs/run-a"
        assert session.span_registry.records[0].analysis.state is AnalysisStatus.COMPLETE
        assert session.span_registry.records[0].analysis.historical_run_ids == ("run-a",)
        assert session.tower_registry.to_dict()["records"] == towers_before
        assert session.project.source_reference.to_dict() == source_before
        assert (session.project.next_tower_sequence, session.project.next_span_sequence) == counters_before

    with open_project_read_only(root) as reopened:
        entry = reopened.run_index.entries[0]
        assert entry.run_id == "run-a"
        assert entry.span_id == "S001"
        assert entry.status is AnalysisStatus.COMPLETE
        assert entry.artifact_root_reference == "artifacts/runs/run-a"
        assert entry.span_dependencies == reopened.span_registry.records[0].current_dependencies
        assert not hasattr(reopened, "register_run")
        assert not hasattr(reopened, "complete_run")


def test_duplicate_and_invalid_run_lifecycle_transitions_fail_closed(tmp_path, tiny_las_project):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _topology(session)
        session.register_run("run-a", "S001")
        generation_before = session.generation_id
        with pytest.raises(DuplicateRunError):
            session.register_run("run-a", "S001")
        with pytest.raises(InvalidRunTransitionError):
            session.complete_run("run-a")
        with pytest.raises(InvalidRunCandidate):
            session.attach_run_artifact_root("run-a", "../escape")
        assert session.generation_id == generation_before
        session.start_run("run-a")
        session.fail_run("run-a")
        with pytest.raises(InvalidRunTransitionError):
            session.complete_run("run-a")
        session.register_run("run-b", "S001")
        assert tuple(entry.run_id for entry in session.run_index.entries) == ("run-a", "run-b")
        assert session.span_registry.records[0].analysis.historical_run_ids == ("run-a",)


def test_multiple_historical_runs_preserve_their_span_bindings(tmp_path, tiny_las_project):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        first_span = _topology(session)
        binding = session.project.source_reference.binding
        session.create_tower(_tower("T003", binding, 20.0))
        second_span = session.create_span(
            _span(binding, from_tower_id="T002", to_tower_id="T003")
        )
        _complete(session, "run-a")
        second = session.register_run("run-b", second_span.span_id)
        assert second.span_id == "S002"
        assert [(entry.run_id, entry.span_id) for entry in session.run_index.entries] == [
            ("run-a", first_span.span_id),
            ("run-b", second_span.span_id),
        ]
        session.archive_span(second_span.span_id)
        with pytest.raises(InvalidRunCandidate):
            session.register_run("run-c", second_span.span_id)


def test_complete_then_failed_run_history_reopens_with_latest_failed_projection(
    tmp_path, tiny_las_project
):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _topology(session)
        _complete(session, "run-a")
        session.register_run("run-b", "S001")
        session.start_run("run-b")
        failed = session.fail_run("run-b")
        assert failed.status is AnalysisStatus.FAILED
        analysis = session.span_registry.records[0].analysis
        assert analysis.active_run_id == "run-b"
        assert analysis.state is AnalysisStatus.FAILED
        assert analysis.historical_run_ids == ("run-a", "run-b")
        assert persistence.open_project(root).generation_id == session.generation_id

    with open_project_read_only(root) as reopened:
        assert [(entry.run_id, entry.status) for entry in reopened.run_index.entries] == [
            ("run-a", AnalysisStatus.COMPLETE),
            ("run-b", AnalysisStatus.FAILED),
        ]
        analysis = reopened.span_registry.records[0].analysis
        assert analysis.active_run_id == "run-b"
        assert analysis.state is AnalysisStatus.FAILED
        assert analysis.historical_run_ids == ("run-a", "run-b")


def test_no_run_not_started_span_reopens_without_run_provenance(tmp_path, tiny_las_project):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _topology(session)
    with open_project_read_only(root) as reopened:
        analysis = reopened.span_registry.records[0].analysis
        assert analysis.state is AnalysisStatus.NOT_STARTED
        assert analysis.active_run_id is None
        assert analysis.run_dependencies is None
        assert analysis.historical_run_ids == ()
        assert reopened.run_index.entries == ()


@pytest.mark.parametrize(
    "unsafe_path",
    ["../escape", "C:/escape", "runs\\run-a", "runs/con/result", "runs/run-a:stream"],
)
def test_artifact_root_path_safety_and_terminal_immutability(
    tmp_path, tiny_las_project, unsafe_path
):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _topology(session)
        session.register_run("run-a", "S001")
        with pytest.raises(InvalidRunCandidate):
            session.attach_run_artifact_root("run-a", unsafe_path)
        session.attach_run_artifact_root("run-a", "artifacts/runs/run-a")
        with pytest.raises(InvalidRunCandidate):
            session.attach_run_artifact_root("run-a", "artifacts/runs/other")
        session.start_run("run-a")
        session.complete_run("run-a")
        with pytest.raises(InvalidRunTransitionError):
            session.attach_run_artifact_root("run-a", "artifacts/runs/run-a")


def test_run_dependencies_are_authoritatively_derived_and_alias_safe(tmp_path, tiny_las_project):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        span = _topology(session)
        entry = session.register_run("run-a", span.span_id, artifact_root_reference="artifacts/runs/run-a")
        assert entry.span_dependencies == span.current_dependencies
        projection = session.run_index.to_dict()
        projection["entries"][0]["span_dependencies"]["from_tower"]["revision"] = 999
        projection["entries"][0]["artifact_root_reference"] = "tampered"
        persisted = session.run_index.entries[0]
        assert persisted.span_dependencies.from_tower.revision == 1
        assert persisted.artifact_root_reference == "artifacts/runs/run-a"
        with pytest.raises(InvalidRunCandidate):
            session.register_run("run-b", "S999")


def test_closed_read_only_fabricated_released_and_wrong_project_authority_are_rejected(
    tmp_path, tiny_las_project
):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    writer = create_project_writable(root, project, source_access_metadata=metadata)
    _topology(writer)
    writer.close()
    with pytest.raises(ProjectSessionClosedError):
        writer.register_run("run-a", "S001")
    with open_project_read_only(root) as readonly:
        with pytest.raises(WriterAuthorityError):
            register_run(readonly, "run-a", "S001")
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
    with pytest.raises(WriterAuthorityError):
        register_run(object.__new__(WritableProjectSession), "run-a", "S001")

    released = open_project_writable(root)
    released_lock = released._writer_lock
    assert released_lock is not None
    released_lock.close()
    with pytest.raises(ProjectSessionError):
        released.register_run("run-a", "S001")
    released.close()

    wrong_root, wrong_project, wrong_metadata, _ = _project(tmp_path / "wrong", tiny_las_project)
    wrong = create_project_writable(wrong_root, wrong_project, source_access_metadata=wrong_metadata)
    right = open_project_writable(root)
    original_lock = right._writer_lock
    assert original_lock is not None and wrong._writer_lock is not None
    right._writer_lock = wrong._writer_lock
    try:
        with pytest.raises(ProjectSessionError):
            right.register_run("run-a", "S001")
    finally:
        right._writer_lock = original_lock
        right.close()
        wrong.close()


def test_pre_pointer_failure_preserves_run_index_and_session_generation(
    tmp_path, tiny_las_project, monkeypatch
):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _topology(session)
        before = session.generation_id

        def fail_before_pointer(*args, **kwargs):
            raise RootPointerReplaceError(
                "injected pre-pointer failure",
                operation="root-pointer-replace",
                path=root / "project.json",
            )

        monkeypatch.setattr(persistence, "_publish_generation", fail_before_pointer)
        with pytest.raises(RootPointerReplaceError):
            session.register_run("run-a", "S001")
        assert session.generation_id == before
        assert session.run_index.entries == ()


def test_post_pointer_uncertainty_revalidates_before_returning(tmp_path, tiny_las_project, monkeypatch):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _topology(session)
        before = session.generation_id

        def fail_after_pointer(*args, **kwargs):
            raise PostPointerDurabilityError(
                "injected post-pointer failure",
                operation="root-directory-fsync-after-pointer-replace",
                path=root,
                pointer_replaced=True,
            )

        monkeypatch.setattr(persistence, "_publish_generation", fail_after_pointer)
        with pytest.raises(RunIndexPublicationUncertainError):
            session.register_run("run-a", "S001")
        assert session.generation_id == before
        assert session.run_index.entries == ()


def test_exact_content_relocation_does_not_stale_completed_run(tmp_path, tiny_las_project):
    root, project, metadata, source_path = _project(tmp_path, tiny_las_project)
    candidate = tmp_path / "relocated" / "same-content.las"
    candidate.parent.mkdir()
    shutil.copy2(source_path, candidate)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _topology(session)
        completed = _complete(session)
        source_content_before = completed.source_content_identity_digest
        source_record_before = completed.source_identity_record_digest
        session.relocate_source(str(candidate))
        assert session.reconcile_run_staleness() == ()
        entry = session.run_index.entries[0]
        assert entry.status is AnalysisStatus.COMPLETE
        assert entry.source_content_identity_digest == source_content_before
        assert entry.source_identity_record_digest == source_record_before
        assert session.source_availability in {
            SourceAvailability.SOURCE_PRESENT_UNVERIFIED,
            SourceAvailability.SOURCE_VERIFIED,
        }


def test_legitimate_stale_run_projection_reopens_without_rewriting_history(
    tmp_path, tiny_las_project
):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _topology(session)
        completed = _complete(session)
    loaded = persistence.open_project(root)
    old_span = loaded.span_registry.records[0]
    old_dependencies = loaded.run_index.entries[0].span_dependencies
    geometry = replace(old_span.geometry, configuration_digest="b" * 64)
    revised_geometry = replace(
        geometry,
        geometry_digest=derive_span_geometry_digest(
            source_binding=old_span.source_binding,
            from_tower=old_span.from_tower_reference,
            to_tower=old_span.to_tower_reference,
            geometry=geometry,
        ),
    )
    stale_entry = replace(loaded.run_index.entries[0], status=AnalysisStatus.STALE)
    stale_analysis = persistence._span_analysis_projection_for_run(
        stale_entry, historical_run_ids=()
    )
    revised_span = replace(old_span, geometry=revised_geometry, analysis=stale_analysis)
    revised_spans = persistence.SpanRegistry(
        loaded.span_registry.schema_version,
        loaded.span_registry.generation_id,
        loaded.span_registry.project_id,
        reserved_sequences=loaded.span_registry.reserved_sequences,
        records=(revised_span,),
    )
    persistence._publish_generation(
        root,
        loaded.project,
        source_sidecar=loaded.source_sidecar,
        tower_registry=loaded.tower_registry,
        span_registry=revised_spans,
        run_index=persistence.RunIndex(
            loaded.run_index.schema_version,
            loaded.run_index.generation_id,
            loaded.run_index.project_id,
            entries=(stale_entry,),
        ),
    )
    with open_project_read_only(root) as session:
        entry = session.run_index.entries[0]
        assert entry.status is AnalysisStatus.STALE
        assert entry.span_dependencies == old_dependencies
        assert session.span_registry.records[0].analysis.state is AnalysisStatus.STALE


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["entries"].append(dict(payload["entries"][0])),
        lambda payload: payload["entries"][0].update(span_id="S999"),
        lambda payload: payload["entries"][0].update(status="NOT_A_STATUS"),
        lambda payload: payload["entries"][0].update(artifact_root_reference="../escape"),
        lambda payload: payload["entries"][0].update(completed_at=None),
    ],
)
def test_corrupted_run_index_fails_closed(tmp_path, tiny_las_project, mutate):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _topology(session)
        _complete(session)
    loaded = persistence.open_project(root)
    _rewrite_active_run_index(loaded, mutate)
    with pytest.raises(ProjectIntegrityError):
        open_project_read_only(root)


def test_complete_run_failed_span_projection_corruption_fails_closed(tmp_path, tiny_las_project):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _topology(session)
        _complete(session)
    loaded = persistence.open_project(root)

    def contradict_status(payload):
        analysis = payload["records"][0]["analysis"]
        analysis["state"] = AnalysisStatus.FAILED.value
        analysis["stage_statuses"] = {
            stage.value: AnalysisStatus.FAILED.value for stage in AnalysisStage
        }

    _rewrite_active_span_registry(loaded, contradict_status)
    with pytest.raises(ProjectIntegrityError, match="status does not match"):
        open_project_read_only(root)


def test_unknown_active_run_id_corruption_fails_closed(tmp_path, tiny_las_project):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _topology(session)
        _complete(session)
    loaded = persistence.open_project(root)
    _rewrite_active_span_registry(
        loaded,
        lambda payload: payload["records"][0]["analysis"].update(active_run_id="run-missing"),
    )
    with pytest.raises(ProjectIntegrityError, match="does not resolve"):
        open_project_read_only(root)


def test_active_run_from_another_span_corruption_fails_closed(tmp_path, tiny_las_project):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _topology(session)
        binding = session.project.source_reference.binding
        session.create_tower(_tower("T003", binding, 20.0))
        session.create_span(_span(binding, from_tower_id="T002", to_tower_id="T003"))
        _complete(session, "run-a", "S001")
        _complete(session, "run-b", "S002")
    loaded = persistence.open_project(root)

    def bind_s001_to_s002_run(payload):
        s001, s002 = payload["records"]
        analysis = s001["analysis"]
        analysis["state"] = AnalysisStatus.STALE.value
        analysis["stage_statuses"] = {
            stage.value: AnalysisStatus.STALE.value for stage in AnalysisStage
        }
        analysis["active_run_id"] = "run-b"
        analysis["run_dependencies"] = s002["analysis"]["run_dependencies"]
        analysis["historical_run_ids"] = ["run-a", "run-b"]

    _rewrite_active_span_registry(loaded, bind_s001_to_s002_run)
    with pytest.raises(ProjectIntegrityError, match="different Span"):
        open_project_read_only(root)


def test_active_run_dependency_projection_mismatch_fails_closed(tmp_path, tiny_las_project):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _topology(session)
        _complete(session)
    loaded = persistence.open_project(root)
    original_span = loaded.span_registry.records[0]
    original_dependencies = loaded.run_index.entries[0].span_dependencies
    geometry = replace(original_span.geometry, configuration_digest="b" * 64)
    revised_geometry = replace(
        geometry,
        geometry_digest=derive_span_geometry_digest(
            source_binding=original_span.source_binding,
            from_tower=original_span.from_tower_reference,
            to_tower=original_span.to_tower_reference,
            geometry=geometry,
        ),
    )
    provisional = replace(
        original_span,
        geometry=revised_geometry,
        analysis=SpanAnalysis(
            schema_version=SPAN_ANALYSIS_SCHEMA_VERSION,
            state=AnalysisStatus.STALE,
            active_run_id="run-a",
            stage_statuses={stage: AnalysisStatus.STALE for stage in AnalysisStage},
            run_dependencies=original_dependencies,
            historical_run_ids=("run-a",),
        ),
    )
    independently_valid_but_wrong = replace(
        provisional,
        analysis=SpanAnalysis(
            schema_version=SPAN_ANALYSIS_SCHEMA_VERSION,
            state=AnalysisStatus.COMPLETE,
            active_run_id="run-a",
            stage_statuses={stage: AnalysisStatus.COMPLETE for stage in AnalysisStage},
            run_dependencies=provisional.current_dependencies,
            historical_run_ids=("run-a",),
        ),
    )
    _rewrite_active_span_registry(
        loaded,
        lambda payload: payload["records"].__setitem__(0, independently_valid_but_wrong.to_dict()),
    )
    with pytest.raises(ProjectIntegrityError, match="dependencies do not match"):
        open_project_read_only(root)


def test_stale_run_complete_span_projection_corruption_fails_closed(tmp_path, tiny_las_project):
    root, project, metadata, _ = _project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _topology(session)
        _complete(session)
    loaded = persistence.open_project(root)
    old_span = loaded.span_registry.records[0]
    old_entry = loaded.run_index.entries[0]
    geometry = replace(old_span.geometry, configuration_digest="b" * 64)
    revised_geometry = replace(
        geometry,
        geometry_digest=derive_span_geometry_digest(
            source_binding=old_span.source_binding,
            from_tower=old_span.from_tower_reference,
            to_tower=old_span.to_tower_reference,
            geometry=geometry,
        ),
    )
    stale_entry = replace(old_entry, status=AnalysisStatus.STALE)
    stale_span = replace(
        old_span,
        geometry=revised_geometry,
        analysis=persistence._span_analysis_projection_for_run(
            stale_entry, historical_run_ids=()
        ),
    )
    persistence._publish_generation(
        root,
        loaded.project,
        source_sidecar=loaded.source_sidecar,
        tower_registry=loaded.tower_registry,
        span_registry=persistence.SpanRegistry(
            loaded.span_registry.schema_version,
            loaded.span_registry.generation_id,
            loaded.span_registry.project_id,
            reserved_sequences=loaded.span_registry.reserved_sequences,
            records=(stale_span,),
        ),
        run_index=persistence.RunIndex(
            loaded.run_index.schema_version,
            loaded.run_index.generation_id,
            loaded.run_index.project_id,
            entries=(stale_entry,),
        ),
    )
    coherent = persistence.open_project(root)
    _rewrite_active_run_index(
        coherent,
        lambda payload: payload["entries"][0].update(status=AnalysisStatus.COMPLETE.value),
    )
    with pytest.raises(ProjectIntegrityError, match="status does not match"):
        open_project_read_only(root)


def test_public_surface_has_narrow_checked_mutations_only():
    for name in (
        "register_run",
        "start_run",
        "complete_run",
        "fail_run",
        "interrupt_run",
        "attach_run_artifact_root",
        "reconcile_run_staleness",
    ):
        assert name in project_package.__all__
        assert hasattr(WritableProjectSession, name)
    for name in ("update_run", "patch_run", "replace_run_index", "mark_run_stale"):
        assert name not in project_package.__all__
        assert not hasattr(WritableProjectSession, name)
