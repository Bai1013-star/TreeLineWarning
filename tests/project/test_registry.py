from __future__ import annotations

import io
import json

import pytest

from core.contracts.project import SpanTopologyState, TowerLifecycleState
from core.project import (
    DependentSpanError,
    DuplicateSpanError,
    InvalidRegistryCandidate,
    ProjectSessionError,
    ProjectSessionMode,
    SourceIdentityMismatchError,
    ProjectSessionClosedError,
    WriterAuthorityError,
    create_project_writable,
    open_project_read_only,
    open_project_writable,
)
from core.project import persistence
from core.project.registry import MutationPublicationUncertainError
from core.project.persistence import (
    PersistenceError,
    PostPointerDurabilityError,
    RootPointerReplaceError,
    file_sha256,
    logical_json_digest,
    published_json_bytes,
)
from core.project.locking import ProjectLockAcquisitionError, ProjectWriterLock
from core.project.session import WritableProjectSession
from tests.contracts.test_project_contracts import _span, _tower


def _project_session(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    return root, project, metadata


def _rewrite_active_tower_registry(loaded, mutate):
    path = loaded.generation_path / "towers" / "towers.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_bytes(published_json_bytes(payload))

    manifest_path = loaded.generation_path / "generation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["files"] if item["relative_path"] == "towers/towers.json")
    data = path.read_bytes()
    record["byte_size"] = len(data)
    record["sha256"] = file_sha256(data)
    manifest_path.write_bytes(published_json_bytes(manifest))

    pointer_path = loaded.project_root / "project.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["generation_manifest_sha256"] = file_sha256(manifest_path.read_bytes())
    pointer_path.write_bytes(published_json_bytes(pointer))


def _rewrite_active_project(loaded, mutate):
    path = loaded.generation_path / "project.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_bytes(published_json_bytes(payload))

    manifest_path = loaded.generation_path / "generation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["files"] if item["relative_path"] == "project.json")
    data = path.read_bytes()
    record["byte_size"] = len(data)
    record["sha256"] = file_sha256(data)
    manifest["project_manifest_logical_digest"] = logical_json_digest(payload)
    manifest_path.write_bytes(published_json_bytes(manifest))

    pointer_path = loaded.project_root / "project.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["project"] = payload
    pointer["project_manifest_sha256"] = file_sha256(path.read_bytes())
    pointer["generation_manifest_sha256"] = file_sha256(manifest_path.read_bytes())
    pointer_path.write_bytes(published_json_bytes(pointer))


def test_primary_shared_tower_topology_reopens_and_preserves_revision_refs(
    tmp_path, tiny_las_project
):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        t1 = session.create_tower(_tower("T001", binding, 0.0))
        t2 = session.create_tower(_tower("T002", binding, 10.0))
        s1 = session.create_span(_span(binding, from_tower_id=t1.tower_id, to_tower_id=t2.tower_id))
        t3 = session.create_tower(_tower("T003", binding, 20.0))
        s2 = session.create_span(_span(binding, from_tower_id=t2.tower_id, to_tower_id=t3.tower_id))
        assert (t1.tower_id, t2.tower_id, t3.tower_id) == ("T001", "T002", "T003")
        assert (s1.span_id, s2.span_id) == ("S001", "S002")
        assert session.generation_id != ""

    with open_project_read_only(root) as reopened:
        assert reopened.project.project_id == project.project_id
        assert {entry.tower_id for entry in reopened.tower_registry.records} == {
            "T001",
            "T002",
            "T003",
        }
        assert {span.span_id for span in reopened.span_registry.records} == {"S001", "S002"}
        actual = {span.span_id: (span.from_tower_id, span.to_tower_id) for span in reopened.span_registry.records}
        assert actual == {"S001": ("T001", "T002"), "S002": ("T002", "T003")}
        assert len([entry for entry in reopened.tower_registry.records if entry.tower_id == "T002"]) == 1
        assert reopened.span_registry.records[0].from_tower_reference.__class__.__name__ == "TowerRevisionReference"


def test_tower_revision_history_and_no_reuse_after_archive(tmp_path, tiny_las_project):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        session.create_tower(_tower("T001", binding, 0.0))
        session.create_tower(_tower("T002", binding, 10.0))
        revised = session.revise_tower("T001", _tower("T999", binding, 0.25))
        assert revised.tower_id == "T001"
        assert revised.revision == 2
        entry = session.tower_registry.get("T001")
        assert [item.revision for item in entry.revisions] == [1, 2]
        archived = session.archive_tower("T002")
        assert archived.state is TowerLifecycleState.ARCHIVED
        new_tower = session.create_tower(_tower("T999", binding, 20.0))
        assert new_tower.tower_id == "T003"
        assert session.project.next_tower_sequence == 4


def test_durable_reservations_are_retained_and_invalid_proposal_does_not_advance_counter(
    tmp_path, tiny_las_project
):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        before = session.project.next_tower_sequence
        with pytest.raises(InvalidRegistryCandidate):
            session.create_tower(object())
        assert session.project.next_tower_sequence == before
        reserved = session.reserve_tower_id()
        assert reserved == "T001"
        assert session.tower_registry.reserved_sequences[0]["display_id"] == "T001"
        created = session.create_tower(_tower("T999", binding, 0.0), reservation=reserved)
        assert created.tower_id == "T001"
        session.create_tower(_tower("T999", binding, 10.0))
        assert session.tower_registry.reserved_sequences[0]["display_id"] == "T001"

        span_reserved = session.reserve_span_id()
        assert span_reserved == "S001"
        session.create_span(_span(binding, from_tower_id="T001", to_tower_id="T002"))
        assert session.span_registry.reserved_sequences[0]["display_id"] == "S001"
        assert session.span_registry.records[0].span_id == "S002"
        assert session.project.next_span_sequence == 3


def test_archive_dependency_requires_explicit_cascade_and_cascade_is_one_generation(
    tmp_path, tiny_las_project
):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        session.create_tower(_tower("T001", binding, 0.0))
        session.create_tower(_tower("T002", binding, 10.0))
        session.create_span(_span(binding, from_tower_id="T001", to_tower_id="T002"))
        generation_before = session.generation_id
        with pytest.raises(DependentSpanError):
            session.archive_tower("T002")
        assert session.generation_id == generation_before
        session.archive_tower("T002", cascade_dependent_spans=True)
        assert session.tower_registry.get("T002").current.state is TowerLifecycleState.ARCHIVED
        assert session.span_registry.records[0].topology_state is SpanTopologyState.ARCHIVED


def test_mutation_requires_writable_authority_and_read_only_has_no_mutator_surface(
    tmp_path, tiny_las_project
):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as writer:
        writer.create_tower(_tower("T001", binding, 0.0))
    readonly = open_project_read_only(root)
    try:
        assert not hasattr(readonly, "create_tower")
        with pytest.raises(WriterAuthorityError):
            from core.project.registry import create_tower

            create_tower(readonly, _tower("T999", binding, 1.0))
    finally:
        readonly.close()
    with pytest.raises(ProjectSessionClosedError):
        writer.create_tower(_tower("T999", binding, 2.0))


def test_source_tamper_is_rejected_before_registry_publication(tmp_path, tiny_las_project, rewrite_tiny_las):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        rewrite_tiny_las(root / "source" / "line.las")
        before_generation = session.generation_id
        with pytest.raises(SourceIdentityMismatchError):
            session.create_tower(_tower("T999", binding, 0.0))
        assert session.generation_id == before_generation
        assert session.tower_registry.records == ()


def test_pre_pointer_publication_failure_keeps_prior_generation_and_does_not_commit_id(
    tmp_path, tiny_las_project, monkeypatch
):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        before = session.generation_id

        def fail_before_pointer(*args, **kwargs):
            raise RootPointerReplaceError(
                "injected pre-pointer failure",
                operation="root-pointer-replace",
                path=root / "project.json",
            )

        monkeypatch.setattr(persistence, "_publish_generation", fail_before_pointer)
        with pytest.raises(RootPointerReplaceError):
            session.create_tower(_tower("T999", binding, 0.0))
        assert session.generation_id == before
        assert session.tower_registry.records == ()
        assert session.project.next_tower_sequence == 1


def test_post_pointer_publication_uncertainty_is_explicit_and_revalidated(
    tmp_path, tiny_las_project, monkeypatch
):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        before = session.generation_id

        def fail_after_pointer(*args, **kwargs):
            raise PostPointerDurabilityError(
                "injected post-pointer failure",
                operation="root-directory-fsync-after-pointer-replace",
                path=root,
                pointer_replaced=True,
            )

        monkeypatch.setattr(persistence, "_publish_generation", fail_after_pointer)
        with pytest.raises(MutationPublicationUncertainError):
            session.create_tower(_tower("T999", binding, 0.0))
        assert session.generation_id == before
        assert session.tower_registry.records == ()


def test_direct_lock_and_direct_writable_session_construction_cannot_forge_authority(
    tmp_path, tiny_las_project
):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        before = session.generation_id
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
    with open_project_read_only(root) as reopened:
        assert reopened.generation_id == before
        assert reopened.project.next_tower_sequence == 1
        assert reopened.tower_registry.records == ()


def test_released_and_wrong_project_lock_capabilities_are_rejected(
    tmp_path, tiny_las_project
):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    wrong_root, wrong_project, wrong_metadata = _project_session(tmp_path / "wrong", tiny_las_project)
    session = create_project_writable(root, project, source_access_metadata=metadata)
    original_lock = session._writer_lock
    assert original_lock is not None
    original_lock.close()
    with pytest.raises(ProjectSessionError):
        session.reserve_tower_id()
    session.close()

    wrong_session = create_project_writable(
        wrong_root, wrong_project, source_access_metadata=wrong_metadata
    )
    wrong_lock = wrong_session._writer_lock
    assert wrong_lock is not None
    right_session = open_project_writable(root)
    right_lock = right_session._writer_lock
    assert right_lock is not None
    right_session._writer_lock = wrong_lock
    try:
        with pytest.raises(ProjectSessionError):
            right_session.reserve_tower_id()
    finally:
        right_session._writer_lock = right_lock
        right_session.close()
        wrong_session.close()


@pytest.mark.parametrize(
    "current_revision,revision_numbers",
    [
        (2, [2]),
        (3, [1, 3]),
        (2, [1, 2, 3]),
        (3, [1, 2, 2, 3]),
    ],
)
def test_persisted_noncontiguous_tower_history_fails_closed(
    tmp_path, tiny_las_project, current_revision, revision_numbers
):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        session.create_tower(_tower("T001", binding, 0.0))
    loaded = persistence.open_project(root)

    def mutate(payload):
        entry = payload["records"][0]
        source_revision = entry["revisions"][0]
        revisions = []
        for revision in revision_numbers:
            item = dict(source_revision)
            item["revision"] = revision
            revisions.append(item)
        entry["current_revision"] = current_revision
        entry["revisions"] = revisions

    _rewrite_active_tower_registry(loaded, mutate)
    with pytest.raises(PersistenceError):
        persistence.open_project(root)


def test_valid_three_revision_history_and_old_span_reference_reopen(
    tmp_path, tiny_las_project
):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        session.create_tower(_tower("T001", binding, 0.0))
        session.create_tower(_tower("T002", binding, 10.0))
        span = session.create_span(_span(binding, from_tower_id="T001", to_tower_id="T002"))
        session.revise_tower("T001", _tower("T999", binding, 0.1))
        session.revise_tower("T001", _tower("T999", binding, 0.2))
        entry = session.tower_registry.get("T001")
        assert entry.current_revision == 3
        assert [item.revision for item in entry.revisions] == [1, 2, 3]
        assert span.from_tower_revision == 1
    with open_project_read_only(root) as reopened:
        assert [item.revision for item in reopened.tower_registry.get("T001").revisions] == [1, 2, 3]
        assert reopened.span_registry.records[0].from_tower_revision == 1


def test_archived_span_persists_and_next_span_id_is_not_reused(tmp_path, tiny_las_project):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        session.create_tower(_tower("T001", binding, 0.0))
        session.create_tower(_tower("T002", binding, 10.0))
        first = session.create_span(_span(binding, from_tower_id="T001", to_tower_id="T002"))
        archived = session.archive_span(first.span_id)
        assert archived.topology_state is SpanTopologyState.ARCHIVED
    with open_project_read_only(root) as reopened:
        assert reopened.span_registry.records[0].topology_state is SpanTopologyState.ARCHIVED
    with open_project_writable(root) as session:
        second = session.create_span(_span(binding, from_tower_id="T001", to_tower_id="T002"))
        assert second.span_id == "S002"


def test_span_endpoint_self_and_duplicate_physical_topology_validation(
    tmp_path, tiny_las_project
):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        session.create_tower(_tower("T001", binding, 0.0))
        session.create_tower(_tower("T002", binding, 10.0))
        with pytest.raises(InvalidRegistryCandidate):
            session.create_span(_span(binding, from_tower_id="T999", to_tower_id="T002"))
        with pytest.raises(InvalidRegistryCandidate):
            session.create_span(_span(binding, from_tower_id="T001", to_tower_id="T999"))
        with pytest.raises(ValueError):
            session.create_span(_span(binding, from_tower_id="T001", to_tower_id="T001"))
        session.create_span(_span(binding, from_tower_id="T001", to_tower_id="T002"))
        with pytest.raises(DuplicateSpanError):
            session.create_span(_span(binding, from_tower_id="T001", to_tower_id="T002"))
        with pytest.raises(DuplicateSpanError):
            session.create_span(_span(binding, from_tower_id="T002", to_tower_id="T001"))


def test_archived_endpoint_rejects_new_active_span(tmp_path, tiny_las_project):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        session.create_tower(_tower("T001", binding, 0.0))
        session.create_tower(_tower("T002", binding, 10.0))
        session.archive_tower("T002")
        with pytest.raises(InvalidRegistryCandidate):
            session.create_span(_span(binding, from_tower_id="T001", to_tower_id="T002"))


def test_run_index_entries_survive_unrelated_registry_mutation(tmp_path, tiny_las_project):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        session.create_tower(_tower("T001", binding, 0.0))
        session.create_tower(_tower("T002", binding, 10.0))
        session.create_span(_span(binding, from_tower_id="T001", to_tower_id="T002"))
    with open_project_writable(root) as session:
        session.register_run("run-1", "S001")
        session.start_run("run-1")
        entry = session.complete_run("run-1")
    with open_project_writable(root) as session:
        session.create_tower(_tower("T003", project.source_reference.binding, 20.0))
        assert session.run_index.to_dict()["entries"] == [entry.to_dict()]


def test_mutable_registry_projections_do_not_alias_authoritative_state(
    tmp_path, tiny_las_project
):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        session.create_tower(_tower("T001", binding, 0.0))
        projection = session.tower_registry.to_dict()
        projection["records"][0]["revisions"][0]["display_name"] = "tampered"
        projection["reserved_sequences"].append({"sequence": 999, "display_id": "T999"})
        assert session.tower_registry.get("T001").current.display_name == "T001"
        assert len(session.tower_registry.reserved_sequences) == 1


def test_corrupt_tower_and_span_counters_fail_closed_without_repair(
    tmp_path, tiny_las_project
):
    root, project, metadata = _project_session(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        session.create_tower(_tower("T001", binding, 0.0))
        session.create_tower(_tower("T002", binding, 10.0))
        session.create_span(_span(binding, from_tower_id="T001", to_tower_id="T002"))
    loaded = persistence.open_project(root)
    _rewrite_active_project(loaded, lambda payload: payload.update(next_tower_sequence=1))
    with pytest.raises(PersistenceError):
        persistence.open_project(root)

    # Recreate a clean project and corrupt the Span counter independently.
    root2, project2, metadata2 = _project_session(tmp_path / "span-counter", tiny_las_project)
    binding2 = project2.source_reference.binding
    with create_project_writable(root2, project2, source_access_metadata=metadata2) as session:
        session.create_tower(_tower("T001", binding2, 0.0))
        session.create_tower(_tower("T002", binding2, 10.0))
        session.create_span(_span(binding2, from_tower_id="T001", to_tower_id="T002"))
    loaded2 = persistence.open_project(root2)
    _rewrite_active_project(loaded2, lambda payload: payload.update(next_span_sequence=1))
    with pytest.raises(PersistenceError):
        persistence.open_project(root2)
