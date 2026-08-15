"""V5A1-B tower-candidate persistence tests (candidate store / manifest)."""

from __future__ import annotations

import json
import secrets

import pytest
from core.contracts.project import PROJECT_SCHEMA_VERSION
from core.contracts.tower_candidates import (
    CandidateSetReference,
    RefinementState,
    ReviewDecision,
)
from core.project import (
    GENERATION_SCHEMA_VERSION,
    AlreadyConfirmedError,
    InvalidTowerCandidateState,
    create_project_writable,
    open_project_read_only,
    persistence,
)
from core.project.persistence import (
    FileIntegrityError,
    ProjectIntegrityError,
    file_sha256,
    logical_json_digest,
    published_json_bytes,
)

from tests.contracts.test_project_contracts import TIMESTAMP
from tests.contracts.test_tower_candidate_contracts import (
    _refined,
    _refinement_evidence,
    _set,
)


def _new_project(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    return root, project, metadata


def _persist_set(session, binding, *, candidate_set_id="cset-001", **overrides):
    candidate_set = _set(binding, candidate_set_id=candidate_set_id, **overrides)
    session.persist_tower_candidate_set(candidate_set)
    return candidate_set


def _append_pending_then_accepted(session, candidate_set, candidate_id="P001"):
    binding = candidate_set.source_binding
    session.append_tower_candidate_review(
        candidate_set_reference=candidate_set.reference,
        candidate_id=candidate_id,
        decision=ReviewDecision.PENDING_REVIEW,
        occurred_at=TIMESTAMP,
        reason="initial review",
        refinement_state=RefinementState.UNREFINED,
        operator_identity="reviewer-1",
    )
    accepted = session.append_tower_candidate_review(
        candidate_set_reference=candidate_set.reference,
        candidate_id=candidate_id,
        decision=ReviewDecision.ACCEPTED,
        occurred_at=TIMESTAMP,
        reason="validated against source",
        refinement_state=RefinementState.RAW_REFINED,
        refined_raw_geometry=_refined(),
        refinement_evidence=_refinement_evidence(binding),
        operator_identity="reviewer-1",
    )
    return accepted


def _rewrite_generation_file(loaded, relative_path, mutate):
    """Rewrite one generation JSON file consistently across manifest/index/pointer."""
    path = loaded.generation_path / relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    data = published_json_bytes(payload)
    path.write_bytes(data)

    manifest_path = loaded.generation_path / "generation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["files"] if item["relative_path"] == relative_path)
    record["byte_size"] = len(data)
    record["sha256"] = file_sha256(data)
    manifest_path.write_bytes(published_json_bytes(manifest))

    if relative_path.startswith("candidates/"):
        index_path = loaded.generation_path / "candidates" / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for group in ("candidate_sets", "review_ledgers", "confirmations"):
            for entry in index.get(group, []):
                if entry["relative_path"] == relative_path:
                    entry["byte_size"] = len(data)
                    entry["sha256"] = file_sha256(data)
        index_path.write_bytes(published_json_bytes(index))

    pointer_path = loaded.project_root / "project.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["generation_manifest_sha256"] = file_sha256(manifest_path.read_bytes())
    pointer_path.write_bytes(published_json_bytes(pointer))


def _write_true_v1_generation(
    root,
    project,
    *,
    source_access_metadata,
    generation_counter: int = 1,
):
    """Hand-assemble a genuine historical v1 generation.

    This deliberately never invokes the current V2 publisher.  It constructs the
    exact pre-V5A1-B v1 layout (five core project files + a v1 manifest + a v1
    active pointer) using the version-agnostic persistence primitives.
    """
    generation_id = f"g-{generation_counter:016d}-{secrets.token_hex(4)}"
    generation_path = root / "state" / "generations" / generation_id
    generation_path.mkdir(parents=True)
    (generation_path / "source").mkdir()
    (generation_path / "towers").mkdir()
    (generation_path / "spans").mkdir()
    (generation_path / "runs").mkdir()

    sidecar = persistence._build_source_sidecar(
        project, generation_id, source_access_metadata
    )
    tower_registry = persistence.TowerRegistry(
        persistence.TOWER_REGISTRY_SCHEMA_VERSION, generation_id, project.project_id
    )
    span_registry = persistence.SpanRegistry(
        persistence.SPAN_REGISTRY_SCHEMA_VERSION, generation_id, project.project_id
    )
    run_index = persistence.RunIndex(
        persistence.RUN_INDEX_SCHEMA_VERSION, generation_id, project.project_id
    )

    project_payload = project.to_dict()
    written = {
        "source/source-reference.json": published_json_bytes(sidecar.to_dict()),
        "towers/towers.json": published_json_bytes(tower_registry.to_dict()),
        "spans/spans.json": published_json_bytes(span_registry.to_dict()),
        "runs/run-index.json": published_json_bytes(run_index.to_dict()),
        "project.json": published_json_bytes(project_payload),
    }
    for relative, data in written.items():
        (generation_path / relative).write_bytes(data)

    manifest = {
        "schema_version": persistence.GENERATION_SCHEMA_VERSION,
        "project_id": project.project_id,
        "generation_id": generation_id,
        "sealed": True,
        "files": [
            {
                "relative_path": relative,
                "byte_size": len(written[relative]),
                "sha256": file_sha256(written[relative]),
            }
            for relative in (
                "project.json",
                "source/source-reference.json",
                "towers/towers.json",
                "spans/spans.json",
                "runs/run-index.json",
            )
        ],
        "project_manifest_logical_digest": logical_json_digest(project_payload),
        "created_at": TIMESTAMP,
        "published_at": TIMESTAMP,
    }
    manifest_bytes = published_json_bytes(manifest)
    (generation_path / "generation.json").write_bytes(manifest_bytes)
    project_bytes = written["project.json"]
    pointer = {
        "schema_version": persistence.ACTIVE_POINTER_SCHEMA_VERSION,
        "project_id": project.project_id,
        "active_generation_id": generation_id,
        "generation_manifest_reference": f"state/generations/{generation_id}/generation.json",
        "generation_manifest_sha256": file_sha256(manifest_bytes),
        "project_manifest_reference": f"state/generations/{generation_id}/project.json",
        "project_manifest_sha256": file_sha256(project_bytes),
        "project": project_payload,
    }
    (root / "project.json").write_bytes(published_json_bytes(pointer))
    return generation_path


# ---------------------------------------------------------------------------
# A. CandidateSet persistence round-trip
# ---------------------------------------------------------------------------


def test_candidate_set_persistence_round_trip(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        persisted = _persist_set(session, binding)
        assert session.candidate_store.candidate_sets[0].reference == persisted.reference

    with open_project_read_only(root) as reopened:
        loaded = reopened.candidate_store.candidate_sets[0]
        assert loaded.reference == persisted.reference
        assert loaded.candidate_set_id == "cset-001"
        assert loaded.digest == persisted.digest
        assert loaded.to_dict() == persisted.to_dict()


# ---------------------------------------------------------------------------
# B. ReviewLedger persistence round-trip
# ---------------------------------------------------------------------------


def test_review_ledger_persistence_round_trip(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        _append_pending_then_accepted(session, candidate_set)
        assert len(session.candidate_store.review_ledgers) == 1

    with open_project_read_only(root) as reopened:
        ledger = reopened.candidate_store.review_ledgers[0]
        # The frozen V5A1-A ledger stores its declared candidate sets through a
        # sorted list, so compare through an explicit tuple projection.
        assert tuple(ledger.candidate_sets) == (candidate_set.reference,)
        revisions = [event.review_revision for event in ledger.events]
        assert revisions == [1, 2]
        assert [event.prior_event_revision for event in ledger.events] == [None, 1]
        assert ledger.events[-1].decision is ReviewDecision.ACCEPTED


# ---------------------------------------------------------------------------
# C. Confirmation persistence round-trip
# ---------------------------------------------------------------------------


def test_confirmation_persistence_round_trip(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        accepted = _append_pending_then_accepted(session, candidate_set)
        confirmation = session.confirm_reviewed_tower_candidate(
            candidate_set_reference=candidate_set.reference,
            candidate_id="P001",
            accepted_review_revision=accepted.review_revision,
            reviewer_identity="reviewer-1",
            confirmed_at=TIMESTAMP,
        )
        assert confirmation.confirmed_tower.tower_id == "T001"
        assert confirmation.confirmed_tower.revision == 1

    with open_project_read_only(root) as reopened:
        confirmation = reopened.candidate_store.confirmations[0]
        assert confirmation.candidate_set_reference == candidate_set.reference
        assert confirmation.candidate_id == "P001"
        assert confirmation.accepted_review_revision == 2
        assert confirmation.confirmed_tower.tower_id == "T001"
        assert reopened.tower_registry.get("T001").current_revision == 1


# ---------------------------------------------------------------------------
# D. Missing candidate artifact -> fail closed
# ---------------------------------------------------------------------------


def test_missing_candidate_artifact_fails_closed(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _persist_set(session, binding)
    loaded = persistence.open_project(root)
    (loaded.generation_path / "candidates" / "candidate-sets" / "cset-001.json").unlink()
    with pytest.raises(FileIntegrityError):
        persistence.open_project(root)


# ---------------------------------------------------------------------------
# E. Modified candidate artifact -> digest failure
# ---------------------------------------------------------------------------


def test_modified_candidate_artifact_digest_failure(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _persist_set(session, binding)
    loaded = persistence.open_project(root)

    # First: an inconsistent byte tamper is caught by the sealed manifest digest.
    path = loaded.generation_path / "candidates" / "candidate-sets" / "cset-001.json"
    path.write_bytes(path.read_bytes().replace(b'"P001"', b'"P001" '))
    with pytest.raises(FileIntegrityError):
        persistence.open_project(root)

    # A consistent rewrite that changes scientific content is caught by the
    # immutable candidate-set content digest even though manifest/index/pointer
    # were updated to match the new bytes.  ``loaded`` still points at the valid
    # generation paths, so no reopen is needed before the consistent tamper.
    def mutate(payload):
        payload["candidates"][0]["candidate_xyz_absolute_m"] = [99.0, 99.0, 99.0]

    _rewrite_generation_file(
        loaded, "candidates/candidate-sets/cset-001.json", mutate
    )
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


# ---------------------------------------------------------------------------
# F. Wrong candidate schema -> fail closed
# ---------------------------------------------------------------------------


def test_wrong_candidate_schema_fails_closed(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _persist_set(session, binding)
    loaded = persistence.open_project(root)

    def mutate(payload):
        payload["schema_version"] = "treeline-tower-candidate-set-v2"

    _rewrite_generation_file(loaded, "candidates/candidate-sets/cset-001.json", mutate)
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


# ---------------------------------------------------------------------------
# T. Legacy manifest v1 without candidate store loads as empty store
# ---------------------------------------------------------------------------


def test_legacy_manifest_v1_loads_as_empty_candidate_store(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    v1_path = _write_true_v1_generation(root, project, source_access_metadata=metadata)
    with open_project_read_only(root) as reopened:
        assert reopened.generation_manifest.schema_version == GENERATION_SCHEMA_VERSION
        assert reopened.generation_id == v1_path.name
        assert reopened.candidate_store.candidate_sets == ()
        assert reopened.candidate_store.review_ledgers == ()
        assert reopened.candidate_store.confirmations == ()
        assert reopened.project.schema_version == PROJECT_SCHEMA_VERSION
        assert not (reopened.loaded_project.generation_path / "candidates").exists()


# ---------------------------------------------------------------------------
# Append-only integrity: gaps / wrong linkage / post-confirmation append
# ---------------------------------------------------------------------------


def test_append_rejects_post_confirmation_review(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        accepted = _append_pending_then_accepted(session, candidate_set)
        session.confirm_reviewed_tower_candidate(
            candidate_set_reference=candidate_set.reference,
            candidate_id="P001",
            accepted_review_revision=accepted.review_revision,
            confirmed_at=TIMESTAMP,
        )
        with pytest.raises(AlreadyConfirmedError):
            session.append_tower_candidate_review(
                candidate_set_reference=candidate_set.reference,
                candidate_id="P001",
                decision=ReviewDecision.SUPERSEDED,
                occurred_at=TIMESTAMP,
                reason="late observation",
                refinement_state=RefinementState.UNREFINED,
            )


def test_append_derives_next_revision_from_authoritative_ledger(
    tmp_path, tiny_las_project
):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        _append_pending_then_accepted(session, candidate_set)
        extra = session.append_tower_candidate_review(
            candidate_set_reference=candidate_set.reference,
            candidate_id="P001",
            decision=ReviewDecision.REJECTED,
            occurred_at=TIMESTAMP,
            reason="manual rejection",
            refinement_state=RefinementState.UNREFINED,
            operator_identity="reviewer-1",
        )
        assert extra.review_revision == 3
        assert extra.prior_event_revision == 2
        ledger = session.candidate_store.review_ledgers[0]
        assert [event.review_revision for event in ledger.events] == [1, 2, 3]


def test_review_cannot_reference_unknown_candidate_set(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        fake_ref = CandidateSetReference(
            candidate_set_id="missing-set", candidate_set_digest="0" * 64
        )
        with pytest.raises(InvalidTowerCandidateState):
            session.append_tower_candidate_review(
                candidate_set_reference=fake_ref,
                candidate_id="P001",
                decision=ReviewDecision.PENDING_REVIEW,
                occurred_at=TIMESTAMP,
                reason="cannot resolve",
                refinement_state=RefinementState.UNREFINED,
            )
        assert session.candidate_store.review_ledgers == ()
