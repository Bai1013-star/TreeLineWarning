"""V5A1-B review-to-TowerRegistry confirmation tests."""

from __future__ import annotations

import json

import pytest
from core.contracts.tower_candidates import (
    CandidateOrigin,
    CandidateSetReference,
    RefinementState,
    ReviewDecision,
)
from core.project import (
    AlreadyConfirmedError,
    InvalidTowerCandidateState,
    create_project_writable,
    open_project_read_only,
    persistence,
)
from core.project.persistence import (
    ProjectIntegrityError,
    file_sha256,
    published_json_bytes,
)

from tests.contracts.test_project_contracts import TIMESTAMP, _span, _tower
from tests.contracts.test_tower_candidate_contracts import (
    _candidate,
)
from tests.project.test_tower_candidate_persistence import (
    _append_pending_then_accepted,
    _persist_set,
    _rewrite_generation_file,
)


def _new_project(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    return root, project, metadata


def _confirm(session, candidate_set, candidate_id="P001", revision=2, **overrides):
    return session.confirm_reviewed_tower_candidate(
        candidate_set_reference=candidate_set.reference,
        candidate_id=candidate_id,
        accepted_review_revision=revision,
        reviewer_identity="reviewer-1",
        confirmed_at=TIMESTAMP,
        **overrides,
    )


# ---------------------------------------------------------------------------
# G. fake P999 cannot confirm
# ---------------------------------------------------------------------------


def test_fake_p999_cannot_confirm(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        with pytest.raises(InvalidTowerCandidateState):
            session.confirm_reviewed_tower_candidate(
                candidate_set_reference=candidate_set.reference,
                candidate_id="P999",
                accepted_review_revision=1,
                confirmed_at=TIMESTAMP,
            )
        assert session.tower_registry.records == ()


# ---------------------------------------------------------------------------
# H. wrong candidate-set digest cannot confirm
# ---------------------------------------------------------------------------


def test_wrong_candidate_set_digest_cannot_confirm(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        wrong_ref = CandidateSetReference(
            candidate_set_id=candidate_set.candidate_set_id,
            candidate_set_digest="0" * 64,
        )
        with pytest.raises(InvalidTowerCandidateState):
            session.confirm_reviewed_tower_candidate(
                candidate_set_reference=wrong_ref,
                candidate_id="P001",
                accepted_review_revision=2,
                confirmed_at=TIMESTAMP,
            )
        assert session.tower_registry.records == ()


# ---------------------------------------------------------------------------
# I. proposal mismatch -> fail closed (ledger grounded to actual candidate)
# ---------------------------------------------------------------------------


def test_proposal_mismatch_fails_closed(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        _append_pending_then_accepted(session, candidate_set)
    loaded = persistence.open_project(root)
    ledger = loaded.candidate_store.review_ledgers[0]
    relative = f"candidates/review-ledgers/{ledger.review_ledger_id}.json"

    def mutate(payload):
        payload["events"][1]["proposal_xyz_absolute_m"] = [7.0, 8.0, 9.0]

    _rewrite_generation_file(loaded, relative, mutate)
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


# ---------------------------------------------------------------------------
# J/K/L/M. non-confirmable effective decisions never allocate a Tower
# ---------------------------------------------------------------------------


def _single_pending(session, candidate_set, candidate_id="P001"):
    return session.append_tower_candidate_review(
        candidate_set_reference=candidate_set.reference,
        candidate_id=candidate_id,
        decision=ReviewDecision.PENDING_REVIEW,
        occurred_at=TIMESTAMP,
        reason="pending",
        refinement_state=RefinementState.UNREFINED,
        operator_identity="reviewer-1",
    )


def test_pending_review_cannot_allocate_tower(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        event = _single_pending(session, candidate_set)
        with pytest.raises(InvalidTowerCandidateState):
            _confirm(session, candidate_set, revision=event.review_revision)
        assert session.tower_registry.records == ()
        assert session.project.next_tower_sequence == 1


def test_rejected_review_cannot_allocate_tower(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        event = session.append_tower_candidate_review(
            candidate_set_reference=candidate_set.reference,
            candidate_id="P001",
            decision=ReviewDecision.REJECTED,
            occurred_at=TIMESTAMP,
            reason="rejected",
            refinement_state=RefinementState.UNREFINED,
            operator_identity="reviewer-1",
        )
        with pytest.raises(InvalidTowerCandidateState):
            _confirm(session, candidate_set, revision=event.review_revision)
        assert session.tower_registry.records == ()


def test_superseded_review_cannot_allocate_tower(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        accepted = _append_pending_then_accepted(session, candidate_set)
        session.append_tower_candidate_review(
            candidate_set_reference=candidate_set.reference,
            candidate_id="P001",
            decision=ReviewDecision.SUPERSEDED,
            occurred_at=TIMESTAMP,
            reason="superseded",
            refinement_state=RefinementState.UNREFINED,
            operator_identity="reviewer-1",
        )
        # The ACCEPTED revision is no longer effective; no Tower may be allocated.
        with pytest.raises(InvalidTowerCandidateState):
            _confirm(session, candidate_set, revision=accepted.review_revision)
        assert session.tower_registry.records == ()


def test_unrefined_review_cannot_allocate_tower(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        event = _single_pending(session, candidate_set)
        assert event.refinement_state is RefinementState.UNREFINED
        # UNREFINED can never satisfy the ACCEPTED + RAW_REFINED requirement.
        with pytest.raises(InvalidTowerCandidateState):
            _confirm(session, candidate_set, revision=event.review_revision)
        assert session.tower_registry.records == ()


# ---------------------------------------------------------------------------
# N. valid ACCEPTED + RAW_REFINED -> next canonical Tower
# ---------------------------------------------------------------------------


def test_valid_accepted_raw_refined_allocates_next_canonical_tower(
    tmp_path, tiny_las_project
):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        accepted = _append_pending_then_accepted(session, candidate_set)
        confirmation = _confirm(session, candidate_set, revision=accepted.review_revision)
        assert confirmation.confirmed_tower.tower_id == "T001"
        assert confirmation.confirmed_tower.revision == 1
        tower = session.tower_registry.get("T001").current
        assert tower.display_name == "P001"
        assert tower.refinement.evidence["candidate_id"] == "P001"
        assert tower.refinement.evidence["candidate_set_id"] == candidate_set.candidate_set_id
        assert tower.refinement.evidence["accepted_review_revision"] == 2
        assert tower.refinement.algorithm_version == "raw-refinement-v1"
        assert session.project.next_tower_sequence == 2


# ---------------------------------------------------------------------------
# O. duplicate confirmation -> exactly one Tower identity
# ---------------------------------------------------------------------------


def test_duplicate_confirmation_yields_exactly_one_tower(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        accepted = _append_pending_then_accepted(session, candidate_set)
        first = _confirm(session, candidate_set, revision=accepted.review_revision)
        assert first.confirmed_tower.tower_id == "T001"
        with pytest.raises(AlreadyConfirmedError):
            _confirm(session, candidate_set, revision=accepted.review_revision)
        assert {entry.tower_id for entry in session.tower_registry.records} == {"T001"}
        assert session.project.next_tower_sequence == 2
        assert len(session.candidate_store.confirmations) == 1
    with open_project_read_only(root) as reopened:
        assert {entry.tower_id for entry in reopened.tower_registry.records} == {"T001"}
        assert len(reopened.candidate_store.confirmations) == 1


# ---------------------------------------------------------------------------
# P. multiple candidates / two sets -> sequential canonical Towers
# ---------------------------------------------------------------------------


def test_multiple_candidates_produce_sequential_canonical_towers(
    tmp_path, tiny_las_project
):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        automatic = _persist_set(
            session,
            binding,
            candidate_set_id="auto-001",
            candidates=[
                _candidate(binding, candidate_id="P001"),
                _candidate(binding, candidate_id="P002"),
                _candidate(binding, candidate_id="P003"),
            ],
        )
        manual = _persist_set(
            session,
            binding,
            candidate_set_id="manual-001",
            candidates=[
                _candidate(binding, candidate_id="P001", origin=CandidateOrigin.MANUAL_ADD)
            ],
        )
        # automatic P001 accepted
        _append_pending_then_accepted(session, automatic, candidate_id="P001")
        # automatic P002 rejected
        session.append_tower_candidate_review(
            candidate_set_reference=automatic.reference,
            candidate_id="P002",
            decision=ReviewDecision.REJECTED,
            occurred_at=TIMESTAMP,
            reason="rejected",
            refinement_state=RefinementState.UNREFINED,
            operator_identity="reviewer-1",
        )
        # automatic P003 accepted
        _append_pending_then_accepted(session, automatic, candidate_id="P003")
        # manual P001 accepted
        _append_pending_then_accepted(session, manual, candidate_id="P001")

        t001 = _confirm(session, automatic, candidate_id="P001", revision=2)
        t002 = _confirm(session, automatic, candidate_id="P003", revision=2)
        t003 = _confirm(session, manual, candidate_id="P001", revision=2)
        assert (t001.confirmed_tower.tower_id, t002.confirmed_tower.tower_id) == ("T001", "T002")
        assert t003.confirmed_tower.tower_id == "T003"
        assert [entry.tower_id for entry in session.tower_registry.records] == [
            "T001",
            "T002",
            "T003",
        ]
        assert len(session.candidate_store.confirmations) == 3
    with open_project_read_only(root) as reopened:
        assert [entry.tower_id for entry in reopened.tower_registry.records] == [
            "T001",
            "T002",
            "T003",
        ]
        assert len(reopened.candidate_store.confirmations) == 3


# ---------------------------------------------------------------------------
# Q. publication failure -> old generation remains authoritative
# ---------------------------------------------------------------------------


def test_publication_failure_keeps_prior_generation_authoritative(
    tmp_path, tiny_las_project, monkeypatch
):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        accepted = _append_pending_then_accepted(session, candidate_set)
        before = session.generation_id

        def fail(*args, **kwargs):
            raise persistence.GenerationPublicationError(
                "injected pre-publication failure",
                operation="generation-directory-replace",
                path=root / "state" / "generations" / "injected",
            )

        monkeypatch.setattr(persistence, "_publish_generation", fail)
        with pytest.raises(persistence.GenerationPublicationError):
            _confirm(session, candidate_set, revision=accepted.review_revision)
        # The in-memory allocation never became authoritative.
        assert session.generation_id == before
        assert session.tower_registry.records == ()
        assert session.project.next_tower_sequence == 1
        assert session.candidate_store.confirmations == ()
    with open_project_read_only(root) as reopened:
        assert reopened.generation_id == before
        assert reopened.tower_registry.records == ()
        assert reopened.candidate_store.confirmations == ()


# ---------------------------------------------------------------------------
# R. confirmation referencing a missing Tower -> fail closed on load
# ---------------------------------------------------------------------------


def _rewrite_active_tower_registry(loaded, mutate):
    path = loaded.generation_path / "towers" / "towers.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    data = published_json_bytes(payload)
    path.write_bytes(data)
    manifest_path = loaded.generation_path / "generation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item for item in manifest["files"] if item["relative_path"] == "towers/towers.json"
    )
    record["byte_size"] = len(data)
    record["sha256"] = file_sha256(data)
    manifest_path.write_bytes(published_json_bytes(manifest))
    pointer_path = loaded.project_root / "project.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["generation_manifest_sha256"] = file_sha256(manifest_path.read_bytes())
    pointer_path.write_bytes(published_json_bytes(pointer))


def test_confirmation_referencing_missing_tower_fails_closed(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        accepted = _append_pending_then_accepted(session, candidate_set)
        _confirm(session, candidate_set, revision=accepted.review_revision)
    loaded = persistence.open_project(root)
    _rewrite_active_tower_registry(loaded, lambda payload: payload.update(records=[]))
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


# ---------------------------------------------------------------------------
# S. confirmation stays tied to T### rev1; revise_tower creates rev2
# ---------------------------------------------------------------------------


def test_confirmation_remains_tied_to_rev1_after_revision(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        accepted = _append_pending_then_accepted(session, candidate_set)
        _confirm(session, candidate_set, revision=accepted.review_revision)
        revised = session.revise_tower("T001", _tower("T999", binding, 30.0))
        assert revised.tower_id == "T001"
        assert revised.revision == 2
        assert session.tower_registry.get("T001").current_revision == 2
    with open_project_read_only(root) as reopened:
        confirmation = reopened.candidate_store.confirmations[0]
        assert confirmation.confirmed_tower.tower_id == "T001"
        assert confirmation.confirmed_tower.revision == 1
        assert reopened.tower_registry.get("T001").current_revision == 2
        assert [item.revision for item in reopened.tower_registry.get("T001").revisions] == [1, 2]


# ---------------------------------------------------------------------------
# U/V. existing create_tower / create_span surface remains compatible
# ---------------------------------------------------------------------------


def test_existing_registry_surface_remains_compatible(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    binding = project.source_reference.binding
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        candidate_set = _persist_set(session, binding)
        accepted = _append_pending_then_accepted(session, candidate_set)
        _confirm(session, candidate_set, revision=accepted.review_revision)
        tower = session.create_tower(_tower("T999", binding, 10.0))
        assert tower.tower_id == "T002"
        span = session.create_span(
            _span(binding, from_tower_id="T001", to_tower_id="T002")
        )
        assert span.span_id == "S001"
        assert session.tower_registry.get("T001").current_revision == 1
        assert session.span_registry.records[0].span_id == "S001"
