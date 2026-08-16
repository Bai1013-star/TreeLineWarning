"""V5A2/V5A3 tower-review workflow tests (thin orchestration over V5A1-B)."""

from __future__ import annotations

import pytest
from algorithms.towers.raw_access import RawTileCache, RawTileCacheConfig
from algorithms.towers.review_workflow import (
    CandidateNotReviewableError,
    accept_candidate,
    manual_add_candidate_set,
    project_tower_review_state,
    reject_candidate,
)
from core.config.fast_baseline import canonical_digest
from core.contracts.project import TowerGeometry
from core.contracts.tower_candidates import (
    CandidateOrigin,
    DetectorKind,
    DetectorProvenance,
    RefinementState,
    ReviewDecision,
    TowerCandidate,
    TowerCandidateQualityFlag,
    TowerCandidateSet,
)
from core.project import (
    AlreadyConfirmedError,
    InvalidTowerCandidateState,
    create_project_writable,
    open_project_read_only,
)

from tests.towers.conftest import project_for_source, write_tower_las


def _detector():
    return DetectorProvenance(
        detector_kind=DetectorKind.GEOMETRY,
        detector_name="test",
        detector_version="1.0.0",
        configuration_digest=canonical_digest({"a": 1}),
    )


def _candidate(binding, candidate_id, xyz):
    return TowerCandidate(
        candidate_id=candidate_id,
        origin=CandidateOrigin.AUTOMATIC,
        coordinate_frame=binding.coordinate_context.frame,
        coordinate_unit=binding.coordinate_context.units,
        candidate_xyz_absolute_m=xyz,
        confidence=0.9,
        quality_flags=TowerCandidateQualityFlag.PROPOSAL_NEEDS_REFINEMENT,
        detector_provenance=_detector(),
        manual_add_provenance=None,
        evidence={},
    )


def _new_session(tmp_path, *, second_tower=False):
    src = tmp_path / "src.las"
    points = write_tower_las(src, extra_tower_xyz=(45.0, 50.0, 0.0) if second_tower else None)
    project, binding, metadata = project_for_source(src, points)
    root = tmp_path / "project"
    session = create_project_writable(root, project, source_access_metadata=metadata)
    cache = RawTileCache.prepare(
        src, tmp_path / "cache", source_binding=binding, config=RawTileCacheConfig()
    )
    return root, session, binding, cache


def _persist_auto_set(session, binding, candidates):
    candidate_set = TowerCandidateSet.create(
        candidate_set_id="auto-001",
        project_id=session.project.project_id,
        generated_at="2026-08-15T00:00:00Z",
        source_binding=binding,
        coordinate_context=binding.coordinate_context,
        detector_provenance=_detector(),
        candidates=candidates,
    )
    session.persist_tower_candidate_set(candidate_set)
    return candidate_set


def test_automatic_pending_candidate_appears_in_state(tmp_path):
    root, session, binding, _ = _new_session(tmp_path)
    with session:
        _persist_auto_set(
            session, binding, [_candidate(binding, "P001", (10.0, 20.0, 0.0))]
        )
        state = project_tower_review_state(session)
        row = state["candidate_sets"][0]["candidates"][0]
        assert row["candidate_id"] == "P001"
        assert row["review_state"] == "PENDING_REVIEW"
        assert row["refinement_state"] == "UNREFINED"
        assert row["confirmed_tower_id"] is None


def test_reject_persists(tmp_path):
    root, session, binding, _ = _new_session(tmp_path)
    with session:
        candidate_set = _persist_auto_set(
            session, binding, [_candidate(binding, "P001", (10.0, 20.0, 0.0))]
        )
        event = reject_candidate(
            session, candidate_set, "P001", reason="not a tower", operator_identity="owner"
        )
        assert event.decision is ReviewDecision.REJECTED
        assert session.tower_registry.records == ()
    with open_project_read_only(root) as reopened:
        state = project_tower_review_state(reopened)
        row = state["candidate_sets"][0]["candidates"][0]
        assert row["review_state"] == "REJECTED"
        assert row["confirmed_tower_id"] is None


def test_accept_refines_and_allocates_tower(tmp_path):
    root, session, binding, cache = _new_session(tmp_path)
    with session:
        candidate_set = _persist_auto_set(
            session, binding, [_candidate(binding, "P001", (10.0, 20.0, 0.0))]
        )
        result = accept_candidate(
            session, candidate_set, "P001", cache, operator_identity="owner"
        )
        assert result.success, result.failure_reason
        assert result.tower_id == "T001"
        assert isinstance(result.geometry, TowerGeometry)
        tower = session.tower_registry.get("T001").current
        assert tower.refinement.algorithm_version == "raw-refinement-v1"
        # The accepted review is effective ACCEPTED + RAW_REFINED.
        state = project_tower_review_state(session)
        row = state["candidate_sets"][0]["candidates"][0]
        assert row["review_state"] == "ACCEPTED"
        assert row["refinement_state"] == "RAW_REFINED"
        assert row["confirmed_tower_id"] == "T001"


def test_failed_refinement_no_tower(tmp_path):
    root, session, binding, cache = _new_session(tmp_path)
    with session:
        # A candidate far from any structure fails RAW refinement.
        candidate_set = _persist_auto_set(
            session, binding, [_candidate(binding, "P001", (500.0, 500.0, 0.0))]
        )
        result = accept_candidate(
            session, candidate_set, "P001", cache, operator_identity="owner"
        )
        assert not result.success
        assert result.tower_id is None
        assert session.tower_registry.records == ()
        state = project_tower_review_state(session)
        row = state["candidate_sets"][0]["candidates"][0]
        assert row["refinement_state"] == "RAW_REFINEMENT_FAILED"
        assert row["failure_reason"] is not None
        assert row["confirmed_tower_id"] is None


def test_adjusted_seed_preserved(tmp_path):
    root, session, binding, cache = _new_session(tmp_path)
    with session:
        candidate_set = _persist_auto_set(
            session, binding, [_candidate(binding, "P001", (10.0, 20.0, 0.0))]
        )
        result = accept_candidate(
            session,
            candidate_set,
            "P001",
            cache,
            review_adjusted_seed_xyz_absolute_m=(10.9, 20.2, 1.0),
            operator_identity="owner",
        )
        assert result.success, result.failure_reason
        assert result.event.review_adjusted_seed_xyz_absolute_m == (10.9, 20.2, 1.0)
        # A successful accept appends exactly one ACCEPTED review event.
        assert result.event.review_revision == 1
        assert result.event.decision is ReviewDecision.ACCEPTED
        assert result.event.refinement_state == RefinementState.RAW_REFINED
        # The immutable proposal is unchanged.
        assert result.event.proposal_xyz_absolute_m == (10.0, 20.0, 0.0)


def test_manual_add_then_accept(tmp_path):
    root, session, binding, cache = _new_session(tmp_path, second_tower=True)
    with session:
        manual = manual_add_candidate_set(
            session,
            project_id=session.project.project_id,
            source_binding=binding,
            coordinate_context=binding.coordinate_context,
            candidate_xyz_absolute_m=(45.0, 50.0, 0.0),
            candidate_set_id="manual-001",
            generated_at="2026-08-15T00:00:01Z",
            operator_identity="owner",
        )
        assert manual.candidate_set_id == "manual-001"
        assert manual.candidates[0].candidate_id == "P001"
        assert manual.candidates[0].origin is CandidateOrigin.MANUAL_ADD
        result = accept_candidate(session, manual, "P001", cache, operator_identity="owner")
        assert result.success, result.failure_reason
        assert result.tower_id == "T001"
        assert result.candidate_set is manual
        state = project_tower_review_state(session)
        assert state["confirmed_towers"][0]["tower_id"] == "T001"


def test_duplicate_accept_no_duplicate_tower(tmp_path):
    root, session, binding, cache = _new_session(tmp_path)
    with session:
        candidate_set = _persist_auto_set(
            session, binding, [_candidate(binding, "P001", (10.0, 20.0, 0.0))]
        )
        first = accept_candidate(session, candidate_set, "P001", cache, operator_identity="owner")
        assert first.success and first.tower_id == "T001"
        with pytest.raises(AlreadyConfirmedError):
            accept_candidate(session, candidate_set, "P001", cache, operator_identity="owner")
        assert [e.tower_id for e in session.tower_registry.records] == ["T001"]
        assert len(session.candidate_store.confirmations) == 1


def test_reload_persists_reviews(tmp_path):
    root, session, binding, cache = _new_session(tmp_path, second_tower=True)
    with session:
        candidate_set = _persist_auto_set(
            session,
            binding,
            [
                _candidate(binding, "P001", (10.0, 20.0, 0.0)),
                _candidate(binding, "P002", (45.0, 50.0, 0.0)),
            ],
        )
        accept_candidate(session, candidate_set, "P001", cache, operator_identity="owner")
        reject_candidate(session, candidate_set, "P002", reason="not a tower")
    with open_project_read_only(root) as reopened:
        state = project_tower_review_state(reopened)
        rows = {row["candidate_id"]: row for row in state["candidate_sets"][0]["candidates"]}
        assert rows["P001"]["review_state"] == "ACCEPTED"
        assert rows["P001"]["confirmed_tower_id"] == "T001"
        assert rows["P002"]["review_state"] == "REJECTED"
        assert state["confirmed_towers"][0]["tower_id"] == "T001"


def test_candidate_numbering_does_not_leak_into_tower_numbering(tmp_path):
    # P002 accepted first -> T001; P001 accepted later -> T002.  Candidate IDs
    # must never drive the Tower registry sequence.
    root, session, binding, cache = _new_session(tmp_path, second_tower=True)
    with session:
        candidate_set = _persist_auto_set(
            session,
            binding,
            [
                _candidate(binding, "P001", (10.0, 20.0, 0.0)),
                _candidate(binding, "P002", (45.0, 50.0, 0.0)),
            ],
        )
        p002 = accept_candidate(session, candidate_set, "P002", cache, operator_identity="owner")
        assert p002.tower_id == "T001"
        p001 = accept_candidate(session, candidate_set, "P001", cache, operator_identity="owner")
        assert p001.tower_id == "T002"


def test_rejected_candidate_cannot_be_accepted(tmp_path):
    root, session, binding, cache = _new_session(tmp_path)
    with session:
        candidate_set = _persist_auto_set(
            session, binding, [_candidate(binding, "P001", (10.0, 20.0, 0.0))]
        )
        reject_candidate(session, candidate_set, "P001", reason="not a tower")
        with pytest.raises(CandidateNotReviewableError):
            accept_candidate(session, candidate_set, "P001", cache, operator_identity="owner")
        assert session.tower_registry.records == ()


def test_unknown_candidate_fails_closed(tmp_path):
    root, session, binding, cache = _new_session(tmp_path)
    with session:
        candidate_set = _persist_auto_set(
            session, binding, [_candidate(binding, "P001", (10.0, 20.0, 0.0))]
        )
        with pytest.raises(InvalidTowerCandidateState):
            accept_candidate(session, candidate_set, "P999", cache, operator_identity="owner")
