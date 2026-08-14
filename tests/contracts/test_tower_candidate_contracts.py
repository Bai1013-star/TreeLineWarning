"""Focused V5A1-A tower candidate / review contract tests (sections A-H)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest
from core.contracts.project import (
    CoordinateFrame,
    CoordinateUnit,
    SourceIdentityBinding,
    TowerGeometry,
    TowerRevisionReference,
)
from core.contracts.tower_candidates import (
    TOWER_CANDIDATE_CONFIRMATION_SCHEMA_VERSION,
    TOWER_CANDIDATE_REVIEW_EVENT_SCHEMA_VERSION,
    TOWER_CANDIDATE_REVIEW_LEDGER_SCHEMA_VERSION,
    TOWER_CANDIDATE_SET_SCHEMA_VERSION,
    CandidateOrigin,
    CandidateSetReference,
    DetectorKind,
    DetectorProvenance,
    RefinementState,
    ReviewDecision,
    TowerCandidate,
    TowerCandidateBounds,
    TowerCandidateConfirmation,
    TowerCandidateManualAddProvenance,
    TowerCandidateQualityFlag,
    TowerCandidateRawRefinementEvidence,
    TowerCandidateReviewEvent,
    TowerCandidateReviewLedger,
    TowerCandidateSet,
    derive_tower_candidate_set_digest,
    validate_tower_candidate_confirmation,
    validate_tower_candidate_review_ledger,
)

from tests.contracts.test_project_contracts import (
    PROJECT_ID,
    SHA,
    TIMESTAMP,
    _source_identity,
)

# ---------------------------------------------------------------------------
# Shared fixtures (reused by the serialization test module)
# ---------------------------------------------------------------------------


def _binding() -> SourceIdentityBinding:
    return SourceIdentityBinding.from_source_identity(_source_identity())


def _detector() -> DetectorProvenance:
    return DetectorProvenance(
        detector_kind=DetectorKind.GEOMETRY,
        detector_name="geometry-detector",
        detector_version="1.0.0",
        configuration_digest=SHA,
        detector_evidence={"method": "local-maxima", "runs": 3},
    )


def _manual_provenance() -> TowerCandidateManualAddProvenance:
    return TowerCandidateManualAddProvenance(
        added_at=TIMESTAMP,
        reason="operator-verified pole",
        operator_identity="reviewer-1",
    )


def _refinement_evidence(binding: SourceIdentityBinding) -> TowerCandidateRawRefinementEvidence:
    return TowerCandidateRawRefinementEvidence(
        method="full-resolution-raw",
        algorithm_version="raw-refinement-v1",
        source_content_identity_digest=binding.source_content_identity_digest,
        source_identity_record_digest=binding.source_identity_record_digest,
        details={"base_points": 40, "top_points": 12},
    )


def _candidate(
    binding: SourceIdentityBinding,
    candidate_id: str = "P001",
    origin: CandidateOrigin = CandidateOrigin.AUTOMATIC,
    **overrides: object,
) -> TowerCandidate:
    detector = None if origin is CandidateOrigin.MANUAL_ADD else _detector()
    manual = _manual_provenance() if origin is CandidateOrigin.MANUAL_ADD else None
    base: dict[str, object] = dict(
        candidate_id=candidate_id,
        origin=origin,
        coordinate_frame=binding.coordinate_context.frame,
        coordinate_unit=binding.coordinate_context.units,
        candidate_xyz_absolute_m=(10.0, 20.0, 2.0),
        candidate_bounds_absolute_m=None,
        confidence=1.0,
        quality_flags=TowerCandidateQualityFlag.NONE,
        detector_provenance=detector,
        manual_add_provenance=manual,
        evidence={},
    )
    base.update(overrides)
    return TowerCandidate(**base)  # type: ignore[arg-type]


def _set(
    binding: SourceIdentityBinding,
    *,
    candidate_set_id: str = "cset-001",
    candidates: list[TowerCandidate] | None = None,
    generated_at: str = TIMESTAMP,
    **overrides: object,
) -> TowerCandidateSet:
    if candidates is None:
        candidates = [_candidate(binding)]
    base: dict[str, object] = dict(
        candidate_set_id=candidate_set_id,
        project_id=PROJECT_ID,
        generated_at=generated_at,
        source_binding=binding,
        coordinate_context=binding.coordinate_context,
        detector_provenance=_detector(),
        candidates=candidates,
        corridor_bounds_absolute_m=None,
    )
    base.update(overrides)
    return TowerCandidateSet.create(**base)  # type: ignore[arg-type]


def _refined() -> TowerGeometry:
    return TowerGeometry(
        base_xyz_absolute_m=(10.0, 20.0, 1.0),
        top_xyz_absolute_m=(10.0, 20.0, 21.0),
        height_m=20.0,
    )


def _event(
    binding: SourceIdentityBinding,
    set_ref: CandidateSetReference,
    candidate_id: str = "P001",
    revision: int = 1,
    decision: ReviewDecision = ReviewDecision.ACCEPTED,
    prior: int | None = None,
    **overrides: object,
) -> TowerCandidateReviewEvent:
    refinement_state = overrides.get(
        "refinement_state", RefinementState.RAW_REFINED
    )
    if refinement_state is RefinementState.RAW_REFINED:
        default_geometry: TowerGeometry | None = _refined()
        default_evidence: TowerCandidateRawRefinementEvidence | None = _refinement_evidence(
            binding
        )
    else:
        default_geometry = None
        default_evidence = None
    base: dict[str, object] = dict(
        schema_version=TOWER_CANDIDATE_REVIEW_EVENT_SCHEMA_VERSION,
        candidate_set_reference=set_ref,
        candidate_id=candidate_id,
        review_revision=revision,
        decision=decision,
        occurred_at=TIMESTAMP,
        reason="validated against source",
        proposal_xyz_absolute_m=(10.0, 20.0, 2.0),
        proposal_bounds_absolute_m=None,
        refinement_state=refinement_state,
        refined_raw_geometry=default_geometry,
        refinement_evidence=default_evidence,
        operator_identity="reviewer-1",
        review_adjusted_seed_xyz_absolute_m=None,
        notes=None,
        prior_event_revision=prior,
    )
    base.update(overrides)
    return TowerCandidateReviewEvent(**base)  # type: ignore[arg-type]


def _ledger(
    set_ref: CandidateSetReference,
    events: list[TowerCandidateReviewEvent],
    review_ledger_id: str = "review-ledger-1",
    candidate_sets: tuple[CandidateSetReference, ...] | None = None,
) -> TowerCandidateReviewLedger:
    if candidate_sets is None:
        candidate_sets = (set_ref,)
    return TowerCandidateReviewLedger(
        schema_version=TOWER_CANDIDATE_REVIEW_LEDGER_SCHEMA_VERSION,
        review_ledger_id=review_ledger_id,
        candidate_sets=candidate_sets,
        events=tuple(events),
    )


def _confirmation(
    set_ref: CandidateSetReference,
    candidate_id: str = "P001",
    **overrides: object,
) -> TowerCandidateConfirmation:
    base: dict[str, object] = dict(
        schema_version=TOWER_CANDIDATE_CONFIRMATION_SCHEMA_VERSION,
        candidate_set_reference=set_ref,
        candidate_id=candidate_id,
        accepted_review_revision=2,
        confirmed_tower=TowerRevisionReference(tower_id="T001", revision=1),
        confirmed_at=TIMESTAMP,
        detector_provenance=_detector(),
        reviewer_identity="reviewer-1",
        notes=None,
    )
    base.update(overrides)
    return TowerCandidateConfirmation(**base)  # type: ignore[arg-type]


def _accepted_ledger(
    binding: SourceIdentityBinding, candidate_set: TowerCandidateSet
) -> TowerCandidateReviewLedger:
    set_ref = candidate_set.reference
    return _ledger(
        set_ref,
        [
            _event(
                binding,
                set_ref,
                revision=1,
                decision=ReviewDecision.PENDING_REVIEW,
                refinement_state=RefinementState.UNREFINED,
                refined_raw_geometry=None,
                refinement_evidence=None,
            ),
            _event(binding, set_ref, revision=2, decision=ReviewDecision.ACCEPTED, prior=1),
        ],
    )


# ---------------------------------------------------------------------------
# A. IDs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate_id", ["P001", "P100", "P999"])
def test_valid_candidate_ids_accepted(candidate_id: str) -> None:
    candidate = _candidate(_binding(), candidate_id=candidate_id)
    assert candidate.candidate_id == candidate_id


@pytest.mark.parametrize(
    "bad",
    [
        "P000",
        "P1",
        "P01",
        "P0001",
        "P0000001",
        "P1000",
        "p001",
        "P001 ",
        " P001",
        "P-01",
        "P+01",
        "P00a",
        "T001",
        "Q001",
        "",
        "P001\n",
    ],
)
def test_malformed_candidate_ids_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="candidate_id"):
        _candidate(_binding(), candidate_id=bad)


def test_duplicate_candidate_ids_rejected_in_one_set() -> None:
    binding = _binding()
    with pytest.raises(ValueError, match="unique"):
        _set(binding, candidates=[_candidate(binding, "P001"), _candidate(binding, "P001")])


def test_same_candidate_id_allowed_in_different_sets() -> None:
    binding = _binding()
    first = _set(binding, candidate_set_id="cset-001")
    second = _set(binding, candidate_set_id="cset-002")
    assert first.candidates[0].candidate_id == "P001"
    assert second.candidates[0].candidate_id == "P001"
    assert first.candidate_set_id != second.candidate_set_id


def test_candidate_id_cannot_masquerade_as_tower_id() -> None:
    # P### must never validate as a T### tower ID.
    with pytest.raises(ValueError):
        TowerRevisionReference(tower_id="P001", revision=1)
    # T### must never validate as a P### candidate ID.
    with pytest.raises(ValueError):
        _candidate(_binding(), candidate_id="T001")
    # The tower domain still accepts canonical T###.
    assert TowerRevisionReference(tower_id="T001", revision=1).tower_id == "T001"


# ---------------------------------------------------------------------------
# B. Geometry
# ---------------------------------------------------------------------------


def test_finite_xyz_accepted() -> None:
    candidate = _candidate(_binding(), candidate_xyz_absolute_m=(1.0, 2.0, 3.0))
    assert candidate.candidate_xyz_absolute_m == (1.0, 2.0, 3.0)


@pytest.mark.parametrize("bad", [nan, inf, -inf])
def test_non_finite_xyz_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _candidate(_binding(), candidate_xyz_absolute_m=(bad, 2.0, 3.0))


def test_bool_as_number_rejected() -> None:
    with pytest.raises(ValueError):
        _candidate(_binding(), candidate_xyz_absolute_m=(True, 2.0, 3.0))


def test_wrong_length_xyz_rejected() -> None:
    with pytest.raises(ValueError, match="exactly 3"):
        _candidate(_binding(), candidate_xyz_absolute_m=(1.0, 2.0))


@pytest.mark.parametrize("bad", [nan, inf, -inf])
def test_non_finite_bounds_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _candidate(
            _binding(),
            candidate_bounds_absolute_m=TowerCandidateBounds(
                (bad, 0.0, 0.0), (1.0, 1.0, 1.0)
            ),
        )


def test_bounds_ordering_rejected() -> None:
    with pytest.raises(ValueError, match="minimum"):
        _candidate(
            _binding(),
            candidate_bounds_absolute_m=TowerCandidateBounds(
                (2.0, 0.0, 0.0), (1.0, 1.0, 1.0)
            ),
        )


def test_valid_bounds_accepted() -> None:
    bounds = TowerCandidateBounds((0.0, 0.0, 0.0), (10.0, 10.0, 5.0))
    candidate = _candidate(_binding(), candidate_bounds_absolute_m=bounds)
    assert candidate.candidate_bounds_absolute_m == bounds


def test_proposal_coordinate_authority_is_source_las_absolute_metres() -> None:
    binding = _binding()
    candidate = _candidate(binding)
    assert candidate.coordinate_frame is CoordinateFrame.SOURCE_LAS_ABSOLUTE
    assert candidate.coordinate_unit is CoordinateUnit.METRE
    # A bare string is not an accepted enum instance.
    with pytest.raises(ValueError, match="coordinate_frame"):
        _candidate(binding, coordinate_frame="SOURCE_LAS_ABSOLUTE")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# C. Confidence / flags
# ---------------------------------------------------------------------------


def test_confidence_endpoints_accepted() -> None:
    binding = _binding()
    assert _candidate(binding, confidence=0.0).confidence == 0.0
    assert _candidate(binding, confidence=1.0).confidence == 1.0
    assert _candidate(binding, confidence=0.5).confidence == 0.5


@pytest.mark.parametrize("bad", [-0.1, 1.1, nan, inf, -inf])
def test_confidence_out_of_range_or_non_finite_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _candidate(_binding(), confidence=bad)


def test_confidence_bool_rejected() -> None:
    with pytest.raises(ValueError):
        _candidate(_binding(), confidence=True)  # type: ignore[arg-type]


def test_known_quality_flags_accepted() -> None:
    flags = (
        TowerCandidateQualityFlag.GEOMETRY_QUALITY_DEGRADED
        | TowerCandidateQualityFlag.OCCLUSION_SUSPECTED
    )
    candidate = _candidate(_binding(), quality_flags=flags)
    assert candidate.quality_flags == flags


def test_unknown_quality_flag_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        _candidate(_binding(), quality_flags=1 << 16)


def test_default_quality_flags_is_none() -> None:
    candidate = _candidate(_binding())
    assert candidate.quality_flags is TowerCandidateQualityFlag.NONE


# ---------------------------------------------------------------------------
# D. Candidate set
# ---------------------------------------------------------------------------


def test_candidate_set_is_immutable() -> None:
    candidate_set = _set(_binding())
    with pytest.raises(FrozenInstanceError):
        candidate_set.candidate_count = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        candidate_set.candidates = ()  # type: ignore[misc]


def test_proposal_geometry_is_immutable() -> None:
    candidate_set = _set(_binding())
    with pytest.raises(FrozenInstanceError):
        candidate_set.candidates[0].candidate_xyz_absolute_m = (0.0, 0.0, 0.0)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        candidate_set.candidates += candidate_set.candidates
    with pytest.raises(AttributeError):
        candidate_set.candidates.append(candidate_set.candidates[0])  # type: ignore[attr-defined]


def test_candidate_count_exact() -> None:
    binding = _binding()
    candidates = [_candidate(binding, "P001"), _candidate(binding, "P002")]
    candidate_set = _set(binding, candidates=candidates)
    assert candidate_set.candidate_count == 2
    digest = derive_tower_candidate_set_digest(
        schema_version=TOWER_CANDIDATE_SET_SCHEMA_VERSION,
        project_id=PROJECT_ID,
        source_binding=binding,
        coordinate_context=binding.coordinate_context,
        detector_provenance=_detector(),
        candidates=candidates,
    )
    with pytest.raises(ValueError, match="candidate_count"):
        TowerCandidateSet(
            schema_version=TOWER_CANDIDATE_SET_SCHEMA_VERSION,
            candidate_set_id="cset-001",
            project_id=PROJECT_ID,
            generated_at=TIMESTAMP,
            source_binding=binding,
            coordinate_context=binding.coordinate_context,
            detector_provenance=_detector(),
            candidates=candidates,
            candidate_count=3,
            corridor_bounds_absolute_m=None,
            digest=digest,
        )


def test_deterministic_serialization() -> None:
    binding = _binding()
    assert _set(binding).to_dict() == _set(binding).to_dict()


def test_candidate_set_canonical_order() -> None:
    binding = _binding()
    unordered = [
        _candidate(binding, "P003"),
        _candidate(binding, "P001"),
        _candidate(binding, "P002"),
    ]
    candidate_set = _set(binding, candidates=unordered)
    assert [candidate.candidate_id for candidate in candidate_set.candidates] == [
        "P001",
        "P002",
        "P003",
    ]
    assert candidate_set.candidate_count == 3


def test_same_content_different_input_order_same_digest() -> None:
    binding = _binding()
    ordered = _set(
        binding,
        candidate_set_id="cset-001",
        candidates=[
            _candidate(binding, "P001"),
            _candidate(binding, "P002"),
            _candidate(binding, "P003"),
        ],
    )
    unordered = _set(
        binding,
        candidate_set_id="cset-002",
        candidates=[
            _candidate(binding, "P003"),
            _candidate(binding, "P001"),
            _candidate(binding, "P002"),
        ],
    )
    assert ordered.digest == unordered.digest
    assert ordered.candidate_set_id != unordered.candidate_set_id
    assert ordered.to_digest_dict() == unordered.to_digest_dict()


def test_digest_is_content_fingerprint_across_reruns() -> None:
    binding = _binding()
    first = _set(binding, candidate_set_id="cset-001", generated_at="2026-01-01T00:00:00Z")
    rerun = _set(binding, candidate_set_id="cset-002", generated_at="2026-02-01T00:00:00Z")
    assert first.digest == rerun.digest
    assert first.candidate_set_id != rerun.candidate_set_id
    assert first.generated_at != rerun.generated_at


def test_digest_changes_on_scientific_content_change() -> None:
    binding = _binding()
    base = _set(binding, candidate_set_id="cset-001")
    changed = _set(
        binding,
        candidate_set_id="cset-001",
        candidates=[
            _candidate(binding, "P001", candidate_xyz_absolute_m=(99.0, 20.0, 2.0))
        ],
    )
    assert base.digest != changed.digest


def test_digest_payload_excludes_generation_fields() -> None:
    candidate_set = _set(_binding())
    payload = candidate_set.to_digest_dict()
    assert "digest" not in payload
    assert "candidate_set_id" not in payload
    assert "generated_at" not in payload
    assert payload["digest_domain"] == "TREELINETCSET1"


# ---------------------------------------------------------------------------
# E. Schema
# ---------------------------------------------------------------------------


def test_unknown_field_rejected_on_set_load() -> None:
    payload = _set(_binding()).to_dict()
    payload["extra"] = True
    with pytest.raises(ValueError, match="unknown"):
        TowerCandidateSet.from_dict(payload)


def test_unsupported_schema_version_rejected() -> None:
    binding = _binding()
    payload = _set(binding).to_dict()
    payload["schema_version"] = "treeline-tower-candidate-set-v2"
    with pytest.raises(ValueError, match="unsupported"):
        TowerCandidateSet.from_dict(payload)


def test_malformed_json_rejected() -> None:
    with pytest.raises(ValueError):
        TowerCandidateSet.from_json_bytes(b"{not json")


# ---------------------------------------------------------------------------
# F. Manual add / origin provenance
# ---------------------------------------------------------------------------


def test_manual_add_is_an_origin() -> None:
    binding = _binding()
    manual = _candidate(binding, candidate_id="P005", origin=CandidateOrigin.MANUAL_ADD)
    assert manual.origin is CandidateOrigin.MANUAL_ADD
    assert manual.manual_add_provenance is not None
    assert manual.detector_provenance is None


def test_manual_add_requires_provenance() -> None:
    with pytest.raises(ValueError, match="manual_add_provenance"):
        _candidate(
            _binding(),
            origin=CandidateOrigin.MANUAL_ADD,
            manual_add_provenance=None,
        )


def test_manual_add_rejects_detector_provenance() -> None:
    with pytest.raises(ValueError, match="detector_provenance"):
        _candidate(
            _binding(),
            origin=CandidateOrigin.MANUAL_ADD,
            detector_provenance=_detector(),
        )


def test_automatic_requires_detector_provenance() -> None:
    with pytest.raises(ValueError, match="detector_provenance"):
        _candidate(_binding(), origin=CandidateOrigin.AUTOMATIC, detector_provenance=None)


def test_automatic_rejects_manual_provenance() -> None:
    with pytest.raises(ValueError, match="manual_add_provenance"):
        _candidate(
            _binding(),
            origin=CandidateOrigin.AUTOMATIC,
            manual_add_provenance=_manual_provenance(),
        )


def test_automatic_with_both_provenances_rejected() -> None:
    with pytest.raises(ValueError, match="manual_add_provenance"):
        _candidate(
            _binding(),
            origin=CandidateOrigin.AUTOMATIC,
            detector_provenance=_detector(),
            manual_add_provenance=_manual_provenance(),
        )


def test_manual_add_is_not_a_review_state() -> None:
    assert CandidateOrigin.MANUAL_ADD not in set(ReviewDecision)
    assert not any("MANUAL" in name for name in ReviewDecision.__members__)


# ---------------------------------------------------------------------------
# G. Review
# ---------------------------------------------------------------------------


def test_valid_review_decisions_accepted() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    pending = _event(
        binding,
        set_ref,
        decision=ReviewDecision.PENDING_REVIEW,
        refinement_state=RefinementState.UNREFINED,
        refined_raw_geometry=None,
        refinement_evidence=None,
    )
    rejected = _event(
        binding,
        set_ref,
        decision=ReviewDecision.REJECTED,
        refinement_state=RefinementState.UNREFINED,
        refined_raw_geometry=None,
        refinement_evidence=None,
    )
    accepted = _event(binding, set_ref, decision=ReviewDecision.ACCEPTED)
    superseded = _event(
        binding,
        set_ref,
        revision=2,
        decision=ReviewDecision.SUPERSEDED,
        prior=1,
        refinement_state=RefinementState.UNREFINED,
        refined_raw_geometry=None,
        refinement_evidence=None,
    )
    assert pending.decision is ReviewDecision.PENDING_REVIEW
    assert rejected.decision is ReviewDecision.REJECTED
    assert accepted.decision is ReviewDecision.ACCEPTED
    assert superseded.decision is ReviewDecision.SUPERSEDED


def test_accept_requires_raw_refined_geometry() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    with pytest.raises(ValueError, match="RAW_REFINED"):
        _event(
            binding,
            set_ref,
            decision=ReviewDecision.ACCEPTED,
            refinement_state=RefinementState.UNREFINED,
            refined_raw_geometry=None,
        )
    with pytest.raises(ValueError, match="RAW_REFINED"):
        _event(
            binding,
            set_ref,
            decision=ReviewDecision.ACCEPTED,
            refinement_state=RefinementState.RAW_REFINED,
            refined_raw_geometry=None,
        )
    with pytest.raises(ValueError, match="RAW_REFINED"):
        _event(
            binding,
            set_ref,
            decision=ReviewDecision.ACCEPTED,
            refinement_state=RefinementState.RAW_REFINEMENT_FAILED,
            refined_raw_geometry=None,
        )


def test_accept_requires_nonempty_refinement_evidence() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    # null evidence is not acceptable for ACCEPTED.
    with pytest.raises(ValueError, match="refinement evidence"):
        _event(binding, set_ref, decision=ReviewDecision.ACCEPTED, refinement_evidence=None)
    # an empty mapping is not a valid closed refinement-evidence structure.
    with pytest.raises(ValueError):
        _event(binding, set_ref, decision=ReviewDecision.ACCEPTED, refinement_evidence={})
    # missing required evidence fields are rejected at the evidence boundary.
    with pytest.raises(ValueError):
        TowerCandidateRawRefinementEvidence(
            method="full-resolution-raw",
            algorithm_version="raw-refinement-v1",
            source_content_identity_digest="not-a-sha",
            source_identity_record_digest=binding.source_identity_record_digest,
        )


def test_accept_with_valid_evidence_accepted() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    event = _event(binding, set_ref, decision=ReviewDecision.ACCEPTED)
    assert event.refinement_evidence is not None
    assert event.refinement_evidence.method == "full-resolution-raw"


def test_raw_refined_requires_geometry() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    with pytest.raises(ValueError, match="requires refined_raw_geometry"):
        _event(
            binding,
            set_ref,
            decision=ReviewDecision.REJECTED,
            refinement_state=RefinementState.RAW_REFINED,
            refined_raw_geometry=None,
        )


def test_raw_refined_requires_evidence() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    with pytest.raises(ValueError, match="requires refinement evidence"):
        _event(
            binding,
            set_ref,
            decision=ReviewDecision.REJECTED,
            refinement_state=RefinementState.RAW_REFINED,
            refinement_evidence=None,
        )


def test_refined_geometry_requires_raw_refined_state() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    with pytest.raises(ValueError, match="requires RAW_REFINED"):
        _event(
            binding,
            set_ref,
            decision=ReviewDecision.REJECTED,
            refinement_state=RefinementState.UNREFINED,
            refined_raw_geometry=_refined(),
        )


def test_evidence_requires_raw_refined_state() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    with pytest.raises(ValueError, match="requires RAW_REFINED"):
        _event(
            binding,
            set_ref,
            decision=ReviewDecision.REJECTED,
            refinement_state=RefinementState.UNREFINED,
            refined_raw_geometry=None,
            refinement_evidence=_refinement_evidence(binding),
        )


def test_rejected_and_pending_without_refinement_valid() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    for decision in (ReviewDecision.PENDING_REVIEW, ReviewDecision.REJECTED):
        event = _event(
            binding,
            set_ref,
            decision=decision,
            refinement_state=RefinementState.UNREFINED,
            refined_raw_geometry=None,
            refinement_evidence=None,
        )
        assert event.refinement_state is RefinementState.UNREFINED


def test_superseded_requires_prior() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    with pytest.raises(ValueError, match="prior"):
        _event(
            binding,
            set_ref,
            revision=2,
            decision=ReviewDecision.SUPERSEDED,
            prior=None,
            refinement_state=RefinementState.UNREFINED,
            refined_raw_geometry=None,
            refinement_evidence=None,
        )
    event = _event(
        binding,
        set_ref,
        revision=2,
        decision=ReviewDecision.SUPERSEDED,
        prior=1,
        refinement_state=RefinementState.UNREFINED,
        refined_raw_geometry=None,
        refinement_evidence=None,
    )
    assert event.prior_event_revision == 1


def test_prior_cannot_point_forward() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    with pytest.raises(ValueError, match="point forward"):
        _event(binding, set_ref, revision=1, prior=1)
    with pytest.raises(ValueError, match="point forward"):
        _event(binding, set_ref, revision=2, prior=3)


def test_invalid_decision_rejected() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    with pytest.raises(ValueError, match="ReviewDecision"):
        _event(binding, set_ref, decision="NOT_A_DECISION")  # type: ignore[arg-type]


def test_ledger_append_only_reconstruction() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    ledger = _ledger(
        set_ref,
        [
            _event(
                binding,
                set_ref,
                revision=1,
                decision=ReviewDecision.PENDING_REVIEW,
                refinement_state=RefinementState.UNREFINED,
                refined_raw_geometry=None,
                refinement_evidence=None,
            )
        ],
    )
    assert ledger.effective_decision(set_ref, "P001") is ReviewDecision.PENDING_REVIEW
    accepted = _event(binding, set_ref, revision=2, decision=ReviewDecision.ACCEPTED, prior=1)
    ledger = ledger.with_event(accepted)
    assert ledger.effective_decision(set_ref, "P001") is ReviewDecision.ACCEPTED
    assert ledger.next_review_revision(set_ref, "P001") == 3
    # The original ledger is untouched: history remains reconstructable.
    assert len(ledger.events) == 2


def test_ledger_duplicate_revision_rejected() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    events = [
        _event(
            binding,
            set_ref,
            revision=1,
            decision=ReviewDecision.REJECTED,
            refinement_state=RefinementState.UNREFINED,
            refined_raw_geometry=None,
            refinement_evidence=None,
        ),
        _event(binding, set_ref, revision=1, decision=ReviewDecision.ACCEPTED),
    ]
    with pytest.raises(ValueError, match="consecutive"):
        _ledger(set_ref, events)


def test_ledger_first_revision_must_be_one() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    with pytest.raises(ValueError, match="consecutive"):
        _ledger(
            set_ref,
            [
                _event(
                    binding,
                    set_ref,
                    revision=2,
                    decision=ReviewDecision.REJECTED,
                    refinement_state=RefinementState.UNREFINED,
                    refined_raw_geometry=None,
                    refinement_evidence=None,
                )
            ],
        )


def test_ledger_prior_must_link_to_immediately_preceding_revision() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    events = [
        _event(
            binding,
            set_ref,
            revision=1,
            decision=ReviewDecision.PENDING_REVIEW,
            refinement_state=RefinementState.UNREFINED,
            refined_raw_geometry=None,
            refinement_evidence=None,
        ),
        _event(
            binding,
            set_ref,
            revision=2,
            decision=ReviewDecision.REJECTED,
            prior=None,
            refinement_state=RefinementState.UNREFINED,
            refined_raw_geometry=None,
            refinement_evidence=None,
        ),
    ]
    with pytest.raises(ValueError, match="prior_event_revision"):
        _ledger(set_ref, events)


def test_ledger_context_resolution() -> None:
    binding = _binding()
    set_ref = _set(binding, candidate_set_id="cset-001").reference
    other_set_ref = _set(binding, candidate_set_id="cset-999").reference
    event = _event(
        binding,
        other_set_ref,
        revision=1,
        decision=ReviewDecision.REJECTED,
        refinement_state=RefinementState.UNREFINED,
        refined_raw_geometry=None,
        refinement_evidence=None,
    )
    with pytest.raises(ValueError, match="resolve within the ledger context"):
        _ledger(set_ref, [event])


def test_ledger_serialization_is_canonical() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    events = [
        _event(
            binding,
            set_ref,
            revision=1,
            decision=ReviewDecision.PENDING_REVIEW,
            refinement_state=RefinementState.UNREFINED,
            refined_raw_geometry=None,
            refinement_evidence=None,
        ),
        _event(
            binding,
            set_ref,
            revision=2,
            decision=ReviewDecision.REJECTED,
            prior=1,
            refinement_state=RefinementState.UNREFINED,
            refined_raw_geometry=None,
            refinement_evidence=None,
        ),
    ]
    ledger = _ledger(set_ref, events)
    assert [event.review_revision for event in ledger.events] == [1, 2]
    assert ledger.to_dict() == _ledger(set_ref, events[::-1]).to_dict()


# ---------------------------------------------------------------------------
# F-002. Cross-artifact ledger validation against the actual candidate set
# ---------------------------------------------------------------------------


def test_ledger_grounded_in_real_set_passes() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    ledger = _accepted_ledger(binding, candidate_set)
    validate_tower_candidate_review_ledger(candidate_set, ledger)


def test_ledger_rejects_candidate_absent_from_actual_set() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    set_ref = candidate_set.reference
    # P999 is a canonical ID but does not exist in the actual set.
    ledger = _ledger(
        set_ref,
        [_event(binding, set_ref, candidate_id="P999", decision=ReviewDecision.ACCEPTED)],
    )
    with pytest.raises(ValueError, match="absent from the candidate set"):
        validate_tower_candidate_review_ledger(candidate_set, ledger)


def test_ledger_rejects_candidate_in_another_set_only() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    _set(binding, candidate_set_id="cset-999", candidates=[_candidate(binding, "P007")])
    # P007 is a canonical ID that only exists in a different candidate set.
    set_ref = candidate_set.reference
    ledger = _ledger(
        set_ref,
        [_event(binding, set_ref, candidate_id="P007", decision=ReviewDecision.ACCEPTED)],
    )
    with pytest.raises(ValueError, match="absent from the candidate set"):
        validate_tower_candidate_review_ledger(candidate_set, ledger)


def test_ledger_rejects_wrong_set_id() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    real_ref = candidate_set.reference
    fake_ref = CandidateSetReference(
        candidate_set_id="fake-001", candidate_set_digest=real_ref.candidate_set_digest
    )
    ledger = _ledger(
        fake_ref,
        [_event(binding, fake_ref, decision=ReviewDecision.ACCEPTED)],
        candidate_sets=(fake_ref,),
    )
    with pytest.raises(ValueError, match="actual set"):
        validate_tower_candidate_review_ledger(candidate_set, ledger)


def test_ledger_rejects_wrong_set_digest() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    real_ref = candidate_set.reference
    fake_ref = CandidateSetReference(
        candidate_set_id=real_ref.candidate_set_id, candidate_set_digest="b" * 64
    )
    ledger = _ledger(
        fake_ref,
        [_event(binding, fake_ref, decision=ReviewDecision.ACCEPTED)],
        candidate_sets=(fake_ref,),
    )
    with pytest.raises(ValueError, match="actual set"):
        validate_tower_candidate_review_ledger(candidate_set, ledger)


def test_ledger_rejects_proposal_mismatch() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    set_ref = candidate_set.reference
    ledger = _ledger(
        set_ref,
        [
            _event(
                binding,
                set_ref,
                decision=ReviewDecision.ACCEPTED,
                proposal_xyz_absolute_m=(99.0, 20.0, 2.0),
            )
        ],
    )
    with pytest.raises(ValueError, match="proposal"):
        validate_tower_candidate_review_ledger(candidate_set, ledger)


def test_ledger_rejects_proposal_bounds_mismatch() -> None:
    binding = _binding()
    candidate_set = _set(
        binding,
        candidates=[
            _candidate(
                binding,
                "P001",
                candidate_bounds_absolute_m=TowerCandidateBounds(
                    (0.0, 0.0, 0.0), (10.0, 10.0, 5.0)
                ),
            )
        ],
    )
    set_ref = candidate_set.reference
    ledger = _ledger(
        set_ref,
        [
            _event(
                binding,
                set_ref,
                decision=ReviewDecision.ACCEPTED,
                proposal_bounds_absolute_m=None,
            )
        ],
    )
    with pytest.raises(ValueError, match="proposal bounds"):
        validate_tower_candidate_review_ledger(candidate_set, ledger)


def test_ledger_rejects_wrong_evidence_source_linkage() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    set_ref = candidate_set.reference
    bad_evidence = TowerCandidateRawRefinementEvidence(
        method="full-resolution-raw",
        algorithm_version="raw-refinement-v1",
        source_content_identity_digest="b" * 64,
        source_identity_record_digest="b" * 64,
        details={"base_points": 40},
    )
    ledger = _ledger(
        set_ref,
        [
            _event(
                binding,
                set_ref,
                decision=ReviewDecision.ACCEPTED,
                refinement_evidence=bad_evidence,
            )
        ],
    )
    with pytest.raises(ValueError, match="source linkage"):
        validate_tower_candidate_review_ledger(candidate_set, ledger)


# ---------------------------------------------------------------------------
# H. Confirmation
# ---------------------------------------------------------------------------


def test_accepted_candidate_confirmation_valid() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    ledger = _accepted_ledger(binding, candidate_set)
    confirmation = _confirmation(candidate_set.reference)
    validate_tower_candidate_confirmation(candidate_set, ledger, confirmation)


def test_invalid_candidate_set_digest_rejected() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    set_ref = candidate_set.reference
    ledger = _accepted_ledger(binding, candidate_set)
    bogus_ref = CandidateSetReference(
        candidate_set_id=set_ref.candidate_set_id,
        candidate_set_digest="b" * 64,
    )
    confirmation = _confirmation(bogus_ref)
    with pytest.raises(ValueError, match="resolve exactly"):
        validate_tower_candidate_confirmation(candidate_set, ledger, confirmation)


def test_confirmation_references_pending_review() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    set_ref = candidate_set.reference
    ledger = _ledger(
        set_ref,
        [
            _event(
                binding,
                set_ref,
                revision=1,
                decision=ReviewDecision.PENDING_REVIEW,
                refinement_state=RefinementState.UNREFINED,
                refined_raw_geometry=None,
                refinement_evidence=None,
            )
        ],
    )
    confirmation = _confirmation(set_ref, accepted_review_revision=1)
    with pytest.raises(ValueError, match="ACCEPTED"):
        validate_tower_candidate_confirmation(candidate_set, ledger, confirmation)


def test_confirmation_references_rejected_review() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    set_ref = candidate_set.reference
    ledger = _ledger(
        set_ref,
        [
            _event(
                binding,
                set_ref,
                revision=1,
                decision=ReviewDecision.REJECTED,
                refinement_state=RefinementState.UNREFINED,
                refined_raw_geometry=None,
                refinement_evidence=None,
            )
        ],
    )
    confirmation = _confirmation(set_ref, accepted_review_revision=1)
    with pytest.raises(ValueError, match="ACCEPTED"):
        validate_tower_candidate_confirmation(candidate_set, ledger, confirmation)


def test_confirmation_references_superseded_review() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    set_ref = candidate_set.reference
    ledger = _ledger(
        set_ref,
        [
            _event(binding, set_ref, revision=1, decision=ReviewDecision.ACCEPTED),
            _event(
                binding,
                set_ref,
                revision=2,
                decision=ReviewDecision.SUPERSEDED,
                prior=1,
                refinement_state=RefinementState.UNREFINED,
                refined_raw_geometry=None,
                refinement_evidence=None,
            ),
        ],
    )
    confirmation = _confirmation(set_ref, accepted_review_revision=1)
    with pytest.raises(ValueError, match="effective"):
        validate_tower_candidate_confirmation(candidate_set, ledger, confirmation)


def test_confirmation_references_nonexistent_review_revision() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    set_ref = candidate_set.reference
    ledger = _accepted_ledger(binding, candidate_set)
    confirmation = _confirmation(set_ref, accepted_review_revision=7)
    with pytest.raises(ValueError, match="exactly one review event"):
        validate_tower_candidate_confirmation(candidate_set, ledger, confirmation)


def test_confirmation_candidate_absent_from_set() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    set_ref = candidate_set.reference
    ledger = _accepted_ledger(binding, candidate_set)
    confirmation = _confirmation(set_ref, candidate_id="P999")
    with pytest.raises(ValueError, match="candidate must exist"):
        validate_tower_candidate_confirmation(candidate_set, ledger, confirmation)


def test_confirmation_candidate_id_differs_from_review_event() -> None:
    binding = _binding()
    candidate_set = _set(
        binding,
        candidates=[_candidate(binding, "P001"), _candidate(binding, "P002")],
    )
    set_ref = candidate_set.reference
    ledger = _accepted_ledger(binding, candidate_set)
    # P002 exists in the set, but the accepted review belongs to P001.
    confirmation = _confirmation(set_ref, candidate_id="P002", accepted_review_revision=2)
    with pytest.raises(ValueError, match="exactly one review event"):
        validate_tower_candidate_confirmation(candidate_set, ledger, confirmation)


def test_fake_p999_confirmation_rejected() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    # A fabricated set reference, candidate, review, and ledger.
    fake_ref = CandidateSetReference(candidate_set_id="fake-001", candidate_set_digest="b" * 64)
    fake_ledger = _ledger(
        fake_ref,
        [_event(binding, fake_ref, candidate_id="P999", decision=ReviewDecision.ACCEPTED)],
        candidate_sets=(fake_ref,),
    )
    fake_confirmation = _confirmation(
        fake_ref,
        candidate_id="P999",
        accepted_review_revision=1,
    )
    with pytest.raises(ValueError, match="actual set"):
        validate_tower_candidate_review_ledger(candidate_set, fake_ledger)
    with pytest.raises(ValueError, match="actual set"):
        validate_tower_candidate_confirmation(candidate_set, fake_ledger, fake_confirmation)


def test_invalid_tower_id_rejected() -> None:
    with pytest.raises(ValueError):
        TowerRevisionReference(tower_id="P001", revision=1)
    with pytest.raises(ValueError):
        TowerRevisionReference(tower_id="T001", revision=0)
    with pytest.raises(ValueError):
        TowerRevisionReference(tower_id="T00", revision=1)


def test_tower_revision_validation() -> None:
    with pytest.raises(ValueError):
        TowerRevisionReference(tower_id="T001", revision=-1)


def test_rejected_observation_cannot_become_confirmation() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    set_ref = candidate_set.reference
    ledger = _ledger(
        set_ref,
        [
            _event(
                binding,
                set_ref,
                revision=1,
                decision=ReviewDecision.PENDING_REVIEW,
                refinement_state=RefinementState.UNREFINED,
                refined_raw_geometry=None,
                refinement_evidence=None,
            ),
            _event(
                binding,
                set_ref,
                revision=2,
                decision=ReviewDecision.REJECTED,
                prior=1,
                refinement_state=RefinementState.UNREFINED,
                refined_raw_geometry=None,
                refinement_evidence=None,
            ),
        ],
    )
    confirmation = _confirmation(set_ref)
    with pytest.raises(ValueError, match="ACCEPTED"):
        validate_tower_candidate_confirmation(candidate_set, ledger, confirmation)


def test_superseded_observation_cannot_become_confirmation() -> None:
    binding = _binding()
    candidate_set = _set(binding)
    set_ref = candidate_set.reference
    ledger = _ledger(
        set_ref,
        [
            _event(binding, set_ref, revision=1, decision=ReviewDecision.ACCEPTED),
            _event(
                binding,
                set_ref,
                revision=2,
                decision=ReviewDecision.SUPERSEDED,
                prior=1,
                refinement_state=RefinementState.UNREFINED,
                refined_raw_geometry=None,
                refinement_evidence=None,
            ),
        ],
    )
    confirmation = _confirmation(set_ref, accepted_review_revision=1)
    with pytest.raises(ValueError, match="effective"):
        validate_tower_candidate_confirmation(candidate_set, ledger, confirmation)


def test_confirmation_provenance_retained() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    confirmation = _confirmation(set_ref)
    assert confirmation.detector_provenance is not None
    assert confirmation.detector_provenance.detector_name == "geometry-detector"
    assert confirmation.reviewer_identity == "reviewer-1"
    assert confirmation.confirmed_tower == TowerRevisionReference("T001", 1)
