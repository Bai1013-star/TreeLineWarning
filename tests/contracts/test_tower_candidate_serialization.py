"""Focused V5A1-A serialization round-trip and strict-JSON tests (section I)."""

from __future__ import annotations

import dataclasses
from math import nan

import pytest
from core.contracts.tower_candidates import (
    RefinementState,
    ReviewDecision,
    TowerCandidateConfirmation,
    TowerCandidateReviewEvent,
    TowerCandidateReviewLedger,
    TowerCandidateSet,
    validate_tower_candidate_confirmation,
    validate_tower_candidate_review_ledger,
)

from tests.contracts.test_tower_candidate_contracts import (
    _binding,
    _candidate,
    _confirmation,
    _event,
    _ledger,
    _set,
)


def test_set_to_dict_exact_field_order() -> None:
    candidate_set = _set(_binding())
    assert list(candidate_set.to_dict()) == [
        field.name for field in dataclasses.fields(TowerCandidateSet)
    ]


def test_event_to_dict_exact_field_order() -> None:
    binding = _binding()
    event = _event(binding, _set(binding).reference)
    assert list(event.to_dict()) == [
        field.name for field in dataclasses.fields(TowerCandidateReviewEvent)
    ]


def test_ledger_to_dict_exact_field_order() -> None:
    binding = _binding()
    set_ref = _set(binding).reference
    ledger = _ledger(set_ref, [_event(binding, set_ref)])
    assert list(ledger.to_dict()) == [
        field.name for field in dataclasses.fields(TowerCandidateReviewLedger)
    ]


def test_confirmation_to_dict_exact_field_order() -> None:
    confirmation = _confirmation(_set(_binding()).reference)
    assert list(confirmation.to_dict()) == [
        field.name for field in dataclasses.fields(TowerCandidateConfirmation)
    ]


def test_enum_encoding_is_canonical_strings() -> None:
    binding = _binding()
    payload = _set(binding).to_dict()
    assert payload["detector_provenance"]["detector_kind"] == "GEOMETRY"
    candidate_payload = payload["candidates"][0]
    assert candidate_payload["origin"] == "AUTOMATIC"
    assert candidate_payload["coordinate_frame"] == "SOURCE_LAS_ABSOLUTE"
    assert candidate_payload["coordinate_unit"] == "m"
    assert candidate_payload["quality_flags"] == 0
    event_payload = _event(binding, _set(binding).reference).to_dict()
    assert event_payload["decision"] == "ACCEPTED"
    assert event_payload["refinement_state"] == "RAW_REFINED"
    assert event_payload["refinement_evidence"]["method"] == "full-resolution-raw"


def test_set_round_trip_exact_equality_and_stable_digest() -> None:
    candidate_set = _set(_binding())
    restored = TowerCandidateSet.from_dict(candidate_set.to_dict())
    assert restored == candidate_set
    assert restored.digest == candidate_set.digest
    assert restored.to_digest_dict() == candidate_set.to_digest_dict()


def test_set_json_round_trip_exact_equality_and_stable_digest() -> None:
    candidate_set = _set(_binding())
    restored = TowerCandidateSet.from_json_bytes(candidate_set.to_json_bytes())
    assert restored == candidate_set
    assert restored.digest == candidate_set.digest
    assert restored.to_json_bytes() == candidate_set.to_json_bytes()


def test_set_json_round_trip_canonicalizes_candidate_order() -> None:
    binding = _binding()
    unordered = [
        _candidate(binding, "P003"),
        _candidate(binding, "P001"),
        _candidate(binding, "P002"),
    ]
    candidate_set = _set(binding, candidates=unordered)
    restored = TowerCandidateSet.from_json_bytes(candidate_set.to_json_bytes())
    assert [item["candidate_id"] for item in restored.to_dict()["candidates"]] == [
        "P001",
        "P002",
        "P003",
    ]
    assert restored.digest == candidate_set.digest


def test_event_round_trip_exact_equality() -> None:
    binding = _binding()
    event = _event(binding, _set(binding).reference)
    restored = TowerCandidateReviewEvent.from_dict(event.to_dict())
    assert restored == event
    assert restored.to_json_bytes() == event.to_json_bytes()


def test_ledger_round_trip_exact_equality() -> None:
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
        _event(binding, set_ref, revision=2, decision=ReviewDecision.ACCEPTED, prior=1),
    ]
    ledger = _ledger(set_ref, events)
    restored = TowerCandidateReviewLedger.from_dict(ledger.to_dict())
    assert restored == ledger
    assert restored.to_dict() == ledger.to_dict()
    json_restored = TowerCandidateReviewLedger.from_json_bytes(ledger.to_json_bytes())
    assert json_restored == ledger


def test_confirmation_round_trip_exact_equality() -> None:
    confirmation = _confirmation(_set(_binding()).reference)
    restored = TowerCandidateConfirmation.from_dict(confirmation.to_dict())
    assert restored == confirmation
    json_restored = TowerCandidateConfirmation.from_json_bytes(
        confirmation.to_json_bytes()
    )
    assert json_restored == confirmation


def test_set_rejects_unknown_field() -> None:
    payload = _set(_binding()).to_dict()
    payload["mystery"] = 1
    with pytest.raises(ValueError, match="unknown"):
        TowerCandidateSet.from_dict(payload)


def test_event_rejects_unknown_field() -> None:
    binding = _binding()
    payload = _event(binding, _set(binding).reference).to_dict()
    payload["mystery"] = 1
    with pytest.raises(ValueError, match="unknown"):
        TowerCandidateReviewEvent.from_dict(payload)


def test_set_rejects_missing_field() -> None:
    payload = _set(_binding()).to_dict()
    del payload["project_id"]
    with pytest.raises(ValueError, match="missing"):
        TowerCandidateSet.from_dict(payload)


def test_set_rejects_wrong_root_type() -> None:
    with pytest.raises(ValueError):
        TowerCandidateSet.from_dict("not-a-mapping")  # type: ignore[arg-type]


def test_set_rejects_nan_in_from_dict() -> None:
    payload = _set(_binding()).to_dict()
    payload["candidates"][0]["candidate_xyz_absolute_m"] = [nan, 20.0, 2.0]
    with pytest.raises(ValueError, match="finite"):
        TowerCandidateSet.from_dict(payload)


def test_event_rejects_bool_as_int_revision() -> None:
    binding = _binding()
    payload = _event(binding, _set(binding).reference).to_dict()
    payload["review_revision"] = True
    with pytest.raises(ValueError):
        TowerCandidateReviewEvent.from_dict(payload)


def test_from_dict_rejects_candidate_p0001() -> None:
    payload = _set(_binding()).to_dict()
    payload["candidates"][0]["candidate_id"] = "P0001"
    with pytest.raises(ValueError, match="candidate_id"):
        TowerCandidateSet.from_dict(payload)


def test_from_dict_rejects_automatic_without_detector() -> None:
    payload = _set(_binding()).to_dict()
    payload["candidates"][0]["detector_provenance"] = None
    with pytest.raises(ValueError, match="detector_provenance"):
        TowerCandidateSet.from_dict(payload)


def test_from_dict_rejects_automatic_with_manual_provenance() -> None:
    payload = _set(_binding()).to_dict()
    payload["candidates"][0]["manual_add_provenance"] = {
        "added_at": "2026-08-12T00:00:00Z",
        "reason": "operator-verified pole",
        "operator_identity": "reviewer-1",
        "note": None,
    }
    with pytest.raises(ValueError, match="manual_add_provenance"):
        TowerCandidateSet.from_dict(payload)


def test_event_from_dict_rejects_accepted_without_evidence() -> None:
    binding = _binding()
    payload = _event(binding, _set(binding).reference).to_dict()
    payload["refinement_evidence"] = None
    with pytest.raises(ValueError, match="refinement evidence"):
        TowerCandidateReviewEvent.from_dict(payload)


def test_event_from_dict_rejects_empty_evidence_mapping() -> None:
    binding = _binding()
    payload = _event(binding, _set(binding).reference).to_dict()
    payload["refinement_evidence"] = {}
    with pytest.raises(ValueError):
        TowerCandidateReviewEvent.from_dict(payload)


def test_json_rejects_nan_literal() -> None:
    with pytest.raises(ValueError):
        TowerCandidateSet.from_json_bytes(b'{"schema_version": NaN}')


def test_json_rejects_infinity_literal() -> None:
    with pytest.raises(ValueError):
        TowerCandidateSet.from_json_bytes(b'{"a": Infinity}')


def test_json_rejects_malformed_document() -> None:
    with pytest.raises(ValueError, match="malformed"):
        TowerCandidateSet.from_json_bytes(b"this is not json")


def test_json_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        TowerCandidateSet.from_json_bytes(b'{"a": 1, "a": 2}')


def test_set_json_round_trip_is_byte_deterministic() -> None:
    binding = _binding()
    assert _set(binding).to_json_bytes() == _set(binding).to_json_bytes()


def test_evidence_rejects_arbitrary_objects() -> None:
    with pytest.raises(ValueError):
        _candidate(_binding(), evidence={"bad": object()})  # type: ignore[arg-type]


def test_proposal_geometry_serialized_as_list() -> None:
    payload = _set(_binding()).to_dict()
    xyz = payload["candidates"][0]["candidate_xyz_absolute_m"]
    assert isinstance(xyz, list)
    assert xyz == [10.0, 20.0, 2.0]


def test_confirmation_retains_provenance_through_round_trip() -> None:
    confirmation = _confirmation(_set(_binding()).reference)
    restored = TowerCandidateConfirmation.from_dict(confirmation.to_dict())
    assert restored.detector_provenance is not None
    assert restored.detector_provenance.detector_name == "geometry-detector"
    assert restored.reviewer_identity == "reviewer-1"
    assert restored.confirmed_tower.tower_id == "T001"


def test_fake_ledger_confirmation_fails_after_json_round_trip() -> None:
    """Cross-artifact fail-closed semantics survive JSON deserialization."""
    binding = _binding()
    candidate_set = _set(binding)
    from core.contracts.tower_candidates import (
        CandidateSetReference,
        TowerCandidateReviewEvent,
    )

    fake_ref = CandidateSetReference(
        candidate_set_id="fake-001", candidate_set_digest="b" * 64
    )
    fake_event = TowerCandidateReviewEvent.from_json_bytes(
        _event(binding, fake_ref, candidate_id="P999").to_json_bytes()
    )
    fake_ledger = TowerCandidateReviewLedger.from_json_bytes(
        _ledger(fake_ref, [fake_event], candidate_sets=(fake_ref,)).to_json_bytes()
    )
    fake_confirmation = TowerCandidateConfirmation.from_json_bytes(
        _confirmation(fake_ref, candidate_id="P999", accepted_review_revision=1).to_json_bytes()
    )
    with pytest.raises(ValueError, match="actual set"):
        validate_tower_candidate_review_ledger(candidate_set, fake_ledger)
    with pytest.raises(ValueError, match="actual set"):
        validate_tower_candidate_confirmation(candidate_set, fake_ledger, fake_confirmation)
