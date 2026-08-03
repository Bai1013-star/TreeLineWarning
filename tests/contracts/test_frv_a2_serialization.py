"""Focused FRV-A2 canonical serialization and digest tests."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from core.contracts.frv import frv_digest
from core.contracts.frv_a2 import (
    FrvA2DomainKind,
    FrvA2FailurePhase,
    FrvA2ProviderRole,
    FrvA2ReadinessRecord,
    frv_a2_index_domain_digest,
    frv_a2_provider_binding_digest,
    frv_a2_readiness_record_digest,
    validate_frv_a2_provider_binding_evidence,
    validate_frv_a2_readiness_record,
)

from tests.contracts.test_frv_a2_contracts import (
    SHA_A,
    _binding,
    _binding_payload,
    _complete_base_providers,
    _failure,
    _readiness,
)


def test_canonical_json_round_trip_has_exact_serializable_shape() -> None:
    providers = _complete_base_providers()
    binding = providers[0]
    readiness = _readiness(providers)
    binding_payload = binding.to_dict()
    readiness_payload = readiness.to_dict()
    assert "digest_domain" not in binding_payload
    assert set(binding_payload) == set(binding.__dataclass_fields__)
    assert set(readiness_payload) == set(readiness.__dataclass_fields__)
    assert json.loads(json.dumps(binding_payload)) == binding_payload
    assert json.loads(json.dumps(readiness_payload)) == readiness_payload
    assert type(binding_payload["artifact_components"]) is list
    assert type(readiness_payload["provider_evidence"]) is list
    assert FrvA2ReadinessRecord.from_dict(readiness_payload) == readiness


def test_frva2idx1_reference_vectors_and_integer_width_equality() -> None:
    assert frv_a2_index_domain_digest(
        "ORIGINAL_INDEX", SHA_A, FrvA2DomainKind.FULL_SOURCE, (0, 1, 2)
    ) == "40a280cccc3f03dbfb32be956f6a3afc05d8772b513d0d594db1cef61c8c7946"
    assert frv_a2_index_domain_digest(
        "SAMPLE_INDEX", "b" * 64, FrvA2DomainKind.SAMPLED_SUBSET, (2, 9)
    ) == "b2150fcb0f5961682a9965457560dfcd83f34942178cce0e8ea3d840f4fe8da7"
    signed = np.array([0, 1, 2], dtype=np.int32)
    unsigned = np.array([0, 1, 2], dtype=np.uint64)
    expected = frv_a2_index_domain_digest(
        "ORIGINAL_INDEX", SHA_A, FrvA2DomainKind.FULL_SOURCE, (0, 1, 2)
    )
    assert frv_a2_index_domain_digest(
        "ORIGINAL_INDEX", SHA_A, FrvA2DomainKind.FULL_SOURCE, signed
    ) == expected
    assert frv_a2_index_domain_digest(
        "ORIGINAL_INDEX", SHA_A, FrvA2DomainKind.FULL_SOURCE, unsigned
    ) == expected


@pytest.mark.parametrize(
    "indices",
    [
        [0, 1],
        (True,),
        (1.0,),
        (-1,),
        (2**64,),
        np.array([[0, 1]], dtype=np.int64),
        np.array([1.0], dtype=np.float64),
        np.array([True], dtype=np.bool_),
        np.array([1], dtype=object),
    ],
)
def test_frva2idx1_rejects_invalid_index_domains(indices: object) -> None:
    with pytest.raises(ValueError):
        frv_a2_index_domain_digest(
            "ORIGINAL_INDEX", SHA_A, FrvA2DomainKind.FULL_SOURCE, indices  # type: ignore[arg-type]
        )


def test_binding_projection_excludes_self_digest_and_runtime_text() -> None:
    binding = _binding()
    original_digest = frv_a2_provider_binding_digest(binding)
    assert original_digest == binding.binding_digest
    object.__setattr__(binding, "binding_digest", "f" * 64)
    assert frv_a2_provider_binding_digest(binding) == original_digest
    with pytest.raises(ValueError):
        validate_frv_a2_provider_binding_evidence(binding)
    with pytest.raises(ValueError):
        _binding(public_module="C:\\machine\\secret")
    with pytest.raises(ValueError):
        _binding(public_loader_name="load_provider@host")


def test_readiness_projection_excludes_nested_binding_and_result_digest() -> None:
    providers = _complete_base_providers()
    binding = providers[0]
    readiness = _readiness(providers)
    digest = frv_a2_readiness_record_digest(readiness)
    object.__setattr__(binding, "binding_digest", "f" * 64)
    assert frv_a2_readiness_record_digest(readiness) == digest
    object.__setattr__(readiness, "result_digest", "e" * 64)
    assert frv_a2_readiness_record_digest(readiness) == digest
    with pytest.raises(ValueError):
        validate_frv_a2_readiness_record(readiness)


def test_generic_dataclass_serialization_is_not_the_binding_projection() -> None:
    binding = _binding()
    generic = frv_digest(dataclasses.asdict(binding))
    assert generic != frv_a2_provider_binding_digest(binding)
    assert "binding_digest" not in str(frv_a2_provider_binding_digest(binding))


def test_readiness_projection_is_field_based_and_deterministic() -> None:
    first = _binding(provider_key="source_provider")
    second = _binding(provider_key="hag_provider", provider_role=FrvA2ProviderRole.HAG)
    third = _binding(provider_key="geometry_provider", provider_role=FrvA2ProviderRole.GEOMETRY)
    readiness = _readiness([second, third, first])
    assert [item.provider_key for item in readiness.provider_evidence] == [
        "source_provider",
        "hag_provider",
        "geometry_provider",
    ]
    # The helper constructs canonical role/key order before the record is decoded.
    assert frv_a2_readiness_record_digest(readiness) == readiness.result_digest
    assert readiness.blocking_failures == ()


def test_readiness_digest_rejects_non_projection_provider_shapes() -> None:
    readiness = _readiness(_complete_base_providers())
    validate_frv_a2_readiness_record(readiness)
    record_data = readiness.to_dict()

    def readiness_payload(provider_projections: list[dict]) -> dict:
        return {
            "digest_domain": "FRVA2RDY1",
            "schema_version": record_data["schema_version"],
            "source_content_identity_digest": record_data["source_content_identity_digest"],
            "source_identity_record_digest": record_data["source_identity_record_digest"],
            "configuration_digest": record_data["configuration_digest"],
            "provider_evidence_projections": provider_projections,
            "aggregate_readiness": record_data["aggregate_readiness"],
            "blocking_failures": record_data["blocking_failures"],
            "review_codes": record_data["review_codes"],
            "limitation_codes": record_data["limitation_codes"],
        }

    full_provider_records = [item.to_dict() for item in readiness.provider_evidence]
    full_provider_records[0]["digest_domain"] = "FRVA2BND1"
    full_dataclass_digest = frv_digest(readiness_payload(full_provider_records))
    object.__setattr__(readiness, "result_digest", full_dataclass_digest)
    with pytest.raises(ValueError):
        validate_frv_a2_readiness_record(readiness)

    projection_only_digest = frv_digest(
        readiness_payload(
            [{"binding_digest": item.binding_digest} for item in readiness.provider_evidence]
        )
    )
    object.__setattr__(readiness, "result_digest", projection_only_digest)
    with pytest.raises(ValueError):
        validate_frv_a2_readiness_record(readiness)

    reference = _readiness(_complete_base_providers())
    reference_projections = [
        _binding_payload(item.to_dict()) for item in reference.provider_evidence
    ]
    assert frv_digest(readiness_payload(reference_projections)) == reference.result_digest


def test_unknown_missing_and_typed_enum_values_are_rejected() -> None:
    payload = _binding().to_dict()
    payload.pop("binding_digest")
    payload["binding_digest"] = "a" * 64
    payload["unknown_field"] = 1
    with pytest.raises(ValueError):
        type(_binding()).from_dict(payload)
    missing = _binding().to_dict()
    del missing["provider_role"]
    with pytest.raises(ValueError):
        type(_binding()).from_dict(missing)
    failure = _failure(phase=FrvA2FailurePhase.PRODUCTION_POLICY)
    assert failure.to_dict()["phase"] == "production_policy"
    assert failure.to_dict()["provider_role"] == FrvA2ProviderRole.SOURCE_SCAN.value
