"""FRV-A1 canonical serialization and digest round-trip tests."""

from __future__ import annotations

import json

import core.contracts.frv as frv
from core.contracts.frv import (
    FRVApprovedHardRejectionEvidence,
    FRVArtifactBindingRecord,
    FRVArtifactBlockDescriptor,
    FRVArtifactPayloadSchema,
    FrvConfiguration,
    FRVManualOverrideArtifact,
    FrvRelocationAudit,
    FRVScopeCoverageManifest,
    FRVScopeCoverageRecord,
    FRVScopeDefinition,
    FrvScopeGeometryEncodingSpec,
    FrvSourceEquivalenceAudit,
    FullResolutionVegetationMembershipArtifact,
    SourceIdentity,
    canonical_frv_json_bytes,
    frv_digest,
)

from tests.contracts.test_frv_contracts import (
    artifact_dict,
    binding_dict,
    block_dict,
    config_dict,
    encoding_dict,
    equiv_dict,
    evidence_dict,
    manifest_dict,
    override_dict,
    rejected_record_dict,
    relocation_dict,
    required_record_dict,
    schema_dict,
    scope_dict,
    source_payload,
)


def _round_trip(cls: type, payload: dict) -> object:
    instance = cls.from_dict(payload)
    restored = cls.from_dict(instance.to_dict())
    assert restored == instance
    return instance


def test_configuration_round_trip() -> None:
    _round_trip(FrvConfiguration, config_dict())


def test_relocation_audit_round_trip() -> None:
    _round_trip(FrvRelocationAudit, relocation_dict())


def test_equivalence_audit_round_trip() -> None:
    _round_trip(FrvSourceEquivalenceAudit, equiv_dict())


def test_source_identity_round_trip() -> None:
    _round_trip(SourceIdentity, source_payload())


def test_encoding_spec_round_trip() -> None:
    _round_trip(FrvScopeGeometryEncodingSpec, encoding_dict())


def test_scope_definition_round_trip() -> None:
    _round_trip(FRVScopeDefinition, scope_dict())


def test_hard_rejection_evidence_round_trip() -> None:
    _round_trip(FRVApprovedHardRejectionEvidence, evidence_dict())


def test_coverage_records_round_trip() -> None:
    _round_trip(FRVScopeCoverageRecord, required_record_dict())
    _round_trip(FRVScopeCoverageRecord, rejected_record_dict())


def test_coverage_manifest_round_trip() -> None:
    records = [
        FRVScopeCoverageRecord.from_dict(required_record_dict()),
        FRVScopeCoverageRecord.from_dict(rejected_record_dict()),
    ]
    _round_trip(FRVScopeCoverageManifest, manifest_dict(records))


def test_override_round_trip() -> None:
    _round_trip(FRVManualOverrideArtifact, override_dict())


def test_payload_schema_round_trip() -> None:
    _round_trip(FRVArtifactPayloadSchema, schema_dict())


def test_block_descriptor_round_trip() -> None:
    schema = FRVArtifactPayloadSchema.from_dict(schema_dict())
    _round_trip(FRVArtifactBlockDescriptor, block_dict(schema))


def test_artifact_round_trip() -> None:
    schema = FRVArtifactPayloadSchema.from_dict(schema_dict())
    block = FRVArtifactBlockDescriptor.from_dict(block_dict(schema))
    artifact = FullResolutionVegetationMembershipArtifact.from_dict(artifact_dict([block], schema))
    restored = FullResolutionVegetationMembershipArtifact.from_dict(artifact.to_dict())
    assert restored == artifact


def test_binding_round_trip() -> None:
    schema = FRVArtifactPayloadSchema.from_dict(schema_dict())
    block = FRVArtifactBlockDescriptor.from_dict(block_dict(schema))
    artifact = FullResolutionVegetationMembershipArtifact.from_dict(artifact_dict([block], schema))
    _round_trip(FRVArtifactBindingRecord, binding_dict(artifact))


def test_all_to_dict_are_json_serializable() -> None:
    schema = FRVArtifactPayloadSchema.from_dict(schema_dict())
    block = FRVArtifactBlockDescriptor.from_dict(block_dict(schema))
    artifact = FullResolutionVegetationMembershipArtifact.from_dict(artifact_dict([block], schema))
    records = [
        FRVScopeCoverageRecord.from_dict(required_record_dict()),
        FRVScopeCoverageRecord.from_dict(rejected_record_dict()),
    ]
    instances = [
        FrvConfiguration.from_dict(config_dict()),
        FrvRelocationAudit.from_dict(relocation_dict()),
        FrvSourceEquivalenceAudit.from_dict(equiv_dict()),
        SourceIdentity.from_dict(source_payload()),
        FrvScopeGeometryEncodingSpec.from_dict(encoding_dict()),
        FRVScopeDefinition.from_dict(scope_dict()),
        FRVApprovedHardRejectionEvidence.from_dict(evidence_dict()),
        FRVScopeCoverageRecord.from_dict(required_record_dict()),
        FRVScopeCoverageManifest.from_dict(manifest_dict(records)),
        FRVManualOverrideArtifact.from_dict(override_dict()),
        schema,
        block,
        artifact,
        FRVArtifactBindingRecord.from_dict(binding_dict(artifact)),
    ]
    for instance in instances:
        json.dumps(instance.to_dict())


def test_canonical_json_serializes_enums_tuples_and_nulls() -> None:
    assert canonical_frv_json_bytes(
        {
            "mode": frv.FrvScopeMode.EXHAUSTIVE_REGION,
            "purpose": frv.FrvConfigurationPurpose.TEST_ONLY,
        }
    ) == b'{"mode":1,"purpose":"TEST_ONLY"}'
    assert canonical_frv_json_bytes({"flags": frv.FrvReasonFlag.GROUND_CONFLICT}) == (
        b'{"flags":8}'
    )
    assert canonical_frv_json_bytes({"items": (1, 2)}) == b'{"items":[1,2]}'
    assert canonical_frv_json_bytes({"missing": None}) == b'{"missing":null}'


def test_canonical_json_orders_keys_lexicographically() -> None:
    assert canonical_frv_json_bytes({"z": 1, "a": 2, "m": 3}) == b'{"a":2,"m":3,"z":1}'


def test_frv_digest_stable_across_key_order_and_round_trip() -> None:
    first = frv_digest({"b": [1, 2], "a": {"x": None}})
    second = frv_digest({"a": {"x": None}, "b": [1, 2]})
    assert first == second
    config = FrvConfiguration.from_dict(config_dict())
    assert frv_digest(config.to_digest_dict()) == config.configuration_digest
    source = SourceIdentity.from_dict(source_payload())
    assert frv_digest(source.to_content_digest_dict()) == source.source_content_identity_digest
    assert frv_digest(source.to_record_digest_dict()) == source.source_identity_record_digest


def test_self_digests_stable_across_round_trip() -> None:
    config = FrvConfiguration.from_dict(config_dict())
    restored = FrvConfiguration.from_dict(config.to_dict())
    assert restored.configuration_digest == config.configuration_digest
    assert restored.computed_digest() == config.computed_digest()
    schema = FRVArtifactPayloadSchema.from_dict(schema_dict())
    restored_schema = FRVArtifactPayloadSchema.from_dict(schema.to_dict())
    assert restored_schema.payload_schema_digest == schema.payload_schema_digest


def test_digest_changes_when_payload_field_changes() -> None:
    config = FrvConfiguration.from_dict(config_dict())
    other = FrvConfiguration.from_dict(config_dict(configuration_version="cfg-2"))
    assert other.configuration_digest != config.configuration_digest


def test_optional_fields_serialize_as_null_not_omitted() -> None:
    config = FrvConfiguration.from_dict(
        config_dict(
            configuration_purpose="TEST_ONLY",
            configuration_is_runtime_complete=False,
            formal_completeness_allowed=False,
            membership_artifact_publication_allowed=False,
            wire_provider_interface_version=None,
            wire_provider_config_digest=None,
            scope_mode=3,
        )
    )
    dumped = config.to_dict()
    assert dumped["wire_provider_config_digest"] is None
    assert "wire_provider_config_digest" in dumped
    source = SourceIdentity.from_dict(source_payload())
    assert source.to_dict()["source_equivalence_audit"] is not None
    no_canonical = SourceIdentity.from_dict(
        source_payload(
            equivalence=None,
            canonical_digest=None,
            canonical_spec_digest=None,
            canonical_spec_version=None,
        )
    )
    assert no_canonical.to_dict()["canonical_point_content_digest"] is None
