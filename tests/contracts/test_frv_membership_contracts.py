"""Focused FRV membership/scope contract tests (PLAN-FROZEN)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import core.contracts.frv as frv
import core.contracts.frv_membership as mem
import pytest
from core.contracts.frv_a2 import (
    FrvA2ArtifactComponentBinding,
    FrvA2ArtifactComponentKind,
    FrvA2FailurePhase,
    FrvA2FailureRecord,
    FrvA2ProviderRole,
)
from core.contracts.frv_membership import (
    FRV_DECISION_EVIDENCE_DIGEST_DOMAIN,
    FRV_MEMBERSHIP_POLICY_DIGEST_DOMAIN,
    FRV_SCOPE_GEOMETRY_DIGEST_DOMAIN,
    FRV_SCOPE_POLICY_DIGEST_DOMAIN,
    FrvDecisionEvidenceArtifact,
    FrvMembershipExecutionResult,
    FrvMembershipExecutionState,
    FrvMembershipPolicyConfig,
    FrvScopeEvidenceReference,
    FrvScopePolicyConfig,
    default_frv_membership_policy_config,
    default_frv_scope_policy_config,
    validate_frv_decision_evidence_artifact,
    validate_frv_membership_execution_result,
    validate_frv_membership_policy_config,
    validate_frv_scope_evidence_reference,
    validate_frv_scope_policy_config,
    validate_membership_result_state_completeness,
)
from core.contracts.frv_source import FrvSourceProviderResult

from tests.contracts.test_frv_contracts import (
    artifact_dict,
    block_dict,
    schema_dict,
)
from tests.io.test_source_provider import _source_identity

SHA = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def _canon(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _component(key: str, kind: FrvA2ArtifactComponentKind) -> FrvA2ArtifactComponentBinding:
    return FrvA2ArtifactComponentBinding(
        component_key=key, component_kind=kind, sha256=SHA, size_bytes=16
    )


# ---------------------------------------------------------------------------
# Scope policy config
# ---------------------------------------------------------------------------


def test_scope_policy_defaults_and_digest_oracle() -> None:
    config = default_frv_scope_policy_config()
    assert config.schema_version == "frv-scope-policy-v1"
    assert config.voxel_size_m == 0.05
    assert config.context_margin_xy_cells == 30
    assert config.context_margin_z_cells == 30
    assert config.tile_edge_cells == 60
    assert config.feature_halo_cells == 30
    assert config.boundary_rule == "min-inclusive-max-exclusive"
    assert config.coordinate_canonicalization == "binary64-subtract-origin-once"
    assert config.geometry_digest_domain == "FRVSCPGEO1"
    validate_frv_scope_policy_config(config)
    payload = {
        "digest_domain": FRV_SCOPE_POLICY_DIGEST_DOMAIN,
        "schema_version": "frv-scope-policy-v1",
        "voxel_size_m": 0.05,
        "context_margin_xy_cells": 30,
        "context_margin_z_cells": 30,
        "tile_edge_cells": 60,
        "feature_halo_cells": 30,
        "boundary_rule": "min-inclusive-max-exclusive",
        "coordinate_canonicalization": "binary64-subtract-origin-once",
        "geometry_digest_domain": "FRVSCPGEO1",
    }
    expected = hashlib.sha256(_canon(payload)).hexdigest()
    assert config.policy_digest == expected


def test_scope_policy_rejections() -> None:
    with pytest.raises(ValueError):
        FrvScopePolicyConfig(
            **{**default_frv_scope_policy_config().to_dict(), "voxel_size_m": 0.0}
        )
    with pytest.raises(ValueError):
        FrvScopePolicyConfig(
            **{**default_frv_scope_policy_config().to_dict(), "boundary_rule": "closed"}
        )
    with pytest.raises(ValueError):
        FrvScopePolicyConfig(
            **{**default_frv_scope_policy_config().to_dict(), "policy_digest": "0" * 64}
        )
    with pytest.raises(ValueError):
        FrvScopePolicyConfig(
            **{
                **default_frv_scope_policy_config().to_dict(),
                "geometry_digest_domain": "OTHER",
            }
        )


# ---------------------------------------------------------------------------
# Membership policy config
# ---------------------------------------------------------------------------


def test_membership_policy_defaults_and_digest_oracle() -> None:
    config = default_frv_membership_policy_config()
    assert config.schema_version == "frv-membership-policy-v1"
    assert config.algorithm_version == "frv-membership-synchronous-v1"
    assert config.decision_rule_version == "frv-membership-decision-v1"
    assert config.ordinary_min_hag_m == 0.30
    assert config.upper_seed_min_hag_m == 1.00
    assert config.ground_max_abs_raw_hag_m == 0.10
    assert config.propagation_min_hag_drop_m == 0.05
    assert config.max_structural_rounds == 12
    assert config.ground_classification_value == 2
    assert config.output_block_record_count == 250000
    assert config.policy_digest_domain == "FRVMEMPOL1"
    validate_frv_membership_policy_config(config)
    payload = {
        "digest_domain": FRV_MEMBERSHIP_POLICY_DIGEST_DOMAIN,
        "schema_version": "frv-membership-policy-v1",
        "algorithm_version": "frv-membership-synchronous-v1",
        "decision_rule_version": "frv-membership-decision-v1",
        "ordinary_min_hag_m": 0.30,
        "upper_seed_min_hag_m": 1.00,
        "ground_max_abs_raw_hag_m": 0.10,
        "propagation_min_hag_drop_m": 0.05,
        "max_structural_rounds": 12,
        "ground_classification_value": 2,
        "output_block_record_count": 250000,
        "policy_digest_domain": "FRVMEMPOL1",
    }
    expected = hashlib.sha256(_canon(payload)).hexdigest()
    assert config.policy_digest == expected


def test_membership_policy_rejections() -> None:
    with pytest.raises(ValueError):
        FrvMembershipPolicyConfig(
            **{**default_frv_membership_policy_config().to_dict(), "max_structural_rounds": 0}
        )
    with pytest.raises(ValueError):
        FrvMembershipPolicyConfig(
            **{**default_frv_membership_policy_config().to_dict(), "policy_digest": "0" * 64}
        )
    with pytest.raises(ValueError):
        FrvMembershipPolicyConfig(
            **{**default_frv_membership_policy_config().to_dict(), "policy_digest_domain": "X"}
        )


def test_policy_config_round_trip() -> None:
    scope = default_frv_scope_policy_config()
    membership = default_frv_membership_policy_config()
    assert FrvScopePolicyConfig.from_dict(scope.to_dict()) == scope
    assert FrvMembershipPolicyConfig.from_dict(membership.to_dict()) == membership


# ---------------------------------------------------------------------------
# Decision evidence artifact
# ---------------------------------------------------------------------------


def _evidence(**overrides) -> FrvDecisionEvidenceArtifact:
    defaults = {
        "schema_version": mem.FRV_DECISION_EVIDENCE_SCHEMA_VERSION,
        "source_content_identity_digest": SHA,
        "source_identity_record_digest": _SHA_B,
        "membership_artifact_manifest_digest": _SHA_C,
        "hag_proof_digest": SHA,
        "geometry_proof_digest": SHA,
        "wire_proof_digest": SHA,
        "original_index_domain_digest": SHA,
        "record_count": 4,
        "block_digests": (SHA,),
        "evidence_digest": None,
    }
    defaults.update(overrides)
    fields = {key: value for key, value in defaults.items() if key != "evidence_digest"}
    payload = {"digest_domain": FRV_DECISION_EVIDENCE_DIGEST_DOMAIN, **fields}
    payload["block_digests"] = list(payload["block_digests"])
    evidence_digest = (
        defaults["evidence_digest"]
        if defaults["evidence_digest"] is not None
        else hashlib.sha256(_canon(payload)).hexdigest()
    )
    return FrvDecisionEvidenceArtifact(**{**fields, "evidence_digest": evidence_digest})


def test_decision_evidence_builds_and_oracle() -> None:
    artifact = _evidence()
    validate_frv_decision_evidence_artifact(artifact)
    assert artifact.record_count == 4
    payload = {
        "digest_domain": FRV_DECISION_EVIDENCE_DIGEST_DOMAIN,
        "schema_version": mem.FRV_DECISION_EVIDENCE_SCHEMA_VERSION,
        "source_content_identity_digest": SHA,
        "source_identity_record_digest": _SHA_B,
        "membership_artifact_manifest_digest": _SHA_C,
        "hag_proof_digest": SHA,
        "geometry_proof_digest": SHA,
        "wire_proof_digest": SHA,
        "original_index_domain_digest": SHA,
        "record_count": 4,
        "block_digests": [SHA],
    }
    expected = hashlib.sha256(_canon(payload)).hexdigest()
    assert artifact.evidence_digest == expected


def test_decision_evidence_round_trip_and_rejections() -> None:
    artifact = _evidence()
    assert FrvDecisionEvidenceArtifact.from_dict(artifact.to_dict()) == artifact
    with pytest.raises(ValueError):
        _evidence(block_digests=(_SHA_B, SHA))  # unsorted
    with pytest.raises(ValueError):
        _evidence(record_count=-1)


# ---------------------------------------------------------------------------
# Scope evidence reference
# ---------------------------------------------------------------------------


def _scope_reference() -> FrvScopeEvidenceReference:
    return FrvScopeEvidenceReference(
        a1_npz_path=Path("a1.npz"),
        a1_json_path=Path("a1.json"),
        v1_npz_path=Path("v1.npz"),
        v1_json_path=Path("v1.json"),
        v2_npz_path=Path("v2.npz"),
        v2_json_path=Path("v2.json"),
        a1_npz_component=_component("a1_npz", FrvA2ArtifactComponentKind.PAYLOAD_NPZ),
        a1_json_component=_component("a1_json", FrvA2ArtifactComponentKind.MANIFEST_JSON),
        v1_npz_component=_component("v1_npz", FrvA2ArtifactComponentKind.PAYLOAD_NPZ),
        v1_json_component=_component("v1_json", FrvA2ArtifactComponentKind.MANIFEST_JSON),
        v2_npz_component=_component("v2_npz", FrvA2ArtifactComponentKind.PAYLOAD_NPZ),
        v2_json_component=_component("v2_json", FrvA2ArtifactComponentKind.MANIFEST_JSON),
    )


def test_scope_evidence_reference_validates() -> None:
    reference = _scope_reference()
    validate_frv_scope_evidence_reference(reference)
    with pytest.raises(ValueError):
        FrvScopeEvidenceReference(
            a1_npz_path="not-a-path",
            a1_json_path=Path("a1.json"),
            v1_npz_path=Path("v1.npz"),
            v1_json_path=Path("v1.json"),
            v2_npz_path=Path("v2.npz"),
            v2_json_path=Path("v2.json"),
            a1_npz_component=_component("a1_npz", FrvA2ArtifactComponentKind.PAYLOAD_NPZ),
            a1_json_component=_component("a1_json", FrvA2ArtifactComponentKind.MANIFEST_JSON),
            v1_npz_component=_component("v1_npz", FrvA2ArtifactComponentKind.PAYLOAD_NPZ),
            v1_json_component=_component("v1_json", FrvA2ArtifactComponentKind.MANIFEST_JSON),
            v2_npz_component=_component("v2_npz", FrvA2ArtifactComponentKind.PAYLOAD_NPZ),
            v2_json_component=_component("v2_json", FrvA2ArtifactComponentKind.MANIFEST_JSON),
        )
    with pytest.raises(ValueError):
        FrvScopeEvidenceReference(
            a1_npz_path=Path("a1.npz"),
            a1_json_path=Path("a1.json"),
            v1_npz_path=Path("v1.npz"),
            v1_json_path=Path("v1.json"),
            v2_npz_path=Path("v2.npz"),
            v2_json_path=Path("v2.json"),
            a1_npz_component="not-a-component",
            a1_json_component=_component(
                "a1_json", FrvA2ArtifactComponentKind.MANIFEST_JSON
            ),
            v1_npz_component=_component(
                "v1_npz", FrvA2ArtifactComponentKind.PAYLOAD_NPZ
            ),
            v1_json_component=_component(
                "v1_json", FrvA2ArtifactComponentKind.MANIFEST_JSON
            ),
            v2_npz_component=_component(
                "v2_npz", FrvA2ArtifactComponentKind.PAYLOAD_NPZ
            ),
            v2_json_component=_component(
                "v2_json", FrvA2ArtifactComponentKind.MANIFEST_JSON
            ),
        )


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------


def _empty_manifest() -> frv.FRVScopeCoverageManifest:
    payload = {
        "manifest_schema_version": frv.FRV_SCOPE_COVERAGE_MANIFEST_SCHEMA_VERSION,
        "source_content_identity_digest": SHA,
        "configuration_digest": SHA,
        "scope_construction_version": "scope-1",
        "canonical_inventory_digest": frv.frv_digest(()),
        "required_inventory_digest": None,
        "scope_union_id": None,
        "scope_union_geometry_digest": None,
        "records": [],
        "canonical_inventory_count": 0,
        "required_inventory_count": 0,
        "approved_hard_rejection_count": 0,
        "scope_created_or_merged_count": 0,
        "coverage_blocked_count": 0,
        "canonical_inventory_complete": True,
        "required_inventory_complete": True,
        "approved_hard_rejection_complete": True,
        "scope_coverage_complete": True,
        "routing_complete": True,
        "expected_semantic_artifact_id": None,
        "provider_failure_reference_digests": [],
        "stream_failure_reference_digests": [],
        "identity_failure_reference_digests": [],
        "warnings": [],
        "limitations": [],
        "provenance_digest": SHA,
        "manifest_digest": SHA,
    }
    digest = frv.frv_digest(
        {key: value for key, value in payload.items() if key != "manifest_digest"}
    )
    payload["manifest_digest"] = digest
    return frv.FRVScopeCoverageManifest.from_dict(payload)


def _source() -> frv.SourceIdentity:
    return _source_identity("C:/s.las", SHA, 8, ([0.0, 0.0, 0.0], [7.0, 7.0, 10.0]))


def _result(**overrides) -> FrvMembershipExecutionResult:
    defaults: dict = {
        "execution_state": FrvMembershipExecutionState.NO_RELEVANT_FRV_SCOPE,
        "source_identity": _source(),
        "a2_readiness": None,
        "coverage_manifest": _empty_manifest(),
        "membership_artifact": None,
        "payload_schema": None,
        "artifact_binding": None,
        "decision_evidence_artifact": None,
        "source_proof_digest": SHA,
        "hag_proof_digest": None,
        "geometry_proof_digest": None,
        "wire_proof_digest": None,
        "provider_binding_digests": (SHA,),
        "override_digests": (),
        "failure_records": (),
        "output_reference": None,
    }
    defaults.update(overrides)
    result = FrvMembershipExecutionResult.__new__(FrvMembershipExecutionResult)
    for name, value in defaults.items():
        object.__setattr__(result, name, value)
    object.__setattr__(result, "result_digest", result.computed_digest())
    return result


def _result_payload(result: FrvMembershipExecutionResult) -> dict:
    return {
        "digest_domain": "FRVMEMRES1",
        "execution_state": result.execution_state.value,
        "source_identity": result.source_identity.to_dict(),
        "a2_readiness": (
            result.a2_readiness.to_dict() if result.a2_readiness is not None else None
        ),
        "coverage_manifest": result.coverage_manifest.to_dict(),
        "membership_artifact": (
            result.membership_artifact.to_dict()
            if result.membership_artifact is not None
            else None
        ),
        "payload_schema": (
            result.payload_schema.to_dict() if result.payload_schema is not None else None
        ),
        "artifact_binding": (
            result.artifact_binding.to_dict()
            if result.artifact_binding is not None
            else None
        ),
        "decision_evidence_artifact": (
            result.decision_evidence_artifact.to_dict()
            if result.decision_evidence_artifact is not None
            else None
        ),
        "source_proof_digest": result.source_proof_digest,
        "hag_proof_digest": result.hag_proof_digest,
        "geometry_proof_digest": result.geometry_proof_digest,
        "wire_proof_digest": result.wire_proof_digest,
        "provider_binding_digests": list(result.provider_binding_digests),
        "override_digests": list(result.override_digests),
        "failure_records": [
            failure.to_dict() for failure in result.failure_records
        ],
    }


def test_execution_result_no_scope_builds() -> None:
    result = _result()
    validate_frv_membership_execution_result(result)
    assert result.execution_state is FrvMembershipExecutionState.NO_RELEVANT_FRV_SCOPE


def test_execution_result_digest_oracle() -> None:
    result = _result()
    expected = hashlib.sha256(_canon(_result_payload(result))).hexdigest()
    assert result.result_digest == expected


def test_execution_result_round_trip() -> None:
    result = _result()
    restored = FrvMembershipExecutionResult.from_dict(result.to_dict())
    assert restored.to_dict() == result.to_dict()


def test_source_proof_digest_rejects_non_finalized_result() -> None:
    """DC-001: a non-finalized/fabricated source result yields no fingerprint."""

    fabricated = FrvSourceProviderResult.__new__(FrvSourceProviderResult)
    object.__setattr__(fabricated, "source_identity", _source())
    with pytest.raises(ValueError):
        mem.frv_source_proof_digest(fabricated)


def test_state_completeness_no_scope() -> None:
    result = _result()
    validate_membership_result_state_completeness(result)
    # A NO_SCOPE result cannot carry a failure record.
    record = FrvA2FailureRecord(
        phase=FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        provider_role=FrvA2ProviderRole.WIRE,
        provider_key="wire_provider",
        field_name="wire_authority",
        error_code=frv.FrvErrorCode.FRV_PROVIDER_INVALID,
        reason_flags=1024,
        detail=(
            "structure_integrity:wire:wire_provider:wire_authority:"
            "FRV_PROVIDER_INVALID:1024"
        ),
    )
    with pytest.raises(ValueError):
        validate_membership_result_state_completeness(
            _result(failure_records=(record,))
        )


def test_state_completeness_blocked_requires_failures() -> None:
    blocked = _result(execution_state=FrvMembershipExecutionState.COVERAGE_BLOCKED)
    with pytest.raises(ValueError):
        validate_membership_result_state_completeness(blocked)


def test_state_completeness_ready_forbids_no_artifact() -> None:
    ready = _result(execution_state=FrvMembershipExecutionState.READY_FOR_V4)
    with pytest.raises(ValueError):
        validate_membership_result_state_completeness(ready)


def test_scope_policy_digest_domains_frozen() -> None:
    assert FRV_SCOPE_POLICY_DIGEST_DOMAIN == "FRVSCPPOL1"
    assert FRV_SCOPE_GEOMETRY_DIGEST_DOMAIN == "FRVSCPGEO1"
    assert FRV_MEMBERSHIP_POLICY_DIGEST_DOMAIN == "FRVMEMPOL1"


# ---------------------------------------------------------------------------
# Membership artifact configuration_purpose contract
# ---------------------------------------------------------------------------


def _artifact(purpose: str) -> frv.FullResolutionVegetationMembershipArtifact:
    schema = frv.FRVArtifactPayloadSchema.from_dict(schema_dict())
    block = frv.FRVArtifactBlockDescriptor.from_dict(block_dict(schema))
    return frv.FullResolutionVegetationMembershipArtifact.from_dict(
        artifact_dict([block], schema, configuration_purpose=purpose)
    )


def test_artifact_configuration_purpose_contract_validation() -> None:
    # The artifact is a faithful record of the validated configuration purpose:
    # every closed enum member is serializable, and the semantic authority is
    # enforced by the formal consumers (which reject test/calibration).
    for purpose in ("TEST_ONLY", "CALIBRATION_CANDIDATE", "REVIEWED_FIXED_POLICY",
                    "PRODUCTION_APPROVED"):
        artifact = _artifact(purpose)
        assert isinstance(artifact.configuration_purpose, frv.FrvConfigurationPurpose)
        assert artifact.configuration_purpose.value == purpose


def test_artifact_supported_formal_purposes_validate_digest_domain() -> None:
    for purpose in (
        frv.FrvConfigurationPurpose.REVIEWED_FIXED_POLICY,
        frv.FrvConfigurationPurpose.PRODUCTION_APPROVED,
    ):
        artifact = _artifact(purpose.value)
        frv.validate_artifact_payload_digest_domain(artifact)
        assert artifact.manifest_digest == artifact.computed_digest()
        assert artifact.artifact_id == artifact.computed_artifact_id()


def test_artifact_missing_or_unsupported_purpose_fails_closed() -> None:
    schema = frv.FRVArtifactPayloadSchema.from_dict(schema_dict())
    block = frv.FRVArtifactBlockDescriptor.from_dict(block_dict(schema))
    with pytest.raises(ValueError, match="configuration_purpose"):
        frv.FullResolutionVegetationMembershipArtifact.from_dict(
            {
                key: value
                for key, value in artifact_dict([block], schema).items()
                if key != "configuration_purpose"
            }
        )
    with pytest.raises(ValueError, match="configuration_purpose"):
        frv.FullResolutionVegetationMembershipArtifact.from_dict(
            {**artifact_dict([block], schema), "configuration_purpose": "OTHER"}
        )


def test_artifact_id_binds_configuration_purpose() -> None:
    reviewed = _artifact("REVIEWED_FIXED_POLICY")
    production = _artifact("PRODUCTION_APPROVED")
    assert reviewed.artifact_id != production.artifact_id
    assert reviewed.artifact_id == reviewed.computed_artifact_id()
    assert production.artifact_id == production.computed_artifact_id()


def test_manifest_digest_binds_configuration_purpose() -> None:
    reviewed = _artifact("REVIEWED_FIXED_POLICY")
    production = _artifact("PRODUCTION_APPROVED")
    assert reviewed.manifest_digest != production.manifest_digest
    assert frv.frv_digest(reviewed.to_digest_dict()) == reviewed.manifest_digest


def test_semantic_authority_cannot_be_altered_independently() -> None:
    artifact = _artifact("REVIEWED_FIXED_POLICY")
    payload = artifact.to_dict()
    payload["configuration_purpose"] = "PRODUCTION_APPROVED"
    with pytest.raises(ValueError):
        frv.FullResolutionVegetationMembershipArtifact.from_dict(payload)
