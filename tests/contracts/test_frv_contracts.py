"""Focused FRV-A1 contract tests: enums, configuration, identity, coverage."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import core.contracts.frv as frv
import numpy as np
import pytest
from core.contracts.frv import (
    FRVApprovedHardRejectionEvidence,
    FRVArtifactBindingRecord,
    FRVArtifactBlockDescriptor,
    FRVArtifactPayloadSchema,
    FrvCanonicalPointContentSpec,
    FrvCanonicalPointFieldSpec,
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
    validate_configuration,
    validate_scope_inventory_conservation,
    validate_source_content_identity_digest,
    validate_source_equivalence_audit_without_cycle,
    validate_source_identity_record_digest,
)

SHA = "a" * 64


def _digest_of(items: dict[str, Any], excluded: str) -> str:
    return frv_digest({key: value for key, value in items.items() if key != excluded})


# -- configuration ----------------------------------------------------------


def config_dict(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema_version": frv.FRV_CONFIGURATION_SCHEMA_VERSION,
        "configuration_version": "cfg-1",
        "algorithm_version": "alg-1",
        "decision_rule_version": "rule-1",
        "canonicalization_version": "canon-1",
        "source_identity_schema_version": frv.FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
        "canonical_point_content_spec_version": (
            frv.FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION
        ),
        "scope_schema_version": frv.FRV_SCOPE_DEFINITION_SCHEMA_VERSION,
        "scope_construction_version": "sc-1",
        "scope_geometry_encoding_version": frv.FRV_SCOPE_GEOMETRY_ENCODING_SCHEMA_VERSION,
        "coverage_manifest_schema_version": frv.FRV_SCOPE_COVERAGE_MANIFEST_SCHEMA_VERSION,
        "artifact_schema_version": frv.FRV_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION,
        "block_schema_version": frv.FRV_ARTIFACT_BLOCK_SCHEMA_VERSION,
        "payload_schema_version": frv.FRV_ARTIFACT_PAYLOAD_SCHEMA_VERSION,
        "override_schema_version": frv.FRV_MANUAL_OVERRIDE_SCHEMA_VERSION,
        "scope_mode": 1,
        "scope_policy_digest": SHA,
        "override_policy_digest": SHA,
        "feature_policy_digest": SHA,
        "resource_policy_digest": SHA,
        "hag_provider_interface_version": "hag-1",
        "geometry_provider_interface_version": "geo-1",
        "scan_provider_interface_version": "scan-1",
        "hag_provider_config_digest": SHA,
        "geometry_provider_config_digest": SHA,
        "scan_provider_config_digest": SHA,
        "wire_provider_interface_version": "wire-1",
        "wire_provider_config_digest": SHA,
        "feature_policy_version": "fp-1",
        "membership_parameter_schema_version": "mp-1",
        "calibration_record_digest": SHA,
        "allow_wire_body_hard_exclusion": False,
        "allow_wire_body_confirmed": False,
        "allow_review_routing_reduction": False,
        "allow_source_specific_selector": False,
        "configuration_purpose": "PRODUCTION_APPROVED",
        "configuration_is_runtime_complete": True,
        "formal_completeness_allowed": True,
        "membership_artifact_publication_allowed": True,
    }
    payload.update(overrides)
    payload["configuration_digest"] = _digest_of(payload, "configuration_digest")
    return payload


# -- relocation / equivalence audits and source identity --------------------


def relocation_dict(**overrides: Any) -> dict[str, Any]:
    payload = {
        "audit_schema_version": frv.FRV_RELOCATION_AUDIT_SCHEMA_VERSION,
        "path_style": "posix",
        "recorded_path_raw": "data/src.las",
        "resolved_path_raw": "data/src.las",
        "recorded_path_lexical_normalized": "data/src.las",
        "resolved_path_lexical_normalized": "data/src.las",
        "resolved_final_path": None,
        "resolution_method": "exact_match",
        "case_comparison_mode": "case_sensitive",
        "platform_name": "linux",
        "run_id": "run-1",
        "audit_timestamp": "2026-01-01T00:00:00Z",
        "content_identity_result": "matched",
        "warning_flags": 0,
    }
    payload.update(overrides)
    payload["audit_digest"] = _digest_of(payload, "audit_digest")
    return payload


def equiv_dict(**overrides: Any) -> dict[str, Any]:
    payload = {
        "audit_schema_version": frv.FRV_SOURCE_EQUIVALENCE_AUDIT_SCHEMA_VERSION,
        "prior_source_content_identity_digest": SHA,
        "current_source_content_identity_digest": SHA,
        "prior_source_file_sha256": SHA,
        "current_source_file_sha256": SHA,
        "canonical_point_content_spec_digest": SHA,
        "canonical_point_content_digest": SHA,
        "source_summary_comparison_digest": SHA,
        "approval_reference": "approval-1",
        "approver": "reviewer-a",
        "approval_timestamp": "2026-01-02T00:00:00Z",
        "decision_digest": SHA,
        "approval_digest": SHA,
    }
    payload.update(overrides)
    payload["audit_digest"] = _digest_of(payload, "audit_digest")
    return payload


def source_payload(
    relocation: FrvRelocationAudit | None = None,
    equivalence: object = "default",
    content_sha: str = SHA,
    path: str = "data/src.las",
    canonical_digest: str | None = SHA,
    canonical_spec_digest: str | None = SHA,
    canonical_spec_version: str | None = frv.FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build a self-consistent SourceIdentity payload.

    The default includes a matching equivalence audit and non-null canonical
    point-content digests (R-008).  Pass ``equivalence=None`` for a source
    without an equivalence audit; pass ``canonical_digest=None`` together with
    ``equivalence=None`` for a source without canonical point-content digests.
    """
    audit = (
        relocation
        if relocation is not None
        else FrvRelocationAudit.from_dict(relocation_dict())
    )
    content_payload = {
        "source_identity_schema_version": frv.FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
        "source_id": "scene-1",
        "source_file_sha256": content_sha,
        "canonical_point_content_digest": canonical_digest,
        "canonical_point_content_spec_digest": canonical_spec_digest,
        "canonical_point_content_spec_version": canonical_spec_version,
        "source_point_count": 1000,
        "source_bounds_min_xyz": [0.0, 0.0, 0.0],
        "source_bounds_max_xyz": [10.0, 10.0, 5.0],
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "origin_context": [0.0, 0.0, 0.0],
    }
    content_digest = frv_digest(content_payload)
    if equivalence == "default":
        equivalence_audit = FrvSourceEquivalenceAudit.from_dict(
            equiv_dict(
                current_source_content_identity_digest=content_digest,
                current_source_file_sha256=content_sha,
                canonical_point_content_digest=canonical_digest,
                canonical_point_content_spec_digest=canonical_spec_digest,
            )
        )
    elif equivalence is None:
        equivalence_audit = None
    else:
        equivalence_audit = equivalence
    record_payload = {
        **content_payload,
        "recorded_source_path": path,
        "resolved_access_path": path,
        "relocation_audit_audit_digest": audit.audit_digest,
        "source_equivalence_audit_audit_digest": (
            equivalence_audit.audit_digest
            if equivalence_audit is not None
            else None
        ),
        "source_content_identity_digest": content_digest,
    }
    return {
        **content_payload,
        "recorded_source_path": path,
        "resolved_access_path": path,
        "relocation_audit": audit.to_dict(),
        "source_equivalence_audit": (
            equivalence_audit.to_dict() if equivalence_audit is not None else None
        ),
        "source_content_identity_digest": content_digest,
        "source_identity_record_digest": frv_digest(record_payload),
    }


# -- geometry encoding spec and scope definition -----------------------------


def encoding_dict(**overrides: Any) -> dict[str, Any]:
    payload = {
        "encoding_schema_version": frv.FRV_SCOPE_GEOMETRY_ENCODING_SCHEMA_VERSION,
        "encoding_spec_id": "geo-spec-1",
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "origin_context_digest": SHA,
        "boundary_rule_version": "boundary-v1",
        "box_order_rule": "lexicographic-six-tuple",
        "float_encoding": "IEEE754_BINARY64",
        "byte_order": "LITTLE_ENDIAN",
        "negative_zero_policy": "normalize-to-positive-zero",
        "box_framing_version": "box-framing-v1",
    }
    payload.update(overrides)
    payload["encoding_spec_digest"] = _digest_of(payload, "encoding_spec_digest")
    return payload


def scope_dict(**overrides: Any) -> dict[str, Any]:
    payload = {
        "scope_schema_version": frv.FRV_SCOPE_DEFINITION_SCHEMA_VERSION,
        "source_content_identity_digest": SHA,
        "scope_id": "scope-1",
        "region_id": "region-1",
        "scope_mode": 1,
        "geometry_encoding_spec_digest": "b" * 64,
        "scope_core_geometry_digest": "c" * 64,
        "scope_context_margin_geometry_digest": "d" * 64,
        "feature_halo_policy_digest": SHA,
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "origin_context_digest": SHA,
        "scope_construction_version": "sc-v1",
        "upstream_reference_digests": [SHA],
        "boundary_margin_policy_digest": SHA,
        "raw_xyz_inclusion_rule_version": "raw-xyz-v1",
        "half_open_boundary_rule": "min-inclusive-max-exclusive",
        "warnings": [],
        "limitations": [],
    }
    payload.update(overrides)
    payload["scope_definition_digest"] = _digest_of(payload, "scope_definition_digest")
    return payload


# -- hard rejection evidence and coverage records ----------------------------


def evidence_dict(**overrides: Any) -> dict[str, Any]:
    payload = {
        "evidence_schema_version": frv.FRV_HARD_REJECTION_EVIDENCE_SCHEMA_VERSION,
        "source_content_identity_digest": SHA,
        "canonical_inventory_key": "k-3",
        "canonical_inventory_record_digest": SHA,
        "approved_rule_id": "rule-1",
        "approved_rule_version": "v1",
        "hard_non_vegetation_evidence_reference": "evidence-ref",
        "hard_non_vegetation_evidence_digest": SHA,
        "decision_digest": SHA,
        "approver": "reviewer-a",
        "approval_timestamp": "2026-01-01T00:00:00Z",
        "approval_digest": "b" * 64,
        "safety_completeness_effect": "PRESERVES_SAFETY_COMPLETENESS",
    }
    key = overrides.pop("key", None)
    if key is not None:
        payload["canonical_inventory_key"] = key
    payload.update(overrides)
    payload["evidence_digest"] = _digest_of(payload, "evidence_digest")
    return payload


def required_record_dict(**overrides: Any) -> dict[str, Any]:
    key = overrides.pop("key", None)
    payload = {
        "record_schema_version": frv.FRV_SCOPE_COVERAGE_RECORD_SCHEMA_VERSION,
        "source_content_identity_digest": SHA,
        "canonical_inventory_key": key if key is not None else "k-2",
        "canonical_inventory_record_digest": SHA,
        "canonical_inventory_digest": SHA,
        "inventory_disposition": 1,
        "required_inventory_record_digest": SHA,
        "required_inventory_digest": SHA,
        "approved_hard_rejection_evidence": None,
        "upstream_candidate_reference": "cand-ref",
        "upstream_candidate_type": "candidate-type",
        "upstream_status": "review",
        "upstream_evidence_reference": "e-ref",
        "scope_id": "scope-1",
        "scope_construction_state": 1,
        "scope_coverage_state": 1,
        "failure_reason_flags": 0,
        "review_required": False,
        "safety_completeness_effect": "PRESERVES_SAFETY_COMPLETENESS",
        "diagnostic_reference": "diag-1",
        "out_of_policy_safety_approval_digest": None,
    }
    payload.update(overrides)
    payload["record_digest"] = _digest_of(payload, "record_digest")
    return payload


def rejected_record_dict(**overrides: Any) -> dict[str, Any]:
    key = overrides.pop("key", None)
    payload = {
        "record_schema_version": frv.FRV_SCOPE_COVERAGE_RECORD_SCHEMA_VERSION,
        "source_content_identity_digest": SHA,
        "canonical_inventory_key": key if key is not None else "k-3",
        "canonical_inventory_record_digest": SHA,
        "canonical_inventory_digest": SHA,
        "inventory_disposition": 2,
        "required_inventory_record_digest": None,
        "required_inventory_digest": None,
        "approved_hard_rejection_evidence": None,
        "upstream_candidate_reference": "cand-ref",
        "upstream_candidate_type": "candidate-type",
        "upstream_status": "hard-rejection",
        "upstream_evidence_reference": "e-ref",
        "scope_id": None,
        "scope_construction_state": 3,
        "scope_coverage_state": 1,
        "failure_reason_flags": 0,
        "review_required": False,
        "safety_completeness_effect": "PRESERVES_SAFETY_COMPLETENESS",
        "diagnostic_reference": "diag-2",
        "out_of_policy_safety_approval_digest": None,
    }
    payload.update(overrides)
    if payload.get("approved_hard_rejection_evidence") is None:
        payload["approved_hard_rejection_evidence"] = FRVApprovedHardRejectionEvidence.from_dict(
            evidence_dict(key=payload["canonical_inventory_key"])
        ).to_dict()
    payload["record_digest"] = _digest_of(payload, "record_digest")
    return payload


def blocked_record_dict(**overrides: Any) -> dict[str, Any]:
    key = overrides.pop("key", None)
    payload = {
        "record_schema_version": frv.FRV_SCOPE_COVERAGE_RECORD_SCHEMA_VERSION,
        "source_content_identity_digest": SHA,
        "canonical_inventory_key": key if key is not None else "k-9",
        "canonical_inventory_record_digest": SHA,
        "canonical_inventory_digest": SHA,
        "inventory_disposition": 1,
        "required_inventory_record_digest": SHA,
        "required_inventory_digest": SHA,
        "approved_hard_rejection_evidence": None,
        "upstream_candidate_reference": "cand-ref",
        "upstream_candidate_type": "candidate-type",
        "upstream_status": "review",
        "upstream_evidence_reference": "e-ref",
        "scope_id": None,
        "scope_construction_state": 3,
        "scope_coverage_state": 2,
        "failure_reason_flags": 16,
        "review_required": True,
        "safety_completeness_effect": "BLOCKS_SAFETY_COMPLETENESS",
        "diagnostic_reference": "diag-3",
        "out_of_policy_safety_approval_digest": None,
    }
    payload.update(overrides)
    payload["record_digest"] = _digest_of(payload, "record_digest")
    return payload


def manifest_dict(records: list[FRVScopeCoverageRecord], **overrides: Any) -> dict[str, Any]:
    required = [record for record in records if record.inventory_disposition == 1]
    payload = {
        "manifest_schema_version": frv.FRV_SCOPE_COVERAGE_MANIFEST_SCHEMA_VERSION,
        "source_content_identity_digest": SHA,
        "configuration_digest": SHA,
        "scope_construction_version": "sc-v1",
        "canonical_inventory_digest": SHA,
        "required_inventory_digest": SHA,
        "scope_union_id": "union-1",
        "scope_union_geometry_digest": SHA,
        "records": [record.to_dict() for record in records],
        "canonical_inventory_count": len(records),
        "required_inventory_count": len(required),
        "approved_hard_rejection_count": len(records) - len(required),
        "scope_created_or_merged_count": sum(
            1 for record in required if record.scope_construction_state in (1, 2)
        ),
        "coverage_blocked_count": sum(
            1 for record in required if record.scope_construction_state == 3
        ),
        "canonical_inventory_complete": True,
        "required_inventory_complete": True,
        "approved_hard_rejection_complete": True,
        "scope_coverage_complete": not any(
            record.scope_construction_state == 3 for record in required
        ),
        "routing_complete": not any(
            record.scope_construction_state == 3 for record in required
        ),
        "expected_semantic_artifact_id": None,
        "provider_failure_reference_digests": [],
        "stream_failure_reference_digests": [],
        "identity_failure_reference_digests": [],
        "warnings": [],
        "limitations": [],
        "provenance_digest": SHA,
    }
    payload.update(overrides)
    payload["manifest_digest"] = _digest_of(payload, "manifest_digest")
    return payload


# -- manual override ---------------------------------------------------------


def override_dict(**overrides: Any) -> dict[str, Any]:
    payload = {
        "override_schema_version": frv.FRV_MANUAL_OVERRIDE_SCHEMA_VERSION,
        "source_content_identity_digest": SHA,
        "override_id": "ov-1",
        "requested_action": 1,
        "requested_original_indexes": [10, 20],
        "requested_geometry_reference": None,
        "requested_geometry_resolution_version": None,
        "requester": "operator-a",
        "request_timestamp": "2026-01-01T00:00:00Z",
        "request_reason": "verified trunk",
        "authority_level": "level-3",
        "predecessor_artifact_digest": None,
        "automatic_state_before_override": 1,
        "automatic_reason_flags": 0,
        "conflict_flags": 0,
        "reviewer": None,
        "review_timestamp": None,
        "approval_state": 1,
        "approval_reason": None,
        "resulting_state": None,
        "superseded_override_digest": None,
    }
    payload.update(overrides)
    payload["override_digest"] = _digest_of(payload, "override_digest")
    return payload


# -- payload schema, blocks, artifact, binding -------------------------------


def payload_arrays() -> list[dict[str, Any]]:
    enum_versions = {
        "membership_state": "ms-v1",
        "reason_flags": "rf-v1",
        "input_quality_flags": "qf-v1",
        "hag_evidence_state": "he-v1",
        "segmentation_hag_readiness": "shr-v1",
    }
    return [
        {
            "field_name": name,
            "dtype_code": dtype,
            "shape_rule": "[N]",
            "required": True,
            "nullable": False,
            "finite_required": False,
            "enum_or_flag_schema_version": enum_versions.get(name),
            "storage_key": name,
        }
        for name, dtype in (
            ("original_index", "<i8"),
            ("membership_state", "|u1"),
            ("reason_flags", "<u4"),
            ("input_quality_flags", "<u4"),
            ("hag_evidence_state", "|u1"),
            ("segmentation_hag_readiness", "|u1"),
            ("scope_union_id_reference", "<u4"),
            ("primary_scope_id_reference", "<u4"),
            ("supporting_region_offset", "<u8"),
            ("supporting_region_count", "<u4"),
            ("manual_override_reference", "<u4"),
        )
    ]


def support_table(name: str = "support-0") -> tuple[str, str]:
    table = (name,)
    return table, frv_digest(table)


def schema_dict(**overrides: Any) -> dict[str, Any]:
    scope_table, scope_digest = support_table("union-0")
    primary_table, primary_digest = support_table("primary-0")
    support_table_value, support_digest = support_table()
    payload = {
        "payload_schema_version": frv.FRV_ARTIFACT_PAYLOAD_SCHEMA_VERSION,
        "block_schema_version": frv.FRV_ARTIFACT_BLOCK_SCHEMA_VERSION,
        "membership_state_schema_version": "ms-v1",
        "reason_flag_schema_version": "rf-v1",
        "quality_flag_schema_version": "qf-v1",
        "hag_evidence_schema_version": "he-v1",
        "segmentation_hag_readiness_schema_version": "shr-v1",
        "payload_encoding_id": "FRV-BLOCK-PAYLOAD-1",
        "payload_encoder_version": "1",
        "payload_compression_algorithm": "NONE",
        "payload_compression_version": "1",
        "arrays": payload_arrays(),
        "scope_union_reference_table": scope_table,
        "scope_union_reference_table_digest": scope_digest,
        "primary_scope_reference_table": primary_table,
        "primary_scope_reference_table_digest": primary_digest,
        "supporting_region_reference_table": support_table_value,
        "supporting_region_reference_table_digest": support_digest,
        "manual_override_reference_table": None,
        "manual_override_reference_table_digest": None,
    }
    payload.update(overrides)
    payload["payload_schema_digest"] = _digest_of(payload, "payload_schema_digest")
    return payload


def block_dict(schema: FRVArtifactPayloadSchema, **overrides: Any) -> dict[str, Any]:
    payload = {
        "block_schema_version": frv.FRV_ARTIFACT_BLOCK_SCHEMA_VERSION,
        "payload_schema_digest": schema.payload_schema_digest,
        "block_id": "block-1",
        "block_digest": SHA,
        "original_index_start": 0,
        "original_index_stop": 10,
        "record_count": 10,
        "first_original_index": 0,
        "last_original_index": 9,
        "storage_reference": "blocks/b1",
        "payload_schema_version": frv.FRV_ARTIFACT_PAYLOAD_SCHEMA_VERSION,
        "array_descriptor_digest": frv_digest([array.to_dict() for array in schema.arrays]),
    }
    payload.update(overrides)
    payload["block_descriptor_digest"] = _digest_of(payload, "block_descriptor_digest")
    return payload


def artifact_dict(
    blocks: list[FRVArtifactBlockDescriptor],
    schema: FRVArtifactPayloadSchema,
    **overrides: Any,
) -> dict[str, Any]:
    payload = {
        "artifact_schema_version": frv.FRV_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION,
        "algorithm_version": "alg-1",
        "decision_rule_version": "rule-1",
        "source_content_identity_digest": SHA,
        "source_identity_record_digest": SHA,
        "source_file_sha256": SHA,
        "canonical_point_content_digest": None,
        "canonical_point_content_spec_digest": None,
        "scope_coverage_manifest_digest": SHA,
        "scope_union_id": "union-1",
        "scope_construction_version": "sc-v1",
        "scope_union_geometry_digest": SHA,
        "canonical_inventory_digest": SHA,
        "required_inventory_digest": SHA,
        "configuration_digest": SHA,
        "payload_schema_digest": schema.payload_schema_digest,
        "block_schema_version": frv.FRV_ARTIFACT_BLOCK_SCHEMA_VERSION,
        "artifact_sha256": "c" * 64,
        "block_descriptors": [block.to_dict() for block in blocks],
        "source_point_count": 1000,
        "unique_in_scope_point_count": 10,
        "evaluated_point_count": 10,
        "verified_vegetation_candidate_count": 6,
        "excluded_non_candidate_count": 2,
        "review_required_membership_count": 1,
        "manual_include_count": 1,
        "manual_exclude_count": 0,
        "unasserted_in_scope_count": 0,
        "canonical_inventory_count": 5,
        "required_inventory_count": 4,
        "approved_hard_rejection_count": 1,
        "scope_created_or_merged_count": 4,
        "coverage_blocked_count": 0,
        "canonical_inventory_complete": True,
        "required_inventory_complete": True,
        "approved_hard_rejection_complete": True,
        "scope_coverage_complete": True,
        "routing_complete": True,
        "membership_decision_complete": True,
        "provider_digest_references": [],
        "override_digest_references": [],
        "block_digest_references": sorted(block.block_digest for block in blocks),
        "warning_digest": SHA,
        "limitation_digest": SHA,
        "provenance_digest": SHA,
    }
    payload.update(overrides)
    id_payload = {
        key: payload[key]
        for key in (
            "artifact_schema_version",
            "algorithm_version",
            "decision_rule_version",
            "source_content_identity_digest",
            "scope_union_id",
            "configuration_digest",
        )
    }
    payload["artifact_id"] = "frv-membership-" + frv_digest(id_payload)
    payload["manifest_digest"] = _digest_of(payload, "manifest_digest")
    return payload


def binding_dict(
    artifact: FullResolutionVegetationMembershipArtifact, **overrides: Any
) -> dict[str, Any]:
    payload = {
        "binding_schema_version": frv.FRV_ARTIFACT_BINDING_SCHEMA_VERSION,
        "source_content_identity_digest": artifact.source_content_identity_digest,
        "source_identity_record_digest": artifact.source_identity_record_digest,
        "scope_union_id": artifact.scope_union_id,
        "configuration_digest": artifact.configuration_digest,
        "semantic_artifact_id": artifact.artifact_id,
        "scope_coverage_manifest_digest": artifact.scope_coverage_manifest_digest,
        "membership_artifact_manifest_digest": artifact.manifest_digest,
        "membership_artifact_sha256": artifact.artifact_sha256,
        "binding_created_by_stage": "A6",
    }
    payload.update(overrides)
    payload["binding_digest"] = _digest_of(payload, "binding_digest")
    return payload


# ---------------------------------------------------------------------------
# 1. Closed enums, flags, and error codes
# ---------------------------------------------------------------------------


def test_membership_state_exact_members_and_values() -> None:
    assert dict(frv.FrvMembershipState.__members__) == {
        "VERIFIED_VEGETATION_CANDIDATE": frv.FrvMembershipState.VERIFIED_VEGETATION_CANDIDATE,
        "EXCLUDED_NON_CANDIDATE": frv.FrvMembershipState.EXCLUDED_NON_CANDIDATE,
        "REVIEW_REQUIRED_MEMBERSHIP": frv.FrvMembershipState.REVIEW_REQUIRED_MEMBERSHIP,
        "MANUAL_INCLUDE": frv.FrvMembershipState.MANUAL_INCLUDE,
        "MANUAL_EXCLUDE": frv.FrvMembershipState.MANUAL_EXCLUDE,
    }
    assert [
        frv.FrvMembershipState.VERIFIED_VEGETATION_CANDIDATE,
        frv.FrvMembershipState.EXCLUDED_NON_CANDIDATE,
        frv.FrvMembershipState.REVIEW_REQUIRED_MEMBERSHIP,
        frv.FrvMembershipState.MANUAL_INCLUDE,
        frv.FrvMembershipState.MANUAL_EXCLUDE,
    ] == [1, 2, 3, 4, 5]


def test_scope_mode_exact_values() -> None:
    assert [frv.FrvScopeMode.EXHAUSTIVE_REGION, frv.FrvScopeMode.EXHAUSTIVE_SOURCE,
            frv.FrvScopeMode.SPARSE_ASSERTIONS_ONLY] == [1, 2, 3]


def test_hag_evidence_state_exact_values() -> None:
    assert [
        frv.FrvHagEvidenceState.HAG_VALID_POSITIVE,
        frv.FrvHagEvidenceState.HAG_VALID_NEAR_ZERO,
        frv.FrvHagEvidenceState.HAG_WITHIN_NEGATIVE_TOLERANCE,
        frv.FrvHagEvidenceState.HAG_MISSING,
        frv.FrvHagEvidenceState.HAG_INVALID,
        frv.FrvHagEvidenceState.HAG_OUT_OF_DOMAIN,
    ] == [1, 2, 3, 4, 5, 6]


def test_segmentation_hag_readiness_exact_values() -> None:
    assert [
        frv.FrvSegmentationHagReadiness.READY,
        frv.FrvSegmentationHagReadiness.REVIEW_REQUIRED,
    ] == [1, 2]


def test_inventory_disposition_exactly_two_values() -> None:
    assert [
        frv.FrvInventoryDisposition.REQUIRED_FRV,
        frv.FrvInventoryDisposition.APPROVED_HARD_REJECTION,
    ] == [1, 2]
    assert len(list(frv.FrvInventoryDisposition)) == 2


def test_scope_construction_state_exact_values() -> None:
    assert [
        frv.FrvScopeConstructionState.SCOPE_CREATED,
        frv.FrvScopeConstructionState.SCOPE_MERGED,
        frv.FrvScopeConstructionState.NOT_CREATED,
    ] == [1, 2, 3]


def test_scope_coverage_state_exact_values_including_reserved_out_of_policy() -> None:
    assert [
        frv.FrvScopeCoverageState.COVERED,
        frv.FrvScopeCoverageState.SCOPE_BLOCKED,
        frv.FrvScopeCoverageState.SOURCE_IDENTITY_FAILED,
        frv.FrvScopeCoverageState.UPSTREAM_EVIDENCE_INVALID,
        frv.FrvScopeCoverageState.REGION_GEOMETRY_INVALID,
        frv.FrvScopeCoverageState.REGION_MAPPING_INCOMPLETE,
        frv.FrvScopeCoverageState.OUT_OF_FRV_POLICY,
    ] == [1, 2, 3, 4, 5, 6, 7]


def test_wire_relation_state_exact_values() -> None:
    assert [
        frv.FrvWireRelationState.WIRE_BODY_CONFIRMED,
        frv.FrvWireRelationState.WIRE_BODY_POSSIBLE,
        frv.FrvWireRelationState.WIRE_PROXIMITY_ONLY,
        frv.FrvWireRelationState.WIRE_VEGETATION_CONFLICT,
    ] == [1, 2, 3, 4]


def test_wire_provider_state_exact_values() -> None:
    assert [
        frv.FrvWireProviderState.WIRE_RELATION_VALID,
        frv.FrvWireProviderState.WIRE_NOT_APPLICABLE_PROVEN,
        frv.FrvWireProviderState.WIRE_PROVIDER_MISSING,
        frv.FrvWireProviderState.WIRE_PROVIDER_INVALID,
        frv.FrvWireProviderState.WIRE_IDENTITY_MISMATCH,
        frv.FrvWireProviderState.WIRE_GEOMETRY_UNAVAILABLE,
    ] == [1, 2, 3, 4, 5, 6]


def test_override_enums_exact_values() -> None:
    assert [
        frv.FrvOverrideRequestedAction.MANUAL_INCLUDE_REQUESTED,
        frv.FrvOverrideRequestedAction.MANUAL_EXCLUDE_REQUESTED,
    ] == [1, 2]
    assert [
        frv.FrvOverrideApprovalState.PENDING_REVIEW,
        frv.FrvOverrideApprovalState.MANUAL_INCLUDE_APPROVED,
        frv.FrvOverrideApprovalState.MANUAL_EXCLUDE_APPROVED,
        frv.FrvOverrideApprovalState.MANUAL_OVERRIDE_REJECTED,
    ] == [1, 2, 3, 4]


def test_override_conflict_flag_exact_bits() -> None:
    assert [
        frv.FrvOverrideConflictFlag.NONE,
        frv.FrvOverrideConflictFlag.SAME_PREDECESSOR_CONFLICT,
        frv.FrvOverrideConflictFlag.MULTIPLE_EFFECTIVE_OVERRIDE,
        frv.FrvOverrideConflictFlag.CHAIN_CYCLE,
        frv.FrvOverrideConflictFlag.UNRESOLVABLE_PREDECESSOR,
        frv.FrvOverrideConflictFlag.ARTIFACT_INTEGRITY_FAILURE,
    ] == [0, 1, 2, 4, 8, 16]


def test_reason_flag_exact_bits() -> None:
    assert [
        frv.FrvReasonFlag.NONE,
        frv.FrvReasonFlag.MANUAL_INCLUDE_PROVENANCE,
        frv.FrvReasonFlag.MANUAL_EXCLUDE_PROVENANCE,
        frv.FrvReasonFlag.INSUFFICIENT_EVIDENCE,
        frv.FrvReasonFlag.GROUND_CONFLICT,
        frv.FrvReasonFlag.VEGETATION_COUNTER_EVIDENCE,
        frv.FrvReasonFlag.WIRE_CONFLICT,
        frv.FrvReasonFlag.SCOPE_EDGE_CONFLICT,
        frv.FrvReasonFlag.HAG_MISSING,
        frv.FrvReasonFlag.HAG_INVALID,
        frv.FrvReasonFlag.HAG_OUT_OF_DOMAIN,
        frv.FrvReasonFlag.PROVIDER_UNCERTAINTY,
        frv.FrvReasonFlag.OVERRIDE_CHAIN_CONFLICT,
        frv.FrvReasonFlag.REVIEW_REQUIRED,
    ] == [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def test_quality_flag_exact_bits() -> None:
    assert [
        frv.FrvQualityFlag.NONE,
        frv.FrvQualityFlag.SOURCE_IDENTITY_FAILURE,
        frv.FrvQualityFlag.ORIGINAL_INDEX_INVALID,
        frv.FrvQualityFlag.ORIGINAL_INDEX_DUPLICATE,
        frv.FrvQualityFlag.STREAM_INCOMPLETE,
        frv.FrvQualityFlag.PROVIDER_MISSING,
        frv.FrvQualityFlag.PROVIDER_INVALID,
        frv.FrvQualityFlag.WIRE_IDENTITY_MISMATCH,
        frv.FrvQualityFlag.WIRE_GEOMETRY_UNAVAILABLE,
        frv.FrvQualityFlag.SCOPE_MAPPING_CONFLICT,
        frv.FrvQualityFlag.SCOPE_GEOMETRY_CONFLICT,
        frv.FrvQualityFlag.ARTIFACT_DIGEST_MISMATCH,
        frv.FrvQualityFlag.MANIFEST_DIGEST_MISMATCH,
        frv.FrvQualityFlag.BLOCK_DIGEST_MISMATCH,
        frv.FrvQualityFlag.OVERRIDE_ARTIFACT_FAILURE,
    ] == [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def test_error_code_exact_inventory() -> None:
    assert {member.value for member in frv.FrvErrorCode} == {
        "FRV_SOURCE_IDENTITY_FAILED",
        "FRV_ORIGINAL_INDEX_INVALID",
        "FRV_ORIGINAL_INDEX_DUPLICATE",
        "FRV_STREAM_INCOMPLETE",
        "FRV_PROVIDER_MISSING",
        "FRV_PROVIDER_INVALID",
        "FRV_WIRE_IDENTITY_MISMATCH",
        "FRV_WIRE_GEOMETRY_UNAVAILABLE",
        "FRV_HAG_MISSING",
        "FRV_HAG_INVALID",
        "FRV_HAG_OUT_OF_DOMAIN",
        "FRV_SCOPE_EDGE_CONFLICT",
        "FRV_SCOPE_MAPPING_INCOMPLETE",
        "FRV_SCOPE_GEOMETRY_INVALID",
        "FRV_ARTIFACT_DIGEST_MISMATCH",
        "FRV_MANIFEST_DIGEST_MISMATCH",
        "FRV_BLOCK_DIGEST_MISMATCH",
        "FRV_OVERRIDE_CHAIN_CONFLICT",
        "FRV_INSUFFICIENT_EVIDENCE",
        "FRV_GROUND_CONFLICT",
        "FRV_VEGETATION_CONFLICT",
        "FRV_WIRE_CONFLICT",
        "FRV_REVIEW_REQUIRED",
    }
    assert len(list(frv.FrvErrorCode)) == 23


def test_safety_completeness_effect_exact_two_values() -> None:
    assert [
        frv.FrvSafetyCompletenessEffect.PRESERVES_SAFETY_COMPLETENESS,
        frv.FrvSafetyCompletenessEffect.BLOCKS_SAFETY_COMPLETENESS,
    ] == [
        "PRESERVES_SAFETY_COMPLETENESS",
        "BLOCKS_SAFETY_COMPLETENESS",
    ]
    with pytest.raises(ValueError):
        frv.FrvSafetyCompletenessEffect("INVALID_EFFECT")


def test_configuration_purpose_exact_three_values() -> None:
    assert {
        frv.FrvConfigurationPurpose.TEST_ONLY,
        frv.FrvConfigurationPurpose.CALIBRATION_CANDIDATE,
        frv.FrvConfigurationPurpose.PRODUCTION_APPROVED,
    } == set(frv.FrvConfigurationPurpose)


# ---------------------------------------------------------------------------
# 2. Bool / numpy.bool_ rejection and unknown enum/flag bits
# ---------------------------------------------------------------------------


def test_integer_fields_reject_bool_and_numpy_bool() -> None:
    payload = source_payload()
    for bad in (True, np.bool_(True)):
        with pytest.raises(ValueError, match="source_point_count"):
            SourceIdentity.from_dict({**payload, "source_point_count": bad})


def test_enum_fields_reject_bool_and_plain_int() -> None:
    payload = config_dict()
    with pytest.raises(ValueError, match="scope_mode"):
        FrvConfiguration.from_dict({**payload, "scope_mode": True})
    with pytest.raises(ValueError, match="scope_mode"):
        FrvConfiguration.from_dict({**payload, "scope_mode": "EXHAUSTIVE_REGION"})


def test_unknown_enum_value_rejected() -> None:
    payload = config_dict()
    with pytest.raises(ValueError):
        FrvConfiguration.from_dict({**payload, "configuration_purpose": "LEGACY_PURPOSE"})


def test_unknown_flag_bit_rejected() -> None:
    payload = required_record_dict()
    with pytest.raises(ValueError, match="unknown flag"):
        FRVScopeCoverageRecord.from_dict({**payload, "failure_reason_flags": 65536})
    with pytest.raises(ValueError, match="unknown flag"):
        frv.validate_frv_flag_mask(65536, frv.FrvReasonFlag, "reason_flags")


def test_combined_known_flag_bits_accepted() -> None:
    combined = int(frv.FrvReasonFlag.GROUND_CONFLICT) | int(frv.FrvReasonFlag.WIRE_CONFLICT)
    record = FRVScopeCoverageRecord.from_dict(blocked_record_dict(failure_reason_flags=combined))
    assert record.failure_reason_flags == combined


# ---------------------------------------------------------------------------
# 3. Frozen dataclasses and immutable nesting
# ---------------------------------------------------------------------------


def test_contracts_are_frozen() -> None:
    config = FrvConfiguration.from_dict(config_dict())
    with pytest.raises(FrozenInstanceError):
        config.scope_mode = frv.FrvScopeMode.EXHAUSTIVE_SOURCE  # type: ignore[misc]


def test_nested_sequences_and_mappings_are_immutable() -> None:
    override = FRVManualOverrideArtifact.from_dict(override_dict())
    assert isinstance(override.requested_original_indexes, tuple)
    with pytest.raises(AttributeError):
        override.requested_original_indexes += (30,)  # type: ignore[union-attr]
    scope = FRVScopeDefinition.from_dict(scope_dict())
    assert isinstance(scope.upstream_reference_digests, tuple)
    with pytest.raises(AttributeError):
        scope.upstream_reference_digests += ("b" * 64,)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 4. Strict from_dict: unknown and missing fields
# ---------------------------------------------------------------------------


def test_from_dict_rejects_unknown_fields() -> None:
    payload = config_dict()
    with pytest.raises(ValueError, match="unknown field"):
        FrvConfiguration.from_dict({**payload, "unplanned_field": 1})


def test_from_dict_rejects_missing_fields() -> None:
    payload = config_dict()
    del payload["scope_mode"]
    with pytest.raises(ValueError, match="missing required"):
        FrvConfiguration.from_dict(payload)


def test_from_dict_rejects_array_digest_as_unknown_field() -> None:
    payload = schema_dict()
    arrays = list(payload["arrays"])
    arrays[0] = {**arrays[0], "array_digest": SHA}
    payload["arrays"] = arrays
    payload["payload_schema_digest"] = frv_digest(
        {key: value for key, value in payload.items() if key != "payload_schema_digest"}
    )
    with pytest.raises(ValueError, match="unknown field"):
        FRVArtifactPayloadSchema.from_dict(payload)


def test_to_dict_returns_new_plain_containers() -> None:
    config = FrvConfiguration.from_dict(config_dict())
    dumped = config.to_dict()
    assert isinstance(dumped, dict)
    assert dumped is not config.__dict__
    payload = schema_dict()
    schema = FRVArtifactPayloadSchema.from_dict(payload)
    assert isinstance(schema.to_dict()["arrays"], list)


# ---------------------------------------------------------------------------
# 5. Canonical JSON determinism and self-digest verification
# ---------------------------------------------------------------------------


def test_canonical_frv_json_bytes_is_deterministic() -> None:
    a = canonical_frv_json_bytes({"b": 1, "a": [2, 3], "c": None})
    b = canonical_frv_json_bytes({"c": None, "a": [2, 3], "b": 1})
    assert a == b
    assert a == b'{"a":[2,3],"b":1,"c":null}'


def test_canonical_frv_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_frv_json_bytes({"value": float("nan")})
    with pytest.raises(ValueError):
        canonical_frv_json_bytes({"value": float("inf")})


def test_canonical_frv_json_rejects_sets_and_non_string_keys() -> None:
    with pytest.raises(ValueError, match="set"):
        canonical_frv_json_bytes({"items": {1, 2}})
    with pytest.raises(ValueError, match="keys must be strings"):
        canonical_frv_json_bytes({1: "x"})


def test_frv_digest_is_sha256_hex() -> None:
    digest = frv_digest({"a": 1})
    assert len(digest) == 64
    assert digest.islower()
    assert all(character in "0123456789abcdef" for character in digest)


def test_self_digest_rejects_wrong_supplied_digest() -> None:
    payload = config_dict()
    payload["configuration_digest"] = "b" * 64
    with pytest.raises(ValueError, match="configuration_digest"):
        FrvConfiguration.from_dict(payload)


# ---------------------------------------------------------------------------
# 6. Configuration purposes and wire-provider binding
# ---------------------------------------------------------------------------


def test_three_configuration_purposes_construct() -> None:
    assert FrvConfiguration.from_dict(config_dict()).configuration_purpose is (
        frv.FrvConfigurationPurpose.PRODUCTION_APPROVED
    )
    calibration = config_dict(
        configuration_purpose="CALIBRATION_CANDIDATE",
        configuration_is_runtime_complete=False,
        formal_completeness_allowed=False,
        membership_artifact_publication_allowed=False,
        wire_provider_config_digest=None,
        scope_mode=3,
    )
    assert FrvConfiguration.from_dict(calibration).configuration_purpose is (
        frv.FrvConfigurationPurpose.CALIBRATION_CANDIDATE
    )
    test_only = config_dict(
        configuration_purpose="TEST_ONLY",
        configuration_is_runtime_complete=False,
        formal_completeness_allowed=False,
        membership_artifact_publication_allowed=False,
        wire_provider_interface_version=None,
        wire_provider_config_digest=None,
        scope_mode=3,
    )
    assert FrvConfiguration.from_dict(test_only).configuration_purpose is (
        frv.FrvConfigurationPurpose.TEST_ONLY
    )


def test_production_wire_provider_binding_fail_closed() -> None:
    config = FrvConfiguration.from_dict(config_dict())
    frv.validate_production_wire_provider_binding(config)
    with pytest.raises(ValueError, match="non-null wire provider binding"):
        validate_configuration(
            FrvConfiguration.from_dict(
                config_dict(wire_provider_config_digest=None)
            )
        )


def test_production_requires_runtime_complete_and_exhaustive_region() -> None:
    with pytest.raises(ValueError, match="runtime complete"):
        validate_configuration(
            FrvConfiguration.from_dict(
                config_dict(configuration_is_runtime_complete=False)
            )
        )
    with pytest.raises(ValueError, match="EXHAUSTIVE_REGION"):
        validate_configuration(
            FrvConfiguration.from_dict(config_dict(scope_mode=3))
        )


def test_calibration_and_test_only_closure() -> None:
    calibration = config_dict(
        configuration_purpose="CALIBRATION_CANDIDATE",
        configuration_is_runtime_complete=False,
        formal_completeness_allowed=False,
        membership_artifact_publication_allowed=False,
        wire_provider_config_digest=None,
        scope_mode=3,
    )
    validate_configuration(FrvConfiguration.from_dict(calibration))
    with pytest.raises(ValueError, match="formal completeness"):
        validate_configuration(
            FrvConfiguration.from_dict(
                config_dict(
                    configuration_purpose="CALIBRATION_CANDIDATE",
                    configuration_is_runtime_complete=False,
                    formal_completeness_allowed=True,
                    membership_artifact_publication_allowed=False,
                    wire_provider_config_digest=None,
                    scope_mode=3,
                )
            )
        )


def test_configuration_cannot_create_wire_not_applicable() -> None:
    # Missing/null wire binding is a provider failure (ValueError), never the
    # WIRE_NOT_APPLICABLE_PROVEN runtime state.
    with pytest.raises(ValueError):
        frv.validate_production_wire_provider_binding(
            FrvConfiguration.from_dict(
                config_dict(wire_provider_config_digest=None)
            )
        )
    assert frv.FrvWireProviderState.WIRE_NOT_APPLICABLE_PROVEN == 2
    assert frv.FrvWireProviderState.WIRE_PROVIDER_MISSING == 3


def test_allow_flags_must_be_exactly_false() -> None:
    for name in (
        "allow_wire_body_hard_exclusion",
        "allow_wire_body_confirmed",
        "allow_review_routing_reduction",
        "allow_source_specific_selector",
    ):
        with pytest.raises(ValueError, match="exactly false"):
            FrvConfiguration.from_dict(config_dict(**{name: True}))


# ---------------------------------------------------------------------------
# 7. Canonical point-content specification
# ---------------------------------------------------------------------------


def test_canonical_point_content_spec_original_index_first_invariant() -> None:
    payload = {
        "spec_schema_version": frv.FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION,
        "spec_id": "point-spec-1",
        "framing_version": "framing-v1",
        "row_order": "ORIGINAL_INDEX_ASCENDING",
        "ordered_field_specs": [
            {
                "field_name": "original_index",
                "semantic_role": "ORIGINAL_INDEX",
                "dtype_code": "<i8",
                "byte_order": "LITTLE_ENDIAN",
                "scale": 1.0,
                "offset": 0.0,
                "required": True,
                "null_policy": "FORBID_NULL",
            },
            {
                "field_name": "height_agl_m",
                "semantic_role": "HAG",
                "dtype_code": "<f8",
                "byte_order": "LITTLE_ENDIAN",
                "scale": 1.0,
                "offset": 0.0,
                "required": True,
                "null_policy": "FORBID_NULL",
            },
        ],
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "quantization_policy": "EXACT_TYPED_VALUE",
        "missing_value_policy": "EXPLICIT_VALIDITY_BYTE",
    }
    payload["spec_digest"] = frv_digest(payload)
    spec = FrvCanonicalPointContentSpec.from_dict(payload)
    assert spec.computed_digest() == spec.spec_digest
    assert spec.ordered_field_specs[0].semantic_role == "ORIGINAL_INDEX"


def test_point_spec_rejects_wrong_first_field() -> None:
    payload = {
        "spec_schema_version": frv.FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION,
        "spec_id": "point-spec-1",
        "framing_version": "framing-v1",
        "row_order": "ORIGINAL_INDEX_ASCENDING",
        "ordered_field_specs": [
            {
                "field_name": "not_original_index",
                "semantic_role": "ORIGINAL_INDEX",
                "dtype_code": "<i8",
                "byte_order": "LITTLE_ENDIAN",
                "scale": 1.0,
                "offset": 0.0,
                "required": True,
                "null_policy": "FORBID_NULL",
            }
        ],
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "quantization_policy": "EXACT_TYPED_VALUE",
        "missing_value_policy": "EXPLICIT_VALIDITY_BYTE",
        "spec_digest": "unused",
    }
    with pytest.raises(ValueError, match="original_index"):
        FrvCanonicalPointContentSpec.from_dict(payload)


def test_point_spec_scaled_integer_rules() -> None:
    with pytest.raises(ValueError, match="integer dtype"):
        FrvCanonicalPointContentSpec.from_dict(
            {
                "spec_schema_version": frv.FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION,
                "spec_id": "s",
                "framing_version": "framing-v1",
                "row_order": "ORIGINAL_INDEX_ASCENDING",
                "ordered_field_specs": [
                    {
                        "field_name": "original_index",
                        "semantic_role": "ORIGINAL_INDEX",
                        "dtype_code": "<i8",
                        "byte_order": "LITTLE_ENDIAN",
                        "scale": 1.0,
                        "offset": 0.0,
                        "required": True,
                        "null_policy": "FORBID_NULL",
                    },
                    {
                        "field_name": "hag",
                        "semantic_role": "HAG",
                        "dtype_code": "<f8",
                        "byte_order": "LITTLE_ENDIAN",
                        "scale": 2.0,
                        "offset": 0.0,
                        "required": True,
                        "null_policy": "FORBID_NULL",
                    },
                ],
                "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
                "units": "m",
                "quantization_policy": "SCALED_INTEGER",
                "missing_value_policy": "EXPLICIT_VALIDITY_BYTE",
                "spec_digest": "unused",
            }
        )


def test_point_spec_rejects_unknown_dtype_and_illegal_null_policy() -> None:
    field = {
        "field_name": "x",
        "semantic_role": "XYZ",
        "dtype_code": "<f16",
        "byte_order": "LITTLE_ENDIAN",
        "scale": 1.0,
        "offset": 0.0,
        "required": True,
        "null_policy": "FORBID_NULL",
    }
    with pytest.raises(ValueError, match="dtype_code"):
        FrvCanonicalPointFieldSpec(**field)
    with pytest.raises(ValueError, match="null_policy"):
        FrvCanonicalPointFieldSpec(
            **{**field, "dtype_code": "<f4", "required": False, "null_policy": "OMIT"}
        )
    with pytest.raises(ValueError, match="FORBID_NULL"):
        FrvCanonicalPointFieldSpec(
            **{**field, "dtype_code": "<f4", "required": True, "null_policy": "VALIDITY_BYTE"}
        )


# ---------------------------------------------------------------------------
# 8. SourceIdentity dual digests and acyclic audits
# ---------------------------------------------------------------------------


def test_source_identity_dual_digests_distinct() -> None:
    source = SourceIdentity.from_dict(source_payload())
    assert source.source_content_identity_digest != source.source_identity_record_digest


def test_path_and_audit_changes_preserve_content_digest() -> None:
    base = SourceIdentity.from_dict(source_payload())
    moved = SourceIdentity.from_dict(source_payload(path="other/relocated.las"))
    assert moved.source_content_identity_digest == base.source_content_identity_digest
    changed_audit = relocation_dict(resolved_final_path="data/final.las")
    audit = FrvRelocationAudit.from_dict(changed_audit)
    audited = SourceIdentity.from_dict(source_payload(relocation=audit))
    assert audited.source_content_identity_digest == base.source_content_identity_digest


def test_audit_changes_alter_record_digest() -> None:
    base = SourceIdentity.from_dict(source_payload())
    audit = FrvRelocationAudit.from_dict(relocation_dict(resolved_final_path="data/final.las"))
    audited = SourceIdentity.from_dict(source_payload(relocation=audit))
    assert audited.source_identity_record_digest != base.source_identity_record_digest


def test_altered_content_changes_content_digest() -> None:
    base = SourceIdentity.from_dict(source_payload())
    altered = SourceIdentity.from_dict(source_payload(content_sha="b" * 64))
    assert altered.source_content_identity_digest != base.source_content_identity_digest


def test_content_digest_ignores_paths_and_audits() -> None:
    source = SourceIdentity.from_dict(source_payload())
    content_payload = source.to_content_digest_dict()
    assert "recorded_source_path" not in content_payload
    assert "resolved_access_path" not in content_payload
    assert "relocation_audit" not in content_payload
    assert "source_equivalence_audit" not in content_payload


def test_equivalence_audit_has_no_record_digest_reference() -> None:
    audit = FrvSourceEquivalenceAudit.from_dict(equiv_dict())
    validate_source_equivalence_audit_without_cycle(audit)
    assert "source_identity_record_digest" not in audit.to_digest_dict()
    source = SourceIdentity.from_dict(source_payload())
    assert audit.audit_digest != source.source_identity_record_digest


def test_record_digest_round_trip_includes_audit_digests() -> None:
    source = SourceIdentity.from_dict(source_payload())
    record_payload = source.to_record_digest_dict()
    assert record_payload["relocation_audit_audit_digest"] == (
        source.relocation_audit.audit_digest
    )
    assert record_payload["source_equivalence_audit_audit_digest"] == (
        source.source_equivalence_audit.audit_digest
    )
    assert record_payload["source_content_identity_digest"] == (
        source.source_content_identity_digest
    )
    validate_source_content_identity_digest(source)
    validate_source_identity_record_digest(source)


def test_canonical_content_tri_state() -> None:
    payload = source_payload()
    payload["canonical_point_content_spec_version"] = None
    with pytest.raises(ValueError, match="all null or all nonnull"):
        SourceIdentity.from_dict(payload)


def test_relocation_audit_warning_flags_zero_only() -> None:
    assert FrvRelocationAudit.from_dict(relocation_dict(warning_flags=0)).warning_flags == 0
    for bad in (1, 2**31, -1, 2**32):
        with pytest.raises(ValueError, match="unknown flag|uint32"):
            FrvRelocationAudit.from_dict(relocation_dict(warning_flags=bad))
    with pytest.raises(ValueError, match="must not be a boolean"):
        FrvRelocationAudit.from_dict(relocation_dict(warning_flags=True))
    with pytest.raises(ValueError, match="must not be a boolean"):
        FrvRelocationAudit.from_dict({**relocation_dict(), "warning_flags": np.bool_(True)})


# ---------------------------------------------------------------------------
# 9. Geometry encoding spec versus content references
# ---------------------------------------------------------------------------


def test_encoding_spec_digest_differs_from_content_digests() -> None:
    spec = FrvScopeGeometryEncodingSpec.from_dict(encoding_dict())
    scope = FRVScopeDefinition.from_dict(scope_dict())
    assert spec.encoding_spec_digest != scope.scope_core_geometry_digest
    assert spec.encoding_spec_digest != scope.scope_context_margin_geometry_digest
    assert scope.scope_core_geometry_digest != scope.scope_context_margin_geometry_digest


def test_scope_geometry_content_reference_validator_rejects_spec_as_content() -> None:
    scope = FRVScopeDefinition.from_dict(
        scope_dict(geometry_encoding_spec_digest=scope_dict()["scope_core_geometry_digest"])
    )
    with pytest.raises(ValueError, match="encoding spec digest"):
        frv.validate_scope_geometry_content_digest_reference(scope)


def test_scope_upstream_references_sorted_unique() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        FRVScopeDefinition.from_dict(scope_dict(upstream_reference_digests=["b" * 64, "a" * 64]))


# ---------------------------------------------------------------------------
# 10. Approved hard rejection evidence
# ---------------------------------------------------------------------------


def test_hard_rejection_requires_complete_approval_fields() -> None:
    evidence = FRVApprovedHardRejectionEvidence.from_dict(evidence_dict())
    assert evidence.evidence_digest != evidence.approval_digest
    for name in ("approved_rule_id", "approved_rule_version", "decision_digest", "approver",
                 "approval_timestamp", "approval_digest", "hard_non_vegetation_evidence_reference"):
        with pytest.raises(ValueError):
            FRVApprovedHardRejectionEvidence.from_dict(evidence_dict(**{name: ""}))


def test_hard_rejection_safety_completeness_effect() -> None:
    evidence = FRVApprovedHardRejectionEvidence.from_dict(
        evidence_dict(safety_completeness_effect="BLOCKS_SAFETY_COMPLETENESS")
    )
    assert evidence.safety_completeness_effect is (
        frv.FrvSafetyCompletenessEffect.BLOCKS_SAFETY_COMPLETENESS
    )
    with pytest.raises(ValueError):
        FRVApprovedHardRejectionEvidence.from_dict(evidence_dict(safety_completeness_effect="X"))


# ---------------------------------------------------------------------------
# 11. Coverage records: disposition exclusivity and binding
# ---------------------------------------------------------------------------


def test_inventory_disposition_exclusive_routes() -> None:
    assert frv.FrvInventoryDisposition.REQUIRED_FRV is not (
        frv.FrvInventoryDisposition.APPROVED_HARD_REJECTION
    )
    required = FRVScopeCoverageRecord.from_dict(required_record_dict())
    assert required.inventory_disposition is frv.FrvInventoryDisposition.REQUIRED_FRV
    rejected = FRVScopeCoverageRecord.from_dict(rejected_record_dict())
    assert rejected.inventory_disposition is frv.FrvInventoryDisposition.APPROVED_HARD_REJECTION


def test_required_row_rejects_evidence_and_missing_digests() -> None:
    with pytest.raises(ValueError, match="cannot carry hard-rejection evidence"):
        FRVScopeCoverageRecord.from_dict(
            required_record_dict(approved_hard_rejection_evidence=evidence_dict())
        )
    with pytest.raises(ValueError, match="required-inventory digests"):
        FRVScopeCoverageRecord.from_dict(
            required_record_dict(required_inventory_record_digest=None)
        )


def test_rejection_row_rejects_required_digests() -> None:
    with pytest.raises(ValueError, match="cannot carry required-inventory digests"):
        FRVScopeCoverageRecord.from_dict(rejected_record_dict(required_inventory_record_digest=SHA))


def test_rejection_evidence_canonical_record_binding() -> None:
    with pytest.raises(ValueError, match="not bound to this canonical key"):
        FRVScopeCoverageRecord.from_dict(
            rejected_record_dict(
                approved_hard_rejection_evidence=evidence_dict(key="different-key")
            )
        )
    with pytest.raises(ValueError, match="not bound to this source"):
        FRVScopeCoverageRecord.from_dict(
            rejected_record_dict(source_content_identity_digest="b" * 64)
        )


def test_out_of_frv_policy_prohibited() -> None:
    with pytest.raises(ValueError, match="OUT_OF_FRV_POLICY"):
        FRVScopeCoverageRecord.from_dict(
            required_record_dict(scope_coverage_state=7, scope_construction_state=3, scope_id=None)
        )
    with pytest.raises(ValueError, match="OUT_OF_FRV_POLICY"):
        FRVScopeCoverageRecord.from_dict(
            rejected_record_dict(out_of_policy_safety_approval_digest=SHA)
        )


# ---------------------------------------------------------------------------
# 12. Coverage manifest: rows, conservation, completeness
# ---------------------------------------------------------------------------


def _required(key: str = "k-2", **overrides: Any) -> FRVScopeCoverageRecord:
    return FRVScopeCoverageRecord.from_dict(required_record_dict(key=key, **overrides))


def _rejected(key: str = "k-3", **overrides: Any) -> FRVScopeCoverageRecord:
    return FRVScopeCoverageRecord.from_dict(rejected_record_dict(key=key, **overrides))


def _blocked(key: str = "k-9", **overrides: Any) -> FRVScopeCoverageRecord:
    return FRVScopeCoverageRecord.from_dict(blocked_record_dict(key=key, **overrides))


def test_manifest_rows_sorted_unique_by_key() -> None:
    manifest = FRVScopeCoverageManifest.from_dict(manifest_dict([_required(), _rejected()]))
    assert [record.canonical_inventory_key for record in manifest.records] == ["k-2", "k-3"]
    with pytest.raises(ValueError, match="sorted"):
        FRVScopeCoverageManifest.from_dict(manifest_dict([_rejected(), _required()]))
    with pytest.raises(ValueError, match="exactly once"):
        FRVScopeCoverageManifest.from_dict(manifest_dict([_required(), _required()]))


def test_manifest_two_conservation_equations() -> None:
    manifest = FRVScopeCoverageManifest.from_dict(manifest_dict([_required(), _rejected()]))
    assert manifest.canonical_inventory_count == (
        manifest.required_inventory_count + manifest.approved_hard_rejection_count
    )
    assert manifest.required_inventory_count == (
        manifest.scope_created_or_merged_count + manifest.coverage_blocked_count
    )


def test_manifest_derives_counts_and_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="canonical_inventory_count"):
        FRVScopeCoverageManifest.from_dict(
            manifest_dict([_required(), _rejected()], canonical_inventory_count=99)
        )
    with pytest.raises(ValueError, match="required_inventory_count"):
        FRVScopeCoverageManifest.from_dict(
            manifest_dict([_required(), _rejected()], required_inventory_count=0)
        )


def test_manifest_completeness_flags_match_rows() -> None:
    healthy = FRVScopeCoverageManifest.from_dict(manifest_dict([_required(), _rejected()]))
    assert healthy.scope_coverage_complete is True
    assert healthy.routing_complete is True
    blocked = FRVScopeCoverageManifest.from_dict(
        manifest_dict(
            [_blocked()],
            required_inventory_digest=SHA,
            scope_union_id=None,
            scope_union_geometry_digest=None,
        )
    )
    assert blocked.scope_coverage_complete is False
    assert blocked.routing_complete is False
    with pytest.raises(ValueError, match="scope_coverage_complete"):
        FRVScopeCoverageManifest.from_dict(
            manifest_dict(
                [_blocked()],
                required_inventory_digest=SHA,
                scope_union_id=None,
                scope_union_geometry_digest=None,
                scope_coverage_complete=True,
                routing_complete=True,
            )
        )


def test_validate_scope_inventory_conservation_passes_healthy_manifest() -> None:
    manifest = FRVScopeCoverageManifest.from_dict(manifest_dict([_required(), _rejected()]))
    validate_scope_inventory_conservation(manifest)


def test_b5_empty_inventory_case() -> None:
    empty = FRVScopeCoverageManifest.from_dict(
        manifest_dict(
            [],
            required_inventory_digest=None,
            scope_union_id=None,
            scope_union_geometry_digest=None,
        )
    )
    frv.validate_empty_inventory_scope_artifact_case(empty)
    assert empty.canonical_inventory_count == 0
    assert empty.required_inventory_count == 0
    assert empty.approved_hard_rejection_count == 0


def test_b5_all_approved_case() -> None:
    approved = FRVScopeCoverageManifest.from_dict(
        manifest_dict(
            [_rejected()],
            required_inventory_digest=None,
            scope_union_id=None,
            scope_union_geometry_digest=None,
        )
    )
    frv.validate_empty_inventory_scope_artifact_case(approved)
    assert approved.approved_hard_rejection_count == approved.canonical_inventory_count
    assert approved.required_inventory_count == 0


# ---------------------------------------------------------------------------
# 13. Manual override request/approval/conflict structure
# ---------------------------------------------------------------------------


def test_override_index_and_geometry_exclusivity() -> None:
    FRVManualOverrideArtifact.from_dict(override_dict())
    FRVManualOverrideArtifact.from_dict(
        override_dict(
            requested_original_indexes=[],
            requested_geometry_reference="geo-1",
            requested_geometry_resolution_version="res-1",
        )
    )
    with pytest.raises(ValueError, match="either indexes or geometry"):
        FRVManualOverrideArtifact.from_dict(
            override_dict(
                requested_original_indexes=[1],
                requested_geometry_reference="geo-1",
                requested_geometry_resolution_version="res-1",
            )
        )
    with pytest.raises(ValueError, match="must request indexes or geometry"):
        FRVManualOverrideArtifact.from_dict(override_dict(requested_original_indexes=[]))


def test_override_requested_indexes_sorted_unique() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        FRVManualOverrideArtifact.from_dict(override_dict(requested_original_indexes=[20, 10]))
    with pytest.raises(ValueError, match="sorted and unique"):
        FRVManualOverrideArtifact.from_dict(override_dict(requested_original_indexes=[10, 10]))
    with pytest.raises(ValueError, match="must not be a boolean"):
        FRVManualOverrideArtifact.from_dict(override_dict(requested_original_indexes=[True]))


def test_override_approval_state_rules() -> None:
    FRVManualOverrideArtifact.from_dict(
        override_dict(
            approval_state=2,
            resulting_state=4,
            reviewer="reviewer-b",
            review_timestamp="2026-01-02T00:00:00Z",
            approval_reason="approved",
        )
    )
    with pytest.raises(ValueError, match="MANUAL_INCLUDE result"):
        FRVManualOverrideArtifact.from_dict(
            override_dict(
                approval_state=2,
                resulting_state=5,
                reviewer="r",
                review_timestamp="t",
                approval_reason="x",
            )
        )
    with pytest.raises(ValueError, match="null reviewer fields"):
        FRVManualOverrideArtifact.from_dict(override_dict(reviewer="r"))
    FRVManualOverrideArtifact.from_dict(
        override_dict(approval_state=4, reviewer="r", review_timestamp="t", approval_reason="x")
    )
    with pytest.raises(ValueError, match="no resulting state"):
        FRVManualOverrideArtifact.from_dict(
            override_dict(
                approval_state=4,
                reviewer="r",
                review_timestamp="t",
                approval_reason="x",
                resulting_state=4,
            )
        )


def test_override_conflict_prohibits_terminal_result() -> None:
    with pytest.raises(ValueError, match="conflicted overrides"):
        FRVManualOverrideArtifact.from_dict(
            override_dict(
                conflict_flags=8,
                approval_state=2,
                resulting_state=4,
                reviewer="r",
                review_timestamp="t",
                approval_reason="x",
            )
        )


def test_override_chain_descriptor_topology() -> None:
    root = FRVManualOverrideArtifact.from_dict(override_dict())
    successor = FRVManualOverrideArtifact.from_dict(
        override_dict(
            override_id="ov-2",
            predecessor_artifact_digest=root.override_digest,
            approval_state=2,
            resulting_state=4,
            reviewer="r",
            review_timestamp="t",
            approval_reason="x",
        )
    )
    frv.validate_override_chain_descriptors([root, successor])
    bad = FRVManualOverrideArtifact.from_dict(
        override_dict(override_id="ov-3", predecessor_artifact_digest="b" * 64)
    )
    with pytest.raises(ValueError, match="UNRESOLVABLE_PREDECESSOR"):
        frv.validate_override_chain_descriptors([bad])


def test_override_cycle_members_empty_for_content_addressed_chain() -> None:
    root = FRVManualOverrideArtifact.from_dict(override_dict())
    successor = FRVManualOverrideArtifact.from_dict(
        override_dict(
            override_id="ov-2",
            predecessor_artifact_digest=root.override_digest,
            approval_state=2,
            resulting_state=4,
            reviewer="r",
            review_timestamp="t",
            approval_reason="x",
        )
    )
    assert frv._chain_cycle_members([root, successor]) == set()


# ---------------------------------------------------------------------------
# 14. Payload schema, supporting-region schema, and blocks
# ---------------------------------------------------------------------------


def _schema() -> FRVArtifactPayloadSchema:
    return FRVArtifactPayloadSchema.from_dict(schema_dict())


def _block(schema: FRVArtifactPayloadSchema, **overrides: Any) -> FRVArtifactBlockDescriptor:
    return FRVArtifactBlockDescriptor.from_dict(block_dict(schema, **overrides))


def _artifact(
    blocks: list[FRVArtifactBlockDescriptor],
    schema: FRVArtifactPayloadSchema,
    **overrides: Any,
) -> FullResolutionVegetationMembershipArtifact:
    return FullResolutionVegetationMembershipArtifact.from_dict(
        artifact_dict(blocks, schema, **overrides)
    )


def test_required_payload_array_inventory() -> None:
    schema = _schema()
    by_name = {array.field_name: array for array in schema.arrays}
    expected_dtypes = {
        "original_index": "<i8",
        "membership_state": "|u1",
        "reason_flags": "<u4",
        "input_quality_flags": "<u4",
        "hag_evidence_state": "|u1",
        "segmentation_hag_readiness": "|u1",
        "scope_union_id_reference": "<u4",
        "primary_scope_id_reference": "<u4",
        "supporting_region_offset": "<u8",
        "supporting_region_count": "<u4",
        "manual_override_reference": "<u4",
    }
    assert set(by_name) == set(expected_dtypes)
    for name, dtype in expected_dtypes.items():
        assert by_name[name].dtype_code == dtype


def test_payload_schema_rejects_unknown_or_old_array() -> None:
    payload = schema_dict()
    payload["arrays"] = [*payload["arrays"], {
        "field_name": "legacy_supporting_upstream_region_reference",
        "dtype_code": "<u8",
        "shape_rule": "[N]",
        "required": False,
        "nullable": False,
        "finite_required": False,
        "enum_or_flag_schema_version": None,
        "storage_key": "legacy",
    }]
    payload["payload_schema_digest"] = _digest_of(payload, "payload_schema_digest")
    with pytest.raises(ValueError, match="unknown or old array"):
        FRVArtifactPayloadSchema.from_dict(payload)


def test_payload_schema_fixed_first_version_values() -> None:
    schema = _schema()
    assert schema.payload_encoding_id == "FRV-BLOCK-PAYLOAD-1"
    assert schema.payload_encoder_version == "1"
    assert schema.payload_compression_algorithm == "NONE"
    assert schema.payload_compression_version == "1"
    with pytest.raises(ValueError, match="NONE"):
        FRVArtifactPayloadSchema.from_dict(schema_dict(payload_compression_algorithm="GZIP"))


def test_supporting_region_offset_count_schema() -> None:
    schema = _schema()
    frv.validate_supporting_region_ragged_reference_schema(schema)
    by_name = {array.field_name: array for array in schema.arrays}
    assert by_name["supporting_region_offset"].dtype_code == "<u8"
    assert by_name["supporting_region_count"].dtype_code == "<u4"
    with pytest.raises(ValueError, match="sorted and unique"):
        FRVArtifactPayloadSchema.from_dict(
            schema_dict(supporting_region_reference_table=("z", "a"))
        )
    with pytest.raises(ValueError, match="sorted and unique"):
        FRVArtifactPayloadSchema.from_dict(
            schema_dict(supporting_region_reference_table=("a", "a"))
        )


def test_reference_table_digest_must_match_table() -> None:
    with pytest.raises(ValueError, match="must match"):
        FRVArtifactPayloadSchema.from_dict(
            schema_dict(scope_union_reference_table_digest="b" * 64)
        )


def test_block_range_record_count_and_first_last() -> None:
    schema = _schema()
    block = _block(schema)
    assert block.original_index_start == 0
    assert block.original_index_stop == 10
    assert block.record_count == 10
    assert block.first_original_index == 0
    assert block.last_original_index == 9
    # Non-contiguous record_count is allowed (no contiguity assumption).
    _block(
        schema,
        block_id="noncontig",
        original_index_start=0,
        original_index_stop=100,
        record_count=10,
        first_original_index=5,
        last_original_index=14,
    )
    # Empty descriptor requires null first/last and an empty [start, start) range.
    empty = _block(
        schema,
        block_id="empty",
        original_index_start=5,
        original_index_stop=5,
        record_count=0,
        first_original_index=None,
        last_original_index=None,
    )
    assert empty.record_count == 0
    assert empty.first_original_index is None
    assert empty.last_original_index is None
    with pytest.raises(ValueError, match="null first/last"):
        _block(
            schema,
            block_id="bad-empty",
            original_index_start=5,
            original_index_stop=5,
            record_count=0,
            first_original_index=0,
            last_original_index=None,
        )
    with pytest.raises(ValueError, match=r"empty \[start, start\)"):
        _block(
            schema,
            block_id="bad-range",
            original_index_start=0,
            original_index_stop=5,
            record_count=0,
            first_original_index=None,
            last_original_index=None,
        )
    with pytest.raises(ValueError, match="require first/last"):
        _block(
            schema,
            block_id="bad-null",
            original_index_start=0,
            original_index_stop=10,
            record_count=10,
            first_original_index=None,
            last_original_index=None,
        )
    with pytest.raises(ValueError, match="start < stop"):
        _block(
            schema,
            block_id="bad-stop",
            original_index_start=5,
            original_index_stop=5,
            record_count=10,
            first_original_index=5,
            last_original_index=14,
        )
    with pytest.raises(ValueError, match="at least original_index_start"):
        _block(
            schema,
            block_id="bad-first",
            original_index_start=10,
            original_index_stop=20,
            record_count=10,
            first_original_index=5,
            last_original_index=14,
        )
    with pytest.raises(ValueError, match="must not exceed"):
        _block(
            schema,
            block_id="bad-order",
            original_index_start=0,
            original_index_stop=20,
            record_count=10,
            first_original_index=15,
            last_original_index=5,
        )
    with pytest.raises(ValueError, match="below original_index_stop"):
        _block(
            schema,
            block_id="bad-last",
            original_index_start=0,
            original_index_stop=10,
            record_count=10,
            first_original_index=5,
            last_original_index=15,
        )
    with pytest.raises(ValueError, match="must not be a boolean"):
        _block(schema, record_count=True)


def test_block_descriptor_inventory_order_and_non_overlap() -> None:
    schema = _schema()
    first = _block(
        schema,
        block_id="b1",
        original_index_start=0,
        original_index_stop=10,
        record_count=10,
        first_original_index=0,
        last_original_index=9,
    )
    second = _block(
        schema,
        block_id="b2",
        block_digest="b" * 64,
        original_index_start=10,
        original_index_stop=20,
        record_count=10,
        first_original_index=10,
        last_original_index=19,
    )
    frv.validate_block_descriptor_inventory([first, second])
    with pytest.raises(ValueError, match="ordered"):
        frv.validate_block_descriptor_inventory([second, first])
    overlapping = _block(
        schema,
        block_id="b3",
        original_index_start=5,
        original_index_stop=15,
        record_count=10,
        first_original_index=5,
        last_original_index=14,
    )
    with pytest.raises(ValueError, match="overlap"):
        frv.validate_block_descriptor_inventory([first, overlapping])


# ---------------------------------------------------------------------------
# 15. Membership artifact: semantic ID, conservation, digest domain
# ---------------------------------------------------------------------------


def test_semantic_artifact_id_derivation() -> None:
    schema = _schema()
    block = _block(schema)
    artifact = _artifact([block], schema)
    assert artifact.artifact_id == artifact.computed_artifact_id()
    assert artifact.artifact_id.startswith("frv-membership-")
    payload = artifact_dict([block], schema)
    payload["artifact_id"] = "frv-membership-" + "b" * 64
    with pytest.raises(ValueError, match="artifact_id"):
        FullResolutionVegetationMembershipArtifact.from_dict(payload)


def test_artifact_id_stable_under_relocation_only() -> None:
    base = SourceIdentity.from_dict(source_payload())
    moved_audit = FrvRelocationAudit.from_dict(
        relocation_dict(resolved_final_path="data/final.las")
    )
    moved = SourceIdentity.from_dict(source_payload(relocation=moved_audit))
    schema = _schema()
    first = _artifact(
        [ _block(schema) ],
        schema,
        source_content_identity_digest=base.source_content_identity_digest,
        source_identity_record_digest=base.source_identity_record_digest,
    )
    second = _artifact(
        [ _block(schema, block_digest="b" * 64) ],
        schema,
        source_content_identity_digest=moved.source_content_identity_digest,
        source_identity_record_digest=moved.source_identity_record_digest,
    )
    assert second.source_identity_record_digest != first.source_identity_record_digest
    assert second.artifact_id == first.artifact_id


def test_five_state_count_conservation() -> None:
    schema = _schema()
    block = _block(schema)
    artifact = _artifact([block], schema)
    frv.validate_membership_state_count_conservation(artifact)
    with pytest.raises(ValueError, match="sum"):
        _artifact([block], schema, verified_vegetation_candidate_count=7)
    with pytest.raises(ValueError, match="evaluated_point_count"):
        _artifact([block], schema, evaluated_point_count=11)


def test_artifact_manifest_digest_includes_artifact_sha256() -> None:
    schema = _schema()
    block = _block(schema)
    artifact = _artifact([block], schema)
    assert artifact.to_digest_dict()["artifact_sha256"] == artifact.artifact_sha256
    altered = _artifact([block], schema, artifact_sha256="d" * 64)
    assert altered.manifest_digest != artifact.manifest_digest


def test_artifact_requires_nonempty_blocks_and_zero_unasserted() -> None:
    schema = _schema()
    with pytest.raises(ValueError, match="nonempty block"):
        _artifact([], schema)
    with pytest.raises(ValueError, match="zero unasserted"):
        _artifact([_block(schema)], schema, unasserted_in_scope_count=1)
    with pytest.raises(ValueError, match="positive in-scope"):
        _artifact([_block(schema)], schema, unique_in_scope_point_count=0)


def test_artifact_block_digest_references_match_descriptors() -> None:
    schema = _schema()
    with pytest.raises(ValueError, match="block_digest_references"):
        _artifact([_block(schema)], schema, block_digest_references=[])


def test_validate_artifact_payload_digest_domain() -> None:
    schema = _schema()
    block = _block(schema)
    artifact = _artifact([block], schema)
    frv.validate_artifact_payload_digest_domain(artifact)
    # The semantic artifact ID and block ordering are enforced by this
    # validator; the artifact constructor already rejects a tampered ID.
    assert artifact.artifact_id == artifact.computed_artifact_id()


# ---------------------------------------------------------------------------
# 16. Binding record consistency and public imports
# ---------------------------------------------------------------------------


def test_binding_record_consistency() -> None:
    schema = _schema()
    block = _block(schema)
    artifact = _artifact([block], schema)
    binding = FRVArtifactBindingRecord.from_dict(binding_dict(artifact))
    frv.validate_artifact_binding_record(binding, artifact)
    with pytest.raises(ValueError, match="artifact sha256"):
        frv.validate_artifact_binding_record(
            FRVArtifactBindingRecord.from_dict(
                binding_dict(artifact, membership_artifact_sha256="d" * 64)
            ),
            artifact,
        )
    with pytest.raises(ValueError, match="manifest digest"):
        frv.validate_artifact_binding_record(
            FRVArtifactBindingRecord.from_dict(
                binding_dict(artifact, membership_artifact_manifest_digest="d" * 64)
            ),
            artifact,
        )


def test_binding_digest_excludes_only_itself() -> None:
    schema = _schema()
    artifact = _artifact([_block(schema)], schema)
    binding = FRVArtifactBindingRecord.from_dict(binding_dict(artifact))
    assert binding.to_digest_dict()["membership_artifact_sha256"] == artifact.artifact_sha256
    assert "binding_digest" not in binding.to_digest_dict()


def test_public_import_smoke() -> None:
    import core.contracts as contracts

    expected_exports = (
        "FRVApprovedHardRejectionEvidence",
        "FRVArtifactArrayDescriptor",
        "FRVArtifactBindingRecord",
        "FRVArtifactBlockDescriptor",
        "FRVArtifactPayloadSchema",
        "FRVManualOverrideArtifact",
        "FRVScopeCoverageManifest",
        "FRVScopeCoverageRecord",
        "FRVScopeDefinition",
        "FrvCanonicalPointContentSpec",
        "FrvCanonicalPointFieldSpec",
        "FrvConfiguration",
        "FrvConfigurationPurpose",
        "FrvErrorCode",
        "FrvHagEvidenceState",
        "FrvInventoryDisposition",
        "FrvMembershipState",
        "FrvOverrideApprovalState",
        "FrvOverrideConflictFlag",
        "FrvOverrideRequestedAction",
        "FrvQualityFlag",
        "FrvReasonFlag",
        "FrvRelocationAudit",
        "FrvSafetyCompletenessEffect",
        "FrvScopeConstructionState",
        "FrvScopeCoverageState",
        "FrvScopeGeometryEncodingSpec",
        "FrvScopeMode",
        "FrvSegmentationHagReadiness",
        "FrvSourceEquivalenceAudit",
        "FrvWireProviderState",
        "FrvWireRelationState",
        "FullResolutionVegetationMembershipArtifact",
        "SourceIdentity",
        "canonical_frv_json_bytes",
        "frv_digest",
    )
    assert all(hasattr(contracts, name) for name in expected_exports)
    assert contracts.FrvMembershipState is frv.FrvMembershipState
    assert contracts.FrvErrorCode is frv.FrvErrorCode
    assert contracts.canonical_frv_json_bytes is frv.canonical_frv_json_bytes


def test_v2_v3_contracts_remain_importable_and_unchanged() -> None:
    from core.contracts.i1 import I1_SCHEMA_VERSION
    from core.contracts.tree_artifacts import TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION
    from core.contracts.tree_inputs import TreeSegmentationInputIdentity

    assert I1_SCHEMA_VERSION == "i1-bundle-v1"
    assert TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION == "tree-segmentation-artifact-v1"
    assert TreeSegmentationInputIdentity.__dataclass_fields__["source_id"]


# ---------------------------------------------------------------------------
# 17. Targeted review-fix tests (R-001 .. R-012)
# ---------------------------------------------------------------------------


def _point_spec_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "spec_schema_version": frv.FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION,
        "spec_id": "point-spec-1",
        "framing_version": "framing-v1",
        "row_order": "ORIGINAL_INDEX_ASCENDING",
        "ordered_field_specs": [
            {
                "field_name": "original_index",
                "semantic_role": "ORIGINAL_INDEX",
                "dtype_code": "<i8",
                "byte_order": "LITTLE_ENDIAN",
                "scale": 1.0,
                "offset": 0.0,
                "required": True,
                "null_policy": "FORBID_NULL",
            }
        ],
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "quantization_policy": "EXACT_TYPED_VALUE",
        "missing_value_policy": "EXPLICIT_VALIDITY_BYTE",
    }
    payload.update(overrides)
    payload["spec_digest"] = _digest_of(payload, "spec_digest")
    return payload


def _frvart1_header(
    artifact: FullResolutionVegetationMembershipArtifact, **overrides: Any
) -> dict[str, Any]:
    header: dict[str, Any] = {
        "magic": "FRVART1",
        "artifact_schema_version": artifact.artifact_schema_version,
        "algorithm_version": artifact.algorithm_version,
        "decision_rule_version": artifact.decision_rule_version,
        "artifact_id": artifact.artifact_id,
        "source_content_identity_digest": artifact.source_content_identity_digest,
        "scope_union_id": artifact.scope_union_id,
        "configuration_digest": artifact.configuration_digest,
        "payload_schema_digest": artifact.payload_schema_digest,
        "block_schema_version": artifact.block_schema_version,
        "block_count": len(artifact.block_descriptors),
        "ordered_block_ids": tuple(
            block.block_id for block in artifact.block_descriptors
        ),
    }
    header.update(overrides)
    return header


# -- R-001 production calibration digest --------------------------------------


def test_production_requires_non_null_calibration_digest() -> None:
    with pytest.raises(ValueError, match="calibration"):
        FrvConfiguration.from_dict(config_dict(calibration_record_digest=None))
    with pytest.raises(ValueError, match="calibration"):
        FrvConfiguration.from_dict(config_dict(calibration_record_digest="not-a-sha"))


def test_calibration_and_test_only_may_allow_null_calibration() -> None:
    calibration = config_dict(
        configuration_purpose="CALIBRATION_CANDIDATE",
        configuration_is_runtime_complete=False,
        formal_completeness_allowed=False,
        membership_artifact_publication_allowed=False,
        wire_provider_config_digest=None,
        scope_mode=3,
        calibration_record_digest=None,
    )
    assert FrvConfiguration.from_dict(calibration).calibration_record_digest is None


# -- R-002 relocation warning mask --------------------------------------------


def test_relocation_warning_mask_zero_only_matrix() -> None:
    valid = FrvRelocationAudit.from_dict(relocation_dict())
    assert valid.warning_flags == 0
    for bad in (1, 2, 2**31, -1, 2**32, 0xFFFFFFFF):
        with pytest.raises(ValueError):
            FrvRelocationAudit.from_dict(relocation_dict(warning_flags=bad))


# -- R-003 contract schema/version exact binding ------------------------------


def test_configuration_schema_references_exact_binding() -> None:
    mismatches = {
        "schema_version": "frv-configuration-v2",
        "canonicalization_version": "canon-2",
        "source_identity_schema_version": "frv-source-identity-v2",
        "canonical_point_content_spec_version": "frv-canonical-point-content-v2",
        "scope_schema_version": "frv-scope-definition-v2",
        "scope_geometry_encoding_version": "frv-scope-geometry-encoding-v2",
        "coverage_manifest_schema_version": "frv-scope-coverage-manifest-v2",
        "artifact_schema_version": "frv-membership-artifact-v2",
        "block_schema_version": "frv-artifact-block-v2",
        "payload_schema_version": "frv-artifact-payload-v2",
        "override_schema_version": "frv-manual-override-v2",
    }
    for name, bad in mismatches.items():
        with pytest.raises(ValueError, match=name):
            FrvConfiguration.from_dict(config_dict(**{name: bad}))
    with pytest.raises(ValueError, match="schema_version"):
        FrvConfiguration.from_dict(config_dict(schema_version=""))


def test_payload_schema_enum_flag_schema_versions_exact() -> None:
    mismatches = {
        "membership_state_schema_version": "ms-v2",
        "reason_flag_schema_version": "rf-v2",
        "quality_flag_schema_version": "qf-v2",
        "hag_evidence_schema_version": "he-v2",
        "segmentation_hag_readiness_schema_version": "shr-v2",
    }
    for name, bad in mismatches.items():
        with pytest.raises(ValueError, match=name):
            FRVArtifactPayloadSchema.from_dict(schema_dict(**{name: bad}))


def test_payload_block_schema_reference_mismatch() -> None:
    with pytest.raises(ValueError, match="block schema version"):
        FRVArtifactPayloadSchema.from_dict(
            schema_dict(block_schema_version="frv-artifact-block-v2")
        )


def test_point_spec_rejects_unknown_framing_version() -> None:
    with pytest.raises(ValueError, match="framing version"):
        FrvCanonicalPointContentSpec.from_dict(_point_spec_payload(framing_version="framing-v2"))


# -- R-004 geometry literals and three-role separation ------------------------


def test_geometry_encoding_rejects_unknown_literals() -> None:
    mismatches = {
        "box_order_rule": "other-order",
        "float_encoding": "FLOAT32",
        "byte_order": "BIG_ENDIAN",
        "negative_zero_policy": "keep-negative-zero",
        "box_framing_version": "box-framing-v2",
    }
    for name, bad in mismatches.items():
        with pytest.raises(ValueError, match="unsupported"):
            FrvScopeGeometryEncodingSpec.from_dict(encoding_dict(**{name: bad}))


def test_geometry_three_roles_non_interchangeable() -> None:
    scope = FRVScopeDefinition.from_dict(scope_dict())
    frv.validate_scope_geometry_content_digest_reference(
        scope, scope_union_geometry_digest="e" * 64
    )
    # union collides with core / context / spec.
    with pytest.raises(ValueError, match="differ by role"):
        frv.validate_scope_geometry_content_digest_reference(
            scope, scope_union_geometry_digest="c" * 64
        )
    with pytest.raises(ValueError, match="differ by role"):
        frv.validate_scope_geometry_content_digest_reference(
            scope, scope_union_geometry_digest="d" * 64
        )
    with pytest.raises(ValueError, match="encoding spec digest"):
        frv.validate_scope_geometry_content_digest_reference(
            scope, scope_union_geometry_digest="b" * 64
        )
    # core and context collide.
    with pytest.raises(ValueError, match="differ by role"):
        frv.validate_scope_geometry_content_digest_reference(
            FRVScopeDefinition.from_dict(
                scope_dict(scope_context_margin_geometry_digest="c" * 64)
            )
        )


# -- R-005 coverage safety closure --------------------------------------------


def test_covered_required_row_safety_closure_matrix() -> None:
    with pytest.raises(ValueError, match="preserve safety completeness"):
        FRVScopeCoverageRecord.from_dict(
            required_record_dict(safety_completeness_effect="BLOCKS_SAFETY_COMPLETENESS")
        )
    with pytest.raises(ValueError, match="cannot require review"):
        FRVScopeCoverageRecord.from_dict(required_record_dict(review_required=True))
    with pytest.raises(ValueError, match="NONE failure flags"):
        FRVScopeCoverageRecord.from_dict(required_record_dict(failure_reason_flags=16))


def test_blocked_required_row_safety_closure_matrix() -> None:
    with pytest.raises(ValueError, match="block safety completeness"):
        FRVScopeCoverageRecord.from_dict(
            blocked_record_dict(safety_completeness_effect="PRESERVES_SAFETY_COMPLETENESS")
        )
    with pytest.raises(ValueError, match="must require review"):
        FRVScopeCoverageRecord.from_dict(blocked_record_dict(review_required=False))
    with pytest.raises(ValueError, match="nonzero failure flags"):
        FRVScopeCoverageRecord.from_dict(blocked_record_dict(failure_reason_flags=0))


# -- R-006 exact payload array descriptors ------------------------------------


def test_payload_schema_duplicate_array_name_rejected() -> None:
    payload = schema_dict()
    payload["arrays"] = [*payload["arrays"], payload["arrays"][0]]
    payload["payload_schema_digest"] = _digest_of(payload, "payload_schema_digest")
    with pytest.raises(ValueError, match="unique"):
        FRVArtifactPayloadSchema.from_dict(payload)


def test_payload_schema_shape_rule_exact() -> None:
    payload = schema_dict()
    for array in payload["arrays"]:
        if array["field_name"] == "original_index":
            array["shape_rule"] = "[M]"
    payload["payload_schema_digest"] = _digest_of(payload, "payload_schema_digest")
    with pytest.raises(ValueError, match="shape_rule"):
        FRVArtifactPayloadSchema.from_dict(payload)


def test_payload_schema_required_and_nullable_exact() -> None:
    for bad in ({"required": False}, {"nullable": True}):
        payload = schema_dict()
        for array in payload["arrays"]:
            if array["field_name"] == "membership_state":
                array.update(bad)
        payload["payload_schema_digest"] = _digest_of(payload, "payload_schema_digest")
        with pytest.raises(ValueError):
            FRVArtifactPayloadSchema.from_dict(payload)


def test_payload_schema_empty_storage_key_rejected() -> None:
    payload = schema_dict()
    for array in payload["arrays"]:
        if array["field_name"] == "original_index":
            array["storage_key"] = ""
    payload["payload_schema_digest"] = _digest_of(payload, "payload_schema_digest")
    with pytest.raises(ValueError, match="storage_key"):
        FRVArtifactPayloadSchema.from_dict(payload)


def test_payload_schema_wrong_enum_flag_version_rejected() -> None:
    payload = schema_dict()
    for array in payload["arrays"]:
        if array["field_name"] == "membership_state":
            array["enum_or_flag_schema_version"] = "ms-v2"
    payload["payload_schema_digest"] = _digest_of(payload, "payload_schema_digest")
    with pytest.raises(ValueError, match="enum_or_flag_schema_version"):
        FRVArtifactPayloadSchema.from_dict(payload)


def test_reference_table_rejects_empty_and_whitespace_entries() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FRVArtifactPayloadSchema.from_dict(schema_dict(scope_union_reference_table=("",)))
    with pytest.raises(ValueError, match="whitespace"):
        FRVArtifactPayloadSchema.from_dict(
            schema_dict(scope_union_reference_table=("  union-0  ",))
        )
    with pytest.raises(ValueError, match="sorted and unique"):
        FRVArtifactPayloadSchema.from_dict(
            schema_dict(scope_union_reference_table=("b", "a"))
        )


# -- R-007 canonical point field uniqueness -----------------------------------


def test_point_spec_rejects_duplicate_field_names() -> None:
    first = _point_spec_payload()["ordered_field_specs"][0]
    duplicate = _point_spec_payload(ordered_field_specs=[first, first])
    with pytest.raises(ValueError, match="field names must be unique"):
        FrvCanonicalPointContentSpec.from_dict(duplicate)
    hag = {
        "field_name": "height_agl_m",
        "semantic_role": "HAG",
        "dtype_code": "<f8",
        "byte_order": "LITTLE_ENDIAN",
        "scale": 1.0,
        "offset": 0.0,
        "required": True,
        "null_policy": "FORBID_NULL",
    }
    duplicate_non_first = _point_spec_payload(ordered_field_specs=[first, hag, hag])
    with pytest.raises(ValueError, match="field names must be unique"):
        FrvCanonicalPointContentSpec.from_dict(duplicate_non_first)


# -- R-008 source equivalence binding -----------------------------------------


def test_equivalence_audit_current_content_binding() -> None:
    base = source_payload()
    with pytest.raises(ValueError, match="current_source_content_identity_digest"):
        SourceIdentity.from_dict(
            source_payload(
                equivalence=FrvSourceEquivalenceAudit.from_dict(
                    equiv_dict(current_source_content_identity_digest="b" * 64)
                )
            )
        )
    with pytest.raises(ValueError, match="current_source_file_sha256"):
        SourceIdentity.from_dict(
            source_payload(
                equivalence=FrvSourceEquivalenceAudit.from_dict(
                    equiv_dict(
                        current_source_content_identity_digest=base[
                            "source_content_identity_digest"
                        ],
                        current_source_file_sha256="b" * 64,
                        canonical_point_content_digest=base["canonical_point_content_digest"],
                        canonical_point_content_spec_digest=base[
                            "canonical_point_content_spec_digest"
                        ],
                    )
                )
            )
        )
    with pytest.raises(ValueError, match="canonical_point_content_digest"):
        SourceIdentity.from_dict(
            source_payload(
                equivalence=FrvSourceEquivalenceAudit.from_dict(
                    equiv_dict(
                        current_source_content_identity_digest=base[
                            "source_content_identity_digest"
                        ],
                        canonical_point_content_digest="b" * 64,
                        canonical_point_content_spec_digest=base[
                            "canonical_point_content_spec_digest"
                        ],
                    )
                )
            )
        )
    with pytest.raises(ValueError, match="canonical_point_content_spec_digest"):
        SourceIdentity.from_dict(
            source_payload(
                equivalence=FrvSourceEquivalenceAudit.from_dict(
                    equiv_dict(
                        current_source_content_identity_digest=base[
                            "source_content_identity_digest"
                        ],
                        canonical_point_content_digest=base["canonical_point_content_digest"],
                        canonical_point_content_spec_digest="b" * 64,
                    )
                )
            )
        )


def test_equivalence_audit_requires_canonical_point_digests() -> None:
    payload = source_payload(
        equivalence=FrvSourceEquivalenceAudit.from_dict(equiv_dict()),
        canonical_digest=None,
        canonical_spec_digest=None,
        canonical_spec_version=None,
    )
    with pytest.raises(ValueError, match="requires a non-null canonical point-content digest"):
        SourceIdentity.from_dict(payload)


# -- R-009 override request/approval/result alignment -------------------------


def test_override_request_approval_result_alignment() -> None:
    with pytest.raises(ValueError, match="exclude request"):
        FRVManualOverrideArtifact.from_dict(
            override_dict(
                requested_action=1,
                approval_state=3,
                resulting_state=5,
                reviewer="r",
                review_timestamp="t",
                approval_reason="x",
            )
        )
    with pytest.raises(ValueError, match="include request"):
        FRVManualOverrideArtifact.from_dict(
            override_dict(
                requested_action=2,
                approval_state=2,
                resulting_state=4,
                reviewer="r",
                review_timestamp="t",
                approval_reason="x",
            )
        )


def test_override_chain_same_source_and_unique_ids() -> None:
    root = FRVManualOverrideArtifact.from_dict(override_dict())
    different_source = FRVManualOverrideArtifact.from_dict(
        override_dict(
            override_id="ov-2",
            source_content_identity_digest="b" * 64,
            predecessor_artifact_digest=root.override_digest,
            approval_state=2,
            resulting_state=4,
            reviewer="r",
            review_timestamp="t",
            approval_reason="x",
        )
    )
    with pytest.raises(ValueError, match="share one source"):
        frv.validate_override_chain_descriptors([root, different_source])
    duplicate_id = FRVManualOverrideArtifact.from_dict(
        override_dict(override_id="dup", request_reason="one")
    )
    duplicate_id_other = FRVManualOverrideArtifact.from_dict(
        override_dict(override_id="dup", request_reason="two")
    )
    with pytest.raises(ValueError, match="override_id"):
        frv.validate_override_chain_descriptors([duplicate_id, duplicate_id_other])
    # Identical content-addressed overrides are rejected (never silently
    # overwritten in the digest map).
    with pytest.raises(ValueError, match="override_id"):
        frv.validate_override_chain_descriptors(
            [
                FRVManualOverrideArtifact.from_dict(override_dict()),
                FRVManualOverrideArtifact.from_dict(override_dict()),
            ]
        )


# -- R-010 manifest/artifact strict counts ------------------------------------


def test_manifest_counts_reject_bool_float_and_negative() -> None:
    healthy = manifest_dict([_required(), _rejected()])
    for bad in (True, False, np.bool_(True), -1, 1.0):
        payload = dict(healthy)
        payload["canonical_inventory_count"] = bad
        with pytest.raises(ValueError):
            FRVScopeCoverageManifest.from_dict(payload)
    with pytest.raises(ValueError, match="coverage_blocked_count"):
        FRVScopeCoverageManifest.from_dict(
            {**healthy, "coverage_blocked_count": 1.0}
        )


def test_artifact_counts_reject_bool_float_and_negative() -> None:
    schema = _schema()
    block = _block(schema)
    for bad in (True, False, np.bool_(True), -1, 1.0):
        with pytest.raises(ValueError):
            FullResolutionVegetationMembershipArtifact.from_dict(
                artifact_dict([block], schema, source_point_count=bad)
            )
    with pytest.raises(ValueError, match="evaluated_point_count"):
        FullResolutionVegetationMembershipArtifact.from_dict(
            artifact_dict([block], schema, evaluated_point_count=1.0)
        )


# -- R-011 block nullable first/last and schema binding -----------------------


def test_block_schema_version_and_payload_version_exact() -> None:
    schema = _schema()
    with pytest.raises(ValueError, match="block schema version"):
        _block(schema, block_schema_version="frv-artifact-block-v2")
    with pytest.raises(ValueError, match="payload schema version reference"):
        _block(schema, payload_schema_version="frv-artifact-payload-v2")


def test_artifact_block_schema_binding() -> None:
    schema = _schema()
    artifact = _artifact([_block(schema)], schema)
    frv.validate_artifact_payload_digest_domain(artifact, payload_schema=schema)
    other_schema = FRVArtifactPayloadSchema.from_dict(
        schema_dict(
            scope_union_reference_table=("union-2",),
            scope_union_reference_table_digest=frv_digest(["union-2"]),
        )
    )
    with pytest.raises(ValueError, match="payload_schema_digest must match"):
        frv.validate_artifact_payload_digest_domain(artifact, payload_schema=other_schema)
    bad_block = _block(
        schema, block_id="b2", block_digest="b" * 64, array_descriptor_digest="c" * 64
    )
    bad_artifact = _artifact([bad_block], schema)
    with pytest.raises(ValueError, match="array_descriptor_digest"):
        frv.validate_artifact_payload_digest_domain(bad_artifact, payload_schema=schema)


# -- R-012 FRVART1 declared header domain -------------------------------------


def test_frvart1_declared_header_domain() -> None:
    schema = _schema()
    artifact = _artifact([_block(schema)], schema)
    header = _frvart1_header(artifact)
    frv.validate_artifact_payload_digest_domain(
        artifact, payload_schema=schema, declared_header=header
    )
    with pytest.raises(ValueError, match="missing key"):
        frv.validate_artifact_payload_digest_domain(
            artifact,
            declared_header={key: value for key, value in header.items() if key != "block_count"},
        )
    with pytest.raises(ValueError, match="extra key"):
        frv.validate_artifact_payload_digest_domain(
            artifact, declared_header={**header, "unexpected_key": 1}
        )
    with pytest.raises(ValueError, match="magic"):
        frv.validate_artifact_payload_digest_domain(
            artifact, declared_header={**header, "magic": "FRVART2"}
        )
    with pytest.raises(ValueError, match="block_count"):
        frv.validate_artifact_payload_digest_domain(
            artifact, declared_header={**header, "block_count": 99}
        )
    with pytest.raises(ValueError, match="ordered_block_ids"):
        frv.validate_artifact_payload_digest_domain(
            artifact, declared_header={**header, "ordered_block_ids": ["wrong"]}
        )
    with pytest.raises(ValueError, match="does not match the artifact"):
        frv.validate_artifact_payload_digest_domain(
            artifact, declared_header={**header, "scope_union_id": "other-union"}
        )
    with pytest.raises(ValueError, match="must not contain"):
        frv.validate_artifact_payload_digest_domain(
            artifact, declared_header={**header, "storage_reference": "blocks/b1"}
        )
    with pytest.raises(ValueError, match="must not contain"):
        frv.validate_artifact_payload_digest_domain(
            artifact, declared_header={**header, "artifact_sha256": SHA}
        )


def test_payload_schema_rejects_encoding_id_mismatch() -> None:
    with pytest.raises(ValueError, match="FRV-BLOCK-PAYLOAD-1"):
        FRVArtifactPayloadSchema.from_dict(schema_dict(payload_encoding_id="FRV-BLOCK-PAYLOAD-2"))


# ---------------------------------------------------------------------------
# 18. Final targeted review-fix tests (F-006 / F-012 / F-013)
# ---------------------------------------------------------------------------

_REQUIRED_REFERENCE_TABLE_FIELDS = (
    "scope_union_reference_table",
    "primary_scope_reference_table",
    "supporting_region_reference_table",
)


def test_required_reference_tables_reject_empty_tuple() -> None:
    for field in _REQUIRED_REFERENCE_TABLE_FIELDS:
        with pytest.raises(ValueError, match="must not be empty"):
            FRVArtifactPayloadSchema.from_dict(schema_dict(**{field: ()}))


def test_optional_manual_override_table_null_is_legal() -> None:
    schema = FRVArtifactPayloadSchema.from_dict(
        schema_dict(
            manual_override_reference_table=None,
            manual_override_reference_table_digest=None,
        )
    )
    assert schema.manual_override_reference_table is None
    assert schema.manual_override_reference_table_digest is None


def test_optional_manual_override_table_rejects_empty_tuple() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        FRVArtifactPayloadSchema.from_dict(
            schema_dict(
                manual_override_reference_table=(),
                manual_override_reference_table_digest="b" * 64,
            )
        )


@pytest.mark.parametrize("field", _REQUIRED_REFERENCE_TABLE_FIELDS)
def test_reference_table_entry_validation(field: str) -> None:
    cases = {
        "empty string": (("a", ""), "non-empty"),
        "whitespace only": ((" ",), "whitespace"),
        "leading whitespace": ((" a",), "whitespace"),
        "trailing whitespace": (("a ",), "whitespace"),
        "unsorted": (("b", "a"), "sorted"),
        "duplicate": (("a", "a"), "sorted"),
    }
    for _label, (table, expected) in cases.items():
        with pytest.raises(ValueError, match=expected):
            FRVArtifactPayloadSchema.from_dict(schema_dict(**{field: table}))


@pytest.mark.parametrize("field", _REQUIRED_REFERENCE_TABLE_FIELDS)
def test_reference_table_digest_mismatch_rejected(field: str) -> None:
    digest_field = field.replace("_reference_table", "_reference_table_digest")
    with pytest.raises(ValueError, match="must match"):
        FRVArtifactPayloadSchema.from_dict(
            schema_dict(**{field: ("a", "b"), digest_field: "b" * 64})
        )


def test_frvart1_ordered_block_ids_use_descriptor_canonical_order() -> None:
    schema = _schema()
    z_block = _block(
        schema,
        block_id="z",
        block_digest="b" * 64,
        original_index_start=0,
        original_index_stop=10,
        record_count=10,
        first_original_index=0,
        last_original_index=9,
    )
    a_block = _block(
        schema,
        block_id="a",
        block_digest="c" * 64,
        original_index_start=10,
        original_index_stop=20,
        record_count=10,
        first_original_index=10,
        last_original_index=19,
    )
    artifact = _artifact([z_block, a_block], schema)
    header = _frvart1_header(artifact)
    assert header["ordered_block_ids"] == ("z", "a")
    frv.validate_artifact_payload_digest_domain(artifact, declared_header=header)
    with pytest.raises(ValueError, match="canonical block order"):
        frv.validate_artifact_payload_digest_domain(
            artifact, declared_header={**header, "ordered_block_ids": ("a", "z")}
        )


def test_frvart1_block_count_rejects_non_integer() -> None:
    schema = _schema()
    artifact = _artifact([_block(schema)], schema)
    header = _frvart1_header(artifact)
    for bad in (True, False, np.bool_(True), 1.0, -1, 99):
        with pytest.raises(ValueError):
            frv.validate_artifact_payload_digest_domain(
                artifact, declared_header={**header, "block_count": bad}
            )
    frv.validate_artifact_payload_digest_domain(artifact, declared_header=header)


def test_production_calibration_digest_rejects_bool() -> None:
    with pytest.raises(ValueError, match="calibration"):
        FrvConfiguration.from_dict(config_dict(calibration_record_digest=True))
    with pytest.raises(ValueError, match="calibration"):
        FrvConfiguration.from_dict(
            {**config_dict(), "calibration_record_digest": np.bool_(True)}
        )


def test_relocation_warning_flags_reject_bool_values() -> None:
    with pytest.raises(ValueError, match="must not be a boolean"):
        FrvRelocationAudit.from_dict(relocation_dict(warning_flags=True))
    with pytest.raises(ValueError, match="must not be a boolean"):
        FrvRelocationAudit.from_dict(
            {**relocation_dict(), "warning_flags": np.bool_(True)}
        )


def test_override_chain_rejects_duplicate_descriptors() -> None:
    first = FRVManualOverrideArtifact.from_dict(override_dict())
    second = FRVManualOverrideArtifact.from_dict(override_dict())
    assert first.override_digest == second.override_digest
    # Identical descriptors carry both a duplicate override_id and a duplicate
    # override_digest; the chain validator must reject them rather than letting
    # the digest map silently overwrite one.
    with pytest.raises(ValueError, match="override_id"):
        frv.validate_override_chain_descriptors([first, second])


# ---------------------------------------------------------------------------
# 19. Final single-point correction tests (F-012 ordered_block_ids strict type)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_container",
    [
        "z",
        b"z",
        bytearray(b"z"),
        {"z": 1},
        {"z"},
        frozenset({"z"}),
        None,
        1,
        1.0,
        True,
        (block_id for block_id in ("z",)),
    ],
)
def test_frvart1_ordered_block_ids_rejects_invalid_container(
    bad_container: object,
) -> None:
    schema = _schema()
    artifact = _artifact([_block(schema)], schema)
    header = _frvart1_header(artifact)
    with pytest.raises(ValueError, match="list or tuple"):
        frv.validate_artifact_payload_digest_domain(
            artifact, declared_header={**header, "ordered_block_ids": bad_container}
        )


@pytest.mark.parametrize(
    "bad_element",
    ["", " ", " z", "z ", 1, True, None, b"z"],
)
def test_frvart1_ordered_block_ids_rejects_invalid_elements(bad_element: object) -> None:
    schema = _schema()
    artifact = _artifact([_block(schema)], schema)
    header = _frvart1_header(artifact)
    with pytest.raises(ValueError, match="non-empty strings"):
        frv.validate_artifact_payload_digest_domain(
            artifact, declared_header={**header, "ordered_block_ids": [bad_element]}
        )


def test_frvart1_ordered_block_ids_accepts_list_and_tuple() -> None:
    schema = _schema()
    artifact = _artifact([_block(schema)], schema)
    header = _frvart1_header(artifact)
    assert header["ordered_block_ids"] == ("block-1",)
    frv.validate_artifact_payload_digest_domain(
        artifact, declared_header={**header, "ordered_block_ids": ("block-1",)}
    )
    frv.validate_artifact_payload_digest_domain(
        artifact, declared_header={**header, "ordered_block_ids": ["block-1"]}
    )


def test_frvart1_ordered_block_ids_list_in_descriptor_order() -> None:
    schema = _schema()
    z_block = _block(
        schema,
        block_id="z",
        block_digest="b" * 64,
        original_index_start=0,
        original_index_stop=10,
        record_count=10,
        first_original_index=0,
        last_original_index=9,
    )
    a_block = _block(
        schema,
        block_id="a",
        block_digest="c" * 64,
        original_index_start=10,
        original_index_stop=20,
        record_count=10,
        first_original_index=10,
        last_original_index=19,
    )
    artifact = _artifact([z_block, a_block], schema)
    header = _frvart1_header(artifact)
    assert header["ordered_block_ids"] == ("z", "a")
    frv.validate_artifact_payload_digest_domain(
        artifact, declared_header={**header, "ordered_block_ids": ["z", "a"]}
    )
    with pytest.raises(ValueError, match="canonical block order"):
        frv.validate_artifact_payload_digest_domain(
            artifact, declared_header={**header, "ordered_block_ids": ("a", "z")}
        )


# ---------------------------------------------------------------------------
# 20. Final strict string-type correction tests (F-012 element type)
# ---------------------------------------------------------------------------


class BlockIdStringSubclass(str):
    """Minimal test-only str subclass; never used in production code."""


def test_frvart1_ordered_block_ids_rejects_str_subclass() -> None:
    schema = _schema()
    artifact = _artifact([_block(schema)], schema)
    header = _frvart1_header(artifact)
    with pytest.raises(ValueError, match="non-empty strings"):
        frv.validate_artifact_payload_digest_domain(
            artifact,
            declared_header={
                **header,
                "ordered_block_ids": [BlockIdStringSubclass("block-1")],
            },
        )


def test_frvart1_ordered_block_ids_rejects_numpy_str() -> None:
    schema = _schema()
    artifact = _artifact([_block(schema)], schema)
    header = _frvart1_header(artifact)
    with pytest.raises(ValueError, match="non-empty strings"):
        frv.validate_artifact_payload_digest_domain(
            artifact,
            declared_header={
                **header,
                "ordered_block_ids": [np.str_("block-1")],
            },
        )


def test_frvart1_ordered_block_ids_accepts_builtin_str() -> None:
    schema = _schema()
    artifact = _artifact([_block(schema)], schema)
    header = _frvart1_header(artifact)
    assert type(header["ordered_block_ids"][0]) is str
    frv.validate_artifact_payload_digest_domain(
        artifact, declared_header={**header, "ordered_block_ids": ["block-1"]}
    )
    frv.validate_artifact_payload_digest_domain(
        artifact, declared_header={**header, "ordered_block_ids": ("block-1",)}
    )
