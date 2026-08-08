"""FRV full-resolution membership orchestrator (PLAN-FROZEN).

``build_frv_membership`` is the synchronous Session B membership execution: it
spools the P-HAG, P-GEOMETRY and P-WIRE point chunks, builds the frozen A2
bundle, constructs the canonical inventory and 0.05 m voxel scope union,
executes the ordered membership decision with the synchronous structural
solver, applies the manual override chains, and atomically publishes the FRV
membership artifact, decision-evidence sidecar, coverage manifest, payload
schema, binding and execution result.  ``load_frv_membership_result`` reloads
and re-validates a published READY or NO_SCOPE result.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts.frv import (
    FRV_ARTIFACT_BINDING_SCHEMA_VERSION,
    FRV_ARTIFACT_BLOCK_SCHEMA_VERSION,
    FRV_ARTIFACT_PAYLOAD_SCHEMA_VERSION,
    FRV_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION,
    FRV_SCOPE_COVERAGE_MANIFEST_SCHEMA_VERSION,
    FRV_SCOPE_COVERAGE_RECORD_SCHEMA_VERSION,
    FRVArtifactBindingRecord,
    FRVArtifactBlockDescriptor,
    FRVArtifactPayloadSchema,
    FrvConfiguration,
    FrvConfigurationPurpose,
    FrvErrorCode,
    FrvHagEvidenceState,
    FrvInventoryDisposition,
    FRVManualOverrideArtifact,
    FrvMembershipState,
    FrvOverrideApprovalState,
    FrvQualityFlag,
    FrvReasonFlag,
    FrvSafetyCompletenessEffect,
    FrvScopeConstructionState,
    FRVScopeCoverageManifest,
    FRVScopeCoverageRecord,
    FrvScopeCoverageState,
    FrvScopeMode,
    FrvSegmentationHagReadiness,
    FrvWireProviderState,
    SourceIdentity,
    canonical_frv_json_bytes,
    frv_digest,
    validate_override_chain_descriptors,
    validate_scope_inventory_conservation,
)
from core.contracts.frv_a2 import (
    FrvA2FailurePhase,
    FrvA2FailureRecord,
    FrvA2ProviderRole,
)
from core.contracts.frv_geometry import (
    FrvGeometryEvidenceFlag,
    FrvGeometryEvidenceState,
    FrvGeometryProviderConfig,
    FrvGeometryProviderResult,
)
from core.contracts.frv_hag import (
    FrvHagProviderConfig,
    FrvHagProviderResult,
)
from core.contracts.frv_membership import (
    FRV_DECISION_EVIDENCE_SCHEMA_VERSION,
    FrvDecisionEvidenceArtifact,
    FrvMembershipExecutionResult,
    FrvMembershipExecutionState,
    FrvMembershipOutputReference,
    FrvScopeEvidenceReference,
    frv_source_proof_digest,
    validate_frv_scope_evidence_reference,
    validate_membership_result_state_completeness,
)
from core.contracts.frv_source import (
    FrvSourcePassThroughDimension,
    FrvSourceProviderReference,
    FrvSourceProviderResult,
)
from core.contracts.frv_wire import (
    FrvWireAuthorityReference,
    FrvWirePointRelationState,
    FrvWireProviderConfig,
    FrvWireProviderResult,
)

from algorithms.io.source_provider import load_frv_source
from algorithms.vegetation.frv_artifacts import (
    FRV_BLOCK_MAGIC,
    FRV_EVIDENCE_BLOCK_MAGIC,
    block_id_for_ordinal,
    block_storage_path,
    build_art1_declared_header,
    build_frv_payload_schema,
    compute_art1_artifact_sha256,
    decode_frv_block,
    encode_frv_block,
    evidence_block_storage_path,
    publish_directory,
    sha256_file,
    write_canonical_json,
)
from algorithms.vegetation.frv_inputs import validate_frv_a2_inputs
from algorithms.vegetation.geometry_provider import load_frv_geometry
from algorithms.vegetation.hag_provider import load_frv_hag
from algorithms.vegetation.wire_provider import load_frv_wire

# ---------------------------------------------------------------------------
# Frozen scope/policy literals
# ---------------------------------------------------------------------------

_VOXEL_SIZE_M = 0.05
_CONTEXT_MARGIN_XY = 30
_CONTEXT_MARGIN_Z = 30

_ROLE_CORE = "SCOPE_CORE"
_ROLE_CONTEXT = "SCOPE_CONTEXT_MARGIN"
_ROLE_UNION = "SCOPE_UNION"

_MEMBERSHIP_ALGORITHM_VERSION = "frv-membership-synchronous-v1"
_MEMBERSHIP_DECISION_RULE_VERSION = "frv-membership-decision-v1"
_MEMBERSHIP_POLICY_SCHEMA_VERSION = "frv-membership-policy-v1"
_SCOPE_POLICY_SCHEMA_VERSION = "frv-scope-policy-v1"

_FRV_SCOPE_DIGEST_DOMAIN = "FRVSCPGEO1"

# Scope construction version bound to the frozen policy (never free text).
_SCOPE_CONSTRUCTION_VERSION = "frv-scope-construction-v1"


# ---------------------------------------------------------------------------
# Internal scratch model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _SpooledPointData:
    """Original-index-keyed per-point provider and source evidence."""

    hag_raw: np.ndarray
    hag_clamped: np.ndarray
    hag_state: np.ndarray
    hag_reason: np.ndarray
    hag_quality: np.ndarray
    geom_state: np.ndarray
    geom_flags: np.ndarray
    geom_neighbor_count: np.ndarray
    geom_support_count: np.ndarray
    geom_support: np.ndarray
    geom_reason: np.ndarray
    geom_quality: np.ndarray
    wire_state: np.ndarray
    wire_relation: np.ndarray
    wire_wire_id: np.ndarray
    wire_segment: np.ndarray
    wire_edge: np.ndarray
    wire_distance: np.ndarray
    wire_reason: np.ndarray
    wire_quality: np.ndarray
    xyz: np.ndarray
    classification: np.ndarray
    scope_hit: np.ndarray


@dataclass(slots=True)
class _ProviderContext:
    """Live finalized providers, their bundle, and the spooled point data."""

    hag_result: FrvHagProviderResult
    geometry_result: FrvGeometryProviderResult
    wire_result: FrvWireProviderResult
    bundle: Any
    spooled: _SpooledPointData


@dataclass(slots=True)
class _CanonicalRow:
    key: str
    disposition: FrvInventoryDisposition
    member_indexes: tuple[int, ...]
    upstream_candidate_type: str
    upstream_status: str
    upstream_evidence_reference: str


@dataclass(slots=True)
class _RowOutcome:
    state: FrvScopeConstructionState
    coverage_state: FrvScopeCoverageState
    scope_id: str | None
    failure_flags: int
    review_required: bool


@dataclass(slots=True)
class _Decision:
    index: int
    membership_state: int
    reason_flags: int
    input_quality_flags: int
    hag_evidence_state: int
    segmentation_hag_readiness: int
    raw_hag: float
    clamped_hag: float
    primary_scope_id_reference: int
    supporting_region_offset: int
    supporting_region_count: int
    supporting_region_string: str
    manual_override_reference: int
    structural_supported: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_frv_membership(
    reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    scope_evidence: FrvScopeEvidenceReference,
    wire_authority: FrvWireAuthorityReference,
    overrides: tuple[FRVManualOverrideArtifact, ...],
    *,
    frv_configuration: FrvConfiguration,
    hag_config: FrvHagProviderConfig,
    geometry_config: FrvGeometryProviderConfig,
    wire_config: FrvWireProviderConfig,
    scope_policy: Any,
    membership_policy: Any,
    output_dir: Path,
    scratch_dir: Path,
    source_chunk_size: int = 1000000,
) -> FrvMembershipExecutionResult:
    """Execute the frozen Session B membership pipeline and publish the result."""

    from core.contracts.frv_membership import (
        FrvMembershipPolicyConfig,
        FrvScopePolicyConfig,
    )

    _check_signature_types(
        reference,
        source_result,
        scope_evidence,
        wire_authority,
        overrides,
        frv_configuration,
        hag_config,
        geometry_config,
        wire_config,
        scope_policy,
        membership_policy,
        output_dir,
        scratch_dir,
        source_chunk_size,
    )
    if type(scope_policy) is not FrvScopePolicyConfig:
        raise TypeError("scope_policy must be a FrvScopePolicyConfig")
    if type(membership_policy) is not FrvMembershipPolicyConfig:
        raise TypeError("membership_policy must be a FrvMembershipPolicyConfig")

    evidence_failure = _load_scope_evidence(
        scope_evidence, source_result, frv_configuration=frv_configuration
    )
    if evidence_failure is not None:
        return evidence_failure

    canonical, inventory_digests = _build_canonical_inventory(
        _scope_data(), source_result.source_identity
    )
    if not canonical:
        return _no_scope_result(source_result)

    production_failure = _production_policy_failure(
        frv_configuration, scope_policy, membership_policy
    )
    if production_failure is not None:
        return _coverage_blocked_result(
            source_result,
            (production_failure,),
            canonical=canonical,
            inventory_digests=inventory_digests,
            configuration=frv_configuration,
        )

    provider_context = _run_provider_spool(
        reference,
        source_result,
        frv_configuration,
        hag_config,
        geometry_config,
        wire_config,
        wire_authority,
        scratch_dir,
        source_chunk_size,
    )
    if isinstance(provider_context, FrvA2FailureRecord):
        return _coverage_blocked_result(
            source_result,
            (provider_context,),
            canonical=canonical,
            inventory_digests=inventory_digests,
            configuration=frv_configuration,
        )
    if isinstance(provider_context, FrvMembershipExecutionResult):
        return provider_context

    scope_outcome = _build_scope_and_manifest(
        canonical,
        inventory_digests,
        source_result,
        frv_configuration,
        _scope_data(),
    )
    if isinstance(scope_outcome, FrvMembershipExecutionResult):
        return scope_outcome
    manifest, union_runs, row_outcomes, scope_context = scope_outcome

    stream_failure = _run_membership_source_pass(
        reference,
        source_result,
        frv_configuration,
        scope_context["union_map"],
        provider_context.spooled,
        source_chunk_size,
    )
    if stream_failure is not None:
        return _coverage_blocked_result(
            source_result,
            (stream_failure,),
            canonical=canonical,
            inventory_digests=inventory_digests,
            configuration=frv_configuration,
        )

    decisions = _decide_membership(
        provider_context.spooled,
        scope_context,
        geometry_config,
        membership_policy,
    )
    final, manual_digests, override_failure = _apply_overrides(
        decisions, overrides
    )
    if override_failure is not None:
        return _coverage_blocked_result(
            source_result,
            (override_failure,),
            canonical=canonical,
            inventory_digests=inventory_digests,
            configuration=frv_configuration,
        )

    try:
        return _publish_membership_result(
            reference=reference,
            source_result=source_result,
            frv_configuration=frv_configuration,
            membership_policy=membership_policy,
            provider_context=provider_context,
            canonical=canonical,
            inventory_digests=inventory_digests,
            manifest=manifest,
            scope_context=scope_context,
            final=final,
            manual_digests=manual_digests,
            output_dir=output_dir,
            scratch_dir=scratch_dir,
        )
    except Exception as exc:
        return _coverage_blocked_result(
            source_result,
            (
                _membership_failure(
                    FrvA2FailurePhase.CROSS_PROVIDER,
                    "digest",
                    FrvErrorCode.FRV_ARTIFACT_DIGEST_MISMATCH,
                ),
            ),
            canonical=canonical,
            inventory_digests=inventory_digests,
            configuration=frv_configuration,
            detail=str(exc),
        )


def load_frv_membership_result(root_path: Path) -> FrvMembershipExecutionResult:
    """Reload and re-validate a published READY or NO_SCOPE membership result."""

    if not isinstance(root_path, Path):
        raise TypeError("root_path must be a pathlib.Path")
    if not root_path.is_dir():
        raise ValueError("root_path is not a directory")
    execution_path = root_path / "execution_result.json"
    if not execution_path.is_file():
        raise ValueError("execution result file is missing")
    execution_payload = json.loads(execution_path.read_text("utf-8"))
    result = FrvMembershipExecutionResult.from_dict(execution_payload)
    state = result.execution_state
    if state not in (
        FrvMembershipExecutionState.READY_FOR_V4,
        FrvMembershipExecutionState.NO_RELEVANT_FRV_SCOPE,
    ):
        raise ValueError("only READY or NO_SCOPE results are loadable")

    coverage = FRVScopeCoverageManifest.from_dict(
        json.loads((root_path / "coverage_manifest.json").read_text("utf-8"))
    )
    if coverage.manifest_digest != coverage.computed_digest():
        raise ValueError("coverage manifest digest mismatch")
    if result.coverage_manifest.manifest_digest != coverage.manifest_digest:
        raise ValueError("execution coverage manifest mismatch")

    if state is FrvMembershipExecutionState.NO_RELEVANT_FRV_SCOPE:
        return result

    # READY: re-validate artifact/schema/binding/decision evidence and blocks.
    for name in (
        "payload_schema.json",
        "membership_manifest.json",
        "binding.json",
        "decision_evidence.json",
    ):
        if not (root_path / name).is_file():
            raise ValueError(f"required file is missing: {name}")
    schema = FRVArtifactPayloadSchema.from_dict(
        json.loads((root_path / "payload_schema.json").read_text("utf-8"))
    )
    artifact = _membership_artifact_from_dict(
        json.loads((root_path / "membership_manifest.json").read_text("utf-8"))
    )
    binding = FRVArtifactBindingRecord.from_dict(
        json.loads((root_path / "binding.json").read_text("utf-8"))
    )
    decision = FrvDecisionEvidenceArtifact.from_dict(
        json.loads((root_path / "decision_evidence.json").read_text("utf-8"))
    )
    if decision.record_count != artifact.evaluated_point_count:
        raise ValueError("decision evidence record count mismatch")
    from core.contracts.frv import (
        validate_artifact_binding_record,
        validate_artifact_payload_digest_domain,
        validate_payload_schema,
    )

    validate_payload_schema(schema)
    validate_artifact_payload_digest_domain(
        artifact,
        payload_schema=schema,
        declared_header=build_art1_declared_header(artifact),
    )
    validate_artifact_binding_record(binding, artifact, coverage_manifest=coverage)

    schema_order = tuple(array.field_name for array in schema.arrays)
    dtypes = {array.field_name: np.dtype(array.dtype_code) for array in schema.arrays}
    row_shapes = {array.field_name: () for array in schema.arrays}
    recovered: list[np.ndarray] = []
    for block in artifact.block_descriptors:
        block_path = root_path / block.storage_reference
        if not block_path.is_file():
            raise ValueError(f"block file is missing: {block.block_id}")
        data = block_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != block.block_digest:
            raise ValueError(f"block digest mismatch: {block.block_id}")
        header, arrays = decode_frv_block(
            data,
            magic=FRV_BLOCK_MAGIC,
            schema_order=schema_order,
            dtypes=dtypes,
            row_shapes=row_shapes,
        )
        if header["block_id"] != block.block_id:
            raise ValueError("block id mismatch")
        recovered.append(arrays["original_index"])
    if artifact.artifact_sha256 != compute_art1_artifact_sha256(artifact):
        raise ValueError("artifact_sha256 mismatch")
    _replay_evaluated_indexes(artifact, recovered)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _membership_artifact_from_dict(data: Mapping[str, Any]) -> Any:
    from core.contracts.frv import FullResolutionVegetationMembershipArtifact

    return FullResolutionVegetationMembershipArtifact.from_dict(data)


def _replay_evaluated_indexes(artifact: Any, block_indexes: list[np.ndarray]) -> None:
    if not block_indexes:
        raise ValueError("artifact contains no blocks")
    recovered = np.concatenate(block_indexes)
    if int(recovered.shape[0]) != artifact.evaluated_point_count:
        raise ValueError("evaluated point count mismatch")
    if not (np.diff(recovered) > 0).all():
        raise ValueError("evaluated indexes must be strictly increasing")


def _check_signature_types(*args: Any) -> None:
    reference = args[0]
    source_result = args[1]
    scope_evidence = args[2]
    wire_authority = args[3]
    overrides = args[4]
    configuration = args[5]
    if type(reference) is not FrvSourceProviderReference:
        raise TypeError("reference must be a FrvSourceProviderReference")
    if type(source_result) is not FrvSourceProviderResult:
        raise TypeError("source_result must be a FrvSourceProviderResult")
    if type(scope_evidence) is not FrvScopeEvidenceReference:
        raise TypeError("scope_evidence must be a FrvScopeEvidenceReference")
    if type(wire_authority) is not FrvWireAuthorityReference:
        raise TypeError("wire_authority must be a FrvWireAuthorityReference")
    if type(overrides) is not tuple or any(
        not isinstance(item, FRVManualOverrideArtifact) for item in overrides
    ):
        raise TypeError("overrides must be a tuple of FRVManualOverrideArtifact")
    if type(configuration) is not FrvConfiguration:
        raise TypeError("frv_configuration must be a FrvConfiguration")
    for name, value in (
        ("hag_config", args[6]),
        ("geometry_config", args[7]),
        ("wire_config", args[8]),
    ):
        if value is None or not isinstance(value, object):
            raise TypeError(f"{name} must be a provider configuration")
    for name, value in (("output_dir", args[11]), ("scratch_dir", args[12])):
        if not isinstance(value, Path):
            raise TypeError(f"{name} must be a pathlib.Path")
    if type(args[13]) is not int or args[13] <= 0:
        raise ValueError("source_chunk_size must be a positive built-in int")


def _production_policy_failure(
    configuration: FrvConfiguration, scope_policy: Any, membership_policy: Any
) -> FrvA2FailureRecord | None:
    from core.contracts.frv_membership import (
        validate_frv_membership_policy_config,
        validate_frv_scope_policy_config,
    )

    validate_frv_scope_policy_config(scope_policy)
    validate_frv_membership_policy_config(membership_policy)
    # Formal FRV execution/publication is limited to reviewed-fixed and
    # production-approved purposes.  TEST_ONLY and CALIBRATION_CANDIDATE never
    # enter this gate.  The reviewed mode is deliberately not data calibrated,
    # so its configuration carries a null calibration record digest; the gate
    # binds the same policy digest and provider closure as production.
    valid = (
        configuration.configuration_purpose
        in (
            FrvConfigurationPurpose.REVIEWED_FIXED_POLICY,
            FrvConfigurationPurpose.PRODUCTION_APPROVED,
        )
        and configuration.configuration_is_runtime_complete is True
        and configuration.formal_completeness_allowed is True
        and configuration.membership_artifact_publication_allowed is True
        and configuration.scope_mode is FrvScopeMode.EXHAUSTIVE_REGION
        and configuration.scope_policy_digest == scope_policy.policy_digest
        and configuration.feature_policy_digest == membership_policy.policy_digest
        and membership_policy.algorithm_version == _MEMBERSHIP_ALGORITHM_VERSION
        and membership_policy.decision_rule_version == _MEMBERSHIP_DECISION_RULE_VERSION
    )
    if valid:
        return None
    return _membership_failure(
        FrvA2FailurePhase.PRODUCTION_POLICY,
        "configuration_purpose",
        FrvErrorCode.FRV_PROVIDER_INVALID,
    )


def _source_proof_digest(source_result: FrvSourceProviderResult) -> str:
    """Canonical P-SOURCE proof fingerprint via the centralized helper."""

    return frv_source_proof_digest(source_result)


def _membership_failure(
    phase: FrvA2FailurePhase,
    field_name: str,
    error_code: FrvErrorCode,
) -> FrvA2FailureRecord:
    detail = (
        f"{phase.value}:{FrvA2ProviderRole.SOURCE_SCAN.value}:source_scan:{field_name}:"
        f"{error_code.value}:{int(FrvReasonFlag.PROVIDER_UNCERTAINTY)}"
    )
    return FrvA2FailureRecord(
        phase=phase,
        provider_role=FrvA2ProviderRole.SOURCE_SCAN,
        provider_key="source_scan",
        field_name=field_name,
        error_code=error_code,
        reason_flags=int(FrvReasonFlag.PROVIDER_UNCERTAINTY),
        detail=detail,
    )


class FrvMembershipFailClosedError(Exception):
    """Membership fail-closed error for unrecoverable scope evidence.

    Raised when invalid scope evidence prevents establishing even one
    canonical/required row identity.  A zero-row COVERAGE_BLOCKED manifest
    would violate the frozen conservation validator, so no formal membership
    artifact is published and the frozen fail-closed/error path is used.
    """

    def __init__(self, failure_record: FrvA2FailureRecord) -> None:
        if type(failure_record) is not FrvA2FailureRecord:
            raise TypeError("failure_record must be a FrvA2FailureRecord")
        self.failure_record = failure_record
        super().__init__(failure_record.detail)


def _empty_manifest(
    source_identity: SourceIdentity, configuration_digest: str
) -> FRVScopeCoverageManifest:
    payload = {
        "manifest_schema_version": FRV_SCOPE_COVERAGE_MANIFEST_SCHEMA_VERSION,
        "source_content_identity_digest": source_identity.source_content_identity_digest,
        "configuration_digest": configuration_digest,
        "scope_construction_version": _SCOPE_CONSTRUCTION_VERSION,
        "canonical_inventory_digest": frv_digest(()),
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
        "provider_failure_reference_digests": (),
        "stream_failure_reference_digests": (),
        "identity_failure_reference_digests": (),
        "warnings": (),
        "limitations": (),
        "provenance_digest": frv_digest(
            {"stage": "empty", "source": source_identity.source_content_identity_digest}
        ),
    }
    payload["manifest_digest"] = frv_digest(payload)
    return FRVScopeCoverageManifest.from_dict(payload)


def _blocked_failure_flags(
    failure_records: tuple[FrvA2FailureRecord, ...],
) -> int:
    """Coverage-row failure flags for a blocked required row (plan 7.5)."""

    for failure in failure_records:
        code = failure.error_code
        if code in (
            FrvErrorCode.FRV_SCOPE_MAPPING_INCOMPLETE,
            FrvErrorCode.FRV_SCOPE_GEOMETRY_INVALID,
        ):
            return int(
                FrvReasonFlag.SCOPE_EDGE_CONFLICT | FrvReasonFlag.REVIEW_REQUIRED
            )
        if code in (
            FrvErrorCode.FRV_WIRE_IDENTITY_MISMATCH,
            FrvErrorCode.FRV_WIRE_GEOMETRY_UNAVAILABLE,
        ):
            return int(
                FrvReasonFlag.WIRE_CONFLICT
                | FrvReasonFlag.PROVIDER_UNCERTAINTY
                | FrvReasonFlag.REVIEW_REQUIRED
            )
    return int(FrvReasonFlag.PROVIDER_UNCERTAINTY | FrvReasonFlag.REVIEW_REQUIRED)


def _all_blocked_manifest(
    canonical: list[_CanonicalRow],
    inventory_digests: dict[str, str],
    source_identity: SourceIdentity,
    configuration: FrvConfiguration,
    failure_flags: int,
) -> FRVScopeCoverageManifest:
    """Build a coverage manifest with every required row failing closed.

    All required canonical rows become ``NOT_CREATED`` blocked records, so the
    derived ``routing_complete`` and ``scope_coverage_complete`` are both False
    and the manifest passes the frozen conservation validator.
    """

    outcomes = {
        row.key: _RowOutcome(
            state=FrvScopeConstructionState.NOT_CREATED,
            coverage_state=FrvScopeCoverageState.SCOPE_BLOCKED,
            scope_id=None,
            failure_flags=failure_flags,
            review_required=True,
        )
        for row in canonical
    }
    records = _build_coverage_records(
        canonical, inventory_digests, source_identity, outcomes
    )
    return _build_manifest(
        canonical,
        inventory_digests,
        source_identity,
        configuration,
        records,
        None,
        None,
    )


def _construct_membership_result(
    *,
    execution_state: FrvMembershipExecutionState,
    source_result: FrvSourceProviderResult,
    coverage_manifest: FRVScopeCoverageManifest,
    failure_records: tuple[FrvA2FailureRecord, ...],
    a2_readiness: Any = None,
    membership_artifact: Any = None,
    payload_schema: Any = None,
    artifact_binding: Any = None,
    decision_evidence_artifact: Any = None,
    hag_proof_digest: str | None = None,
    geometry_proof_digest: str | None = None,
    wire_proof_digest: str | None = None,
    provider_binding_digests: tuple[str, ...] = (),
    override_digests: tuple[str, ...] = (),
    output_reference: Any = None,
) -> FrvMembershipExecutionResult:
    """Construct a membership result through the validated path.

    The canonical FRVMEMRES1 digest is derived from the exact field projections,
    then the real dataclass constructor runs ``_check_execution_result`` and
    re-verifies the digest, so no ``__new__`` result ever bypasses the contract.
    State completeness is enforced by the caller.
    """

    source_identity = source_result.source_identity
    fingerprint = _source_proof_digest(source_result)
    fields = {
        "execution_state": execution_state,
        "source_identity": source_identity,
        "a2_readiness": a2_readiness,
        "coverage_manifest": coverage_manifest,
        "membership_artifact": membership_artifact,
        "payload_schema": payload_schema,
        "artifact_binding": artifact_binding,
        "decision_evidence_artifact": decision_evidence_artifact,
        "source_proof_digest": fingerprint,
        "hag_proof_digest": hag_proof_digest,
        "geometry_proof_digest": geometry_proof_digest,
        "wire_proof_digest": wire_proof_digest,
        "provider_binding_digests": provider_binding_digests,
        "override_digests": override_digests,
        "failure_records": failure_records,
        "output_reference": output_reference,
    }
    scratch = FrvMembershipExecutionResult.__new__(FrvMembershipExecutionResult)
    for name, value in fields.items():
        object.__setattr__(scratch, name, value)
    fields["result_digest"] = scratch.computed_digest()
    return FrvMembershipExecutionResult(**fields)


def _coverage_blocked_result(
    source_result: FrvSourceProviderResult,
    failure_records: tuple[FrvA2FailureRecord, ...],
    *,
    canonical: list[_CanonicalRow] | None = None,
    inventory_digests: dict[str, str] | None = None,
    configuration: FrvConfiguration | None = None,
    detail: str | None = None,
) -> FrvMembershipExecutionResult:
    failure_flags = _blocked_failure_flags(failure_records)
    if canonical is None or inventory_digests is None or configuration is None:
        raise FrvMembershipFailClosedError(failure_records[0])
    manifest = _all_blocked_manifest(
        canonical,
        inventory_digests,
        source_result.source_identity,
        configuration,
        failure_flags,
    )
    result = _construct_membership_result(
        execution_state=FrvMembershipExecutionState.COVERAGE_BLOCKED,
        source_result=source_result,
        coverage_manifest=manifest,
        failure_records=failure_records,
    )
    validate_membership_result_state_completeness(result)
    return result


def _no_scope_result(source_result: FrvSourceProviderResult) -> FrvMembershipExecutionResult:
    manifest = _empty_manifest(source_result.source_identity, "0" * 64)
    result = _construct_membership_result(
        execution_state=FrvMembershipExecutionState.NO_RELEVANT_FRV_SCOPE,
        source_result=source_result,
        coverage_manifest=manifest,
        failure_records=(),
    )
    validate_membership_result_state_completeness(result)
    return result


# ---------------------------------------------------------------------------
# Scope evidence loading (A1/V1/V2)
# ---------------------------------------------------------------------------

_scope_data_store: dict[str, Any] = {}


def _scope_data() -> dict[str, Any]:
    if not _scope_data_store:
        raise ValueError("scope evidence is not loaded")
    return _scope_data_store


def _load_scope_evidence(
    reference: FrvScopeEvidenceReference,
    source_result: FrvSourceProviderResult,
    *,
    frv_configuration: FrvConfiguration,
) -> FrvMembershipExecutionResult | None:
    """Load and strictly validate the A1/V1/V2 evidence pairs."""

    validate_frv_scope_evidence_reference(reference)
    source_identity = source_result.source_identity
    from core.contracts.preprocessing import PreprocessingResult

    v1_arrays: dict[str, np.ndarray] | None = None
    v2_arrays: dict[str, np.ndarray] | None = None
    try:
        a1_json = json.loads(reference.a1_json_path.read_text("utf-8"))
        a1_result = PreprocessingResult.from_dict(a1_json)
        a1_arrays = dict(np.load(reference.a1_npz_path, allow_pickle=False))
        v1_result, v1_arrays = _load_v1(reference)
        v2_result, v2_arrays = _load_v2(reference)
        _verify_components(reference)
        _verify_provenance(reference, v1_result, v2_result)
        _verify_identity(a1_result, v1_result, v2_result, source_identity)
        _verify_point_mapping(a1_arrays, v1_arrays, v2_arrays, source_identity)
    except Exception:
        return _scope_evidence_failure(
            source_result,
            _membership_failure(
                FrvA2FailurePhase.IDENTITY_CONTEXT,
                "scope_evidence",
                FrvErrorCode.FRV_SOURCE_IDENTITY_FAILED,
            ),
            configuration=frv_configuration,
            v1_arrays=v1_arrays,
            v2_arrays=v2_arrays,
            reference=reference,
        )
    _scope_data_store.clear()
    _scope_data_store.update(
        {
            "a1_arrays": a1_arrays,
            "v1_arrays": v1_arrays,
            "v2_arrays": v2_arrays,
            "a1_json": a1_json,
            "v1_result": v1_result,
            "v2_result": v2_result,
            "components": tuple(
                sorted(
                    {
                        reference.a1_npz_component.sha256,
                        reference.a1_json_component.sha256,
                        reference.v1_npz_component.sha256,
                        reference.v1_json_component.sha256,
                        reference.v2_npz_component.sha256,
                        reference.v2_json_component.sha256,
                    }
                )
            ),
        }
    )
    return None


def _scope_evidence_failure(
    source_result: FrvSourceProviderResult,
    failure_record: FrvA2FailureRecord,
    *,
    configuration: FrvConfiguration,
    v1_arrays: dict[str, np.ndarray] | None,
    v2_arrays: dict[str, np.ndarray] | None,
    reference: FrvScopeEvidenceReference,
) -> FrvMembershipExecutionResult:
    """Fail closed for invalid scope evidence (frozen plan 7.5/9.1).

    When one or more canonical/required row identities are determinable from
    the partially loaded evidence, every known row becomes a formal blocked
    coverage record through ``_coverage_blocked_result``; the manifest passes
    the formal constructor and validator with ``routing_complete=False``.  When
    even one row identity cannot be established, no zero-row COVERAGE_BLOCKED
    manifest is fabricated: the frozen fail-closed/error path raises
    ``FrvMembershipFailClosedError`` and no formal membership artifact is
    published.
    """

    canonical: list[_CanonicalRow] | None = None
    inventory_digests: dict[str, str] | None = None
    if v1_arrays is not None and v2_arrays is not None:
        try:
            data = {
                "v1_arrays": v1_arrays,
                "v2_arrays": v2_arrays,
                "components": tuple(
                    sorted(
                        {
                            reference.a1_npz_component.sha256,
                            reference.a1_json_component.sha256,
                            reference.v1_npz_component.sha256,
                            reference.v1_json_component.sha256,
                            reference.v2_npz_component.sha256,
                            reference.v2_json_component.sha256,
                        }
                    )
                ),
            }
            canonical, inventory_digests = _build_canonical_inventory(
                data, source_result.source_identity
            )
        except Exception:
            canonical = None
            inventory_digests = None
    if canonical and inventory_digests is not None:
        return _coverage_blocked_result(
            source_result,
            (failure_record,),
            canonical=canonical,
            inventory_digests=inventory_digests,
            configuration=configuration,
        )
    raise FrvMembershipFailClosedError(failure_record)


def _load_v1(reference: FrvScopeEvidenceReference) -> tuple[Any, dict[str, np.ndarray]]:
    from algorithms.vegetation.candidates import load_vegetation_candidates

    return load_vegetation_candidates(reference.v1_npz_path, reference.v1_json_path)


def _load_v2(reference: FrvScopeEvidenceReference) -> tuple[Any, dict[str, np.ndarray]]:
    from algorithms.vegetation.instances import load_vegetation_instances

    return load_vegetation_instances(reference.v2_npz_path, reference.v2_json_path)


def _verify_components(reference: FrvScopeEvidenceReference) -> None:
    for component, path in (
        (reference.a1_npz_component, reference.a1_npz_path),
        (reference.a1_json_component, reference.a1_json_path),
        (reference.v1_npz_component, reference.v1_npz_path),
        (reference.v1_json_component, reference.v1_json_path),
        (reference.v2_npz_component, reference.v2_npz_path),
        (reference.v2_json_component, reference.v2_json_path),
    ):
        if sha256_file(path) != component.sha256:
            raise ValueError("scope evidence SHA mismatch")
        if path.stat().st_size != component.size_bytes:
            raise ValueError("scope evidence size mismatch")


def _verify_provenance(
    reference: FrvScopeEvidenceReference, v1_result: Any, v2_result: Any
) -> None:
    a1_pair = {reference.a1_npz_component.sha256, reference.a1_json_component.sha256}
    provenance = getattr(v1_result, "input_provenance", None)
    if provenance is not None and not a1_pair.intersection(_collect_sha(provenance)):
        raise ValueError("V1 provenance must reference the A1 pair SHA")
    manual_log = getattr(v2_result, "manual_log", None)
    if manual_log is not None:
        recorded = getattr(manual_log, "v1_npz_sha256", None)
        if reference.v1_npz_component.sha256 not in _collect_sha(recorded):
            raise ValueError("V2 manual log must reference the V1 NPZ SHA")


def _collect_sha(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    ):
        found.add(value)
        return found
    if isinstance(value, Mapping):
        for item in value.values():
            found |= _collect_sha(item)
        return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found |= _collect_sha(item)
        return found
    return found


def _verify_identity(
    a1_result: Any,
    v1_result: Any,
    v2_result: Any,
    source_identity: SourceIdentity,
) -> None:
    source_id = source_identity.source_id
    origin = tuple(float(v) for v in source_identity.origin_context)
    for result in (a1_result, v1_result, v2_result):
        result_source = getattr(result, "manifest_source_id", None)
        if result_source is not None and result_source != source_id:
            raise ValueError("upstream source id mismatch")
        result_origin = getattr(result, "origin_xyz", None)
        if result_origin is not None and tuple(float(v) for v in result_origin) != origin:
            raise ValueError("upstream origin mismatch")


def _verify_point_mapping(
    a1_arrays: Mapping[str, np.ndarray],
    v1_arrays: Mapping[str, np.ndarray],
    v2_arrays: Mapping[str, np.ndarray],
    source_identity: SourceIdentity,
) -> None:
    count = source_identity.source_point_count
    for name, arrays in (("a1", a1_arrays), ("v1", v1_arrays), ("v2", v2_arrays)):
        original = arrays.get("original_index")
        if original is None:
            raise ValueError(f"{name} is missing original_index")
        values = np.asarray(original, dtype=np.int64)
        if int(values.shape[0]) > count:
            raise ValueError(f"{name} original_index exceeds the source count")
        if int(values.min() if values.size else -1) < 0 or int(
            values.max() if values.size else -1
        ) >= count:
            raise ValueError(f"{name} original_index out of source range")
        if len({int(v) for v in values}) != int(values.shape[0]):
            raise ValueError(f"{name} original_index must be unique")
    if "cluster_id_by_point" not in v2_arrays:
        raise ValueError("V2 is missing cluster_id_by_point")


# ---------------------------------------------------------------------------
# Canonical inventory
# ---------------------------------------------------------------------------


def _build_canonical_inventory(
    data: dict[str, Any], source_identity: SourceIdentity
) -> tuple[list[_CanonicalRow], dict[str, str]]:
    v1 = data["v1_arrays"]
    v2 = data["v2_arrays"]
    component_digests = sorted(set(data["components"]))

    rows: dict[str, _CanonicalRow] = {}

    v2_cluster_id = v2.get("cluster_id")
    v2_cluster_status = v2.get("cluster_status")
    v2_member_offsets = v2.get("cluster_member_offsets")
    v2_ordered_members = v2.get("ordered_cluster_member_row_index")
    if v2_cluster_id is not None:
        from core.contracts.vegetation_instances import VegetationClusterStatus

        valid_statuses = {
            VegetationClusterStatus.CONFIRMED.value,
            VegetationClusterStatus.PROVISIONAL.value,
            VegetationClusterStatus.REVIEW_REQUIRED.value,
            VegetationClusterStatus.REJECTED_AS_INSTANCE.value,
        }
        cluster_ids = np.asarray(v2_cluster_id, dtype=np.int64)
        statuses = np.asarray(v2_cluster_status, dtype=np.int64)
        offsets = np.asarray(v2_member_offsets, dtype=np.int64)
        ordered = np.asarray(v2_ordered_members, dtype=np.int64)
        for position in range(len(cluster_ids)):
            if int(statuses[position]) not in valid_statuses:
                continue
            start = int(offsets[position])
            stop = int(offsets[position + 1])
            members = tuple(int(v) for v in ordered[start:stop])
            key = f"V2:{int(cluster_ids[position]):010d}"
            rows[key] = _CanonicalRow(
                key=key,
                disposition=FrvInventoryDisposition.REQUIRED_FRV,
                member_indexes=members,
                upstream_candidate_type="v2_cluster",
                upstream_status=str(int(statuses[position])),
                upstream_evidence_reference=key,
            )

    v1_level = v1.get("vegetation_level")
    v1_original = v1.get("original_index")
    v1_cluster = v2.get("cluster_id_by_point")
    if v1_level is not None and v1_original is not None and v1_cluster is not None:
        from core.contracts.vegetation_candidates import VegetationCandidateLevel

        levels = np.asarray(v1_level, dtype=np.int64)
        originals = np.asarray(v1_original, dtype=np.int64)
        clusters = np.asarray(v1_cluster, dtype=np.int64)
        for position in range(len(originals)):
            if int(levels[position]) == int(VegetationCandidateLevel.REJECTED):
                continue
            if int(clusters[position]) != -1:
                continue
            key = f"V1:{int(originals[position]):020d}"
            rows[key] = _CanonicalRow(
                key=key,
                disposition=FrvInventoryDisposition.REQUIRED_FRV,
                member_indexes=(int(originals[position]),),
                upstream_candidate_type="v1_point",
                upstream_status=str(int(levels[position])),
                upstream_evidence_reference=key,
            )

    canonical_keys = sorted(rows)
    canonical_records = {
        key: frv_digest(
            {
                "source_content_identity_digest": source_identity.source_content_identity_digest,
                "canonical_key": key,
                "type": rows[key].upstream_candidate_type,
                "status": rows[key].upstream_status,
                "members": list(rows[key].member_indexes),
                "components": list(component_digests),
            }
        )
        for key in canonical_keys
    }
    canonical_inventory_digest = frv_digest(
        tuple(canonical_records[key] for key in canonical_keys)
    )
    required_keys = [
        key
        for key in canonical_keys
        if rows[key].disposition is FrvInventoryDisposition.REQUIRED_FRV
    ]
    required_digest = frv_digest(tuple(canonical_records[key] for key in required_keys))
    digests = {
        "canonical_inventory_digest": canonical_inventory_digest,
        "required_inventory_digest": required_digest,
        "canonical_records": canonical_records,
    }
    return [rows[key] for key in canonical_keys], digests


# ---------------------------------------------------------------------------
# Provider spooling
# ---------------------------------------------------------------------------


def _run_provider_spool(
    reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    configuration: FrvConfiguration,
    hag_config: FrvHagProviderConfig,
    geometry_config: FrvGeometryProviderConfig,
    wire_config: FrvWireProviderConfig,
    wire_authority: FrvWireAuthorityReference,
    scratch_dir: Path,
    source_chunk_size: int,
) -> Any:
    """Spool the three provider streams; return ``_ProviderContext`` or a failure."""

    try:
        with load_frv_hag(
            reference,
            source_result,
            frv_configuration=configuration,
            provider_config=hag_config,
            source_chunk_size=source_chunk_size,
        ) as session:
            hag_chunks = list(session.iter_chunks())
            hag_result = session.finalize()
        with load_frv_geometry(
            reference,
            source_result,
            frv_configuration=configuration,
            provider_config=geometry_config,
            scratch_dir=scratch_dir,
            source_chunk_size=source_chunk_size,
        ) as session:
            geometry_chunks = list(session.iter_chunks())
            geometry_result = session.finalize()
        with load_frv_wire(
            reference,
            source_result,
            wire_authority,
            frv_configuration=configuration,
            provider_config=wire_config,
            source_chunk_size=source_chunk_size,
        ) as session:
            wire_chunks = list(session.iter_chunks())
            wire_result = session.finalize()
    except Exception as exc:
        failure = getattr(exc, "failure_record", None)
        if isinstance(failure, FrvA2FailureRecord):
            return failure
        return _membership_failure(
            FrvA2FailurePhase.CROSS_PROVIDER,
            "digest",
            FrvErrorCode.FRV_ARTIFACT_DIGEST_MISMATCH,
        )

    try:
        bundle = validate_frv_a2_inputs(
            source_result,
            hag_result,
            geometry_result,
            wire_result,
            frv_configuration=configuration,
        )
    except Exception:
        return _membership_failure(
            FrvA2FailurePhase.CROSS_PROVIDER,
            "readiness",
            FrvErrorCode.FRV_PROVIDER_INVALID,
        )

    spooled = _assemble_spool(hag_chunks, geometry_chunks, wire_chunks, source_result)
    if isinstance(spooled, FrvA2FailureRecord):
        return spooled
    return _ProviderContext(
        hag_result=hag_result,
        geometry_result=geometry_result,
        wire_result=wire_result,
        bundle=bundle,
        spooled=spooled,
    )


def _assemble_spool(
    hag_chunks: list[Any],
    geometry_chunks: list[Any],
    wire_chunks: list[Any],
    source_result: FrvSourceProviderResult,
) -> Any:
    count = source_result.source_identity.source_point_count
    point = _SpooledPointData(
        hag_raw=np.full(count, 0.0, dtype=np.float32),
        hag_clamped=np.full(count, 0.0, dtype=np.float32),
        hag_state=np.zeros(count, dtype=np.uint8),
        hag_reason=np.zeros(count, dtype=np.uint32),
        hag_quality=np.zeros(count, dtype=np.uint32),
        geom_state=np.zeros(count, dtype=np.uint8),
        geom_flags=np.zeros(count, dtype=np.uint32),
        geom_neighbor_count=np.zeros((count, 3), dtype=np.uint16),
        geom_support_count=np.zeros(count, dtype=np.uint8),
        geom_support=np.full((count, 16), -1, dtype=np.int64),
        geom_reason=np.zeros(count, dtype=np.uint32),
        geom_quality=np.zeros(count, dtype=np.uint32),
        wire_state=np.zeros(count, dtype=np.uint8),
        wire_relation=np.zeros(count, dtype=np.uint8),
        wire_wire_id=np.full(count, -1, dtype=np.int32),
        wire_segment=np.full(count, -1, dtype=np.int32),
        wire_edge=np.full(count, -1, dtype=np.int32),
        wire_distance=np.zeros(count, dtype=np.float64),
        wire_reason=np.zeros(count, dtype=np.uint32),
        wire_quality=np.zeros(count, dtype=np.uint32),
        xyz=np.zeros((count, 3), dtype=np.float64),
        classification=np.zeros(count, dtype=np.uint8),
        scope_hit=np.zeros(count, dtype=bool),
    )
    try:
        for chunk in hag_chunks:
            for row in range(chunk.point_count):
                index = int(chunk.original_index[row])
                point.hag_raw[index] = chunk.raw_hag_m[row]
                point.hag_clamped[index] = chunk.clamped_hag_m[row]
                point.hag_state[index] = chunk.hag_evidence_state[row]
                point.hag_reason[index] = chunk.reason_flags[row]
                point.hag_quality[index] = chunk.quality_flags[row]
        for chunk in geometry_chunks:
            for row in range(chunk.point_count):
                index = int(chunk.original_index[row])
                point.geom_state[index] = chunk.geometry_evidence_state[row]
                point.geom_flags[index] = chunk.geometry_evidence_flags[row]
                point.geom_neighbor_count[index, :] = chunk.neighbor_count[row]
                support_count = int(chunk.support_neighbor_count[row])
                point.geom_support_count[index] = support_count
                point.geom_support[index, :support_count] = (
                    chunk.support_neighbor_original_index[row, :support_count]
                )
                point.geom_reason[index] = chunk.reason_flags[row]
                point.geom_quality[index] = chunk.quality_flags[row]
        for chunk in wire_chunks:
            for row in range(chunk.point_count):
                index = int(chunk.original_index[row])
                point.wire_state[index] = chunk.provider_state[row]
                point.wire_relation[index] = chunk.relation_state[row]
                point.wire_wire_id[index] = chunk.nearest_wire_id[row]
                point.wire_segment[index] = chunk.nearest_segment_index[row]
                point.wire_edge[index] = chunk.nearest_edge_index[row]
                point.wire_distance[index] = chunk.centerline_distance_m[row]
                point.wire_reason[index] = chunk.reason_flags[row]
                point.wire_quality[index] = chunk.quality_flags[row]
    except Exception:
        return _membership_failure(
            FrvA2FailurePhase.DOMAIN_ALIGNMENT,
            "original_index",
            FrvErrorCode.FRV_ORIGINAL_INDEX_INVALID,
        )
    return point


# ---------------------------------------------------------------------------
# Scope construction
# ---------------------------------------------------------------------------


def _build_scope_and_manifest(
    canonical: list[_CanonicalRow],
    inventory_digests: dict[str, str],
    source_result: FrvSourceProviderResult,
    configuration: FrvConfiguration,
    data: dict[str, Any],
) -> Any:
    source_identity = source_result.source_identity
    voxel_keys_by_index = _build_voxel_index(data["a1_arrays"], source_identity)

    row_outcomes: dict[str, _RowOutcome] = {}
    scope_ids: dict[str, str] = {}
    row_core_cells: dict[str, set[tuple[int, int, int]]] = {}
    union_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    scope_ordinal = 0

    for row in canonical:
        core_keys = _row_core_keys(row, voxel_keys_by_index, source_identity)
        if core_keys is None:
            row_outcomes[row.key] = _RowOutcome(
                state=FrvScopeConstructionState.NOT_CREATED,
                coverage_state=FrvScopeCoverageState.REGION_MAPPING_INCOMPLETE,
                scope_id=None,
                failure_flags=int(
                    FrvReasonFlag.SCOPE_EDGE_CONFLICT | FrvReasonFlag.REVIEW_REQUIRED
                ),
                review_required=True,
            )
            continue
        context_map = _dilate_intervals(core_keys)
        intersects = _interval_map_intersects(context_map, union_map)
        state = (
            FrvScopeConstructionState.SCOPE_MERGED
            if intersects
            else FrvScopeConstructionState.SCOPE_CREATED
        )
        scope_id = (
            f"frv-scope-{source_identity.source_content_identity_digest[:12]}-{scope_ordinal:04d}"
        )
        scope_ordinal += 1
        scope_ids[row.key] = scope_id
        row_core_cells[row.key] = core_keys
        _merge_interval_maps(union_map, context_map)
        row_outcomes[row.key] = _RowOutcome(
            state=state,
            coverage_state=FrvScopeCoverageState.COVERED,
            scope_id=scope_id,
            failure_flags=0,
            review_required=False,
        )

    union_runs = _runs_from_interval_map(union_map)
    if canonical and not union_runs:
        return _coverage_blocked_result(
            source_result,
            (
                _membership_failure(
                    FrvA2FailurePhase.SEMANTIC_READINESS,
                    "scope_union",
                    FrvErrorCode.FRV_SCOPE_MAPPING_INCOMPLETE,
                ),
            ),
            canonical=canonical,
            inventory_digests=inventory_digests,
            configuration=configuration,
        )

    union_geometry_digest = _scope_geometry_digest(_ROLE_UNION, union_runs, source_identity)
    scope_union_id = f"frv-scope-union-{union_geometry_digest[:16]}"
    scope_context = {
        "union_runs": union_runs,
        "union_map": union_map,
        "scope_union_id": scope_union_id,
        "scope_union_geometry_digest": union_geometry_digest,
        "scope_ids": scope_ids,
        "row_core_cells": row_core_cells,
        "canonical": canonical,
    }
    records = _build_coverage_records(
        canonical, inventory_digests, source_identity, row_outcomes
    )
    manifest = _build_manifest(
        canonical, inventory_digests, source_identity, configuration, records,
        scope_union_id, union_geometry_digest,
    )
    return manifest, union_runs, row_outcomes, scope_context


def _build_voxel_index(
    a1_arrays: Mapping[str, np.ndarray], source_identity: SourceIdentity
) -> dict[int, tuple[int, int, int]]:
    original = a1_arrays.get("original_index")
    voxel_key = a1_arrays.get("voxel_key")
    if original is None or voxel_key is None:
        return {}
    result: dict[int, tuple[int, int, int]] = {}
    for position, index in enumerate(np.asarray(original, dtype=np.int64)):
        key = tuple(int(v) for v in np.asarray(voxel_key)[position])
        if int(index) in result:
            raise ValueError("duplicate A1 original index")
        result[int(index)] = key
    return result


def _cell_in_row_context(
    cell: tuple[int, int, int], core_cells: set[tuple[int, int, int]]
) -> bool:
    """Return whether *cell* is within the ±30 context of any core cell."""

    ix, iy, iz = cell
    for cx, cy, cz in core_cells:
        if (
            abs(ix - cx) <= _CONTEXT_MARGIN_XY
            and abs(iy - cy) <= _CONTEXT_MARGIN_XY
            and abs(iz - cz) <= _CONTEXT_MARGIN_Z
        ):
            return True
    return False


def _row_core_keys(
    row: _CanonicalRow,
    voxel_keys_by_index: dict[int, tuple[int, int, int]],
    source_identity: SourceIdentity,
) -> set[tuple[int, int, int]] | None:
    core: set[tuple[int, int, int]] = set()
    for index in row.member_indexes:
        key = voxel_keys_by_index.get(index)
        if key is None:
            return None
        core.add(key)
    if not core:
        return None
    return core


def _dilate_intervals(
    core: set[tuple[int, int, int]],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Return ``{(ix, iy): [(z_start, z_stop_exclusive)]}`` for the ±30 context.

    Each core cell (cx, cy, cz) contributes the box
    ``[cx-30, cx+30] x [cy-30, cy+30] x [cz-30, cz+31)``; z intervals are merged
    per (ix, iy) so the result is a compact interval map rather than a dense
    cell set.
    """

    by_xy: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cx, cy, cz in core:
        for ix in range(cx - _CONTEXT_MARGIN_XY, cx + _CONTEXT_MARGIN_XY + 1):
            for iy in range(cy - _CONTEXT_MARGIN_XY, cy + _CONTEXT_MARGIN_XY + 1):
                by_xy.setdefault((ix, iy), []).append(
                    (cz - _CONTEXT_MARGIN_Z, cz + _CONTEXT_MARGIN_Z + 1)
                )
    merged: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for key, intervals in by_xy.items():
        intervals.sort()
        result: list[tuple[int, int]] = []
        for start, stop in intervals:
            if result and start <= result[-1][1]:
                result[-1] = (result[-1][0], max(result[-1][1], stop))
            else:
                result.append((start, stop))
        merged[key] = result
    return merged


def _interval_map_intersects(
    a: dict[tuple[int, int], list[tuple[int, int]]],
    b: dict[tuple[int, int], list[tuple[int, int]]],
) -> bool:
    for key, a_intervals in a.items():
        b_intervals = b.get(key)
        if b_intervals is None:
            continue
        for a_start, a_stop in a_intervals:
            for b_start, b_stop in b_intervals:
                if a_start < b_stop and b_start < a_stop:
                    return True
    return False


def _merge_interval_maps(
    target: dict[tuple[int, int], list[tuple[int, int]]],
    source: dict[tuple[int, int], list[tuple[int, int]]],
) -> None:
    for key, intervals in source.items():
        target.setdefault(key, []).extend(intervals)
    for key, intervals in list(target.items()):
        intervals.sort()
        result: list[tuple[int, int]] = []
        for start, stop in intervals:
            if result and start <= result[-1][1]:
                result[-1] = (result[-1][0], max(result[-1][1], stop))
            else:
                result.append((start, stop))
        target[key] = result


def _runs_from_interval_map(
    interval_map: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[tuple[int, int, int, int], ...]:
    runs: list[tuple[int, int, int, int]] = []
    for (ix, iy), intervals in sorted(interval_map.items()):
        for start, stop in intervals:
            runs.append((ix, iy, start, stop))
    return tuple(runs)


def _scope_geometry_digest(
    role: str, runs: tuple[tuple[int, int, int, int], ...], source_identity: SourceIdentity
) -> str:
    hasher = hashlib.sha256()
    hasher.update(_FRV_SCOPE_DIGEST_DOMAIN.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(role.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(source_identity.source_content_identity_digest.encode("utf-8"))
    hasher.update(b"\0")
    origin = source_identity.origin_context
    hasher.update(struct.pack("<ddd", *origin))
    hasher.update(struct.pack("<d", _VOXEL_SIZE_M))
    hasher.update(struct.pack("<Q", len(runs)))
    for ix, iy, z_start, z_stop in runs:
        hasher.update(struct.pack("<qqqq", ix, iy, z_start, z_stop))
    return hasher.hexdigest()


def _build_coverage_records(
    canonical: list[_CanonicalRow],
    inventory_digests: dict[str, str],
    source_identity: SourceIdentity,
    row_outcomes: Mapping[str, _RowOutcome],
) -> tuple[FRVScopeCoverageRecord, ...]:
    records: list[FRVScopeCoverageRecord] = []
    for row in canonical:
        outcome = row_outcomes[row.key]
        record_digest = inventory_digests["canonical_records"][row.key]
        if outcome.state in (
            FrvScopeConstructionState.SCOPE_CREATED,
            FrvScopeConstructionState.SCOPE_MERGED,
        ):
            payload = {
                "record_schema_version": FRV_SCOPE_COVERAGE_RECORD_SCHEMA_VERSION,
                "source_content_identity_digest": source_identity.source_content_identity_digest,
                "canonical_inventory_key": row.key,
                "canonical_inventory_record_digest": record_digest,
                "canonical_inventory_digest": inventory_digests["canonical_inventory_digest"],
                "inventory_disposition": FrvInventoryDisposition.REQUIRED_FRV,
                "required_inventory_record_digest": record_digest,
                "required_inventory_digest": inventory_digests["required_inventory_digest"],
                "approved_hard_rejection_evidence": None,
                "upstream_candidate_reference": row.upstream_evidence_reference,
                "upstream_candidate_type": row.upstream_candidate_type,
                "upstream_status": row.upstream_status,
                "upstream_evidence_reference": row.upstream_evidence_reference,
                "scope_id": outcome.scope_id,
                "scope_construction_state": outcome.state,
                "scope_coverage_state": FrvScopeCoverageState.COVERED,
                "failure_reason_flags": 0,
                "review_required": False,
                "safety_completeness_effect": (
                    FrvSafetyCompletenessEffect.PRESERVES_SAFETY_COMPLETENESS
                ),
                "diagnostic_reference": row.key,
                "out_of_policy_safety_approval_digest": None,
            }
        else:
            payload = {
                "record_schema_version": FRV_SCOPE_COVERAGE_RECORD_SCHEMA_VERSION,
                "source_content_identity_digest": source_identity.source_content_identity_digest,
                "canonical_inventory_key": row.key,
                "canonical_inventory_record_digest": record_digest,
                "canonical_inventory_digest": inventory_digests["canonical_inventory_digest"],
                "inventory_disposition": FrvInventoryDisposition.REQUIRED_FRV,
                "required_inventory_record_digest": record_digest,
                "required_inventory_digest": inventory_digests["required_inventory_digest"],
                "approved_hard_rejection_evidence": None,
                "upstream_candidate_reference": row.upstream_evidence_reference,
                "upstream_candidate_type": row.upstream_candidate_type,
                "upstream_status": row.upstream_status,
                "upstream_evidence_reference": row.upstream_evidence_reference,
                "scope_id": None,
                "scope_construction_state": FrvScopeConstructionState.NOT_CREATED,
                "scope_coverage_state": outcome.coverage_state,
                "failure_reason_flags": outcome.failure_flags,
                "review_required": True,
                "safety_completeness_effect": (
                    FrvSafetyCompletenessEffect.BLOCKS_SAFETY_COMPLETENESS
                ),
                "diagnostic_reference": row.key,
                "out_of_policy_safety_approval_digest": None,
            }
        record = FRVScopeCoverageRecord.__new__(FRVScopeCoverageRecord)
        for name, value in payload.items():
            object.__setattr__(record, name, value)
        object.__setattr__(record, "record_digest", record.computed_digest())
        records.append(record)
    return tuple(records)


def _build_manifest(
    canonical: list[_CanonicalRow],
    inventory_digests: dict[str, str],
    source_identity: SourceIdentity,
    configuration: FrvConfiguration,
    records: tuple[FRVScopeCoverageRecord, ...],
    scope_union_id: str,
    scope_union_geometry_digest: str,
) -> FRVScopeCoverageManifest:
    required_count = sum(
        1 for row in canonical if row.disposition is FrvInventoryDisposition.REQUIRED_FRV
    )
    approved_count = len(canonical) - required_count
    created_merged = sum(
        1
        for record in records
        if record.scope_construction_state
        in (FrvScopeConstructionState.SCOPE_CREATED, FrvScopeConstructionState.SCOPE_MERGED)
    )
    blocked = required_count - created_merged
    payload = {
        "manifest_schema_version": FRV_SCOPE_COVERAGE_MANIFEST_SCHEMA_VERSION,
        "source_content_identity_digest": source_identity.source_content_identity_digest,
        "configuration_digest": configuration.configuration_digest,
        "scope_construction_version": _SCOPE_CONSTRUCTION_VERSION,
        "canonical_inventory_digest": inventory_digests["canonical_inventory_digest"],
        "required_inventory_digest": inventory_digests["required_inventory_digest"],
        "scope_union_id": scope_union_id,
        "scope_union_geometry_digest": scope_union_geometry_digest,
        "records": [record.to_dict() for record in records],
        "canonical_inventory_count": len(canonical),
        "required_inventory_count": required_count,
        "approved_hard_rejection_count": approved_count,
        "scope_created_or_merged_count": created_merged,
        "coverage_blocked_count": blocked,
        "canonical_inventory_complete": True,
        "required_inventory_complete": True,
        "approved_hard_rejection_complete": True,
        "scope_coverage_complete": blocked == 0,
        "routing_complete": blocked == 0,
        "expected_semantic_artifact_id": None,
        "provider_failure_reference_digests": (),
        "stream_failure_reference_digests": (),
        "identity_failure_reference_digests": (),
        "warnings": (),
        "limitations": (),
        "provenance_digest": frv_digest(
            {"stage": "scope", "source": source_identity.source_content_identity_digest}
        ),
    }
    payload["manifest_digest"] = frv_digest(payload)
    manifest = FRVScopeCoverageManifest.from_dict(payload)
    validate_scope_inventory_conservation(manifest)
    return manifest


# ---------------------------------------------------------------------------
# Membership source pass
# ---------------------------------------------------------------------------


def _run_membership_source_pass(
    reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    configuration: FrvConfiguration,
    union_map: dict[tuple[int, int], list[tuple[int, int]]],
    spooled: _SpooledPointData,
    source_chunk_size: int,
) -> FrvA2FailureRecord | None:
    from core.contracts.frv_source import (
        FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
        FrvSourceCanonicalContentPolicy,
        FrvSourceCoordinateConversionPolicy,
        FrvSourceInvalidPointPolicy,
        FrvSourceModel,
        FrvSourceProviderConfig,
        derive_frv_source_provider_config_digest,
    )

    source_config = FrvSourceProviderConfig(
        schema_version=FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
        source_model=FrvSourceModel.SINGLE_FILE_ONLY,
        coordinate_conversion_policy=FrvSourceCoordinateConversionPolicy.FORBID,
        invalid_point_policy=FrvSourceInvalidPointPolicy.FAIL_CLOSED,
        canonical_content_policy=FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
        pass_through_dimensions=(FrvSourcePassThroughDimension.CLASSIFICATION,),
        provider_config_digest=derive_frv_source_provider_config_digest(
            schema_version=FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
            source_model=FrvSourceModel.SINGLE_FILE_ONLY,
            coordinate_conversion_policy=FrvSourceCoordinateConversionPolicy.FORBID,
            invalid_point_policy=FrvSourceInvalidPointPolicy.FAIL_CLOSED,
            canonical_content_policy=FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
            pass_through_dimensions=(FrvSourcePassThroughDimension.CLASSIFICATION,),
        ),
    )
    origin = reference.source_identity.origin_context
    try:
        with load_frv_source(
            reference,
            frv_configuration=configuration,
            provider_config=source_config,
            chunk_size=source_chunk_size,
        ) as session:
            for chunk in session.iter_chunks():
                for row in range(chunk.point_count):
                    index = int(chunk.original_index[row])
                    spooled.xyz[index, :] = chunk.xyz[row]
                    classification = chunk.attributes.get("classification")
                    if classification is not None:
                        spooled.classification[index] = int(classification[row])
            session.finalize()
    except Exception:
        return _membership_failure(
            FrvA2FailurePhase.DOMAIN_ALIGNMENT,
            "source_pass",
            FrvErrorCode.FRV_STREAM_INCOMPLETE,
        )
    # Write the formal scope hit into the spooled store as a derived bitmap.
    hit = np.zeros(spooled.xyz.shape[0], dtype=bool)
    for index in range(spooled.xyz.shape[0]):
        x, y, z = spooled.xyz[index]
        ix = int(math.floor((float(x) - float(origin[0])) / _VOXEL_SIZE_M))
        iy = int(math.floor((float(y) - float(origin[1])) / _VOXEL_SIZE_M))
        iz = int(math.floor((float(z) - float(origin[2])) / _VOXEL_SIZE_M))
        hit[index] = _in_union(ix, iy, iz, union_map)
    spooled.scope_hit = hit
    return None


# ---------------------------------------------------------------------------
# Membership decisions
# ---------------------------------------------------------------------------


def _in_union(
    ix: int, iy: int, iz: int, union_map: dict[tuple[int, int], list[tuple[int, int]]]
) -> bool:
    intervals = union_map.get((ix, iy))
    if intervals is None:
        return False
    for start, stop in intervals:
        if start <= iz < stop:
            return True
    return False


def _decide_membership(
    spooled: _SpooledPointData,
    scope_context: dict[str, Any],
    geometry_config: FrvGeometryProviderConfig,
    membership_policy: Any,
) -> dict[int, _Decision]:
    ordinary_min = membership_policy.ordinary_min_hag_m
    upper_seed_min = membership_policy.upper_seed_min_hag_m
    ground_max_abs = membership_policy.ground_max_abs_raw_hag_m
    propagation_drop = membership_policy.propagation_min_hag_drop_m
    max_rounds = membership_policy.max_structural_rounds
    ground_class = membership_policy.ground_classification_value

    scope_ids = scope_context["scope_ids"]
    row_core_cells = scope_context["row_core_cells"]
    canonical = scope_context["canonical"]
    primary_scope_table = sorted(set(scope_ids.values()))
    scope_reference = {scope_id: position for position, scope_id in enumerate(primary_scope_table)}

    in_scope = np.asarray(spooled.scope_hit)
    in_scope_indexes = [int(index) for index in np.flatnonzero(in_scope)]

    solver = _structural_solver(
        spooled,
        in_scope_indexes,
        geometry_config,
        upper_seed_min,
        propagation_drop,
        max_rounds,
        ground_class,
    )
    nonconverged = solver["nonconverged"]
    supported = solver["supported"]

    origin = (0.0, 0.0, 0.0)
    covering_by_index: dict[int, list[str]] = {}
    for index in in_scope_indexes:
        x, y, z = spooled.xyz[index]
        cell = (
            int(math.floor((float(x) - float(origin[0])) / _VOXEL_SIZE_M)),
            int(math.floor((float(y) - float(origin[1])) / _VOXEL_SIZE_M)),
            int(math.floor((float(z) - float(origin[2])) / _VOXEL_SIZE_M)),
        )
        covering = sorted(
            row.key
            for row in canonical
            if _cell_in_row_context(cell, row_core_cells[row.key])
        )
        covering_by_index[index] = covering

    decisions: dict[int, _Decision] = {}
    for index in in_scope_indexes:
        covering = covering_by_index[index]
        decisions[index] = _decide_point(
            index,
            spooled,
            geometry_config,
            ordinary_min,
            ground_max_abs,
            ground_class,
            supported,
            nonconverged,
            scope_reference,
            scope_ids,
            covering,
        )
    return decisions


def _structural_solver(
    spooled: _SpooledPointData,
    in_scope: list[int],
    geometry_config: FrvGeometryProviderConfig,
    upper_seed_min: float,
    propagation_drop: float,
    max_rounds: int,
    ground_class: int,
) -> dict[str, Any]:
    supported_mask = np.zeros(spooled.xyz.shape[0], dtype=bool)
    frontier: set[int] = set()
    for index in in_scope:
        if _is_upper_seed(spooled, index, geometry_config, upper_seed_min, ground_class):
            supported_mask[index] = True
            frontier.add(index)

    rounds = 0
    while rounds < max_rounds:
        new_points: set[int] = set()
        for index in in_scope:
            if supported_mask[index]:
                continue
            if not _hag_eligible(spooled, index):
                continue
            if not _woody_eligible(spooled, index, geometry_config, ground_class):
                continue
            if any(
                supported_mask[neighbor]
                and float(spooled.hag_clamped[neighbor])
                >= float(spooled.hag_clamped[index]) + propagation_drop
                for neighbor in _support_neighbors(spooled, index)
                if 0 <= neighbor < spooled.xyz.shape[0]
            ):
                new_points.add(index)
        if not new_points:
            break
        for index in new_points:
            supported_mask[index] = True
        frontier = new_points
        rounds += 1

    nonconverged: set[int] = set()
    if rounds >= max_rounds:
        for index in in_scope:
            if supported_mask[index]:
                continue
            if not _hag_eligible(spooled, index):
                continue
            if not _woody_eligible(spooled, index, geometry_config, ground_class):
                continue
            if any(
                supported_mask[neighbor]
                and float(spooled.hag_clamped[neighbor])
                >= float(spooled.hag_clamped[index]) + propagation_drop
                for neighbor in _support_neighbors(spooled, index)
                if 0 <= neighbor < spooled.xyz.shape[0]
            ):
                nonconverged.add(index)
    return {"supported": supported_mask, "frontier": frontier, "nonconverged": nonconverged}


def _is_upper_seed(
    spooled: _SpooledPointData,
    index: int,
    geometry_config: FrvGeometryProviderConfig,
    upper_seed_min: float,
    ground_class: int,
) -> bool:
    if int(spooled.hag_state[index]) != int(FrvHagEvidenceState.HAG_VALID_POSITIVE):
        return False
    if not (float(spooled.hag_clamped[index]) >= upper_seed_min):
        return False
    if not _geometry_positive(spooled, index):
        return False
    if not _density_supported(spooled, index):
        return False
    if not _scan_adequate(spooled, index, geometry_config):
        return False
    if int(spooled.classification[index]) == ground_class:
        return False
    if not _wire_clear(spooled, index):
        return False
    return True


def _hag_eligible(spooled: _SpooledPointData, index: int) -> bool:
    return int(spooled.hag_state[index]) in (
        int(FrvHagEvidenceState.HAG_VALID_POSITIVE),
        int(FrvHagEvidenceState.HAG_VALID_NEAR_ZERO),
        int(FrvHagEvidenceState.HAG_WITHIN_NEGATIVE_TOLERANCE),
    )


def _woody_eligible(
    spooled: _SpooledPointData,
    index: int,
    geometry_config: FrvGeometryProviderConfig,
    ground_class: int,
) -> bool:
    if not (int(spooled.geom_flags[index]) & int(FrvGeometryEvidenceFlag.WOODY_COLUMN_COMPATIBLE)):
        return False
    if not _scan_adequate(spooled, index, geometry_config):
        return False
    if int(spooled.classification[index]) == ground_class:
        return False
    if not _wire_clear(spooled, index):
        return False
    return True


def _support_neighbors(spooled: _SpooledPointData, index: int) -> list[int]:
    count = int(spooled.geom_support_count[index])
    return [int(v) for v in spooled.geom_support[index, :count]]


def _scan_adequate(
    spooled: _SpooledPointData, index: int, geometry_config: FrvGeometryProviderConfig
) -> bool:
    if int(spooled.geom_state[index]) != int(FrvGeometryEvidenceState.VALID):
        return False
    counts = spooled.geom_neighbor_count[index]
    return bool(
        int(counts[1]) >= geometry_config.min_neighbor_counts[1]
        and int(counts[2]) >= geometry_config.min_neighbor_counts[2]
    )


def _geometry_positive(spooled: _SpooledPointData, index: int) -> bool:
    flags = int(spooled.geom_flags[index])
    return bool(
        flags & int(FrvGeometryEvidenceFlag.CANOPY_BRANCHING_COMPATIBLE)
        or flags & int(FrvGeometryEvidenceFlag.WOODY_COLUMN_COMPATIBLE)
    )


def _density_supported(spooled: _SpooledPointData, index: int) -> bool:
    return bool(int(spooled.geom_flags[index]) & int(FrvGeometryEvidenceFlag.DENSITY_SUPPORTED))


def _ground_compatible(spooled: _SpooledPointData, index: int) -> bool:
    return bool(
        int(spooled.geom_flags[index]) & int(FrvGeometryEvidenceFlag.GROUND_SURFACE_COMPATIBLE)
    )


def _wire_clear(spooled: _SpooledPointData, index: int) -> bool:
    state = int(spooled.wire_state[index])
    relation = int(spooled.wire_relation[index])
    if state == int(FrvWireProviderState.WIRE_NOT_APPLICABLE_PROVEN):
        return True
    if state == int(FrvWireProviderState.WIRE_RELATION_VALID):
        return relation == int(FrvWirePointRelationState.OUTSIDE_INFLUENCE_DOMAIN)
    return False


def _decide_point(
    index: int,
    spooled: _SpooledPointData,
    geometry_config: FrvGeometryProviderConfig,
    ordinary_min: float,
    ground_max_abs: float,
    ground_class: int,
    supported: np.ndarray,
    nonconverged: set[int],
    scope_reference: dict[str, int],
    scope_ids: Mapping[str, str],
    covering: list[str],
) -> _Decision:
    hag_state = int(spooled.hag_state[index])
    raw_hag = float(spooled.hag_raw[index])
    clamped_hag = float(spooled.hag_clamped[index])
    geom_state = int(spooled.geom_state[index])
    classification = int(spooled.classification[index])
    wire_relation = int(spooled.wire_relation[index])
    is_supported = bool(supported[index])
    quality = int(spooled.hag_quality[index])
    covering_scope_ids = tuple(sorted(scope_ids[key] for key in covering))
    if covering_scope_ids:
        primary_scope_id = covering_scope_ids[0]
    else:
        primary_scope_id = sorted(scope_ids.values())[0]
    supporting_region_string = canonical_frv_json_bytes(list(covering_scope_ids)).decode("utf-8")

    membership = int(FrvMembershipState.REVIEW_REQUIRED_MEMBERSHIP)
    reason = int(FrvReasonFlag.INSUFFICIENT_EVIDENCE | FrvReasonFlag.REVIEW_REQUIRED)
    if hag_state in (
        int(FrvHagEvidenceState.HAG_MISSING),
        int(FrvHagEvidenceState.HAG_INVALID),
        int(FrvHagEvidenceState.HAG_OUT_OF_DOMAIN),
    ):
        reason = _hag_reason_flag(hag_state)
    elif geom_state != int(FrvGeometryEvidenceState.VALID):
        reason = int(FrvReasonFlag.INSUFFICIENT_EVIDENCE | FrvReasonFlag.REVIEW_REQUIRED)
    elif wire_relation in (
        int(FrvWirePointRelationState.WIRE_BODY_POSSIBLE),
        int(FrvWirePointRelationState.WIRE_PROXIMITY_ONLY),
        int(FrvWirePointRelationState.WIRE_VEGETATION_CONFLICT),
    ):
        reason = int(FrvReasonFlag.WIRE_CONFLICT | FrvReasonFlag.REVIEW_REQUIRED)
    elif _geometry_positive(spooled, index) and _ground_compatible(spooled, index):
        reason = int(FrvReasonFlag.VEGETATION_COUNTER_EVIDENCE | FrvReasonFlag.REVIEW_REQUIRED)
    elif index in nonconverged:
        reason = int(FrvReasonFlag.PROVIDER_UNCERTAINTY | FrvReasonFlag.REVIEW_REQUIRED)
    elif (
        classification == ground_class
        and abs(raw_hag) <= ground_max_abs
        and _ground_compatible(spooled, index)
        and not _geometry_positive(spooled, index)
        and not is_supported
        and _scan_adequate(spooled, index, geometry_config)
        and _wire_clear(spooled, index)
    ):
        membership = int(FrvMembershipState.EXCLUDED_NON_CANDIDATE)
        reason = int(FrvReasonFlag.GROUND_CONFLICT)
    elif (
        hag_state == int(FrvHagEvidenceState.HAG_VALID_POSITIVE)
        and clamped_hag >= ordinary_min
        and _geometry_positive(spooled, index)
        and _density_supported(spooled, index)
        and _scan_adequate(spooled, index, geometry_config)
        and classification != ground_class
        and _wire_clear(spooled, index)
    ):
        membership = int(FrvMembershipState.VERIFIED_VEGETATION_CANDIDATE)
        reason = int(FrvReasonFlag.NONE)
    elif (
        _hag_eligible(spooled, index)
        and is_supported
        and bool(
            int(spooled.geom_flags[index]) & int(FrvGeometryEvidenceFlag.WOODY_COLUMN_COMPATIBLE)
        )
        and _scan_adequate(spooled, index, geometry_config)
        and classification != ground_class
        and _wire_clear(spooled, index)
    ):
        membership = int(FrvMembershipState.VERIFIED_VEGETATION_CANDIDATE)
        reason = int(FrvReasonFlag.NONE)

    readiness = _hag_readiness(membership, hag_state, quality)
    return _Decision(
        index=index,
        membership_state=membership,
        reason_flags=reason,
        input_quality_flags=quality,
        hag_evidence_state=hag_state,
        segmentation_hag_readiness=readiness,
        raw_hag=raw_hag,
        clamped_hag=clamped_hag,
        primary_scope_id_reference=scope_reference[primary_scope_id],
        supporting_region_offset=0,
        supporting_region_count=1,
        supporting_region_string=supporting_region_string,
        manual_override_reference=0xFFFFFFFF,
        structural_supported=is_supported,
    )


def _hag_reason_flag(hag_state: int) -> int:
    if hag_state == int(FrvHagEvidenceState.HAG_MISSING):
        return int(FrvReasonFlag.HAG_MISSING | FrvReasonFlag.REVIEW_REQUIRED)
    if hag_state == int(FrvHagEvidenceState.HAG_INVALID):
        return int(FrvReasonFlag.HAG_INVALID | FrvReasonFlag.REVIEW_REQUIRED)
    return int(FrvReasonFlag.HAG_OUT_OF_DOMAIN | FrvReasonFlag.REVIEW_REQUIRED)


def _hag_readiness(membership: int, hag_state: int, quality: int) -> int:
    if membership in (
        int(FrvMembershipState.VERIFIED_VEGETATION_CANDIDATE),
        int(FrvMembershipState.MANUAL_INCLUDE),
    ) and hag_state in (
        int(FrvHagEvidenceState.HAG_VALID_POSITIVE),
        int(FrvHagEvidenceState.HAG_VALID_NEAR_ZERO),
        int(FrvHagEvidenceState.HAG_WITHIN_NEGATIVE_TOLERANCE),
    ) and not (quality & int(FrvQualityFlag.PROVIDER_INVALID)):
        return int(FrvSegmentationHagReadiness.READY)
    return int(FrvSegmentationHagReadiness.REVIEW_REQUIRED)


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


def _apply_overrides(
    decisions: dict[int, _Decision],
    overrides: tuple[FRVManualOverrideArtifact, ...],
) -> tuple[dict[int, _Decision], tuple[str, ...], FrvA2FailureRecord | None]:
    if overrides:
        try:
            validate_override_chain_descriptors(overrides)
        except Exception:
            return decisions, (), _membership_failure(
                FrvA2FailurePhase.STRUCTURE_INTEGRITY,
                "override_chain",
                FrvErrorCode.FRV_OVERRIDE_CHAIN_CONFLICT,
            )
    by_digest = {item.override_digest: item for item in overrides}
    by_index: dict[int, list[FRVManualOverrideArtifact]] = {}
    for item in overrides:
        for index in item.requested_original_indexes:
            by_index.setdefault(int(index), []).append(item)

    effective: dict[int, FRVManualOverrideArtifact] = {}
    for index, items in by_index.items():
        leaves = [
            item
            for item in items
            if item.resulting_state is not None
            and item.conflict_flags == 0
            and (
                item.superseded_override_digest is None
                or item.superseded_override_digest not in by_digest
            )
        ]
        if len(leaves) == 1:
            effective[index] = leaves[0]

    manual_digests: set[str] = set()
    for index, decision in decisions.items():
        leaf = effective.get(index)
        if leaf is None or index not in by_index:
            continue
        if leaf.automatic_state_before_override != decision.membership_state:
            continue
        if leaf.approval_state is FrvOverrideApprovalState.MANUAL_INCLUDE_APPROVED:
            decision.membership_state = int(FrvMembershipState.MANUAL_INCLUDE)
            decision.reason_flags = int(
                decision.reason_flags | FrvReasonFlag.MANUAL_INCLUDE_PROVENANCE
            )
        elif leaf.approval_state is FrvOverrideApprovalState.MANUAL_EXCLUDE_APPROVED:
            decision.membership_state = int(FrvMembershipState.MANUAL_EXCLUDE)
            decision.reason_flags = int(
                decision.reason_flags | FrvReasonFlag.MANUAL_EXCLUDE_PROVENANCE
            )
        manual_digests.add(leaf.override_digest)
    return decisions, tuple(sorted(manual_digests)), None


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------


def _publish_membership_result(
    *,
    reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    frv_configuration: FrvConfiguration,
    membership_policy: Any,
    provider_context: _ProviderContext,
    canonical: list[_CanonicalRow],
    inventory_digests: dict[str, str],
    manifest: FRVScopeCoverageManifest,
    scope_context: dict[str, Any],
    final: dict[int, _Decision],
    manual_digests: tuple[str, ...],
    output_dir: Path,
    scratch_dir: Path,
) -> FrvMembershipExecutionResult:
    evaluated = sorted(final.values(), key=lambda decision: decision.index)
    count = len(evaluated)
    if count <= 0:
        raise ValueError("READY membership requires evaluated points")

    block_capacity = int(membership_policy.output_block_record_count)
    supporting_table = tuple(
        sorted({decision.supporting_region_string for decision in evaluated})
    )
    supporting_index = {text: position for position, text in enumerate(supporting_table)}
    for decision in evaluated:
        decision.supporting_region_offset = supporting_index[decision.supporting_region_string]
    schema = _payload_schema_for(scope_context, manual_digests, supporting_table)
    schema_order = tuple(array.field_name for array in schema.arrays)
    payload_arrays = _build_payload_arrays(evaluated, schema, scope_context)
    evidence_arrays = _build_evidence_arrays(evaluated, provider_context.spooled)
    evidence_order = tuple(evidence_arrays)

    blocks = _split_blocks(count, block_capacity)
    block_descriptors: list[FRVArtifactBlockDescriptor] = []
    evidence_descriptors: list[FRVArtifactBlockDescriptor] = []
    block_bytes_by_id: dict[str, bytes] = {}
    evidence_bytes_by_id: dict[str, bytes] = {}

    artifact_id = _compute_artifact_id(source_result, scope_context, frv_configuration)
    for ordinal, (start, stop) in enumerate(blocks):
        block_id = block_id_for_ordinal(ordinal)
        payload = _slice(payload_arrays, start, stop)
        data = encode_frv_block(
            magic=FRV_BLOCK_MAGIC,
            artifact_id=artifact_id,
            block_id=block_id,
            payload_schema_digest=schema.payload_schema_digest,
            arrays=payload,
            schema_order=schema_order,
        )
        block_bytes_by_id[block_id] = data
        block_descriptors.append(
            _block_descriptor(
                block_id=block_id,
                block_digest=hashlib.sha256(data).hexdigest(),
                schema=schema,
                start=start,
                stop=stop,
                evaluated=evaluated,
                storage_reference=block_storage_path(block_id),
            )
        )
        evidence_block_id = block_id_for_ordinal(ordinal, evidence=True)
        ev_payload = _slice(evidence_arrays, start, stop)
        ev_data = encode_frv_block(
            magic=FRV_EVIDENCE_BLOCK_MAGIC,
            artifact_id=artifact_id,
            block_id=evidence_block_id,
            payload_schema_digest=schema.payload_schema_digest,
            arrays=ev_payload,
            schema_order=evidence_order,
        )
        evidence_bytes_by_id[evidence_block_id] = ev_data
        evidence_descriptors.append(
            _block_descriptor(
                block_id=evidence_block_id,
                block_digest=hashlib.sha256(ev_data).hexdigest(),
                schema=None,
                start=start,
                stop=stop,
                evaluated=evaluated,
                storage_reference=evidence_block_storage_path(evidence_block_id),
            )
        )

    artifact = _build_artifact(
        reference=reference,
        source_result=source_result,
        frv_configuration=frv_configuration,
        manifest=manifest,
        schema=schema,
        block_descriptors=block_descriptors,
        final=final,
        canonical=canonical,
        inventory_digests=inventory_digests,
        scope_context=scope_context,
        manual_digests=manual_digests,
        artifact_id=artifact_id,
    )
    binding = _build_binding(artifact, manifest, frv_configuration, source_result)
    decision = _build_decision_evidence(
        artifact=artifact,
        provider_context=provider_context,
        evaluated=evaluated,
        evidence_descriptors=evidence_descriptors,
    )
    output_reference = FrvMembershipOutputReference(
        root_path=output_dir,
        execution_result_path=output_dir / "execution_result.json",
        payload_schema_path=output_dir / "payload_schema.json",
        coverage_manifest_path=output_dir / "coverage_manifest.json",
        membership_manifest_path=output_dir / "membership_manifest.json",
        binding_path=output_dir / "binding.json",
        block_paths=tuple(
            output_dir / block.storage_reference for block in block_descriptors
        ),
        decision_evidence_manifest_path=output_dir / "decision_evidence.json",
        decision_evidence_block_paths=tuple(
            output_dir / block.storage_reference for block in evidence_descriptors
        ),
    )
    result = _build_execution_result(
        source_result=source_result,
        provider_context=provider_context,
        manifest=manifest,
        artifact=artifact,
        schema=schema,
        binding=binding,
        decision=decision,
        output_reference=output_reference,
        manual_digests=manual_digests,
    )
    _write_output_directory(
        output_dir=output_dir,
        scratch_dir=scratch_dir,
        result=result,
        block_descriptors=block_descriptors,
        block_bytes=block_bytes_by_id,
        evidence_descriptors=evidence_descriptors,
        evidence_bytes=evidence_bytes_by_id,
    )
    return result


def _compute_artifact_id(
    source_result: FrvSourceProviderResult,
    scope_context: dict[str, Any],
    configuration: FrvConfiguration,
) -> str:
    payload = {
        "artifact_schema_version": FRV_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION,
        "algorithm_version": _MEMBERSHIP_ALGORITHM_VERSION,
        "decision_rule_version": _MEMBERSHIP_DECISION_RULE_VERSION,
        "source_content_identity_digest": (
            source_result.source_identity.source_content_identity_digest
        ),
        "scope_union_id": scope_context["scope_union_id"],
        "configuration_digest": configuration.configuration_digest,
        "configuration_purpose": configuration.configuration_purpose,
    }
    return "frv-membership-" + frv_digest(payload)


def _payload_schema_for(
    scope_context: dict[str, Any],
    manual_digests: tuple[str, ...],
    supporting_table: tuple[str, ...],
) -> FRVArtifactPayloadSchema:
    scope_ids = tuple(sorted(set(scope_context["scope_ids"].values())))
    return build_frv_payload_schema(
        scope_union_id=scope_context["scope_union_id"],
        primary_scope_ids=scope_ids,
        supporting_region_strings=supporting_table,
        manual_override_digests=manual_digests if manual_digests else None,
    )


def _build_payload_arrays(
    evaluated: list[_Decision],
    schema: FRVArtifactPayloadSchema,
    scope_context: dict[str, Any],
) -> dict[str, np.ndarray]:
    count = len(evaluated)
    arrays = {
        "original_index": np.asarray([d.index for d in evaluated], dtype="<i8"),
        "membership_state": np.asarray([d.membership_state for d in evaluated], dtype="|u1"),
        "reason_flags": np.asarray([d.reason_flags for d in evaluated], dtype="<u4"),
        "input_quality_flags": np.asarray([d.input_quality_flags for d in evaluated], dtype="<u4"),
        "hag_evidence_state": np.asarray([d.hag_evidence_state for d in evaluated], dtype="|u1"),
        "segmentation_hag_readiness": np.asarray(
            [d.segmentation_hag_readiness for d in evaluated], dtype="|u1"
        ),
        "scope_union_id_reference": np.zeros(count, dtype="<u4"),
        "primary_scope_id_reference": np.asarray(
            [d.primary_scope_id_reference for d in evaluated], dtype="<u4"
        ),
        "supporting_region_offset": np.asarray(
            [d.supporting_region_offset for d in evaluated], dtype="<u8"
        ),
        "supporting_region_count": np.asarray(
            [d.supporting_region_count for d in evaluated], dtype="<u4"
        ),
        "manual_override_reference": np.asarray(
            [d.manual_override_reference for d in evaluated], dtype="<u4"
        ),
        "raw_hag": np.asarray([d.raw_hag for d in evaluated], dtype="<f4"),
        "clamped_hag": np.asarray([d.clamped_hag for d in evaluated], dtype="<f4"),
        "hag_quality_flags": np.asarray([d.input_quality_flags for d in evaluated], dtype="<u4"),
        "hag_reason_flags": np.asarray([d.reason_flags for d in evaluated], dtype="<u4"),
    }
    for _name, array in arrays.items():
        array.flags.writeable = False
    return arrays


def _build_evidence_arrays(
    evaluated: list[_Decision], spooled: _SpooledPointData
) -> dict[str, np.ndarray]:
    arrays = {
        "original_index": np.asarray([d.index for d in evaluated], dtype="<i8"),
        "geometry_evidence_state": np.asarray(
            [int(spooled.geom_state[d.index]) for d in evaluated], dtype="|u1"
        ),
        "geometry_evidence_flags": np.asarray(
            [int(spooled.geom_flags[d.index]) for d in evaluated], dtype="<u4"
        ),
        "wire_provider_state": np.asarray(
            [int(spooled.wire_state[d.index]) for d in evaluated], dtype="|u1"
        ),
        "wire_relation_state": np.asarray(
            [int(spooled.wire_relation[d.index]) for d in evaluated], dtype="|u1"
        ),
        "nearest_wire_id": np.asarray(
            [int(spooled.wire_wire_id[d.index]) for d in evaluated], dtype="<i4"
        ),
        "nearest_segment_index": np.asarray(
            [int(spooled.wire_segment[d.index]) for d in evaluated], dtype="<i4"
        ),
        "centerline_distance_m": np.asarray(
            [float(spooled.wire_distance[d.index]) for d in evaluated], dtype="<f8"
        ),
    }
    for _name, array in arrays.items():
        array.flags.writeable = False
    return arrays


def _slice(arrays: dict[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    return {name: array[start:stop] for name, array in arrays.items()}


def _split_blocks(count: int, block_capacity: int) -> list[tuple[int, int]]:
    if count == 0:
        return []
    blocks: list[tuple[int, int]] = []
    start = 0
    while start < count:
        stop = min(start + block_capacity, count)
        blocks.append((start, stop))
        start = stop
    return blocks


def _block_descriptor(
    *,
    block_id: str,
    block_digest: str,
    schema: FRVArtifactPayloadSchema | None,
    start: int,
    stop: int,
    evaluated: list[_Decision],
    storage_reference: str,
) -> FRVArtifactBlockDescriptor:
    if schema is None:
        payload_schema_digest = ""
        payload_schema_version = FRV_ARTIFACT_PAYLOAD_SCHEMA_VERSION
        array_descriptor_digest = ""
    else:
        payload_schema_digest = schema.payload_schema_digest
        payload_schema_version = schema.payload_schema_version
        array_descriptor_digest = frv_digest([array.to_dict() for array in schema.arrays])
    slice_indexes = [d.index for d in evaluated[start:stop]]
    record_count = len(slice_indexes)
    first = slice_indexes[0] if record_count else None
    last = slice_indexes[-1] if record_count else None
    payload = {
        "block_schema_version": FRV_ARTIFACT_BLOCK_SCHEMA_VERSION,
        "payload_schema_digest": payload_schema_digest,
        "block_id": block_id,
        "block_digest": block_digest,
        "original_index_start": first if first is not None else 0,
        "original_index_stop": (last + 1) if last is not None else 0,
        "record_count": record_count,
        "first_original_index": first,
        "last_original_index": last,
        "storage_reference": storage_reference,
        "payload_schema_version": payload_schema_version,
        "array_descriptor_digest": array_descriptor_digest,
    }
    descriptor = FRVArtifactBlockDescriptor.__new__(FRVArtifactBlockDescriptor)
    for name, value in payload.items():
        object.__setattr__(descriptor, name, value)
    object.__setattr__(descriptor, "block_descriptor_digest", descriptor.computed_digest())
    return descriptor


def _build_artifact(
    *,
    reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    frv_configuration: FrvConfiguration,
    manifest: FRVScopeCoverageManifest,
    schema: FRVArtifactPayloadSchema,
    block_descriptors: list[FRVArtifactBlockDescriptor],
    final: dict[int, _Decision],
    canonical: list[_CanonicalRow],
    inventory_digests: dict[str, str],
    scope_context: dict[str, Any],
    manual_digests: tuple[str, ...],
    artifact_id: str,
) -> Any:
    from core.contracts.frv import FullResolutionVegetationMembershipArtifact

    counts = _state_counts(final)
    payload = {
        "artifact_schema_version": FRV_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION,
        "algorithm_version": _MEMBERSHIP_ALGORITHM_VERSION,
        "decision_rule_version": _MEMBERSHIP_DECISION_RULE_VERSION,
        "artifact_id": artifact_id,
        "source_content_identity_digest": (
            source_result.source_identity.source_content_identity_digest
        ),
        "source_identity_record_digest": (
            source_result.source_identity.source_identity_record_digest
        ),
        "source_file_sha256": reference.artifact_components[0].sha256,
        "canonical_point_content_digest": None,
        "canonical_point_content_spec_digest": None,
        "scope_coverage_manifest_digest": manifest.manifest_digest,
        "scope_union_id": scope_context["scope_union_id"],
        "scope_construction_version": _SCOPE_CONSTRUCTION_VERSION,
        "scope_union_geometry_digest": scope_context["scope_union_geometry_digest"],
        "canonical_inventory_digest": inventory_digests["canonical_inventory_digest"],
        "required_inventory_digest": inventory_digests["required_inventory_digest"],
        "configuration_digest": frv_configuration.configuration_digest,
        "configuration_purpose": frv_configuration.configuration_purpose,
        "payload_schema_digest": schema.payload_schema_digest,
        "block_schema_version": FRV_ARTIFACT_BLOCK_SCHEMA_VERSION,
        "artifact_sha256": "",
        "block_descriptors": tuple(block_descriptors),
        "source_point_count": source_result.source_identity.source_point_count,
        "unique_in_scope_point_count": len(final),
        "evaluated_point_count": len(final),
        "verified_vegetation_candidate_count": counts["verified"],
        "excluded_non_candidate_count": counts["excluded"],
        "review_required_membership_count": counts["review"],
        "manual_include_count": counts["manual_include"],
        "manual_exclude_count": counts["manual_exclude"],
        "unasserted_in_scope_count": 0,
        "canonical_inventory_count": len(canonical),
        "required_inventory_count": sum(
            1 for row in canonical if row.disposition is FrvInventoryDisposition.REQUIRED_FRV
        ),
        "approved_hard_rejection_count": sum(
            1
            for row in canonical
            if row.disposition is FrvInventoryDisposition.APPROVED_HARD_REJECTION
        ),
        "scope_created_or_merged_count": manifest.scope_created_or_merged_count,
        "coverage_blocked_count": 0,
        "canonical_inventory_complete": True,
        "required_inventory_complete": True,
        "approved_hard_rejection_complete": True,
        "scope_coverage_complete": True,
        "routing_complete": True,
        "membership_decision_complete": True,
        "provider_digest_references": (),
        "override_digest_references": tuple(sorted(set(manual_digests))),
        "block_digest_references": tuple(
            sorted(block.block_digest for block in block_descriptors)
        ),
        "warning_digest": frv_digest(()),
        "limitation_digest": frv_digest(()),
        "provenance_digest": frv_digest(
            {
                "stage": "frv-membership",
                "source": source_result.source_identity.source_content_identity_digest,
            }
        ),
    }
    artifact = FullResolutionVegetationMembershipArtifact.__new__(
        FullResolutionVegetationMembershipArtifact
    )
    for name, value in payload.items():
        object.__setattr__(artifact, name, value)
    if artifact.artifact_id != artifact.computed_artifact_id():
        raise ValueError("artifact id mismatch")
    object.__setattr__(artifact, "artifact_sha256", compute_art1_artifact_sha256(artifact))
    object.__setattr__(artifact, "manifest_digest", artifact.computed_digest())
    return artifact


def _state_counts(final: dict[int, _Decision]) -> dict[str, int]:
    counts = {"verified": 0, "excluded": 0, "review": 0, "manual_include": 0, "manual_exclude": 0}
    for decision in final.values():
        state = decision.membership_state
        if state == int(FrvMembershipState.VERIFIED_VEGETATION_CANDIDATE):
            counts["verified"] += 1
        elif state == int(FrvMembershipState.EXCLUDED_NON_CANDIDATE):
            counts["excluded"] += 1
        elif state == int(FrvMembershipState.MANUAL_INCLUDE):
            counts["manual_include"] += 1
        elif state == int(FrvMembershipState.MANUAL_EXCLUDE):
            counts["manual_exclude"] += 1
        else:
            counts["review"] += 1
    return counts


def _build_binding(
    artifact: Any,
    manifest: FRVScopeCoverageManifest,
    configuration: FrvConfiguration,
    source_result: FrvSourceProviderResult,
) -> FRVArtifactBindingRecord:
    payload = {
        "binding_schema_version": FRV_ARTIFACT_BINDING_SCHEMA_VERSION,
        "source_content_identity_digest": (
            source_result.source_identity.source_content_identity_digest
        ),
        "source_identity_record_digest": (
            source_result.source_identity.source_identity_record_digest
        ),
        "scope_union_id": artifact.scope_union_id,
        "configuration_digest": configuration.configuration_digest,
        "semantic_artifact_id": artifact.artifact_id,
        "scope_coverage_manifest_digest": manifest.manifest_digest,
        "membership_artifact_manifest_digest": artifact.manifest_digest,
        "membership_artifact_sha256": artifact.artifact_sha256,
        "binding_created_by_stage": "frv-membership",
    }
    binding = FRVArtifactBindingRecord.__new__(FRVArtifactBindingRecord)
    for name, value in payload.items():
        object.__setattr__(binding, name, value)
    object.__setattr__(binding, "binding_digest", binding.computed_digest())
    return binding


def _build_decision_evidence(
    *,
    artifact: Any,
    provider_context: _ProviderContext,
    evaluated: list[_Decision],
    evidence_descriptors: list[FRVArtifactBlockDescriptor],
) -> FrvDecisionEvidenceArtifact:
    decision = FrvDecisionEvidenceArtifact.__new__(FrvDecisionEvidenceArtifact)
    object.__setattr__(decision, "schema_version", FRV_DECISION_EVIDENCE_SCHEMA_VERSION)
    object.__setattr__(
        decision,
        "source_content_identity_digest",
        provider_context.hag_result.source_identity.source_content_identity_digest,
    )
    object.__setattr__(
        decision,
        "source_identity_record_digest",
        provider_context.hag_result.source_identity.source_identity_record_digest,
    )
    object.__setattr__(decision, "membership_artifact_manifest_digest", artifact.manifest_digest)
    object.__setattr__(
        decision, "hag_proof_digest", provider_context.hag_result.proof.proof_digest
    )
    object.__setattr__(
        decision,
        "geometry_proof_digest",
        provider_context.geometry_result.proof.proof_digest,
    )
    object.__setattr__(
        decision, "wire_proof_digest", provider_context.wire_result.proof.proof_digest
    )
    object.__setattr__(
        decision, "original_index_domain_digest", _index_domain_digest(evaluated, provider_context)
    )
    object.__setattr__(decision, "record_count", len(evaluated))
    object.__setattr__(
        decision,
        "block_digests",
        tuple(sorted(block.block_digest for block in evidence_descriptors)),
    )
    object.__setattr__(decision, "evidence_digest", decision.computed_digest())
    return decision


def _index_domain_digest(
    evaluated: list[_Decision], provider_context: _ProviderContext
) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"FRVA2IDX1\0")
    hasher.update(b"ORIGINAL_INDEX\0")
    hasher.update(
        provider_context.hag_result.source_identity.source_content_identity_digest.encode("utf-8")
    )
    hasher.update(b"\0")
    hasher.update(b"full_source\0")
    hasher.update(struct.pack("<Q", len(evaluated)))
    for decision in evaluated:
        hasher.update(struct.pack("<Q", decision.index))
    return hasher.hexdigest()


def _write_output_directory(
    *,
    output_dir: Path,
    scratch_dir: Path,
    result: FrvMembershipExecutionResult,
    block_descriptors: list[FRVArtifactBlockDescriptor],
    block_bytes: dict[str, bytes],
    evidence_descriptors: list[FRVArtifactBlockDescriptor],
    evidence_bytes: dict[str, bytes],
) -> None:
    artifact = result.membership_artifact
    schema = result.payload_schema
    manifest = result.coverage_manifest
    binding = result.artifact_binding
    decision = result.decision_evidence_artifact
    assert (
        artifact is not None
        and schema is not None
        and binding is not None
        and decision is not None
    )

    def publish(staging: Path) -> None:
        (staging / "blocks").mkdir()
        (staging / "evidence_blocks").mkdir()
        for descriptor in block_descriptors:
            (staging / descriptor.storage_reference).write_bytes(
                block_bytes[descriptor.block_id]
            )
        for descriptor in evidence_descriptors:
            (staging / descriptor.storage_reference).write_bytes(
                evidence_bytes[descriptor.block_id]
            )
        write_canonical_json(staging / "payload_schema.json", schema.to_dict())
        write_canonical_json(staging / "coverage_manifest.json", manifest.to_dict())
        write_canonical_json(staging / "membership_manifest.json", artifact.to_dict())
        write_canonical_json(staging / "binding.json", binding.to_dict())
        write_canonical_json(staging / "decision_evidence.json", decision.to_dict())
        write_canonical_json(staging / "execution_result.json", result.to_dict())

    publish_directory(scratch_dir=scratch_dir, target_dir=output_dir, publish=publish)


def _build_execution_result(
    *,
    source_result: FrvSourceProviderResult | None,
    provider_context: _ProviderContext | None,
    manifest: FRVScopeCoverageManifest,
    artifact: Any,
    schema: FRVArtifactPayloadSchema,
    binding: FRVArtifactBindingRecord,
    decision: FrvDecisionEvidenceArtifact,
    output_reference: FrvMembershipOutputReference,
    manual_digests: tuple[str, ...],
) -> FrvMembershipExecutionResult:
    if source_result is None or provider_context is None:
        raise ValueError("final execution result requires live provider context")
    bundle = provider_context.bundle
    provider_bindings = tuple(
        sorted(item.binding_digest for item in bundle.readiness_record.provider_evidence)
    )
    result = _construct_membership_result(
        execution_state=FrvMembershipExecutionState.READY_FOR_V4,
        source_result=source_result,
        coverage_manifest=manifest,
        failure_records=(),
        a2_readiness=bundle.readiness_record,
        membership_artifact=artifact,
        payload_schema=schema,
        artifact_binding=binding,
        decision_evidence_artifact=decision,
        hag_proof_digest=provider_context.hag_result.proof.proof_digest,
        geometry_proof_digest=provider_context.geometry_result.proof.proof_digest,
        wire_proof_digest=provider_context.wire_result.proof.proof_digest,
        provider_binding_digests=provider_bindings,
        override_digests=tuple(sorted(set(manual_digests))),
        output_reference=output_reference,
    )
    validate_membership_result_state_completeness(result)
    return result


__all__ = [
    "build_frv_membership",
    "load_frv_membership_result",
]
