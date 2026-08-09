"""FRV-to-V4 input handoff contracts (PLAN-FROZEN).

This module freezes the closed V4 recovery/routing enums, the read-only
validated membership view, the recovered vegetation sets, the unrecovered
region records, the coverage report, the handoff result, and the closed
``V4InputLoadError``.  It performs no file access and no source replay.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any

import numpy as np

from core.contracts.frv import (
    FRVArtifactBindingRecord,
    FRVArtifactPayloadSchema,
    FRVScopeCoverageManifest,
    FullResolutionVegetationMembershipArtifact,
    SourceIdentity,
    frv_digest,
    validate_source_content_identity_digest,
    validate_source_identity_record_digest,
)
from core.contracts.frv_membership import (
    FrvDecisionEvidenceArtifact,
    validate_frv_decision_evidence_artifact,
)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

V4_VALIDATED_VIEW_SCHEMA_VERSION = "v4-validated-frv-membership-view-v1"
V4_RECOVERED_SET_SCHEMA_VERSION = "v4-recovered-vegetation-set-v1"
V4_UNRESOLVED_SET_SCHEMA_VERSION = "v4-unresolved-measured-point-set-v1"
V4_UNRECOVERED_REGION_SCHEMA_VERSION = "v4-unrecovered-region-record-v1"
V4_RECOVERY_COVERAGE_SCHEMA_VERSION = "v4-recovery-coverage-v1"
V4_HANDOFF_SCHEMA_VERSION = "v4-input-handoff-v1"
V4_HANDOFF_MANIFEST_SCHEMA_VERSION_V2 = "v4-input-handoff-v2"
V4_VALIDATED_VIEW_DIGEST_DOMAIN = "V4FRVVIEW1"
V4_RECOVERED_INDEX_DIGEST_DOMAIN = "V4RECIDX1"
V4_RECOVERED_SET_DIGEST_DOMAIN = "V4NRM1"
V4_UNRESOLVED_SET_DIGEST_DOMAIN = "V4REV1"
V4_COVERAGE_DIGEST_DOMAIN = "V4COV1"
V4_HANDOFF_DIGEST_DOMAIN = "V4INP1"
V4_INPUT_MANIFEST_DIGEST_DOMAIN = "V4INPMF1"
V4_BLOCK_MAGIC_NORMAL = "V4NRMBLK1"
V4_BLOCK_MAGIC_REVIEW = "V4REVBLK1"
V4_DEFAULT_BLOCK_RECORD_COUNT = 250000


# ---------------------------------------------------------------------------
# Closed enums
# ---------------------------------------------------------------------------


class V4PointRecoveryState(IntEnum):
    POINT_RECOVERY_FAILED = 1
    POINT_RECOVERED_HAG_READY = 2
    POINT_RECOVERED_HAG_REVIEW_REQUIRED = 3
    POINT_RECOVERED_HAG_MISSING = 4
    POINT_RECOVERED_HAG_INVALID = 5


class V4PrimaryRoute(IntEnum):
    NORMAL_SEGMENTATION = 1
    MEASURED_POINT_REVIEW = 2
    UNRECOVERED_REGION_REVIEW = 3
    EXCLUDED_AUDIT_ONLY = 4


class V4InputErrorCode(str, Enum):
    FRV_RESULT_NOT_READY = "FRV_RESULT_NOT_READY"
    FRV_ARTIFACT_INVALID = "FRV_ARTIFACT_INVALID"
    SOURCE_IDENTITY_FAILED = "SOURCE_IDENTITY_FAILED"
    SOURCE_STREAM_INCOMPLETE = "SOURCE_STREAM_INCOMPLETE"
    ORIGINAL_INDEX_INVALID = "ORIGINAL_INDEX_INVALID"
    ORIGINAL_INDEX_DUPLICATE = "ORIGINAL_INDEX_DUPLICATE"
    HAG_EVIDENCE_INVALID = "HAG_EVIDENCE_INVALID"
    ROUTING_CONSERVATION_FAILED = "ROUTING_CONSERVATION_FAILED"
    OUTPUT_PUBLICATION_FAILED = "OUTPUT_PUBLICATION_FAILED"


# ---------------------------------------------------------------------------
# Private strict validation helpers
# ---------------------------------------------------------------------------


def _require_exact_str(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a built-in str")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if value != value.strip():
        raise ValueError(f"{name} must not have surrounding whitespace")
    return value


def _require_sha(value: Any, name: str) -> str:
    text = _require_exact_str(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _require_nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a nonnegative built-in int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_finite_float(value: Any, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite built-in float")
    return value


def _require_enum(value: Any, enum_type: type, name: str) -> Any:
    if type(value) is not enum_type:
        raise ValueError(f"{name} must be a {enum_type.__name__}")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a built-in bool")
    return value


def _exact_fields(data: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(f"{name} must be a mapping")
    actual = set(data)
    missing = fields - actual
    unknown = actual - fields
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    return dict(data)


def _tuple_from_json(value: Any, name: str) -> tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be a JSON array")
    return tuple(value)


def _enum_from_value(value: Any, enum_type: type, name: str) -> Any:
    text = _require_exact_str(value, name)
    try:
        return enum_type(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is unsupported") from exc


def _require_ndarray(
    value: Any,
    name: str,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    finite: bool,
    zero_normalized: bool,
) -> None:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a numpy ndarray")
    if value.dtype != dtype:
        raise ValueError(f"{name} must use dtype {dtype.str}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not value.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be C-contiguous")
    if value.flags.writeable:
        raise ValueError(f"{name} must be read-only")
    if value.dtype.hasobject:
        raise ValueError(f"{name} must not contain object values")
    if finite and not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if zero_normalized:
        for item in value.ravel():
            if item == 0.0 and math.copysign(1.0, float(item)) < 0.0:
                raise ValueError(f"{name} must not contain IEEE negative zero")


def _unique_sha_tuple(value: Any, name: str) -> tuple[str, ...]:
    """Validate a positional SHA-256 inventory without imposing sort order.

    The block digest tuple is positionally aligned with the canonical block
    path inventory (ordinal/path order); SHA values themselves are NOT required
    to be lexicographically sorted.  Only SHA format and uniqueness are
    enforced here; positional alignment is checked against the canonical paths.
    """
    items = _tuple_from_json(value, name)
    for item in items:
        _require_sha(item, f"{name}[]")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} must be unique")
    return items


# ---------------------------------------------------------------------------
# V4ValidatedFrvMembershipView
# ---------------------------------------------------------------------------

_COUNT_FIELDS = (
    "frv_canonical_inventory_count",
    "frv_required_inventory_count",
    "frv_approved_hard_rejection_count",
    "frv_scope_created_or_merged_count",
    "frv_coverage_blocked_count",
    "frv_unique_in_scope_point_count",
    "frv_evaluated_point_count",
    "frv_unasserted_in_scope_count",
    "frv_verified_vegetation_candidate_count",
    "frv_excluded_non_candidate_count",
    "frv_review_required_membership_count",
    "frv_manual_include_count",
    "frv_manual_exclude_count",
)
_BOOL_FIELDS = (
    "frv_canonical_inventory_complete",
    "frv_required_inventory_complete",
    "frv_approved_hard_rejection_complete",
    "frv_scope_coverage_complete",
    "frv_routing_complete",
    "frv_membership_decision_complete",
)


def _check_view(view: V4ValidatedFrvMembershipView) -> None:
    if type(view.source_identity) is not SourceIdentity:
        raise ValueError("source_identity must be a SourceIdentity")
    _require_sha(view.frv_configuration_digest, "frv_configuration_digest")
    if type(view.coverage_manifest) is not FRVScopeCoverageManifest:
        raise ValueError("coverage_manifest must be a FRVScopeCoverageManifest")
    if type(view.membership_artifact) is not FullResolutionVegetationMembershipArtifact:
        raise ValueError("membership_artifact must be a FullResolutionVegetationMembershipArtifact")
    if type(view.payload_schema) is not FRVArtifactPayloadSchema:
        raise ValueError("payload_schema must be a FRVArtifactPayloadSchema")
    if type(view.artifact_binding) is not FRVArtifactBindingRecord:
        raise ValueError("artifact_binding must be a FRVArtifactBindingRecord")
    if type(view.decision_evidence_artifact) is not FrvDecisionEvidenceArtifact:
        raise ValueError(
            "decision_evidence_artifact must be a FrvDecisionEvidenceArtifact"
        )
    for name in ("membership_block_references", "decision_evidence_block_references"):
        value = getattr(view, name)
        if type(value) is not tuple or any(not isinstance(item, Path) for item in value):
            raise ValueError(f"{name} must be a tuple of pathlib.Path values")
    for name in _COUNT_FIELDS:
        object.__setattr__(
            view, name, _require_nonnegative_int(getattr(view, name), name)
        )
    for name in _BOOL_FIELDS:
        _require_bool(getattr(view, name), name)
    _require_sha(view.view_digest, "view_digest")
    if view.view_digest != view.computed_digest():
        raise ValueError("view_digest does not match canonical V4FRVVIEW1 fields")


@dataclass(frozen=True, slots=True)
class V4ValidatedFrvMembershipView:
    """Read-only consumer view of a ready FRV membership execution."""

    source_identity: SourceIdentity
    frv_configuration_digest: str
    coverage_manifest: FRVScopeCoverageManifest
    membership_artifact: FullResolutionVegetationMembershipArtifact
    payload_schema: FRVArtifactPayloadSchema
    artifact_binding: FRVArtifactBindingRecord
    decision_evidence_artifact: FrvDecisionEvidenceArtifact
    membership_block_references: tuple[Path, ...]
    decision_evidence_block_references: tuple[Path, ...]
    frv_canonical_inventory_count: int
    frv_required_inventory_count: int
    frv_approved_hard_rejection_count: int
    frv_scope_created_or_merged_count: int
    frv_coverage_blocked_count: int
    frv_unique_in_scope_point_count: int
    frv_evaluated_point_count: int
    frv_unasserted_in_scope_count: int
    frv_verified_vegetation_candidate_count: int
    frv_excluded_non_candidate_count: int
    frv_review_required_membership_count: int
    frv_manual_include_count: int
    frv_manual_exclude_count: int
    frv_canonical_inventory_complete: bool
    frv_required_inventory_complete: bool
    frv_approved_hard_rejection_complete: bool
    frv_scope_coverage_complete: bool
    frv_routing_complete: bool
    frv_membership_decision_complete: bool
    view_digest: str

    def __post_init__(self) -> None:
        _check_view(self)

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": V4_VALIDATED_VIEW_DIGEST_DOMAIN,
            "source_identity": self.source_identity.to_dict(),
            "frv_configuration_digest": self.frv_configuration_digest,
            "coverage_manifest": self.coverage_manifest.to_dict(),
            "membership_artifact": self.membership_artifact.to_dict(),
            "payload_schema": self.payload_schema.to_dict(),
            "artifact_binding": self.artifact_binding.to_dict(),
            "decision_evidence_artifact": self.decision_evidence_artifact.to_dict(),
            "frv_canonical_inventory_count": self.frv_canonical_inventory_count,
            "frv_required_inventory_count": self.frv_required_inventory_count,
            "frv_approved_hard_rejection_count": self.frv_approved_hard_rejection_count,
            "frv_scope_created_or_merged_count": self.frv_scope_created_or_merged_count,
            "frv_coverage_blocked_count": self.frv_coverage_blocked_count,
            "frv_unique_in_scope_point_count": self.frv_unique_in_scope_point_count,
            "frv_evaluated_point_count": self.frv_evaluated_point_count,
            "frv_unasserted_in_scope_count": self.frv_unasserted_in_scope_count,
            "frv_verified_vegetation_candidate_count": (
                self.frv_verified_vegetation_candidate_count
            ),
            "frv_excluded_non_candidate_count": self.frv_excluded_non_candidate_count,
            "frv_review_required_membership_count": self.frv_review_required_membership_count,
            "frv_manual_include_count": self.frv_manual_include_count,
            "frv_manual_exclude_count": self.frv_manual_exclude_count,
            "frv_canonical_inventory_complete": self.frv_canonical_inventory_complete,
            "frv_required_inventory_complete": self.frv_required_inventory_complete,
            "frv_approved_hard_rejection_complete": (
                self.frv_approved_hard_rejection_complete
            ),
            "frv_scope_coverage_complete": self.frv_scope_coverage_complete,
            "frv_routing_complete": self.frv_routing_complete,
            "frv_membership_decision_complete": self.frv_membership_decision_complete,
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.to_digest_dict())
        payload.pop("digest_domain")
        payload["view_digest"] = self.view_digest
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4ValidatedFrvMembershipView:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["source_identity"] = SourceIdentity.from_dict(
            _mapping(values["source_identity"], "source_identity")
        )
        values["coverage_manifest"] = FRVScopeCoverageManifest.from_dict(
            _mapping(values["coverage_manifest"], "coverage_manifest")
        )
        values["membership_artifact"] = FullResolutionVegetationMembershipArtifact.from_dict(
            _mapping(values["membership_artifact"], "membership_artifact")
        )
        values["payload_schema"] = FRVArtifactPayloadSchema.from_dict(
            _mapping(values["payload_schema"], "payload_schema")
        )
        values["artifact_binding"] = FRVArtifactBindingRecord.from_dict(
            _mapping(values["artifact_binding"], "artifact_binding")
        )
        values["decision_evidence_artifact"] = FrvDecisionEvidenceArtifact.from_dict(
            _mapping(values["decision_evidence_artifact"], "decision_evidence_artifact")
        )
        return cls(**values)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def validate_frv_validated_membership_view(view: V4ValidatedFrvMembershipView) -> None:
    _check_view(view)
    validate_source_content_identity_digest(view.source_identity)
    validate_source_identity_record_digest(view.source_identity)
    validate_frv_decision_evidence_artifact(view.decision_evidence_artifact)


# ---------------------------------------------------------------------------
# V4RecoveredVegetationSet
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4RecoveredVegetationSet:
    """Exact recovered V4 normal-input point set (no downsampled XYZ)."""

    source_identity: SourceIdentity
    original_index: np.ndarray
    xyz_local_m: np.ndarray
    hag_m: np.ndarray
    membership_state: np.ndarray
    hag_evidence_state: np.ndarray
    segmentation_hag_readiness: np.ndarray
    candidate_reason_flags: np.ndarray
    candidate_confidence: None
    frv_membership_artifact_digest: str
    recovered_index_digest: str
    raw_weight: np.ndarray | None = None

    def __post_init__(self) -> None:
        if type(self.source_identity) is not SourceIdentity:
            raise ValueError("source_identity must be a SourceIdentity")
        if self.candidate_confidence is not None:
            raise ValueError("candidate_confidence must be null in this version")
        count = int(self.original_index.shape[0])
        for name, dtype, shape, finite, zero in (
            ("original_index", "<i8", (count,), False, False),
            ("xyz_local_m", "<f8", (count, 3), True, True),
            ("hag_m", "<f4", (count,), True, True),
            ("membership_state", "|u1", (count,), False, False),
            ("hag_evidence_state", "|u1", (count,), False, False),
            ("segmentation_hag_readiness", "|u1", (count,), False, False),
            ("candidate_reason_flags", "<u4", (count,), False, False),
        ):
            _require_ndarray(
                getattr(self, name),
                name,
                dtype=np.dtype(dtype),
                shape=shape,
                finite=finite,
                zero_normalized=zero,
            )
        _require_sha(
            self.frv_membership_artifact_digest, "frv_membership_artifact_digest"
        )
        _require_sha(self.recovered_index_digest, "recovered_index_digest")
        if self.raw_weight is not None:
            _require_ndarray(
                self.raw_weight,
                "raw_weight",
                dtype=np.dtype("<u4"),
                shape=(count,),
                finite=False,
                zero_normalized=False,
            )
            if np.any(np.asarray(self.raw_weight, dtype=np.uint32) == 0):
                raise ValueError("raw_weight must contain strictly positive source populations")


# ---------------------------------------------------------------------------
# V4UnresolvedMeasuredPointSet
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4UnresolvedMeasuredPointSet:
    """Recovered review/HAG-not-ready measured points routed to review."""

    source_identity: SourceIdentity
    original_index: np.ndarray
    xyz_local_m: np.ndarray
    membership_state: np.ndarray
    hag_evidence_state: np.ndarray
    segmentation_hag_readiness: np.ndarray
    raw_hag_m: np.ndarray
    clamped_hag_m: np.ndarray
    reason_flags: np.ndarray
    quality_flags: np.ndarray
    wire_provider_state: np.ndarray
    wire_relation_state: np.ndarray
    nearest_wire_id: np.ndarray
    centerline_distance_m: np.ndarray
    primary_route: np.ndarray
    frv_membership_artifact_digest: str
    source_proof_digest: str
    hag_proof_digest: str
    geometry_proof_digest: str
    wire_proof_digest: str
    set_digest: str

    def __post_init__(self) -> None:
        if type(self.source_identity) is not SourceIdentity:
            raise ValueError("source_identity must be a SourceIdentity")
        count = int(self.original_index.shape[0])
        for name, dtype, shape, finite, zero in (
            ("original_index", "<i8", (count,), False, False),
            ("xyz_local_m", "<f8", (count, 3), True, True),
            ("membership_state", "|u1", (count,), False, False),
            ("hag_evidence_state", "|u1", (count,), False, False),
            ("segmentation_hag_readiness", "|u1", (count,), False, False),
            ("raw_hag_m", "<f8", (count,), True, True),
            ("clamped_hag_m", "<f8", (count,), True, True),
            ("reason_flags", "<u4", (count,), False, False),
            ("quality_flags", "<u4", (count,), False, False),
            ("wire_provider_state", "|u1", (count,), False, False),
            ("wire_relation_state", "|u1", (count,), False, False),
            ("nearest_wire_id", "<i4", (count,), False, False),
            ("centerline_distance_m", "<f8", (count,), True, True),
            ("primary_route", "|u1", (count,), False, False),
        ):
            _require_ndarray(
                getattr(self, name),
                name,
                dtype=np.dtype(dtype),
                shape=shape,
                finite=finite,
                zero_normalized=zero,
            )
        for name in (
            "frv_membership_artifact_digest",
            "source_proof_digest",
            "hag_proof_digest",
            "geometry_proof_digest",
            "wire_proof_digest",
            "set_digest",
        ):
            _require_sha(getattr(self, name), name)


# ---------------------------------------------------------------------------
# V4UnrecoveredRegionRecord
# ---------------------------------------------------------------------------


def _check_unrecovered_region_record(
    record: V4UnrecoveredRegionRecord,
) -> None:
    _require_exact_str(record.schema_version, "schema_version")
    if record.schema_version != V4_UNRECOVERED_REGION_SCHEMA_VERSION:
        raise ValueError("unsupported unrecovered region schema version")
    _require_sha(
        record.source_content_identity_digest, "source_content_identity_digest"
    )
    _require_sha(
        record.frv_scope_coverage_manifest_digest,
        "frv_scope_coverage_manifest_digest",
    )
    _require_exact_str(record.canonical_inventory_key, "canonical_inventory_key")
    if record.scope_id is not None:
        _require_exact_str(record.scope_id, "scope_id")
    if record.frv_scope_union_id is not None:
        _require_exact_str(record.frv_scope_union_id, "frv_scope_union_id")
    if record.scope_union_geometry_digest is not None:
        _require_sha(
            record.scope_union_geometry_digest, "scope_union_geometry_digest"
        )
    if (record.frv_scope_union_id is None) != (
        record.scope_union_geometry_digest is None
    ):
        raise ValueError("union id and geometry digest must be both set or both null")
    if type(record.failure_reason_flags) is not int or record.failure_reason_flags <= 0:
        raise ValueError("failure_reason_flags must be a positive built-in int")
    _require_exact_str(record.safety_relevance, "safety_relevance")
    if record.safety_relevance != "UNKNOWN":
        raise ValueError("safety_relevance must be UNKNOWN in this version")
    _require_bool(record.review_required, "review_required")
    _require_bool(
        record.measured_clearance_complete, "measured_clearance_complete"
    )
    if record.measured_clearance_complete is not False:
        raise ValueError("measured_clearance_complete must be false in this version")
    _require_exact_str(record.diagnostic_reference, "diagnostic_reference")
    _require_sha(record.record_digest, "record_digest")
    if record.record_digest != record.computed_digest():
        raise ValueError("record_digest does not match canonical fields")


@dataclass(frozen=True, slots=True)
class V4UnrecoveredRegionRecord:
    """One blocked required region with no fabricated point/HAG/tree/distance."""

    schema_version: str
    source_content_identity_digest: str
    frv_scope_coverage_manifest_digest: str
    canonical_inventory_key: str
    scope_id: str | None
    frv_scope_union_id: str | None
    scope_union_geometry_digest: str | None
    failure_reason_flags: int
    safety_relevance: str
    review_required: bool
    measured_clearance_complete: bool
    diagnostic_reference: str
    record_digest: str

    def __post_init__(self) -> None:
        _check_unrecovered_region_record(self)

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_content_identity_digest": self.source_content_identity_digest,
            "frv_scope_coverage_manifest_digest": self.frv_scope_coverage_manifest_digest,
            "canonical_inventory_key": self.canonical_inventory_key,
            "scope_id": self.scope_id,
            "frv_scope_union_id": self.frv_scope_union_id,
            "scope_union_geometry_digest": self.scope_union_geometry_digest,
            "failure_reason_flags": self.failure_reason_flags,
            "safety_relevance": self.safety_relevance,
            "review_required": self.review_required,
            "measured_clearance_complete": self.measured_clearance_complete,
            "diagnostic_reference": self.diagnostic_reference,
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4UnrecoveredRegionRecord:
        return cls(**_exact_fields(data, set(cls.__dataclass_fields__), cls.__name__))


# ---------------------------------------------------------------------------
# V4RecoveryCoverageReport
# ---------------------------------------------------------------------------

_REPORT_COUNT_FIELDS = (
    "frv_canonical_inventory_count",
    "frv_required_inventory_count",
    "frv_approved_hard_rejection_count",
    "frv_scope_created_or_merged_count",
    "frv_coverage_blocked_count",
    "frv_unique_in_scope_point_count",
    "frv_evaluated_point_count",
    "frv_unasserted_in_scope_count",
    "frv_verified_vegetation_candidate_count",
    "frv_excluded_non_candidate_count",
    "frv_review_required_membership_count",
    "frv_manual_include_count",
    "frv_manual_exclude_count",
    "normal_segmentation_count",
    "measured_point_review_count",
    "excluded_audit_count",
    "point_recovery_failed_count",
    "unrecovered_region_count",
)
_REPORT_BOOL_FIELDS = (
    "frv_canonical_inventory_complete",
    "frv_required_inventory_complete",
    "frv_approved_hard_rejection_complete",
    "frv_scope_coverage_complete",
    "frv_routing_complete",
    "frv_membership_decision_complete",
    "source_identity_complete",
    "xyz_recovery_complete",
    "hag_ready_complete",
    "segmentation_input_complete",
    "v4_routing_complete",
    "measured_clearance_complete",
    "review_required",
)


def _check_coverage_report(report: V4RecoveryCoverageReport) -> None:
    _require_exact_str(report.schema_version, "schema_version")
    if report.schema_version != V4_RECOVERY_COVERAGE_SCHEMA_VERSION:
        raise ValueError("unsupported recovery coverage schema version")
    _require_sha(
        report.source_content_identity_digest, "source_content_identity_digest"
    )
    _require_sha(
        report.source_identity_record_digest, "source_identity_record_digest"
    )
    _require_sha(
        report.frv_scope_coverage_manifest_digest,
        "frv_scope_coverage_manifest_digest",
    )
    _require_exact_str(report.frv_scope_union_id, "frv_scope_union_id")
    _require_sha(report.frv_upstream_inventory_digest, "frv_upstream_inventory_digest")
    _require_sha(report.frv_membership_artifact_digest, "frv_membership_artifact_digest")
    _require_sha(report.recovered_index_digest, "recovered_index_digest")
    for name in _REPORT_COUNT_FIELDS:
        object.__setattr__(
            report, name, _require_nonnegative_int(getattr(report, name), name)
        )
    for name in _REPORT_BOOL_FIELDS:
        _require_bool(getattr(report, name), name)
    object.__setattr__(
        report, "failure_codes", _failure_codes(report.failure_codes)
    )
    _require_sha(report.report_digest, "report_digest")
    if report.report_digest != report.computed_digest():
        raise ValueError("report_digest does not match canonical V4COV1 fields")


def _failure_codes(value: Any) -> tuple[str, ...]:
    items = _tuple_from_json(value, "failure_codes")
    order = {member.value: index for index, member in enumerate(V4InputErrorCode)}
    validated = []
    for item in items:
        text = _require_exact_str(item, "failure_codes[]")
        if text not in order:
            raise ValueError("failure_codes must use V4InputErrorCode values")
        validated.append(text)
    if len(set(validated)) != len(validated):
        raise ValueError("failure_codes must be unique")
    if tuple(sorted(validated, key=order.__getitem__)) != tuple(validated):
        raise ValueError("failure_codes must use declaration order")
    return tuple(validated)


@dataclass(frozen=True, slots=True)
class V4RecoveryCoverageReport:
    """Exact D05 recovery-coverage report (schema v4-recovery-coverage-v1)."""

    schema_version: str
    source_content_identity_digest: str
    source_identity_record_digest: str
    frv_scope_coverage_manifest_digest: str
    frv_scope_union_id: str
    frv_upstream_inventory_digest: str
    frv_membership_artifact_digest: str
    recovered_index_digest: str
    frv_canonical_inventory_count: int
    frv_required_inventory_count: int
    frv_approved_hard_rejection_count: int
    frv_scope_created_or_merged_count: int
    frv_coverage_blocked_count: int
    frv_unique_in_scope_point_count: int
    frv_evaluated_point_count: int
    frv_unasserted_in_scope_count: int
    frv_verified_vegetation_candidate_count: int
    frv_excluded_non_candidate_count: int
    frv_review_required_membership_count: int
    frv_manual_include_count: int
    frv_manual_exclude_count: int
    normal_segmentation_count: int
    measured_point_review_count: int
    excluded_audit_count: int
    point_recovery_failed_count: int
    unrecovered_region_count: int
    frv_canonical_inventory_complete: bool
    frv_required_inventory_complete: bool
    frv_approved_hard_rejection_complete: bool
    frv_scope_coverage_complete: bool
    frv_routing_complete: bool
    frv_membership_decision_complete: bool
    source_identity_complete: bool
    xyz_recovery_complete: bool
    hag_ready_complete: bool
    segmentation_input_complete: bool
    v4_routing_complete: bool
    measured_clearance_complete: bool
    review_required: bool
    failure_codes: tuple[str, ...]
    report_digest: str

    def __post_init__(self) -> None:
        _check_coverage_report(self)

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": V4_COVERAGE_DIGEST_DOMAIN,
            "schema_version": self.schema_version,
            "source_content_identity_digest": self.source_content_identity_digest,
            "source_identity_record_digest": self.source_identity_record_digest,
            "frv_scope_coverage_manifest_digest": self.frv_scope_coverage_manifest_digest,
            "frv_scope_union_id": self.frv_scope_union_id,
            "frv_upstream_inventory_digest": self.frv_upstream_inventory_digest,
            "frv_membership_artifact_digest": self.frv_membership_artifact_digest,
            "recovered_index_digest": self.recovered_index_digest,
            "frv_canonical_inventory_count": self.frv_canonical_inventory_count,
            "frv_required_inventory_count": self.frv_required_inventory_count,
            "frv_approved_hard_rejection_count": self.frv_approved_hard_rejection_count,
            "frv_scope_created_or_merged_count": self.frv_scope_created_or_merged_count,
            "frv_coverage_blocked_count": self.frv_coverage_blocked_count,
            "frv_unique_in_scope_point_count": self.frv_unique_in_scope_point_count,
            "frv_evaluated_point_count": self.frv_evaluated_point_count,
            "frv_unasserted_in_scope_count": self.frv_unasserted_in_scope_count,
            "frv_verified_vegetation_candidate_count": (
                self.frv_verified_vegetation_candidate_count
            ),
            "frv_excluded_non_candidate_count": self.frv_excluded_non_candidate_count,
            "frv_review_required_membership_count": self.frv_review_required_membership_count,
            "frv_manual_include_count": self.frv_manual_include_count,
            "frv_manual_exclude_count": self.frv_manual_exclude_count,
            "normal_segmentation_count": self.normal_segmentation_count,
            "measured_point_review_count": self.measured_point_review_count,
            "excluded_audit_count": self.excluded_audit_count,
            "point_recovery_failed_count": self.point_recovery_failed_count,
            "unrecovered_region_count": self.unrecovered_region_count,
            "frv_canonical_inventory_complete": self.frv_canonical_inventory_complete,
            "frv_required_inventory_complete": self.frv_required_inventory_complete,
            "frv_approved_hard_rejection_complete": (
                self.frv_approved_hard_rejection_complete
            ),
            "frv_scope_coverage_complete": self.frv_scope_coverage_complete,
            "frv_routing_complete": self.frv_routing_complete,
            "frv_membership_decision_complete": self.frv_membership_decision_complete,
            "source_identity_complete": self.source_identity_complete,
            "xyz_recovery_complete": self.xyz_recovery_complete,
            "hag_ready_complete": self.hag_ready_complete,
            "segmentation_input_complete": self.segmentation_input_complete,
            "v4_routing_complete": self.v4_routing_complete,
            "measured_clearance_complete": self.measured_clearance_complete,
            "review_required": self.review_required,
            "failure_codes": list(self.failure_codes),
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.to_digest_dict())
        payload.pop("digest_domain")
        payload["report_digest"] = self.report_digest
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4RecoveryCoverageReport:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["failure_codes"] = _tuple_from_json(values["failure_codes"], "failure_codes")
        return cls(**values)


def validate_frv_recovery_coverage_report(report: V4RecoveryCoverageReport) -> None:
    _check_coverage_report(report)
    if report.measured_clearance_complete is not False:
        raise ValueError("measured_clearance_complete must be false in this version")


# ---------------------------------------------------------------------------
# V4InputOutputReference (runtime-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4InputOutputReference:
    """Runtime access information for a published V4 input handoff."""

    root_path: Path
    validated_view_json_path: Path
    coverage_json_path: Path
    normal_block_paths: tuple[Path, ...]
    review_block_paths: tuple[Path, ...]
    handoff_manifest_path: Path

    def __post_init__(self) -> None:
        for name in (
            "root_path",
            "validated_view_json_path",
            "coverage_json_path",
            "handoff_manifest_path",
        ):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a pathlib.Path instance")
        for name in ("normal_block_paths", "review_block_paths"):
            value = getattr(self, name)
            if type(value) is not tuple or any(
                not isinstance(item, Path) for item in value
            ):
                raise ValueError(f"{name} must be a tuple of pathlib.Path values")


# ---------------------------------------------------------------------------
# V4InputArtifactManifest (v2 digest-bound handoff manifest)
# ---------------------------------------------------------------------------


def _require_relative_block_paths(value: Any, name: str) -> tuple[str, ...]:
    items = _tuple_from_json(value, name)
    for item in items:
        text = _require_exact_str(item, f"{name}[]")
        if text != text.strip() or "\\" in text:
            raise ValueError(f"{name}[] must be a canonical relative path")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} must be unique")
    return items


def _require_finite_vec3(value: Any, name: str) -> tuple[float, float, float]:
    if type(value) not in (list, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a three-value JSON array")
    return (
        _require_finite_float(value[0], f"{name}[0]"),
        _require_finite_float(value[1], f"{name}[1]"),
        _require_finite_float(value[2], f"{name}[2]"),
    )


def _check_input_artifact_manifest(manifest: V4InputArtifactManifest) -> None:
    if type(manifest.schema_version) is not str:
        raise ValueError("schema_version must be a built-in str")
    if manifest.schema_version != V4_HANDOFF_MANIFEST_SCHEMA_VERSION_V2:
        raise ValueError("unsupported V4 input manifest schema version")
    _require_sha(
        manifest.source_content_identity_digest, "source_content_identity_digest"
    )
    _require_sha(
        manifest.source_identity_record_digest, "source_identity_record_digest"
    )
    _require_sha(
        manifest.frv_membership_artifact_digest, "frv_membership_artifact_digest"
    )
    _require_sha(manifest.report_digest, "report_digest")
    for name in ("normal_block_count", "review_block_count"):
        _require_nonnegative_int(getattr(manifest, name), name)
    for name in ("normal_row_count", "review_row_count"):
        _require_nonnegative_int(getattr(manifest, name), name)
    normal_paths = _require_relative_block_paths(
        manifest.normal_block_paths, "normal_block_paths"
    )
    review_paths = _require_relative_block_paths(
        manifest.review_block_paths, "review_block_paths"
    )
    normal_digests = _unique_sha_tuple(
        manifest.normal_block_digests, "normal_block_digests"
    )
    review_digests = _unique_sha_tuple(
        manifest.review_block_digests, "review_block_digests"
    )
    if len(normal_paths) != manifest.normal_block_count:
        raise ValueError("normal_block_paths length must equal normal_block_count")
    if len(review_paths) != manifest.review_block_count:
        raise ValueError("review_block_paths length must equal review_block_count")
    if len(normal_digests) != manifest.normal_block_count:
        raise ValueError("normal_block_digests length must equal normal_block_count")
    if len(review_digests) != manifest.review_block_count:
        raise ValueError("review_block_digests length must equal review_block_count")
    if not all(
        path == f"normal/v4nrm-block-{ordinal:08d}.v4blk"
        for ordinal, path in enumerate(normal_paths)
    ):
        raise ValueError("normal_block_paths must use the canonical block inventory")
    if not all(
        path == f"review/v4rev-block-{ordinal:08d}.v4blk"
        for ordinal, path in enumerate(review_paths)
    ):
        raise ValueError("review_block_paths must use the canonical block inventory")
    for name in (
        "source_proof_digest",
        "hag_proof_digest",
        "geometry_proof_digest",
        "wire_proof_digest",
    ):
        _require_sha(getattr(manifest, name), name)
    _require_sha(manifest.normal_set_digest, "normal_set_digest")
    _require_sha(manifest.review_set_digest, "review_set_digest")
    _require_sha(manifest.handoff_digest, "handoff_digest")
    _require_sha(manifest.manifest_digest, "manifest_digest")
    if manifest.manifest_digest != manifest.computed_digest():
        raise ValueError("manifest_digest does not match canonical V4INPMF1 fields")


@dataclass(frozen=True, slots=True)
class V4InputArtifactManifest:
    """Digest-bound V4 input handoff manifest (schema v2).

    The v2 manifest persists the canonical normal/review block relative paths,
    per-block SHA-256 digests, the set digests and the final handoff digest so a
    separate process can re-validate every published byte before segmentation.
    The ``manifest_digest`` binds the complete v2 manifest inventory.
    """

    schema_version: str
    source_content_identity_digest: str
    source_identity_record_digest: str
    origin_context: tuple[float, float, float]
    frv_membership_artifact_digest: str
    report_digest: str
    source_proof_digest: str
    hag_proof_digest: str
    geometry_proof_digest: str
    wire_proof_digest: str
    normal_block_count: int
    review_block_count: int
    normal_row_count: int
    review_row_count: int
    normal_block_paths: tuple[str, ...]
    review_block_paths: tuple[str, ...]
    normal_block_digests: tuple[str, ...]
    review_block_digests: tuple[str, ...]
    normal_set_digest: str
    review_set_digest: str
    handoff_digest: str
    manifest_digest: str

    def __post_init__(self) -> None:
        _check_input_artifact_manifest(self)

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": V4_INPUT_MANIFEST_DIGEST_DOMAIN,
            "schema_version": self.schema_version,
            "source_content_identity_digest": self.source_content_identity_digest,
            "source_identity_record_digest": self.source_identity_record_digest,
            "origin_context": list(self.origin_context),
            "frv_membership_artifact_digest": self.frv_membership_artifact_digest,
            "report_digest": self.report_digest,
            "source_proof_digest": self.source_proof_digest,
            "hag_proof_digest": self.hag_proof_digest,
            "geometry_proof_digest": self.geometry_proof_digest,
            "wire_proof_digest": self.wire_proof_digest,
            "normal_block_count": self.normal_block_count,
            "review_block_count": self.review_block_count,
            "normal_row_count": self.normal_row_count,
            "review_row_count": self.review_row_count,
            "normal_block_paths": list(self.normal_block_paths),
            "review_block_paths": list(self.review_block_paths),
            "normal_block_digests": list(self.normal_block_digests),
            "review_block_digests": list(self.review_block_digests),
            "normal_set_digest": self.normal_set_digest,
            "review_set_digest": self.review_set_digest,
            "handoff_digest": self.handoff_digest,
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.to_digest_dict())
        payload.pop("digest_domain")
        payload["manifest_digest"] = self.manifest_digest
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4InputArtifactManifest:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["origin_context"] = _require_finite_vec3(
            values["origin_context"], "origin_context"
        )
        values["normal_block_paths"] = _require_relative_block_paths(
            values["normal_block_paths"], "normal_block_paths"
        )
        values["review_block_paths"] = _require_relative_block_paths(
            values["review_block_paths"], "review_block_paths"
        )
        values["normal_block_digests"] = _unique_sha_tuple(
            values["normal_block_digests"], "normal_block_digests"
        )
        values["review_block_digests"] = _unique_sha_tuple(
            values["review_block_digests"], "review_block_digests"
        )
        return cls(**values)


def validate_v4_input_artifact_manifest(manifest: V4InputArtifactManifest) -> None:
    _check_input_artifact_manifest(manifest)


# ---------------------------------------------------------------------------
# V4InputHandoffResult
# ---------------------------------------------------------------------------


def _check_handoff_result(result: V4InputHandoffResult) -> None:
    if result.validated_membership is not None and type(
        result.validated_membership
    ) is not V4ValidatedFrvMembershipView:
        raise ValueError(
            "validated_membership must be null or a V4ValidatedFrvMembershipView"
        )
    if result.recovered_vegetation is not None and type(
        result.recovered_vegetation
    ) is not V4RecoveredVegetationSet:
        raise ValueError(
            "recovered_vegetation must be null or a V4RecoveredVegetationSet"
        )
    if result.unresolved_measured_points is not None and type(
        result.unresolved_measured_points
    ) is not V4UnresolvedMeasuredPointSet:
        raise ValueError(
            "unresolved_measured_points must be null or a V4UnresolvedMeasuredPointSet"
        )
    for item in result.unrecovered_regions:
        if type(item) is not V4UnrecoveredRegionRecord:
            raise ValueError(
                "unrecovered_regions must contain V4UnrecoveredRegionRecord values"
            )
    if type(result.coverage_report) is not V4RecoveryCoverageReport:
        raise ValueError("coverage_report must be a V4RecoveryCoverageReport")
    if result.output_reference is not None and type(
        result.output_reference
    ) is not V4InputOutputReference:
        raise ValueError("output_reference must be null or a V4InputOutputReference")
    nonnull = (
        result.validated_membership is not None,
        result.recovered_vegetation is not None,
        result.unresolved_measured_points is not None,
    )
    if nonnull not in ((True, True, True), (False, False, False)):
        raise ValueError(
            "validated view, recovered set, and unresolved set must be "
            "all null or all nonnull"
        )
    _require_sha(result.handoff_digest, "handoff_digest")


@dataclass(frozen=True, slots=True)
class V4InputHandoffResult:
    """Exact V4 input handoff result; never contains tree/apex/crown/CHM fields."""

    validated_membership: V4ValidatedFrvMembershipView | None
    recovered_vegetation: V4RecoveredVegetationSet | None
    unresolved_measured_points: V4UnresolvedMeasuredPointSet | None
    unrecovered_regions: tuple[V4UnrecoveredRegionRecord, ...]
    coverage_report: V4RecoveryCoverageReport
    output_reference: V4InputOutputReference | None
    handoff_digest: str

    def __post_init__(self) -> None:
        _check_handoff_result(self)


# ---------------------------------------------------------------------------
# V4RECIDX1 recovered-index digest
# ---------------------------------------------------------------------------


def frv_v4_recovered_index_digest(
    *,
    source_content_identity_digest: str,
    original_index: Any,
) -> str:
    """Return the V4RECIDX1 byte-stream digest over the recovered indexes."""


    _require_sha(source_content_identity_digest, "source_content_identity_digest")
    indexes = np.asarray(original_index)
    if indexes.ndim != 1 or indexes.dtype.kind not in "iu":
        raise ValueError("original_index must be a one-dimensional integer array")
    count = int(indexes.shape[0])
    hasher = hashlib.sha256()
    hasher.update(b"V4RECIDX1\0")
    hasher.update(source_content_identity_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(struct.pack("<Q", count))
    for value in indexes:
        hasher.update(struct.pack("<Q", int(value)))
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# V4INP1 handoff byte-framing digest
# ---------------------------------------------------------------------------


def frv_v4_handoff_digest(
    *,
    source_content_identity_digest: str,
    frv_membership_artifact_digest: str,
    report_digest: str,
    normal_set_digest: str,
    review_set_digest: str,
    unrecovered_regions: tuple[V4UnrecoveredRegionRecord, ...],
) -> str:
    """Return the V4INP1 byte-framing digest of a complete V4 input handoff."""


    _require_sha(source_content_identity_digest, "source_content_identity_digest")
    _require_sha(frv_membership_artifact_digest, "frv_membership_artifact_digest")
    _require_sha(report_digest, "report_digest")
    _require_sha(normal_set_digest, "normal_set_digest")
    _require_sha(review_set_digest, "review_set_digest")
    records = tuple(unrecovered_regions)
    for record in records:
        if type(record) is not V4UnrecoveredRegionRecord:
            raise ValueError(
                "unrecovered_regions must contain V4UnrecoveredRegionRecord values"
            )
    hasher = hashlib.sha256()
    hasher.update(b"V4INP1\0")
    hasher.update(source_content_identity_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(frv_membership_artifact_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(report_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(normal_set_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(review_set_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(struct.pack("<Q", len(records)))
    for record in records:
        hasher.update(bytes.fromhex(record.record_digest))
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# V4InputLoadError
# ---------------------------------------------------------------------------


class V4InputLoadError(Exception):
    """Closed V4 handoff load failure."""

    def __init__(self, *, code: V4InputErrorCode, field_name: str) -> None:
        if type(code) is not V4InputErrorCode:
            raise TypeError("code must be a V4InputErrorCode")
        _require_exact_str(field_name, "field_name")
        self.code = code
        self.field_name = field_name
        super().__init__(f"{code.value}:{field_name}")


__all__ = [
    "V4_VALIDATED_VIEW_SCHEMA_VERSION",
    "V4_RECOVERED_SET_SCHEMA_VERSION",
    "V4_UNRESOLVED_SET_SCHEMA_VERSION",
    "V4_UNRECOVERED_REGION_SCHEMA_VERSION",
    "V4_RECOVERY_COVERAGE_SCHEMA_VERSION",
    "V4_HANDOFF_SCHEMA_VERSION",
    "V4_HANDOFF_MANIFEST_SCHEMA_VERSION_V2",
    "V4_VALIDATED_VIEW_DIGEST_DOMAIN",
    "V4_RECOVERED_INDEX_DIGEST_DOMAIN",
    "V4_RECOVERED_SET_DIGEST_DOMAIN",
    "V4_UNRESOLVED_SET_DIGEST_DOMAIN",
    "V4_COVERAGE_DIGEST_DOMAIN",
    "V4_HANDOFF_DIGEST_DOMAIN",
    "V4_INPUT_MANIFEST_DIGEST_DOMAIN",
    "V4_BLOCK_MAGIC_NORMAL",
    "V4_BLOCK_MAGIC_REVIEW",
    "V4_DEFAULT_BLOCK_RECORD_COUNT",
    "V4PointRecoveryState",
    "V4PrimaryRoute",
    "V4InputErrorCode",
    "V4ValidatedFrvMembershipView",
    "V4RecoveredVegetationSet",
    "V4UnresolvedMeasuredPointSet",
    "V4UnrecoveredRegionRecord",
    "V4RecoveryCoverageReport",
    "V4InputOutputReference",
    "V4InputArtifactManifest",
    "V4InputHandoffResult",
    "V4InputLoadError",
    "frv_v4_recovered_index_digest",
    "frv_v4_handoff_digest",
    "validate_frv_validated_membership_view",
    "validate_frv_recovery_coverage_report",
    "validate_v4_input_artifact_manifest",
]
