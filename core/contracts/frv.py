"""FRV-A1 immutable contracts and pure validation foundation.

FRV (Full-Resolution Vegetation Membership Evidence) is the frozen upstream
vegetation-candidate boundary consumed by V4.  FRV-A1 defines closed enums,
immutable configuration/identity/scope/evidence/payload/manifest contracts,
canonical serialization with self-digests, and pure structural validators.

A1 never accesses a filesystem, reads point or payload bytes, computes a
source/payload/geometry hash from data, discovers inventory, constructs a
scope/union/partition, validates provider truth, executes membership or
overrides, publishes/replays an artifact, or supplies V4 runtime data.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum, IntFlag
from typing import Any

import numpy as np

from core.contracts._json import reject_unknown_fields, require_fields

# ---------------------------------------------------------------------------
# Frozen schema/version vocabulary (module-level, not exported publicly)
# ---------------------------------------------------------------------------

FRV_CONFIGURATION_SCHEMA_VERSION = "frv-configuration-v1"
FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION = "frv-canonical-point-content-v1"
FRV_RELOCATION_AUDIT_SCHEMA_VERSION = "frv-relocation-audit-v1"
FRV_SOURCE_EQUIVALENCE_AUDIT_SCHEMA_VERSION = "frv-source-equivalence-audit-v1"
FRV_SOURCE_IDENTITY_SCHEMA_VERSION = "frv-source-identity-v1"
FRV_SCOPE_GEOMETRY_ENCODING_SCHEMA_VERSION = "frv-scope-geometry-encoding-v1"
FRV_SCOPE_DEFINITION_SCHEMA_VERSION = "frv-scope-definition-v1"
FRV_HARD_REJECTION_EVIDENCE_SCHEMA_VERSION = "frv-hard-rejection-evidence-v1"
FRV_SCOPE_COVERAGE_RECORD_SCHEMA_VERSION = "frv-scope-coverage-record-v1"
FRV_SCOPE_COVERAGE_MANIFEST_SCHEMA_VERSION = "frv-scope-coverage-manifest-v1"
FRV_MANUAL_OVERRIDE_SCHEMA_VERSION = "frv-manual-override-v1"
FRV_ARTIFACT_PAYLOAD_SCHEMA_VERSION = "frv-artifact-payload-v1"
FRV_ARTIFACT_BLOCK_SCHEMA_VERSION = "frv-artifact-block-v1"
FRV_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION = "frv-membership-artifact-v2"
FRV_MEMBERSHIP_ARTIFACT_LEGACY_SCHEMA_VERSION = "frv-membership-artifact-v1"
FRV_ARTIFACT_BINDING_SCHEMA_VERSION = "frv-artifact-binding-v1"

FRV_PAYLOAD_ENCODING_ID = "FRV-BLOCK-PAYLOAD-1"
FRV_PAYLOAD_ENCODER_VERSION = "1"
FRV_PAYLOAD_COMPRESSION_ALGORITHM = "NONE"
FRV_PAYLOAD_COMPRESSION_VERSION = "1"

# Closed B1 canonical point-content policy vocabulary (first version).
_FRV_POINT_ROW_ORDERS = frozenset({"ORIGINAL_INDEX_ASCENDING"})
_FRV_POINT_BYTE_ORDERS = frozenset({"LITTLE_ENDIAN"})
_FRV_POINT_QUANTIZATION_POLICIES = frozenset({"EXACT_TYPED_VALUE", "SCALED_INTEGER"})
_FRV_POINT_NULL_POLICIES = frozenset({"FORBID_NULL", "VALIDITY_BYTE"})
_FRV_POINT_MISSING_VALUE_POLICIES = frozenset({"EXPLICIT_VALIDITY_BYTE"})
_FRV_POINT_DTYPE_CODES = frozenset({"<i8", "<u8", "<i4", "<u4", "<f4", "<f8", "|u1"})
_FRV_INTEGER_DTYPE_CODES = frozenset({"<i8", "<u8", "<i4", "<u4"})
_FRV_PAYLOAD_DTYPE_CODES = frozenset({"<i8", "<u8", "<i4", "<u4", "<f4", "<f8", "|u1"})

# Optional arrays permitted by the frozen payload inventory.
ALLOWED_OPTIONAL_FRV_ARRAYS = frozenset(
    {"raw_hag", "clamped_hag", "hag_quality_flags", "hag_reason_flags", "candidate_confidence"}
)

# First-version protocol literals (owner-approved; reused by the test factories).
_FRV_CANONICALIZATION_VERSION = "canon-1"
_FRV_POINT_FRAMING_VERSION = "framing-v1"
_FRV_POINT_SHAPE_RULE = "[N]"
_FRV_MEMBERSHIP_STATE_SCHEMA_VERSION = "ms-v1"
_FRV_REASON_FLAG_SCHEMA_VERSION = "rf-v1"
_FRV_QUALITY_FLAG_SCHEMA_VERSION = "qf-v1"
_FRV_HAG_EVIDENCE_SCHEMA_VERSION = "he-v1"
_FRV_SEGMENTATION_HAG_READINESS_SCHEMA_VERSION = "shr-v1"

# First-version geometry encoding literals (owner-approved).
_FRV_GEOMETRY_BOX_ORDER_RULE = "lexicographic-six-tuple"
_FRV_GEOMETRY_FLOAT_ENCODING = "IEEE754_BINARY64"
_FRV_GEOMETRY_BYTE_ORDER = "LITTLE_ENDIAN"
_FRV_GEOMETRY_NEGATIVE_ZERO_POLICY = "normalize-to-positive-zero"
_FRV_GEOMETRY_BOX_FRAMING_VERSION = "box-framing-v1"
_FRV_GEOMETRY_ROLES = frozenset({"SCOPE_CORE", "SCOPE_CONTEXT_MARGIN", "SCOPE_UNION"})

# Relocation warning flags: the plan defines no first-version public warning
# bit, so the only valid mask is zero.
_FRV_RELOCATION_WARNING_KNOWN_MASK = 0

# Declared membership-payload domain identifier (FRVART1).
_FRV_ART1_MAGIC = "FRVART1"

# Exact first-version required point-aligned payload arrays: field_name,
# dtype_code, shape_rule, enum_or_flag_schema_version.
REQUIRED_FRV_ARRAY_SPECS = (
    ("original_index", "<i8", _FRV_POINT_SHAPE_RULE, None),
    ("membership_state", "|u1", _FRV_POINT_SHAPE_RULE, _FRV_MEMBERSHIP_STATE_SCHEMA_VERSION),
    ("reason_flags", "<u4", _FRV_POINT_SHAPE_RULE, _FRV_REASON_FLAG_SCHEMA_VERSION),
    ("input_quality_flags", "<u4", _FRV_POINT_SHAPE_RULE, _FRV_QUALITY_FLAG_SCHEMA_VERSION),
    ("hag_evidence_state", "|u1", _FRV_POINT_SHAPE_RULE, _FRV_HAG_EVIDENCE_SCHEMA_VERSION),
    (
        "segmentation_hag_readiness",
        "|u1",
        _FRV_POINT_SHAPE_RULE,
        _FRV_SEGMENTATION_HAG_READINESS_SCHEMA_VERSION,
    ),
    ("scope_union_id_reference", "<u4", _FRV_POINT_SHAPE_RULE, None),
    ("primary_scope_id_reference", "<u4", _FRV_POINT_SHAPE_RULE, None),
    ("supporting_region_offset", "<u8", _FRV_POINT_SHAPE_RULE, None),
    ("supporting_region_count", "<u4", _FRV_POINT_SHAPE_RULE, None),
    ("manual_override_reference", "<u4", _FRV_POINT_SHAPE_RULE, None),
)
_REQUIRED_ARRAY_NAMES = frozenset(spec[0] for spec in REQUIRED_FRV_ARRAY_SPECS)

# ---------------------------------------------------------------------------
# Canonical serialization and digest
# ---------------------------------------------------------------------------


def _frv_json_value(value: Any) -> Any:
    """Normalize *value* into the exact FRV canonical JSON value tree."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("canonical FRV JSON mapping keys must be strings")
            result[key] = _frv_json_value(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_frv_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise ValueError("canonical FRV JSON forbids set values")
    raise ValueError(f"canonical FRV JSON forbids {type(value).__name__}")


def canonical_frv_json_bytes(value: Any) -> bytes:
    """Return the fixed FRV canonical JSON byte representation of *value*.

    UTF-8, ``ensure_ascii=False``, lexicographically sorted keys, compact
    separators, ``allow_nan=False``, tuples serialized as lists, mapping keys
    restricted to strings, and set/arbitrary objects rejected.
    """

    return json.dumps(
        _frv_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def frv_digest(value: Any) -> str:
    """Return the lowercase SHA-256 of the canonical FRV JSON bytes of *value*."""

    return hashlib.sha256(canonical_frv_json_bytes(value)).hexdigest()


# ---------------------------------------------------------------------------
# Strict scalar validation helpers
# ---------------------------------------------------------------------------


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _sha_or_null(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, name)


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must not be a boolean")
    if not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _uint32(value: Any, name: str) -> int:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must not be a boolean")
    if not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be a uint32 integer")
    integer = int(value)
    if integer < 0 or integer > 0xFFFFFFFF:
        raise ValueError(f"{name} must fit in uint32")
    return integer


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must not be a boolean")
    if not isinstance(value, int | float | np.integer | np.floating):
        raise ValueError(f"{name} must be a finite numeric value")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _finite_vec3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a three-item numeric sequence")
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return (
        _finite_float(value[0], f"{name}[0]"),
        _finite_float(value[1], f"{name}[1]"),
        _finite_float(value[2], f"{name}[2]"),
    )


def _required_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a non-string sequence")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must contain only strings")
    return tuple(value)


def _sorted_unique_sha_tuple(value: Any, name: str) -> tuple[str, ...]:
    items = _string_tuple(value, name)
    for item in items:
        _sha256(item, f"{name}[]")
    if items != tuple(sorted(items)) or len(items) != len(set(items)):
        raise ValueError(f"{name} must be lexicographically sorted and unique")
    return items


def _enum_value(value: Any, enum_cls: type, name: str) -> Any:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must not be a boolean")
    if not isinstance(value, enum_cls):
        raise ValueError(f"{name} must be a {enum_cls.__name__}")
    return value


def _known_flag_mask(flag_cls: type) -> int:
    known = 0
    for member in flag_cls.__members__.values():
        known |= int(member.value)
    return known


def _flag_mask(value: Any, flag_cls: type, name: str) -> Any:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must not be a boolean")
    if not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer flag mask")
    integer = int(value)
    if integer < 0 or integer > 0xFFFFFFFF:
        raise ValueError(f"{name} must fit in uint32")
    known = _known_flag_mask(flag_cls)
    if integer & ~known:
        raise ValueError(f"{name} contains unknown flag bit(s)")
    return flag_cls(integer)


def _enum_from_value(value: Any, enum_cls: type, name: str) -> Any:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must not be a boolean")
    try:
        return enum_cls(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is unsupported") from exc


def _enum_from_value_or_none(value: Any, enum_cls: type, name: str) -> Any:
    if value is None:
        return None
    return _enum_from_value(value, enum_cls, name)


def _flag_from_value(value: Any, flag_cls: type, name: str) -> Any:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must not be a boolean")
    if not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer flag mask")
    integer = int(value)
    if integer < 0 or integer > 0xFFFFFFFF:
        raise ValueError(f"{name} must fit in uint32")
    known = _known_flag_mask(flag_cls)
    if integer & ~known:
        raise ValueError(f"{name} contains unknown flag bit(s)")
    return flag_cls(integer)


def _strict_mapping(data: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return data


def _from_dict_values(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    values = _strict_mapping(data, cls.__name__)
    fields = set(cls.__dataclass_fields__)
    require_fields(values, fields, cls.__name__)
    reject_unknown_fields(values, fields, cls.__name__)
    return dict(values)


# ---------------------------------------------------------------------------
# A1.1 Closed enums, flags, and error codes
# ---------------------------------------------------------------------------


class FrvMembershipState(IntEnum):
    """Five terminal full-resolution membership states."""

    VERIFIED_VEGETATION_CANDIDATE = 1
    EXCLUDED_NON_CANDIDATE = 2
    REVIEW_REQUIRED_MEMBERSHIP = 3
    MANUAL_INCLUDE = 4
    MANUAL_EXCLUDE = 5


class FrvScopeMode(IntEnum):
    """Scope construction modes; first production version is EXHAUSTIVE_REGION."""

    EXHAUSTIVE_REGION = 1
    EXHAUSTIVE_SOURCE = 2
    SPARSE_ASSERTIONS_ONLY = 3


class FrvHagEvidenceState(IntEnum):
    """Closed HAG evidence states evaluated at full resolution."""

    HAG_VALID_POSITIVE = 1
    HAG_VALID_NEAR_ZERO = 2
    HAG_WITHIN_NEGATIVE_TOLERANCE = 3
    HAG_MISSING = 4
    HAG_INVALID = 5
    HAG_OUT_OF_DOMAIN = 6


class FrvSegmentationHagReadiness(IntEnum):
    """Independent segmentation HAG readiness flag for V4 hand-off."""

    READY = 1
    REVIEW_REQUIRED = 2


class FrvInventoryDisposition(IntEnum):
    """The only two canonical-inventory dispositions; never inferred."""

    REQUIRED_FRV = 1
    APPROVED_HARD_REJECTION = 2


class FrvScopeConstructionState(IntEnum):
    """Routing state of one canonical inventory record into scope construction."""

    SCOPE_CREATED = 1
    SCOPE_MERGED = 2
    NOT_CREATED = 3


class FrvScopeCoverageState(IntEnum):
    """Coverage outcome for one canonical inventory record or required region."""

    COVERED = 1
    SCOPE_BLOCKED = 2
    SOURCE_IDENTITY_FAILED = 3
    UPSTREAM_EVIDENCE_INVALID = 4
    REGION_GEOMETRY_INVALID = 5
    REGION_MAPPING_INCOMPLETE = 6
    OUT_OF_FRV_POLICY = 7  # reserved; prohibited from first-version production output


class FrvWireRelationState(IntEnum):
    """Point-level wire relation; confirmed body has no first-version producer."""

    WIRE_BODY_CONFIRMED = 1
    WIRE_BODY_POSSIBLE = 2
    WIRE_PROXIMITY_ONLY = 3
    WIRE_VEGETATION_CONFLICT = 4


class FrvWireProviderState(IntEnum):
    """Provider/output state; not-applicable is a runtime result only."""

    WIRE_RELATION_VALID = 1
    WIRE_NOT_APPLICABLE_PROVEN = 2
    WIRE_PROVIDER_MISSING = 3
    WIRE_PROVIDER_INVALID = 4
    WIRE_IDENTITY_MISMATCH = 5
    WIRE_GEOMETRY_UNAVAILABLE = 6


class FrvOverrideRequestedAction(IntEnum):
    """Requested manual action for one override."""

    MANUAL_INCLUDE_REQUESTED = 1
    MANUAL_EXCLUDE_REQUESTED = 2


class FrvOverrideApprovalState(IntEnum):
    """Approval routing state for one override request."""

    PENDING_REVIEW = 1
    MANUAL_INCLUDE_APPROVED = 2
    MANUAL_EXCLUDE_APPROVED = 3
    MANUAL_OVERRIDE_REJECTED = 4


class FrvOverrideConflictFlag(IntFlag):
    """Override-chain conflict types; a set flag prohibits a manual terminal."""

    NONE = 0
    SAME_PREDECESSOR_CONFLICT = 1
    MULTIPLE_EFFECTIVE_OVERRIDE = 2
    CHAIN_CYCLE = 4
    UNRESOLVABLE_PREDECESSOR = 8
    ARTIFACT_INTEGRITY_FAILURE = 16


class FrvReasonFlag(IntFlag):
    """Closed automatic membership reason flags."""

    NONE = 0
    MANUAL_INCLUDE_PROVENANCE = 1
    MANUAL_EXCLUDE_PROVENANCE = 2
    INSUFFICIENT_EVIDENCE = 4
    GROUND_CONFLICT = 8
    VEGETATION_COUNTER_EVIDENCE = 16
    WIRE_CONFLICT = 32
    SCOPE_EDGE_CONFLICT = 64
    HAG_MISSING = 128
    HAG_INVALID = 256
    HAG_OUT_OF_DOMAIN = 512
    PROVIDER_UNCERTAINTY = 1024
    OVERRIDE_CHAIN_CONFLICT = 2048
    REVIEW_REQUIRED = 4096


class FrvQualityFlag(IntFlag):
    """Closed input/provider/stream/artifact quality flags."""

    NONE = 0
    SOURCE_IDENTITY_FAILURE = 1
    ORIGINAL_INDEX_INVALID = 2
    ORIGINAL_INDEX_DUPLICATE = 4
    STREAM_INCOMPLETE = 8
    PROVIDER_MISSING = 16
    PROVIDER_INVALID = 32
    WIRE_IDENTITY_MISMATCH = 64
    WIRE_GEOMETRY_UNAVAILABLE = 128
    SCOPE_MAPPING_CONFLICT = 256
    SCOPE_GEOMETRY_CONFLICT = 512
    ARTIFACT_DIGEST_MISMATCH = 1024
    MANIFEST_DIGEST_MISMATCH = 2048
    BLOCK_DIGEST_MISMATCH = 4096
    OVERRIDE_ARTIFACT_FAILURE = 8192


class FrvErrorCode(str, Enum):
    """Exact public FRV error-code library."""

    FRV_SOURCE_IDENTITY_FAILED = "FRV_SOURCE_IDENTITY_FAILED"
    FRV_ORIGINAL_INDEX_INVALID = "FRV_ORIGINAL_INDEX_INVALID"
    FRV_ORIGINAL_INDEX_DUPLICATE = "FRV_ORIGINAL_INDEX_DUPLICATE"
    FRV_STREAM_INCOMPLETE = "FRV_STREAM_INCOMPLETE"
    FRV_PROVIDER_MISSING = "FRV_PROVIDER_MISSING"
    FRV_PROVIDER_INVALID = "FRV_PROVIDER_INVALID"
    FRV_WIRE_IDENTITY_MISMATCH = "FRV_WIRE_IDENTITY_MISMATCH"
    FRV_WIRE_GEOMETRY_UNAVAILABLE = "FRV_WIRE_GEOMETRY_UNAVAILABLE"
    FRV_HAG_MISSING = "FRV_HAG_MISSING"
    FRV_HAG_INVALID = "FRV_HAG_INVALID"
    FRV_HAG_OUT_OF_DOMAIN = "FRV_HAG_OUT_OF_DOMAIN"
    FRV_SCOPE_EDGE_CONFLICT = "FRV_SCOPE_EDGE_CONFLICT"
    FRV_SCOPE_MAPPING_INCOMPLETE = "FRV_SCOPE_MAPPING_INCOMPLETE"
    FRV_SCOPE_GEOMETRY_INVALID = "FRV_SCOPE_GEOMETRY_INVALID"
    FRV_ARTIFACT_DIGEST_MISMATCH = "FRV_ARTIFACT_DIGEST_MISMATCH"
    FRV_MANIFEST_DIGEST_MISMATCH = "FRV_MANIFEST_DIGEST_MISMATCH"
    FRV_BLOCK_DIGEST_MISMATCH = "FRV_BLOCK_DIGEST_MISMATCH"
    FRV_OVERRIDE_CHAIN_CONFLICT = "FRV_OVERRIDE_CHAIN_CONFLICT"
    FRV_INSUFFICIENT_EVIDENCE = "FRV_INSUFFICIENT_EVIDENCE"
    FRV_GROUND_CONFLICT = "FRV_GROUND_CONFLICT"
    FRV_VEGETATION_CONFLICT = "FRV_VEGETATION_CONFLICT"
    FRV_WIRE_CONFLICT = "FRV_WIRE_CONFLICT"
    FRV_REVIEW_REQUIRED = "FRV_REVIEW_REQUIRED"


class FrvSafetyCompletenessEffect(str, Enum):
    """Closed safety-completeness routing effect for coverage rows."""

    PRESERVES_SAFETY_COMPLETENESS = "PRESERVES_SAFETY_COMPLETENESS"
    BLOCKS_SAFETY_COMPLETENESS = "BLOCKS_SAFETY_COMPLETENESS"


class FrvConfigurationPurpose(str, Enum):
    """Closed configuration purpose; only formal purposes carry runtime binding.

    ``REVIEWED_FIXED_POLICY`` is the reviewed/frozen pure-algorithm policy: it is
    deterministic, traceable, and a formal review/evaluation authority, but it is
    deliberately *not* statistically calibrated, independently held-out
    validated, production-calibrated, or production-certified.
    """

    TEST_ONLY = "TEST_ONLY"
    CALIBRATION_CANDIDATE = "CALIBRATION_CANDIDATE"
    REVIEWED_FIXED_POLICY = "REVIEWED_FIXED_POLICY"
    PRODUCTION_APPROVED = "PRODUCTION_APPROVED"


# ---------------------------------------------------------------------------
# A1.2 Pure schema/validators (frozen ValueError failure type)
# ---------------------------------------------------------------------------


def validate_exact_schema_fields(
    data: Mapping[str, Any], allowed_fields: set[str], model_name: str
) -> None:
    """Require an exact field schema: no unknown and no missing fields."""

    _strict_mapping(data, model_name)
    require_fields(data, allowed_fields, model_name)
    reject_unknown_fields(data, allowed_fields, model_name)


def validate_frv_enum_value(value: Any, enum_cls: type, name: str) -> None:
    """Reject bool values and require a member of the closed enum."""

    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must not be a boolean")
    if not isinstance(value, enum_cls):
        raise ValueError(f"{name} must be a {enum_cls.__name__}")


def validate_frv_flag_mask(value: Any, flag_cls: type, name: str) -> None:
    """Reject bool and any unknown flag bit; never silently mask."""

    _flag_mask(value, flag_cls, name)


def _nonempty_or_null(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, name)


# ---------------------------------------------------------------------------
# A1.3 FrvConfiguration
# ---------------------------------------------------------------------------


def _check_configuration(config: FrvConfiguration) -> None:
    for name in (
        "configuration_version",
        "algorithm_version",
        "decision_rule_version",
        "scope_construction_version",
        "feature_policy_version",
        "membership_parameter_schema_version",
    ):
        _nonempty(getattr(config, name), name)
    # Contract protocol schema versions are exactly bound (not operational IDs).
    protocol_versions = {
        "schema_version": FRV_CONFIGURATION_SCHEMA_VERSION,
        "canonicalization_version": _FRV_CANONICALIZATION_VERSION,
        "source_identity_schema_version": FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
        "canonical_point_content_spec_version": (
            FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION
        ),
        "scope_schema_version": FRV_SCOPE_DEFINITION_SCHEMA_VERSION,
        "scope_geometry_encoding_version": FRV_SCOPE_GEOMETRY_ENCODING_SCHEMA_VERSION,
        "coverage_manifest_schema_version": FRV_SCOPE_COVERAGE_MANIFEST_SCHEMA_VERSION,
        "artifact_schema_version": FRV_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION,
        "block_schema_version": FRV_ARTIFACT_BLOCK_SCHEMA_VERSION,
        "payload_schema_version": FRV_ARTIFACT_PAYLOAD_SCHEMA_VERSION,
        "override_schema_version": FRV_MANUAL_OVERRIDE_SCHEMA_VERSION,
    }
    for name, expected in protocol_versions.items():
        _nonempty(getattr(config, name), name)
        if getattr(config, name) != expected:
            raise ValueError(f"unsupported {name}")
    _enum_value(config.scope_mode, FrvScopeMode, "scope_mode")
    for name in (
        "scope_policy_digest",
        "override_policy_digest",
        "feature_policy_digest",
        "resource_policy_digest",
    ):
        _sha256(getattr(config, name), name)
    for name in (
        "hag_provider_interface_version",
        "geometry_provider_interface_version",
        "scan_provider_interface_version",
    ):
        _nonempty(getattr(config, name), name)
    for name in (
        "hag_provider_config_digest",
        "geometry_provider_config_digest",
        "scan_provider_config_digest",
    ):
        _sha256(getattr(config, name), name)
    _nonempty_or_null(config.wire_provider_interface_version, "wire_provider_interface_version")
    _sha_or_null(config.wire_provider_config_digest, "wire_provider_config_digest")
    _sha_or_null(config.calibration_record_digest, "calibration_record_digest")
    for name in (
        "allow_wire_body_hard_exclusion",
        "allow_wire_body_confirmed",
        "allow_review_routing_reduction",
        "allow_source_specific_selector",
    ):
        if getattr(config, name) is not False:
            raise ValueError(f"{name} must be exactly false")
    _enum_value(config.configuration_purpose, FrvConfigurationPurpose, "configuration_purpose")
    for name in (
        "configuration_is_runtime_complete",
        "formal_completeness_allowed",
        "membership_artifact_publication_allowed",
    ):
        _required_bool(getattr(config, name), name)

    purpose = config.configuration_purpose
    if purpose is FrvConfigurationPurpose.PRODUCTION_APPROVED:
        if config.configuration_is_runtime_complete is not True:
            raise ValueError("PRODUCTION_APPROVED configuration must be runtime complete")
        if (
            config.wire_provider_interface_version is None
            or config.wire_provider_config_digest is None
        ):
            raise ValueError("PRODUCTION_APPROVED requires a non-null wire provider binding")
        if config.scope_mode is not FrvScopeMode.EXHAUSTIVE_REGION:
            raise ValueError("production-v1 requires EXHAUSTIVE_REGION scope mode")
        if config.calibration_record_digest is None:
            raise ValueError("PRODUCTION_APPROVED requires a non-null calibration record digest")
        _sha256(config.calibration_record_digest, "calibration_record_digest")
    elif purpose is FrvConfigurationPurpose.REVIEWED_FIXED_POLICY:
        if config.configuration_is_runtime_complete is not True:
            raise ValueError("REVIEWED_FIXED_POLICY configuration must be runtime complete")
        if (
            config.wire_provider_interface_version is None
            or config.wire_provider_config_digest is None
        ):
            raise ValueError(
                "REVIEWED_FIXED_POLICY requires a non-null wire provider binding"
            )
        if config.scope_mode is not FrvScopeMode.EXHAUSTIVE_REGION:
            raise ValueError("REVIEWED_FIXED_POLICY requires EXHAUSTIVE_REGION scope mode")
        if config.calibration_record_digest is not None:
            raise ValueError(
                "REVIEWED_FIXED_POLICY requires a null calibration record digest"
            )
    elif purpose is FrvConfigurationPurpose.CALIBRATION_CANDIDATE:
        if config.configuration_is_runtime_complete is not False:
            raise ValueError("CALIBRATION_CANDIDATE configuration cannot be runtime complete")
        if (
            config.formal_completeness_allowed is not False
            or config.membership_artifact_publication_allowed is not False
        ):
            raise ValueError(
                "CALIBRATION_CANDIDATE forbids formal completeness and official publication"
            )
        if config.wire_provider_interface_version is None:
            raise ValueError("CALIBRATION_CANDIDATE requires a wire provider interface version")
    else:  # TEST_ONLY
        if config.configuration_is_runtime_complete is not False:
            raise ValueError("TEST_ONLY configuration cannot be runtime complete")
        if (
            config.formal_completeness_allowed is not False
            or config.membership_artifact_publication_allowed is not False
        ):
            raise ValueError("TEST_ONLY forbids formal completeness and official publication")


@dataclass(frozen=True)
class FrvConfiguration:
    """Exact immutable FRV configuration with purpose-derived closure.

    Contains no paths, source/scene/device/test selectors, provider objects,
    callables, runtime arrays, or file handles.  ``configuration_digest`` is
    derived from every other field and excludes only itself.
    """

    schema_version: str
    configuration_version: str
    algorithm_version: str
    decision_rule_version: str
    canonicalization_version: str
    source_identity_schema_version: str
    canonical_point_content_spec_version: str
    scope_schema_version: str
    scope_construction_version: str
    scope_geometry_encoding_version: str
    coverage_manifest_schema_version: str
    artifact_schema_version: str
    block_schema_version: str
    payload_schema_version: str
    override_schema_version: str
    scope_mode: FrvScopeMode
    scope_policy_digest: str
    override_policy_digest: str
    feature_policy_digest: str
    resource_policy_digest: str
    hag_provider_interface_version: str
    geometry_provider_interface_version: str
    scan_provider_interface_version: str
    hag_provider_config_digest: str
    geometry_provider_config_digest: str
    scan_provider_config_digest: str
    wire_provider_interface_version: str | None
    wire_provider_config_digest: str | None
    feature_policy_version: str
    membership_parameter_schema_version: str
    calibration_record_digest: str | None
    allow_wire_body_hard_exclusion: bool
    allow_wire_body_confirmed: bool
    allow_review_routing_reduction: bool
    allow_source_specific_selector: bool
    configuration_purpose: FrvConfigurationPurpose
    configuration_is_runtime_complete: bool
    formal_completeness_allowed: bool
    membership_artifact_publication_allowed: bool
    configuration_digest: str

    def __post_init__(self) -> None:
        _check_configuration(self)
        computed = self.computed_digest()
        if self.configuration_digest != computed:
            raise ValueError("configuration_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "configuration_digest"
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "configuration_digest": self.configuration_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvConfiguration:
        values = _from_dict_values(cls, data)
        values["scope_mode"] = _enum_from_value(values["scope_mode"], FrvScopeMode, "scope_mode")
        values["configuration_purpose"] = _enum_from_value(
            values["configuration_purpose"], FrvConfigurationPurpose, "configuration_purpose"
        )
        return cls(**values)


def validate_configuration(config: FrvConfiguration) -> None:
    """Validate purpose booleans, SHA/string fields, and wire-provider closure."""

    _check_configuration(config)


def validate_production_wire_provider_binding(config: FrvConfiguration) -> None:
    """Require the complete production wire-provider binding or fail closed.

    A configuration can never create ``WIRE_NOT_APPLICABLE_PROVEN``: that state
    requires authoritative inventory, complete influence-domain geometry, and a
    successful spatial query.  A production configuration with a missing or null
    wire binding is a provider failure, never non-applicability.
    """

    if not isinstance(config, FrvConfiguration):
        raise ValueError("configuration must be a FrvConfiguration")
    if config.configuration_purpose is FrvConfigurationPurpose.PRODUCTION_APPROVED:
        if (
            config.wire_provider_interface_version is None
            or config.wire_provider_config_digest is None
        ):
            raise ValueError("PRODUCTION_APPROVED requires a non-null wire provider binding")


# ---------------------------------------------------------------------------
# A1.3 B1 canonical decoded-point content specification
# ---------------------------------------------------------------------------


def _check_point_field_spec(field: FrvCanonicalPointFieldSpec) -> None:
    _nonempty(field.field_name, "field_name")
    _nonempty(field.semantic_role, "semantic_role")
    if field.dtype_code not in _FRV_POINT_DTYPE_CODES:
        raise ValueError(f"unsupported dtype_code {field.dtype_code!r}")
    if field.byte_order not in _FRV_POINT_BYTE_ORDERS:
        raise ValueError(f"unsupported byte_order {field.byte_order!r}")
    object.__setattr__(field, "scale", _finite_float(field.scale, "scale"))
    object.__setattr__(field, "offset", _finite_float(field.offset, "offset"))
    _required_bool(field.required, "required")
    if field.null_policy not in _FRV_POINT_NULL_POLICIES:
        raise ValueError(f"unsupported null_policy {field.null_policy!r}")
    if field.required and field.null_policy != "FORBID_NULL":
        raise ValueError("FORBID_NULL is mandatory for required fields")
    if field.null_policy == "VALIDITY_BYTE" and field.required:
        raise ValueError("VALIDITY_BYTE is legal only for non-required fields")


@dataclass(frozen=True)
class FrvCanonicalPointFieldSpec:
    """One fixed-width field of the canonical decoded-point content framing."""

    field_name: str
    semantic_role: str
    dtype_code: str
    byte_order: str
    scale: float
    offset: float
    required: bool
    null_policy: str

    def __post_init__(self) -> None:
        _check_point_field_spec(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "semantic_role": self.semantic_role,
            "dtype_code": self.dtype_code,
            "byte_order": self.byte_order,
            "scale": self.scale,
            "offset": self.offset,
            "required": self.required,
            "null_policy": self.null_policy,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvCanonicalPointFieldSpec:
        return cls(**_from_dict_values(cls, data))


def _check_point_content_spec(spec: FrvCanonicalPointContentSpec) -> None:
    if spec.spec_schema_version != FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION:
        raise ValueError("unsupported canonical point-content schema version")
    _nonempty(spec.spec_id, "spec_id")
    if spec.framing_version != _FRV_POINT_FRAMING_VERSION:
        raise ValueError("unsupported canonical point-content framing version")
    if spec.row_order != "ORIGINAL_INDEX_ASCENDING":
        raise ValueError("first version supports only ORIGINAL_INDEX_ASCENDING")
    if spec.quantization_policy not in _FRV_POINT_QUANTIZATION_POLICIES:
        raise ValueError(f"unsupported quantization_policy {spec.quantization_policy!r}")
    if spec.missing_value_policy != "EXPLICIT_VALIDITY_BYTE":
        raise ValueError("first version supports only EXPLICIT_VALIDITY_BYTE")
    _nonempty(spec.coordinate_frame, "coordinate_frame")
    _nonempty(spec.units, "units")
    if not isinstance(spec.ordered_field_specs, tuple) or any(
        not isinstance(field, FrvCanonicalPointFieldSpec) for field in spec.ordered_field_specs
    ):
        raise ValueError("ordered_field_specs must be a tuple of field specs")
    if not spec.ordered_field_specs:
        raise ValueError("ordered_field_specs must not be empty")
    field_names = [field.field_name for field in spec.ordered_field_specs]
    if len(field_names) != len(set(field_names)):
        raise ValueError("ordered_field_specs field names must be unique")
    for field in spec.ordered_field_specs:
        _check_point_field_spec(field)

    first = spec.ordered_field_specs[0]
    if (
        first.field_name != "original_index"
        or first.semantic_role != "ORIGINAL_INDEX"
        or first.dtype_code != "<i8"
        or first.byte_order != "LITTLE_ENDIAN"
        or first.required is not True
        or first.null_policy != "FORBID_NULL"
    ):
        raise ValueError(
            "ORIGINAL_INDEX_ASCENDING requires original_index as the first <i8 required field"
        )
    for field in spec.ordered_field_specs:
        if spec.quantization_policy == "EXACT_TYPED_VALUE":
            if field.scale != 1.0 or field.offset != 0.0:
                raise ValueError("EXACT_TYPED_VALUE permits no additional quantization")
        else:  # SCALED_INTEGER
            if field.dtype_code not in _FRV_INTEGER_DTYPE_CODES:
                raise ValueError("SCALED_INTEGER is legal only for integer dtype_code")
            if field.scale == 0.0:
                raise ValueError("SCALED_INTEGER requires a finite nonzero scale")


@dataclass(frozen=True)
class FrvCanonicalPointContentSpec:
    """B1 canonical decoded-point content framing specification."""

    spec_schema_version: str
    spec_id: str
    framing_version: str
    row_order: str
    ordered_field_specs: tuple[FrvCanonicalPointFieldSpec, ...]
    coordinate_frame: str
    units: str
    quantization_policy: str
    missing_value_policy: str
    spec_digest: str

    def __post_init__(self) -> None:
        _check_point_content_spec(self)
        computed = self.computed_digest()
        if self.spec_digest != computed:
            raise ValueError("spec_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "spec_schema_version": self.spec_schema_version,
            "spec_id": self.spec_id,
            "framing_version": self.framing_version,
            "row_order": self.row_order,
            "ordered_field_specs": [field.to_dict() for field in self.ordered_field_specs],
            "coordinate_frame": self.coordinate_frame,
            "units": self.units,
            "quantization_policy": self.quantization_policy,
            "missing_value_policy": self.missing_value_policy,
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "spec_digest": self.spec_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvCanonicalPointContentSpec:
        values = _from_dict_values(cls, data)
        raw_fields = values["ordered_field_specs"]
        if not isinstance(raw_fields, Sequence) or isinstance(
            raw_fields, (str, bytes, bytearray)
        ):
            raise ValueError("ordered_field_specs must be a non-string sequence")
        values["ordered_field_specs"] = tuple(
            FrvCanonicalPointFieldSpec.from_dict(_strict_mapping(field, "field"))
            for field in raw_fields
        )
        return cls(**values)


def validate_canonical_point_content_spec(spec: FrvCanonicalPointContentSpec) -> None:
    """Validate the B1 field/framing specification (never reads points)."""

    _check_point_content_spec(spec)


# ---------------------------------------------------------------------------
# A1.4 Relocation audit, source-equivalence audit, and SourceIdentity
# ---------------------------------------------------------------------------


def _check_relocation_audit(audit: FrvRelocationAudit) -> None:
    if audit.audit_schema_version != FRV_RELOCATION_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported relocation audit schema version")
    for name in (
        "path_style",
        "recorded_path_raw",
        "resolved_path_raw",
        "recorded_path_lexical_normalized",
        "resolved_path_lexical_normalized",
        "resolution_method",
        "case_comparison_mode",
        "platform_name",
        "run_id",
        "audit_timestamp",
        "content_identity_result",
    ):
        _nonempty(getattr(audit, name), name)
    _nonempty_or_null(audit.resolved_final_path, "resolved_final_path")
    if isinstance(audit.warning_flags, bool | np.bool_):
        raise ValueError("warning_flags must not be a boolean")
    if not isinstance(audit.warning_flags, int | np.integer):
        raise ValueError("warning_flags must be an integer flag mask")
    warning_flags = int(audit.warning_flags)
    if warning_flags < 0 or warning_flags > 0xFFFFFFFF:
        raise ValueError("warning_flags must fit in uint32")
    if warning_flags & ~_FRV_RELOCATION_WARNING_KNOWN_MASK:
        raise ValueError("warning_flags contains unknown flag bit(s)")
    object.__setattr__(audit, "warning_flags", warning_flags)


@dataclass(frozen=True)
class FrvRelocationAudit:
    """Self-contained immutable relocation audit; never resolves paths itself."""

    audit_schema_version: str
    path_style: str
    recorded_path_raw: str
    resolved_path_raw: str
    recorded_path_lexical_normalized: str
    resolved_path_lexical_normalized: str
    resolved_final_path: str | None
    resolution_method: str
    case_comparison_mode: str
    platform_name: str
    run_id: str
    audit_timestamp: str
    content_identity_result: str
    warning_flags: int
    audit_digest: str

    def __post_init__(self) -> None:
        _check_relocation_audit(self)
        computed = self.computed_digest()
        if self.audit_digest != computed:
            raise ValueError("audit_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "audit_digest"
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "audit_digest": self.audit_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvRelocationAudit:
        return cls(**_from_dict_values(cls, data))


def _check_source_equivalence_audit(audit: FrvSourceEquivalenceAudit) -> None:
    if audit.audit_schema_version != FRV_SOURCE_EQUIVALENCE_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported source-equivalence audit schema version")
    for name in ("approval_reference", "approver", "approval_timestamp"):
        _nonempty(getattr(audit, name), name)
    for name in (
        "prior_source_content_identity_digest",
        "current_source_content_identity_digest",
        "prior_source_file_sha256",
        "current_source_file_sha256",
        "canonical_point_content_spec_digest",
        "canonical_point_content_digest",
        "source_summary_comparison_digest",
        "decision_digest",
        "approval_digest",
    ):
        _sha256(getattr(audit, name), name)


@dataclass(frozen=True)
class FrvSourceEquivalenceAudit:
    """Acyclic source-equivalence audit; never references the record digest."""

    audit_schema_version: str
    prior_source_content_identity_digest: str
    current_source_content_identity_digest: str
    prior_source_file_sha256: str
    current_source_file_sha256: str
    canonical_point_content_spec_digest: str
    canonical_point_content_digest: str
    source_summary_comparison_digest: str
    approval_reference: str
    approver: str
    approval_timestamp: str
    decision_digest: str
    approval_digest: str
    audit_digest: str

    def __post_init__(self) -> None:
        _check_source_equivalence_audit(self)
        computed = self.computed_digest()
        if self.audit_digest != computed:
            raise ValueError("audit_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "audit_digest"
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "audit_digest": self.audit_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvSourceEquivalenceAudit:
        return cls(**_from_dict_values(cls, data))


def _check_source_identity(source: SourceIdentity) -> None:
    if source.source_identity_schema_version != FRV_SOURCE_IDENTITY_SCHEMA_VERSION:
        raise ValueError("unsupported source-identity schema version")
    for name in ("source_id", "recorded_source_path", "resolved_access_path"):
        _nonempty(getattr(source, name), name)
    _sha256(source.source_file_sha256, "source_file_sha256")
    _nonempty(source.coordinate_frame, "coordinate_frame")
    _nonempty(source.units, "units")
    object.__setattr__(source, "source_point_count", _non_negative_int(
        source.source_point_count, "source_point_count"
    ))
    object.__setattr__(
        source,
        "source_bounds_min_xyz",
        _finite_vec3(source.source_bounds_min_xyz, "source_bounds_min_xyz"),
    )
    object.__setattr__(
        source,
        "source_bounds_max_xyz",
        _finite_vec3(source.source_bounds_max_xyz, "source_bounds_max_xyz"),
    )
    object.__setattr__(
        source, "origin_context", _finite_vec3(source.origin_context, "origin_context")
    )
    _sha_or_null(source.canonical_point_content_digest, "canonical_point_content_digest")
    _sha_or_null(source.canonical_point_content_spec_digest, "canonical_point_content_spec_digest")
    _nonempty_or_null(
        source.canonical_point_content_spec_version, "canonical_point_content_spec_version"
    )
    content_parts = (
        source.canonical_point_content_digest,
        source.canonical_point_content_spec_digest,
        source.canonical_point_content_spec_version,
    )
    if not all(part is None for part in content_parts) and not all(
        part is not None for part in content_parts
    ):
        raise ValueError(
            "canonical point-content digest, spec digest, and spec version "
            "must be all null or all nonnull"
        )
    if not isinstance(source.relocation_audit, FrvRelocationAudit):
        raise ValueError("relocation_audit must be a FrvRelocationAudit")
    if source.source_equivalence_audit is not None and not isinstance(
        source.source_equivalence_audit, FrvSourceEquivalenceAudit
    ):
        raise ValueError("source_equivalence_audit must be null or a FrvSourceEquivalenceAudit")
    equivalence = source.source_equivalence_audit
    if equivalence is not None:
        if source.canonical_point_content_digest is None:
            raise ValueError(
                "an equivalence audit requires a non-null canonical point-content digest"
            )
        if source.canonical_point_content_spec_digest is None:
            raise ValueError(
                "an equivalence audit requires a non-null canonical point-content spec digest"
            )
        bindings = {
            "current_source_content_identity_digest": (
                equivalence.current_source_content_identity_digest,
                source.source_content_identity_digest,
            ),
            "current_source_file_sha256": (
                equivalence.current_source_file_sha256,
                source.source_file_sha256,
            ),
            "canonical_point_content_digest": (
                equivalence.canonical_point_content_digest,
                source.canonical_point_content_digest,
            ),
            "canonical_point_content_spec_digest": (
                equivalence.canonical_point_content_spec_digest,
                source.canonical_point_content_spec_digest,
            ),
        }
        for name, (audit_value, source_value) in bindings.items():
            if audit_value != source_value:
                raise ValueError(f"equivalence audit {name} does not match the source identity")
    if source.source_content_identity_digest != source.computed_content_digest():
        raise ValueError("source_content_identity_digest does not match canonical content fields")
    if source.source_identity_record_digest != source.computed_record_digest():
        raise ValueError("source_identity_record_digest does not match canonical record fields")


@dataclass(frozen=True)
class SourceIdentity:
    """Immutable source content identity with acyclic content/record digests."""

    source_identity_schema_version: str
    source_id: str
    recorded_source_path: str
    resolved_access_path: str
    source_file_sha256: str
    canonical_point_content_digest: str | None
    canonical_point_content_spec_digest: str | None
    canonical_point_content_spec_version: str | None
    source_point_count: int
    source_bounds_min_xyz: tuple[float, float, float]
    source_bounds_max_xyz: tuple[float, float, float]
    coordinate_frame: str
    units: str
    origin_context: tuple[float, float, float]
    relocation_audit: FrvRelocationAudit
    source_equivalence_audit: FrvSourceEquivalenceAudit | None
    source_content_identity_digest: str
    source_identity_record_digest: str

    def __post_init__(self) -> None:
        _check_source_identity(self)

    def to_content_digest_dict(self) -> dict[str, Any]:
        return {
            "source_identity_schema_version": self.source_identity_schema_version,
            "source_id": self.source_id,
            "source_file_sha256": self.source_file_sha256,
            "canonical_point_content_digest": self.canonical_point_content_digest,
            "canonical_point_content_spec_digest": self.canonical_point_content_spec_digest,
            "canonical_point_content_spec_version": self.canonical_point_content_spec_version,
            "source_point_count": self.source_point_count,
            "source_bounds_min_xyz": list(self.source_bounds_min_xyz),
            "source_bounds_max_xyz": list(self.source_bounds_max_xyz),
            "coordinate_frame": self.coordinate_frame,
            "units": self.units,
            "origin_context": list(self.origin_context),
        }

    def computed_content_digest(self) -> str:
        return frv_digest(self.to_content_digest_dict())

    def to_record_digest_dict(self) -> dict[str, Any]:
        return {
            **self.to_content_digest_dict(),
            "recorded_source_path": self.recorded_source_path,
            "resolved_access_path": self.resolved_access_path,
            "relocation_audit_audit_digest": self.relocation_audit.audit_digest,
            "source_equivalence_audit_audit_digest": (
                self.source_equivalence_audit.audit_digest
                if self.source_equivalence_audit is not None
                else None
            ),
            "source_content_identity_digest": self.source_content_identity_digest,
        }

    def computed_record_digest(self) -> str:
        return frv_digest(self.to_record_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_identity_schema_version": self.source_identity_schema_version,
            "source_id": self.source_id,
            "recorded_source_path": self.recorded_source_path,
            "resolved_access_path": self.resolved_access_path,
            "source_file_sha256": self.source_file_sha256,
            "canonical_point_content_digest": self.canonical_point_content_digest,
            "canonical_point_content_spec_digest": self.canonical_point_content_spec_digest,
            "canonical_point_content_spec_version": self.canonical_point_content_spec_version,
            "source_point_count": self.source_point_count,
            "source_bounds_min_xyz": list(self.source_bounds_min_xyz),
            "source_bounds_max_xyz": list(self.source_bounds_max_xyz),
            "coordinate_frame": self.coordinate_frame,
            "units": self.units,
            "origin_context": list(self.origin_context),
            "relocation_audit": self.relocation_audit.to_dict(),
            "source_equivalence_audit": (
                self.source_equivalence_audit.to_dict()
                if self.source_equivalence_audit is not None
                else None
            ),
            "source_content_identity_digest": self.source_content_identity_digest,
            "source_identity_record_digest": self.source_identity_record_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SourceIdentity:
        values = _from_dict_values(cls, data)
        values["relocation_audit"] = FrvRelocationAudit.from_dict(
            _strict_mapping(values["relocation_audit"], "relocation_audit")
        )
        values["source_equivalence_audit"] = (
            FrvSourceEquivalenceAudit.from_dict(
                _strict_mapping(values["source_equivalence_audit"], "source_equivalence_audit")
            )
            if values["source_equivalence_audit"] is not None
            else None
        )
        return cls(**values)


def validate_source_content_identity_digest(source: SourceIdentity) -> None:
    """Verify the content identity digest against the content payload."""

    if not isinstance(source, SourceIdentity):
        raise ValueError("source must be a SourceIdentity")
    _check_source_identity(source)


def validate_source_identity_record_digest(source: SourceIdentity) -> None:
    """Verify the complete audited record digest against the full record."""

    if not isinstance(source, SourceIdentity):
        raise ValueError("source must be a SourceIdentity")
    _check_source_identity(source)


def validate_source_equivalence_audit_without_cycle(audit: FrvSourceEquivalenceAudit) -> None:
    """Verify the equivalence audit digest and the absence of a record-digest cycle.

    The audit payload contains only its own fields (content digests and approval
    metadata) and never a ``source_identity_record_digest`` reference, so the
    dependency direction is content digest -> equivalence audit -> record digest.
    """

    if not isinstance(audit, FrvSourceEquivalenceAudit):
        raise ValueError("source_equivalence_audit must be a FrvSourceEquivalenceAudit")
    _check_source_equivalence_audit(audit)
    if audit.audit_digest != audit.computed_digest():
        raise ValueError("audit_digest does not match canonical fields")
    if "source_identity_record_digest" in audit.to_digest_dict():
        raise ValueError("source equivalence audit must not reference the source record digest")


# ---------------------------------------------------------------------------
# A1.5 Geometry encoding specification and scope definition
# ---------------------------------------------------------------------------


def _check_scope_geometry_encoding_spec(spec: FrvScopeGeometryEncodingSpec) -> None:
    if spec.encoding_schema_version != FRV_SCOPE_GEOMETRY_ENCODING_SCHEMA_VERSION:
        raise ValueError("unsupported scope geometry encoding schema version")
    for name in ("encoding_spec_id", "coordinate_frame", "units", "boundary_rule_version"):
        _nonempty(getattr(spec, name), name)
    encoding_literals = {
        "box_order_rule": _FRV_GEOMETRY_BOX_ORDER_RULE,
        "float_encoding": _FRV_GEOMETRY_FLOAT_ENCODING,
        "byte_order": _FRV_GEOMETRY_BYTE_ORDER,
        "negative_zero_policy": _FRV_GEOMETRY_NEGATIVE_ZERO_POLICY,
        "box_framing_version": _FRV_GEOMETRY_BOX_FRAMING_VERSION,
    }
    for name, expected in encoding_literals.items():
        if getattr(spec, name) != expected:
            raise ValueError(f"unsupported {name} {getattr(spec, name)!r}")
    _sha256(spec.origin_context_digest, "origin_context_digest")


@dataclass(frozen=True)
class FrvScopeGeometryEncodingSpec:
    """Encoding specification only; it proves rules, never real geometry."""

    encoding_schema_version: str
    encoding_spec_id: str
    coordinate_frame: str
    units: str
    origin_context_digest: str
    boundary_rule_version: str
    box_order_rule: str
    float_encoding: str
    byte_order: str
    negative_zero_policy: str
    box_framing_version: str
    encoding_spec_digest: str

    def __post_init__(self) -> None:
        _check_scope_geometry_encoding_spec(self)
        computed = self.computed_digest()
        if self.encoding_spec_digest != computed:
            raise ValueError("encoding_spec_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "encoding_spec_digest"
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "encoding_spec_digest": self.encoding_spec_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvScopeGeometryEncodingSpec:
        return cls(**_from_dict_values(cls, data))


def _check_scope_definition(scope: FRVScopeDefinition) -> None:
    if scope.scope_schema_version != FRV_SCOPE_DEFINITION_SCHEMA_VERSION:
        raise ValueError("unsupported scope definition schema version")
    for name in ("scope_id", "region_id"):
        _nonempty(getattr(scope, name), name)
    _sha256(scope.source_content_identity_digest, "source_content_identity_digest")
    _sha256(scope.geometry_encoding_spec_digest, "geometry_encoding_spec_digest")
    _sha256(scope.scope_core_geometry_digest, "scope_core_geometry_digest")
    _sha256(
        scope.scope_context_margin_geometry_digest,
        "scope_context_margin_geometry_digest",
    )
    _sha256(scope.feature_halo_policy_digest, "feature_halo_policy_digest")
    _nonempty(scope.coordinate_frame, "coordinate_frame")
    _nonempty(scope.units, "units")
    _sha256(scope.origin_context_digest, "origin_context_digest")
    _nonempty(scope.scope_construction_version, "scope_construction_version")
    _enum_value(scope.scope_mode, FrvScopeMode, "scope_mode")
    object.__setattr__(
        scope,
        "upstream_reference_digests",
        _sorted_unique_sha_tuple(scope.upstream_reference_digests, "upstream_reference_digests"),
    )
    _sha256(scope.boundary_margin_policy_digest, "boundary_margin_policy_digest")
    _nonempty(scope.raw_xyz_inclusion_rule_version, "raw_xyz_inclusion_rule_version")
    _nonempty(scope.half_open_boundary_rule, "half_open_boundary_rule")
    object.__setattr__(scope, "warnings", _string_tuple(scope.warnings, "warnings"))
    object.__setattr__(scope, "limitations", _string_tuple(scope.limitations, "limitations"))


@dataclass(frozen=True)
class FRVScopeDefinition:
    """Immutable scope definition binding content digest references and rules."""

    scope_schema_version: str
    source_content_identity_digest: str
    scope_id: str
    region_id: str
    scope_mode: FrvScopeMode
    geometry_encoding_spec_digest: str
    scope_core_geometry_digest: str
    scope_context_margin_geometry_digest: str
    feature_halo_policy_digest: str
    coordinate_frame: str
    units: str
    origin_context_digest: str
    scope_construction_version: str
    upstream_reference_digests: tuple[str, ...]
    boundary_margin_policy_digest: str
    raw_xyz_inclusion_rule_version: str
    half_open_boundary_rule: str
    scope_definition_digest: str
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_scope_definition(self)
        computed = self.computed_digest()
        if self.scope_definition_digest != computed:
            raise ValueError("scope_definition_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "scope_schema_version": self.scope_schema_version,
            "source_content_identity_digest": self.source_content_identity_digest,
            "scope_id": self.scope_id,
            "region_id": self.region_id,
            "scope_mode": self.scope_mode,
            "geometry_encoding_spec_digest": self.geometry_encoding_spec_digest,
            "scope_core_geometry_digest": self.scope_core_geometry_digest,
            "scope_context_margin_geometry_digest": self.scope_context_margin_geometry_digest,
            "feature_halo_policy_digest": self.feature_halo_policy_digest,
            "coordinate_frame": self.coordinate_frame,
            "units": self.units,
            "origin_context_digest": self.origin_context_digest,
            "scope_construction_version": self.scope_construction_version,
            "upstream_reference_digests": list(self.upstream_reference_digests),
            "boundary_margin_policy_digest": self.boundary_margin_policy_digest,
            "raw_xyz_inclusion_rule_version": self.raw_xyz_inclusion_rule_version,
            "half_open_boundary_rule": self.half_open_boundary_rule,
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "scope_definition_digest": self.scope_definition_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FRVScopeDefinition:
        values = _from_dict_values(cls, data)
        values["scope_mode"] = _enum_from_value(values["scope_mode"], FrvScopeMode, "scope_mode")
        return cls(**values)


def validate_scope_geometry_encoding_spec(spec: FrvScopeGeometryEncodingSpec) -> None:
    """Validate the geometry encoding specification (rules only, no real boxes)."""

    _check_scope_geometry_encoding_spec(spec)


def validate_scope_geometry_content_digest_reference(
    scope: FRVScopeDefinition,
    scope_union_geometry_digest: str | None = None,
) -> None:
    """Keep the encoding-spec digest separate from real geometry content digests.

    The content digests are references to real boxed geometry.  Roles
    SCOPE_CORE, SCOPE_CONTEXT_MARGIN, and SCOPE_UNION are not interchangeable:
    the same digest must never serve two roles or equal the encoding
    specification digest.  A1 never computes real geometry bytes.
    """

    if not isinstance(scope, FRVScopeDefinition):
        raise ValueError("scope must be a FRVScopeDefinition")
    _sha256(scope.geometry_encoding_spec_digest, "geometry_encoding_spec_digest")
    _sha256(scope.scope_core_geometry_digest, "scope_core_geometry_digest")
    _sha256(scope.scope_context_margin_geometry_digest, "scope_context_margin_geometry_digest")
    if scope_union_geometry_digest is not None:
        _sha256(scope_union_geometry_digest, "scope_union_geometry_digest")
    references = {
        "scope core": scope.scope_core_geometry_digest,
        "scope context margin": scope.scope_context_margin_geometry_digest,
    }
    if scope_union_geometry_digest is not None:
        references["scope union"] = scope_union_geometry_digest
    for role, digest in references.items():
        if digest == scope.geometry_encoding_spec_digest:
            raise ValueError(f"geometry encoding spec digest must not equal {role} content digest")
    seen: dict[str, str] = {}
    for role, digest in references.items():
        if digest in seen:
            raise ValueError(f"{seen[digest]} and {role} content digests must differ by role")
        seen[digest] = role


# ---------------------------------------------------------------------------
# A1.6 Approved hard rejection, coverage record, and coverage manifest
# ---------------------------------------------------------------------------

_COVERAGE_BLOCKED_STATES = frozenset(
    {
        FrvScopeCoverageState.SCOPE_BLOCKED,
        FrvScopeCoverageState.SOURCE_IDENTITY_FAILED,
        FrvScopeCoverageState.UPSTREAM_EVIDENCE_INVALID,
        FrvScopeCoverageState.REGION_GEOMETRY_INVALID,
        FrvScopeCoverageState.REGION_MAPPING_INCOMPLETE,
    }
)


def _semantic_artifact_id(value: Any, name: str) -> str:
    _nonempty(value, name)
    prefix = "frv-membership-"
    if not value.startswith(prefix):
        raise ValueError(f"{name} must start with {prefix!r}")
    digest = value[len(prefix):]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must embed a lowercase SHA-256")
    return value


def _check_approved_hard_rejection_evidence(evidence: FRVApprovedHardRejectionEvidence) -> None:
    if evidence.evidence_schema_version != FRV_HARD_REJECTION_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported hard-rejection evidence schema version")
    _sha256(evidence.source_content_identity_digest, "source_content_identity_digest")
    _nonempty(evidence.canonical_inventory_key, "canonical_inventory_key")
    _sha256(evidence.canonical_inventory_record_digest, "canonical_inventory_record_digest")
    for name in (
        "approved_rule_id",
        "approved_rule_version",
        "hard_non_vegetation_evidence_reference",
        "approver",
        "approval_timestamp",
    ):
        _nonempty(getattr(evidence, name), name)
    for name in (
        "hard_non_vegetation_evidence_digest",
        "decision_digest",
        "approval_digest",
    ):
        _sha256(getattr(evidence, name), name)
    _enum_value(
        evidence.safety_completeness_effect,
        FrvSafetyCompletenessEffect,
        "safety_completeness_effect",
    )
    if evidence.approval_digest == evidence.evidence_digest:
        raise ValueError("approval_digest must differ from evidence_digest")


@dataclass(frozen=True)
class FRVApprovedHardRejectionEvidence:
    """Complete approved upstream hard-rejection evidence; A1 never judges truth."""

    evidence_schema_version: str
    source_content_identity_digest: str
    canonical_inventory_key: str
    canonical_inventory_record_digest: str
    approved_rule_id: str
    approved_rule_version: str
    hard_non_vegetation_evidence_reference: str
    hard_non_vegetation_evidence_digest: str
    decision_digest: str
    approver: str
    approval_timestamp: str
    approval_digest: str
    safety_completeness_effect: FrvSafetyCompletenessEffect
    evidence_digest: str

    def __post_init__(self) -> None:
        _check_approved_hard_rejection_evidence(self)
        computed = self.computed_digest()
        if self.evidence_digest != computed:
            raise ValueError("evidence_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "evidence_digest"
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "evidence_digest": self.evidence_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FRVApprovedHardRejectionEvidence:
        values = _from_dict_values(cls, data)
        values["safety_completeness_effect"] = _enum_from_value(
            values["safety_completeness_effect"],
            FrvSafetyCompletenessEffect,
            "safety_completeness_effect",
        )
        return cls(**values)


def _check_scope_coverage_record(record: FRVScopeCoverageRecord) -> None:
    if record.record_schema_version != FRV_SCOPE_COVERAGE_RECORD_SCHEMA_VERSION:
        raise ValueError("unsupported scope coverage record schema version")
    _sha256(record.source_content_identity_digest, "source_content_identity_digest")
    _nonempty(record.canonical_inventory_key, "canonical_inventory_key")
    _sha256(record.canonical_inventory_record_digest, "canonical_inventory_record_digest")
    _sha256(record.canonical_inventory_digest, "canonical_inventory_digest")
    _enum_value(record.inventory_disposition, FrvInventoryDisposition, "inventory_disposition")
    _sha_or_null(record.required_inventory_record_digest, "required_inventory_record_digest")
    _sha_or_null(record.required_inventory_digest, "required_inventory_digest")
    if record.approved_hard_rejection_evidence is not None and not isinstance(
        record.approved_hard_rejection_evidence, FRVApprovedHardRejectionEvidence
    ):
        raise ValueError(
            "approved_hard_rejection_evidence must be null or a FRVApprovedHardRejectionEvidence"
        )
    for name in (
        "upstream_candidate_reference",
        "upstream_candidate_type",
        "upstream_status",
        "upstream_evidence_reference",
    ):
        _nonempty(getattr(record, name), name)
    _nonempty_or_null(record.scope_id, "scope_id")
    _enum_value(
        record.scope_construction_state, FrvScopeConstructionState, "scope_construction_state"
    )
    _enum_value(record.scope_coverage_state, FrvScopeCoverageState, "scope_coverage_state")
    object.__setattr__(
        record,
        "failure_reason_flags",
        _flag_mask(record.failure_reason_flags, FrvReasonFlag, "failure_reason_flags"),
    )
    _required_bool(record.review_required, "review_required")
    _enum_value(
        record.safety_completeness_effect,
        FrvSafetyCompletenessEffect,
        "safety_completeness_effect",
    )
    _nonempty(record.diagnostic_reference, "diagnostic_reference")
    _sha_or_null(
        record.out_of_policy_safety_approval_digest, "out_of_policy_safety_approval_digest"
    )
    if record.out_of_policy_safety_approval_digest is not None:
        raise ValueError("OUT_OF_FRV_POLICY is prohibited from first-version output")
    if record.scope_coverage_state is FrvScopeCoverageState.OUT_OF_FRV_POLICY:
        raise ValueError("OUT_OF_FRV_POLICY is prohibited from first-version output")

    disposition = record.inventory_disposition
    if disposition is FrvInventoryDisposition.REQUIRED_FRV:
        if (
            record.required_inventory_record_digest is None
            or record.required_inventory_digest is None
        ):
            raise ValueError("REQUIRED_FRV rows require both required-inventory digests")
        if record.approved_hard_rejection_evidence is not None:
            raise ValueError("REQUIRED_FRV rows cannot carry hard-rejection evidence")
        if record.scope_construction_state in (
            FrvScopeConstructionState.SCOPE_CREATED,
            FrvScopeConstructionState.SCOPE_MERGED,
        ):
            if record.scope_coverage_state is not FrvScopeCoverageState.COVERED:
                raise ValueError("created or merged required rows must be COVERED")
            if record.scope_id is None:
                raise ValueError("created or merged required rows require a scope_id")
            if (
                record.safety_completeness_effect
                is not FrvSafetyCompletenessEffect.PRESERVES_SAFETY_COMPLETENESS
            ):
                raise ValueError("covered required rows must preserve safety completeness")
            if record.review_required is not False:
                raise ValueError("covered required rows cannot require review")
            if record.failure_reason_flags != 0:
                raise ValueError("covered required rows require NONE failure flags")
        else:
            if record.scope_construction_state is not FrvScopeConstructionState.NOT_CREATED:
                raise ValueError("required rows route to SCOPE_CREATED/SCOPE_MERGED or NOT_CREATED")
            if record.scope_coverage_state not in _COVERAGE_BLOCKED_STATES:
                raise ValueError("NOT_CREATED required rows require a coverage-blocked state")
            if record.failure_reason_flags == 0:
                raise ValueError("blocked required rows require nonzero failure flags")
            if record.scope_id is not None:
                raise ValueError("NOT_CREATED required rows cannot carry a scope_id")
            if (
                record.safety_completeness_effect
                is not FrvSafetyCompletenessEffect.BLOCKS_SAFETY_COMPLETENESS
            ):
                raise ValueError("blocked required rows must block safety completeness")
            if record.review_required is not True:
                raise ValueError("blocked required rows must require review")
    else:  # APPROVED_HARD_REJECTION
        if (
            record.required_inventory_record_digest is not None
            or record.required_inventory_digest is not None
        ):
            raise ValueError("approved hard-rejection rows cannot carry required-inventory digests")
        evidence = record.approved_hard_rejection_evidence
        if evidence is None:
            raise ValueError("approved hard-rejection rows require complete evidence")
        if evidence.source_content_identity_digest != record.source_content_identity_digest:
            raise ValueError("hard-rejection evidence is not bound to this source")
        if evidence.canonical_inventory_key != record.canonical_inventory_key:
            raise ValueError("hard-rejection evidence is not bound to this canonical key")
        if evidence.canonical_inventory_record_digest != record.canonical_inventory_record_digest:
            raise ValueError("hard-rejection evidence is not bound to this canonical record")
        if record.scope_coverage_state is not FrvScopeCoverageState.COVERED:
            raise ValueError("approved hard-rejection rows must be COVERED")
        if record.scope_construction_state is not FrvScopeConstructionState.NOT_CREATED:
            raise ValueError("approved hard-rejection rows must be NOT_CREATED")
        if record.scope_id is not None:
            raise ValueError("approved hard-rejection rows cannot carry a scope_id")
        if record.failure_reason_flags != 0:
            raise ValueError("approved hard-rejection rows require NONE failure flags")
        if record.review_required is not False:
            raise ValueError("approved hard-rejection rows cannot require review")
        if (
            record.safety_completeness_effect
            is not FrvSafetyCompletenessEffect.PRESERVES_SAFETY_COMPLETENESS
        ):
            raise ValueError("approved hard rejection must preserve safety completeness")


@dataclass(frozen=True)
class FRVScopeCoverageRecord:
    """One canonical-inventory coverage row; never inferred from optional fields."""

    record_schema_version: str
    source_content_identity_digest: str
    canonical_inventory_key: str
    canonical_inventory_record_digest: str
    canonical_inventory_digest: str
    inventory_disposition: FrvInventoryDisposition
    required_inventory_record_digest: str | None
    required_inventory_digest: str | None
    approved_hard_rejection_evidence: FRVApprovedHardRejectionEvidence | None
    upstream_candidate_reference: str
    upstream_candidate_type: str
    upstream_status: str
    upstream_evidence_reference: str
    scope_id: str | None
    scope_construction_state: FrvScopeConstructionState
    scope_coverage_state: FrvScopeCoverageState
    failure_reason_flags: FrvReasonFlag
    review_required: bool
    safety_completeness_effect: FrvSafetyCompletenessEffect
    diagnostic_reference: str
    out_of_policy_safety_approval_digest: str | None
    record_digest: str

    def __post_init__(self) -> None:
        _check_scope_coverage_record(self)
        computed = self.computed_digest()
        if self.record_digest != computed:
            raise ValueError("record_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "record_schema_version": self.record_schema_version,
            "source_content_identity_digest": self.source_content_identity_digest,
            "canonical_inventory_key": self.canonical_inventory_key,
            "canonical_inventory_record_digest": self.canonical_inventory_record_digest,
            "canonical_inventory_digest": self.canonical_inventory_digest,
            "inventory_disposition": self.inventory_disposition,
            "required_inventory_record_digest": self.required_inventory_record_digest,
            "required_inventory_digest": self.required_inventory_digest,
            "approved_hard_rejection_evidence": (
                self.approved_hard_rejection_evidence.to_dict()
                if self.approved_hard_rejection_evidence is not None
                else None
            ),
            "upstream_candidate_reference": self.upstream_candidate_reference,
            "upstream_candidate_type": self.upstream_candidate_type,
            "upstream_status": self.upstream_status,
            "upstream_evidence_reference": self.upstream_evidence_reference,
            "scope_id": self.scope_id,
            "scope_construction_state": self.scope_construction_state,
            "scope_coverage_state": self.scope_coverage_state,
            "failure_reason_flags": self.failure_reason_flags,
            "review_required": self.review_required,
            "safety_completeness_effect": self.safety_completeness_effect,
            "diagnostic_reference": self.diagnostic_reference,
            "out_of_policy_safety_approval_digest": self.out_of_policy_safety_approval_digest,
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FRVScopeCoverageRecord:
        values = _from_dict_values(cls, data)
        values["inventory_disposition"] = _enum_from_value(
            values["inventory_disposition"], FrvInventoryDisposition, "inventory_disposition"
        )
        values["approved_hard_rejection_evidence"] = (
            FRVApprovedHardRejectionEvidence.from_dict(
                _strict_mapping(
                    values["approved_hard_rejection_evidence"], "approved_hard_rejection_evidence"
                )
            )
            if values["approved_hard_rejection_evidence"] is not None
            else None
        )
        values["scope_construction_state"] = _enum_from_value(
            values["scope_construction_state"],
            FrvScopeConstructionState,
            "scope_construction_state",
        )
        values["scope_coverage_state"] = _enum_from_value(
            values["scope_coverage_state"], FrvScopeCoverageState, "scope_coverage_state"
        )
        values["failure_reason_flags"] = _flag_from_value(
            values["failure_reason_flags"], FrvReasonFlag, "failure_reason_flags"
        )
        values["safety_completeness_effect"] = _enum_from_value(
            values["safety_completeness_effect"],
            FrvSafetyCompletenessEffect,
            "safety_completeness_effect",
        )
        return cls(**values)


def _check_scope_coverage_manifest(manifest: FRVScopeCoverageManifest) -> None:
    if manifest.manifest_schema_version != FRV_SCOPE_COVERAGE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported scope coverage manifest schema version")
    _sha256(manifest.source_content_identity_digest, "source_content_identity_digest")
    _sha256(manifest.configuration_digest, "configuration_digest")
    _nonempty(manifest.scope_construction_version, "scope_construction_version")
    _sha256(manifest.canonical_inventory_digest, "canonical_inventory_digest")
    _sha_or_null(manifest.required_inventory_digest, "required_inventory_digest")
    _nonempty_or_null(manifest.scope_union_id, "scope_union_id")
    _sha_or_null(manifest.scope_union_geometry_digest, "scope_union_geometry_digest")
    if not isinstance(manifest.records, tuple) or any(
        not isinstance(record, FRVScopeCoverageRecord) for record in manifest.records
    ):
        raise ValueError("records must be a tuple of coverage records")
    keys = tuple(record.canonical_inventory_key for record in manifest.records)
    if keys != tuple(sorted(keys)):
        raise ValueError("records must be sorted by canonical_inventory_key")
    if len(keys) != len(set(keys)):
        raise ValueError("each canonical_inventory_key must appear exactly once")
    for record in manifest.records:
        _check_scope_coverage_record(record)

    derived = _derive_manifest_counts(manifest)
    count_fields = {
        "canonical_inventory_count": "canonical",
        "required_inventory_count": "required",
        "approved_hard_rejection_count": "approved",
        "scope_created_or_merged_count": "created_or_merged",
        "coverage_blocked_count": "blocked",
    }
    for name, derived_key in count_fields.items():
        value = _non_negative_int(getattr(manifest, name), name)
        object.__setattr__(manifest, name, value)
        if value != derived[derived_key]:
            raise ValueError(f"{name} must match the derived coverage rows")

    if derived["required"] != derived["created_or_merged"] + derived["blocked"]:
        raise ValueError("required rows must equal created/merged rows plus blocked rows")
    if derived["canonical"] != derived["required"] + derived["approved"]:
        raise ValueError("canonical rows must equal required rows plus approved rows")

    completeness = {
        "canonical_inventory_complete": True,
        "required_inventory_complete": True,
        "approved_hard_rejection_complete": True,
        "scope_coverage_complete": derived["blocked"] == 0,
        "routing_complete": derived["blocked"] == 0,
    }
    for name, expected in completeness.items():
        if getattr(manifest, name) is not expected:
            raise ValueError(f"{name} must match the coverage rows")

    _nonempty_or_null(manifest.expected_semantic_artifact_id, "expected_semantic_artifact_id")
    if manifest.expected_semantic_artifact_id is not None:
        _semantic_artifact_id(
            manifest.expected_semantic_artifact_id, "expected_semantic_artifact_id"
        )
    for name in (
        "provider_failure_reference_digests",
        "stream_failure_reference_digests",
        "identity_failure_reference_digests",
    ):
        object.__setattr__(
            manifest,
            name,
            _sorted_unique_sha_tuple(getattr(manifest, name), name),
        )
    object.__setattr__(manifest, "warnings", _string_tuple(manifest.warnings, "warnings"))
    object.__setattr__(manifest, "limitations", _string_tuple(manifest.limitations, "limitations"))
    _sha256(manifest.provenance_digest, "provenance_digest")


def _derive_manifest_counts(manifest: FRVScopeCoverageManifest) -> dict[str, int]:
    required = 0
    approved = 0
    created_or_merged = 0
    blocked = 0
    for record in manifest.records:
        if record.inventory_disposition is FrvInventoryDisposition.REQUIRED_FRV:
            required += 1
            if record.scope_construction_state in (
                FrvScopeConstructionState.SCOPE_CREATED,
                FrvScopeConstructionState.SCOPE_MERGED,
            ):
                created_or_merged += 1
            else:
                blocked += 1
        else:
            approved += 1
    return {
        "canonical": len(manifest.records),
        "required": required,
        "approved": approved,
        "created_or_merged": created_or_merged,
        "blocked": blocked,
    }


@dataclass(frozen=True)
class FRVScopeCoverageManifest:
    """Immutable coverage accounting over canonical/required inventory rows."""

    manifest_schema_version: str
    source_content_identity_digest: str
    configuration_digest: str
    scope_construction_version: str
    canonical_inventory_digest: str
    required_inventory_digest: str | None
    scope_union_id: str | None
    scope_union_geometry_digest: str | None
    records: tuple[FRVScopeCoverageRecord, ...]
    canonical_inventory_count: int
    required_inventory_count: int
    approved_hard_rejection_count: int
    scope_created_or_merged_count: int
    coverage_blocked_count: int
    canonical_inventory_complete: bool
    required_inventory_complete: bool
    approved_hard_rejection_complete: bool
    scope_coverage_complete: bool
    routing_complete: bool
    expected_semantic_artifact_id: str | None
    provider_failure_reference_digests: tuple[str, ...]
    stream_failure_reference_digests: tuple[str, ...]
    identity_failure_reference_digests: tuple[str, ...]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    provenance_digest: str
    manifest_digest: str

    def __post_init__(self) -> None:
        _check_scope_coverage_manifest(self)
        computed = self.computed_digest()
        if self.manifest_digest != computed:
            raise ValueError("manifest_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "manifest_schema_version": self.manifest_schema_version,
            "source_content_identity_digest": self.source_content_identity_digest,
            "configuration_digest": self.configuration_digest,
            "scope_construction_version": self.scope_construction_version,
            "canonical_inventory_digest": self.canonical_inventory_digest,
            "required_inventory_digest": self.required_inventory_digest,
            "scope_union_id": self.scope_union_id,
            "scope_union_geometry_digest": self.scope_union_geometry_digest,
            "records": [record.to_dict() for record in self.records],
            "canonical_inventory_count": self.canonical_inventory_count,
            "required_inventory_count": self.required_inventory_count,
            "approved_hard_rejection_count": self.approved_hard_rejection_count,
            "scope_created_or_merged_count": self.scope_created_or_merged_count,
            "coverage_blocked_count": self.coverage_blocked_count,
            "canonical_inventory_complete": self.canonical_inventory_complete,
            "required_inventory_complete": self.required_inventory_complete,
            "approved_hard_rejection_complete": self.approved_hard_rejection_complete,
            "scope_coverage_complete": self.scope_coverage_complete,
            "routing_complete": self.routing_complete,
            "expected_semantic_artifact_id": self.expected_semantic_artifact_id,
            "provider_failure_reference_digests": list(self.provider_failure_reference_digests),
            "stream_failure_reference_digests": list(self.stream_failure_reference_digests),
            "identity_failure_reference_digests": list(self.identity_failure_reference_digests),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "provenance_digest": self.provenance_digest,
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "manifest_digest": self.manifest_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FRVScopeCoverageManifest:
        values = _from_dict_values(cls, data)
        raw_records = values["records"]
        if not isinstance(raw_records, Sequence) or isinstance(
            raw_records, (str, bytes, bytearray)
        ):
            raise ValueError("records must be a non-string sequence")
        values["records"] = tuple(
            FRVScopeCoverageRecord.from_dict(_strict_mapping(record, "record"))
            for record in raw_records
        )
        return cls(**values)


def validate_inventory_disposition(disposition: FrvInventoryDisposition) -> None:
    """Require an exact closed inventory disposition (never inferred)."""

    if isinstance(disposition, bool | np.bool_):
        raise ValueError("inventory_disposition must not be a boolean")
    if not isinstance(disposition, FrvInventoryDisposition):
        raise ValueError("inventory_disposition must be a FrvInventoryDisposition")


def validate_approved_hard_rejection_evidence(
    evidence: FRVApprovedHardRejectionEvidence,
) -> None:
    """Validate the complete approval/canonical binding of hard-rejection evidence."""

    if not isinstance(evidence, FRVApprovedHardRejectionEvidence):
        raise ValueError("evidence must be a FRVApprovedHardRejectionEvidence")
    _check_approved_hard_rejection_evidence(evidence)


def validate_scope_coverage_record(record: FRVScopeCoverageRecord) -> None:
    """Validate exact exclusive disposition and approval/canonical binding."""

    _check_scope_coverage_record(record)


def validate_scope_inventory_conservation(manifest: FRVScopeCoverageManifest) -> None:
    """Validate row order, uniqueness, both inventory equations, and derived counts."""

    _check_scope_coverage_manifest(manifest)


def validate_empty_inventory_scope_artifact_case(manifest: FRVScopeCoverageManifest) -> None:
    """Validate the B5 empty-case truth-table relations for a coverage manifest."""

    if not isinstance(manifest, FRVScopeCoverageManifest):
        raise ValueError("manifest must be a FRVScopeCoverageManifest")
    derived = _derive_manifest_counts(manifest)
    if derived["canonical"] == 0:
        if manifest.records:
            raise ValueError("empty canonical inventory must have no records")
        if manifest.required_inventory_count != 0 or manifest.approved_hard_rejection_count != 0:
            raise ValueError("empty canonical inventory requires zero counts")
        if manifest.required_inventory_digest is not None:
            raise ValueError("empty canonical inventory requires a null required-inventory digest")
        if manifest.scope_union_id is not None or manifest.scope_union_geometry_digest is not None:
            raise ValueError("empty canonical inventory requires a null union")
    elif derived["approved"] == derived["canonical"]:
        if manifest.required_inventory_count != 0:
            raise ValueError("all-approved inventory requires zero required rows")
        if manifest.scope_union_id is not None:
            raise ValueError("all-approved inventory cannot admit a union")
    elif derived["required"] > 0 and derived["created_or_merged"] == 0:
        if manifest.coverage_blocked_count != manifest.required_inventory_count:
            raise ValueError("all-blocked inventory requires blocked equals required")
        if manifest.scope_coverage_complete is not False or manifest.routing_complete is not False:
            raise ValueError("all-blocked inventory is not scope/routing complete")


# ---------------------------------------------------------------------------
# A1.7 Manual override descriptor
# ---------------------------------------------------------------------------


def _sorted_unique_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a non-string sequence")
    items = tuple(_non_negative_int(item, f"{name}[]") for item in value)
    if items != tuple(sorted(items)) or len(items) != len(set(items)):
        raise ValueError(f"{name} must be sorted and unique")
    return items


def _check_manual_override_artifact(override: FRVManualOverrideArtifact) -> None:
    if override.override_schema_version != FRV_MANUAL_OVERRIDE_SCHEMA_VERSION:
        raise ValueError("unsupported manual override schema version")
    _sha256(override.source_content_identity_digest, "source_content_identity_digest")
    _nonempty(override.override_id, "override_id")
    _enum_value(override.requested_action, FrvOverrideRequestedAction, "requested_action")
    object.__setattr__(
        override,
        "requested_original_indexes",
        _sorted_unique_int_tuple(override.requested_original_indexes, "requested_original_indexes"),
    )
    _nonempty_or_null(override.requested_geometry_reference, "requested_geometry_reference")
    _nonempty_or_null(
        override.requested_geometry_resolution_version, "requested_geometry_resolution_version"
    )
    for name in ("requester", "request_timestamp", "request_reason", "authority_level"):
        _nonempty(getattr(override, name), name)
    _sha_or_null(override.predecessor_artifact_digest, "predecessor_artifact_digest")
    _enum_value(
        override.automatic_state_before_override,
        FrvMembershipState,
        "automatic_state_before_override",
    )
    object.__setattr__(
        override,
        "automatic_reason_flags",
        _flag_mask(override.automatic_reason_flags, FrvReasonFlag, "automatic_reason_flags"),
    )
    object.__setattr__(
        override,
        "conflict_flags",
        _flag_mask(override.conflict_flags, FrvOverrideConflictFlag, "conflict_flags"),
    )
    _nonempty_or_null(override.reviewer, "reviewer")
    _nonempty_or_null(override.review_timestamp, "review_timestamp")
    _enum_value(override.approval_state, FrvOverrideApprovalState, "approval_state")
    _nonempty_or_null(override.approval_reason, "approval_reason")
    if override.resulting_state is not None and not isinstance(
        override.resulting_state, FrvMembershipState
    ):
        raise ValueError("resulting_state must be null or a FrvMembershipState")
    _sha_or_null(override.superseded_override_digest, "superseded_override_digest")

    has_indexes = bool(override.requested_original_indexes)
    has_geometry = (
        override.requested_geometry_reference is not None
        or override.requested_geometry_resolution_version is not None
    )
    if has_indexes and has_geometry:
        raise ValueError("an override requests either indexes or geometry, not both")
    if not has_indexes and not has_geometry:
        raise ValueError("an override must request indexes or geometry")
    if has_geometry:
        if override.requested_geometry_reference is None:
            raise ValueError("a geometry request requires a geometry reference")
        if override.requested_geometry_resolution_version is None:
            raise ValueError("a geometry request requires a resolution version")

    requested = override.requested_action
    approval_state = override.approval_state
    if approval_state is FrvOverrideApprovalState.MANUAL_INCLUDE_APPROVED:
        if requested is not FrvOverrideRequestedAction.MANUAL_INCLUDE_REQUESTED:
            raise ValueError("MANUAL_INCLUDE_APPROVED requires an include request")
        if override.resulting_state is not FrvMembershipState.MANUAL_INCLUDE:
            raise ValueError("MANUAL_INCLUDE_APPROVED requires MANUAL_INCLUDE result")
    elif approval_state is FrvOverrideApprovalState.MANUAL_EXCLUDE_APPROVED:
        if requested is not FrvOverrideRequestedAction.MANUAL_EXCLUDE_REQUESTED:
            raise ValueError("MANUAL_EXCLUDE_APPROVED requires an exclude request")
        if override.resulting_state is not FrvMembershipState.MANUAL_EXCLUDE:
            raise ValueError("MANUAL_EXCLUDE_APPROVED requires MANUAL_EXCLUDE result")
    elif approval_state is FrvOverrideApprovalState.PENDING_REVIEW:
        if override.resulting_state is not None:
            raise ValueError("PENDING_REVIEW requires no resulting state")
        if (
            override.reviewer is not None
            or override.review_timestamp is not None
            or override.approval_reason is not None
        ):
            raise ValueError("PENDING_REVIEW requires null reviewer fields")
    else:  # MANUAL_OVERRIDE_REJECTED
        if override.resulting_state is not None:
            raise ValueError("MANUAL_OVERRIDE_REJECTED requires no resulting state")
        if (
            override.reviewer is None
            or override.review_timestamp is None
            or override.approval_reason is None
        ):
            raise ValueError("MANUAL_OVERRIDE_REJECTED requires reviewer fields")
    if override.resulting_state is FrvMembershipState.MANUAL_INCLUDE:
        if requested is not FrvOverrideRequestedAction.MANUAL_INCLUDE_REQUESTED:
            raise ValueError("MANUAL_INCLUDE result requires an include request")
    if override.resulting_state is FrvMembershipState.MANUAL_EXCLUDE:
        if requested is not FrvOverrideRequestedAction.MANUAL_EXCLUDE_REQUESTED:
            raise ValueError("MANUAL_EXCLUDE result requires an exclude request")

    if approval_state in (
        FrvOverrideApprovalState.MANUAL_INCLUDE_APPROVED,
        FrvOverrideApprovalState.MANUAL_EXCLUDE_APPROVED,
    ):
        if (
            override.reviewer is None
            or override.review_timestamp is None
            or override.approval_reason is None
        ):
            raise ValueError("approved overrides require reviewer, timestamp, and reason")

    if override.conflict_flags != 0:
        if override.resulting_state is not None:
            raise ValueError("conflicted overrides cannot have a terminal result")


@dataclass(frozen=True)
class FRVManualOverrideArtifact:
    """Immutable override request/approval/chain descriptor only."""

    override_schema_version: str
    source_content_identity_digest: str
    override_id: str
    requested_action: FrvOverrideRequestedAction
    requested_original_indexes: tuple[int, ...]
    requested_geometry_reference: str | None
    requested_geometry_resolution_version: str | None
    requester: str
    request_timestamp: str
    request_reason: str
    authority_level: str
    predecessor_artifact_digest: str | None
    automatic_state_before_override: FrvMembershipState
    automatic_reason_flags: FrvReasonFlag
    conflict_flags: FrvOverrideConflictFlag
    reviewer: str | None
    review_timestamp: str | None
    approval_state: FrvOverrideApprovalState
    approval_reason: str | None
    resulting_state: FrvMembershipState | None
    superseded_override_digest: str | None
    override_digest: str

    def __post_init__(self) -> None:
        _check_manual_override_artifact(self)
        computed = self.computed_digest()
        if self.override_digest != computed:
            raise ValueError("override_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "override_schema_version": self.override_schema_version,
            "source_content_identity_digest": self.source_content_identity_digest,
            "override_id": self.override_id,
            "requested_action": self.requested_action,
            "requested_original_indexes": list(self.requested_original_indexes),
            "requested_geometry_reference": self.requested_geometry_reference,
            "requested_geometry_resolution_version": self.requested_geometry_resolution_version,
            "requester": self.requester,
            "request_timestamp": self.request_timestamp,
            "request_reason": self.request_reason,
            "authority_level": self.authority_level,
            "predecessor_artifact_digest": self.predecessor_artifact_digest,
            "automatic_state_before_override": self.automatic_state_before_override,
            "automatic_reason_flags": self.automatic_reason_flags,
            "conflict_flags": self.conflict_flags,
            "reviewer": self.reviewer,
            "review_timestamp": self.review_timestamp,
            "approval_state": self.approval_state,
            "approval_reason": self.approval_reason,
            "resulting_state": self.resulting_state,
            "superseded_override_digest": self.superseded_override_digest,
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "override_digest": self.override_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FRVManualOverrideArtifact:
        values = _from_dict_values(cls, data)
        values["requested_action"] = _enum_from_value(
            values["requested_action"], FrvOverrideRequestedAction, "requested_action"
        )
        values["automatic_state_before_override"] = _enum_from_value(
            values["automatic_state_before_override"],
            FrvMembershipState,
            "automatic_state_before_override",
        )
        values["automatic_reason_flags"] = _flag_from_value(
            values["automatic_reason_flags"], FrvReasonFlag, "automatic_reason_flags"
        )
        values["conflict_flags"] = _flag_from_value(
            values["conflict_flags"], FrvOverrideConflictFlag, "conflict_flags"
        )
        values["approval_state"] = _enum_from_value(
            values["approval_state"], FrvOverrideApprovalState, "approval_state"
        )
        values["resulting_state"] = _enum_from_value_or_none(
            values["resulting_state"], FrvMembershipState, "resulting_state"
        )
        return cls(**values)


def validate_manual_override_artifact(override: FRVManualOverrideArtifact) -> None:
    """Validate the supplied override descriptor structure and approval separation."""

    if not isinstance(override, FRVManualOverrideArtifact):
        raise ValueError("override must be a FRVManualOverrideArtifact")
    _check_manual_override_artifact(override)


def validate_override_chain_descriptors(
    overrides: Sequence[FRVManualOverrideArtifact],
) -> None:
    """Validate supplied chain-descriptor topology and approval separation only.

    Predecessor resolution, acyclicity, and at-most-one-effective override are
    checked; execution and effectiveness routing remain A5.
    """

    if not isinstance(overrides, Sequence) or isinstance(overrides, (str, bytes, bytearray)):
        raise ValueError("overrides must be a non-string sequence")
    items = tuple(overrides)
    if any(not isinstance(override, FRVManualOverrideArtifact) for override in items):
        raise ValueError("overrides must contain FRVManualOverrideArtifact values")
    for override in items:
        _check_manual_override_artifact(override)

    sources = {override.source_content_identity_digest for override in items}
    if len(sources) != 1:
        raise ValueError("all chain overrides must share one source content identity digest")
    override_ids = [override.override_id for override in items]
    if len(override_ids) != len(set(override_ids)):
        raise ValueError("override_id values must be unique")
    override_digests = [override.override_digest for override in items]
    if len(override_digests) != len(set(override_digests)):
        raise ValueError("override_digest values must be unique")

    by_digest: dict[str, FRVManualOverrideArtifact] = {}
    for override in items:
        if override.override_digest in by_digest:
            raise ValueError("duplicate override_digest must not be silently overwritten")
        by_digest[override.override_digest] = override
    digests = set(by_digest)

    for override in items:
        predecessor = override.predecessor_artifact_digest
        if predecessor is None:
            continue
        resolvable = predecessor in digests and predecessor != override.override_digest
        if not resolvable:
            if not (override.conflict_flags & FrvOverrideConflictFlag.UNRESOLVABLE_PREDECESSOR):
                raise ValueError(
                    "an unresolvable predecessor requires UNRESOLVABLE_PREDECESSOR"
                )
            if override.resulting_state is not None:
                raise ValueError("a conflicted override cannot have a terminal result")

    cycle_members = _chain_cycle_members(items)
    for override in items:
        if override.override_digest in cycle_members:
            if not (override.conflict_flags & FrvOverrideConflictFlag.CHAIN_CYCLE):
                raise ValueError("an override chain cycle requires CHAIN_CYCLE")
            if override.resulting_state is not None:
                raise ValueError("a cycle-member override cannot have a terminal result")

    terminal_by_predecessor: dict[str | None, list[FRVManualOverrideArtifact]] = {}
    for override in items:
        if override.resulting_state is not None:
            terminal_by_predecessor.setdefault(override.predecessor_artifact_digest, []).append(
                override
            )
    for _predecessors, group in terminal_by_predecessor.items():
        if len(group) > 1:
            for override in group:
                has_conflict = override.conflict_flags & (
                    FrvOverrideConflictFlag.SAME_PREDECESSOR_CONFLICT
                    | FrvOverrideConflictFlag.MULTIPLE_EFFECTIVE_OVERRIDE
                )
                if not has_conflict:
                    raise ValueError(
                        "multiple effective overrides for one predecessor require a conflict flag"
                    )
                if override.resulting_state is not None:
                    raise ValueError("a conflicted override cannot have a terminal result")


def _chain_cycle_members(
    items: Sequence[FRVManualOverrideArtifact],
) -> set[str]:
    by_digest = {override.override_digest: override for override in items}
    color: dict[str, int] = {}
    for override in items:
        color[override.override_digest] = 0
    in_cycle: set[str] = set()

    def visit(node_digest: str, stack: list[str]) -> None:
        color[node_digest] = 1
        stack.append(node_digest)
        override = by_digest[node_digest]
        predecessor = override.predecessor_artifact_digest
        if predecessor is not None and predecessor in by_digest and predecessor != node_digest:
            if color[predecessor] == 1:
                start = stack.index(predecessor)
                for member in stack[start:]:
                    in_cycle.add(member)
            elif color[predecessor] == 0:
                visit(predecessor, stack)
        stack.pop()
        color[node_digest] = 2

    for override in items:
        if color[override.override_digest] == 0:
            visit(override.override_digest, [])
    return in_cycle


# ---------------------------------------------------------------------------
# A1.8 Payload array schema and block descriptors
# ---------------------------------------------------------------------------

def _table_tuple(value: Any, name: str) -> tuple[str, ...]:
    items = _string_tuple(value, name)
    if not items:
        raise ValueError(f"{name} must not be empty")
    if any(not item or item != item.strip() for item in items):
        raise ValueError(
            f"{name} entries must be non-empty and without surrounding whitespace"
        )
    if items != tuple(sorted(items)) or len(items) != len(set(items)):
        raise ValueError(f"{name} must be lexicographically sorted and unique")
    return items


def _check_array_descriptor(array: FRVArtifactArrayDescriptor) -> None:
    _nonempty(array.field_name, "field_name")
    if array.dtype_code not in _FRV_PAYLOAD_DTYPE_CODES:
        raise ValueError(f"unsupported array dtype_code {array.dtype_code!r}")
    _nonempty(array.shape_rule, "shape_rule")
    _required_bool(array.required, "required")
    _required_bool(array.nullable, "nullable")
    _required_bool(array.finite_required, "finite_required")
    _nonempty_or_null(array.enum_or_flag_schema_version, "enum_or_flag_schema_version")
    _nonempty(array.storage_key, "storage_key")


@dataclass(frozen=True)
class FRVArtifactArrayDescriptor:
    """Metadata descriptor for one point-aligned payload array.

    Deliberately has no ``array_digest``: array content integrity is owned by
    each block's ``block_digest`` and the artifact's ``artifact_sha256``.
    """

    field_name: str
    dtype_code: str
    shape_rule: str
    required: bool
    nullable: bool
    finite_required: bool
    enum_or_flag_schema_version: str | None
    storage_key: str

    def __post_init__(self) -> None:
        _check_array_descriptor(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "dtype_code": self.dtype_code,
            "shape_rule": self.shape_rule,
            "required": self.required,
            "nullable": self.nullable,
            "finite_required": self.finite_required,
            "enum_or_flag_schema_version": self.enum_or_flag_schema_version,
            "storage_key": self.storage_key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FRVArtifactArrayDescriptor:
        return cls(**_from_dict_values(cls, data))


def _check_payload_schema(schema: FRVArtifactPayloadSchema) -> None:
    if schema.payload_schema_version != FRV_ARTIFACT_PAYLOAD_SCHEMA_VERSION:
        raise ValueError("unsupported payload schema version")
    if schema.block_schema_version != FRV_ARTIFACT_BLOCK_SCHEMA_VERSION:
        raise ValueError("unsupported block schema version")
    enum_flag_versions = {
        "membership_state_schema_version": _FRV_MEMBERSHIP_STATE_SCHEMA_VERSION,
        "reason_flag_schema_version": _FRV_REASON_FLAG_SCHEMA_VERSION,
        "quality_flag_schema_version": _FRV_QUALITY_FLAG_SCHEMA_VERSION,
        "hag_evidence_schema_version": _FRV_HAG_EVIDENCE_SCHEMA_VERSION,
        "segmentation_hag_readiness_schema_version": (
            _FRV_SEGMENTATION_HAG_READINESS_SCHEMA_VERSION
        ),
    }
    for name, expected in enum_flag_versions.items():
        _nonempty(getattr(schema, name), name)
        if getattr(schema, name) != expected:
            raise ValueError(f"unsupported {name} {getattr(schema, name)!r}")
    if schema.payload_encoding_id != FRV_PAYLOAD_ENCODING_ID:
        raise ValueError("first version payload_encoding_id must be FRV-BLOCK-PAYLOAD-1")
    if schema.payload_encoder_version != FRV_PAYLOAD_ENCODER_VERSION:
        raise ValueError("first version payload_encoder_version must be 1")
    if schema.payload_compression_algorithm != FRV_PAYLOAD_COMPRESSION_ALGORITHM:
        raise ValueError("first version payload_compression_algorithm must be NONE")
    if schema.payload_compression_version != FRV_PAYLOAD_COMPRESSION_VERSION:
        raise ValueError("first version payload_compression_version must be 1")
    if not isinstance(schema.arrays, tuple) or any(
        not isinstance(array, FRVArtifactArrayDescriptor) for array in schema.arrays
    ):
        raise ValueError("arrays must be a tuple of array descriptors")
    if not schema.arrays:
        raise ValueError("arrays must not be empty")
    for array in schema.arrays:
        _check_array_descriptor(array)

    array_names = [array.field_name for array in schema.arrays]
    if len(array_names) != len(set(array_names)):
        raise ValueError("array field names must be unique")
    by_name = {array.field_name: array for array in schema.arrays}
    for (
        field_name,
        expected_dtype,
        expected_shape,
        expected_enum_version,
    ) in REQUIRED_FRV_ARRAY_SPECS:
        array = by_name.get(field_name)
        if array is None:
            raise ValueError(f"payload schema is missing required array {field_name}")
        if array.dtype_code != expected_dtype:
            raise ValueError(f"{field_name} requires dtype {expected_dtype!r}")
        if array.shape_rule != expected_shape:
            raise ValueError(f"{field_name} requires shape_rule {expected_shape!r}")
        if array.required is not True:
            raise ValueError(f"{field_name} must be required")
        if array.nullable is not False:
            raise ValueError(f"{field_name} must not be nullable")
        _nonempty(array.storage_key, f"{field_name}.storage_key")
        if array.enum_or_flag_schema_version != expected_enum_version:
            raise ValueError(f"{field_name} enum_or_flag_schema_version is unsupported")
    for array in schema.arrays:
        if array.field_name not in _REQUIRED_ARRAY_NAMES and (
            array.field_name not in ALLOWED_OPTIONAL_FRV_ARRAYS
        ):
            raise ValueError(f"unknown or old array field {array.field_name!r}")

    table_specs = (
        ("scope_union_reference_table", "scope_union_reference_table_digest", False),
        ("primary_scope_reference_table", "primary_scope_reference_table_digest", False),
        ("supporting_region_reference_table", "supporting_region_reference_table_digest", False),
        ("manual_override_reference_table", "manual_override_reference_table_digest", True),
    )
    for table_field, digest_field, nullable in table_specs:
        table = getattr(schema, table_field)
        digest = getattr(schema, digest_field)
        if table is None:
            if not nullable:
                raise ValueError(f"{table_field} must not be null")
            if digest is not None:
                raise ValueError(f"{digest_field} must be null when the table is null")
            continue
        items = _table_tuple(table, table_field)
        object.__setattr__(schema, table_field, items)
        if digest is None:
            raise ValueError(f"{digest_field} must be present when the table is present")
        _sha256(digest, digest_field)
        if digest != frv_digest(items):
            raise ValueError(f"{digest_field} must match {table_field}")


@dataclass(frozen=True)
class FRVArtifactPayloadSchema:
    """Exact first-version payload schema and reference-table inventory."""

    payload_schema_version: str
    block_schema_version: str
    membership_state_schema_version: str
    reason_flag_schema_version: str
    quality_flag_schema_version: str
    hag_evidence_schema_version: str
    segmentation_hag_readiness_schema_version: str
    payload_encoding_id: str
    payload_encoder_version: str
    payload_compression_algorithm: str
    payload_compression_version: str
    arrays: tuple[FRVArtifactArrayDescriptor, ...]
    scope_union_reference_table: tuple[str, ...]
    scope_union_reference_table_digest: str
    primary_scope_reference_table: tuple[str, ...]
    primary_scope_reference_table_digest: str
    supporting_region_reference_table: tuple[str, ...]
    supporting_region_reference_table_digest: str
    manual_override_reference_table: tuple[str, ...] | None
    manual_override_reference_table_digest: str | None
    payload_schema_digest: str

    def __post_init__(self) -> None:
        _check_payload_schema(self)
        computed = self.computed_digest()
        if self.payload_schema_digest != computed:
            raise ValueError("payload_schema_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "payload_schema_version": self.payload_schema_version,
            "block_schema_version": self.block_schema_version,
            "membership_state_schema_version": self.membership_state_schema_version,
            "reason_flag_schema_version": self.reason_flag_schema_version,
            "quality_flag_schema_version": self.quality_flag_schema_version,
            "hag_evidence_schema_version": self.hag_evidence_schema_version,
            "segmentation_hag_readiness_schema_version": (
                self.segmentation_hag_readiness_schema_version
            ),
            "payload_encoding_id": self.payload_encoding_id,
            "payload_encoder_version": self.payload_encoder_version,
            "payload_compression_algorithm": self.payload_compression_algorithm,
            "payload_compression_version": self.payload_compression_version,
            "arrays": [array.to_dict() for array in self.arrays],
            "scope_union_reference_table": list(self.scope_union_reference_table),
            "scope_union_reference_table_digest": self.scope_union_reference_table_digest,
            "primary_scope_reference_table": list(self.primary_scope_reference_table),
            "primary_scope_reference_table_digest": self.primary_scope_reference_table_digest,
            "supporting_region_reference_table": list(self.supporting_region_reference_table),
            "supporting_region_reference_table_digest": (
                self.supporting_region_reference_table_digest
            ),
            "manual_override_reference_table": (
                list(self.manual_override_reference_table)
                if self.manual_override_reference_table is not None
                else None
            ),
            "manual_override_reference_table_digest": self.manual_override_reference_table_digest,
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "payload_schema_digest": self.payload_schema_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FRVArtifactPayloadSchema:
        values = _from_dict_values(cls, data)
        raw_arrays = values["arrays"]
        if not isinstance(raw_arrays, Sequence) or isinstance(
            raw_arrays, (str, bytes, bytearray)
        ):
            raise ValueError("arrays must be a non-string sequence")
        values["arrays"] = tuple(
            FRVArtifactArrayDescriptor.from_dict(_strict_mapping(array, "array"))
            for array in raw_arrays
        )
        return cls(**values)


def _non_negative_int_or_null(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, name)


def _check_block_descriptor(block: FRVArtifactBlockDescriptor) -> None:
    if block.block_schema_version != FRV_ARTIFACT_BLOCK_SCHEMA_VERSION:
        raise ValueError("unsupported block schema version")
    if block.payload_schema_version != FRV_ARTIFACT_PAYLOAD_SCHEMA_VERSION:
        raise ValueError("unsupported payload schema version reference")
    _sha256(block.payload_schema_digest, "payload_schema_digest")
    _nonempty(block.block_id, "block_id")
    _sha256(block.block_digest, "block_digest")
    object.__setattr__(
        block,
        "original_index_start",
        _non_negative_int(block.original_index_start, "original_index_start"),
    )
    object.__setattr__(
        block,
        "original_index_stop",
        _non_negative_int(block.original_index_stop, "original_index_stop"),
    )
    object.__setattr__(block, "record_count", _non_negative_int(block.record_count, "record_count"))
    object.__setattr__(
        block,
        "first_original_index",
        _non_negative_int_or_null(block.first_original_index, "first_original_index"),
    )
    object.__setattr__(
        block,
        "last_original_index",
        _non_negative_int_or_null(block.last_original_index, "last_original_index"),
    )
    if block.record_count == 0:
        if block.first_original_index is not None or block.last_original_index is not None:
            raise ValueError("empty blocks require null first/last original indexes")
        if block.original_index_stop != block.original_index_start:
            raise ValueError("empty blocks require an empty [start, start) range")
    else:
        if block.first_original_index is None or block.last_original_index is None:
            raise ValueError("nonempty blocks require first/last original indexes")
        if block.original_index_stop <= block.original_index_start:
            raise ValueError("nonempty blocks require start < stop")
        if block.original_index_start > block.first_original_index:
            raise ValueError("first_original_index must be at least original_index_start")
        if block.first_original_index > block.last_original_index:
            raise ValueError("first_original_index must not exceed last_original_index")
        if block.last_original_index >= block.original_index_stop:
            raise ValueError("last_original_index must be below original_index_stop")
    _nonempty(block.storage_reference, "storage_reference")
    _sha256(block.array_descriptor_digest, "array_descriptor_digest")


@dataclass(frozen=True)
class FRVArtifactBlockDescriptor:
    """Metadata-only block descriptor; A1 never opens a block."""

    block_schema_version: str
    payload_schema_digest: str
    block_id: str
    block_digest: str
    original_index_start: int
    original_index_stop: int
    record_count: int
    first_original_index: int | None
    last_original_index: int | None
    storage_reference: str
    payload_schema_version: str
    array_descriptor_digest: str
    block_descriptor_digest: str

    def __post_init__(self) -> None:
        _check_block_descriptor(self)
        computed = self.computed_digest()
        if self.block_descriptor_digest != computed:
            raise ValueError("block_descriptor_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "block_descriptor_digest"
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "block_descriptor_digest": self.block_descriptor_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FRVArtifactBlockDescriptor:
        return cls(**_from_dict_values(cls, data))


def _check_block_descriptor_inventory(descriptors: Sequence[FRVArtifactBlockDescriptor]) -> None:
    if not isinstance(descriptors, Sequence) or isinstance(descriptors, (str, bytes, bytearray)):
        raise ValueError("block descriptors must be a non-string sequence")
    if not descriptors:
        raise ValueError("block descriptors must not be empty")
    if any(not isinstance(block, FRVArtifactBlockDescriptor) for block in descriptors):
        raise ValueError("block descriptors must contain FRVArtifactBlockDescriptor values")
    block_ids = [block.block_id for block in descriptors]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("block_id values must be unique")
    keys = [
        (block.original_index_start, block.original_index_stop, block.block_id)
        for block in descriptors
    ]
    if keys != sorted(keys):
        raise ValueError("blocks must be ordered by start, then stop, then block_id")
    for previous, current in zip(descriptors, descriptors[1:], strict=False):
        if current.original_index_start < previous.original_index_stop:
            raise ValueError("block ranges must not overlap")


def validate_supporting_region_ragged_reference_schema(
    schema: FRVArtifactPayloadSchema,
) -> None:
    """Validate offset/count dtypes and the sorted-unique supporting-region table."""

    if not isinstance(schema, FRVArtifactPayloadSchema):
        raise ValueError("schema must be a FRVArtifactPayloadSchema")
    by_name = {array.field_name: array for array in schema.arrays}
    offset = by_name.get("supporting_region_offset")
    count = by_name.get("supporting_region_count")
    if offset is None or offset.dtype_code != "<u8":
        raise ValueError("supporting_region_offset requires dtype <u8[N]")
    if count is None or count.dtype_code != "<u4":
        raise ValueError("supporting_region_count requires dtype <u4[N]")
    _table_tuple(schema.supporting_region_reference_table, "supporting_region_reference_table")
    if schema.supporting_region_reference_table_digest != frv_digest(
        schema.supporting_region_reference_table
    ):
        raise ValueError("supporting_region_reference_table_digest must match the table")


def validate_payload_schema(schema: FRVArtifactPayloadSchema) -> None:
    """Validate exact arrays, dtypes, sorted-unique tables, and no object dtype."""

    _check_payload_schema(schema)


def validate_block_descriptor_inventory(
    descriptors: Sequence[FRVArtifactBlockDescriptor],
) -> None:
    """Validate descriptor IDs, ranges, order, and non-overlap."""

    _check_block_descriptor_inventory(descriptors)


# ---------------------------------------------------------------------------
# A1.9 Membership artifact and binding record
# ---------------------------------------------------------------------------


def _artifact_id_payload(artifact: FullResolutionVegetationMembershipArtifact) -> dict[str, Any]:
    return {
        "artifact_schema_version": artifact.artifact_schema_version,
        "algorithm_version": artifact.algorithm_version,
        "decision_rule_version": artifact.decision_rule_version,
        "source_content_identity_digest": artifact.source_content_identity_digest,
        "scope_union_id": artifact.scope_union_id,
        "configuration_digest": artifact.configuration_digest,
        "configuration_purpose": artifact.configuration_purpose,
    }


def _check_artifact(artifact: FullResolutionVegetationMembershipArtifact) -> None:
    if artifact.artifact_schema_version != FRV_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported membership artifact schema version")
    for name in ("algorithm_version", "decision_rule_version"):
        _nonempty(getattr(artifact, name), name)
    expected_id = "frv-membership-" + frv_digest(_artifact_id_payload(artifact))
    if artifact.artifact_id != expected_id:
        raise ValueError("artifact_id does not match canonical semantic inputs")
    # The configuration purpose is mandatory artifact authority: no artifact may
    # be silently assigned a trusted purpose, and a missing/None purpose fails
    # closed here rather than defaulting to any authority.
    validate_frv_enum_value(
        artifact.configuration_purpose,
        FrvConfigurationPurpose,
        "configuration_purpose",
    )
    _sha256(artifact.source_content_identity_digest, "source_content_identity_digest")
    _sha256(artifact.source_identity_record_digest, "source_identity_record_digest")
    _sha256(artifact.source_file_sha256, "source_file_sha256")
    _sha_or_null(artifact.canonical_point_content_digest, "canonical_point_content_digest")
    _sha_or_null(
        artifact.canonical_point_content_spec_digest, "canonical_point_content_spec_digest"
    )
    if (artifact.canonical_point_content_digest is None) != (
        artifact.canonical_point_content_spec_digest is None
    ):
        raise ValueError("canonical point-content digests must be all null or all nonnull")
    _sha256(artifact.scope_coverage_manifest_digest, "scope_coverage_manifest_digest")
    _nonempty(artifact.scope_union_id, "scope_union_id")
    _nonempty(artifact.scope_construction_version, "scope_construction_version")
    _sha256(artifact.scope_union_geometry_digest, "scope_union_geometry_digest")
    _sha256(artifact.canonical_inventory_digest, "canonical_inventory_digest")
    _sha_or_null(artifact.required_inventory_digest, "required_inventory_digest")
    _sha256(artifact.configuration_digest, "configuration_digest")
    _sha256(artifact.payload_schema_digest, "payload_schema_digest")
    _nonempty(artifact.block_schema_version, "block_schema_version")
    _sha256(artifact.artifact_sha256, "artifact_sha256")
    if not isinstance(artifact.block_descriptors, tuple) or any(
        not isinstance(block, FRVArtifactBlockDescriptor) for block in artifact.block_descriptors
    ):
        raise ValueError("block_descriptors must be a tuple of block descriptors")
    if not artifact.block_descriptors:
        raise ValueError("a legal artifact requires at least one nonempty block")
    for block in artifact.block_descriptors:
        _check_block_descriptor(block)
        if block.record_count <= 0:
            raise ValueError("a legal artifact has no empty blocks")
        if block.block_schema_version != artifact.block_schema_version:
            raise ValueError("block block_schema_version must match the artifact")
        if block.payload_schema_digest != artifact.payload_schema_digest:
            raise ValueError("block payload_schema_digest must match the artifact")
    _check_block_descriptor_inventory(artifact.block_descriptors)

    for name in (
        "source_point_count",
        "unique_in_scope_point_count",
        "evaluated_point_count",
        "verified_vegetation_candidate_count",
        "excluded_non_candidate_count",
        "review_required_membership_count",
        "manual_include_count",
        "manual_exclude_count",
        "unasserted_in_scope_count",
        "canonical_inventory_count",
        "required_inventory_count",
        "approved_hard_rejection_count",
        "scope_created_or_merged_count",
        "coverage_blocked_count",
    ):
        object.__setattr__(
            artifact, name, _non_negative_int(getattr(artifact, name), name)
        )
    for name in (
        "canonical_inventory_complete",
        "required_inventory_complete",
        "approved_hard_rejection_complete",
        "scope_coverage_complete",
        "routing_complete",
        "membership_decision_complete",
    ):
        _required_bool(getattr(artifact, name), name)

    five_state_sum = (
        artifact.verified_vegetation_candidate_count
        + artifact.excluded_non_candidate_count
        + artifact.review_required_membership_count
        + artifact.manual_include_count
        + artifact.manual_exclude_count
    )
    if five_state_sum != artifact.evaluated_point_count:
        raise ValueError("five terminal state counts must sum to evaluated_point_count")
    if artifact.unique_in_scope_point_count <= 0:
        raise ValueError("a legal artifact requires positive in-scope point count")
    if artifact.unique_in_scope_point_count != artifact.evaluated_point_count:
        raise ValueError("evaluated_point_count must equal unique_in_scope_point_count")
    if artifact.unasserted_in_scope_count != 0:
        raise ValueError("a legal artifact requires zero unasserted in-scope points")

    if artifact.coverage_blocked_count != 0:
        raise ValueError("a legal artifact cannot have coverage-blocked required regions")
    if artifact.required_inventory_count != (
        artifact.scope_created_or_merged_count + artifact.coverage_blocked_count
    ):
        raise ValueError("required inventory must equal created/merged plus blocked")
    if artifact.canonical_inventory_count != (
        artifact.required_inventory_count + artifact.approved_hard_rejection_count
    ):
        raise ValueError("canonical inventory must equal required plus approved")
    for name in (
        "canonical_inventory_complete",
        "required_inventory_complete",
        "approved_hard_rejection_complete",
        "scope_coverage_complete",
        "routing_complete",
        "membership_decision_complete",
    ):
        if getattr(artifact, name) is not True:
            raise ValueError(f"a legal artifact requires {name}")

    object.__setattr__(
        artifact,
        "provider_digest_references",
        _sorted_unique_sha_tuple(artifact.provider_digest_references, "provider_digest_references"),
    )
    object.__setattr__(
        artifact,
        "override_digest_references",
        _sorted_unique_sha_tuple(artifact.override_digest_references, "override_digest_references"),
    )
    object.__setattr__(
        artifact,
        "block_digest_references",
        _sorted_unique_sha_tuple(artifact.block_digest_references, "block_digest_references"),
    )
    expected_block_digests = tuple(
        sorted(block.block_digest for block in artifact.block_descriptors)
    )
    if artifact.block_digest_references != expected_block_digests:
        raise ValueError("block_digest_references must match the block descriptors")
    for name in ("warning_digest", "limitation_digest", "provenance_digest"):
        _sha256(getattr(artifact, name), name)


@dataclass(frozen=True)
class FullResolutionVegetationMembershipArtifact:
    """Semantic membership artifact metadata; never embeds payload arrays."""

    artifact_schema_version: str
    algorithm_version: str
    decision_rule_version: str
    artifact_id: str
    source_content_identity_digest: str
    source_identity_record_digest: str
    source_file_sha256: str
    canonical_point_content_digest: str | None
    canonical_point_content_spec_digest: str | None
    scope_coverage_manifest_digest: str
    scope_union_id: str
    scope_construction_version: str
    scope_union_geometry_digest: str
    canonical_inventory_digest: str
    required_inventory_digest: str | None
    configuration_digest: str
    payload_schema_digest: str
    block_schema_version: str
    artifact_sha256: str
    block_descriptors: tuple[FRVArtifactBlockDescriptor, ...]
    source_point_count: int
    unique_in_scope_point_count: int
    evaluated_point_count: int
    verified_vegetation_candidate_count: int
    excluded_non_candidate_count: int
    review_required_membership_count: int
    manual_include_count: int
    manual_exclude_count: int
    unasserted_in_scope_count: int
    canonical_inventory_count: int
    required_inventory_count: int
    approved_hard_rejection_count: int
    scope_created_or_merged_count: int
    coverage_blocked_count: int
    canonical_inventory_complete: bool
    required_inventory_complete: bool
    approved_hard_rejection_complete: bool
    scope_coverage_complete: bool
    routing_complete: bool
    membership_decision_complete: bool
    provider_digest_references: tuple[str, ...]
    override_digest_references: tuple[str, ...]
    block_digest_references: tuple[str, ...]
    warning_digest: str
    limitation_digest: str
    provenance_digest: str
    manifest_digest: str
    configuration_purpose: FrvConfigurationPurpose | None = None

    def __post_init__(self) -> None:
        _check_artifact(self)
        computed = self.computed_digest()
        if self.manifest_digest != computed:
            raise ValueError("manifest_digest does not match canonical fields")

    def computed_artifact_id(self) -> str:
        return "frv-membership-" + frv_digest(_artifact_id_payload(self))

    def to_digest_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            if name == "manifest_digest":
                continue
            if name == "block_descriptors":
                payload[name] = [block.to_dict() for block in self.block_descriptors]
            else:
                payload[name] = getattr(self, name)
        return payload

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "manifest_digest": self.manifest_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FullResolutionVegetationMembershipArtifact:
        values = _from_dict_values(cls, data)
        values["configuration_purpose"] = _enum_from_value(
            values["configuration_purpose"],
            FrvConfigurationPurpose,
            "configuration_purpose",
        )
        raw_blocks = values["block_descriptors"]
        if not isinstance(raw_blocks, Sequence) or isinstance(raw_blocks, (str, bytes, bytearray)):
            raise ValueError("block_descriptors must be a non-string sequence")
        values["block_descriptors"] = tuple(
            FRVArtifactBlockDescriptor.from_dict(_strict_mapping(block, "block"))
            for block in raw_blocks
        )
        return cls(**values)


def _check_binding_record(binding: FRVArtifactBindingRecord) -> None:
    if binding.binding_schema_version != FRV_ARTIFACT_BINDING_SCHEMA_VERSION:
        raise ValueError("unsupported artifact binding schema version")
    _sha256(binding.source_content_identity_digest, "source_content_identity_digest")
    _sha256(binding.source_identity_record_digest, "source_identity_record_digest")
    _nonempty(binding.scope_union_id, "scope_union_id")
    _sha256(binding.configuration_digest, "configuration_digest")
    _semantic_artifact_id(binding.semantic_artifact_id, "semantic_artifact_id")
    _sha256(binding.scope_coverage_manifest_digest, "scope_coverage_manifest_digest")
    _sha256(binding.membership_artifact_manifest_digest, "membership_artifact_manifest_digest")
    _sha256(binding.membership_artifact_sha256, "membership_artifact_sha256")
    _nonempty(binding.binding_created_by_stage, "binding_created_by_stage")


@dataclass(frozen=True)
class FRVArtifactBindingRecord:
    """Immutable semantic/audit/artifact binding; digest excludes only itself."""

    binding_schema_version: str
    source_content_identity_digest: str
    source_identity_record_digest: str
    scope_union_id: str
    configuration_digest: str
    semantic_artifact_id: str
    scope_coverage_manifest_digest: str
    membership_artifact_manifest_digest: str
    membership_artifact_sha256: str
    binding_created_by_stage: str
    binding_digest: str

    def __post_init__(self) -> None:
        _check_binding_record(self)
        computed = self.computed_digest()
        if self.binding_digest != computed:
            raise ValueError("binding_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "binding_digest"
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "binding_digest": self.binding_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FRVArtifactBindingRecord:
        return cls(**_from_dict_values(cls, data))


def validate_membership_state_count_conservation(
    artifact: FullResolutionVegetationMembershipArtifact,
) -> None:
    """Validate the five-terminal-state conservation sum."""

    if not isinstance(artifact, FullResolutionVegetationMembershipArtifact):
        raise ValueError("artifact must be a FullResolutionVegetationMembershipArtifact")
    five_state_sum = (
        artifact.verified_vegetation_candidate_count
        + artifact.excluded_non_candidate_count
        + artifact.review_required_membership_count
        + artifact.manual_include_count
        + artifact.manual_exclude_count
    )
    if five_state_sum != artifact.evaluated_point_count:
        raise ValueError("five terminal state counts must sum to evaluated_point_count")


_FRV_ART1_HEADER_FIELDS = (
    "artifact_schema_version",
    "algorithm_version",
    "decision_rule_version",
    "artifact_id",
    "source_content_identity_digest",
    "scope_union_id",
    "configuration_digest",
    "payload_schema_digest",
    "block_schema_version",
    "block_count",
    "ordered_block_ids",
)
_FRV_ART1_FORBIDDEN_HEADER_KEYS = frozenset(
    {
        "artifact_sha256",
        "manifest_digest",
        "binding_digest",
        "storage_reference",
        "recorded_source_path",
        "resolved_access_path",
        "mtime",
        "host",
    }
)


def _array_descriptor_digest(schema: FRVArtifactPayloadSchema) -> str:
    return frv_digest([array.to_dict() for array in schema.arrays])


def _check_frvart1_declared_header(
    artifact: FullResolutionVegetationMembershipArtifact,
    declared_header: Mapping[str, Any],
) -> None:
    if not isinstance(declared_header, Mapping):
        raise ValueError("declared_header must be a mapping")
    keys = set(declared_header)
    if declared_header.get("magic") != _FRV_ART1_MAGIC:
        raise ValueError(f"declared domain magic must be {_FRV_ART1_MAGIC!r}")
    for key in _FRV_ART1_FORBIDDEN_HEADER_KEYS:
        if key in keys:
            raise ValueError(f"declared FRVART1 header must not contain {key}")
    expected_keys = set(_FRV_ART1_HEADER_FIELDS) | {"magic"}
    if keys != expected_keys:
        missing = sorted(expected_keys - keys)
        extra = sorted(keys - expected_keys)
        if missing:
            raise ValueError(f"declared FRVART1 header is missing key(s): {', '.join(missing)}")
        raise ValueError(f"declared FRVART1 header has extra key(s): {', '.join(extra)}")
    artifact_fields = {
        "artifact_schema_version": artifact.artifact_schema_version,
        "algorithm_version": artifact.algorithm_version,
        "decision_rule_version": artifact.decision_rule_version,
        "artifact_id": artifact.artifact_id,
        "source_content_identity_digest": artifact.source_content_identity_digest,
        "scope_union_id": artifact.scope_union_id,
        "configuration_digest": artifact.configuration_digest,
        "payload_schema_digest": artifact.payload_schema_digest,
        "block_schema_version": artifact.block_schema_version,
    }
    for name, expected in artifact_fields.items():
        if declared_header[name] != expected:
            raise ValueError(f"declared FRVART1 header {name} does not match the artifact")
    declared_block_count = _non_negative_int(declared_header["block_count"], "block_count")
    if declared_block_count != len(artifact.block_descriptors):
        raise ValueError("declared FRVART1 header block_count must match the block descriptors")
    raw_ordered_block_ids = declared_header["ordered_block_ids"]
    if type(raw_ordered_block_ids) not in (list, tuple):
        raise ValueError("declared FRVART1 header ordered_block_ids must be a list or tuple")
    for block_id in raw_ordered_block_ids:
        if (
            type(block_id) is not str
            or not block_id
            or block_id != block_id.strip()
        ):
            raise ValueError(
                "declared FRVART1 header ordered_block_ids entries must be "
                "non-empty strings without surrounding whitespace"
            )
    ordered_block_ids = tuple(raw_ordered_block_ids)
    canonical_block_ids = tuple(block.block_id for block in artifact.block_descriptors)
    if ordered_block_ids != canonical_block_ids:
        raise ValueError("declared FRVART1 header ordered_block_ids must be canonical block order")


def _check_artifact_block_schema_binding(
    artifact: FullResolutionVegetationMembershipArtifact,
    schema: FRVArtifactPayloadSchema,
) -> None:
    if not isinstance(schema, FRVArtifactPayloadSchema):
        raise ValueError("payload_schema must be a FRVArtifactPayloadSchema")
    if artifact.payload_schema_digest != schema.payload_schema_digest:
        raise ValueError("artifact payload_schema_digest must match the payload schema")
    if artifact.block_schema_version != schema.block_schema_version:
        raise ValueError("artifact block_schema_version must match the payload schema")
    if schema.payload_encoding_id != FRV_PAYLOAD_ENCODING_ID:
        raise ValueError("declared payload encoding must be FRV-BLOCK-PAYLOAD-1")
    if schema.payload_encoder_version != FRV_PAYLOAD_ENCODER_VERSION:
        raise ValueError("declared payload encoder version must be 1")
    if schema.payload_compression_algorithm != FRV_PAYLOAD_COMPRESSION_ALGORITHM:
        raise ValueError("declared payload compression algorithm must be NONE")
    if schema.payload_compression_version != FRV_PAYLOAD_COMPRESSION_VERSION:
        raise ValueError("declared payload compression version must be 1")
    expected_array_digest = _array_descriptor_digest(schema)
    for block in artifact.block_descriptors:
        if block.payload_schema_digest != schema.payload_schema_digest:
            raise ValueError("block payload_schema_digest must match the payload schema")
        if block.payload_schema_version != schema.payload_schema_version:
            raise ValueError("block payload_schema_version must match the payload schema")
        if block.block_schema_version != schema.block_schema_version:
            raise ValueError("block block_schema_version must match the payload schema")
        if block.array_descriptor_digest != expected_array_digest:
            raise ValueError("block array_descriptor_digest must match the payload schema arrays")


def validate_artifact_payload_digest_domain(
    artifact: FullResolutionVegetationMembershipArtifact,
    payload_schema: FRVArtifactPayloadSchema | None = None,
    declared_header: Mapping[str, Any] | None = None,
) -> None:
    """Validate the semantic artifact ID, canonical block order, and header domain.

    The exact payload byte domain is A6's concern; A1 validates the declared
    FRVART1 header fields, canonical block ordering, and the declared payload
    encoding/schema bindings that the domain depends on.
    """

    if not isinstance(artifact, FullResolutionVegetationMembershipArtifact):
        raise ValueError("artifact must be a FullResolutionVegetationMembershipArtifact")
    if artifact.artifact_id != artifact.computed_artifact_id():
        raise ValueError("artifact_id does not match canonical semantic inputs")
    _check_block_descriptor_inventory(artifact.block_descriptors)
    if artifact.block_digest_references != tuple(
        sorted(block.block_digest for block in artifact.block_descriptors)
    ):
        raise ValueError("block_digest_references must match the block descriptors")
    if declared_header is not None:
        _check_frvart1_declared_header(artifact, declared_header)
    if payload_schema is not None:
        _check_artifact_block_schema_binding(artifact, payload_schema)


def validate_artifact_binding_record(
    binding: FRVArtifactBindingRecord,
    artifact: FullResolutionVegetationMembershipArtifact,
    coverage_manifest: FRVScopeCoverageManifest | None = None,
) -> None:
    """Validate semantic identity, artifact-ID derivation, and digest equality."""

    if not isinstance(binding, FRVArtifactBindingRecord):
        raise ValueError("binding must be a FRVArtifactBindingRecord")
    if not isinstance(artifact, FullResolutionVegetationMembershipArtifact):
        raise ValueError("artifact must be a FullResolutionVegetationMembershipArtifact")
    if binding.source_content_identity_digest != artifact.source_content_identity_digest:
        raise ValueError("binding content identity does not match the artifact")
    if binding.source_identity_record_digest != artifact.source_identity_record_digest:
        raise ValueError("binding record audit does not match the artifact")
    if binding.semantic_artifact_id != artifact.artifact_id:
        raise ValueError("binding semantic artifact id does not match the artifact id")
    if binding.scope_union_id != artifact.scope_union_id:
        raise ValueError("binding scope union id does not match the artifact")
    if binding.configuration_digest != artifact.configuration_digest:
        raise ValueError("binding configuration digest does not match the artifact")
    if binding.scope_coverage_manifest_digest != artifact.scope_coverage_manifest_digest:
        raise ValueError("binding coverage manifest digest does not match the artifact")
    if binding.membership_artifact_manifest_digest != artifact.manifest_digest:
        raise ValueError("binding manifest digest does not match the artifact manifest")
    if binding.membership_artifact_sha256 != artifact.artifact_sha256:
        raise ValueError("binding artifact sha256 does not match the artifact")
    if coverage_manifest is not None:
        if not isinstance(coverage_manifest, FRVScopeCoverageManifest):
            raise ValueError("coverage_manifest must be a FRVScopeCoverageManifest")
        if binding.scope_coverage_manifest_digest != coverage_manifest.manifest_digest:
            raise ValueError("binding coverage manifest digest does not match the manifest")
