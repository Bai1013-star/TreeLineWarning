"""Strict, non-I/O contracts for TREE_WIRE_DISTANCE_ALGORITHM_V1 clearance.

The contracts bind a deterministic tree-wire 3D minimum-clearance result that
is computed from full-resolution confirmed V4 tree memberships and distance-
ready W1 polyline edges only.  Every record is frozen, digest-bound and
round-trips through canonical JSON with the exact repository conventions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Any

from core.contracts._json import reject_unknown_fields, require_fields
from core.contracts.frv import frv_digest

TREE_WIRE_DISTANCE_ALGORITHM_V1 = "tree-wire-distance-algorithm-v1"
TREE_WIRE_CLEARANCE_SCHEMA_VERSION = "tree-wire-clearance-v1"
TREE_WIRE_CLEARANCE_CONFIG_SCHEMA_VERSION = "tree-wire-clearance-config-v1"
TREE_WIRE_CLEARANCE_RESULT_DIGEST_DOMAIN = "TWC1"
TREE_WIRE_CLEARANCE_CONFIG_DIGEST_DOMAIN = "TWCCFG1"


class TreeWireClearanceStatus(IntEnum):
    """Global result status.  Provenance failures never produce a result."""

    VALID = 0
    NO_DISTANCE_READY_EDGE = 1


class TreeWireClearanceRecordStatus(IntEnum):
    """Per-tree record status."""

    VALID = 0
    INVALID_TREE_NO_VALID_POINT = 1
    NO_DISTANCE_READY_EDGE = 2


class TreeWireClearanceQualityFlag(IntFlag):
    """Quality context preserved from the winning W1 wire/segment.

    Distance is never silently promoted: a non-CONFIRMED nearest wire or a
    nearest segment carrying fallback/review quality is reported explicitly.
    """

    NONE = 0
    NEAREST_WIRE_NOT_CONFIRMED = 1
    NEAREST_SEGMENT_FALLBACK_OR_REVIEW = 2


# ---------------------------------------------------------------------------
# Private strict validation helpers
# ---------------------------------------------------------------------------


def _exact_str(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a built-in str")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if value != value.strip():
        raise ValueError(f"{name} must not have surrounding whitespace")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive built-in int")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative built-in int")
    return value


def _finite_float(value: Any, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite built-in float")
    return value


def _finite_vec3(value: Any, name: str) -> tuple[float, float, float]:
    if type(value) not in (list, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a three-value sequence")
    return (
        _finite_float(value[0], f"{name}[0]"),
        _finite_float(value[1], f"{name}[1]"),
        _finite_float(value[2], f"{name}[2]"),
    )


def _sha(value: Any, name: str) -> str:
    text = _exact_str(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _flag_mask(value: Any, flag_type: type, name: str) -> int:
    if type(value) is not int or value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"{name} must be a nonnegative 32-bit flag mask")
    return value


def _vec3_from_json(value: Any, name: str) -> tuple[float, float, float]:
    if value is None:
        return None
    return _finite_vec3(value, name)


# ---------------------------------------------------------------------------
# TreeWireClearanceConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TreeWireClearanceConfig:
    """Frozen TREE_WIRE_DISTANCE_ALGORITHM_V1 clearance configuration.

    Binds the algorithm/version, the bounded BVH leaf capacity and the exact
    distance tie tolerance, and carries a canonical config digest.
    """

    schema_version: str
    algorithm_version: str
    bvh_leaf_cap: int
    tie_tolerance_m: float
    config_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != TREE_WIRE_CLEARANCE_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported tree-wire clearance config schema")
        if self.algorithm_version != TREE_WIRE_DISTANCE_ALGORITHM_V1:
            raise ValueError("unsupported tree-wire distance algorithm version")
        object.__setattr__(self, "bvh_leaf_cap", _positive_int(self.bvh_leaf_cap, "bvh_leaf_cap"))
        object.__setattr__(
            self,
            "tie_tolerance_m",
            _finite_float(self.tie_tolerance_m, "tie_tolerance_m"),
        )
        if self.tie_tolerance_m <= 0.0:
            raise ValueError("tie_tolerance_m must be positive")
        _sha(self.config_digest, "config_digest")
        if self.config_digest != self.computed_digest():
            raise ValueError("config_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": TREE_WIRE_CLEARANCE_CONFIG_DIGEST_DOMAIN,
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "bvh_leaf_cap": self.bvh_leaf_cap,
            "tie_tolerance_m": self.tie_tolerance_m,
        }

    def computed_digest(self) -> str:
        payload = dict(self.to_digest_dict())
        payload.pop("digest_domain")
        return frv_digest(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "bvh_leaf_cap": self.bvh_leaf_cap,
            "tie_tolerance_m": self.tie_tolerance_m,
            "config_digest": self.config_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeWireClearanceConfig:
        values = _mapping(data, cls.__name__)
        require_fields(values, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(values, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(values))


def default_tree_wire_clearance_config(**overrides: Any) -> TreeWireClearanceConfig:
    """Return the canonical V1 clearance config with deterministic defaults."""

    fields: dict[str, Any] = {
        "schema_version": TREE_WIRE_CLEARANCE_CONFIG_SCHEMA_VERSION,
        "algorithm_version": TREE_WIRE_DISTANCE_ALGORITHM_V1,
        "bvh_leaf_cap": 16,
        "tie_tolerance_m": 1e-9,
    }
    fields.update(overrides)
    digest_fields = dict(fields)
    digest_fields.pop("config_digest", None)
    return TreeWireClearanceConfig(**fields, config_digest=frv_digest(digest_fields))


# ---------------------------------------------------------------------------
# TreeWireClearanceRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TreeWireClearanceRecord:
    """One deterministic per-tree minimum-clearance result."""

    tree_id: int
    status: TreeWireClearanceRecordStatus
    nearest_wire_id: int | None
    minimum_clearance_m: float | None
    nearest_tree_original_index: int | None
    nearest_tree_xyz_local_m: tuple[float, float, float] | None
    nearest_wire_xyz_local_m: tuple[float, float, float] | None
    w1_segment_index: int | None
    polyline_edge_index: int | None
    edge_interpolation_t: float | None
    tree_point_count: int
    candidate_edge_count: int
    evaluated_edge_count: int
    quality_flags: int
    nearest_wire_status: int | None
    nearest_segment_quality_flags: int | None

    def __post_init__(self) -> None:
        _non_negative_int(self.tree_id, "tree_id")
        if type(self.status) is not TreeWireClearanceRecordStatus:
            raise ValueError("status must be a TreeWireClearanceRecordStatus")
        object.__setattr__(
            self, "tree_point_count", _non_negative_int(self.tree_point_count, "tree_point_count")
        )
        object.__setattr__(
            self,
            "candidate_edge_count",
            _non_negative_int(self.candidate_edge_count, "candidate_edge_count"),
        )
        object.__setattr__(
            self,
            "evaluated_edge_count",
            _non_negative_int(self.evaluated_edge_count, "evaluated_edge_count"),
        )
        object.__setattr__(
            self,
            "quality_flags",
            _flag_mask(self.quality_flags, TreeWireClearanceQualityFlag, "quality_flags"),
        )
        if self.status is TreeWireClearanceRecordStatus.VALID:
            if self.nearest_wire_id is None:
                raise ValueError("nearest_wire_id is required for VALID records")
            _non_negative_int(self.nearest_wire_id, "nearest_wire_id")
            if self.minimum_clearance_m is None:
                raise ValueError("minimum_clearance_m is required for VALID records")
            _finite_float(self.minimum_clearance_m, "minimum_clearance_m")
            if self.minimum_clearance_m < 0.0:
                raise ValueError("minimum_clearance_m must be nonnegative")
            if self.nearest_tree_original_index is None:
                raise ValueError("nearest_tree_original_index is required for VALID records")
            _non_negative_int(self.nearest_tree_original_index, "nearest_tree_original_index")
            object.__setattr__(
                self,
                "nearest_tree_xyz_local_m",
                _finite_vec3(self.nearest_tree_xyz_local_m, "nearest_tree_xyz_local_m"),
            )
            object.__setattr__(
                self,
                "nearest_wire_xyz_local_m",
                _finite_vec3(self.nearest_wire_xyz_local_m, "nearest_wire_xyz_local_m"),
            )
            if self.w1_segment_index is None:
                raise ValueError("w1_segment_index is required for VALID records")
            _non_negative_int(self.w1_segment_index, "w1_segment_index")
            if self.polyline_edge_index is None:
                raise ValueError("polyline_edge_index is required for VALID records")
            _non_negative_int(self.polyline_edge_index, "polyline_edge_index")
            if self.edge_interpolation_t is None:
                raise ValueError("edge_interpolation_t is required for VALID records")
            object.__setattr__(
                self,
                "edge_interpolation_t",
                _finite_float(self.edge_interpolation_t, "edge_interpolation_t"),
            )
            if not (0.0 <= self.edge_interpolation_t <= 1.0):
                raise ValueError("edge_interpolation_t must be in [0, 1]")
            if self.nearest_wire_status is None:
                raise ValueError("nearest_wire_status is required for VALID records")
            _non_negative_int(self.nearest_wire_status, "nearest_wire_status")
            if self.nearest_segment_quality_flags is None:
                raise ValueError("nearest_segment_quality_flags is required for VALID records")
            _non_negative_int(
                self.nearest_segment_quality_flags, "nearest_segment_quality_flags"
            )
        else:
            for name in (
                "nearest_wire_id",
                "minimum_clearance_m",
                "nearest_tree_original_index",
                "nearest_tree_xyz_local_m",
                "nearest_wire_xyz_local_m",
                "w1_segment_index",
                "polyline_edge_index",
                "edge_interpolation_t",
                "nearest_wire_status",
                "nearest_segment_quality_flags",
            ):
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} must be None for non-VALID records")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "status": int(self.status),
            "nearest_wire_id": self.nearest_wire_id,
            "minimum_clearance_m": self.minimum_clearance_m,
            "nearest_tree_original_index": self.nearest_tree_original_index,
            "nearest_tree_xyz_local_m": (
                None
                if self.nearest_tree_xyz_local_m is None
                else list(self.nearest_tree_xyz_local_m)
            ),
            "nearest_wire_xyz_local_m": (
                None
                if self.nearest_wire_xyz_local_m is None
                else list(self.nearest_wire_xyz_local_m)
            ),
            "w1_segment_index": self.w1_segment_index,
            "polyline_edge_index": self.polyline_edge_index,
            "edge_interpolation_t": self.edge_interpolation_t,
            "tree_point_count": self.tree_point_count,
            "candidate_edge_count": self.candidate_edge_count,
            "evaluated_edge_count": self.evaluated_edge_count,
            "quality_flags": self.quality_flags,
            "nearest_wire_status": self.nearest_wire_status,
            "nearest_segment_quality_flags": self.nearest_segment_quality_flags,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_digest_dict()
        payload["status"] = self.status.name
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeWireClearanceRecord:
        values = _mapping(data, cls.__name__)
        require_fields(values, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(values, set(cls.__dataclass_fields__), cls.__name__)
        values["status"] = TreeWireClearanceRecordStatus[values["status"]]
        values["nearest_tree_xyz_local_m"] = _vec3_from_json(
            values["nearest_tree_xyz_local_m"], "nearest_tree_xyz_local_m"
        )
        values["nearest_wire_xyz_local_m"] = _vec3_from_json(
            values["nearest_wire_xyz_local_m"], "nearest_wire_xyz_local_m"
        )
        return cls(**values)


# ---------------------------------------------------------------------------
# TreeWireClearanceResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TreeWireClearanceResult:
    """The full digest-bound tree-wire clearance result.

    Binds the V4 result/source identity, the W1 NPZ/JSON SHA-256 values, the
    hash-verified A4 parent SHA-256 values, the shared local origin context
    and the deterministic per-tree records.
    """

    schema_version: str
    algorithm_version: str
    status: TreeWireClearanceStatus
    v4_result_digest: str
    v4_source_id: str
    v4_source_identity_record_digest: str
    v4_origin_context: tuple[float, float, float]
    a4_source_id: str
    a4_manifest_source_id: str
    a4_origin_xyz: tuple[float, float, float]
    w1_npz_sha256: str
    w1_json_sha256: str
    a4_npz_sha256: str
    a4_json_sha256: str
    config_digest: str
    records: tuple[TreeWireClearanceRecord, ...]
    result_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != TREE_WIRE_CLEARANCE_SCHEMA_VERSION:
            raise ValueError("unsupported tree-wire clearance schema version")
        if self.algorithm_version != TREE_WIRE_DISTANCE_ALGORITHM_V1:
            raise ValueError("unsupported tree-wire distance algorithm version")
        if type(self.status) is not TreeWireClearanceStatus:
            raise ValueError("status must be a TreeWireClearanceStatus")
        for name in (
            "v4_result_digest",
            "v4_source_identity_record_digest",
            "w1_npz_sha256",
            "w1_json_sha256",
            "a4_npz_sha256",
            "a4_json_sha256",
            "config_digest",
        ):
            _sha(getattr(self, name), name)
        _exact_str(self.v4_source_id, "v4_source_id")
        _exact_str(self.a4_source_id, "a4_source_id")
        _exact_str(self.a4_manifest_source_id, "a4_manifest_source_id")
        object.__setattr__(
            self, "v4_origin_context", _finite_vec3(self.v4_origin_context, "v4_origin_context")
        )
        object.__setattr__(self, "a4_origin_xyz", _finite_vec3(self.a4_origin_xyz, "a4_origin_xyz"))
        if self.v4_origin_context != self.a4_origin_xyz:
            raise ValueError("v4_origin_context must equal a4_origin_xyz")
        if self.v4_source_id != self.a4_source_id:
            raise ValueError("v4_source_id must equal a4_source_id")
        if not isinstance(self.records, tuple) or any(
            type(record) is not TreeWireClearanceRecord for record in self.records
        ):
            raise ValueError("records must be a tuple of TreeWireClearanceRecord values")
        _sha(self.result_digest, "result_digest")

    def validate_digest(self) -> None:
        """Fail if ``result_digest`` does not match the canonical fields."""

        if self.result_digest != self.computed_result_digest():
            raise ValueError("result_digest does not match canonical fields")

    def computed_result_digest(self) -> str:
        return frv_digest(
            {
                "digest_domain": TREE_WIRE_CLEARANCE_RESULT_DIGEST_DOMAIN,
                "schema_version": self.schema_version,
                "algorithm_version": self.algorithm_version,
                "status": int(self.status),
                "v4_result_digest": self.v4_result_digest,
                "v4_source_id": self.v4_source_id,
                "v4_source_identity_record_digest": self.v4_source_identity_record_digest,
                "v4_origin_context": [float(value) for value in self.v4_origin_context],
                "a4_source_id": self.a4_source_id,
                "a4_manifest_source_id": self.a4_manifest_source_id,
                "a4_origin_xyz": [float(value) for value in self.a4_origin_xyz],
                "w1_npz_sha256": self.w1_npz_sha256,
                "w1_json_sha256": self.w1_json_sha256,
                "a4_npz_sha256": self.a4_npz_sha256,
                "a4_json_sha256": self.a4_json_sha256,
                "config_digest": self.config_digest,
                "records": [record.to_digest_dict() for record in self.records],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "status": self.status.name,
            "v4_result_digest": self.v4_result_digest,
            "v4_source_id": self.v4_source_id,
            "v4_source_identity_record_digest": self.v4_source_identity_record_digest,
            "v4_origin_context": list(self.v4_origin_context),
            "a4_source_id": self.a4_source_id,
            "a4_manifest_source_id": self.a4_manifest_source_id,
            "a4_origin_xyz": list(self.a4_origin_xyz),
            "w1_npz_sha256": self.w1_npz_sha256,
            "w1_json_sha256": self.w1_json_sha256,
            "a4_npz_sha256": self.a4_npz_sha256,
            "a4_json_sha256": self.a4_json_sha256,
            "config_digest": self.config_digest,
            "records": [record.to_dict() for record in self.records],
            "result_digest": self.result_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeWireClearanceResult:
        values = _mapping(data, cls.__name__)
        require_fields(values, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(values, set(cls.__dataclass_fields__), cls.__name__)
        values["status"] = TreeWireClearanceStatus[values["status"]]
        values["v4_origin_context"] = _finite_vec3(values["v4_origin_context"], "v4_origin_context")
        values["a4_origin_xyz"] = _finite_vec3(values["a4_origin_xyz"], "a4_origin_xyz")
        raw_records = values["records"]
        if not isinstance(raw_records, list | tuple):
            raise ValueError("records must be a sequence")
        values["records"] = tuple(
            TreeWireClearanceRecord.from_dict(_mapping(item, "record")) for item in raw_records
        )
        result = cls(**values)
        result.validate_digest()
        return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


__all__ = [
    "TREE_WIRE_DISTANCE_ALGORITHM_V1",
    "TREE_WIRE_CLEARANCE_SCHEMA_VERSION",
    "TREE_WIRE_CLEARANCE_CONFIG_SCHEMA_VERSION",
    "TREE_WIRE_CLEARANCE_RESULT_DIGEST_DOMAIN",
    "TREE_WIRE_CLEARANCE_CONFIG_DIGEST_DOMAIN",
    "TreeWireClearanceStatus",
    "TreeWireClearanceRecordStatus",
    "TreeWireClearanceQualityFlag",
    "TreeWireClearanceConfig",
    "TreeWireClearanceRecord",
    "TreeWireClearanceResult",
    "default_tree_wire_clearance_config",
]
