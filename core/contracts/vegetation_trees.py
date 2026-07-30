"""Portable V3 single-scene tree-segmentation contracts.

This module freezes V3 identity and safety invariants only.  It deliberately
does not implement input loading, structural evidence, tree segmentation, or
cross-cluster association.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, IntEnum, IntFlag
from types import MappingProxyType
from typing import Any

import numpy as np

from core.contracts._json import json_safe_mapping, reject_unknown_fields, require_fields
from core.contracts.vegetation_instances import VegetationClusterStatus

TREE_SEGMENTATION_SCHEMA_VERSION = "tree-segmentation-v1"
TREE_SEGMENTATION_ALGORITHM_VERSION = "v3-single-tree-segmentation-v1"


class TreeSegmentationStatus(IntEnum):
    CONFIRMED_TREE = 0
    REVIEW_REQUIRED = 1
    PROVISIONAL = 2
    REJECTED = 3
    UNRESOLVED = 4


class TreeSegmentationErrorCode(str, Enum):
    """Stable V3 fatal error categories for a future runner and loader."""

    SCHEMA_MISMATCH = "V3_SCHEMA_MISMATCH"
    ALGORITHM_VERSION_MISMATCH = "V3_ALGORITHM_VERSION_MISMATCH"
    CORRUPT_SIBLING_ARTIFACT = "V3_CORRUPT_SIBLING_ARTIFACT"
    NONFINITE_REQUIRED_SOURCE_GEOMETRY = "V3_NONFINITE_REQUIRED_SOURCE_GEOMETRY"
    INVALID_INDEXES = "V3_INVALID_INDEXES"
    INCONSISTENT_CLUSTER_MEMBERSHIP = "V3_INCONSISTENT_CLUSTER_MEMBERSHIP"
    COORDINATE_MISMATCH = "V3_COORDINATE_MISMATCH"
    MISSING_REQUIRED_FIELD = "V3_MISSING_REQUIRED_FIELD"
    UNSUPPORTED_STATUS = "V3_UNSUPPORTED_STATUS"
    DETERMINISTIC_ORDERING_FAILURE = "V3_DETERMINISTIC_ORDERING_FAILURE"
    PUBLICATION_FAILURE = "V3_PUBLICATION_FAILURE"
    CANCELLED = "V3_CANCELLED"


class TreeSegmentationQualityFlag(IntFlag):
    NONE = 0
    RESOURCE_SAFETY_LIMIT = 1
    INSUFFICIENT_SUPPORT = 2
    MISSING_TRUNK_EVIDENCE = 4
    MISSING_CROWN_EVIDENCE = 8
    STRUCTURAL_CONFLICT = 16
    PARTIAL_GEOMETRY = 32
    CROSS_CLUSTER_REVIEW = 64
    SOURCE_NOT_V2_CONFIRMED = 128
    OTHER_REVIEW_REASON = 256


_TREE_QUALITY_FLAG_MASK = sum(int(value) for value in TreeSegmentationQualityFlag)
_TREE_RESULT_VALIDATION_CHUNK_SIZE = 65_536


def _finite(value: float, name: str, minimum: float | None = None) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int | np.integer) or isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int | np.integer) or isinstance(value, bool) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _array(value: Any, dtype: Any, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != np.dtype(dtype) or value.ndim != len(shape):
        raise ValueError(f"{name} has invalid dtype or rank")
    if any(
        expected is not None and expected != actual
        for expected, actual in zip(shape, value.shape, strict=True)
    ):
        raise ValueError(f"{name} has invalid shape")
    if value.dtype == object:
        raise ValueError(f"{name} must not use object dtype")
    if not value.flags.owndata or not value.flags.c_contiguous or value.flags.writeable:
        raise ValueError(f"{name} has invalid ownership, contiguity, or mutability")
    return value


def _freeze_array(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _json_integer_array(value: Any, dtype: Any, name: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{name} must be a one-dimensional JSON sequence")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must contain only JSON integer values")
    return _freeze_array(np.array(value, dtype=dtype, order="C", copy=True, subok=False))


def _json_bool_array(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{name} must be a one-dimensional JSON sequence")
    if any(not isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must contain only JSON bool values")
    return _freeze_array(np.array(value, dtype=np.bool_, order="C", copy=True, subok=False))


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        backing = {key: _freeze_json_value(value[key]) for key in sorted(value)}
        return MappingProxyType(backing)
    if isinstance(value, list | tuple):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _immutable_json_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _freeze_json_value(json_safe_mapping(value, name))


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _iter_chunks(length: int) -> range:
    return range(0, length, _TREE_RESULT_VALIDATION_CHUNK_SIZE)


def _origin_xyz(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError("origin_xyz must be a one-dimensional XYZ sequence")
    if len(value) != 3:
        raise ValueError("origin_xyz must contain exactly three values")
    origin = []
    for coordinate in value:
        if not isinstance(coordinate, int | float | np.integer | np.floating) or isinstance(
            coordinate, bool | np.bool_
        ):
            raise ValueError("origin_xyz must contain only finite numeric values")
        numeric = float(coordinate)
        if not math.isfinite(numeric):
            raise ValueError("origin_xyz must contain only finite numeric values")
        origin.append(numeric)
    return (origin[0], origin[1], origin[2])


def _geometry_scalar(value: Any, name: str, minimum: float | None = None) -> float:
    if not isinstance(value, int | float | np.integer | np.floating) or isinstance(
        value, bool | np.bool_
    ):
        raise ValueError(f"{name} must be a finite numeric value")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite numeric value")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _apex_xyz_local(value: Any) -> tuple[float, float, float]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError("apex_xyz_local must be a one-dimensional XYZ sequence")
        coordinates: Sequence[Any] = value
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        coordinates = value
    else:
        raise ValueError("apex_xyz_local must be a one-dimensional XYZ sequence")
    if len(coordinates) != 3:
        raise ValueError("apex_xyz_local must contain exactly three values")
    apex = tuple(_geometry_scalar(coordinate, "apex_xyz_local") for coordinate in coordinates)
    return (apex[0], apex[1], apex[2])


def _crown_xy_bounds_local(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError("crown_xy_bounds_local must be a one-dimensional XY bounds sequence")
        coordinates: Sequence[Any] = value
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        coordinates = value
    else:
        raise ValueError("crown_xy_bounds_local must be a one-dimensional XY bounds sequence")
    if len(coordinates) != 4:
        raise ValueError("crown_xy_bounds_local must contain exactly four values")
    bounds = tuple(
        _geometry_scalar(coordinate, "crown_xy_bounds_local") for coordinate in coordinates
    )
    min_x, min_y, max_x, max_y = bounds
    if min_x > max_x or min_y > max_y:
        raise ValueError("crown_xy_bounds_local minima must not exceed maxima")
    return (min_x, min_y, max_x, max_y)


def _enum_value(value: Any, enum_type: type[IntEnum], name: str) -> IntEnum:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be a non-bool integer enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{name} contains an invalid enum value") from exc


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{name} must be a non-string sequence")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must contain only strings")
    return tuple(value)


@dataclass(frozen=True)
class TreeSegmentationConfig:
    """V3A1 safety and identity policy; algorithm thresholds are deferred to V3B."""

    allow_cross_cluster_confirmation: bool = False
    allow_crown_only_confirmation: bool = False
    allow_trunk_only_confirmation: bool = False
    maximum_cluster_points: int = 250_000
    maximum_graph_edges: int = 2_000_000
    maximum_seeds: int = 128
    deterministic_quantization_m: float = 1e-6
    record_memory: bool = True

    def __post_init__(self) -> None:
        for name in (
            "allow_cross_cluster_confirmation",
            "allow_crown_only_confirmation",
            "allow_trunk_only_confirmation",
            "record_memory",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        for name in ("maximum_cluster_points", "maximum_graph_edges", "maximum_seeds"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        object.__setattr__(
            self,
            "deterministic_quantization_m",
            _finite(self.deterministic_quantization_m, "deterministic_quantization_m", 0.0),
        )
        if self.deterministic_quantization_m == 0.0:
            raise ValueError("deterministic_quantization_m must be positive")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeSegmentationConfig:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class TreeCrownSummary:
    """Minimal planimetric crown geometry for one retained V3 segment.

    ``crown_xy_bounds_local`` is an axis-aligned bounding box in the result's
    local Cartesian metre frame, ordered as ``(min_x, min_y, max_x, max_y)``.
    ``crown_area_m2`` is an algorithm-estimated horizontal footprint area, not
    necessarily the bounding-box area. ``crown_equivalent_radius_m`` is the
    radius of a circle with the same area.
    """

    crown_xy_bounds_local: tuple[float, float, float, float]
    crown_area_m2: float
    crown_equivalent_radius_m: float

    def __post_init__(self) -> None:
        bounds = _crown_xy_bounds_local(self.crown_xy_bounds_local)
        area = _geometry_scalar(self.crown_area_m2, "crown_area_m2", 0.0)
        radius = _geometry_scalar(
            self.crown_equivalent_radius_m,
            "crown_equivalent_radius_m",
            0.0,
        )
        min_x, min_y, max_x, max_y = bounds
        box_area = (max_x - min_x) * (max_y - min_y)
        if box_area == 0.0:
            if area != 0.0:
                raise ValueError("crown_area_m2 must not exceed the crown XY bounding-box area")
        elif area > box_area and not math.isclose(
            area,
            box_area,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("crown_area_m2 must not exceed the crown XY bounding-box area")
        expected_radius = math.sqrt(area / math.pi)
        if area == 0.0:
            if radius != 0.0:
                raise ValueError(
                    "crown_equivalent_radius_m must be zero when crown_area_m2 is zero"
                )
        elif not math.isclose(
            radius,
            expected_radius,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("crown_equivalent_radius_m is inconsistent with crown_area_m2")
        object.__setattr__(self, "crown_xy_bounds_local", bounds)
        object.__setattr__(self, "crown_area_m2", area)
        object.__setattr__(self, "crown_equivalent_radius_m", radius)

    def to_dict(self) -> dict[str, Any]:
        return {
            "crown_xy_bounds_local": list(self.crown_xy_bounds_local),
            "crown_area_m2": self.crown_area_m2,
            "crown_equivalent_radius_m": self.crown_equivalent_radius_m,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeCrownSummary:
        if not isinstance(data, Mapping):
            raise ValueError("TreeCrownSummary must be a mapping")
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class TreeGeometrySummary:
    """Minimal vertical geometry for one retained V3 segment.

    ``apex_xyz_local`` is an algorithm-selected representative in the result's
    local Cartesian metre frame; V3C1B does not freeze its selection policy.
    HAG values use metres above ground and may be absent only as a pair.
    ``vertical_extent_m`` is maximum local Z minus minimum local Z over valid
    local-geometry support and is intentionally not compared with either HAG
    value.
    """

    apex_xyz_local: tuple[float, float, float]
    maximum_hag_m: float | None
    robust_height_hag_m: float | None
    vertical_extent_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "apex_xyz_local", _apex_xyz_local(self.apex_xyz_local))
        maximum_missing = self.maximum_hag_m is None
        robust_missing = self.robust_height_hag_m is None
        if maximum_missing != robust_missing:
            raise ValueError("maximum_hag_m and robust_height_hag_m must both be present or None")
        if not maximum_missing:
            maximum = _geometry_scalar(self.maximum_hag_m, "maximum_hag_m", 0.0)
            robust = _geometry_scalar(self.robust_height_hag_m, "robust_height_hag_m", 0.0)
            if robust > maximum:
                raise ValueError("robust_height_hag_m must not exceed maximum_hag_m")
            object.__setattr__(self, "maximum_hag_m", maximum)
            object.__setattr__(self, "robust_height_hag_m", robust)
        object.__setattr__(
            self,
            "vertical_extent_m",
            _geometry_scalar(self.vertical_extent_m, "vertical_extent_m", 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "apex_xyz_local": list(self.apex_xyz_local),
            "maximum_hag_m": self.maximum_hag_m,
            "robust_height_hag_m": self.robust_height_hag_m,
            "vertical_extent_m": self.vertical_extent_m,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeGeometrySummary:
        if not isinstance(data, Mapping):
            raise ValueError("TreeGeometrySummary must be a mapping")
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class TreeInstanceRecord:
    """One V3 hypothesis; only ``CONFIRMED_TREE`` carries a formal tree ID.

    Confirmed trees require complete geometry including both HAG values.  Other
    statuses may retain complete, local-only, or no geometry.  The containing
    result requires a positive-area crown for confirmation unless its existing
    trunk-only confirmation policy is enabled.
    """

    tree_id: int
    segment_id: int
    status: TreeSegmentationStatus
    source_cluster_ids: tuple[int, ...]
    source_cluster_statuses: tuple[VegetationClusterStatus, ...]
    source_cluster_distance_ready: tuple[bool, ...]
    tree_distance_ready: bool
    confidence: float
    geometry_summary: TreeGeometrySummary | None
    crown_summary: TreeCrownSummary | None
    quality_flags: TreeSegmentationQualityFlag = TreeSegmentationQualityFlag.NONE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "segment_id",
            _non_negative_int(self.segment_id, "segment_id"),
        )
        status = _enum_value(self.status, TreeSegmentationStatus, "status")
        object.__setattr__(self, "status", TreeSegmentationStatus(status))
        if not isinstance(self.tree_id, int | np.integer) or isinstance(self.tree_id, bool):
            raise ValueError("tree_id must be -1 or a non-negative integer")
        tree_id = int(self.tree_id)
        if tree_id < -1:
            raise ValueError("tree_id must be -1 or a non-negative integer")
        object.__setattr__(self, "tree_id", tree_id)
        source_ids = tuple(
            _non_negative_int(value, "source_cluster_ids") for value in self.source_cluster_ids
        )
        if (
            not source_ids
            or source_ids != tuple(sorted(source_ids))
            or len(source_ids) != len(set(source_ids))
        ):
            raise ValueError("source_cluster_ids must be non-empty, sorted, and unique")
        object.__setattr__(self, "source_cluster_ids", source_ids)
        if not isinstance(self.source_cluster_statuses, Sequence) or isinstance(
            self.source_cluster_statuses, str | bytes | bytearray
        ):
            raise ValueError("source_cluster_statuses must be a non-string sequence")
        statuses = tuple(
            VegetationClusterStatus(
                _enum_value(value, VegetationClusterStatus, "source_cluster_statuses")
            )
            for value in self.source_cluster_statuses
        )
        readiness = tuple(bool(value) for value in self.source_cluster_distance_ready)
        if any(
            not isinstance(value, bool | np.bool_) for value in self.source_cluster_distance_ready
        ):
            raise ValueError("source_cluster_distance_ready must contain bool values")
        if len(statuses) != len(source_ids) or len(readiness) != len(source_ids):
            raise ValueError("source cluster metadata must align with source_cluster_ids")
        object.__setattr__(self, "source_cluster_statuses", statuses)
        object.__setattr__(self, "source_cluster_distance_ready", readiness)
        if not isinstance(self.tree_distance_ready, bool):
            raise ValueError("tree_distance_ready must be bool")
        object.__setattr__(self, "confidence", _finite(self.confidence, "confidence", 0.0))
        if self.confidence > 1.0:
            raise ValueError("confidence must not exceed one")
        if self.geometry_summary is not None and not isinstance(
            self.geometry_summary, TreeGeometrySummary
        ):
            raise ValueError("geometry_summary must be a TreeGeometrySummary or None")
        if self.crown_summary is not None and not isinstance(self.crown_summary, TreeCrownSummary):
            raise ValueError("crown_summary must be a TreeCrownSummary or None")
        if not isinstance(self.quality_flags, int | np.integer) or isinstance(
            self.quality_flags, bool
        ):
            raise ValueError("invalid tree quality flags")
        raw_flags = int(self.quality_flags)
        if raw_flags < 0 or raw_flags & ~_TREE_QUALITY_FLAG_MASK:
            raise ValueError("invalid tree quality flags")
        try:
            flags = TreeSegmentationQualityFlag(raw_flags)
        except ValueError as exc:
            raise ValueError("invalid tree quality flags") from exc
        object.__setattr__(self, "quality_flags", flags)
        confirmed = self.status is TreeSegmentationStatus.CONFIRMED_TREE
        if confirmed != (self.tree_id >= 0):
            raise ValueError("only confirmed trees may have a non-negative tree_id")
        if self.tree_distance_ready and not confirmed:
            raise ValueError("tree_distance_ready requires CONFIRMED_TREE")
        if confirmed and (
            self.geometry_summary is None
            or self.geometry_summary.maximum_hag_m is None
            or self.geometry_summary.robust_height_hag_m is None
        ):
            raise ValueError("confirmed trees require complete geometry with HAG values")
        if confirmed and (
            any(status is not VegetationClusterStatus.CONFIRMED for status in statuses)
            or not all(readiness)
        ):
            raise ValueError("confirmed trees require V2 confirmed and distance-ready sources")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "segment_id": self.segment_id,
            "status": int(self.status),
            "source_cluster_ids": list(self.source_cluster_ids),
            "source_cluster_statuses": [int(value) for value in self.source_cluster_statuses],
            "source_cluster_distance_ready": list(self.source_cluster_distance_ready),
            "tree_distance_ready": self.tree_distance_ready,
            "confidence": self.confidence,
            "geometry_summary": (
                None if self.geometry_summary is None else self.geometry_summary.to_dict()
            ),
            "crown_summary": (None if self.crown_summary is None else self.crown_summary.to_dict()),
            "quality_flags": int(self.quality_flags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeInstanceRecord:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values = dict(data)
        geometry_summary = values["geometry_summary"]
        if geometry_summary is not None:
            if not isinstance(geometry_summary, Mapping):
                raise ValueError("geometry_summary must be a mapping or null")
            values["geometry_summary"] = TreeGeometrySummary.from_dict(geometry_summary)
        crown_summary = values["crown_summary"]
        if crown_summary is not None:
            if not isinstance(crown_summary, Mapping):
                raise ValueError("crown_summary must be a mapping or null")
            values["crown_summary"] = TreeCrownSummary.from_dict(crown_summary)
        return cls(**values)


@dataclass(frozen=True)
class TreeSegmentationStatistics:
    input_point_count: int
    record_count: int
    confirmed_tree_count: int
    review_required_count: int
    provisional_count: int
    rejected_count: int
    unresolved_count: int
    elapsed_seconds: float
    phase_timings_seconds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "input_point_count",
            "record_count",
            "confirmed_tree_count",
            "review_required_count",
            "provisional_count",
            "rejected_count",
            "unresolved_count",
        ):
            object.__setattr__(self, name, _non_negative_int(getattr(self, name), name))
        if self.record_count != sum(
            (
                self.confirmed_tree_count,
                self.review_required_count,
                self.provisional_count,
                self.rejected_count,
                self.unresolved_count,
            )
        ):
            raise ValueError("status counts must sum to record_count")
        object.__setattr__(
            self,
            "elapsed_seconds",
            _finite(self.elapsed_seconds, "elapsed_seconds", 0.0),
        )
        if not isinstance(self.phase_timings_seconds, Mapping) or any(
            not isinstance(key, str)
            or not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            for key, value in self.phase_timings_seconds.items()
        ):
            raise ValueError("phase_timings_seconds must contain finite non-negative values")
        object.__setattr__(
            self,
            "phase_timings_seconds",
            _immutable_json_mapping(self.phase_timings_seconds, "phase_timings_seconds"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_point_count": self.input_point_count,
            "record_count": self.record_count,
            "confirmed_tree_count": self.confirmed_tree_count,
            "review_required_count": self.review_required_count,
            "provisional_count": self.provisional_count,
            "rejected_count": self.rejected_count,
            "unresolved_count": self.unresolved_count,
            "elapsed_seconds": self.elapsed_seconds,
            "phase_timings_seconds": _thaw_json_value(self.phase_timings_seconds),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeSegmentationStatistics:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class TreeSegmentationResult:
    """Canonical V3 point-aligned output contract.

    ``segment_id_by_point``, ``tree_id_by_point``, and
    ``tree_distance_ready_by_point`` always have one row for every V2 point
    row.  V3A1 intentionally has no member digest: a future artifact layer
    must use a documented cryptographic digest, never Python's
    process-randomized ``hash()``.
    """

    schema_version: str
    algorithm_version: str
    source_id: str
    manifest_source_id: str
    origin_xyz: tuple[float, float, float]
    config: TreeSegmentationConfig
    statistics: TreeSegmentationStatistics
    records: tuple[TreeInstanceRecord, ...]
    segment_id_by_point: Any
    tree_id_by_point: Any
    tree_distance_ready_by_point: Any
    tree_member_offsets: Any
    ordered_tree_member_row_index: Any
    input_provenance: Mapping[str, Any] = field(default_factory=dict)
    coordinate_context: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != TREE_SEGMENTATION_SCHEMA_VERSION:
            raise ValueError("unsupported V3 tree segmentation schema")
        if self.algorithm_version != TREE_SEGMENTATION_ALGORITHM_VERSION:
            raise ValueError("unsupported V3 tree segmentation algorithm version")
        if (
            not isinstance(self.source_id, str)
            or not self.source_id
            or not isinstance(self.manifest_source_id, str)
            or not self.manifest_source_id
        ):
            raise ValueError("source identifiers must be non-empty")
        object.__setattr__(self, "origin_xyz", _origin_xyz(self.origin_xyz))
        if not isinstance(self.config, TreeSegmentationConfig) or not isinstance(
            self.statistics, TreeSegmentationStatistics
        ):
            raise ValueError("invalid V3 result contracts")
        input_records = tuple(self.records)
        if any(not isinstance(record, TreeInstanceRecord) for record in input_records):
            raise ValueError("records must contain TreeInstanceRecord values")
        for record in input_records:
            if record.status is not TreeSegmentationStatus.CONFIRMED_TREE:
                continue
            if record.crown_summary is None:
                if not self.config.allow_trunk_only_confirmation:
                    raise ValueError("confirmed trees require a complete crown summary")
            elif record.crown_summary.crown_area_m2 == 0.0:
                raise ValueError("confirmed tree crown area must be positive")
        actual_status_counts = {
            TreeSegmentationStatus.CONFIRMED_TREE: 0,
            TreeSegmentationStatus.REVIEW_REQUIRED: 0,
            TreeSegmentationStatus.PROVISIONAL: 0,
            TreeSegmentationStatus.REJECTED: 0,
            TreeSegmentationStatus.UNRESOLVED: 0,
        }
        for record in input_records:
            actual_status_counts[record.status] += 1
        expected_statistics = {
            "record_count": len(input_records),
            "confirmed_tree_count": actual_status_counts[TreeSegmentationStatus.CONFIRMED_TREE],
            "review_required_count": actual_status_counts[TreeSegmentationStatus.REVIEW_REQUIRED],
            "provisional_count": actual_status_counts[TreeSegmentationStatus.PROVISIONAL],
            "rejected_count": actual_status_counts[TreeSegmentationStatus.REJECTED],
            "unresolved_count": actual_status_counts[TreeSegmentationStatus.UNRESOLVED],
        }
        for name, expected in expected_statistics.items():
            if getattr(self.statistics, name) != expected:
                raise ValueError(f"statistics.{name} does not match records")
        point_count = self.statistics.input_point_count
        point_segments = _array(
            self.segment_id_by_point,
            np.int32,
            (point_count,),
            "segment_id_by_point",
        )
        point_ids = _array(
            self.tree_id_by_point,
            np.int32,
            (point_count,),
            "tree_id_by_point",
        )
        point_ready = _array(
            self.tree_distance_ready_by_point,
            np.bool_,
            (point_count,),
            "tree_distance_ready_by_point",
        )
        offsets = _array(
            self.tree_member_offsets,
            np.int64,
            (len(input_records) + 1,),
            "tree_member_offsets",
        )
        members = _array(
            self.ordered_tree_member_row_index,
            np.int64,
            (None,),
            "ordered_tree_member_row_index",
        )
        record_count = len(input_records)
        member_count = len(members)
        if offsets[0] != 0 or offsets[-1] != member_count:
            raise ValueError("tree_member_offsets must be monotonic and span member rows")
        previous_offset = 0
        for chunk_start in _iter_chunks(len(offsets)):
            chunk = offsets[chunk_start : chunk_start + _TREE_RESULT_VALIDATION_CHUNK_SIZE]
            if np.any(chunk < 0) or np.any(chunk > member_count):
                raise ValueError("tree_member_offsets must be monotonic and span member rows")
            if len(chunk) and int(chunk[0]) < previous_offset:
                raise ValueError("tree_member_offsets must be monotonic and span member rows")
            if len(chunk) > 1 and np.any(chunk[1:] < chunk[:-1]):
                raise ValueError("tree_member_offsets must be monotonic and span member rows")
            if len(chunk):
                previous_offset = int(chunk[-1])

        assigned_point_count = 0
        for chunk_start in _iter_chunks(point_count):
            chunk = point_segments[chunk_start : chunk_start + _TREE_RESULT_VALIDATION_CHUNK_SIZE]
            if np.any(chunk < -1) or np.any(chunk >= record_count):
                raise ValueError("segment_id_by_point contains an unknown or invalid segment_id")
            assigned_point_count += int(np.count_nonzero(chunk >= 0))
        records_by_segment: list[TreeInstanceRecord | None] = [None] * record_count
        expected_tree_id_by_segment = np.full(record_count, -1, dtype=np.int32)
        expected_ready_by_segment = np.zeros(record_count, dtype=np.bool_)
        confirmed_count = actual_status_counts[TreeSegmentationStatus.CONFIRMED_TREE]
        confirmed_tree_ids = np.zeros(confirmed_count, dtype=np.bool_)
        block_lengths_by_segment = np.empty(record_count, dtype=np.int64)

        for input_index, record in enumerate(input_records):
            segment_id = record.segment_id
            if segment_id >= record_count:
                raise ValueError("segment_id values must start at zero and be continuous")
            if records_by_segment[segment_id] is not None:
                raise ValueError("record segment_id values must be unique")
            records_by_segment[segment_id] = record
            start = int(offsets[input_index])
            end = int(offsets[input_index + 1])
            block_lengths_by_segment[segment_id] = end - start
            previous_member: int | None = None
            for chunk_start in range(start, end, _TREE_RESULT_VALIDATION_CHUNK_SIZE):
                rows = members[
                    chunk_start : min(
                        end,
                        chunk_start + _TREE_RESULT_VALIDATION_CHUNK_SIZE,
                    )
                ]
                if np.any(rows < 0) or np.any(rows >= point_count):
                    raise ValueError("tree member index is outside V2 point-row range")
                if previous_member is not None and len(rows) and int(rows[0]) <= previous_member:
                    raise ValueError("member rows must be strictly increasing within a record")
                if len(rows) > 1 and np.any(rows[1:] <= rows[:-1]):
                    raise ValueError("member rows must be strictly increasing within a record")
                if len(rows) and np.any(point_segments[rows] != segment_id):
                    raise ValueError(
                        "segment member rows must exactly match segment_id_by_point; "
                        "retained segments must not share member rows"
                    )
                if len(rows):
                    previous_member = int(rows[-1])

            confirmed = record.status is TreeSegmentationStatus.CONFIRMED_TREE
            if confirmed:
                if end == start:
                    raise ValueError("confirmed trees must have member rows")
                tree_id = record.tree_id
                if tree_id < 0:
                    raise ValueError("confirmed trees require a non-negative tree_id")
                if tree_id >= confirmed_count or confirmed_tree_ids[tree_id]:
                    raise ValueError(
                        "confirmed tree_id values must start at zero and be continuous"
                    )
                confirmed_tree_ids[tree_id] = True
                expected_tree_id_by_segment[segment_id] = tree_id
                expected_ready_by_segment[segment_id] = record.tree_distance_ready
            elif record.tree_id != -1:
                raise ValueError("only confirmed trees may have a non-negative tree_id")
            elif record.tree_distance_ready:
                raise ValueError("tree_distance_ready requires CONFIRMED_TREE")

            if (
                confirmed
                and len(record.source_cluster_ids) > 1
                and not self.config.allow_cross_cluster_confirmation
            ):
                raise ValueError("cross-cluster confirmed trees are disabled by configuration")

        if any(record is None for record in records_by_segment):
            raise ValueError("segment_id values must start at zero and be continuous")
        if confirmed_count and not np.all(confirmed_tree_ids):
            raise ValueError("confirmed tree_id values must start at zero and be continuous")
        if member_count != assigned_point_count:
            raise ValueError(
                "segment member rows must exactly match segment_id_by_point assigned count"
            )

        for chunk_start in _iter_chunks(point_count):
            chunk_end = min(
                point_count,
                chunk_start + _TREE_RESULT_VALIDATION_CHUNK_SIZE,
            )
            segment_chunk = point_segments[chunk_start:chunk_end]
            tree_id_chunk = point_ids[chunk_start:chunk_end]
            ready_chunk = point_ready[chunk_start:chunk_end]
            unassigned = segment_chunk == -1
            if np.any(tree_id_chunk[unassigned] != -1):
                raise ValueError("unassigned points must not receive a formal tree_id")
            if np.any(ready_chunk[unassigned]):
                raise ValueError("tree_distance_ready_by_point requires a confirmed tree_id")
            assigned = ~unassigned
            assigned_segments = segment_chunk[assigned]
            actual_tree_ids = tree_id_chunk[assigned]
            expected_tree_ids = expected_tree_id_by_segment[assigned_segments]
            tree_id_mismatch = actual_tree_ids != expected_tree_ids
            if np.any(tree_id_mismatch & (expected_tree_ids == -1)):
                raise ValueError(
                    "non-confirmed member rows must not receive a formal tree assignment; "
                    "tree_distance_ready_by_point requires a confirmed tree_id"
                )
            if np.any(tree_id_mismatch):
                raise ValueError("confirmed member rows must exactly match tree_id_by_point")
            actual_ready = ready_chunk[assigned]
            expected_ready = expected_ready_by_segment[assigned_segments]
            readiness_mismatch = actual_ready != expected_ready
            if np.any(readiness_mismatch & (expected_tree_ids == -1)):
                raise ValueError(
                    "non-confirmed member rows must not receive a formal tree assignment; "
                    "tree_distance_ready_by_point requires a confirmed tree_id"
                )
            if np.any(readiness_mismatch):
                raise ValueError("confirmed point readiness must match tree_distance_ready")

        # Each strictly increasing block is unique, distinct segment labels prevent
        # cross-block duplicates, and equal dense/grouped counts prove exact membership.
        canonical_input = all(
            record.segment_id == index for index, record in enumerate(input_records)
        )
        if canonical_input:
            canonical_records = input_records
            canonical_offsets = offsets
            canonical_members = members
        else:
            canonical_records = tuple(record for record in records_by_segment if record is not None)
            canonical_offsets = np.empty(record_count + 1, dtype=np.int64)
            canonical_offsets[0] = 0
            canonical_offsets[1:] = block_lengths_by_segment
            np.cumsum(
                canonical_offsets[1:],
                dtype=np.int64,
                out=canonical_offsets[1:],
            )
            canonical_members = np.empty(member_count, dtype=np.int64)
            for input_index, record in enumerate(input_records):
                source_start = int(offsets[input_index])
                source_end = int(offsets[input_index + 1])
                destination_start = int(canonical_offsets[record.segment_id])
                destination_end = int(canonical_offsets[record.segment_id + 1])
                canonical_members[destination_start:destination_end] = members[
                    source_start:source_end
                ]
            canonical_offsets.setflags(write=False)
            canonical_members.setflags(write=False)

        object.__setattr__(self, "records", canonical_records)
        object.__setattr__(self, "segment_id_by_point", point_segments)
        object.__setattr__(self, "tree_id_by_point", point_ids)
        object.__setattr__(self, "tree_distance_ready_by_point", point_ready)
        object.__setattr__(self, "tree_member_offsets", canonical_offsets)
        object.__setattr__(self, "ordered_tree_member_row_index", canonical_members)
        object.__setattr__(
            self,
            "input_provenance",
            _immutable_json_mapping(self.input_provenance, "input_provenance"),
        )
        object.__setattr__(
            self,
            "coordinate_context",
            _immutable_json_mapping(self.coordinate_context, "coordinate_context"),
        )
        object.__setattr__(self, "warnings", _string_sequence(self.warnings, "warnings"))
        object.__setattr__(self, "limitations", _string_sequence(self.limitations, "limitations"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "source_id": self.source_id,
            "manifest_source_id": self.manifest_source_id,
            "origin_xyz": list(self.origin_xyz),
            "config": self.config.to_dict(),
            "statistics": self.statistics.to_dict(),
            "records": [record.to_dict() for record in self.records],
            "segment_id_by_point": self.segment_id_by_point.tolist(),
            "tree_id_by_point": self.tree_id_by_point.tolist(),
            "tree_distance_ready_by_point": self.tree_distance_ready_by_point.tolist(),
            "tree_member_offsets": self.tree_member_offsets.tolist(),
            "ordered_tree_member_row_index": self.ordered_tree_member_row_index.tolist(),
            "input_provenance": _thaw_json_value(self.input_provenance),
            "coordinate_context": _thaw_json_value(self.coordinate_context),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeSegmentationResult:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values = dict(data)
        values["config"] = TreeSegmentationConfig.from_dict(values["config"])
        values["statistics"] = TreeSegmentationStatistics.from_dict(values["statistics"])
        values["records"] = tuple(
            TreeInstanceRecord.from_dict(value) for value in values["records"]
        )
        values["segment_id_by_point"] = _json_integer_array(
            values["segment_id_by_point"], np.int32, "segment_id_by_point"
        )
        values["tree_id_by_point"] = _json_integer_array(
            values["tree_id_by_point"], np.int32, "tree_id_by_point"
        )
        values["tree_distance_ready_by_point"] = _json_bool_array(
            values["tree_distance_ready_by_point"], "tree_distance_ready_by_point"
        )
        values["tree_member_offsets"] = _json_integer_array(
            values["tree_member_offsets"], np.int64, "tree_member_offsets"
        )
        values["ordered_tree_member_row_index"] = _json_integer_array(
            values["ordered_tree_member_row_index"],
            np.int64,
            "ordered_tree_member_row_index",
        )
        return cls(**values)
