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
    if getattr(value, "dtype", None) != np.dtype(dtype) or getattr(value, "ndim", None) != len(
        shape
    ):
        raise ValueError(f"{name} has invalid dtype or rank")
    if any(
        expected is not None and expected != actual
        for expected, actual in zip(shape, value.shape, strict=True)
    ):
        raise ValueError(f"{name} has invalid shape")
    if value.dtype == object:
        raise ValueError(f"{name} must not use object dtype")
    return np.asarray(value)


def _json_integer_array(value: Any, dtype: Any, name: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{name} must be a one-dimensional JSON sequence")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must contain only JSON integer values")
    return np.asarray(value, dtype=dtype)


def _json_bool_array(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{name} must be a one-dimensional JSON sequence")
    if any(not isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must contain only JSON bool values")
    return np.asarray(value, dtype=np.bool_)


def _origin_xyz(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError("origin_xyz must be a one-dimensional XYZ sequence")
    if len(value) != 3:
        raise ValueError("origin_xyz must contain exactly three values")
    origin = []
    for coordinate in value:
        if (
            not isinstance(coordinate, int | float | np.integer | np.floating)
            or isinstance(coordinate, bool | np.bool_)
        ):
            raise ValueError("origin_xyz must contain only finite numeric values")
        numeric = float(coordinate)
        if not math.isfinite(numeric):
            raise ValueError("origin_xyz must contain only finite numeric values")
        origin.append(numeric)
    return (origin[0], origin[1], origin[2])


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
class TreeInstanceRecord:
    """One V3 hypothesis; only ``CONFIRMED_TREE`` carries a formal tree ID."""

    tree_id: int
    segment_id: int
    status: TreeSegmentationStatus
    source_cluster_ids: tuple[int, ...]
    source_cluster_statuses: tuple[VegetationClusterStatus, ...]
    source_cluster_distance_ready: tuple[bool, ...]
    tree_distance_ready: bool
    confidence: float
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
            not isinstance(value, bool | np.bool_)
            for value in self.source_cluster_distance_ready
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
            "quality_flags": int(self.quality_flags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeInstanceRecord:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


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
        object.__setattr__(self, "phase_timings_seconds", dict(self.phase_timings_seconds))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

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
        records = tuple(self.records)
        if any(not isinstance(record, TreeInstanceRecord) for record in records):
            raise ValueError("records must contain TreeInstanceRecord values")
        actual_status_counts = {
            TreeSegmentationStatus.CONFIRMED_TREE: 0,
            TreeSegmentationStatus.REVIEW_REQUIRED: 0,
            TreeSegmentationStatus.PROVISIONAL: 0,
            TreeSegmentationStatus.REJECTED: 0,
            TreeSegmentationStatus.UNRESOLVED: 0,
        }
        for record in records:
            actual_status_counts[record.status] += 1
        expected_statistics = {
            "record_count": len(records),
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
            (len(records) + 1,),
            "tree_member_offsets",
        )
        members = _array(
            self.ordered_tree_member_row_index,
            np.int64,
            (None,),
            "ordered_tree_member_row_index",
        )
        if (
            offsets[0] != 0
            or offsets[-1] != len(members)
            or np.any(np.diff(offsets) < 0)
        ):
            raise ValueError("tree_member_offsets must be monotonic and span member rows")
        if np.any(members < 0) or np.any(members >= len(point_segments)):
            raise ValueError("tree member index is outside V2 point-row range")
        if len(np.unique(members)) != len(members):
            raise ValueError("retained segments must not share member rows")
        segment_ids = [record.segment_id for record in records]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("record segment_id values must be unique")
        if sorted(segment_ids) != list(range(len(segment_ids))):
            raise ValueError("segment_id values must start at zero and be continuous")
        allowed_segment_ids = {-1, *segment_ids}
        if any(int(value) not in allowed_segment_ids for value in point_segments):
            raise ValueError(
                "segment_id_by_point contains an unknown or invalid segment_id"
            )
        confirmed_ids = [
            record.tree_id
            for record in records
            if record.status is TreeSegmentationStatus.CONFIRMED_TREE
        ]
        if sorted(confirmed_ids) != list(range(len(confirmed_ids))):
            raise ValueError("confirmed tree_id values must start at zero and be continuous")
        allowed_ids = {-1, *confirmed_ids}
        if any(int(value) not in allowed_ids for value in point_ids):
            raise ValueError("tree_id_by_point contains an unknown or invalid tree_id")
        if np.any(point_ready & (point_ids < 0)):
            raise ValueError("tree_distance_ready_by_point requires a confirmed tree_id")
        entries: list[tuple[TreeInstanceRecord, np.ndarray]] = []
        retained_member_rows: set[int] = set()
        for index, record in enumerate(records):
            start, end = offsets[index : index + 2]
            rows = members[start:end]
            if len(rows) > 1 and np.any(np.diff(rows) <= 0):
                raise ValueError("member rows must be strictly increasing within a record")
            overlap = retained_member_rows.intersection(int(value) for value in rows)
            if overlap:
                raise ValueError("retained segments must not share member rows")
            retained_member_rows.update(int(value) for value in rows)
            if not np.array_equal(
                np.flatnonzero(point_segments == record.segment_id),
                rows,
            ):
                raise ValueError(
                    "segment member rows must exactly match segment_id_by_point"
                )
            if record.status is TreeSegmentationStatus.CONFIRMED_TREE:
                if not len(rows):
                    raise ValueError("confirmed trees must have member rows")
                if not np.array_equal(np.flatnonzero(point_ids == record.tree_id), rows):
                    raise ValueError("confirmed member rows must exactly match tree_id_by_point")
                if not np.array_equal(
                    point_ready[rows],
                    np.full(len(rows), record.tree_distance_ready, dtype=np.bool_),
                ):
                    raise ValueError("confirmed point readiness must match tree_distance_ready")
            elif len(rows) and (
                np.any(point_ids[rows] != -1) or np.any(point_ready[rows])
            ):
                raise ValueError(
                    "non-confirmed member rows must not receive a formal tree assignment"
                )
            entries.append((record, rows))
        for record in records:
            if (
                record.status is TreeSegmentationStatus.CONFIRMED_TREE
                and len(record.source_cluster_ids) > 1
                and not self.config.allow_cross_cluster_confirmation
            ):
                raise ValueError("cross-cluster confirmed trees are disabled by configuration")
        entries.sort(key=lambda item: self._record_sort_key(item[0], item[1]))
        canonical_offsets = np.zeros(len(entries) + 1, dtype=np.int64)
        canonical_offsets[1:] = np.cumsum(
            [len(rows) for _record, rows in entries], dtype=np.int64
        )
        canonical_members = (
            np.concatenate([rows for _record, rows in entries]).astype(np.int64, copy=False)
            if entries
            else np.empty(0, dtype=np.int64)
        )
        object.__setattr__(self, "records", tuple(record for record, _rows in entries))
        object.__setattr__(self, "tree_member_offsets", canonical_offsets)
        object.__setattr__(self, "ordered_tree_member_row_index", canonical_members)
        object.__setattr__(
            self,
            "input_provenance",
            json_safe_mapping(self.input_provenance, "input_provenance"),
        )
        object.__setattr__(
            self,
            "coordinate_context",
            json_safe_mapping(self.coordinate_context, "coordinate_context"),
        )
        object.__setattr__(self, "warnings", _string_sequence(self.warnings, "warnings"))
        object.__setattr__(self, "limitations", _string_sequence(self.limitations, "limitations"))

    @staticmethod
    def _record_sort_key(record: TreeInstanceRecord, rows: np.ndarray) -> tuple[Any, ...]:
        return (
            record.segment_id,
            record.tree_id,
            int(record.status),
            record.source_cluster_ids,
            tuple(int(value) for value in record.source_cluster_statuses),
            record.source_cluster_distance_ready,
            tuple(int(value) for value in rows),
            record.tree_distance_ready,
            record.confidence,
            int(record.quality_flags),
        )

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
            "input_provenance": dict(self.input_provenance),
            "coordinate_context": dict(self.coordinate_context),
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
