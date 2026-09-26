"""Portable contracts for V2 vegetation clusters and confirmed instances."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import Any

import numpy as np

from core.contracts._json import json_safe_mapping, reject_unknown_fields, require_fields


class VegetationClusterStatus(IntEnum):
    UNASSIGNED = 0
    CONFIRMED = 1
    REVIEW_REQUIRED = 2
    PROVISIONAL = 3
    REJECTED_AS_INSTANCE = 4


class VegetationPlausibilityLabel(IntEnum):
    MIXED_OR_INSUFFICIENT_EVIDENCE = 0
    VERTICALLY_DEVELOPED_CROWN_LIKE = 1
    LOW_SHRUB_LIKE = 2
    SHALLOW_HORIZONTAL_SUSPECT = 3
    PLANAR_STRUCTURE_SUSPECT = 4
    LINEAR_STRUCTURE_SUSPECT = 5
    FRAGMENTED_OR_UNRESOLVED = 6
    ISOLATED_SLENDER_AMBIGUOUS = 7
    COMPACT_LAYERED_SUSPECT = 8
    MIXED_MORPHOLOGY_REVIEW = 9


class LocalScaleState(IntEnum):
    LOCAL_SCALE_UNAVAILABLE = 0
    LOCAL_SCALE_VALID = 1
    LOCAL_SCALE_CLAMPED_LOW = 2
    LOCAL_SCALE_CLAMPED_HIGH = 3


class VegetationInstanceReasonFlag(IntFlag):
    NONE = 0
    UNSUPPORTED_CORE_FRAGMENT = 1
    LOW_AMBIGUOUS_MULTI_CLUSTER = 2
    LOW_NO_CORE_SUPPORT = 4
    CONFIRMED_WIRE_BLOCKED = 8
    UNRESOLVED_BRIDGE = 16
    MANUAL_STATUS_OVERRIDE = 32
    MANUAL_EXCLUDED = 64
    MANUAL_INCLUDED = 128
    PLANAR_AMBIGUITY = 256
    ISOLATED_SLENDER_AMBIGUITY = 512
    COMPACT_LAYERED_AMBIGUITY = 1024
    MIXED_MORPHOLOGY_CONFLICT = 2048
    INSUFFICIENT_POSITIVE_EVIDENCE = 4096
    UNRESOLVED_TOPOLOGY = 8192


class VegetationInstanceQualityFlag(IntFlag):
    NONE = 0
    LOCAL_SCALE_UNAVAILABLE = 1
    HAG_REDUCED_CONFIDENCE = 2
    GEOMETRY_DEGRADED = 4
    BRIDGE_WARNING = 8
    FRAGMENTED = 16
    GROUPING_SUPPORT_DEGRADED = 32
    MORPHOLOGY_DEGRADED = 64


class VegetationInstanceManualFlag(IntFlag):
    NONE = 0
    CONFIRMED = 1
    REJECTED = 2
    MERGED = 4
    SPLIT = 8
    INCLUDED = 16
    EXCLUDED = 32
    WIRE_BLOCKED = 64


_ACTIONS = {
    "confirm_cluster",
    "reject_cluster",
    "merge_clusters",
    "split_cluster_by_seeds",
    "split_cluster_by_boxes",
    "include_unassigned_candidates",
    "exclude_points_or_boxes",
}


def _array(value: Any, shape: tuple[int | None, ...], dtype: Any, name: str) -> None:
    if getattr(value, "dtype", None) != np.dtype(dtype) or getattr(value, "ndim", None) != len(
        shape
    ):
        raise ValueError(f"{name} has invalid dtype or rank")
    if any(
        expected is not None and expected != actual
        for expected, actual in zip(shape, value.shape, strict=True)
    ):
        raise ValueError(f"{name} has invalid shape")


def _finite(value: float, name: str, minimum: float | None = None) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and float(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _sha(value: str, name: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be lowercase SHA-256")


@dataclass(frozen=True)
class VegetationInstanceConfig:
    source_id: str = "scene_simple"
    local_scale_lower_m: float = 0.030
    local_scale_upper_m: float = 0.120
    local_scale_support_multiplier: float = 2.5
    local_scale_k: int = 8
    local_scale_min_valid_neighbors: int = 5
    core_neighbor_cap: int = 12
    core_radius_multiplier: float = 1.4
    core_radius_lower_m: float = 0.035
    core_radius_upper_m: float = 0.160
    vertical_scale_multiplier: float = 1.25
    low_search_radius_m: float = 0.180
    low_support_cap: int = 24
    low_radius_multiplier: float = 1.15
    min_core_points: int = 12
    min_high_points: int = 2
    min_medium_points: int = 8
    min_core_voxels: int = 3
    min_core_extent_m: float = 0.150
    voxel_lower_m: float = 0.080
    voxel_upper_m: float = 0.200
    fragmentation_review_threshold: float = 0.35
    confirmation_plausibility_threshold: float = 0.62
    artificial_suspicion_threshold: float = 0.55
    label_evidence_threshold: float = 0.55
    primary_neighbor_cap: int = 12
    expanded_neighbor_cap: int = 24
    neighbour_tie_epsilon_m: float = 1e-6
    secondary_edge_normalized_distance_max: float = 0.75
    secondary_edge_min_shared_neighbors: int = 3
    secondary_edge_max_per_point: int = 2
    grouping_min_median_degree: int = 2
    grouping_min_median_shared_neighbors: int = 2
    grouping_min_nonunavailable_fraction: float = 0.50
    grouping_min_scale_quality: float = 0.70
    grouping_support_threshold: float = 0.65
    bridge_normalized_length_threshold: float = 0.85
    bridge_min_shared_neighbors: int = 2
    bridge_max_split_depth: int = 3
    bridge_max_children_per_parent: int = 8
    bridge_vertical_gap_scale_multiplier: float = 2.0
    bridge_min_weak_slice_run: int = 1
    bridge_section_change_ratio: float = 1.60
    gap_weak_endpoint_degree_threshold: int = 2
    gap_weak_voxel_support_threshold: int = 2
    slice_count: int = 6
    slice_min_points: int = 8
    promotion_morphology_threshold: float = 0.70
    promotion_confidence_threshold: float = 0.70
    review_confidence_threshold: float = 0.45
    review_morphology_threshold: float = 0.40
    minimum_independent_positive_channels: int = 3
    promotion_min_crown_expansion_ratio: float = 1.15
    promotion_max_crown_contraction_ratio: float = 1.00
    record_memory: bool = True

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name in {"source_id", "record_memory"}:
                continue
            if name in {
                "local_scale_k",
                "local_scale_min_valid_neighbors",
                "core_neighbor_cap",
                "low_support_cap",
                "min_core_points",
                "min_high_points",
                "min_medium_points",
                "min_core_voxels",
                "primary_neighbor_cap",
                "expanded_neighbor_cap",
                "secondary_edge_min_shared_neighbors",
                "secondary_edge_max_per_point",
                "grouping_min_median_degree",
                "grouping_min_median_shared_neighbors",
                "bridge_min_shared_neighbors",
                "bridge_max_split_depth",
                "bridge_max_children_per_parent",
                "bridge_min_weak_slice_run",
                "gap_weak_endpoint_degree_threshold",
                "gap_weak_voxel_support_threshold",
                "slice_count",
                "slice_min_points",
                "minimum_independent_positive_channels",
            }:
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            else:
                _finite(value, name, 0.0)
        if not isinstance(self.record_memory, bool):
            raise ValueError("record_memory must be bool")
        if not (
            self.local_scale_lower_m < self.local_scale_upper_m
            and self.local_scale_k >= self.local_scale_min_valid_neighbors
        ):
            raise ValueError("local scale bounds are invalid")
        if not (
            self.core_radius_lower_m <= self.core_radius_upper_m
            and self.voxel_lower_m <= self.voxel_upper_m
        ):
            raise ValueError("radius or voxel bounds are invalid")
        for name in (
            "fragmentation_review_threshold",
            "confirmation_plausibility_threshold",
            "artificial_suspicion_threshold",
            "label_evidence_threshold",
            "grouping_min_nonunavailable_fraction",
            "grouping_min_scale_quality",
            "grouping_support_threshold",
            "secondary_edge_normalized_distance_max",
            "bridge_normalized_length_threshold",
            "promotion_morphology_threshold",
            "promotion_confidence_threshold",
            "review_confidence_threshold",
            "review_morphology_threshold",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must not exceed one")
        if self.expanded_neighbor_cap < self.primary_neighbor_cap:
            raise ValueError("expanded_neighbor_cap must be at least primary_neighbor_cap")

    @property
    def local_scale_support_radius_m(self) -> float:
        return self.local_scale_upper_m * self.local_scale_support_multiplier

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VegetationInstanceConfig:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class VegetationManualActionRecord:
    action: str
    cluster_ids: tuple[int, ...] = ()
    original_indices: tuple[int, ...] = ()
    boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = ()

    def __post_init__(self) -> None:
        if self.action not in _ACTIONS:
            raise ValueError("unsupported manual action")
        for name in ("cluster_ids", "original_indices"):
            values = tuple(int(v) for v in getattr(self, name))
            if len(values) != len(set(values)) or any(v < 0 for v in values):
                raise ValueError(f"{name} must contain unique non-negative values")
            object.__setattr__(self, name, values)
        boxes = []
        for lower, upper in self.boxes:
            lo, hi = tuple(map(float, lower)), tuple(map(float, upper))
            if (
                len(lo) != 3
                or len(hi) != 3
                or not all(math.isfinite(v) for v in (*lo, *hi))
                or any(a > b for a, b in zip(lo, hi, strict=True))
            ):
                raise ValueError("boxes must be finite inclusive XYZ boxes")
            boxes.append((lo, hi))
        object.__setattr__(self, "boxes", tuple(boxes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "cluster_ids": list(self.cluster_ids),
            "original_indices": list(self.original_indices),
            "boxes": [[list(a), list(b)] for a, b in self.boxes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VegetationManualActionRecord:
        require_fields(data, {"action", "cluster_ids", "original_indices", "boxes"}, cls.__name__)
        reject_unknown_fields(
            data, {"action", "cluster_ids", "original_indices", "boxes"}, cls.__name__
        )
        return cls(**dict(data))


@dataclass(frozen=True)
class VegetationManualLog:
    v1_npz_sha256: str = ""
    a2a_npz_sha256: str = ""
    a2b_npz_sha256: str = ""
    source_id: str = ""
    manifest_source_id: str = ""
    actions: tuple[VegetationManualActionRecord, ...] = ()

    def __post_init__(self) -> None:
        if not self.actions and not any(
            (
                self.v1_npz_sha256,
                self.a2a_npz_sha256,
                self.a2b_npz_sha256,
                self.source_id,
                self.manifest_source_id,
            )
        ):
            return
        for name in ("v1_npz_sha256", "a2a_npz_sha256", "a2b_npz_sha256"):
            _sha(getattr(self, name), name)
        if not self.source_id or not self.manifest_source_id:
            raise ValueError("manual log source identities are required")
        object.__setattr__(self, "actions", tuple(self.actions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "v1_npz_sha256": self.v1_npz_sha256,
            "a2a_npz_sha256": self.a2a_npz_sha256,
            "a2b_npz_sha256": self.a2b_npz_sha256,
            "source_id": self.source_id,
            "manifest_source_id": self.manifest_source_id,
            "actions": [a.to_dict() for a in self.actions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VegetationManualLog:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values = dict(data)
        values["actions"] = tuple(
            VegetationManualActionRecord.from_dict(v) for v in values["actions"]
        )
        return cls(**values)


@dataclass(frozen=True)
class VegetationInstancePointSet:
    original_index: Any
    sample_index: Any
    xyz_local: Any
    wire_id_by_point: Any
    vegetation_rule_score: Any
    vegetation_confidence: Any
    automatic_vegetation_level: Any
    vegetation_level: Any
    hag_state: Any
    local_scale_raw_m: Any
    local_scale_m: Any
    local_scale_state: Any
    cluster_id_by_point: Any
    vegetation_id_by_point: Any
    cluster_status_by_point: Any
    distance_ready_by_point: Any
    assignment_confidence: Any
    assignment_reason_flags: Any
    assignment_quality_flags: Any
    manual_provenance_flags: Any

    def __post_init__(self) -> None:
        count = len(self.original_index)
        _array(self.xyz_local, (count, 3), np.float32, "xyz_local")
        for name, dtype in (
            ("original_index", np.int64),
            ("sample_index", np.int64),
            ("wire_id_by_point", np.int32),
            ("automatic_vegetation_level", np.uint8),
            ("vegetation_level", np.uint8),
            ("hag_state", np.uint8),
            ("local_scale_state", np.uint8),
            ("cluster_id_by_point", np.int32),
            ("vegetation_id_by_point", np.int32),
            ("cluster_status_by_point", np.uint8),
            ("distance_ready_by_point", np.bool_),
            ("assignment_reason_flags", np.uint16),
            ("assignment_quality_flags", np.uint16),
            ("manual_provenance_flags", np.uint16),
        ):
            _array(getattr(self, name), (count,), dtype, name)
        for name in (
            "vegetation_rule_score",
            "vegetation_confidence",
            "local_scale_raw_m",
            "local_scale_m",
            "assignment_confidence",
        ):
            _array(getattr(self, name), (count,), np.float32, name)
        if (
            np.any(np.asarray(self.original_index) < 0)
            or len(np.unique(self.original_index)) != count
        ):
            raise ValueError("original_index must be unique non-negative")
        if np.any(np.asarray(self.sample_index) < 0) or len(np.unique(self.sample_index)) != count:
            raise ValueError("sample_index must be unique non-negative")
        if (
            not np.isfinite(self.xyz_local).all()
            or not np.isfinite(self.vegetation_rule_score).all()
            or not np.isfinite(self.vegetation_confidence).all()
        ):
            raise ValueError("coordinates, score and confidence must be finite")
        if np.any((self.vegetation_rule_score < 0) | (self.vegetation_rule_score > 1)) or np.any(
            (self.vegetation_confidence < 0) | (self.vegetation_confidence > 1)
        ):
            raise ValueError("score and confidence must be in [0, 1]")
        if not np.isin(
            self.cluster_status_by_point, [int(v) for v in VegetationClusterStatus]
        ).all():
            raise ValueError("invalid cluster status")
        if np.any(
            (self.vegetation_id_by_point >= 0)
            & (self.cluster_status_by_point != int(VegetationClusterStatus.CONFIRMED))
        ):
            raise ValueError("only confirmed points may have vegetation IDs")
        if np.any(self.distance_ready_by_point != (self.vegetation_id_by_point >= 0)):
            raise ValueError("distance readiness must equal formal vegetation assignment")


@dataclass(frozen=True)
class VegetationInstanceStatistics:
    input_point_count: int
    candidate_count: int
    core_point_count: int
    cluster_count: int
    confirmed_count: int
    review_count: int
    provisional_count: int
    rejected_count: int
    assigned_candidate_count: int
    unassigned_candidate_count: int
    elapsed_seconds: float
    rss_start_bytes: int | None = None
    rss_peak_bytes: int | None = None
    rss_end_bytes: int | None = None
    phase_timings_seconds: Mapping[str, float] = field(default_factory=dict)
    safe_split_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "phase_timings_seconds":
                if not isinstance(value, Mapping) or any(
                    not isinstance(key, str)
                    or not isinstance(seconds, int | float)
                    or isinstance(seconds, bool)
                    or not math.isfinite(seconds)
                    or seconds < 0
                    for key, seconds in value.items()
                ):
                    raise ValueError("phase_timings_seconds must be finite non-negative values")
                object.__setattr__(self, name, dict(value))
                continue
            if name == "safe_split_counts":
                if not isinstance(value, Mapping) or any(
                    not isinstance(key, str)
                    or not isinstance(count, int)
                    or isinstance(count, bool)
                    or count < 0
                    for key, count in value.items()
                ):
                    raise ValueError("safe_split_counts must be non-negative integer values")
                object.__setattr__(self, name, dict(value))
                continue
            if name == "elapsed_seconds":
                _finite(value, name, 0.0)
            elif value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VegetationInstanceStatistics:
        require_fields(
            data,
            set(cls.__dataclass_fields__)
            - {
                "rss_start_bytes",
                "rss_peak_bytes",
                "rss_end_bytes",
                "phase_timings_seconds",
                "safe_split_counts",
            },
            cls.__name__,
        )
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class VegetationInstanceResult:
    schema_version: str
    source_id: str
    manifest_source_id: str
    origin_xyz: tuple[float, float, float]
    output_npz_filename: str
    output_npz_sha256: str
    output_npz_bytes: int
    config: VegetationInstanceConfig
    manual_log: VegetationManualLog
    input_provenance: Mapping[str, Any]
    statistics: VegetationInstanceStatistics
    arrays: Mapping[str, Any]
    clusters: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...] = ()
    algorithm_version: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != "vegetation-instances-v2":
            raise ValueError("unsupported V2 schema version")
        if self.algorithm_version != "v2-generalized-morphology-safe-vegetation-instances-v2":
            raise ValueError("unsupported V2 algorithm version")
        if not all(
            (
                self.schema_version,
                self.source_id,
                self.manifest_source_id,
                self.output_npz_filename,
                self.algorithm_version,
                self.created_at,
            )
        ):
            raise ValueError("required result text is missing")
        if any(c in self.output_npz_filename for c in "/\\:"):
            raise ValueError("output NPZ must be sibling filename")
        _sha(self.output_npz_sha256, "output_npz_sha256")
        if not isinstance(self.output_npz_bytes, int) or self.output_npz_bytes < 0:
            raise ValueError("output_npz_bytes must be non-negative")
        origin = tuple(float(v) for v in self.origin_xyz)
        if len(origin) != 3 or not all(math.isfinite(v) for v in origin):
            raise ValueError("origin_xyz must be finite XYZ")
        object.__setattr__(self, "origin_xyz", origin)
        if (
            not isinstance(self.config, VegetationInstanceConfig)
            or not isinstance(self.manual_log, VegetationManualLog)
            or not isinstance(self.statistics, VegetationInstanceStatistics)
        ):
            raise ValueError("invalid V2 result contracts")
        object.__setattr__(
            self, "input_provenance", json_safe_mapping(self.input_provenance, "input_provenance")
        )
        object.__setattr__(self, "arrays", json_safe_mapping(self.arrays, "arrays"))
        object.__setattr__(
            self, "clusters", tuple(json_safe_mapping(v, "cluster") for v in self.clusters)
        )
        object.__setattr__(self, "warnings", tuple(str(v) for v in self.warnings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "manifest_source_id": self.manifest_source_id,
            "origin_xyz": list(self.origin_xyz),
            "output_npz_filename": self.output_npz_filename,
            "output_npz_sha256": self.output_npz_sha256,
            "output_npz_bytes": self.output_npz_bytes,
            "config": self.config.to_dict(),
            "manual_log": self.manual_log.to_dict(),
            "input_provenance": dict(self.input_provenance),
            "statistics": self.statistics.to_dict(),
            "arrays": dict(self.arrays),
            "clusters": [dict(v) for v in self.clusters],
            "warnings": list(self.warnings),
            "algorithm_version": self.algorithm_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VegetationInstanceResult:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values = dict(data)
        values["config"] = VegetationInstanceConfig.from_dict(values["config"])
        values["manual_log"] = VegetationManualLog.from_dict(values["manual_log"])
        values["statistics"] = VegetationInstanceStatistics.from_dict(values["statistics"])
        return cls(**values)
