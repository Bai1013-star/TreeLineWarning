"""Contracts for A4 deterministic wire-instance separation and association."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import Any

import numpy as np

from core.contracts._json import json_safe_mapping, reject_unknown_fields, require_fields


class WireTrackStatus(IntEnum):
    PROVISIONAL = 0
    CONFIRMED = 1
    REVIEW_REQUIRED = 2


class WireTrackSource(IntEnum):
    AUTOMATIC = 0
    MANUAL_SEED = 1
    HYBRID = 2


class WireConfirmationChannel(IntEnum):
    NONE = 0
    OBVIOUS = 1
    FINE = 2


class WireTrackQualityFlag(IntFlag):
    NONE = 0
    PROVISIONAL_SHORT_TRACK = 1
    LOW_COVERAGE = 2
    LARGE_GAP = 4
    LOW_DIRECTION_CONSISTENCY = 8
    HIGH_TRANSVERSE_RESIDUAL = 16
    HIGH_CURVATURE = 32
    BRANCH_CONFLICT = 64
    LOW_MEAN_SCORE = 128
    OFF_DOMINANT_AXIS = 256
    MANUAL_REVIEW_REQUIRED = 512
    NUMERIC_INVALID = 1024
    MANUAL_SEEDED = 2048


class WireAssignmentReasonFlag(IntFlag):
    NONE = 0
    HIGH_CONFIDENCE_SEED = 1
    MEDIUM_CONFIDENCE_SEED = 2
    TRACK_EXTENSION = 4
    LOW_CONFIDENCE_GAP_FILL = 8
    HUNGARIAN_MATCH = 16
    MANUAL_SEED = 32
    MANUAL_EXCLUDE = 64
    BIDIRECTIONAL_EXTENSION = 128
    PROVISIONAL_TRACK = 256
    UNASSIGNED_OUTLIER = 512


class WireAssignmentQualityFlag(IntFlag):
    NONE = 0
    TRACK_PROVISIONAL = 1
    LOW_ASSIGNMENT_SCORE = 2
    LARGE_LOCAL_RESIDUAL = 4
    DIRECTION_MISMATCH = 8
    GAP_BRIDGED = 16
    BRANCH_AMBIGUITY = 32
    MANUAL_OVERRIDE = 64
    INVALID_INPUT = 128


def _shape(value: Any, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(item) for item in shape)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return (len(value),)
    raise ValueError(f"{name} must expose shape")


def _array(value: Any, shape: tuple[int, ...], dtype: np.dtype[Any], name: str) -> None:
    if _shape(value, name) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if getattr(value, "dtype", None) is None or np.dtype(value.dtype) != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")


def _finite(value: float, name: str, *, minimum: float | None = None) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _positive(value: float, name: str) -> float:
    return _finite(value, name, minimum=np.nextafter(0.0, 1.0))


def _non_negative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _metadata(value: Any, unit: str) -> dict[str, Any]:
    return {"shape": list(_shape(value, "array")), "dtype": str(value.dtype), "unit": unit}


@dataclass(frozen=True)
class WireSeedConfig:
    """Minimal deterministic manual seed/exclusion inputs for A4."""

    seed_original_indices: tuple[int, ...] = ()
    seed_group_ids: tuple[int, ...] = ()
    local_xyz_seed_boxes: tuple[
        tuple[int, tuple[float, float, float], tuple[float, float, float]], ...
    ] = ()
    excluded_original_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.seed_original_indices) != len(self.seed_group_ids):
            raise ValueError("seed_original_indices and seed_group_ids must have equal length")
        for name in ("seed_original_indices", "excluded_original_indices", "seed_group_ids"):
            values = tuple(int(value) for value in getattr(self, name))
            if any(value < 0 for value in values):
                raise ValueError(f"{name} must contain non-negative integers")
            if name != "seed_group_ids" and len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique values")
            object.__setattr__(self, name, values)
        boxes = []
        for group_id, lower, upper in self.local_xyz_seed_boxes:
            lower = tuple(float(value) for value in lower)
            upper = tuple(float(value) for value in upper)
            if (
                int(group_id) < 0
                or len(lower) != 3
                or len(upper) != 3
                or not all(math.isfinite(value) for value in (*lower, *upper))
                or any(low > high for low, high in zip(lower, upper, strict=True))
            ):
                raise ValueError("local_xyz_seed_boxes must contain finite inclusive XYZ boxes")
            boxes.append((int(group_id), lower, upper))
        object.__setattr__(self, "local_xyz_seed_boxes", tuple(boxes))

    @property
    def has_seeds(self) -> bool:
        return bool(self.seed_original_indices or self.local_xyz_seed_boxes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_original_indices": list(self.seed_original_indices),
            "seed_group_ids": list(self.seed_group_ids),
            "local_xyz_seed_boxes": [
                [group_id, list(lower), list(upper)]
                for group_id, lower, upper in self.local_xyz_seed_boxes
            ],
            "excluded_original_indices": list(self.excluded_original_indices),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WireSeedConfig:
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class WireSeparationConfig:
    """All A4 axis, slicing, association, and quality thresholds."""

    source_id: str = "scene_simple"
    expected_input_point_count: int = 146_126
    high_score_threshold: float = 0.80
    robust_axis_iterations: int = 3
    robust_axis_mad_multiplier: float = 4.0
    robust_axis_direction_cosine: float = 0.50
    min_axis_seed_points: int = 20
    min_axis_span_m: float = 0.60
    slice_length_m: float = 0.20
    slice_overlap_m: float = 0.10
    transverse_eps_m: float = 0.16
    vertical_eps_m: float = 0.20
    max_cluster_diameter_m: float = 0.45
    weak_cluster_min_points: int = 2
    confirmed_cluster_min_points: int = 3
    max_transverse_distance_m: float = 0.30
    max_height_distance_m: float = 0.35
    max_prediction_distance_m: float = 0.40
    max_association_cost: float = 1.00
    max_gap_slices: int = 4
    weight_transverse: float = 0.30
    weight_height: float = 0.15
    weight_prediction: float = 0.20
    weight_direction: float = 0.10
    weight_score: float = 0.10
    weight_gap: float = 0.15
    low_max_transverse_distance_m: float = 0.20
    low_max_height_distance_m: float = 0.25
    low_max_prediction_distance_m: float = 0.30
    low_min_direction_cosine: float = 0.80
    medium_seed_min_consecutive_slices: int = 7
    medium_seed_min_span_m: float = 0.60
    confirmation_min_span_m: float = 1.50
    confirmation_min_occupied_slices: int = 8
    confirmation_min_coverage_ratio: float = 0.35
    confirmation_min_direction_consistency: float = 0.80
    confirmation_min_mean_score: float = 0.55
    low_assignment_score_threshold: float = 0.55
    high_transverse_residual_m: float = 0.20
    high_curvature_p90_rad: float = 0.35
    branch_cost_closeness: float = 0.08
    obvious_min_span_m: float = 10.0
    obvious_min_coverage_ratio: float = 0.80
    obvious_min_direction_consistency: float = 0.99
    obvious_max_transverse_residual_m: float = 0.04
    obvious_max_curvature_proxy_rad: float = 0.45
    obvious_max_gap_m: float = 0.20
    obvious_max_axis_angle_deg: float = 2.0
    obvious_min_mean_candidate_score: float = 0.90
    obvious_min_high_candidate_fraction: float = 0.95
    obvious_max_branch_conflicts: int = 0
    fine_min_span_m: float = 6.0
    fine_min_coverage_ratio: float = 0.55
    fine_min_direction_consistency: float = 0.99
    fine_max_transverse_residual_m: float = 0.04
    fine_max_curvature_proxy_rad: float = 1.05
    fine_max_gap_m: float = 0.20
    fine_max_axis_angle_deg: float = 2.0
    fine_min_mean_candidate_score: float = 0.63
    fine_min_medium_low_fraction: float = 0.40
    fine_max_branch_conflicts: int = 0
    preview_background_limit: int = 40_000
    preview_candidate_limit: int = 50_000
    record_memory: bool = True

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        for name in self.__dataclass_fields__:
            if name in {"source_id", "record_memory"}:
                continue
            value = getattr(self, name)
            if name in {"obvious_max_branch_conflicts", "fine_max_branch_conflicts"}:
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"{name} must be a non-negative integer")
            elif isinstance(value, int) and not isinstance(value, bool):
                if value <= 0:
                    raise ValueError(f"{name} must be positive")
            else:
                object.__setattr__(self, name, _positive(value, name))
        if not isinstance(self.record_memory, bool):
            raise ValueError("record_memory must be bool")
        if not (
            0.0 < self.high_score_threshold <= 1.0
            and 0.0 < self.robust_axis_direction_cosine <= 1.0
            and self.slice_overlap_m < self.slice_length_m
            and self.weak_cluster_min_points <= self.confirmed_cluster_min_points
            and 0.0 < self.low_min_direction_cosine <= 1.0
            and 0.0 < self.confirmation_min_coverage_ratio <= 1.0
            and 0.0 < self.confirmation_min_direction_consistency <= 1.0
            and 0.0 < self.confirmation_min_mean_score <= 1.0
            and 0.0 < self.low_assignment_score_threshold <= 1.0
            and 0.0 < self.obvious_min_coverage_ratio <= 1.0
            and 0.0 < self.obvious_min_direction_consistency <= 1.0
            and 0.0 < self.obvious_min_mean_candidate_score <= 1.0
            and 0.0 < self.obvious_min_high_candidate_fraction <= 1.0
            and 0.0 < self.fine_min_coverage_ratio <= 1.0
            and 0.0 < self.fine_min_direction_consistency <= 1.0
            and 0.0 < self.fine_min_mean_candidate_score <= 1.0
            and 0.0 < self.fine_min_medium_low_fraction <= 1.0
        ):
            raise ValueError("WireSeparationConfig threshold ordering is invalid")
        if not math.isclose(
            sum(
                (
                    self.weight_transverse,
                    self.weight_height,
                    self.weight_prediction,
                    self.weight_direction,
                    self.weight_score,
                    self.weight_gap,
                )
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("association weights must sum to 1")

    @property
    def slice_step_m(self) -> float:
        return self.slice_length_m - self.slice_overlap_m

    def to_dict(self) -> dict[str, Any]:
        data = {name: getattr(self, name) for name in self.__dataclass_fields__}
        data["slice_step_m"] = self.slice_step_m
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WireSeparationConfig:
        allowed = set(cls.__dataclass_fields__) | {"slice_step_m"}
        reject_unknown_fields(data, allowed, cls.__name__)
        values = dict(data)
        values.pop("slice_step_m", None)
        return cls(**values)


@dataclass(frozen=True)
class WireAssignmentSet:
    """Full A3-order track and confirmed-wire assignment arrays."""

    source_id: str
    original_index: Any
    sample_index: Any
    xyz_local: Any
    track_id_by_point: Any
    wire_id_by_point: Any
    assignment_score: Any
    assignment_reason_flags: Any
    assignment_quality_flags: Any

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        point_count, dims = _shape(self.xyz_local, "xyz_local")
        if dims != 3:
            raise ValueError("xyz_local must have shape (M, 3)")
        _array(self.xyz_local, (point_count, 3), np.dtype(np.float32), "xyz_local")
        for name in ("original_index", "sample_index"):
            value = getattr(self, name)
            _array(value, (point_count,), np.dtype(np.int64), name)
            if np.any(np.asarray(value) < 0) or len(np.unique(value)) != point_count:
                raise ValueError(f"{name} must be unique and non-negative")
        for name in ("track_id_by_point", "wire_id_by_point"):
            _array(getattr(self, name), (point_count,), np.dtype(np.int32), name)
        _array(self.assignment_score, (point_count,), np.dtype(np.float32), "assignment_score")
        for name in ("assignment_reason_flags", "assignment_quality_flags"):
            _array(getattr(self, name), (point_count,), np.dtype(np.uint16), name)
        if (
            not np.isfinite(np.asarray(self.xyz_local)).all()
            or not np.isfinite(np.asarray(self.assignment_score)).all()
        ):
            raise ValueError("assignment coordinates and scores must be finite")
        score = np.asarray(self.assignment_score)
        if not np.all((score >= 0.0) & (score <= 1.0)):
            raise ValueError("assignment_score must be in [0, 1]")
        track_id = np.asarray(self.track_id_by_point)
        wire_id = np.asarray(self.wire_id_by_point)
        if np.any(track_id < -1) or np.any(wire_id < -1):
            raise ValueError("track and wire IDs must be -1 or non-negative")
        if np.any((wire_id >= 0) & (track_id < 0)):
            raise ValueError("assigned wires must also have a track ID")

    @property
    def point_count(self) -> int:
        return _shape(self.track_id_by_point, "track_id_by_point")[0]

    def array_metadata(self) -> dict[str, Any]:
        values = {
            "original_index": self.original_index,
            "sample_index": self.sample_index,
            "xyz_local": self.xyz_local,
            "track_id_by_point": self.track_id_by_point,
            "wire_id_by_point": self.wire_id_by_point,
            "assignment_score": self.assignment_score,
            "assignment_reason_flags": self.assignment_reason_flags,
            "assignment_quality_flags": self.assignment_quality_flags,
        }
        units = {
            "original_index": "index",
            "sample_index": "index",
            "xyz_local": "m",
            "track_id_by_point": "id",
            "wire_id_by_point": "id",
            "assignment_score": "unitless",
            "assignment_reason_flags": "bitmask",
            "assignment_quality_flags": "bitmask",
        }
        return {name: _metadata(value, units[name]) for name, value in values.items()}


@dataclass(frozen=True)
class WireTrackSummary:
    """Per-track diagnostics, with a wire ID only after final confirmation."""

    track_id: int
    wire_id: int
    status: WireTrackStatus
    source: WireTrackSource
    point_count: int
    longitudinal_span_m: float
    occupied_slice_count: int
    coverage_ratio: float
    mean_candidate_score: float
    high_candidate_fraction: float
    medium_candidate_fraction: float
    low_candidate_fraction: float
    direction_consistency: float
    transverse_residual: float
    curvature_proxy: float
    maximum_gap_m: float
    branch_conflict_count: int
    median_u: float
    median_v: float
    first_s: float
    actual_main_direction: tuple[float, float, float]
    axis_angle_deg: float
    confirmation_channel: WireConfirmationChannel
    start_xyz: tuple[float, float, float]
    end_xyz: tuple[float, float, float]
    quality_flags: int

    def __post_init__(self) -> None:
        _non_negative_int(self.track_id, "track_id")
        if not isinstance(self.wire_id, int) or isinstance(self.wire_id, bool) or self.wire_id < -1:
            raise ValueError("wire_id must be -1 or a non-negative integer")
        if not isinstance(self.status, WireTrackStatus) or not isinstance(
            self.source, WireTrackSource
        ):
            raise ValueError("status and source must use A4 enums")
        if not isinstance(self.confirmation_channel, WireConfirmationChannel):
            raise ValueError("confirmation_channel must use the A4 enum")
        if (self.status == WireTrackStatus.CONFIRMED) != (self.wire_id >= 0):
            raise ValueError("only confirmed tracks may have a wire_id")
        for name in (
            "point_count",
            "occupied_slice_count",
            "branch_conflict_count",
            "quality_flags",
        ):
            _non_negative_int(getattr(self, name), name)
        for name in (
            "longitudinal_span_m",
            "coverage_ratio",
            "mean_candidate_score",
            "high_candidate_fraction",
            "medium_candidate_fraction",
            "low_candidate_fraction",
            "direction_consistency",
            "transverse_residual",
            "curvature_proxy",
            "maximum_gap_m",
            "axis_angle_deg",
        ):
            _finite(getattr(self, name), name, minimum=0.0)
        for name in (
            "coverage_ratio",
            "mean_candidate_score",
            "high_candidate_fraction",
            "medium_candidate_fraction",
            "low_candidate_fraction",
            "direction_consistency",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("median_u", "median_v", "first_s"):
            _finite(getattr(self, name), name)
        direction = tuple(float(value) for value in self.actual_main_direction)
        if len(direction) != 3 or not all(math.isfinite(value) for value in direction):
            raise ValueError("actual_main_direction must contain three finite values")
        norm = math.sqrt(sum(value * value for value in direction))
        invalid = bool(self.quality_flags & int(WireTrackQualityFlag.NUMERIC_INVALID))
        valid_direction = math.isclose(norm, 1.0, abs_tol=1e-6)
        invalid_sentinel = invalid and math.isclose(norm, 0.0, abs_tol=1e-12)
        if not (valid_direction or invalid_sentinel):
            raise ValueError(
                "actual_main_direction must be unit length or zero when numeric-invalid"
            )
        object.__setattr__(self, "actual_main_direction", direction)
        for name in ("start_xyz", "end_xyz"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, Any]:
        data = {name: getattr(self, name) for name in self.__dataclass_fields__}
        data["status"] = int(self.status)
        data["source"] = int(self.source)
        data["confirmation_channel"] = int(self.confirmation_channel)
        for name in ("actual_main_direction", "start_xyz", "end_xyz"):
            data[name] = list(data[name])
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WireTrackSummary:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values = dict(data)
        values["status"] = WireTrackStatus(values["status"])
        values["source"] = WireTrackSource(values["source"])
        values["confirmation_channel"] = WireConfirmationChannel(values["confirmation_channel"])
        return cls(**values)


@dataclass(frozen=True)
class WireInstanceSet:
    """Flattened row positions for all tracks and for confirmed wires only."""

    ordered_track_point_indices: Any
    track_offsets: Any
    ordered_wire_point_indices: Any
    wire_offsets: Any
    summaries: tuple[WireTrackSummary, ...]

    def __post_init__(self) -> None:
        track_count = len(self.summaries)
        _array(
            self.ordered_track_point_indices,
            (_shape(self.ordered_track_point_indices, "ordered_track_point_indices")[0],),
            np.dtype(np.int64),
            "ordered_track_point_indices",
        )
        _array(self.track_offsets, (track_count + 1,), np.dtype(np.int64), "track_offsets")
        _array(
            self.ordered_wire_point_indices,
            (_shape(self.ordered_wire_point_indices, "ordered_wire_point_indices")[0],),
            np.dtype(np.int64),
            "ordered_wire_point_indices",
        )
        wire_count = sum(summary.status == WireTrackStatus.CONFIRMED for summary in self.summaries)
        _array(self.wire_offsets, (wire_count + 1,), np.dtype(np.int64), "wire_offsets")
        offsets = np.asarray(self.track_offsets)
        if (
            offsets[0] != 0
            or offsets[-1] != len(self.ordered_track_point_indices)
            or np.any(np.diff(offsets) < 0)
        ):
            raise ValueError("track_offsets must monotonically span ordered_track_point_indices")
        offsets = np.asarray(self.wire_offsets)
        if (
            offsets[0] != 0
            or offsets[-1] != len(self.ordered_wire_point_indices)
            or np.any(np.diff(offsets) < 0)
        ):
            raise ValueError("wire_offsets must monotonically span ordered_wire_point_indices")
        if [summary.track_id for summary in self.summaries] != list(range(track_count)):
            raise ValueError("summaries must use continuous track IDs")
        wire_ids = [summary.wire_id for summary in self.summaries if summary.wire_id >= 0]
        if wire_ids != list(range(wire_count)):
            raise ValueError("confirmed summaries must use continuous wire IDs")

    @property
    def track_count(self) -> int:
        return len(self.summaries)

    @property
    def wire_count(self) -> int:
        return sum(summary.wire_id >= 0 for summary in self.summaries)


@dataclass(frozen=True)
class WireSeparationStatistics:
    input_point_count: int
    candidate_point_count: int
    primary_slice_count: int
    slice_cluster_count: int
    initial_provisional_track_count: int
    total_track_count: int
    confirmed_wire_count: int
    provisional_track_count: int
    review_required_track_count: int
    assigned_track_point_count: int
    assigned_wire_point_count: int
    unassigned_candidate_count: int
    duplicate_assignment_count: int
    overlap_duplicate_assignment_count: int
    elapsed_seconds: float
    rss_start_bytes: int | None = None
    rss_peak_bytes: int | None = None
    rss_end_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name == "elapsed_seconds":
                continue
            value = getattr(self, name)
            if value is not None:
                _non_negative_int(value, name)
        _finite(self.elapsed_seconds, "elapsed_seconds", minimum=0.0)
        if (
            self.total_track_count
            != self.confirmed_wire_count
            + self.provisional_track_count
            + self.review_required_track_count
        ):
            raise ValueError("track status counts must equal wire_count")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WireSeparationStatistics:
        required = set(cls.__dataclass_fields__) - {
            "rss_start_bytes",
            "rss_peak_bytes",
            "rss_end_bytes",
        }
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class WireSeparationResult:
    source_id: str
    manifest_source_id: str
    input_a3_npz_path: Path
    input_a3_json_path: Path
    input_a3_npz_sha256: str
    input_a3_json_sha256: str
    output_npz_path: Path
    output_json_path: Path
    output_npz_sha256: str
    output_npz_bytes: int
    origin_xyz: tuple[float, float, float]
    center_xyz: tuple[float, float, float]
    s_axis: tuple[float, float, float]
    u_axis: tuple[float, float, float]
    v_axis: tuple[float, float, float]
    config: WireSeparationConfig
    seeds: WireSeedConfig
    statistics: WireSeparationStatistics
    arrays: Mapping[str, Any]
    summaries: tuple[WireTrackSummary, ...]
    assignment_reason_counts: Mapping[str, Any]
    assignment_quality_counts: Mapping[str, Any]
    track_quality_counts: Mapping[str, Any]
    confirmation_transition_counts: Mapping[str, Any]
    formulas: Mapping[str, Any]
    warnings: tuple[str, ...]
    algorithm_version: str
    created_at: str

    def __post_init__(self) -> None:
        if not self.source_id or not self.manifest_source_id:
            raise ValueError("source identifiers must be non-empty")
        for name in (
            "input_a3_npz_path",
            "input_a3_json_path",
            "output_npz_path",
            "output_json_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in ("input_a3_npz_sha256", "input_a3_json_sha256", "output_npz_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        _non_negative_int(self.output_npz_bytes, "output_npz_bytes")
        for name in ("origin_xyz", "center_xyz", "s_axis", "u_axis", "v_axis"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain three finite coordinates")
            object.__setattr__(self, name, values)
        if not isinstance(self.config, WireSeparationConfig) or not isinstance(
            self.seeds, WireSeedConfig
        ):
            raise ValueError("config and seeds must use A4 contracts")
        if not isinstance(self.statistics, WireSeparationStatistics):
            raise ValueError("statistics must use WireSeparationStatistics")
        if not self.algorithm_version or not self.created_at:
            raise ValueError("algorithm_version and created_at must be non-empty")
        for name in (
            "arrays",
            "assignment_reason_counts",
            "assignment_quality_counts",
            "track_quality_counts",
            "confirmation_transition_counts",
            "formulas",
        ):
            object.__setattr__(self, name, json_safe_mapping(getattr(self, name), name))
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "manifest_source_id": self.manifest_source_id,
            "input_a3_npz_path": str(self.input_a3_npz_path),
            "input_a3_json_path": str(self.input_a3_json_path),
            "input_a3_npz_sha256": self.input_a3_npz_sha256,
            "input_a3_json_sha256": self.input_a3_json_sha256,
            "output_npz_path": str(self.output_npz_path),
            "output_json_path": str(self.output_json_path),
            "output_npz_sha256": self.output_npz_sha256,
            "output_npz_bytes": self.output_npz_bytes,
            "origin_xyz": list(self.origin_xyz),
            "center_xyz": list(self.center_xyz),
            "s_axis": list(self.s_axis),
            "u_axis": list(self.u_axis),
            "v_axis": list(self.v_axis),
            "config": self.config.to_dict(),
            "seeds": self.seeds.to_dict(),
            "statistics": self.statistics.to_dict(),
            "arrays": dict(self.arrays),
            "summaries": [summary.to_dict() for summary in self.summaries],
            "assignment_reason_counts": dict(self.assignment_reason_counts),
            "assignment_quality_counts": dict(self.assignment_quality_counts),
            "track_quality_counts": dict(self.track_quality_counts),
            "confirmation_transition_counts": dict(self.confirmation_transition_counts),
            "formulas": dict(self.formulas),
            "warnings": list(self.warnings),
            "algorithm_version": self.algorithm_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WireSeparationResult:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values = dict(data)
        values["config"] = WireSeparationConfig.from_dict(values["config"])
        values["seeds"] = WireSeedConfig.from_dict(values["seeds"])
        values["statistics"] = WireSeparationStatistics.from_dict(values["statistics"])
        values["summaries"] = tuple(
            WireTrackSummary.from_dict(value) for value in values["summaries"]
        )
        return cls(**values)
