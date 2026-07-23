"""Contracts for W1 confirmed-wire centreline reconstruction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Any

import numpy as np

from core.contracts._json import json_safe_mapping, reject_unknown_fields, require_fields


class WireModelStatus(IntEnum):
    CONFIRMED_MODEL = 0
    FALLBACK_MODEL = 1
    REVIEW_REQUIRED = 2
    FAILED = 3


class WireModelMethod(IntEnum):
    CUBIC_BSPLINE = 0
    SMOOTHED_POLYLINE = 1
    PIECEWISE_LINEAR = 2
    NONE = 3
    MIXED = 4


class WireModelQualityFlag(IntFlag):
    NONE = 0
    INSUFFICIENT_POINTS = 1
    INSUFFICIENT_SPAN = 2
    LARGE_GAP = 4
    HIGH_RESIDUAL = 8
    HIGH_TURN_ANGLE = 16
    UNSTABLE_DIRECTION = 32
    ENDPOINT_UNCERTAINTY = 64
    FALLBACK_USED = 128
    RECONSTRUCTION_FAILURE = 256
    NONFINITE_INPUT = 512
    FRAGMENTED_COVERAGE = 1024
    OUTLIER_HEAVY = 2048


def _array(value: Any, shape: tuple[int, ...], dtype: np.dtype[Any], name: str) -> None:
    if getattr(value, "shape", None) != shape or getattr(value, "dtype", None) is None:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    if np.dtype(value.dtype) != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")


def _finite(value: float, name: str, minimum: float | None = None) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _identifier(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class WireReconstructionConfig:
    """Deterministic W1 thresholds; independent from A4 confirmation settings."""

    source_id: str = "scene_simple"
    representative_bin_m: float = 0.10
    minimum_segment_points: int = 4
    minimum_segment_span_m: float = 0.30
    gap_relative_multiplier: float = 8.0
    gap_absolute_min_m: float = 0.35
    gap_absolute_max_m: float = 1.00
    maximum_residual_m: float = 0.15
    maximum_turn_angle_rad: float = 1.25
    maximum_turn_angle_spike_multiplier: float = 3.0
    maximum_sample_step_multiplier: float = 2.0
    maximum_arc_length_inflation_ratio: float = 1.02
    sample_spacing_m: float = 0.05
    maximum_samples_per_segment: int = 4000
    smoothing_candidates: tuple[float, ...] = (0.0, 0.002, 0.01)
    record_memory: bool = True

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        for name in (
            "representative_bin_m",
            "minimum_segment_span_m",
            "gap_relative_multiplier",
            "gap_absolute_min_m",
            "gap_absolute_max_m",
            "maximum_residual_m",
            "maximum_turn_angle_rad",
            "maximum_turn_angle_spike_multiplier",
            "maximum_sample_step_multiplier",
            "maximum_arc_length_inflation_ratio",
            "sample_spacing_m",
        ):
            object.__setattr__(
                self, name, _finite(getattr(self, name), name, np.nextafter(0.0, 1.0))
            )
        if self.gap_absolute_min_m > self.gap_absolute_max_m:
            raise ValueError("gap absolute limits are invalid")
        _identifier(self.minimum_segment_points, "minimum_segment_points")
        if self.minimum_segment_points < 2:
            raise ValueError("minimum_segment_points must be at least 2")
        _identifier(self.maximum_samples_per_segment, "maximum_samples_per_segment")
        if not self.smoothing_candidates or any(
            value < 0 or not math.isfinite(value) for value in self.smoothing_candidates
        ):
            raise ValueError("smoothing_candidates must be finite and non-negative")
        if tuple(self.smoothing_candidates) != tuple(sorted(set(self.smoothing_candidates))):
            raise ValueError("smoothing_candidates must be unique and increasing")
        object.__setattr__(
            self, "smoothing_candidates", tuple(float(value) for value in self.smoothing_candidates)
        )
        if not isinstance(self.record_memory, bool):
            raise ValueError("record_memory must be bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: list(value) if name == "smoothing_candidates" else value
            for name, value in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WireReconstructionConfig:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class WireModelSet:
    """Core flattened numerical W1 layout, with non-contiguous preserved wire IDs allowed."""

    wire_id: Any
    source_track_id: Any
    wire_status: Any
    wire_method: Any
    wire_quality_flags: Any
    wire_has_distance_ready_segments: Any
    wire_continuous_distance_ready: Any
    wire_segment_offsets: Any
    segment_wire_id: Any
    segment_index: Any
    segment_method: Any
    segment_quality_flags: Any
    segment_distance_ready: Any
    segment_parameter_domain: Any
    sampled_curve_xyz: Any
    sampled_curve_parameter: Any
    sampled_curve_offsets: Any

    def __post_init__(self) -> None:
        count = len(self.wire_id)
        for name, dtype in (
            ("wire_id", np.dtype(np.int32)),
            ("source_track_id", np.dtype(np.int32)),
            ("wire_status", np.dtype(np.uint8)),
            ("wire_method", np.dtype(np.uint8)),
            ("wire_quality_flags", np.dtype(np.uint16)),
            ("wire_has_distance_ready_segments", np.dtype(np.bool_)),
            ("wire_continuous_distance_ready", np.dtype(np.bool_)),
        ):
            _array(getattr(self, name), (count,), dtype, name)
        if np.any(np.asarray(self.wire_id) < 0) or len(np.unique(self.wire_id)) != count:
            raise ValueError("wire_id must be unique and non-negative")
        if list(np.asarray(self.wire_id)) != sorted(np.asarray(self.wire_id).tolist()):
            raise ValueError("wire_id must be deterministically ordered")
        _array(self.wire_segment_offsets, (count + 1,), np.dtype(np.int64), "wire_segment_offsets")
        segment_count = len(self.segment_wire_id)
        for name, shape, dtype in (
            ("segment_wire_id", (segment_count,), np.dtype(np.int32)),
            ("segment_index", (segment_count,), np.dtype(np.int32)),
            ("segment_method", (segment_count,), np.dtype(np.uint8)),
            ("segment_quality_flags", (segment_count,), np.dtype(np.uint16)),
            ("segment_distance_ready", (segment_count,), np.dtype(np.bool_)),
            ("segment_parameter_domain", (segment_count, 2), np.dtype(np.float64)),
            ("sampled_curve_parameter", (len(self.sampled_curve_parameter),), np.dtype(np.float32)),
            ("sampled_curve_xyz", (len(self.sampled_curve_parameter), 3), np.dtype(np.float32)),
            ("sampled_curve_offsets", (segment_count + 1,), np.dtype(np.int64)),
        ):
            _array(getattr(self, name), shape, dtype, name)
        for offsets, total, name in (
            (self.wire_segment_offsets, segment_count, "wire_segment_offsets"),
            (
                self.sampled_curve_offsets,
                len(self.sampled_curve_parameter),
                "sampled_curve_offsets",
            ),
        ):
            values = np.asarray(offsets)
            if values[0] != 0 or values[-1] != total or np.any(np.diff(values) < 0):
                raise ValueError(f"{name} must monotonically span its values")
        if (
            not np.isfinite(self.segment_parameter_domain).all()
            or not np.isfinite(self.sampled_curve_xyz).all()
        ):
            raise ValueError("model coordinates and parameter domains must be finite")
        if np.any(self.segment_parameter_domain[:, 1] < self.segment_parameter_domain[:, 0]):
            raise ValueError("segment parameter domains must be ordered")
        for wire_pos, identifier in enumerate(np.asarray(self.wire_id)):
            start, end = self.wire_segment_offsets[wire_pos : wire_pos + 2]
            if np.any(np.asarray(self.segment_wire_id)[start:end] != identifier):
                raise ValueError("segment parent wire IDs must match offsets")
        if np.any(np.asarray(self.wire_status) == int(WireModelStatus.FAILED)):
            failed = np.asarray(self.wire_status) == int(WireModelStatus.FAILED)
            for pos in np.flatnonzero(failed):
                start, end = self.wire_segment_offsets[pos : pos + 2]
                if np.any(np.asarray(self.segment_distance_ready)[start:end]):
                    raise ValueError("FAILED wires cannot contain distance-ready segments")


@dataclass(frozen=True)
class WireModelSummary:
    wire_id: int
    source_track_id: int
    status: WireModelStatus
    method: WireModelMethod
    quality_flags: int
    wire_has_distance_ready_segments: bool
    wire_continuous_distance_ready: bool
    observed_segment_count: int
    point_count: int
    representative_count: int
    observed_total_span_m: float
    observed_envelope_span_m: float
    total_unbridged_gap_span_m: float
    observed_coverage_ratio: float
    observed_total_arc_length_m: float
    start_xyz: tuple[float, float, float]
    end_xyz: tuple[float, float, float]
    main_direction: tuple[float, float, float]
    fallback_or_review_reason: str = ""

    def __post_init__(self) -> None:
        _identifier(self.wire_id, "wire_id")
        _identifier(self.source_track_id, "source_track_id")
        if not isinstance(self.status, WireModelStatus) or not isinstance(
            self.method, WireModelMethod
        ):
            raise ValueError("status and method must use W1 enums")
        for name in (
            "quality_flags",
            "observed_segment_count",
            "point_count",
            "representative_count",
        ):
            _identifier(getattr(self, name), name)
        for name in (
            "observed_total_span_m",
            "observed_envelope_span_m",
            "total_unbridged_gap_span_m",
            "observed_coverage_ratio",
            "observed_total_arc_length_m",
        ):
            _finite(getattr(self, name), name, 0.0)
        if self.observed_coverage_ratio > 1.0 + 1e-9:
            raise ValueError("observed_coverage_ratio must not exceed one")
        if self.status == WireModelStatus.FAILED and self.wire_has_distance_ready_segments:
            raise ValueError("FAILED wire cannot have distance-ready segments")
        for name in ("start_xyz", "end_xyz", "main_direction"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["status"] = self.status.name
        result["method"] = self.method.name
        for name in ("start_xyz", "end_xyz", "main_direction"):
            result[name] = list(result[name])
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WireModelSummary:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values = dict(data)
        values["status"] = WireModelStatus[values["status"]]
        values["method"] = WireModelMethod[values["method"]]
        return cls(**values)


@dataclass(frozen=True)
class WireReconstructionStatistics:
    input_point_count: int
    confirmed_wire_point_count: int
    wire_count: int
    segment_count: int
    gap_count: int
    distance_ready_segment_count: int
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
                _identifier(value, name)
        _finite(self.elapsed_seconds, "elapsed_seconds", 0.0)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WireReconstructionStatistics:
        required = set(cls.__dataclass_fields__) - {
            "rss_start_bytes",
            "rss_peak_bytes",
            "rss_end_bytes",
        }
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class WireReconstructionResult:
    schema_version: str
    source_id: str
    manifest_source_id: str
    input_a4_npz_relpath: str
    input_a4_json_relpath: str
    input_a4_npz_sha256: str
    input_a4_json_sha256: str
    output_npz_relpath: str
    output_json_relpath: str
    output_npz_sha256: str
    output_npz_bytes: int
    config: WireReconstructionConfig
    statistics: WireReconstructionStatistics
    arrays: Mapping[str, Any]
    wires: tuple[WireModelSummary, ...]
    segments: tuple[Mapping[str, Any], ...] = ()
    gaps: tuple[Mapping[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    algorithm_version: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.schema_version or not self.source_id or not self.manifest_source_id:
            raise ValueError("schema and source identifiers must be non-empty")
        for name in (
            "input_a4_npz_relpath",
            "input_a4_json_relpath",
            "output_npz_relpath",
            "output_json_relpath",
        ):
            value = getattr(self, name)
            if not value or ":" in value or "\\" in value or value.startswith("/"):
                raise ValueError(f"{name} must be a relative POSIX path")
        for name in ("input_a4_npz_sha256", "input_a4_json_sha256", "output_npz_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        _identifier(self.output_npz_bytes, "output_npz_bytes")
        if not isinstance(self.config, WireReconstructionConfig) or not isinstance(
            self.statistics, WireReconstructionStatistics
        ):
            raise ValueError("config and statistics must use W1 contracts")
        if [item.wire_id for item in self.wires] != sorted(item.wire_id for item in self.wires):
            raise ValueError("wire summaries must be deterministically ordered")
        if len({item.wire_id for item in self.wires}) != len(self.wires):
            raise ValueError("wire summaries must have unique IDs")
        object.__setattr__(self, "arrays", json_safe_mapping(self.arrays, "arrays"))
        object.__setattr__(
            self, "segments", tuple(json_safe_mapping(item, "segments") for item in self.segments)
        )
        object.__setattr__(
            self, "gaps", tuple(json_safe_mapping(item, "gaps") for item in self.gaps)
        )
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))
        if not self.algorithm_version or not self.created_at:
            raise ValueError("algorithm_version and created_at must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["config"] = self.config.to_dict()
        result["statistics"] = self.statistics.to_dict()
        result["wires"] = [item.to_dict() for item in self.wires]
        result["segments"] = [dict(item) for item in self.segments]
        result["gaps"] = [dict(item) for item in self.gaps]
        result["warnings"] = list(self.warnings)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WireReconstructionResult:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values = dict(data)
        values["config"] = WireReconstructionConfig.from_dict(values["config"])
        values["statistics"] = WireReconstructionStatistics.from_dict(values["statistics"])
        values["wires"] = tuple(WireModelSummary.from_dict(item) for item in values["wires"])
        return cls(**values)
