"""Contracts for A2B grid ground estimation and height-above-ground outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path
from typing import Any

import numpy as np

from core.contracts._json import json_safe_mapping, reject_unknown_fields, require_fields


class GroundQualityFlag(IntFlag):
    """Stable uint16 quality bits for A2B ground and HAG consumers."""

    NONE = 0
    SPARSE_GROUND_CELL = 1
    EMPTY_GROUND_CELL = 2
    INTERPOLATED_GROUND = 4
    EXTRAPOLATION_TOO_FAR = 8
    HIGH_LOCAL_GROUND_VARIATION = 16
    BELOW_GROUND_TOLERANCE = 32
    EXTREME_HEIGHT_AGL = 64
    INVALID_GROUND_ESTIMATE = 128
    SOURCE_CLASSIFICATION_DISAGREES = 256


def _array_shape(value: Any, field_name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(item) for item in shape)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        if not value:
            return (0,)
        first = value[0]
        if isinstance(first, Sequence) and not isinstance(first, str | bytes | bytearray):
            return (len(value), len(first))
        return (len(value),)
    raise ValueError(f"{field_name} must expose a shape or sequence length")


def _require_dtype(value: Any, expected: np.dtype[Any], field_name: str) -> None:
    dtype = getattr(value, "dtype", None)
    if dtype is None or np.dtype(dtype) != expected:
        actual = "missing" if dtype is None else str(dtype)
        raise ValueError(f"{field_name} must have dtype {expected}, got {actual}")


def _require_shape(value: Any, expected: tuple[int, ...], field_name: str) -> None:
    if _array_shape(value, field_name) != expected:
        raise ValueError(f"{field_name} must have shape {expected}")


def _positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _positive_float(value: float, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive finite number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")


def _non_negative_float(value: float, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative finite number")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{field_name} must be a non-negative finite number")


def _shape_dtype(value: Any) -> dict[str, Any]:
    return {"shape": list(_array_shape(value, "array")), "dtype": str(value.dtype)}


@dataclass(frozen=True)
class GroundEstimationConfig:
    """Configuration for deterministic A2B grid ground estimation."""

    source_id: str = "scene_simple"
    grid_size_m: float = 0.5
    ground_quantile: float = 0.05
    quantile_method: str = "linear"
    min_points_per_cell: int = 32
    interpolation_method: str = "idw"
    idw_neighbors: int = 4
    idw_power: float = 2.0
    interpolation_max_distance_m: float = 1.5
    smoothing_radius_cells: int = 0
    high_ground_gradient_threshold_m_per_m: float = 2.0
    below_ground_tolerance_m: float = 0.15
    max_reasonable_hag_m: float = 20.0
    source_ground_code: int = 2
    classification_disagreement_tolerance_m: float = 0.30
    record_memory: bool = True

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("GroundEstimationConfig.source_id must be non-empty")
        _positive_float(self.grid_size_m, "GroundEstimationConfig.grid_size_m")
        if not isinstance(self.ground_quantile, int | float) or isinstance(
            self.ground_quantile, bool
        ):
            raise ValueError("GroundEstimationConfig.ground_quantile must be numeric")
        if not 0.0 < float(self.ground_quantile) < 1.0:
            raise ValueError("GroundEstimationConfig.ground_quantile must be in (0, 1)")
        if self.quantile_method != "linear":
            raise ValueError("GroundEstimationConfig.quantile_method must be linear")
        _positive_int(self.min_points_per_cell, "GroundEstimationConfig.min_points_per_cell")
        if self.interpolation_method != "idw":
            raise ValueError("GroundEstimationConfig.interpolation_method must be idw")
        _positive_int(self.idw_neighbors, "GroundEstimationConfig.idw_neighbors")
        _positive_float(self.idw_power, "GroundEstimationConfig.idw_power")
        _positive_float(
            self.interpolation_max_distance_m,
            "GroundEstimationConfig.interpolation_max_distance_m",
        )
        _non_negative_int(
            self.smoothing_radius_cells,
            "GroundEstimationConfig.smoothing_radius_cells",
        )
        _positive_float(
            self.high_ground_gradient_threshold_m_per_m,
            "GroundEstimationConfig.high_ground_gradient_threshold_m_per_m",
        )
        _non_negative_float(
            self.below_ground_tolerance_m,
            "GroundEstimationConfig.below_ground_tolerance_m",
        )
        _positive_float(
            self.max_reasonable_hag_m,
            "GroundEstimationConfig.max_reasonable_hag_m",
        )
        if not isinstance(self.source_ground_code, int) or not 0 <= self.source_ground_code <= 255:
            raise ValueError("GroundEstimationConfig.source_ground_code must fit uint8")
        _non_negative_float(
            self.classification_disagreement_tolerance_m,
            "GroundEstimationConfig.classification_disagreement_tolerance_m",
        )
        for field_name in (
            "grid_size_m",
            "ground_quantile",
            "idw_power",
            "interpolation_max_distance_m",
            "high_ground_gradient_threshold_m_per_m",
            "below_ground_tolerance_m",
            "max_reasonable_hag_m",
            "classification_disagreement_tolerance_m",
        ):
            object.__setattr__(self, field_name, float(getattr(self, field_name)))

    def to_dict(self) -> dict[str, Any]:
        return {
            field_name: getattr(self, field_name)
            for field_name in (
                "source_id",
                "grid_size_m",
                "ground_quantile",
                "quantile_method",
                "min_points_per_cell",
                "interpolation_method",
                "idw_neighbors",
                "idw_power",
                "interpolation_max_distance_m",
                "smoothing_radius_cells",
                "high_ground_gradient_threshold_m_per_m",
                "below_ground_tolerance_m",
                "max_reasonable_hag_m",
                "source_ground_code",
                "classification_disagreement_tolerance_m",
                "record_memory",
            )
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GroundEstimationConfig:
        allowed = set(cls().to_dict())
        reject_unknown_fields(data, allowed, cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class GroundSurfaceModel:
    """Grid control surface retained for A2B traceability and diagnostics."""

    grid_origin_xy: Any
    grid_size_m: float
    grid_shape: tuple[int, int]
    cell_point_count: Any
    cell_ground_z_m: Any
    cell_control_mask: Any
    cell_quality_flags: Any
    control_cell_xy: Any
    control_ground_z_m: Any

    def __post_init__(self) -> None:
        _require_shape(self.grid_origin_xy, (2,), "GroundSurfaceModel.grid_origin_xy")
        _require_dtype(
            self.grid_origin_xy, np.dtype(np.float64), "GroundSurfaceModel.grid_origin_xy"
        )
        if not np.isfinite(np.asarray(self.grid_origin_xy)).all():
            raise ValueError("GroundSurfaceModel.grid_origin_xy must be finite")
        _positive_float(self.grid_size_m, "GroundSurfaceModel.grid_size_m")
        if len(self.grid_shape) != 2 or any(
            not isinstance(value, int) or value <= 0 for value in self.grid_shape
        ):
            raise ValueError("GroundSurfaceModel.grid_shape must contain two positive integers")
        expected = tuple(self.grid_shape)
        for field_name, dtype in (
            ("cell_point_count", np.dtype(np.int32)),
            ("cell_ground_z_m", np.dtype(np.float32)),
            ("cell_control_mask", np.dtype(np.bool_)),
            ("cell_quality_flags", np.dtype(np.uint16)),
        ):
            value = getattr(self, field_name)
            _require_shape(value, expected, f"GroundSurfaceModel.{field_name}")
            _require_dtype(value, dtype, f"GroundSurfaceModel.{field_name}")
        if np.any(np.asarray(self.cell_point_count) < 0):
            raise ValueError("GroundSurfaceModel.cell_point_count must be non-negative")
        control_count = int(np.count_nonzero(self.cell_control_mask))
        _require_shape(
            self.control_cell_xy, (control_count, 2), "GroundSurfaceModel.control_cell_xy"
        )
        _require_dtype(
            self.control_cell_xy, np.dtype(np.float32), "GroundSurfaceModel.control_cell_xy"
        )
        _require_shape(
            self.control_ground_z_m,
            (control_count,),
            "GroundSurfaceModel.control_ground_z_m",
        )
        _require_dtype(
            self.control_ground_z_m,
            np.dtype(np.float32),
            "GroundSurfaceModel.control_ground_z_m",
        )
        if (
            not np.isfinite(np.asarray(self.control_cell_xy)).all()
            or not np.isfinite(np.asarray(self.control_ground_z_m)).all()
        ):
            raise ValueError("GroundSurfaceModel control arrays must be finite")

    @property
    def control_cell_count(self) -> int:
        return int(np.count_nonzero(self.cell_control_mask))

    def array_metadata(self) -> dict[str, Any]:
        values = {
            "grid_origin_xy": self.grid_origin_xy,
            "cell_point_count": self.cell_point_count,
            "cell_ground_z_m": self.cell_ground_z_m,
            "cell_control_mask": self.cell_control_mask,
            "cell_quality_flags": self.cell_quality_flags,
            "control_cell_xy": self.control_cell_xy,
            "control_ground_z_m": self.control_ground_z_m,
        }
        units = {
            "grid_origin_xy": "m",
            "cell_point_count": "points",
            "cell_ground_z_m": "m",
            "cell_control_mask": "bool",
            "cell_quality_flags": "bitmask",
            "control_cell_xy": "m",
            "control_ground_z_m": "m",
        }
        return {
            name: {**_shape_dtype(value), "unit": units[name]} for name, value in values.items()
        }


@dataclass(frozen=True)
class HeightAboveGroundSet:
    """A2B point arrays aligned exactly with A1 and A2A input order."""

    source_id: str
    manifest_source_id: str
    original_index: Any
    sample_index: Any
    ground_z_m: Any
    raw_height_agl_m: Any
    height_agl_m: Any
    ground_cell_id: Any
    ground_distance_m: Any
    quality_flags: Any

    def __post_init__(self) -> None:
        if not self.source_id or not self.manifest_source_id:
            raise ValueError("HeightAboveGroundSet source identifiers must be non-empty")
        point_count = _array_shape(self.original_index, "HeightAboveGroundSet.original_index")
        if len(point_count) != 1:
            raise ValueError("HeightAboveGroundSet.original_index must have shape (M,)")
        point_count = point_count[0]
        for field_name in ("original_index", "sample_index"):
            value = getattr(self, field_name)
            _require_shape(value, (point_count,), f"HeightAboveGroundSet.{field_name}")
            _require_dtype(value, np.dtype(np.int64), f"HeightAboveGroundSet.{field_name}")
        for field_name in ("ground_z_m", "raw_height_agl_m", "height_agl_m", "ground_distance_m"):
            value = getattr(self, field_name)
            _require_shape(value, (point_count,), f"HeightAboveGroundSet.{field_name}")
            _require_dtype(value, np.dtype(np.float32), f"HeightAboveGroundSet.{field_name}")
        _require_shape(self.ground_cell_id, (point_count,), "HeightAboveGroundSet.ground_cell_id")
        _require_dtype(
            self.ground_cell_id, np.dtype(np.int32), "HeightAboveGroundSet.ground_cell_id"
        )
        _require_shape(self.quality_flags, (point_count,), "HeightAboveGroundSet.quality_flags")
        _require_dtype(
            self.quality_flags, np.dtype(np.uint16), "HeightAboveGroundSet.quality_flags"
        )
        if np.any(np.asarray(self.original_index) < 0) or np.any(np.asarray(self.sample_index) < 0):
            raise ValueError("HeightAboveGroundSet indexes must be non-negative")
        invalid = (
            np.asarray(self.quality_flags) & int(GroundQualityFlag.INVALID_GROUND_ESTIMATE)
        ) != 0
        for field_name in ("ground_z_m", "raw_height_agl_m", "height_agl_m", "ground_distance_m"):
            value = np.asarray(getattr(self, field_name))
            if not np.isfinite(value[~invalid]).all():
                raise ValueError(
                    f"HeightAboveGroundSet.{field_name} is non-finite without invalid flag"
                )
        if np.any(np.asarray(self.height_agl_m)[~invalid] < 0.0):
            raise ValueError("HeightAboveGroundSet.height_agl_m must be non-negative when valid")

    @property
    def point_count(self) -> int:
        return _array_shape(self.original_index, "HeightAboveGroundSet.original_index")[0]

    def array_metadata(self) -> dict[str, Any]:
        values = {
            "original_index": self.original_index,
            "sample_index": self.sample_index,
            "ground_z_m": self.ground_z_m,
            "raw_height_agl_m": self.raw_height_agl_m,
            "height_agl_m": self.height_agl_m,
            "ground_cell_id": self.ground_cell_id,
            "ground_distance_m": self.ground_distance_m,
            "quality_flags": self.quality_flags,
        }
        units = {
            "original_index": "index",
            "sample_index": "index",
            "ground_z_m": "m",
            "raw_height_agl_m": "m",
            "height_agl_m": "m",
            "ground_cell_id": "index",
            "ground_distance_m": "m",
            "quality_flags": "bitmask",
        }
        return {
            name: {**_shape_dtype(value), "unit": units[name]} for name, value in values.items()
        }


@dataclass(frozen=True)
class GroundEstimationStatistics:
    """Counts, diagnostics, time, and RSS values from A2B execution."""

    input_point_count: int
    output_point_count: int
    total_cell_count: int
    nonempty_cell_count: int
    direct_control_cell_count: int
    sparse_cell_count: int
    empty_cell_count: int
    interpolated_cell_count: int
    extrapolation_too_far_cell_count: int
    valid_ground_point_count: int
    invalid_ground_point_count: int
    high_gradient_cell_count: int
    high_gradient_point_count: int
    raw_negative_hag_count: int
    below_ground_tolerance_count: int
    extreme_hag_count: int
    classification_ground_count: int
    classification_disagreement_count: int
    elapsed_seconds: float
    rss_start_bytes: int | None = None
    rss_peak_bytes: int | None = None
    rss_end_bytes: int | None = None
    control_tree_coordinate_bytes: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "input_point_count",
            "output_point_count",
            "total_cell_count",
            "nonempty_cell_count",
            "direct_control_cell_count",
            "sparse_cell_count",
            "empty_cell_count",
            "interpolated_cell_count",
            "extrapolation_too_far_cell_count",
            "valid_ground_point_count",
            "invalid_ground_point_count",
            "high_gradient_cell_count",
            "high_gradient_point_count",
            "raw_negative_hag_count",
            "below_ground_tolerance_count",
            "extreme_hag_count",
            "classification_ground_count",
            "classification_disagreement_count",
        ):
            _non_negative_int(getattr(self, field_name), f"GroundEstimationStatistics.{field_name}")
        _non_negative_float(self.elapsed_seconds, "GroundEstimationStatistics.elapsed_seconds")
        for field_name in (
            "rss_start_bytes",
            "rss_peak_bytes",
            "rss_end_bytes",
            "control_tree_coordinate_bytes",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _non_negative_int(value, f"GroundEstimationStatistics.{field_name}")

    def to_dict(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GroundEstimationStatistics:
        optional = {
            "rss_start_bytes",
            "rss_peak_bytes",
            "rss_end_bytes",
            "control_tree_coordinate_bytes",
        }
        required = set(cls.__dataclass_fields__) - optional
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class GroundEstimationResult:
    """JSON-safe A2B artifact metadata, provenance, and summary statistics."""

    source_id: str
    manifest_source_id: str
    input_a1_npz_path: Path
    input_a1_json_path: Path
    input_a1_npz_sha256: str
    input_a1_json_sha256: str
    input_a2a_npz_path: Path
    input_a2a_json_path: Path
    input_a2a_npz_sha256: str
    input_a2a_json_sha256: str
    output_npz_path: Path
    output_json_path: Path
    output_npz_sha256: str
    output_npz_bytes: int
    origin_xyz: tuple[float, float, float]
    config: GroundEstimationConfig
    statistics: GroundEstimationStatistics
    grid: Mapping[str, Any]
    arrays: Mapping[str, Any]
    quality_flag_counts: Mapping[str, Any]
    distributions: Mapping[str, Any]
    formulas: Mapping[str, Any]
    classification_validation: Mapping[str, Any]
    algorithm_version: str
    created_at: str

    def __post_init__(self) -> None:
        if not self.source_id or not self.manifest_source_id:
            raise ValueError("GroundEstimationResult source identifiers must be non-empty")
        for field_name in (
            "input_a1_npz_path",
            "input_a1_json_path",
            "input_a2a_npz_path",
            "input_a2a_json_path",
            "output_npz_path",
            "output_json_path",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))
        for field_name in (
            "input_a1_npz_sha256",
            "input_a1_json_sha256",
            "input_a2a_npz_sha256",
            "input_a2a_json_sha256",
            "output_npz_sha256",
        ):
            value = getattr(self, field_name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(
                    f"GroundEstimationResult.{field_name} must be a SHA-256 hex digest"
                )
        _non_negative_int(self.output_npz_bytes, "GroundEstimationResult.output_npz_bytes")
        origin = tuple(float(value) for value in self.origin_xyz)
        if len(origin) != 3 or not all(math.isfinite(value) for value in origin):
            raise ValueError("GroundEstimationResult.origin_xyz must contain three finite values")
        object.__setattr__(self, "origin_xyz", origin)
        if not isinstance(self.config, GroundEstimationConfig):
            raise ValueError("GroundEstimationResult.config must be GroundEstimationConfig")
        if not isinstance(self.statistics, GroundEstimationStatistics):
            raise ValueError("GroundEstimationResult.statistics must be GroundEstimationStatistics")
        if not self.algorithm_version or not self.created_at:
            raise ValueError("GroundEstimationResult version and timestamp must be non-empty")
        for field_name in (
            "grid",
            "arrays",
            "quality_flag_counts",
            "distributions",
            "formulas",
            "classification_validation",
        ):
            object.__setattr__(
                self, field_name, json_safe_mapping(getattr(self, field_name), field_name)
            )

    def to_dict(self) -> dict[str, Any]:
        data = {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
            if field_name
            not in {
                "input_a1_npz_path",
                "input_a1_json_path",
                "input_a2a_npz_path",
                "input_a2a_json_path",
                "output_npz_path",
                "output_json_path",
                "config",
                "statistics",
                "origin_xyz",
            }
        }
        data.update(
            {
                "input_a1_npz_path": str(self.input_a1_npz_path),
                "input_a1_json_path": str(self.input_a1_json_path),
                "input_a2a_npz_path": str(self.input_a2a_npz_path),
                "input_a2a_json_path": str(self.input_a2a_json_path),
                "output_npz_path": str(self.output_npz_path),
                "output_json_path": str(self.output_json_path),
                "origin_xyz": list(self.origin_xyz),
                "config": self.config.to_dict(),
                "statistics": self.statistics.to_dict(),
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GroundEstimationResult:
        required = set(cls.__dataclass_fields__)
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, required, cls.__name__)
        values = dict(data)
        values["config"] = GroundEstimationConfig.from_dict(values["config"])
        values["statistics"] = GroundEstimationStatistics.from_dict(values["statistics"])
        return cls(**values)
