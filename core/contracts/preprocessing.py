"""Contracts for point cloud preprocessing outputs.

Coordinates and distances are expressed in meters. Large point arrays are
represented by NumPy-like objects in memory and by NPZ files on disk.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from core.contracts._json import json_safe_mapping, reject_unknown_fields, require_fields
from core.contracts.pointcloud import Vector3, _as_vector3

OriginMode = Literal["header_min", "first_valid", "explicit"]
CoordinateDtype = Literal["float32", "float64"]
VoxelRepresentative = Literal["nearest_centroid", "first"]


def _positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _positive_float(value: float, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive finite number")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{field_name} must be a positive finite number")


def _optional_non_negative_int(value: int | None, field_name: str) -> None:
    if value is not None:
        _non_negative_int(value, field_name)


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


def _validate_vector_array(value: Any, field_name: str, width: int) -> int:
    shape = _array_shape(value, field_name)
    if len(shape) != 2 or shape[1] != width:
        raise ValueError(f"{field_name} must have shape (M, {width})")
    return shape[0]


def _validate_index_array(value: Any, field_name: str, expected_length: int) -> None:
    shape = _array_shape(value, field_name)
    if shape != (expected_length,):
        raise ValueError(f"{field_name} must have shape (M,)")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for the A1 basic preprocessing pipeline."""

    source_id: str = "scene_simple"
    chunk_size: int = 250_000
    origin_mode: OriginMode = "header_min"
    origin_xyz: Vector3 | None = None
    coordinate_dtype: CoordinateDtype = "float32"
    reject_nan_inf: bool = True
    max_abs_local_coord_m: float | None = 10_000.0
    enable_statistical_outlier_filter: bool = False
    sor_k: int = 16
    sor_std_ratio: float = 2.5
    sor_max_points: int = 300_000
    voxel_size_m: float = 0.05
    voxel_representative: VoxelRepresentative = "nearest_centroid"
    max_input_points: int | None = None
    max_output_points: int | None = 1_000_000
    record_memory: bool = True

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("PreprocessingConfig.source_id must be non-empty")
        _positive_int(self.chunk_size, "PreprocessingConfig.chunk_size")
        if self.origin_mode not in {"header_min", "first_valid", "explicit"}:
            raise ValueError("PreprocessingConfig.origin_mode is invalid")
        if self.origin_mode == "explicit" and self.origin_xyz is None:
            raise ValueError("PreprocessingConfig.origin_xyz is required for explicit origin")
        if self.origin_xyz is not None:
            object.__setattr__(
                self,
                "origin_xyz",
                _as_vector3(self.origin_xyz, "PreprocessingConfig.origin_xyz"),
            )
        if self.coordinate_dtype not in {"float32", "float64"}:
            raise ValueError("PreprocessingConfig.coordinate_dtype must be float32 or float64")
        if self.max_abs_local_coord_m is not None:
            _positive_float(
                self.max_abs_local_coord_m,
                "PreprocessingConfig.max_abs_local_coord_m",
            )
            object.__setattr__(
                self,
                "max_abs_local_coord_m",
                float(self.max_abs_local_coord_m),
            )
        _positive_int(self.sor_k, "PreprocessingConfig.sor_k")
        _positive_float(self.sor_std_ratio, "PreprocessingConfig.sor_std_ratio")
        object.__setattr__(self, "sor_std_ratio", float(self.sor_std_ratio))
        _positive_int(self.sor_max_points, "PreprocessingConfig.sor_max_points")
        _positive_float(self.voxel_size_m, "PreprocessingConfig.voxel_size_m")
        object.__setattr__(self, "voxel_size_m", float(self.voxel_size_m))
        if self.voxel_representative not in {"nearest_centroid", "first"}:
            raise ValueError("PreprocessingConfig.voxel_representative is invalid")
        _optional_non_negative_int(self.max_input_points, "PreprocessingConfig.max_input_points")
        _optional_non_negative_int(
            self.max_output_points,
            "PreprocessingConfig.max_output_points",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "chunk_size": self.chunk_size,
            "origin_mode": self.origin_mode,
            "origin_xyz": list(self.origin_xyz) if self.origin_xyz is not None else None,
            "coordinate_dtype": self.coordinate_dtype,
            "reject_nan_inf": self.reject_nan_inf,
            "max_abs_local_coord_m": self.max_abs_local_coord_m,
            "enable_statistical_outlier_filter": self.enable_statistical_outlier_filter,
            "sor_k": self.sor_k,
            "sor_std_ratio": self.sor_std_ratio,
            "sor_max_points": self.sor_max_points,
            "voxel_size_m": self.voxel_size_m,
            "voxel_representative": self.voxel_representative,
            "max_input_points": self.max_input_points,
            "max_output_points": self.max_output_points,
            "record_memory": self.record_memory,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreprocessingConfig:
        allowed = {
            "source_id",
            "chunk_size",
            "origin_mode",
            "origin_xyz",
            "coordinate_dtype",
            "reject_nan_inf",
            "max_abs_local_coord_m",
            "enable_statistical_outlier_filter",
            "sor_k",
            "sor_std_ratio",
            "sor_max_points",
            "voxel_size_m",
            "voxel_representative",
            "max_input_points",
            "max_output_points",
            "record_memory",
        }
        reject_unknown_fields(data, allowed, cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class ProcessingStatistics:
    """Point counts, runtime, and memory counters for preprocessing."""

    input_point_count: int
    processed_input_point_count: int
    valid_point_count: int
    nan_inf_removed_count: int
    abnormal_coord_removed_count: int
    outlier_removed_count: int
    downsampled_point_count: int
    downsample_ratio: float
    chunk_size: int
    voxel_size_m: float
    elapsed_seconds: float
    rss_start_bytes: int | None = None
    rss_peak_bytes: int | None = None
    rss_end_bytes: int | None = None
    estimated_input_array_bytes: int | None = None
    estimated_output_array_bytes: int | None = None
    sor_skipped_reason: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "input_point_count",
            "processed_input_point_count",
            "valid_point_count",
            "nan_inf_removed_count",
            "abnormal_coord_removed_count",
            "outlier_removed_count",
            "downsampled_point_count",
        ):
            _non_negative_int(getattr(self, field_name), f"ProcessingStatistics.{field_name}")
        _positive_int(self.chunk_size, "ProcessingStatistics.chunk_size")
        _positive_float(self.voxel_size_m, "ProcessingStatistics.voxel_size_m")
        if not math.isfinite(self.downsample_ratio) or self.downsample_ratio < 0:
            raise ValueError("ProcessingStatistics.downsample_ratio must be non-negative")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("ProcessingStatistics.elapsed_seconds must be non-negative")
        for field_name in (
            "rss_start_bytes",
            "rss_peak_bytes",
            "rss_end_bytes",
            "estimated_input_array_bytes",
            "estimated_output_array_bytes",
        ):
            _optional_non_negative_int(
                getattr(self, field_name),
                f"ProcessingStatistics.{field_name}",
            )
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_point_count": self.input_point_count,
            "processed_input_point_count": self.processed_input_point_count,
            "valid_point_count": self.valid_point_count,
            "nan_inf_removed_count": self.nan_inf_removed_count,
            "abnormal_coord_removed_count": self.abnormal_coord_removed_count,
            "outlier_removed_count": self.outlier_removed_count,
            "downsampled_point_count": self.downsampled_point_count,
            "downsample_ratio": self.downsample_ratio,
            "chunk_size": self.chunk_size,
            "voxel_size_m": self.voxel_size_m,
            "elapsed_seconds": self.elapsed_seconds,
            "rss_start_bytes": self.rss_start_bytes,
            "rss_peak_bytes": self.rss_peak_bytes,
            "rss_end_bytes": self.rss_end_bytes,
            "estimated_input_array_bytes": self.estimated_input_array_bytes,
            "estimated_output_array_bytes": self.estimated_output_array_bytes,
            "sor_skipped_reason": self.sor_skipped_reason,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProcessingStatistics:
        required = {
            "input_point_count",
            "processed_input_point_count",
            "valid_point_count",
            "nan_inf_removed_count",
            "abnormal_coord_removed_count",
            "outlier_removed_count",
            "downsampled_point_count",
            "downsample_ratio",
            "chunk_size",
            "voxel_size_m",
            "elapsed_seconds",
        }
        allowed = required | {
            "rss_start_bytes",
            "rss_peak_bytes",
            "rss_end_bytes",
            "estimated_input_array_bytes",
            "estimated_output_array_bytes",
            "sor_skipped_reason",
            "warnings",
        }
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, allowed, cls.__name__)
        return cls(
            input_point_count=data["input_point_count"],
            processed_input_point_count=data["processed_input_point_count"],
            valid_point_count=data["valid_point_count"],
            nan_inf_removed_count=data["nan_inf_removed_count"],
            abnormal_coord_removed_count=data["abnormal_coord_removed_count"],
            outlier_removed_count=data["outlier_removed_count"],
            downsampled_point_count=data["downsampled_point_count"],
            downsample_ratio=data["downsample_ratio"],
            chunk_size=data["chunk_size"],
            voxel_size_m=data["voxel_size_m"],
            elapsed_seconds=data["elapsed_seconds"],
            rss_start_bytes=data.get("rss_start_bytes"),
            rss_peak_bytes=data.get("rss_peak_bytes"),
            rss_end_bytes=data.get("rss_end_bytes"),
            estimated_input_array_bytes=data.get("estimated_input_array_bytes"),
            estimated_output_array_bytes=data.get("estimated_output_array_bytes"),
            sor_skipped_reason=data.get("sor_skipped_reason"),
            warnings=tuple(data.get("warnings", ())),
        )


@dataclass(frozen=True)
class DownsampledPointCloud:
    """A voxel-downsampled point cloud with raw LAS index traceability."""

    source_id: str
    xyz_local: Any
    origin_xyz: Any
    original_index: Any
    sample_index: Any
    voxel_key: Any
    voxel_point_count: Any
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("DownsampledPointCloud.source_id must be non-empty")
        point_count = _validate_vector_array(
            self.xyz_local,
            "DownsampledPointCloud.xyz_local",
            3,
        )
        if _array_shape(self.origin_xyz, "DownsampledPointCloud.origin_xyz") != (3,):
            raise ValueError("DownsampledPointCloud.origin_xyz must have shape (3,)")
        _validate_index_array(
            self.original_index,
            "DownsampledPointCloud.original_index",
            point_count,
        )
        _validate_index_array(
            self.sample_index,
            "DownsampledPointCloud.sample_index",
            point_count,
        )
        if _array_shape(self.voxel_key, "DownsampledPointCloud.voxel_key") != (point_count, 3):
            raise ValueError("DownsampledPointCloud.voxel_key must have shape (M, 3)")
        _validate_index_array(
            self.voxel_point_count,
            "DownsampledPointCloud.voxel_point_count",
            point_count,
        )
        for name, value in self.attributes.items():
            shape = _array_shape(value, f"DownsampledPointCloud.attributes.{name}")
            if not shape or shape[0] != point_count:
                raise ValueError(f"DownsampledPointCloud.attributes.{name} length must match M")

    @property
    def point_count(self) -> int:
        return _array_shape(self.xyz_local, "DownsampledPointCloud.xyz_local")[0]

    def array_metadata(self) -> dict[str, Any]:
        result = {
            "xyz_local": _shape_dtype(self.xyz_local),
            "origin_xyz": _shape_dtype(self.origin_xyz),
            "original_index": _shape_dtype(self.original_index),
            "sample_index": _shape_dtype(self.sample_index),
            "voxel_key": _shape_dtype(self.voxel_key),
            "voxel_point_count": _shape_dtype(self.voxel_point_count),
            "attributes": {
                name: _shape_dtype(value) for name, value in sorted(self.attributes.items())
            },
        }
        return result


@dataclass(frozen=True)
class PreprocessingResult:
    """Preprocessing output paths and JSON-safe metadata."""

    source_id: str
    input_path: Path
    index_map_path: Path
    manifest_path: Path
    output_npz_path: Path
    output_json_path: Path
    config: PreprocessingConfig
    statistics: ProcessingStatistics
    manifest_source_id: str
    origin_xyz: Vector3
    preview_html_path: Path | None = None
    arrays: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    created_at: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("PreprocessingResult.source_id must be non-empty")
        for field_name in (
            "input_path",
            "index_map_path",
            "manifest_path",
            "output_npz_path",
            "output_json_path",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))
        if self.preview_html_path is not None:
            object.__setattr__(self, "preview_html_path", Path(self.preview_html_path))
        if not isinstance(self.config, PreprocessingConfig):
            raise ValueError("PreprocessingResult.config must be a PreprocessingConfig")
        if not isinstance(self.statistics, ProcessingStatistics):
            raise ValueError("PreprocessingResult.statistics must be a ProcessingStatistics")
        if not self.manifest_source_id:
            raise ValueError("PreprocessingResult.manifest_source_id must be non-empty")
        object.__setattr__(self, "origin_xyz", _as_vector3(self.origin_xyz, "origin_xyz"))
        object.__setattr__(self, "arrays", json_safe_mapping(self.arrays, "arrays"))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "input_path": str(self.input_path),
            "index_map_path": str(self.index_map_path),
            "manifest_path": str(self.manifest_path),
            "output_npz_path": str(self.output_npz_path),
            "output_json_path": str(self.output_json_path),
            "preview_html_path": (
                str(self.preview_html_path) if self.preview_html_path is not None else None
            ),
            "config": self.config.to_dict(),
            "statistics": self.statistics.to_dict(),
            "manifest_source_id": self.manifest_source_id,
            "origin_xyz": list(self.origin_xyz),
            "arrays": dict(self.arrays),
            "warnings": list(self.warnings),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreprocessingResult:
        required = {
            "source_id",
            "input_path",
            "index_map_path",
            "manifest_path",
            "output_npz_path",
            "output_json_path",
            "config",
            "statistics",
            "manifest_source_id",
            "origin_xyz",
        }
        allowed = required | {"preview_html_path", "arrays", "warnings", "created_at"}
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, allowed, cls.__name__)
        return cls(
            source_id=data["source_id"],
            input_path=Path(data["input_path"]),
            index_map_path=Path(data["index_map_path"]),
            manifest_path=Path(data["manifest_path"]),
            output_npz_path=Path(data["output_npz_path"]),
            output_json_path=Path(data["output_json_path"]),
            preview_html_path=(
                Path(data["preview_html_path"])
                if data.get("preview_html_path") is not None
                else None
            ),
            config=PreprocessingConfig.from_dict(data["config"]),
            statistics=ProcessingStatistics.from_dict(data["statistics"]),
            manifest_source_id=data["manifest_source_id"],
            origin_xyz=data["origin_xyz"],
            arrays=data.get("arrays", {}),
            warnings=tuple(data.get("warnings", ())),
            created_at=data.get("created_at"),
        )


def _shape_dtype(value: Any) -> dict[str, Any]:
    shape = _array_shape(value, "array")
    dtype = getattr(value, "dtype", None)
    return {
        "shape": list(shape),
        "dtype": str(dtype) if dtype is not None else type(value).__name__,
    }
