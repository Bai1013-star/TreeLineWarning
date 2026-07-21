"""Contracts for A2A local geometric point features.

Coordinates are expressed in meters.  Large arrays are represented in memory by
NumPy-like objects and are persisted only in NPZ artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path
from typing import Any

import numpy as np

from core.contracts._json import json_safe_mapping, reject_unknown_fields, require_fields


class FeatureQualityFlag(IntFlag):
    """Stable uint16 quality bits for A2A feature consumers."""

    NONE = 0
    INSUFFICIENT_NEIGHBORS = 1
    DEGENERATE_COVARIANCE = 2
    DUPLICATE_NEIGHBOR_COORDINATES = 4
    NEGATIVE_EIGENVALUE_CLIPPED = 8
    NORMAL_DIRECTION_AMBIGUOUS = 16
    PRINCIPAL_DIRECTION_AMBIGUOUS = 32
    NUMERIC_INVALID = 64
    DENSITY_UNDEFINED = 128


def _positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _positive_float(value: float, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive finite number")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{field_name} must be a positive finite number")


def _non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


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


def _require_dtype(value: Any, expected: np.dtype[Any], field_name: str) -> None:
    dtype = getattr(value, "dtype", None)
    if dtype is None or np.dtype(dtype) != expected:
        actual = "missing" if dtype is None else str(dtype)
        raise ValueError(f"{field_name} must have dtype {expected}, got {actual}")


def _require_shape(value: Any, expected: tuple[int, ...], field_name: str) -> None:
    if _array_shape(value, field_name) != expected:
        raise ValueError(f"{field_name} must have shape {expected}")


def _shape_dtype(value: Any) -> dict[str, Any]:
    return {"shape": list(_array_shape(value, "array")), "dtype": str(value.dtype)}


@dataclass(frozen=True)
class GeometryFeatureConfig:
    """Controlled-memory configuration for KNN PCA feature extraction."""

    source_id: str = "scene_simple"
    k_neighbors: int = 32
    min_neighbors: int = 8
    query_batch_size: int = 12_000
    workers: int = 1
    density_epsilon_m: float = 1e-6
    feature_epsilon_m2: float = 1e-12
    eigenvalue_absolute_tolerance_m2: float = 1e-12
    eigenvalue_relative_tolerance: float = 1e-10
    direction_gap_relative_tolerance: float = 1e-3
    record_memory: bool = True

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("GeometryFeatureConfig.source_id must be non-empty")
        _positive_int(self.k_neighbors, "GeometryFeatureConfig.k_neighbors")
        if self.k_neighbors > np.iinfo(np.uint16).max:
            raise ValueError("GeometryFeatureConfig.k_neighbors must fit uint16")
        _positive_int(self.min_neighbors, "GeometryFeatureConfig.min_neighbors")
        if self.min_neighbors > self.k_neighbors:
            raise ValueError("GeometryFeatureConfig.min_neighbors cannot exceed k_neighbors")
        _positive_int(self.query_batch_size, "GeometryFeatureConfig.query_batch_size")
        _positive_int(self.workers, "GeometryFeatureConfig.workers")
        for field_name in (
            "density_epsilon_m",
            "feature_epsilon_m2",
            "eigenvalue_absolute_tolerance_m2",
            "eigenvalue_relative_tolerance",
            "direction_gap_relative_tolerance",
        ):
            _positive_float(getattr(self, field_name), f"GeometryFeatureConfig.{field_name}")
            object.__setattr__(self, field_name, float(getattr(self, field_name)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "k_neighbors": self.k_neighbors,
            "min_neighbors": self.min_neighbors,
            "query_batch_size": self.query_batch_size,
            "workers": self.workers,
            "density_epsilon_m": self.density_epsilon_m,
            "feature_epsilon_m2": self.feature_epsilon_m2,
            "eigenvalue_absolute_tolerance_m2": self.eigenvalue_absolute_tolerance_m2,
            "eigenvalue_relative_tolerance": self.eigenvalue_relative_tolerance,
            "direction_gap_relative_tolerance": self.direction_gap_relative_tolerance,
            "record_memory": self.record_memory,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GeometryFeatureConfig:
        allowed = {
            "source_id",
            "k_neighbors",
            "min_neighbors",
            "query_batch_size",
            "workers",
            "density_epsilon_m",
            "feature_epsilon_m2",
            "eigenvalue_absolute_tolerance_m2",
            "eigenvalue_relative_tolerance",
            "direction_gap_relative_tolerance",
            "record_memory",
        }
        reject_unknown_fields(data, allowed, cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class PointFeatureSet:
    """A2A features that preserve the exact A1 point ordering and traceability."""

    source_id: str
    manifest_source_id: str
    xyz_local: Any
    origin_xyz: Any
    original_index: Any
    sample_index: Any
    linearity: Any
    planarity: Any
    scattering: Any
    verticality: Any
    local_density: Any
    height_m: Any
    neighbor_count: Any
    eigenvalues: Any
    principal_direction: Any
    normal: Any
    quality_flags: Any

    def __post_init__(self) -> None:
        if not self.source_id or not self.manifest_source_id:
            raise ValueError("PointFeatureSet source identifiers must be non-empty")
        xyz_shape = _array_shape(self.xyz_local, "PointFeatureSet.xyz_local")
        if len(xyz_shape) != 2 or xyz_shape[1] != 3:
            raise ValueError("PointFeatureSet.xyz_local must have shape (M, 3)")
        point_count = xyz_shape[0]
        _require_dtype(self.xyz_local, np.dtype(np.float32), "PointFeatureSet.xyz_local")
        _require_shape(self.origin_xyz, (3,), "PointFeatureSet.origin_xyz")
        _require_dtype(self.origin_xyz, np.dtype(np.float64), "PointFeatureSet.origin_xyz")
        for field_name in ("original_index", "sample_index"):
            value = getattr(self, field_name)
            _require_shape(value, (point_count,), f"PointFeatureSet.{field_name}")
            _require_dtype(value, np.dtype(np.int64), f"PointFeatureSet.{field_name}")
        for field_name in (
            "linearity",
            "planarity",
            "scattering",
            "verticality",
            "local_density",
            "height_m",
        ):
            value = getattr(self, field_name)
            _require_shape(value, (point_count,), f"PointFeatureSet.{field_name}")
            _require_dtype(value, np.dtype(np.float32), f"PointFeatureSet.{field_name}")
        _require_shape(self.neighbor_count, (point_count,), "PointFeatureSet.neighbor_count")
        _require_dtype(self.neighbor_count, np.dtype(np.uint16), "PointFeatureSet.neighbor_count")
        for field_name in ("eigenvalues", "principal_direction", "normal"):
            value = getattr(self, field_name)
            _require_shape(value, (point_count, 3), f"PointFeatureSet.{field_name}")
            _require_dtype(value, np.dtype(np.float32), f"PointFeatureSet.{field_name}")
        _require_shape(self.quality_flags, (point_count,), "PointFeatureSet.quality_flags")
        _require_dtype(self.quality_flags, np.dtype(np.uint16), "PointFeatureSet.quality_flags")
        for field_name in (
            "xyz_local",
            "origin_xyz",
            "linearity",
            "planarity",
            "scattering",
            "verticality",
            "local_density",
            "height_m",
            "eigenvalues",
            "principal_direction",
            "normal",
        ):
            if not np.isfinite(np.asarray(getattr(self, field_name))).all():
                raise ValueError(f"PointFeatureSet.{field_name} must contain finite values")

    @property
    def point_count(self) -> int:
        return _array_shape(self.xyz_local, "PointFeatureSet.xyz_local")[0]

    def array_metadata(self) -> dict[str, Any]:
        units = {
            "xyz_local": "m",
            "origin_xyz": "m",
            "linearity": "unitless",
            "planarity": "unitless",
            "scattering": "unitless",
            "verticality": "unitless",
            "local_density": "points/m^3",
            "height_m": "m",
            "neighbor_count": "points",
            "eigenvalues": "m^2",
            "principal_direction": "unitless",
            "normal": "unitless",
            "quality_flags": "bitmask",
        }
        values = {
            "xyz_local": self.xyz_local,
            "origin_xyz": self.origin_xyz,
            "original_index": self.original_index,
            "sample_index": self.sample_index,
            "linearity": self.linearity,
            "planarity": self.planarity,
            "scattering": self.scattering,
            "verticality": self.verticality,
            "local_density": self.local_density,
            "height_m": self.height_m,
            "neighbor_count": self.neighbor_count,
            "eigenvalues": self.eigenvalues,
            "principal_direction": self.principal_direction,
            "normal": self.normal,
            "quality_flags": self.quality_flags,
        }
        return {
            name: {**_shape_dtype(value), "unit": units.get(name, "index")}
            for name, value in values.items()
        }


@dataclass(frozen=True)
class GeometryFeatureStatistics:
    """Runtime, memory, and quality counters for A2A feature extraction."""

    input_point_count: int
    output_point_count: int
    pca_computed_count: int
    insufficient_neighbor_count: int
    degenerate_covariance_count: int
    duplicate_neighbor_coordinate_count: int
    negative_eigenvalue_clipped_count: int
    normal_direction_ambiguous_count: int
    principal_direction_ambiguous_count: int
    density_undefined_count: int
    numeric_invalid_count: int
    requested_k: int
    query_batch_size: int
    elapsed_seconds: float
    rss_start_bytes: int | None = None
    rss_peak_bytes: int | None = None
    rss_end_bytes: int | None = None
    kd_tree_coordinate_bytes: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "input_point_count",
            "output_point_count",
            "pca_computed_count",
            "insufficient_neighbor_count",
            "degenerate_covariance_count",
            "duplicate_neighbor_coordinate_count",
            "negative_eigenvalue_clipped_count",
            "normal_direction_ambiguous_count",
            "principal_direction_ambiguous_count",
            "density_undefined_count",
            "numeric_invalid_count",
        ):
            _non_negative_int(getattr(self, field_name), f"GeometryFeatureStatistics.{field_name}")
        _positive_int(self.requested_k, "GeometryFeatureStatistics.requested_k")
        _positive_int(self.query_batch_size, "GeometryFeatureStatistics.query_batch_size")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("GeometryFeatureStatistics.elapsed_seconds must be non-negative")
        for field_name in (
            "rss_start_bytes",
            "rss_peak_bytes",
            "rss_end_bytes",
            "kd_tree_coordinate_bytes",
        ):
            _optional_non_negative_int(
                getattr(self, field_name), f"GeometryFeatureStatistics.{field_name}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "input_point_count",
                "output_point_count",
                "pca_computed_count",
                "insufficient_neighbor_count",
                "degenerate_covariance_count",
                "duplicate_neighbor_coordinate_count",
                "negative_eigenvalue_clipped_count",
                "normal_direction_ambiguous_count",
                "principal_direction_ambiguous_count",
                "density_undefined_count",
                "numeric_invalid_count",
                "requested_k",
                "query_batch_size",
                "elapsed_seconds",
                "rss_start_bytes",
                "rss_peak_bytes",
                "rss_end_bytes",
                "kd_tree_coordinate_bytes",
            )
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GeometryFeatureStatistics:
        required = {
            "input_point_count",
            "output_point_count",
            "pca_computed_count",
            "insufficient_neighbor_count",
            "degenerate_covariance_count",
            "duplicate_neighbor_coordinate_count",
            "negative_eigenvalue_clipped_count",
            "normal_direction_ambiguous_count",
            "principal_direction_ambiguous_count",
            "density_undefined_count",
            "numeric_invalid_count",
            "requested_k",
            "query_batch_size",
            "elapsed_seconds",
        }
        allowed = required | {
            "rss_start_bytes",
            "rss_peak_bytes",
            "rss_end_bytes",
            "kd_tree_coordinate_bytes",
        }
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, allowed, cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class GeometryFeatureResult:
    """JSON-safe A2A artifact metadata and provenance."""

    source_id: str
    manifest_source_id: str
    input_a1_npz_path: Path
    input_a1_json_path: Path
    input_a1_npz_sha256: str
    input_a1_json_sha256: str
    output_npz_path: Path
    output_json_path: Path
    output_npz_sha256: str
    output_npz_bytes: int
    origin_xyz: tuple[float, float, float]
    config: GeometryFeatureConfig
    statistics: GeometryFeatureStatistics
    arrays: Mapping[str, Any]
    quality_flag_counts: Mapping[str, Any]
    formulas: Mapping[str, Any]
    algorithm_version: str
    created_at: str

    def __post_init__(self) -> None:
        if not self.source_id or not self.manifest_source_id:
            raise ValueError("GeometryFeatureResult source identifiers must be non-empty")
        for field_name in (
            "input_a1_npz_path",
            "input_a1_json_path",
            "output_npz_path",
            "output_json_path",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))
        for field_name in ("input_a1_npz_sha256", "input_a1_json_sha256", "output_npz_sha256"):
            value = getattr(self, field_name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"GeometryFeatureResult.{field_name} must be a SHA-256 hex digest")
        _non_negative_int(self.output_npz_bytes, "GeometryFeatureResult.output_npz_bytes")
        origin = tuple(float(value) for value in self.origin_xyz)
        if len(origin) != 3 or not all(math.isfinite(value) for value in origin):
            raise ValueError("GeometryFeatureResult.origin_xyz must contain three finite values")
        object.__setattr__(self, "origin_xyz", origin)
        if not isinstance(self.config, GeometryFeatureConfig):
            raise ValueError("GeometryFeatureResult.config must be a GeometryFeatureConfig")
        if not isinstance(self.statistics, GeometryFeatureStatistics):
            raise ValueError("GeometryFeatureResult.statistics must be a GeometryFeatureStatistics")
        if not self.algorithm_version or not self.created_at:
            raise ValueError(
                "GeometryFeatureResult algorithm_version and created_at must be non-empty"
            )
        object.__setattr__(self, "arrays", json_safe_mapping(self.arrays, "arrays"))
        object.__setattr__(
            self,
            "quality_flag_counts",
            json_safe_mapping(self.quality_flag_counts, "quality_flag_counts"),
        )
        object.__setattr__(self, "formulas", json_safe_mapping(self.formulas, "formulas"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "manifest_source_id": self.manifest_source_id,
            "input_a1_npz_path": str(self.input_a1_npz_path),
            "input_a1_json_path": str(self.input_a1_json_path),
            "input_a1_npz_sha256": self.input_a1_npz_sha256,
            "input_a1_json_sha256": self.input_a1_json_sha256,
            "output_npz_path": str(self.output_npz_path),
            "output_json_path": str(self.output_json_path),
            "output_npz_sha256": self.output_npz_sha256,
            "output_npz_bytes": self.output_npz_bytes,
            "origin_xyz": list(self.origin_xyz),
            "config": self.config.to_dict(),
            "statistics": self.statistics.to_dict(),
            "arrays": dict(self.arrays),
            "quality_flag_counts": dict(self.quality_flag_counts),
            "formulas": dict(self.formulas),
            "algorithm_version": self.algorithm_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GeometryFeatureResult:
        required = {
            "source_id",
            "manifest_source_id",
            "input_a1_npz_path",
            "input_a1_json_path",
            "input_a1_npz_sha256",
            "input_a1_json_sha256",
            "output_npz_path",
            "output_json_path",
            "output_npz_sha256",
            "output_npz_bytes",
            "origin_xyz",
            "config",
            "statistics",
            "arrays",
            "quality_flag_counts",
            "formulas",
            "algorithm_version",
            "created_at",
        }
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, required, cls.__name__)
        return cls(
            source_id=data["source_id"],
            manifest_source_id=data["manifest_source_id"],
            input_a1_npz_path=Path(data["input_a1_npz_path"]),
            input_a1_json_path=Path(data["input_a1_json_path"]),
            input_a1_npz_sha256=data["input_a1_npz_sha256"],
            input_a1_json_sha256=data["input_a1_json_sha256"],
            output_npz_path=Path(data["output_npz_path"]),
            output_json_path=Path(data["output_json_path"]),
            output_npz_sha256=data["output_npz_sha256"],
            output_npz_bytes=data["output_npz_bytes"],
            origin_xyz=tuple(data["origin_xyz"]),
            config=GeometryFeatureConfig.from_dict(data["config"]),
            statistics=GeometryFeatureStatistics.from_dict(data["statistics"]),
            arrays=data["arrays"],
            quality_flag_counts=data["quality_flag_counts"],
            formulas=data["formulas"],
            algorithm_version=data["algorithm_version"],
            created_at=data["created_at"],
        )
