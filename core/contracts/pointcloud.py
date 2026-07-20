"""Minimal point cloud contracts for LAS IO and crop pipeline stages.

Coordinates, distances, scales, offsets, and bounds are expressed in meters.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from core.contracts._json import json_safe_mapping, reject_unknown_fields, require_fields

Vector3 = tuple[float, float, float]
SourceRole = Literal["raw", "sample", "processed"]
CoordinateMode = Literal["source", "local"]


def _as_vector3(value: Any, field_name: str) -> Vector3:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three numeric meter values")
    vector = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{field_name} must contain only finite meter values")
    return vector


def _as_optional_bounds(value: Any, field_name: str) -> tuple[Vector3, Vector3] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{field_name} must contain min_xyz and max_xyz")
    min_xyz = _as_vector3(value[0], f"{field_name}.min_xyz")
    max_xyz = _as_vector3(value[1], f"{field_name}.max_xyz")
    if any(
        min_value > max_value for min_value, max_value in zip(min_xyz, max_xyz, strict=True)
    ):
        raise ValueError(f"{field_name}.min_xyz cannot exceed max_xyz")
    return min_xyz, max_xyz


def _non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _array_length(value: Any, field_name: str) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 0:
            raise ValueError(f"{field_name} must have a length")
        return int(shape[0])
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return len(value)
    return None


def _validate_xyz(value: Any, point_count: int) -> None:
    if value is None:
        return
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) != 2 or shape[1] != 3:
            raise ValueError("PointCloudChunk.xyz must have shape (N, 3) in meters")
        if int(shape[0]) != point_count:
            raise ValueError("PointCloudChunk.point_count must match xyz length")
        return
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError("PointCloudChunk.xyz must be a sequence or array shaped (N, 3)")
    if len(value) != point_count:
        raise ValueError("PointCloudChunk.point_count must match xyz length")
    for row in value:
        _as_vector3(row, "PointCloudChunk.xyz row")


@dataclass(frozen=True)
class PointCloudSource:
    """Metadata for a point cloud path without opening or loading points."""

    source_id: str
    path: Path
    role: SourceRole
    size_bytes: int | None = None
    readonly_required: bool = True
    sha256: str | None = None
    crs_wkt: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("PointCloudSource.source_id must be non-empty")
        object.__setattr__(self, "path", Path(self.path))
        if self.role not in {"raw", "sample", "processed"}:
            raise ValueError("PointCloudSource.role must be one of: raw, sample, processed")
        if self.role == "raw" and not self.readonly_required:
            raise ValueError("PointCloudSource.readonly_required must be True for raw sources")
        if self.size_bytes is not None:
            _non_negative_int(self.size_bytes, "PointCloudSource.size_bytes")
        object.__setattr__(self, "metadata", json_safe_mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path": str(self.path),
            "role": self.role,
            "size_bytes": self.size_bytes,
            "readonly_required": self.readonly_required,
            "sha256": self.sha256,
            "crs_wkt": self.crs_wkt,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PointCloudSource:
        required = {"source_id", "path", "role"}
        allowed = required | {
            "size_bytes",
            "readonly_required",
            "sha256",
            "crs_wkt",
            "metadata",
        }
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, allowed, cls.__name__)
        return cls(
            source_id=data["source_id"],
            path=Path(data["path"]),
            role=data["role"],
            size_bytes=data.get("size_bytes"),
            readonly_required=data.get("readonly_required", True),
            sha256=data.get("sha256"),
            crs_wkt=data.get("crs_wkt"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class LasHeaderSummary:
    """LAS header-only summary; no point arrays are loaded by this contract."""

    source_id: str
    source_path: Path
    file_size_bytes: int
    las_version: str
    point_format_id: int
    point_record_length_bytes: int
    point_count: int
    min_xyz: Vector3
    max_xyz: Vector3
    scale: Vector3
    offset: Vector3
    dimensions: tuple[str, ...]
    extra_dimensions: tuple[str, ...] = ()
    has_rgb: bool = False
    has_intensity: bool = False
    has_classification: bool = False
    estimated_record_memory_bytes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("LasHeaderSummary.source_id must be non-empty")
        object.__setattr__(self, "source_path", Path(self.source_path))
        _non_negative_int(self.file_size_bytes, "LasHeaderSummary.file_size_bytes")
        if not self.las_version:
            raise ValueError("LasHeaderSummary.las_version must be non-empty")
        _non_negative_int(self.point_format_id, "LasHeaderSummary.point_format_id")
        _positive_int(
            self.point_record_length_bytes,
            "LasHeaderSummary.point_record_length_bytes",
        )
        _non_negative_int(self.point_count, "LasHeaderSummary.point_count")
        min_xyz = _as_vector3(self.min_xyz, "LasHeaderSummary.min_xyz")
        max_xyz = _as_vector3(self.max_xyz, "LasHeaderSummary.max_xyz")
        if any(
            min_value > max_value for min_value, max_value in zip(min_xyz, max_xyz, strict=True)
        ):
            raise ValueError("LasHeaderSummary.min_xyz cannot exceed max_xyz")
        object.__setattr__(self, "min_xyz", min_xyz)
        object.__setattr__(self, "max_xyz", max_xyz)
        object.__setattr__(self, "scale", _as_vector3(self.scale, "LasHeaderSummary.scale"))
        object.__setattr__(self, "offset", _as_vector3(self.offset, "LasHeaderSummary.offset"))
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        object.__setattr__(self, "extra_dimensions", tuple(self.extra_dimensions))
        if self.estimated_record_memory_bytes is not None:
            _non_negative_int(
                self.estimated_record_memory_bytes,
                "LasHeaderSummary.estimated_record_memory_bytes",
            )
        object.__setattr__(self, "metadata", json_safe_mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_path": str(self.source_path),
            "file_size_bytes": self.file_size_bytes,
            "las_version": self.las_version,
            "point_format_id": self.point_format_id,
            "point_record_length_bytes": self.point_record_length_bytes,
            "point_count": self.point_count,
            "min_xyz": list(self.min_xyz),
            "max_xyz": list(self.max_xyz),
            "scale": list(self.scale),
            "offset": list(self.offset),
            "dimensions": list(self.dimensions),
            "extra_dimensions": list(self.extra_dimensions),
            "has_rgb": self.has_rgb,
            "has_intensity": self.has_intensity,
            "has_classification": self.has_classification,
            "estimated_record_memory_bytes": self.estimated_record_memory_bytes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LasHeaderSummary:
        required = {
            "source_id",
            "source_path",
            "file_size_bytes",
            "las_version",
            "point_format_id",
            "point_record_length_bytes",
            "point_count",
            "min_xyz",
            "max_xyz",
            "scale",
            "offset",
            "dimensions",
        }
        allowed = required | {
            "extra_dimensions",
            "has_rgb",
            "has_intensity",
            "has_classification",
            "estimated_record_memory_bytes",
            "metadata",
        }
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, allowed, cls.__name__)
        return cls(
            source_id=data["source_id"],
            source_path=Path(data["source_path"]),
            file_size_bytes=data["file_size_bytes"],
            las_version=data["las_version"],
            point_format_id=data["point_format_id"],
            point_record_length_bytes=data["point_record_length_bytes"],
            point_count=data["point_count"],
            min_xyz=data["min_xyz"],
            max_xyz=data["max_xyz"],
            scale=data["scale"],
            offset=data["offset"],
            dimensions=tuple(data["dimensions"]),
            extra_dimensions=tuple(data.get("extra_dimensions", ())),
            has_rgb=data.get("has_rgb", False),
            has_intensity=data.get("has_intensity", False),
            has_classification=data.get("has_classification", False),
            estimated_record_memory_bytes=data.get("estimated_record_memory_bytes"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class PointCloudChunk:
    """In-memory chunk container for later streaming LAS readers.

    ``xyz`` may be a NumPy array shaped ``(N, 3)`` with meter coordinates.
    ``original_index`` is retained to trace chunk rows back to source LAS points.
    """

    source_id: str
    chunk_id: int
    start_index: int
    point_count: int
    xyz: Any | None = None
    original_index: Any | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    bounds_xyz: tuple[Vector3, Vector3] | None = None
    is_last: bool = False

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("PointCloudChunk.source_id must be non-empty")
        _non_negative_int(self.chunk_id, "PointCloudChunk.chunk_id")
        _non_negative_int(self.start_index, "PointCloudChunk.start_index")
        _non_negative_int(self.point_count, "PointCloudChunk.point_count")
        _validate_xyz(self.xyz, self.point_count)
        if self.original_index is not None:
            original_length = _array_length(self.original_index, "PointCloudChunk.original_index")
            if original_length is None:
                raise ValueError("PointCloudChunk.original_index must expose a length")
            if original_length != self.point_count:
                raise ValueError("PointCloudChunk.original_index length must match point_count")
        object.__setattr__(
            self,
            "bounds_xyz",
            _as_optional_bounds(self.bounds_xyz, "PointCloudChunk.bounds_xyz"),
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "start_index": self.start_index,
            "point_count": self.point_count,
            "has_xyz": self.xyz is not None,
            "has_original_index": self.original_index is not None,
            "attribute_names": sorted(self.attributes),
            "bounds_xyz": (
                [list(self.bounds_xyz[0]), list(self.bounds_xyz[1])]
                if self.bounds_xyz is not None
                else None
            ),
            "is_last": self.is_last,
        }


@dataclass(frozen=True)
class CropBounds:
    """Axis-aligned crop bounds in meters."""

    min_xyz: Vector3
    max_xyz: Vector3
    coordinate_mode: CoordinateMode = "source"
    origin_xyz: Vector3 | None = None

    def __post_init__(self) -> None:
        min_xyz = _as_vector3(self.min_xyz, "CropBounds.min_xyz")
        max_xyz = _as_vector3(self.max_xyz, "CropBounds.max_xyz")
        if any(
            min_value >= max_value for min_value, max_value in zip(min_xyz, max_xyz, strict=True)
        ):
            raise ValueError("CropBounds.min_xyz must be strictly less than max_xyz on every axis")
        if self.coordinate_mode not in {"source", "local"}:
            raise ValueError("CropBounds.coordinate_mode must be one of: source, local")
        if self.coordinate_mode == "local" and self.origin_xyz is None:
            raise ValueError("CropBounds.origin_xyz is required when coordinate_mode is local")
        origin_xyz = None if self.origin_xyz is None else _as_vector3(self.origin_xyz, "origin_xyz")
        object.__setattr__(self, "min_xyz", min_xyz)
        object.__setattr__(self, "max_xyz", max_xyz)
        object.__setattr__(self, "origin_xyz", origin_xyz)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_xyz": list(self.min_xyz),
            "max_xyz": list(self.max_xyz),
            "coordinate_mode": self.coordinate_mode,
            "origin_xyz": list(self.origin_xyz) if self.origin_xyz is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CropBounds:
        required = {"min_xyz", "max_xyz"}
        allowed = required | {"coordinate_mode", "origin_xyz"}
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, allowed, cls.__name__)
        return cls(
            min_xyz=data["min_xyz"],
            max_xyz=data["max_xyz"],
            coordinate_mode=data.get("coordinate_mode", "source"),
            origin_xyz=data.get("origin_xyz"),
        )


@dataclass(frozen=True)
class CropResult:
    """Result metadata for a crop operation without performing the crop."""

    source_id: str
    bounds: CropBounds
    input_point_count: int
    output_point_count: int
    output_path: Path
    header_preserved: bool
    dimension_names: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    started_at: str | None = None
    finished_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("CropResult.source_id must be non-empty")
        if not isinstance(self.bounds, CropBounds):
            raise ValueError("CropResult.bounds must be a CropBounds")
        _non_negative_int(self.input_point_count, "CropResult.input_point_count")
        _non_negative_int(self.output_point_count, "CropResult.output_point_count")
        if self.output_point_count > self.input_point_count:
            raise ValueError("CropResult.output_point_count cannot exceed input_point_count")
        object.__setattr__(self, "output_path", Path(self.output_path))
        object.__setattr__(self, "dimension_names", tuple(self.dimension_names))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "metadata", json_safe_mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "bounds": self.bounds.to_dict(),
            "input_point_count": self.input_point_count,
            "output_point_count": self.output_point_count,
            "output_path": str(self.output_path),
            "header_preserved": self.header_preserved,
            "dimension_names": list(self.dimension_names),
            "warnings": list(self.warnings),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CropResult:
        required = {
            "source_id",
            "bounds",
            "input_point_count",
            "output_point_count",
            "output_path",
            "header_preserved",
            "dimension_names",
        }
        allowed = required | {"warnings", "started_at", "finished_at", "metadata"}
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, allowed, cls.__name__)
        return cls(
            source_id=data["source_id"],
            bounds=CropBounds.from_dict(data["bounds"]),
            input_point_count=data["input_point_count"],
            output_point_count=data["output_point_count"],
            output_path=Path(data["output_path"]),
            header_preserved=data["header_preserved"],
            dimension_names=tuple(data["dimension_names"]),
            warnings=tuple(data.get("warnings", ())),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            metadata=data.get("metadata", {}),
        )
