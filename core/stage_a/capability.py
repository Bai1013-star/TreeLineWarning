"""Vendor-independent input capability inspection.

Inspection is metadata/header-only.  It does not read a complete point cloud
and it never infers a CRS, unit, or camera capability from a filename.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from core.stage_a.models import CoordinateDeclaration, SourceDescriptor, UnitStatus


class CapabilityState(str, Enum):
    AVAILABLE = "AVAILABLE"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    UNKNOWN = "UNKNOWN"
    REQUIRES_CONFIRMATION = "REQUIRES_CONFIRMATION"


CAPABILITY_NAMES = (
    "POINT_CLOUD",
    "XYZ",
    "RGB",
    "INTENSITY",
    "CLASSIFICATION",
    "GPS_TIME",
    "RETURN_NUMBER",
    "SCAN_ANGLE",
    "CRS",
    "HORIZONTAL_CRS",
    "VERTICAL_CRS",
    "LINEAR_UNIT",
    "TRAJECTORY",
    "GNSS",
    "CAMERA",
    "CAMERA_POSE",
    "CALIBRATION",
    "GIS",
)


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    source: SourceDescriptor
    capabilities: Mapping[str, CapabilityState]
    coordinate_declaration: CoordinateDeclaration
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.source) is not SourceDescriptor:
            raise ValueError("source must be a SourceDescriptor")
        values = dict(self.capabilities)
        unknown = set(values) - set(CAPABILITY_NAMES)
        if unknown:
            raise ValueError(f"unsupported capability names: {sorted(unknown)}")
        if any(type(value) is not CapabilityState for value in values.values()):
            raise ValueError("capabilities must contain CapabilityState values")
        missing = set(CAPABILITY_NAMES) - set(values)
        if missing:
            raise ValueError(f"capability report is missing fields: {sorted(missing)}")
        if type(self.coordinate_declaration) is not CoordinateDeclaration:
            raise ValueError("coordinate_declaration must be a CoordinateDeclaration")

    @property
    def requires_confirmation(self) -> bool:
        return any(
            value in {CapabilityState.UNKNOWN, CapabilityState.REQUIRES_CONFIRMATION}
            for value in self.capabilities.values()
        ) or self.coordinate_declaration.unit_status in {
            UnitStatus.UNKNOWN,
            UnitStatus.REQUIRES_CONFIRMATION,
        }

    def has(self, capability: str) -> bool:
        return self.capabilities.get(capability) is CapabilityState.AVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "capabilities": {key: value.value for key, value in self.capabilities.items()},
            "coordinate_declaration": self.coordinate_declaration.to_dict(),
            "details": dict(self.details),
            "requires_confirmation": self.requires_confirmation,
        }


def _optional_dimension(dimensions: set[str], *names: str) -> CapabilityState:
    return (
        CapabilityState.AVAILABLE
        if any(name in dimensions for name in names)
        else CapabilityState.NOT_AVAILABLE
    )


def inspect_point_cloud(
    path: Path | str, *, source_id: str | None = None, declared_unit: UnitStatus | None = None
) -> CapabilityReport:
    """Inspect a LAS/LAZ header and point-format dimensions without full reads."""

    import laspy
    from laspy.errors import LaspyException

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if resolved.suffix.lower() not in {".las", ".laz"}:
        raise ValueError("point-cloud capability inspection currently accepts LAS/LAZ")
    source = SourceDescriptor(
        source_id=source_id
        or f"source-{hashlib.sha256(str(resolved).encode('utf-8')).hexdigest()[:16]}",
        path=str(resolved),
        format=resolved.suffix.lower()[1:].upper(),
        file_size_bytes=resolved.stat().st_size,
    )
    try:
        reader_context = laspy.open(resolved, mode="r")
    except LaspyException as exc:
        raise ValueError("point-cloud header is not readable") from exc
    with reader_context as reader:
        header = reader.header
        dimensions = {str(item).lower() for item in header.point_format.dimension_names}
        parsed_crs = None
        try:
            parsed_crs = header.parse_crs()
        except (AttributeError, ValueError, RuntimeError):
            parsed_crs = None
        horizontal_crs = parsed_crs.to_string() if parsed_crs is not None else None
        linear_unit = None
        if parsed_crs is not None:
            axis_info = getattr(parsed_crs, "axis_info", ()) or ()
            units = {str(getattr(axis, "unit_name", "")).lower() for axis in axis_info}
            if "metre" in units or "meter" in units:
                linear_unit = "m"
            elif "foot" in units or "feet" in units:
                linear_unit = "ft"
        unit_status = declared_unit
        if unit_status is None:
            if linear_unit == "m":
                unit_status = UnitStatus.METRE
            elif linear_unit == "ft":
                unit_status = UnitStatus.FOOT
            else:
                unit_status = UnitStatus.REQUIRES_CONFIRMATION
        coordinate = CoordinateDeclaration(
            horizontal_crs=horizontal_crs,
            vertical_reference=None,
            linear_unit=linear_unit,
            unit_status=unit_status,
            coordinate_kind="GEOREFERENCED" if horizontal_crs else "LOCAL",
        )
        capabilities = {
            "POINT_CLOUD": CapabilityState.AVAILABLE,
            "XYZ": CapabilityState.AVAILABLE
            if {"x", "y", "z"}.issubset(dimensions)
            else CapabilityState.UNKNOWN,
            "RGB": CapabilityState.AVAILABLE
            if {"red", "green", "blue"}.issubset(dimensions)
            else CapabilityState.NOT_AVAILABLE,
            "INTENSITY": _optional_dimension(dimensions, "intensity"),
            "CLASSIFICATION": _optional_dimension(dimensions, "classification"),
            "GPS_TIME": _optional_dimension(dimensions, "gps_time"),
            "RETURN_NUMBER": _optional_dimension(dimensions, "return_number"),
            "SCAN_ANGLE": _optional_dimension(dimensions, "scan_angle", "scan_angle_rank"),
            "CRS": CapabilityState.AVAILABLE if parsed_crs is not None else CapabilityState.UNKNOWN,
            "HORIZONTAL_CRS": CapabilityState.AVAILABLE
            if horizontal_crs
            else CapabilityState.UNKNOWN,
            "VERTICAL_CRS": CapabilityState.AVAILABLE
            if parsed_crs is not None and getattr(parsed_crs, "is_compound", False)
            else CapabilityState.UNKNOWN,
            "LINEAR_UNIT": CapabilityState.AVAILABLE
            if linear_unit
            else CapabilityState.REQUIRES_CONFIRMATION,
            "TRAJECTORY": CapabilityState.NOT_AVAILABLE,
            "GNSS": CapabilityState.NOT_AVAILABLE,
            "CAMERA": CapabilityState.NOT_AVAILABLE,
            "CAMERA_POSE": CapabilityState.NOT_AVAILABLE,
            "CALIBRATION": CapabilityState.NOT_AVAILABLE,
            "GIS": CapabilityState.NOT_AVAILABLE,
        }
        details = {
            "format": source.format,
            "point_count": int(header.point_count),
            "point_format_id": int(header.point_format.id),
            "dimensions": sorted(dimensions),
            "bounds_min_xyz": [float(value) for value in header.mins],
            "bounds_max_xyz": [float(value) for value in header.maxs],
            "crs_string": horizontal_crs,
        }
    return CapabilityReport(source, capabilities, coordinate, details)


__all__ = ["CAPABILITY_NAMES", "CapabilityReport", "CapabilityState", "inspect_point_cloud"]
