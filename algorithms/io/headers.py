"""Read-only LAS header summaries.

This module intentionally uses ``laspy.open(...).header`` only. It must not call
``laspy.read()``, ``reader.read()``, or access full point arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import laspy
from core.contracts import LasHeaderSummary, PointCloudSource, UnifiedError

from algorithms.io.discovery import POINT_CLOUD_SUFFIXES, stable_source_id


class LasIoError(Exception):
    """Exception carrying a JSON-safe unified IO error."""

    def __init__(self, error: UnifiedError) -> None:
        super().__init__(error.message)
        self.error = error


def read_las_header_summary_for_source(source: PointCloudSource) -> LasHeaderSummary:
    """Read a header summary for a discovered source while preserving its source id."""
    if source.role != "raw":
        raise _error(
            code="LAS_SOURCE_ROLE_INVALID",
            message="LAS header summary expects a raw PointCloudSource",
            details={"source_id": source.source_id, "role": source.role},
        )
    return read_las_header_summary(source.path, source_id=source.source_id)


def read_las_header_summary(path: Path | str, source_id: str | None = None) -> LasHeaderSummary:
    """Read only LAS/LAZ header metadata and return a contract summary."""
    file_path = _validate_las_input(Path(path))
    stat = file_path.stat()
    stable_id = source_id or stable_source_id(file_path, file_path.parent)

    try:
        with laspy.open(file_path, mode="r") as reader:
            header = reader.header
            point_format = header.point_format
            point_record_length = int(point_format.size)
            point_count = int(header.point_count)
            dimensions = tuple(str(name) for name in point_format.dimension_names)
            extra_dimensions = tuple(str(name) for name in point_format.extra_dimension_names)
            warnings = _header_warnings(point_count)
            metadata = {
                "warnings": warnings,
                "vlrs": _safe_vlr_summary(header),
            }
            return LasHeaderSummary(
                source_id=stable_id,
                source_path=file_path,
                file_size_bytes=stat.st_size,
                las_version=str(header.version),
                point_format_id=int(point_format.id),
                point_record_length_bytes=point_record_length,
                point_count=point_count,
                min_xyz=_triple(header.mins),
                max_xyz=_triple(header.maxs),
                scale=_triple(header.scales),
                offset=_triple(header.offsets),
                dimensions=dimensions,
                extra_dimensions=extra_dimensions,
                has_rgb=_has_all_dimensions(dimensions, {"red", "green", "blue"}),
                has_intensity="intensity" in dimensions,
                has_classification="classification" in dimensions,
                estimated_record_memory_bytes=point_count * point_record_length,
                metadata=metadata,
            )
    except LasIoError:
        raise
    except Exception as exc:
        raise _error(
            code="LAS_HEADER_READ_FAILED",
            message=f"Failed to read LAS header: {file_path}",
            recoverable=False,
            details={"path": str(file_path), "cause_type": type(exc).__name__},
            cause=str(exc),
        ) from exc


def _validate_las_input(path: Path) -> Path:
    if not path.exists():
        raise _error(
            code="LAS_INPUT_NOT_FOUND",
            message=f"LAS input does not exist: {path}",
            details={"path": str(path)},
        )
    if path.is_dir():
        raise _error(
            code="LAS_INPUT_IS_DIRECTORY",
            message=f"LAS input must be a file, not a directory: {path}",
            details={"path": str(path)},
        )
    if path.suffix.lower() not in POINT_CLOUD_SUFFIXES:
        raise _error(
            code="LAS_INPUT_EXTENSION_INVALID",
            message=f"LAS input must have .las or .laz extension: {path}",
            details={"path": str(path), "suffix": path.suffix},
        )
    return path


def _triple(values: Any) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _has_all_dimensions(dimensions: tuple[str, ...], expected: set[str]) -> bool:
    return expected.issubset(set(dimensions))


def _header_warnings(point_count: int) -> list[str]:
    if point_count == 0:
        return ["LAS file contains zero points"]
    return []


def _safe_vlr_summary(header: Any) -> list[dict[str, Any]]:
    vlrs = getattr(header, "vlrs", ())
    result: list[dict[str, Any]] = []
    for vlr in vlrs:
        result.append(
            {
                "type": type(vlr).__name__,
                "user_id": str(getattr(vlr, "user_id", "")),
                "record_id": int(getattr(vlr, "record_id", -1)),
                "description": str(getattr(vlr, "description", "")),
            }
        )
    return result


def _error(
    *,
    code: str,
    message: str,
    details: dict[str, Any],
    recoverable: bool = True,
    cause: str | None = None,
) -> LasIoError:
    return LasIoError(
        UnifiedError(
            code=code,
            message=message,
            severity="error",
            module="algorithms.io",
            recoverable=recoverable,
            details=details,
            cause=cause,
        )
    )
