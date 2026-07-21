"""Safe LAS/LAZ chunk iteration."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from core.contracts import CancellationToken, PipelineProgress, PointCloudChunk, ProgressReporter

from algorithms.io.discovery import stable_source_id
from algorithms.io.headers import _error, _validate_las_input

DEFAULT_CHUNK_SIZE = 250_000

APPROVED_ATTRIBUTE_NAMES = (
    "intensity",
    "return_number",
    "number_of_returns",
    "classification",
    "user_data",
    "scan_angle",
    "point_source_id",
    "gps_time",
    "red",
    "green",
    "blue",
    "NormalX",
    "NormalY",
    "NormalZ",
)


def iter_las_chunks(
    path: Path | str,
    *,
    source_id: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    reporter: ProgressReporter | None = None,
    cancellation_token: CancellationToken | None = None,
    task_id: str | None = None,
) -> Iterable[PointCloudChunk]:
    """Yield point chunks using ``laspy.open().chunk_iterator()`` only."""
    file_path = _validate_las_input(Path(path))
    if chunk_size <= 0:
        raise _error(
            code="LAS_CHUNK_SIZE_INVALID",
            message="chunk_size must be a positive integer",
            details={"chunk_size": chunk_size},
        )

    stable_id = source_id or stable_source_id(file_path, file_path.parent)
    progress_task_id = task_id or f"{stable_id}:chunk-read"

    with laspy.open(file_path, mode="r") as reader:
        total_points = int(reader.header.point_count)
        start_index = 0
        chunk_id = 0
        for point_record in reader.chunk_iterator(chunk_size):
            if _cancelled(cancellation_token):
                _report(
                    reporter,
                    progress_task_id,
                    "chunk-read",
                    "cancelled",
                    start_index,
                    total_points,
                    "LAS chunk reading cancelled before next chunk",
                )
                break

            point_count = len(point_record)
            next_start = start_index + point_count
            xyz = _extract_xyz(point_record)
            chunk = PointCloudChunk(
                source_id=stable_id,
                chunk_id=chunk_id,
                start_index=start_index,
                point_count=point_count,
                xyz=xyz,
                original_index=np.arange(start_index, next_start, dtype=np.int64),
                attributes=_extract_attributes(point_record),
                bounds_xyz=_bounds_xyz(xyz) if point_count else None,
                is_last=next_start >= total_points,
            )

            yield chunk

            start_index = next_start
            chunk_id += 1
            _report(
                reporter,
                progress_task_id,
                "chunk-read",
                "running" if start_index < total_points else "completed",
                start_index,
                total_points,
                f"Read {start_index} of {total_points} LAS points",
            )
            if _cancelled(cancellation_token):
                _report(
                    reporter,
                    progress_task_id,
                    "chunk-read",
                    "cancelled",
                    start_index,
                    total_points,
                    "LAS chunk reading cancelled after chunk",
                )
                break


def _extract_xyz(point_record: Any) -> np.ndarray:
    return np.column_stack(
        (
            np.asarray(point_record.x, dtype=np.float64),
            np.asarray(point_record.y, dtype=np.float64),
            np.asarray(point_record.z, dtype=np.float64),
        )
    )


def _extract_attributes(point_record: Any) -> dict[str, np.ndarray]:
    dimensions = set(str(name) for name in point_record.point_format.dimension_names)
    attributes: dict[str, np.ndarray] = {}
    for name in APPROVED_ATTRIBUTE_NAMES:
        if name in dimensions:
            attributes[name] = np.asarray(getattr(point_record, name))
    return attributes


def _bounds_xyz(xyz: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    minimum = tuple(float(value) for value in xyz.min(axis=0))
    maximum = tuple(float(value) for value in xyz.max(axis=0))
    return minimum, maximum


def _cancelled(cancellation_token: CancellationToken | None) -> bool:
    return cancellation_token is not None and cancellation_token.is_cancelled()


def _report(
    reporter: ProgressReporter | None,
    task_id: str,
    stage: str,
    status: str,
    processed_points: int,
    total_points: int,
    message: str,
) -> None:
    if reporter is None:
        return
    percent = 100.0 if total_points == 0 else processed_points / total_points * 100.0
    reporter.update(
        PipelineProgress(
            task_id=task_id,
            stage=stage,
            status=status,  # type: ignore[arg-type]
            processed_points=processed_points,
            total_points=total_points,
            percent=percent,
            message=message,
        )
    )
