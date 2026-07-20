"""Chunked XYZ crop engine for LAS/LAZ files."""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from core.contracts import (
    CancellationToken,
    CropBounds,
    CropResult,
    PipelineProgress,
    ProgressReporter,
)

from algorithms.io.chunks import DEFAULT_CHUNK_SIZE
from algorithms.io.headers import _error, _validate_las_input


def crop_las_xyz(
    source_path: Path | str,
    output_path: Path | str,
    bounds: CropBounds,
    *,
    source_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overwrite: bool = False,
    reporter: ProgressReporter | None = None,
    cancellation_token: CancellationToken | None = None,
    project_root: Path | str | None = None,
    task_id: str | None = None,
) -> CropResult:
    """Crop LAS points using inclusive XYZ bounds and write a new LAS/LAZ file."""
    source = _validate_las_input(Path(source_path))
    output = Path(output_path)
    if chunk_size <= 0:
        raise _error(
            code="LAS_CHUNK_SIZE_INVALID",
            message="chunk_size must be a positive integer",
            details={"chunk_size": chunk_size},
        )
    _validate_output_path(source, output, overwrite=overwrite, project_root=project_root)

    started_at = _now()
    tmp_output = _temporary_output_path(output)
    input_point_count = 0
    output_point_count = 0
    progress_task_id = task_id or f"{source_id}:crop"

    try:
        if tmp_output.exists():
            tmp_output.unlink()
        output.parent.mkdir(parents=True, exist_ok=True)

        with laspy.open(source, mode="r") as reader:
            total_points = int(reader.header.point_count)
            output_header = reader.header.copy()
            dimension_names = tuple(
                str(name) for name in output_header.point_format.dimension_names
            )
            effective_min, effective_max = _effective_bounds(bounds)

            with laspy.open(tmp_output, mode="w", header=output_header) as writer:
                for point_record in reader.chunk_iterator(chunk_size):
                    if _cancelled(cancellation_token):
                        raise _error(
                            code="LAS_CROP_CANCELLED",
                            message="LAS crop cancelled before next chunk",
                            details={"source_path": str(source), "output_path": str(output)},
                        )

                    point_count = len(point_record)
                    input_point_count += point_count
                    mask = _crop_mask(point_record, effective_min, effective_max)
                    selected_count = int(np.count_nonzero(mask))
                    if selected_count:
                        writer.write_points(point_record[mask])
                        output_point_count += selected_count

                    _report(
                        reporter,
                        progress_task_id,
                        "crop",
                        "running" if input_point_count < total_points else "completed",
                        input_point_count,
                        total_points,
                        f"Cropped {output_point_count} of {input_point_count} processed points",
                    )
                    if _cancelled(cancellation_token):
                        raise _error(
                            code="LAS_CROP_CANCELLED",
                            message="LAS crop cancelled after chunk",
                            details={"source_path": str(source), "output_path": str(output)},
                        )

        if overwrite and output.exists():
            output.unlink()
        os.replace(tmp_output, output)
        return CropResult(
            source_id=source_id,
            bounds=bounds,
            input_point_count=input_point_count,
            output_point_count=output_point_count,
            output_path=output,
            header_preserved=True,
            dimension_names=dimension_names,
            warnings=() if output_point_count else ("Crop selected zero points",),
            started_at=started_at,
            finished_at=_now(),
        )
    except Exception:
        if tmp_output.exists():
            tmp_output.unlink()
        raise


def _effective_bounds(bounds: CropBounds) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray(bounds.min_xyz, dtype=np.float64)
    maximum = np.asarray(bounds.max_xyz, dtype=np.float64)
    if bounds.coordinate_mode == "source":
        return minimum, maximum
    if bounds.origin_xyz is None:
        raise _error(
            code="LAS_CROP_LOCAL_ORIGIN_REQUIRED",
            message="Local crop bounds require origin_xyz",
            details={"coordinate_mode": bounds.coordinate_mode},
        )
    origin = np.asarray(bounds.origin_xyz, dtype=np.float64)
    return minimum + origin, maximum + origin


def _crop_mask(point_record: Any, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    x = np.asarray(point_record.x, dtype=np.float64)
    y = np.asarray(point_record.y, dtype=np.float64)
    z = np.asarray(point_record.z, dtype=np.float64)
    return (
        (x >= minimum[0])
        & (x <= maximum[0])
        & (y >= minimum[1])
        & (y <= maximum[1])
        & (z >= minimum[2])
        & (z <= maximum[2])
    )


def _validate_output_path(
    source: Path,
    output: Path,
    *,
    overwrite: bool,
    project_root: Path | str | None,
) -> None:
    if output.suffix.lower() not in {".las", ".laz"}:
        raise _error(
            code="LAS_CROP_OUTPUT_EXTENSION_INVALID",
            message="Crop output must have .las or .laz extension",
            details={"output_path": str(output), "suffix": output.suffix},
        )
    source_resolved = source.resolve()
    output_resolved = output.resolve(strict=False)
    if output_resolved == source_resolved:
        raise _error(
            code="LAS_CROP_OUTPUT_IS_SOURCE",
            message="Crop output cannot overwrite the source LAS",
            details={"source_path": str(source), "output_path": str(output)},
        )
    if output.exists() and not overwrite:
        raise _error(
            code="LAS_CROP_OUTPUT_EXISTS",
            message="Crop output already exists and overwrite is disabled",
            details={"output_path": str(output)},
        )
    root = Path(project_root) if project_root is not None else Path.cwd()
    if not _is_allowed_output(output_resolved, root.resolve()):
        raise _error(
            code="LAS_CROP_OUTPUT_FORBIDDEN",
            message=(
                "Crop output must be under data/samples, data/processed, outputs, or pytest temp"
            ),
            details={"output_path": str(output)},
        )


def _is_allowed_output(output: Path, project_root: Path) -> bool:
    for allowed in (
        project_root / "data" / "samples",
        project_root / "data" / "processed",
        project_root / "outputs",
    ):
        if _is_relative_to(output, allowed.resolve(strict=False)):
            return True
    raw_root = (project_root / "data" / "raw").resolve(strict=False)
    if _is_relative_to(output, raw_root):
        return False
    temp_root = Path(tempfile.gettempdir()).resolve(strict=False)
    return _is_relative_to(output, temp_root)


def _temporary_output_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.tmp{output.suffix}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


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


def _now() -> str:
    return datetime.now(UTC).isoformat()
