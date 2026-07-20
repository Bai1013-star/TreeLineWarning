"""Formal sample crop helpers built on the chunked LAS crop engine."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from core.contracts import CropBounds

from algorithms.io.chunks import DEFAULT_CHUNK_SIZE, iter_las_chunks
from algorithms.io.crop import crop_las_xyz
from algorithms.io.headers import _error, read_las_header_summary

BOUNDARY_TOLERANCE_M = 0.10
ALGORITHM_NAME = "chunked_xyz_scene_sample_crop"
ALGORITHM_VERSION = "0.1"


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    size_bytes: int
    last_write_time: str
    attributes: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "last_write_time": self.last_write_time,
            "attributes": self.attributes,
        }


@dataclass(frozen=True)
class CropPrecheck:
    source_id: str
    source_path: Path
    input_point_count: int
    kept_point_count: int
    kept_ratio: float
    boundary_counts: dict[str, int]
    z_max_boundary_count: int
    z_max_boundary_ratio: float
    kept_min_xyz: tuple[float, float, float] | None
    kept_max_xyz: tuple[float, float, float] | None
    original_index: np.ndarray

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_path": str(self.source_path),
            "input_point_count": self.input_point_count,
            "kept_point_count": self.kept_point_count,
            "kept_ratio": self.kept_ratio,
            "boundary_counts": dict(self.boundary_counts),
            "z_max_boundary_count": self.z_max_boundary_count,
            "z_max_boundary_ratio": self.z_max_boundary_ratio,
            "kept_min_xyz": (
                list(self.kept_min_xyz) if self.kept_min_xyz is not None else None
            ),
            "kept_max_xyz": (
                list(self.kept_max_xyz) if self.kept_max_xyz is not None else None
            ),
        }


def file_snapshot(path: Path | str) -> FileSnapshot:
    file_path = Path(path)
    stat = file_path.stat()
    return FileSnapshot(
        path=file_path,
        size_bytes=int(stat.st_size),
        last_write_time=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        attributes=getattr(stat, "st_file_attributes", None),
    )


def load_and_validate_scene_simple_selection(
    selection_path: Path | str,
    *,
    expected_source_path: Path | str,
    expected_min_xyz: tuple[float, float, float],
    expected_max_xyz: tuple[float, float, float],
    expected_label: str = "scene_simple",
) -> tuple[dict[str, Any], CropBounds]:
    selection_file = Path(selection_path)
    payload = json.loads(selection_file.read_text(encoding="utf-8"))
    source_path = str(Path(payload["source_path"]))
    expected_path = str(Path(expected_source_path))
    if source_path != expected_path:
        raise ValueError(
            f"selection source_path mismatch: expected {expected_path}, got {source_path}"
        )
    if payload.get("coordinate_mode") != "source":
        raise ValueError("selection coordinate_mode must be source")
    if payload.get("label") != expected_label:
        raise ValueError(f"selection label must be {expected_label}")
    bounds = CropBounds(
        min_xyz=tuple(float(value) for value in payload["min_xyz"]),
        max_xyz=tuple(float(value) for value in payload["max_xyz"]),
        coordinate_mode="source",
    )
    if bounds.min_xyz != expected_min_xyz or bounds.max_xyz != expected_max_xyz:
        raise ValueError(
            "selection bounds do not match approved scene_simple range: "
            f"{bounds.min_xyz} - {bounds.max_xyz}"
        )
    return payload, bounds


def precheck_crop(
    source_path: Path | str,
    bounds: CropBounds,
    *,
    source_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    boundary_tolerance_m: float = BOUNDARY_TOLERANCE_M,
) -> CropPrecheck:
    minimum = np.asarray(bounds.min_xyz, dtype=np.float64)
    maximum = np.asarray(bounds.max_xyz, dtype=np.float64)
    total_points = 0
    kept_count = 0
    boundary_counts = {
        "x_min": 0,
        "x_max": 0,
        "y_min": 0,
        "y_max": 0,
        "z_min": 0,
        "z_max": 0,
    }
    kept_min: np.ndarray | None = None
    kept_max: np.ndarray | None = None
    index_parts: list[np.ndarray] = []

    for chunk in iter_las_chunks(source_path, source_id=source_id, chunk_size=chunk_size):
        xyz = np.asarray(chunk.xyz, dtype=np.float64)
        mask = np.all((xyz >= minimum) & (xyz <= maximum), axis=1)
        selected_count = int(np.count_nonzero(mask))
        total_points += chunk.point_count
        if selected_count == 0:
            continue

        selected_xyz = xyz[mask]
        kept_count += selected_count
        index_parts.append(np.asarray(chunk.original_index, dtype=np.int64)[mask])
        selected_min = selected_xyz.min(axis=0)
        selected_max = selected_xyz.max(axis=0)
        kept_min = selected_min if kept_min is None else np.minimum(kept_min, selected_min)
        kept_max = selected_max if kept_max is None else np.maximum(kept_max, selected_max)
        boundary_counts["x_min"] += int(
            np.count_nonzero(np.abs(selected_xyz[:, 0] - minimum[0]) <= boundary_tolerance_m)
        )
        boundary_counts["x_max"] += int(
            np.count_nonzero(np.abs(selected_xyz[:, 0] - maximum[0]) <= boundary_tolerance_m)
        )
        boundary_counts["y_min"] += int(
            np.count_nonzero(np.abs(selected_xyz[:, 1] - minimum[1]) <= boundary_tolerance_m)
        )
        boundary_counts["y_max"] += int(
            np.count_nonzero(np.abs(selected_xyz[:, 1] - maximum[1]) <= boundary_tolerance_m)
        )
        boundary_counts["z_min"] += int(
            np.count_nonzero(np.abs(selected_xyz[:, 2] - minimum[2]) <= boundary_tolerance_m)
        )
        boundary_counts["z_max"] += int(
            np.count_nonzero(np.abs(selected_xyz[:, 2] - maximum[2]) <= boundary_tolerance_m)
        )

    original_index = (
        np.concatenate(index_parts).astype(np.int64, copy=False)
        if index_parts
        else np.empty(0, dtype=np.int64)
    )
    return CropPrecheck(
        source_id=source_id,
        source_path=Path(source_path),
        input_point_count=total_points,
        kept_point_count=kept_count,
        kept_ratio=0.0 if total_points == 0 else kept_count / total_points,
        boundary_counts=boundary_counts,
        z_max_boundary_count=boundary_counts["z_max"],
        z_max_boundary_ratio=0.0 if kept_count == 0 else boundary_counts["z_max"] / kept_count,
        kept_min_xyz=_triple_or_none(kept_min),
        kept_max_xyz=_triple_or_none(kept_max),
        original_index=original_index,
    )


def create_scene_sample(
    *,
    source_path: Path | str,
    output_path: Path | str,
    index_map_path: Path | str,
    manifest_path: Path | str,
    bounds: CropBounds,
    source_id: str,
    selection_record: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    project_root: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create the scene_simple LAS, original-index sidecar, and manifest."""
    source = Path(source_path)
    output = Path(output_path)
    index_map = Path(index_map_path)
    manifest = Path(manifest_path)
    root = Path(project_root) if project_root is not None else Path.cwd()
    _ensure_outputs_absent((output, index_map, manifest), overwrite=overwrite)
    _ensure_under_samples(output, root)
    _ensure_under_samples(index_map, root)
    _ensure_under_samples(manifest, root)

    warnings: list[str] = []
    started = time.perf_counter()
    memory_before = _process_memory_bytes()
    source_before = file_snapshot(source)
    source_header = read_las_header_summary(source, source_id=source_id)
    precheck = precheck_crop(source, bounds, source_id=source_id, chunk_size=chunk_size)
    if precheck.kept_point_count == 0:
        warnings.append("Crop selected zero points")
    if precheck.z_max_boundary_ratio > 0.05:
        warnings.append("More than 5% of kept points are within 0.10 m of z_max")

    try:
        crop_result = crop_las_xyz(
            source,
            output,
            bounds,
            source_id=source_id,
            chunk_size=chunk_size,
            overwrite=overwrite,
            project_root=root,
        )
        _write_index_map(
            index_map,
            precheck.original_index,
            source_id=source_id,
            bounds=bounds,
            output_point_count=crop_result.output_point_count,
        )
        validation = validate_scene_sample_outputs(
            source_path=source,
            output_path=output,
            index_map_path=index_map,
            manifest_path=None,
            bounds=bounds,
            source_header=source_header.to_dict(),
            source_snapshot_before=source_before.to_dict(),
            source_point_count=source_header.point_count,
            expected_point_count=crop_result.output_point_count,
            chunk_size=chunk_size,
        )
        source_after = file_snapshot(source)
        memory_after = _process_memory_bytes()
        manifest_payload = {
            "source_id": source_id,
            "source_path": str(source),
            "source_file_size": source_before.size_bytes,
            "source_point_count": source_header.point_count,
            "source_header": source_header.to_dict(),
            "crop_bounds": bounds.to_dict(),
            "coordinate_mode": bounds.coordinate_mode,
            "output_path": str(output),
            "output_point_count": crop_result.output_point_count,
            "kept_ratio": precheck.kept_ratio,
            "output_file_size": output.stat().st_size,
            "index_map_path": str(index_map),
            "created_at": datetime.now(UTC).isoformat(),
            "chunk_size": chunk_size,
            "processing_seconds": time.perf_counter() - started,
            "process_memory_bytes": {
                "before": memory_before,
                "after": memory_after,
            },
            "warnings": warnings,
            "precheck": precheck.to_manifest_dict(),
            "selection_record": selection_record,
            "source_file_before": source_before.to_dict(),
            "source_file_after": source_after.to_dict(),
            "source_file_unchanged": source_before == source_after,
            "validation": validation,
            "algorithm": {
                "name": ALGORITHM_NAME,
                "version": ALGORITHM_VERSION,
            },
        }
        _write_json_atomic(manifest, manifest_payload)
        manifest_validation = validate_scene_sample_outputs(
            source_path=source,
            output_path=output,
            index_map_path=index_map,
            manifest_path=manifest,
            bounds=bounds,
            source_header=source_header.to_dict(),
            source_snapshot_before=source_before.to_dict(),
            source_point_count=source_header.point_count,
            expected_point_count=crop_result.output_point_count,
            chunk_size=chunk_size,
        )
        manifest_payload["validation"] = manifest_validation
        _write_json_atomic(manifest, manifest_payload)
        return manifest_payload
    except Exception:
        _cleanup_paths((output, index_map, manifest))
        raise


def validate_scene_sample_outputs(
    *,
    source_path: Path | str,
    output_path: Path | str,
    index_map_path: Path | str,
    manifest_path: Path | str | None,
    bounds: CropBounds,
    source_header: dict[str, Any],
    source_snapshot_before: dict[str, Any],
    source_point_count: int,
    expected_point_count: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    source = Path(source_path)
    output = Path(output_path)
    index_map = Path(index_map_path)
    output_summary = read_las_header_summary(output, source_id="scene_simple")
    bounds_check = _validate_output_points_within_bounds(output, bounds, chunk_size=chunk_size)
    with np.load(index_map, allow_pickle=False) as data:
        original_index = np.asarray(data["original_index"], dtype=np.int64)
        output_point_count = int(np.asarray(data["output_point_count"]).item())
    if len(original_index) != output_summary.point_count:
        raise ValueError("index_map original_index length does not match output LAS point count")
    if output_point_count != output_summary.point_count:
        raise ValueError("index_map output_point_count does not match output LAS point count")
    if len(original_index) and int(original_index.min()) < 0:
        raise ValueError("index_map contains negative original_index values")
    if len(original_index) and int(original_index.max()) >= source_point_count:
        raise ValueError("index_map contains out-of-range original_index values")
    if len(np.unique(original_index)) != len(original_index):
        raise ValueError("index_map contains duplicate original_index values")

    _validate_header_preserved(output_summary.to_dict(), source_header)
    source_after = file_snapshot(source).to_dict()
    source_unchanged = source_snapshot_before == source_after
    if not source_unchanged:
        raise ValueError("source LAS file changed during sample crop")

    manifest_matches = True
    if manifest_path is not None:
        manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest_matches = (
            manifest_payload["output_point_count"] == output_summary.point_count
            and manifest_payload["output_file_size"] == output.stat().st_size
            and manifest_payload["source_file_size"] == source_snapshot_before["size_bytes"]
        )
        if not manifest_matches:
            raise ValueError("manifest values do not match generated files")

    return {
        "output_las_opened": True,
        "output_point_count": output_summary.point_count,
        "output_points_within_bounds": bounds_check["all_within_bounds"],
        "output_min_xyz": bounds_check["min_xyz"],
        "output_max_xyz": bounds_check["max_xyz"],
        "header_preserved": True,
        "dimensions": output_summary.dimensions,
        "extra_dimensions": output_summary.extra_dimensions,
        "index_map_length": len(original_index),
        "index_map_min": int(original_index.min()) if len(original_index) else None,
        "index_map_max": int(original_index.max()) if len(original_index) else None,
        "index_map_unique": len(np.unique(original_index)) == len(original_index),
        "manifest_matches_files": manifest_matches,
        "source_file_unchanged": source_unchanged,
        "expected_point_count": expected_point_count,
    }


def _validate_output_points_within_bounds(
    output_path: Path,
    bounds: CropBounds,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    minimum = np.asarray(bounds.min_xyz, dtype=np.float64)
    maximum = np.asarray(bounds.max_xyz, dtype=np.float64)
    point_count = 0
    found_min: np.ndarray | None = None
    found_max: np.ndarray | None = None
    with laspy.open(output_path, mode="r") as reader:
        for point_record in reader.chunk_iterator(chunk_size):
            xyz = np.column_stack(
                (
                    np.asarray(point_record.x, dtype=np.float64),
                    np.asarray(point_record.y, dtype=np.float64),
                    np.asarray(point_record.z, dtype=np.float64),
                )
            )
            point_count += len(xyz)
            if len(xyz) == 0:
                continue
            if not bool(np.all((xyz >= minimum) & (xyz <= maximum))):
                raise ValueError("output LAS contains points outside approved crop bounds")
            chunk_min = xyz.min(axis=0)
            chunk_max = xyz.max(axis=0)
            found_min = chunk_min if found_min is None else np.minimum(found_min, chunk_min)
            found_max = chunk_max if found_max is None else np.maximum(found_max, chunk_max)
    return {
        "all_within_bounds": True,
        "point_count": point_count,
        "min_xyz": _list_or_none(found_min),
        "max_xyz": _list_or_none(found_max),
    }


def _write_index_map(
    path: Path,
    original_index: np.ndarray,
    *,
    source_id: str,
    bounds: CropBounds,
    output_point_count: int,
) -> None:
    if len(original_index) != output_point_count:
        raise ValueError("original_index length must match output_point_count")
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tmp_path,
            original_index=np.asarray(original_index, dtype=np.int64),
            source_id=np.asarray(source_id),
            crop_bounds=np.asarray(json.dumps(bounds.to_dict())),
            output_point_count=np.asarray(output_point_count, dtype=np.int64),
        )
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _ensure_outputs_absent(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        details = {str(path): path.stat().st_size for path in existing}
        raise _error(
            code="SCENE_SAMPLE_OUTPUT_EXISTS",
            message="scene_simple output already exists and overwrite is disabled",
            details=details,
        )


def _ensure_under_samples(path: Path, project_root: Path) -> None:
    samples_root = (project_root / "data" / "samples").resolve(strict=False)
    resolved = path.resolve(strict=False)
    if not _is_relative_to(resolved, samples_root):
        raise _error(
            code="SCENE_SAMPLE_OUTPUT_FORBIDDEN",
            message="scene_simple outputs must be under data/samples",
            details={"output_path": str(path)},
        )


def _cleanup_paths(paths: tuple[Path, ...]) -> None:
    for path in paths:
        tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
        if tmp_path.exists():
            tmp_path.unlink()
        if path.exists():
            path.unlink()


def _validate_header_preserved(
    output_header: dict[str, Any],
    source_header: dict[str, Any],
) -> None:
    for field_name in ("las_version", "point_format_id", "scale", "offset"):
        if output_header[field_name] != source_header[field_name]:
            raise ValueError(f"output LAS header field not preserved: {field_name}")
    source_dimensions = set(source_header["dimensions"])
    output_dimensions = set(output_header["dimensions"])
    missing_dimensions = source_dimensions - output_dimensions
    if missing_dimensions:
        raise ValueError(f"output LAS missing source dimensions: {sorted(missing_dimensions)}")
    source_extra = set(source_header.get("extra_dimensions", ()))
    output_extra = set(output_header.get("extra_dimensions", ()))
    missing_extra = source_extra - output_extra
    if missing_extra:
        raise ValueError(f"output LAS missing source extra dimensions: {sorted(missing_extra)}")


def _triple_or_none(values: np.ndarray | None) -> tuple[float, float, float] | None:
    if values is None:
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def _list_or_none(values: np.ndarray | None) -> list[float] | None:
    if values is None:
        return None
    return [float(values[0]), float(values[1]), float(values[2])]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _process_memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception:
        return None
    return int(psutil.Process().memory_info().rss)
