"""Formal sample crop helpers built on the chunked LAS crop engine."""

from __future__ import annotations

import hashlib
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
# This only absorbs JSON float representation noise and is far below the 0.0001 m LAS scale.
CROP_BOUNDS_COMPARISON_ATOL_M = 1e-9
ALGORITHM_NAME = "chunked_xyz_scene_sample_crop"
ALGORITHM_VERSION = "0.1"
BOUNDARY_NAMES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
_CLASSIFIED_LINEAGE_SCHEMA_VERSION = "fgc-classified-parent-lineage-v1"


def _file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 of the actual file bytes on disk."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            "kept_min_xyz": (list(self.kept_min_xyz) if self.kept_min_xyz is not None else None),
            "kept_max_xyz": (list(self.kept_max_xyz) if self.kept_max_xyz is not None else None),
        }


@dataclass(frozen=True)
class BoundaryProximityReport:
    point_count: int
    tolerances_m: tuple[float, ...]
    counts: dict[str, dict[str, int]]
    ratios: dict[str, dict[str, float]]
    z_max_detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_count": self.point_count,
            "tolerances_m": list(self.tolerances_m),
            "counts": self.counts,
            "ratios": self.ratios,
            "z_max_detail": self.z_max_detail,
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


def validate_scene_sample_artifacts(
    *,
    sample_path: Path | str,
    index_map_path: Path | str,
    manifest_path: Path | str,
    bounds: CropBounds,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Validate scene_simple LAS, sidecar index map, and manifest without writing."""
    sample = Path(sample_path)
    index_map = Path(index_map_path)
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    sample_snapshot = file_snapshot(sample)
    sample_header = read_las_header_summary(sample, source_id="scene_simple")
    bounds_check = _validate_output_points_within_bounds(sample, bounds, chunk_size=chunk_size)

    with np.load(index_map, allow_pickle=False) as data:
        original_index = np.asarray(data["original_index"], dtype=np.int64)
        index_output_count = int(np.asarray(data["output_point_count"]).item())
        index_source_id = str(np.asarray(data["source_id"]).item())
        index_bounds = json.loads(str(np.asarray(data["crop_bounds"]).item()))

    source_point_count = int(manifest["source_point_count"])
    if len(original_index) != sample_header.point_count:
        raise ValueError("index_map original_index length does not match scene_simple point count")
    if index_output_count != sample_header.point_count:
        raise ValueError("index_map output_point_count does not match scene_simple point count")
    if len(original_index) and int(original_index.min()) < 0:
        raise ValueError("index_map contains negative original_index values")
    if len(original_index) and int(original_index.max()) >= source_point_count:
        raise ValueError("index_map contains original_index values outside source point count")
    index_unique = len(np.unique(original_index)) == len(original_index)
    if not index_unique:
        raise ValueError("index_map contains duplicate original_index values")

    expected_output = Path(manifest["output_path"])
    expected_index = Path(manifest["index_map_path"])
    manifest_matches = (
        Path(manifest["output_path"]) == sample
        and expected_output.exists()
        and expected_index == index_map
        and manifest["output_point_count"] == sample_header.point_count
        and manifest["output_file_size"] == sample_snapshot.size_bytes
        and _crop_bounds_match(manifest.get("crop_bounds"), bounds)
        and manifest["coordinate_mode"] == bounds.coordinate_mode
    )
    if not manifest_matches:
        raise ValueError("manifest does not match scene_simple sample artifacts")
    lineage = manifest.get("parent_lineage")
    if lineage is not None:
        _recheck_parent_lineage(
            lineage,
            sample=sample,
            sample_point_count=sample_header.point_count,
            index_map_path=index_map,
            index_map_length=len(original_index),
            index_map_source_id=index_source_id,
        )

    return {
        "sample_header": sample_header.to_dict(),
        "sample_file": sample_snapshot.to_dict(),
        "points_within_bounds": bounds_check["all_within_bounds"],
        "actual_min_xyz": bounds_check["min_xyz"],
        "actual_max_xyz": bounds_check["max_xyz"],
        "index_map_length": len(original_index),
        "index_map_min": int(original_index.min()) if len(original_index) else None,
        "index_map_max": int(original_index.max()) if len(original_index) else None,
        "index_map_unique": index_unique,
        "index_map_source_id": index_source_id,
        "index_map_crop_bounds": index_bounds,
        "manifest_matches_files": manifest_matches,
        "manifest": manifest,
    }


def _crop_bounds_match(manifest_bounds: Any, bounds: CropBounds) -> bool:
    """Compare finite manifest coordinates with a serialization-only tolerance."""
    if not isinstance(manifest_bounds, dict):
        return False
    try:
        manifest_min = np.asarray(manifest_bounds["min_xyz"], dtype=np.float64)
        manifest_max = np.asarray(manifest_bounds["max_xyz"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False

    expected_min = np.asarray(bounds.min_xyz, dtype=np.float64)
    expected_max = np.asarray(bounds.max_xyz, dtype=np.float64)
    coordinates = (manifest_min, manifest_max, expected_min, expected_max)
    if any(values.shape != (3,) or not np.isfinite(values).all() for values in coordinates):
        return False

    return bool(
        np.allclose(manifest_min, expected_min, rtol=0.0, atol=CROP_BOUNDS_COMPARISON_ATOL_M)
        and np.allclose(manifest_max, expected_max, rtol=0.0, atol=CROP_BOUNDS_COMPARISON_ATOL_M)
    )


def boundary_proximity_report(
    sample_path: Path | str,
    bounds: CropBounds,
    *,
    source_id: str,
    tolerances_m: tuple[float, ...] = (0.05, 0.10, 0.20),
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> BoundaryProximityReport:
    """Count points near each approved crop boundary with z_max detail."""
    tolerances = tuple(float(value) for value in tolerances_m)
    counts = {
        boundary: {f"{tolerance:.2f}": 0 for tolerance in tolerances} for boundary in BOUNDARY_NAMES
    }
    total_points = 0
    z_max_xyz_parts: list[np.ndarray] = []
    z_max_intensity_parts: list[np.ndarray] = []
    z_max_class_parts: list[np.ndarray] = []
    z_max_rgb_parts: list[np.ndarray] = []
    maximum = np.asarray(bounds.max_xyz, dtype=np.float64)

    for chunk in iter_las_chunks(sample_path, source_id=source_id, chunk_size=chunk_size):
        xyz = np.asarray(chunk.xyz, dtype=np.float64)
        total_points += chunk.point_count
        distances = {
            "x_min": np.abs(xyz[:, 0] - bounds.min_xyz[0]),
            "x_max": np.abs(xyz[:, 0] - bounds.max_xyz[0]),
            "y_min": np.abs(xyz[:, 1] - bounds.min_xyz[1]),
            "y_max": np.abs(xyz[:, 1] - bounds.max_xyz[1]),
            "z_min": np.abs(xyz[:, 2] - bounds.min_xyz[2]),
            "z_max": np.abs(xyz[:, 2] - bounds.max_xyz[2]),
        }
        for boundary, values in distances.items():
            for tolerance in tolerances:
                counts[boundary][f"{tolerance:.2f}"] += int(np.count_nonzero(values <= tolerance))

        z_mask = distances["z_max"] <= 0.10
        if int(np.count_nonzero(z_mask)):
            z_max_xyz_parts.append(xyz[z_mask])
            intensity = chunk.attributes.get("intensity")
            classification = chunk.attributes.get("classification")
            red = chunk.attributes.get("red")
            green = chunk.attributes.get("green")
            blue = chunk.attributes.get("blue")
            if intensity is not None:
                z_max_intensity_parts.append(np.asarray(intensity)[z_mask])
            if classification is not None:
                z_max_class_parts.append(np.asarray(classification)[z_mask])
            if red is not None and green is not None and blue is not None:
                z_max_rgb_parts.append(
                    np.column_stack(
                        (
                            np.asarray(red)[z_mask],
                            np.asarray(green)[z_mask],
                            np.asarray(blue)[z_mask],
                        )
                    )
                )

    ratios = {
        boundary: {
            tolerance: 0.0 if total_points == 0 else count / total_points
            for tolerance, count in tolerance_counts.items()
        }
        for boundary, tolerance_counts in counts.items()
    }
    z_max_detail = _z_max_detail(
        xyz_parts=z_max_xyz_parts,
        intensity_parts=z_max_intensity_parts,
        class_parts=z_max_class_parts,
        rgb_parts=z_max_rgb_parts,
        max_xyz=maximum,
        total_points=total_points,
    )
    return BoundaryProximityReport(
        point_count=total_points,
        tolerances_m=tolerances,
        counts=counts,
        ratios=ratios,
        z_max_detail=z_max_detail,
    )


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
    classified_parent_root: Path | str | None = None,
) -> dict[str, Any]:
    """Create the scene_simple LAS, original-index sidecar, and manifest.

    ``classified_parent_root`` optionally binds an FGC classified parent
    publication: the sample source must be the actual classified LAS (digest
    bound), the sample ``source_id`` must represent the classified parent
    authority, and the produced sample LAS / index-map digests and point counts
    are derived from the actual generated files and recorded in a verified
    ``parent_lineage`` manifest record.  Binding never changes crop selection
    geometry.
    """
    source = Path(source_path)
    output = Path(output_path)
    index_map = Path(index_map_path)
    manifest = Path(manifest_path)
    root = Path(project_root) if project_root is not None else Path.cwd()
    _ensure_outputs_absent((output, index_map, manifest), overwrite=overwrite)
    _ensure_under_samples(output, root)
    _ensure_under_samples(index_map, root)
    _ensure_under_samples(manifest, root)

    publication = None
    if classified_parent_root is not None:
        publication = _verify_classified_parent_source(
            classified_parent_root, source, source_id
        )

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
        parent_lineage = None
        if publication is not None:
            parent_lineage = _derive_classified_lineage(
                publication=publication,
                source_id=source_id,
                output_path=output,
                index_map_path=index_map,
                output_point_count=crop_result.output_point_count,
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
            "parent_lineage": _json_safe_lineage(parent_lineage),
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
        index_source_id = str(np.asarray(data["source_id"]).item())
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
        lineage = manifest_payload.get("parent_lineage")
        if lineage is not None:
            _recheck_parent_lineage(
                lineage,
                sample=output,
                sample_point_count=output_summary.point_count,
                index_map_path=index_map,
                index_map_length=len(original_index),
                index_map_source_id=index_source_id,
            )

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


def _json_safe_lineage(lineage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a JSON-safe copy of the classified-parent lineage binding."""
    if lineage is None:
        return None
    from core.contracts._json import json_safe

    return json_safe(lineage, "parent_lineage")  # type: ignore[return-value]


def _verify_classified_parent_source(
    publication_root: Path | str,
    source_path: Path,
    source_id: str,
) -> Any:
    """Validate the FGC classified parent and bind the sample source to it.

    Uses the existing fully validated FGC publication (completion manifest,
    decision manifest, publication proof, classified identity) and requires the
    sample source to be the byte-identical classified LAS and the sample
    ``source_id`` to represent the classified parent authority.  This rejects
    metadata relabeling of an unrelated scene.
    """

    from algorithms.ground.visualization import validate_fgc_publication

    publication = validate_fgc_publication(publication_root)
    completion = publication.completion_manifest
    classified = completion.classified_source_identity
    source_sha = _file_sha256(source_path)
    if source_sha != completion.classified_las_sha256:
        raise ValueError("sample source LAS digest does not match the classified parent")
    if classified.source_id != source_id:
        raise ValueError("sample source_id must represent the classified parent authority")
    return publication


def _derive_classified_lineage(
    *,
    publication: Any,
    source_id: str,
    output_path: Path,
    index_map_path: Path,
    output_point_count: int,
) -> dict[str, Any]:
    """Derive the verified classified-parent lineage from actual generated files.

    All hashes are computed from the produced/read files; no caller-supplied
    hash string is accepted as authority.
    """

    completion = publication.completion_manifest
    proof = publication.publication_proof
    classified = completion.classified_source_identity
    with np.load(index_map_path, allow_pickle=False) as data:
        original_index = np.asarray(data["original_index"], dtype=np.int64)
    if int(original_index.shape[0]) != output_point_count:
        raise ValueError("index-map point count does not match the sample output")
    return {
        "lineage_schema_version": _CLASSIFIED_LINEAGE_SCHEMA_VERSION,
        "completion_manifest_digest": completion.completion_digest,
        "publication_proof_digest": proof.proof_digest,
        "classified_las_sha256": completion.classified_las_sha256,
        "classified_las_point_count": completion.classified_las_point_count,
        "classified_parent_source_id": classified.source_id,
        "classified_parent_source_file_sha256": classified.source_file_sha256,
        "classified_parent_identity_record_digest": (
            classified.source_identity_record_digest
        ),
        "sample_source_id": source_id,
        "sample_las_sha256": _file_sha256(output_path),
        "sample_point_count": output_point_count,
        "sample_index_map_sha256": _file_sha256(index_map_path),
        "index_map_original_index_count": int(original_index.shape[0]),
    }


def _recheck_parent_lineage(
    lineage: dict[str, Any],
    *,
    sample: Path,
    sample_point_count: int,
    index_map_path: Path,
    index_map_length: int,
    index_map_source_id: str,
) -> None:
    """Re-check a recorded parent lineage against the actual sample artifacts."""

    if lineage.get("lineage_schema_version") != _CLASSIFIED_LINEAGE_SCHEMA_VERSION:
        raise ValueError("unsupported parent lineage schema version")
    if lineage.get("sample_las_sha256") != _file_sha256(sample):
        raise ValueError("recorded sample LAS digest does not match the sample file")
    if lineage.get("sample_point_count") != sample_point_count:
        raise ValueError("recorded sample point count does not match the sample")
    if lineage.get("index_map_original_index_count") != index_map_length:
        raise ValueError("recorded index-map length does not match the index map")
    if lineage.get("sample_index_map_sha256") != _file_sha256(index_map_path):
        raise ValueError("recorded index-map digest does not match the index map")
    if lineage.get("sample_source_id") != index_map_source_id:
        raise ValueError("recorded sample source id does not match the index map")
    for name in (
        "completion_manifest_digest",
        "publication_proof_digest",
        "classified_las_sha256",
        "classified_parent_source_file_sha256",
        "classified_parent_identity_record_digest",
    ):
        value = lineage.get(name)
        if not (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"parent lineage {name} must be a valid SHA-256")
    parent_point_count = lineage.get("classified_las_point_count")
    if not isinstance(parent_point_count, int) or parent_point_count < 0:
        raise ValueError("classified_las_point_count must be a non-negative integer")


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


def _z_max_detail(
    *,
    xyz_parts: list[np.ndarray],
    intensity_parts: list[np.ndarray],
    class_parts: list[np.ndarray],
    rgb_parts: list[np.ndarray],
    max_xyz: np.ndarray,
    total_points: int,
) -> dict[str, Any]:
    xyz = _concat_or_empty_2d(xyz_parts, width=3, dtype=np.float64)
    intensity = _concat_or_empty_1d(intensity_parts, dtype=np.float64)
    classification = _concat_or_empty_1d(class_parts, dtype=np.int64)
    rgb = _concat_or_empty_2d(rgb_parts, width=3, dtype=np.float64)
    count = int(len(xyz))
    detail: dict[str, Any] = {
        "count": count,
        "ratio": 0.0 if total_points == 0 else count / total_points,
        "xyz_range": {
            "min": _list_or_none(xyz.min(axis=0)) if count else None,
            "max": _list_or_none(xyz.max(axis=0)) if count else None,
        },
        "touches_z_max": bool(count and float(xyz[:, 2].max()) >= float(max_xyz[2])),
    }
    if len(intensity):
        detail["intensity"] = {
            "min": float(intensity.min()),
            "max": float(intensity.max()),
            "mean": float(intensity.mean()),
        }
    if len(classification):
        values, counts = np.unique(classification, return_counts=True)
        detail["classification_counts"] = {
            str(int(value)): int(count_value)
            for value, count_value in zip(values, counts, strict=True)
        }
    if len(rgb):
        detail["rgb"] = {
            "min": [float(value) for value in rgb.min(axis=0)],
            "max": [float(value) for value in rgb.max(axis=0)],
            "mean": [float(value) for value in rgb.mean(axis=0)],
        }
    return detail


def _concat_or_empty_1d(parts: list[np.ndarray], *, dtype: Any) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


def _concat_or_empty_2d(
    parts: list[np.ndarray],
    *,
    width: int,
    dtype: Any,
) -> np.ndarray:
    if not parts:
        return np.empty((0, width), dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)
