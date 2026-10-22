"""Bounded, provenance-closed Ground/HAG worksets for Stage-B spans.

The validated Ground implementation is intentionally unchanged.  This module
only streams the source once and reproduces the established source-wide coarse
representative domain without holding all source points in memory.  SMRF's
raster domain is source-wide; empirical comparison showed that cropping before
SMRF changes classification and HAG even with a window-sized context.  The
resulting compact LAS is consumed by the existing SMRF + HAG-NN pipeline.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import laspy
import numpy as np

from core.config.fast_baseline import canonical_digest

if TYPE_CHECKING:
    from algorithms.spans.project_adapter import SpanAnalysisInput


GROUND_WORKSET_SCHEMA = "treeline-stage-b-ground-workset-v3"
GROUND_WORKSET_ATTRIBUTE_POLICY = "LAS14_POINT_FORMAT_6_STANDARD_NO_RGB_NO_EXTRA"
GROUND_WORKSET_DOMAIN = "FULL_SOURCE_COARSE_REPRESENTATIVES"
DEFAULT_SMRF_CONTEXT_M = 16.0
_MIN_CHUNK_POINTS = 65_536
_MAX_CHUNK_POINTS = 1_000_000
_FALLBACK_AVAILABLE_BYTES = 4 * 1024**3


class GroundWorksetError(RuntimeError):
    """Fail-closed workset preparation or validation error."""


@dataclass(frozen=True, slots=True)
class GroundWorkset:
    schema: str
    status: str
    source_binding_digest: str
    source_fingerprint: tuple[int, int]
    region_digest: str
    region_count: int
    smrf_context_m: float
    source_point_count: int
    active_region_point_count: int
    workset_point_count: int
    source_read_points: int
    source_full_pass_count: int
    chunk_size_points: int
    available_memory_start_bytes: int
    input_point_size_bytes: int
    workset_point_size_bytes: int
    representative_voxel_size_m: float
    merge_run_count: int
    workset_las_path: str
    original_index_path: str
    workset_las_sha256: str
    original_index_sha256: str
    attribute_policy: str = GROUND_WORKSET_ATTRIBUTE_POLICY
    workset_domain: str = GROUND_WORKSET_DOMAIN

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_fingerprint"] = list(self.source_fingerprint)
        return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _available_memory_bytes() -> int:
    try:
        import psutil

        value = int(psutil.virtual_memory().available)
        if value > 0:
            return value
    except (ImportError, OSError, ValueError):
        pass
    return _FALLBACK_AVAILABLE_BYTES


def adaptive_source_chunk_size(
    point_size_bytes: int, *, available_memory_bytes: int | None = None
) -> int:
    """Choose a conservative source chunk without machine-specific constants.

    A source chunk temporarily owns the packed point record, scaled XY arrays,
    union masks and a compact output record.  Budgeting eight source-record
    equivalents per point is intentionally conservative and leaves at least
    75% of currently available RAM to PDAL, the desktop, and the OS.
    """

    if type(point_size_bytes) is not int or point_size_bytes <= 0:
        raise ValueError("point_size_bytes must be a positive integer")
    available = (
        _available_memory_bytes()
        if available_memory_bytes is None
        else int(available_memory_bytes)
    )
    if available <= 0:
        available = _FALLBACK_AVAILABLE_BYTES
    budget = max(32 * 1024**2, available // 4)
    estimated_bytes_per_point = max(128, point_size_bytes * 8)
    selected = budget // estimated_bytes_per_point
    return int(min(_MAX_CHUNK_POINTS, max(_MIN_CHUNK_POINTS, selected)))


def _xyz(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise GroundWorksetError(f"GROUND_WORKSET_REGION_INVALID:{name}")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise GroundWorksetError(f"GROUND_WORKSET_REGION_INVALID:{name}")
    return result


def _region_record(span: SpanAnalysisInput, smrf_context_m: float) -> dict[str, Any]:
    start = _xyz(span.from_endpoint_geometry.get("base_xyz_absolute_m"), "from_base")
    end = _xyz(span.to_endpoint_geometry.get("base_xyz_absolute_m"), "to_base")
    delta = end[:2] - start[:2]
    length = float(np.linalg.norm(delta))
    if not math.isfinite(length) or length <= 0.0:
        raise GroundWorksetError("GROUND_WORKSET_REGION_INVALID:length")
    axis = delta / length
    perpendicular = np.asarray((-axis[1], axis[0]), dtype=np.float64)
    geometry = dict(span.span_geometry)
    half_width = float(geometry.get("core_half_width_m", 9.0))
    halo = float(geometry.get("longitudinal_halo_m", geometry.get("halo_m", 6.0)))
    if not math.isfinite(half_width) or half_width <= 0.0 or not math.isfinite(halo) or halo < 0.0:
        raise GroundWorksetError("GROUND_WORKSET_REGION_INVALID:margins")
    return {
        "span_id": span.span_id,
        "start_xy": [float(start[0]), float(start[1])],
        "axis_xy": [float(axis[0]), float(axis[1])],
        "perpendicular_xy": [float(perpendicular[0]), float(perpendicular[1])],
        "length_m": length,
        "half_width_m": half_width + smrf_context_m,
        "before_m": halo + smrf_context_m,
        "after_m": halo + smrf_context_m,
        "publish_half_width_m": half_width,
        "publish_halo_m": halo,
    }


def ground_region_plan(
    span_inputs: Sequence[SpanAnalysisInput], *, smrf_context_m: float = DEFAULT_SMRF_CONTEXT_M
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Return a deterministic union plan for all logical spans."""

    if not math.isfinite(smrf_context_m) or smrf_context_m <= 0.0:
        raise GroundWorksetError("GROUND_WORKSET_CONTEXT_INVALID")
    if not span_inputs:
        raise GroundWorksetError("GROUND_WORKSET_SPANS_EMPTY")
    regions = tuple(
        _region_record(item, float(smrf_context_m))
        for item in sorted(span_inputs, key=lambda value: (value.group_id, value.span_id))
    )
    digest = canonical_digest(
        {
            "schema": GROUND_WORKSET_SCHEMA,
            "smrf_context_m": float(smrf_context_m),
            "regions": regions,
            "attribute_policy": GROUND_WORKSET_ATTRIBUTE_POLICY,
        }
    )
    return digest, regions


def _union_mask(x: np.ndarray, y: np.ndarray, regions: Sequence[Mapping[str, Any]]) -> np.ndarray:
    mask = np.zeros(len(x), dtype=bool)
    for region in regions:
        start = np.asarray(region["start_xy"], dtype=np.float64)
        axis = np.asarray(region["axis_xy"], dtype=np.float64)
        perpendicular = np.asarray(region["perpendicular_xy"], dtype=np.float64)
        dx = x - start[0]
        dy = y - start[1]
        longitudinal = dx * axis[0] + dy * axis[1]
        lateral = dx * perpendicular[0] + dy * perpendicular[1]
        mask |= (
            (longitudinal >= -float(region["before_m"]))
            & (longitudinal <= float(region["length_m"]) + float(region["after_m"]))
            & (np.abs(lateral) <= float(region["half_width_m"]))
        )
    return mask


def _workset_header(source_header: laspy.LasHeader) -> laspy.LasHeader:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.asarray(source_header.scales, dtype=np.float64)
    header.offsets = np.asarray(source_header.offsets, dtype=np.float64)
    try:
        crs = source_header.parse_crs()
        if crs is not None:
            header.add_crs(crs)
    except (ImportError, ValueError, RuntimeError):
        pass
    return header


def _copy_standard_points(
    selected: laspy.ScaleAwarePointRecord, header: laspy.LasHeader
) -> laspy.ScaleAwarePointRecord:
    output = laspy.ScaleAwarePointRecord.zeros(len(selected), header=header)
    source_dimensions = set(selected.point_format.dimension_names)
    for name in header.point_format.dimension_names:
        if name in source_dimensions:
            output[name] = selected[name]
    return output


_REPRESENTATIVE_RUN_DTYPE = np.dtype(
    [
        ("voxel_key", "<i8"),
        ("source_index", "<i8"),
        ("X", "<i4"),
        ("Y", "<i4"),
        ("Z", "<i4"),
        ("classification", "u1"),
    ],
    align=False,
)


def _voxel_keys(
    xyz: np.ndarray,
    source_min: np.ndarray,
    voxel_size_m: float,
    cell_shape: np.ndarray,
) -> np.ndarray:
    cell = np.floor((xyz - source_min) / voxel_size_m).astype(np.int64)
    if np.any(cell < 0) or np.any(cell >= cell_shape):
        raise GroundWorksetError("GROUND_WORKSET_VOXEL_OUT_OF_RANGE")
    return (cell[:, 0] * cell_shape[1] + cell[:, 1]) * cell_shape[2] + cell[:, 2]


def _write_representative_run(
    path: Path,
    points: laspy.ScaleAwarePointRecord,
    mask: np.ndarray,
    *,
    source_offset: int,
    source_min: np.ndarray,
    voxel_size_m: float,
    cell_shape: np.ndarray,
) -> tuple[int, int]:
    rows = np.flatnonzero(mask)
    selected = points[rows]
    xyz = np.column_stack(
        (np.asarray(selected.x), np.asarray(selected.y), np.asarray(selected.z))
    ).astype(np.float64, copy=False)
    keys = _voxel_keys(xyz, source_min, voxel_size_m, cell_shape)
    unique_keys, first = np.unique(keys, return_index=True)
    records = np.empty(len(unique_keys), dtype=_REPRESENTATIVE_RUN_DTYPE)
    records["voxel_key"] = unique_keys
    records["source_index"] = rows[first].astype(np.int64, copy=False) + source_offset
    records["X"] = np.asarray(selected.X)[first]
    records["Y"] = np.asarray(selected.Y)[first]
    records["Z"] = np.asarray(selected.Z)[first]
    records["classification"] = np.asarray(selected.classification, dtype=np.uint8)[first]
    records.tofile(path)
    return len(rows), len(records)


def _write_merged_representatives(
    run_paths: Sequence[Path],
    output_las: Path,
    output_original_index: Path,
    header: laspy.LasHeader,
    *,
    batch_size: int = 65_536,
) -> int:
    runs = [np.memmap(path, mode="r", dtype=_REPRESENTATIVE_RUN_DTYPE) for path in run_paths]
    positions = np.zeros(len(runs), dtype=np.int64)
    heap: list[tuple[int, int]] = []
    for run_index, run in enumerate(runs):
        if len(run):
            heapq.heappush(heap, (int(run[0]["voxel_key"]), run_index))
    batches: list[np.void] = []
    output_count = 0

    def flush(writer: laspy.LasWriter, original_stream: Any) -> None:
        nonlocal output_count
        if not batches:
            return
        records = np.asarray(batches, dtype=_REPRESENTATIVE_RUN_DTYPE)
        points = laspy.ScaleAwarePointRecord.zeros(len(records), header=header)
        points.X = records["X"]
        points.Y = records["Y"]
        points.Z = records["Z"]
        points.classification = records["classification"]
        writer.write_points(points)
        original_stream.write(records["source_index"].astype("<i8", copy=False).tobytes())
        output_count += len(records)
        batches.clear()

    with (
        laspy.open(output_las, mode="w", header=header) as writer,
        output_original_index.open("wb") as original_stream,
    ):
        while heap:
            voxel_key, run_index = heapq.heappop(heap)
            candidates: list[tuple[int, np.void]] = []
            while True:
                row = runs[run_index][positions[run_index]]
                candidates.append((int(row["source_index"]), row))
                positions[run_index] += 1
                if positions[run_index] < len(runs[run_index]):
                    next_key = int(runs[run_index][positions[run_index]]["voxel_key"])
                    heapq.heappush(heap, (next_key, run_index))
                if not heap or heap[0][0] != voxel_key:
                    break
                _same_key, run_index = heapq.heappop(heap)
            batches.append(min(candidates, key=lambda item: item[0])[1])
            if len(batches) >= batch_size:
                flush(writer, original_stream)
        flush(writer, original_stream)
    del runs
    return output_count


def _load_cached(
    manifest_path: Path,
    *,
    source_binding_digest: str,
    source_fingerprint: tuple[int, int],
    region_digest: str,
) -> GroundWorkset | None:
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["source_fingerprint"] = tuple(int(value) for value in payload["source_fingerprint"])
        result = GroundWorkset(**payload)
        workset = Path(result.workset_las_path)
        original = Path(result.original_index_path)
        if (
            result.schema != GROUND_WORKSET_SCHEMA
            or result.status != "COMPLETE"
            or result.source_binding_digest != source_binding_digest
            or result.source_fingerprint != source_fingerprint
            or result.region_digest != region_digest
            or result.attribute_policy != GROUND_WORKSET_ATTRIBUTE_POLICY
            or result.workset_domain != GROUND_WORKSET_DOMAIN
            or not workset.is_file()
            or not original.is_file()
            or original.stat().st_size != result.workset_point_count * np.dtype("<i8").itemsize
        ):
            return None
        with laspy.open(workset) as reader:
            if int(reader.header.point_count) != result.workset_point_count:
                return None
        if _sha256_file(workset) != result.workset_las_sha256:
            return None
        if _sha256_file(original) != result.original_index_sha256:
            return None
        return result
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def ensure_ground_workset(
    source: Path,
    output_root: Path,
    source_identity: Mapping[str, Any],
    span_inputs: Sequence[SpanAnalysisInput],
    *,
    smrf_context_m: float = DEFAULT_SMRF_CONTEXT_M,
) -> tuple[str, GroundWorkset]:
    """Build or validate one bounded union workset without rewriting *source*."""

    source = Path(source).resolve()
    output_root = Path(output_root).resolve()
    if not source.is_file():
        raise GroundWorksetError(f"GROUND_WORKSET_SOURCE_MISSING:{source}")
    stat = source.stat()
    fingerprint = (int(stat.st_size), int(stat.st_mtime_ns))
    source_binding_digest = canonical_digest(dict(source_identity))
    _publication_digest, regions = ground_region_plan(
        span_inputs, smrf_context_m=smrf_context_m
    )
    region_digest = canonical_digest(
        {
            "schema": GROUND_WORKSET_SCHEMA,
            "domain": GROUND_WORKSET_DOMAIN,
            "representative": "first_source_row_per_xyz_voxel",
            "attribute_policy": GROUND_WORKSET_ATTRIBUTE_POLICY,
        }
    )
    # The full digest is persisted and validated in the manifest.  A bounded
    # prefix gives ample collision resistance while preventing WinError 206
    # once temporary run filenames are appended on default Windows systems.
    root = output_root / region_digest[:20]
    manifest_path = root / "ground-workset.json"
    cached = _load_cached(
        manifest_path,
        source_binding_digest=source_binding_digest,
        source_fingerprint=fingerprint,
        region_digest=region_digest,
    )
    if cached is not None:
        return "SKIPPED_FROM_CACHE", cached

    root.mkdir(parents=True, exist_ok=True)
    workset_path = root / "ground-workset.las"
    original_path = root / "source-original-index.bin"
    temporary_las = root / "ground-workset.tmp.las"
    temporary_original = root / "source-original-index.tmp.bin"
    temporary_manifest = root / "ground-workset.json.tmp"
    for path in (temporary_las, temporary_original, temporary_manifest):
        path.unlink(missing_ok=True)

    available = _available_memory_bytes()
    try:
        with laspy.open(source, mode="r") as reader:
            source_point_count = int(reader.header.point_count)
            input_point_size = int(reader.header.point_format.size)
            header = _workset_header(reader.header)
            source_min = np.asarray(reader.header.mins, dtype=np.float64)
            source_max = np.asarray(reader.header.maxs, dtype=np.float64)
            xy_area = float(source_max[0] - source_min[0]) * float(
                source_max[1] - source_min[1]
            )
            from algorithms.vegetation.fast_baseline_pipeline import (
                choose_representative_voxel_size,
            )

            voxel_size = choose_representative_voxel_size(source_point_count, xy_area)
            cell_shape = np.floor((source_max - source_min) / voxel_size).astype(np.int64) + 2
            chunk_size = adaptive_source_chunk_size(
                input_point_size, available_memory_bytes=available
            )
            source_read_points = 0
            active_region_point_count = 0
            run_root = root / "runs"
            run_root.mkdir(parents=True, exist_ok=True)
            run_paths: list[Path] = []
            source_offset = 0
            while True:
                start = source_offset
                points = reader.read_points(chunk_size)
                if len(points) == 0:
                    break
                source_offset += len(points)
                source_read_points += len(points)
                # The mature/manual and role-preparation routes define Ground
                # on the source-wide coarse representative domain.  Keep that
                # domain exact; the Span union is a downstream publication and
                # query boundary, not an SMRF input crop.
                mask = np.ones(len(points), dtype=bool)
                if np.any(mask):
                    run_path = run_root / f"run-{len(run_paths):06d}.bin"
                    active_count, _representative_count = _write_representative_run(
                        run_path,
                        points,
                        mask,
                        source_offset=start,
                        source_min=source_min,
                        voxel_size_m=voxel_size,
                        cell_shape=cell_shape,
                    )
                    active_region_point_count += active_count
                    run_paths.append(run_path)
        if source_read_points != source_point_count:
            raise GroundWorksetError("GROUND_WORKSET_SOURCE_READ_INCOMPLETE")
        if not run_paths:
            raise GroundWorksetError("GROUND_WORKSET_EMPTY")
        workset_point_count = _write_merged_representatives(
            run_paths,
            temporary_las,
            temporary_original,
            header,
        )
        for run_path in run_paths:
            run_path.unlink(missing_ok=True)
        if workset_point_count <= 0:
            raise GroundWorksetError("GROUND_WORKSET_EMPTY")
        os.replace(temporary_las, workset_path)
        os.replace(temporary_original, original_path)
        result = GroundWorkset(
            schema=GROUND_WORKSET_SCHEMA,
            status="COMPLETE",
            source_binding_digest=source_binding_digest,
            source_fingerprint=fingerprint,
            region_digest=region_digest,
            region_count=len(regions),
            smrf_context_m=float(smrf_context_m),
            source_point_count=source_point_count,
            active_region_point_count=active_region_point_count,
            workset_point_count=workset_point_count,
            source_read_points=source_read_points,
            source_full_pass_count=1,
            chunk_size_points=chunk_size,
            available_memory_start_bytes=available,
            input_point_size_bytes=input_point_size,
            workset_point_size_bytes=int(header.point_format.size),
            representative_voxel_size_m=float(voxel_size),
            merge_run_count=len(run_paths),
            workset_las_path=str(workset_path),
            original_index_path=str(original_path),
            workset_las_sha256=_sha256_file(workset_path),
            original_index_sha256=_sha256_file(original_path),
        )
        temporary_manifest.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
        return "PASS", result
    except Exception:
        for path in (temporary_las, temporary_original, temporary_manifest):
            path.unlink(missing_ok=True)
        raise


__all__ = [
    "DEFAULT_SMRF_CONTEXT_M",
    "GROUND_WORKSET_SCHEMA",
    "GroundWorkset",
    "GroundWorksetError",
    "adaptive_source_chunk_size",
    "ensure_ground_workset",
    "ground_region_plan",
]
