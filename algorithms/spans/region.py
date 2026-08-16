"""Span-local raw-region subsetting and fine representative helpers.

A scene-wide recovered ``RawRegion`` (vegetation or wire) is a memmap; per
span we stream a spatial box subset into a fresh ``RawRegion`` so fine
voxelization and downstream processing stay strictly span-local.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.vegetation.coarse_to_fine import (
    _RAW_REGION_CHUNK,
    FastResearchError,
    RawRegion,
    _RawRegionWriter,
)


def subset_raw_region(
    region: RawRegion,
    bounds_min_xyz: tuple[float, float, float],
    bounds_max_xyz: tuple[float, float, float],
    output_root: Any,
    *,
    chunk_size: int = _RAW_REGION_CHUNK,
) -> RawRegion:
    """Stream rows of *region* inside the box into a new span-local RawRegion.

    Bounds are in the same scene-local frame as *region*.xyz().  The result is
    a deterministic subset in raw row order with an unchanged local origin.
    """

    minimum = np.asarray(bounds_min_xyz, dtype=np.float64)
    maximum = np.asarray(bounds_max_xyz, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or np.any(maximum < minimum):
        raise FastResearchError("SPAN_REGION_BOUNDS_INVALID")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    writer = _RawRegionWriter(
        output_root, origin_xyz=tuple(float(value) for value in region.origin_xyz)
    )
    xyz = region.xyz()
    original_index = region.original_index()
    classification = region.classification()
    try:
        for start in range(0, region.count, chunk_size):
            stop = min(region.count, start + chunk_size)
            chunk_xyz = xyz[start:stop].astype(np.float64, copy=False)
            mask = np.all(chunk_xyz >= minimum, axis=1) & np.all(
                chunk_xyz <= maximum, axis=1
            )
            if np.any(mask):
                writer.append(
                    chunk_xyz[mask],
                    np.asarray(original_index[start:stop], dtype=np.int64)[mask],
                    np.asarray(classification[start:stop], dtype=np.uint8)[mask],
                )
        try:
            return writer.close()
        except FastResearchError:
            # A span halo may legitimately contain zero rows (sparse/occluded
            # span).  Return a deterministic empty RawRegion so the per-span
            # tree/wire pipelines fail closed instead of aborting the run.
            return RawRegion(
                xyz_local_m_path=str(output_root / "xyz_local.bin"),
                original_index_path=str(output_root / "original_index.bin"),
                classification_path=str(output_root / "classification.bin"),
                count=0,
                bounds_min_xyz=tuple(float(value) for value in minimum),
                bounds_max_xyz=tuple(float(value) for value in maximum),
                origin_xyz=tuple(float(value) for value in region.origin_xyz),
            )
    except Exception:
        writer.abort()
        raise


@dataclass(frozen=True, slots=True)
class SpanRegionSubset:
    """A span-local raw subset plus the scene-local origin it shares."""

    region: RawRegion
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SpanLocalFrame:
    """Horizontal local frame derived only from authoritative endpoints."""

    start_xyz_absolute_m: tuple[float, float, float]
    axis_xy: tuple[float, float]
    length_m: float
    origin_xyz: tuple[float, float, float]

    @classmethod
    def from_endpoints(
        cls,
        start_xyz_absolute_m: tuple[float, float, float],
        end_xyz_absolute_m: tuple[float, float, float],
        origin_xyz: tuple[float, float, float],
    ) -> SpanLocalFrame:
        start = np.asarray(start_xyz_absolute_m, dtype=np.float64)
        end = np.asarray(end_xyz_absolute_m, dtype=np.float64)
        delta = end[:2] - start[:2]
        length = float(np.linalg.norm(delta))
        if length <= 0.0:
            raise FastResearchError("SPAN_LOCAL_FRAME_DEGENERATE")
        return cls(
            tuple(float(value) for value in start),
            (float(delta[0] / length), float(delta[1] / length)),
            length,
            tuple(float(value) for value in origin_xyz),
        )


@dataclass(frozen=True, slots=True)
class SpanRegionSpec:
    """One auditable scientific region role in the Span-local frame."""

    role: str
    half_width_m: float
    longitudinal_before_m: float
    longitudinal_after_m: float
    z_min_m: float
    z_max_m: float

    def __post_init__(self) -> None:
        if not self.role or self.half_width_m <= 0.0:
            raise ValueError("invalid Span region specification")
        if self.longitudinal_before_m < 0.0 or self.longitudinal_after_m < 0.0:
            raise ValueError("longitudinal margins must be non-negative")
        if self.z_max_m < self.z_min_m:
            raise ValueError("z_max_m must not be below z_min_m")


@dataclass(frozen=True, slots=True)
class SpanRegionQueryResult:
    """Local region plus coarse/exact counts for persisted stage telemetry."""

    region: RawRegion
    aabb_query_points: int
    oriented_region_points: int
    tiles: tuple[str, ...] = ()
    total_tiles: int = 0
    intersecting_aabb_tiles: int = 0
    opened_tiles: int = 0
    points_read: int = 0


def span_region_bounds_absolute(
    frame: SpanLocalFrame, spec: SpanRegionSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Return the AABB used only as a coarse source query."""

    axis = np.asarray(frame.axis_xy, dtype=np.float64)
    perp = np.asarray((-axis[1], axis[0]), dtype=np.float64)
    start_xy = np.asarray(frame.start_xyz_absolute_m[:2], dtype=np.float64)
    corners = []
    for s in (-spec.longitudinal_before_m, frame.length_m + spec.longitudinal_after_m):
        for n in (-spec.half_width_m, spec.half_width_m):
            corners.append(start_xy + axis * s + perp * n)
    xy = np.asarray(corners, dtype=np.float64)
    return (
        np.asarray((xy[:, 0].min(), xy[:, 1].min(), spec.z_min_m), dtype=np.float64),
        np.asarray((xy[:, 0].max(), xy[:, 1].max(), spec.z_max_m), dtype=np.float64),
    )


def oriented_span_mask_absolute(
    xyz_absolute_m: np.ndarray, frame: SpanLocalFrame, spec: SpanRegionSpec
) -> np.ndarray:
    """Return the exact OBB/prism mask after a coarse AABB query."""

    xyz = np.asarray(xyz_absolute_m, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz_absolute_m must have shape (N, 3)")
    delta = xyz[:, :2] - np.asarray(frame.start_xyz_absolute_m[:2], dtype=np.float64)
    axis = np.asarray(frame.axis_xy, dtype=np.float64)
    s = delta @ axis
    n = delta @ np.asarray((-axis[1], axis[0]), dtype=np.float64)
    return (
        (s >= -spec.longitudinal_before_m)
        & (s <= frame.length_m + spec.longitudinal_after_m)
        & (np.abs(n) <= spec.half_width_m)
        & (xyz[:, 2] >= spec.z_min_m)
        & (xyz[:, 2] <= spec.z_max_m)
    )


def _empty_span_region(
    output_root: Path,
    *,
    origin_xyz: tuple[float, float, float],
    bounds_min_xyz: np.ndarray,
    bounds_max_xyz: np.ndarray,
) -> RawRegion:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("xyz_local.bin", "original_index.bin", "classification.bin"):
        (output_root / name).touch()
    return RawRegion(
        xyz_local_m_path=str(output_root / "xyz_local.bin"),
        original_index_path=str(output_root / "original_index.bin"),
        classification_path=str(output_root / "classification.bin"),
        count=0,
        bounds_min_xyz=tuple(float(value) for value in bounds_min_xyz),
        bounds_max_xyz=tuple(float(value) for value in bounds_max_xyz),
        origin_xyz=origin_xyz,
    )


def subset_raw_region_oriented(
    region: RawRegion,
    frame: SpanLocalFrame,
    spec: SpanRegionSpec,
    output_root: Any,
    *,
    chunk_size: int = _RAW_REGION_CHUNK,
) -> SpanRegionQueryResult:
    """Apply AABB broad query then exact Span-local mask to a raw memmap."""

    bounds_min, bounds_max = span_region_bounds_absolute(frame, spec)
    local_min = bounds_min - np.asarray(region.origin_xyz, dtype=np.float64)
    local_max = bounds_max - np.asarray(region.origin_xyz, dtype=np.float64)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    writer = _RawRegionWriter(output_root, origin_xyz=region.origin_xyz)
    xyz = region.xyz()
    original_index = region.original_index()
    classification = region.classification()
    aabb_count = 0
    oriented_count = 0
    try:
        for start in range(0, region.count, chunk_size):
            stop = min(region.count, start + chunk_size)
            chunk = np.asarray(xyz[start:stop], dtype=np.float64)
            aabb = np.all(chunk >= local_min, axis=1) & np.all(chunk <= local_max, axis=1)
            aabb_count += int(np.count_nonzero(aabb))
            if not np.any(aabb):
                continue
            exact = oriented_span_mask_absolute(chunk[aabb] + region.origin_xyz, frame, spec)
            if not np.any(exact):
                continue
            rows = np.flatnonzero(aabb)[exact]
            oriented_count += len(rows)
            writer.append(
                chunk[rows],
                np.asarray(original_index[start:stop], dtype=np.int64)[rows],
                np.asarray(classification[start:stop], dtype=np.uint8)[rows],
            )
        try:
            subset = writer.close()
        except FastResearchError:
            subset = _empty_span_region(
                output_root,
                origin_xyz=region.origin_xyz,
                bounds_min_xyz=local_min,
                bounds_max_xyz=local_max,
            )
        return SpanRegionQueryResult(subset, aabb_count, oriented_count)
    except Exception:
        writer.abort()
        raise


def subset_raw_tile_cache_oriented(
    tile_cache: Any,
    frame: SpanLocalFrame,
    spec: SpanRegionSpec,
    output_root: Any,
) -> SpanRegionQueryResult:
    """Query an existing validated RAW tile cache and apply the exact OBB mask."""

    bounds_min, bounds_max = span_region_bounds_absolute(frame, spec)
    midpoint = np.asarray(frame.start_xyz_absolute_m, dtype=np.float64)
    midpoint[:2] += np.asarray(frame.axis_xy, dtype=np.float64) * (frame.length_m / 2.0)
    radius = float(
        np.hypot(
            frame.length_m / 2.0
            + max(spec.longitudinal_before_m, spec.longitudinal_after_m),
            spec.half_width_m,
        )
    )
    if hasattr(tile_cache, "plan_query") and hasattr(tile_cache, "iter_tile_blocks"):
        if hasattr(tile_cache, "plan_oriented_query"):
            total_tiles, tile_ids = tile_cache.plan_oriented_query(frame, spec)
        else:
            total_tiles, tile_ids = tile_cache.plan_query(
                tuple(float(value) for value in midpoint),
                xy_radius_m=radius,
                z_context_below_m=max(0.0, float(midpoint[2] - spec.z_min_m)),
                z_context_above_m=max(0.0, float(spec.z_max_m - midpoint[2])),
            )
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        writer = _RawRegionWriter(output_root, origin_xyz=frame.origin_xyz)
        aabb_count = 0
        oriented_count = 0
        points_read = 0
        try:
            for _tile_id, block in tile_cache.iter_tile_blocks(tile_ids):
                xyz_absolute = np.column_stack((block["x"], block["y"], block["z"]))
                points_read += len(block)
                aabb_count += len(block)
                mask = oriented_span_mask_absolute(xyz_absolute, frame, spec)
                if np.any(mask):
                    rows = np.flatnonzero(mask)
                    oriented_count += len(rows)
                    writer.append(
                        xyz_absolute[rows] - np.asarray(frame.origin_xyz, dtype=np.float64),
                        np.asarray(block["source_index"], dtype=np.int64)[rows],
                        np.asarray(block["classification"], dtype=np.uint8)[rows],
                    )
            try:
                subset = writer.close()
            except FastResearchError:
                subset = _empty_span_region(
                    output_root,
                    origin_xyz=frame.origin_xyz,
                    bounds_min_xyz=bounds_min - np.asarray(frame.origin_xyz),
                    bounds_max_xyz=bounds_max - np.asarray(frame.origin_xyz),
                )
        except Exception:
            writer.abort()
            raise
        return SpanRegionQueryResult(
            subset,
            aabb_count,
            oriented_count,
            tile_ids,
            total_tiles,
            len(tile_ids),
            len(tile_ids),
            points_read,
        )
    result = tile_cache.query(
        tuple(float(value) for value in midpoint),
        xy_radius_m=radius,
        z_context_below_m=max(0.0, float(midpoint[2] - spec.z_min_m)),
        z_context_above_m=max(0.0, float(spec.z_max_m - midpoint[2])),
    )
    xyz_absolute = np.column_stack((result.x, result.y, result.z)).astype(np.float64, copy=False)
    mask = oriented_span_mask_absolute(xyz_absolute, frame, spec)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    writer = _RawRegionWriter(output_root, origin_xyz=frame.origin_xyz)
    classification = getattr(result, "classification", None)
    if classification is None:
        # Legacy test doubles and non-role adapters have no semantic class
        # channel.  Production role caches always provide it and are rejected
        # before this function when their contract is not validated.
        classification = np.zeros(len(xyz_absolute), dtype=np.uint8)
    try:
        if np.any(mask):
            writer.append(
                xyz_absolute[mask] - np.asarray(frame.origin_xyz, dtype=np.float64),
                np.asarray(result.source_index, dtype=np.int64)[mask],
                np.asarray(classification, dtype=np.uint8)[mask],
            )
            subset = writer.close()
        else:
            writer.abort()
            subset = _empty_span_region(
                output_root,
                origin_xyz=frame.origin_xyz,
                bounds_min_xyz=bounds_min - np.asarray(frame.origin_xyz),
                bounds_max_xyz=bounds_max - np.asarray(frame.origin_xyz),
            )
    except Exception:
        writer.abort()
        raise
    return SpanRegionQueryResult(
        subset,
        int(result.point_count),
        int(np.count_nonzero(mask)),
        tuple(result.tiles),
        int(getattr(result, "total_tiles", 0)),
        int(getattr(result, "intersecting_aabb_tiles", len(result.tiles))),
        int(getattr(result, "opened_tiles", len(result.tiles))),
        int(getattr(result, "points_read", result.point_count)),
    )


__all__ = [
    "SpanLocalFrame",
    "SpanRegionQueryResult",
    "SpanRegionSpec",
    "SpanRegionSubset",
    "oriented_span_mask_absolute",
    "span_region_bounds_absolute",
    "subset_raw_region",
    "subset_raw_region_oriented",
    "subset_raw_tile_cache_oriented",
]
