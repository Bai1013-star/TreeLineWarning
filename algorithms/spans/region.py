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


__all__ = [
    "SpanRegionSubset",
    "subset_raw_region",
]
