"""Span-local raw region recovery for the manual single span.

The expensive full-resolution source stream happens exactly ONCE for the whole
manual run: a bounding-box pass recovers every raw row inside the span halo.
The halo rows are then split by coarse-representative membership into the wire
region (rigid wire-candidate representatives) and the vegetation region
(retained-vegetation representatives) using the working-set raw-to-rep map.
All coordinates are scene-local (absolute minus the working-set origin), the
frame the span pipelines expect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.vegetation.coarse_to_fine import (
    FastResearchError,
    RawRegion,
    _RawRegionWriter,
    _write_region_sidecar,
    recover_wire_region_raw,
)
from algorithms.vegetation.fast_weighted_preprocess import WeightedWorkingSet


def recover_halo_box_region(
    source_las: Path,
    region_bounds_absolute: tuple[tuple[float, float, float], tuple[float, float, float]],
    output_root: Path,
    *,
    origin_xyz: tuple[float, float, float],
    chunk_size: int = 1_000_000,
) -> RawRegion:
    """Recover all raw source rows inside the span halo box (one full stream)."""

    return recover_wire_region_raw(
        source_las,
        region_bounds_absolute,
        output_root,
        origin_xyz=origin_xyz,
        chunk_size=chunk_size,
    )


def split_halo_region(
    halo_region: RawRegion,
    working_set: WeightedWorkingSet,
    wire_representative_ids: np.ndarray,
    vegetation_representative_ids: np.ndarray,
    wire_output_root: Path,
    vegetation_output_root: Path,
) -> tuple[RawRegion, RawRegion]:
    """Split the recovered halo rows into wire region and vegetation region.

    A halo raw row belongs to the wire region when its coarse representative is
    a wire candidate and to the vegetation region when its representative is
    retained vegetation.  Both outputs share the halo region's origin and raw
    row order.
    """

    wire_ids = np.asarray(wire_representative_ids, dtype=np.uint32)
    veg_ids = np.asarray(vegetation_representative_ids, dtype=np.uint32)
    is_wire = np.zeros(working_set.representative_count, dtype=bool)
    is_veg = np.zeros(working_set.representative_count, dtype=bool)
    if len(wire_ids):
        if int(wire_ids.max()) >= working_set.representative_count:
            raise FastResearchError("MANUAL_SPAN_WIRE_IDS_RANGE")
        is_wire[wire_ids] = True
    if len(veg_ids):
        if int(veg_ids.max()) >= working_set.representative_count:
            raise FastResearchError("MANUAL_SPAN_VEG_IDS_RANGE")
        is_veg[veg_ids] = True

    raw_to_rep = np.load(working_set.raw_to_representative_npy_path, mmap_mode="r")
    wire_root = Path(wire_output_root).resolve()
    veg_root = Path(vegetation_output_root).resolve()
    wire_root.mkdir(parents=True, exist_ok=True)
    veg_root.mkdir(parents=True, exist_ok=True)
    origin_xyz = tuple(float(value) for value in halo_region.origin_xyz)
    wire_writer = _RawRegionWriter(wire_root, origin_xyz=origin_xyz)
    veg_writer = _RawRegionWriter(veg_root, origin_xyz=origin_xyz)
    try:
        xyz = halo_region.xyz()
        original_index = halo_region.original_index()
        classification = halo_region.classification()
        for start in range(0, halo_region.count, 1_000_000):
            stop = min(halo_region.count, start + 1_000_000)
            chunk_index = np.asarray(original_index[start:stop], dtype=np.int64)
            reps = np.asarray(raw_to_rep[chunk_index], dtype=np.uint32)
            chunk_xyz = xyz[start:stop]
            chunk_cls = np.asarray(classification[start:stop], dtype=np.uint8)
            wire_mask = is_wire[reps]
            veg_mask = is_veg[reps]
            if np.any(wire_mask):
                wire_writer.append(
                    np.asarray(chunk_xyz[wire_mask], dtype=np.float64),
                    chunk_index[wire_mask],
                    chunk_cls[wire_mask],
                )
            if np.any(veg_mask):
                veg_writer.append(
                    np.asarray(chunk_xyz[veg_mask], dtype=np.float64),
                    chunk_index[veg_mask],
                    chunk_cls[veg_mask],
                )
        try:
            wire_region = wire_writer.close()
        except FastResearchError:
            from algorithms.vegetation.coarse_to_fine import RawRegion as _RawRegion

            wire_region = _RawRegion(
                xyz_local_m_path=str(wire_root / "xyz_local.bin"),
                original_index_path=str(wire_root / "original_index.bin"),
                classification_path=str(wire_root / "classification.bin"),
                count=0,
                bounds_min_xyz=halo_region.bounds_min_xyz,
                bounds_max_xyz=halo_region.bounds_max_xyz,
                origin_xyz=origin_xyz,
            )
        try:
            veg_region = veg_writer.close()
        except FastResearchError:
            from algorithms.vegetation.coarse_to_fine import RawRegion as _RawRegion

            veg_region = _RawRegion(
                xyz_local_m_path=str(veg_root / "xyz_local.bin"),
                original_index_path=str(veg_root / "original_index.bin"),
                classification_path=str(veg_root / "classification.bin"),
                count=0,
                bounds_min_xyz=halo_region.bounds_min_xyz,
                bounds_max_xyz=halo_region.bounds_max_xyz,
                origin_xyz=origin_xyz,
            )
    except Exception:
        wire_writer.abort()
        veg_writer.abort()
        raise
    finally:
        del raw_to_rep
    _write_region_sidecar(wire_region, wire_root / "region_sidecar.json")
    _write_region_sidecar(veg_region, veg_root / "region_sidecar.json")
    return wire_region, veg_region


__all__ = [
    "recover_halo_box_region",
    "split_halo_region",
]
