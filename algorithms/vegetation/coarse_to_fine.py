"""Coarse-to-fine source-native building blocks for the FAST one-click route.

The reviewed 0.25 m representative cloud is retained exclusively as a **coarse
routing** representation: global scene inspection, Ground/HAG, broad vegetation
region routing and broad wire region routing.  It is never used as the final
geometric representation for single-tree segmentation or conductor
reconstruction.

This module provides the coarse-to-fine handoff:

* recovering the original raw source rows represented by an accepted coarse
  region (exclusive ``raw_to_representative_id.npy`` membership),
* building a deterministic, memory-bounded fine representative working set at
  a centrally configured resolution,
* writing a contract-compliant A1 artifact pair for the recovered wire region
  so the existing A2A/A2B/A3/A4/W1 production chain can reconstruct a
  source-native W1.

``raw_count`` remains the physical/raw weighting used by V4 density and is never
replaced by representative count.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from algorithms.vegetation.fast_research import (
    FAST_RESEARCH_SCHEMA_VERSION,
    NON_CERTIFIED_INPUT_MARKER,
    FastGroundResult,
    FastResearchError,
    FastVegetationInputResult,
    canonical_digest,
    fast_ground_content_digest,
    sha256_file,
    validate_fast_v4_arrays,
)
from algorithms.vegetation.fast_weighted_preprocess import WeightedWorkingSet

# ---------------------------------------------------------------------------
# Central coarse-to-fine configuration
# ---------------------------------------------------------------------------

# Reviewed coarse routing scale; must stay the default for scalability.
COARSE_VOXEL_SIZE_M = 0.25
# First practical fine vegetation representative resolution.  Chosen because
# the current 0.25 m representation visually collapses broad tree-crown
# geometry required by V4.  This is deliberately NOT an optimization sweep.
VEGETATION_FINE_VOXEL_SIZE_M = 0.10
# Fine wire representative resolution.  Conductors are thin; 0.10 m voxel
# downsampling keeps one point every 0.10 m along a conductor (line topology
# preserved) while bounding the wire crop volume.  It is wire-specific and not
# merely symmetric with the vegetation scale.
WIRE_FINE_VOXEL_SIZE_M = 0.10
WIRE_FINE_MODE = "FINE_REPRESENTATIVES"
COARSE_TO_FINE_SCHEMA = "coarse-to-fine-v1"
_RAW_REGION_CHUNK = 1_000_000


# ---------------------------------------------------------------------------
# Recovered raw region store
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawRegion:
    """Bounded, deterministic recovered raw source rows in the scene frame."""

    xyz_local_m_path: str
    original_index_path: str
    classification_path: str
    count: int
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]

    def xyz(self) -> np.memmap:
        return np.memmap(self.xyz_local_m_path, dtype="<f4", mode="r", shape=(self.count, 3))

    def original_index(self) -> np.memmap:
        return np.memmap(self.original_index_path, dtype="<i8", mode="r", shape=(self.count,))

    def classification(self) -> np.memmap:
        return np.memmap(self.classification_path, dtype="<u1", mode="r", shape=(self.count,))


class _RawRegionWriter:
    """Streaming binary writer so a large raw region never enters RAM at once."""

    def __init__(self, root: Path, *, origin_xyz: tuple[float, float, float]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        self._xyz = (root / "xyz_local.bin").open("wb")
        self._idx = (root / "original_index.bin").open("wb")
        self._cls = (root / "classification.bin").open("wb")
        self._count = 0
        self._origin = tuple(float(value) for value in origin_xyz)
        self._min = np.full(3, np.inf, dtype=np.float64)
        self._max = np.full(3, -np.inf, dtype=np.float64)

    def append(
        self,
        xyz_local: np.ndarray,
        original_index: np.ndarray,
        classification: np.ndarray,
    ) -> None:
        xyz = np.asarray(xyz_local, dtype="<f4")
        idx = np.asarray(original_index, dtype="<i8")
        cls = np.asarray(classification, dtype="<u1")
        count = len(xyz)
        if idx.shape != (count,) or cls.shape != (count,):
            raise FastResearchError("COARSE_TO_FINE_REGION_ALIGNMENT_INVALID")
        if count:
            if not np.isfinite(xyz).all() or np.any(idx < 0):
                raise FastResearchError("COARSE_TO_FINE_REGION_POINT_INVALID")
            xyz.tofile(self._xyz)
            idx.tofile(self._idx)
            cls.tofile(self._cls)
            self._count += count
            values = xyz.astype("<f8", copy=False)
            self._min = np.minimum(self._min, values.min(axis=0))
            self._max = np.maximum(self._max, values.max(axis=0))

    def close(self) -> RawRegion:
        self._close_handles()
        if self._count <= 0:
            raise FastResearchError("COARSE_TO_FINE_REGION_EMPTY")
        return RawRegion(
            xyz_local_m_path=str(self._root / "xyz_local.bin"),
            original_index_path=str(self._root / "original_index.bin"),
            classification_path=str(self._root / "classification.bin"),
            count=self._count,
            bounds_min_xyz=tuple(float(value) for value in self._min),
            bounds_max_xyz=tuple(float(value) for value in self._max),
            origin_xyz=self._origin,
        )

    def abort(self) -> None:
        """Close open handles without validation (used on a failed recovery)."""
        self._close_handles()

    def _close_handles(self) -> None:
        for handle in (self._xyz, self._idx, self._cls):
            if handle is not None and not handle.closed:
                handle.close()
        self._xyz = self._idx = self._cls = None


def _write_region_sidecar(region: RawRegion, sidecar_path: Path) -> None:
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = sidecar_path.with_name(sidecar_path.name + ".tmp")
    temporary.write_text(json.dumps(asdict(region), sort_keys=True), encoding="utf-8")
    os.replace(temporary, sidecar_path)


def load_raw_region(sidecar_path: Path | str) -> RawRegion:
    """Reconstruct a recovered raw region from its persisted sidecar."""
    return _load_region_sidecar(Path(sidecar_path))


def _load_region_sidecar(sidecar_path: Path) -> RawRegion:
    payload = json.loads(sidecar_path.read_text("utf-8"))
    return RawRegion(
        xyz_local_m_path=str(payload["xyz_local_m_path"]),
        original_index_path=str(payload["original_index_path"]),
        classification_path=str(payload["classification_path"]),
        count=int(payload["count"]),
        bounds_min_xyz=tuple(float(value) for value in payload["bounds_min_xyz"]),
        bounds_max_xyz=tuple(float(value) for value in payload["bounds_max_xyz"]),
        origin_xyz=tuple(float(value) for value in payload["origin_xyz"]),
    )


# ---------------------------------------------------------------------------
# Raw region recovery
# ---------------------------------------------------------------------------


def recover_vegetation_region_raw(
    source_las: Path,
    working_set: WeightedWorkingSet,
    retained_representative_ids: np.ndarray,
    output_root: Path,
    *,
    exclude_original_index: np.ndarray | None = None,
    chunk_size: int = _RAW_REGION_CHUNK,
) -> RawRegion:
    """Recover the raw source rows represented by accepted coarse vegetation.

    Every source row inherits exactly one coarse representative through
    ``raw_to_representative_id.npy``; rows whose representative is retained are
    recovered in raw row order.  Optional ``exclude_original_index`` (exact
    source-native wire members) never enters the region.
    """
    source_las = Path(source_las).resolve()
    if not source_las.is_file():
        raise FastResearchError("COARSE_TO_FINE_VEG_SOURCE_MISSING")
    retained = np.asarray(retained_representative_ids, dtype=np.uint32)
    if retained.ndim != 1:
        raise FastResearchError("COARSE_TO_FINE_VEG_RETAINED_INVALID")
    if len(retained) and int(retained.max()) >= working_set.representative_count:
        raise FastResearchError("COARSE_TO_FINE_VEG_RETAINED_RANGE")
    is_retained = np.zeros(working_set.representative_count, dtype=bool)
    is_retained[retained] = True
    excluded = (
        None
        if exclude_original_index is None
        else np.asarray(exclude_original_index, dtype=np.int64)
    )
    if excluded is not None and (excluded.ndim != 1 or len(excluded) and excluded.min() < 0):
        raise FastResearchError("COARSE_TO_FINE_VEG_EXCLUDE_INVALID")

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    raw_to_rep = np.load(working_set.raw_to_representative_npy_path, mmap_mode="r")
    if raw_to_rep.shape != (working_set.source_point_count,):
        raise FastResearchError("COARSE_TO_FINE_VEG_MAPPING_INVALID")
    origin = np.asarray(working_set.source_bounds_min_xyz, dtype=np.float64)
    writer = _RawRegionWriter(output_root, origin_xyz=tuple(float(value) for value in origin))
    offset = 0
    try:
        with laspy.open(source_las) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                count = len(chunk)
                original = np.arange(offset, offset + count, dtype=np.int64)
                reps = np.asarray(raw_to_rep[original], dtype=np.uint32)
                mask = is_retained[reps]
                if excluded is not None and len(excluded):
                    mask &= ~np.isin(original, excluded)
                if np.any(mask):
                    xyz_absolute = np.column_stack(
                        (np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z))
                    ).astype(np.float64, copy=False)
                    writer.append(
                        xyz_absolute[mask] - origin,
                        original[mask],
                        np.asarray(chunk.classification, dtype=np.uint8)[mask],
                    )
                offset += count
        if offset != working_set.source_point_count:
            raise FastResearchError("COARSE_TO_FINE_VEG_SOURCE_COUNT_CHANGED")
        region = writer.close()
    except Exception:
        writer.abort()
        raise
    finally:
        del raw_to_rep
    _write_region_sidecar(region, output_root / "region_sidecar.json")
    return region


def recover_wire_and_high_recall_regions_raw(
    source_las: Path,
    working_set: WeightedWorkingSet,
    wire_representative_ids: np.ndarray,
    wire_output_root: Path,
    vegetation_output_root: Path,
    *,
    chunk_size: int = _RAW_REGION_CHUNK,
) -> tuple[RawRegion, RawRegion]:
    """Recover Wire and all-source Vegetation regions in one exact source pass.

    Vegetation HIGH RECALL retains every source row.  Wire retains exactly the
    rows whose coarse representative is in ``wire_representative_ids``.  The
    two writers receive the same decoded source chunk, preserving original row
    order and the established role semantics while eliminating a redundant
    second LAS traversal.
    """

    source_las = Path(source_las).resolve()
    if not source_las.is_file():
        raise FastResearchError("COARSE_TO_FINE_VEG_SOURCE_MISSING")
    retained = np.asarray(wire_representative_ids, dtype=np.uint32)
    if retained.ndim != 1 or (
        len(retained) and int(retained.max()) >= working_set.representative_count
    ):
        raise FastResearchError("COARSE_TO_FINE_VEG_RETAINED_INVALID")
    is_wire = np.zeros(working_set.representative_count, dtype=bool)
    is_wire[retained] = True
    raw_to_rep = np.load(working_set.raw_to_representative_npy_path, mmap_mode="r")
    if raw_to_rep.shape != (working_set.source_point_count,):
        raise FastResearchError("COARSE_TO_FINE_VEG_MAPPING_INVALID")
    origin = np.asarray(working_set.source_bounds_min_xyz, dtype=np.float64)
    wire_writer = _RawRegionWriter(
        Path(wire_output_root).resolve(),
        origin_xyz=tuple(float(value) for value in origin),
    )
    vegetation_writer = _RawRegionWriter(
        Path(vegetation_output_root).resolve(),
        origin_xyz=tuple(float(value) for value in origin),
    )
    offset = 0
    try:
        with laspy.open(source_las) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                count = len(chunk)
                original = np.arange(offset, offset + count, dtype=np.int64)
                xyz_absolute = np.column_stack(
                    (np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z))
                ).astype(np.float64, copy=False)
                classification = np.asarray(chunk.classification, dtype=np.uint8)
                xyz_local = xyz_absolute - origin
                vegetation_writer.append(xyz_local, original, classification)
                wire_mask = is_wire[np.asarray(raw_to_rep[original], dtype=np.uint32)]
                if np.any(wire_mask):
                    wire_writer.append(
                        xyz_local[wire_mask],
                        original[wire_mask],
                        classification[wire_mask],
                    )
                offset += count
        if offset != working_set.source_point_count:
            raise FastResearchError("COARSE_TO_FINE_VEG_SOURCE_COUNT_CHANGED")
        wire_region = wire_writer.close()
        vegetation_region = vegetation_writer.close()
    except Exception:
        wire_writer.abort()
        vegetation_writer.abort()
        raise
    finally:
        del raw_to_rep
    _write_region_sidecar(
        wire_region, Path(wire_output_root).resolve() / "region_sidecar.json"
    )
    _write_region_sidecar(
        vegetation_region,
        Path(vegetation_output_root).resolve() / "region_sidecar.json",
    )
    return wire_region, vegetation_region


def recover_wire_region_raw(
    source_las: Path,
    region_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    output_root: Path,
    *,
    origin_xyz: tuple[float, float, float],
    chunk_size: int = _RAW_REGION_CHUNK,
) -> RawRegion:
    """Recover the raw source rows spatially belonging to a wire region box."""
    source_las = Path(source_las).resolve()
    if not source_las.is_file():
        raise FastResearchError("COARSE_TO_FINE_WIRE_SOURCE_MISSING")
    minimum = np.asarray(region_bounds[0], dtype=np.float64)
    maximum = np.asarray(region_bounds[1], dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or np.any(maximum < minimum):
        raise FastResearchError("COARSE_TO_FINE_WIRE_BOUNDS_INVALID")
    origin = np.asarray(origin_xyz, dtype=np.float64)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    writer = _RawRegionWriter(output_root, origin_xyz=tuple(float(value) for value in origin))
    offset = 0
    try:
        with laspy.open(source_las) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                count = len(chunk)
                xyz_absolute = np.column_stack(
                    (np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z))
                ).astype(np.float64, copy=False)
                mask = np.all(xyz_absolute >= minimum, axis=1) & np.all(
                    xyz_absolute <= maximum, axis=1
                )
                if np.any(mask):
                    original = np.arange(offset, offset + count, dtype=np.int64)
                    writer.append(
                        xyz_absolute[mask] - origin,
                        original[mask],
                        np.asarray(chunk.classification, dtype=np.uint8)[mask],
                    )
                offset += count
        region = writer.close()
    except Exception:
        writer.abort()
        raise
    _write_region_sidecar(region, output_root / "region_sidecar.json")
    return region


# ---------------------------------------------------------------------------
# Deterministic fine voxelization (memory-bounded)
# ---------------------------------------------------------------------------


def build_fine_representatives(
    region: RawRegion,
    output_path: Path,
    *,
    voxel_size_m: float,
    chunk_size: int = _RAW_REGION_CHUNK,
) -> dict[str, np.ndarray]:
    """Voxelize a recovered raw region at ``voxel_size_m`` without RAM blowup.

    The key array is sorted on disk and unique runs are merged by a bounded
    scan; the representative per voxel is the first raw row (lowest original
    index) because the region is stored in raw row order.
    """
    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise FastResearchError("COARSE_TO_FINE_FINE_VOXEL_SIZE_INVALID")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = region.count
    if count <= 0:
        raise FastResearchError("COARSE_TO_FINE_FINE_EMPTY_REGION")
    bounds_min = np.asarray(region.bounds_min_xyz, dtype=np.float64)
    cell_shape = np.floor((np.asarray(region.bounds_max_xyz) - bounds_min) / voxel_size_m).astype(
        np.int64
    ) + 2
    if np.any(cell_shape <= 0):
        raise FastResearchError("COARSE_TO_FINE_FINE_GRID_INVALID")

    key_path = output_path.with_name(output_path.stem + ".keys")
    key_path.unlink(missing_ok=True)
    keys = np.lib.format.open_memmap(key_path, mode="w+", dtype="<i8", shape=(count,))
    xyz = region.xyz()
    try:
        for start in range(0, count, chunk_size):
            stop = min(count, start + chunk_size)
            keys[start:stop] = _linear_keys(
                xyz[start:stop].astype("<f8", copy=False), bounds_min, voxel_size_m, cell_shape
            )
        keys.flush()
        keys.sort()
        keys.flush()
        unique_keys, run_counts = _unique_sorted(keys, chunk_size)
    finally:
        del keys
        key_path.unlink(missing_ok=True)
    representative_count = len(unique_keys)
    if representative_count == 0:
        raise FastResearchError("COARSE_TO_FINE_FINE_NO_VOXELS")
    if representative_count > np.iinfo(np.uint32).max:
        raise FastResearchError("COARSE_TO_FINE_FINE_REPRESENTATIVE_LIMIT")

    claimed = np.zeros(representative_count, dtype=bool)
    representative_xyz = np.empty((representative_count, 3), dtype="<f4")
    representative_original = np.empty(representative_count, dtype="<i8")
    representative_classification = np.empty(representative_count, dtype="<u1")
    raw_count = np.zeros(representative_count, dtype="<u4")
    original_index = region.original_index()
    classification = region.classification()
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        keys_chunk = _linear_keys(
            xyz[start:stop].astype("<f8", copy=False), bounds_min, voxel_size_m, cell_shape
        )
        positions = np.searchsorted(unique_keys, keys_chunk)
        if np.any(positions >= representative_count) or np.any(
            unique_keys[positions] != keys_chunk
        ):
            raise FastResearchError("COARSE_TO_FINE_FINE_KEY_MISMATCH")
        first_positions, rows = _first_unclaimed_rows(positions, claimed)
        if len(first_positions):
            # Keep the FIRST raw row per voxel: because the region is streamed
            # in raw row order, the first occurrence is the lowest original
            # index.  ``minimum.at`` avoids sorting the million-row chunk only
            # to locate that first occurrence; it has identical duplicate and
            # cross-chunk semantics to ``np.unique(..., return_index=True)``.
            representative_xyz[first_positions] = xyz[start:stop][rows]
            representative_original[first_positions] = original_index[start:stop][rows]
            representative_classification[first_positions] = classification[start:stop][rows]
            claimed[first_positions] = True
        chunk_counts = np.bincount(positions, minlength=representative_count)
        raw_count += chunk_counts.astype(np.uint32, copy=False)
    del xyz, original_index, classification
    if not bool(claimed.all()):
        raise FastResearchError("COARSE_TO_FINE_FINE_COVERAGE_INVALID")
    if int(np.sum(raw_count, dtype=np.int64)) != count:
        raise FastResearchError("COARSE_TO_FINE_FINE_RAW_COUNT_CONSERVATION")
    voxel_key = _decode_keys(unique_keys, cell_shape)
    arrays = {
        "representative_id": np.arange(representative_count, dtype="<u4"),
        "original_index": representative_original,
        "xyz_local_m": representative_xyz,
        "raw_count": raw_count,
        "classification": representative_classification,
        "voxel_key": voxel_key,
        "voxel_point_count": run_counts.astype("<i4", copy=False),
        "voxel_size_m": np.asarray(float(voxel_size_m), dtype="<f8"),
        "grid_min_xyz": bounds_min.astype("<f8", copy=False),
        "grid_shape": cell_shape.astype("<i8", copy=False),
    }
    _write_npz_atomic(output_path, **arrays)
    return arrays


def _first_unclaimed_rows(
    positions: np.ndarray, claimed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return unclaimed voxel ids and their first row without an O(N log N) sort."""

    position = np.asarray(positions, dtype=np.int64)
    is_claimed = np.asarray(claimed, dtype=bool)
    new_rows = np.flatnonzero(~is_claimed[position])
    if not len(new_rows):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    sentinel = np.int64(len(position))
    first_row = np.full(len(is_claimed), sentinel, dtype=np.int64)
    np.minimum.at(first_row, position[new_rows], new_rows)
    first_positions = np.flatnonzero(first_row != sentinel)
    return first_positions, first_row[first_positions]


def map_region_rows_to_tree(
    region: RawRegion,
    fine_reps_npz_path: Path,
    fine_original_index: np.ndarray,
    fine_tree_id: np.ndarray,
    *,
    chunk_size: int = _RAW_REGION_CHUNK,
) -> np.ndarray:
    """Return the per-raw-region-row V4 tree id (``-1`` when unassigned).

    Recomputes each recovered raw row's fine voxel key and maps it to the fine
    representative's V4 tree id.  The returned array is aligned with the region
    raw rows (raw row order), so ``original_index[i]`` carries ``result[i]``.
    """
    fine_original = np.asarray(fine_original_index, dtype=np.int64)
    tree = np.asarray(fine_tree_id, dtype=np.int32)
    if fine_original.ndim != 1 or tree.shape != fine_original.shape:
        raise FastResearchError("COARSE_TO_FINE_BACKMAP_ARRAY_INVALID")
    with np.load(fine_reps_npz_path, allow_pickle=False) as stream:
        voxel_size_m = float(np.asarray(stream["voxel_size_m"]).item())
        grid_min = np.asarray(stream["grid_min_xyz"], dtype=np.float64)
        cell_shape = np.asarray(stream["grid_shape"], dtype=np.int64)
        rep_voxel_key = np.asarray(stream["voxel_key"], dtype=np.int64)
        rep_original = np.asarray(stream["original_index"], dtype=np.int64)
    unique_keys = _fine_region_keys(fine_reps_npz_path, cell_shape)
    tree_by_key = np.full(len(unique_keys), -1, dtype=np.int32)
    # The V4 input is a HAG/continuation-filtered subset of the persisted fine
    # representatives; locate each V4 input rep inside the persisted set.
    rep_order = np.argsort(rep_original, kind="stable")
    selected = np.searchsorted(rep_original[rep_order], fine_original)
    if np.any(selected >= len(rep_order)) or not np.array_equal(
        rep_original[rep_order][selected], fine_original
    ):
        raise FastResearchError("COARSE_TO_FINE_BACKMAP_REP_MISMATCH")
    selected_voxel_key = rep_voxel_key[rep_order[selected]]
    rep_linear = (
        selected_voxel_key[:, 0] * cell_shape[1] + selected_voxel_key[:, 1]
    ) * cell_shape[2] + selected_voxel_key[:, 2]
    positions = np.searchsorted(unique_keys, rep_linear)
    if np.any(positions >= len(unique_keys)) or np.any(
        unique_keys[positions] != rep_linear
    ):
        raise FastResearchError("COARSE_TO_FINE_BACKMAP_KEY_MISMATCH")
    tree_by_key[positions] = tree

    result = np.full(region.count, -1, dtype="<i4")
    xyz = region.xyz()
    for start in range(0, region.count, chunk_size):
        stop = min(region.count, start + chunk_size)
        keys_chunk = _linear_keys(
            xyz[start:stop].astype("<f8", copy=False), grid_min, voxel_size_m, cell_shape
        )
        row_positions = np.searchsorted(unique_keys, keys_chunk)
        if np.any(row_positions >= len(unique_keys)) or np.any(
            unique_keys[row_positions] != keys_chunk
        ):
            raise FastResearchError("COARSE_TO_FINE_BACKMAP_ROW_KEY_MISMATCH")
        result[start:stop] = tree_by_key[row_positions]
    del xyz
    return result


def _fine_region_keys(fine_reps_npz_path: Path, cell_shape: np.ndarray) -> np.ndarray:
    with np.load(fine_reps_npz_path, allow_pickle=False) as stream:
        voxel_key = np.asarray(stream["voxel_key"], dtype=np.int64)
    linear = (voxel_key[:, 0] * cell_shape[1] + voxel_key[:, 1]) * cell_shape[2] + voxel_key[:, 2]
    return np.unique(linear)


def _linear_keys(
    xyz: np.ndarray,
    bounds_min: np.ndarray,
    voxel_size_m: float,
    cell_shape: np.ndarray,
) -> np.ndarray:
    cell = np.floor((xyz - bounds_min) / voxel_size_m).astype(np.int64)
    if np.any(cell < 0) or np.any(cell >= cell_shape):
        raise FastResearchError("COARSE_TO_FINE_FINE_VOXEL_OUT_OF_RANGE")
    return (cell[:, 0] * cell_shape[1] + cell[:, 1]) * cell_shape[2] + cell[:, 2]


def _decode_keys(unique_keys: np.ndarray, cell_shape: np.ndarray) -> np.ndarray:
    remainder = unique_keys % (cell_shape[1] * cell_shape[2])
    x_cell = unique_keys // (cell_shape[1] * cell_shape[2])
    y_cell = remainder // cell_shape[2]
    z_cell = remainder % cell_shape[2]
    return np.column_stack((x_cell, y_cell, z_cell)).astype(np.int64)


def _unique_sorted(values: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return unique values and run counts from a sorted (possibly memmap) array."""
    unique_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    current_value: int | None = None
    current_count = 0
    length = len(values)
    for start in range(0, length, chunk_size):
        stop = min(length, start + chunk_size)
        chunk = values[start:stop]
        if not len(chunk):
            continue
        boundaries = np.flatnonzero(np.diff(chunk) != 0)
        run_starts = np.concatenate(([0], boundaries + 1))
        run_lengths = np.concatenate((boundaries + 1, [len(chunk)])) - run_starts
        run_values = chunk[run_starts]
        first = int(run_values[0])
        if current_value is not None and current_value == first:
            current_count += int(run_lengths[0])
            if len(run_values) > 1:
                # The first run (which continues the pending run) ends inside
                # this chunk because a different value follows; flush it now.
                unique_parts.append(np.asarray([current_value], dtype="<i8"))
                count_parts.append(np.asarray([current_count], dtype="<i8"))
                if len(run_values) > 2:
                    unique_parts.append(run_values[1:-1])
                    count_parts.append(run_lengths[1:-1])
                current_value = int(run_values[-1])
                current_count = int(run_lengths[-1])
            # else: the whole chunk continues the pending run; keep it pending
            # because it may extend into the next chunk.
        else:
            if current_value is not None and current_count > 0:
                # A different value begins the chunk, so the pending run ended.
                unique_parts.append(np.asarray([current_value], dtype="<i8"))
                count_parts.append(np.asarray([current_count], dtype="<i8"))
            if len(run_values):
                # Only the final run stays pending; every earlier run completed.
                if len(run_values) > 1:
                    unique_parts.append(run_values[:-1])
                    count_parts.append(run_lengths[:-1])
                current_value = int(run_values[-1])
                current_count = int(run_lengths[-1])
            else:
                current_value = None
                current_count = 0
    if current_value is not None and current_count > 0:
        unique_parts.append(np.asarray([current_value], dtype="<i8"))
        count_parts.append(np.asarray([current_count], dtype="<i8"))
    unique = np.concatenate(unique_parts) if unique_parts else np.empty(0, dtype="<i8")
    counts = np.concatenate(count_parts) if count_parts else np.empty(0, dtype="<i8")
    if len(unique) and len(unique) != int(np.count_nonzero(np.diff(unique)) + 1):
        raise FastResearchError("COARSE_TO_FINE_UNIQUE_SCAN_INVALID")
    return unique, counts


# ---------------------------------------------------------------------------
# Fine vegetation HAG (coarse ground surface, HAG-NN semantics)
# ---------------------------------------------------------------------------


def estimate_fine_vegetation_hag(
    fine_xyz_local_m: np.ndarray,
    origin_xyz: tuple[float, float, float],
    coarse_ground_xyz_absolute: np.ndarray,
    *,
    neighbor_count: int = 8,
    chunk_size: int = 1_000_000,
) -> np.ndarray:
    """Estimate HAG for fine representatives from the reviewed coarse ground.

    Uses the same nearest-ground-neighbour semantics as the reviewed
    ``filters.hag_nn`` stage: for every fine point the ground height is the
    median of the ``k`` nearest coarse ground representatives' Z.
    """
    from scipy.spatial import cKDTree

    ground_xy = np.asarray(coarse_ground_xyz_absolute[:, :2], dtype=np.float64)
    ground_z = np.asarray(coarse_ground_xyz_absolute[:, 2], dtype=np.float64)
    if not len(ground_xy):
        raise FastResearchError("COARSE_TO_FINE_HAG_NO_GROUND")
    if neighbor_count <= 0:
        raise FastResearchError("COARSE_TO_FINE_HAG_NEIGHBOR_INVALID")
    origin = np.asarray(origin_xyz, dtype=np.float64)
    xyz = np.asarray(fine_xyz_local_m, dtype=np.float64)
    tree = cKDTree(ground_xy)
    hag = np.empty(len(xyz), dtype="<f4")
    for start in range(0, len(xyz), chunk_size):
        stop = min(len(xyz), start + chunk_size)
        points_xy = xyz[start:stop, :2] + origin[:2]
        _distances, neighbors = tree.query(points_xy, k=min(neighbor_count, len(ground_xy)))
        if neighbors.ndim == 1:
            neighbors = neighbors[:, None]
        ground_estimate = np.median(ground_z[neighbors], axis=1)
        hag[start:stop] = (xyz[start:stop, 2] + origin[2]) - ground_estimate
    return hag


def load_coarse_ground_representatives(ground: FastGroundResult) -> np.ndarray:
    """Return absolute XYZ of coarse ground (class 2) representatives only."""
    output_las = Path(ground.output_las_path)
    if not output_las.is_file() or sha256_file(output_las) != ground.output_las_sha256:
        raise FastResearchError("COARSE_TO_FINE_GROUND_OUTPUT_INVALID")
    parts: list[np.ndarray] = []
    with laspy.open(output_las) as reader:
        for chunk in reader.chunk_iterator(1_000_000):
            classification = np.asarray(chunk.classification, dtype=np.uint8)
            rows = np.flatnonzero(classification == 2)
            if len(rows):
                parts.append(
                    np.column_stack(
                        (
                            np.asarray(chunk.x)[rows],
                            np.asarray(chunk.y)[rows],
                            np.asarray(chunk.z)[rows],
                        )
                    ).astype(np.float64, copy=False)
                )
    if not parts:
        raise FastResearchError("COARSE_TO_FINE_GROUND_NO_CLASS_2")
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Fine vegetation V4 input
# ---------------------------------------------------------------------------


def build_fine_vegetation_v4_input(
    *,
    source_las: Path,
    working_set: WeightedWorkingSet,
    ground: FastGroundResult,
    retained_vegetation_representative_ids: np.ndarray,
    origin_context: tuple[float, float, float],
    source_id: str,
    voxel_size_m: float,
    output_root: Path,
    wire_exclusion_original_index: np.ndarray | None = None,
    w1_continuation_xyz_local: np.ndarray | None = None,
    w1_continuation_tolerance_m: float = 0.30,
    minimum_hag_m: float = 0.3,
) -> tuple[FastVegetationInputResult, dict[str, Any]]:
    """Build the 0.10 m fine vegetation working set that V4 actually consumes.

    ``raw_count`` is conserved from the recovered region and V4 density remains
    raw-weighted.  Exact source-native wire members and (inside the wire region)
    points within ``w1_continuation_tolerance_m`` of a distance-ready W1 segment
    are excluded as conductor continuation.
    """
    output_root = Path(output_root).resolve()
    started = time.perf_counter()
    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise FastResearchError("COARSE_TO_FINE_FINE_VEG_VOXEL_SIZE_INVALID")

    region_root = output_root / "recovered_region"
    if region_root.exists():
        raise FastResearchError("COARSE_TO_FINE_FINE_VEG_OUTPUT_EXISTS")
    region = recover_vegetation_region_raw(
        source_las,
        working_set,
        retained_vegetation_representative_ids,
        region_root,
        exclude_original_index=wire_exclusion_original_index,
    )
    fine_root = output_root / "fine_representatives"
    fine_root.mkdir(parents=True, exist_ok=True)
    fine = build_fine_representatives(
        region, fine_root / "fine_representatives.npz", voxel_size_m=voxel_size_m
    )
    representative_count = len(fine["original_index"])
    if representative_count == 0:
        raise FastResearchError("COARSE_TO_FINE_FINE_VEG_EMPTY")

    coarse_ground = load_coarse_ground_representatives(ground)
    hag = estimate_fine_vegetation_hag(
        fine["xyz_local_m"],
        region.origin_xyz,
        coarse_ground,
    )
    origin = np.asarray(origin_context, dtype=np.float64)
    if not np.array_equal(origin, np.asarray(region.origin_xyz)):
        raise FastResearchError("COARSE_TO_FINE_FINE_VEG_ORIGIN_MISMATCH")

    valid = np.isfinite(hag) & (hag >= np.float32(minimum_hag_m))
    if w1_continuation_xyz_local is not None and len(w1_continuation_xyz_local):
        valid &= ~_near_w1_continuation(
            fine["xyz_local_m"], np.asarray(w1_continuation_xyz_local, dtype=np.float64),
            tolerance_m=w1_continuation_tolerance_m,
            fine_chunk_size=500_000,
        )

    clean_original = np.asarray(fine["original_index"][valid], dtype="<i8")
    clean_xyz = np.asarray(fine["xyz_local_m"][valid], dtype="<f8")
    clean_hag = np.asarray(hag[valid], dtype="<f4")
    clean_weight = np.asarray(fine["raw_count"][valid], dtype="<u4")
    clean_original, clean_xyz, clean_hag, clean_weight = _order_fine_arrays(
        clean_original, clean_xyz, clean_hag, clean_weight
    )
    validate_fast_v4_arrays(clean_original, clean_xyz, clean_hag)
    if len(clean_original) == 0:
        raise FastResearchError("COARSE_TO_FINE_FINE_VEG_EMPTY_AFTER_FILTER")
    if np.any(clean_weight == 0):
        raise FastResearchError("COARSE_TO_FINE_FINE_VEG_WEIGHT_INVALID")

    config_payload = {
        "schema_version": COARSE_TO_FINE_SCHEMA,
        "ground_digest": fast_ground_content_digest(ground),
        "working_set_config_digest": working_set.config_digest,
        "voxel_size_m": float(voxel_size_m),
        "source_id": source_id,
        "minimum_hag_m": float(minimum_hag_m),
        "region_raw_point_count": int(region.count),
        "wire_exclusion": (
            None if wire_exclusion_original_index is None
            else _array_digest(np.asarray(wire_exclusion_original_index))
        ),
        "w1_continuation_tolerance_m": float(w1_continuation_tolerance_m),
    }
    config_digest = canonical_digest(config_payload)
    output_npz = output_root / "fast_v4_input.npz"
    _write_npz_atomic(
        output_npz,
        original_index=clean_original,
        xyz_local_m=clean_xyz,
        hag_m=clean_hag,
        raw_count=clean_weight,
        candidate_reason_flags=np.zeros(len(clean_original), dtype="<u4"),
    )
    output_sha = sha256_file(output_npz)
    vegetation = FastVegetationInputResult(
        schema_version=FAST_RESEARCH_SCHEMA_VERSION,
        non_certified_marker=NON_CERTIFIED_INPUT_MARKER,
        ground_result_digest=fast_ground_content_digest(ground),
        config_digest=config_digest,
        output_npz_path=str(output_npz),
        output_npz_sha256=output_sha,
        point_count=len(clean_original),
        source_id=source_id,
        source_las_path=str(Path(source_las).resolve()),
        source_point_count=working_set.source_point_count,
        source_file_sha256=working_set.source_las_sha256,
        origin_context=tuple(float(value) for value in origin),
        selected_bounds_min_xyz=(
            float(region.bounds_min_xyz[0] + origin[0]),
            float(region.bounds_min_xyz[1] + origin[1]),
            float(region.bounds_min_xyz[2] + origin[2]),
        ),
        selected_bounds_max_xyz=(
            float(region.bounds_max_xyz[0] + origin[0]),
            float(region.bounds_max_xyz[1] + origin[1]),
            float(region.bounds_max_xyz[2] + origin[2]),
        ),
        minimum_hag_m=float(minimum_hag_m),
        admission_voxel_size_m=float(voxel_size_m),
        original_index_sha256=_array_digest(clean_original),
        xyz_sha256=_array_digest(clean_xyz),
        hag_sha256=_array_digest(clean_hag),
        elapsed_seconds=time.perf_counter() - started,
    )
    _write_json_atomic(output_root / "fast_v4_input.json", vegetation.to_dict())
    counts = {
        "region_raw_point_count": int(region.count),
        "fine_representative_count": representative_count,
        "hag_filtered_out": int(np.count_nonzero(~valid)),
        "final_vegetation_representatives": int(len(clean_original)),
        "final_vegetation_raw_weight": int(np.sum(clean_weight, dtype=np.int64)),
        "fine_reduction_ratio": (
            float(len(clean_original) / region.count) if region.count else None
        ),
        "raw_to_fine_ratio": (
            float(region.count / len(clean_original)) if len(clean_original) else None
        ),
    }
    return vegetation, {
        "counts": counts,
        "config_digest": config_digest,
        "schema": COARSE_TO_FINE_SCHEMA,
        "recovered_region_sidecar_path": str(region_root / "region_sidecar.json"),
        "fine_representatives_npz_path": str(fine_root / "fine_representatives.npz"),
        "region_bounds_min_xyz": list(region.bounds_min_xyz),
        "region_bounds_max_xyz": list(region.bounds_max_xyz),
    }


def _near_w1_continuation(
    fine_xyz_local_m: np.ndarray,
    w1_xyz_local: np.ndarray,
    *,
    tolerance_m: float,
    fine_chunk_size: int,
) -> np.ndarray:
    """Mark fine representatives close to distance-ready W1 sampled points."""
    from scipy.spatial import cKDTree

    near = np.zeros(len(fine_xyz_local_m), dtype=bool)
    if tolerance_m <= 0 or not len(w1_xyz_local):
        return near
    tree = cKDTree(np.asarray(w1_xyz_local, dtype=np.float64))
    for start in range(0, len(fine_xyz_local_m), fine_chunk_size):
        stop = min(len(fine_xyz_local_m), start + fine_chunk_size)
        distances, _indexes = tree.query(
            np.asarray(fine_xyz_local_m[start:stop], dtype=np.float64),
            k=1,
            distance_upper_bound=tolerance_m,
        )
        near[start:stop] = np.isfinite(distances)
    return near


def _order_fine_arrays(
    original_index: np.ndarray,
    xyz_local_m: np.ndarray,
    hag_m: np.ndarray,
    raw_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(original_index, kind="stable")
    return (
        original_index[order],
        xyz_local_m[order],
        hag_m[order],
        raw_count[order],
    )


# ---------------------------------------------------------------------------
# Wire region selection from the coarse scene
# ---------------------------------------------------------------------------


def compute_wire_region_bounds(
    candidate_representative_ids: np.ndarray,
    ground: FastGroundResult,
    *,
    origin_xyz: tuple[float, float, float],
    xy_margin_m: float = 6.0,
    z_ground_clearance_m: float = 1.5,
    z_top_margin_m: float = 2.0,
) -> dict[str, Any] | None:
    """Compute an absolute XYZ box around the accepted coarse wire candidates.

    The box deliberately reaches down to the recovered coarse ground surface so
    the A2B ground-grid stage can estimate a valid ground within the crop.
    """
    if candidate_representative_ids is None or not len(candidate_representative_ids):
        return None
    candidate = np.asarray(candidate_representative_ids, dtype=np.uint32)
    output_las = Path(ground.output_las_path)
    if not output_las.is_file() or sha256_file(output_las) != ground.output_las_sha256:
        raise FastResearchError("COARSE_TO_FINE_WIRE_GROUND_INVALID")
    want = set(int(value) for value in candidate)
    candidate_xyz: list[np.ndarray] = []
    all_ground_xyz: list[np.ndarray] = []
    with laspy.open(output_las) as reader:
        for chunk in reader.chunk_iterator(1_000_000):
            ids = np.asarray(chunk.representative_id, dtype=np.uint32)
            classification = np.asarray(chunk.classification, dtype=np.uint8)
            is_ground = classification == 2
            rows = np.flatnonzero(is_ground)
            if len(rows):
                ground_rows = np.column_stack(
                    (
                        np.asarray(chunk.x)[rows],
                        np.asarray(chunk.y)[rows],
                        np.asarray(chunk.z)[rows],
                    )
                )
                all_ground_xyz.append(ground_rows.astype(np.float64, copy=False))
            selected = np.fromiter(
                (int(value) in want for value in ids), dtype=bool, count=len(ids)
            )
            if np.any(selected):
                candidate_xyz.append(
                    np.column_stack(
                        (
                            np.asarray(chunk.x)[selected],
                            np.asarray(chunk.y)[selected],
                            np.asarray(chunk.z)[selected],
                        )
                    ).astype(np.float64, copy=False)
                )
    if not candidate_xyz:
        return None
    candidate_points = np.concatenate(candidate_xyz, axis=0)
    candidate_min = candidate_points.min(axis=0)
    candidate_max = candidate_points.max(axis=0)
    # Ground level: median Z of coarse ground representatives inside the
    # candidate XY extent; fall back to the scene-wide coarse ground median.
    box_ground = [
        part
        for part in _ground_within_xy(
            all_ground_xyz, candidate_min[:2], candidate_max[:2], chunk_size=1_000_000
        )
        if len(part)
    ]
    if box_ground:
        ground_level = float(np.median(np.concatenate(box_ground, axis=0)[:, 2]))
    elif all_ground_xyz:
        scene_ground = np.concatenate(all_ground_xyz, axis=0)
        ground_level = float(np.median(scene_ground[:, 2])) if len(scene_ground) else None
    else:
        ground_level = None
    if ground_level is None:
        raise FastResearchError("COARSE_TO_FINE_WIRE_NO_GROUND_LEVEL")
    minimum_xy = candidate_min[:2] - xy_margin_m
    maximum_xy = candidate_max[:2] + xy_margin_m
    z_min = ground_level - z_ground_clearance_m
    z_max = float(candidate_max[2]) + z_top_margin_m
    if z_max < z_min:
        z_max = z_min + 1.0
    origin = np.asarray(origin_xyz, dtype=np.float64)
    return {
        "xy_bounds_min": [float(minimum_xy[0]), float(minimum_xy[1])],
        "xy_bounds_max": [float(maximum_xy[0]), float(maximum_xy[1])],
        "z_bounds_min": z_min,
        "z_bounds_max": z_max,
        "bounds_min_xyz": [float(minimum_xy[0]), float(minimum_xy[1]), z_min],
        "bounds_max_xyz": [float(maximum_xy[0]), float(maximum_xy[1]), z_max],
        "candidate_representative_count": int(len(candidate_points)),
        "candidate_xy_span_m": [
            float(candidate_max[0] - candidate_min[0]),
            float(candidate_max[1] - candidate_min[1]),
        ],
        "candidate_z_span_m": float(candidate_max[2] - candidate_min[2]),
        "candidate_hag_range_m": None,
        "ground_level_median_z_m": ground_level,
        "origin_xyz": [float(value) for value in origin],
    }


# ---------------------------------------------------------------------------
# Wire A1 artifact + production chain
# ---------------------------------------------------------------------------

_WIRE_A1_ATTRIBUTE_NAMES = ("classification",)


def _ground_within_xy(
    ground_parts: list[np.ndarray],
    minimum_xy: np.ndarray,
    maximum_xy: np.ndarray,
    *,
    chunk_size: int,
) -> list[np.ndarray]:
    """Return ground parts whose XY coordinates fall inside the given box."""
    result: list[np.ndarray] = []
    for part in ground_parts:
        if not len(part):
            continue
        for start in range(0, len(part), chunk_size):
            stop = min(len(part), start + chunk_size)
            block = part[start:stop]
            inside = (
                (block[:, 0] >= minimum_xy[0])
                & (block[:, 0] <= maximum_xy[0])
                & (block[:, 1] >= minimum_xy[1])
                & (block[:, 1] <= maximum_xy[1])
            )
            if np.any(inside):
                result.append(block[inside])
    return result


def write_wire_region_a1(
    region: RawRegion,
    output_dir: Path,
    *,
    source_id: str,
    manifest_source_id: str,
    voxel_size_m: float,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write a contract-compliant A1 NPZ/JSON pair from the recovered region."""
    if not source_id:
        raise FastResearchError("COARSE_TO_FINE_A1_SOURCE_ID_INVALID")
    if not manifest_source_id:
        raise FastResearchError("COARSE_TO_FINE_A1_MANIFEST_INVALID")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fine_root = output_dir / "fine"
    fine_root.mkdir(parents=True, exist_ok=True)
    fine = build_fine_representatives(
        region, fine_root / "wire_fine_representatives.npz", voxel_size_m=voxel_size_m
    )
    origin_xyz = np.asarray(region.origin_xyz, dtype=np.float64)
    npz_path = output_dir / "scene_simple_preprocessed.npz"
    json_path = output_dir / "scene_simple_preprocessed.json"
    if npz_path.exists() or json_path.exists():
        raise FastResearchError("COARSE_TO_FINE_A1_OUTPUT_EXISTS")
    sample_index = np.arange(len(fine["original_index"]), dtype="<i8")
    classification = np.asarray(fine["classification"], dtype="<u1")
    arrays = {
        "xyz_local": np.asarray(fine["xyz_local_m"], dtype="<f4"),
        "origin_xyz": origin_xyz,
        "original_index": np.asarray(fine["original_index"], dtype="<i8"),
        "sample_index": sample_index,
        "voxel_key": np.asarray(fine["voxel_key"], dtype="<i8"),
        "voxel_point_count": np.asarray(fine["voxel_point_count"], dtype="<i4"),
        "source_id": np.asarray(source_id),
        "manifest_source_id": np.asarray(manifest_source_id),
        "attribute_names": np.asarray(list(_WIRE_A1_ATTRIBUTE_NAMES)),
    }
    arrays["classification"] = classification
    array_metadata = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }
    attributes_metadata = {
        "classification": {"shape": list(classification.shape), "dtype": str(classification.dtype)}
    }
    payload = {
        "source_id": source_id,
        "manifest_source_id": manifest_source_id,
        "output_npz_path": str(npz_path),
        "output_json_path": str(json_path),
        "origin_xyz": [float(value) for value in origin_xyz],
        "coordinate_mode": "FAST_RECOVERED_RAW_REGION_SCENE_FRAME",
        "arrays": {
            "xyz_local": array_metadata["xyz_local"],
            "origin_xyz": array_metadata["origin_xyz"],
            "original_index": array_metadata["original_index"],
            "sample_index": array_metadata["sample_index"],
            "voxel_key": array_metadata["voxel_key"],
            "voxel_point_count": array_metadata["voxel_point_count"],
            "attributes": attributes_metadata,
        },
        "statistics": {
            "downsampled_point_count": int(len(fine["original_index"])),
            "input_point_count": int(region.count),
            "voxel_size_m": float(voxel_size_m),
        },
        "algorithm": {
            "name": "fast_coarse_to_fine_wire_a1",
            "version": COARSE_TO_FINE_SCHEMA,
            "wire_fine_mode": WIRE_FINE_MODE,
        },
        "non_certified_marker": NON_CERTIFIED_INPUT_MARKER,
    }
    _write_npz_atomic(npz_path, **arrays)
    _write_json_atomic(json_path, payload)
    _validate_a1_arrays_quick(npz_path, json_path, expected_count=len(fine["original_index"]))
    return npz_path, json_path, {
        "fine_representative_count": int(len(fine["original_index"])),
        "region_raw_point_count": int(region.count),
        "voxel_size_m": float(voxel_size_m),
        "raw_count_sum": int(np.sum(fine["raw_count"], dtype=np.int64)),
    }


def _validate_a1_arrays_quick(npz_path: Path, json_path: Path, *, expected_count: int) -> None:
    payload = json.loads(json_path.read_text("utf-8"))
    with np.load(npz_path, allow_pickle=False) as stream:
        arrays = {name: stream[name] for name in stream.files}
    required = {
        "xyz_local",
        "origin_xyz",
        "original_index",
        "sample_index",
        "voxel_key",
        "voxel_point_count",
        "source_id",
        "manifest_source_id",
        "attribute_names",
        "classification",
    }
    missing = required - set(arrays)
    if missing:
        raise FastResearchError(f"COARSE_TO_FINE_A1_MISSING_ARRAYS: {sorted(missing)}")
    if arrays["xyz_local"].shape != (expected_count, 3) or arrays["xyz_local"].dtype != np.float32:
        raise FastResearchError("COARSE_TO_FINE_A1_XYZ_INVALID")
    for name in ("original_index", "sample_index"):
        if arrays[name].shape != (expected_count,) or arrays[name].dtype != np.int64:
            raise FastResearchError(f"COARSE_TO_FINE_A1_INDEX_INVALID: {name}")
        if len(np.unique(arrays[name])) != expected_count or np.any(arrays[name] < 0):
            raise FastResearchError(f"COARSE_TO_FINE_A1_INDEX_UNIQUE_INVALID: {name}")
    if payload.get("statistics", {}).get("downsampled_point_count") != expected_count:
        raise FastResearchError("COARSE_TO_FINE_A1_STATISTICS_INVALID")
    if Path(payload["output_npz_path"]).resolve() != npz_path.resolve():
        raise FastResearchError("COARSE_TO_FINE_A1_PATH_MISMATCH")


def fast_wire_separation_config(
    source_id: str, expected_input_point_count: int
) -> Any:
    """FAST-route A4 config: identical production math, route-appropriate confirmation.

    The formal ``fine_min_medium_low_fraction=0.40`` gate requires a mixed
    HIGH/MEDIUM/LOW candidate population that a clean conductor crop never has,
    so the production default would never confirm a real wire.  The formal
    ``*_max_axis_angle_deg=2.0`` gate is also too strict for a slightly sagging
    real conductor.  For the bounded source-native wire crop the confirmation
    mean-score gate, the mixed-population gate and the axis-angle gate are
    relaxed to physically reasonable values; every other geometric guard (span,
    coverage, direction consistency, transverse residual, gap, curvature) is
    unchanged.
    """
    from core.contracts import WireSeparationConfig

    return WireSeparationConfig(
        source_id=source_id,
        expected_input_point_count=int(expected_input_point_count),
        obvious_min_mean_candidate_score=0.70,
        fine_min_medium_low_fraction=1e-6,
        obvious_max_axis_angle_deg=5.0,
        fine_max_axis_angle_deg=5.0,
    )


def run_source_native_wire_chain(
    a1_npz: Path,
    a1_json: Path,
    chain_root: Path,
    *,
    source_id: str,
    a2b_min_points_per_cell: int = 4,
    expected_input_point_count: int | None = None,
) -> dict[str, Any]:
    """Run the existing A2A/A2B/A3/A4/W1 production chain on a recovered A1.

    Returns a manifest describing exactly which stages ran and whether a valid
    distance-ready W1 exists.  Chain execution failures fail closed with a
    stable code and reason instead of fabricating W1.
    """
    from core.contracts import (
        GeometryFeatureConfig,
        GroundEstimationConfig,
        WireCandidateConfig,
        WireReconstructionConfig,
    )

    from algorithms.candidates.wire import run_wire_candidate_selection
    from algorithms.features.geometry import run_geometry_feature_extraction
    from algorithms.features.ground import run_ground_height_estimation
    from algorithms.wires.reconstruct import (
        WireReconstructionError,
        load_wire_models,
        run_wire_reconstruction,
    )
    from algorithms.wires.separate import run_wire_separation

    chain_root = Path(chain_root).resolve()
    chain_root.mkdir(parents=True, exist_ok=True)
    stages: dict[str, Any] = {}
    blocked = {"TREE_WIRE_STATUS": "BLOCKED_NO_VALID_W1"}

    def fail(stage: str, error: Exception) -> dict[str, Any]:
        code = getattr(error, "code", None)
        code_value = code.value if hasattr(code, "value") else str(code)
        return {
            **blocked,
            "wire_chain_root": str(chain_root),
            "failed_stage": stage,
            "reason": f"{stage} failed: {code_value or type(error).__name__}: {error}",
            "stages": stages,
            "w1_distance_ready_segments": 0,
            "w1_polyline_count": 0,
        }

    try:
        a2a_config = GeometryFeatureConfig(source_id=source_id)
        run_geometry_feature_extraction(
            a1_npz,
            a1_json,
            chain_root / "outputs" / "features",
            config=a2a_config,
            project_root=chain_root,
        )
        stages["a2a"] = "PASS"
    except Exception as exc:  # noqa: BLE001 - fail closed, never fabricate W1
        return fail("a2a", exc)

    a2a_npz = chain_root / "outputs" / "features" / "scene_simple_geometry_features.npz"
    a2a_json = chain_root / "outputs" / "features" / "scene_simple_geometry_features.json"

    try:
        a2b_config = GroundEstimationConfig(
            source_id=source_id, min_points_per_cell=int(a2b_min_points_per_cell)
        )
        run_ground_height_estimation(
            a1_npz,
            a1_json,
            a2a_npz,
            a2a_json,
            chain_root / "outputs" / "features",
            config=a2b_config,
            project_root=chain_root,
        )
        stages["a2b"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        return fail("a2b", exc)

    a2b_npz = chain_root / "outputs" / "features" / "scene_simple_ground_hag.npz"
    a2b_json = chain_root / "outputs" / "features" / "scene_simple_ground_hag.json"

    try:
        a3_config = WireCandidateConfig(source_id=source_id)
        run_wire_candidate_selection(
            a1_npz,
            a1_json,
            a2a_npz,
            a2a_json,
            a2b_npz,
            a2b_json,
            chain_root / "outputs" / "candidates",
            config=a3_config,
            mode="rule",
            project_root=chain_root,
            generate_preview=False,
        )
        stages["a3"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        return fail("a3", exc)

    a3_npz = chain_root / "outputs" / "candidates" / "scene_simple_wire_candidates.npz"
    a3_json = chain_root / "outputs" / "candidates" / "scene_simple_wire_candidates.json"

    try:
        a3_payload = json.loads(a3_json.read_text("utf-8"))
        a3_point_count = int(a3_payload["statistics"]["input_point_count"])
        if expected_input_point_count is not None:
            a3_point_count = int(expected_input_point_count)
        a4_config = fast_wire_separation_config(source_id, a3_point_count)
        run_wire_separation(
            a3_npz,
            a3_json,
            chain_root / "outputs" / "wires",
            config=a4_config,
            project_root=chain_root,
            generate_preview=False,
        )
        stages["a4"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        return fail("a4", exc)

    a4_npz = chain_root / "outputs" / "wires" / "scene_simple_wire_instances.npz"
    a4_json = chain_root / "outputs" / "wires" / "scene_simple_wire_instances.json"

    # A4 confirmed no wire: W1 has nothing to reconstruct.  Fail closed here so
    # the reason is precise and the W1 zero-segment publish edge is never hit.
    try:
        a4_payload = json.loads(a4_json.read_text("utf-8"))
        confirmed_count = int(a4_payload["statistics"]["confirmed_wire_count"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return fail("a4_readback", exc)
    if confirmed_count <= 0:
        return {
            **blocked,
            "wire_chain_root": str(chain_root),
            "failed_stage": "a4_confirmation",
            "reason": "A4 未确认任何导线（无足够可靠的连续线状证据），禁止合成 W1。",
            "stages": stages,
            "a4_npz_path": str(a4_npz),
            "a4_json_path": str(a4_json),
            "w1_distance_ready_segments": 0,
            "w1_polyline_count": 0,
        }

    try:
        w1_config = WireReconstructionConfig(source_id=source_id)
        run_wire_reconstruction(
            a4_npz,
            a4_json,
            chain_root / "outputs" / "wire_models",
            config=w1_config,
            project_root=chain_root,
            generate_preview=False,
        )
        stages["w1"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        return fail("w1", exc)

    w1_npz = chain_root / "outputs" / "wire_models" / f"{source_id}_wire_models.npz"
    w1_json = chain_root / "outputs" / "wire_models" / f"{source_id}_wire_models.json"
    if not w1_npz.is_file() or not w1_json.is_file():
        return fail("w1_publish", FastResearchError("COARSE_TO_FINE_W1_OUTPUT_MISSING"))

    try:
        w1_result, w1_arrays = load_wire_models(w1_npz, w1_json)
    except WireReconstructionError as exc:
        return fail("w1_load", exc)

    distance_ready = np.asarray(w1_arrays["segment_distance_ready"], dtype=bool)
    distance_ready_segments = int(np.count_nonzero(distance_ready))
    polyline_count = int(np.count_nonzero(
        np.asarray(w1_arrays["wire_has_distance_ready_segments"], dtype=bool)
    ))
    if distance_ready_segments <= 0:
        return {
            **blocked,
            "wire_chain_root": str(chain_root),
            "failed_stage": "w1_validity",
            "reason": "W1 produced no distance-ready segments",
            "stages": stages,
            "w1_npz_path": str(w1_npz),
            "w1_json_path": str(w1_json),
            "a4_npz_path": str(a4_npz),
            "a4_json_path": str(a4_json),
            "w1_distance_ready_segments": 0,
            "w1_polyline_count": 0,
        }

    wire_member_original = _a4_confirmed_original_index(a4_npz)
    continuation_xyz = _w1_distance_ready_xyz(w1_arrays)
    a4_origin = _a4_origin_xyz(a4_json)
    return {
        "TREE_WIRE_STATUS": "VALID_W1",
        "wire_chain_root": str(chain_root),
        "failed_stage": None,
        "reason": None,
        "stages": stages,
        "w1_npz_path": str(w1_npz),
        "w1_json_path": str(w1_json),
        "a4_npz_path": str(a4_npz),
        "a4_json_path": str(a4_json),
        "w1_distance_ready_segments": distance_ready_segments,
        "w1_polyline_count": polyline_count,
        "w1_continuous_count": int(np.count_nonzero(
            np.asarray(w1_arrays["wire_continuous_distance_ready"], dtype=bool)
        )),
        "wire_member_original_index": wire_member_original,
        "w1_continuation_xyz_local": continuation_xyz,
        "w1_origin_xyz": [float(value) for value in a4_origin],
    }


def _a4_origin_xyz(a4_json: Path) -> tuple[float, float, float]:
    payload = json.loads(a4_json.read_text("utf-8"))
    origin = payload.get("origin_xyz")
    if origin is None or not isinstance(origin, list) or len(origin) != 3:
        raise FastResearchError("COARSE_TO_FINE_W1_A4_ORIGIN_INVALID")
    return (float(origin[0]), float(origin[1]), float(origin[2]))


def _a4_confirmed_original_index(a4_npz: Path) -> np.ndarray:
    with np.load(a4_npz, allow_pickle=False) as stream:
        original_index = np.asarray(stream["original_index"], dtype=np.int64)
        wire_id = np.asarray(stream["wire_id_by_point"], dtype=np.int32)
    return np.asarray(original_index[wire_id >= 0], dtype=np.int64)


def _w1_distance_ready_xyz(w1_arrays: dict[str, np.ndarray]) -> np.ndarray:
    ready = np.asarray(w1_arrays["segment_distance_ready"], dtype=bool)
    sample_offsets = np.asarray(w1_arrays["sampled_curve_offsets"], dtype=np.int64)
    sampled = np.asarray(w1_arrays["sampled_curve_xyz"], dtype=np.float64)
    parts: list[np.ndarray] = []
    for index, is_ready in enumerate(ready):
        if not bool(is_ready):
            continue
        start = int(sample_offsets[index])
        stop = int(sample_offsets[index + 1])
        if stop > start:
            parts.append(sampled[start:stop])
    if not parts:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------


def _array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), "utf-8")
    os.replace(temporary, path)


__all__ = [
    "COARSE_VOXEL_SIZE_M",
    "VEGETATION_FINE_VOXEL_SIZE_M",
    "WIRE_FINE_VOXEL_SIZE_M",
    "WIRE_FINE_MODE",
    "RawRegion",
    "build_fine_representatives",
    "build_fine_vegetation_v4_input",
    "compute_wire_region_bounds",
    "estimate_fine_vegetation_hag",
    "load_coarse_ground_representatives",
    "load_raw_region",
    "map_region_rows_to_tree",
    "recover_vegetation_region_raw",
    "recover_wire_region_raw",
    "run_source_native_wire_chain",
    "write_wire_region_a1",
]
