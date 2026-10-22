# ruff: noqa: E501
"""Per-span tree pipeline driver (vegetation region -> full-resolution trees).

Given one span, a scene-wide recovered vegetation ``RawRegion`` and the coarse
ground representatives, this driver streams the span halo subset, builds
0.10 m fine vegetation representatives, estimates HAG, runs the CHM /
adaptive-tree-top / marker-watershed algorithm, back-maps crown ids to the
full-resolution raw source rows and persists the per-span artifacts used for
review, reconciliation and clearance.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from core.config.fast_baseline import (
    TREE_BACKEND_CROWN_STEM_HYBRID,
    TREE_BACKEND_RAW_STEM_FIRST,
    TREE_BACKEND_LEGACY_WATERSHED,
    FastBaselineConfig,
)

from algorithms.poles.evidence import PoleEvidenceError
from algorithms.poles.topology import SpanRecord
from algorithms.spans.region import (
    SpanLocalFrame,
    SpanRegionQueryResult,
    SpanRegionSpec,
    oriented_span_mask_absolute,
    subset_raw_region_oriented,
)
from algorithms.spans.stem_guided_hybrid import (
    StemGuidedHybridConfig,
    segment_stem_guided_hybrid,
)
from algorithms.spans.tree import (
    assign_span_tree_membership,
    build_span_chm,
    build_span_tree_result,
    detect_span_tree_tops,
    segment_span_crowns,
    smooth_span_chm,
)
from algorithms.vegetation.coarse_to_fine import (
    build_fine_representatives,
    estimate_fine_vegetation_hag,
)

SPAN_TREE_RUNNER_SCHEMA = "fast-baseline-span-tree-runner-v1"


def _restore_validated_trunk_membership(
    region: Any,
    raw_tree_id: np.ndarray,
    hybrid: Any,
    coarse_ground_xyz_absolute: np.ndarray,
    origin_xyz: tuple[float, float, float],
    *,
    chunk_size: int = 500_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Restore original low-trunk rows around validated stems.

    Crown/HAG membership deliberately rejects near-ground rows.  That is safe
    for finding crowns but it must not become destructive denoising: the
    original trunk is required for the ground+1.30 m DBH section.  This pass
    only fills *unassigned* source rows inside a narrow cylinder around a
    stem that already passed the multi-slice verticality checks.  Ground and
    LAS noise classes are never restored, so vehicles and unrelated clusters
    away from a validated stem stay excluded.
    """

    from scipy.spatial import cKDTree

    result = np.asarray(raw_tree_id, dtype=np.int32)
    if result.shape != (int(region.count),):
        raise PoleEvidenceError("TREE_TRUNK_RESTORE_ALIGNMENT_INVALID")
    confirmed = {
        int(candidate.tree_id): candidate
        for candidate in hybrid.candidates
        if candidate.quality_state.value == "CONFIRMED_TREE"
        and candidate.stem_seed_ids
    }
    stem_by_id = {
        int(stem.stem_seed_id): stem for stem in hybrid.validated_stems
    }
    restore_specs: list[tuple[int, float, float, float, float]] = []
    for tree_id, candidate in sorted(confirmed.items()):
        for stem_id in candidate.stem_seed_ids:
            stem = stem_by_id.get(int(stem_id))
            if stem is None:
                continue
            # Keep the tube deliberately narrower than the candidate search
            # radius.  It is wide enough for a mature trunk/DBH slice while
            # avoiding nearby cars, poles and neighbouring crowns.
            radius = max(
                0.18,
                min(0.48, float(stem.cross_section_radius_m) * 1.25 + 0.06),
            )
            max_hag = max(
                1.65,
                min(5.5, float(stem.stem_vertical_extent_m) + 0.75),
            )
            restore_specs.append(
                (
                    tree_id,
                    float(stem.xy_local_m[0]),
                    float(stem.xy_local_m[1]),
                    radius,
                    max_hag,
                )
            )
    if not restore_specs:
        return result, {
            "mode": "VALIDATED_STEM_CYLINDER",
            "restored_raw_points": 0,
            "validated_stem_tubes": 0,
        }

    ground = np.asarray(coarse_ground_xyz_absolute, dtype=np.float64)
    if ground.ndim != 2 or ground.shape[1] != 3 or not len(ground):
        raise PoleEvidenceError("TREE_TRUNK_RESTORE_GROUND_INVALID")
    ground_index = cKDTree(ground[:, :2])
    origin = np.asarray(origin_xyz, dtype=np.float64)
    xyz = region.xyz()
    classification = region.classification()
    restored = 0
    quarantined_raw = 0
    restored_by_tree: dict[str, int] = {}
    planes = (getattr(hybrid, 'diagnostics', None) or {}).get('structural_planes', [])
    keep_stems = tuple(stem_by_id[sid] for candidate in confirmed.values()
        for sid in candidate.stem_seed_ids if sid in stem_by_id)
    for start in range(0, int(region.count), chunk_size):
        stop = min(int(region.count), start + chunk_size)
        points = np.asarray(xyz[start:stop], dtype=np.float64)
        current = result[start:stop]
        cls = np.asarray(classification[start:stop], dtype=np.uint8)
        non_tree = np.isin(cls, np.asarray([2, 6, 7, 18], dtype=np.uint8))
        quarantined_raw += int(np.count_nonzero((current >= 0) & non_tree))
        current[non_tree] = -1
        unassigned = current < 0
        if not np.any(unassigned) and not planes:
            continue
        # ASPRS: 2=ground, 6=building, 7=low noise, 18=high noise. Class 1 is retained
        # because photogrammetric tree trunks are commonly unclassified.
        eligible = unassigned & ~np.isin(cls, np.asarray([2, 6, 7, 18], dtype=np.uint8))
        if not np.any(eligible) and not planes:
            continue
        absolute_xy = points[:, :2] + origin[:2]
        _distance, neighbours = ground_index.query(
            absolute_xy, k=min(8, len(ground))
        )
        if np.ndim(neighbours) == 1:
            neighbours = np.asarray(neighbours)[:, None]
        ground_z = np.median(ground[np.asarray(neighbours), 2], axis=1)
        hag = points[:, 2] + origin[2] - ground_z
        if planes:
            from algorithms.spans.raw_stem_first import structural_plane_mask
            planar_junk = structural_plane_mask(points, hag, keep_stems, planes)
            eligible &= ~planar_junk
            quarantined_raw += int(np.count_nonzero((current >= 0) & planar_junk))
            current[planar_junk] = -1  # fine-cell backmapping must not reintroduce panels
        # A point can fall in two close tubes.  Deterministic nearest-axis
        # ownership prevents duplicates and preserves exact source rows.
        best_distance = np.full(len(points), np.inf, dtype=np.float64)
        best_tree = np.full(len(points), -1, dtype=np.int32)
        for tree_id, stem_x, stem_y, radius, max_hag in restore_specs:
            radial = np.hypot(points[:, 0] - stem_x, points[:, 1] - stem_y)
            candidate_rows = (
                eligible
                & (hag >= 0.05)
                & (hag <= max_hag)
                & (radial <= radius)
                & (radial < best_distance)
            )
            best_distance[candidate_rows] = radial[candidate_rows]
            best_tree[candidate_rows] = np.int32(tree_id)
        rows = best_tree >= 0
        if np.any(rows):
            current[rows] = best_tree[rows]
            count = int(np.count_nonzero(rows))
            restored += count
            for tree_id in np.unique(best_tree[rows]):
                tree_count = int(np.count_nonzero(best_tree[rows] == tree_id))
                key = str(int(tree_id))
                restored_by_tree[key] = restored_by_tree.get(key, 0) + tree_count
    del xyz, classification
    return result, {
        "mode": "VALIDATED_STEM_CYLINDER",
        "restored_raw_points": restored,
        "quarantined_raw_object_points": quarantined_raw,
        "validated_stem_tubes": len(restore_specs),
        "restored_by_tree": restored_by_tree,
        "excluded_las_classes": [2, 6, 7, 18],
        "minimum_hag_m": 0.05,
        "dbh_reference": "ORIGINAL_GROUND_PLUS_1.30_M",
    }


@dataclass(frozen=True, slots=True)
class SpanTreeOutcome:
    """Everything reconciliation and clearance need from one span's trees."""

    span: SpanRecord
    tree_records: tuple[dict[str, Any], ...]
    raw_original_index: np.ndarray  # int64, per assigned raw row
    raw_xyz_local_m: np.ndarray  # float64 (N, 3) scene-local
    raw_tree_id: np.ndarray  # int32, per assigned raw row (per-span ids)
    in_core: np.ndarray  # bool, per assigned raw row
    raw_local_maxima: int
    accepted_tree_tops: int
    assigned_raw_point_count: int
    artifact_root: Path
    tree_top_diagnostics: dict[str, Any] | None = None
    membership_diagnostics: dict[str, Any] | None = None
    fine_representative_count: int = 0
    aabb_query_points: int = 0
    oriented_region_points: int = 0
    source_tiles: tuple[str, ...] = ()
    source_total_tiles: int = 0
    source_intersecting_aabb_tiles: int = 0
    source_opened_tiles: int = 0
    source_points_read: int = 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": SPAN_TREE_RUNNER_SCHEMA,
            "span_name": self.span.span_name,
            "span_id": self.span.span_id,
            "line_id": self.span.line_id,
            "trees": [dict(record) for record in self.tree_records],
            "raw_local_maxima": self.raw_local_maxima,
            "accepted_tree_tops": self.accepted_tree_tops,
            "assigned_raw_point_count": self.assigned_raw_point_count,
            "tree_top_diagnostics": dict(self.tree_top_diagnostics or {}),
            "membership_diagnostics": dict(self.membership_diagnostics or {}),
            "fine_representative_count": self.fine_representative_count,
            "aabb_query_points": self.aabb_query_points,
            "oriented_region_points": self.oriented_region_points,
            "source_tiles": list(self.source_tiles),
            "source_total_tiles": self.source_total_tiles,
            "source_intersecting_aabb_tiles": self.source_intersecting_aabb_tiles,
            "source_opened_tiles": self.source_opened_tiles,
            "source_points_read": self.source_points_read,
        }


def _row_in_span_core(
    xyz_local_m: np.ndarray, span: SpanRecord, origin_xyz: tuple[float, float, float]
) -> np.ndarray:
    """Return ownership only for the exact oriented span core, never its AABB."""

    # Narrow legacy/test adapter: callers that provide only bounds cannot
    # express an oriented core, so retain their documented AABB conversion.
    if not hasattr(span, "start_xyz_absolute_m") or not hasattr(span, "end_xyz_absolute_m"):
        core_min, core_max = _span_bounds_in_scene_local_frame(
            span.core_bounds_min_xyz, span.core_bounds_max_xyz, origin_xyz
        )
        return np.all(xyz_local_m >= core_min, axis=1) & np.all(xyz_local_m <= core_max, axis=1)
    frame = SpanLocalFrame.from_endpoints(
        span.start_xyz_absolute_m, span.end_xyz_absolute_m, origin_xyz
    )
    core = SpanRegionSpec(
        role="span-core-ownership",
        half_width_m=float(span.half_width_m),
        longitudinal_before_m=0.0,
        longitudinal_after_m=0.0,
        z_min_m=float(span.core_bounds_min_xyz[2]),
        z_max_m=float(span.core_bounds_max_xyz[2]),
    )
    return oriented_span_mask_absolute(
        np.asarray(xyz_local_m, dtype=np.float64) + np.asarray(origin_xyz, dtype=np.float64),
        frame,
        core,
    )


def _span_bounds_in_scene_local_frame(
    bounds_min_xyz: tuple[float, float, float],
    bounds_max_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert authoritative absolute bounds to the RawRegion local frame."""

    origin = np.asarray(origin_xyz, dtype=np.float64)
    return (
        np.asarray(bounds_min_xyz, dtype=np.float64) - origin,
        np.asarray(bounds_max_xyz, dtype=np.float64) - origin,
    )


# Bounded full-resolution output assembly.  The frozen CROWN_STEM_HYBRID
# geometry runs on ~434k fine representatives (~120 MB); the only memory hotspot
# was the FULL-RESOLUTION membership output assembly (full float64 XYZ copies,
# a global int64 flatnonzero, two redundant 26.8M np.unique validations and the
# span-core temporaries).  These helpers assemble the same output with bounded
# memory: a two-pass chunked scan writing exact-size disk-backed memmaps.
_TREE_MEMBERS_CHUNK_ROWS = 500_000  # engineering-only bounded default, not science


def _validate_region_original_index_unique(
    region: Any, *, chunk_size: int = _TREE_MEMBERS_CHUNK_ROWS
) -> None:
    """Bounded duplicate validation of a RawRegion's source rows.

    Direct source-recovery regions are ascending and use the constant-memory
    fast path.  Role-tile queries are deterministic but tile-major, so their
    globally unique source rows are not necessarily ascending.  That fallback
    uses a packed source-index bitmap plus bounded per-chunk sorts; it does not
    allocate a full-region ``np.unique``/sort or change scientific row order.
    """

    if region.count == 0:
        return
    original = region.original_index()
    previous = -1
    ascending = True
    maximum = -1
    for start in range(0, region.count, chunk_size):
        stop = min(region.count, start + chunk_size)
        chunk = np.asarray(original[start:stop], dtype=np.int64)
        if np.any(chunk < 0):
            raise PoleEvidenceError("TREE_MEMBERS_SOURCE_INDEX_NOT_UNIQUE")
        maximum = max(maximum, int(np.max(chunk)))
        if np.any(chunk[1:] <= chunk[:-1]) or (
            start > 0 and chunk[0] <= previous
        ):
            ascending = False
        previous = int(chunk[-1])
    if ascending:
        return

    seen = np.zeros((maximum // 8) + 1, dtype=np.uint8)
    for start in range(0, region.count, chunk_size):
        stop = min(region.count, start + chunk_size)
        chunk = np.asarray(original[start:stop], dtype=np.int64)
        ordered = chunk.copy()
        ordered.sort()
        if np.any(ordered[1:] == ordered[:-1]):
            raise PoleEvidenceError("TREE_MEMBERS_SOURCE_INDEX_NOT_UNIQUE")
        byte = chunk >> 3
        bit = np.left_shift(np.uint8(1), (chunk & 7).astype(np.uint8))
        if np.any(seen[byte] & bit):
            raise PoleEvidenceError("TREE_MEMBERS_SOURCE_INDEX_NOT_UNIQUE")
        np.bitwise_or.at(seen, byte, bit)


def _assemble_tree_members_store(
    region: Any,
    raw_tree: np.ndarray,
    span: SpanRecord,
    origin_xyz: tuple[float, float, float],
    output_root: Any,
    *,
    chunk_size: int = _TREE_MEMBERS_CHUNK_ROWS,
) -> tuple[np.memmap, np.memmap, np.memmap, np.memmap, int, Path]:
    """Two-pass disk-backed full-resolution tree membership assembly.

    PASS 1 counts assigned rows (``tree_id >= 0``) in bounded chunks.
    ALLOCATE exact-size disk-backed memmaps (float32 XYZ, int64 source rows,
    int32 tree id, bool in-core).
    PASS 2 recomputes the same chunk masks and fills the memmaps, advancing an
    output cursor; the final cursor must equal PASS 1's count.

    The span-core (``in_core``) flag is a per-row COLUMN, not a filter — exactly
    matching the legacy contract (assigned core AND halo rows are both kept).
    The chunked span-core uses the SAME canonical span math as ``_row_in_span_core``
    on each bounded chunk, so it is bit-identical to the legacy full-array mask.
    """

    _validate_region_original_index_unique(region, chunk_size=chunk_size)
    raw_tree = np.asarray(raw_tree, dtype=np.int32)
    if raw_tree.shape != (region.count,):
        raise PoleEvidenceError("TREE_MEMBERS_RAW_TREE_ALIGNMENT")

    # PASS 1 — count assigned rows (tree members) only.
    final_count = 0
    for start in range(0, region.count, chunk_size):
        stop = min(region.count, start + chunk_size)
        final_count += int(np.count_nonzero(raw_tree[start:stop] >= 0))

    store_root = Path(output_root) / "tree_members_store"
    store_root.mkdir(parents=True, exist_ok=True)
    xyz_path = store_root / "xyz.npy"
    original_path = store_root / "original_index.npy"
    tree_path = store_root / "tree_id.npy"
    core_path = store_root / "in_core.npy"
    xyz_out = np.lib.format.open_memmap(xyz_path, mode="w+", dtype="<f4", shape=(final_count, 3))
    original_out = np.lib.format.open_memmap(original_path, mode="w+", dtype="<i8", shape=(final_count,))
    tree_out = np.lib.format.open_memmap(tree_path, mode="w+", dtype="<i4", shape=(final_count,))
    core_out = np.lib.format.open_memmap(core_path, mode="w+", dtype=bool, shape=(final_count,))

    # PASS 2 — fill the memmaps chunk by chunk (bounded temporaries).
    xyz_mem = region.xyz()
    original_mem = region.original_index()
    cursor = 0
    for start in range(0, region.count, chunk_size):
        stop = min(region.count, start + chunk_size)
        xyz_chunk = np.asarray(xyz_mem[start:stop], dtype=np.float32)
        tree_chunk = raw_tree[start:stop]
        assigned = tree_chunk >= 0
        count = int(np.count_nonzero(assigned))
        if count == 0:
            continue
        # Same canonical span-core math as the legacy full-array mask, per chunk.
        core_mask = _row_in_span_core(xyz_chunk, span, origin_xyz)
        end = cursor + count
        xyz_out[cursor:end] = xyz_chunk[assigned]
        original_out[cursor:end] = np.asarray(original_mem[start:stop], dtype=np.int64)[assigned]
        tree_out[cursor:end] = tree_chunk[assigned]
        core_out[cursor:end] = core_mask[assigned]
        cursor = end
        del xyz_chunk, tree_chunk, assigned, core_mask
    if cursor != final_count:
        raise PoleEvidenceError("TREE_MEMBERS_COUNT_MISMATCH")
    xyz_out.flush()
    original_out.flush()
    tree_out.flush()
    core_out.flush()
    _write_json(
        store_root / "manifest.json",
        {
            "schema": "tree-members-store-v1",
            "count": int(final_count),
            "xyz_dtype": "<f4",
            "original_index_dtype": "<i8",
            "tree_id_dtype": "<i4",
            "in_core_dtype": "bool",
            "chunk_rows": int(chunk_size),
        },
    )
    return xyz_out, original_out, tree_out, core_out, int(final_count), store_root


def _run_span_tree_pipeline_legacy(
    span: SpanRecord,
    vegetation_region: Any,
    coarse_ground_xyz_absolute: np.ndarray,
    config: FastBaselineConfig,
    *,
    output_dir: Any,
    origin_xyz: tuple[float, float, float],
    vegetation_query: SpanRegionQueryResult | None = None,
    publish_full_resolution: bool = True,
) -> SpanTreeOutcome:
    """Run the span-local tree pipeline and persist per-span artifacts.

    *vegetation_region* is the scene-wide recovered vegetation ``RawRegion``
    (memmap).  *origin_xyz* is the shared scene-local origin.
    """

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if vegetation_query is None:
        frame = SpanLocalFrame.from_endpoints(
            span.start_xyz_absolute_m, span.end_xyz_absolute_m, origin_xyz
        )
        vegetation_query = subset_raw_region_oriented(
            vegetation_region,
            frame,
            SpanRegionSpec(
                role="vegetation-search",
                half_width_m=float(span.half_width_m),
                longitudinal_before_m=float(span.halo_m),
                longitudinal_after_m=float(span.halo_m),
                z_min_m=float(span.halo_bounds_min_xyz[2]),
                z_max_m=float(span.halo_bounds_max_xyz[2]),
            ),
            output_root / "recovered_region",
        )
    subset = vegetation_query.region
    if subset.count == 0:
        empty = SpanTreeOutcome(
            span=span,
            tree_records=(),
            raw_original_index=np.empty(0, dtype=np.int64),
            raw_xyz_local_m=np.empty((0, 3), dtype=np.float64),
            raw_tree_id=np.empty(0, dtype=np.int32),
            in_core=np.empty(0, dtype=bool),
            raw_local_maxima=0,
            accepted_tree_tops=0,
            assigned_raw_point_count=0,
            artifact_root=output_root,
            tree_top_diagnostics={},
            membership_diagnostics={},
            aabb_query_points=vegetation_query.aabb_query_points,
            oriented_region_points=vegetation_query.oriented_region_points,
            source_tiles=vegetation_query.tiles,
            source_total_tiles=vegetation_query.total_tiles,
            source_intersecting_aabb_tiles=vegetation_query.intersecting_aabb_tiles,
            source_opened_tiles=vegetation_query.opened_tiles,
            source_points_read=vegetation_query.points_read,
        )
        _write_json(
            output_root / "span_tree_manifest.json",
            {**empty.to_manifest(), "runtime_seconds": time.perf_counter() - started},
        )
        return empty
    fine_path = output_root / "fine_representatives.npz"
    fine_arrays = build_fine_representatives(
        subset, fine_path, voxel_size_m=config.vegetation_fine_voxel_size_m
    )
    fine_xyz = fine_arrays["xyz_local_m"].astype(np.float64, copy=False)
    fine_original = fine_arrays["original_index"].astype(np.int64, copy=False)
    origin = np.asarray(origin_xyz, dtype=np.float64)
    # ``estimate_fine_vegetation_hag`` expects scene-LOCAL coordinates and adds
    # the shared origin internally (same contract as the existing callers).
    fine_hag = estimate_fine_vegetation_hag(
        fine_xyz,
        tuple(float(value) for value in origin),
        coarse_ground_xyz_absolute,
    )
    chm = build_span_chm(fine_xyz, fine_hag, config)
    smoothed = smooth_span_chm(chm, config)
    tops = detect_span_tree_tops(chm, smoothed, config)
    crowns = segment_span_crowns(chm, smoothed, tops, config)
    membership_diagnostics: dict[str, Any] = {}
    tree_id_by_fine = assign_span_tree_membership(
        fine_xyz, fine_hag, chm, smoothed, crowns, config, membership_diagnostics
    )
    if not publish_full_resolution:
        # The promoted CROWN_STEM backend consumes only the exact fine
        # candidate artifacts below.  Restoring and serializing the legacy raw
        # membership here would immediately be replaced by hybrid membership,
        # doubling full-resolution reconstruction and NPZ publication.
        _write_npz(
            output_root / "tree_fine.npz",
            fine_xyz_local_m=fine_arrays["xyz_local_m"],
            fine_original_index=fine_original,
            fine_hag_m=fine_hag,
            fine_tree_id=tree_id_by_fine,
        )
        _write_npz(
            output_root / "chm_review.npz",
            chm_height_m=chm.height_m,
            chm_smoothed_m=smoothed,
            chm_occupied=chm.occupied,
            chm_origin_xy=np.asarray(chm.origin_xy, dtype=np.float64),
            chm_cell_size_m=np.asarray(chm.cell_size_m, dtype=np.float64),
            crown_label=crowns.region_label_by_cell,
            marker_seed_mask=crowns.marker_seed_mask,
            accepted_tops_xy=np.asarray(
                [list(top.xy_local_m) for top in tops.accepted], dtype=np.float64
            )
            if tops.accepted
            else np.empty((0, 2), dtype=np.float64),
        )
        outcome = SpanTreeOutcome(
            span=span,
            tree_records=(),
            raw_original_index=np.empty(0, dtype=np.int64),
            raw_xyz_local_m=np.empty((0, 3), dtype=np.float32),
            raw_tree_id=np.empty(0, dtype=np.int32),
            in_core=np.empty(0, dtype=bool),
            raw_local_maxima=tops.raw_local_maxima_count,
            accepted_tree_tops=tops.accepted_count,
            assigned_raw_point_count=0,
            artifact_root=output_root,
            tree_top_diagnostics=dict(tops.diagnostics),
            membership_diagnostics=membership_diagnostics,
            fine_representative_count=len(fine_xyz),
            aabb_query_points=vegetation_query.aabb_query_points,
            oriented_region_points=vegetation_query.oriented_region_points,
            source_tiles=vegetation_query.tiles,
            source_total_tiles=vegetation_query.total_tiles,
            source_intersecting_aabb_tiles=vegetation_query.intersecting_aabb_tiles,
            source_opened_tiles=vegetation_query.opened_tiles,
            source_points_read=vegetation_query.points_read,
        )
        manifest = outcome.to_manifest()
        manifest["runtime_seconds"] = time.perf_counter() - started
        manifest["full_resolution_publication"] = "DEFERRED_TO_CROWN_STEM_HYBRID"
        _write_json(output_root / "span_tree_manifest.json", manifest)
        return outcome
    result = build_span_tree_result(
        subset,
        fine_path,
        fine_original,
        tree_id_by_fine,
        tops,
        crowns,
        chm,
        config,
    )
    raw_tree = np.asarray(result["raw_tree_id"], dtype=np.int32)
    # Bounded two-pass full-resolution output assembly (disk-backed memmaps).
    # Replaces the full float64 XYZ copy + global flatnonzero + full span-core
    # mask that previously peaked at ~3 GB RSS and OOM'd on the 16 GB machine.
    xyz_out, original_out, tree_out, core_out, member_count, _store_root = (
        _assemble_tree_members_store(subset, raw_tree, span, origin_xyz, output_root)
    )

    # Persist full-resolution tree members (scene-local XYZ + source rows).
    _write_npz(
        output_root / "tree_members.npz",
        original_index=original_out,
        xyz_local_m=xyz_out,
        tree_id=tree_out,
        in_core=core_out,
    )
    _write_npz(
        output_root / "tree_fine.npz",
        fine_xyz_local_m=fine_arrays["xyz_local_m"],
        fine_original_index=fine_original,
        fine_hag_m=fine_hag,
        fine_tree_id=tree_id_by_fine,
    )
    # CHM review arrays.
    _write_npz(
        output_root / "chm_review.npz",
        chm_height_m=chm.height_m,
        chm_smoothed_m=smoothed,
        chm_occupied=chm.occupied,
        chm_origin_xy=np.asarray(chm.origin_xy, dtype=np.float64),
        chm_cell_size_m=np.asarray(chm.cell_size_m, dtype=np.float64),
        crown_label=crowns.region_label_by_cell,
        marker_seed_mask=crowns.marker_seed_mask,
        accepted_tops_xy=np.asarray(
            [list(top.xy_local_m) for top in tops.accepted], dtype=np.float64
        )
        if tops.accepted
        else np.empty((0, 2), dtype=np.float64),
    )
    outcome = SpanTreeOutcome(
        span=span,
        tree_records=tuple(result["trees"]),
        raw_original_index=original_out,
        raw_xyz_local_m=xyz_out,
        raw_tree_id=tree_out,
        in_core=core_out,
        raw_local_maxima=tops.raw_local_maxima_count,
        accepted_tree_tops=tops.accepted_count,
        assigned_raw_point_count=int(member_count),
        artifact_root=output_root,
        tree_top_diagnostics=dict(tops.diagnostics),
        membership_diagnostics=membership_diagnostics,
        fine_representative_count=len(fine_xyz),
        aabb_query_points=vegetation_query.aabb_query_points,
        oriented_region_points=vegetation_query.oriented_region_points,
        source_tiles=vegetation_query.tiles,
        source_total_tiles=vegetation_query.total_tiles,
        source_intersecting_aabb_tiles=vegetation_query.intersecting_aabb_tiles,
        source_opened_tiles=vegetation_query.opened_tiles,
        source_points_read=vegetation_query.points_read,
    )
    manifest = outcome.to_manifest()
    manifest["runtime_seconds"] = time.perf_counter() - started
    _write_json(output_root / "span_tree_manifest.json", manifest)
    return outcome


def run_span_tree_pipeline(
    span: SpanRecord,
    vegetation_region: Any,
    coarse_ground_xyz_absolute: np.ndarray,
    config: FastBaselineConfig,
    *,
    output_dir: Any,
    origin_xyz: tuple[float, float, float],
    vegetation_query: SpanRegionQueryResult | None = None,
) -> SpanTreeOutcome:
    """Run the configured canonical tree backend.

    The promoted route uses the legacy CHM/watershed result solely to generate
    crown candidates, then replaces membership with crown-stem refinement. The
    legacy artifacts remain in ``legacy_reference`` for audit/fallback.
    """

    if config.tree_backend == TREE_BACKEND_LEGACY_WATERSHED:
        return _run_span_tree_pipeline_legacy(
            span, vegetation_region, coarse_ground_xyz_absolute, config,
            output_dir=output_dir, origin_xyz=origin_xyz, vegetation_query=vegetation_query,
        )
    if config.tree_backend not in {TREE_BACKEND_CROWN_STEM_HYBRID, TREE_BACKEND_RAW_STEM_FIRST}:
        raise PoleEvidenceError("TREE_BACKEND_INVALID")
    total_started = time.perf_counter()
    root = Path(output_dir)
    if vegetation_query is None:
        frame = SpanLocalFrame.from_endpoints(
            span.start_xyz_absolute_m, span.end_xyz_absolute_m, origin_xyz
        )
        vegetation_query = subset_raw_region_oriented(
            vegetation_region,
            frame,
            SpanRegionSpec(
                role="vegetation-search",
                half_width_m=float(span.half_width_m),
                longitudinal_before_m=float(span.halo_m),
                longitudinal_after_m=float(span.halo_m),
                z_min_m=float(span.halo_bounds_min_xyz[2]),
                z_max_m=float(span.halo_bounds_max_xyz[2]),
            ),
            root / "recovered_region",
        )
    legacy_root = root / "legacy_reference"
    raw_stems, raw_stem_diagnostics = (), {}
    excluded_pole_axes = ()
    if config.tree_backend == TREE_BACKEND_RAW_STEM_FIRST:
        from algorithms.spans.raw_stem_first import detect_raw_stem_axes
        raw_stems, raw_stem_diagnostics = detect_raw_stem_axes(
            vegetation_query.region, coarse_ground_xyz_absolute, origin_xyz
        )
        # Operator-confirmed endpoints are poles, not tree seeds. Exclude only
        # a detected shaft matching that endpoint, never its broad surroundings.
        endpoints = np.asarray([span.start_xyz_absolute_m[:2], span.end_xyz_absolute_m[:2]]) - np.asarray(origin_xyz[:2])
        excluded_pole_axes = tuple(stem for stem in raw_stems
            if np.min(np.linalg.norm(endpoints - stem.xy_local_m, axis=1)) <= .60)
        excluded_ids = {stem.stem_seed_id for stem in excluded_pole_axes}
        raw_stems = tuple(replace(stem, stem_seed_id=i) for i, stem in enumerate(
            stem for stem in raw_stems if stem.stem_seed_id not in excluded_ids))
        raw_stem_diagnostics["confirmed_endpoint_shafts_excluded"] = len(excluded_pole_axes)
    candidate_started = time.perf_counter()
    legacy = _run_span_tree_pipeline_legacy(
        span, vegetation_region, coarse_ground_xyz_absolute, config,
        output_dir=legacy_root,
        origin_xyz=origin_xyz,
        vegetation_query=vegetation_query,
        publish_full_resolution=False,
    )
    candidate_runtime = time.perf_counter() - candidate_started
    from algorithms.vegetation.coarse_to_fine import map_region_rows_to_tree

    with np.load(legacy_root / "tree_fine.npz", allow_pickle=False) as stream:
        fine_xyz = np.asarray(stream["fine_xyz_local_m"], dtype=np.float64)
        fine_original = np.asarray(stream["fine_original_index"], dtype=np.int64)
        fine_hag = np.asarray(stream["fine_hag_m"], dtype=np.float32)
        baseline_id = np.asarray(stream["fine_tree_id"], dtype=np.int32)
    with np.load(legacy_root / "fine_representatives.npz", allow_pickle=False) as stream:
        raw_count = np.asarray(stream["raw_count"], dtype=np.uint32)
    segmentation_started = time.perf_counter()
    if config.tree_backend == TREE_BACKEND_RAW_STEM_FIRST:
        from algorithms.spans.raw_stem_first import segment_from_raw_stems
        with np.load(legacy_root / "fine_representatives.npz", allow_pickle=False) as stream:
            fine_class = np.asarray(stream["classification"], dtype=np.uint8)
        hybrid = segment_from_raw_stems(fine_xyz, fine_hag, raw_stems, classification=fine_class,
            excluded_pole_axes=excluded_pole_axes)
    else:
        hybrid = segment_stem_guided_hybrid(
            fine_xyz, fine_hag, baseline_id, raw_count=raw_count
        )
    segmentation_runtime = time.perf_counter() - segmentation_started
    reconstruction_started = time.perf_counter()
    raw_tree = map_region_rows_to_tree(
        vegetation_query.region,
        legacy_root / "fine_representatives.npz",
        fine_original,
        hybrid.fine_tree_id,
    )
    raw_tree, trunk_restore = _restore_validated_trunk_membership(
        vegetation_query.region,
        raw_tree,
        hybrid,
        coarse_ground_xyz_absolute,
        origin_xyz,
    )
    reconstruction_runtime = time.perf_counter() - reconstruction_started
    # Per-tree full-resolution raw point counts in one O(N) pass (never
    # O(tree_count x N) full-array rescans).
    geometry_started = time.perf_counter()
    assigned_tree_ids = raw_tree[raw_tree >= 0]
    per_tree_raw_count = (
        np.bincount(assigned_tree_ids, minlength=int(assigned_tree_ids.max()) + 1)
        if assigned_tree_ids.size
        else np.zeros(0, dtype=np.int64)
    )
    records = []
    for candidate in hybrid.candidates:
        if candidate.quality_state.value != "CONFIRMED_TREE":
            continue
        records.append({
            "tree_id": candidate.tree_id,
            "tree_top_id": candidate.source_crown_ids[0],
            "apex_xy_local_m": list(candidate.apex_xy_local_m),
            "apex_height_m": float(np.max(fine_hag[candidate.member_rows])),
            "quality_state": candidate.quality_state.value,
            "structural_evidence": dict(candidate.structural_metrics or {}),
            "quality_reasons": list(candidate.quality_reasons),
            "fine_representative_count": int(len(candidate.member_rows)),
            "raw_point_count": int(per_tree_raw_count[candidate.tree_id]),
        })
    del assigned_tree_ids, per_tree_raw_count
    geometry_runtime = time.perf_counter() - geometry_started
    # Bounded two-pass full-resolution output assembly (disk-backed memmaps).
    membership_started = time.perf_counter()
    xyz_out, original_out, tree_out, core_out, member_count, _store_root = (
        _assemble_tree_members_store(
            vegetation_query.region, raw_tree, span, origin_xyz, root
        )
    )
    del raw_tree
    membership_runtime = time.perf_counter() - membership_started
    root.mkdir(parents=True, exist_ok=True)
    publish_started = time.perf_counter()
    _write_npz(root / "tree_members.npz", original_index=original_out, xyz_local_m=xyz_out, tree_id=tree_out, in_core=core_out)
    _write_npz(root / "tree_fine.npz", fine_xyz_local_m=fine_xyz, fine_original_index=fine_original, fine_hag_m=fine_hag, fine_tree_id=hybrid.fine_tree_id)
    for filename in ("fine_representatives.npz", "chm_review.npz"):
        source = legacy_root / filename
        if source.is_file():
            shutil.copy2(source, root / filename)
    publish_runtime = time.perf_counter() - publish_started
    diagnostics = dict(hybrid.diagnostics)
    if config.tree_backend == TREE_BACKEND_RAW_STEM_FIRST:
        diagnostics["raw_stem_detection"] = raw_stem_diagnostics
        diagnostics["validated_raw_stems"] = [stem.to_dict() for stem in raw_stems]
        diagnostics["excluded_confirmed_pole_axes"] = [stem.to_dict() for stem in excluded_pole_axes]
    diagnostics["original_trunk_restoration"] = trunk_restore
    diagnostics["pipeline_runtime_seconds"] = {
        "candidate_creation": candidate_runtime,
        "crown_stem_segmentation": segmentation_runtime,
        "full_resolution_mapping": reconstruction_runtime,
        "tree_geometry": geometry_runtime,
        "membership_assembly": membership_runtime,
        "artifact_publication": publish_runtime,
        "total": time.perf_counter() - total_started,
    }
    outcome = SpanTreeOutcome(
        span=span, tree_records=tuple(records), raw_original_index=original_out,
        raw_xyz_local_m=xyz_out, raw_tree_id=tree_out, in_core=core_out,
        raw_local_maxima=legacy.raw_local_maxima, accepted_tree_tops=legacy.accepted_tree_tops,
        assigned_raw_point_count=int(member_count), artifact_root=root,
        tree_top_diagnostics={**(legacy.tree_top_diagnostics or {}), "tree_backend": config.tree_backend},
        membership_diagnostics=diagnostics,
        fine_representative_count=legacy.fine_representative_count,
        aabb_query_points=legacy.aabb_query_points, oriented_region_points=legacy.oriented_region_points,
        source_tiles=legacy.source_tiles, source_total_tiles=legacy.source_total_tiles,
        source_intersecting_aabb_tiles=legacy.source_intersecting_aabb_tiles,
        source_opened_tiles=legacy.source_opened_tiles, source_points_read=legacy.source_points_read,
    )
    _write_json(root / "span_tree_manifest.json", outcome.to_manifest())
    return outcome


def load_span_tree_outcome(span: SpanRecord, artifact_root: Any) -> SpanTreeOutcome:
    """Reload a persisted per-span tree outcome from its artifact directory."""

    root = Path(artifact_root)
    members_path = root / "tree_members.npz"
    manifest_path = root / "span_tree_manifest.json"
    if not members_path.is_file() or not manifest_path.is_file():
        raise PoleEvidenceError("SPAN_TREE_ARTIFACT_MISSING")
    with np.load(members_path, allow_pickle=False) as stream:
        original_index = np.asarray(stream["original_index"], dtype=np.int64)
        # tree_members.npz xyz is stored at the source float32 precision to keep
        # the bounded-output contract; never upcast the full array on reload.
        xyz_local_m = np.asarray(stream["xyz_local_m"], dtype=np.float32)
        tree_id = np.asarray(stream["tree_id"], dtype=np.int32)
        in_core = np.asarray(stream["in_core"], dtype=bool)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    return SpanTreeOutcome(
        span=span,
        tree_records=tuple(manifest["trees"]),
        raw_original_index=original_index,
        raw_xyz_local_m=xyz_local_m,
        raw_tree_id=tree_id,
        in_core=in_core,
        raw_local_maxima=int(manifest.get("raw_local_maxima", 0)),
        accepted_tree_tops=int(manifest.get("accepted_tree_tops", 0)),
        assigned_raw_point_count=int(len(original_index)),
        artifact_root=root,
        tree_top_diagnostics=dict(manifest.get("tree_top_diagnostics") or {}),
        membership_diagnostics=dict(manifest.get("membership_diagnostics") or {}),
        fine_representative_count=int(manifest.get("fine_representative_count", 0)),
        aabb_query_points=int(manifest.get("aabb_query_points", 0)),
        oriented_region_points=int(manifest.get("oriented_region_points", 0)),
        source_tiles=tuple(str(value) for value in manifest.get("source_tiles", ())),
        source_total_tiles=int(manifest.get("source_total_tiles", 0)),
        source_intersecting_aabb_tiles=int(manifest.get("source_intersecting_aabb_tiles", 0)),
        source_opened_tiles=int(manifest.get("source_opened_tiles", 0)),
        source_points_read=int(manifest.get("source_points_read", 0)),
    )


def run_span_stem_guided_hybrid_candidate(
    span: SpanRecord,
    vegetation_region: Any,
    coarse_ground_xyz_absolute: np.ndarray,
    config: FastBaselineConfig,
    *,
    output_dir: Any,
    origin_xyz: tuple[float, float, float],
    stem_config: StemGuidedHybridConfig | None = None,
    vegetation_query: SpanRegionQueryResult | None = None,
) -> dict[str, Any]:
    """Run the non-authoritative stem-guided candidate beside the baseline.

    The function deliberately does not call reconciliation or identity code.
    It writes a separate result directory and keeps the existing span route as
    the baseline/reference implementation.  Mapping back to raw rows uses the
    exact same ``original_index`` map as the canonical tree runner.
    """

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if vegetation_query is None:
        frame = SpanLocalFrame.from_endpoints(
            span.start_xyz_absolute_m, span.end_xyz_absolute_m, origin_xyz
        )
        vegetation_query = subset_raw_region_oriented(
            vegetation_region,
            frame,
            SpanRegionSpec(
                role="vegetation-search",
                half_width_m=float(span.half_width_m),
                longitudinal_before_m=float(span.halo_m),
                longitudinal_after_m=float(span.halo_m),
                z_min_m=float(span.halo_bounds_min_xyz[2]),
                z_max_m=float(span.halo_bounds_max_xyz[2]),
            ),
            output_root / "recovered_region",
        )
    subset = vegetation_query.region
    fine_path = output_root / "fine_representatives.npz"
    fine = build_fine_representatives(
        subset, fine_path, voxel_size_m=config.vegetation_fine_voxel_size_m
    )
    xyz = np.asarray(fine["xyz_local_m"], dtype=np.float64)
    original = np.asarray(fine["original_index"], dtype=np.int64)
    hag = estimate_fine_vegetation_hag(xyz, origin_xyz, coarse_ground_xyz_absolute)
    chm = build_span_chm(xyz, hag, config)
    smoothed = smooth_span_chm(chm, config)
    tops = detect_span_tree_tops(chm, smoothed, config)
    crowns = segment_span_crowns(chm, smoothed, tops, config)
    baseline_ids = assign_span_tree_membership(xyz, hag, chm, smoothed, crowns, config)
    hybrid = segment_stem_guided_hybrid(
        xyz, hag, baseline_ids, raw_count=np.asarray(fine["raw_count"]), config=stem_config
    )
    from algorithms.vegetation.coarse_to_fine import map_region_rows_to_tree

    raw_tree = map_region_rows_to_tree(subset, fine_path, original, hybrid.fine_tree_id)
    raw_tree, trunk_restore = _restore_validated_trunk_membership(
        subset,
        raw_tree,
        hybrid,
        coarse_ground_xyz_absolute,
        origin_xyz,
    )
    # Bounded two-pass assembly (same store as the canonical runner); the
    # candidate artifact omits the in_core column.
    xyz_out, original_out, tree_out, _core_out, member_count, _store = (
        _assemble_tree_members_store(subset, raw_tree, span, origin_xyz, output_root)
    )
    del raw_tree
    _write_npz(
        output_root / "stem_guided_members.npz",
        original_index=original_out,
        xyz_local_m=xyz_out,
        tree_id=tree_out,
    )
    _write_npz(
        output_root / "stem_guided_fine.npz",
        fine_xyz_local_m=xyz,
        fine_original_index=original,
        fine_hag_m=hag,
        baseline_tree_id=baseline_ids,
        hybrid_tree_id=hybrid.fine_tree_id,
    )
    payload = {
        "schema": "stem-guided-hybrid-span-runner-v1",
        "span_name": span.span_name,
        "candidate_only": True,
        "trees": [candidate.to_dict() for candidate in hybrid.candidates],
        "validated_stems": [stem.to_dict() for stem in hybrid.validated_stems],
        "diagnostics": {
            **hybrid.diagnostics,
            "original_trunk_restoration": trunk_restore,
        },
        "full_resolution_original_index_unique": True,
        "assigned_raw_point_count": int(member_count),
    }
    _write_json(output_root / "stem_guided_manifest.json", payload)
    return payload


def _write_npz(path: Any, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def _write_json(path: Any, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), "utf-8")
    temporary.replace(path)


__all__ = [
    "SPAN_TREE_RUNNER_SCHEMA",
    "SpanTreeOutcome",
    "run_span_tree_pipeline",
    "run_span_stem_guided_hybrid_candidate",
]
