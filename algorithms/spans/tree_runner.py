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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

from algorithms.poles.evidence import PoleEvidenceError
from algorithms.poles.topology import SpanRecord
from algorithms.spans.region import (
    SpanLocalFrame,
    SpanRegionQueryResult,
    SpanRegionSpec,
    oriented_span_mask_absolute,
    subset_raw_region_oriented,
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
    raw_xyz_all = subset.xyz().astype(np.float64, copy=False)
    assigned_rows = np.flatnonzero(raw_tree >= 0)
    in_core = _row_in_span_core(raw_xyz_all, span, origin_xyz)
    assigned_original = np.asarray(subset.original_index(), dtype=np.int64)[assigned_rows]
    assigned_xyz = raw_xyz_all[assigned_rows]
    assigned_tree = raw_tree[assigned_rows]
    assigned_in_core = in_core[assigned_rows]

    # Persist full-resolution tree members (scene-local XYZ + source rows).
    _write_npz(
        output_root / "tree_members.npz",
        original_index=assigned_original,
        xyz_local_m=assigned_xyz,
        tree_id=assigned_tree,
        in_core=assigned_in_core,
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
        raw_original_index=assigned_original,
        raw_xyz_local_m=assigned_xyz,
        raw_tree_id=assigned_tree,
        in_core=assigned_in_core,
        raw_local_maxima=tops.raw_local_maxima_count,
        accepted_tree_tops=tops.accepted_count,
        assigned_raw_point_count=int(len(assigned_rows)),
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


def load_span_tree_outcome(span: SpanRecord, artifact_root: Any) -> SpanTreeOutcome:
    """Reload a persisted per-span tree outcome from its artifact directory."""

    root = Path(artifact_root)
    members_path = root / "tree_members.npz"
    manifest_path = root / "span_tree_manifest.json"
    if not members_path.is_file() or not manifest_path.is_file():
        raise PoleEvidenceError("SPAN_TREE_ARTIFACT_MISSING")
    with np.load(members_path, allow_pickle=False) as stream:
        original_index = np.asarray(stream["original_index"], dtype=np.int64)
        xyz_local_m = np.asarray(stream["xyz_local_m"], dtype=np.float64)
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
]
