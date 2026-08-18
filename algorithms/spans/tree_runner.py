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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from core.config.fast_baseline import (
    TREE_BACKEND_CROWN_STEM_HYBRID,
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


def _run_span_tree_pipeline_legacy(
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
    if config.tree_backend != TREE_BACKEND_CROWN_STEM_HYBRID:
        raise PoleEvidenceError("TREE_BACKEND_INVALID")
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
    legacy = _run_span_tree_pipeline_legacy(
        span, vegetation_region, coarse_ground_xyz_absolute, config,
        output_dir=legacy_root, origin_xyz=origin_xyz, vegetation_query=vegetation_query,
    )
    from algorithms.vegetation.coarse_to_fine import map_region_rows_to_tree

    with np.load(legacy_root / "tree_fine.npz", allow_pickle=False) as stream:
        fine_xyz = np.asarray(stream["fine_xyz_local_m"], dtype=np.float64)
        fine_original = np.asarray(stream["fine_original_index"], dtype=np.int64)
        fine_hag = np.asarray(stream["fine_hag_m"], dtype=np.float32)
        baseline_id = np.asarray(stream["fine_tree_id"], dtype=np.int32)
    with np.load(legacy_root / "fine_representatives.npz", allow_pickle=False) as stream:
        raw_count = np.asarray(stream["raw_count"], dtype=np.uint32)
    hybrid = segment_stem_guided_hybrid(
        fine_xyz, fine_hag, baseline_id, raw_count=raw_count
    )
    raw_tree = map_region_rows_to_tree(
        vegetation_query.region,
        legacy_root / "fine_representatives.npz",
        fine_original,
        hybrid.fine_tree_id,
    )
    raw_xyz = np.asarray(vegetation_query.region.xyz(), dtype=np.float64)
    raw_original_all = np.asarray(vegetation_query.region.original_index(), dtype=np.int64)
    assigned = np.flatnonzero(raw_tree >= 0)
    assigned_original = raw_original_all[assigned]
    if len(assigned_original) != len(np.unique(assigned_original)):
        raise PoleEvidenceError("CROWN_STEM_ORIGINAL_INDEX_DUPLICATE")
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
            "raw_point_count": int(np.count_nonzero(raw_tree == candidate.tree_id)),
        })
    root.mkdir(parents=True, exist_ok=True)
    _write_npz(root / "tree_members.npz", original_index=assigned_original, xyz_local_m=raw_xyz[assigned], tree_id=raw_tree[assigned], in_core=_row_in_span_core(raw_xyz, span, origin_xyz)[assigned])
    _write_npz(root / "tree_fine.npz", fine_xyz_local_m=fine_xyz, fine_original_index=fine_original, fine_hag_m=fine_hag, fine_tree_id=hybrid.fine_tree_id)
    for filename in ("fine_representatives.npz", "chm_review.npz"):
        source = legacy_root / filename
        if source.is_file():
            shutil.copy2(source, root / filename)
    outcome = SpanTreeOutcome(
        span=span, tree_records=tuple(records), raw_original_index=assigned_original,
        raw_xyz_local_m=raw_xyz[assigned], raw_tree_id=raw_tree[assigned],
        in_core=_row_in_span_core(raw_xyz, span, origin_xyz)[assigned],
        raw_local_maxima=legacy.raw_local_maxima, accepted_tree_tops=legacy.accepted_tree_tops,
        assigned_raw_point_count=int(len(assigned)), artifact_root=root,
        tree_top_diagnostics={**(legacy.tree_top_diagnostics or {}), "tree_backend": config.tree_backend},
        membership_diagnostics=dict(hybrid.diagnostics), fine_representative_count=legacy.fine_representative_count,
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
    assigned = np.flatnonzero(raw_tree >= 0)
    raw_original = np.asarray(subset.original_index(), dtype=np.int64)[assigned]
    if len(raw_original) != len(np.unique(raw_original)):
        raise PoleEvidenceError("STEM_GUIDED_ORIGINAL_INDEX_NOT_UNIQUE")
    _write_npz(
        output_root / "stem_guided_members.npz",
        original_index=raw_original,
        xyz_local_m=np.asarray(subset.xyz(), dtype=np.float64)[assigned],
        tree_id=raw_tree[assigned],
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
        "diagnostics": hybrid.diagnostics,
        "full_resolution_original_index_unique": True,
        "assigned_raw_point_count": int(len(raw_original)),
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
