"""Full-resolution V4 tree artifact for exact Tree-Wire clearance.

The 0.10 m representatives are the geometry V4 reasons over, but they are NOT
the authoritative final geometry for minimum clearance: a fine representative
may summarize many raw source points, and the nearest branch tip can be a raw
member that is not the representative.  This module rebuilds a valid
``V4TreeSegmentationResult`` whose points are the full-resolution raw tree
members recovered through the coarse-to-fine back-map, so the existing
point-to-segment Tree-Wire engine consumes the exact raw tree geometry.

No V4 segmentation reasoning is repeated and no V4 semantic is changed; the
full-resolution result reuses the fine result's segmentation records and only
replaces the point input with the back-mapped raw members.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts.v4_tree_segmentation import (
    V4TreeAssignmentState,
    V4TreeSegmentationResult,
    validate_v4_tree_segmentation_result,
)

from algorithms.vegetation.coarse_to_fine import (
    RawRegion,
    estimate_fine_vegetation_hag,
    load_coarse_ground_representatives,
)
from algorithms.vegetation.fast_research import FastGroundResult
from algorithms.vegetation.v4_tree_artifacts import publish_v4_tree_segmentation_artifact
from algorithms.vegetation.v4_tree_segmentation import (
    _member_digest,
    _ro_array,
    _with_result_digest,
)

_FULL_RESOLUTION_TREE_GEOMETRY_SOURCE = "FULL_RESOLUTION_RAW"


def build_full_resolution_v4_artifact(
    fine_result: V4TreeSegmentationResult,
    region: RawRegion,
    region_tree: np.ndarray,
    origin_xyz: tuple[float, float, float],
    ground: FastGroundResult,
    output_root: Path,
) -> tuple[V4TreeSegmentationResult, dict[str, Any]]:
    """Publish a full-resolution V4 artifact from the fine result + back-map.

    ``region`` holds the recovered raw vegetation-region rows; ``region_tree``
    is the per-row V4 tree id (``-1`` for rows outside an accepted tree).
    Returns the constructed result and a summary manifest.
    """
    region_tree = np.asarray(region_tree, dtype=np.int32)
    if region_tree.shape != (region.count,):
        raise ValueError("FULL_RES_CLEARANCE_REGION_TREE_MISMATCH")
    tree_count = len(fine_result.tree_records)
    if tree_count == 0:
        raise ValueError("FULL_RES_CLEARANCE_NO_TREES")

    # Full-resolution HAG uses the same nearest-ground-neighbour method as the
    # fine representatives (the reviewed ``filters.hag_nn`` semantics).
    assigned_rows = np.flatnonzero(region_tree >= 0)
    if not len(assigned_rows):
        raise ValueError("FULL_RES_CLEARANCE_NO_ASSIGNED_ROWS")
    coarse_ground = load_coarse_ground_representatives(ground)
    assigned_xyz = region.xyz()[assigned_rows]
    assigned_hag = estimate_fine_vegetation_hag(
        assigned_xyz, origin_xyz, coarse_ground
    )

    # Per-tree member rows sorted by original index (deterministic ordering).
    member_rows: list[np.ndarray] = []
    member_indexes: list[np.ndarray] = []
    for tree_id in range(tree_count):
        rows = np.flatnonzero(region_tree == tree_id)
        order = np.argsort(region.original_index()[rows], kind="stable")
        rows = rows[order]
        member_rows.append(rows)
        member_indexes.append(region.original_index()[rows])
    if any(not len(rows) for rows in member_rows):
        raise ValueError("FULL_RES_CLEARANCE_EMPTY_TREE")

    # Dense arrays are globally ordered by original index so the dense
    # ``original_index`` is strictly increasing.  The region rows are stored in
    # raw LAS order (ascending original index), so the assigned rows are already
    # ascending and ``dense_rows == assigned_rows``.
    all_rows = np.concatenate(member_rows)
    global_order = np.argsort(region.original_index()[all_rows], kind="stable")
    dense_rows = all_rows[global_order]
    dense_original = region.original_index()[dense_rows].astype("<i8", copy=False)
    dense_xyz = region.xyz()[dense_rows].astype("<f8", copy=False)
    dense_hag = assigned_hag.astype("<f4", copy=False)
    dense_tree = region_tree[dense_rows].astype("<i4", copy=False)

    ordered_members = np.concatenate(member_indexes).astype("<i8", copy=False)
    offsets = np.concatenate(([0], np.cumsum([len(rows) for rows in member_rows]))).astype(
        "<i8", copy=False
    )

    tree_records = _rebuild_tree_records(fine_result, region, member_rows, member_indexes)
    state_by_point = np.full(len(dense_original), int(V4TreeAssignmentState.ASSIGNED), "|u1")
    # The audit's total_points is the population the result reports; for the
    # full-resolution artifact that is the full-res member count.
    full_audit = replace(fine_result.audit, total_points=int(len(dense_original)))

    full = V4TreeSegmentationResult(
        schema_version=fine_result.schema_version,
        algorithm_version=fine_result.algorithm_version,
        source_identity=fine_result.source_identity,
        config=fine_result.config,
        handoff_digest=fine_result.handoff_digest,
        validated_view_digest=fine_result.validated_view_digest,
        coverage_report_digest=fine_result.coverage_report_digest,
        original_index=_ro_array(dense_original),
        xyz_local_m=_ro_array(dense_xyz),
        hag_m=_ro_array(dense_hag),
        assignment_state_by_point=_ro_array(state_by_point),
        tree_id_by_point=_ro_array(dense_tree),
        assignment_confidence_by_point=_ro_array(np.ones(len(dense_original), "<f4")),
        assignment_reason_flags_by_point=_ro_array(np.zeros(len(dense_original), "<u4")),
        tree_member_offsets=_ro_array(offsets),
        ordered_tree_member_original_index=_ro_array(ordered_members),
        apex_records=fine_result.apex_records,
        trunk_records=fine_result.trunk_records,
        trunk_apex_association_records=fine_result.trunk_apex_association_records,
        structural_path_records=fine_result.structural_path_records,
        rotating_profile_records=fine_result.rotating_profile_records,
        hypothesis_records=fine_result.hypothesis_records,
        slice_records=fine_result.slice_records,
        relation_records=fine_result.relation_records,
        tree_records=tuple(tree_records),
        crown_boundary_vertices=_ro_array(np.zeros((0, 2), "<f8")),
        crown_boundary_offsets=_ro_array(np.zeros(len(tree_records) + 1, "<i8")),
        audit=full_audit,
        result_digest="0" * 64,
    )
    full = _with_result_digest(full)
    validate_v4_tree_segmentation_result(full)
    reference = publish_v4_tree_segmentation_artifact(full, output_root)
    manifest = {
        "tree_geometry_source": _FULL_RESOLUTION_TREE_GEOMETRY_SOURCE,
        "full_resolution_tree_point_count": int(len(dense_original)),
        "tree_count": int(tree_count),
        "artifact_root": str(reference.root_path),
        "per_tree_full_resolution_members": [
            int(len(rows)) for rows in member_rows
        ],
    }
    return full, manifest


def _rebuild_tree_records(
    fine_result: V4TreeSegmentationResult,
    region: RawRegion,
    member_rows: list[np.ndarray],
    member_indexes: list[np.ndarray],
) -> list[Any]:
    """Re-derive member-dependent tree record fields for full-res members."""

    records: list[Any] = []
    for tree_id, fine_record in enumerate(fine_result.tree_records):
        rows = member_rows[tree_id]
        indexes = member_indexes[tree_id]
        member_count = int(len(rows))
        xyz = region.xyz()[rows].astype("<f8")
        xs = xyz[:, 0]
        ys = xyz[:, 1]
        zs = xyz[:, 2]
        geometry = replace(
            fine_record.geometry_summary,
            member_count=member_count,
            xy_centroid=(
                float(np.mean(xs)),
                float(np.mean(ys)),
                float(np.mean(zs)),
            ),
            xy_bounds_min=(float(np.min(xs)), float(np.min(ys)), float(np.min(zs))),
            xy_bounds_max=(float(np.max(xs)), float(np.max(ys)), float(np.max(zs))),
            crown_area_m2=float((np.max(xs) - np.min(xs)) * (np.max(ys) - np.min(ys))),
            equivalent_radius_m=math.sqrt(
                max(0.0, float((np.max(xs) - np.min(xs)) * (np.max(ys) - np.min(ys))))
                / math.pi
            ),
        )
        records.append(
            replace(
                fine_record,
                member_count=member_count,
                member_original_index_digest=_member_digest([int(v) for v in indexes]),
                geometry_summary=geometry,
            )
        )
    return records
