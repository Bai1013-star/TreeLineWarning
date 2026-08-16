"""Production binding for the tracked per-Span FAST science chain.

The project orchestrator owns lifecycle and persistence.  This module owns only
the narrow conversion from project contracts to the existing tracked science
records.  It deliberately requires a completed source-shared preparation
manifest; it never rebuilds global RAW, ground/HAG, or tower caches.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

from algorithms.poles.topology import SpanRecord
from algorithms.spans.clearance import run_span_tree_wire_clearance
from algorithms.spans.project_adapter import (
    PROJECT_SPAN_CORE_HALF_WIDTH_M,
    PROJECT_SPAN_GEOMETRY_CONFIGURATION_DIGEST,
    PROJECT_SPAN_LONGITUDINAL_HALO_M,
    RISK_THRESHOLD_M,
    ScienceAdapterError,
    SourceSharedPreparation,
    SpanAnalysisInput,
    SpanScienceResult,
)
from algorithms.spans.reconcile import reconcile_trees, reconciliation_to_dict
from algorithms.spans.region import SpanLocalFrame, SpanRegionSpec, subset_raw_tile_cache_oriented
from algorithms.spans.role_tiles import (
    ROLE_TILE_CELL_SIZE_M,
    RoleSpecificTileCache,
    RoleTileCacheError,
)
from algorithms.spans.tree_runner import run_span_tree_pipeline
from algorithms.spans.wire import run_span_wire_pipeline

_DISPLAY_ID_RE = re.compile(r"^[A-Z](\d+)$")


def _project_root_for_output(output_root: Path) -> Path:
    for candidate in (output_root, *output_root.parents):
        if (candidate / "project.json").is_file():
            return candidate
    raise ScienceAdapterError("PROJECT_ROOT_NOT_FOUND")


def _read_preparation(
    project_root: Path,
    preparation: SourceSharedPreparation,
    input_data: SpanAnalysisInput,
) -> dict[str, Any]:
    path = project_root / Path(preparation.manifest_reference)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ScienceAdapterError("SOURCE_SHARED_PREPARATION_INVALID") from exc
    if not isinstance(payload, dict):
        raise ScienceAdapterError("SOURCE_SHARED_PREPARATION_INVALID")
    if payload.get("source_binding_digest") != preparation.source_binding_digest:
        raise ScienceAdapterError("SOURCE_SHARED_SOURCE_DIGEST_MISMATCH")
    if payload.get("configuration_digest") != input_data.science_configuration_digest:
        raise ScienceAdapterError("SOURCE_SHARED_CONFIGURATION_DIGEST_MISMATCH")
    if payload.get("authority") != "FULL_RESOLUTION_RAW":
        raise ScienceAdapterError("INVALID_SOURCE_AUTHORITY")
    if payload.get("tower_membership_used_for_vegetation") is not False:
        raise ScienceAdapterError("INVALID_SOURCE_AUTHORITY")
    if (
        input_data.source_point_count is not None
        and payload.get("source_point_count") is not None
        and payload.get("source_point_count") != input_data.source_point_count
    ):
        raise ScienceAdapterError("SOURCE_SHARED_SOURCE_COUNT_MISMATCH")
    return payload


def _required_reference(payload: dict[str, Any], *names: str) -> str:
    for name in names:
        value = payload.get(name)
        if isinstance(value, str) and value:
            return value
    raise ScienceAdapterError("SOURCE_SHARED_PREPARATION_NOT_READY")


def _resolve_reference(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    if not path.is_file():
        raise ScienceAdapterError("SOURCE_SHARED_ARTIFACT_MISSING")
    return path


def _resolve_directory_reference(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    if not path.is_dir():
        raise ScienceAdapterError("SOURCE_SPATIAL_INDEX_MISSING")
    return path


def _xyz(payload: dict[str, Any], key: str) -> tuple[float, float, float]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ScienceAdapterError(f"SPAN_GEOMETRY_INVALID:{key}")
    return tuple(float(item) for item in value)


def _span_record(input_data: SpanAnalysisInput, config: FastBaselineConfig) -> SpanRecord:
    geometry = dict(input_data.span_geometry)
    start_endpoint = dict(input_data.from_endpoint_geometry)
    end_endpoint = dict(input_data.to_endpoint_geometry)
    start = _xyz(start_endpoint, "base_xyz_absolute_m")
    end = _xyz(end_endpoint, "base_xyz_absolute_m")
    start_top = _xyz(start_endpoint, "top_xyz_absolute_m")
    end_top = _xyz(end_endpoint, "top_xyz_absolute_m")
    direction = geometry.get("direction_xy")
    if not isinstance(direction, (list, tuple)) or len(direction) != 2:
        raise ScienceAdapterError("SPAN_GEOMETRY_INVALID:direction_xy")
    core_min = _xyz(geometry, "core_bounds_min_xyz")
    core_max = _xyz(geometry, "core_bounds_max_xyz")
    halo_min = _xyz(geometry, "halo_bounds_min_xyz")
    halo_max = _xyz(geometry, "halo_bounds_max_xyz")

    def numeric_id(value: str, prefix: str) -> int:
        match = _DISPLAY_ID_RE.fullmatch(value)
        if match is None or not value.startswith(prefix):
            raise ScienceAdapterError(f"DISPLAY_ID_INVALID:{value}")
        return int(match.group(1))

    return SpanRecord(
        line_id=numeric_id(input_data.group_id, "G"),
        line_name=input_data.group_id,
        span_id=numeric_id(input_data.span_id, "S") - 1,
        span_name=input_data.span_id,
        start_pole_candidate_id=-1,
        end_pole_candidate_id=-1,
        start_xyz_absolute_m=start,
        end_xyz_absolute_m=end,
        start_pole_top_z_m=float(start_top[2]),
        end_pole_top_z_m=float(end_top[2]),
        length_m=float(geometry["length_m"]),
        direction_xy=(float(direction[0]), float(direction[1]), 0.0),
        half_width_m=config.span_half_width_m,
        halo_m=config.span_halo_m,
        core_bounds_min_xyz=core_min,
        core_bounds_max_xyz=core_max,
        halo_bounds_min_xyz=halo_min,
        halo_bounds_max_xyz=halo_max,
        quality="PROJECT_CONFIRMED",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _tile_cache_from_preparation(
    project_root: Path,
    payload: dict[str, Any],
    input_data: SpanAnalysisInput,
    role: str,
    *,
    expected_source_index_exclusion: dict[str, Any] | None = None,
) -> Any | None:
    reference = payload.get(f"{role}_role_tile_index_reference")
    if not reference:
        return None
    root = _resolve_directory_reference(project_root, reference)
    try:
        config_digest = (
            FastBaselineConfig().computed_digest()
            if role == "wire"
            else PROJECT_SPAN_GEOMETRY_CONFIGURATION_DIGEST
        )
        return RoleSpecificTileCache.load(
            root,
            role=role,
            source_identity=input_data.source_identity,
            config_digest=config_digest,
            source_point_count=input_data.source_point_count,
            expected_source_index_exclusion=expected_source_index_exclusion,
            expected_cell_size_m=ROLE_TILE_CELL_SIZE_M,
        )
    except (RoleTileCacheError, OSError, ValueError) as exc:
        raise ScienceAdapterError("SOURCE_SPATIAL_INDEX_INVALID") from exc


def run_tracked_canonical_span_science(
    input_data: SpanAnalysisInput,
    preparation: SourceSharedPreparation,
    output_root: Path,
) -> SpanScienceResult:
    """Run one project Span through the existing tracked science modules."""

    total_started = time.perf_counter()
    project_root = _project_root_for_output(output_root)
    payload = _read_preparation(project_root, preparation, input_data)
    ground_reference = _required_reference(payload, "coarse_ground_reference")
    origin_value = payload.get("origin_xyz")
    if not isinstance(origin_value, (list, tuple)) or len(origin_value) != 3:
        raise ScienceAdapterError("SOURCE_SHARED_PREPARATION_NOT_READY")

    role_index_started = time.perf_counter()
    wire_tile_cache = _tile_cache_from_preparation(
        project_root, payload, input_data, "wire"
    )
    vegetation_tile_cache = _tile_cache_from_preparation(
        project_root,
        payload,
        input_data,
        "vegetation",
        expected_source_index_exclusion=(
            {
                "role": "wire",
                "source_index_digest": wire_tile_cache.metadata.get("source_index_digest"),
            }
            if wire_tile_cache is not None
            and wire_tile_cache.metadata.get("source_index_digest")
            else None
        ),
    )
    if wire_tile_cache is None or vegetation_tile_cache is None:
        raise ScienceAdapterError("SPAN_LOCAL_ROLE_TILE_INDEX_NOT_BOUND")
    role_index_runtime = time.perf_counter() - role_index_started
    ground_path = _resolve_reference(project_root, ground_reference)
    try:
        coarse_ground = np.asarray(np.load(ground_path, allow_pickle=False), dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ScienceAdapterError("SOURCE_SHARED_GROUND_REFERENCE_INVALID") from exc
    if coarse_ground.ndim != 2 or coarse_ground.shape[1] != 3:
        raise ScienceAdapterError("SOURCE_SHARED_GROUND_REFERENCE_INVALID")

    config = FastBaselineConfig()
    span = _span_record(input_data, config)
    frame = SpanLocalFrame.from_endpoints(
        span.start_xyz_absolute_m, span.end_xyz_absolute_m, tuple(origin_value)
    )
    wire_spec = SpanRegionSpec(
        role="wire",
        half_width_m=config.span_half_width_m,
        longitudinal_before_m=config.span_halo_m,
        longitudinal_after_m=config.span_halo_m,
        z_min_m=float(span.halo_bounds_min_xyz[2]),
        z_max_m=float(span.halo_bounds_max_xyz[2]),
    )
    vegetation_spec = SpanRegionSpec(
        role="vegetation-search",
        half_width_m=PROJECT_SPAN_CORE_HALF_WIDTH_M,
        longitudinal_before_m=PROJECT_SPAN_LONGITUDINAL_HALO_M,
        longitudinal_after_m=PROJECT_SPAN_LONGITUDINAL_HALO_M,
        z_min_m=float(span.halo_bounds_min_xyz[2]),
        z_max_m=float(span.halo_bounds_max_xyz[2]),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    wire_query_started = time.perf_counter()
    wire_query = subset_raw_tile_cache_oriented(
        wire_tile_cache, frame, wire_spec, output_root / "regions" / "wire"
    )
    wire_query_runtime = time.perf_counter() - wire_query_started
    vegetation_query_started = time.perf_counter()
    vegetation_query = subset_raw_tile_cache_oriented(
        vegetation_tile_cache,
        frame,
        vegetation_spec,
        output_root / "regions" / "vegetation",
    )
    vegetation_query_runtime = time.perf_counter() - vegetation_query_started
    vegetation_exclusion = vegetation_tile_cache.metadata.get("source_index_exclusion")
    if not isinstance(vegetation_exclusion, dict):
        raise ScienceAdapterError("SOURCE_SPATIAL_INDEX_EXCLUSION_NOT_VALIDATED")
    wire_vegetation_source_overlap_in_tree_input = 0
    wire_started = time.perf_counter()
    wire_result = run_span_wire_pipeline(
        span,
        wire_query.region,
        config,
        origin_xyz=tuple(float(item) for item in origin_value),
        output_dir=output_root / "wire",
    )
    wire_runtime = time.perf_counter() - wire_started
    tree_started = time.perf_counter()
    tree_outcome = run_span_tree_pipeline(
        span,
        None,
        coarse_ground,
        config,
        output_dir=output_root / "tree",
        origin_xyz=tuple(float(item) for item in origin_value),
        vegetation_query=vegetation_query,
    )
    reconciled = reconcile_trees(
        [tree_outcome], config, origin_xyz=tuple(float(item) for item in origin_value)
    )
    tree_runtime = time.perf_counter() - tree_started
    clearance_started = time.perf_counter()
    clearance = run_span_tree_wire_clearance(reconciled, [wire_result], config)
    clearance_runtime = time.perf_counter() - clearance_started
    _write_json(output_root / "reconciliation.json", reconciliation_to_dict(reconciled))
    _write_json(
        output_root / "clearance.json",
        {
            "schema": "treeline-span-clearance-result-v1",
            "status": clearance.status,
            "global_minimum_clearance_m": clearance.global_minimum_clearance_m,
            "records": [record.to_dict() for record in clearance.records],
            "diagnostics": dict(clearance.diagnostics),
        },
    )
    threshold = RISK_THRESHOLD_M
    below_threshold = sum(
        record.minimum_clearance_m is not None
        and record.minimum_clearance_m < threshold
        for record in clearance.records
    )
    risk_counts = {
        "trees_evaluated": len(clearance.records),
        "below_threshold": int(below_threshold),
        "not_below_threshold": len(clearance.records) - int(below_threshold),
        "no_distance_ready_edge": sum(
            record.status == "NO_DISTANCE_READY_EDGE" for record in clearance.records
        ),
    }
    return SpanScienceResult(
        tree_count=reconciled.final_tree_count,
        minimum_clearance_m=clearance.global_minimum_clearance_m,
        risk_threshold_m=threshold,
        risk_counts=risk_counts,
        artifact_files={
            "wire": str(output_root / "wire" / "span_wire_result.json"),
            "tree": str(output_root / "tree" / "span_tree_manifest.json"),
            "clearance": str(output_root / "clearance.json"),
        },
        diagnostics={
            "executor": "tracked.algorithms.spans",
            "timings_seconds": {
                "role_index_load": role_index_runtime,
                "wire_role_query": wire_query_runtime,
                "vegetation_role_query": vegetation_query_runtime,
                "role_query": wire_query_runtime + vegetation_query_runtime,
                "raw_restoration": wire_query_runtime + vegetation_query_runtime,
                "wire": wire_runtime,
                "tree": tree_runtime,
                "clearance": clearance_runtime,
                "total": time.perf_counter() - total_started,
            },
            "wire": dict(wire_result.diagnostics),
            "tree": dict(tree_outcome.to_manifest()),
            "clearance": dict(clearance.diagnostics),
            "source_spatial_query": {
                "backend": "ROLE_SPECIFIC_SOURCE_TILE_CACHE",
                "global_workset_used": False,
                "project_source_points": int(payload.get("source_point_count", 0)),
                "span_aabb_query_points": int(
                    wire_query.aabb_query_points + vegetation_query.aabb_query_points
                ),
                "wire_tile_telemetry": {
                    "total_tiles": int(wire_query.total_tiles),
                    "intersecting_aabb_tiles": int(wire_query.intersecting_aabb_tiles),
                    "opened_tiles": int(wire_query.opened_tiles),
                    "points_read": int(wire_query.points_read),
                    "points_after_mask": int(wire_query.oriented_region_points),
                },
                "vegetation_tile_telemetry": {
                    "total_tiles": int(vegetation_query.total_tiles),
                    "intersecting_aabb_tiles": int(vegetation_query.intersecting_aabb_tiles),
                    "opened_tiles": int(vegetation_query.opened_tiles),
                    "points_read": int(vegetation_query.points_read),
                    "points_after_mask": int(vegetation_query.oriented_region_points),
                },
                "wire_region_points": int(wire_query.oriented_region_points),
                "vegetation_search_region_points": int(
                    vegetation_query.oriented_region_points
                ),
                "vegetation_candidates": int(tree_outcome.fine_representative_count),
                "full_resolution_vegetation_points": int(
                    vegetation_query.oriented_region_points
                ),
                "tree_instances": int(reconciled.final_tree_count),
                "tree_associated_raw_points": int(tree_outcome.assigned_raw_point_count),
                "wire_vegetation_source_overlap_in_tree_input": (
                    wire_vegetation_source_overlap_in_tree_input
                ),
                "wire_vegetation_source_overlap_validation": (
                    "EXACT_SOURCE_INDEX_EXCLUSION_CONTRACT_VALIDATED"
                ),
            },
        },
    )


__all__ = ["run_tracked_canonical_span_science"]
