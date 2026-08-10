"""Source-native coarse-to-fine orchestration for the one-click LAS demo.

The module deliberately creates a new isolated output namespace for every LAS
input.  It never looks for historical W1, A2A, V1, V2, or V4 artifacts.

Architecture (owner-reviewed coarse-to-fine):

    FULL-RESOLUTION LAS
        -> coarse 0.25 m representatives  (routing / Ground/HAG only)
        -> broad vegetation region + broad wire region
        -> recover original raw source rows per region
        -> fine vegetation 0.10 m representatives -> V4
        -> fine wire region A1 -> A2A/A2B/A3/A4/W1 (source-native)
        -> full-resolution tree back-map
        -> exact tree-wire clearance (existing point-to-segment engine)

The coarse representative cloud is a routing representation only; it is never
the final geometric representation used for single-tree segmentation or
conductor reconstruction.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from core.contracts.v4_tree_segmentation import V4TreeAssignmentState

from algorithms.vegetation.coarse_to_fine import (
    COARSE_VOXEL_SIZE_M,
    VEGETATION_FINE_VOXEL_SIZE_M,
    WIRE_FINE_MODE,
    WIRE_FINE_VOXEL_SIZE_M,
    RawRegion,
    build_fine_vegetation_v4_input,
    compute_wire_region_bounds,
    load_raw_region,
    map_region_rows_to_tree,
    recover_wire_region_raw,
    run_source_native_wire_chain,
    write_wire_region_a1,
)
from algorithms.vegetation.fast_research import (
    FastResearchError,
    generate_fast_ground_preview,
    run_fast_ground_hag,
    run_fast_v4,
    sha256_file,
)
from algorithms.vegetation.fast_vegetation_ab import (
    FastVegetationEvidenceConfig,
    _pdal_covariance_features,
)
from algorithms.vegetation.fast_vegetation_v3 import (
    FastV3DiagnosticState,
    _grid_component_labels,
)
from algorithms.vegetation.fast_weighted_preprocess import (
    WeightedWorkingSet,
    build_weighted_fast_vegetation,
    build_weighted_working_set,
)
from algorithms.vegetation.v4_tree_artifacts import load_v4_tree_segmentation_artifact
from algorithms.vegetation.v4_tree_visualization import run_v4_tree_visualization, tree_color

SUPPORTED_SUFFIXES = {".las", ".laz"}
NON_CERTIFIED_MARKER = "NON_CERTIFIED_RESEARCH_DEMONSTRATION"
_Progress = Callable[[int, str, bool, float | None], None]
_W1_CONTINUATION_TOLERANCE_M = 0.30


@dataclass(frozen=True, slots=True)
class DemoRunResult:
    """Published, human-facing one-click run outcome."""

    run_root: Path
    summary_path: Path
    summary_html_path: Path
    overall_status: str
    tree_count: int
    tree_wire_status: str
    minimum_clearance_m: float | None


def discover_demo_inputs(input_dir: Path | str) -> list[Path]:
    """Return direct LAS/LAZ children in deterministic case-insensitive order."""

    root = Path(input_dir)
    if not root.exists():
        return []
    if not root.is_dir():
        raise NotADirectoryError(f"input_las is not a folder: {root}")
    files = [
        path
        for path in root.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files, key=lambda path: (path.name.casefold(), path.name))


def create_isolated_run_dir(
    runs_root: Path | str, input_path: Path | str, *, timestamp: datetime | None = None
) -> Path:
    """Allocate, but never overwrite, a stable per-run output directory."""

    root = Path(runs_root)
    stem = _safe_stem(Path(input_path).stem)
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    for suffix in range(1, 10_000):
        candidate_name = f"{stem}_{stamp}" if suffix == 1 else f"{stem}_{stamp}_{suffix:02d}"
        candidate = root / candidate_name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FastResearchError("NEW_LAS_DEMO_OUTPUT_NAMESPACE_EXHAUSTED")


def new_source_id(input_sha256: str) -> str:
    """Derive a content-bound source ID with no historical source alias."""

    if not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
        raise FastResearchError("NEW_LAS_DEMO_SOURCE_SHA_INVALID")
    return f"new_las_{input_sha256[:16]}"


def choose_representative_voxel_size(raw_point_count: int) -> float:
    """Choose a bounded FAST coarse-routing scale from the new source size.

    The reviewed 0.25 m setting remains the normal coarse routing scale.  Larger
    sources get a modestly coarser computation representation instead of
    silently assuming that every future LAS has the same density or memory
    budget.  Raw-row mapping and raw_count remain lossless at every scale.
    """

    if type(raw_point_count) is not int or raw_point_count <= 0:
        raise FastResearchError("NEW_LAS_DEMO_RAW_POINT_COUNT_INVALID")
    if raw_point_count <= 5_000_000:
        return COARSE_VOXEL_SIZE_M
    if raw_point_count <= 20_000_000:
        return 0.35
    if raw_point_count <= 50_000_000:
        return 0.5
    return 0.75


def run_new_las_demo(
    input_path: Path | str,
    run_root: Path | str,
    *,
    progress: _Progress | None = None,
    voxel_size_m: float | None = None,
    vegetation_fine_voxel_size_m: float | None = None,
    wire_fine_voxel_size_m: float | None = None,
) -> DemoRunResult:
    """Run the reviewed coarse-to-fine FAST route in an isolated namespace.

    The formal W1 producer is driven through a fresh A1/A2A/A2B/A3/A4 chain on
    the recovered source-native wire region.  If W1 cannot be generated
    reliably the route fails closed as ``BLOCKED_NO_VALID_W1`` and never
    fabricates clearance.
    """
    source = Path(input_path).resolve()
    root = Path(run_root).resolve()
    if not source.is_file() or source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise FastResearchError(f"NEW_LAS_DEMO_INPUT_INVALID: {source}")
    _ensure_empty_run_root(root)
    for stage in (
        "00_input",
        "01_preprocess",
        "02_ground",
        "03_wire",
        "04_vegetation",
        "05_v4",
        "06_clearance",
        "07_report",
    ):
        (root / stage).mkdir(parents=True, exist_ok=False)

    total_started = time.perf_counter()
    input_sha = sha256_file(source)
    source_id = new_source_id(input_sha)
    source_inspection = _inspect_source(source, input_sha, source_id)
    coarse_voxel_size_m = (
        choose_representative_voxel_size(int(source_inspection["point_count"]))
        if voxel_size_m is None
        else float(voxel_size_m)
    )
    if not np.isfinite(coarse_voxel_size_m) or coarse_voxel_size_m <= 0:
        raise FastResearchError("NEW_LAS_DEMO_VOXEL_SIZE_INVALID")
    fine_veg_voxel_size_m = (
        float(VEGETATION_FINE_VOXEL_SIZE_M)
        if vegetation_fine_voxel_size_m is None
        else float(vegetation_fine_voxel_size_m)
    )
    fine_wire_voxel_size_m = (
        float(WIRE_FINE_VOXEL_SIZE_M)
        if wire_fine_voxel_size_m is None
        else float(wire_fine_voxel_size_m)
    )
    if fine_veg_voxel_size_m <= 0 or fine_wire_voxel_size_m <= 0:
        raise FastResearchError("NEW_LAS_DEMO_FINE_VOXEL_SIZE_INVALID")
    source_inspection["coarse_voxel_size_m"] = coarse_voxel_size_m
    source_inspection["vegetation_fine_voxel_size_m"] = fine_veg_voxel_size_m
    source_inspection["wire_fine_voxel_size_m"] = fine_wire_voxel_size_m
    _write_json(root / "00_input" / "source_inspection.json", source_inspection)
    timings: dict[str, float] = {}
    _report(progress, 1, "正在读取点云...", False, None)
    timings["source_inspection"] = float(source_inspection["inspection_runtime_seconds"])
    _report(progress, 1, "读取点云完成", True, timings["source_inspection"])

    _report(progress, 2, "正在进行粗点云预处理...", False, None)
    started = time.perf_counter()
    _preprocess_status, working = build_weighted_working_set(
        source,
        root / "01_preprocess",
        voxel_size_m=coarse_voxel_size_m,
        artifact_stem="new_las",
    )
    generate_preprocessing_review(working, root / "01_preprocess" / "coarse_preprocess_review.html")
    timings["preprocess"] = time.perf_counter() - started
    _report(progress, 2, "粗点云预处理完成", True, timings["preprocess"])

    _report(progress, 3, "正在进行地面分离和高度计算...", False, None)
    started = time.perf_counter()
    _ground_status, ground = run_fast_ground_hag(
        Path(working.representative_las_path),
        root / "02_ground",
    )
    generate_fast_ground_preview(ground, root / "02_ground" / "ground_hag_review.html")
    timings["ground"] = time.perf_counter() - started
    _report(progress, 3, "地面分离和高度计算完成", True, timings["ground"])

    _report(progress, 4, "正在恢复导线区域并重建导线...", False, None)
    started = time.perf_counter()
    coarse_wire = run_source_native_wire_candidate_path(
        ground.output_las_path,
        root / "03_wire" / "coarse_candidate",
        origin_xyz=working.source_bounds_min_xyz,
    )
    wire = run_coarse_to_fine_wire_route(
        source,
        working,
        ground,
        root / "03_wire",
        source_id=source_id,
        coarse_wire_result=coarse_wire,
        voxel_size_m=fine_wire_voxel_size_m,
    )
    generate_wire_candidate_raw_review(wire, root / "03_wire" / "wire_candidate_raw_review.html")
    generate_w1_reconstruction_review(wire, root / "03_wire" / "w1_reconstruction_review.html")
    timings["wire"] = time.perf_counter() - started
    _report(progress, 4, "导线区域恢复与导线重建完成", True, timings["wire"])

    _report(progress, 5, "正在恢复植被区域并精细预处理...", False, None)
    started = time.perf_counter()
    veg_route = run_fine_vegetation_route(
        source,
        working,
        ground,
        root / "04_vegetation",
        source_id=source_id,
        coarse_wire_candidate_ids=coarse_wire["candidate_representative_ids"],
        wire=wire,
        voxel_size_m=fine_veg_voxel_size_m,
    )
    generate_vegetation_review(
        veg_route["coarse_manifest"],
        root / "04_vegetation" / "vegetation_review.html",
    )
    generate_fine_vegetation_input_review(
        veg_route["fine_vegetation"],
        root / "04_vegetation" / "fine_vegetation_input_review.html",
    )
    generate_raw_vs_fine_vegetation_overlay(
        veg_route["region"],
        veg_route["fine_vegetation"],
        root / "04_vegetation" / "raw_vs_fine_vegetation_overlay.html",
    )
    timings["vegetation"] = time.perf_counter() - started
    _report(progress, 5, "植被区域恢复与精细预处理完成", True, timings["vegetation"])

    _report(progress, 6, "正在进行单木分割与全分辨率回映射...", False, None)
    started = time.perf_counter()
    _v4_status, v4_root, _ = run_fast_v4(
        veg_route["fine_vegetation"],
        root / "05_v4",
        source_bounds_min_xyz=working.source_bounds_min_xyz,
        source_bounds_max_xyz=working.source_bounds_max_xyz,
    )
    loaded_v4 = load_v4_tree_segmentation_artifact(v4_root)
    run_v4_tree_visualization(
        v4_root,
        root / "05_v4" / "v4_tree_segmentation_review.html",
        root / "05_v4" / "v4_tree_segmentation_review.json",
        overwrite=False,
    )
    generate_v4_clean_review(loaded_v4.result, root / "05_v4" / "v4_tree_segmentation_clean.html")
    assigned_tree_ids = np.where(
        loaded_v4.result.assignment_state_by_point
        == int(V4TreeAssignmentState.ASSIGNED),
        loaded_v4.result.tree_id_by_point,
        -1,
    ).astype(np.int32)
    region = load_raw_region(veg_route["region_sidecar_path"])
    region_tree = map_region_rows_to_tree(
        region,
        veg_route["fine_reps_npz_path"],
        loaded_v4.result.original_index,
        assigned_tree_ids,
    )
    tree_summaries = summarize_fine_full_resolution_trees(
        region,
        region_tree,
        loaded_v4.result,
        working,
    )
    generate_v4_fullres_tree_preview(
        region,
        region_tree,
        root / "05_v4" / "v4_fullres_tree_preview.html",
    )
    _write_json(root / "05_v4" / "full_resolution_tree_summary.json", tree_summaries)
    timings["v4"] = time.perf_counter() - started
    _report(progress, 6, "单木分割与全分辨率回映射完成", True, timings["v4"])

    _report(progress, 7, "正在计算树线最小距离...", False, None)
    started = time.perf_counter()
    clearance = run_tree_wire_clearance_route(
        wire,
        v4_root,
        root / "06_clearance",
    )
    timings["clearance"] = time.perf_counter() - started
    _report(progress, 7, "树线最小距离与结果生成完成", True, timings["clearance"])

    quality = build_quality_flags(veg_route["coarse_manifest"], wire, tree_summaries)
    backmap = {
        "mapping_semantics": (
            "Each recovered raw vegetation-region row inherits the V4 tree id of "
            "its fine 0.10 m representative voxel."
        ),
        "assigned_full_resolution_points": tree_summaries["assigned_full_resolution_points"],
        "region_raw_point_count": region.count,
        "region_unassigned_points": (
            region.count - tree_summaries["assigned_full_resolution_points"]
        ),
        "fine_representative_count": veg_route["fine_manifest"]["counts"][
            "final_vegetation_representatives"
        ],
    }
    summary = _build_run_summary(
        source=source,
        input_sha=input_sha,
        source_id=source_id,
        coarse_voxel_size_m=coarse_voxel_size_m,
        fine_veg_voxel_size_m=fine_veg_voxel_size_m,
        fine_wire_voxel_size_m=fine_wire_voxel_size_m,
        working=working,
        ground=ground,
        wire=wire,
        veg_route=veg_route,
        v4_result=loaded_v4.result,
        tree_summaries=tree_summaries,
        clearance=clearance,
        timings={**timings, "total": time.perf_counter() - total_started},
        quality=quality,
        backmap=backmap,
        root=root,
    )
    report_dir = root / "07_report"
    summary_path = report_dir / "run_summary.json"
    summary_text_path = report_dir / "run_summary.txt"
    summary_html_path = report_dir / "RUN_SUMMARY.html"
    _write_json(summary_path, summary)
    _write_text(summary_text_path, render_summary_text(summary))
    write_summary_html(summary, root, summary_html_path)
    return DemoRunResult(
        run_root=root,
        summary_path=summary_path,
        summary_html_path=summary_html_path,
        overall_status=str(summary["RUN_STATUS"]),
        tree_count=int(summary["TREE_COUNT"]),
        tree_wire_status=str(summary["TREE_WIRE_STATUS"]),
        minimum_clearance_m=summary["GLOBAL_MINIMUM_CLEARANCE"],
    )


def generate_preprocessing_review(working: WeightedWorkingSet, output_html: Path) -> Path:
    """Write a bounded coarse 0.25 m routing representative review."""

    import plotly.graph_objects as go

    with np.load(working.representatives_npz_path, allow_pickle=False) as stream:
        xyz = np.asarray(stream["xyz_absolute"], dtype=np.float64)
        raw_count = np.asarray(stream["raw_count"], dtype=np.uint32)
        original = np.asarray(stream["original_index"], dtype=np.int64)
    rows = _even_rows(len(xyz), 60_000)
    figure = go.Figure(
        go.Scatter3d(
            x=xyz[rows, 0],
            y=xyz[rows, 1],
            z=xyz[rows, 2],
            mode="markers",
            marker={"size": 1.8, "color": raw_count[rows], "colorscale": "Viridis"},
            customdata=np.column_stack((original[rows], raw_count[rows])),
            hovertemplate=(
                "source_row=%{customdata[0]}<br>raw_count=%{customdata[1]}<extra></extra>"
            ),
            name="coarse routing representatives",
        )
    )
    figure.update_layout(
        title="粗点云预处理审查（0.25 m 粗路由代表点；raw_count 保留原始密度）",
        scene={"xaxis_title": "X (m)", "yaxis_title": "Y (m)", "zaxis_title": "Z (m)"},
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
    )
    return _write_figure(figure, output_html)


def run_source_native_wire_candidate_path(
    ground_las_path: Path | str,
    output_dir: Path | str,
    *,
    origin_xyz: tuple[float, float, float],
) -> dict[str, Any]:
    """Coarse routing step: detect broad line-like wire candidate regions.

    This is a coarse pre-V4 safety exclusion and routing handoff only; it never
    emits W1.  The recovered candidate representative IDs select the wire region
    whose raw source rows feed the real A1/A2A/A2B/A3/A4/W1 chain.
    """

    import plotly.graph_objects as go

    started = time.perf_counter()
    output = Path(output_dir)
    las = laspy.read(Path(ground_las_path))
    xyz_absolute = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)
    xyz_local = xyz_absolute - np.asarray(origin_xyz, dtype=np.float64)
    hag = np.asarray(las.HeightAboveGround, dtype=np.float32)
    classification = np.asarray(las.classification, dtype=np.uint8)
    representative_id = np.asarray(las.representative_id, dtype=np.uint32)
    original_index = np.asarray(las.source_original_index, dtype=np.int64)
    count = len(xyz_local)
    aligned = (hag, classification, representative_id, original_index)
    if any(values.shape != (count,) for values in aligned):
        raise FastResearchError("NEW_LAS_DEMO_WIRE_ALIGNMENT_INVALID")
    eligible = (classification != 2) & np.isfinite(hag) & (hag >= 3.0)
    linearity = np.zeros(count, dtype=np.float32)
    scattering = np.zeros(count, dtype=np.float32)
    feature_runtime = 0.0
    eligible_rows = np.flatnonzero(eligible)
    if len(eligible_rows) >= 16:
        features, feature_runtime = _pdal_covariance_features(
            xyz_local[eligible_rows],
            config=FastVegetationEvidenceConfig(),
        )
        linearity[eligible_rows] = features["linearity"]
        scattering[eligible_rows] = features["scattering"]
    seed = eligible & (linearity >= 0.76) & (scattering <= 0.04)
    labels, _cell, _origin = _grid_component_labels(xyz_local, seed, 1.0)
    candidate = np.zeros(count, dtype=bool)
    segments: list[dict[str, Any]] = []
    for component_id in np.unique(labels[labels > 0]):
        rows = np.flatnonzero(labels == component_id)
        points = xyz_local[rows]
        if len(rows) < 8:
            continue
        span = np.ptp(points, axis=0)
        xy_span = float(np.hypot(span[0], span[1]))
        z_span = float(span[2])
        if xy_span < 10.0 or z_span > max(3.0, 0.12 * xy_span):
            continue
        centered = points - points.mean(axis=0)
        _values, vectors = np.linalg.eigh(centered.T @ centered)
        direction = vectors[:, -1]
        if float(abs(direction[2])) > 0.35:
            continue
        projection = centered @ direction
        start = points.mean(axis=0) + direction * float(np.min(projection))
        stop = points.mean(axis=0) + direction * float(np.max(projection))
        candidate[rows] = True
        segments.append(
            {
                "candidate_id": int(component_id),
                "representative_count": int(len(rows)),
                "xy_span_m": xy_span,
                "z_span_m": z_span,
                "median_hag_m": float(np.median(hag[rows])),
                "start_xyz_local_m": [float(value) for value in start],
                "end_xyz_local_m": [float(value) for value in stop],
                "usage": "COARSE_WIRE_REGION_ROUTING_ONLY",
            }
        )
    candidate_ids = representative_id[candidate]
    _write_npz(
        output / "source_native_wire_candidates.npz",
        representative_id=representative_id,
        source_original_index=original_index,
        hag_m=hag,
        linearity=linearity,
        scattering=scattering,
        wire_geometry_candidate_mask=candidate,
    )
    manifest = {
        "schema_version": "new-las-fast-wire-candidate-v1",
        "non_certified_marker": NON_CERTIFIED_MARKER,
        "input_ground_las": str(Path(ground_las_path).resolve()),
        "wire_candidates": len(segments),
        "candidate_representatives": int(np.count_nonzero(candidate)),
        "candidate_segments": segments,
        "feature_runtime_seconds": feature_runtime,
        "runtime_seconds": time.perf_counter() - started,
        "w1_distance_ready_segments": 0,
        "tree_wire_status": "PENDING_COARSE_TO_FINE_CHAIN",
        "usage": (
            "Coarse line-like region routing handoff.  The candidate IDs select "
            "a wire region whose raw source rows feed the source-native "
            "A1/A2A/A2B/A3/A4/W1 chain; this manifest is not W1."
        ),
    }
    _write_json(output / "source_native_wire_manifest.json", manifest)
    review_rows = _even_rows(count, 60_000)
    candidate_rows = _even_rows(np.count_nonzero(candidate), 30_000)
    candidate_positions = np.flatnonzero(candidate)[candidate_rows]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=xyz_local[review_rows, 0],
            y=xyz_local[review_rows, 1],
            z=xyz_local[review_rows, 2],
            mode="markers",
            name="非地面代表点（粗路由样本）",
            marker={"size": 1.4, "color": "#9ca3af", "opacity": 0.34},
            hoverinfo="skip",
        )
    )
    if len(candidate_positions):
        figure.add_trace(
            go.Scatter3d(
                x=xyz_local[candidate_positions, 0],
                y=xyz_local[candidate_positions, 1],
                z=xyz_local[candidate_positions, 2],
                mode="markers",
                name="粗导线区域候选（仅路由）",
                marker={"size": 2.5, "color": "#dc2626", "opacity": 0.88},
            )
        )
    for segment in segments:
        endpoints = np.asarray(
            (segment["start_xyz_local_m"], segment["end_xyz_local_m"]), dtype=np.float64
        )
        figure.add_trace(
            go.Scatter3d(
                x=endpoints[:, 0],
                y=endpoints[:, 1],
                z=endpoints[:, 2],
                mode="lines",
                name=f"候选线段 {segment['candidate_id']}（路由）",
                line={"color": "#991b1b", "width": 5},
            )
        )
    figure.update_layout(
        title="粗导线候选路由审查（供精细导线重建的区域定位）",
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
    )
    _write_figure(figure, output / "wire_review.html")
    return {**manifest, "candidate_representative_ids": candidate_ids}


def run_coarse_to_fine_wire_route(
    source: Path,
    working: WeightedWorkingSet,
    ground: Any,
    output_dir: Path | str,
    *,
    source_id: str,
    coarse_wire_result: dict[str, Any],
    voxel_size_m: float,
) -> dict[str, Any]:
    """Recover the wire-region raw rows and run the source-native W1 chain."""
    output_dir = Path(output_dir)
    candidate_ids = coarse_wire_result.get("candidate_representative_ids")
    region_bounds = compute_wire_region_bounds(
        candidate_ids, ground, origin_xyz=working.source_bounds_min_xyz
    )
    base = {
        "TREE_WIRE_STATUS": "BLOCKED_NO_VALID_W1",
        "wire_candidates": int(coarse_wire_result.get("wire_candidates", 0)),
        "candidate_representative_ids": candidate_ids,
        "region_bounds": region_bounds,
        "wire_fine_mode": WIRE_FINE_MODE,
        "w1_distance_ready_segments": 0,
        "w1_polyline_count": 0,
        "stages": {},
    }
    if region_bounds is None:
        return {
            **base,
            "reason": "粗点云中未检测到可用的导线候选区域，禁止合成 W1。",
            "wire_region_raw_point_count": 0,
            "wire_fine_point_count": 0,
            "wire_region_sidecar_path": None,
        }
    region = recover_wire_region_raw(
        source,
        (
            tuple(float(value) for value in region_bounds["bounds_min_xyz"]),
            tuple(float(value) for value in region_bounds["bounds_max_xyz"]),
        ),
        output_dir / "raw_region",
        origin_xyz=working.source_bounds_min_xyz,
    )
    a1_dir = output_dir / "a1"
    a1_npz, a1_json, a1_stats = write_wire_region_a1(
        region,
        a1_dir,
        source_id=source_id,
        manifest_source_id=source_id,
        voxel_size_m=voxel_size_m,
    )
    chain = run_source_native_wire_chain(
        a1_npz,
        a1_json,
        output_dir / "wire_chain",
        source_id=source_id,
    )
    return {
        **base,
        **chain,
        "region_bounds": region_bounds,
        "wire_region_raw_point_count": int(region.count),
        "wire_fine_point_count": int(a1_stats["fine_representative_count"]),
        "wire_fine_voxel_size_m": float(voxel_size_m),
        "wire_region_sidecar_path": str(output_dir / "raw_region" / "region_sidecar.json"),
    }


def run_fine_vegetation_route(
    source: Path,
    working: WeightedWorkingSet,
    ground: Any,
    output_dir: Path | str,
    *,
    source_id: str,
    coarse_wire_candidate_ids: np.ndarray | None,
    wire: dict[str, Any],
    voxel_size_m: float,
) -> dict[str, Any]:
    """Coarse vegetation routing then raw recovery and fine V4 input."""
    output_dir = Path(output_dir)
    all_scene_roi = {
        "xy_bounds_min": [
            float(working.source_bounds_min_xyz[0]),
            float(working.source_bounds_min_xyz[1]),
        ],
        "xy_bounds_max": [
            float(working.source_bounds_max_xyz[0]),
            float(working.source_bounds_max_xyz[1]),
        ],
        "z_bounds_min": float(working.source_bounds_min_xyz[2]),
        "z_bounds_max": float(working.source_bounds_max_xyz[2]),
    }
    _coarse_vegetation, coarse_manifest = build_weighted_fast_vegetation(
        working,
        all_scene_roi,
        ground,
        output_dir / "coarse_routing",
        source_id=source_id,
        wire_geometry_representative_ids=coarse_wire_candidate_ids,
        artifact_stem="new_las",
    )
    with np.load(coarse_manifest["diagnostics_npz_path"], allow_pickle=False) as stream:
        diag_state = np.asarray(stream["diagnostic_state"], dtype=np.uint8)
        diag_ids = np.asarray(stream["representative_id"], dtype=np.uint32)
    retained_ids = diag_ids[diag_state == int(FastV3DiagnosticState.VEGETATION_RETAINED)]
    wire_members = wire.get("wire_member_original_index")
    continuation = wire.get("w1_continuation_xyz_local")
    if wire.get("TREE_WIRE_STATUS") != "VALID_W1":
        wire_members = None
        continuation = None
    fine_vegetation, fine_manifest = build_fine_vegetation_v4_input(
        source_las=source,
        working_set=working,
        ground=ground,
        retained_vegetation_representative_ids=retained_ids,
        origin_context=working.source_bounds_min_xyz,
        source_id=source_id,
        voxel_size_m=voxel_size_m,
        output_root=output_dir / "fine",
        wire_exclusion_original_index=wire_members,
        w1_continuation_xyz_local=continuation,
        w1_continuation_tolerance_m=_W1_CONTINUATION_TOLERANCE_M,
    )
    return {
        "fine_vegetation": fine_vegetation,
        "fine_manifest": fine_manifest,
        "coarse_manifest": coarse_manifest,
        "region_sidecar_path": fine_manifest["recovered_region_sidecar_path"],
        "fine_reps_npz_path": fine_manifest["fine_representatives_npz_path"],
        "region": load_raw_region(fine_manifest["recovered_region_sidecar_path"]),
        "retained_coarse_vegetation_representatives": int(len(retained_ids)),
    }


def generate_vegetation_review(manifest: dict[str, Any], output_html: Path) -> Path:
    """Render retained and excluded source-native FAST vegetation categories."""

    import plotly.graph_objects as go

    diagnostics = Path(manifest["diagnostics_npz_path"])
    with np.load(diagnostics, allow_pickle=False) as stream:
        xyz = np.asarray(stream["xyz_local_m"], dtype=np.float64)
        state = np.asarray(stream["diagnostic_state"], dtype=np.uint8)
        original = np.asarray(stream["original_index"], dtype=np.int64)
        raw_count = np.asarray(stream["raw_count"], dtype=np.uint32)
    labels = (
        (FastV3DiagnosticState.VEGETATION_RETAINED, "保留植被（粗路由）", "#228b22"),
        (FastV3DiagnosticState.WIRE_GEOMETRY_EXCLUDED, "粗导线候选排除（路由）", "#dc2626"),
        (FastV3DiagnosticState.RIGID_LINEAR_EXCLUDED, "刚性线状排除", "#f59e0b"),
        (FastV3DiagnosticState.RIGID_SURFACE_EXCLUDED, "刚性平面排除", "#9333ea"),
        (FastV3DiagnosticState.GROUND_EXCLUDED, "SMRF 地面排除", "#8b5e3c"),
        (FastV3DiagnosticState.NO_POSITIVE_VEGETATION_EVIDENCE, "缺少正植被证据", "#9ca3af"),
    )
    figure = go.Figure()
    for code, label, color in labels:
        rows = np.flatnonzero(state == int(code))
        shown = rows[_even_rows(len(rows), 60_000)]
        if not len(shown):
            continue
        figure.add_trace(
            go.Scatter3d(
                x=xyz[shown, 0],
                y=xyz[shown, 1],
                z=xyz[shown, 2],
                mode="markers",
                name=f"{label} ({len(rows):,})",
                marker={"size": 1.8, "color": color, "opacity": 0.72},
                customdata=np.column_stack((original[shown], raw_count[shown])),
                hovertemplate=(
                    "source_row=%{customdata[0]}<br>raw_count=%{customdata[1]}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="FAST 粗植被路由审查（非认证研究演示）",
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
        legend={"orientation": "h"},
    )
    return _write_figure(figure, output_html)


def generate_fine_vegetation_input_review(vegetation: Any, output_html: Path) -> Path:
    """Render the exact fine vegetation representatives actually fed into V4."""

    import plotly.graph_objects as go

    with np.load(vegetation.output_npz_path, allow_pickle=False) as stream:
        xyz = np.asarray(stream["xyz_local_m"], dtype=np.float64)
        hag = np.asarray(stream["hag_m"], dtype=np.float32)
        raw_count = np.asarray(stream["raw_count"], dtype=np.uint32)
        original = np.asarray(stream["original_index"], dtype=np.int64)
    rows = _even_rows(len(xyz), 80_000)
    figure = go.Figure(
        go.Scatter3d(
            x=xyz[rows, 0],
            y=xyz[rows, 1],
            z=xyz[rows, 2],
            mode="markers",
            marker={"size": 1.8, "color": hag[rows], "colorscale": "Viridis"},
            customdata=np.column_stack((original[rows], raw_count[rows], hag[rows])),
            hovertemplate=(
                "source_row=%{customdata[0]}<br>raw_count=%{customdata[1]}"
                "<br>HAG=%{customdata[2]:.2f}<extra></extra>"
            ),
            name="fine vegetation representatives",
        )
    )
    figure.update_layout(
        title="精细植被输入审查（实际送入 V4 的 0.10 m 代表点，颜色为 HAG）",
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
    )
    return _write_figure(figure, output_html)


def generate_raw_vs_fine_vegetation_overlay(
    region: RawRegion, vegetation: Any, output_html: Path
) -> Path:
    """Overlay raw recovered tree geometry and fine V4 input in one frame."""

    import plotly.graph_objects as go

    origin = np.asarray(region.origin_xyz, dtype=np.float64)
    raw_xyz = region.xyz().astype(np.float64, copy=False) + origin
    raw_rows = _even_rows(region.count, 60_000)
    with np.load(vegetation.output_npz_path, allow_pickle=False) as stream:
        fine_xyz = np.asarray(stream["xyz_local_m"], dtype=np.float64) + origin
        fine_count = np.asarray(stream["raw_count"], dtype=np.uint32)
    fine_rows = _even_rows(len(fine_xyz), 60_000)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=raw_xyz[raw_rows, 0],
            y=raw_xyz[raw_rows, 1],
            z=raw_xyz[raw_rows, 2],
            mode="markers",
            name="原始恢复植被区域点（样本）",
            marker={"size": 1.2, "color": "#9ca3af", "opacity": 0.22},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=fine_xyz[fine_rows, 0],
            y=fine_xyz[fine_rows, 1],
            z=fine_xyz[fine_rows, 2],
            mode="markers",
            name="0.10 m 精细植被代表点",
            marker={"size": 2.2, "color": fine_count[fine_rows], "colorscale": "Viridis"},
            hovertemplate="raw_count=%{marker.color}<extra></extra>",
        )
    )
    figure.update_layout(
        title="原始树几何与精细 V4 输入叠加（同一坐标框架）",
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
    )
    return _write_figure(figure, output_html)


def generate_wire_candidate_raw_review(wire: dict[str, Any], output_html: Path) -> Path:
    """Render the recovered raw wire-region points and the coarse region box."""

    import plotly.graph_objects as go

    figure = go.Figure()
    sidecar = wire.get("wire_region_sidecar_path")
    if sidecar:
        region = load_raw_region(sidecar)
        origin = np.asarray(region.origin_xyz, dtype=np.float64)
        raw_xyz = region.xyz().astype(np.float64, copy=False) + origin
        rows = _even_rows(region.count, 60_000)
        figure.add_trace(
            go.Scatter3d(
                x=raw_xyz[rows, 0],
                y=raw_xyz[rows, 1],
                z=raw_xyz[rows, 2],
                mode="markers",
                name=f"恢复的原始导线区域点（{region.count:,}，样本）",
                marker={"size": 1.4, "color": "#6b7280", "opacity": 0.30},
                hoverinfo="skip",
            )
        )
    bounds = wire.get("region_bounds")
    if bounds:
        minimum = np.asarray(bounds["bounds_min_xyz"], dtype=np.float64)
        maximum = np.asarray(bounds["bounds_max_xyz"], dtype=np.float64)
        corners = np.asarray(
            [
                [minimum[0], minimum[1], minimum[2]],
                [maximum[0], minimum[1], minimum[2]],
                [maximum[0], maximum[1], minimum[2]],
                [minimum[0], maximum[1], minimum[2]],
                [minimum[0], minimum[1], maximum[2]],
                [maximum[0], minimum[1], maximum[2]],
                [maximum[0], maximum[1], maximum[2]],
                [minimum[0], maximum[1], maximum[2]],
            ]
        )
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        for first, second in edges:
            figure.add_trace(
                go.Scatter3d(
                    x=corners[[first, second], 0],
                    y=corners[[first, second], 1],
                    z=corners[[first, second], 2],
                    mode="lines",
                    name="导线区域包围盒" if first == 0 and second == 1 else "",
                    line={"color": "#dc2626", "width": 3},
                    showlegend=first == 0 and second == 1,
                )
            )
    figure.update_layout(
        title="导线候选原始点恢复审查（粗路由区域 -> 原始源点）",
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
    )
    return _write_figure(figure, output_html)


def generate_w1_reconstruction_review(wire: dict[str, Any], output_html: Path) -> Path:
    """Overlay reconstructed W1 segments with source-native conductor points."""

    import plotly.graph_objects as go

    figure = go.Figure()
    if wire.get("TREE_WIRE_STATUS") != "VALID_W1":
        figure.update_layout(
            title=(
                f"W1 重建审查：{wire.get('reason') or '未生成有效 W1'}"
                "（fail-closed，未伪造导线模型）"
            ),
            margin={"l": 0, "r": 0, "t": 44, "b": 0},
        )
        return _write_figure(figure, output_html)

    from algorithms.wires.reconstruct import load_wire_models

    try:
        _result, arrays = load_wire_models(
            Path(wire["w1_npz_path"]), Path(wire["w1_json_path"])
        )
    except Exception as exc:  # noqa: BLE001 - review layer must never crash the run
        figure.update_layout(
            title=f"W1 重建审查：无法读取 W1 产物：{exc}",
            margin={"l": 0, "r": 0, "t": 44, "b": 0},
        )
        return _write_figure(figure, output_html)

    with np.load(wire["a4_npz_path"], allow_pickle=False) as stream:
        a4_xyz = np.asarray(stream["xyz_local"], dtype=np.float64)
        a4_wire = np.asarray(stream["wire_id_by_point"], dtype=np.int32)
    confirmed = a4_wire >= 0
    shown = np.flatnonzero(confirmed)[_even_rows(int(np.count_nonzero(confirmed)), 50_000)]
    figure.add_trace(
        go.Scatter3d(
            x=a4_xyz[shown, 0],
            y=a4_xyz[shown, 1],
            z=a4_xyz[shown, 2],
            mode="markers",
            name="source-native 导线点（A4 确认成员，样本）",
            marker={"size": 2.0, "color": "#94a3b8", "opacity": 0.55},
            hoverinfo="skip",
        )
    )
    ready = np.asarray(arrays["segment_distance_ready"], dtype=bool)
    offsets = np.asarray(arrays["sampled_curve_offsets"], dtype=np.int64)
    sampled = np.asarray(arrays["sampled_curve_xyz"], dtype=np.float64)
    palette = ["#dc2626", "#2563eb", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    wire_ids = np.asarray(arrays["segment_wire_id"], dtype=np.int32)
    for position, is_ready in enumerate(ready):
        if not bool(is_ready):
            continue
        start, stop = int(offsets[position]), int(offsets[position + 1])
        curve = sampled[start:stop]
        color = palette[int(wire_ids[position]) % len(palette)]
        figure.add_trace(
            go.Scatter3d(
                x=curve[:, 0],
                y=curve[:, 1],
                z=curve[:, 2],
                mode="lines+markers",
                name=f"W1 距离就绪线段 wire {int(wire_ids[position])}",
                line={"color": color, "width": 6},
                marker={"size": 1.4, "color": color},
            )
        )
    figure.update_layout(
        title="W1 重建审查：重建导线与 source-native 导线点叠加",
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
    )
    return _write_figure(figure, output_html)


def generate_v4_clean_review(result: Any, output_html: Path) -> Path:
    """Render assigned V4 tree memberships only, separate from the review layers."""

    import plotly.graph_objects as go

    figure = go.Figure()
    state = np.asarray(result.assignment_state_by_point, dtype=np.uint8)
    tree_ids = np.asarray(result.tree_id_by_point, dtype=np.int32)
    xyz = np.asarray(result.xyz_local_m, dtype=np.float64)
    for tree_id in sorted(int(value) for value in np.unique(tree_ids) if value >= 0):
        rows = np.flatnonzero(
            (tree_ids == tree_id) & (state == int(V4TreeAssignmentState.ASSIGNED))
        )
        if not len(rows):
            continue
        shown = rows[_even_rows(len(rows), 80_000)]
        figure.add_trace(
            go.Scatter3d(
                x=xyz[shown, 0],
                y=xyz[shown, 1],
                z=xyz[shown, 2],
                mode="markers",
                name=f"Tree {tree_id} ({len(rows):,})",
                marker={"size": 2.0, "color": tree_color(tree_id), "opacity": 0.86},
            )
        )
    figure.update_layout(
        title="V4 单木分割：仅显示已分配树点（0.10 m 精细代表点）",
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
    )
    return _write_figure(figure, output_html)


def generate_v4_fullres_tree_preview(
    region: RawRegion, region_tree: np.ndarray, output_html: Path
) -> Path:
    """Render V4 tree assignments mapped back onto original raw tree points."""

    import plotly.graph_objects as go

    origin = np.asarray(region.origin_xyz, dtype=np.float64)
    xyz = region.xyz().astype(np.float64, copy=False) + origin
    figure = go.Figure()
    assigned = region_tree >= 0
    unassigned_rows = np.flatnonzero(~assigned)
    shown_unassigned = unassigned_rows[_even_rows(len(unassigned_rows), 40_000)]
    if len(shown_unassigned):
        figure.add_trace(
            go.Scatter3d(
                x=xyz[shown_unassigned, 0],
                y=xyz[shown_unassigned, 1],
                z=xyz[shown_unassigned, 2],
                mode="markers",
                name="未分配原始点（样本）",
                marker={"size": 1.2, "color": "#9ca3af", "opacity": 0.20},
                hoverinfo="skip",
            )
        )
    for tree_id in sorted(int(value) for value in np.unique(region_tree[assigned]) if value >= 0):
        rows = np.flatnonzero(region_tree == tree_id)
        shown = rows[_even_rows(len(rows), 100_000)]
        figure.add_trace(
            go.Scatter3d(
                x=xyz[shown, 0],
                y=xyz[shown, 1],
                z=xyz[shown, 2],
                mode="markers",
                name=f"Tree {tree_id} ({len(rows):,})",
                marker={"size": 1.6, "color": tree_color(tree_id), "opacity": 0.88},
            )
        )
    figure.update_layout(
        title="V4 单木全分辨率回映射：树分配映射回原始源点",
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
    )
    return _write_figure(figure, output_html)


def summarize_fine_full_resolution_trees(
    region: RawRegion,
    region_tree: np.ndarray,
    v4_result: Any,
    working: WeightedWorkingSet,
) -> dict[str, Any]:
    """Compute exact raw XYZ extents and counts from fine back-mapped members."""

    assigned_state = np.asarray(v4_result.assignment_state_by_point, dtype=np.uint8)
    assigned = assigned_state == int(V4TreeAssignmentState.ASSIGNED)
    tree_by_point = np.asarray(v4_result.tree_id_by_point, dtype=np.int32)
    unique_tree_ids = sorted(
        int(value) for value in np.unique(tree_by_point[assigned]) if value >= 0
    )
    region_tree_array = np.asarray(region_tree, dtype=np.int32)
    if len(region_tree_array) != region.count:
        raise FastResearchError("NEW_LAS_DEMO_FULLRES_REGION_TREE_MISMATCH")
    tree_data: dict[int, dict[str, Any]] = {
        tree_id: {
            "tree_id": tree_id,
            "representative_count": int(
                np.count_nonzero(assigned & (tree_by_point == tree_id))
            ),
            "full_resolution_point_count": 0,
            "full_resolution_min_xyz": np.full(3, np.inf, dtype=np.float64),
            "full_resolution_max_xyz": np.full(3, -np.inf, dtype=np.float64),
        }
        for tree_id in unique_tree_ids
    }
    origin = np.asarray(working.source_bounds_min_xyz, dtype=np.float64)
    xyz = region.xyz()
    assigned_region_rows = 0
    for tree_id in unique_tree_ids:
        rows = np.flatnonzero(region_tree_array == tree_id)
        if not len(rows):
            raise FastResearchError("NEW_LAS_DEMO_FULLRES_TREE_MEMBERSHIP_EMPTY")
        selected = xyz[rows].astype(np.float64, copy=False) + origin
        entry = tree_data[tree_id]
        entry["full_resolution_point_count"] = int(len(rows))
        entry["full_resolution_min_xyz"] = selected.min(axis=0)
        entry["full_resolution_max_xyz"] = selected.max(axis=0)
        assigned_region_rows += len(rows)
    del xyz
    apex_by_id = {record.candidate_id: record for record in v4_result.apex_records}
    tree_record_by_id = {int(record.tree_id): record for record in v4_result.tree_records}
    hag = np.asarray(v4_result.hag_m, dtype=np.float32)
    rows_by_tree: dict[int, np.ndarray] = {}
    for tree_id in unique_tree_ids:
        rows_by_tree[tree_id] = np.flatnonzero(assigned & (tree_by_point == tree_id))
    for tree_id in unique_tree_ids:
        entry = tree_data[tree_id]
        record = tree_record_by_id[tree_id]
        apex = apex_by_id.get(record.primary_apex_candidate_id)
        minimum = entry.pop("full_resolution_min_xyz")
        maximum = entry.pop("full_resolution_max_xyz")
        if entry["full_resolution_point_count"] <= 0:
            raise FastResearchError("NEW_LAS_DEMO_FULLRES_TREE_MEMBERSHIP_EMPTY")
        tree_rows = rows_by_tree[tree_id]
        entry["apex_xyz_local_m"] = (
            None if apex is None else [float(value) for value in apex.xyz_local_m]
        )
        entry["apex_xyz_absolute_m"] = (
            None
            if apex is None
            else [float(value) for value in np.asarray(apex.xyz_local_m, dtype=np.float64) + origin]
        )
        hag_range = [float(np.min(hag[tree_rows])), float(np.max(hag[tree_rows]))]
        entry["HAG_RANGE_M"] = hag_range
        entry["HAG_RANGE_SEMANTICS"] = (
            "HAG is evaluated on the fine 0.10 m representatives V4 consumed; "
            "full-resolution XYZ membership is back-mapped from raw region rows."
        )
        entry["representative_hag_range_m"] = hag_range
        entry["full_resolution_xy_extent_m"] = [
            float(minimum[0]),
            float(maximum[0]),
            float(minimum[1]),
            float(maximum[1]),
        ]
        entry["full_resolution_z_extent_m"] = [float(minimum[2]), float(maximum[2])]
    return {
        "mapping_semantics": (
            "Raw vegetation-region rows inherit the V4 tree id of their fine "
            "0.10 m representative voxel; rows outside the region are unassigned."
        ),
        "trees": list(tree_data.values()),
        "assigned_full_resolution_points": assigned_region_rows,
        "region_raw_point_count": region.count,
    }


def run_tree_wire_clearance_route(
    wire: dict[str, Any],
    v4_root: Path,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Run the existing exact tree-wire engine when a valid source-native W1 exists."""
    from core.contracts.tree_wire_clearance import (
        TreeWireClearanceStatus,
        default_tree_wire_clearance_config,
    )

    from algorithms.vegetation.tree_wire_clearance import (
        TreeWireClearanceError,
        run_tree_wire_clearance,
    )
    from algorithms.vegetation.tree_wire_clearance_visualization import (
        run_tree_wire_clearance_visualization,
    )

    output_dir = Path(output_dir)
    if wire.get("TREE_WIRE_STATUS") != "VALID_W1":
        reason = wire.get("reason") or (
            "本次新源 FAST 路径未生成经 A1/A2A/A2B/A3/A4/W1 验证的 source-native W1，"
            "因此禁止计算树线距离。"
        )
        return blocked_clearance_result(output_dir, reason=reason)
    try:
        clearance = run_tree_wire_clearance(
            v4_root,
            Path(wire["w1_npz_path"]),
            Path(wire["w1_json_path"]),
            Path(wire["wire_chain_root"]),
            config=default_tree_wire_clearance_config(),
        )
    except TreeWireClearanceError as exc:
        return blocked_clearance_result(
            output_dir,
            reason=f"树线距离引擎 fail-closed：{exc}",
        )
    clearance_json = output_dir / "tree_wire_clearance.json"
    _write_json(clearance_json, clearance.to_dict())
    run_tree_wire_clearance_visualization(
        clearance_json,
        v4_root,
        Path(wire["w1_npz_path"]),
        Path(wire["w1_json_path"]),
        output_dir / "tree_wire_clearance_review.html",
        overwrite=False,
    )
    minimum_values = [
        float(record.minimum_clearance_m)
        for record in clearance.records
        if record.minimum_clearance_m is not None
    ]
    status = str(clearance.status.name) if clearance.status is not None else "UNKNOWN"
    if clearance.status == TreeWireClearanceStatus.NO_DISTANCE_READY_EDGE:
        status = "NO_DISTANCE_READY_EDGE"
    return {
        "TREE_WIRE_STATUS": (
            "VALID_W1"
            if clearance.status == TreeWireClearanceStatus.VALID
            else "BLOCKED_NO_VALID_W1"
        ),
        "TREE_WIRE_RESULT_STATUS": status,
        "GLOBAL_MINIMUM_CLEARANCE": min(minimum_values) if minimum_values else None,
        "reason": None,
        "result_digest": clearance.result_digest,
    }


def blocked_clearance_result(output_dir: Path, *, reason: str) -> dict[str, Any]:
    """Write explicit blocked clearance pages instead of a misleading distance."""

    payload = {
        "TREE_WIRE_STATUS": "BLOCKED_NO_VALID_W1",
        "TREE_WIRE_RESULT_STATUS": "BLOCKED_NO_VALID_W1",
        "GLOBAL_MINIMUM_CLEARANCE": None,
        "reason": reason,
        "non_certified_marker": NON_CERTIFIED_MARKER,
    }
    _write_json(output_dir / "tree_wire_clearance.json", payload)
    escaped = html.escape(reason)
    page = (
        "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
        "<title>树线距离状态</title><body><h1>树线距离：已阻止</h1>"
        "<p><strong>TREE_WIRE_STATUS: BLOCKED_NO_VALID_W1</strong></p>"
        f"<p>{escaped}</p><p>未使用任何历史 W1 或旧源导线模型。</p></body></html>"
    )
    _write_text(output_dir / "tree_wire_clearance_review.html", page)
    _write_text(output_dir / "minimum_clearance_review.html", page)
    return payload


def build_quality_flags(
    vegetation_manifest: dict[str, Any], wire: dict[str, Any], tree_summaries: dict[str, Any]
) -> dict[str, str]:
    """Produce visible, deterministic warnings without asserting semantic truth."""

    counts = vegetation_manifest["counts"]
    excluded_weight_proxy = int(counts["wire_geometry_excluded"]) + int(
        counts["rigid_linear_excluded"]
    )
    contamination_fraction = excluded_weight_proxy / max(1, int(counts["input_representatives"]))
    vegetation_quality = (
        "HEAVILY_CONTAMINATED"
        if contamination_fraction >= 0.25
        else "MODERATELY_CONTAMINATED"
        if contamination_fraction >= 0.08
        else "CLEAN"
    )
    trees = tree_summaries["trees"]
    overmerge = "NO"
    if len(trees) == 1:
        xy_extent = trees[0]["full_resolution_xy_extent_m"]
        if xy_extent[1] - xy_extent[0] > 35.0:
            overmerge = "YES"
    oversplit = "YES" if any(tree["representative_count"] < 25 for tree in trees) else "NO"
    return {
        "VEGETATION_QUALITY": vegetation_quality,
        "V4_OVERMERGE_SUSPECTED": overmerge,
        "V4_OVERSPLIT_SUSPECTED": oversplit,
        "TREE_WIRE_RESULT_STATUS": str(wire.get("TREE_WIRE_STATUS", "BLOCKED_NO_VALID_W1")),
    }


def render_summary_text(summary: dict[str, Any]) -> str:
    """Return a compact Chinese-friendly plain text report."""

    labels = {
        "INPUT_FILE": "输入文件",
        "RAW_POINTS": "原始点数",
        "COARSE_REPRESENTATIVES": "粗路由代表点",
        "COARSE_VOXEL_SIZE_M": "粗体素尺寸（米）",
        "VEGETATION_FINE_REPRESENTATIVES": "植被精细代表点",
        "VEGETATION_FINE_VOXEL_SIZE_M": "植被精细体素尺寸（米）",
        "V4_INPUT_REPRESENTATIVES": "V4 输入代表点",
        "TREE_COUNT": "识别单木",
        "ASSIGNED_FULLRES_POINTS": "已分配全分辨率点数",
        "W1_POLYLINE_COUNT": "W1 导线数量",
        "W1_DISTANCE_READY_SEGMENTS": "W1 距离就绪线段",
        "TREE_WIRE_STATUS": "树线距离状态",
        "GLOBAL_MINIMUM_CLEARANCE": "场景最小距离（米）",
        "VEGETATION_QUALITY": "植被质量",
        "V4_OVERMERGE_SUSPECTED": "疑似过合并",
        "V4_OVERSPLIT_SUSPECTED": "疑似过分割",
        "NON_CERTIFIED": "非认证研究演示",
    }
    lines = ["新 LAS 一键运行摘要", "=" * 40]
    for key, label in labels.items():
        lines.append(f"{label}: {summary.get(key)}")
    lines.append("")
    lines.append(f"结果目录: {summary['OUTPUT_DIRECTORY']}")
    return "\n".join(lines) + "\n"


def write_summary_html(summary: dict[str, Any], run_root: Path, output_html: Path) -> Path:
    """Write a local final page with only the principal review links."""

    links = (
        ("粗预处理审查", run_root / "01_preprocess" / "coarse_preprocess_review.html"),
        ("地面/HAG 审查", run_root / "02_ground" / "ground_hag_review.html"),
        ("导线候选原始点审查", run_root / "03_wire" / "wire_candidate_raw_review.html"),
        ("W1 重建审查", run_root / "03_wire" / "w1_reconstruction_review.html"),
        ("植被路由审查", run_root / "04_vegetation" / "vegetation_review.html"),
        ("精细植被输入审查", run_root / "04_vegetation" / "fine_vegetation_input_review.html"),
        (
            "原始与精细植被叠加",
            run_root / "04_vegetation" / "raw_vs_fine_vegetation_overlay.html",
        ),
        ("V4 单木（干净视图）", run_root / "05_v4" / "v4_tree_segmentation_clean.html"),
        ("V4 全分辨率回映射", run_root / "05_v4" / "v4_fullres_tree_preview.html"),
        ("树线距离状态", run_root / "06_clearance" / "tree_wire_clearance_review.html"),
    )
    rendered_links = "".join(
        _summary_link(label, path, output_html.parent)
        for label, path in links
        if path.is_file()
    )
    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
        if key not in {"TREE_DETAILS", "VISUALIZATIONS", "WIRE_MANIFEST", "BACKMAP"}
    )
    page = (
        "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
        "<title>新 LAS 运行摘要</title><style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;"
        "max-width:1100px;margin:36px auto;padding:0 20px}th{text-align:left;background:#eef2ff}"
        "th,td{padding:7px 10px;border:1px solid #d1d5db;vertical-align:top}"
        "table{border-collapse:collapse;"
        "width:100%}</style><body><h1>新 LAS 一键运行摘要</h1>"
        "<p><strong>NON-CERTIFIED: TRUE</strong> — 研究演示，非认证结果。</p>"
        f"<p><strong>树线距离状态:</strong> {html.escape(str(summary['TREE_WIRE_STATUS']))}</p>"
        f"<h2>主要可视化</h2><ul>{rendered_links}</ul><h2>运行指标</h2><table>{rows}</table>"
        "</body></html>"
    )
    _write_text(output_html, page)
    return output_html


def _build_run_summary(
    *,
    source: Path,
    input_sha: str,
    source_id: str,
    coarse_voxel_size_m: float,
    fine_veg_voxel_size_m: float,
    fine_wire_voxel_size_m: float,
    working: WeightedWorkingSet,
    ground: Any,
    wire: dict[str, Any],
    veg_route: dict[str, Any],
    v4_result: Any,
    tree_summaries: dict[str, Any],
    clearance: dict[str, Any],
    timings: dict[str, float],
    quality: dict[str, str],
    backmap: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    coarse_counts = veg_route["coarse_manifest"]["counts"]
    fine_counts = veg_route["fine_manifest"]["counts"]
    tree_records = tree_summaries["trees"]
    wire_manifest = {
        key: value
        for key, value in wire.items()
        if not isinstance(value, np.ndarray) and key != "candidate_representative_ids"
    }
    return {
        "RUN_STATUS": (
            "PASS" if clearance["TREE_WIRE_STATUS"] == "VALID_W1" else "PASS_WITH_TREE_WIRE_BLOCKED"
        ),
        "NON_CERTIFIED": True,
        "NON_CERTIFIED_MARKER": NON_CERTIFIED_MARKER,
        "INPUT_FILE": str(source),
        "INPUT_SHA256": input_sha,
        "SOURCE_ID": source_id,
        "OUTPUT_DIRECTORY": str(root),
        "RAW_POINTS": working.source_point_count,
        "COARSE_REPRESENTATIVES": working.representative_count,
        "COARSE_VOXEL_SIZE_M": coarse_voxel_size_m,
        "VEGETATION_FINE_VOXEL_SIZE_M": fine_veg_voxel_size_m,
        "VEGETATION_RAW_REGION_POINTS": fine_counts["region_raw_point_count"],
        "VEGETATION_FINE_REPRESENTATIVES": fine_counts["final_vegetation_representatives"],
        "FINE_REDUCTION_RATIO": fine_counts["fine_reduction_ratio"],
        "WIRE_RAW_REGION_POINTS": wire.get("wire_region_raw_point_count", 0),
        "WIRE_FINE_POINTS_OR_REPRESENTATIVES": wire.get("wire_fine_point_count", 0),
        "WIRE_FINE_VOXEL_SIZE_M": fine_wire_voxel_size_m,
        "WIRE_FINE_MODE": wire.get("wire_fine_mode", WIRE_FINE_MODE),
        "V4_INPUT_REPRESENTATIVES": veg_route["fine_vegetation"].point_count,
        "V4_TREE_COUNT": len(tree_records),
        "ASSIGNED_FULLRES_POINTS": tree_summaries["assigned_full_resolution_points"],
        "W1_POLYLINE_COUNT": wire.get("w1_polyline_count", 0),
        "W1_DISTANCE_READY_SEGMENTS": wire.get("w1_distance_ready_segments", 0),
        "TREE_WIRE_STATUS": clearance["TREE_WIRE_STATUS"],
        "TREE_WIRE_RESULT_STATUS": clearance.get(
            "TREE_WIRE_RESULT_STATUS", clearance["TREE_WIRE_STATUS"]
        ),
        "GLOBAL_MINIMUM_CLEARANCE": clearance.get("GLOBAL_MINIMUM_CLEARANCE"),
        # Legacy FAST keys retained for compatibility.
        "REPRESENTATIVE_VOXEL_SIZE_M": coarse_voxel_size_m,
        "REPRESENTATIVES": working.representative_count,
        "REDUCTION_RATIO": working.representative_count / working.source_point_count,
        "GROUND_POINTS": ground.ground_class_2_count,
        "HAG_VALID_POINTS": ground.hag_valid_count,
        "WIRE_CANDIDATES": int(wire.get("wire_candidates", 0)),
        "VEGETATION_RETAINED_REPRESENTATIVES": coarse_counts["final_vegetation_representatives"],
        "VEGETATION_RETAINED_RAW_WEIGHT": coarse_counts["final_vegetation_raw_weight"],
        "APEX_CANDIDATES": len(v4_result.apex_records),
        "ACCEPTED_APICES": int(v4_result.audit.hypothesis_count),
        "TREE_COUNT": len(tree_records),
        "ASSIGNED_REPRESENTATIVES": veg_route["fine_manifest"]["counts"][
            "final_vegetation_representatives"
        ],
        "ASSIGNED_FULL_RESOLUTION_POINTS": tree_summaries["assigned_full_resolution_points"],
        "PREPROCESS_RUNTIME": timings["preprocess"],
        "GROUND_RUNTIME": timings["ground"],
        "WIRE_RUNTIME": timings["wire"],
        "VEGETATION_RUNTIME": timings["vegetation"],
        "V4_RUNTIME": timings["v4"],
        "CLEARANCE_RUNTIME": timings["clearance"],
        "TOTAL_RUNTIME": timings["total"],
        **quality,
        "TREE_DETAILS": tree_records,
        "BACKMAP": backmap,
        "WIRE_MANIFEST": wire_manifest,
        "VISUALIZATIONS": {
            "coarse_preprocess_review": str(
                root / "01_preprocess" / "coarse_preprocess_review.html"
            ),
            "ground_hag_review": str(root / "02_ground" / "ground_hag_review.html"),
            "wire_candidate_raw_review": str(root / "03_wire" / "wire_candidate_raw_review.html"),
            "w1_reconstruction_review": str(root / "03_wire" / "w1_reconstruction_review.html"),
            "vegetation_review": str(root / "04_vegetation" / "vegetation_review.html"),
            "fine_vegetation_input_review": str(
                root / "04_vegetation" / "fine_vegetation_input_review.html"
            ),
            "raw_vs_fine_vegetation_overlay": str(
                root / "04_vegetation" / "raw_vs_fine_vegetation_overlay.html"
            ),
            "v4_tree_segmentation_review": str(root / "05_v4" / "v4_tree_segmentation_review.html"),
            "v4_tree_segmentation_clean": str(root / "05_v4" / "v4_tree_segmentation_clean.html"),
            "v4_fullres_tree_preview": str(root / "05_v4" / "v4_fullres_tree_preview.html"),
            "tree_wire_clearance_review": str(
                root / "06_clearance" / "tree_wire_clearance_review.html"
            ),
            "minimum_clearance_review": str(
                root / "06_clearance" / "minimum_clearance_review.html"
            ),
            "run_summary": str(root / "07_report" / "RUN_SUMMARY.html"),
        },
    }


def _inspect_source(source: Path, input_sha: str, source_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    with laspy.open(source) as reader:
        header = reader.header
        payload = {
            "input_file": str(source),
            "input_sha256": input_sha,
            "source_id": source_id,
            "point_count": int(header.point_count),
            "bounds_min_xyz": [float(value) for value in header.mins],
            "bounds_max_xyz": [float(value) for value in header.maxs],
            "scales": [float(value) for value in header.scales],
            "offsets": [float(value) for value in header.offsets],
            "point_format": int(header.point_format.id),
            "version": str(header.version),
        }
    if payload["point_count"] <= 0:
        raise FastResearchError("NEW_LAS_DEMO_EMPTY_SOURCE")
    payload["inspection_runtime_seconds"] = time.perf_counter() - started
    return payload


def _ensure_empty_run_root(root: Path) -> None:
    if root.exists():
        # The CLI creates run.log before orchestration so every later failure
        # has a durable traceback.  It is the sole permitted pre-existing file.
        unexpected = [path for path in root.iterdir() if path.name != "run.log"]
        if unexpected:
            raise FastResearchError(f"NEW_LAS_DEMO_OUTPUT_EXISTS: {root}")
    else:
        root.mkdir(parents=True)


def _safe_stem(stem: str) -> str:
    value = re.sub(r'[<>:"/\\|?*]+', "_", stem).strip(" ._")
    return value[:80] or "new_las"


def _even_rows(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _report(
    progress: _Progress | None,
    stage: int,
    message: str,
    done: bool,
    elapsed: float | None,
) -> None:
    if progress is not None:
        progress(stage, message, done, elapsed)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _write_figure(figure: Any, output_html: Path) -> Path:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_html.with_name(output_html.name + ".tmp")
    figure.write_html(temporary, include_plotlyjs=True, full_html=True)
    os.replace(temporary, output_html)
    return output_html


def _summary_link(label: str, path: Path, parent: Path) -> str:
    relative = os.path.relpath(path, parent).replace(os.sep, "/")
    return f'<li><a href="{html.escape(relative)}">{html.escape(label)}</a></li>'


__all__ = [
    "DemoRunResult",
    "SUPPORTED_SUFFIXES",
    "blocked_clearance_result",
    "choose_representative_voxel_size",
    "create_isolated_run_dir",
    "discover_demo_inputs",
    "new_source_id",
    "run_new_las_demo",
]
