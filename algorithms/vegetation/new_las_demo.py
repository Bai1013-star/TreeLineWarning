"""Non-certified, source-native orchestration for the one-click LAS demo.

The module deliberately creates a new isolated output namespace for every LAS
input.  It never looks for historical W1, A2A, V1, V2, or V4 artifacts.  A
fresh wire-like geometry candidate may be used as a conservative vegetation
exclusion, but it is never relabelled as W1 and never used for clearance.
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
    backmap_full_working_set_tree_ids,
    build_weighted_fast_vegetation,
    build_weighted_working_set,
)
from algorithms.vegetation.v4_tree_artifacts import load_v4_tree_segmentation_artifact
from algorithms.vegetation.v4_tree_visualization import run_v4_tree_visualization, tree_color

SUPPORTED_SUFFIXES = {".las", ".laz"}
NON_CERTIFIED_MARKER = "NON_CERTIFIED_RESEARCH_DEMONSTRATION"
_Progress = Callable[[int, str, bool, float | None], None]


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
    """Choose a bounded FAST representative scale from the new source size.

    The reviewed 0.25 m setting remains the normal choice. Larger sources get
    a modestly coarser computation representation instead of silently assuming
    that every future LAS has the same density or memory budget. Raw-row
    mapping and raw_count remain lossless at every selected scale.
    """

    if type(raw_point_count) is not int or raw_point_count <= 0:
        raise FastResearchError("NEW_LAS_DEMO_RAW_POINT_COUNT_INVALID")
    if raw_point_count <= 5_000_000:
        return 0.25
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
) -> DemoRunResult:
    """Run the reviewed FAST route in one isolated source-native namespace.

    The formal W1 producer requires a valid A1→A4 provenance chain.  This
    deliberately does not synthesize one from a fresh LAS.  Therefore the
    output will explicitly block Tree-Wire clearance unless a future source-
    native formal producer is added within this run directory.
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
    selected_voxel_size_m = (
        choose_representative_voxel_size(int(source_inspection["point_count"]))
        if voxel_size_m is None
        else float(voxel_size_m)
    )
    if not np.isfinite(selected_voxel_size_m) or selected_voxel_size_m <= 0:
        raise FastResearchError("NEW_LAS_DEMO_VOXEL_SIZE_INVALID")
    source_inspection["representative_voxel_size_m"] = selected_voxel_size_m
    _write_json(root / "00_input" / "source_inspection.json", source_inspection)
    timings: dict[str, float] = {}
    _report(progress, 1, "正在读取点云...", False, None)
    timings["source_inspection"] = float(source_inspection["inspection_runtime_seconds"])
    _report(progress, 1, "读取点云完成", True, timings["source_inspection"])

    _report(progress, 2, "正在进行点云预处理...", False, None)
    started = time.perf_counter()
    _preprocess_status, working = build_weighted_working_set(
        source,
        root / "01_preprocess",
        voxel_size_m=selected_voxel_size_m,
        artifact_stem="new_las",
    )
    generate_preprocessing_review(working, root / "01_preprocess" / "preprocessing_review.html")
    timings["preprocess"] = time.perf_counter() - started
    _report(progress, 2, "点云预处理完成", True, timings["preprocess"])

    _report(progress, 3, "正在进行地面分离和高度计算...", False, None)
    started = time.perf_counter()
    _ground_status, ground = run_fast_ground_hag(
        Path(working.representative_las_path),
        root / "02_ground",
    )
    generate_fast_ground_preview(ground, root / "02_ground" / "ground_hag_review.html")
    timings["ground"] = time.perf_counter() - started
    _report(progress, 3, "地面分离和高度计算完成", True, timings["ground"])

    _report(progress, 4, "正在识别和重建导线...", False, None)
    started = time.perf_counter()
    wire = run_source_native_wire_candidate_path(
        ground.output_las_path,
        root / "03_wire",
        origin_xyz=working.source_bounds_min_xyz,
    )
    timings["wire"] = time.perf_counter() - started
    _report(progress, 4, "导线候选审查完成", True, timings["wire"])

    _report(progress, 5, "正在提取植被...", False, None)
    started = time.perf_counter()
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
    vegetation, vegetation_manifest = build_weighted_fast_vegetation(
        working,
        all_scene_roi,
        ground,
        root / "04_vegetation",
        source_id=source_id,
        wire_geometry_representative_ids=wire["candidate_representative_ids"],
        artifact_stem="new_las",
    )
    generate_vegetation_review(
        vegetation_manifest,
        root / "04_vegetation" / "vegetation_review.html",
    )
    timings["vegetation"] = time.perf_counter() - started
    _report(progress, 5, "植被提取完成", True, timings["vegetation"])

    _report(progress, 6, "正在进行单木分割...", False, None)
    started = time.perf_counter()
    _v4_status, v4_root, _ = run_fast_v4(
        vegetation,
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
    backmap = backmap_full_working_set_tree_ids(
        working,
        loaded_v4.result.original_index,
        assigned_tree_ids,
        root / "05_v4" / "full_resolution_representative_backmap.npz",
    )
    tree_summaries = summarize_full_resolution_trees(
        source,
        working,
        loaded_v4.result,
        backmap,
    )
    _write_json(root / "05_v4" / "full_resolution_tree_summary.json", tree_summaries)
    timings["v4"] = time.perf_counter() - started
    _report(progress, 6, "单木分割完成", True, timings["v4"])

    _report(progress, 7, "正在计算树线最小距离...", False, None)
    started = time.perf_counter()
    clearance = blocked_clearance_result(
        root / "06_clearance",
        reason=(
            "本次新源 FAST 路径仅产生导线候选与植被排除证据；未生成经 A1/A2A/A2B/A3/A4 "
            "验证的 source-native W1，因此禁止计算树线距离。"
        ),
    )
    timings["clearance"] = time.perf_counter() - started
    _report(progress, 7, "树线距离状态已确定", True, timings["clearance"])

    quality = build_quality_flags(vegetation_manifest, wire, tree_summaries)
    summary = _build_run_summary(
        source=source,
        input_sha=input_sha,
        source_id=source_id,
        representative_voxel_size_m=selected_voxel_size_m,
        working=working,
        ground=ground,
        wire=wire,
        vegetation_manifest=vegetation_manifest,
        v4_result=loaded_v4.result,
        backmap=backmap,
        tree_summaries=tree_summaries,
        clearance=clearance,
        timings={**timings, "total": time.perf_counter() - total_started},
        quality=quality,
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
    """Write a bounded source-native representative/raw-weight review."""

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
            name="0.25 m computation representatives",
        )
    )
    figure.update_layout(
        title="新 LAS 预处理审查（代表点用于计算，raw_count 保留原始密度）",
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
    """Produce source-native wire-like candidate evidence without manufacturing W1."""

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
                "usage": "VISUAL_REVIEW_AND_VEGETATION_EXCLUSION_ONLY",
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
        "candidate_raw_weight": None,
        "candidate_segments": segments,
        "feature_runtime_seconds": feature_runtime,
        "runtime_seconds": time.perf_counter() - started,
        "w1_distance_ready_segments": 0,
        "tree_wire_status": "BLOCKED_NO_VALID_W1",
        "w1_policy": (
            "No W1 is emitted from heuristic candidates. Formal W1 requires a fresh, "
            "validated A1/A2A/A2B/A3/A4 chain for this exact source."
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
            name="非地面代表点（审查样本）",
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
                name="新源导线样候选（仅排除/审查，非 W1）",
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
                name=f"候选线段 {segment['candidate_id']}（非距离模型）",
                line={"color": "#991b1b", "width": 5},
            )
        )
    figure.update_layout(
        title="导线候选审查：未生成距离就绪 W1，树线距离已阻止",
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
    )
    _write_figure(figure, output / "wire_review.html")
    return {**manifest, "candidate_representative_ids": candidate_ids}


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
        (FastV3DiagnosticState.VEGETATION_RETAINED, "保留植被", "#228b22"),
        (FastV3DiagnosticState.WIRE_GEOMETRY_EXCLUDED, "导线样候选排除（非 W1）", "#dc2626"),
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
        title="FAST 植被保留审查（非认证研究演示）",
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
        legend={"orientation": "h"},
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
        title="V4 单木分割：仅显示已分配树点（代表点）",
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
    )
    return _write_figure(figure, output_html)


def summarize_full_resolution_trees(
    source: Path,
    working: WeightedWorkingSet,
    v4_result: Any,
    backmap: dict[str, Any],
) -> dict[str, Any]:
    """Compute exact raw XYZ extents and counts from source-row memberships."""

    raw_tree_path = Path(backmap["full_resolution_tree_id_by_source_row_path"])
    raw_tree = np.load(raw_tree_path, mmap_mode="r")
    assigned_state = np.asarray(v4_result.assignment_state_by_point, dtype=np.uint8)
    assigned = assigned_state == int(V4TreeAssignmentState.ASSIGNED)
    tree_by_point = np.asarray(v4_result.tree_id_by_point, dtype=np.int32)
    unique_tree_ids = sorted(
        int(value) for value in np.unique(tree_by_point[assigned]) if value >= 0
    )
    tree_data: dict[int, dict[str, Any]] = {
        tree_id: {
            "tree_id": tree_id,
            "representative_count": int(np.count_nonzero(assigned & (tree_by_point == tree_id))),
            "full_resolution_point_count": 0,
            "full_resolution_min_xyz": np.full(3, np.inf, dtype=np.float64),
            "full_resolution_max_xyz": np.full(3, -np.inf, dtype=np.float64),
        }
        for tree_id in unique_tree_ids
    }
    offset = 0
    with laspy.open(source) as reader:
        for chunk in reader.chunk_iterator(500_000):
            stop = offset + len(chunk)
            ids = np.asarray(raw_tree[offset:stop], dtype=np.int32)
            xyz = np.column_stack((chunk.x, chunk.y, chunk.z)).astype(np.float64, copy=False)
            for tree_id in np.unique(ids[ids >= 0]):
                row = tree_data[int(tree_id)]
                selected = xyz[ids == tree_id]
                row["full_resolution_point_count"] += int(len(selected))
                row["full_resolution_min_xyz"] = np.minimum(
                    row["full_resolution_min_xyz"], selected.min(axis=0)
                )
                row["full_resolution_max_xyz"] = np.maximum(
                    row["full_resolution_max_xyz"], selected.max(axis=0)
                )
            offset = stop
    if offset != working.source_point_count:
        raise FastResearchError("NEW_LAS_DEMO_FULLRES_SOURCE_COUNT_MISMATCH")
    apex_by_id = {record.candidate_id: record for record in v4_result.apex_records}
    tree_record_by_id = {int(record.tree_id): record for record in v4_result.tree_records}
    origin = np.asarray(working.source_bounds_min_xyz, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    hag = np.asarray(v4_result.hag_m, dtype=np.float32)
    for tree_id in unique_tree_ids:
        entry = tree_data[tree_id]
        record = tree_record_by_id[tree_id]
        apex = apex_by_id.get(record.primary_apex_candidate_id)
        tree_rows = assigned & (tree_by_point == tree_id)
        minimum = entry.pop("full_resolution_min_xyz")
        maximum = entry.pop("full_resolution_max_xyz")
        if entry["full_resolution_point_count"] <= 0:
            raise FastResearchError("NEW_LAS_DEMO_FULLRES_TREE_MEMBERSHIP_EMPTY")
        entry["apex_xyz_local_m"] = (
            None if apex is None else [float(value) for value in apex.xyz_local_m]
        )
        entry["apex_xyz_absolute_m"] = (
            None
            if apex is None
            else [float(value) for value in np.asarray(apex.xyz_local_m, dtype=np.float64) + origin]
        )
        hag_range = [
            float(np.min(hag[tree_rows])),
            float(np.max(hag[tree_rows])),
        ]
        entry["HAG_RANGE_M"] = hag_range
        entry["HAG_RANGE_SEMANTICS"] = (
            "HAG is evaluated on compact representatives; full-resolution XYZ membership "
            "and extents are streamed from the original source rows."
        )
        entry["representative_hag_range_m"] = hag_range
        entry["full_resolution_xy_extent_m"] = [
            float(minimum[0]),
            float(maximum[0]),
            float(minimum[1]),
            float(maximum[1]),
        ]
        entry["full_resolution_z_extent_m"] = [float(minimum[2]), float(maximum[2])]
        rows.append(entry)
    return {
        "mapping_semantics": backmap["mapping_semantics"],
        "full_resolution_tree_id_by_source_row_path": str(raw_tree_path),
        "trees": rows,
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
    del wire  # Counts are raw-weighted where available; flags stay conservative.
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
        "TREE_WIRE_RESULT_STATUS": "BLOCKED_NO_VALID_W1",
    }


def render_summary_text(summary: dict[str, Any]) -> str:
    """Return a compact Chinese-friendly plain text report."""

    labels = {
        "INPUT_FILE": "输入文件",
        "RAW_POINTS": "原始点数",
        "REPRESENTATIVES": "计算代表点",
        "TREE_COUNT": "识别单木",
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
        ("预处理审查", run_root / "01_preprocess" / "preprocessing_review.html"),
        ("地面/HAG 审查", run_root / "02_ground" / "ground_hag_review.html"),
        ("导线候选审查", run_root / "03_wire" / "wire_review.html"),
        ("植被审查", run_root / "04_vegetation" / "vegetation_review.html"),
        ("V4 单木（干净视图）", run_root / "05_v4" / "v4_tree_segmentation_clean.html"),
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
        if key not in {"TREE_DETAILS", "VISUALIZATIONS"}
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
    representative_voxel_size_m: float,
    working: WeightedWorkingSet,
    ground: Any,
    wire: dict[str, Any],
    vegetation_manifest: dict[str, Any],
    v4_result: Any,
    backmap: dict[str, Any],
    tree_summaries: dict[str, Any],
    clearance: dict[str, Any],
    timings: dict[str, float],
    quality: dict[str, str],
    root: Path,
) -> dict[str, Any]:
    vegetation_counts = vegetation_manifest["counts"]
    tree_records = tree_summaries["trees"]
    return {
        "RUN_STATUS": "PASS_WITH_TREE_WIRE_BLOCKED",
        "NON_CERTIFIED": True,
        "NON_CERTIFIED_MARKER": NON_CERTIFIED_MARKER,
        "INPUT_FILE": str(source),
        "INPUT_SHA256": input_sha,
        "SOURCE_ID": source_id,
        "OUTPUT_DIRECTORY": str(root),
        "RAW_POINTS": working.source_point_count,
        "REPRESENTATIVES": working.representative_count,
        "REPRESENTATIVE_VOXEL_SIZE_M": representative_voxel_size_m,
        "REDUCTION_RATIO": working.representative_count / working.source_point_count,
        "GROUND_POINTS": ground.ground_class_2_count,
        "HAG_VALID_POINTS": ground.hag_valid_count,
        "WIRE_CANDIDATES": wire["wire_candidates"],
        "W1_DISTANCE_READY_SEGMENTS": 0,
        "VEGETATION_RETAINED_REPRESENTATIVES": vegetation_counts[
            "final_vegetation_representatives"
        ],
        "VEGETATION_RETAINED_RAW_WEIGHT": vegetation_counts["final_vegetation_raw_weight"],
        "APEX_CANDIDATES": len(v4_result.apex_records),
        "ACCEPTED_APICES": len(tree_records),
        "TREE_COUNT": len(tree_records),
        "ASSIGNED_REPRESENTATIVES": backmap["assigned_representatives"],
        "ASSIGNED_FULL_RESOLUTION_POINTS": backmap["assigned_full_resolution_points"],
        "TREE_WIRE_STATUS": clearance["TREE_WIRE_STATUS"],
        "TREE_WIRE_RESULT_STATUS": clearance["TREE_WIRE_RESULT_STATUS"],
        "GLOBAL_MINIMUM_CLEARANCE": clearance["GLOBAL_MINIMUM_CLEARANCE"],
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
        "WIRE_MANIFEST": {
            key: value for key, value in wire.items() if key != "candidate_representative_ids"
        },
        "VISUALIZATIONS": {
            "preprocessing_review": str(root / "01_preprocess" / "preprocessing_review.html"),
            "ground_hag_review": str(root / "02_ground" / "ground_hag_review.html"),
            "wire_review": str(root / "03_wire" / "wire_review.html"),
            "vegetation_review": str(root / "04_vegetation" / "vegetation_review.html"),
            "v4_tree_segmentation_review": str(root / "05_v4" / "v4_tree_segmentation_review.html"),
            "v4_tree_segmentation_clean": str(root / "05_v4" / "v4_tree_segmentation_clean.html"),
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
