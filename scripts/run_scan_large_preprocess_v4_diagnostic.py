"""Run the FAST-only preprocess-first, raw-weighted scan_large V4 diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import plotly.graph_objects as go

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from algorithms.vegetation.fast_research import run_fast_ground_hag, run_fast_v4  # noqa: E402
from algorithms.vegetation.fast_weighted_preprocess import (  # noqa: E402
    FAST_WEIGHTED_MARKER,
    WeightedWorkingSet,
    backmap_representative_tree_ids,
    build_weighted_fast_vegetation,
    build_weighted_working_set,
    select_weighted_multitree_roi,
    write_weighted_roi_las,
)
from algorithms.vegetation.v4_tree_visualization import run_v4_tree_visualization  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.25)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), "utf-8")
    os.replace(temporary, path)


def _even_rows(count: int, maximum: int = 120_000) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _write_preprocessing_review(working: WeightedWorkingSet, output: Path) -> Path:
    with np.load(working.representatives_npz_path, allow_pickle=False) as stream:
        xyz = np.asarray(stream["xyz_absolute"], dtype=np.float64)
        raw_count = np.asarray(stream["raw_count"], dtype=np.uint32)
    rows = _even_rows(len(xyz))
    figure = go.Figure(
        go.Scatter3d(
            x=xyz[rows, 0],
            y=xyz[rows, 1],
            z=xyz[rows, 2],
            mode="markers",
            marker={"size": 2, "color": np.log1p(raw_count[rows]), "colorscale": "Viridis"},
            customdata=np.column_stack((raw_count[rows],)),
            hovertemplate="raw_count=%{customdata[0]}<extra></extra>",
            name="compact representatives",
        )
    )
    figure.update_layout(
        title="NON-CERTIFIED scan_large preprocessing — representative raw populations",
        scene={"aspectmode": "data"},
        annotations=[
            {
                "text": (
                    f"raw={working.source_point_count:,}; "
                    f"representatives={working.representative_count:,}; "
                    f"raw_count_sum={working.raw_count_sum:,}; voxel={working.voxel_size_m:.2f} m"
                ),
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.08,
            }
        ],
    )
    figure.write_html(output, include_plotlyjs="cdn", full_html=True)
    return output


def _write_vegetation_review(diagnostic_npz: Path, output: Path) -> Path:
    with np.load(diagnostic_npz, allow_pickle=False) as stream:
        xyz = np.asarray(stream["xyz_local_m"], dtype=np.float64)
        state = np.asarray(stream["diagnostic_state"], dtype=np.uint8)
        raw_count = np.asarray(stream["raw_count"], dtype=np.uint32)
    labels = {
        1: ("ground excluded", "#8c8c8c"),
        4: ("rigid linear excluded", "#756bb1"),
        5: ("rigid surface excluded", "#54278f"),
        6: ("no positive vegetation evidence", "#bdbdbd"),
        7: ("retained vegetation", "#31a354"),
    }
    figure = go.Figure()
    for code, (name, color) in labels.items():
        rows = np.flatnonzero(state == code)
        rows = rows[_even_rows(len(rows), 45_000)] if len(rows) else rows
        figure.add_trace(
            go.Scatter3d(
                x=xyz[rows, 0],
                y=xyz[rows, 1],
                z=xyz[rows, 2],
                mode="markers",
                name=name,
                marker={"size": 2, "color": color, "opacity": 0.65},
                customdata=np.column_stack((raw_count[rows],)),
                hovertemplate="raw_count=%{customdata[0]}<extra></extra>",
            )
        )
    figure.update_layout(
        title="scan_large ROI FAST vegetation — no native W1 is available",
        scene={"aspectmode": "data"},
    )
    figure.write_html(output, include_plotlyjs="cdn", full_html=True)
    return output


def _write_clean_v4_overview(v4_npz: Path, v4_json: Path, output: Path) -> Path:
    with np.load(v4_npz, allow_pickle=False) as stream:
        xyz = np.asarray(stream["xyz_local_m"], dtype=np.float64)
        tree = np.asarray(stream["tree_id_by_point"], dtype=np.int32)
    payload = json.loads(v4_json.read_text("utf-8"))
    figure = go.Figure()
    for tree_id in sorted(int(value) for value in np.unique(tree) if value >= 0):
        rows = np.flatnonzero(tree == tree_id)
        rows = rows[_even_rows(len(rows), 20_000)]
        figure.add_trace(
            go.Scatter3d(
                x=xyz[rows, 0],
                y=xyz[rows, 1],
                z=xyz[rows, 2],
                mode="markers",
                name=f"tree {tree_id}",
                marker={"size": 2},
            )
        )
    apexes = payload.get("apex_records", [])
    if apexes:
        apex_xyz = np.asarray([record["xyz_local_m"] for record in apexes], dtype=np.float64)
        figure.add_trace(
            go.Scatter3d(
                x=apex_xyz[:, 0],
                y=apex_xyz[:, 1],
                z=apex_xyz[:, 2],
                mode="markers",
                name="apex candidates",
                marker={"size": 7, "color": "#e31a1c", "symbol": "diamond"},
            )
        )
    figure.update_layout(
        title="scan_large ROI V4 clean overview — tree IDs and apices",
        scene={"aspectmode": "data"},
    )
    figure.write_html(output, include_plotlyjs="cdn", full_html=True)
    return output


def _write_fullres_preview(
    working: WeightedWorkingSet,
    backmap_npz: Path,
    output: Path,
    *,
    blocks: int = 20,
    rows: int = 6_000,
) -> Path:
    with np.load(backmap_npz, allow_pickle=False) as stream:
        representative_id = np.asarray(stream["representative_id"], dtype=np.uint32)
        tree_id = np.asarray(stream["tree_id_by_representative"], dtype=np.int32)
    tree_by_representative = np.full(working.representative_count, -1, dtype=np.int32)
    tree_by_representative[representative_id] = tree_id
    raw_to_representative = np.load(working.raw_to_representative_npy_path, mmap_mode="r")
    samples: list[np.ndarray] = []
    tree_samples: list[np.ndarray] = []
    with laspy.open(working.source_las_path) as reader:
        starts = np.linspace(0, max(0, working.source_point_count - rows), blocks, dtype=np.int64)
        for start in starts:
            reader.seek(int(start))
            points = reader.read_points(rows)
            positions = np.arange(int(start), int(start) + len(points), dtype=np.int64)
            assigned = tree_by_representative[raw_to_representative[positions]]
            keep = assigned >= 0
            if np.any(keep):
                samples.append(np.column_stack((points.x[keep], points.y[keep], points.z[keep])))
                tree_samples.append(assigned[keep])
    xyz = np.concatenate(samples) if samples else np.empty((0, 3), dtype=np.float64)
    tree = np.concatenate(tree_samples) if tree_samples else np.empty(0, dtype=np.int32)
    figure = go.Figure()
    for tree_id_value in sorted(int(value) for value in np.unique(tree)):
        rows_index = np.flatnonzero(tree == tree_id_value)
        figure.add_trace(
            go.Scatter3d(
                x=xyz[rows_index, 0],
                y=xyz[rows_index, 1],
                z=xyz[rows_index, 2],
                mode="markers",
                name=f"tree {tree_id_value}",
                marker={"size": 2},
            )
        )
    figure.update_layout(
        title="scan_large ROI — deterministic full-resolution back-mapping preview",
        scene={"aspectmode": "data"},
    )
    figure.write_html(output, include_plotlyjs="cdn", full_html=True)
    return output


def _tree_diagnostics(
    v4_json: Path, backmap: dict[str, Any]
) -> tuple[list[dict[str, Any]], bool, bool]:
    payload = json.loads(v4_json.read_text("utf-8"))
    trees: list[dict[str, Any]] = []
    overmerge = False
    for record in payload["tree_records"]:
        geometry = record["geometry_summary"]
        lower = geometry["xy_bounds_min"]
        upper = geometry["xy_bounds_max"]
        extent = [float(upper[0] - lower[0]), float(upper[1] - lower[1])]
        secondary_apex_count = len(record["secondary_apex_candidate_ids"])
        flag = bool(
            secondary_apex_count > 0
            or max(extent) > 12.0
            or float(geometry["crown_area_m2"]) > 100.0
        )
        overmerge = overmerge or flag
        trees.append(
            {
                "tree_id": int(record["tree_id"]),
                "representative_member_count": int(record["member_count"]),
                "hag_min_m": float(geometry["hag_min_m"]),
                "hag_max_m": float(geometry["hag_max_m"]),
                "hag_median_m": float(geometry["robust_hag_m"]),
                "xy_extent_m": extent,
                "z_extent_m": float(geometry["vertical_extent_m"]),
                "footprint_area_m2": float(geometry["crown_area_m2"]),
                "primary_apex_candidate_id": record["primary_apex_candidate_id"],
                "secondary_apex_count": secondary_apex_count,
                "possible_overmerge": flag,
            }
        )
    centroids = [
        record["geometry_summary"]["xy_centroid"][:2]
        for record in payload["tree_records"]
    ]
    oversplit = any(
        np.hypot(first[0] - second[0], first[1] - second[1]) < 2.0
        for index, first in enumerate(centroids)
        for second in centroids[index + 1 :]
    )
    return trees, overmerge, oversplit


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    input_las = args.input.resolve()
    output = args.output.resolve()
    if output.exists() and not output.is_dir():
        raise RuntimeError("scan_large output must be a directory")
    started = time.perf_counter()
    preprocess_status, working = build_weighted_working_set(
        input_las, output / "preprocess", voxel_size_m=args.voxel_size, chunk_size=args.chunk_size
    )
    visual = output / "visualizations"
    preprocessing_review = _write_preprocessing_review(
        working, visual / "scan_large_preprocessing_review.html"
    )
    selection_started = time.perf_counter()
    roi = select_weighted_multitree_roi(working, output)
    roi_las = write_weighted_roi_las(working, roi, output)
    selection_seconds = time.perf_counter() - selection_started
    ground_status, ground = run_fast_ground_hag(roi_las, output / "ground")
    vegetation, vegetation_manifest = build_weighted_fast_vegetation(
        working, roi, ground, output / "vegetation"
    )
    vegetation_review = _write_vegetation_review(
        output / "vegetation" / "scan_large_fast_vegetation_diagnostics.npz",
        visual / "scan_large_fast_vegetation_review.html",
    )
    v4_status, v4_root, v4_summary = run_fast_v4(
        vegetation,
        output / "v4",
        source_bounds_min_xyz=working.source_bounds_min_xyz,
        source_bounds_max_xyz=working.source_bounds_max_xyz,
    )
    v4_review = visual / "scan_large_v4_review.html"
    v4_review_json = visual / "scan_large_v4_review.json"
    run_v4_tree_visualization(v4_root, v4_review, v4_review_json, overwrite=False)
    clean_overview = _write_clean_v4_overview(
        v4_root / "v4_tree_segmentation.npz",
        v4_root / "v4_tree_segmentation.json",
        visual / "scan_large_v4_clean_overview.html",
    )
    backmap_started = time.perf_counter()
    backmap_path = output / "fullres_back_mapping.npz"
    backmap = backmap_representative_tree_ids(
        working,
        output,
        Path(vegetation.output_npz_path),
        v4_root / "v4_tree_segmentation.npz",
        backmap_path,
    )
    fullres_preview = _write_fullres_preview(
        working, backmap_path, visual / "scan_large_v4_fullres_preview.html"
    )
    backmap_seconds = time.perf_counter() - backmap_started
    v4_payload = json.loads((v4_root / "v4_tree_segmentation.json").read_text("utf-8"))
    trees, overmerge, oversplit = _tree_diagnostics(v4_root / "v4_tree_segmentation.json", backmap)
    accepted_apices = sum(
        1 for record in v4_payload["apex_records"] if int(record["quality_flags"]) == 0
    )
    diagnostics = {
        "schema_version": "scan-large-preprocess-weighted-v4-diagnostic-v1",
        "non_certified_marker": FAST_WEIGHTED_MARKER,
        "input": "SCAN_LARGE_MULTI_TREE_ROI",
        "preprocess_status": preprocess_status,
        "ground_status": ground_status,
        "v4_status": v4_status,
        "working_set": working.to_dict(),
        "roi": roi,
        "roi_las_path": str(roi_las),
        "stage_seconds": {
            "preprocess": working.elapsed_seconds,
            "roi_selection": selection_seconds,
            "ground_hag": ground.elapsed_seconds,
            "geometry_vegetation": vegetation.elapsed_seconds,
            "v4": float(
                json.loads(
                    (output / "v4" / "fast_v4_adapter.json").read_text("utf-8")
                )["elapsed_seconds"]
            ),
            "back_mapping": backmap_seconds,
            "total": time.perf_counter() - started,
        },
        "vegetation": vegetation_manifest,
        "current_v4_density_source": "RAW_WEIGHTED_COUNT",
        "v4_summary": v4_summary,
        "apex_candidates": len(v4_payload["apex_records"]),
        "accepted_apices": accepted_apices,
        "back_mapping": backmap,
        "trees": trees,
        "overmerge_suspected": overmerge,
        "oversplit_suspected": oversplit,
        "outputs": {
            "preprocessing_review": str(preprocessing_review),
            "vegetation_review": str(vegetation_review),
            "v4_review": str(v4_review),
            "v4_clean_overview": str(clean_overview),
            "fullres_preview": str(fullres_preview),
        },
    }
    _write_json(output / "scan_large_v4_diagnostics.json", diagnostics)
    comparison = {
        "scan_small": {
            "raw_points": 6905520,
            "fast_v3_vegetation": 94516,
            "v4_trees": 10,
            "assigned_points": 72224,
            "owner_visual_finding": "not convincing one-tree-per-crown segmentation",
        },
        "scan_large": {
            "raw_points": working.source_point_count,
            "representatives": working.representative_count,
            "reduction_ratio": working.source_point_count / working.representative_count,
            "fast_vegetation_representatives": vegetation.point_count,
            "v4_trees": v4_summary["tree_count"],
            "assigned_representatives": v4_summary["assigned_point_count"],
            "assigned_raw_weight": backmap["assigned_raw_weight"],
            "overmerge_suspected": overmerge,
            "oversplit_suspected": oversplit,
        },
    }
    _write_json(output / "scan_small_vs_scan_large_v4_comparison.json", comparison)
    return diagnostics


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
