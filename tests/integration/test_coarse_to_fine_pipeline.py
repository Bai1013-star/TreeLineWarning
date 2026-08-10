"""Bounded synthetic integration for the source-native coarse-to-fine pipeline.

Runs the whole one-click route on a small deterministic LAS and asserts the
owner-mandated summary fields and fail-closed semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
import pytest
from algorithms.vegetation.new_las_demo import run_new_las_demo


def _build_scene(path: Path, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    gx, gy = np.meshgrid(np.arange(0, 60.5, 0.4), np.arange(-10, 10.5, 0.4))
    ground = np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, 0.0)))
    parts.append(ground)
    classes.append(np.full(gx.size, 2, np.uint8))
    s = np.arange(0, 30.0, 0.05)
    wire = np.column_stack((s, np.full(s.size, 2.0), np.full(s.size, 10.0)))
    parts.append(wire)
    classes.append(np.full(s.size, 6, np.uint8))
    s2 = np.arange(0, 18.0, 0.05)
    wire2 = np.column_stack((s2, np.full(s2.size, 4.0), np.full(s2.size, 10.5)))
    parts.append(wire2)
    classes.append(np.full(s2.size, 6, np.uint8))
    trunk = np.column_stack(
        (np.full(40, 15.0), rng.uniform(-0.15, 0.15, 40), np.linspace(0, 3.0, 40))
    )
    parts.append(trunk)
    classes.append(np.full(40, 5, np.uint8))
    n = 5000
    while True:
        pts = rng.uniform((-3, -3, 0), (3, 3, 6), (n, 3))
        ell = (
            (pts[:, 0]) ** 2 / 2.5**2
            + (pts[:, 1]) ** 2 / 2.5**2
            + (pts[:, 2] - 3.5) ** 2 / 3.0**2
        ) <= 1.0
        pts = pts[ell]
        if len(pts) >= 1600:
            break
        n *= 2
    crown = pts[:2000] + np.array([15.0, 0.0, 0.0])
    parts.append(crown)
    classes.append(np.full(len(crown), 5, np.uint8))
    trunk2 = np.column_stack(
        (np.full(30, 25.0), rng.uniform(-0.15, 0.15, 30), np.linspace(0, 2.5, 30))
    )
    parts.append(trunk2)
    classes.append(np.full(30, 5, np.uint8))
    crown2 = rng.normal((25.0, -3.0, 3.0), (1.4, 1.4, 1.2), (900, 3))
    crown2[:, 2] = np.clip(crown2[:, 2], 0.5, None)
    parts.append(crown2)
    classes.append(np.full(len(crown2), 5, np.uint8))
    xyz = np.concatenate(parts, axis=0)
    classification = np.concatenate(classes, axis=0)
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    xyz = xyz[order]
    classification = classification[order]
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray((0.001, 0.001, 0.001))
    header.offsets = np.zeros(3, dtype=np.float64)
    data = laspy.LasData(header)
    data.x, data.y, data.z = xyz.T
    data.classification = classification
    data.write(path)


def test_one_click_summary_reports_both_resolutions_and_wire_mode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scene.las"
    _build_scene(source)
    run_root = tmp_path / "run"
    result = run_new_las_demo(source, run_root, progress=None)
    assert result.overall_status == "PASS"
    assert result.tree_wire_status == "VALID_W1"
    assert result.minimum_clearance_m is not None
    assert result.tree_count >= 1

    summary = json.loads((run_root / "07_report" / "run_summary.json").read_text("utf-8"))
    required = {
        "COARSE_VOXEL_SIZE_M",
        "VEGETATION_FINE_VOXEL_SIZE_M",
        "RAW_POINTS",
        "COARSE_REPRESENTATIVES",
        "VEGETATION_RAW_REGION_POINTS",
        "VEGETATION_FINE_REPRESENTATIVES",
        "FINE_REDUCTION_RATIO",
        "WIRE_RAW_REGION_POINTS",
        "WIRE_FINE_POINTS_OR_REPRESENTATIVES",
        "WIRE_FINE_MODE",
        "V4_INPUT_REPRESENTATIVES",
        "V4_TREE_COUNT",
        "ASSIGNED_FULLRES_POINTS",
        "W1_POLYLINE_COUNT",
        "W1_DISTANCE_READY_SEGMENTS",
        "TREE_WIRE_STATUS",
    }
    missing = required - set(summary)
    assert not missing, f"summary missing fields: {sorted(missing)}"
    assert summary["COARSE_VOXEL_SIZE_M"] == 0.25
    assert summary["VEGETATION_FINE_VOXEL_SIZE_M"] == 0.10
    assert summary["TREE_WIRE_STATUS"] == "VALID_W1"
    assert summary["W1_DISTANCE_READY_SEGMENTS"] >= 1
    assert summary["V4_INPUT_REPRESENTATIVES"] > 0
    assert summary["ASSIGNED_FULLRES_POINTS"] > 0
    assert summary["WIRE_FINE_MODE"] in {"RAW_REGION", "FINE_REPRESENTATIVES"}

    # Every owner-mandated diagnostic is present.
    diagnostics = {
        "01": run_root / "01_preprocess" / "coarse_preprocess_review.html",
        "02": run_root / "04_vegetation" / "fine_vegetation_input_review.html",
        "03": run_root / "04_vegetation" / "raw_vs_fine_vegetation_overlay.html",
        "04": run_root / "03_wire" / "wire_candidate_raw_review.html",
        "05": run_root / "03_wire" / "w1_reconstruction_review.html",
        "06": run_root / "05_v4" / "v4_tree_segmentation_clean.html",
        "07": run_root / "05_v4" / "v4_tree_segmentation_review.html",
        "08": run_root / "05_v4" / "v4_fullres_tree_preview.html",
        "09": run_root / "06_clearance" / "tree_wire_clearance_review.html",
        "10": run_root / "07_report" / "RUN_SUMMARY.html",
    }
    for label, path in diagnostics.items():
        assert path.is_file(), f"missing diagnostic {label}: {path}"

    # The clearance review is a rendered Plotly page, not the blocked stub.
    clearance_html = (run_root / "06_clearance" / "tree_wire_clearance_review.html").read_text(
        "utf-8"
    )
    assert "BLOCKED_NO_VALID_W1" not in clearance_html


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pipeline_fails_closed_for_a_scene_without_wires(tmp_path: Path) -> None:
    source = tmp_path / "scene_no_wire.las"
    rng = np.random.default_rng(3)
    gx, gy = np.meshgrid(np.arange(0, 40.5, 0.4), np.arange(-5, 5.5, 0.4))
    ground = np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, 0.0)))
    trunk = np.column_stack(
        (np.full(30, 10.0), rng.uniform(-0.15, 0.15, 30), np.linspace(0, 3.0, 30))
    )
    crown = rng.normal((10.0, 0.0, 4.0), (2.0, 2.0, 2.0), (800, 3))
    crown[:, 2] = np.clip(crown[:, 2], 0.5, None)
    xyz = np.concatenate((ground, trunk, crown), axis=0)
    classification = np.concatenate(
        (
            np.full(len(ground), 2, np.uint8),
            np.full(len(trunk) + len(crown), 5, np.uint8),
        )
    )
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray((0.001, 0.001, 0.001))
    header.offsets = np.zeros(3, dtype=np.float64)
    data = laspy.LasData(header)
    data.x, data.y, data.z = xyz[order].T
    data.classification = classification[order]
    data.write(source)

    result = run_new_las_demo(source, tmp_path / "run", progress=None)
    assert result.overall_status == "PASS_WITH_TREE_WIRE_BLOCKED"
    assert result.tree_wire_status == "BLOCKED_NO_VALID_W1"
    assert result.minimum_clearance_m is None
    summary = json.loads(
        (tmp_path / "run" / "07_report" / "run_summary.json").read_text("utf-8")
    )
    assert summary["TREE_WIRE_STATUS"] == "BLOCKED_NO_VALID_W1"
    assert summary["GLOBAL_MINIMUM_CLEARANCE"] is None
    clearance_json = json.loads(
        (tmp_path / "run" / "06_clearance" / "tree_wire_clearance.json").read_text("utf-8")
    )
    assert clearance_json["TREE_WIRE_STATUS"] == "BLOCKED_NO_VALID_W1"
