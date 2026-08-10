"""End-to-end integration test for POLE_SPAN_MATURE_BASELINE on synthetic LAS.

Builds a small scene (ground + 3 poles + parallel conductors + trees) and runs
the complete pole-first span-based pipeline, asserting the corridor topology,
per-span trees, per-span wires and exact Tree-Wire clearance all close the
loop together.
"""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
from algorithms.vegetation.fast_baseline_pipeline import run_fast_baseline_pipeline


def _build_synthetic_scene() -> np.ndarray:
    rng = np.random.default_rng(1234)
    rows: list[np.ndarray] = []

    def add(points: np.ndarray, classification: int) -> None:
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        rows.append(
            np.column_stack((p, np.full(len(p), classification, dtype=np.uint8)))
        )

    ground = np.asarray(
        [[x, y, 0.0] for x in np.arange(-30, 150, 3.0) for y in np.arange(-90, 10, 3.0)]
    )
    add(ground + rng.normal(0, 0.02, ground.shape), 2)
    for x in (0.0, 50.0, 100.0):
        z = np.arange(0.1, 13.0, 0.3)
        add(np.column_stack((np.full(len(z), x), np.full(len(z), -50.0), z)), 5)
    for span in ((0.0, 50.0), (50.0, 100.0)):
        for u in (-1.0, 1.0):
            s = np.arange(span[0], span[1] + 0.01, 0.2)
            z = 11.0 - 0.3 * np.sin(np.pi * (s - span[0]) / 50.0)
            add(np.column_stack((s, np.full(len(s), -50.0 + u), z)), 5)
    for cx, u, height in ((15.0, -4.0, 10.0), (70.0, 3.0, 8.0)):
        for _ in range(600):
            r = 2.5 * np.sqrt(rng.uniform(0.0, 1.0))
            angle = rng.uniform(0.0, 2 * np.pi)
            x = cx + r * np.cos(angle)
            y = -50.0 + u + r * np.sin(angle)
            z = height * (1.0 - (r / 2.5) ** 2) * rng.uniform(0.2, 1.0)
            add(np.asarray([[x, y, max(1.5, z)]]), 5)
    clutter = np.column_stack(
        (
            rng.uniform(-20, 140, 400),
            rng.uniform(-90, 10, 400),
            rng.uniform(2, 8, 400),
        )
    )
    add(clutter, 5)
    return np.concatenate(rows, axis=0)


def _write_las(path: Path, rows: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.asarray((0.001, 0.001, 0.001))
    header.offsets = np.zeros(3, dtype=np.float64)
    data = laspy.LasData(header)
    data.x, data.y, data.z = rows[:, 0], rows[:, 1], rows[:, 2]
    data.classification = rows[:, 3].astype(np.uint8)
    data.write(path)


def test_full_pipeline_closes_the_loop(tmp_path: Path) -> None:
    source = tmp_path / "scene.las"
    _write_las(source, _build_synthetic_scene())
    run_root = tmp_path / "run"
    progress_events: list[int] = []

    def progress(stage: int, _message: str, _done: bool, _elapsed: float | None) -> None:
        progress_events.append(stage)

    summary = run_fast_baseline_pipeline(
        source, run_root, progress=progress, voxel_size_m=0.5
    )
    # Corridor topology: 3 poles on one line -> 2 spans.
    assert summary["POLE_CANDIDATES"] >= 3
    assert summary["CONFIRMED_POLES"] >= 3
    assert summary["LINE_COUNT"] == 1
    assert summary["SPAN_COUNT"] == 2
    # Trees: two synthetic crowns.
    assert summary["FINAL_TREE_COUNT"] >= 2
    # Wires: two spans x two strands.
    assert summary["WIRE_STRAND_COUNT"] >= 4
    # Exact Tree-Wire clearance must be VALID (fail-closed otherwise).
    assert summary["TREE_WIRE_STATUS"] == "VALID"
    assert summary["GLOBAL_MINIMUM_CLEARANCE"] is not None
    assert summary["GLOBAL_MINIMUM_CLEARANCE"] > 0.0
    # All eight user-visible stages ran.
    assert sorted(set(progress_events)) == list(range(1, 9))
    # Required review pages exist.
    required = {
        "01_pole_candidates_review.html",
        "02_confirmed_poles_review.html",
        "03_pole_topology_review.html",
        "04_span_partition_review.html",
        "05_chm_review.html",
        "06_tree_tops_review.html",
        "07_watershed_crowns_review.html",
        "08_tree_segmentation_fullres_preview.html",
        "09_wire_candidates_review.html",
        "10_wire_clusters_review.html",
        "11_wire_strands_review.html",
        "12_wire_reconstruction_review.html",
        "13_raw_vs_wire_overlay.html",
        "14_tree_wire_clearance_review.html",
        "15_minimum_clearance_review.html",
        "16_RUN_SUMMARY.html",
    }
    found = {path.name for path in run_root.rglob("*.html")}
    assert required.issubset(found)


def test_pipeline_fails_closed_without_poles(tmp_path: Path) -> None:
    # A scene with no poles must never fabricate spans.
    rng = np.random.default_rng(9)
    rows = np.column_stack(
        (
            rng.uniform(0, 60, 1500),
            rng.uniform(0, 60, 1500),
            rng.uniform(0, 15, 1500),
        )
    )
    source = tmp_path / "no_poles.las"
    _write_las(source, np.column_stack((rows, np.full(len(rows), 5, dtype=np.uint8))))
    summary = run_fast_baseline_pipeline(
        source, tmp_path / "run", voxel_size_m=0.5
    )
    assert summary["RUN_STATUS"] == "BLOCKED"
    assert summary["SPAN_PIPELINE"].startswith("SPAN_PIPELINE:")
    assert summary["SPAN_COUNT"] == 0
