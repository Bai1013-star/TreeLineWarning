from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from core.stage_b.tree_metrics import compute_stage_b_tree_metrics


def _write_preparation(tmp_path, ground: np.ndarray) -> str:
    shared = tmp_path / "artifacts" / "project-runs" / "source-shared" / "fixture"
    shared.mkdir(parents=True)
    ground_path = shared / "coarse-ground.npy"
    np.save(ground_path, ground)
    preparation = shared / "preparation.json"
    preparation.write_text(
        json.dumps(
            {
                "schema": "treeline-project-source-shared-preparation-v1",
                "authority": "FULL_RESOLUTION_RAW",
                "tower_membership_used_for_vegetation": False,
                "source_point_count": 123,
                "coarse_ground_reference": ground_path.relative_to(tmp_path).as_posix(),
                "origin_xyz": [0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    return preparation.relative_to(tmp_path).as_posix()


def _tree_points() -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    stem = np.column_stack(
        (
            0.15 * np.cos(angle),
            0.13 * np.sin(angle),
            np.full(len(angle), 1.30),
        )
    )
    crown = np.asarray(
        [
            [-1.2, -0.6, 3.5],
            [-1.0, 0.7, 4.0],
            [-0.5, -0.8, 4.4],
            [0.0, 0.0, 5.0],
            [0.6, 0.9, 4.2],
            [1.1, -0.7, 3.8],
            [1.3, 0.5, 3.6],
            [0.4, -1.0, 4.1],
            [-0.3, 1.0, 4.0],
            [0.8, 0.2, 4.6],
        ],
        dtype=np.float64,
    )
    return np.vstack((stem, crown))


def test_stage_b_metrics_reuse_fullres_ground_bound_implementation(tmp_path) -> None:
    ground = np.asarray(
        [
            [-2.0, -2.0, 0.0],
            [-2.0, 0.0, 0.0],
            [-2.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, -2.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    reference = _write_preparation(tmp_path, ground)
    xyz = _tree_points()
    reconciled = SimpleNamespace(
        final_xyz_local_m=xyz,
        final_tree_id=np.ones(len(xyz), dtype=np.int32),
    )
    result = SimpleNamespace(
        scan_id="scan-fixture",
        telemetry=SimpleNamespace(source_point_count=123),
        batch_diagnostics={"source_preparation_reference": reference},
    )

    metrics = compute_stage_b_tree_metrics(
        result,
        reconciled,
        tmp_path / "exports" / "Prediction_Input.xlsx",
        project_root=tmp_path,
    )

    assert set(metrics) == {1}
    metric = metrics[1]
    assert metric.tree_height_m is not None and metric.tree_height_m > 4.0
    assert metric.crown_major_width_m > metric.crown_minor_width_m > 0.2
    assert metric.crown_mean_width_m > 0.2
    assert metric.crown_area_m2 > 0.0
    assert metric.dbh.dbh_m is not None
    assert metric.dbh.dbh_quality == "MEASURED"
    assert 0.20 < metric.dbh.dbh_m < 0.40


def test_stage_b_metrics_fail_closed_when_ground_context_is_invalid(tmp_path) -> None:
    shared = tmp_path / "artifacts" / "project-runs" / "source-shared" / "bad"
    shared.mkdir(parents=True)
    preparation = shared / "preparation.json"
    preparation.write_text(
        json.dumps(
            {
                "authority": "DISPLAY_SAMPLE_ONLY",
                "tower_membership_used_for_vegetation": False,
                "source_point_count": 123,
                "coarse_ground_reference": "missing.npy",
                "origin_xyz": [0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    xyz = _tree_points()
    result = SimpleNamespace(
        scan_id="scan-fixture",
        telemetry=SimpleNamespace(source_point_count=123),
        batch_diagnostics={
            "source_preparation_reference": preparation.relative_to(tmp_path).as_posix()
        },
    )
    reconciled = SimpleNamespace(
        final_xyz_local_m=xyz,
        final_tree_id=np.ones(len(xyz), dtype=np.int32),
    )

    metrics = compute_stage_b_tree_metrics(
        result,
        reconciled,
        tmp_path / "exports" / "Prediction_Input.xlsx",
        project_root=tmp_path,
    )

    assert metrics == {}
