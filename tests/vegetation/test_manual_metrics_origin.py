"""Regression tests for manual-span structural metric coordinate frames."""

from __future__ import annotations

import numpy as np
import pytest

from algorithms.manual_span.config import default_manual_span_config
from algorithms.manual_span.metrics import compute_tree_metrics
from algorithms.spans.reconcile import GlobalTreeRecord, TreeReconciliationResult


def _fake_reconciliation(points: np.ndarray) -> TreeReconciliationResult:
    count = int(len(points))
    record = GlobalTreeRecord(
        global_tree_id=0,
        line_id=0,
        span_id=0,
        span_name="ManualLine-Span001",
        external_name="T1",
        primary_span_tree_id=0,
        apex_xy_local_m=(10.0, 10.0),
        apex_height_m=8.0,
        full_resolution_point_count=count,
        core_point_count=count,
        halo_point_count=0,
        core_fraction=1.0,
        review=False,
        merged_span_tree_keys=((0, 0),),
    )
    return TreeReconciliationResult(
        global_tree_records=(record,),
        final_original_index=np.arange(count, dtype=np.int64),
        final_xyz_local_m=np.asarray(points, dtype=np.float64),
        final_tree_id=np.zeros(count, dtype=np.int32),
        review_tree_ids=(),
        span_tree_to_global={(0, 0): 0},
        diagnostics={"schema": "metric-origin-regression"},
    )


def _tree_points() -> np.ndarray:
    rng = np.random.default_rng(20260827)
    trunk = np.column_stack(
        (
            10.0 + rng.normal(0.0, 0.08, 240),
            10.0 + rng.normal(0.0, 0.08, 240),
            np.linspace(0.1, 1.8, 240),
        )
    )
    canopy = np.column_stack(
        (
            10.0 + rng.normal(0.0, 1.4, 480),
            10.0 + rng.normal(0.0, 1.1, 480),
            rng.uniform(2.0, 8.0, 480),
        )
    )
    return np.concatenate((trunk, canopy), axis=0)


def _ground_patch(center_x: float, center_y: float, z: float) -> np.ndarray:
    offsets = np.linspace(-6.0, 6.0, 13)
    return np.asarray(
        [[center_x + dx, center_y + dy, z] for dx in offsets for dy in offsets],
        dtype=np.float64,
    )


def test_metrics_hag_uses_absolute_xy_when_origin_is_nonzero() -> None:
    """Local tree XY must be translated before querying absolute Ground XYZ."""

    points = _tree_points()
    reconciliation = _fake_reconciliation(points)
    manual = default_manual_span_config()

    reference_ground = _ground_patch(10.0, 10.0, 0.0)
    reference = compute_tree_metrics(
        reconciliation,
        reference_ground,
        (0.0, 0.0, 0.0),
        manual,
    )[0]

    origin = np.asarray((1000.0, 2000.0, 3000.0), dtype=np.float64)
    correct_absolute_ground = reference_ground + origin[None, :]

    # Deliberately place an unrelated terrain patch near the LOCAL tree XY.
    # A buggy query using local XY against the absolute-ground KD-tree will
    # select this patch and inflate HAG by roughly 15 metres.
    local_xy_distractor = _ground_patch(10.0, 10.0, 2985.0)
    shifted_ground = np.concatenate(
        (correct_absolute_ground, local_xy_distractor), axis=0
    )

    shifted = compute_tree_metrics(
        reconciliation,
        shifted_ground,
        tuple(float(value) for value in origin),
        manual,
    )[0]

    assert reference.tree_height_m is not None
    assert shifted.tree_height_m is not None
    assert 7.0 <= shifted.tree_height_m <= 8.2
    assert shifted.tree_height_m == pytest.approx(reference.tree_height_m, abs=1e-5)

    # Crown geometry is local-horizontal geometry and must remain invariant.
    assert shifted.crown_major_width_m == pytest.approx(
        reference.crown_major_width_m, abs=1e-10
    )
    assert shifted.crown_minor_width_m == pytest.approx(
        reference.crown_minor_width_m, abs=1e-10
    )
    assert shifted.crown_area_m2 == pytest.approx(reference.crown_area_m2, abs=1e-10)

    # DBH uses the HAG breast-height band, so its evidence classification must
    # also be identical after a pure source-frame translation.
    assert shifted.dbh.dbh_quality == reference.dbh.dbh_quality
    if reference.dbh.dbh_m is None:
        assert shifted.dbh.dbh_m is None
    else:
        assert shifted.dbh.dbh_m == pytest.approx(reference.dbh.dbh_m, abs=1e-10)
