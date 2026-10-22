from __future__ import annotations

import numpy as np

from algorithms.manual_span.config import default_manual_span_config
from algorithms.manual_span.metrics import compute_tree_metrics
from algorithms.spans.reconcile import GlobalTreeRecord, TreeReconciliationResult


def test_dbh_uses_raw_tree_lowest_plus_1p30_and_crown_is_available():
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    trunk = np.column_stack((0.15 * np.cos(angles), 0.15 * np.sin(angles), np.full(96, 101.30)))
    base = np.array([[0.0, 0.0, 100.0], [0.02, 0.01, 100.2], [-0.02, 0.01, 100.5]])
    crown = np.array(
        [[x, y, z] for x in (-2.0, -1.0, 1.0, 2.0) for y in (-1.5, 1.5) for z in (106.0, 108.0)]
    )
    xyz = np.vstack((base, trunk, crown))
    record = GlobalTreeRecord(
        global_tree_id=1, line_id=1, span_id=1, span_name="档距1", external_name="T-1-001",
        primary_span_tree_id=1, apex_xy_local_m=(0.0, 0.0, 8.0), apex_height_m=8.0,
        full_resolution_point_count=len(xyz), core_point_count=len(xyz), halo_point_count=0,
        core_fraction=1.0, review=False, merged_span_tree_keys=((1, 1),),
    )
    reconciled = TreeReconciliationResult(
        global_tree_records=(record,), final_original_index=np.arange(len(xyz), dtype=np.int64),
        final_xyz_local_m=xyz, final_tree_id=np.ones(len(xyz), dtype=np.int32),
        review_tree_ids=(), span_tree_to_global={(1, 1): 1}, diagnostics={},
    )
    # Deliberately wrong terrain Z: the result must follow the raw tree's own
    # lowest point (100.0), so the 101.30 m ring is the DBH slice.
    ground = np.array([[-5.0, -5.0, 95.0], [5.0, -5.0, 95.0], [-5.0, 5.0, 95.0], [5.0, 5.0, 95.0]])
    metric = compute_tree_metrics(reconciled, ground, (0.0, 0.0, 0.0), default_manual_span_config())[0]
    assert metric.tree_height_m == 8.0
    assert metric.dbh.dbh_m is not None
    assert metric.crown_major_width_m > 3.0
    assert metric.crown_minor_width_m > 2.0

