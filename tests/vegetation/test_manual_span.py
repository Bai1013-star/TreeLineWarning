"""Tests for the manual single-span route: pole refinement, span, metrics, risk."""

from __future__ import annotations

import numpy as np
import pytest
from algorithms.manual_span.clearance_fast import ManualClearanceResult
from algorithms.manual_span.config import (
    ManualSpanConfig,
    default_manual_span_config,
)
from algorithms.manual_span.dbh import DBH_MEASURED, DBH_REVIEW, DBH_UNAVAILABLE
from algorithms.manual_span.metrics import compute_tree_metrics
from algorithms.manual_span.risk import apply_risk_threshold
from algorithms.manual_span.span import build_manual_span_record, refine_pole_from_click
from algorithms.poles.evidence import GroundEvidence
from algorithms.spans.clearance import (
    SpanClearanceResult,
    SpanTreeWireRecord,
)
from algorithms.spans.reconcile import GlobalTreeRecord, TreeReconciliationResult
from core.config.fast_baseline import default_fast_baseline_config


def _make_evidence(
    pole_points: np.ndarray,
    ground_z: float = 0.0,
    ground_centers: list[tuple[float, float]] | None = None,
) -> GroundEvidence:
    """Build a synthetic GroundEvidence around a pole-like point column."""

    xyz = pole_points.copy()
    n = len(xyz)
    hag = xyz[:, 2].astype(np.float32) - ground_z
    classification = np.ones(n, dtype=np.uint8)
    verticality = np.full(n, 0.9, dtype=np.float32)
    linearity = np.full(n, 0.6, dtype=np.float32)
    planarity = np.full(n, 0.1, dtype=np.float32)
    scattering = np.full(n, 0.2, dtype=np.float32)
    main_direction_z = np.full(n, 0.95, dtype=np.float32)
    # Add ground support points near each pole centre.
    centres = ground_centers or [(5.0, 5.0)]
    ground_rows = []
    for (gx, gy) in centres:
        for dx in (-1.0, 0.0, 1.0, -0.5, 0.5):
            for dy in (-1.0, 0.0, 1.0):
                ground_rows.append([gx + dx, gy + dy, ground_z])
    ground = np.asarray(ground_rows, dtype=np.float64)
    xyz = np.concatenate((xyz, ground), axis=0)
    hag = np.concatenate((hag, np.zeros(len(ground), dtype=np.float32)))
    classification = np.concatenate(
        (classification, np.full(len(ground), 2, dtype=np.uint8))
    )
    verticality = np.concatenate((verticality, np.zeros(len(ground), dtype=np.float32)))
    linearity = np.concatenate((linearity, np.zeros(len(ground), dtype=np.float32)))
    planarity = np.concatenate((planarity, np.zeros(len(ground), dtype=np.float32)))
    scattering = np.concatenate((scattering, np.zeros(len(ground), dtype=np.float32)))
    main_direction_z = np.concatenate((main_direction_z, np.zeros(len(ground), dtype=np.float32)))
    total = len(xyz)
    return GroundEvidence(
        xyz_absolute_m=xyz + np.asarray((1000.0, 2000.0, 3000.0)),
        xyz_local_m=xyz,
        hag_m=hag,
        classification=classification,
        representative_id=np.arange(total, dtype=np.uint32),
        source_original_index=np.arange(total, dtype=np.int64),
        raw_count=np.ones(total, dtype=np.uint32),
        linearity=linearity,
        planarity=planarity,
        scattering=scattering,
        verticality=verticality,
        main_direction_z=main_direction_z,
        count=total,
        origin_xyz=(1000.0, 2000.0, 3000.0),
        ground_class_2_count=int(np.count_nonzero(classification == 2)),
        non_ground_count=int(np.count_nonzero(classification != 2)),
        feature_runtime_seconds=0.0,
    )


def _pole_column(
    base_xy: tuple[float, float], height: float = 9.0, footprint: float = 0.3
) -> np.ndarray:
    rng = np.random.default_rng(5)
    points = []
    for z in np.linspace(0.1, height, 40):
        xy = base_xy + rng.normal(0, footprint / 3.0, size=2)
        points.append([xy[0], xy[1], z])
    return np.asarray(points, dtype=np.float64)


def _tree_blob(base_xy: tuple[float, float]) -> np.ndarray:
    rng = np.random.default_rng(9)
    points = []
    # A wide canopy > refine_max_footprint.
    for z in np.linspace(0.5, 7.0, 30):
        xy = base_xy + rng.normal(0, 1.6, size=2)
        points.append([xy[0], xy[1], z])
    return np.asarray(points, dtype=np.float64)


def test_refine_defensible_pole() -> None:
    evidence = _make_evidence(_pole_column((5.0, 5.0)))
    cfg = default_fast_baseline_config()
    manual = default_manual_span_config()
    pole = refine_pole_from_click("A", (1005.0, 2005.0, 3004.0), evidence, cfg, manual)
    assert pole.defensible, pole.reason
    assert pole.label == "A"
    assert 6.0 <= pole.height_m <= 10.0
    assert abs(pole.base_xyz_local_m[0] - 5.0) < 1.0
    assert abs(pole.base_xyz_local_m[1] - 5.0) < 1.0
    assert pole.xy_footprint_m < manual.refine_max_footprint_m


def test_refine_rejects_tree_blob_as_pole() -> None:
    evidence = _make_evidence(_tree_blob((5.0, 5.0)))
    cfg = default_fast_baseline_config()
    manual = default_manual_span_config()
    pole = refine_pole_from_click("B", (1005.0, 2005.0, 3004.0), evidence, cfg, manual)
    assert not pole.defensible
    assert pole.reason is not None


def test_refine_rejects_short_object() -> None:
    evidence = _make_evidence(_pole_column((5.0, 5.0), height=2.0))
    cfg = default_fast_baseline_config()
    manual = default_manual_span_config()
    pole = refine_pole_from_click("A", (1005.0, 2005.0, 3003.0), evidence, cfg, manual)
    assert not pole.defensible


def test_build_manual_span() -> None:
    cfg = default_fast_baseline_config()
    manual = default_manual_span_config()
    both = np.concatenate((_pole_column((5.0, 5.0)), _pole_column((85.0, 5.0))), axis=0)
    evidence = _make_evidence(both, ground_centers=[(5.0, 5.0), (85.0, 5.0)])
    pole_a = refine_pole_from_click("A", (1005.0, 2005.0, 3004.0), evidence, cfg, manual)
    pole_b = refine_pole_from_click(
        "B", (1085.0, 2005.0, 3004.0), evidence, cfg, manual
    )
    assert pole_a.defensible and pole_b.defensible
    span = build_manual_span_record(pole_a, pole_b, cfg, manual, (1000.0, 2000.0, 3000.0))
    assert span.length_m == pytest.approx(80.0, abs=2.0)
    assert span.direction_xy[0] == pytest.approx(1.0, abs=0.05)
    # Halo bounds must be wider than core bounds in XY.
    core_area = (
        (span.core_bounds_max_xyz[0] - span.core_bounds_min_xyz[0])
        * (span.core_bounds_max_xyz[1] - span.core_bounds_min_xyz[1])
    )
    halo_area = (
        (span.halo_bounds_max_xyz[0] - span.halo_bounds_min_xyz[0])
        * (span.halo_bounds_max_xyz[1] - span.halo_bounds_min_xyz[1])
    )
    assert halo_area > core_area


def test_build_manual_span_too_short() -> None:
    cfg = default_fast_baseline_config()
    manual = default_manual_span_config()
    evidence = _make_evidence(_pole_column((5.0, 5.0)))
    pole_a = refine_pole_from_click("A", (1005.0, 2005.0, 3004.0), evidence, cfg, manual)
    pole_b = refine_pole_from_click("B", (1007.0, 2005.0, 3004.0), evidence, cfg, manual)
    with pytest.raises(ValueError):
        build_manual_span_record(pole_a, pole_b, cfg, manual, (1000.0, 2000.0, 3000.0))


def _fake_reconciliation(points: np.ndarray, tree_ids: np.ndarray) -> TreeReconciliationResult:
    ids = sorted(set(int(value) for value in tree_ids))
    records = []
    for tree_id in ids:
        count = int(np.count_nonzero(tree_ids == tree_id))
        records.append(
            GlobalTreeRecord(
                global_tree_id=tree_id,
                line_id=0,
                span_id=0,
                span_name="ManualLine-Span001",
                external_name=f"T{tree_id + 1}",
                primary_span_tree_id=tree_id,
                apex_xy_local_m=(0.0, 0.0),
                apex_height_m=8.0,
                full_resolution_point_count=count,
                core_point_count=count,
                halo_point_count=0,
                core_fraction=1.0,
                review=False,
                merged_span_tree_keys=((0, tree_id),),
            )
        )
    return TreeReconciliationResult(
        global_tree_records=tuple(records),
        final_original_index=np.arange(len(points), dtype=np.int64),
        final_xyz_local_m=np.asarray(points, dtype=np.float64),
        final_tree_id=np.asarray(tree_ids, dtype=np.int32),
        review_tree_ids=(),
        span_tree_to_global={(0, tree_id): tree_id for tree_id in ids},
        diagnostics={"schema": "test"},
    )


def test_metrics_height_crown_dbh() -> None:
    rng = np.random.default_rng(21)
    # One tree: a trunk column + canopy.
    trunk = np.column_stack(
        (
            10.0 + rng.normal(0, 0.1, 200),
            10.0 + rng.normal(0, 0.1, 200),
            np.linspace(0.1, 1.7, 200),
        )
    )
    canopy_z = np.random.uniform(2.0, 8.0, 400)
    canopy = np.column_stack(
        (
            10.0 + rng.normal(0, 1.5, 400),
            10.0 + rng.normal(0, 1.5, 400),
            canopy_z,
        )
    )
    points = np.concatenate((trunk, canopy), axis=0)
    reconciliation = _fake_reconciliation(points, np.zeros(len(points), dtype=np.int32))
    ground = np.asarray(
        [[x, y, 0.0] for x in np.linspace(-5, 25, 12) for y in np.linspace(-5, 25, 12)],
        dtype=np.float64,
    )
    manual = default_manual_span_config()
    metrics = compute_tree_metrics(reconciliation, ground, (0.0, 0.0, 0.0), manual)
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.tree_height_m is not None
    assert 7.0 <= metric.tree_height_m <= 8.2
    assert metric.crown_major_width_m > 1.0
    assert metric.crown_area_m2 > 0.5
    assert metric.dbh.dbh_quality in {DBH_MEASURED, DBH_REVIEW, DBH_UNAVAILABLE}


def test_apply_risk_threshold_counts() -> None:
    # Two trees: tree 0 dangerous (near wire), tree 1 safe (far).
    points = np.asarray(
        [
            [10.0, 0.0, 8.0],
            [10.0, 0.2, 8.1],
            [10.0, -0.2, 7.9],
            [30.0, 20.0, 5.0],
            [30.0, 20.2, 5.1],
        ],
        dtype=np.float64,
    )
    tree_ids = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
    reconciliation = _fake_reconciliation(points, tree_ids)
    records = (
        SpanTreeWireRecord(
            global_tree_id=0,
            external_name="T1",
            line_id=0,
            span_id=0,
            span_name="ManualLine-Span001",
            nearest_wire_id=0,
            minimum_clearance_m=1.0,
            nearest_tree_original_index=0,
            nearest_tree_xyz_local_m=(10.0, 0.0, 8.0),
            nearest_wire_xyz_local_m=(10.0, 0.0, 9.0),
            tree_point_count=3,
            evaluated_edge_count=10,
            status="VALID",
        ),
        SpanTreeWireRecord(
            global_tree_id=1,
            external_name="T2",
            line_id=0,
            span_id=0,
            span_name="ManualLine-Span001",
            nearest_wire_id=0,
            minimum_clearance_m=12.0,
            nearest_tree_original_index=3,
            nearest_tree_xyz_local_m=(30.0, 20.0, 5.0),
            nearest_wire_xyz_local_m=(30.0, 20.0, 9.0),
            tree_point_count=2,
            evaluated_edge_count=10,
            status="VALID",
        ),
    )
    clearance = SpanClearanceResult(
        records=records,
        status="VALID",
        global_minimum_clearance_m=1.0,
        edge_count=10,
        diagnostics={},
    )
    manual_clearance = ManualClearanceResult(
        clearance=clearance,
        danger_original_index=np.asarray([0, 1, 2], dtype=np.int64),
        danger_xyz_local_m=np.asarray(points[:3], dtype=np.float32),
        danger_distance_m=np.asarray([1.0, 1.1, 0.9], dtype=np.float64),
        danger_wire_id=np.zeros(3, dtype=np.int32),
        danger_segment_index=np.zeros(3, dtype=np.int32),
        danger_edge_index=np.zeros(3, dtype=np.int32),
        danger_tree_id=np.zeros(3, dtype=np.int32),
        danger_nearest_wire_xyz_local_m=np.tile(
            np.asarray([10.0, 0.0, 9.0], dtype=np.float32), (3, 1)
        ),
    )
    metrics_by_tree = {}
    risk = apply_risk_threshold(
        manual_clearance, metrics_by_tree, reconciliation, (0.0, 0.0, 0.0), 3.5
    )
    assert risk.dangerous_tree_count == 1
    assert risk.danger_point_count == 3
    assert risk.records[0].dangerous is True
    assert risk.records[1].dangerous is False
    # Horizontal and vertical clearance must refer to the SAME pair as 3D.
    record = risk.records[0]
    assert record.clearance_3d_m == pytest.approx(1.0)
    assert record.horizontal_clearance_m == pytest.approx(0.0, abs=1e-9)
    assert record.vertical_clearance_m == pytest.approx(1.0)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        default_manual_span_config(refine_search_radius_m=-1.0)
    with pytest.raises(ValueError):
        default_manual_span_config(display_sample_points=0)
    with pytest.raises(ValueError):
        ManualSpanConfig(span_half_width_m=-0.5)
