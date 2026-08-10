"""Synthetic tests for reconciliation and exact Tree-Wire clearance (44, 46)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from algorithms.poles.topology import SpanRecord
from algorithms.spans.clearance import run_span_tree_wire_clearance
from algorithms.spans.reconcile import reconcile_trees
from algorithms.spans.tree_runner import SpanTreeOutcome
from algorithms.spans.wire import SpanWireResult
from algorithms.vegetation.tree_wire_clearance import (
    SegmentEdgeBvh,
    build_wire_edge_bundle,
    exact_point_to_segment_distance,
)
from core.config.fast_baseline import default_fast_baseline_config

_CFG = default_fast_baseline_config()


def _span(span_id: int, start_x: float, end_x: float) -> SpanRecord:
    return SpanRecord(
        line_id=0,
        line_name="Line01",
        span_id=span_id,
        span_name=f"Line01-Span{span_id + 1:03d}",
        start_pole_candidate_id=span_id,
        end_pole_candidate_id=span_id + 1,
        start_xyz_absolute_m=(start_x, -50.0, 0.0),
        end_xyz_absolute_m=(end_x, -50.0, 0.0),
        start_pole_top_z_m=15.0,
        end_pole_top_z_m=15.0,
        length_m=end_x - start_x,
        direction_xy=(1.0, 0.0, 0.0),
        half_width_m=9.0,
        halo_m=6.0,
        core_bounds_min_xyz=(start_x, -59.0, 0.0),
        core_bounds_max_xyz=(end_x, -41.0, 20.0),
        halo_bounds_min_xyz=(start_x - 6.0, -59.0, 0.0),
        halo_bounds_max_xyz=(end_x + 6.0, -41.0, 20.0),
        quality="CONFIRMED_SPAN",
    )


def _outcome(
    span: SpanRecord,
    tree_records: list[dict],
    original_index: np.ndarray,
    xyz: np.ndarray,
    tree_id: np.ndarray,
    in_core: np.ndarray,
) -> SpanTreeOutcome:
    return SpanTreeOutcome(
        span=span,
        tree_records=tuple(tree_records),
        raw_original_index=original_index,
        raw_xyz_local_m=xyz,
        raw_tree_id=tree_id,
        in_core=in_core,
        raw_local_maxima=0,
        accepted_tree_tops=0,
        assigned_raw_point_count=int(len(original_index)),
        artifact_root=Path("."),
    )


def test_cross_span_duplicate_crown_merges_into_one_global_tree() -> None:
    span_a = _span(0, -20.0, 30.0)
    span_b = _span(1, 30.0, 80.0)
    # The same physical crown appears in both spans' halos near the shared pole.
    shared = np.arange(50, 100, dtype=np.int64)
    a_only = np.arange(0, 50, dtype=np.int64)
    b_only = np.arange(100, 150, dtype=np.int64)
    xyz_shared = np.column_stack(
        (np.full(len(shared), 30.0), np.full(len(shared), -50.0), np.linspace(2, 12, len(shared)))
    )
    outcome_a = _outcome(
        span_a,
        [{"tree_id": 0, "apex_xy_local_m": [30.0, -50.0, 12.0], "apex_height_m": 12.0}],
        np.concatenate((a_only, shared)),
        np.vstack(
            (
                np.column_stack(
                    (
                        np.full(len(a_only), 20.0),
                        np.full(len(a_only), -50.0),
                        np.linspace(2, 12, len(a_only)),
                    )
                ),
                xyz_shared,
            )
        ),
        np.concatenate((np.zeros(len(a_only)), np.zeros(len(shared)))).astype(np.int32),
        np.ones(len(a_only) + len(shared), dtype=bool),
    )
    outcome_b = _outcome(
        span_b,
        [{"tree_id": 0, "apex_xy_local_m": [30.0, -50.0, 12.0], "apex_height_m": 12.0}],
        np.concatenate((shared, b_only)),
        np.vstack(
            (
                xyz_shared,
                np.column_stack(
                    (
                        np.full(len(b_only), 40.0),
                        np.full(len(b_only), -50.0),
                        np.linspace(2, 12, len(b_only)),
                    )
                ),
            )
        ),
        np.concatenate((np.zeros(len(shared)), np.zeros(len(b_only)))).astype(np.int32),
        np.ones(len(shared) + len(b_only), dtype=bool),
    )
    result = reconcile_trees([outcome_a, outcome_b], _CFG, origin_xyz=(0.0, 0.0, 0.0))
    assert result.final_tree_count == 1
    # No raw point is duplicated into two final memberships.
    unique = len(np.unique(result.final_original_index))
    assert unique == len(result.final_original_index)


def test_no_proximity_only_merge_of_separate_crowns() -> None:
    span_a = _span(0, -20.0, 30.0)
    span_b = _span(1, 30.0, 80.0)
    outcome_a = _outcome(
        span_a,
        [{"tree_id": 0, "apex_xy_local_m": [5.0, -50.0, 10.0], "apex_height_m": 10.0}],
        np.arange(0, 100, dtype=np.int64),
        np.column_stack((np.full(100, 5.0), np.full(100, -50.0), np.linspace(2, 10, 100))),
        np.zeros(100, dtype=np.int32),
        np.ones(100, dtype=bool),
    )
    outcome_b = _outcome(
        span_b,
        [{"tree_id": 0, "apex_xy_local_m": [40.0, -50.0, 10.0], "apex_height_m": 10.0}],
        np.arange(200, 300, dtype=np.int64),
        np.column_stack((np.full(100, 40.0), np.full(100, -50.0), np.linspace(2, 10, 100))),
        np.zeros(100, dtype=np.int32),
        np.ones(100, dtype=bool),
    )
    result = reconcile_trees([outcome_a, outcome_b], _CFG, origin_xyz=(0.0, 0.0, 0.0))
    # No shared original_index -> two separate trees, never merged.
    assert result.final_tree_count == 2


def test_deterministic_global_tree_ids() -> None:
    span_a = _span(0, -20.0, 30.0)
    outcome_a = _outcome(
        span_a,
        [{"tree_id": 0, "apex_xy_local_m": [5.0, -50.0, 10.0], "apex_height_m": 10.0}],
        np.arange(0, 100, dtype=np.int64),
        np.column_stack((np.full(100, 5.0), np.full(100, -50.0), np.linspace(2, 10, 100))),
        np.zeros(100, dtype=np.int32),
        np.ones(100, dtype=bool),
    )
    first = reconcile_trees([outcome_a], _CFG, origin_xyz=(0.0, 0.0, 0.0))
    second = reconcile_trees([outcome_a], _CFG, origin_xyz=(0.0, 0.0, 0.0))
    assert [record.external_name for record in first.global_tree_records] == [
        record.external_name for record in second.global_tree_records
    ]
    assert first.global_tree_records[0].external_name.startswith("Line01-Span001")


def _wire_arrays(curve: np.ndarray) -> dict:
    return {
        "wire_id": np.asarray([0], dtype=np.int32),
        "wire_status": np.asarray([0], dtype=np.uint8),
        "wire_method": np.asarray([1], dtype=np.uint8),
        "wire_quality_flags": np.zeros(1, dtype=np.uint16),
        "wire_has_distance_ready_segments": np.ones(1, dtype=np.bool_),
        "wire_continuous_distance_ready": np.ones(1, dtype=np.bool_),
        "wire_segment_offsets": np.asarray([0, 1], dtype=np.int64),
        "segment_wire_id": np.asarray([0], dtype=np.int32),
        "segment_index": np.asarray([0], dtype=np.int32),
        "segment_method": np.asarray([1], dtype=np.uint8),
        "segment_quality_flags": np.zeros(1, dtype=np.uint16),
        "segment_distance_ready": np.ones(1, dtype=np.bool_),
        "segment_parameter_domain": np.asarray([[0.0, 1.0]], dtype=np.float64),
        "sampled_curve_xyz": np.asarray(curve, dtype=np.float32),
        "sampled_curve_parameter": np.linspace(0, 1, len(curve), dtype=np.float32),
        "sampled_curve_offsets": np.asarray([0, len(curve)], dtype=np.int64),
    }


def test_bvh_matches_brute_force_exact_distance() -> None:
    rng = np.random.default_rng(5)
    curve = np.column_stack((np.linspace(0, 40, 100), np.zeros(100), np.full(100, 10.0)))
    arrays = _wire_arrays(curve)
    bundle = build_wire_edge_bundle(arrays)
    bvh = SegmentEdgeBvh(bundle, _CFG.clearance_bvh_leaf_cap)
    for _ in range(20):
        point = np.asarray([rng.uniform(0, 40), rng.uniform(-5, 5), rng.uniform(5, 15)])
        edge_position, distance, q, t, _evaluated = bvh.nearest_edge(
            point, _CFG.clearance_tie_tolerance_m
        )
        # Brute force over every edge.
        best = None
        best_edge = -1
        for position in range(bundle.edge_count):
            a = bundle.a_xyz[position]
            b = bundle.b_xyz[position]
            d, _, _ = exact_point_to_segment_distance(point, a, b)
            if best is None or d < best - _CFG.clearance_tie_tolerance_m:
                best = d
                best_edge = position
        assert edge_position == best_edge
        assert abs(distance - best) < 1e-9


def test_clearance_status_fails_closed_without_wires() -> None:
    span_a = _span(0, -20.0, 30.0)
    outcome = _outcome(
        span_a,
        [{"tree_id": 0, "apex_xy_local_m": [5.0, -50.0, 10.0], "apex_height_m": 10.0}],
        np.arange(0, 100, dtype=np.int64),
        np.column_stack((np.full(100, 5.0), np.full(100, -50.0), np.linspace(2, 10, 100))),
        np.zeros(100, dtype=np.int32),
        np.ones(100, dtype=bool),
    )
    reconciled = reconcile_trees([outcome], _CFG, origin_xyz=(0.0, 0.0, 0.0))
    empty_wire = SpanWireResult(
        strands=(),
        w1_arrays=_wire_arrays(np.empty((0, 3))),
        candidate_mask=np.empty(0, dtype=bool),
        cluster_id_by_rep=np.empty(0, dtype=np.int32),
        diagnostics={},
    )
    result = run_span_tree_wire_clearance(reconciled, [empty_wire], _CFG)
    assert result.status == "NO_DISTANCE_READY_EDGE"
    assert result.global_minimum_clearance_m is None
