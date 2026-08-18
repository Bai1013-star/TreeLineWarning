"""Exact, streaming candidate-clearance regression tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from algorithms.spans.clearance import (
    ClearanceCancelled,
    _candidate_exact_nearest_edges,
    run_span_tree_wire_clearance,
)
from algorithms.spans.wire import SpanWireResult
from algorithms.vegetation.tree_wire_clearance import SegmentEdgeBvh, WireEdgeBundle
from core.config.fast_baseline import FastBaselineConfig


def _bundle(edge_count: int = 307) -> WireEdgeBundle:
    y = np.arange(edge_count, dtype=np.float64)
    return WireEdgeBundle(
        a_xyz=np.column_stack((np.zeros(edge_count), y, np.zeros(edge_count))),
        b_xyz=np.column_stack((np.full(edge_count, 10.0), y, np.zeros(edge_count))),
        wire_id=np.arange(edge_count, dtype=np.int32),
        segment_index=np.zeros(edge_count, dtype=np.int32),
        polyline_edge_index=np.zeros(edge_count, dtype=np.int32),
        wire_status=np.zeros(edge_count, dtype=np.uint8),
        segment_quality_flags=np.zeros(edge_count, dtype=np.uint16),
        edge_count=edge_count,
    )


def test_candidate_backend_is_exact_above_legacy_direct_cap_and_chunk_invariant():
    rng = np.random.default_rng(20260818)
    edges = _bundle()
    points = rng.uniform((-5.0, -10.0, -3.0), (15.0, 320.0, 3.0), size=(513, 3))
    bvh = SegmentEdgeBvh(edges, leaf_cap=16)
    fast = _candidate_exact_nearest_edges(points, bvh, edges, 1e-9)
    workers_two = _candidate_exact_nearest_edges(points, bvh, edges, 1e-9, workers=2)
    reference = [bvh.nearest_edge(point, 1e-9) for point in points]
    np.testing.assert_array_equal(fast[0], [item[0] for item in reference])
    np.testing.assert_allclose(fast[1], [item[1] for item in reference], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(fast[2], [item[2] for item in reference], rtol=0.0, atol=1e-10)
    np.testing.assert_array_equal(workers_two[0], fast[0])
    np.testing.assert_allclose(workers_two[1], fast[1], rtol=0.0, atol=1e-12)
    assert fast[5]["fallback_points"] == 0


def _w1() -> SpanWireResult:
    return SpanWireResult(
        strands=(),
        candidate_mask=np.empty(0, dtype=bool),
        cluster_id_by_rep=np.empty(0, dtype=np.int32),
        diagnostics={},
        w1_arrays={
            "wire_id": np.asarray((0,), dtype=np.int32),
            "wire_status": np.asarray((0,), dtype=np.uint8),
            "wire_method": np.asarray((0,), dtype=np.uint8),
            "wire_quality_flags": np.asarray((0,), dtype=np.uint16),
            "wire_has_distance_ready_segments": np.asarray((True,)),
            "wire_continuous_distance_ready": np.asarray((True,)),
            "wire_segment_offsets": np.asarray((0, 1), dtype=np.int64),
            "segment_wire_id": np.asarray((0,), dtype=np.int32),
            "segment_index": np.asarray((0,), dtype=np.int32),
            "segment_method": np.asarray((0,), dtype=np.uint8),
            "segment_quality_flags": np.asarray((0,), dtype=np.uint16),
            "segment_distance_ready": np.asarray((True,)),
            "segment_parameter_domain": np.asarray(((0.0, 1.0),)),
            "sampled_curve_xyz": np.asarray(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))),
            "sampled_curve_parameter": np.asarray((0.0, 1.0)),
            "sampled_curve_offsets": np.asarray((0, 2), dtype=np.int64),
        },
    )


def _reconciled() -> SimpleNamespace:
    return SimpleNamespace(
        global_tree_records=(
            SimpleNamespace(
                global_tree_id=1,
                external_name="T001",
                line_id=1,
                span_id=1,
                span_name="S001",
                full_resolution_point_count=3,
                persistent_tree_id="tree-1",
            ),
        ),
        final_tree_id=np.asarray((1, 1, 1), dtype=np.int32),
        final_xyz_local_m=np.asarray(((2.0, 1.0, 0.0), (3.0, 2.0, 0.0), (4.0, 2.01, 0.0))),
        final_original_index=np.asarray((10, 11, 12), dtype=np.int64),
    )


def test_danger_points_are_exact_inclusive_and_streamed_with_progress():
    progress: list[dict] = []
    result = run_span_tree_wire_clearance(
        _reconciled(),
        [_w1()],
        FastBaselineConfig(),
        danger_threshold_m=2.0,
        chunk_size=1,
        progress_callback=progress.append,
    )
    assert [point.original_index for point in result.danger_points] == [10, 11]
    assert result.records[0].danger_point_count == 2
    assert result.diagnostics["full_per_point_clearance_stored"] is False
    assert progress[-1]["processed_points"] == 3
    assert progress[-1]["total_points"] == 3


def test_cancellation_is_checked_at_chunk_boundaries():
    with pytest.raises(ClearanceCancelled, match="CHUNK_BOUNDARY"):
        run_span_tree_wire_clearance(
            _reconciled(),
            [_w1()],
            FastBaselineConfig(),
            chunk_size=1,
            cancellation_requested=lambda: True,
        )
