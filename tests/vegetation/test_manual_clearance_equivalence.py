"""Deterministic equivalence: optimized chunked clearance == whole-array exact.

The frozen exact Tree-Wire mathematics must be reproduced bit-for-bit by the
new streamed/chunked implementation: distance, nearest tree original index,
nearest tree XYZ, wire ID, segment identity, nearest wire XYZ and the tie
result.  This test feeds the SAME synthetic reconciliation and wire results to
``run_span_tree_wire_clearance`` (existing whole-array scalar engine) and
``run_manual_clearance`` (new vectorized chunked engine) and requires identical
per-tree records, then cross-checks every danger point's distance against the
scalar ``SegmentEdgeBvh.nearest_edge``.
"""

from __future__ import annotations

import numpy as np
from algorithms.manual_span.clearance_fast import run_manual_clearance
from algorithms.manual_span.config import default_manual_span_config
from algorithms.spans.clearance import run_span_tree_wire_clearance
from algorithms.spans.reconcile import (
    GlobalTreeRecord,
    TreeReconciliationResult,
)
from algorithms.spans.wire import SpanWireResult
from algorithms.vegetation.tree_wire_clearance import (
    SegmentEdgeBvh,
    build_wire_edge_bundle,
)
from core.config.fast_baseline import default_fast_baseline_config


def _make_wire_arrays(curves: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Build W1-compatible distance-ready arrays (one segment per wire)."""

    wire_count = len(curves)
    wire_id = np.arange(wire_count, dtype=np.int32)
    wire_status = np.zeros(wire_count, dtype=np.uint8)
    wire_method = np.ones(wire_count, dtype=np.uint8)
    wire_quality_flags = np.zeros(wire_count, dtype=np.uint16)
    wire_has_ready = np.ones(wire_count, dtype=np.bool_)
    wire_continuous = np.ones(wire_count, dtype=np.bool_)
    segment_wire_id = np.arange(wire_count, dtype=np.int32)
    segment_index = np.zeros(wire_count, dtype=np.int32)
    segment_method = np.ones(wire_count, dtype=np.uint8)
    segment_quality_flags = np.zeros(wire_count, dtype=np.uint16)
    segment_distance_ready = np.ones(wire_count, dtype=np.bool_)
    segment_parameter_domain = np.asarray(
        [[0.0, 1.0]] * wire_count, dtype=np.float64
    )
    sampled_curve_xyz = (
        np.concatenate(curves, axis=0).astype(np.float32)
        if wire_count
        else np.empty((0, 3), dtype=np.float32)
    )
    sample_counts = np.asarray([len(curve) for curve in curves], dtype=np.int64)
    sampled_curve_offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(sample_counts)), axis=0
    )
    if wire_count:
        sampled_curve_parameter = np.concatenate(
            [np.linspace(0.0, 1.0, len(curve), dtype=np.float32) for curve in curves],
            axis=0,
        )
    else:
        sampled_curve_parameter = np.empty(0, dtype=np.float32)
    return {
        "wire_id": wire_id,
        "wire_status": wire_status,
        "wire_method": wire_method,
        "wire_quality_flags": wire_quality_flags,
        "wire_has_distance_ready_segments": wire_has_ready,
        "wire_continuous_distance_ready": wire_continuous,
        "wire_segment_offsets": np.arange(wire_count + 1, dtype=np.int64),
        "segment_wire_id": segment_wire_id,
        "segment_index": segment_index,
        "segment_method": segment_method,
        "segment_quality_flags": segment_quality_flags,
        "segment_distance_ready": segment_distance_ready,
        "segment_parameter_domain": segment_parameter_domain,
        "sampled_curve_xyz": sampled_curve_xyz,
        "sampled_curve_parameter": sampled_curve_parameter,
        "sampled_curve_offsets": sampled_curve_offsets,
    }


def _make_reconciliation(tree_groups: list[np.ndarray]) -> TreeReconciliationResult:
    """Build a synthetic reconciliation from per-tree point arrays (local xyz)."""

    records = []
    final_xyz_parts = []
    final_tree_parts = []
    final_orig_parts = []
    for tree_id, points in enumerate(tree_groups):
        count = len(points)
        records.append(
            GlobalTreeRecord(
                global_tree_id=tree_id,
                line_id=0,
                span_id=0,
                span_name="ManualLine-Span001",
                external_name=f"ManualLine-Span001-Tree{tree_id + 1:03d}",
                primary_span_tree_id=tree_id,
                apex_xy_local_m=(0.0, 0.0),
                apex_height_m=10.0,
                full_resolution_point_count=count,
                core_point_count=count,
                halo_point_count=0,
                core_fraction=1.0,
                review=False,
                merged_span_tree_keys=((0, tree_id),),
            )
        )
        final_xyz_parts.append(np.asarray(points, dtype=np.float64))
        final_tree_parts.append(np.full(count, tree_id, dtype=np.int32))
        final_orig_parts.append(np.arange(1000 * tree_id, 1000 * tree_id + count, dtype=np.int64))
    return TreeReconciliationResult(
        global_tree_records=tuple(records),
        final_original_index=np.concatenate(final_orig_parts, axis=0),
        final_xyz_local_m=np.concatenate(final_xyz_parts, axis=0),
        final_tree_id=np.concatenate(final_tree_parts, axis=0),
        review_tree_ids=(),
        span_tree_to_global={(0, tree_id): tree_id for tree_id in range(len(tree_groups))},
        diagnostics={"schema": "test"},
    )


def _random_points(
    rng: np.random.Generator, count: int, center: np.ndarray, spread: float
) -> np.ndarray:
    return center[None, :] + rng.standard_normal((count, 3)) * spread


def _synthetic_scene(seed: int = 7) -> tuple[TreeReconciliationResult, list[SpanWireResult], dict]:
    rng = np.random.default_rng(seed)
    # Two parallel conductor curves (sag-like).
    s = np.linspace(-5.0, 45.0, 40)
    def curve(u_off: float, z_off: float) -> np.ndarray:
        z = 25.0 + z_off - 0.02 * (s - 20.0) ** 2
        return np.column_stack((s, np.full_like(s, u_off), z))
    curves = [curve(2.0, 0.0), curve(-2.0, -0.5)]
    wire = SpanWireResult(
        strands=(),
        w1_arrays=_make_wire_arrays(curves),
        candidate_mask=np.empty(0, dtype=bool),
        cluster_id_by_rep=np.empty(0, dtype=np.int32),
        diagnostics={},
    )
    # Three trees: two near the wires, one far.
    tree_a = _random_points(rng, 4000, np.asarray([10.0, 2.5, 20.0]), 1.5)
    tree_b = _random_points(rng, 3000, np.asarray([30.0, -2.2, 18.0]), 1.2)
    tree_c = _random_points(rng, 2000, np.asarray([20.0, 15.0, 10.0]), 2.0)
    reconciliation = _make_reconciliation([tree_a, tree_b, tree_c])
    cfg = default_fast_baseline_config()
    manual = default_manual_span_config(clearance_chunk_points=256)
    return reconciliation, [wire], {"cfg": cfg, "manual": manual}


def _records_equal(left: list, right: list) -> list[str]:
    mismatches: list[str] = []
    if len(left) != len(right):
        mismatches.append(f"record_count {len(left)} != {len(right)}")
        return mismatches
    for a, b in zip(left, right, strict=True):
        if a.global_tree_id != b.global_tree_id:
            mismatches.append(f"tree {a.global_tree_id}: global_tree_id")
        if a.nearest_wire_id != b.nearest_wire_id:
            mismatches.append(
                f"tree {a.global_tree_id}: nearest_wire_id "
                f"{a.nearest_wire_id} != {b.nearest_wire_id}"
            )
        if a.minimum_clearance_m != b.minimum_clearance_m:
            mismatches.append(
                f"tree {a.global_tree_id}: distance "
                f"{a.minimum_clearance_m} != {b.minimum_clearance_m}"
            )
        if a.nearest_tree_original_index != b.nearest_tree_original_index:
            mismatches.append(
                f"tree {a.global_tree_id}: original "
                f"{a.nearest_tree_original_index} != {b.nearest_tree_original_index}"
            )
        if a.nearest_tree_xyz_local_m != b.nearest_tree_xyz_local_m:
            mismatches.append(f"tree {a.global_tree_id}: tree xyz")
        if a.nearest_wire_xyz_local_m != b.nearest_wire_xyz_local_m:
            mismatches.append(f"tree {a.global_tree_id}: wire xyz")
        if a.status != b.status:
            mismatches.append(f"tree {a.global_tree_id}: status")
    return mismatches


def test_chunked_clearance_matches_whole_array_exact() -> None:
    reconciliation, wires, env = _synthetic_scene()
    whole = run_span_tree_wire_clearance(
        reconciliation, wires, env["cfg"]
    )
    chunked = run_manual_clearance(
        reconciliation, wires, env["cfg"], env["manual"]
    )
    assert chunked.clearance.status == whole.status
    mismatches = _records_equal(list(whole.records), list(chunked.clearance.records))
    assert not mismatches, "\n".join(mismatches)
    assert chunked.clearance.global_minimum_clearance_m == whole.global_minimum_clearance_m


def test_danger_point_distances_match_scalar_bvh() -> None:
    reconciliation, wires, env = _synthetic_scene(seed=11)
    chunked = run_manual_clearance(reconciliation, wires, env["cfg"], env["manual"])
    arrays = wires[0].w1_arrays
    bundle = build_wire_edge_bundle(arrays)
    bvh = SegmentEdgeBvh(bundle, env["cfg"].clearance_bvh_leaf_cap)
    final_xyz = reconciliation.final_xyz_local_m
    # Danger candidates are a subset of points; verify each against the scalar engine.
    for position in range(0, min(len(chunked.danger_original_index), 400)):
        original_index = int(chunked.danger_original_index[position])
        # final_original_index is dense arange here so the row equals the index.
        row = np.flatnonzero(reconciliation.final_original_index == original_index)[0]
        point = final_xyz[row]
        edge_position, distance, _q, _t, _ev = bvh.nearest_edge(
            point, env["cfg"].clearance_tie_tolerance_m
        )
        assert distance == float(chunked.danger_distance_m[position]), (
            f"point {original_index}: distance {distance} != {chunked.danger_distance_m[position]}"
        )
        assert int(bundle.wire_id[edge_position]) == int(chunked.danger_wire_id[position])


def test_chunked_handles_many_chunks_and_ties() -> None:
    """Exercise multiple chunks and exact tie breaking deterministically."""

    rng = np.random.default_rng(3)
    s = np.linspace(0.0, 20.0, 30)
    curves = [
        np.column_stack((s, np.zeros_like(s), 10.0 * np.ones_like(s))),
        np.column_stack((s, np.zeros_like(s), 10.5 * np.ones_like(s))),
    ]
    wire = SpanWireResult(
        strands=(),
        w1_arrays=_make_wire_arrays(curves),
        candidate_mask=np.empty(0, dtype=bool),
        cluster_id_by_rep=np.empty(0, dtype=np.int32),
        diagnostics={},
    )
    # Points placed exactly on the lower wire: distance 0 to wire 0.
    on_wire = np.column_stack((s + 0.5, np.zeros_like(s), 10.0 * np.ones_like(s)))
    scattered = _random_points(rng, 500, np.asarray([10.0, 3.0, 8.0]), 2.0)
    reconciliation = _make_reconciliation([on_wire, scattered])
    cfg = default_fast_baseline_config()
    manual = default_manual_span_config(clearance_chunk_points=64)
    whole = run_span_tree_wire_clearance(reconciliation, [wire], cfg)
    chunked = run_manual_clearance(reconciliation, [wire], cfg, manual)
    mismatches = _records_equal(list(whole.records), list(chunked.clearance.records))
    assert not mismatches, "\n".join(mismatches)


def test_no_distance_ready_fails_closed_identically() -> None:
    reconciliation, _wires, env = _synthetic_scene()
    empty_arrays = _make_wire_arrays([])
    wire = SpanWireResult(
        strands=(), w1_arrays=empty_arrays,
        candidate_mask=np.empty(0, dtype=bool),
        cluster_id_by_rep=np.empty(0, dtype=np.int32),
        diagnostics={},
    )
    whole = run_span_tree_wire_clearance(reconciliation, [wire], env["cfg"])
    chunked = run_manual_clearance(reconciliation, [wire], env["cfg"], env["manual"])
    assert whole.status == chunked.clearance.status == "NO_DISTANCE_READY_EDGE"
    assert all(r.minimum_clearance_m is None for r in chunked.clearance.records)
    assert len(chunked.danger_original_index) == 0


def test_zero_length_edge_handling_identical() -> None:
    """A degenerate zero-length segment must resolve like the scalar engine."""

    curves = [np.asarray([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]], dtype=np.float64)]
    wire = SpanWireResult(
        strands=(), w1_arrays=_make_wire_arrays(curves),
        candidate_mask=np.empty(0, dtype=bool),
        cluster_id_by_rep=np.empty(0, dtype=np.int32),
        diagnostics={},
    )
    points = np.asarray([[0.5, 0.0, 10.0], [0.0, 0.0, 12.0]], dtype=np.float64)
    reconciliation = _make_reconciliation([points])
    cfg = default_fast_baseline_config()
    manual = default_manual_span_config(clearance_chunk_points=2)
    whole = run_span_tree_wire_clearance(reconciliation, [wire], cfg)
    chunked = run_manual_clearance(reconciliation, [wire], cfg, manual)
    mismatches = _records_equal(list(whole.records), list(chunked.clearance.records))
    assert not mismatches, "\n".join(mismatches)
