"""Optimized full-resolution Tree-Wire clearance with exact equivalence.

The frozen exact mathematics (float64 point-to-segment formula, zero-length
handling, tie tolerance and lexicographic tie ordering, deterministic
segment-AABB BVH) lives in ``algorithms.vegetation.tree_wire_clearance`` and
is reused unchanged.  This module only changes the *execution strategy*:
instead of one scalar Python BVH traversal per tree point, it traverses the
same BVH with vectorized point masks (one NumPy call per visited node) and
streamed per-tree point chunks.  Every visited node and every refined leaf
edge is decided by the *same* pruning rule and the *same* ``candidate_better``
winner rule as the scalar engine, so the per-tree winners are bit-identical.

The pointwise nearest-wire distance is a by-product: every full-resolution raw
tree point whose exact distance to the nearest valid wire is within
``danger_point_cap_m`` is persisted as a danger-point candidate.  Risk
classification later filters these by the teacher-entered threshold without
re-running Ground / trees / wires / membership.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

from algorithms.manual_span.config import MANUAL_CLEARANCE_SCHEMA, ManualSpanConfig
from algorithms.spans.clearance import (
    NO_DISTANCE_READY_EDGE,
    SpanClearanceResult,
    SpanTreeWireRecord,
    _concatenate_span_wire_edges,
)
from algorithms.spans.reconcile import TreeReconciliationResult
from algorithms.spans.wire import SpanWireResult
from algorithms.vegetation.tree_wire_clearance import (
    SegmentEdgeBvh,
    WireEdgeBundle,
    build_wire_edge_bundle,
)


def _vectorized_nearest_edges(
    points: np.ndarray,
    edges: WireEdgeBundle,
    bvh: SegmentEdgeBvh,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact nearest edge per point, vectorized with the same pruning/winner rules.

    Returns ``(distance, wire_id, segment_index, edge_index, q, evaluated)``
    where *evaluated* is the total number of point-edge exact computations.
    Results are bit-identical to ``SegmentEdgeBvh.nearest_edge`` per point.
    """

    count = len(points)
    best_dist = np.full(count, np.inf, dtype=np.float64)
    best_wire = np.zeros(count, dtype=np.int32)
    best_seg = np.zeros(count, dtype=np.int32)
    best_edge = np.zeros(count, dtype=np.int32)
    best_q = np.zeros((count, 3), dtype=np.float64)
    evaluated = 0
    stack: list[tuple[Any, np.ndarray]] = [(bvh.root, np.arange(count))]
    while stack:
        node, idx = stack.pop()
        if len(idx) == 0:
            continue
        active = points[idx]
        mn = node.min_corner
        mx = node.max_corner
        delta = np.maximum(np.maximum(mn - active, 0.0), np.maximum(active - mx, 0.0))
        bound = np.linalg.norm(delta, axis=1)
        current = best_dist[idx]
        keep = bound <= current + tolerance
        if not keep.any():
            continue
        idx = idx[keep]
        if node.leaf:
            order = bvh.order[node.lo:node.hi]
            if len(order):
                evaluated += len(idx) * len(order)
            for edge_position in order:
                ep = int(edge_position)
                a = edges.a_xyz[ep]
                b = edges.b_xyz[ep]
                d = b - a
                denom = float(np.dot(d, d))
                active = points[idx]
                if denom == 0.0:
                    t = np.zeros(len(idx), dtype=np.float64)
                    q = np.repeat(a[None, :], len(idx), axis=0)
                else:
                    # ``(diff * d).sum(axis=1)`` is bit-identical to the frozen
                    # scalar ``float(np.dot(p - a, d))`` for every row, whereas
                    # BLAS dgemv (``np.dot((N,3),(3,))``) differs in the last
                    # ULP.  Bit-identity is required by the equivalence test.
                    t_raw = ((active - a) * d).sum(axis=1) / denom
                    t = np.minimum(1.0, np.maximum(0.0, t_raw))
                    q = a + t[:, None] * d
                dist = np.linalg.norm(active - q, axis=1)
                wire_id = int(edges.wire_id[ep])
                seg = int(edges.segment_index[ep])
                ei = int(edges.polyline_edge_index[ep])
                cur_d = best_dist[idx]
                cur_w = best_wire[idx]
                cur_s = best_seg[idx]
                cur_e = best_edge[idx]
                key_less = (wire_id < cur_w) | (
                    (wire_id == cur_w)
                    & ((seg < cur_s) | ((seg == cur_s) & (ei < cur_e)))
                )
                better = (dist < cur_d - tolerance) | (
                    (np.abs(dist - cur_d) <= tolerance) & key_less
                )
                if better.any():
                    chosen = idx[better]
                    best_dist[chosen] = dist[better]
                    best_wire[chosen] = wire_id
                    best_seg[chosen] = seg
                    best_edge[chosen] = ei
                    best_q[chosen] = q[better]
        else:
            stack.append((node.right, idx))
            stack.append((node.left, idx))
    return best_dist, best_wire, best_seg, best_edge, best_q, evaluated


@dataclass(frozen=True, slots=True)
class ManualClearanceResult:
    """SpanClearanceResult-equivalent records plus pointwise danger candidates."""

    clearance: SpanClearanceResult
    danger_original_index: np.ndarray  # int64
    danger_xyz_local_m: np.ndarray  # float32 (M,3)
    danger_distance_m: np.ndarray  # float64 (M,)
    danger_wire_id: np.ndarray  # int32 (M,)
    danger_segment_index: np.ndarray  # int32 (M,)
    danger_edge_index: np.ndarray  # int32 (M,)
    danger_tree_id: np.ndarray  # int32 (M,)
    danger_nearest_wire_xyz_local_m: np.ndarray  # float32 (M,3)


def run_manual_clearance(
    reconciled: TreeReconciliationResult,
    wire_results: list[SpanWireResult],
    cfg: FastBaselineConfig,
    manual: ManualSpanConfig,
) -> ManualClearanceResult:
    """Full-resolution exact clearance with streamed chunked execution.

    The per-tree ``SpanTreeWireRecord`` results are deterministic and identical
    to ``run_span_tree_wire_clearance`` (verified by regression test).  In
    addition every full-res raw tree point within ``danger_point_cap_m`` of its
    exact nearest valid wire is persisted for the later risk stage.
    """

    started = time.perf_counter()
    arrays, distance_ready_segments, total_wires = _concatenate_span_wire_edges(
        wire_results
    )
    edge_bundle = build_wire_edge_bundle(arrays)
    edge_count = edge_bundle.edge_count
    records: list[SpanTreeWireRecord] = []

    record_by_tree: dict[int, dict[str, Any]] = {
        record.global_tree_id: {
            "global_tree_id": record.global_tree_id,
            "external_name": record.external_name,
            "line_id": record.line_id,
            "span_id": record.span_id,
            "span_name": record.span_name,
        }
        for record in reconciled.global_tree_records
    }

    danger_index_parts: list[np.ndarray] = []
    danger_xyz_parts: list[np.ndarray] = []
    danger_dist_parts: list[np.ndarray] = []
    danger_wire_parts: list[np.ndarray] = []
    danger_seg_parts: list[np.ndarray] = []
    danger_edge_parts: list[np.ndarray] = []
    danger_tree_parts: list[np.ndarray] = []
    danger_q_parts: list[np.ndarray] = []

    if edge_count == 0 or distance_ready_segments == 0:
        status = NO_DISTANCE_READY_EDGE
        for record in reconciled.global_tree_records:
            records.append(
                SpanTreeWireRecord(
                    global_tree_id=record.global_tree_id,
                    external_name=record.external_name,
                    line_id=record.line_id,
                    span_id=record.span_id,
                    span_name=record.span_name,
                    nearest_wire_id=None,
                    minimum_clearance_m=None,
                    nearest_tree_original_index=None,
                    nearest_tree_xyz_local_m=None,
                    nearest_wire_xyz_local_m=None,
                    tree_point_count=record.full_resolution_point_count,
                    evaluated_edge_count=0,
                    status=NO_DISTANCE_READY_EDGE,
                )
            )
        clearance = SpanClearanceResult(
            records=tuple(records),
            status=status,
            global_minimum_clearance_m=None,
            edge_count=0,
            diagnostics={
                "schema": MANUAL_CLEARANCE_SCHEMA,
                "status": status,
                "total_wires": total_wires,
                "distance_ready_segments": distance_ready_segments,
                "runtime_seconds": time.perf_counter() - started,
            },
        )
        return ManualClearanceResult(
            clearance=clearance,
            danger_original_index=np.empty(0, dtype=np.int64),
            danger_xyz_local_m=np.empty((0, 3), dtype=np.float32),
            danger_distance_m=np.empty(0, dtype=np.float64),
            danger_wire_id=np.empty(0, dtype=np.int32),
            danger_segment_index=np.empty(0, dtype=np.int32),
            danger_edge_index=np.empty(0, dtype=np.int32),
            danger_tree_id=np.empty(0, dtype=np.int32),
            danger_nearest_wire_xyz_local_m=np.empty((0, 3), dtype=np.float32),
        )

    bvh = SegmentEdgeBvh(edge_bundle, cfg.clearance_bvh_leaf_cap)
    final_tree = np.asarray(reconciled.final_tree_id, dtype=np.int32)
    final_xyz = np.asarray(reconciled.final_xyz_local_m, dtype=np.float64)
    final_original = np.asarray(reconciled.final_original_index, dtype=np.int64)
    tree_ids_to_process = sorted(
        set(int(value) for value in final_tree if value >= 0)
    )
    tolerance = cfg.clearance_tie_tolerance_m
    chunk = manual.clearance_chunk_points
    danger_cap = manual.danger_point_cap_m
    evaluated_total = 0
    danger_candidate_count = 0

    for tree_id in tree_ids_to_process:
        rows = np.flatnonzero(final_tree == tree_id)
        point_count = int(len(rows))
        tree_xyz = final_xyz[rows]
        tree_orig = final_original[rows]
        best_distance: float | None = None
        best_key: tuple[int, int, int, int] | None = None
        best_q: np.ndarray | None = None
        best_original = -1
        best_point: np.ndarray | None = None
        per_tree_evaluated = 0
        tree_danger: list[np.ndarray] = []

        for start in range(0, point_count, chunk):
            stop = min(point_count, start + chunk)
            points = tree_xyz[start:stop]
            originals = tree_orig[start:stop]
            dist, wire_id, seg, ei, q, evaluated = _vectorized_nearest_edges(
                points, edge_bundle, bvh, tolerance
            )
            evaluated_total += evaluated
            per_tree_evaluated += evaluated
            # Per-tree minimum over this chunk with the frozen winner rule.
            for position in range(len(dist)):
                original_index = int(originals[position])
                key = (
                    int(wire_id[position]),
                    int(seg[position]),
                    int(ei[position]),
                    original_index,
                )
                if _candidate_better_scalar(
                    float(dist[position]),
                    key,
                    best_distance,
                    best_key,
                    tolerance,
                ):
                    best_distance = float(dist[position])
                    best_key = key
                    best_q = q[position]
                    best_original = original_index
                    best_point = points[position]
            # Danger candidates (pointwise, within the storage cap).
            near = dist <= danger_cap
            if near.any():
                tree_danger.append(
                    (
                        originals[near],
                        points[near].astype(np.float32),
                        dist[near],
                        wire_id[near],
                        seg[near],
                        ei[near],
                        q[near].astype(np.float32),
                    )
                )
                danger_candidate_count += int(np.count_nonzero(near))
        if tree_danger:
            danger_index_parts.append(np.concatenate([part[0] for part in tree_danger]))
            danger_xyz_parts.append(np.concatenate([part[1] for part in tree_danger]))
            danger_dist_parts.append(np.concatenate([part[2] for part in tree_danger]))
            danger_wire_parts.append(np.concatenate([part[3] for part in tree_danger]))
            danger_seg_parts.append(np.concatenate([part[4] for part in tree_danger]))
            danger_edge_parts.append(np.concatenate([part[5] for part in tree_danger]))
            danger_tree_parts.append(
                np.full(
                    int(np.count_nonzero(near)),
                    tree_id,
                    dtype=np.int32,
                )
            )
            danger_q_parts.append(np.concatenate([part[6] for part in tree_danger]))

        tree_context = record_by_tree.get(tree_id, {})
        records.append(
            SpanTreeWireRecord(
                global_tree_id=tree_id,
                external_name=str(tree_context.get("external_name", "")),
                line_id=int(tree_context.get("line_id", -1)),
                span_id=int(tree_context.get("span_id", -1)),
                span_name=str(tree_context.get("span_name", "")),
                nearest_wire_id=best_key[0] if best_key is not None else None,
                minimum_clearance_m=best_distance,
                nearest_tree_original_index=best_original,
                nearest_tree_xyz_local_m=(
                    None
                    if best_point is None
                    else tuple(float(value) for value in best_point)
                ),
                nearest_wire_xyz_local_m=(
                    None if best_q is None else tuple(float(value) for value in best_q)
                ),
                tree_point_count=point_count,
                evaluated_edge_count=per_tree_evaluated,
                status="VALID",
            )
        )

    valid_values = [
        float(record.minimum_clearance_m)
        for record in records
        if record.minimum_clearance_m is not None
    ]
    clearance = SpanClearanceResult(
        records=tuple(records),
        status="VALID",
        global_minimum_clearance_m=min(valid_values) if valid_values else None,
        edge_count=edge_count,
        diagnostics={
            "schema": MANUAL_CLEARANCE_SCHEMA,
            "status": "VALID",
            "total_wires": total_wires,
            "distance_ready_segments": distance_ready_segments,
            "edge_count": edge_count,
            "tree_count": len(records),
            "evaluated_edge_total": evaluated_total,
            "danger_candidate_point_count": danger_candidate_count,
            "runtime_seconds": time.perf_counter() - started,
        },
    )

    def _concat(parts: list[np.ndarray], dtype: Any, shape2: tuple[int, int] | None = None):
        if not parts:
            if shape2 is not None:
                return np.empty((0,) + shape2, dtype=dtype)
            return np.empty(0, dtype=dtype)
        return np.concatenate(parts, axis=0)

    danger_xyz = _concat(danger_xyz_parts, np.float32, (3,))
    danger_q = _concat(danger_q_parts, np.float32, (3,))
    return ManualClearanceResult(
        clearance=clearance,
        danger_original_index=_concat(danger_index_parts, np.int64),
        danger_xyz_local_m=danger_xyz,
        danger_distance_m=_concat(danger_dist_parts, np.float64),
        danger_wire_id=_concat(danger_wire_parts, np.int32),
        danger_segment_index=_concat(danger_seg_parts, np.int32),
        danger_edge_index=_concat(danger_edge_parts, np.int32),
        danger_tree_id=_concat(danger_tree_parts, np.int32),
        danger_nearest_wire_xyz_local_m=danger_q,
    )


def _candidate_better_scalar(
    distance: float,
    key: tuple[int, int, int, int],
    best_distance: float | None,
    best_key: tuple[int, ...] | None,
    tolerance: float,
) -> bool:
    """Frozen V1 winner rule (mirror of ``candidate_better`` in the exact engine)."""

    if best_distance is None or best_key is None:
        return True
    if distance < best_distance - tolerance:
        return True
    if distance > best_distance + tolerance:
        return False
    return key < best_key


__all__ = [
    "MANUAL_CLEARANCE_SCHEMA",
    "ManualClearanceResult",
    "run_manual_clearance",
    "_vectorized_nearest_edges",
]
