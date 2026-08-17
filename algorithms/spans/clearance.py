"""Per-span tree-wire clearance reusing the exact distance mathematics.

The FAST span baseline reuses the EXISTING Tree-Wire engine's exact building
blocks — the deterministic segment-AABB BVH, the float64 point-to-segment
formula, zero-length handling, tie tolerance and tie ordering — without
modifying any of them.  Full-resolution raw tree points are compared against
the distance-ready FAST span wire curves.  The output is explicitly
NON_CERTIFIED and never claims Formal approval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

from algorithms.spans.reconcile import TreeReconciliationResult
from algorithms.spans.wire import SpanWireResult
from algorithms.vegetation.tree_wire_clearance import (
    SegmentEdgeBvh,
    build_wire_edge_bundle,
    candidate_better,
)

SPAN_CLEARANCE_SCHEMA = "fast-baseline-span-clearance-v1"

# The current FAST wire reconstruction produces small edge bundles.  For those
# bundles a bounded, vectorized direct evaluation is materially cheaper than
# entering the Python BVH for every raw point.  The reference BVH remains the
# oracle and is retained for larger bundles until measured evidence says the
# crossover should move.
# Benchmark evidence on the real S002 edge bundle shows bounded NumPy direct
# evaluation remains substantially faster through 256 edges.  The BVH remains
# the exact reference oracle and the cap is deliberately conservative for
# larger/future bundles until another benchmark justifies extending it.
DIRECT_EDGE_CAP = 256
DIRECT_CHUNK_SIZE = 65_536

VALID = "VALID"
BLOCKED = "BLOCKED"
NO_DISTANCE_READY_EDGE = "NO_DISTANCE_READY_EDGE"


@dataclass(frozen=True, slots=True)
class SpanTreeWireRecord:
    """Per-tree minimum clearance against the span conductor curves."""

    global_tree_id: int
    external_name: str
    line_id: int
    span_id: int
    span_name: str
    nearest_wire_id: int | None
    minimum_clearance_m: float | None
    nearest_tree_original_index: int | None
    nearest_tree_xyz_local_m: tuple[float, float, float] | None
    nearest_wire_xyz_local_m: tuple[float, float, float] | None
    tree_point_count: int
    evaluated_edge_count: int
    status: str
    persistent_tree_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_tree_id": self.global_tree_id,
            "external_name": self.external_name,
            "persistent_tree_id": self.persistent_tree_id,
            "line_id": self.line_id,
            "span_id": self.span_id,
            "span_name": self.span_name,
            "nearest_wire_id": self.nearest_wire_id,
            "minimum_clearance_m": self.minimum_clearance_m,
            "nearest_tree_original_index": self.nearest_tree_original_index,
            "nearest_tree_xyz_local_m": (
                None
                if self.nearest_tree_xyz_local_m is None
                else list(self.nearest_tree_xyz_local_m)
            ),
            "nearest_wire_xyz_local_m": (
                None
                if self.nearest_wire_xyz_local_m is None
                else list(self.nearest_wire_xyz_local_m)
            ),
            "tree_point_count": self.tree_point_count,
            "evaluated_edge_count": self.evaluated_edge_count,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class SpanClearanceResult:
    """Deterministic span-based tree-wire clearance outcome."""

    records: tuple[SpanTreeWireRecord, ...]
    status: str
    global_minimum_clearance_m: float | None
    edge_count: int
    diagnostics: dict[str, Any]


def _concatenate_span_wire_edges(
    wire_results: list[SpanWireResult],
) -> tuple[dict[str, np.ndarray], int, int]:
    """Concatenate per-span W1 arrays with globally-unique deterministic wire ids.

    Wire ids are re-based to be globally unique and sorted in span order; the
    ``sampled_curve_offsets`` and ``wire_segment_offsets`` are re-based onto the
    concatenated arrays so the exact edge-bundle builder receives valid offsets.
    """

    wire_offsets: list[int] = []
    total_wires = 0
    sample_offsets: list[int] = []
    total_samples = 0
    for result in wire_results:
        wire_offsets.append(total_wires)
        total_wires += int(result.w1_arrays.get("wire_id", np.empty(0)).shape[0])
        sample_offsets.append(total_samples)
        total_samples += int(
            result.w1_arrays.get("sampled_curve_xyz", np.empty((0, 3))).shape[0]
        )
    keys = (
        "wire_id",
        "wire_status",
        "wire_method",
        "wire_quality_flags",
        "wire_has_distance_ready_segments",
        "wire_continuous_distance_ready",
        "wire_segment_offsets",
        "segment_wire_id",
        "segment_index",
        "segment_method",
        "segment_quality_flags",
        "segment_distance_ready",
        "segment_parameter_domain",
        "sampled_curve_xyz",
        "sampled_curve_parameter",
        "sampled_curve_offsets",
    )
    arrays: dict[str, np.ndarray] = {}
    for key in keys:
        parts = [result.w1_arrays.get(key) for result in wire_results]
        parts = [part for part in parts if part is not None and len(part)]
        if parts:
            arrays[key] = np.concatenate(parts, axis=0)
        else:
            dtype = np.dtype("f8") if key == "segment_parameter_domain" else np.dtype("i4")
            if key == "sampled_curve_xyz":
                dtype = np.dtype("f4")
            shape = (
                (0, 2)
                if key == "segment_parameter_domain"
                else (0, 3)
                if key == "sampled_curve_xyz"
                else (0,)
            )
            arrays[key] = np.empty(shape, dtype=dtype)
    # Re-base offsets and remap wire ids per span.
    new_segment_wire: list[int] = []
    new_wire_id: list[int] = []
    new_wire_segment_offsets: list[int] = []
    new_sampled_offsets: list[int] = []
    segment_cursor = 0
    wire_cursor = 0
    sample_cursor = 0
    for span_index, result in enumerate(wire_results):
        local_segments = int(result.w1_arrays.get("segment_wire_id", np.empty(0)).shape[0])
        local_wires = int(result.w1_arrays.get("wire_id", np.empty(0)).shape[0])
        local_samples = int(result.w1_arrays.get("sampled_curve_xyz", np.empty((0, 3))).shape[0])
        offset = wire_offsets[span_index]
        for position in range(local_segments):
            local_wire = int(segment_wire_id_at(result, position))
            new_segment_wire.append(offset + local_wire)
        for wire_position in range(local_wires):
            new_wire_id.append(offset + wire_position)
        base_wo = result.w1_arrays.get("wire_segment_offsets", np.zeros(1, dtype=np.int64))
        new_wire_segment_offsets.extend(
            int(value) + wire_cursor for value in np.asarray(base_wo)
        )
        base_so = result.w1_arrays.get("sampled_curve_offsets", np.zeros(1, dtype=np.int64))
        new_sampled_offsets.extend(
            int(value) + sample_cursor for value in np.asarray(base_so)
        )
        segment_cursor += local_segments
        wire_cursor += local_wires
        sample_cursor += local_samples
    arrays["segment_wire_id"] = np.asarray(new_segment_wire, dtype=np.int32)
    arrays["wire_id"] = np.asarray(new_wire_id, dtype=np.int32)
    arrays["wire_segment_offsets"] = np.asarray(new_wire_segment_offsets, dtype=np.int64)
    arrays["sampled_curve_offsets"] = np.asarray(new_sampled_offsets, dtype=np.int64)
    distance_ready = np.asarray(arrays["segment_distance_ready"], dtype=bool)
    return arrays, int(np.count_nonzero(distance_ready)), total_wires


def segment_wire_id_at(result: SpanWireResult, position: int) -> int:
    """Return the local wire id of *result*'s segment at *position*."""

    return int(result.w1_arrays["segment_wire_id"][position])


def _edge_order_by_key(edge_bundle: Any) -> np.ndarray:
    """Return edge positions in the frozen wire/segment/edge key order."""

    return np.lexsort(
        (
            np.asarray(edge_bundle.polyline_edge_index, dtype=np.int64),
            np.asarray(edge_bundle.segment_index, dtype=np.int64),
            np.asarray(edge_bundle.wire_id, dtype=np.int64),
        )
    ).astype(np.int64, copy=False)


def _direct_nearest_edges(
    points: np.ndarray,
    edge_bundle: Any,
    tolerance: float,
    *,
    chunk_size: int = DIRECT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Evaluate exact point-to-edge distances in bounded NumPy batches.

    This is not an approximation: every point is evaluated against every edge
    using the same float64 projection formula as the reference implementation.
    Edges are visited in canonical key order, which is equivalent to the
    reference tie rule (minimum distance, then lexicographically smallest edge
    key within the frozen tolerance).
    """

    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    edge_count = int(edge_bundle.edge_count)
    if edge_count <= 0:
        raise ValueError("at least one edge is required")
    order = _edge_order_by_key(edge_bundle)
    a_xyz = np.asarray(edge_bundle.a_xyz, dtype=np.float64)
    b_xyz = np.asarray(edge_bundle.b_xyz, dtype=np.float64)
    if a_xyz.shape != (edge_count, 3) or b_xyz.shape != (edge_count, 3):
        raise ValueError("edge bundle coordinates are malformed")
    edge_d = b_xyz - a_xyz
    edge_denom = np.einsum("ij,ij->i", edge_d, edge_d)
    edge_denom_safe = np.where(edge_denom == 0.0, 1.0, edge_denom)
    selected_edge = np.empty(len(xyz), dtype=np.int64)
    selected_distance = np.empty(len(xyz), dtype=np.float64)
    selected_q = np.empty((len(xyz), 3), dtype=np.float64)
    selected_t = np.empty(len(xyz), dtype=np.float64)
    for start in range(0, len(xyz), chunk_size):
        stop = min(len(xyz), start + chunk_size)
        p = xyz[start:stop]
        best_distance = np.full(len(p), np.inf, dtype=np.float64)
        best_edge = np.full(len(p), -1, dtype=np.int64)
        best_q = np.empty((len(p), 3), dtype=np.float64)
        best_t = np.zeros(len(p), dtype=np.float64)
        for edge_position in order:
            edge_position = int(edge_position)
            d = edge_d[edge_position]
            if edge_denom[edge_position] == 0.0:
                t = np.zeros(len(p), dtype=np.float64)
            else:
                t = np.einsum("ij,j->i", p - a_xyz[edge_position], d) / edge_denom_safe[
                    edge_position
                ]
                np.clip(t, 0.0, 1.0, out=t)
            q = a_xyz[edge_position] + t[:, None] * d
            delta = p - q
            distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            # The canonical edge order makes the lexicographically smaller key
            # win all tolerance ties without a per-point Python reduction.
            better = distance < best_distance - tolerance
            better |= ~np.isfinite(best_distance)
            if np.any(better):
                best_distance[better] = distance[better]
                best_edge[better] = edge_position
                best_q[better] = q[better]
                best_t[better] = t[better]
        if np.any(best_edge < 0):  # pragma: no cover - guarded by edge_count
            raise ValueError("direct exact backend found no edge")
        selected_edge[start:stop] = best_edge
        selected_distance[start:stop] = best_distance
        selected_q[start:stop] = best_q
        selected_t[start:stop] = best_t
    return selected_edge, selected_distance, selected_q, selected_t, len(xyz) * edge_count


def _nearest_edges(
    points: np.ndarray,
    bvh: SegmentEdgeBvh,
    edge_bundle: Any,
    tolerance: float,
    *,
    backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Dispatch one tree's exact distance search to the selected backend."""

    if backend == "NUMPY_DIRECT":
        return _direct_nearest_edges(points, edge_bundle, tolerance)
    if backend != "PYTHON_BVH":
        raise ValueError(f"unknown clearance backend: {backend}")
    edge_positions = np.empty(len(points), dtype=np.int64)
    distances = np.empty(len(points), dtype=np.float64)
    nearest = np.empty((len(points), 3), dtype=np.float64)
    parameters = np.empty(len(points), dtype=np.float64)
    evaluated_total = 0
    for position, point in enumerate(points):
        edge_position, distance, q, t, evaluated = bvh.nearest_edge(point, tolerance)
        edge_positions[position] = edge_position
        distances[position] = distance
        nearest[position] = q
        parameters[position] = t
        evaluated_total += evaluated
    return edge_positions, distances, nearest, parameters, evaluated_total


def run_span_tree_wire_clearance(
    reconciled: TreeReconciliationResult,
    wire_results: list[SpanWireResult],
    config: FastBaselineConfig,
    *,
    backend: str | None = None,
) -> SpanClearanceResult:
    """Compute full-resolution tree-wire clearance per span, fail closed.

    Wire curves come from the FAST span wire pipeline; trees are the
    reconciled full-resolution raw memberships.  The exact BVH /
    point-to-segment engine is reused unchanged.
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
            # Accept legacy reconciliation-shaped records in reference/backend
            # tests while production GlobalTreeRecord always supplies this.
            "persistent_tree_id": getattr(record, "persistent_tree_id", None),
            "line_id": record.line_id,
            "span_id": record.span_id,
            "span_name": record.span_name,
        }
        for record in reconciled.global_tree_records
    }
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
                    persistent_tree_id=getattr(record, "persistent_tree_id", None),
                )
            )
        return SpanClearanceResult(
            records=tuple(records),
            status=status,
            global_minimum_clearance_m=None,
            edge_count=0,
            diagnostics={
                "schema": SPAN_CLEARANCE_SCHEMA,
                "status": status,
                "total_wires": total_wires,
                "distance_ready_segments": distance_ready_segments,
                "runtime_seconds": time.perf_counter() - started,
            },
        )

    bvh = SegmentEdgeBvh(edge_bundle, config.clearance_bvh_leaf_cap)
    selected_backend = backend or (
        "NUMPY_DIRECT" if edge_count <= DIRECT_EDGE_CAP else "PYTHON_BVH"
    )
    if selected_backend not in {"NUMPY_DIRECT", "PYTHON_BVH"}:
        raise ValueError(f"unknown clearance backend: {selected_backend}")
    # Group final members by global tree id (deterministic).
    final_tree = np.asarray(reconciled.final_tree_id, dtype=np.int32)
    final_xyz = np.asarray(reconciled.final_xyz_local_m, dtype=np.float64)
    final_original = np.asarray(reconciled.final_original_index, dtype=np.int64)
    tree_ids_to_process = sorted(
        set(int(value) for value in final_tree if value >= 0)
    )
    for tree_id in tree_ids_to_process:
        rows = np.flatnonzero(final_tree == tree_id)
        point_count = int(len(rows))
        best_distance: float | None = None
        best_key: tuple[int, int, int, int] | None = None
        best_q: np.ndarray | None = None
        best_original = -1
        best_point: np.ndarray | None = None
        evaluated_total = 0
        points = final_xyz[rows]
        edge_positions, distances, nearest_xyz, _parameters, evaluated_total = _nearest_edges(
            points,
            bvh,
            edge_bundle,
            config.clearance_tie_tolerance_m,
            backend=selected_backend,
        )
        for local_position, position in enumerate(rows):
            edge_position = int(edge_positions[local_position])
            distance = float(distances[local_position])
            q = nearest_xyz[local_position]
            point = final_xyz[position]
            original_index = int(final_original[position])
            key = (
                int(edge_bundle.wire_id[edge_position]),
                int(edge_bundle.segment_index[edge_position]),
                int(edge_bundle.polyline_edge_index[edge_position]),
                original_index,
            )
            if candidate_better(
                distance, key, best_distance, best_key, config.clearance_tie_tolerance_m
            ):
                best_distance = distance
                best_key = key
                best_q = q
                best_original = original_index
                best_point = point
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
                evaluated_edge_count=evaluated_total,
                status=VALID,
                persistent_tree_id=tree_context.get("persistent_tree_id"),
            )
        )
    valid_values = [
        float(record.minimum_clearance_m)
        for record in records
        if record.minimum_clearance_m is not None
    ]
    return SpanClearanceResult(
        records=tuple(records),
        status=VALID,
        global_minimum_clearance_m=min(valid_values) if valid_values else None,
        edge_count=edge_count,
        diagnostics={
            "schema": SPAN_CLEARANCE_SCHEMA,
            "status": VALID,
            "total_wires": total_wires,
            "distance_ready_segments": distance_ready_segments,
            "edge_count": edge_count,
            "tree_count": len(records),
            "evaluated_edge_total": sum(record.evaluated_edge_count for record in records),
            "backend": selected_backend,
            "chunk_size": DIRECT_CHUNK_SIZE if selected_backend == "NUMPY_DIRECT" else None,
            "clearance_broad_phase_points": int(len(final_xyz)),
            "clearance_exact_points": int(len(final_xyz)),
            "runtime_seconds": time.perf_counter() - started,
        },
    )


__all__ = [
    "SPAN_CLEARANCE_SCHEMA",
    "SpanTreeWireRecord",
    "SpanClearanceResult",
    "run_span_tree_wire_clearance",
]
