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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig
from scipy.spatial import cKDTree

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
# This is deliberately a point chunk, not a point-by-edge matrix size.  The
# candidate backend only holds ``chunk_size * candidate_count`` temporaries.
CLEARANCE_CHUNK_SIZE = 32_768
CANDIDATE_INITIAL_K = 16
CANDIDATE_FALLBACK_CAP = 64

VALID = "VALID"
BLOCKED = "BLOCKED"
NO_DISTANCE_READY_EDGE = "NO_DISTANCE_READY_EDGE"


class ClearanceCancelled(RuntimeError):
    """Raised at a clearance chunk boundary when the owner cancels a run."""


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
    danger_point_count: int = 0

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
            "danger_point_count": self.danger_point_count,
        }


@dataclass(frozen=True, slots=True)
class SpanDangerPoint:
    """A single exact-danger observation; no scene-wide distance array is kept."""

    global_tree_id: int
    persistent_tree_id: str | None
    original_index: int
    xyz_local_m: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_tree_id": self.global_tree_id,
            "persistent_tree_id": self.persistent_tree_id,
            "original_index": self.original_index,
            "xyz_local_m": list(self.xyz_local_m),
        }


@dataclass(frozen=True, slots=True)
class SpanClearanceResult:
    """Deterministic span-based tree-wire clearance outcome."""

    records: tuple[SpanTreeWireRecord, ...]
    status: str
    global_minimum_clearance_m: float | None
    edge_count: int
    diagnostics: dict[str, Any]
    danger_points: tuple[SpanDangerPoint, ...] = ()


def _validated_local_prefix_offsets(
    result: SpanWireResult,
    key: str,
    *,
    expected_count: int,
    expected_end: int,
    span_index: int,
) -> np.ndarray:
    """Return one validated local W1 prefix array as int64 values.

    W1 requires one boundary per item plus the initial zero.  A missing offset
    array is only safe for an empty collection; accepting a synthetic ``[0]``
    for non-empty wire/segment data would make later geometry indexing unsafe.
    """

    raw = result.w1_arrays.get(key)
    if raw is None:
        if expected_count == 0 and expected_end == 0:
            return np.zeros(1, dtype=np.int64)
        raise ValueError(
            f"Span {span_index} {key} is missing for {expected_count} items"
        )
    values = np.asarray(raw)
    if values.ndim != 1 or len(values) != expected_count + 1:
        raise ValueError(
            f"Span {span_index} {key} must have length {expected_count + 1}"
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"Span {span_index} {key} must contain integer offsets")
    values = values.astype(np.int64, copy=False)
    if (
        values[0] != 0
        or values[-1] != expected_end
        or np.any(values[1:] < values[:-1])
    ):
        raise ValueError(
            f"Span {span_index} {key} must monotonically span 0..{expected_end}"
        )
    return values


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
        total_samples += int(result.w1_arrays.get("sampled_curve_xyz", np.empty((0, 3))).shape[0])
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
    # Re-base offsets and remap wire ids per span.  W1 offsets are prefix
    # arrays, so each local array starts with its own boundary zero.  Keep one
    # global zero and append only each local suffix; appending the full local
    # array would introduce a duplicate boundary between adjacent Spans.
    new_segment_wire: list[int] = []
    new_wire_id: list[int] = []
    new_wire_segment_offsets: list[int] = [0]
    new_sampled_offsets: list[int] = [0]
    segment_cursor = 0
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
        base_wo = _validated_local_prefix_offsets(
            result,
            "wire_segment_offsets",
            expected_count=local_wires,
            expected_end=local_segments,
            span_index=span_index,
        )
        # wire_segment_offsets maps wire positions to segment positions; its
        # cumulative rebase is therefore the number of preceding segments.
        new_wire_segment_offsets.extend(
            int(value) + segment_cursor for value in base_wo[1:]
        )
        base_so = _validated_local_prefix_offsets(
            result,
            "sampled_curve_offsets",
            expected_count=local_segments,
            expected_end=local_samples,
            span_index=span_index,
        )
        new_sampled_offsets.extend(int(value) + sample_cursor for value in base_so[1:])
        segment_cursor += local_segments
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
                t = (
                    np.einsum("ij,j->i", p - a_xyz[edge_position], d)
                    / edge_denom_safe[edge_position]
                )
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


def _candidate_exact_nearest_edges(
    points: np.ndarray,
    bvh: SegmentEdgeBvh,
    edge_bundle: Any,
    tolerance: float,
    *,
    initial_k: int = CANDIDATE_INITIAL_K,
    fallback_cap: int = CANDIDATE_FALLBACK_CAP,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict[str, int]]:
    """Batch exact nearest-edge search with a conservative midpoint certificate.

    A midpoint KD-tree is used only to *order* exact segment candidates.  For a
    segment midpoint ``M`` and half length ``h``, ``dist(P, segment)`` is at
    least ``max(0, dist(P, M) - h)``.  The next unvisited midpoint distance and
    the largest segment half-length therefore provide a conservative bound for
    every unvisited segment.  Equality never prunes, preserving the frozen
    tolerance/tie rule.  Points that cannot be certified within a bounded
    candidate set fall back to the existing reference BVH.
    """

    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    edge_count = int(edge_bundle.edge_count)
    if edge_count <= 0:
        raise ValueError("at least one edge is required")
    if type(initial_k) is not int or initial_k <= 0:
        raise ValueError("initial_k must be a positive integer")
    if type(fallback_cap) is not int or fallback_cap <= 0:
        raise ValueError("fallback_cap must be a positive integer")
    if type(workers) is not int or workers == 0:
        raise ValueError("workers must be a non-zero integer")

    a_xyz = np.asarray(edge_bundle.a_xyz, dtype=np.float64)
    b_xyz = np.asarray(edge_bundle.b_xyz, dtype=np.float64)
    edge_d = b_xyz - a_xyz
    denominator = np.einsum("ij,ij->i", edge_d, edge_d)
    denominator_safe = np.where(denominator == 0.0, 1.0, denominator)
    midpoint = 0.5 * (a_xyz + b_xyz)
    half_length = 0.5 * np.sqrt(denominator)
    maximum_half_length = float(np.max(half_length))
    tree = cKDTree(midpoint)
    canonical_rank = np.empty(edge_count, dtype=np.int64)
    canonical_rank[_edge_order_by_key(edge_bundle)] = np.arange(edge_count, dtype=np.int64)

    selected_edge = np.empty(len(xyz), dtype=np.int64)
    selected_distance = np.empty(len(xyz), dtype=np.float64)
    selected_q = np.empty((len(xyz), 3), dtype=np.float64)
    selected_t = np.empty(len(xyz), dtype=np.float64)
    unresolved = np.arange(len(xyz), dtype=np.int64)
    k = min(edge_count, initial_k)
    evaluated_total = 0
    certificate_success = 0
    fallback_points = 0
    expansion_rounds = 0

    while len(unresolved):
        active = xyz[unresolved]
        midpoint_distance, candidate_edge = tree.query(active, k=k, workers=workers)
        if k == 1:
            midpoint_distance = np.asarray(midpoint_distance, dtype=np.float64)[:, None]
            candidate_edge = np.asarray(candidate_edge, dtype=np.int64)[:, None]
        else:
            midpoint_distance = np.asarray(midpoint_distance, dtype=np.float64)
            candidate_edge = np.asarray(candidate_edge, dtype=np.int64)
        if candidate_edge.ndim != 2 or candidate_edge.shape != (len(active), k):
            raise ValueError("midpoint candidate query returned an invalid shape")
        candidate_a = a_xyz[candidate_edge]
        candidate_d = edge_d[candidate_edge]
        candidate_denominator = denominator[candidate_edge]
        t = np.einsum("nkj,nkj->nk", active[:, None, :] - candidate_a, candidate_d)
        t /= denominator_safe[candidate_edge]
        t[candidate_denominator == 0.0] = 0.0
        np.clip(t, 0.0, 1.0, out=t)
        q = candidate_a + t[:, :, None] * candidate_d
        delta = active[:, None, :] - q
        distance = np.sqrt(np.einsum("nkj,nkj->nk", delta, delta))
        best_column = np.zeros(len(active), dtype=np.int64)
        best_distance = distance[:, 0].copy()
        best_rank = canonical_rank[candidate_edge[:, 0]].copy()
        for column in range(1, k):
            current_distance = distance[:, column]
            current_rank = canonical_rank[candidate_edge[:, column]]
            within_tie = np.abs(current_distance - best_distance) <= tolerance
            better = current_distance < best_distance - tolerance
            better |= within_tie & (current_rank < best_rank)
            if np.any(better):
                best_column[better] = column
                best_distance[better] = current_distance[better]
                best_rank[better] = current_rank[better]
        rows = np.arange(len(active), dtype=np.int64)
        best_edge = candidate_edge[rows, best_column]
        best_q = q[rows, best_column]
        best_t = t[rows, best_column]
        evaluated_total += int(len(active) * k)

        # If all edges have been refined the result is exact without a BVH.
        if k == edge_count:
            certified = np.ones(len(active), dtype=bool)
        else:
            lower_bound_unvisited = np.maximum(0.0, midpoint_distance[:, -1] - maximum_half_length)
            # Strictly greater is required: an equal lower bound can still
            # produce a tolerance tie whose canonical edge key wins.
            certified = lower_bound_unvisited > best_distance + tolerance
        if np.any(certified):
            target = unresolved[certified]
            selected_edge[target] = best_edge[certified]
            selected_distance[target] = best_distance[certified]
            selected_q[target] = best_q[certified]
            selected_t[target] = best_t[certified]
            certificate_success += int(np.count_nonzero(certified))
        unresolved = unresolved[~certified]
        if not len(unresolved):
            break
        if k < min(edge_count, fallback_cap):
            k = min(edge_count, max(k + 1, k * 2), fallback_cap)
            expansion_rounds += 1
            continue
        # Keep the existing deterministic segment-AABB implementation as the
        # reference/fallback for the geometries a midpoint certificate cannot
        # prove cheaply.
        for target in unresolved:
            edge, value, nearest, parameter, evaluated = bvh.nearest_edge(xyz[target], tolerance)
            selected_edge[target] = edge
            selected_distance[target] = value
            selected_q[target] = nearest
            selected_t[target] = parameter
            evaluated_total += evaluated
            fallback_points += 1
        unresolved = np.empty(0, dtype=np.int64)

    return (
        selected_edge,
        selected_distance,
        selected_q,
        selected_t,
        evaluated_total,
        {
            "certificate_success_points": certificate_success,
            "fallback_points": fallback_points,
            "candidate_expansion_rounds": expansion_rounds,
        },
    )


def _nearest_edges(
    points: np.ndarray,
    bvh: SegmentEdgeBvh,
    edge_bundle: Any,
    tolerance: float,
    *,
    backend: str,
    candidate_workers: int = 1,
    gpu_accelerator: Any | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict[str, Any]]:
    """Dispatch one tree's exact distance search to the selected backend."""

    if backend == "NUMPY_DIRECT":
        edge, distance, nearest, parameter, evaluated = _direct_nearest_edges(
            points, edge_bundle, tolerance
        )
        return edge, distance, nearest, parameter, evaluated, {}
    if backend == "SCIPY_CANDIDATE_EXACT":
        if gpu_accelerator is not None:
            return gpu_accelerator.nearest(
                points,
                lambda values: _candidate_exact_nearest_edges(
                    values,
                    bvh,
                    edge_bundle,
                    tolerance,
                    workers=candidate_workers,
                ),
                cancellation_requested=cancellation_requested,
            )
        return _candidate_exact_nearest_edges(
            points, bvh, edge_bundle, tolerance, workers=candidate_workers
        )
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
    return edge_positions, distances, nearest, parameters, evaluated_total, {}


def run_span_tree_wire_clearance(
    reconciled: TreeReconciliationResult,
    wire_results: list[SpanWireResult],
    config: FastBaselineConfig,
    *,
    backend: str | None = None,
    danger_threshold_m: float | None = None,
    chunk_size: int = CLEARANCE_CHUNK_SIZE,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
    candidate_workers: int = 1,
    acceleration_mode: str = "AUTO",
) -> SpanClearanceResult:
    """Compute full-resolution tree-wire clearance per span, fail closed.

    Wire curves come from the FAST span wire pipeline; trees are reconciled
    full-resolution raw memberships.  The default backend is batched exact
    candidate search; the original BVH is retained as an explicit reference
    backend and as a candidate fallback.  Only per-tree minima and exact
    danger observations survive the streaming pass.
    """

    started = time.perf_counter()
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if danger_threshold_m is not None and (
        not np.isfinite(danger_threshold_m) or danger_threshold_m <= 0.0
    ):
        raise ValueError("danger_threshold_m must be a finite positive number or None")
    index_started = time.perf_counter()
    arrays, distance_ready_segments, total_wires = _concatenate_span_wire_edges(wire_results)
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
    index_runtime = time.perf_counter() - index_started
    selected_backend = backend or "SCIPY_CANDIDATE_EXACT"
    if selected_backend not in {"NUMPY_DIRECT", "PYTHON_BVH", "SCIPY_CANDIDATE_EXACT"}:
        raise ValueError(f"unknown clearance backend: {selected_backend}")
    from algorithms.spans.gpu_clearance import (
        GpuClearanceCancelled,
        create_gpu_accelerator,
        normalize_gpu_mode,
    )

    requested_acceleration = normalize_gpu_mode(acceleration_mode)
    if selected_backend != "SCIPY_CANDIDATE_EXACT" and requested_acceleration == "GPU":
        raise ValueError("GPU acceleration requires SCIPY_CANDIDATE_EXACT backend")
    gpu_accelerator = None
    acceleration_diagnostics: dict[str, Any] = {
        "requested_mode": requested_acceleration,
        "effective_mode": "CPU",
        "reason": "NON_CANDIDATE_BACKEND",
    }
    if selected_backend == "SCIPY_CANDIDATE_EXACT":
        gpu_accelerator, acceleration_diagnostics = create_gpu_accelerator(
            edge_bundle,
            config.clearance_tie_tolerance_m,
            mode=requested_acceleration,
        )
    processing_chunk_size = (
        max(chunk_size, int(gpu_accelerator.chunk_size))
        if gpu_accelerator is not None
        else chunk_size
    )
    # Group final members by global tree id (deterministic).
    final_tree = np.asarray(reconciled.final_tree_id, dtype=np.int32)
    final_xyz = np.asarray(reconciled.final_xyz_local_m, dtype=np.float64)
    final_original = np.asarray(reconciled.final_original_index, dtype=np.int64)
    tree_ids_to_process = sorted(set(int(value) for value in final_tree if value >= 0))
    total_points = int(len(final_xyz))
    processed_points = 0
    danger_points: list[SpanDangerPoint] = []
    candidate_diagnostics = {
        "certificate_success_points": 0,
        "fallback_points": 0,
        "candidate_expansion_rounds": 0,
    }
    maximum_processing_chunk_size = processing_chunk_size
    distance_started = time.perf_counter()
    for tree_ordinal, tree_id in enumerate(tree_ids_to_process, start=1):
        rows = np.flatnonzero(final_tree == tree_id)
        point_count = int(len(rows))
        best_distance: float | None = None
        best_key: tuple[int, int, int, int] | None = None
        best_q: np.ndarray | None = None
        best_original = -1
        best_point: np.ndarray | None = None
        evaluated_total = 0
        danger_count = 0
        local_start = 0
        chunk_ordinal = 0
        while local_start < point_count:
            chunk_ordinal += 1
            if cancellation_requested is not None and cancellation_requested():
                raise ClearanceCancelled("CLEARANCE_CANCELLED_AT_CHUNK_BOUNDARY")
            local_stop = min(local_start + processing_chunk_size, point_count)
            chunk_rows = rows[local_start:local_stop]
            try:
                (
                    edge_positions,
                    distances,
                    nearest_xyz,
                    _parameters,
                    evaluated,
                    backend_diagnostics,
                ) = _nearest_edges(
                    final_xyz[chunk_rows],
                    bvh,
                    edge_bundle,
                    config.clearance_tie_tolerance_m,
                    backend=selected_backend,
                    candidate_workers=candidate_workers,
                    gpu_accelerator=gpu_accelerator,
                    cancellation_requested=cancellation_requested,
                )
            except GpuClearanceCancelled as exc:
                raise ClearanceCancelled("CLEARANCE_CANCELLED_AT_GPU_BLOCK_BOUNDARY") from exc
            except Exception as exc:
                if gpu_accelerator is None or requested_acceleration == "GPU":
                    raise
                acceleration_diagnostics = {
                    "requested_mode": requested_acceleration,
                    "effective_mode": "CPU",
                    "reason": f"GPU_RUNTIME_FAILED:{type(exc).__name__}:{exc}",
                    "capability": gpu_accelerator.capability.to_dict(),
                }
                gpu_accelerator = None
                processing_chunk_size = chunk_size
                (
                    edge_positions,
                    distances,
                    nearest_xyz,
                    _parameters,
                    evaluated,
                    backend_diagnostics,
                ) = _nearest_edges(
                    final_xyz[chunk_rows],
                    bvh,
                    edge_bundle,
                    config.clearance_tie_tolerance_m,
                    backend=selected_backend,
                    candidate_workers=candidate_workers,
                )
            evaluated_total += evaluated
            for name, value in backend_diagnostics.items():
                candidate_diagnostics[name] = candidate_diagnostics.get(name, 0) + int(value)
            # A chunk contributes one deterministic minimum candidate.  This
            # avoids another 29M-row Python reduction loop while retaining the
            # frozen tolerance then canonical wire/segment/edge/source key.
            local_minimum = float(np.min(distances))
            local_candidates = np.flatnonzero(
                distances <= local_minimum + config.clearance_tie_tolerance_m
            )
            if len(local_candidates) > 1:
                candidate_edges = edge_positions[local_candidates]
                candidate_original = final_original[chunk_rows[local_candidates]]
                candidate_order = np.lexsort(
                    (
                        candidate_original,
                        edge_bundle.polyline_edge_index[candidate_edges],
                        edge_bundle.segment_index[candidate_edges],
                        edge_bundle.wire_id[candidate_edges],
                    )
                )
                local_position = int(local_candidates[candidate_order[0]])
            else:
                local_position = int(local_candidates[0])
            position = int(chunk_rows[local_position])
            edge_position = int(edge_positions[local_position])
            distance = float(distances[local_position])
            key = (
                int(edge_bundle.wire_id[edge_position]),
                int(edge_bundle.segment_index[edge_position]),
                int(edge_bundle.polyline_edge_index[edge_position]),
                int(final_original[position]),
            )
            if candidate_better(
                distance, key, best_distance, best_key, config.clearance_tie_tolerance_m
            ):
                best_distance = distance
                best_key = key
                best_q = nearest_xyz[local_position]
                best_original = int(final_original[position])
                best_point = final_xyz[position]
            if danger_threshold_m is not None:
                danger_positions = np.flatnonzero(distances <= danger_threshold_m)
                danger_count += int(len(danger_positions))
                persistent_tree_id = record_by_tree.get(tree_id, {}).get("persistent_tree_id")
                danger_points.extend(
                    SpanDangerPoint(
                        global_tree_id=tree_id,
                        persistent_tree_id=persistent_tree_id,
                        original_index=int(final_original[int(chunk_rows[local_position])]),
                        xyz_local_m=tuple(
                            float(value) for value in final_xyz[int(chunk_rows[local_position])]
                        ),
                    )
                    for local_position in danger_positions
                )
            processed_points += len(chunk_rows)
            if progress_callback is not None:
                elapsed = time.perf_counter() - started
                throughput = processed_points / elapsed if elapsed > 0.0 else None
                eta = (
                    (total_points - processed_points) / throughput
                    if throughput and processed_points < total_points
                    else 0.0
                )
                progress_callback(
                    {
                        "stage": "clearance",
                        "processed_points": processed_points,
                        "total_points": total_points,
                        "percent": (
                            100.0 * processed_points / total_points if total_points else 100.0
                        ),
                        "current_tree_id": tree_id,
                        "current_tree_ordinal": tree_ordinal,
                        "tree_count": len(tree_ids_to_process),
                        "current_chunk": chunk_ordinal,
                        "elapsed_seconds": elapsed,
                        "recent_throughput_points_per_second": throughput,
                        "eta_seconds": eta,
                    }
                )
            local_start = local_stop
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
                    None if best_point is None else tuple(float(value) for value in best_point)
                ),
                nearest_wire_xyz_local_m=(
                    None if best_q is None else tuple(float(value) for value in best_q)
                ),
                tree_point_count=point_count,
                evaluated_edge_count=evaluated_total,
                status=VALID,
                persistent_tree_id=tree_context.get("persistent_tree_id"),
                danger_point_count=danger_count,
            )
        )
    distance_runtime = time.perf_counter() - distance_started
    valid_values = [
        float(record.minimum_clearance_m)
        for record in records
        if record.minimum_clearance_m is not None
    ]
    if gpu_accelerator is not None:
        acceleration_diagnostics = dict(gpu_accelerator.diagnostics)
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
            "backend": (
                "GPU_CERTIFIED_CPU_EXACT"
                if acceleration_diagnostics.get("effective_mode")
                == "GPU_CERTIFIED_CPU_EXACT"
                else selected_backend
            ),
            "chunk_size": maximum_processing_chunk_size,
            "candidate_workers": (
                candidate_workers if selected_backend == "SCIPY_CANDIDATE_EXACT" else None
            ),
            "danger_threshold_m": danger_threshold_m,
            "danger_point_count": len(danger_points),
            "full_per_point_clearance_stored": False,
            **candidate_diagnostics,
            "clearance_broad_phase_points": int(len(final_xyz)),
            "clearance_exact_points": int(len(final_xyz)),
            "index_build_seconds": index_runtime,
            "distance_evaluation_seconds": distance_runtime,
            "runtime_seconds": time.perf_counter() - started,
            "acceleration": acceleration_diagnostics,
        },
        danger_points=tuple(danger_points),
    )


__all__ = [
    "SPAN_CLEARANCE_SCHEMA",
    "ClearanceCancelled",
    "SpanTreeWireRecord",
    "SpanDangerPoint",
    "SpanClearanceResult",
    "run_span_tree_wire_clearance",
]
