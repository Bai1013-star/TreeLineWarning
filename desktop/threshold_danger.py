"""Current-policy viewer mask over full-resolution, published tree membership.

This is a threshold display projection, not a refit or a clearance publication.
Only distance-ready W1 edges are used. A midpoint sphere is a conservative
candidate screen; every candidate is judged against the actual line segment.
No sampled display point, nearest-tree classification or schematic is used to
decide point membership.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree

from algorithms.vegetation.tree_wire_clearance import build_wire_edge_bundle
from core.stage_a.risk import RiskPolicy
from desktop.current_report import _wire_arrays


def _empty(authority: str, threshold: float) -> dict[str, Any]:
    return dict(xyz=[], tree_id_by_index=[], original_index=[], distance_m=[],
                display_point_count=0, full_point_count=0, authority=authority,
                threshold_m=threshold, display_only_sampling=False)


def _edges(result: Any) -> tuple[np.ndarray, np.ndarray]:
    starts, ends = [], []
    for span in (getattr(result, "span_results", {}) or {}).values():
        wire = span.get("wire_result") if isinstance(span, dict) else getattr(span, "wire_result", None)
        if wire is None:
            continue
        bundle = build_wire_edge_bundle(_wire_arrays(wire))
        if bundle.edge_count:
            starts.append(bundle.a_xyz)
            ends.append(bundle.b_xyz)
    if not starts:
        return np.empty((0, 3)), np.empty((0, 3))
    a, b = np.concatenate(starts), np.concatenate(ends)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("THRESHOLD_WIRE_GEOMETRY_NONFINITE")
    return a, b


def threshold_membership(xyz: np.ndarray, a: np.ndarray, b: np.ndarray,
                         threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return exact selected row indices and their nearest distance, bounded RAM.

    Query radius = threshold + the longest edge half-length, rounded outward.
    Thus even an edge endpoint inside the threshold cannot be missed by the
    midpoint screen. Curves, conductor boundaries and unsupported gaps are
    never joined. The final comparison includes the configured boundary.
    """
    threshold = RiskPolicy(warning_clearance_threshold_m=threshold).warning_clearance_threshold_m
    xyz, a, b = (np.asarray(value, dtype=np.float64) for value in (xyz, a, b))
    if any(value.ndim != 2 or value.shape[1] != 3 for value in (xyz, a, b)) or a.shape != b.shape:
        raise ValueError("THRESHOLD_GEOMETRY_SHAPE")
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("THRESHOLD_GEOMETRY_NONFINITE")
    if not len(xyz) or not len(a):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    d = b - a
    denominator = np.einsum("ij,ij->i", d, d)
    safe = np.where(denominator > 0, denominator, 1.0)
    midpoint = a + d * 0.5
    radius = np.nextafter(threshold + np.sqrt(denominator.max()) * 0.5, np.inf)
    index = cKDTree(midpoint)
    chosen, distances = [], []
    # At most 131072 candidate pairs per vectorized judge. Even a large user
    # threshold cannot allocate a points-by-all-edges distance tensor.
    for start in range(0, len(xyz), 1024):
        block = xyz[start:start + 1024]
        best = np.full(len(block), np.inf)
        candidates = index.query_ball_point(block, radius, eps=0.0)
        point_rows, edge_rows = [], []

        def judge() -> None:
            if not point_rows:
                return
            rows = np.asarray(point_rows, dtype=np.int64)
            edges = np.asarray(edge_rows, dtype=np.int64)
            delta = block[rows] - a[edges]
            parameter = np.einsum("ij,ij->i", delta, d[edges]) / safe[edges]
            np.clip(parameter, 0.0, 1.0, out=parameter)
            residual = delta - parameter[:, None] * d[edges]
            distance = np.sqrt(np.einsum("ij,ij->i", residual, residual))
            np.minimum.at(best, rows, distance)
            point_rows.clear()
            edge_rows.clear()

        for row, edges in enumerate(candidates):
            for offset in range(0, len(edges), 65536):
                chunk = edges[offset:offset + 65536]
                point_rows.extend([row] * len(chunk))
                edge_rows.extend(chunk)
                if len(edge_rows) >= 65536:
                    judge()
        judge()
        selected = np.flatnonzero(best <= threshold)
        if len(selected):
            chosen.append(selected + start)
            distances.append(best[selected])
    return (np.concatenate(chosen), np.concatenate(distances)) if chosen else (
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))


def nearest_wire_points(xyz: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Closest points for displayed danger samples only; bounded 64 x 1024 tiles."""
    closest = np.empty_like(xyz, dtype=np.float64)
    for start in range(0, len(xyz), 64):
        block = xyz[start:start + 64]
        best = np.full(len(block), np.inf)
        found = np.empty_like(block, dtype=np.float64)
        for edge_start in range(0, len(a), 1024):
            s, e = a[edge_start:edge_start + 1024], b[edge_start:edge_start + 1024]
            direction = e - s
            denominator = np.einsum('ij,ij->i', direction, direction)
            delta = block[:, None, :] - s
            parameter = np.einsum('ijk,jk->ij', delta, direction) / np.where(denominator > 0, denominator, 1)
            np.clip(parameter, 0, 1, out=parameter)
            projected = s + parameter[:, :, None] * direction
            residual = block[:, None, :] - projected
            square = np.einsum('ijk,ijk->ij', residual, residual)
            edge = np.argmin(square, axis=1)
            value = square[np.arange(len(block)), edge]
            better = value < best
            found[better] = projected[np.arange(len(block)), edge][better]
            best[better] = value[better]
        closest[start:start + len(block)] = found
    return closest


def build_threshold_danger_display(result: Any, *, visible_tree_ids: Iterable[int],
                                  display_budget: int = 20000,
                                  cache: dict[str, Any] | None = None,
                                  reconciled_override: Any | None = None) -> dict[str, Any]:
    snapshot = getattr(result, "risk_policy_snapshot", {}) or {}
    threshold = RiskPolicy(warning_clearance_threshold_m=snapshot.get(
        "warning_clearance_threshold_m", 0.5)).warning_clearance_threshold_m
    reconciled = reconciled_override or getattr(result, "global_reconciled", None)
    if reconciled is None:
        return _empty("UNAVAILABLE", threshold)
    # Identity holds strong references in the cache: no id recycling; policy
    # changes invalidate only the display mask, not science or wire geometry.
    wires = tuple((getattr(result, "span_results", {}) or {}).values())
    if cache is not None and cache.get("geometry") is reconciled and cache.get("threshold") == threshold and all(
        old is new for old, new in zip(cache.get("wires", ()), wires)
    ) and len(cache.get("wires", ())) == len(wires):
        selected, distance = cache["selected"], cache["distance"]
        a, b = cache['a'], cache['b']
    else:
        a, b = _edges(result)
        if not len(a):
            return _empty("NO_DISTANCE_READY_WIRE", threshold)
        selected, distance = threshold_membership(reconciled.final_xyz_local_m, a, b, threshold)
        if cache is not None:
            cache.update(geometry=reconciled, wires=wires, threshold=threshold,
                         selected=selected, distance=distance, a=a, b=b)
    ids = np.asarray(reconciled.final_tree_id)[selected]
    visible = np.isin(ids, list(visible_tree_ids))
    selected, distance, ids = selected[visible], distance[visible], ids[visible]
    full_count = len(selected)
    budget = max(1, int(display_budget))
    sample = np.linspace(0, full_count - 1, budget, dtype=np.int64) if full_count > budget else np.arange(full_count)
    rows = selected[sample]
    xyz = np.asarray(reconciled.final_xyz_local_m)[rows]
    nearest = nearest_wire_points(xyz, a, b)
    return dict(xyz=xyz.tolist(), wire_xyz=nearest.tolist(),
                tree_id_by_index=ids[sample].tolist(),
                original_index=np.asarray(reconciled.final_original_index)[rows].tolist(),
                distance_m=distance[sample].tolist(), threshold_m=threshold,
                full_point_count=full_count, display_point_count=len(rows),
                authority="EXACT_CURRENT_THRESHOLD_FULL_RESOLUTION",
                display_only_sampling=full_count > len(rows))
