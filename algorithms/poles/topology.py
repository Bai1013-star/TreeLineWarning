"""Pole topology: corridor graph, deterministic line ordering and spans.

Poles are NOT sorted by global X or Y; a corridor may curve.  Confirmed/likely
poles are connected into a graph using pole-to-pole distance, coarse-wire
continuity along the corridor, heading compatibility with each pole's wire
evidence and top-height compatibility.  Connected chains become Lines; poles
inside each line are numbered deterministically.  Adjacent pole pairs become
Spans with a local (s, u, z) coordinate system and a core + halo ownership
model.

Only CONFIRMED / LIKELY poles define authoritative spans.  Review poles never
participate in the corridor graph.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

from algorithms.poles.candidates import PoleCandidateResult
from algorithms.poles.evidence import GroundEvidence
from algorithms.poles.validation import PoleValidationResult
from algorithms.poles.wire_evidence import CoarseWireResult

TOPOLOGY_SCHEMA = "fast-baseline-pole-topology-v1"
SPAN_SCHEMA = "fast-baseline-span-v1"


@dataclass(frozen=True, slots=True)
class LineRecord:
    """One connected, deterministically-ordered power-line corridor."""

    line_id: int
    line_name: str
    ordered_pole_candidate_ids: tuple[int, ...]
    start_xyz_absolute_m: tuple[float, float, float]
    end_xyz_absolute_m: tuple[float, float, float]
    corridor_length_m: float
    pole_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id,
            "line_name": self.line_name,
            "ordered_pole_candidate_ids": list(self.ordered_pole_candidate_ids),
            "start_xyz_absolute_m": list(self.start_xyz_absolute_m),
            "end_xyz_absolute_m": list(self.end_xyz_absolute_m),
            "corridor_length_m": self.corridor_length_m,
            "pole_count": self.pole_count,
        }


@dataclass(frozen=True, slots=True)
class SpanRecord:
    """One span between two adjacent poles in a line, with core + halo."""

    line_id: int
    line_name: str
    span_id: int
    span_name: str
    start_pole_candidate_id: int
    end_pole_candidate_id: int
    start_xyz_absolute_m: tuple[float, float, float]
    end_xyz_absolute_m: tuple[float, float, float]
    start_pole_top_z_m: float
    end_pole_top_z_m: float
    length_m: float
    direction_xy: tuple[float, float, float]
    half_width_m: float
    halo_m: float
    core_bounds_min_xyz: tuple[float, float, float]
    core_bounds_max_xyz: tuple[float, float, float]
    halo_bounds_min_xyz: tuple[float, float, float]
    halo_bounds_max_xyz: tuple[float, float, float]
    quality: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id,
            "line_name": self.line_name,
            "span_id": self.span_id,
            "span_name": self.span_name,
            "start_pole_candidate_id": self.start_pole_candidate_id,
            "end_pole_candidate_id": self.end_pole_candidate_id,
            "start_xyz_absolute_m": list(self.start_xyz_absolute_m),
            "end_xyz_absolute_m": list(self.end_xyz_absolute_m),
            "start_pole_top_z_m": self.start_pole_top_z_m,
            "end_pole_top_z_m": self.end_pole_top_z_m,
            "length_m": self.length_m,
            "direction_xy": list(self.direction_xy),
            "half_width_m": self.half_width_m,
            "halo_m": self.halo_m,
            "core_bounds_min_xyz": list(self.core_bounds_min_xyz),
            "core_bounds_max_xyz": list(self.core_bounds_max_xyz),
            "halo_bounds_min_xyz": list(self.halo_bounds_min_xyz),
            "halo_bounds_max_xyz": list(self.halo_bounds_max_xyz),
            "quality": self.quality,
        }


@dataclass(frozen=True, slots=True)
class SpanTopologyResult:
    """Lines, spans, and the underlying pole-pair edge graph."""

    lines: tuple[LineRecord, ...]
    spans: tuple[SpanRecord, ...]
    edge_pairs: tuple[tuple[int, int], ...]  # (pole_candidate_id_a, pole_candidate_id_b)
    edge_scores: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def span_count(self) -> int:
        return len(self.spans)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SpanTopologyResult:
        lines = tuple(
            LineRecord(
                line_id=int(item["line_id"]),
                line_name=str(item["line_name"]),
                ordered_pole_candidate_ids=tuple(
                    int(value) for value in item["ordered_pole_candidate_ids"]
                ),
                start_xyz_absolute_m=tuple(
                    float(value) for value in item["start_xyz_absolute_m"]
                ),
                end_xyz_absolute_m=tuple(
                    float(value) for value in item["end_xyz_absolute_m"]
                ),
                corridor_length_m=float(item["corridor_length_m"]),
                pole_count=int(item["pole_count"]),
            )
            for item in payload.get("lines", [])
        )
        spans = tuple(
            SpanRecord(
                line_id=int(item["line_id"]),
                line_name=str(item["line_name"]),
                span_id=int(item["span_id"]),
                span_name=str(item["span_name"]),
                start_pole_candidate_id=int(item["start_pole_candidate_id"]),
                end_pole_candidate_id=int(item["end_pole_candidate_id"]),
                start_xyz_absolute_m=tuple(
                    float(value) for value in item["start_xyz_absolute_m"]
                ),
                end_xyz_absolute_m=tuple(
                    float(value) for value in item["end_xyz_absolute_m"]
                ),
                start_pole_top_z_m=float(item["start_pole_top_z_m"]),
                end_pole_top_z_m=float(item["end_pole_top_z_m"]),
                length_m=float(item["length_m"]),
                direction_xy=tuple(float(value) for value in item["direction_xy"]),
                half_width_m=float(item["half_width_m"]),
                halo_m=float(item["halo_m"]),
                core_bounds_min_xyz=tuple(
                    float(value) for value in item["core_bounds_min_xyz"]
                ),
                core_bounds_max_xyz=tuple(
                    float(value) for value in item["core_bounds_max_xyz"]
                ),
                halo_bounds_min_xyz=tuple(
                    float(value) for value in item["halo_bounds_min_xyz"]
                ),
                halo_bounds_max_xyz=tuple(
                    float(value) for value in item["halo_bounds_max_xyz"]
                ),
                quality=str(item["quality"]),
            )
            for item in payload.get("spans", [])
        )
        return cls(
            lines=lines,
            spans=spans,
            edge_pairs=tuple(
                (int(pair[0]), int(pair[1])) for pair in payload.get("edge_pairs", [])
            ),
            edge_scores=tuple(dict(item) for item in payload.get("edge_scores", [])),
            diagnostics=dict(payload.get("diagnostics", {})),
        )


def _unit_xy(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.asarray((1.0, 0.0), dtype=np.float64)
    return np.asarray(vector[:2], dtype=np.float64) / norm


def _heading_alignment(direction_xy: np.ndarray, reference: np.ndarray | None) -> float | None:
    """Return |dot| of *direction_xy* with *reference* (or None when absent)."""

    if reference is None:
        return None
    ref = np.asarray(reference[:2], dtype=np.float64)
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm <= 1e-12:
        return None
    return float(abs(float(np.dot(direction_xy, ref / ref_norm))))


def _corridor_continuity(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    wire_evidence: CoarseWireResult,
    evidence: GroundEvidence,
    config: FastBaselineConfig,
) -> float:
    """Fraction of the straight corridor covered by aligned wire candidates."""

    axis = _unit_xy(end_xy - start_xy)
    length = float(np.linalg.norm(end_xy - start_xy))
    if length <= 1e-9:
        return 0.0
    candidates = np.flatnonzero(wire_evidence.candidate_mask)
    if not len(candidates):
        return 0.0
    relative = evidence.xyz_local_m[candidates, :2] - start_xy
    s = relative @ axis
    transverse = np.abs(
        relative[:, 0] * (-axis[1]) + relative[:, 1] * axis[0]
    )
    band = (s >= -0.5) & (s <= length + 0.5) & (transverse <= config.span_half_width_m)
    selected = s[band]
    if not len(selected):
        return 0.0
    selected = np.sort(selected)
    gaps = np.diff(selected)
    gap_threshold = max(4.0, 0.08 * length)
    run_starts = np.concatenate(([selected[0]], selected[1:][gaps > gap_threshold]))
    run_ends = np.concatenate((selected[:-1][gaps > gap_threshold], [selected[-1]]))
    max_run = float(np.max(run_ends - run_starts)) if len(run_starts) else 0.0
    return float(min(1.0, max_run / length))


def _top_height_compatible(
    candidate_a: Any, candidate_b: Any, config: FastBaselineConfig
) -> bool:
    tolerance = max(4.0, 0.25 * max(candidate_a.height_m, candidate_b.height_m))
    return abs(candidate_a.top_xyz_local_m[2] - candidate_b.top_xyz_local_m[2]) <= tolerance


def build_pole_graph(
    candidates: PoleCandidateResult,
    validation: PoleValidationResult,
    wire_evidence: CoarseWireResult,
    evidence: GroundEvidence,
    config: FastBaselineConfig,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]], dict[int, Any]]:
    """Return accepted pole-pair edges with deterministic scoring."""

    by_id = {candidate.pole_candidate_id: candidate for candidate in candidates.records}
    validation_by_id = {
        record.pole_candidate_id: record for record in validation.records
    }
    usable = set(validation.usable_pole_ids)
    nodes = sorted(usable)
    edges: list[tuple[int, int]] = []
    scores: list[dict[str, Any]] = []
    for first, node_a in enumerate(nodes):
        for node_b in nodes[first + 1:]:
            a = by_id[node_a]
            b = by_id[node_b]
            start_xy = np.asarray(a.base_xyz_local_m[:2], dtype=np.float64)
            end_xy = np.asarray(b.base_xyz_local_m[:2], dtype=np.float64)
            distance = float(np.linalg.norm(end_xy - start_xy))
            if not (config.pole_min_span_m <= distance <= config.pole_max_span_m):
                continue
            if not _top_height_compatible(a, b, config):
                continue
            direction = _unit_xy(end_xy - start_xy)
            record_a = validation_by_id[node_a]
            record_b = validation_by_id[node_b]
            heading_a = _heading_alignment(direction, record_a.primary_wire_direction_xy)
            heading_b = _heading_alignment(direction, record_b.primary_wire_direction_xy)
            tolerance = np.cos(config.topology_direction_tolerance_rad)
            if heading_a is not None and heading_a < tolerance:
                continue
            if heading_b is not None and heading_b < tolerance:
                continue
            continuity = _corridor_continuity(
                start_xy, end_xy, wire_evidence, evidence, config
            )
            if continuity < 0.50:
                continue
            edges.append((node_a, node_b))
            scores.append(
                {
                    "pole_a": node_a,
                    "pole_b": node_b,
                    "distance_m": distance,
                    "heading_alignment_a": heading_a,
                    "heading_alignment_b": heading_b,
                    "corridor_continuity": continuity,
                }
            )
    return edges, scores, by_id


def _connected_components(
    nodes: list[int], edges: list[tuple[int, int]]
) -> list[list[int]]:
    adjacency: dict[int, list[int]] = {node: [] for node in nodes}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    components: list[list[int]] = []
    visited: set[int] = set()
    for node in nodes:
        if node in visited:
            continue
        stack = [node]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _order_component(
    component: list[int], adjacency: dict[int, list[int]]
) -> list[int]:
    """Deterministically order a corridor component as a simple path.

    For a pure chain (all degrees <= 2) the path starts at the endpoint with
    the lexicographically smaller (x, y, z) and walks to the other endpoint.
    Branched components keep their longest simple path (by node count, ties by
    lexicographic endpoint); leftover nodes are not part of the main line.
    """

    if not component:
        return []
    ordered = sorted(component)
    endpoints = [node for node in ordered if len(adjacency[node]) == 1]
    if len(endpoints) == 2:
        start = endpoints[0]
        path = [start]
        previous: int | None = None
        current = start
        while True:
            neighbors = sorted(adjacency[current])
            nxt = next((n for n in neighbors if n != previous), None)
            if nxt is None:
                break
            path.append(nxt)
            previous, current = current, nxt
        if len(path) == len(component):
            return path
    # Fallback: longest simple path via BFS from each candidate start.
    best: list[int] = []
    for start in ordered:
        path = _longest_path_from(start, adjacency, set())
        if len(path) > len(best) or (
            len(path) == len(best) and path < best
        ):
            best = path
    return best


def _longest_path_from(
    start: int, adjacency: dict[int, list[int]], used: frozenset[int]
) -> list[int]:
    used = used | {start}
    best: list[int] = [start]
    for neighbor in sorted(adjacency[start]):
        if neighbor in used:
            continue
        path = [start] + _longest_path_from(neighbor, adjacency, used)
        if len(path) > len(best):
            best = path
    return best


def _span_bounds(
    start_xyz: np.ndarray,
    direction_xy: np.ndarray,
    length: float,
    half_width: float,
    s_lo: float,
    s_hi: float,
    z_lo: float,
    z_hi: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    perp = np.asarray((-direction_xy[1], direction_xy[0]), dtype=np.float64)
    corners_s = np.asarray((s_lo, s_lo, s_hi, s_hi), dtype=np.float64)
    corners_u = np.asarray((-half_width, half_width, -half_width, half_width), dtype=np.float64)
    corners = start_xyz[:2][None, :] + corners_s[:, None] * direction_xy[None, :] + (
        corners_u[:, None] * perp[None, :]
    )
    minimum = np.asarray((np.min(corners[:, 0]), np.min(corners[:, 1]), z_lo))
    maximum = np.asarray((np.max(corners[:, 0]), np.max(corners[:, 1]), z_hi))
    return tuple(float(v) for v in minimum), tuple(float(v) for v in maximum)


def build_span_topology(
    candidates: PoleCandidateResult,
    validation: PoleValidationResult,
    wire_evidence: CoarseWireResult,
    evidence: GroundEvidence,
    config: FastBaselineConfig,
) -> SpanTopologyResult:
    """Build lines and spans from validated poles and wire continuity."""

    started = time.perf_counter()
    edges, edge_scores, by_id = build_pole_graph(
        candidates, validation, wire_evidence, evidence, config
    )
    adjacency: dict[int, list[int]] = {}
    for node in validation.usable_pole_ids:
        adjacency[node] = []
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    components = _connected_components(list(validation.usable_pole_ids), edges)
    ordered_lines: list[list[int]] = []
    for component in components:
        if len(component) < 2:
            continue
        path = _order_component(component, adjacency)
        if len(path) >= 2:
            ordered_lines.append(path)
    # Deterministic line ordering by first pole base (x, y, z) then line length.
    def line_sort_key(path: list[int]) -> tuple[float, float, float, int]:
        base = by_id[path[0]].base_xyz_local_m
        return (-float(base[0]), -float(base[1]), -float(base[2]), int(path[0]))
    ordered_lines.sort(key=line_sort_key)

    origin = np.asarray(evidence.origin_xyz, dtype=np.float64)
    lines: list[LineRecord] = []
    spans: list[SpanRecord] = []
    for line_index, path in enumerate(ordered_lines):
        line_name = f"Line{line_index + 1:02d}"
        start_base = np.asarray(by_id[path[0]].base_xyz_local_m, dtype=np.float64)
        end_base = np.asarray(by_id[path[-1]].base_xyz_local_m, dtype=np.float64)
        corridor_length = float(np.linalg.norm(end_base[:2] - start_base[:2]))
        lines.append(
            LineRecord(
                line_id=line_index,
                line_name=line_name,
                ordered_pole_candidate_ids=tuple(path),
                start_xyz_absolute_m=tuple(
                    float(value) for value in start_base + origin
                ),
                end_xyz_absolute_m=tuple(float(value) for value in end_base + origin),
                corridor_length_m=corridor_length,
                pole_count=len(path),
            )
        )
        for span_index in range(len(path) - 1):
            start_pole_id = path[span_index]
            end_pole_id = path[span_index + 1]
            start_pole = by_id[start_pole_id]
            end_pole = by_id[end_pole_id]
            start_xy = np.asarray(start_pole.base_xyz_local_m[:2], dtype=np.float64)
            end_xy = np.asarray(end_pole.base_xyz_local_m[:2], dtype=np.float64)
            length = float(np.linalg.norm(end_xy - start_xy))
            if length < config.span_min_length_m:
                continue
            direction_xy = _unit_xy(end_xy - start_xy)
            z_lo = (
                min(float(start_pole.base_xyz_local_m[2]), float(end_pole.base_xyz_local_m[2]))
                - config.span_z_ground_clearance_m
            )
            z_hi = (
                max(float(start_pole.top_xyz_local_m[2]), float(end_pole.top_xyz_local_m[2]))
                + config.span_z_top_margin_m
            )
            core_min, core_max = _span_bounds(
                start_xy,
                direction_xy,
                length,
                config.span_half_width_m,
                -0.5,
                length + 0.5,
                z_lo,
                z_hi,
            )
            halo_min, halo_max = _span_bounds(
                start_xy,
                direction_xy,
                length,
                config.span_half_width_m,
                -config.span_halo_m,
                length + config.span_halo_m,
                z_lo,
                z_hi,
            )
            span_name = f"{line_name}-Span{span_index + 1:03d}"
            spans.append(
                SpanRecord(
                    line_id=line_index,
                    line_name=line_name,
                    span_id=span_index,
                    span_name=span_name,
                    start_pole_candidate_id=start_pole_id,
                    end_pole_candidate_id=end_pole_id,
                    start_xyz_absolute_m=tuple(
                        float(value) for value in start_xy + origin[:2]
                    ) + (float(start_pole.base_xyz_local_m[2] + origin[2]),),
                    end_xyz_absolute_m=tuple(
                        float(value) for value in end_xy + origin[:2]
                    ) + (float(end_pole.base_xyz_local_m[2] + origin[2]),),
                    start_pole_top_z_m=float(start_pole.top_xyz_local_m[2] + origin[2]),
                    end_pole_top_z_m=float(end_pole.top_xyz_local_m[2] + origin[2]),
                    length_m=length,
                    direction_xy=(float(direction_xy[0]), float(direction_xy[1]), 0.0),
                    half_width_m=config.span_half_width_m,
                    halo_m=config.span_halo_m,
                    core_bounds_min_xyz=core_min,
                    core_bounds_max_xyz=core_max,
                    halo_bounds_min_xyz=halo_min,
                    halo_bounds_max_xyz=halo_max,
                    quality="CONFIRMED_SPAN",
                )
            )
    return SpanTopologyResult(
        lines=tuple(lines),
        spans=tuple(spans),
        edge_pairs=tuple(edges),
        edge_scores=tuple(edge_scores),
        diagnostics={
            "schema": TOPOLOGY_SCHEMA,
            "usable_pole_count": len(validation.usable_pole_ids),
            "edge_candidate_pairs": len(edges),
            "line_count": len(lines),
            "span_count": len(spans),
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def span_to_suz(
    span: SpanRecord,
    xyz_absolute: np.ndarray,
    origin_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform absolute XYZ into a span-local (s, u, z) coordinate system.

    ``s`` is the longitudinal coordinate from the start pole toward the end
    pole, ``u`` the horizontal transverse offset and ``z`` the elevation.
    The start pole base is the local origin of the (s, u) plane.  ``xyz_absolute``
    and ``span.start_xyz_absolute_m`` share the same absolute frame; the local
    origin is subtracted from both.
    """

    del origin_xyz  # coordinates are already absolute in the same frame
    axis = np.asarray(span.direction_xy[:2], dtype=np.float64)
    perp = np.asarray((-axis[1], axis[0]), dtype=np.float64)
    start = np.asarray(span.start_xyz_absolute_m[:2], dtype=np.float64)
    xyz = np.asarray(xyz_absolute, dtype=np.float64)
    xy = xyz[:, :2] - start[None, :]
    s = xy @ axis
    u = xy @ perp
    z = xyz[:, 2]
    return s, u, z


def topology_to_dict(result: SpanTopologyResult) -> dict[str, Any]:
    return {
        "schema": TOPOLOGY_SCHEMA,
        "lines": [line.to_dict() for line in result.lines],
        "spans": [span.to_dict() for span in result.spans],
        "edge_pairs": [list(pair) for pair in result.edge_pairs],
        "edge_scores": [dict(score) for score in result.edge_scores],
        "diagnostics": dict(result.diagnostics),
    }


__all__ = [
    "TOPOLOGY_SCHEMA",
    "SPAN_SCHEMA",
    "LineRecord",
    "SpanRecord",
    "SpanTopologyResult",
    "build_pole_graph",
    "build_span_topology",
    "span_to_suz",
    "topology_to_dict",
]
