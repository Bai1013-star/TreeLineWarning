"""Fail-closed GIS route-to-tower association for the desktop corridor workflow.

This module only creates logical topology. It never promotes a GIS seed to an
operator-confirmed LiDAR position and it never reads the point cloud. Distances
are evaluated in a local metre CRS built with bundled, network-disabled PROJ.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from core.stage_a import (
    Project,
    Route,
    Tower,
    build_anchor_metric_transform,
    build_crs_transform,
    generate_spans_from_ordered_towers,
    transform_xy,
)
from core.stage_a.models import Line, Span, stable_id

DEFAULT_ROUTE_ASSIGNMENT_MAX_DISTANCE_M = 100.0
DEFAULT_ROUTE_AMBIGUITY_MARGIN_M = 10.0
_REBUILDABLE_MEMBERSHIP = {
    "DEFERRED_UNTIL_REGISTRATION",
    "SPATIAL_METRIC_ASSIGNMENT_V1",
    "SPATIAL_ASSIGNMENT_BLOCKED_V1",
}


def _source(value: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw = value.get("gis_source")
    return raw if isinstance(raw, Mapping) else None


def _same_source(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> bool:
    if left is None or right is None:
        return False
    return (
        str(left.get("path")) == str(right.get("path"))
        and str(left.get("sha256")) == str(right.get("sha256"))
    )


def _route_crs(line: Line) -> str:
    raw = line.source_provenance.get("route_position_crs")
    if not isinstance(raw, Mapping):
        raise ValueError("ROUTE_CRS_REQUIRED_FOR_SPATIAL_ASSIGNMENT")
    value = raw.get("horizontal_crs")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ROUTE_CRS_REQUIRED_FOR_SPATIAL_ASSIGNMENT")
    return value.strip()


def _tower_seed_position(tower: Tower) -> tuple[float, float, float]:
    raw = tower.attributes.get("gis_original_position")
    if isinstance(raw, Sequence) and not isinstance(raw, str | bytes) and len(raw) == 3:
        return tuple(float(item) for item in raw)  # type: ignore[return-value]
    if tower.position_source.startswith("GIS:"):
        return tuple(float(item) for item in tower.position)
    raise ValueError("TOWER_GIS_ORIGINAL_POSITION_REQUIRED")


def _tower_seed_crs(tower: Tower) -> str:
    raw = tower.attributes.get("gis_original_position_crs")
    if isinstance(raw, Mapping):
        value = raw.get("horizontal_crs")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if tower.position_source.startswith("GIS:"):
        value = tower.position_crs.horizontal_crs
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("TOWER_GIS_CRS_REQUIRED_FOR_SPATIAL_ASSIGNMENT")


def _polyline_projection(
    geometry: Sequence[tuple[float, float]],
    point: tuple[float, float],
) -> tuple[float, float]:
    if len(geometry) < 2:
        raise ValueError("ROUTE_GEOMETRY_REQUIRES_TWO_POINTS")
    best_distance = math.inf
    best_chainage = 0.0
    chainage = 0.0
    px, py = point
    for first, second in zip(geometry, geometry[1:], strict=False):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length_sq = dx * dx + dy * dy
        if length_sq <= 0.0:
            continue
        length = math.sqrt(length_sq)
        t = ((px - first[0]) * dx + (py - first[1]) * dy) / length_sq
        t = min(1.0, max(0.0, t))
        qx = first[0] + t * dx
        qy = first[1] + t * dy
        distance = math.hypot(px - qx, py - qy)
        if distance < best_distance:
            best_distance = distance
            best_chainage = chainage + t * length
        chainage += length
    if not math.isfinite(best_distance):
        raise ValueError("ROUTE_GEOMETRY_HAS_NO_NONZERO_SEGMENT")
    return best_distance, best_chainage


def _metric_route(line: Line) -> tuple[Any, tuple[tuple[float, float], ...]]:
    if len(line.route_geometry) < 2:
        raise ValueError("ROUTE_GEOMETRY_REQUIRES_TWO_POINTS")
    crs = _route_crs(line)
    plan = build_anchor_metric_transform(crs, line.route_geometry[0])
    xy = transform_xy(plan, tuple(item[:2] for item in line.route_geometry))
    return plan, xy


def _tower_in_route_metric(
    tower: Tower,
    route_crs: str,
    route_metric_plan: Any,
) -> tuple[float, float]:
    seed = _tower_seed_position(tower)
    tower_crs = _tower_seed_crs(tower)
    xy = seed[:2]
    if tower_crs != route_crs:
        xy = transform_xy(build_crs_transform(tower_crs, route_crs), (xy,))[0]
    return transform_xy(route_metric_plan, (xy,))[0]


def _span_factory(line: Line, existing: Sequence[Span]):
    pairs = list(zip(line.ordered_tower_ids, line.ordered_tower_ids[1:], strict=False))
    existing_ids = {
        (span.start_tower_id, span.end_tower_id): span.span_id
        for span in existing
        if span.line_id == line.line_id
    }

    def factory(_owner: str, index: int) -> str:
        pair = pairs[index]
        return existing_ids.get(pair, stable_id("span"))

    return factory


def rebuild_deferred_gis_topology(
    project: Project,
    *,
    maximum_distance_m: float = DEFAULT_ROUTE_ASSIGNMENT_MAX_DISTANCE_M,
    ambiguity_margin_m: float = DEFAULT_ROUTE_AMBIGUITY_MARGIN_M,
) -> tuple[Project, dict[str, Any]]:
    """Assign/reassign GIS towers to route geometry and order them by chainage.

    Ambiguity is route-fatal for the affected routes. We deliberately do not
    bridge around an unresolved tower because that could manufacture a false
    long Span between its neighbours.
    """

    if type(project) is not Project:
        raise ValueError("project must be a Project")
    if maximum_distance_m <= 0.0 or not math.isfinite(maximum_distance_m):
        raise ValueError("maximum_distance_m must be positive and finite")
    if ambiguity_margin_m < 0.0 or not math.isfinite(ambiguity_margin_m):
        raise ValueError("ambiguity_margin_m must be finite and non-negative")

    rebuildable = [
        line
        for line in project.lines
        if line.source_provenance.get("tower_membership") in _REBUILDABLE_MEMBERSHIP
        and len(line.route_geometry) >= 2
    ]
    if not rebuildable:
        return project, {
            "status": "NO_DEFERRED_ROUTES",
            "assigned_tower_count": 0,
            "ambiguous_tower_ids": [],
            "unassigned_tower_ids": [],
            "blocked_route_ids": [],
        }

    route_cache: dict[str, tuple[Any, tuple[tuple[float, float], ...], str]] = {}
    for line in rebuildable:
        plan, metric = _metric_route(line)
        route_cache[line.line_id] = (plan, metric, _route_crs(line))

    assigned: dict[str, list[tuple[float, str]]] = {line.line_id: [] for line in rebuildable}
    ambiguous: list[str] = []
    unassigned: list[str] = []
    blocked_routes: set[str] = set()
    considered = 0

    for tower in project.towers:
        tower_source = _source(tower.attributes)
        candidate_lines = [
            line
            for line in rebuildable
            if _same_source(tower_source, _source(line.source_provenance))
        ]
        if not candidate_lines:
            continue
        considered += 1
        scores: list[tuple[float, float, str]] = []
        try:
            for line in candidate_lines:
                plan, metric_geometry, route_crs = route_cache[line.line_id]
                tower_metric = _tower_in_route_metric(tower, route_crs, plan)
                distance, chainage = _polyline_projection(metric_geometry, tower_metric)
                scores.append((distance, chainage, line.line_id))
        except (TypeError, ValueError):
            unassigned.append(tower.tower_id)
            continue
        scores.sort(key=lambda item: (item[0], item[2]))
        best = scores[0]
        if best[0] > maximum_distance_m:
            unassigned.append(tower.tower_id)
            continue
        if len(scores) > 1 and scores[1][0] - best[0] < ambiguity_margin_m:
            ambiguous.append(tower.tower_id)
            # Both plausible routes are blocked. This prevents T1→T3 spans
            # when an unresolved T2 sits between two corridor candidates.
            blocked_routes.update(item[2] for item in scores[:2])
            continue
        assigned[best[2]].append((best[1], tower.tower_id))

    line_updates: dict[str, Line] = {}
    for line in rebuildable:
        provenance = dict(line.source_provenance)
        if line.line_id in blocked_routes:
            provenance["tower_membership"] = "SPATIAL_ASSIGNMENT_BLOCKED_V1"
            provenance["route_assignment_v1"] = {
                "maximum_distance_m": float(maximum_distance_m),
                "ambiguity_margin_m": float(ambiguity_margin_m),
                "assigned_tower_count": 0,
                "blocked_by_ambiguity": True,
            }
            line_updates[line.line_id] = replace(
                line,
                ordered_tower_ids=(),
                source_provenance=provenance,
            )
            continue
        ordered = tuple(
            tower_id
            for _chainage, tower_id in sorted(
                assigned[line.line_id], key=lambda item: (item[0], item[1])
            )
        )
        provenance["tower_membership"] = "SPATIAL_METRIC_ASSIGNMENT_V1"
        provenance["route_assignment_v1"] = {
            "maximum_distance_m": float(maximum_distance_m),
            "ambiguity_margin_m": float(ambiguity_margin_m),
            "assigned_tower_count": len(ordered),
            "blocked_by_ambiguity": False,
        }
        line_updates[line.line_id] = replace(
            line,
            ordered_tower_ids=ordered,
            source_provenance=provenance,
        )

    updated_lines = tuple(line_updates.get(line.line_id, line) for line in project.lines)
    updated_routes: list[Route] = []
    for route in project.routes:
        line = line_updates.get(route.line_id)
        if line is None:
            updated_routes.append(route)
            continue
        provenance = dict(route.source_provenance)
        provenance.update(line.source_provenance)
        updated_routes.append(
            replace(
                route,
                ordered_tower_ids=line.ordered_tower_ids,
                source_provenance=provenance,
            )
        )

    processed_line_ids = set(line_updates)
    retained_spans = [span for span in project.spans if span.line_id not in processed_line_ids]
    generated: list[Span] = []
    for line in updated_lines:
        if (
            line.line_id not in processed_line_ids
            or line.line_id in blocked_routes
            or len(line.ordered_tower_ids) < 2
        ):
            continue
        generated.extend(
            generate_spans_from_ordered_towers(
                line,
                span_id_factory=_span_factory(line, project.spans),
            )
        )

    diagnostics = {
        "status": "REQUIRES_CONFIRMATION" if blocked_routes else "ASSIGNED",
        "considered_tower_count": considered,
        "assigned_tower_count": sum(
            len(items) for line_id, items in assigned.items() if line_id not in blocked_routes
        ),
        "route_count": len(rebuildable),
        "generated_span_count": len(generated),
        "ambiguous_tower_ids": sorted(ambiguous),
        "unassigned_tower_ids": sorted(unassigned),
        "blocked_route_ids": sorted(blocked_routes),
        "maximum_distance_m": float(maximum_distance_m),
        "ambiguity_margin_m": float(ambiguity_margin_m),
        "authority": "GIS_TOPOLOGY_ONLY_NOT_SCIENCE_CONFIRMATION",
    }
    metadata = dict(project.metadata)
    metadata["corridor_topology_v1"] = diagnostics
    return (
        replace(
            project,
            metadata=metadata,
            lines=updated_lines,
            routes=tuple(updated_routes),
            spans=tuple((*retained_spans, *generated)),
        ),
        diagnostics,
    )


__all__ = [
    "DEFAULT_ROUTE_AMBIGUITY_MARGIN_M",
    "DEFAULT_ROUTE_ASSIGNMENT_MAX_DISTANCE_M",
    "rebuild_deferred_gis_topology",
]
