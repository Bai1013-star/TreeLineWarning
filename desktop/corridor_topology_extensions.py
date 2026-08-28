"""Desktop integration for GIS route assignment and ready-span execution."""

from __future__ import annotations

from functools import wraps
from typing import Any

from core.stage_a import save_project
from desktop.corridor_topology import (
    DEFAULT_ROUTE_AMBIGUITY_MARGIN_M,
    DEFAULT_ROUTE_ASSIGNMENT_MAX_DISTANCE_M,
    rebuild_deferred_gis_topology,
)

_INSTALLED = False


def install_corridor_topology_extensions() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from desktop.workflow import DesktopWorkflow

    original_prepare = DesktopWorkflow.prepare_corridor_registration
    original_set_anchor = DesktopWorkflow.set_corridor_anchor
    original_project_projection = DesktopWorkflow.project_projection

    def rebuild_corridor_topology(
        self: Any,
        payload: Any | None = None,
    ) -> dict[str, Any]:
        options = payload if isinstance(payload, dict) else {}
        maximum_distance_m = float(
            options.get(
                "maximum_distance_m",
                DEFAULT_ROUTE_ASSIGNMENT_MAX_DISTANCE_M,
            )
        )
        ambiguity_margin_m = float(
            options.get(
                "ambiguity_margin_m",
                DEFAULT_ROUTE_AMBIGUITY_MARGIN_M,
            )
        )
        project = self._stage_project()
        updated, diagnostics = rebuild_deferred_gis_topology(
            project,
            maximum_distance_m=maximum_distance_m,
            ambiguity_margin_m=ambiguity_margin_m,
        )
        if updated != project:
            save_project(updated, self._root)
        self._log(
            "corridor_topology_rebuilt",
            status=diagnostics["status"],
            assigned_tower_count=diagnostics.get("assigned_tower_count", 0),
            generated_span_count=diagnostics.get("generated_span_count", 0),
            ambiguous_tower_ids=diagnostics.get("ambiguous_tower_ids", []),
        )
        return diagnostics

    @wraps(original_prepare)
    def prepare_with_topology(self: Any) -> dict[str, Any]:
        registration = original_prepare(self)
        if registration.get("ready_for_science"):
            rebuild_corridor_topology(self)
        return self.corridor_registration_projection()

    @wraps(original_set_anchor)
    def set_anchor_with_topology(self: Any, payload: Any) -> dict[str, Any]:
        registration = original_set_anchor(self, payload)
        if registration.get("ready_for_science"):
            rebuild_corridor_topology(self)
        return self.corridor_registration_projection()

    @wraps(original_project_projection)
    def project_projection_with_topology(self: Any) -> dict[str, Any]:
        projection = original_project_projection(self)
        try:
            project = self._stage_project()
        except (FileNotFoundError, ValueError):
            return projection
        diagnostics = project.metadata.get("corridor_topology_v1")
        projection["corridor_topology"] = (
            dict(diagnostics) if isinstance(diagnostics, dict) else None
        )
        readiness = self.span_readiness_projection()
        projection["span_readiness_summary"] = {
            "ready": sum(item["status"] == "READY" for item in readiness),
            "blocked": sum(item["status"] != "READY" for item in readiness),
            "total": len(readiness),
        }
        return projection

    def start_ready_batch(
        self: Any,
        span_ids: list[str] | None = None,
        *,
        mode: str = "REAL",
    ) -> dict[str, Any]:
        readiness = self.span_readiness_projection()
        requested = set(str(item) for item in span_ids) if span_ids else None
        ready = [
            item["span_id"]
            for item in readiness
            if item["status"] == "READY"
            and (requested is None or item["span_id"] in requested)
        ]
        blocked = [
            item["span_id"]
            for item in readiness
            if item["status"] != "READY"
            and (requested is None or item["span_id"] in requested)
        ]
        if not ready:
            raise ValueError("NO_READY_SPANS_FOR_BATCH")
        batch = self.start_batch(ready, mode=mode)
        return {
            "batch": batch,
            "selected_ready_span_ids": ready,
            "blocked_span_ids": blocked,
            "selection_policy": "READY_ONLY_EXPLICITLY_REPORTED",
        }

    DesktopWorkflow.rebuild_corridor_topology = rebuild_corridor_topology
    DesktopWorkflow.prepare_corridor_registration = prepare_with_topology
    DesktopWorkflow.set_corridor_anchor = set_anchor_with_topology
    DesktopWorkflow.project_projection = project_projection_with_topology
    DesktopWorkflow.start_ready_batch = start_ready_batch
    _INSTALLED = True


__all__ = ["install_corridor_topology_extensions"]
