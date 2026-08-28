"""Idempotent canonical GIS ingestion into the Stage-A project topology."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.stage_a import (
    CanonicalGISImport,
    Line,
    LineRole,
    Project,
    Route,
    Tower,
    generate_spans_from_ordered_towers,
    import_gis,
    load_project,
    save_project,
)
from core.stage_a.models import stable_id


def _source_from_mapping(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    source = value.get("gis_source")
    return source if isinstance(source, Mapping) else None


def _source_matches(
    value: Mapping[str, Any] | None,
    source: Mapping[str, Any],
    *,
    require_sha: bool,
) -> bool:
    if value is None or str(value.get("path")) != str(source.get("path")):
        return False
    return not require_sha or str(value.get("sha256")) == str(source.get("sha256"))


def _history_contains(project: Project, source: Mapping[str, Any]) -> bool:
    history = project.metadata.get("gis_import_history")
    if not isinstance(history, list):
        return False
    return any(
        isinstance(item, Mapping)
        and _source_matches(item.get("source"), source, require_sha=True)
        for item in history
    )


def _already_imported(project: Project, imported: CanonicalGISImport) -> bool:
    source = imported.source.to_dict()
    if _history_contains(project, source):
        return True
    return any(
        _source_matches(
            _source_from_mapping(line.source_provenance),
            source,
            require_sha=True,
        )
        for line in project.lines
    )


def _remove_previous_revision(project: Project, imported: CanonicalGISImport) -> Project:
    """Replace an older revision from the same absolute GIS source path."""

    source = imported.source.to_dict()
    old_lines = {
        line.line_id
        for line in project.lines
        if _source_matches(
            _source_from_mapping(line.source_provenance),
            source,
            require_sha=False,
        )
        and not _source_matches(
            _source_from_mapping(line.source_provenance),
            source,
            require_sha=True,
        )
    }
    old_towers_from_source = {
        tower.tower_id
        for tower in project.towers
        if _source_matches(
            _source_from_mapping(tower.attributes),
            source,
            require_sha=False,
        )
        and not _source_matches(
            _source_from_mapping(tower.attributes),
            source,
            require_sha=True,
        )
    }
    if not old_lines and not old_towers_from_source:
        return project

    for tower in project.towers:
        if tower.tower_id not in old_towers_from_source:
            continue
        if not tower.position_source.startswith("GIS:"):
            raise ValueError("GIS_SOURCE_REVISION_REQUIRES_CONFIRMATION")

    old_routes = {route.route_id for route in project.routes if route.line_id in old_lines}
    candidate_towers = {
        tower_id
        for line in project.lines
        if line.line_id in old_lines
        for tower_id in line.ordered_tower_ids
    }
    candidate_towers.update(old_towers_from_source)
    retained_lines = tuple(line for line in project.lines if line.line_id not in old_lines)
    retained_line_towers = {
        tower_id for line in retained_lines for tower_id in line.ordered_tower_ids
    }
    removable_towers = candidate_towers - retained_line_towers
    return replace(
        project,
        lines=retained_lines,
        routes=tuple(route for route in project.routes if route.route_id not in old_routes),
        spans=tuple(span for span in project.spans if span.line_id not in old_lines),
        towers=tuple(
            tower for tower in project.towers if tower.tower_id not in removable_towers
        ),
    )


def _existing_line_ids_by_external(project: Project) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in project.lines:
        value = line.source_provenance.get("external_route_id")
        if isinstance(value, str) and value:
            result[value] = line.line_id
    return result


def _existing_route_ids_by_external(project: Project) -> dict[str, str]:
    result: dict[str, str] = {}
    for route in project.routes:
        value = route.source_provenance.get("external_route_id")
        if isinstance(value, str) and value:
            result[value] = route.route_id
    return result


def apply_gis_import(project: Project, imported: CanonicalGISImport) -> Project:
    """Apply one canonical GIS import without duplicating an existing revision."""

    if _already_imported(project, imported):
        return project
    project = _remove_previous_revision(project, imported)
    source = imported.source.to_dict()

    towers = list(project.towers)
    tower_by_external = {
        tower.external_asset_id: tower for tower in towers if tower.external_asset_id
    }
    imported_tower_ids: list[str] = []
    for seed in imported.towers:
        tower = tower_by_external.get(seed.external_asset_id)
        if tower is None:
            attributes = dict(seed.attributes)
            attributes.update(
                {
                    "gis_source": source,
                    "gis_original_position": list(seed.position),
                }
            )
            tower = Tower(
                tower_id=stable_id("tower"),
                external_asset_id=seed.external_asset_id,
                display_name=seed.display_name,
                position=seed.position,
                position_crs=seed.position_crs,
                position_source=f"GIS:{imported.source.format}",
                attributes=attributes,
            )
            towers.append(tower)
            tower_by_external[seed.external_asset_id] = tower
        imported_tower_ids.append(tower.tower_id)

    lines = list(project.lines)
    routes = list(project.routes)
    spans = list(project.spans)
    line_ids = {route.external_route_id: stable_id("line") for route in imported.routes}
    route_ids = {route.external_route_id: stable_id("route") for route in imported.routes}
    external_line_ids = _existing_line_ids_by_external(project)
    external_line_ids.update(line_ids)
    external_route_ids = _existing_route_ids_by_external(project)
    external_route_ids.update(route_ids)

    for seed in imported.routes:
        ordered = tuple(
            tower_by_external[item].tower_id
            for item in seed.ordered_external_asset_ids
            if item in tower_by_external
        )
        parent_line_id = (
            external_line_ids.get(seed.parent_external_route_id)
            if seed.parent_external_route_id
            else None
        )
        parent_route_id = (
            external_route_ids.get(seed.parent_external_route_id)
            if seed.parent_external_route_id
            else None
        )
        if seed.route_role is LineRole.BRANCH and (
            parent_line_id is None or parent_route_id is None
        ):
            raise ValueError("BRANCH_PARENT_ROUTE_NOT_FOUND")
        provenance = {
            "gis_source": source,
            "external_route_id": seed.external_route_id,
            "route_position_crs": seed.position_crs.to_dict() if seed.position_crs else None,
            "tower_membership": (
                "EXPLICIT_FROM_SOURCE" if ordered else "DEFERRED_UNTIL_REGISTRATION"
            ),
        }
        line = Line(
            line_ids[seed.external_route_id],
            seed.display_name,
            ordered,
            seed.geometry,
            provenance,
            seed.route_role,
            parent_line_id,
        )
        lines.append(line)
        routes.append(
            Route(
                route_ids[seed.external_route_id],
                line.line_id,
                seed.display_name,
                ordered,
                seed.geometry,
                seed.route_role,
                parent_route_id,
                provenance,
            )
        )
        spans.extend(generate_spans_from_ordered_towers(line))

    if not imported.routes and imported_tower_ids:
        provenance = {
            "gis_source": source,
            "tower_membership": "EXPLICIT_ROW_ORDER",
        }
        line = Line(
            stable_id("line"),
            "GIS Towers",
            tuple(imported_tower_ids),
            (),
            provenance,
        )
        lines.append(line)
        routes.append(
            Route(
                stable_id("route"),
                line.line_id,
                line.display_name,
                line.ordered_tower_ids,
                (),
                LineRole.MAIN,
                None,
                provenance,
            )
        )
        spans.extend(generate_spans_from_ordered_towers(line))

    metadata = dict(project.metadata)
    metadata["gis_import"] = imported.report.to_dict()
    history = list(metadata.get("gis_import_history") or [])
    history.append(imported.report.to_dict())
    metadata["gis_import_history"] = history
    return replace(
        project,
        metadata=metadata,
        lines=tuple(lines),
        routes=tuple(routes),
        towers=tuple(towers),
        spans=tuple(spans),
    )


def import_gis_into_project(
    project_root: Path | str,
    source: Path | str,
    **kwargs: Any,
) -> Project:
    """Read a GIS source, apply it idempotently, and persist Stage-A topology."""

    root = Path(project_root).resolve()
    project = load_project(root)
    imported = import_gis(source, **kwargs)
    updated = apply_gis_import(project, imported)
    if updated is not project:
        save_project(updated, root)
    return updated


__all__ = ["apply_gis_import", "import_gis_into_project"]
