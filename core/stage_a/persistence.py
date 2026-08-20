"""Sidecar persistence and explicit legacy compatibility for Stage A."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.stage_a.models import (
    STAGE_A_SCHEMA_VERSION,
    AnalysisRun,
    AnalysisRunStatus,
    CoordinateDeclaration,
    Line,
    LineRole,
    Project,
    Route,
    Scan,
    SourceDescriptor,
    Span,
    Tower,
    UnitStatus,
)

STAGE_A_FILENAME = "stage-a-project.json"


def _mapping(value: Any, name: str, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if set(value) != fields:
        raise ValueError(
            f"{name} fields do not match schema; missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )
    return value


def _coordinate(value: Any) -> CoordinateDeclaration:
    data = _mapping(
        value,
        "CoordinateDeclaration",
        {"horizontal_crs", "vertical_reference", "linear_unit", "unit_status", "coordinate_kind"},
    )
    return CoordinateDeclaration(
        data["horizontal_crs"],
        data["vertical_reference"],
        data["linear_unit"],
        UnitStatus(data["unit_status"]),
        data["coordinate_kind"],
    )


def _source(value: Any) -> SourceDescriptor:
    data = _mapping(
        value,
        "SourceDescriptor",
        {"source_id", "path", "format", "file_size_bytes", "sha256", "read_only"},
    )
    return SourceDescriptor(**dict(data))


def _line(value: Any) -> Line:
    value = dict(value)
    # Stage-A sidecars created before the main/branch topology extension are
    # still valid; absent topology fields mean a standalone main line.
    value.setdefault("line_role", LineRole.MAIN.value)
    value.setdefault("parent_line_id", None)
    data = _mapping(
        value,
        "Line",
        {
            "line_id",
            "display_name",
            "ordered_tower_ids",
            "route_geometry",
            "source_provenance",
            "line_role",
            "parent_line_id",
        },
    )
    return Line(
        data["line_id"],
        data["display_name"],
        tuple(data["ordered_tower_ids"]),
        tuple(tuple(item) for item in data["route_geometry"]),
        data["source_provenance"],
        LineRole(data["line_role"]),
        data["parent_line_id"],
    )


def _route(value: Any) -> Route:
    data = _mapping(
        value,
        "Route",
        {
            "route_id",
            "line_id",
            "display_name",
            "ordered_tower_ids",
            "route_geometry",
            "route_role",
            "parent_route_id",
            "source_provenance",
        },
    )
    return Route(
        data["route_id"],
        data["line_id"],
        data["display_name"],
        tuple(data["ordered_tower_ids"]),
        tuple(tuple(item) for item in data["route_geometry"]),
        LineRole(data["route_role"]),
        data["parent_route_id"],
        data["source_provenance"],
    )


def _tower(value: Any) -> Tower:
    data = _mapping(
        value,
        "Tower",
        {
            "tower_id",
            "external_asset_id",
            "display_name",
            "position",
            "position_crs",
            "position_source",
            "attributes",
        },
    )
    return Tower(
        data["tower_id"],
        data["external_asset_id"],
        data["display_name"],
        tuple(data["position"]),
        _coordinate(data["position_crs"]),
        data["position_source"],
        data["attributes"],
    )


def _span(value: Any) -> Span:
    value = dict(value)
    # Preserve pre-topology Stage-A sidecars with the shared-source default.
    value.setdefault(
        "source_partition",
        {"kind": "LOGICAL", "access_mode": "SHARED_SOURCE_LOCAL", "full_source_copy": False},
    )
    data = _mapping(
        value,
        "Span",
        {
            "span_id",
            "line_id",
            "start_tower_id",
            "end_tower_id",
            "core_tower_ids",
            "attributes",
            "source_partition",
        },
    )
    return Span(
        data["span_id"],
        data["line_id"],
        data["start_tower_id"],
        data["end_tower_id"],
        tuple(data["core_tower_ids"]),
        data["attributes"],
        data["source_partition"],
    )


def _scan(value: Any) -> Scan:
    data = _mapping(
        value,
        "Scan",
        {
            "scan_id",
            "source_identity",
            "point_cloud_source",
            "capability_report",
            "registration_status",
            "coverage",
            "imported_at",
            "metadata",
        },
    )
    return Scan(
        data["scan_id"],
        _source(data["source_identity"]),
        data["point_cloud_source"],
        data["capability_report"],
        data["registration_status"],
        data["coverage"],
        data["imported_at"],
        data["metadata"],
    )


def _run(value: Any) -> AnalysisRun:
    data = _mapping(
        value,
        "AnalysisRun",
        {
            "analysis_run_id",
            "scan_id",
            "span_ids",
            "algorithm_versions",
            "configuration_snapshot",
            "risk_policy_snapshot",
            "status",
            "artifact_references",
            "created_at",
        },
    )
    return AnalysisRun(
        data["analysis_run_id"],
        data["scan_id"],
        tuple(data["span_ids"]),
        data["algorithm_versions"],
        data["configuration_snapshot"],
        data["risk_policy_snapshot"],
        AnalysisRunStatus(data["status"]),
        data["artifact_references"],
        data["created_at"],
    )


def project_from_dict(value: Mapping[str, Any]) -> Project:
    value = dict(value)
    value.setdefault("routes", [])
    data = _mapping(
        value,
        "Project",
        {
            "project_id",
            "schema_version",
            "name",
            "created_at",
            "metadata",
            "lines",
            "towers",
            "spans",
            "scans",
            "analysis_runs",
            "basemap_policy",
            "routes",
        },
    )
    if data["schema_version"] != STAGE_A_SCHEMA_VERSION:
        raise ValueError("unsupported Stage-A project schema version")
    return Project(
        data["project_id"],
        data["schema_version"],
        data["name"],
        data["created_at"],
        data["metadata"],
        tuple(_line(item) for item in data["lines"]),
        tuple(_tower(item) for item in data["towers"]),
        tuple(_span(item) for item in data["spans"]),
        tuple(_scan(item) for item in data["scans"]),
        tuple(_run(item) for item in data["analysis_runs"]),
        data["basemap_policy"],
        tuple(_route(item) for item in data["routes"]),
    )


@dataclass(frozen=True, slots=True)
class LegacyProjectCompatibility:
    """Read-only marker returned for legacy workspaces until explicit migration."""

    path: str
    source_schema: str
    project_id: str | None
    name: str | None
    migration_required: bool = True


def save_project(project: Project, root: Path | str) -> Path:
    if type(project) is not Project:
        raise ValueError("save_project expects a Stage-A Project")
    project_root = Path(root).expanduser().resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    target = project_root / STAGE_A_FILENAME
    payload = {"schema_version": STAGE_A_SCHEMA_VERSION, "project": project.to_dict()}
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target


def load_project(root: Path | str) -> Project:
    project_root = Path(root).expanduser().resolve()
    target = project_root / STAGE_A_FILENAME
    if not target.is_file():
        raise FileNotFoundError(f"Stage-A sidecar is absent: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        envelope = _mapping(payload, "Stage-A envelope", {"schema_version", "project"})
        if envelope["schema_version"] != STAGE_A_SCHEMA_VERSION:
            raise ValueError("unsupported Stage-A envelope version")
        return project_from_dict(envelope["project"])
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid Stage-A project sidecar: {target}") from exc


def open_project(root: Path | str) -> Project | LegacyProjectCompatibility:
    """Open Stage A, or explicitly identify a legacy project without reinterpreting it."""

    project_root = Path(root).expanduser().resolve()
    if (project_root / STAGE_A_FILENAME).is_file():
        return load_project(project_root)
    legacy = project_root / "project.json"
    desktop = project_root / "desktop-project.json"
    if legacy.is_file():
        try:
            payload = json.loads(legacy.read_text(encoding="utf-8"))
            project = payload.get("project", {}) if isinstance(payload, dict) else {}
            return LegacyProjectCompatibility(
                str(project_root),
                str(payload.get("schema_version", "legacy")),
                project.get("project_id"),
                project.get("project_name"),
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            return LegacyProjectCompatibility(str(project_root), "legacy-invalid", None, None)
    if desktop.is_file():
        try:
            payload = json.loads(desktop.read_text(encoding="utf-8"))
            return LegacyProjectCompatibility(
                str(project_root),
                str(payload.get("schema_version", "legacy-desktop")),
                payload.get("project_id"),
                payload.get("project_name"),
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            return LegacyProjectCompatibility(
                str(project_root), "legacy-desktop-invalid", None, None
            )
    raise FileNotFoundError(f"no TreeLineWarning project manifest found under {project_root}")


def migrate_legacy_project(root: Path | str, *, name: str | None = None) -> Project:
    """Create a Stage-A sidecar only after an explicit migration call."""

    opened = open_project(root)
    if not isinstance(opened, LegacyProjectCompatibility):
        return opened
    project = Project.new(
        name or opened.name or "Migrated TreeLineWarning Project",
        metadata={
            "compatibility": {
                "source_schema": opened.source_schema,
                "source_path": opened.path,
                "explicit_migration": True,
            }
        },
    )
    save_project(project, root)
    return project


__all__ = [
    "LegacyProjectCompatibility",
    "STAGE_A_FILENAME",
    "load_project",
    "migrate_legacy_project",
    "open_project",
    "project_from_dict",
    "save_project",
]
