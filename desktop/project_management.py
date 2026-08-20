"""Desktop-only project lifecycle metadata.

This deliberately does not manufacture scientific artifacts.  A newly created
project is a valid TreeLineWarning desktop workspace with an explicit empty
state; it becomes an analysis project only after the existing ingestion and
tower workflow has produced the normal ``project.json`` contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from core.stage_a import (
    CapabilityReport,
    CapabilityState,
    CoordinateDeclaration,
    Line,
    LineRole,
    Project,
    Route,
    Scan,
    SourceDescriptor,
    Tower,
    generate_spans_from_ordered_towers,
    import_gis,
    inspect_point_cloud,
    load_project,
    save_project,
)
from core.stage_a.capability import CAPABILITY_NAMES
from core.stage_a.models import stable_id

DESKTOP_MANIFEST = "desktop-project.json"
DESKTOP_SCHEMA = "treeline-desktop-project-v1"
_INVALID_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_project_name(name: str) -> str:
    value = str(name).strip()
    if not value or value in {".", ".."} or _INVALID_NAME.search(value):
        raise ValueError("项目名称不能为空，且不能包含 Windows 非法文件字符")
    if value.rstrip(". ") != value:
        raise ValueError("项目名称不能以句点或空格结尾")
    return value


def _manifest_path(root: Path) -> Path:
    return root / DESKTOP_MANIFEST


def create_empty_project(parent: Path, project_name: str) -> dict:
    """Create an empty, non-science project workspace below ``parent``."""
    name = validate_project_name(project_name)
    root = (parent / name).resolve()
    if root.exists():
        raise ValueError("目标项目目录已存在，请选择其他名称")
    root.mkdir(parents=True)
    for relative in ("data/raw", "data/processed", "outputs", "artifacts", "exports"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    now = _now()
    payload = {
        "schema_version": DESKTOP_SCHEMA,
        "project_id": str(uuid4()),
        "project_name": name,
        "created_at": now,
        "updated_at": now,
        "lifecycle": {
            "pointcloud": "NOT_IMPORTED",
            "towers": 0,
            "lines": 0,
            "routes": 0,
            "spans": 0,
            "analysis": "NOT_STARTED",
            "prediction": "NOT_COMPUTED",
        },
        "pointcloud_source": None,
    }
    _manifest_path(root).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_project(Project.new(name), root)
    return describe_project(root)


def _read_desktop_manifest(root: Path) -> dict:
    try:
        payload = json.loads(_manifest_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("不是有效的 TreeLineWarning Desktop 项目") from exc
    if payload.get("schema_version") != DESKTOP_SCHEMA:
        raise ValueError("不支持的 Desktop 项目版本")
    if not isinstance(payload.get("project_id"), str) or not isinstance(
        payload.get("project_name"), str
    ):
        raise ValueError("Desktop 项目元数据不完整")
    return payload


def describe_project(root: Path) -> dict:
    """Return a small presentation projection for empty and analysis projects."""
    root = root.resolve()
    stage_a = root / "stage-a-project.json"
    if stage_a.is_file():
        project = load_project(root)
        scan = project.scans[-1] if project.scans else None
        capability = scan.capability_report if scan else None
        imported = scan is not None
        return {
            "kind": "analysis" if project.towers or project.spans else "empty",
            "root": str(root),
            "project_id": project.project_id,
            "project_name": project.name,
            "pointcloud": "IMPORTED_PENDING_TOWER_REGISTRATION" if imported else "NOT_IMPORTED",
            "pointcloud_source": scan.point_cloud_source if scan else None,
            "towers": len(project.towers),
            "spans": len(project.spans),
            "lines": len(project.lines),
            "routes": len(project.routes),
            "main_lines": sum(line.line_role is LineRole.MAIN for line in project.lines),
            "branch_lines": sum(line.line_role is LineRole.BRANCH for line in project.lines),
            "topology_workflow": "LINE_ROUTE_ORDERED_TOWERS_AUTO_SPANS",
            "analysis": "AVAILABLE" if project.analysis_runs else "NOT_STARTED",
            "prediction": "NOT_COMPUTED",
            "project_schema": project.schema_version,
            "capability_report": capability,
            "registration_status": scan.registration_status if scan else "UNREGISTERED",
            "source_cache_state": "NOT_PREPARED",
        }
    project_file = root / "project.json"
    if project_file.is_file():
        try:
            active = json.loads(project_file.read_text(encoding="utf-8"))
            project = active["project"]
            source = project.get("source_reference", {}).get("source_identity", {})
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("项目 project.json 无法验证") from exc
        return {
            "kind": "analysis",
            "root": str(root),
            "project_id": project["project_id"],
            "project_name": project["project_name"],
            "pointcloud": "IMPORTED",
            "pointcloud_source": source.get("resolved_access_path")
            or source.get("recorded_source_path"),
            "towers": None,
            "spans": None,
            "lines": None,
            "routes": None,
            "analysis": "AVAILABLE",
            "prediction": "AVAILABLE",
        }
    payload = _read_desktop_manifest(root)
    lifecycle = payload.get("lifecycle") or {}
    return {
        "kind": "empty",
        "root": str(root),
        "project_id": payload["project_id"],
        "project_name": payload["project_name"],
        "pointcloud": lifecycle.get("pointcloud", "NOT_IMPORTED"),
        "pointcloud_source": payload.get("pointcloud_source"),
        "towers": int(lifecycle.get("towers", 0)),
        "spans": int(lifecycle.get("spans", 0)),
        "lines": int(lifecycle.get("lines", 0)),
        "routes": int(lifecycle.get("routes", 0)),
        "analysis": lifecycle.get("analysis", "NOT_STARTED"),
        "prediction": lifecycle.get("prediction", "NOT_COMPUTED"),
    }


def import_pointcloud(root: Path, source: Path) -> dict:
    """Record a selected LAS/LAZ source without starting scientific analysis."""
    root = root.resolve()
    payload = _read_desktop_manifest(root)
    source = source.resolve()
    if source.suffix.lower() not in {".las", ".laz"} or not source.is_file():
        raise ValueError("请选择可读取的 LAS 或 LAZ 点云文件")
    payload["pointcloud_source"] = {
        "path": str(source),
        "format": source.suffix.lower()[1:].upper(),
        "file_size_bytes": source.stat().st_size,
        "selected_at": _now(),
    }
    payload["lifecycle"] = {
        "pointcloud": "IMPORTED_PENDING_TOWER_REGISTRATION",
        "towers": 0,
        "lines": 0,
        "routes": 0,
        "spans": 0,
        "analysis": "NOT_STARTED",
        "prediction": "NOT_COMPUTED",
    }
    payload["updated_at"] = _now()
    _manifest_path(root).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        report = inspect_point_cloud(source)
    except (ImportError, OSError, ValueError, RuntimeError):
        source_descriptor = SourceDescriptor(
            source_id=f"source-{uuid4().hex}",
            path=str(source),
            format=source.suffix.lower()[1:].upper(),
            file_size_bytes=source.stat().st_size,
        )
        report = CapabilityReport(
            source_descriptor,
            {name: CapabilityState.UNKNOWN for name in CAPABILITY_NAMES},
            CoordinateDeclaration.unresolved(),
            {"diagnostic": "header inspection unavailable; explicit confirmation required"},
        )
    stage_project = load_project(root)
    scan = Scan(
        scan_id=stable_id("scan"),
        source_identity=report.source,
        point_cloud_source=str(source.resolve()),
        capability_report=report.to_dict(),
        registration_status="UNREGISTERED",
        coverage=report.details,
        metadata={"offline_runtime": True},
    )
    save_project(replace(stage_project, scans=(*stage_project.scans, scan)), root)
    return describe_project(root)


def import_gis_file(root: Path, source: Path, **kwargs) -> dict:
    """Import canonical GIS seeds and persist only product topology metadata."""

    project_root = root.resolve()
    project = load_project(project_root)
    imported = import_gis(source, **kwargs)
    existing_towers = list(project.towers)
    tower_by_external = {
        tower.external_asset_id: tower for tower in existing_towers if tower.external_asset_id
    }
    for seed in imported.towers:
        if seed.external_asset_id in tower_by_external:
            continue
        tower = Tower(
            tower_id=stable_id("tower"),
            external_asset_id=seed.external_asset_id,
            display_name=seed.display_name,
            position=seed.position,
            position_crs=seed.position_crs,
            position_source=f"GIS:{imported.source.format}",
            attributes=seed.attributes,
        )
        existing_towers.append(tower)
        tower_by_external[seed.external_asset_id] = tower
    lines = list(project.lines)
    routes = list(project.routes)
    spans = list(project.spans)
    line_ids = {route.external_route_id: stable_id("line") for route in imported.routes}
    route_ids = {route.external_route_id: stable_id("route") for route in imported.routes}
    for route in imported.routes:
        ordered = tuple(
            tower_by_external[item].tower_id
            for item in route.ordered_external_asset_ids
            if item in tower_by_external
        )
        line = Line(
            line_ids[route.external_route_id],
            route.display_name,
            ordered,
            route.geometry,
            {"gis_source": imported.source.to_dict(), "external_route_id": route.external_route_id},
            route.route_role,
            line_ids.get(route.parent_external_route_id),
        )
        lines.append(line)
        routes.append(
            Route(
                route_ids[route.external_route_id],
                line.line_id,
                route.display_name,
                ordered,
                route.geometry,
                route.route_role,
                route_ids.get(route.parent_external_route_id),
                {
                    "gis_source": imported.source.to_dict(),
                    "external_route_id": route.external_route_id,
                },
            )
        )
        spans.extend(
            generate_spans_from_ordered_towers(
                line,
                span_id_factory=lambda _owner, index: stable_id(f"span-{index + 1}"),
            )
        )
    if not imported.routes and existing_towers:
        line = Line(
            stable_id("line"),
            "GIS Towers",
            tuple(tower.tower_id for tower in existing_towers),
            (),
            {"gis_source": imported.source.to_dict()},
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
                {"gis_source": imported.source.to_dict()},
            )
        )
        spans.extend(generate_spans_from_ordered_towers(line))
    metadata = dict(project.metadata)
    metadata["gis_import"] = imported.report.to_dict()
    save_project(
        replace(
            project,
            metadata=metadata,
            lines=tuple(lines),
            routes=tuple(routes),
            towers=tuple(existing_towers),
            spans=tuple(spans),
        ),
        project_root,
    )
    return describe_project(project_root)
