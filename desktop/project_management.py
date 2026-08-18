"""Desktop-only project lifecycle metadata.

This deliberately does not manufacture scientific artifacts.  A newly created
project is a valid TreeLineWarning desktop workspace with an explicit empty
state; it becomes an analysis project only after the existing ingestion and
tower workflow has produced the normal ``project.json`` contract.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

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
            "spans": 0,
            "analysis": "NOT_STARTED",
            "prediction": "NOT_COMPUTED",
        },
        "pointcloud_source": None,
    }
    _manifest_path(root).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
        "spans": 0,
        "analysis": "NOT_STARTED",
        "prediction": "NOT_COMPUTED",
    }
    payload["updated_at"] = _now()
    _manifest_path(root).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return describe_project(root)
