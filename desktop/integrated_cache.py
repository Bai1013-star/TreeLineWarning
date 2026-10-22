"""Small persisted snapshot shared by data view, warning, and digital twin."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


CACHE_SCHEMA = "treeline-integrated-current-report-cache-v1"


def _root(workflow: Any) -> Path | None:
    value = getattr(workflow, "project_root", None) or getattr(workflow, "_root", None)
    try:
        root = Path(value).resolve()
    except (OSError, TypeError, ValueError):
        return None
    return root if root.is_dir() else None


def _latest_batch_path(workflow: Any) -> Path | None:
    getter = getattr(workflow, "_latest_batch_path", None)
    try:
        value = getter() if callable(getter) else None
        return Path(value).resolve() if value is not None else None
    except (OSError, TypeError, ValueError, RuntimeError):
        return None


def _file_stamp(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def revision_fingerprint(workflow: Any) -> dict[str, Any] | None:
    root = _root(workflow)
    batch = _latest_batch_path(workflow)
    if root is None or batch is None:
        return None
    edits = batch.parent / "tree_edits.json"
    dbh_final = batch.parent / "dbh_final.json"
    workflow_path = root / "desktop-workflow.json"
    state = getattr(workflow, "_workflow", {})
    state = state if isinstance(state, Mapping) else {}
    return {
        "project_root": str(root),
        "run_id": str(state.get("last_batch_id") or ""),
        "batch": _file_stamp(batch),
        "tree_edits": _file_stamp(edits),
        "dbh_final": _file_stamp(dbh_final),
        "workflow": _file_stamp(workflow_path),
    }


def cache_path(workflow: Any) -> Path | None:
    root = _root(workflow)
    return None if root is None else root / "runtime" / "warning" / "current_report_cache.json"


def load_current_report_cache(workflow: Any) -> dict[str, Any] | None:
    path = cache_path(workflow)
    fingerprint = revision_fingerprint(workflow)
    if path is None or fingerprint is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != CACHE_SCHEMA:
        return None
    if payload.get("revision") != fingerprint:
        return None
    report = payload.get("report")
    if not isinstance(report, dict) or report.get("status") != "FINALIZED":
        return None
    return report


def save_current_report_cache(workflow: Any, report: Mapping[str, Any]) -> Path | None:
    if report.get("status") != "FINALIZED":
        return None
    path = cache_path(workflow)
    fingerprint = revision_fingerprint(workflow)
    if path is None or fingerprint is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"schema": CACHE_SCHEMA, "revision": fingerprint, "report": dict(report)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


__all__ = [
    "CACHE_SCHEMA",
    "cache_path",
    "load_current_report_cache",
    "revision_fingerprint",
    "save_current_report_cache",
]
