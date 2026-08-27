"""Persistence and display projection for exact Stage-B danger points.

Danger points are produced by the existing exact tree-to-wire clearance engine.
This module never recalculates or approximates distance.  It only persists the
already-authoritative ``original_index`` + XYZ observations and builds a
bounded deterministic viewer sample.  Full-resolution counts remain explicit
so display LOD cannot be mistaken for science membership.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DANGER_POINTS_SCHEMA = "treeline-stage-b-danger-points-v1"
_DEFAULT_DISPLAY_BUDGET = 20_000


def _artifact_root(project_root: Path | str, analysis_run_id: str) -> Path:
    return (
        Path(project_root).expanduser().resolve()
        / "artifacts"
        / "project-runs"
        / "batches"
        / str(analysis_run_id)
    )


def danger_points_path(project_root: Path | str, analysis_run_id: str) -> Path:
    return _artifact_root(project_root, analysis_run_id) / "danger_points_final.npz"


def _metadata_path(project_root: Path | str, analysis_run_id: str) -> Path:
    return _artifact_root(project_root, analysis_run_id) / "danger_points_final.json"


def persist_final_danger_points(
    project_root: Path | str,
    analysis_run_id: str,
    clearance: Any | None,
) -> Path | None:
    """Persist exact post-editor danger observations atomically.

    An empty, valid clearance result is persisted as empty arrays so a restart
    can distinguish "computed: zero danger points" from "artifact absent".
    """

    if clearance is None:
        return None
    points = tuple(getattr(clearance, "danger_points", ()) or ())
    original_index = np.asarray(
        [int(point.original_index) for point in points],
        dtype=np.int64,
    )
    tree_id = np.asarray(
        [int(point.global_tree_id) for point in points],
        dtype=np.int64,
    )
    xyz = np.asarray(
        [tuple(float(value) for value in point.xyz_local_m) for point in points],
        dtype=np.float64,
    ).reshape((-1, 3))
    if len(original_index) != len(np.unique(original_index)):
        raise ValueError("DANGER_POINT_ORIGINAL_INDEX_DUPLICATE")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("DANGER_POINT_XYZ_NONFINITE")

    path = danger_points_path(project_root, analysis_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(
        temporary,
        original_index=original_index,
        global_tree_id=tree_id,
        xyz_local_m=xyz,
    )
    temporary.replace(path)
    metadata = {
        "schema": DANGER_POINTS_SCHEMA,
        "analysis_run_id": str(analysis_run_id),
        "authority": "EXACT_FINAL_CLEARANCE",
        "point_count": int(len(original_index)),
        "clearance_status": str(getattr(clearance, "status", "UNKNOWN")),
        "global_minimum_clearance_m": getattr(
            clearance,
            "global_minimum_clearance_m",
            None,
        ),
    }
    metadata_path = _metadata_path(project_root, analysis_run_id)
    metadata_tmp = metadata_path.with_name(metadata_path.name + ".tmp")
    metadata_tmp.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata_tmp.replace(metadata_path)
    return path


def _arrays_from_clearance(clearance: Any | None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if clearance is None:
        return None
    points = tuple(getattr(clearance, "danger_points", ()) or ())
    if not points:
        return None
    original_index = np.asarray([int(point.original_index) for point in points], dtype=np.int64)
    tree_id = np.asarray([int(point.global_tree_id) for point in points], dtype=np.int64)
    xyz = np.asarray(
        [tuple(float(value) for value in point.xyz_local_m) for point in points],
        dtype=np.float64,
    ).reshape((-1, 3))
    return original_index, tree_id, xyz


def load_final_danger_points(
    project_root: Path | str,
    analysis_run_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    path = danger_points_path(project_root, analysis_run_id)
    metadata_path = _metadata_path(project_root, analysis_run_id)
    if not path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != DANGER_POINTS_SCHEMA:
            return None
        if metadata.get("authority") != "EXACT_FINAL_CLEARANCE":
            return None
        with np.load(path, allow_pickle=False) as stream:
            original_index = np.asarray(stream["original_index"], dtype=np.int64)
            tree_id = np.asarray(stream["global_tree_id"], dtype=np.int64)
            xyz = np.asarray(stream["xyz_local_m"], dtype=np.float64)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if original_index.ndim != 1 or tree_id.shape != original_index.shape:
        return None
    if xyz.shape != (len(original_index), 3):
        return None
    if len(original_index) != len(np.unique(original_index)) or not np.all(np.isfinite(xyz)):
        return None
    if metadata.get("point_count") != int(len(original_index)):
        return None
    return original_index, tree_id, xyz


def build_danger_point_display(
    project_root: Path | str,
    result: Any | None,
    *,
    visible_tree_ids: Iterable[int] | None = None,
    display_budget: int = _DEFAULT_DISPLAY_BUDGET,
) -> dict[str, Any]:
    """Build a bounded red-layer payload without changing danger membership."""

    if result is None:
        return {
            "xyz": [],
            "tree_id_by_index": [],
            "original_index": [],
            "display_point_count": 0,
            "full_point_count": 0,
            "authority": "UNAVAILABLE",
        }
    arrays = _arrays_from_clearance(getattr(result, "final_clearance", None))
    authority = "EXACT_FINAL_CLEARANCE_MEMORY"
    if arrays is None:
        arrays = load_final_danger_points(project_root, str(result.analysis_run_id))
        authority = "EXACT_FINAL_CLEARANCE_PERSISTED"
    if arrays is None:
        return {
            "xyz": [],
            "tree_id_by_index": [],
            "original_index": [],
            "display_point_count": 0,
            "full_point_count": 0,
            "authority": "UNAVAILABLE",
        }
    original_index, tree_id, xyz = arrays
    if visible_tree_ids is not None:
        visible = np.asarray(sorted({int(value) for value in visible_tree_ids}), dtype=np.int64)
        mask = np.isin(tree_id, visible) if len(visible) else np.zeros(len(tree_id), dtype=bool)
        original_index = original_index[mask]
        tree_id = tree_id[mask]
        xyz = xyz[mask]
    full_count = int(len(original_index))
    budget = max(1, int(display_budget))
    if full_count > budget:
        rows = np.linspace(0, full_count - 1, budget, dtype=np.int64)
        original_index = original_index[rows]
        tree_id = tree_id[rows]
        xyz = xyz[rows]
    return {
        "xyz": xyz.tolist(),
        "tree_id_by_index": tree_id.astype(np.int64).tolist(),
        "original_index": original_index.astype(np.int64).tolist(),
        "display_point_count": int(len(original_index)),
        "full_point_count": full_count,
        "authority": authority,
        "display_only_sampling": full_count > len(original_index),
    }


__all__ = [
    "DANGER_POINTS_SCHEMA",
    "build_danger_point_display",
    "danger_points_path",
    "load_final_danger_points",
    "persist_final_danger_points",
]
