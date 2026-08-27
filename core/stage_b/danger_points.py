"""Persistence and viewer projection for exact Stage-B clearance evidence.

The existing clearance engine remains authoritative.  This module never
recalculates point-to-wire distance: it only persists exact danger observations
and projects already-computed per-tree nearest-point evidence into bounded UI
payloads.  Display LOD is explicit and cannot become science membership.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core.stage_a.risk import RiskPolicy
from core.stage_b.contact import contact_evidence_from_clearance

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


def _remove_persisted_danger_points(
    project_root: Path | str,
    analysis_run_id: str,
) -> None:
    """Remove viewer-only exact-danger sidecars that no longer match science."""

    for path in (
        danger_points_path(project_root, analysis_run_id),
        _metadata_path(project_root, analysis_run_id),
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # A stale sidecar is never authoritative without a matching current
            # clearance object.  Best-effort cleanup must not block finalization.
            pass


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


def persist_final_danger_points(
    project_root: Path | str,
    analysis_run_id: str,
    clearance: Any | None,
) -> Path | None:
    """Persist exact post-editor danger observations when XYZ evidence exists.

    A hydrated historical ``batch.json`` restores per-tree danger counts but
    does not restore every danger XYZ.  That optional display limitation must
    not block finalization or Excel export.  In that case an already-valid
    sidecar is reused when its count matches; otherwise the display layer stays
    unavailable until exact clearance is recomputed.

    ``clearance is None`` explicitly invalidates any prior sidecar for this run.
    This prevents edited tree membership from resurrecting pre-edit danger XYZ.
    """

    if clearance is None:
        _remove_persisted_danger_points(project_root, analysis_run_id)
        return None
    points = tuple(getattr(clearance, "danger_points", ()) or ())
    expected_count = int(
        sum(
            int(getattr(record, "danger_point_count", 0) or 0)
            for record in getattr(clearance, "records", ())
        )
    )
    path = danger_points_path(project_root, analysis_run_id)
    if expected_count > 0 and not points:
        existing = load_final_danger_points(project_root, analysis_run_id)
        if existing is not None and len(existing[0]) == expected_count:
            return path
        _remove_persisted_danger_points(project_root, analysis_run_id)
        return None

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
    if expected_count != len(original_index):
        raise ValueError("DANGER_POINT_COUNT_MISMATCH")

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


def _arrays_from_clearance(
    clearance: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if clearance is None:
        return None
    points = tuple(getattr(clearance, "danger_points", ()) or ())
    if not points:
        return None
    original_index = np.asarray(
        [int(point.original_index) for point in points], dtype=np.int64
    )
    tree_id = np.asarray(
        [int(point.global_tree_id) for point in points], dtype=np.int64
    )
    xyz = np.asarray(
        [tuple(float(value) for value in point.xyz_local_m) for point in points],
        dtype=np.float64,
    ).reshape((-1, 3))
    return original_index, tree_id, xyz


def _empty_danger(authority: str) -> dict[str, Any]:
    return {
        "xyz": [],
        "tree_id_by_index": [],
        "original_index": [],
        "display_point_count": 0,
        "full_point_count": 0,
        "authority": authority,
        "display_only_sampling": False,
    }


def build_danger_point_display(
    project_root: Path | str,
    result: Any | None,
    *,
    visible_tree_ids: Iterable[int] | None = None,
    display_budget: int = _DEFAULT_DISPLAY_BUDGET,
) -> dict[str, Any]:
    """Build a bounded red-point payload from exact current clearance evidence."""

    if result is None:
        return _empty_danger("UNAVAILABLE")
    clearance = getattr(result, "final_clearance", None)
    if clearance is None:
        return _empty_danger("UNAVAILABLE")

    expected_count = int(
        sum(
            int(getattr(record, "danger_point_count", 0) or 0)
            for record in getattr(clearance, "records", ())
        )
    )
    arrays = _arrays_from_clearance(clearance)
    authority = "EXACT_FINAL_CLEARANCE_MEMORY"
    if arrays is None:
        if expected_count == 0:
            return _empty_danger(authority)
        arrays = load_final_danger_points(project_root, str(result.analysis_run_id))
        authority = "EXACT_FINAL_CLEARANCE_PERSISTED"
    if arrays is None:
        return _empty_danger("UNAVAILABLE")
    original_index, tree_id, xyz = arrays
    if len(original_index) != expected_count:
        return _empty_danger("COUNT_MISMATCH")
    if visible_tree_ids is not None:
        visible = np.asarray(
            sorted({int(value) for value in visible_tree_ids}), dtype=np.int64
        )
        mask = (
            np.isin(tree_id, visible)
            if len(visible)
            else np.zeros(len(tree_id), dtype=bool)
        )
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


def build_clearance_evidence_display(
    result: Any | None,
    *,
    visible_tree_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Project exact per-tree closest tree/wire pairs for the Giro3D reviewer."""

    if result is None or getattr(result, "final_clearance", None) is None:
        return {"records": [], "authority": "UNAVAILABLE"}
    clearance = result.final_clearance
    visible = (
        None
        if visible_tree_ids is None
        else {int(value) for value in visible_tree_ids}
    )
    tree_records = {
        int(record.global_tree_id): record
        for record in getattr(
            getattr(result, "global_reconciled", None), "global_tree_records", ()
        )
    }
    snapshot = dict(getattr(result, "risk_policy_snapshot", {}) or {})
    policy = RiskPolicy(
        risk_policy_version=str(
            snapshot.get("risk_policy_version", "risk-policy-v1")
        ),
        warning_clearance_threshold_m=float(
            snapshot.get("warning_clearance_threshold_m", 0.5)
        ),
    )
    rows: list[dict[str, Any]] = []
    for record in getattr(clearance, "records", ()):
        tree_id = int(record.global_tree_id)
        if visible is not None and tree_id not in visible:
            continue
        minimum = record.minimum_clearance_m
        tree_xyz = record.nearest_tree_xyz_local_m
        wire_xyz = record.nearest_wire_xyz_local_m
        if minimum is None or tree_xyz is None or wire_xyz is None:
            continue
        if not np.isfinite(float(minimum)):
            continue
        tree_xyz_values = tuple(float(value) for value in tree_xyz)
        wire_xyz_values = tuple(float(value) for value in wire_xyz)
        if not np.all(
            np.isfinite(
                np.asarray((*tree_xyz_values, *wire_xyz_values), dtype=np.float64)
            )
        ):
            continue
        evidence = contact_evidence_from_clearance(
            float(minimum), clearance_status=str(record.status)
        )
        source_tree = tree_records.get(tree_id)
        review_required = (
            bool(getattr(source_tree, "review", False)) or evidence.review_required
        )
        assessment = policy.classify(
            clearance_m=float(minimum),
            contact=(
                False
                if evidence.state.value == "CONTACT_NOT_CONFIRMED"
                else None
            ),
            review_required=review_required,
        )
        display_class = (
            "REVIEW_REQUIRED"
            if assessment.review.value == "REVIEW_REQUIRED"
            else assessment.risk.value
        )
        midpoint = [
            0.5 * (tree_xyz_values[axis] + wire_xyz_values[axis])
            for axis in range(3)
        ]
        rows.append(
            {
                "global_tree_id": tree_id,
                "persistent_tree_id": getattr(record, "persistent_tree_id", None),
                "span_name": str(record.span_name),
                "nearest_wire_id": record.nearest_wire_id,
                "minimum_clearance_m": float(minimum),
                "nearest_tree_original_index": record.nearest_tree_original_index,
                "tree_xyz": list(tree_xyz_values),
                "wire_xyz": list(wire_xyz_values),
                "midpoint_xyz": midpoint,
                "risk": assessment.risk.value,
                "review_state": assessment.review.value,
                "display_class": display_class,
                "contact_state": evidence.state.value,
                "clearance_status": str(record.status),
            }
        )
    return {
        "records": rows,
        "authority": "EXACT_FINAL_CLEARANCE_NEAREST_PAIR",
        "record_count": len(rows),
        "warning_threshold_m": policy.warning_clearance_threshold_m,
    }


__all__ = [
    "DANGER_POINTS_SCHEMA",
    "build_clearance_evidence_display",
    "build_danger_point_display",
    "danger_points_path",
    "load_final_danger_points",
    "persist_final_danger_points",
]
