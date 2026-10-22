"""Read-only adapter for the local experimental geometry viewer.

The in-memory snapshot contains bounded detached display points. It is never a
science result, membership map, export input, or persisted project artifact.
Finalized results are preferred, but a current reviewed/edited tree cloud may
also be previewed before publication.  That provisional path never unlocks the
formal report or export gates.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from desktop.current_report import current_finalized_gate, current_wire_curves

PREVIEW_SCHEMA = "treeline-local-future-preview-v1"
MAX_DISPLAY_POINTS = 30_000
_LOCK_MESSAGE = "请先完成点云分析并确认至少一棵树。"


def _locked(gate: Mapping[str, Any], reason: str | None = None, *, retryable: bool = False) -> dict[str, Any]:
    return {
        "schema": PREVIEW_SCHEMA,
        "status": "LOCKED",
        "retryable": retryable,
        "message": reason if retryable and reason else (_LOCK_MESSAGE if reason is None else f"{_LOCK_MESSAGE} {reason}"),
        "gate": dict(gate),
        "experimental": True,
        "prediction_mode": "LOCAL_SCENARIO",
        "trees": [],
        "wires": {"segments": []},
        "future_clearance_status": "NOT_IMPLEMENTED",
        "future_wire_status": "NOT_IMPLEMENTED",
    }


def preview_gate(workflow: Any, *, hydrate_editor: bool = False) -> dict[str, Any]:
    """Authorize a read-only twin from either finalized or current tree data.

    Formal reports still use :func:`current_finalized_gate`.  This gate only
    confirms that the current editor can produce a non-empty, internally
    aligned tree cloud.  It deliberately exposes the editor revision so the
    browser can invalidate a preview immediately after any tree edit.
    """

    final = current_finalized_gate(workflow, hydrate_editor=hydrate_editor)
    if final.get("status") == "PASS":
        return {
            **dict(final),
            "publication_state": "FINALIZED",
            "provisional": False,
        }
    try:
        editor = workflow._ensure_editor()
        result = workflow._current_result()
        if editor is None or result is None:
            return {**dict(final), "publication_state": "UNAVAILABLE", "provisional": True}
        reconciled = editor.final_reconciled()
        tree_id = np.asarray(reconciled.final_tree_id)
        xyz = np.asarray(reconciled.final_xyz_local_m)
        records = tuple(reconciled.global_tree_records)
        if (
            tree_id.ndim != 1
            or xyz.ndim != 2
            or xyz.shape != (len(tree_id), 3)
            or not len(tree_id)
            or not records
            or not np.all(np.isfinite(xyz))
        ):
            raise ValueError("当前处理结果没有可用的单木点云")
        inventory = {int(record.global_tree_id) for record in records}
        if set(np.unique(tree_id).tolist()) != inventory:
            raise ValueError("当前单木身份与点云成员不一致")
        operations = getattr(getattr(editor, "session", None), "operations", None)
        if not isinstance(operations, list):
            raise ValueError("当前树木编辑记录不可用")
        return {
            **dict(final),
            "status": "PASS",
            "run_id": str(getattr(result, "analysis_run_id", final.get("run_id") or "")),
            "publication_state": "REVIEW_READY",
            "provisional": True,
            "editor_operation_count": len(operations),
            "published_editor_operation_count": None,
            "tree_count": len(records),
            "reasons": [],
        }
    except (AttributeError, KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        return {
            **dict(final),
            "status": "LOCKED",
            "publication_state": "UNAVAILABLE",
            "provisional": True,
            "reasons": [*list(final.get("reasons") or []), str(exc)],
        }


def _revision_key(workflow: Any, gate: Mapping[str, Any], result: Any) -> tuple[Any, ...]:
    editor = workflow._ensure_editor()
    operations = getattr(getattr(editor, "session", None), "operations", None)
    if not isinstance(operations, list):
        raise ValueError("final editor operation journal unavailable")
    return (
        str(workflow.project_root),
        id(result),
        id(result.global_reconciled),
        json.dumps(dict(gate), sort_keys=True, ensure_ascii=False, allow_nan=False),
        json.dumps(operations, sort_keys=True, ensure_ascii=False, allow_nan=False),
    )


def _snapshot(workflow: Any, result: Any) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    from core.future_preview import FinalizedTreeGeometry
    from core.stage_b.tree_metrics import compute_stage_b_tree_metrics
    from core.stage_b.tree_display_id import ordered_product_spans
    from desktop.tree_display_numbering_extensions import _current_display_identities

    editor = workflow._ensure_editor()
    if editor is None:
        raise ValueError("current tree editor unavailable")
    reconciled = editor.final_reconciled()
    metrics = compute_stage_b_tree_metrics(
        result,
        reconciled,
        workflow.project_root / "exports" / "Prediction_Input.xlsx",
        project_root=workflow.project_root,
    )
    identities = _current_display_identities(workflow, editor)
    projection = workflow.final_projection(persist_display_evidence=False)
    rows = projection.get("trees") if isinstance(projection, Mapping) else None
    rows = rows if isinstance(rows, list) else []
    by_persistent: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        persistent = row.get("persistent_tree_id")
        if not isinstance(persistent, str) or not persistent or persistent in by_persistent:
            raise ValueError("finalized tree identities are missing or duplicated")
        by_persistent[persistent] = row
    records = tuple(
        sorted(reconciled.global_tree_records, key=lambda record: record.global_tree_id)
    )
    product_spans = ordered_product_spans(workflow._stage_project())
    if len(records) > MAX_DISPLAY_POINTS // 3:
        raise ValueError("tree inventory cannot be represented within the preview budget")
    tree_ids = np.asarray(reconciled.final_tree_id)
    xyz = np.asarray(reconciled.final_xyz_local_m)
    if tree_ids.ndim != 1 or xyz.shape != (len(tree_ids), 3):
        raise ValueError("finalized geometry and ownership are not aligned")
    if set(np.unique(tree_ids).tolist()) != {record.global_tree_id for record in records}:
        raise ValueError("finalized geometry inventory does not match tree records")
    per_tree_budget = MAX_DISPLAY_POINTS // len(records)
    snapshot = []
    seen_persistent: set[str] = set()
    for record in records:
        persistent = str(record.persistent_tree_id)
        if persistent in seen_persistent:
            raise ValueError("tree geometry identity is duplicated")
        seen_persistent.add(persistent)
        row = by_persistent.get(persistent, {})
        metric = metrics.get(int(record.global_tree_id))
        identity = identities.get(int(record.global_tree_id), {})
        span_index = int(record.span_id)
        if not 0 <= span_index < len(product_spans):
            raise ValueError("finalized parent Span cannot be bound to project topology")
        span = product_spans[span_index]
        if span.span_id not in result.span_statuses:
            raise ValueError("finalized parent Span is outside the accepted batch")
        indices = np.flatnonzero(tree_ids == record.global_tree_id)
        if len(indices) != record.full_resolution_point_count or not len(indices):
            raise ValueError("finalized tree point count does not match authoritative record")
        points = xyz[indices]
        if not np.all(np.isfinite(points)):
            raise ValueError("finalized parent geometry contains non-finite coordinates")
        # Include Z extremes, then evenly spaced members in authoritative order.
        # Each tree gets its own budget, including small trees in large projects.
        keep = np.unique(
            np.concatenate(
                (
                    np.asarray([np.argmin(points[:, 2]), np.argmax(points[:, 2])]),
                    np.linspace(
                        0, len(points) - 1, min(len(points), per_tree_budget - 2), dtype=int
                    ),
                )
            )
        )
        anchor = (*np.mean(points[:, :2], axis=0).tolist(), float(np.min(points[:, 2])))
        geometry = FinalizedTreeGeometry(
            persistent_tree_id=persistent,
            span_id=span.span_id,
            xyz_local_m=points[keep],
            height_m=(metric.tree_height_m if metric is not None else row.get("height_m")),
            crown_major_m=(metric.crown_major_width_m if metric is not None else row.get("crown_width_major_m")),
            crown_minor_m=(metric.crown_minor_width_m if metric is not None else row.get("crown_width_minor_m")),
            dbh_cm=(
                metric.dbh.dbh_m * 100.0
                if metric is not None and metric.dbh.dbh_m is not None
                else row.get("diameter_cm")
            ),
            anchor_xyz_local_m=anchor,
        )
        snapshot.append(
            {
                "tree_id": int(record.global_tree_id),
                "persistent_tree_id": persistent,
                "display_tree_id": identity.get("display_tree_id") or row.get("display_tree_id"),
                "span_id": span.span_id,
                "internal_span_id": span_index,
                "analysis_run_id": str(result.analysis_run_id),
                "start_tower_id": span.start_tower_id,
                "end_tower_id": span.end_tower_id,
                "current_height_m": geometry.height_m,
                "current_crown_major_m": geometry.crown_major_m,
                "current_crown_minor_m": geometry.crown_minor_m,
                "current_dbh_cm": geometry.dbh_cm,
                "current_minimum_clearance_m": row.get("minimum_clearance_m"),
                "current_horizontal_clearance_m": row.get("horizontal_clearance_m"),
                "current_vertical_clearance_m": row.get("vertical_clearance_m"),
                "parent_point_count": int(len(points)),
                "display_point_count": int(len(keep)),
                "geometry": geometry,
            }
        )
    return tuple(snapshot), current_wire_curves(result)


def _cached_snapshot(
    workflow: Any,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any], str, float] | None:
    """Build the preview from the persisted bounded current-result snapshot.

    A finalized project can contain tens of millions of member points.  The
    current-result page has already certified the full-resolution metrics and
    saved a bounded, identity-labelled display sample.  Reusing that sample
    makes reopening a project and entering the twin immediate while the
    parent point counts and measurements remain the authoritative values.
    """

    from core.future_preview import FinalizedTreeGeometry
    from core.stage_b.tree_display_id import ordered_product_spans
    from desktop.integrated_cache import load_current_report_cache, revision_fingerprint

    report = load_current_report_cache(workflow)
    fingerprint = revision_fingerprint(workflow)
    if report is None or fingerprint is None:
        return None
    rows = report.get("trees")
    display = (report.get("display") or {}).get("trees3d")
    if not isinstance(rows, list) or not rows or not isinstance(display, Mapping):
        return None
    xyz = np.asarray(display.get("xyz", []), dtype=np.float64)
    tree_ids = np.asarray(display.get("tree_id_by_index", []))
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(tree_ids) != len(xyz) or not len(xyz):
        return None
    if not np.all(np.isfinite(xyz)):
        return None
    manifest_getter = getattr(workflow, "_latest_batch_manifest", None)
    manifest = manifest_getter() if callable(manifest_getter) else None
    raw_records = ((manifest or {}).get("global_reconciliation") or {}).get("trees")
    if not isinstance(raw_records, list):
        return None
    record_by_persistent = {
        str(item.get("persistent_tree_id")): item
        for item in raw_records
        if isinstance(item, Mapping) and item.get("persistent_tree_id")
    }
    row_by_persistent = {
        str(item.get("persistent_tree_id")): item
        for item in rows
        if isinstance(item, Mapping) and item.get("persistent_tree_id")
    }
    if set(record_by_persistent) != set(row_by_persistent):
        return None
    product_spans = ordered_product_spans(workflow._stage_project())
    snapshot: list[dict[str, Any]] = []
    for persistent, row in row_by_persistent.items():
        record = record_by_persistent[persistent]
        global_id = int(record["global_tree_id"])
        points = xyz[tree_ids == global_id]
        if not len(points):
            return None
        span_index = int(record["span_id"])
        if not 0 <= span_index < len(product_spans):
            return None
        span = product_spans[span_index]
        anchor = (*np.mean(points[:, :2], axis=0).tolist(), float(np.min(points[:, 2])))
        geometry = FinalizedTreeGeometry(
            persistent_tree_id=persistent,
            span_id=span.span_id,
            xyz_local_m=points,
            height_m=row.get("height_m"),
            crown_major_m=row.get("crown_width_major_m"),
            crown_minor_m=row.get("crown_width_minor_m"),
            dbh_cm=row.get("diameter_cm"),
            anchor_xyz_local_m=anchor,
        )
        snapshot.append(
            {
                "tree_id": global_id,
                "persistent_tree_id": persistent,
                "display_tree_id": row.get("display_tree_id"),
                "span_id": span.span_id,
                "internal_span_id": span_index,
                "analysis_run_id": str((manifest or {}).get("analysis_run_id") or ""),
                "start_tower_id": span.start_tower_id,
                "end_tower_id": span.end_tower_id,
                "current_height_m": row.get("height_m"),
                "current_crown_major_m": row.get("crown_width_major_m"),
                "current_crown_minor_m": row.get("crown_width_minor_m"),
                "current_dbh_cm": row.get("diameter_cm"),
                "current_minimum_clearance_m": row.get("minimum_clearance_m"),
                "current_horizontal_clearance_m": row.get("horizontal_clearance_m"),
                "current_vertical_clearance_m": row.get("vertical_clearance_m"),
                "parent_point_count": int(record.get("full_resolution_point_count") or len(points)),
                "display_point_count": int(len(points)),
                "geometry": geometry,
            }
        )
    wires = report.get("wires")
    if not isinstance(wires, Mapping):
        return None
    threshold = (report.get("kpis") or {}).get("warning_threshold_m", 0.5)
    return (
        tuple(sorted(snapshot, key=lambda item: int(item["tree_id"]))),
        dict(wires),
        json.dumps(fingerprint, sort_keys=True, ensure_ascii=False),
        float(0.5 if threshold is None else threshold),
    )


def _twin_profiles(workflow: Any) -> dict[str, dict[str, Any]]:
    path = Path(workflow.project_root) / "runtime" / "warning" / "twin_profiles.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    profiles = value.get("profiles") if isinstance(value, dict) else None
    return profiles if isinstance(profiles, dict) else {}


def _calendar_days(start_iso: str, value: int, unit: str) -> int:
    from algorithms.prediction.forecast import add_months

    start = date.fromisoformat(start_iso[:10])
    if unit == "days":
        return value
    months = value if unit == "months" else value * 12
    return (add_months(start, months) - start).days


def _display_clearance(output: list[dict[str, Any]], wires: Mapping[str, Any]) -> str:
    points = []
    for segment in wires.get("segments", []):
        xyz = np.asarray(segment.get("xyz", []), dtype=np.float64)
        if xyz.ndim == 2 and xyz.shape[1] == 3 and len(xyz) and np.all(np.isfinite(xyz)):
            points.append(xyz)
    if not points:
        return "UNAVAILABLE_NO_WIRE_DISPLAY_POINTS"
    try:
        from scipy.spatial import cKDTree

        index = cKDTree(np.concatenate(points, axis=0))
        for tree in output:
            future = np.asarray(tree["future_xyz"], dtype=np.float64)
            tree["estimated_future_clearance_m"] = round(
                float(np.min(index.query(future, k=1, workers=1)[0])), 3
            )
            tree["future_clearance_source"] = "BOUNDED_DISPLAY_POINT_ESTIMATE"
        return "BOUNDED_DISPLAY_POINT_ESTIMATE"
    except (ImportError, TypeError, ValueError, MemoryError):
        return "UNAVAILABLE_LOCAL_DISTANCE_INDEX"


def _clearance_envelope(tree: Mapping[str, Any], metrics: Mapping[str, Any]) -> float | None:
    """Conservative branch-growth envelope relative to the current nearest pair.

    The authoritative current components come from the full-resolution
    tree/wire nearest pair.  Future vertical extension and half of crown-width
    expansion consume those orthogonal components.  Unlike a sparse display
    point distance, the envelope can reach zero when a branch crosses the
    conductor path.
    """

    height = float(metrics.get("height_increment_m") or 0.0)
    radial = float(metrics.get("crown_width_increment_m") or 0.0) / 2.0
    horizontal = tree.get("current_horizontal_clearance_m")
    vertical = tree.get("current_vertical_clearance_m")
    if isinstance(horizontal, (int, float)) and isinstance(vertical, (int, float)):
        return float(np.hypot(max(0.0, float(horizontal) - radial), max(0.0, float(vertical) - height)))
    minimum = tree.get("current_minimum_clearance_m")
    if isinstance(minimum, (int, float)):
        return max(0.0, float(minimum) - max(radial, height))
    return None


def _trajectory(
    tree: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    warning_threshold_m: float,
    search_days: int = 1826,
) -> dict[str, Any]:
    """Build a weekly five-year trend plus day-accurate threshold crossings."""

    from core.future_preview.predictor import seasonal_growth_metrics

    start_iso = str(profile.get("observation_date") or date.today().isoformat())[:10]
    species = str(profile.get("species") or "其他")
    age = float(profile.get("age", 20.0))
    weather = profile.get("weather") if isinstance(profile.get("weather"), dict) else {}
    base_height = tree.get("current_height_m")
    base_major = tree.get("current_crown_major_m")
    base_minor = tree.get("current_crown_minor_m")
    base_dbh = tree.get("current_dbh_cm")

    def values(day_count: int) -> dict[str, Any]:
        if day_count == 0:
            metrics = {
                "height_increment_m": 0.0,
                "crown_width_increment_m": 0.0,
                "dbh_increment_cm": 0.0,
                "seasonal_growth_years": 0.0,
                "weather_multiplier": 1.0,
            }
        else:
            metrics = seasonal_growth_metrics(
                horizon_days=day_count,
                start_date=start_iso,
                species=species,
                age=age,
                weather=weather,
            )
        clearance = _clearance_envelope(tree, metrics)
        return {
            "day": day_count,
            "date": (date.fromisoformat(start_iso) + timedelta(days=day_count)).isoformat(),
            "height_m": None if base_height is None else round(float(base_height) + float(metrics["height_increment_m"]), 4),
            "crown_major_m": None if base_major is None else round(float(base_major) + float(metrics["crown_width_increment_m"]), 4),
            "crown_minor_m": None if base_minor is None else round(
                float(base_minor)
                + float(metrics["crown_width_increment_m"])
                * float(metrics.get("minor_to_major_increment", 0.75)),
                4,
            ),
            "dbh_cm": None if base_dbh is None else round(float(base_dbh) + float(metrics["dbh_increment_cm"]), 4),
            "clearance_m": None if clearance is None else round(clearance, 4),
            "seasonal_growth_years": round(float(metrics["seasonal_growth_years"]), 6),
        }

    crossing_day = None
    contact_day = None
    points = []
    sample_days = list(range(0, search_days + 1, 7))
    if sample_days[-1] != search_days:
        sample_days.append(search_days)
    for day_count in sample_days:
        item = values(day_count)
        points.append(item)
        clearance = item["clearance_m"]
        if clearance is not None:
            if crossing_day is None and clearance <= warning_threshold_m:
                crossing_day = day_count
            if contact_day is None and clearance <= 0.0:
                contact_day = day_count
        if crossing_day is not None and contact_day is not None:
            # Keep the full curve for the chart; only crossing discovery is complete.
            continue

    def refine(rough_day: int | None, threshold: float) -> int | None:
        if rough_day is None:
            return None
        low = max(0, rough_day - 7)
        for day_count in range(low, rough_day + 1):
            clearance = values(day_count)["clearance_m"]
            if clearance is not None and clearance <= threshold:
                return day_count
        return rough_day

    crossing_day = refine(crossing_day, warning_threshold_m)
    contact_day = refine(contact_day, 0.0)
    return {
        "points": points,
        "standard_crossing_day": crossing_day,
        "standard_crossing_date": None if crossing_day is None else values(crossing_day)["date"],
        "contact_day": contact_day,
        "contact_date": None if contact_day is None else values(contact_day)["date"],
        "search_horizon_days": search_days,
        "warning_threshold_m": warning_threshold_m,
        "clearance_method": "FULL_RESOLUTION_NEAREST_PAIR_GROWTH_ENVELOPE",
    }


def build_future_preview(
    workflow: Any,
    *,
    horizon_years: int = 1,
    horizon_value: int | None = None,
    horizon_unit: str = "years",
) -> dict[str, Any]:
    """Validate publication twice and transform detached parent display geometry."""

    legacy_request = horizon_value is None
    if legacy_request:
        if type(horizon_years) is not int or horizon_years not in {1, 3, 5}:
            raise ValueError("horizon must be 1, 3, or 5 years")
        horizon_value = horizon_years
        horizon_unit = "years"
    if type(horizon_value) is not int or horizon_unit not in {"days", "months", "years"}:
        raise ValueError("time horizon must use integer days, months, or years")
    limits = {"days": (1, 1826), "months": (1, 60), "years": (1, 5)}
    low, high = limits[horizon_unit]
    if not low <= horizon_value <= high:
        raise ValueError(f"{horizon_unit} horizon must be in {low}..{high}")
    gate = preview_gate(workflow, hydrate_editor=False)
    if gate["status"] != "PASS":
        return _locked(gate)
    # Do not wait behind a running analysis or another long workflow operation.
    lock = workflow._lock
    if not lock.acquire(blocking=False):
        return _locked(gate, "当前项目正在处理其他操作，请稍后重试。", retryable=True)
    try:
        persisted = _cached_snapshot(workflow)
        using_persisted_snapshot = persisted is not None
        if using_persisted_snapshot:
            trees, wires, persisted_revision, persisted_threshold = persisted
            key = ("PERSISTED_CURRENT_REPORT", persisted_revision)
            result = None
        else:
            gate = preview_gate(workflow, hydrate_editor=True)
            if gate["status"] != "PASS":
                return _locked(gate)
            result = workflow._current_result()
            key = _revision_key(workflow, gate, result)
            cache = getattr(workflow, "_local_future_preview_snapshot", None)
            if not isinstance(cache, tuple) or cache[0] != key:
                trees, wires = _snapshot(workflow, result)
            else:
                _, trees, wires = cache
        from core.future_preview import LocalScenarioPredictor
        from core.future_preview.predictor import seasonal_nonuniform_prediction

        profiles = _twin_profiles(workflow)
        legacy_predictor = LocalScenarioPredictor() if legacy_request else None
        output = []
        warning_threshold_m = (
            persisted_threshold
            if using_persisted_snapshot
            else float(
                (getattr(result, "risk_policy_snapshot", {}) or {}).get(
                    "warning_clearance_threshold_m", 0.5
                )
            )
        )
        for tree in trees:
            geometry = tree["geometry"]
            profile = profiles.get(str(tree["persistent_tree_id"]), {})
            start_date = str(profile.get("observation_date") or date.today().isoformat())[:10]
            days = _calendar_days(start_date, horizon_value, horizon_unit)
            if legacy_request:
                legacy = legacy_predictor.predict(geometry, horizon_value)
                predicted = {
                    "xyz_local_m": legacy.xyz_local_m,
                    "predicted_height_m": legacy.predicted_height_m,
                    "predicted_crown_major_m": legacy.predicted_crown_major_m,
                    "predicted_crown_minor_m": legacy.predicted_crown_minor_m,
                    "predicted_dbh_cm": legacy.predicted_dbh_cm,
                    "target_date": (date.fromisoformat(start_date).replace()).isoformat(),
                    "species": str(profile.get("species") or "其他"),
                    "age": float(profile.get("age", 20.0)),
                    "height_increment_m": None,
                    "crown_width_increment_m": None,
                    "dbh_increment_cm": None,
                    "seasonal_growth_years": float(horizon_value),
                    "weather_multiplier": 1.0,
                    "transformation": "LEGACY_LOCAL_SCENARIO",
                    "assumptions": list(legacy.assumptions),
                }
            else:
                predicted = seasonal_nonuniform_prediction(
                    geometry,
                    horizon_days=days,
                    start_date=start_date,
                    species=str(profile.get("species") or "其他"),
                    age=float(profile.get("age", 20.0)),
                    weather=(profile.get("weather") if isinstance(profile.get("weather"), dict) else {}),
                )
            row = {
                    **{name: value for name, value in tree.items() if name != "geometry"},
                    "current_xyz": geometry.xyz_local_m.tolist(),
                    "future_xyz": predicted["xyz_local_m"].tolist(),
                    "predicted_height_m": predicted["predicted_height_m"],
                    "predicted_crown_major_m": predicted["predicted_crown_major_m"],
                    "predicted_crown_minor_m": predicted["predicted_crown_minor_m"],
                    "predicted_dbh_cm": predicted["predicted_dbh_cm"],
                    "forecast_horizon_days": days,
                    "forecast_horizon_years": horizon_value if horizon_unit == "years" else None,
                    "forecast_horizon_value": horizon_value,
                    "forecast_horizon_unit": horizon_unit,
                    "target_date": predicted["target_date"],
                    "species": predicted["species"],
                    "species_architecture": predicted.get("species_architecture"),
                    "tree_age": predicted["age"],
                    "height_increment_m": predicted["height_increment_m"],
                    "crown_width_increment_m": predicted["crown_width_increment_m"],
                    "dbh_increment_cm": predicted["dbh_increment_cm"],
                    "seasonal_growth_years": predicted["seasonal_growth_years"],
                    "weather_multiplier": predicted["weather_multiplier"],
                    "branch_growth_mode": predicted.get("branch_growth_mode"),
                    "profile_source": "USER_REPORT" if profile else "NEUTRAL_FALLBACK",
                    "prediction_mode": (
                        legacy.prediction_mode if legacy_request else "SEASONAL_NON_RIGID_TWIN"
                    ),
                    "model_name": (
                        legacy.model_name if legacy_request else "seasonal-nonrigid-tree-twin"
                    ),
                    "model_version": legacy.model_version if legacy_request else "2.0.0",
                    "experimental": True,
                    "geometry_kind": "SYNTHETIC_FUTURE_GEOMETRY",
                    "science_status": "NOT_CURRENT_SURVEY_SCIENCE",
                    "transformation": predicted["transformation"],
                    "assumptions": list(predicted["assumptions"]),
                }
            if not legacy_request:
                envelope = _clearance_envelope(tree, predicted)
                row["predicted_clearance_envelope_m"] = (
                    None if envelope is None else round(envelope, 4)
                )
                row["warning_threshold_m"] = warning_threshold_m
                row["risk_state"] = (
                    "UNKNOWN"
                    if envelope is None
                    else "STANDARD_EXCEEDED"
                    if envelope <= warning_threshold_m
                    else "SAFE"
                )
                profile_key = json.dumps(profile, sort_keys=True, ensure_ascii=False, default=str)
                contact_cache = getattr(workflow, "_local_future_contact_cache", None)
                if not isinstance(contact_cache, dict):
                    contact_cache = {}
                cache_key = (repr(key), str(tree["persistent_tree_id"]), profile_key, warning_threshold_m)
                trajectory = contact_cache.get(cache_key)
                if trajectory is None:
                    trajectory = _trajectory(
                        tree,
                        profile,
                        warning_threshold_m=warning_threshold_m,
                    )
                    contact_cache[cache_key] = trajectory
                    if len(contact_cache) > 128:
                        contact_cache = {cache_key: trajectory}
                    workflow._local_future_contact_cache = contact_cache
                row["trajectory"] = copy.deepcopy(trajectory)
            output.append(row)
        clearance_status = (
            "NOT_IMPLEMENTED" if legacy_request else _display_clearance(output, wires)
        )
        after = preview_gate(
            workflow, hydrate_editor=not using_persisted_snapshot
        )
        if using_persisted_snapshot:
            refreshed = _cached_snapshot(workflow)
            revision_unchanged = (
                refreshed is not None
                and ("PERSISTED_CURRENT_REPORT", refreshed[2]) == key
            )
        else:
            revision_unchanged = (
                _revision_key(workflow, after, workflow._current_result()) == key
            )
        if (
            after["status"] != "PASS"
            or not revision_unchanged
        ):
            workflow._local_future_preview_snapshot = None
            return _locked(after, "当前树木点云已变化，请刷新后重新打开预测。")
        # Cache only detached, bounded visualization inputs after both guards.
        workflow._local_future_preview_snapshot = (key, trees, wires)
        return {
            "schema": PREVIEW_SCHEMA,
            "status": "PASS",
            "message": (
                "当前处理点云的实验性预测 · 尚未定稿，不替代正式勘察结果"
                if after.get("provisional")
                else "实验性预测 · 不替代当前正式勘察结果"
            ),
            "gate": after,
            "experimental": True,
            "prediction_mode": (
                "LOCAL_SCENARIO" if legacy_request else "SEASONAL_NON_RIGID_TWIN"
            ),
            "model_name": output[0]["model_name"],
            "model_version": output[0]["model_version"],
            "branch_growth_mode": output[0].get("branch_growth_mode"),
            "forecast_horizon_years": horizon_value if horizon_unit == "years" else None,
            "forecast_horizon_value": horizon_value,
            "forecast_horizon_unit": horizon_unit,
            "assumptions": output[0]["assumptions"],
            "validation_status": "NOT_VALIDATED_LONG_TERM_BIOLOGICAL_FORECAST",
            "geometry_kind": "SYNTHETIC_FUTURE_GEOMETRY",
            "science_status": "NOT_CURRENT_SURVEY_SCIENCE",
            "coordinate_space": "scene_local_m",
            "display_only": True,
            "display_point_budget": MAX_DISPLAY_POINTS,
            "trees": output,
            "wires": copy.deepcopy(wires),
            "future_clearance_status": clearance_status,
            "future_wire_status": "NOT_IMPLEMENTED",
            "warning_threshold_m": warning_threshold_m,
            "trajectory_resolution_days": 7,
            "trajectory_horizon_days": 1826,
        }
    except (AttributeError, KeyError, TypeError, ValueError, OSError) as exc:
        workflow._local_future_preview_snapshot = None
        return _locked(gate, f"本地预览不可用：{type(exc).__name__}: {exc}")
    finally:
        lock.release()
