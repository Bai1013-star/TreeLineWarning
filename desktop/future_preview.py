"""Read-only, finalized-only adapter for the local experimental geometry viewer.

The in-memory snapshot contains bounded detached display points. It is never a
science result, membership map, export input, or persisted project artifact.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from desktop.current_report import current_finalized_gate, current_wire_curves

PREVIEW_SCHEMA = "treeline-local-future-preview-v1"
MAX_DISPLAY_POINTS = 30_000
_LOCK_MESSAGE = "请先完成当前项目的单木复核。"


def _locked(gate: Mapping[str, Any], reason: str | None = None) -> dict[str, Any]:
    return {
        "schema": PREVIEW_SCHEMA,
        "status": "LOCKED",
        "message": _LOCK_MESSAGE if reason is None else f"{_LOCK_MESSAGE} {reason}",
        "gate": dict(gate),
        "experimental": True,
        "prediction_mode": "LOCAL_SCENARIO",
        "trees": [],
        "wires": {"segments": []},
        "future_clearance_status": "NOT_IMPLEMENTED",
        "future_wire_status": "NOT_IMPLEMENTED",
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
    from core.stage_b.tree_display_id import ordered_product_spans

    projection = workflow.final_projection(persist_display_evidence=False)
    if projection.get("metadata", {}).get("tree_metrics_source") != "CANONICAL_FULLRES_POST_EDITOR":
        raise ValueError("canonical finalized metrics unavailable")
    rows = projection.get("trees")
    if not isinstance(rows, list) or not rows:
        raise ValueError("no finalized trees available for preview")
    by_persistent: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        persistent = row.get("persistent_tree_id")
        if not isinstance(persistent, str) or not persistent or persistent in by_persistent:
            raise ValueError("finalized tree identities are missing or duplicated")
        by_persistent[persistent] = row
    reconciled = result.global_reconciled
    records = tuple(
        sorted(reconciled.global_tree_records, key=lambda record: record.global_tree_id)
    )
    product_spans = ordered_product_spans(workflow._stage_project())
    if len(records) != len(rows) or len(records) > MAX_DISPLAY_POINTS // 3:
        raise ValueError("finalized tree inventory cannot be represented within the preview budget")
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
        if persistent not in by_persistent or persistent in seen_persistent:
            raise ValueError("finalized geometry identity does not match current metrics")
        seen_persistent.add(persistent)
        row = by_persistent[persistent]
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
            height_m=row.get("height_m"),
            crown_major_m=row.get("crown_width_major_m"),
            crown_minor_m=row.get("crown_width_minor_m"),
            dbh_cm=row.get("diameter_cm"),
            anchor_xyz_local_m=anchor,
        )
        snapshot.append(
            {
                "tree_id": int(record.global_tree_id),
                "persistent_tree_id": persistent,
                "display_tree_id": row.get("display_tree_id"),
                "span_id": span.span_id,
                "internal_span_id": span_index,
                "analysis_run_id": str(result.analysis_run_id),
                "start_tower_id": span.start_tower_id,
                "end_tower_id": span.end_tower_id,
                "current_height_m": row.get("height_m"),
                "current_crown_major_m": row.get("crown_width_major_m"),
                "current_crown_minor_m": row.get("crown_width_minor_m"),
                "current_dbh_cm": row.get("diameter_cm"),
                "parent_point_count": int(len(points)),
                "display_point_count": int(len(keep)),
                "geometry": geometry,
            }
        )
    return tuple(snapshot), current_wire_curves(result)


def build_future_preview(workflow: Any, *, horizon_years: int = 1) -> dict[str, Any]:
    """Validate publication twice and transform detached parent display geometry."""

    if type(horizon_years) is not int or horizon_years not in {1, 3, 5}:
        raise ValueError("horizon must be 1, 3, or 5 years")
    gate = current_finalized_gate(workflow, hydrate_editor=False)
    if gate["status"] != "PASS":
        return _locked(gate)
    # Do not wait behind a running analysis or another long workflow operation.
    lock = workflow._lock
    if not lock.acquire(blocking=False):
        return _locked(gate, "当前项目正在处理其他操作，请稍后重试。")
    try:
        gate = current_finalized_gate(workflow)
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

        predictor = LocalScenarioPredictor()
        output = []
        for tree in trees:
            geometry = tree["geometry"]
            predicted = predictor.predict(geometry, horizon_years)
            output.append(
                {
                    **{name: value for name, value in tree.items() if name != "geometry"},
                    "current_xyz": geometry.xyz_local_m.tolist(),
                    "future_xyz": predicted.xyz_local_m.tolist(),
                    "predicted_height_m": predicted.predicted_height_m,
                    "predicted_crown_major_m": predicted.predicted_crown_major_m,
                    "predicted_crown_minor_m": predicted.predicted_crown_minor_m,
                    "predicted_dbh_cm": predicted.predicted_dbh_cm,
                    "forecast_horizon_years": predicted.forecast_horizon_years,
                    "prediction_mode": predicted.prediction_mode,
                    "model_name": predicted.model_name,
                    "model_version": predicted.model_version,
                    "experimental": True,
                    "geometry_kind": "SYNTHETIC_FUTURE_GEOMETRY",
                    "science_status": "NOT_CURRENT_SURVEY_SCIENCE",
                    "assumptions": list(predicted.assumptions),
                }
            )
        after = current_finalized_gate(workflow)
        if (
            after["status"] != "PASS"
            or _revision_key(workflow, after, workflow._current_result()) != key
        ):
            workflow._local_future_preview_snapshot = None
            return _locked(after, "当前结果已变化，请重新完成复核后打开预测。")
        # Cache only detached, bounded visualization inputs after both guards.
        workflow._local_future_preview_snapshot = (key, trees, wires)
        return {
            "schema": PREVIEW_SCHEMA,
            "status": "PASS",
            "message": "实验性预测 · 不替代当前正式勘察结果",
            "gate": after,
            "experimental": True,
            "prediction_mode": "LOCAL_SCENARIO",
            "model_name": output[0]["model_name"],
            "model_version": output[0]["model_version"],
            "forecast_horizon_years": horizon_years,
            "assumptions": output[0]["assumptions"],
            "validation_status": "NOT_VALIDATED_LONG_TERM_BIOLOGICAL_FORECAST",
            "geometry_kind": "SYNTHETIC_FUTURE_GEOMETRY",
            "science_status": "NOT_CURRENT_SURVEY_SCIENCE",
            "coordinate_space": "scene_local_m",
            "display_only": True,
            "display_point_budget": MAX_DISPLAY_POINTS,
            "trees": output,
            "wires": copy.deepcopy(wires),
            "future_clearance_status": "NOT_IMPLEMENTED",
            "future_wire_status": "NOT_IMPLEMENTED",
        }
    except (AttributeError, KeyError, TypeError, ValueError, OSError) as exc:
        workflow._local_future_preview_snapshot = None
        return _locked(gate, f"本地预览不可用：{type(exc).__name__}: {exc}")
    finally:
        lock.release()
