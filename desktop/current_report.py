"""Read-only projection for the modern current-result report.

The desktop workflow has a legacy Step-3 HTML contract and a newer editor
projection.  This module keeps the modern page at the product boundary: it
checks that the persisted result is still finalized, then exposes only the
canonical final rows and bounded display samples.  It does not finalize,
export, rerun science, or write project artifacts.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from core.stage_b import BatchSpanStatus
from core.stage_b.dbh_recovery import DBH_FIELDS

CURRENT_REPORT_SCHEMA = "treeline-current-final-report-v1"
CURRENT_REPORT_MODE = "CURRENT_FINAL"
CURRENT_WIRE_SCHEMA = "current-wire-curves-v1"
MAX_TREE_DISPLAY_POINTS = 30_000
MAX_WIRE_DISPLAY_POINTS = 20_000

_ACTIVE_FINALIZE_STAGES = {
    "APPLYING_EDITS",
    "RECALCULATING_CLEARANCE",
    "VALIDATING_CLEARANCE",
    "RECALCULATING_RISK",
    "GENERATING_EXCEL",
}

_CURRENT_TREE_FIELDS = (
    "persistent_tree_id",
    "display_tree_id",
    "height_m",
    "crown_width_m",
    "crown_width_major_m",
    "crown_width_minor_m",
    "crown_area_m2",
    "diameter_cm",
    "diameter_source",
    "minimum_clearance_m",
    "danger_point_count",
    "current_risk",
    "quality_state",
    "human_edited",
) + DBH_FIELDS


def _status_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).upper() if raw is not None else ""


def _thread_is_alive(value: Any) -> bool:
    checker = getattr(value, "is_alive", None)
    return bool(checker()) if callable(checker) else False


def _mapping_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _project_root(workflow: Any) -> Path:
    root = getattr(workflow, "project_root", None)
    if root is None:
        root = getattr(workflow, "_root", ".")
    return Path(root).expanduser().resolve()


def _workflow_state(workflow: Any) -> Mapping[str, Any]:
    value = getattr(workflow, "_workflow", {})
    return value if isinstance(value, Mapping) else {}


def _latest_manifest(workflow: Any) -> tuple[dict[str, Any] | None, Path | None]:
    getter = getattr(workflow, "_latest_batch_manifest", None)
    try:
        manifest = getter() if callable(getter) else None
    except (OSError, TypeError, ValueError, RuntimeError):
        manifest = None
    if not isinstance(manifest, dict):
        manifest = None
    path_getter = getattr(workflow, "_latest_batch_path", None)
    try:
        path = path_getter() if callable(path_getter) else None
    except (OSError, TypeError, ValueError, RuntimeError):
        path = None
    return manifest, Path(path) if path is not None else None


def _load_mapping(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _artifact_root(
    workflow: Any,
    run_id: str | None,
    manifest_path: Path | None,
) -> Path | None:
    if manifest_path is not None and run_id:
        candidate = manifest_path.parent
        if candidate.name == run_id:
            return candidate
    if not run_id:
        return None
    return _project_root(workflow) / "artifacts" / "project-runs" / "batches" / run_id


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _revision_from_disk(
    batch_manifest: Mapping[str, Any] | None,
    edits_path: Path,
    *,
    expected_run_id: str | None = None,
) -> tuple[bool | None, int | None, int | None]:
    """Compare the persisted journal revision with published batch metadata."""

    edits = _load_mapping(edits_path) if edits_path.is_file() else None
    journal_run_id = edits.get("analysis_run_id") if edits is not None else None
    if (
        expected_run_id
        and journal_run_id is not None
        and str(journal_run_id) != expected_run_id
    ):
        return None, None, None
    operations = edits.get("operations") if edits is not None else None
    operation_count = len(operations) if isinstance(operations, list) else None
    reconciliation = (
        batch_manifest.get("global_reconciliation") if batch_manifest else None
    )
    diagnostics = reconciliation.get("diagnostics") if isinstance(reconciliation, Mapping) else None
    published = (
        diagnostics.get("editor_operation_count")
        if isinstance(diagnostics, Mapping)
        else None
    )
    published_count = published if type(published) is int and published >= 0 else None
    if operation_count is None:
        return None, published_count, operation_count
    if published_count is None:
        # Legacy accepted runs had no post-editor marker.  They are safe only
        # when their journal is still empty; a nonempty journal is unverified.
        return operation_count == 0, published_count, operation_count
    return operation_count == published_count, published_count, operation_count


def current_finalized_gate(workflow: Any, *, hydrate_editor: bool = True) -> dict[str, Any]:
    """Return the current-result finalization gate without changing artifacts.

    ``hydrate_editor=False`` is used by the desktop polling projection.  It
    reads the persisted batch and edit journal only, so a status refresh does
    not build the full editor membership arrays.  The report uses the hydrated
    form before calling ``final_projection``.
    """

    reasons: list[str] = []
    workflow_state = _workflow_state(workflow)
    cached_manifest, manifest_path = _latest_manifest(workflow)
    # The object returned by the workflow is only a lookup convenience.  The
    # gate must independently verify the on-disk publication so a stale
    # in-memory manifest cannot authorize the current report after restart.
    persisted_manifest = (
        _load_mapping(manifest_path)
        if isinstance(cached_manifest, dict)
        and manifest_path is not None
        and manifest_path.is_file()
        else None
    )
    manifest = persisted_manifest
    result: Any | None = None
    run_id = None

    manifest_valid = isinstance(manifest, Mapping)
    manifest_run_id = None
    manifest_status = ""
    manifest_statuses: dict[Any, Any] = {}
    manifest_has_reconciliation = False
    if manifest_valid:
        raw_run_id = manifest.get("analysis_run_id")
        manifest_run_id = str(raw_run_id) if raw_run_id else None
        run_id = manifest_run_id
        manifest_status = _status_text(manifest.get("status"))
        statuses_raw = manifest.get("span_statuses") or {}
        manifest_statuses = dict(statuses_raw) if isinstance(statuses_raw, Mapping) else {}
        reconciliation = manifest.get("global_reconciliation")
        manifest_has_reconciliation = isinstance(reconciliation, Mapping) and (
            (
                type(reconciliation.get("tree_count")) is int
                and reconciliation.get("tree_count") >= 0
            )
            or isinstance(reconciliation.get("trees"), list)
        )
        if not manifest_run_id:
            _append_reason(reasons, "批次发布缺少分析运行标识")
        if not manifest_statuses:
            _append_reason(reasons, "批次发布没有可用的已选档距结果")
        elif any(
            _status_text(status) != BatchSpanStatus.FINALIZED.value
            for status in manifest_statuses.values()
        ):
            _append_reason(reasons, "批次发布中的所有已选档距必须完成最终定稿")
        if manifest_status != "COMPLETE":
            _append_reason(reasons, "批次发布尚未完成")
        if not manifest_has_reconciliation:
            _append_reason(reasons, "批次发布缺少最终全局归并结果")
    else:
        _append_reason(reasons, "缺少或无法读取批次发布文件")

    workflow_run_id = workflow_state.get("last_batch_id")
    if workflow_run_id and (not run_id or str(workflow_run_id) != run_id):
        _append_reason(reasons, "工作流与批次发布不是同一分析运行")

    result_status = manifest_status
    result_statuses: dict[Any, Any] = {}
    result_has_reconciliation = False
    result_run_id = None
    if hydrate_editor:
        current_result = getattr(workflow, "_current_result", None)
        try:
            result = current_result() if callable(current_result) else None
        except (OSError, TypeError, ValueError, RuntimeError):
            result = None
            _append_reason(reasons, "无法加载当前批次结果")
        if result is not None:
            raw_run_id = getattr(result, "analysis_run_id", None)
            result_run_id = str(raw_run_id) if raw_run_id else None
            if not result_run_id:
                _append_reason(reasons, "当前结果缺少分析运行标识")
            if manifest_run_id and result_run_id != manifest_run_id:
                _append_reason(reasons, "内存结果与批次发布不是同一分析运行")
            if workflow_run_id and result_run_id != str(workflow_run_id):
                _append_reason(reasons, "工作流与当前结果不是同一分析运行")
            if run_id is None:
                run_id = result_run_id
            result_status = _status_text(getattr(result, "status", None))
            statuses_raw = getattr(result, "span_statuses", {}) or {}
            result_statuses = dict(statuses_raw) if isinstance(statuses_raw, Mapping) else {}
            result_has_reconciliation = getattr(result, "global_reconciled", None) is not None
            if not result_statuses:
                _append_reason(reasons, "当前结果没有可用的已选档距结果")
            elif any(
                _status_text(status) != BatchSpanStatus.FINALIZED.value
                for status in result_statuses.values()
            ):
                _append_reason(reasons, "当前结果中的所有已选档距必须完成最终定稿")
            if result_status != "COMPLETE":
                _append_reason(reasons, "当前结果尚未完成")
            if not result_has_reconciliation:
                _append_reason(reasons, "当前结果缺少最终全局归并结果")
            if set(result_statuses) != set(manifest_statuses):
                _append_reason(reasons, "当前结果与批次发布的档距集合不一致")
        elif manifest is None:
            _append_reason(reasons, "尚未加载当前批次结果")
        else:
            _append_reason(reasons, "无法加载当前批次结果")

    manifest_all_finalized = bool(manifest_statuses) and all(
        _status_text(status) == BatchSpanStatus.FINALIZED.value
        for status in manifest_statuses.values()
    )
    result_all_finalized = bool(result_statuses) and all(
        _status_text(status) == BatchSpanStatus.FINALIZED.value
        for status in result_statuses.values()
    )
    all_finalized = manifest_all_finalized and (result_all_finalized if hydrate_editor else True)
    statuses = result_statuses if hydrate_editor and result is not None else manifest_statuses

    batch_active = _thread_is_alive(getattr(workflow, "_thread", None)) or (
        _status_text(workflow_state.get("last_batch_status")) == BatchSpanStatus.RUNNING.value
    )
    if batch_active:
        _append_reason(reasons, "批量分析仍在运行")

    finalize_stage = str(getattr(workflow, "_finalize_stage", "IDLE") or "IDLE").upper()
    finalize_active = _thread_is_alive(getattr(workflow, "_finalize_thread", None)) or (
        finalize_stage in _ACTIVE_FINALIZE_STAGES
    )
    finalize_failed = finalize_stage == "FAILED"
    if finalize_active:
        _append_reason(reasons, "复核结果仍在生成")
    if finalize_failed:
        _append_reason(reasons, "上一次复核结果生成失败，请重新完成复核")

    finalize_state = workflow_state.get("finalize_state")
    if isinstance(finalize_state, Mapping):
        state_run_id = finalize_state.get("run_id")
        state_status = _status_text(finalize_state.get("status"))
        bound_to_current = state_run_id is None or str(state_run_id) == run_id
        if bound_to_current and state_status == "RUNNING":
            finalize_active = True
            _append_reason(reasons, "复核结果仍在生成（已持久化状态）")
        if bound_to_current and state_status == "FAILED":
            finalize_failed = True
            _append_reason(reasons, "持久化的复核状态为失败，请重新完成复核")

    root = _artifact_root(workflow, run_id, manifest_path)
    canonical_batch_path = root / "batch.json" if root else None
    batch_path = canonical_batch_path or manifest_path
    if canonical_batch_path is not None and manifest_path is not None:
        try:
            paths_differ = manifest_path.resolve() != canonical_batch_path.resolve()
        except OSError:
            paths_differ = True
        if paths_differ:
            _append_reason(reasons, "批次发布路径与分析运行不一致")
    edits_path = root / "tree_edits.json" if root else None
    batch_mtime_ns = None
    journal_mtime_ns = None
    revision_match: bool | None = None
    published_operation_count = None
    operation_count = None
    if batch_path is None or not batch_path.is_file():
        _append_reason(reasons, "缺少批次发布文件")
    else:
        try:
            batch_mtime_ns = batch_path.stat().st_mtime_ns
        except OSError:
            _append_reason(reasons, "无法读取批次发布文件时间")
        if not manifest_valid:
            _append_reason(reasons, "批次发布文件内容无效")
    if edits_path is None or not edits_path.is_file():
        _append_reason(reasons, "缺少当前编辑日志")
    else:
        try:
            journal_mtime_ns = edits_path.stat().st_mtime_ns
        except OSError:
            _append_reason(reasons, "无法读取当前编辑日志时间")
        if batch_mtime_ns is not None and journal_mtime_ns > batch_mtime_ns:
            _append_reason(reasons, "编辑日志晚于批次发布，当前结果已变脏")
        revision_match, published_operation_count, operation_count = _revision_from_disk(
            manifest, edits_path, expected_run_id=run_id
        )
        if revision_match is not True:
            _append_reason(reasons, "编辑日志与已发布编辑修订不一致")

    if hydrate_editor and not reasons:
        editor_getter = getattr(workflow, "_ensure_editor", None)
        try:
            editor = editor_getter() if callable(editor_getter) else None
        except (OSError, TypeError, ValueError, RuntimeError):
            editor = None
        if editor is None:
            _append_reason(reasons, "无法加载当前编辑结果")
        else:
            try:
                from desktop.review_extensions import _result_matches_editor_revision

                revision_match = bool(_result_matches_editor_revision(editor, result))
            except (AttributeError, TypeError, ValueError):
                revision_match = False
            if not revision_match:
                _append_reason(reasons, "当前编辑修订与最终归并结果不一致")

    return {
        "status": "PASS" if not reasons else "LOCKED",
        "run_id": run_id,
        "result_status": result_status,
        "span_statuses": {str(key): _status_text(value) for key, value in statuses.items()},
        "all_spans_finalized": all_finalized,
        "batch_active": batch_active,
        "finalize_active": finalize_active,
        "finalize_failed": finalize_failed,
        "editor_revision_match": revision_match,
        "published_editor_operation_count": published_operation_count,
        "editor_operation_count": operation_count,
        "journal_mtime_ns": journal_mtime_ns,
        "batch_mtime_ns": batch_mtime_ns,
        "reasons": reasons,
    }


def _wire_arrays(wire_result: Any) -> Mapping[str, Any]:
    arrays = getattr(wire_result, "w1_arrays", None)
    if not isinstance(arrays, Mapping):
        raise ValueError("persisted W1 wire arrays are unavailable")
    return arrays


def _array(arrays: Mapping[str, Any], name: str, *, ndim: int | None = None) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"persisted W1 array missing: {name}")
    value = np.asarray(arrays[name])
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"persisted W1 array has invalid shape: {name}")
    return value


def _prefix_offsets(
    arrays: Mapping[str, Any], name: str, expected_count: int, expected_end: int
) -> np.ndarray:
    offsets = _array(arrays, name, ndim=1).astype(np.int64, copy=False)
    if len(offsets) != expected_count + 1 or len(offsets) == 0:
        raise ValueError(f"persisted W1 offsets have invalid length: {name}")
    if int(offsets[0]) != 0 or int(offsets[-1]) != expected_end:
        raise ValueError(f"persisted W1 offsets have invalid boundary: {name}")
    if np.any(np.diff(offsets) < 0):
        raise ValueError(f"persisted W1 offsets are not monotone: {name}")
    return offsets


def _span_wire_segments(
    span_id: str, wire_result: Any, *, max_points: int
) -> tuple[list[dict[str, Any]], int, int, bool]:
    arrays = _wire_arrays(wire_result)
    wire_ids = _array(arrays, "wire_id", ndim=1).astype(np.int64, copy=False)
    segment_wire_id = _array(arrays, "segment_wire_id", ndim=1).astype(np.int64, copy=False)
    segment_index = _array(arrays, "segment_index", ndim=1).astype(np.int64, copy=False)
    distance_ready = _array(arrays, "segment_distance_ready", ndim=1).astype(bool, copy=False)
    sampled_xyz = _array(arrays, "sampled_curve_xyz", ndim=2).astype(np.float64, copy=False)
    if sampled_xyz.shape[1] != 3:
        raise ValueError("persisted W1 sampled_curve_xyz must have shape (N, 3)")
    segment_count = len(segment_wire_id)
    if len(segment_index) != segment_count or len(distance_ready) != segment_count:
        raise ValueError("persisted W1 segment arrays have inconsistent lengths")
    wire_offsets = _prefix_offsets(
        arrays, "wire_segment_offsets", len(wire_ids), segment_count
    )
    sample_offsets = _prefix_offsets(
        arrays, "sampled_curve_offsets", segment_count, len(sampled_xyz)
    )
    if np.any(~np.isfinite(sampled_xyz)):
        raise ValueError("persisted W1 sampled_curve_xyz contains non-finite values")
    for wire_position, wire_id in enumerate(wire_ids):
        start = int(wire_offsets[wire_position])
        stop = int(wire_offsets[wire_position + 1])
        if np.any(segment_wire_id[start:stop] != wire_id):
            raise ValueError("persisted W1 wire/segment ownership is inconsistent")

    segments: list[dict[str, Any]] = []
    source_point_count = int(sample_offsets[-1])
    remaining = int(max_points)
    truncated = False
    for position in range(segment_count):
        start = int(sample_offsets[position])
        stop = int(sample_offsets[position + 1])
        curve = sampled_xyz[start:stop]
        if len(curve) < 2:
            if bool(distance_ready[position]):
                raise ValueError("distance-ready W1 segment has fewer than two samples")
            continue
        if remaining < 2:
            truncated = True
            break
        if len(curve) > remaining:
            indices = np.linspace(0, len(curve) - 1, remaining, dtype=np.int64)
            curve = curve[indices]
            truncated = True
        remaining -= len(curve)
        segments.append(
            {
                "span_id": str(span_id),
                "wire_id": int(segment_wire_id[position]),
                "segment_index": int(segment_index[position]),
                "distance_ready": bool(distance_ready[position]),
                "xyz": curve.tolist(),
            }
        )
    return segments, source_point_count, sum(len(item["xyz"]) for item in segments), truncated


def current_wire_curves(
    result: Any, *, max_points: int = MAX_WIRE_DISPLAY_POINTS
) -> dict[str, Any]:
    """Return bounded, separately indexed persisted W1 wire segment curves.

    The helper intentionally does not consult the editor's tower schematic.
    Each returned curve is sliced through ``wire_segment_offsets`` and
    ``sampled_curve_offsets`` from the persisted ``SpanWireResult.w1_arrays``.
    """

    if type(max_points) is not int or max_points < 2:
        raise ValueError("max_points must be an integer of at least two")
    span_results = getattr(result, "span_results", {}) or {}
    if not isinstance(span_results, Mapping):
        span_results = {}
    segments: list[dict[str, Any]] = []
    source_point_count = 0
    returned_point_count = 0
    truncated = False
    diagnostics: list[str] = []
    for span_id in sorted(span_results, key=str):
        span_result = span_results[span_id]
        wire_result = _mapping_value(span_result, "wire_result")
        if wire_result is None:
            diagnostics.append(f"{span_id}: wire result unavailable")
            continue
        remaining = max_points - returned_point_count
        if remaining < 2:
            truncated = True
            break
        try:
            local, local_source_count, _, local_truncated = _span_wire_segments(
                str(span_id), wire_result, max_points=remaining
            )
        except (TypeError, ValueError, KeyError) as exc:
            diagnostics.append(f"{span_id}: {exc}")
            continue
        segments.extend(local)
        source_point_count += local_source_count
        returned_point_count += sum(len(item["xyz"]) for item in local)
        truncated = truncated or local_truncated
        if returned_point_count >= max_points:
            truncated = True
            break
    return {
        "schema": CURRENT_WIRE_SCHEMA,
        "source": "PERSISTED_SPAN_WIRE_RESULT_W1",
        "geometry_kind": "FITTED_CURRENT_GEOMETRY",
        "coordinate_space": "scene_local_m",
        "status": (
            "AVAILABLE"
            if segments and not diagnostics
            else "PARTIAL"
            if segments
            else "UNAVAILABLE"
        ),
        "segments": segments,
        "segment_count": len(segments),
        "point_count": returned_point_count,
        "source_point_count": source_point_count,
        "truncated": truncated,
        "diagnostics": diagnostics,
    }


def _bounded_tree_display(display: Any, *, max_points: int) -> dict[str, Any]:
    trees3d = display.get("trees3d") if isinstance(display, Mapping) else None
    trees3d = trees3d if isinstance(trees3d, Mapping) else {}
    raw_xyz = np.asarray(trees3d.get("xyz", []), dtype=np.float64)
    if raw_xyz.size == 0:
        raw_xyz = np.empty((0, 3), dtype=np.float64)
    if raw_xyz.ndim != 2 or raw_xyz.shape[1] != 3 or np.any(~np.isfinite(raw_xyz)):
        raise ValueError("final display trees3d.xyz must have shape (N, 3)")
    count = len(raw_xyz)
    ids = list(trees3d.get("tree_id_by_index", []))
    colors = list(trees3d.get("color", []))
    if len(ids) not in {0, count} or len(colors) not in {0, count}:
        raise ValueError("final display tree IDs/colors are not aligned with XYZ")
    keep = (
        np.linspace(0, count - 1, min(count, max_points), dtype=np.int64)
        if count
        else np.empty(0, dtype=np.int64)
    )
    return {
        "xyz": raw_xyz[keep].tolist(),
        "tree_id_by_index": [ids[int(index)] for index in keep] if ids else [],
        "color": [colors[int(index)] for index in keep] if colors else [],
        "point_count": int(len(keep)),
        "source_point_count": count,
        "bounded": True,
        "source": "FINAL_EDITOR_DISPLAY_SAMPLE",
    }


def _current_row(row: Any) -> dict[str, Any]:
    source = row if isinstance(row, Mapping) else vars(row)
    output: dict[str, Any] = {}
    for field in _CURRENT_TREE_FIELDS:
        value = source.get(field)
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        output[field] = value
    if output["persistent_tree_id"] is not None:
        output["persistent_tree_id"] = str(output["persistent_tree_id"])
    return output


def _kpis(rows: list[dict[str, Any]], result: Any) -> dict[str, Any]:
    clearance = getattr(result, "final_clearance", None)
    minimum = getattr(clearance, "global_minimum_clearance_m", None)
    if minimum is None:
        values = [
            float(row["minimum_clearance_m"])
            for row in rows
            if isinstance(row.get("minimum_clearance_m"), (int, float))
            and math.isfinite(float(row["minimum_clearance_m"]))
        ]
        minimum = min(values) if values else None
    diagnostics = getattr(clearance, "diagnostics", {})
    exact_points = (
        diagnostics.get("clearance_exact_points")
        if isinstance(diagnostics, Mapping)
        else None
    )
    return {
        "tree_count": len(rows),
        "minimum_clearance_m": minimum,
        "warning_threshold_m": (
            getattr(result, "risk_policy_snapshot", {}) or {}
        ).get("warning_clearance_threshold_m"),
        "risk_counts": dict(
            Counter(str(row.get("current_risk") or "UNKNOWN") for row in rows)
        ),
        "final_member_point_count": (
            int(exact_points) if type(exact_points) is int and exact_points >= 0 else None
        ),
    }


def build_current_report(workflow: Any) -> dict[str, Any]:
    """Build the modern current-only report, or a locked gate response."""

    gate = current_finalized_gate(workflow, hydrate_editor=True)
    base = {
        "schema": CURRENT_REPORT_SCHEMA,
        "mode": CURRENT_REPORT_MODE,
        "current_only": True,
        "gate": gate,
    }
    if gate["status"] != "PASS":
        return {
            **base,
            "status": "LOCKED",
            "message": "当前结果尚未定稿，报告已锁定；完成复核后可查看当前最终结果。",
        }

    result_getter = getattr(workflow, "_current_result", None)
    try:
        result = result_getter() if callable(result_getter) else None
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        return {
            **base,
            "status": "ERROR",
            "message": f"当前最终结果暂时不可用：{type(exc).__name__}: {exc}",
        }
    if result is None:
        return {
            **base,
            "status": "ERROR",
            "message": "当前最终结果暂时不可用，请重新打开项目。",
        }
    try:
        projection = workflow.final_projection(persist_display_evidence=False)
        rows = [_current_row(row) for row in projection.get("trees", [])]
        display = {
            "trees3d": _bounded_tree_display(
                projection.get("display"), max_points=MAX_TREE_DISPLAY_POINTS
            )
        }
        wires = current_wire_curves(result)
    except (AttributeError, KeyError, TypeError, ValueError, OSError) as exc:
        return {
            **base,
            "status": "ERROR",
            "message": f"当前最终结果暂时不可用：{type(exc).__name__}: {exc}",
        }
    return {
        **base,
        "status": "FINALIZED",
        "message": "当前最终结果（只读）",
        "kpis": _kpis(rows, result),
        "trees": rows,
        "display": display,
        "wires": wires,
        "metadata": {
            "tree_metrics_source": projection.get("metadata", {}).get("tree_metrics_source"),
            "wire_source": wires["source"],
            "wire_geometry_kind": wires["geometry_kind"],
            "coordinate_space": wires["coordinate_space"],
        },
    }


__all__ = [
    "CURRENT_REPORT_MODE",
    "CURRENT_REPORT_SCHEMA",
    "CURRENT_WIRE_SCHEMA",
    "MAX_TREE_DISPLAY_POINTS",
    "MAX_WIRE_DISPLAY_POINTS",
    "build_current_report",
    "current_finalized_gate",
    "current_wire_curves",
]
