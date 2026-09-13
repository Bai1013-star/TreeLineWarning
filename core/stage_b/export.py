"""Final-tree-only Prediction_Input.xlsx export for Stage-B batches."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from algorithms.folder_pipeline.export_excel import write_prediction_input_excel
from algorithms.manual_span.metrics import CROWN_WIDTH_STATISTIC, TREE_HEIGHT_STATISTIC

from core.stage_b.batch import BatchAnalysisResult, BatchSpanStatus
from core.stage_b.contact import contact_evidence_from_clearance
from core.stage_b.dbh_recovery import load_finalized_dbh
from core.stage_b.measurement_fields import clearance_components, dbh_fields
from core.stage_b.tree_display_id import assign_tree_display_ids, ordered_product_spans
from core.stage_b.tree_metrics import compute_stage_b_tree_metrics


def _project_for_output(output_path: Path) -> Any | None:
    """Resolve the Stage-A project owning a desktop export without hard-coded paths."""

    from core.stage_a import load_project

    for candidate in (output_path.parent, *output_path.parents):
        if (candidate / "stage-a-project.json").is_file():
            try:
                return load_project(candidate)
            except (OSError, TypeError, ValueError):
                return None
    return None


def _automatic_display_identity(
    output_path: Path,
    reconciled: Any,
    *,
    project_id: str | None = None,
) -> dict[int, dict[str, Any]]:
    project = _project_for_output(output_path)
    if project is None or (
        project_id is not None and project.project_id != project_id
    ):
        return {}
    return assign_tree_display_ids(project, reconciled.global_tree_records)


def _topology_identity(
    output_path: Path,
    reconciled: Any,
    *,
    project_id: str | None = None,
) -> dict[int, dict[str, str]]:
    """Resolve each record's primary Span to persisted Tower identities.

    Missing project metadata, malformed legacy records, or an out-of-range
    internal Span ID produce no automatic identity.  This preserves the
    historical fail-closed export behavior for old projects while allowing
    current projects to recover the exact endpoint pair after restart.
    """

    project = _project_for_output(output_path)
    if project is None or (
        project_id is not None and project.project_id != project_id
    ):
        return {}
    ordered_spans = ordered_product_spans(project)
    result: dict[int, dict[str, str]] = {}
    for record in getattr(reconciled, "global_tree_records", ()):
        try:
            tree_id = int(record.global_tree_id)
            span_index = int(record.span_id)
        except (AttributeError, TypeError, ValueError):
            continue
        if not 0 <= span_index < len(ordered_spans):
            continue
        span = ordered_spans[span_index]
        result[tree_id] = {
            "span_id": span.span_id,
            "start_tower_id": span.start_tower_id,
            "end_tower_id": span.end_tower_id,
        }
    return result


def _merge_identity(
    automatic: Mapping[str, Any],
    override: Mapping[str, Any],
    topology: Mapping[str, Any],
    *,
    display_span_id: Any = None,
) -> dict[str, Any]:
    """Merge presentation/caller metadata while enforcing persisted topology."""

    extra = dict(automatic)
    extra.update(dict(override))
    authoritative_span_id = topology.get("span_id")
    supplied_span_id = override.get("span_id")
    if (
        authoritative_span_id is not None
        and supplied_span_id is not None
        and supplied_span_id not in {authoritative_span_id, display_span_id}
    ):
        raise ValueError("EXPORT_IDENTITY_CONFLICTS_WITH_PROJECT_TOPOLOGY")
    for key in ("start_tower_id", "end_tower_id"):
        authoritative = topology.get(key)
        if authoritative is None:
            continue
        supplied = override.get(key)
        if supplied is not None and supplied != authoritative:
            raise ValueError("EXPORT_IDENTITY_CONFLICTS_WITH_PROJECT_TOPOLOGY")
        extra[key] = authoritative
    return extra


def export_batch_prediction(
    result: BatchAnalysisResult,
    output_path: Path | str,
    *,
    final_reconciled: Any | None = None,
    identity_by_tree: Mapping[int, Mapping[str, Any]] | None = None,
    progress: Any = None,
    timing_sink: dict[str, float] | None = None,
) -> Path:
    """Export only a globally reconciled, finalized tree set.

    ``final_reconciled`` is the post-editor result.  Requiring it when the
    batch is not explicitly finalized prevents raw candidates, legacy
    references, or pre-editor memberships from entering the prediction file.

    DBH/crown values are computed from the same exact post-editor membership
    using the existing canonical full-resolution metric implementation.  If an
    unambiguous source-bound Ground context cannot be recovered, optional
    geometry cells remain blank rather than being guessed.

    ``display_tree_id`` is presentation-only.  The durable
    ``persistent_tree_id`` remains unchanged for Prediction and restart logic.
    """

    total_started = time.perf_counter()
    output_path = Path(output_path)
    statuses = tuple(result.span_statuses.values())
    if any(
        status not in {BatchSpanStatus.FINALIZED, BatchSpanStatus.USER_REVIEWED}
        for status in statuses
    ):
        raise ValueError("PREDICTION_EXPORT_REQUIRES_FINALIZED_BATCH")
    reconciled = final_reconciled or result.global_reconciled
    if reconciled is None:
        raise ValueError("PREDICTION_EXPORT_REQUIRES_FINAL_RECONCILIATION")
    if reconciled.diagnostics.get("editor_schema") != "treeline-stage-b-batch-editor-v1":
        raise ValueError("PREDICTION_EXPORT_REQUIRES_POST_EDITOR_MEMBERSHIP")
    metrics_by_tree = compute_stage_b_tree_metrics(result, reconciled, output_path)
    dbh_rows = load_finalized_dbh(output_path.parent.parent, result, reconciled)
    clearance_by_tree = {
        record.global_tree_id: record
        for record in (result.final_clearance.records if result.final_clearance else ())
    }
    automatic_identity = _automatic_display_identity(
        output_path,
        reconciled,
        project_id=result.project_id,
    )
    topology_identity = _topology_identity(
        output_path,
        reconciled,
        project_id=result.project_id,
    )
    if progress is not None:
        progress("RECALCULATING_RISK")
    risk_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for record in reconciled.global_tree_records:
        automatic_for_record = dict(automatic_identity.get(record.global_tree_id, {}))
        extra = _merge_identity(
            automatic_for_record,
            (identity_by_tree or {}).get(record.global_tree_id, {}),
            topology_identity.get(record.global_tree_id, {}),
            display_span_id=record.span_name,
        )
        metric = metrics_by_tree.get(int(record.global_tree_id))
        identity_state = str(getattr(record, "identity_state", ""))
        if identity_state.startswith("HUMAN_"):
            extra.setdefault("human_edited", True)
            extra.setdefault("edit_provenance", identity_state.removeprefix("HUMAN_"))
        clearance = clearance_by_tree.get(record.global_tree_id)
        evidence = contact_evidence_from_clearance(
            None if clearance is None else clearance.minimum_clearance_m,
            clearance_status="UNKNOWN" if clearance is None else clearance.status,
        )
        clearance_value = None if clearance is None else clearance.minimum_clearance_m
        assessment = None
        if clearance_value is not None:
            from core.stage_a.risk import RiskPolicy

            assessment = RiskPolicy(
                risk_policy_version=str(
                    result.risk_policy_snapshot.get("risk_policy_version", "risk-policy-v1")
                ),
                warning_clearance_threshold_m=float(
                    result.risk_policy_snapshot["warning_clearance_threshold_m"]
                ),
            ).classify(
                clearance_m=clearance_value,
                contact=(
                    False
                    if evidence.state.value == "CONTACT_NOT_CONFIRMED"
                    else None
                ),
                review_required=record.review or evidence.review_required,
            )
        dbh_m = None if metric is None else metric.dbh.dbh_m
        rows.append(
            {
                "persistent_tree_id": record.persistent_tree_id,
                "display_tree_id": extra.get("display_tree_id"),
                "analysis_tree_id": f"analysis-tree-{record.global_tree_id:06d}",
                "project_id": result.project_id,
                "line_id": extra.get("line_id", record.line_id),
                "group_id": extra.get("group_id", record.line_id),
                "span_id": extra.get("span_id", record.span_name),
                "start_tower_id": extra.get("start_tower_id"),
                "end_tower_id": extra.get("end_tower_id"),
                "height_m": (
                    metric.tree_height_m
                    if metric is not None and metric.tree_height_m is not None
                    else record.apex_height_m
                ),
                "height_source": (
                    TREE_HEIGHT_STATISTIC
                    if metric is not None and metric.tree_height_m is not None
                    else None
                ),
                "crown_width_m": None if metric is None else metric.crown_mean_width_m,
                "crown_width_major_m": (
                    None if metric is None else metric.crown_major_width_m
                ),
                "crown_width_minor_m": (
                    None if metric is None else metric.crown_minor_width_m
                ),
                "crown_area_m2": None if metric is None else metric.crown_area_m2,
                "crown_source": None if metric is None else CROWN_WIDTH_STATISTIC,
                "diameter_cm": None if dbh_m is None else float(dbh_m) * 100.0,
                "diameter_source": (
                    None
                    if metric is None
                    else f"BREAST_HEIGHT_1P30M_{metric.dbh.dbh_quality}"
                ),
                "tree_x_m": record.apex_xy_local_m[0],
                "tree_y_m": record.apex_xy_local_m[1],
                "quality_state": (
                    "REVIEW_REQUIRED"
                    if record.review or evidence.review_required
                    else "VALID"
                ),
                "review_status": "USER_REVIEWED",
                "minimum_clearance_m": clearance_value,
                **clearance_components(clearance),
                "danger_point_count": (
                    None if clearance is None else clearance.danger_point_count
                ),
                "current_risk": None if assessment is None else assessment.risk.value,
                "contact_state": extra.get("contact_state", evidence.state.value),
                "contact_evidence": extra.get("contact_evidence", evidence.evidence),
                "warning_threshold_used_m": result.risk_policy_snapshot.get(
                    "warning_clearance_threshold_m"
                ),
                "human_edited": bool(extra.get("human_edited", False)),
                "edit_provenance": extra.get("edit_provenance", "TREE_EDITOR_FINAL"),
            }
        )
    for row in rows:
        tree_id = int(str(row["analysis_tree_id"]).removeprefix("analysis-tree-"))
        row.update(dbh_fields(metrics_by_tree.get(tree_id), dbh_rows.get(tree_id)))
    risk_seconds = time.perf_counter() - risk_started
    metadata = {
        "project_id": result.project_id,
        "science_run_id": result.analysis_run_id,
        "tree_backend": "crown_stem_hybrid",
        "wire_exclusion_policy": "EXACT_ORIGINAL_INDEX",
        "science_scope": "FINAL_GLOBAL_RECONCILED_POST_EDITOR",
        "batch_status": result.status,
        "risk_policy_version": result.risk_policy_snapshot.get("risk_policy_version"),
        "warning_threshold_used_m": result.risk_policy_snapshot.get(
            "warning_clearance_threshold_m"
        ),
        "prediction_source": "FINAL_GLOBAL_RECONCILED_POST_EDITOR",
        "tree_metrics_source": "CANONICAL_FULLRES_POST_EDITOR",
        "tree_display_id_schema": "treeline-tree-display-id-v1",
    }
    if progress is not None:
        progress("GENERATING_EXCEL")
    excel_started = time.perf_counter()
    written = write_prediction_input_excel(output_path, rows, metadata)
    if timing_sink is not None:
        timing_sink.update(
            {
                "RISK_UPDATE_TIME": float(risk_seconds),
                "XLSX_GENERATION_TIME": float(time.perf_counter() - excel_started),
                "EXPORT_TOTAL_TIME": float(time.perf_counter() - total_started),
            }
        )
    return written


__all__ = ["export_batch_prediction"]
