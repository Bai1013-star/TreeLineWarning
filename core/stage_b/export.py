"""Final-tree-only Prediction_Input.xlsx export for Stage-B batches."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from algorithms.folder_pipeline.export_excel import write_prediction_input_excel
from algorithms.manual_span.metrics import CROWN_WIDTH_STATISTIC, TREE_HEIGHT_STATISTIC

from core.stage_b.batch import BatchAnalysisResult, BatchSpanStatus
from core.stage_b.contact import contact_evidence_from_clearance
from core.stage_b.tree_metrics import compute_stage_b_tree_metrics


def export_batch_prediction(
    result: BatchAnalysisResult,
    output_path: Path | str,
    *,
    final_reconciled: Any | None = None,
    identity_by_tree: Mapping[int, Mapping[str, Any]] | None = None,
    progress: Any = None,
) -> Path:
    """Export only a globally reconciled, finalized tree set.

    ``final_reconciled`` is the post-editor result.  Requiring it when the
    batch is not explicitly finalized prevents raw candidates, legacy
    references, or pre-editor memberships from entering the prediction file.

    DBH/crown values are computed from the same exact post-editor membership
    using the existing canonical full-resolution metric implementation.  If an
    unambiguous source-bound Ground context cannot be recovered, optional
    geometry cells remain blank rather than being guessed.
    """

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
    clearance_by_tree = {
        record.global_tree_id: record
        for record in (result.final_clearance.records if result.final_clearance else ())
    }
    if progress is not None:
        progress("RECALCULATING_RISK")
    rows: list[dict[str, Any]] = []
    for record in reconciled.global_tree_records:
        extra = dict((identity_by_tree or {}).get(record.global_tree_id, {}))
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
    }
    if progress is not None:
        progress("GENERATING_EXCEL")
    return write_prediction_input_excel(Path(output_path), rows, metadata)


__all__ = ["export_batch_prediction"]
