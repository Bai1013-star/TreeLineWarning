"""Prediction_Input.xlsx export: ``Trees`` + ``Run_Metadata`` sheets.

Missing-value policy (strict):

- ``unknown`` / ``unavailable`` values are written as EMPTY cells (never ``0``).
- A missing ``crown_width_m`` must NEVER become ``0``; a crown below the
  plausibility floor (or NaN/Inf) is written as blank.
- A missing ``diameter_cm`` must NEVER become ``0``; it is blank while an
  explicit ``REVIEW``/``UNAVAILABLE`` evidence status is retained.
- Real zero and missing are therefore semantically distinct in the workbook.
- Any NaN / Inf numeric is sanitized to blank before writing.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

PREDICTION_INPUT_SCHEMA = "prediction-input-v1"
EXPORT_SCHEMA_VERSION = "folder-to-excel-export-v1"

_HEADER_FONT = Font(bold=True, color="000000")
_HEADER_FILL = PatternFill("solid", fgColor="DDEBF7")
_DANGER_FILL = PatternFill("solid", fgColor="FFC7CE")
_THIN = Side(style="thin", color="B0B0B0")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)

# Minimum plausible crown width (m) for a point-cloud-measured tree crown.
CROWN_WIDTH_PLAUSIBILITY_FLOOR_M = 0.2

TREE_SHEET = "Trees"
METADATA_SHEET = "Run_Metadata"

_TREE_COLUMNS = [
    "persistent_tree_id",
    "project_id",
    "group_id",
    "span_id",
    "start_tower_id",
    "end_tower_id",
    "height_m",
    "height_source",
    "crown_width_m",
    "crown_width_major_m",
    "crown_width_minor_m",
    "crown_area_m2",
    "crown_source",
    "diameter_cm",
    "diameter_source",
    "tree_x_m",
    "tree_y_m",
    "longitudinal_s_m",
    "lateral_n_m",
    "minimum_clearance_m",
    "danger_point_count",
    "current_risk",
    "quality_state",
    "human_edited",
    "edit_provenance",
    "observation_date",
    "science_run_id",
    # Appended Stage-B fields preserve the existing leading column contract.
    "analysis_tree_id",
    "line_id",
    "contact_state",
    "contact_evidence",
    "warning_threshold_used_m",
    "review_status",
    # Additive finalized measurement contract; retain every established column
    # position, including the terminal human-facing presentation identity.
    "dbh_cm",
    "dbh_method",
    "dbh_quality",
    "dbh_point_count",
    "dbh_fit_rmse",
    "dbh_slice_count",
    "dbh_fit_status",
    "dbh_source",
    "dbh_model_version",
    "dbh_uncertainty_cm",
    "horizontal_clearance_m",
    "vertical_clearance_m",
    "clearance_component_source",
    "display_tree_id",
    "dbh_ground_z_absolute_m",
    "dbh_measurement_z_absolute_m",
    "dbh_height_reference",
    "dbh_ground_reference_method",
    "dbh_ground_reference_point_count",
    "dbh_measurement_height_m",
    "dbh_section_lower_hag_m",
    "dbh_section_upper_hag_m",
    "dbh_coordinate_source",
]

_NUMERIC_COLUMNS = {
    "horizontal_clearance_m",
    "vertical_clearance_m",
    "height_m",
    "crown_width_m",
    "crown_width_major_m",
    "crown_width_minor_m",
    "crown_area_m2",
    "diameter_cm",
    "tree_x_m",
    "tree_y_m",
    "longitudinal_s_m",
    "lateral_n_m",
    "minimum_clearance_m",
    "danger_point_count",
    "warning_threshold_used_m",
    "dbh_cm",
    "dbh_point_count",
    "dbh_fit_rmse",
    "dbh_slice_count",
    "dbh_uncertainty_cm",
    "dbh_ground_z_absolute_m",
    "dbh_measurement_z_absolute_m",
    "dbh_ground_reference_point_count",
    "dbh_measurement_height_m",
    "dbh_section_lower_hag_m",
    "dbh_section_upper_hag_m",
}

_METADATA_ROWS = [
    ("source_file", "source_file", "源点云文件"),
    ("source_identity", "source_identity", "源点云身份"),
    ("source_sha256", "source_sha256", "源点云 SHA256"),
    ("project_id", "project_id", "项目 ID"),
    ("group_id", "group_id", "杆塔组 ID"),
    ("span_id", "span_id", "档距 ID"),
    ("start_tower_id", "start_tower_id", "起始杆塔 ID"),
    ("end_tower_id", "end_tower_id", "终止杆塔 ID"),
    ("science_run_id", "science_run_id", "科学运行 ID"),
    ("risk_policy_version", "risk_policy_version", "风险策略版本"),
    ("warning_threshold_used_m", "warning_threshold_used_m", "预警阈值(m)"),
    ("analysis_tree_id", "analysis_tree_id", "分析树 ID"),
    ("batch_status", "batch_status", "批次状态"),
    ("tree_backend", "tree_backend", "单木后端"),
    ("wire_exclusion_policy", "wire_exclusion_policy", "导线剔除策略"),
    ("tower_exclusion_policy", "tower_exclusion_policy", "杆塔剔除策略"),
    ("tower_exclusion_coverage", "tower_exclusion_coverage", "杆塔剔除覆盖"),
    ("tower_membership_source", "tower_membership_source", "杆塔成员来源"),
    ("tower_authoritative_membership", "tower_authoritative_membership", "杆塔权威成员(是/否)"),
    ("GROUND_BACKEND", "GROUND_BACKEND", "地面后端"),
    ("GROUND_CACHE", "GROUND_CACHE", "地面缓存"),
    ("GROUND_SOURCE", "GROUND_SOURCE", "地面来源"),
    (
        "ground_cache_source_identity_check",
        "ground_cache_source_identity_check",
        "地面缓存源身份校验",
    ),
    ("vegetation_input_count", "vegetation_input_count", "植被输入点数"),
    (
        "confirmed_wire_excluded_count",
        "confirmed_wire_excluded_count",
        "确认导线剔除点数",
    ),
    ("confirmed_tower_excluded_count", "confirmed_tower_excluded_count", "确认杆塔剔除点数"),
    ("final_tree_input_count", "final_tree_input_count", "最终单木输入点数"),
    ("wire_tree_input_intersection", "wire_tree_input_intersection", "导线-单木输入交集"),
    ("tower_tree_input_intersection", "tower_tree_input_intersection", "杆塔-单木输入交集"),
    ("algorithm_tree_count", "algorithm_tree_count", "算法单木数"),
    ("deleted_tree_count", "deleted_tree_count", "人工删除单木数"),
    ("merge_operation_count", "merge_operation_count", "合并操作数"),
    ("final_tree_count", "final_tree_count", "最终单木数"),
    ("wire_count", "wire_count", "导线根数"),
    ("danger_tree_count", "danger_tree_count", "危险单木数"),
    ("danger_point_count", "danger_point_count", "危险点数"),
    ("global_minimum_clearance_m", "global_minimum_clearance_m", "场景最小净空(m)"),
    ("risk_threshold_m", "risk_threshold_m", "危险净空阈值(m)"),
    ("risk_threshold_value", "risk_threshold_value", "危险净空阈值取值"),
    ("risk_threshold_source", "risk_threshold_source", "危险净空阈值来源"),
    ("risk_config_digest", "risk_config_digest", "危险净空配置摘要"),
    ("viewer_mode", "viewer_mode", "浏览器模式"),
    ("science_scope", "science_scope", "科学范围"),
    ("created_at", "created_at", "创建时间"),
    ("git_commit", "git_commit", "Git 提交"),
]


def _clean_cell(value: Any, *, numeric: bool = False) -> Any:
    """Sanitize a cell value: None -> blank, NaN/Inf -> blank, no silent 0."""

    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return None
    if numeric and isinstance(value, (int, float)):
        return value
    return value


def _sanitize_tree_row(
    row: dict[str, Any], science_run_id: str, observation_date: str
) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("science_run_id", science_run_id)
    out.setdefault("observation_date", observation_date)
    # Crown missing semantics: blank, never 0.
    crown = out.get("crown_width_m")
    crown_bad = (
        crown is None
        or not math.isfinite(float(crown))
        or float(crown) <= 0.0
        or float(crown) < CROWN_WIDTH_PLAUSIBILITY_FLOOR_M
    )
    if crown_bad:
        for key in (
            "crown_width_m",
            "crown_width_major_m",
            "crown_width_minor_m",
            "crown_area_m2",
        ):
            out[key] = None
    # DBH missing semantics: blank, never 0.
    diameter = out.get("diameter_cm")
    if diameter is None or not math.isfinite(float(diameter)) or float(diameter) <= 0.0:
        out["diameter_cm"] = None
        source = out.get("diameter_source")
        if not isinstance(source, str) or not source.strip().upper().endswith(
            ("REVIEW", "UNAVAILABLE")
        ):
            out["diameter_source"] = None
    return out


def write_prediction_input_excel(
    output_path: Path,
    rows: list[dict[str, Any]],
    run_metadata: dict[str, Any],
    *,
    created_at: str | None = None,
) -> Path:
    """Write Prediction_Input.xlsx atomically (temp + replace)."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created_at = created_at or time.strftime("%Y-%m-%dT%H:%M:%S")
    science_run_id = str(
        run_metadata.get("science_run_id")
        or f"folder_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    observation_date = time.strftime("%Y-%m-%d")

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = TREE_SHEET
    _write_tree_sheet(sheet, rows, science_run_id, observation_date)
    _write_metadata_sheet(workbook.create_sheet(METADATA_SHEET), run_metadata, created_at)

    temporary = output_path.with_name(output_path.name + ".tmp")
    workbook.save(temporary)
    temporary.replace(output_path)
    return output_path


def _write_tree_sheet(
    sheet: Any, rows: list[dict[str, Any]], science_run_id: str, observation_date: str
) -> None:
    headers = list(_TREE_COLUMNS)
    for column, name in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=column, value=name)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.border = _BORDER
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"
    for row_index, raw_row in enumerate(rows, start=2):
        row = _sanitize_tree_row(raw_row, science_run_id, observation_date)
        dangerous = row.get("current_risk") == "DANGEROUS"
        for column, name in enumerate(headers, start=1):
            cell = sheet.cell(row=row_index, column=column)
            cell.value = _clean_cell(row.get(name), numeric=name in _NUMERIC_COLUMNS)
            cell.border = _BORDER
            if name in _NUMERIC_COLUMNS and isinstance(cell.value, (int, float)):
                cell.number_format = "0.000"
            if dangerous and name in {"current_risk", "minimum_clearance_m", "danger_point_count"}:
                cell.fill = _DANGER_FILL


def _write_metadata_sheet(sheet: Any, run_metadata: dict[str, Any], created_at: str) -> None:
    sheet.cell(row=1, column=1, value="Key").font = _HEADER_FONT
    sheet.cell(row=1, column=2, value="Value").font = _HEADER_FONT
    sheet.cell(row=1, column=3, value="说明").font = _HEADER_FONT
    for cell in sheet[1]:
        cell.fill = _HEADER_FILL
        cell.border = _BORDER
    sheet.freeze_panes = "A2"
    for row_index, (key, _, label) in enumerate(_METADATA_ROWS, start=2):
        sheet.cell(row=row_index, column=1, value=key).border = _BORDER
        _value = _clean_cell(run_metadata.get(key))
        sheet.cell(row=row_index, column=2, value=_value).border = _BORDER
        sheet.cell(row=row_index, column=3, value=label).border = _BORDER
    # Explicit mandatory markers (fail-closed policy reporting).
    marker_row = len(_METADATA_ROWS) + 3
    sheet.cell(row=marker_row, column=1, value="tree_backend_required").font = _HEADER_FONT
    sheet.cell(
        row=marker_row,
        column=2,
        value=str(run_metadata.get("tree_backend") or "UNSPECIFIED"),
    ).border = _BORDER
    sheet.cell(
        row=marker_row + 1, column=1, value="wire_exclusion_policy_required"
    ).font = _HEADER_FONT
    sheet.cell(
        row=marker_row + 1,
        column=2,
        value=str(run_metadata.get("wire_exclusion_policy") or "UNSPECIFIED"),
    ).border = _BORDER
    sheet.cell(
        row=marker_row + 2, column=1, value="tower_exclusion_policy_required"
    ).font = _HEADER_FONT
    sheet.cell(
        row=marker_row + 2,
        column=2,
        value=str(run_metadata.get("tower_exclusion_policy") or "UNSPECIFIED"),
    ).border = _BORDER
    sheet.column_dimensions["A"].width = 34
    sheet.column_dimensions["B"].width = 46
    sheet.column_dimensions["C"].width = 30


__all__ = [
    "PREDICTION_INPUT_SCHEMA",
    "EXPORT_SCHEMA_VERSION",
    "TREE_SHEET",
    "METADATA_SHEET",
    "CROWN_WIDTH_PLAUSIBILITY_FLOOR_M",
    "write_prediction_input_excel",
]
