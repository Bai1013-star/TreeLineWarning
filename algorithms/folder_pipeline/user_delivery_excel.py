"""Human-facing, one-observation-per-tree workbook matching the supplied CSV."""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

DELIVERY_FILENAME = "树木风险结果.xlsx"
DELIVERY_COLUMNS = (
    "tree_id", "species", "observation_date", "observation_year",
    "observation_month", "observation_day", "height", "age", "diameter",
    "vertical_distance", "horizontal_distance", "clearance_distance",
    "wind_offset_distance", "voltage_level_kv", "maximum_sag_winter",
    "maximum_sag_summer", "wind_speed", "rainfall", "humidity",
    "temperature", "sunlight_hours", "risk_score", "risk_level",
)
_RISK_LABELS = {
    "DANGER": "高风险",
    "DANGEROUS": "高风险",  # folder_pipeline's existing risk vocabulary
    "WARNING": "中风险",
    "SAFE": "低风险",
}
_DECIMAL_COLUMNS = {
    "height", "age", "diameter", "vertical_distance", "horizontal_distance",
    "clearance_distance", "wind_offset_distance", "maximum_sag_winter",
    "maximum_sag_summer", "wind_speed", "rainfall", "humidity",
    "temperature", "sunlight_hours", "risk_score",
}
_FIT_REASON_ZH = {
    "NO_CROWN_TO_STEM_IDENTITY": "未找到与当前树冠可靠对应的树干",
    "NO_ABSOLUTE_CANDIDATE": "树干几何证据未通过自动确认",
    "TOP_CANDIDATE_CONFLICT": "候选树干与其他树存在归属冲突",
    "EDITOR_REMOVED_BH_EVIDENCE": "胸高处点已由人工删点操作剔除",
    "FOLDER_BH_GEOMETRY_UNAVAILABLE": "当前树缺少可用的胸高宽度数据",
    "LOCAL_BASE_HEIGHT_AMBIGUOUS": "本树根部有多个可能地面高度，需人工确认树根地面",
    "LOCAL_BASE_HEIGHT_UNVERIFIED": "本树根部地面尚未确认，需指定本树地面点",
    "MULTIPLE_STEM_AXES_REVIEW_REQUIRED": "本树有多根候选树干轴，需选择对应的一根",
    "PER_TREE_STEM_EVIDENCE_REQUIRED": "缺少本树可核验的树干和胸高点证据，需逐树复核",
    "SEARCH_BUDGET_EXHAUSTED": "树干原始点搜索达到预算上限，需缩小范围或指定树干证据",
}
_ACTIONABLE_FIT_STATUSES = {
    "LOCAL_BASE_HEIGHT_AMBIGUOUS",
    "LOCAL_BASE_HEIGHT_UNVERIFIED",
    "MULTIPLE_STEM_AXES_REVIEW_REQUIRED",
    "PER_TREE_STEM_EVIDENCE_REQUIRED",
}
_LEGACY_MEASURED_SOURCES = {
    "DIRECT_STEM", "RECOVERED_STEM", "REVIEW_CONFIRMED_STEM",
    "FOLDER_BH_WIDTH_MEASUREMENT",
}


class DeliveryDiameterMissingError(ValueError):
    """Compatibility name for callers of older delivery behavior."""


def _measurement_reason(code: Any) -> str:
    if not code:
        return "当前没有可信的本树胸高几何测量"
    selected = str(code)
    reason = _FIT_REASON_ZH.get(selected)
    if reason is None and selected.startswith("SEARCH_BUDGET"):
        reason = _FIT_REASON_ZH["SEARCH_BUDGET_EXHAUSTED"]
    return f"{reason or '本树胸高几何测量证据不足'}（{selected}）"


def _unmeasured_reason_code(source: dict[str, Any]) -> Any:
    status = str(source.get("dbh_fit_status") or "")
    if status in _ACTIONABLE_FIT_STATUSES or status.startswith("SEARCH_BUDGET"):
        return status
    return source.get("dbh_pre_reference_failure_reason") or status


def _current_diameter(source: dict[str, Any]) -> float | None:
    method = source.get("dbh_method") or source.get("diameter_source")
    if method in {"REFERENCE_ESTIMATE", "UNAVAILABLE"} or source.get(
        "diameter_source"
    ) == "REFERENCE_ESTIMATE":
        return None
    value = source.get("dbh_cm")
    if value is None and source.get("dbh_method") is None and source.get(
        "diameter_source"
    ) in _LEGACY_MEASURED_SOURCES:
        value = source.get("diameter_cm")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def _diameter_comment(source: dict[str, Any], diameter: Any) -> Comment:
    method = source.get("dbh_method") or source.get("diameter_source")
    quality = source.get("dbh_quality")
    uncertainty = source.get("dbh_uncertainty_cm")
    model = source.get("dbh_model_version")
    if method == "REFERENCE_ESTIMATE":
        lines = ["胸径留空：旧项目参考估算已撤销，不能作为本树胸径交付。"]
        lines.append("本树未测原因：" + _measurement_reason(_unmeasured_reason_code(source)))
    elif diameter is None:
        lines = ["胸径留空：当前没有可信的本树胸高直径证据。"]
        lines.append("本树未测原因：" + _measurement_reason(_unmeasured_reason_code(source)))
    else:
        estimated = str(quality).upper() in {"ESTIMATE", "ESTIMATED"} or (
            "ESTIMATE" in str(method).upper()
        )
        if estimated:
            if "UPPER_PARTIAL_ARC" in str(model):
                lines = ["胸径来源：本树胸高点云拟合（单侧估算）。"]
            elif "UPPER_CIRCULAR" in str(model):
                lines = ["胸径来源：本树上部树干外推估算。"]
            else:
                lines = ["胸径来源：本树点云几何估算（ESTIMATE）。"]
        elif method == "FOLDER_BH_WIDTH_MEASUREMENT":
            lines = ["胸径来源：点云胸高宽度测量；不作为参考均值训练样本。"]
        elif method in {"DIRECT_STEM", "RECOVERED_STEM", "REVIEW_CONFIRMED_STEM"}:
            lines = ["胸径来源：点云实测。"]
        else:
            lines = ["胸径为当前结果值；请核对下列方法和质量标记。"]
        if method and method != "UNAVAILABLE":
            lines.append(f"方法：{method}")
        if quality and quality != "UNAVAILABLE":
            lines.append(f"质量：{quality}")
        if model:
            lines.append(f"模型：{model}")
        if isinstance(uncertainty, (int, float)):
            if estimated:
                lines.append(f"估算不确定度约 {uncertainty:.2f} cm（非置信区间）")
            else:
                lines.append(f"不确定度：{uncertainty:.2f} cm")
    comment = Comment("\n".join(lines), "TreeLineWarning")
    comment.width = 380
    comment.height = max(140, min(420, 24 * len(lines)))
    return comment


def _observation(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            pass
    # The internal writer always supplies this date. An invalid value must not
    # silently become the export date, because that would misstate provenance.
    raise ValueError("DELIVERY_OBSERVATION_DATE_INVALID")


def _delivery_row(source: dict[str, Any]) -> dict[str, Any]:
    observed = _observation(source.get("observation_date"))
    risk = source.get("current_risk")
    return {
        "tree_id": source.get("display_tree_id") or source.get("persistent_tree_id"),
        "species": None,
        "observation_date": f"{observed.year}/{observed.month}/{observed.day}",
        "observation_year": observed.year,
        "observation_month": observed.month,
        "observation_day": observed.day,
        "height": source.get("height_m"),
        "age": None,
        "diameter": _current_diameter(source),
        "vertical_distance": source.get("vertical_clearance_m"),
        "horizontal_distance": source.get("horizontal_clearance_m"),
        "clearance_distance": source.get("minimum_clearance_m"),
        "wind_offset_distance": None,
        "voltage_level_kv": None,
        "maximum_sag_winter": None,
        "maximum_sag_summer": None,
        "wind_speed": None,
        "rainfall": None,
        "humidity": None,
        "temperature": None,
        "sunlight_hours": None,
        "risk_score": None,
        "risk_level": _RISK_LABELS.get(str(risk)) if risk is not None else None,
    }


def archive_internal_prediction(private_path: Path | str) -> Path:
    """Retain a prior internal workbook before replacing it with a new run."""
    private = Path(private_path)
    archived = private.with_name(f"Prediction_Input.legacy-{uuid4().hex}.xlsx")
    private.replace(archived)
    return archived


def archive_stale_delivery(delivery_path: Path | str, private_dir: Path | str) -> Path | None:
    """Keep an obsolete user workbook privately when current DBH blocks delivery."""
    delivery = Path(delivery_path)
    if not delivery.is_file():
        return None
    private = Path(private_dir)
    private.mkdir(parents=True, exist_ok=True)
    archived = private / f"树木风险结果.legacy-{uuid4().hex}.xlsx"
    delivery.replace(archived)
    return archived


def migrate_legacy_prediction(
    legacy_path: Path | str,
    private_path: Path | str,
    *,
    replacing: bool = False,
) -> Path:
    """Move a former public internal workbook out of the user export folder."""
    legacy = Path(legacy_path)
    private = Path(private_path)
    if not legacy.is_file() or legacy.resolve() == private.resolve():
        return private
    private.parent.mkdir(parents=True, exist_ok=True)
    destination = private
    if private.exists() or replacing:
        destination = private.with_name(f"Prediction_Input.legacy-{uuid4().hex}.xlsx")
    legacy.replace(destination)
    return private


def write_delivery_from_prediction(
    prediction_path: Path | str, output_path: Path | str | None = None
) -> Path:
    """Project the validated internal workbook into the strict 23-column view."""
    source_path = Path(prediction_path)
    output_path = (
        Path(output_path) if output_path is not None
        else source_path.with_name(DELIVERY_FILENAME)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_book = load_workbook(source_path, read_only=True, data_only=True)
    try:
        values = source_book["Trees"].iter_rows(values_only=True)
        headers = tuple(next(values))
        required = {
            "persistent_tree_id", "observation_date", "height_m", "dbh_cm",
            "vertical_clearance_m", "horizontal_clearance_m",
            "minimum_clearance_m", "current_risk",
        }
        if not required.issubset(headers):
            raise ValueError("DELIVERY_INTERNAL_COLUMNS_MISSING")
        output_book = Workbook()
        sheet = output_book.active
        sheet.title = "树木风险结果"
        sheet.append(DELIVERY_COLUMNS)
        sheet.row_dimensions[1].height = 32
        for index, cell in enumerate(sheet[1], start=1):
            cell.font = Font(bold=True)
            cell.fill = PatternFill("solid", fgColor="DDEBF7")
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            sheet.column_dimensions[get_column_letter(index)].width = max(
                16, len(DELIVERY_COLUMNS[index - 1]) + 3
            )
        for raw in values:
            if not any(value is not None for value in raw):
                continue
            source = dict(zip(headers, raw, strict=True))
            mapped = _delivery_row(source)
            sheet.append([mapped[name] for name in DELIVERY_COLUMNS])
            for index, name in enumerate(DELIVERY_COLUMNS, start=1):
                if name in _DECIMAL_COLUMNS:
                    sheet.cell(sheet.max_row, index).number_format = "0.00"
            sheet.cell(sheet.max_row, DELIVERY_COLUMNS.index("diameter") + 1).comment = (
                _diameter_comment(source, mapped["diameter"])
            )
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = f"A1:W{sheet.max_row}"
        temporary = output_path.with_name(output_path.name + f".{uuid4().hex}.tmp")
        try:
            output_book.save(temporary)
            temporary.replace(output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return output_path
    finally:
        source_book.close()
