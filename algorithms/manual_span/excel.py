"""Excel delivery: ``tree_clearance_report.xlsx``.

One row per final tree with every required metric; a bold frozen filtered
header; dangerous rows clearly highlighted red; an additional ``Span Summary``
sheet and a ``Parameters`` sheet.  Normal text is black, headers bold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from algorithms.manual_span.pipeline import ManualSingleSpanRun

_HEADER_FONT = Font(bold=True, color="000000")
_HEADER_FILL = PatternFill("solid", fgColor="DDEBF7")
_DANGER_FILL = PatternFill("solid", fgColor="FFC7CE")
_THIN = Side(style="thin", color="B0B0B0")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)

_HEADERS = [
    "Tree ID",
    "Span ID",
    "Tree Height (m)",
    "DBH (m)",
    "DBH Quality",
    "Crown Major Width (m)",
    "Crown Minor Width (m)",
    "Crown Mean Width (m)",
    "Crown Area (m²)",
    "Full-resolution Point Count",
    "Nearest Wire ID",
    "Nearest Tree Point Original Index",
    "Nearest Tree Point X",
    "Nearest Tree Point Y",
    "Nearest Tree Point Z",
    "Nearest Wire Point X",
    "Nearest Wire Point Y",
    "Nearest Wire Point Z",
    "Horizontal Clearance (m)",
    "Vertical Clearance (m)",
    "3D Minimum Clearance (m)",
    "Risk Threshold (m)",
    "Danger Point Count",
    "Dangerous?",
    "Quality / Review Notes",
]

_FLOAT_FIELDS = {
    "Tree Height (m)": "0.00",
    "DBH (m)": "0.000",
    "Crown Major Width (m)": "0.00",
    "Crown Minor Width (m)": "0.00",
    "Crown Mean Width (m)": "0.00",
    "Crown Area (m²)": "0.00",
    "Horizontal Clearance (m)": "0.00",
    "Vertical Clearance (m)": "0.00",
    "3D Minimum Clearance (m)": "0.00",
    "Risk Threshold (m)": "0.00",
}

_INT_FIELDS = {
    "Tree ID",
    "Full-resolution Point Count",
    "Nearest Wire ID",
    "Nearest Tree Point Original Index",
    "Danger Point Count",
}


def _row_values(record: Any) -> list[Any]:
    nearest_tree = record.nearest_tree_xyz_absolute_m or (None, None, None)
    nearest_wire = record.nearest_wire_xyz_absolute_m or (None, None, None)
    return [
        record.global_tree_id + 1,
        record.span_name,
        record.tree_height_m,
        record.dbh_m,
        record.dbh_quality,
        record.crown_major_width_m,
        record.crown_minor_width_m,
        record.crown_mean_width_m,
        record.crown_area_m2,
        record.fullres_point_count,
        record.nearest_wire_id,
        record.nearest_tree_original_index,
        nearest_tree[0],
        nearest_tree[1],
        nearest_tree[2],
        nearest_wire[0],
        nearest_wire[1],
        nearest_wire[2],
        record.horizontal_clearance_m,
        record.vertical_clearance_m,
        record.clearance_3d_m,
        record.risk_threshold_m,
        record.danger_point_count,
        "是" if record.dangerous else "否",
        record.quality_notes,
    ]


def write_tree_clearance_excel(run: ManualSingleSpanRun, output_path: Path) -> Path:
    """Write the full Excel workbook and return its path."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    report = workbook.active
    report.title = "Tree Clearance Report"

    report.append(_HEADERS)
    for col in range(1, len(_HEADERS) + 1):
        cell = report.cell(row=1, column=col)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = _BORDER
    report.freeze_panes = "A2"
    report.auto_filter.ref = f"A1:{get_column_letter(len(_HEADERS))}1"

    for record in run.risk.records:
        report.append(_row_values(record))
        row_index = report.max_row
        dangerous = bool(record.dangerous)
        for col in range(1, len(_HEADERS) + 1):
            cell = report.cell(row=row_index, column=col)
            cell.border = _BORDER
            header = _HEADERS[col - 1]
            if header in _FLOAT_FIELDS and isinstance(cell.value, (int, float)):
                cell.number_format = _FLOAT_FIELDS[header]
            if header in _INT_FIELDS and isinstance(cell.value, (int, float)):
                cell.number_format = "0"
            if dangerous:
                cell.fill = _DANGER_FILL
                cell.font = Font(color="9C0006")

    widths = [
        8, 16, 14, 10, 12, 16, 16, 16, 12, 16, 12, 16, 14, 14, 14,
        14, 14, 14, 16, 14, 16, 12, 12, 10, 30,
    ]
    for index, width in enumerate(widths, start=1):
        report.column_dimensions[get_column_letter(index)].width = width

    # ---- Span Summary ----
    summary = workbook.create_sheet("Span Summary")
    summary_rows = _span_summary_rows(run)
    for position, (label, value) in enumerate(summary_rows, start=1):
        summary.cell(row=position, column=1, value=label).font = Font(bold=True)
        summary.cell(row=position, column=2, value=value)
    summary.column_dimensions["A"].width = 30
    summary.column_dimensions["B"].width = 60

    # ---- Parameters ----
    params = workbook.create_sheet("Parameters")
    params.cell(row=1, column=1, value="Parameter").font = Font(bold=True)
    params.cell(row=1, column=2, value="Value").font = Font(bold=True)
    params.cell(row=1, column=3, value="Source").font = Font(bold=True)
    for col in range(1, 4):
        params.cell(row=1, column=col).fill = _HEADER_FILL
    parameter_rows = _parameter_rows(run)
    for position, (name, value, source) in enumerate(parameter_rows, start=2):
        params.cell(row=position, column=1, value=name)
        params.cell(row=position, column=2, value=value)
        params.cell(row=position, column=3, value=source)
    params.column_dimensions["A"].width = 40
    params.column_dimensions["B"].width = 22
    params.column_dimensions["C"].width = 22
    params.freeze_panes = "A2"

    temporary = output_path.with_name(output_path.name + ".tmp.xlsx")
    workbook.save(temporary)
    temporary.replace(output_path)
    return output_path


def _span_summary_rows(run: ManualSingleSpanRun) -> list[tuple[str, Any]]:
    span = run.span
    risk = run.risk
    return [
        ("source", str(run.source)),
        ("source_sha256", run.source_sha256),
        ("source_id", run.source_id),
        ("Span", span.span_name if span else ""),
        ("Pole A (base)", _fmt_xyz(span.start_xyz_absolute_m if span else None)),
        ("Pole A (top z)", f"{span.start_pole_top_z_m:.2f}" if span else ""),
        ("Pole B (base)", _fmt_xyz(span.end_xyz_absolute_m if span else None)),
        ("Pole B (top z)", f"{span.end_pole_top_z_m:.2f}" if span else ""),
        ("span_length_m", f"{span.length_m:.2f}" if span else ""),
        ("tree_count", len(run.risk.records) if risk else 0),
        ("wire_count", run.wire_outcome.strand_count if run.wire_outcome else 0),
        ("risk_threshold_m", f"{risk.threshold_m:.2f}" if risk else ""),
        ("dangerous_tree_count", risk.dangerous_tree_count if risk else 0),
        ("danger_point_count", risk.danger_point_count if risk else 0),
        (
            "global_minimum_clearance_m",
            _fmt_opt(run.clearance.clearance.global_minimum_clearance_m if run.clearance else None),
        ),
        (
            "clearance_runtime_s",
            _fmt_opt(
                run.clearance.clearance.diagnostics.get("runtime_seconds")
                if run.clearance
                else None
            ),
        ),
        ("pipeline_version", "MANUAL_SINGLE_SPAN_v1"),
        ("method_marker", "NON_CERTIFIED_RESEARCH_DEMONSTRATION"),
    ]


def _parameter_rows(run: ManualSingleSpanRun) -> list[tuple[str, Any, str]]:
    rows: list[tuple[str, Any, str]] = []
    cfg = run.cfg
    for name in (
        "vegetation_fine_voxel_size_m",
        "wire_fine_voxel_size_m",
        "chm_resolution_m",
        "chm_smooth_sigma_m",
        "tree_minimum_hag_m",
        "tree_top_min_prominence_m",
        "tree_min_final_members",
        "wire_dbscan_eps_m",
        "wire_dbscan_min_samples",
        "wire_min_strand_points",
        "wire_residual_tolerance_m",
        "span_half_width_m",
        "span_halo_m",
        "clearance_bvh_leaf_cap",
        "clearance_tie_tolerance_m",
    ):
        rows.append((name, getattr(cfg, name), "FastBaselineConfig"))
    manual = run.manual
    for name in (
        "refine_search_radius_m",
        "refine_min_pole_height_m",
        "refine_max_footprint_m",
        "refine_verticality_min",
        "refine_linearity_min",
        "clearance_chunk_points",
        "danger_point_cap_m",
        "dbh_band_center_m",
        "dbh_measured_min_points",
    ):
        rows.append((name, getattr(manual, name), "ManualSpanConfig"))
    return rows


def _fmt_xyz(xyz: Any) -> str:
    if xyz is None:
        return ""
    return f"({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})"


def _fmt_opt(value: Any) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


__all__ = ["write_tree_clearance_excel"]
