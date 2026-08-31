from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from core.stage_a import CoordinateDeclaration, Line, Project, Span, Tower, save_project
from core.stage_b import BatchSpanStatus
from core.stage_b.export import export_batch_prediction
from openpyxl import load_workbook


def test_stage_b_export_adds_display_id_without_replacing_persistent_id(tmp_path) -> None:
    coordinate = CoordinateDeclaration.local_metric()
    towers = (
        Tower(
            "tower-107",
            "T107",
            "T107",
            (0.0, 0.0, 0.0),
            coordinate,
            "TEST",
        ),
        Tower(
            "tower-108",
            "T108",
            "T108",
            (100.0, 0.0, 0.0),
            coordinate,
            "TEST",
        ),
    )
    line = Line("line-1", "Line 1", tuple(tower.tower_id for tower in towers))
    span = Span("span-1", line.line_id, towers[0].tower_id, towers[1].tower_id)
    project = replace(Project.new("display-id"), lines=(line,), towers=towers, spans=(span,))
    save_project(project, tmp_path)

    record = SimpleNamespace(
        global_tree_id=7,
        persistent_tree_id="machine-tree-7",
        line_id=1,
        span_id=0,
        span_name="S001",
        longitudinal_s_m=12.0,
        apex_xy_local_m=(12.0, 2.0, 8.0),
        apex_height_m=8.0,
        review=False,
        identity_state="BATCH_EDITOR_FINAL",
    )
    reconciled = SimpleNamespace(
        global_tree_records=(record,),
        diagnostics={"editor_schema": "treeline-stage-b-batch-editor-v1"},
    )
    result = SimpleNamespace(
        span_statuses={"span-1": BatchSpanStatus.FINALIZED},
        global_reconciled=reconciled,
        final_clearance=None,
        risk_policy_snapshot={
            "risk_policy_version": "risk-policy-v1",
            "warning_clearance_threshold_m": 0.5,
        },
        project_id=project.project_id,
        analysis_run_id="analysis-1",
        scan_id="scan-not-needed-for-display-test",
        telemetry=SimpleNamespace(source_point_count=0),
        batch_diagnostics={},
        status="COMPLETE",
    )

    output = export_batch_prediction(
        result,
        tmp_path / "exports" / "Prediction_Input.xlsx",
        final_reconciled=reconciled,
        identity_by_tree={7: {"span_id": "S001"}},
    )
    workbook = load_workbook(output, read_only=True)
    sheet = workbook["Trees"]
    headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    values = [cell.value for cell in next(sheet.iter_rows(min_row=2, max_row=2))]
    row = dict(zip(headers, values, strict=True))
    workbook.close()

    assert row["persistent_tree_id"] == "machine-tree-7"
    assert row["display_tree_id"] == "T-1-001"
    assert row["start_tower_id"] == "tower-107"
    assert row["end_tower_id"] == "tower-108"
    assert headers[0] == "persistent_tree_id"
    assert headers[33] == "display_tree_id"
    assert headers[34:37] == ["dbh_cm", "dbh_method", "dbh_quality"]

    mismatched_data = vars(result).copy()
    mismatched_data["project_id"] = "unrelated-project"
    mismatched_result = SimpleNamespace(**mismatched_data)
    mismatch_output = tmp_path / "exports" / "mismatched.xlsx"
    export_batch_prediction(
        mismatched_result,
        mismatch_output,
        final_reconciled=reconciled,
    )
    mismatch_workbook = load_workbook(mismatch_output, read_only=True)
    mismatch_sheet = mismatch_workbook["Trees"]
    mismatch_headers = [
        cell.value for cell in next(mismatch_sheet.iter_rows(min_row=1, max_row=1))
    ]
    mismatch_values = [
        cell.value for cell in next(mismatch_sheet.iter_rows(min_row=2, max_row=2))
    ]
    mismatch_row = dict(zip(mismatch_headers, mismatch_values, strict=True))
    mismatch_workbook.close()
    assert mismatch_row["start_tower_id"] is None
    assert mismatch_row["end_tower_id"] is None


def test_stage_b_export_rejects_topology_override_conflict(tmp_path) -> None:
    coordinate = CoordinateDeclaration.local_metric()
    towers = (
        Tower("tower-1", "T001", "T001", (0.0, 0.0, 0.0), coordinate, "TEST"),
        Tower("tower-2", "T002", "T002", (100.0, 0.0, 0.0), coordinate, "TEST"),
    )
    line = Line("line-1", "Line 1", tuple(tower.tower_id for tower in towers))
    span = Span("span-1", line.line_id, towers[0].tower_id, towers[1].tower_id)
    project = replace(Project.new("topology-conflict"), lines=(line,), towers=towers, spans=(span,))
    save_project(project, tmp_path)

    record = SimpleNamespace(
        global_tree_id=0,
        persistent_tree_id="machine-tree-0",
        line_id=1,
        span_id=0,
        span_name="S001",
        longitudinal_s_m=12.0,
        apex_xy_local_m=(12.0, 2.0, 8.0),
        apex_height_m=8.0,
        review=False,
        identity_state="BATCH_EDITOR_FINAL",
    )
    reconciled = SimpleNamespace(
        global_tree_records=(record,),
        diagnostics={"editor_schema": "treeline-stage-b-batch-editor-v1"},
    )
    result = SimpleNamespace(
        span_statuses={"span-1": BatchSpanStatus.FINALIZED},
        global_reconciled=reconciled,
        final_clearance=None,
        risk_policy_snapshot={
            "risk_policy_version": "risk-policy-v1",
            "warning_clearance_threshold_m": 0.5,
        },
        project_id=project.project_id,
        analysis_run_id="analysis-1",
        scan_id="scan-not-needed-for-display-test",
        telemetry=SimpleNamespace(source_point_count=0),
        batch_diagnostics={},
        status="COMPLETE",
    )

    with pytest.raises(ValueError, match="EXPORT_IDENTITY_CONFLICTS_WITH_PROJECT_TOPOLOGY"):
        export_batch_prediction(
            result,
            tmp_path / "exports" / "Prediction_Input.xlsx",
            final_reconciled=reconciled,
            identity_by_tree={
                0: {
                    "start_tower_id": "tower-2",
                    "end_tower_id": "tower-1",
                }
            },
        )

    with pytest.raises(ValueError, match="EXPORT_IDENTITY_CONFLICTS_WITH_PROJECT_TOPOLOGY"):
        export_batch_prediction(
            result,
            tmp_path / "exports" / "wrong-span-display.xlsx",
            final_reconciled=reconciled,
            identity_by_tree={0: {"span_id": "S002"}},
        )


def test_stage_b_export_binds_primary_multispan_record_by_topology_order(tmp_path) -> None:
    coordinate = CoordinateDeclaration.local_metric()
    towers = tuple(
        Tower(
            f"tower-{index}",
            f"T00{index}",
            f"T00{index}",
            (float((index - 1) * 100), 0.0, 0.0),
            coordinate,
            "TEST",
        )
        for index in (1, 2, 3)
    )
    line = Line("line-1", "Line 1", tuple(tower.tower_id for tower in towers))
    spans = (
        Span("span-2", line.line_id, towers[1].tower_id, towers[2].tower_id),
        Span("span-1", line.line_id, towers[0].tower_id, towers[1].tower_id),
    )
    project = replace(Project.new("multispan-topology"), lines=(line,), towers=towers, spans=spans)
    save_project(project, tmp_path)
    record = SimpleNamespace(
        global_tree_id=4,
        persistent_tree_id="machine-tree-4",
        line_id=1,
        span_id=1,
        span_name="S002",
        longitudinal_s_m=12.0,
        apex_xy_local_m=(112.0, 2.0, 8.0),
        apex_height_m=8.0,
        review=False,
        identity_state="BATCH_EDITOR_FINAL",
    )
    reconciled = SimpleNamespace(
        global_tree_records=(record,),
        diagnostics={"editor_schema": "treeline-stage-b-batch-editor-v1"},
    )
    result = SimpleNamespace(
        span_statuses={"span-2": BatchSpanStatus.FINALIZED},
        global_reconciled=reconciled,
        final_clearance=None,
        risk_policy_snapshot={
            "risk_policy_version": "risk-policy-v1",
            "warning_clearance_threshold_m": 0.5,
        },
        project_id=project.project_id,
        analysis_run_id="analysis-4",
        scan_id="scan-not-needed-for-display-test",
        telemetry=SimpleNamespace(source_point_count=0),
        batch_diagnostics={},
        status="COMPLETE",
    )

    output = export_batch_prediction(
        result,
        tmp_path / "exports" / "Prediction_Input.xlsx",
        final_reconciled=reconciled,
    )
    workbook = load_workbook(output, read_only=True)
    sheet = workbook["Trees"]
    headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    values = [cell.value for cell in next(sheet.iter_rows(min_row=2, max_row=2))]
    workbook.close()
    row = dict(zip(headers, values, strict=True))

    assert row["span_id"] == "S002"
    assert row["start_tower_id"] == "tower-2"
    assert row["end_tower_id"] == "tower-3"
