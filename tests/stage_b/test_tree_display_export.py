from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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
    )
    workbook = load_workbook(output, read_only=True)
    sheet = workbook["Trees"]
    headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    values = [cell.value for cell in next(sheet.iter_rows(min_row=2, max_row=2))]
    row = dict(zip(headers, values, strict=True))

    assert row["persistent_tree_id"] == "machine-tree-7"
    assert row["display_tree_id"] == "T-1-001"
    assert headers[0] == "persistent_tree_id"
    assert headers[-1] == "display_tree_id"
