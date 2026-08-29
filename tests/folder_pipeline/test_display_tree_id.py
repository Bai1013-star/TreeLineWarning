from __future__ import annotations

import json
from pathlib import Path

from openpyxl import load_workbook

from algorithms.folder_pipeline.export_excel import write_prediction_input_excel
from algorithms.folder_pipeline.identity import assign_persistent_tree_ids
from algorithms.folder_pipeline.workspace import discover_workspace
from tests.folder_pipeline.conftest import make_workspace


def _workspace_with_selection(
    tmp_path: Path,
    *,
    span_id: str,
    start_tower_id: str,
    end_tower_id: str,
):
    workspace = make_workspace(tmp_path)
    resolved = discover_workspace(workspace)
    resolved.ensure_dirs()
    resolved.span_selection_path.write_text(
        json.dumps(
            {
                "project_id": "project-1",
                "group_id": "G001",
                "span_id": span_id,
                "start_tower_id": start_tower_id,
                "end_tower_id": end_tower_id,
            }
        ),
        encoding="utf-8",
    )
    return resolved


def test_folder_display_ids_start_near_smaller_tower(tmp_path: Path) -> None:
    workspace = _workspace_with_selection(
        tmp_path,
        span_id="S001",
        start_tower_id="T107",
        end_tower_id="T108",
    )
    rows = [
        {"tree_id": 5, "longitudinal_s_m": 75.0},
        {"tree_id": 6, "longitudinal_s_m": 10.0},
        {"tree_id": 7, "longitudinal_s_m": 40.0},
    ]

    assign_persistent_tree_ids(rows, workspace)

    assert rows[1]["display_tree_id"] == "T-1-001"
    assert rows[2]["display_tree_id"] == "T-1-002"
    assert rows[0]["display_tree_id"] == "T-1-003"
    assert rows[1]["numbering_origin_tower_id"] == "T107"


def test_folder_display_ids_reverse_when_smaller_tower_is_span_end(tmp_path: Path) -> None:
    workspace = _workspace_with_selection(
        tmp_path,
        span_id="S002",
        start_tower_id="T108",
        end_tower_id="T107",
    )
    rows = [
        {"tree_id": 1, "longitudinal_s_m": 10.0},
        {"tree_id": 2, "longitudinal_s_m": 50.0},
        {"tree_id": 3, "longitudinal_s_m": 90.0},
    ]

    assign_persistent_tree_ids(rows, workspace)

    assert rows[2]["display_tree_id"] == "T-2-001"
    assert rows[1]["display_tree_id"] == "T-2-002"
    assert rows[0]["display_tree_id"] == "T-2-003"
    assert rows[2]["numbering_origin_tower_id"] == "T107"
    # Durable machine IDs keep their historical start->end allocation.
    assert rows[0]["persistent_tree_id"] == "G001-T001"
    assert rows[2]["persistent_tree_id"] == "G001-T003"


def test_excel_keeps_persistent_id_first_and_appends_display_id(tmp_path: Path) -> None:
    workspace = _workspace_with_selection(
        tmp_path,
        span_id="S001",
        start_tower_id="T107",
        end_tower_id="T108",
    )
    rows = [
        {
            "tree_id": 1,
            "longitudinal_s_m": 12.0,
            "height_m": 10.0,
            "crown_width_m": 3.0,
            "diameter_cm": 20.0,
            "current_risk": "SAFE",
        }
    ]
    assign_persistent_tree_ids(rows, workspace)

    output = write_prediction_input_excel(
        tmp_path / "Prediction_Input.xlsx",
        rows,
        {"science_run_id": "run-1"},
    )
    workbook = load_workbook(output, read_only=True)
    sheet = workbook["Trees"]
    headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    first_row = [cell.value for cell in next(sheet.iter_rows(min_row=2, max_row=2))]

    assert headers[0] == "persistent_tree_id"
    assert first_row[0] == "G001-T001"
    assert headers[-1] == "display_tree_id"
    assert first_row[-1] == "T-1-001"
