from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from algorithms.folder_pipeline.export_excel import write_prediction_input_excel
from core.stage_a import CoordinateDeclaration, Line, Project, Span, Tower, save_project
from core.stage_b import BatchSpanStatus
from desktop.workflow import DesktopWorkflow


def _project(tmp_path: Path) -> Project:
    coordinate = CoordinateDeclaration.local_metric()
    towers = (
        Tower("tower-1", "T001", "T001", (0.0, 0.0, 0.0), coordinate, "TEST"),
        Tower("tower-2", "T002", "T002", (100.0, 0.0, 0.0), coordinate, "TEST"),
    )
    line = Line("line-1", "Line 1", tuple(tower.tower_id for tower in towers))
    span = Span("span-1", line.line_id, towers[0].tower_id, towers[1].tower_id)
    project = replace(Project.new("export-cache"), lines=(line,), towers=towers, spans=(span,))
    save_project(project, tmp_path)
    return project


def _workflow(tmp_path: Path, project: Project) -> DesktopWorkflow:
    workflow = DesktopWorkflow.__new__(DesktopWorkflow)
    workflow._root = tmp_path
    workflow._final_tree_records = {
        7: {
            "global_tree_id": 7,
            "persistent_tree_id": "tree-7",
            "span_id": 0,
            "span_name": "S001",
        }
    }
    workflow._result = None
    return workflow


def _result(project: Project, *, run_id: str = "analysis-7") -> SimpleNamespace:
    record = SimpleNamespace(
        global_tree_id=7,
        persistent_tree_id="tree-7",
        span_id=0,
        span_name="S001",
    )
    reconciled = SimpleNamespace(
        global_tree_records=(record,),
        diagnostics={
            "editor_schema": "treeline-stage-b-batch-editor-v1",
            "editor_operation_count": 0,
        },
    )
    return SimpleNamespace(
        project_id=project.project_id,
        analysis_run_id=run_id,
        span_statuses={"span-1": BatchSpanStatus.FINALIZED},
        global_reconciled=reconciled,
    )


def _write_cache(
    path: Path,
    project: Project,
    *,
    run_id: str = "analysis-7",
    row_start: str | None = "tower-1",
    row_end: str | None = "tower-2",
    rows: bool = True,
) -> None:
    write_prediction_input_excel(
        path,
        [
            {
                "persistent_tree_id": "tree-7",
                "project_id": project.project_id,
                "science_run_id": run_id,
                "analysis_tree_id": "analysis-tree-000007",
                "start_tower_id": row_start,
                "end_tower_id": row_end,
            }
        ]
        if rows
        else [],
        {
            "project_id": project.project_id,
            "science_run_id": run_id,
            "science_scope": "FINAL_GLOBAL_RECONCILED_POST_EDITOR",
        },
    )


def test_export_cache_rejects_wrong_run_and_wrong_tower(tmp_path: Path) -> None:
    project = _project(tmp_path)
    workflow = _workflow(tmp_path, project)
    result = _result(project)
    path = tmp_path / "exports" / "Prediction_Input.xlsx"

    _write_cache(path, project)
    assert workflow._prediction_export_cache_matches_final(path, result)

    _write_cache(path, project, run_id="analysis-other")
    assert not workflow._prediction_export_cache_matches_final(path, result)

    _write_cache(path, project, row_start="tower-2")
    assert not workflow._prediction_export_cache_matches_final(path, result)


def test_export_cache_rejects_empty_tree_sheet_and_corrupt_workbook(tmp_path: Path) -> None:
    project = _project(tmp_path)
    workflow = _workflow(tmp_path, project)
    result = _result(project)
    path = tmp_path / "exports" / "Prediction_Input.xlsx"

    _write_cache(path, project, rows=False)
    assert not workflow._prediction_export_cache_matches_final(path, result)

    path.write_bytes(b"not an xlsx archive")
    assert not workflow._prediction_export_cache_matches_final(path, result)


def test_export_prediction_requires_finalized_status_without_implicit_finalize(
    tmp_path: Path,
) -> None:
    workflow = DesktopWorkflow.__new__(DesktopWorkflow)
    workflow._root = tmp_path
    workflow._finalize_thread = None
    workflow._result = SimpleNamespace(
        span_statuses={"span-1": BatchSpanStatus.REVIEW_REQUIRED}
    )
    calls: list[str] = []
    editor = SimpleNamespace(
        result=workflow._result,
        artifact_root=tmp_path,
        session=SimpleNamespace(operations=[]),
        mark_reviewed=lambda: calls.append("mark_reviewed"),
        finalize=lambda: calls.append("finalize"),
    )
    workflow._ensure_editor = lambda: editor

    with pytest.raises(ValueError, match="PREDICTION_EXPORT_REQUIRES_FINALIZED_BATCH"):
        workflow.export_prediction()
    assert calls == []


def test_newer_cached_workbook_cannot_hide_stale_editor_revision(tmp_path: Path) -> None:
    workflow = DesktopWorkflow.__new__(DesktopWorkflow)
    workflow._root = tmp_path
    workflow._finalize_thread = None
    result = SimpleNamespace(
        span_statuses={"span-1": BatchSpanStatus.FINALIZED},
        global_reconciled=SimpleNamespace(
            diagnostics={
                "editor_schema": "treeline-stage-b-batch-editor-v1",
                "editor_operation_count": 0,
            }
        ),
    )
    workflow._result = result
    editor = SimpleNamespace(
        result=result,
        artifact_root=tmp_path,
        session=SimpleNamespace(operations=[{"type": "DELETE", "tree_ids": [7]}]),
    )
    workflow._ensure_editor = lambda: editor
    (tmp_path / "exports").mkdir(parents=True)
    (tmp_path / "exports" / "Prediction_Input.xlsx").write_bytes(b"cached")

    with pytest.raises(ValueError, match="人工修改已更新"):
        workflow.export_prediction()
