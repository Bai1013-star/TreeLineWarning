from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from algorithms.spans.project_adapter import SpanScienceResult
from core.stage_b.batch import BatchTelemetry
from desktop.project_management import create_empty_project, import_pointcloud
from desktop.workflow import DesktopWorkflow, derive_workspace_state


def _workflow(tmp_path: Path) -> DesktopWorkflow:
    project = create_empty_project(tmp_path, "Stage C")
    source = tmp_path / "fixture.las"
    source.write_bytes(b"LASF-stage-c-fixture")
    import_pointcloud(Path(project["root"]), source)
    workflow = DesktopWorkflow(Path(project["root"]))
    workflow.confirm_unit("METRE")
    workflow.save_manual_towers(
        {
            "line_name": "Main Line",
            "line_role": "MAIN",
            "towers": [
                {"display_name": "T001", "x": 0, "y": 0, "z": 0, "height_m": 12},
                {"display_name": "T002", "x": 20, "y": 0, "z": 0, "height_m": 12},
                {"display_name": "T003", "x": 40, "y": 0, "z": 0, "height_m": 12},
            ],
        }
    )
    return workflow


def test_stage_c_manual_topology_is_ordered_logical_and_safe(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    project = workflow.project_projection()

    assert project["lines"] == 1
    assert project["towers"] == 3
    assert project["spans"] == 2
    assert [item["display_name"] for item in project["tower_records"]] == [
        "T001",
        "T002",
        "T003",
    ]
    assert all(
        item["source_partition"]
        == {
            "kind": "LOGICAL",
            "access_mode": "SHARED_SOURCE_LOCAL",
            "full_source_copy": False,
            "sequence_index": item["attributes"]["sequence_index"],
            "owner_kind": "LINE",
            "owner_id": item["attributes"]["topology_owner_id"],
        }
        for item in project["span_records"]
    )


@pytest.mark.parametrize("execution_mode", ["AUTO", "SERIAL"])
def test_stage_c_fixture_batch_reconciles_and_resumes_without_rerun(
    tmp_path: Path, execution_mode: str
) -> None:
    workflow = _workflow(tmp_path)
    span_ids = [item["span_id"] for item in workflow.project_projection()["span_records"]]
    workflow.start_batch(span_ids, mode="FIXTURE", execution_mode=execution_mode)
    for _ in range(100):
        batch = workflow.batch_projection()
        if batch["status"] != "RUNNING":
            break
        time.sleep(0.01)

    assert batch["status"] == "COMPLETE"
    assert batch["acceptance_mode"] == "FIXTURE"
    assert batch["duplicate_final_original_index"] == 0
    editor_data = workflow.editor_data()
    assert isinstance(editor_data["trees"], list)
    assert set(editor_data["trees3d"]) == {
        "xyz",
        "color",
        "tree_id_by_index",
        "source_tree_id_by_index",
        "original_index_by_index",
        "visible_by_index",
        "labels",
    }
    assert len(editor_data["trees3d"]["xyz"]) == len(editor_data["trees3d"]["tree_id_by_index"])
    assert len(editor_data["trees3d"]["xyz"]) == len(
        editor_data["trees3d"]["original_index_by_index"]
    )
    span_editor_data = workflow.editor_data(span_id=span_ids[0])
    assert span_editor_data["review_span_id"] == span_ids[0]
    assert {
        int(value) for value in span_editor_data["trees3d"]["tree_id_by_index"]
    } <= {int(item["tree_id"]) for item in span_editor_data["trees"]}

    restarted = DesktopWorkflow(workflow.project_root)
    assert restarted.project_projection()["resume_state"] == "RESUMABLE"
    assert restarted.batch_projection()["status"] == "COMPLETE"
    assert len(restarted.editor_data()["trees"]) == 2


def test_batch_telemetry_preserves_measured_canonical_stage_profile() -> None:
    telemetry = BatchTelemetry()
    telemetry.record_span_profile(
        SpanScienceResult(
            tree_count=0,
            minimum_clearance_m=None,
            risk_threshold_m=0.5,
            risk_counts={},
            diagnostics={"timings_seconds": {"wire": 1.25, "tree": 2.5, "bad": "n/a"}},
        )
    )

    assert telemetry.to_dict()["STAGE_TIME"] == {"wire": 1.25, "tree": 2.5}


def test_stage_c_restart_replays_persisted_tree_editor_operations(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    span_ids = [item["span_id"] for item in workflow.project_projection()["span_records"]]
    workflow.start_batch(span_ids, mode="FIXTURE")
    for _ in range(100):
        if workflow.batch_projection()["status"] != "RUNNING":
            break
        time.sleep(0.01)

    workflow.select_editor_trees([0, 1])
    workflow.merge_selected()
    restarted = DesktopWorkflow(workflow.project_root)

    restored = restarted.editor_state()
    assert restored["tree_snapshot"]["current_tree_count"] == 1
    assert restored["tree_snapshot"]["operations"] == [
        {"type": "MERGE", "tree_ids": [0, 1], "merged_id": 2}
    ]


def test_stage_c_restart_retains_finalized_editor_revision(tmp_path: Path) -> None:
    from desktop.review_extensions import _result_matches_editor_revision

    workflow = _workflow(tmp_path)
    span_ids = [item["span_id"] for item in workflow.project_projection()["span_records"]]
    workflow.start_batch(span_ids, mode="FIXTURE")
    for _ in range(100):
        if workflow.batch_projection()["status"] != "RUNNING":
            break
        time.sleep(0.01)

    workflow.select_editor_trees([0, 1])
    workflow.merge_selected()
    workflow.finalize()

    restarted = DesktopWorkflow(workflow.project_root)
    result = restarted._current_result()
    editor = restarted._ensure_editor()

    assert result is not None
    assert editor is not None
    assert result.global_reconciled.diagnostics["editor_operation_count"] == 1
    assert _result_matches_editor_revision(editor, result)


def test_stage_b_scheduler_snapshot_binds_latest_production_science() -> None:
    from core.stage_b.batch import BatchAnalysisScheduler

    assert (
        "algorithms.spans.wire.run_span_wire_pipeline"
        in BatchAnalysisScheduler._new_analysis_run.__code__.co_consts
    )
    assert (
        "algorithms.spans.tree_runner.run_span_tree_pipeline"
        in BatchAnalysisScheduler._new_analysis_run.__code__.co_consts
    )
    assert "CROWN_STEM_HYBRID" in BatchAnalysisScheduler._new_analysis_run.__code__.co_consts


def test_workspace_state_drives_each_primary_product_view() -> None:
    base_project = {"is_desktop_home": False, "pointcloud": "NOT_IMPORTED", "spans": 0}
    no_capability = {"available": False, "unit_confirmed": False}
    ready_capability = {"available": True, "unit_confirmed": True}
    not_started = {"status": "NOT_STARTED"}

    assert derive_workspace_state(
        {**base_project, "is_desktop_home": True},
        no_capability,
        not_started,
        tree_review_ready=False,
        finalized=False,
    ) == "NO_PROJECT"
    assert derive_workspace_state(
        base_project,
        no_capability,
        not_started,
        tree_review_ready=False,
        finalized=False,
    ) == "PROJECT_CREATED"
    assert derive_workspace_state(
        {**base_project, "pointcloud": "IMPORTED_PENDING_TOWER_REGISTRATION"},
        {"available": True, "unit_confirmed": False},
        not_started,
        tree_review_ready=False,
        finalized=False,
    ) == "SOURCE_IMPORTED_UNIT_PENDING"
    assert derive_workspace_state(
        {**base_project, "pointcloud": "IMPORTED_PENDING_TOWER_REGISTRATION"},
        ready_capability,
        not_started,
        tree_review_ready=False,
        finalized=False,
    ) == "TOWER_SELECTION"
    assert derive_workspace_state(
        {**base_project, "pointcloud": "IMPORTED", "spans": 2},
        ready_capability,
        not_started,
        tree_review_ready=False,
        finalized=False,
    ) == "TOPOLOGY_READY"
    assert derive_workspace_state(
        {**base_project, "pointcloud": "IMPORTED", "spans": 2},
        ready_capability,
        {"status": "RUNNING"},
        tree_review_ready=False,
        finalized=False,
    ) == "ANALYSIS_RUNNING"
    assert derive_workspace_state(
        {**base_project, "pointcloud": "IMPORTED", "spans": 2},
        ready_capability,
        {"status": "COMPLETE"},
        tree_review_ready=True,
        finalized=False,
    ) == "TREE_REVIEW_READY"
    assert derive_workspace_state(
        {**base_project, "pointcloud": "IMPORTED", "spans": 2},
        ready_capability,
        {"status": "COMPLETE"},
        tree_review_ready=True,
        finalized=True,
    ) == "FINALIZED"


def test_workspace_projection_marks_home_without_creating_science_state(tmp_path: Path) -> None:
    project = create_empty_project(tmp_path, "Desktop Home")
    workflow = DesktopWorkflow(Path(project["root"]), desktop_home=True)

    projection = workflow.workspace_projection()

    assert projection["ui_state"] == "NO_PROJECT"
    assert projection["project"]["is_desktop_home"] is True
    assert projection["tree_review_ready"] is False


def test_failed_span_projection_surfaces_chinese_reason_and_retry_policy(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    span_id = workflow.project_projection()["span_records"][0]["span_id"]
    run_id = "analysis-failed-projection"
    batch_path = (
        workflow.project_root / "artifacts/project-runs/batches" / run_id / "batch.json"
    )
    batch_path.parent.mkdir(parents=True)
    batch_path.write_text(
        json.dumps(
            {
                "analysis_run_id": run_id,
                "status": "FAILED",
                "span_statuses": {span_id: "FAILED"},
                "span_results": {
                    span_id: {
                        "diagnostics": {
                            "failure": "FastResearchError: FAST_WIRE_EVIDENCE_INSUFFICIENT",
                            "failure_stage": "WIRE",
                            "failure_stage_label": "导线识别",
                            "failure_code": "FAST_WIRE_EVIDENCE_INSUFFICIENT",
                            "user_message": "当前档距未获得可确认的导线证据，分析保持未完成。",
                            "technical_message": (
                                "FastResearchError: FAST_WIRE_EVIDENCE_INSUFFICIENT"
                            ),
                            "recoverable": False,
                            "recommended_action": "请返回杆塔选取重新确认。",
                        }
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    workflow._workflow.update(
        {"last_batch_id": run_id, "last_batch_status": "FAILED", "selected_span_ids": [span_id]}
    )
    workflow._save_workflow()

    restarted = DesktopWorkflow(workflow.project_root)
    span = restarted.batch_projection()["spans"][0]
    assert span["stage"] == "导线识别"
    assert span["failure"]["user_message"].startswith("当前档距")
    assert span["retryable"] is False
    try:
        restarted.retry_failed()
    except ValueError as exc:
        assert "返回杆塔选取" in str(exc)
    else:
        raise AssertionError("non-recoverable evidence failure must not retry endlessly")


def test_serial_release_rejects_parallel_before_start(monkeypatch: pytest.MonkeyPatch) -> None:
    import desktop.workflow as workflow_module

    class SerialScheduler:
        def __init__(self, root, project, scan_id, *, risk_policy=None):
            raise AssertionError("validation must not construct the scheduler")

    monkeypatch.setattr(workflow_module, "BatchAnalysisScheduler", SerialScheduler)
    assert DesktopWorkflow._validate_execution_mode("AUTO") == "AUTO"
    assert DesktopWorkflow._validate_execution_mode("SERIAL") == "SERIAL"
    with pytest.raises(ValueError, match="PARALLEL_UNAVAILABLE"):
        DesktopWorkflow._validate_execution_mode("PARALLEL")
    assert DesktopWorkflow._validate_acceleration_mode("auto") == "AUTO"
    assert DesktopWorkflow._validate_acceleration_mode("cpu") == "CPU"
    assert DesktopWorkflow._validate_acceleration_mode("gpu") == "GPU"
    with pytest.raises(ValueError, match="acceleration_mode"):
        DesktopWorkflow._validate_acceleration_mode("CUDA")
