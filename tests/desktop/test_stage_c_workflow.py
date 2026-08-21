from __future__ import annotations

import time
from pathlib import Path

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


def test_stage_c_fixture_batch_reconciles_and_resumes_without_rerun(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    span_ids = [item["span_id"] for item in workflow.project_projection()["span_records"]]
    workflow.start_batch(span_ids, mode="FIXTURE")
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
    assert set(editor_data["trees3d"]) == {"xyz", "color", "tree_id_by_index"}
    assert len(editor_data["trees3d"]["xyz"]) == len(editor_data["trees3d"]["tree_id_by_index"])

    restarted = DesktopWorkflow(workflow.project_root)
    assert restarted.project_projection()["resume_state"] == "RESUMABLE"
    assert restarted.batch_projection()["status"] == "COMPLETE"
    assert len(restarted.editor_data()["trees"]) == 2


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
