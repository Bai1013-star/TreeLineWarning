from __future__ import annotations

import time
from pathlib import Path

from desktop.project_management import create_empty_project, import_pointcloud
from desktop.workflow import DesktopWorkflow


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
