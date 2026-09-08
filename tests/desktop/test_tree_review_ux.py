from __future__ import annotations

import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from algorithms.folder_pipeline.tree_editor import TreeEditSession
from core.stage_b.editor import BatchTreeEditor
from desktop.workflow import DesktopWorkflow, _load_final_clearance
from tests.folder_pipeline.test_tree_trim import _separated_tree


def _bare_workflow(tmp_path: Path) -> DesktopWorkflow:
    workflow = DesktopWorkflow.__new__(DesktopWorkflow)
    workflow._root = tmp_path
    workflow._lock = threading.RLock()
    workflow._editor_snapshot_cache = None
    workflow._editor_display_cache = None
    workflow._finalize_thread = None
    workflow._finalize_stage = "IDLE"
    workflow._finalize_message = ""
    workflow._finalize_error = None
    workflow._finalize_detail = None
    workflow._final_tree_records = {}
    workflow._workflow = {"last_batch_status": "COMPLETE", "selected_span_ids": []}
    workflow._result = None
    return workflow


def test_editor_display_reuses_stable_basis_across_committed_revisions(tmp_path: Path) -> None:
    workflow = _bare_workflow(tmp_path)

    class Session:
        def __init__(self) -> None:
            self.operations: list[dict] = []
            self.build_calls = 0

        def display_sample(self, _limit):
            self.build_calls += 1
            return (
                np.asarray([10, 11], dtype=np.int64),
                np.asarray([0, 1], dtype=np.int32),
                np.asarray([[0.0, 0.0, 1.0], [2.0, 0.0, 3.0]], dtype=np.float64),
            )

    session = Session()
    editor = SimpleNamespace(
        session=session,
        snapshot=lambda: {
            "trees": [
                {"tree_id": 0, "point_count": 1},
                {"tree_id": 1, "point_count": 1},
            ]
        },
    )
    workflow._ensure_editor = lambda: editor
    workflow._stage_project = lambda: SimpleNamespace(towers=(), spans=())

    first = workflow.editor_data()
    second = workflow.editor_data()
    assert first is second
    assert session.build_calls == 1

    session.operations.append({"type": "DELETE", "tree_ids": [1]})
    workflow.editor_data()
    assert session.build_calls == 1


def test_display_sample_tracks_exact_committed_membership_without_full_final_build() -> None:
    original = _separated_tree()
    session = TreeEditSession(original)
    polygon = np.asarray([[-1.0, -1.0], [0.0, -1.0], [0.0, 1.0], [-1.0, 1.0]])
    preview = session.preview_trim_lasso(1, np.eye(4), polygon)
    session.commit_trim(preview["preview_id"])

    sample_index, sample_tree, sample_xyz = session.display_sample(len(original.original_index))
    final_index, final_tree, final_xyz, _summary = session.build_final()

    assert np.array_equal(sample_index, final_index)
    assert np.array_equal(sample_tree, final_tree)
    assert np.array_equal(sample_xyz, final_xyz)


def test_interactive_edit_persistence_defers_reconstructible_final_npz(tmp_path: Path) -> None:
    editor = BatchTreeEditor.__new__(BatchTreeEditor)
    editor.artifact_root = tmp_path
    editor.original = _separated_tree()
    editor.session = TreeEditSession(editor.original)
    editor.result = SimpleNamespace(analysis_run_id="analysis-edits")
    editor.tree_ids_by_span = {"span-1": (1, 2)}
    (tmp_path / "tree_segmentation_original.npz").write_bytes(b"existing-original")
    editor.session.select([2])
    editor.session.delete_selected()

    editor.persist_edits()

    assert (tmp_path / "tree_edits.json").is_file()
    assert not (tmp_path / "tree_segmentation_final.npz").exists()


def test_finalize_disk_precheck_uses_current_artifact_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = _bare_workflow(tmp_path)
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    (artifact / "tree_segmentation_original.npz").write_bytes(b"x" * 1024)
    editor = SimpleNamespace(artifact_root=artifact)
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))

    with pytest.raises(OSError) as caught:
        workflow._require_finalize_disk_space(editor)
    assert caught.value.errno == 28


def test_restart_hydrates_final_clearance_for_excel_export() -> None:
    clearance = _load_final_clearance(
        {
            "status": "VALID",
            "global_minimum_clearance_m": 0.42,
            "diagnostics": {"edge_count": 7},
            "records": [
                {
                    "global_tree_id": 3,
                    "external_name": "Tree-3",
                    "persistent_tree_id": "tree-3",
                    "line_id": 1,
                    "span_id": 2,
                    "span_name": "S003",
                    "nearest_wire_id": 5,
                    "minimum_clearance_m": 0.42,
                    "nearest_tree_original_index": 99,
                    "nearest_tree_xyz_local_m": [1.0, 2.0, 3.0],
                    "nearest_wire_xyz_local_m": [1.0, 2.0, 3.42],
                    "tree_point_count": 1000,
                    "evaluated_edge_count": 7,
                    "status": "VALID",
                    "danger_point_count": 4,
                }
            ],
        }
    )

    assert clearance is not None
    assert clearance.global_minimum_clearance_m == 0.42
    assert clearance.records[0].minimum_clearance_m == 0.42
    assert clearance.records[0].danger_point_count == 4


def test_step3_projection_uses_final_clearance_instead_of_placeholder(tmp_path: Path) -> None:
    workflow = _bare_workflow(tmp_path)
    record = SimpleNamespace(
        global_tree_id=3,
        persistent_tree_id="tree-3",
        apex_height_m=8.5,
        review=False,
        identity_state="HUMAN_LASSO",
    )
    clearance = SimpleNamespace(
        global_tree_id=3,
        minimum_clearance_m=0.42,
        danger_point_count=4,
        status="VALID",
    )
    result = SimpleNamespace(
        global_reconciled=SimpleNamespace(global_tree_records=(record,)),
        final_clearance=SimpleNamespace(records=(clearance,)),
    )
    editor = SimpleNamespace(
        result=result,
        session=SimpleNamespace(operations=[]),
        snapshot=lambda: {"trees": [{"tree_id": 3, "apex_height_m": 8.0}]},
    )
    workflow._ensure_editor = lambda: editor
    workflow.editor_data = lambda: {"trees3d": {"xyz": []}}

    projection = workflow.final_projection()

    assert projection["trees"][0]["persistent_tree_id"] == "tree-3"
    assert projection["trees"][0]["minimum_clearance_m"] == 0.42
    assert projection["trees"][0]["danger_point_count"] == 4
    assert projection["trees"][0]["human_edited"] is True


def test_step3_projection_preserves_persisted_human_edit_provenance_after_restart(
    tmp_path: Path,
) -> None:
    workflow = _bare_workflow(tmp_path)
    workflow._final_tree_records = {
        12: {
            "global_tree_id": 12,
            "persistent_tree_id": "tree-12",
            "apex_height_m": 9.25,
            "review": False,
            "identity_state": "HUMAN_LASSO",
        }
    }
    stale_record = SimpleNamespace(
        global_tree_id=12,
        persistent_tree_id="tree-12",
        apex_height_m=9.0,
        review=False,
        identity_state="RUN_LOCAL_ONLY",
    )
    result = SimpleNamespace(
        global_reconciled=SimpleNamespace(global_tree_records=(stale_record,)),
        final_clearance=None,
    )
    editor = SimpleNamespace(
        result=result,
        session=SimpleNamespace(operations=[]),
        snapshot=lambda: {"trees": [{"tree_id": 12, "apex_height_m": 9.0}]},
    )
    workflow._ensure_editor = lambda: editor
    workflow.editor_data = lambda: {"trees3d": {"xyz": []}}

    tree = workflow.final_projection()["trees"][0]

    assert tree["height_m"] == 9.25
    assert tree["human_edited"] is True


def test_background_finalize_reports_real_stages_and_finishes_excel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = _bare_workflow(tmp_path)
    stages: list[str] = []
    result = SimpleNamespace(
        status="COMPLETE",
        analysis_run_id="analysis-ux",
        global_reconciled=SimpleNamespace(global_tree_records=()),
    )

    class Editor:
        artifact_root = tmp_path

        @staticmethod
        def mark_reviewed() -> None:
            pass

        @staticmethod
        def finalize(*, progress):
            for stage in ("APPLYING_EDITS", "RECALCULATING_CLEARANCE"):
                stages.append(stage)
                progress(stage)
            return result

    workflow._ensure_editor = lambda: Editor()
    workflow._require_finalize_disk_space = lambda _editor: None
    workflow._save_workflow = lambda: None
    workflow._log = lambda *_args, **_kwargs: None

    def fake_export(_result, output, *, final_reconciled, progress):
        assert final_reconciled is not None
        for stage in ("RECALCULATING_RISK", "GENERATING_EXCEL"):
            stages.append(stage)
            progress(stage)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"xlsx")

    monkeypatch.setattr("core.stage_b.export.export_batch_prediction", fake_export)
    workflow._run_finalize()

    assert stages == [
        "APPLYING_EDITS",
        "RECALCULATING_CLEARANCE",
        "RECALCULATING_RISK",
        "GENERATING_EXCEL",
    ]
    assert workflow._finalize_stage == "DONE"
    assert workflow._finalize_message == "完成"
    assert (tmp_path / "exports" / "Prediction_Input.xlsx").read_bytes() == b"xlsx"
