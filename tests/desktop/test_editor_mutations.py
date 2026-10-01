from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from desktop.editor_tasks import EditorTaskError, editor_tasks
from desktop.server import DesktopApplication
from desktop.workflow import DesktopWorkflow
from openpyxl import load_workbook
from tests.desktop.test_stage_c_workflow import _workflow


def ready(tmp_path):
    workflow = _workflow(tmp_path)
    workflow.start_batch(
        [s["span_id"] for s in workflow.project_projection()["span_records"]], mode="FIXTURE"
    )
    workflow._thread.join(10)
    editor = workflow._ensure_editor()
    editor.config = replace(editor.config, tree_min_final_members=1)
    matrix = np.eye(4)
    matrix[0, 0] = matrix[1, 1] = 0.01
    point = editor.original.xyz_local_m[0, :2] * 0.01
    x, y = point
    polygon = [
        [x - 0.005, y - 0.005],
        [x + 0.005, y - 0.005],
        [x + 0.005, y + 0.005],
        [x - 0.005, y + 0.005],
    ]
    return workflow, matrix.tolist(), polygon


def membership(workflow):
    return tuple(a.copy() for a in workflow._ensure_editor().session.build_final()[:3])


def assert_membership(actual, expected):
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, e)


def disk_membership(editor):
    with np.load(editor.artifact_root / "tree_segmentation_final.npz", allow_pickle=False) as data:
        return data["original_index"], data["tree_id"], data["xyz_local_m"]


@pytest.mark.parametrize("action", ["DELETE", "REMOVE_POINTS"])
def test_previewed_membership_is_persisted_before_final_publication(tmp_path: Path, action):
    workflow, matrix, polygon = ready(tmp_path)
    workflow.finalize()
    editor = workflow._ensure_editor()
    original = disk_membership(editor)

    if action == "DELETE":
        workflow.select_editor_trees([0])
        workflow.delete_selected()
    else:
        preview = workflow.preview_trim_lasso(-1, matrix, polygon)
        workflow.commit_trim(preview["preview_id"])

    # A UI preview may fill the in-memory cache while the disk still holds the
    # previous generation. The next finalize must publish the previewed arrays.
    previewed = editor.final_reconciled()
    expected = (previewed.final_original_index, previewed.final_tree_id,
                previewed.final_xyz_local_m)
    assert len(expected[0]) < len(original[0])
    assert_membership(disk_membership(editor), original)
    workflow.finalize()
    assert_membership(disk_membership(editor), expected)
    assert workflow.current_report()["status"] == "FINALIZED"

    reopened = DesktopWorkflow(workflow.project_root)
    assert reopened.current_report()["status"] == "FINALIZED"
    assert_membership(disk_membership(reopened._ensure_editor()), expected)

    workflow.undo_editor()
    editor.final_reconciled()  # Preview the restored membership before publish.
    workflow.finalize()
    assert_membership(disk_membership(editor), original)
    assert DesktopWorkflow(workflow.project_root).current_report()["status"] == "FINALIZED"


def test_persisted_membership_reuses_file_until_missing(tmp_path: Path, monkeypatch):
    workflow, _matrix, _polygon = ready(tmp_path)
    workflow.select_editor_trees([0])
    workflow.delete_selected()
    workflow.finalize()
    editor = workflow._ensure_editor()
    expected = disk_membership(editor)
    final_path = editor.artifact_root / "tree_segmentation_final.npz"
    original_write = editor._write_npz
    monkeypatch.setattr(editor, "_write_npz", lambda *a, **k: pytest.fail("redundant NPZ write"))
    workflow.finalize()
    assert_membership(disk_membership(editor), expected)

    final_path.unlink()
    monkeypatch.setattr(editor, "_write_npz", original_write)
    editor.persist()
    assert_membership(disk_membership(editor), expected)


def test_reopen_replays_equal_count_different_source_membership(tmp_path: Path):
    from algorithms.folder_pipeline.tree_editor import _encode_indices

    workflow, matrix, polygon = ready(tmp_path)
    first = workflow.preview_trim_lasso(-1, matrix, polygon)
    assert first["point_count"] == 1
    workflow.commit_trim(first["preview_id"])
    workflow.finalize()
    editor = workflow._ensure_editor()
    final_path = editor.artifact_root / "tree_segmentation_final.npz"
    stale_npz = final_path.read_bytes()
    stale_indices = disk_membership(editor)[0]

    workflow.undo_editor()
    source = editor.original.original_index[editor.original.tree_id == 0]
    survivors = set(stale_indices)
    second_index = next(int(index) for index in source if index in survivors)
    editor.session.operations.append({
        "type": "REMOVE_POINTS", "tree_ids": [0], "point_count": 1,
        "removed_indices_b64": _encode_indices(np.array([second_index], dtype=np.int64)),
    })
    workflow.finalize()
    correct = disk_membership(editor)
    assert len(correct[0]) == len(stale_indices)
    assert not np.array_equal(correct[0], stale_indices)
    # Recreate a legacy failure: the new batch/journal was published but an
    # older, equal-length NPZ remained. Record counts and generation still pass.
    final_path.write_bytes(stale_npz)

    reopened = DesktopWorkflow(workflow.project_root)
    assert reopened.current_report()["status"] == "LOCKED"
    recovered_editor = reopened._ensure_editor()
    assert recovered_editor._final_membership_disk_mismatch is True
    np.testing.assert_array_equal(recovered_editor.final_reconciled().final_original_index,
                                  correct[0])
    assert recovered_editor._persisted_final_membership_key is None
    reopened.finalize()
    assert_membership(disk_membership(recovered_editor), correct)
    assert reopened.current_report()["status"] == "FINALIZED"


def test_journal_identity_distinguishes_equal_size_point_removals():
    from core.stage_b.final_generation import editor_journal_digest

    first = {"type": "REMOVE_POINTS", "tree_ids": [0], "removed_indices_b64": "first"}
    second = {"type": "REMOVE_POINTS", "tree_ids": [0], "removed_indices_b64": "second"}
    assert editor_journal_digest([first]) != editor_journal_digest([second])


def test_exact_lasso_undo_and_reopen_without_raw_access(tmp_path: Path, monkeypatch):
    workflow, matrix, polygon = ready(tmp_path)
    before = membership(workflow)
    raw = tmp_path / "fixture.las"
    raw_before = raw.read_bytes()
    import laspy

    monkeypatch.setattr(laspy, "open", lambda *a, **k: pytest.fail("raw LAS accessed by edit"))
    preview = workflow.preview_trim_lasso(-1, matrix, polygon)
    assert preview["point_count"] == 1
    result = workflow.commit_trim(preview["preview_id"])
    after = membership(workflow)
    np.testing.assert_array_equal(after[0], before[0][1:])
    np.testing.assert_array_equal(after[1], before[1][1:])
    np.testing.assert_array_equal(after[2], before[2][1:])
    assert editor_tasks(workflow).snapshot()["state"] == "COMMITTED"
    assert set(result["timings_seconds"]) == {
        "selection_decode",
        "candidate_lookup",
        "geometric_inclusion",
        "membership_mutation",
        "journal_write",
        "viewer_artifact_rebuild",
        "persist",
    }
    workflow.undo_editor()
    assert_membership(membership(workflow), before)
    preview = workflow.preview_trim_lasso(-1, matrix, polygon)
    workflow.commit_trim(preview["preview_id"])
    reopened = DesktopWorkflow(workflow.project_root)
    assert_membership(membership(reopened), after)
    display = reopened.editor_data(display_budget=1_500)["trees3d"]
    visible_indices = {
        int(index)
        for index, visible in zip(
            display["original_index_by_index"], display["visible_by_index"], strict=True
        )
        if visible
    }
    assert visible_indices == set(map(int, after[0]))
    reopened.undo_editor()
    assert_membership(membership(reopened), before)
    assert raw.read_bytes() == raw_before


def test_failed_journal_commit_preserves_live_membership_and_preview(
    tmp_path: Path,
    monkeypatch,
):
    workflow, matrix, polygon = ready(tmp_path)
    editor = workflow._ensure_editor()
    before = membership(workflow)
    journal = editor.artifact_root / "tree_edits.json"
    journal_before = journal.read_bytes()
    preview = workflow.preview_trim_lasso(-1, matrix, polygon)

    def fail(*args, **kwargs):
        raise OSError("injected disk or viewer failure")

    monkeypatch.setattr(type(editor), "_write_json", fail)
    with pytest.raises(OSError, match="injected"):
        workflow.commit_trim(preview["preview_id"])
    assert_membership(membership(workflow), before)
    assert journal.read_bytes() == journal_before
    assert preview["preview_id"] in editor.session._previews
    assert editor_tasks(workflow).snapshot()["state"] == "FAILED"


def test_cancel_preview_and_atomic_journal_replace_failure(tmp_path: Path, monkeypatch):
    workflow, matrix, polygon = ready(tmp_path)
    before = membership(workflow)
    preview = workflow.preview_trim_lasso(-1, matrix, polygon)
    workflow.cancel_trim(preview["preview_id"])
    assert editor_tasks(workflow).snapshot()["state"] == "CANCELLED"
    assert_membership(membership(workflow), before)

    editor = workflow._ensure_editor()
    journal = editor.artifact_root / "tree_edits.json"
    previous = journal.read_bytes()
    preview = workflow.preview_trim_lasso(-1, matrix, polygon)
    original_replace = Path.replace

    def fail_replace(path, target):
        if Path(target) == journal:
            raise OSError("atomic rename denied")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="atomic rename"):
        workflow.commit_trim(preview["preview_id"])
    assert journal.read_bytes() == previous
    assert not list(journal.parent.glob("tree_edits.json.*.tmp"))
    assert_membership(membership(workflow), before)


def test_display_label_cache_tracks_commit_undo_and_topology(tmp_path: Path, monkeypatch):
    from desktop.tree_display_numbering_extensions import (
        _compute_display_identities,
        _current_display_identities,
    )

    workflow, matrix, polygon = ready(tmp_path)
    editor = workflow._ensure_editor()
    original = type(editor).final_reconciled
    calls = []

    def measured(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(editor), "final_reconciled", measured)
    first = _current_display_identities(workflow, editor)
    for _ in range(5):
        assert _current_display_identities(workflow, editor) == first
    assert len(calls) == 0
    preview = workflow.preview_trim_lasso(-1, matrix, polygon)
    workflow.commit_trim(preview["preview_id"])
    count = len(calls)
    cached = _current_display_identities(workflow, editor)
    assert len(calls) == count  # rebuilt inside the task, not on viewer refresh
    assert cached == _compute_display_identities(workflow, editor)
    assert len(calls) == 0
    workflow.undo_editor()
    count = len(calls)
    assert _current_display_identities(workflow, editor) == first
    assert len(calls) == count
    manifest = workflow.project_root / "stage-a-project.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _current_display_identities(workflow, editor)
    assert len(calls) == count


def test_revision_delta_replays_delete_merge_trim_and_undo_on_stable_basis(tmp_path: Path):
    workflow, matrix, polygon = ready(tmp_path)
    initial = workflow.editor_data(display_budget=1_500)
    base_id = initial["display_base_id"]
    initial_xyz = initial["trees3d"]["xyz"]

    workflow.select_editor_trees([0])
    deleted = workflow.delete_selected()["display_delta"]
    assert deleted["available"] is True
    assert deleted["base_id"] == base_id
    assert deleted["from_revision"] == 0 and deleted["revision"] == 1
    assert "0" not in deleted["tree_id_by_source"]

    restored = workflow.undo_editor()["display_delta"]
    assert restored["from_revision"] == 1 and restored["revision"] == 0
    assert restored["tree_id_by_source"]["0"] == 0

    preview = workflow.preview_trim_lasso(-1, matrix, polygon)
    trimmed = workflow.commit_trim(preview["preview_id"])["display_delta"]
    assert trimmed["revision"] == 1
    assert trimmed["removed_original_index"]
    assert workflow.editor_data(display_budget=1_500)["trees3d"]["xyz"] == initial_xyz

    workflow.undo_editor()
    workflow.select_editor_trees([0, 1])
    merged = workflow.merge_selected()["display_delta"]
    merged_id = merged["trees"][0]["tree_id"]
    assert set(merged["tree_id_by_source"].values()) == {merged_id}


def test_lightweight_display_numbering_matches_full_reconciliation(tmp_path: Path):
    from core.stage_b.tree_display_id import assign_tree_display_ids
    from desktop.tree_display_numbering_extensions import _current_display_identities

    workflow, _matrix, _polygon = ready(tmp_path)
    editor = workflow._ensure_editor()
    workflow.select_editor_trees([0, 1])
    workflow.merge_selected()

    lightweight = _current_display_identities(workflow, editor)
    full = assign_tree_display_ids(
        workflow._stage_project(), editor.final_reconciled().global_tree_records
    )
    assert lightweight == full


def test_delete_and_undo_reset_done_state_and_drive_final_projection(tmp_path: Path):
    workflow, _matrix, _polygon = ready(tmp_path)
    workflow.finalize()
    initial = workflow.final_projection()
    editor = workflow._ensure_editor()
    tree_id = int(editor.snapshot()["trees"][0]["tree_id"])
    persistent_id = initial["trees"][0]["persistent_tree_id"]

    workflow.select_editor_trees([tree_id])
    deleted = workflow.delete_selected()

    assert deleted["editor_revision"] == 1
    assert workflow.editor_state()["step"] == "TREE_EDITING"
    assert workflow._workflow["finalize_state"]["status"] == "IDLE"
    assert persistent_id not in {
        row["persistent_tree_id"] for row in workflow.final_projection()["trees"]
    }
    assert workflow.batch_projection()["tree_count_source"] == "EDITOR_FINAL_PROJECTION"
    assert workflow.current_report()["status"] == "LOCKED"
    with pytest.raises(ValueError, match="完成复核"):
        workflow.export_prediction()

    workflow.finalize()
    assert persistent_id not in {
        row["persistent_tree_id"] for row in workflow.current_report()["trees"]
    }

    workflow.undo_editor()
    assert workflow.editor_state()["step"] == "TREE_EDITING"
    assert persistent_id in {
        row["persistent_tree_id"] for row in workflow.final_projection()["trees"]
    }
    workflow.finalize()
    assert persistent_id in {
        row["persistent_tree_id"] for row in workflow.current_report()["trees"]
    }


def test_same_length_new_delete_stays_stale_after_restart_until_refinalized(tmp_path: Path):
    from desktop.review_extensions import _result_matches_editor_revision

    workflow, _matrix, _polygon = ready(tmp_path)
    workflow.select_editor_trees([0])
    workflow.delete_selected()
    workflow.finalize()
    assert _result_matches_editor_revision(workflow._ensure_editor(), workflow._result)

    workflow.undo_editor()
    workflow.select_editor_trees([1])
    workflow.delete_selected()
    assert not _result_matches_editor_revision(workflow._ensure_editor(), workflow._result)
    projection = workflow.final_projection()
    assert len(projection["trees"]) == 1
    assert projection["trees"][0]["minimum_clearance_m"] is None
    assert projection["metadata"]["revision_current"] is False
    assert workflow.batch_projection()["risk_unavailable_tree_count"] == 1
    assert workflow.current_report()["status"] == "LOCKED"

    restarted = DesktopWorkflow(workflow.project_root)
    assert restarted.editor_state()["step"] == "TREE_EDITING"
    assert len(restarted.final_projection()["trees"]) == 1
    assert restarted.final_projection()["trees"][0]["minimum_clearance_m"] is None
    assert restarted.batch_projection()["final_tree_count"] == 1
    assert restarted.current_report()["status"] == "LOCKED"
    restarted.finalize()
    assert {row["persistent_tree_id"] for row in restarted.current_report()["trees"]} == {
        row["persistent_tree_id"] for row in restarted.final_projection()["trees"]
    }


def test_legacy_published_edit_is_invalidated_after_same_length_change(tmp_path: Path):
    from desktop.review_extensions import _result_matches_editor_revision

    workflow, _matrix, _polygon = ready(tmp_path)
    workflow.select_editor_trees([0])
    workflow.delete_selected()
    workflow.finalize()
    batch_path = workflow._ensure_editor().artifact_root / "batch.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch["global_reconciliation"]["diagnostics"].pop("editor_journal_digest", None)
    batch["batch_diagnostics"].pop("editor_journal_digest", None)
    batch_path.write_text(json.dumps(batch), encoding="utf-8")

    legacy = DesktopWorkflow(workflow.project_root)
    assert _result_matches_editor_revision(legacy._ensure_editor(), legacy._result)
    legacy.undo_editor()
    legacy.select_editor_trees([1])
    legacy.delete_selected()

    restarted = DesktopWorkflow(workflow.project_root)
    editor = restarted._ensure_editor()
    assert not _result_matches_editor_revision(editor, restarted._result)
    assert restarted.editor_state()["step"] == "TREE_EDITING"
    assert restarted.final_projection()["metadata"]["revision_current"] is False
    assert restarted.current_report()["status"] == "LOCKED"


def test_background_finalize_keeps_generated_excel_cache_current(tmp_path: Path):
    workflow, _matrix, _polygon = ready(tmp_path)
    legacy = workflow.project_root / "exports" / "Prediction_Input.xlsx"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"prior user workbook")
    workflow.start_finalize()
    thread = workflow._finalize_thread
    assert thread is not None
    thread.join(10)
    assert workflow.editor_state()["step"] == "DONE"

    editor = workflow._ensure_editor()
    workbook = workflow.project_root / "artifacts" / "Prediction_Input.xlsx"
    batch = editor.artifact_root / "batch.json"
    assert workbook.is_file() and batch.is_file()
    assert not legacy.exists()
    archived = list(workbook.parent.glob("Prediction_Input.legacy-*.xlsx"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == b"prior user workbook"
    assert workbook.stat().st_mtime_ns >= batch.stat().st_mtime_ns
    modified = workbook.stat().st_mtime_ns

    assert workflow.export_prediction() == workbook
    assert workbook.stat().st_mtime_ns == modified


def test_missing_dbh_keeps_finalized_result_and_exports_blank_with_reason(tmp_path: Path):
    workflow, _matrix, _polygon = ready(tmp_path)
    workflow.start_finalize()
    workflow._finalize_thread.join(10)
    assert workflow.editor_state()["step"] == "DONE"
    assert workflow.current_report()["status"] == "FINALIZED"
    delivery = workflow.export_delivery()
    workbook = load_workbook(delivery)
    try:
        diameter = workbook.active["I2"]
        assert diameter.value is None
        assert diameter.comment is not None
        assert "胸径留空" in diameter.comment.text
    finally:
        workbook.close()


def request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path.lstrip("/"), data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as response:
        return response.code, json.load(response)


def terminal(base):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        code, result = request(base, "/api/editor/mutation")
        assert code == 200
        if result["state"] != "RUNNING":
            return result
        time.sleep(0.01)
    pytest.fail("editor job did not finish")


def test_http_running_conflicts_failure_recovery_and_undo(tmp_path: Path, monkeypatch):
    workflow, matrix, polygon = ready(tmp_path)
    app = DesktopApplication.__new__(DesktopApplication)
    app.workflow, app.resource_root = workflow, Path(__file__).parents[2]
    app._server = app._thread = None
    app.ui_test_mode = False
    base = app.start()
    entered, release = threading.Event(), threading.Event()
    original = type(workflow._ensure_editor().session)._selected_rows_chunked

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        type(workflow._ensure_editor().session), "_selected_rows_chunked", staticmethod(slow)
    )
    before = membership(workflow)
    try:
        code, job = request(
            base,
            "/api/trim/lasso-preview",
            {
                "async": True,
                "request_id": "lasso-one",
                "tree_id": -1,
                "view_projection": matrix,
                "polygon_ndc": polygon,
            },
        )
        assert code == 202 and job["status"] == "EDIT_RUNNING"
        assert entered.wait(2)
        t = time.monotonic()
        assert request(base, "/api/editor/mutation")[1]["state"] == "RUNNING"
        assert time.monotonic() - t < 1
        for path in (
            "/api/undo",
            "/api/delete",
            "/api/merge",
            "/api/confirm_trees",
            "/api/trim/commit",
            "/api/trim/lasso-preview",
        ):
            code, result = request(base, path, {"async": True})
            assert code == 409 and result["error"]["code"] == "EDIT_RUNNING"
        with pytest.raises(EditorTaskError, match="EDIT_RUNNING"):
            workflow.undo_editor()
        assert_membership(membership(workflow), before)
        release.set()
        preview = terminal(base)["result"]
        code, duplicate = request(
            base,
            "/api/trim/lasso-preview",
            {
                "async": True,
                "request_id": "lasso-one",
            },
        )
        assert code == 409 and duplicate["status"] == "EDIT_DUPLICATE"
        code, _ = request(
            base,
            "/api/trim/commit",
            {
                "async": True,
                "request_id": "commit-one",
                "preview_id": preview["preview_id"],
            },
        )
        assert code == 202
        assert terminal(base)["state"] == "COMMITTED"
        request(base, "/api/undo", {"async": True, "request_id": "undo-one"})
        assert terminal(base)["state"] == "COMMITTED"
        assert_membership(membership(workflow), before)
        # Unexpected exceptions are task failures, not a closed HTTP connection.
        monkeypatch.setattr(workflow, "preview_trim_lasso", lambda *a: 1 / 0)
        request(
            base,
            "/api/trim/lasso-preview",
            {
                "async": True,
                "request_id": "fail-one",
                "tree_id": -1,
            },
        )
        failed = terminal(base)
        assert failed["status"] == "EDIT_FAILED"
        assert "ZeroDivisionError" in failed["error"]["details"]
        request(base, "/api/undo", {"async": True, "request_id": "recovered"})
        assert terminal(base)["state"] == "COMMITTED"
        with monkeypatch.context() as patch:

            def log_denied(*args, **kwargs):
                raise PermissionError("application log denied")

            patch.setattr(workflow, "_log", log_denied)
            assert request(base, "/api/editor/mutation")[0] == 200
            code, error = request(base, "/api/trim/lasso-preview", {"tree_id": -1})
            assert code == 409 and "ZeroDivisionError" in error["error"]["message"]
    finally:
        release.set()
        app.stop()
