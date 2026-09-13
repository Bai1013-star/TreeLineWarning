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


def test_background_finalize_keeps_generated_excel_cache_current(tmp_path: Path):
    workflow, _matrix, _polygon = ready(tmp_path)
    workflow.start_finalize()
    thread = workflow._finalize_thread
    assert thread is not None
    thread.join(10)
    assert workflow.editor_state()["step"] == "DONE"

    editor = workflow._ensure_editor()
    workbook = workflow.project_root / "exports" / "Prediction_Input.xlsx"
    batch = editor.artifact_root / "batch.json"
    assert workbook.is_file() and batch.is_file()
    assert workbook.stat().st_mtime_ns >= batch.stat().st_mtime_ns
    modified = workbook.stat().st_mtime_ns

    assert workflow.export_prediction() == workbook
    assert workbook.stat().st_mtime_ns == modified


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
