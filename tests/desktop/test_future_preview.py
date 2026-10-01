from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from desktop.future_preview import MAX_DISPLAY_POINTS, build_future_preview
from desktop.server import _DesktopHandler
from desktop.workflow import DesktopWorkflow
from tests.desktop.test_current_report import _final_projection, _workflow


def _preview_workflow(tmp_path: Path) -> SimpleNamespace:
    workflow = _workflow(tmp_path)
    workflow._lock = threading.RLock()
    workflow._stage_project = lambda: SimpleNamespace(
        lines=[],
        spans=[
            SimpleNamespace(span_id="S1", start_tower_id="T1", end_tower_id="T2"),
            SimpleNamespace(span_id="S2", start_tower_id="T2", end_tower_id="T3"),
        ],
    )
    result = workflow._current_result()
    records = []
    points = []
    ids = []
    for tree_id in range(16):
        count = 40_000 if tree_id == 0 else 5
        t = np.linspace(0.0, 1.0, count)
        tree = np.column_stack((tree_id * 10 + np.cos(t * 6), np.sin(t * 6), 3 + t * 8))
        points.append(tree)
        ids.extend([tree_id] * count)
        records.append(
            SimpleNamespace(
                global_tree_id=tree_id,
                persistent_tree_id=f"tree-{tree_id:02d}",
                span_id=tree_id // 8,
                full_resolution_point_count=count,
            )
        )
    reconciled = result.global_reconciled
    reconciled.global_tree_records = tuple(records)
    reconciled.final_xyz_local_m = np.concatenate(points)
    reconciled.final_tree_id = np.asarray(ids, dtype=np.int64)
    reconciled.final_original_index = np.arange(len(ids), dtype=np.int64)
    # This synthetic preview fixture replaces the final tree membership, so
    # the unrelated clearance stub no longer describes the new exact trees.
    result.final_clearance = None
    workflow.projection_calls = []

    def projection(**kwargs):
        workflow.projection_calls.append(kwargs)
        return _final_projection()

    workflow.final_projection = projection
    (tmp_path / "Prediction_Input.xlsx").write_bytes(b"authoritative workbook sentinel")
    (tmp_path / "source.las").write_bytes(b"raw source sentinel")
    return workflow


def _files(root: Path) -> dict:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_all_horizons_are_detached_and_cover_every_final_tree_without_writes(tmp_path):
    workflow = _preview_workflow(tmp_path)
    result = workflow._current_result()
    original = result.global_reconciled
    before_arrays = (
        original.final_xyz_local_m.copy(),
        original.final_tree_id.copy(),
        original.final_original_index.copy(),
    )
    before_files = _files(tmp_path)
    before_current = copy.deepcopy(workflow.final_projection())
    workflow.projection_calls.clear()
    previous_height = 0.0
    for horizon in (1, 3, 5):
        preview = build_future_preview(workflow, horizon_years=horizon)
        assert preview["status"] == "PASS", preview
        assert len(preview["trees"]) == 16
        assert sum(row["display_point_count"] for row in preview["trees"]) <= MAX_DISPLAY_POINTS
        assert preview["forecast_horizon_years"] == horizon
        assert preview["future_clearance_status"] == "NOT_IMPLEMENTED"
        assert preview["science_status"] == "NOT_CURRENT_SURVEY_SCIENCE"
        assert preview["trees"][0]["parent_point_count"] == 40_000
        for row in preview["trees"]:
            assert row["persistent_tree_id"] == f"tree-{row['tree_id']:02d}"
            assert row["predicted_dbh_cm"] is None and row["current_dbh_cm"] is None
            assert row["experimental"] is True
            assert row["forecast_horizon_years"] == horizon
            assert row["prediction_mode"] == "LOCAL_SCENARIO"
            assert row["future_xyz"] != row["current_xyz"]
            parents = original.final_xyz_local_m[original.final_tree_id == row["tree_id"]]
            parent_rows = {tuple(point) for point in parents}
            assert all(tuple(point) in parent_rows for point in row["current_xyz"])
            assert "original_index" not in row
        height = preview["trees"][0]["predicted_height_m"]
        assert height > previous_height
        previous_height = height
        # A caller cannot modify the next response or the authoritative arrays.
        preview["trees"][0]["current_xyz"][0][0] = 1e9
        preview["wires"]["segments"][0]["xyz"][0][0] = 1e9
    assert workflow.projection_calls == [{"persist_display_evidence": False}]
    assert workflow.final_projection() == before_current
    for before, after in zip(
        before_arrays,
        (original.final_xyz_local_m, original.final_tree_id, original.final_original_index),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)
    assert _files(tmp_path) == before_files


@pytest.mark.parametrize(
    "state", ["unfinalized", "batch_active", "finalize_running", "failed", "stale"]
)
def test_preview_gate_blocks_without_geometry_work(tmp_path, state):
    workflow = _preview_workflow(tmp_path)
    if state == "unfinalized":
        path = workflow._latest_batch_path()
        manifest = json.loads(path.read_text())
        manifest["span_statuses"]["S1"] = "AUTO_ANALYZED"
        path.write_text(json.dumps(manifest))
    elif state == "batch_active":
        workflow._workflow["last_batch_status"] = "RUNNING"
    elif state == "finalize_running":
        workflow._workflow["finalize_state"] = {"status": "RUNNING"}
    elif state == "failed":
        workflow._finalize_stage = "FAILED"
    else:
        workflow._current_result().analysis_run_id = "stale-run"
    result = build_future_preview(workflow)
    assert result["status"] == "LOCKED"
    assert result["trees"] == []
    assert workflow.projection_calls == []


def test_gate_change_during_prediction_discards_all_geometry(tmp_path, monkeypatch):
    from core.future_preview import LocalScenarioPredictor

    workflow = _preview_workflow(tmp_path)
    original_predict = LocalScenarioPredictor.predict

    def concurrent_finalize(self, tree, horizon_years):
        prediction = original_predict(self, tree, horizon_years)
        workflow._workflow["finalize_state"] = {"status": "RUNNING"}
        return prediction

    monkeypatch.setattr(LocalScenarioPredictor, "predict", concurrent_finalize)
    preview = build_future_preview(workflow)
    assert preview["status"] == "LOCKED"
    assert preview["trees"] == [] and preview["wires"]["segments"] == []
    assert workflow._local_future_preview_snapshot is None


def test_cached_snapshot_does_not_bypass_same_run_edit_guard(tmp_path):
    workflow = _preview_workflow(tmp_path)
    assert build_future_preview(workflow)["status"] == "PASS"
    workflow._ensure_editor().session.operations.append({"type": "DELETE", "tree_id": 8})
    assert build_future_preview(workflow, horizon_years=3)["status"] == "LOCKED"


def test_preview_does_not_restore_stale_reference_diameter(tmp_path):
    workflow = _preview_workflow(tmp_path)
    def projection(**kwargs):
        rows = _final_projection()
        for row in rows["trees"]:
            row["diameter_cm"] = 23.1
            row["dbh_cm"] = None
            row["dbh_method"] = "UNAVAILABLE"
        return rows
    workflow.final_projection = projection
    preview = build_future_preview(workflow)
    assert preview["status"] == "PASS", preview
    assert all(tree["current_dbh_cm"] is None for tree in preview["trees"])
    assert all(tree["predicted_dbh_cm"] is None for tree in preview["trees"])


def test_all_deleted_final_membership_never_shows_old_preview_points(tmp_path):
    workflow = _preview_workflow(tmp_path)
    result = workflow._current_result()
    reconciled = result.global_reconciled
    reconciled.global_tree_records = ()
    reconciled.final_original_index = np.empty(0, dtype=np.int64)
    reconciled.final_tree_id = np.empty(0, dtype=np.int32)
    reconciled.final_xyz_local_m = np.empty((0, 3), dtype=np.float64)
    workflow.final_projection = lambda **_kwargs: {
        "trees": [], "display": {"trees3d": {"xyz": []}},
        "metadata": {"tree_metrics_source": "CANONICAL_FULLRES_POST_EDITOR"},
    }
    for horizon in (1, 3, 5):
        preview = build_future_preview(workflow, horizon_years=horizon)
        assert preview["trees"] == []
        assert preview["status"] == "LOCKED"


@pytest.mark.parametrize("horizon", [0, 2, 4, 6, -1, True, 1.0, "1"])
def test_invalid_horizon_fails_before_any_project_access(horizon):
    with pytest.raises(ValueError, match="horizon"):
        build_future_preview(None, horizon_years=horizon)


def test_invalid_execution_mode_fails_before_project_access():
    workflow = DesktopWorkflow.__new__(DesktopWorkflow)
    with pytest.raises(ValueError, match="execution_mode"):
        workflow.start_batch([], execution_mode="UNBOUNDED")
    workflow._workflow = {}
    with pytest.raises(ValueError, match="execution_mode"):
        workflow.retry_failed(execution_mode="UNBOUNDED")


@pytest.mark.parametrize("requested, expected", [(None, "SERIAL"), ("parallel", "PARALLEL")])
def test_retry_preserves_or_explicitly_updates_execution_policy(monkeypatch, requested, expected):
    workflow = DesktopWorkflow.__new__(DesktopWorkflow)
    workflow._lock = threading.RLock()
    workflow._thread = None
    workflow._workflow = {"acceptance_mode": "REAL", "execution_mode": "SERIAL"}
    workflow._latest_batch_manifest = lambda: {
        "analysis_run_id": "retry-run",
        "span_statuses": {"S2": "FAILED", "S1": "AUTO_ANALYZED"},
    }
    workflow._stage_project = lambda: SimpleNamespace(
        spans=[SimpleNamespace(span_id=value) for value in ("S1", "S2", "S3")]
    )
    workflow._save_workflow = lambda: None
    workflow.batch_projection = lambda: {}
    captured = []

    class Thread:
        def __init__(self, *, target, args, daemon):
            captured.append(args)

        def start(self):
            pass

    monkeypatch.setattr("desktop.workflow.threading.Thread", Thread)
    workflow.retry_failed(execution_mode=requested)
    assert workflow._workflow["execution_mode"] == expected
    assert captured == [(("S1", "S2"), "REAL", "retry-run")]


@pytest.mark.parametrize("query", ["horizon=2", "horizon=1&horizon=3", "horizon=nan"])
def test_http_rejects_invalid_preview_horizon_without_calling_workflow(query):
    handler = _DesktopHandler.__new__(_DesktopHandler)
    calls = []
    responses = []
    handler.application = SimpleNamespace(
        workflow=SimpleNamespace(future_preview=lambda horizon: calls.append(horizon))
    )
    handler.path = "/api/future-preview?" + query
    handler._send_json = lambda payload, **kwargs: responses.append((payload, kwargs))
    handler._handle_get("/api/future-preview")
    assert calls == []
    assert responses[0][1]["status"] == 400


def test_http_preview_and_lightweight_gate_are_read_only_routes(tmp_path):
    workflow = _preview_workflow(tmp_path)
    workflow.future_preview = lambda horizon: build_future_preview(workflow, horizon_years=horizon)
    handler = _DesktopHandler.__new__(_DesktopHandler)
    handler.application = SimpleNamespace(workflow=workflow)
    responses = []
    handler._send_json = lambda payload, **kwargs: responses.append(payload)
    before = _files(tmp_path)
    handler.path = "/api/future-preview?horizon=3"
    handler._handle_get("/api/future-preview")
    assert responses[-1]["status"] == "PASS"
    assert responses[-1]["forecast_horizon_years"] == 3
    handler.path = "/api/future-preview/gate"
    handler._handle_get("/api/future-preview/gate")
    assert responses[-1]["gate"]["status"] == "PASS"
    assert _files(tmp_path) == before
