from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from core.stage_b import BatchSpanStatus
from desktop.current_report import (
    build_current_report,
    current_finalized_gate,
    current_wire_curves,
)
from desktop.workflow import DesktopWorkflow

RUN_ID = "analysis-current-report"


def _wire_result() -> SimpleNamespace:
    xyz = np.asarray(
        [
            [0.0, 0.0, 8.0],
            [1.0, 0.0, 8.1],
            [2.0, 0.0, 8.2],
            [3.0, 0.0, 8.3],
            [4.0, 0.0, 8.4],
            [10.0, 2.0, 7.0],
            [11.0, 2.0, 7.1],
            [12.0, 2.0, 7.2],
        ],
        dtype=np.float32,
    )
    arrays = {
        "wire_id": np.asarray([10, 20], dtype=np.int32),
        "wire_segment_offsets": np.asarray([0, 2, 3], dtype=np.int64),
        "segment_wire_id": np.asarray([10, 10, 20], dtype=np.int32),
        "segment_index": np.asarray([0, 1, 0], dtype=np.int32),
        "segment_distance_ready": np.asarray([True, True, True], dtype=bool),
        "sampled_curve_xyz": xyz,
        "sampled_curve_offsets": np.asarray([0, 3, 5, 8], dtype=np.int64),
    }
    return SimpleNamespace(w1_arrays=arrays)


def _final_projection(row_count: int = 16) -> dict:
    rows = []
    for tree_id in range(row_count):
        rows.append(
            {
                "persistent_tree_id": f"tree-{tree_id:02d}",
                "display_tree_id": f"T-{tree_id + 1:03d}",
                "height_m": 12.0 + tree_id,
                "crown_width_m": 3.0,
                "crown_width_major_m": 3.5,
                "crown_width_minor_m": 2.5,
                "crown_area_m2": 7.0,
                "diameter_cm": None,
                "diameter_source": "BREAST_HEIGHT_1P30M_UNAVAILABLE",
                "minimum_clearance_m": 4.543721626531224 + tree_id,
                "danger_point_count": 0,
                "current_risk": "SAFE" if tree_id else "WARNING",
                "quality_state": "VALID",
                "human_edited": tree_id == 0,
            }
        )
    return {
        "trees": rows,
        "display": {
            "trees3d": {
                "xyz": [[0.0, 0.0, 8.0], [1.0, 0.0, 8.1], [2.0, 0.0, 8.2]],
                "tree_id_by_index": [0, 0, 1],
                "color": ["#fff", "#fff", "#0ff"],
            }
        },
        "metadata": {"tree_metrics_source": "CANONICAL_FULLRES_POST_EDITOR"},
    }


def _workflow(tmp_path: Path, *, finalize_state: dict | None = None) -> SimpleNamespace:
    artifact_root = tmp_path / "artifacts" / "project-runs" / "batches" / RUN_ID
    artifact_root.mkdir(parents=True)
    manifest = {
        "analysis_run_id": RUN_ID,
        "status": "COMPLETE",
        "span_statuses": {"S1": "FINALIZED", "S2": "FINALIZED"},
        "global_reconciliation": {
            "tree_count": 16,
            "diagnostics": {
                "editor_schema": "treeline-stage-b-batch-editor-v1",
                "editor_operation_count": 1,
            },
        },
    }
    batch_path = artifact_root / "batch.json"
    batch_path.write_text(json.dumps(manifest), encoding="utf-8")
    edits_path = artifact_root / "tree_edits.json"
    edits_path.write_text(json.dumps({"operations": [{"type": "DELETE"}]}), encoding="utf-8")
    os.utime(batch_path, ns=(2_000_000_000, 2_000_000_000))
    os.utime(edits_path, ns=(1_000_000_000, 1_000_000_000))

    reconciled = SimpleNamespace(
        diagnostics={
            "editor_schema": "treeline-stage-b-batch-editor-v1",
            "editor_operation_count": 1,
        }
    )
    clearance = SimpleNamespace(
        global_minimum_clearance_m=4.543721626531224,
        diagnostics={"clearance_exact_points": 3_750_317},
    )
    result = SimpleNamespace(
        status="COMPLETE",
        analysis_run_id=RUN_ID,
        span_statuses={"S1": BatchSpanStatus.FINALIZED, "S2": BatchSpanStatus.FINALIZED},
        global_reconciled=reconciled,
        final_clearance=clearance,
        span_results={
            "S1": SimpleNamespace(wire_result=_wire_result()),
            "S2": SimpleNamespace(wire_result=None),
        },
    )
    editor = SimpleNamespace(
        result=result,
        artifact_root=artifact_root,
        session=SimpleNamespace(operations=[{"type": "DELETE"}]),
    )
    workflow = SimpleNamespace(
        project_root=tmp_path,
        _root=tmp_path,
        _workflow={
            "last_batch_id": RUN_ID,
            "last_batch_status": "COMPLETE",
            "finalize_state": finalize_state or {},
        },
        _thread=None,
        _finalize_thread=None,
        _finalize_stage="IDLE",
        _current_result=lambda: result,
        _ensure_editor=lambda: editor,
        _latest_batch_manifest=lambda: manifest,
        _latest_batch_path=lambda: batch_path,
        final_projection=lambda: _final_projection(),
    )
    return workflow


def test_current_report_uses_global_rows_and_separate_persisted_w1_curves(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)
    projection_kwargs: dict[str, object] = {}
    workflow.final_projection = (
        lambda **kwargs: projection_kwargs.update(kwargs) or _final_projection()
    )
    report = build_current_report(workflow)

    assert report["status"] == "FINALIZED"
    assert report["current_only"] is True
    assert report["kpis"]["tree_count"] == 16
    assert report["kpis"]["minimum_clearance_m"] == pytest.approx(4.543721626531224)
    assert report["kpis"]["risk_counts"] == {"WARNING": 1, "SAFE": 15}
    assert report["kpis"]["final_member_point_count"] == 3_750_317
    assert len(report["trees"]) == 16
    assert report["trees"][0]["display_tree_id"] == "T-001"
    assert report["trees"][0]["diameter_cm"] is None
    assert report["trees"][0]["diameter_source"] == "BREAST_HEIGHT_1P30M_UNAVAILABLE"

    assert report["display"]["trees3d"]["xyz"]
    wires = report["wires"]
    assert wires["source"] == "PERSISTED_SPAN_WIRE_RESULT_W1"
    assert wires["geometry_kind"] == "FITTED_CURRENT_GEOMETRY"
    assert wires["segment_count"] == 3
    np.testing.assert_allclose(
        wires["segments"][0]["xyz"],
        [[0.0, 0.0, 8.0], [1.0, 0.0, 8.1], [2.0, 0.0, 8.2]],
    )
    np.testing.assert_allclose(
        wires["segments"][1]["xyz"], [[3.0, 0.0, 8.3], [4.0, 0.0, 8.4]]
    )
    np.testing.assert_allclose(
        wires["segments"][2]["xyz"],
        [[10.0, 2.0, 7.0], [11.0, 2.0, 7.1], [12.0, 2.0, 7.2]],
    )
    assert "wire" not in report["display"]
    assert "wires3d" not in report["display"]
    assert projection_kwargs == {"persist_display_evidence": False}


def test_read_only_display_projection_never_persists_danger_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import desktop.review_extensions as review_extensions

    result = SimpleNamespace(
        analysis_run_id="analysis-sidecar-test",
        global_reconciled=SimpleNamespace(diagnostics={}),
        final_clearance=SimpleNamespace(danger_points=(object(),)),
    )
    editor = SimpleNamespace(
        result=result,
        session=SimpleNamespace(
            operations=[],
            display_sample=lambda _budget: (
                np.asarray([0], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
            ),
        ),
        snapshot=lambda: {
            "trees": [{"tree_id": 0, "source_tree_ids": [0]}],
        },
    )
    workflow = SimpleNamespace(
        project_root=Path("."),
        _workflow={"selected_span_ids": []},
        _editor=editor,
        _editor_snapshot_cache=None,
        _editor_display_cache=None,
        _ensure_editor=lambda: editor,
        _stage_project=lambda: SimpleNamespace(towers=[], spans=[]),
        _editor_revision_key=lambda value: DesktopWorkflow._editor_revision_key(value),
        _editor_snapshot=lambda value: DesktopWorkflow._editor_snapshot(workflow, value),
    )
    persisted: list[object] = []
    monkeypatch.setattr(
        review_extensions,
        "persist_final_danger_points",
        lambda *args: persisted.append(args),
    )
    monkeypatch.setattr(
        review_extensions,
        "build_danger_point_display",
        lambda *args, **kwargs: {"xyz": []},
    )
    monkeypatch.setattr(
        review_extensions,
        "build_clearance_evidence_display",
        lambda *args, **kwargs: {"records": []},
    )

    review_editor_data = DesktopWorkflow.editor_data.__wrapped__
    review_editor_data(workflow, persist_display_evidence=False)
    assert persisted == []
    review_editor_data(workflow, persist_display_evidence=True)
    assert len(persisted) == 1


@pytest.mark.parametrize(
    "change",
    [
        "pre_final",
        "dirty",
        "active_batch",
        "active_finalize",
        "failed_finalize",
        "persisted_running",
        "persisted_failed",
    ],
)
def test_current_report_locks_before_a_verified_final_state(
    tmp_path: Path, change: str
) -> None:
    workflow = _workflow(tmp_path)
    calls: list[str] = []
    workflow.final_projection = (
        lambda **_kwargs: calls.append("projection") or _final_projection()
    )
    if change == "pre_final":
        workflow._current_result().span_statuses["S2"] = BatchSpanStatus.REVIEW_REQUIRED
    elif change == "dirty":
        edits_path = Path(workflow._latest_batch_path()).with_name("tree_edits.json")
        os.utime(edits_path, ns=(3_000_000_000, 3_000_000_000))
    elif change == "active_batch":
        class AliveThread:
            def is_alive(self) -> bool:
                return True

        workflow._thread = AliveThread()
        workflow._workflow["last_batch_status"] = "RUNNING"
    elif change == "active_finalize":
        workflow._finalize_stage = "RECALCULATING_CLEARANCE"
    elif change == "failed_finalize":
        workflow._finalize_stage = "FAILED"
    elif change == "persisted_running":
        workflow._workflow["finalize_state"] = {"run_id": RUN_ID, "status": "RUNNING"}
    elif change == "persisted_failed":
        workflow._workflow["finalize_state"] = {"run_id": RUN_ID, "status": "FAILED"}

    report = build_current_report(workflow)

    assert report["status"] == "LOCKED"
    assert calls == []
    assert report["gate"]["reasons"]


def test_persisted_nonfinal_manifest_cannot_be_hidden_by_same_run_cached_final(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)
    batch_path = Path(workflow._latest_batch_path())
    persisted = json.loads(batch_path.read_text(encoding="utf-8"))
    persisted["span_statuses"]["S2"] = "REVIEW_REQUIRED"
    batch_path.write_text(json.dumps(persisted), encoding="utf-8")
    os.utime(batch_path, ns=(2_000_000_000, 2_000_000_000))

    gate = current_finalized_gate(workflow)

    assert gate["status"] == "LOCKED"
    assert any("批次发布中的所有已选档距" in reason for reason in gate["reasons"])


def test_old_persisted_finalize_failure_does_not_lock_a_new_run(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path, finalize_state={"run_id": "analysis-old", "status": "FAILED"})
    gate = current_finalized_gate(workflow)
    assert gate["status"] == "PASS"


def test_corrupt_persisted_batch_cannot_be_replaced_by_cached_result(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    workflow._latest_batch_manifest = lambda: None
    gate = current_finalized_gate(workflow)
    assert gate["status"] == "LOCKED"
    assert any("发布文件" in reason for reason in gate["reasons"])


def test_malformed_edit_journal_fails_closed_even_when_mtime_is_old(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    edits_path = Path(workflow._latest_batch_path()).with_name("tree_edits.json")
    edits_path.write_text("{not-json", encoding="utf-8")
    os.utime(edits_path, ns=(1_000_000_000, 1_000_000_000))
    gate = current_finalized_gate(workflow)
    assert gate["status"] == "LOCKED"
    assert gate["editor_revision_match"] is not True


def test_cached_result_from_another_run_cannot_satisfy_current_gate(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    workflow._current_result().analysis_run_id = "analysis-stale"
    gate = current_finalized_gate(workflow)
    assert gate["status"] == "LOCKED"
    assert any("同一分析运行" in reason for reason in gate["reasons"])


def test_current_report_does_not_mutate_batch_or_edit_artifacts(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    batch_path = Path(workflow._latest_batch_path())
    edits_path = batch_path.with_name("tree_edits.json")
    before = (
        batch_path.read_bytes(),
        batch_path.stat().st_mtime_ns,
        edits_path.read_bytes(),
        edits_path.stat().st_mtime_ns,
    )

    build_current_report(workflow)

    after = (
        batch_path.read_bytes(),
        batch_path.stat().st_mtime_ns,
        edits_path.read_bytes(),
        edits_path.stat().st_mtime_ns,
    )
    assert after == before


def test_current_wire_helper_honours_segment_and_sample_offsets(tmp_path: Path) -> None:
    result = SimpleNamespace(span_results={"S1": SimpleNamespace(wire_result=_wire_result())})
    payload = current_wire_curves(result, max_points=5)
    assert payload["truncated"] is True
    assert payload["point_count"] == 5
    assert payload["segments"][0]["segment_index"] == 0
    assert payload["segments"][1]["segment_index"] == 1
    np.testing.assert_allclose(payload["segments"][0]["xyz"][0], [0.0, 0.0, 8.0])
    np.testing.assert_allclose(payload["segments"][1]["xyz"][0], [3.0, 0.0, 8.3])


def test_batch_projection_uses_global_reconciliation_count() -> None:
    workflow = DesktopWorkflow.__new__(DesktopWorkflow)
    workflow._workflow = {"selected_span_ids": ["S1", "S2"]}
    workflow._batch_error = None
    workflow._stage_project = lambda: SimpleNamespace(
        spans=[
            SimpleNamespace(span_id="S1", line_id="L1", start_tower_id="T1", end_tower_id="T2"),
            SimpleNamespace(span_id="S2", line_id="L1", start_tower_id="T2", end_tower_id="T3"),
            SimpleNamespace(span_id="S3", line_id="L1", start_tower_id="T3", end_tower_id="T4"),
        ]
    )
    workflow._span_name = lambda _project, span: span.span_id
    workflow._latest_batch_manifest = lambda: {
        "analysis_run_id": RUN_ID,
        "status": "COMPLETE",
        "span_statuses": {"S1": "FINALIZED", "S2": "FINALIZED"},
        "span_results": {"S1": {"tree_count": 8}, "S2": {"tree_count": 7}},
        "global_reconciliation": {"tree_count": 16},
    }
    workflow._risk_policy = lambda: SimpleNamespace(to_dict=lambda: {})

    projection = workflow.batch_projection()

    assert projection["final_tree_count"] == 16
    assert projection["global_reconciliation_tree_count"] == 16
    assert [span["summary"].get("tree_count", 0) for span in projection["spans"][:2]] == [8, 7]
    assert projection["spans"][2]["status"] == "WAITING"


def test_retry_replays_original_selected_batch_and_excludes_unselected_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = DesktopWorkflow.__new__(DesktopWorkflow)
    workflow._lock = threading.RLock()
    workflow._thread = None
    workflow._workflow = {"acceptance_mode": "REAL", "selected_span_ids": ["S1", "S2"]}
    workflow._batch_error = None
    manifest = {
        "analysis_run_id": RUN_ID,
        "span_statuses": {"S1": "FAILED", "S2": "AUTO_ANALYZED"},
        "span_results": {
            "S1": {"diagnostics": {"failure": "FAST_GROUND_BACKEND_UNAVAILABLE"}},
            "S2": {"tree_count": 8},
        },
    }
    workflow._latest_batch_manifest = lambda: manifest
    workflow._stage_project = lambda: SimpleNamespace(
        spans=[
            SimpleNamespace(span_id="S1"),
            SimpleNamespace(span_id="S2"),
            SimpleNamespace(span_id="S3"),
        ]
    )
    workflow._save_workflow = lambda: None
    workflow.batch_projection = lambda: {"status": "RUNNING"}
    captured: list[tuple[str, ...]] = []

    class CapturedThread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            captured.append(self.args[0])

    monkeypatch.setattr("desktop.workflow.threading.Thread", CapturedThread)

    workflow.retry_failed()

    assert captured == [("S1", "S2")]
    assert workflow._workflow["selected_span_ids"] == ["S1", "S2"]
