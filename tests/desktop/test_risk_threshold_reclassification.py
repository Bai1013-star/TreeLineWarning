from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from algorithms.spans.clearance import SpanClearanceResult, SpanTreeWireRecord
from core.stage_a.risk import RiskClass, RiskPolicy
from core.stage_b.batch import BatchAnalysisResult, BatchTelemetry
from core.stage_b.danger_points import build_clearance_evidence_display
from desktop.project_management import create_empty_project, import_pointcloud
from desktop.review_extensions import _apply_risk_tree_colors
from desktop.workflow import DesktopWorkflow

RUN_ID = "analysis-risk-reclassification"


def _record(tree_id: int, clearance_m: float) -> SpanTreeWireRecord:
    return SpanTreeWireRecord(
        global_tree_id=tree_id,
        external_name=f"Tree-{tree_id}",
        persistent_tree_id=f"tree-{tree_id}",
        line_id=1,
        span_id=1,
        span_name="S001",
        nearest_wire_id=0,
        minimum_clearance_m=clearance_m,
        nearest_tree_original_index=tree_id,
        nearest_tree_xyz_local_m=(float(tree_id), 0.0, 1.0),
        nearest_wire_xyz_local_m=(float(tree_id), 0.0, 1.0 + clearance_m),
        tree_point_count=1,
        evaluated_edge_count=1,
        status="VALID",
    )


def _clearance() -> SpanClearanceResult:
    return SpanClearanceResult(
        records=(_record(0, 0.4), _record(1, 2.0)),
        status="VALID",
        global_minimum_clearance_m=0.4,
        edge_count=1,
        diagnostics={
            "schema": "fast-baseline-span-clearance-v1",
            "clearance_exact_points": 2,
            "danger_threshold_m": 0.5,
        },
    )


def _result(clearance: SpanClearanceResult) -> BatchAnalysisResult:
    return BatchAnalysisResult(
        analysis_run_id=RUN_ID,
        project_id="project-risk",
        scan_id="scan-risk",
        status="COMPLETE",
        span_statuses={},
        span_results={},
        telemetry=BatchTelemetry(),
        risk_policy_snapshot={
            "risk_policy_version": "risk-policy-v1",
            "warning_clearance_threshold_m": 0.5,
        },
        final_clearance=clearance,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_danger_requires_confirmed_contact_at_every_warning_threshold() -> None:
    for threshold in (0.5, 3.0):
        policy = RiskPolicy(warning_clearance_threshold_m=threshold)
        assert policy.classify(clearance_m=0.0, contact=True).risk is RiskClass.DANGER
        assert policy.classify(clearance_m=0.0, contact=None).risk is RiskClass.WARNING
        assert policy.classify(clearance_m=threshold, contact=False).risk is RiskClass.WARNING
        assert policy.classify(clearance_m=threshold + 0.001, contact=False).risk is RiskClass.SAFE


def test_threshold_change_reclassifies_saved_clearance_without_science_rewrite(
    tmp_path: Path,
) -> None:
    project = create_empty_project(tmp_path, "Risk reclassification")
    workflow = DesktopWorkflow(Path(project["root"]))
    source = tmp_path / "fixture.las"
    source.write_bytes(b"LASF-risk-reclassification-fixture")
    import_pointcloud(workflow.project_root, source)
    clearance = _clearance()
    result = _result(clearance)
    workflow._result = result
    workflow._workflow["last_batch_id"] = RUN_ID

    batch_root = (
        workflow.project_root / "artifacts" / "project-runs" / "batches" / RUN_ID
    )
    batch_root.mkdir(parents=True)
    batch_path = batch_root / "batch.json"
    batch_path.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    science_paths = []
    for relative in (
        "ground/preparation.json",
        "wire/span_wire_result.json",
        "tree/span_tree_manifest.json",
        "batch-clearance.json",
        "dbh_final.json",
    ):
        path = batch_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{{"authority":"{relative}"}}', encoding="utf-8")
        science_paths.append(path)
    membership_path = batch_root / "tree_segmentation_final.npz"
    np.savez(
        membership_path,
        original_index=np.asarray([0, 1], dtype=np.int64),
        tree_id=np.asarray([0, 1], dtype=np.int32),
        xyz_local_m=np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
    )
    science_paths.append(membership_path)
    science_before = {
        path: (path.stat().st_mtime_ns, _sha256(path))
        for path in science_paths
    }

    response = workflow.set_risk_policy(3.0)

    assert response["warning_clearance_threshold_m"] == 3.0
    assert response["reclassification"]["risk_counts"] == {
        "DANGER": 0,
        "WARNING": 2,
        "SAFE": 0,
    }
    assert response["reclassification"]["science_recomputed"] is False
    assert workflow._result is not result
    assert workflow._result.final_clearance is clearance
    assert workflow._result.risk_policy_snapshot["warning_clearance_threshold_m"] == 3.0
    assert workflow._result.batch_diagnostics["risk_reclassification"]["risk_counts"][
        "WARNING"
    ] == 2
    assert workflow.batch_projection()["risk_counts"] == {
        "DANGER": 0,
        "WARNING": 2,
        "SAFE": 0,
    }

    persisted = json.loads(batch_path.read_text(encoding="utf-8"))
    assert persisted["risk_policy_snapshot"]["warning_clearance_threshold_m"] == 3.0
    assert persisted["batch_diagnostics"]["risk_reclassification"]["risk_counts"] == {
        "DANGER": 0,
        "WARNING": 2,
        "SAFE": 0,
    }
    assert {
        path: (path.stat().st_mtime_ns, _sha256(path))
        for path in science_paths
    } == science_before

    restarted = DesktopWorkflow(workflow.project_root)
    restarted_result = restarted._current_result()
    assert restarted_result is not None
    assert restarted_result.risk_policy_snapshot["warning_clearance_threshold_m"] == 3.0
    assert restarted_result.batch_diagnostics["risk_reclassification"]["risk_counts"][
        "WARNING"
    ] == 2


def test_saved_clearance_projection_changes_safe_to_warning_at_three_metres() -> None:
    clearance = _clearance()
    result = SimpleNamespace(
        global_reconciled=SimpleNamespace(global_tree_records=()),
        final_clearance=clearance,
        risk_policy_snapshot=RiskPolicy(
            warning_clearance_threshold_m=0.5
        ).to_dict(),
    )

    initial = build_clearance_evidence_display(result)
    assert [row["risk"] for row in initial["records"]] == ["WARNING", "SAFE"]

    result.risk_policy_snapshot = RiskPolicy(
        warning_clearance_threshold_m=3.0
    ).to_dict()
    updated = build_clearance_evidence_display(result)
    assert [row["risk"] for row in updated["records"]] == ["WARNING", "WARNING"]
    assert updated["warning_threshold_m"] == 3.0

    display = {
        "trees3d": {
            "tree_id_by_index": [0, 1, 1],
            "color": ["old-0", "old-1", "old-1"],
        }
    }
    _apply_risk_tree_colors(display, updated)
    assert display["trees3d"]["color"] == ["#ffa000", "#ffa000", "#ffa000"]


def test_desktop_threshold_save_forces_open_review_view_to_reload() -> None:
    source = Path("viewer_poc/giro3d/desktop.js").read_text(encoding="utf-8")
    save_risk = source[source.index("async function saveRisk()") : source.index(
        "$('newProject').addEventListener"
    )]

    assert "state.viewerLoaded = false" in save_risk
    assert "await loadWorkspace()" in save_risk
    assert "结果已即时重分类" in save_risk
