from __future__ import annotations

import numpy as np

from algorithms.spans.clearance import (
    SpanClearanceResult,
    SpanDangerPoint,
    SpanTreeWireRecord,
)
from algorithms.spans.reconcile import GlobalTreeRecord, TreeReconciliationResult
from core.stage_b.batch import (
    BatchAnalysisResult,
    BatchSpanStatus,
    BatchTelemetry,
)
from core.stage_b.danger_points import (
    load_final_danger_points,
    persist_final_danger_points,
)
from core.stage_b.editor import BatchTreeEditor


def _reconciled() -> TreeReconciliationResult:
    records = (
        GlobalTreeRecord(
            global_tree_id=0,
            line_id=1,
            span_id=0,
            span_name="S001",
            external_name="Tree-0",
            primary_span_tree_id=0,
            apex_xy_local_m=(0.1, 0.0, 3.2),
            apex_height_m=3.2,
            full_resolution_point_count=2,
            core_point_count=2,
            halo_point_count=0,
            core_fraction=1.0,
            review=False,
            merged_span_tree_keys=((0, 0),),
            persistent_tree_id="tree-0",
        ),
        GlobalTreeRecord(
            global_tree_id=1,
            line_id=1,
            span_id=0,
            span_name="S001",
            external_name="Tree-1",
            primary_span_tree_id=1,
            apex_xy_local_m=(2.1, 0.0, 4.2),
            apex_height_m=4.2,
            full_resolution_point_count=2,
            core_point_count=2,
            halo_point_count=0,
            core_fraction=1.0,
            review=False,
            merged_span_tree_keys=((0, 1),),
            persistent_tree_id="tree-1",
        ),
    )
    return TreeReconciliationResult(
        global_tree_records=records,
        final_original_index=np.asarray([10, 11, 20, 21], dtype=np.int64),
        final_xyz_local_m=np.asarray(
            [
                [0.0, 0.0, 2.5],
                [0.1, 0.0, 3.2],
                [2.0, 0.0, 3.5],
                [2.1, 0.0, 4.2],
            ],
            dtype=np.float64,
        ),
        final_tree_id=np.asarray([0, 0, 1, 1], dtype=np.int32),
        review_tree_ids=(),
        span_tree_to_global={(0, 0): 0, (0, 1): 1},
        diagnostics={"duplicate_final_original_index": 0},
    )


def _clearance() -> SpanClearanceResult:
    records = (
        SpanTreeWireRecord(
            global_tree_id=0,
            external_name="Tree-0",
            line_id=1,
            span_id=0,
            span_name="S001",
            nearest_wire_id=0,
            minimum_clearance_m=0.2,
            nearest_tree_original_index=10,
            nearest_tree_xyz_local_m=(0.0, 0.0, 2.5),
            nearest_wire_xyz_local_m=(0.0, 0.0, 2.7),
            tree_point_count=2,
            evaluated_edge_count=4,
            status="VALID",
            persistent_tree_id="tree-0",
            danger_point_count=1,
        ),
        SpanTreeWireRecord(
            global_tree_id=1,
            external_name="Tree-1",
            line_id=1,
            span_id=0,
            span_name="S001",
            nearest_wire_id=0,
            minimum_clearance_m=0.9,
            nearest_tree_original_index=20,
            nearest_tree_xyz_local_m=(2.0, 0.0, 3.5),
            nearest_wire_xyz_local_m=(2.0, 0.0, 4.4),
            tree_point_count=2,
            evaluated_edge_count=4,
            status="VALID",
            persistent_tree_id="tree-1",
            danger_point_count=0,
        ),
    )
    return SpanClearanceResult(
        records=records,
        status="VALID",
        global_minimum_clearance_m=0.2,
        edge_count=2,
        diagnostics={"schema": "fixture"},
        danger_points=(
            SpanDangerPoint(
                global_tree_id=0,
                persistent_tree_id="tree-0",
                original_index=10,
                xyz_local_m=(0.0, 0.0, 2.5),
            ),
        ),
    )


def _result(clearance: SpanClearanceResult) -> BatchAnalysisResult:
    return BatchAnalysisResult(
        analysis_run_id="analysis-clearance-invalidation",
        project_id="project-fixture",
        scan_id="scan-fixture",
        status="COMPLETE",
        span_statuses={"span-a": BatchSpanStatus.USER_REVIEWED},
        # No runtime wire_result is available.  This is the restart/incomplete
        # handoff that must never keep pre-edit clearance after membership edits.
        span_results={},
        telemetry=BatchTelemetry(source_point_count=4, span_count=1),
        risk_policy_snapshot={
            "risk_policy_version": "risk-policy-v1",
            "warning_clearance_threshold_m": 0.5,
        },
        global_reconciled=_reconciled(),
        final_clearance=clearance,
    )


def test_batch_publication_persists_exact_danger_points_before_viewer_open(tmp_path) -> None:
    clearance = _clearance()
    result = _result(clearance)
    editor = BatchTreeEditor(
        tmp_path,
        result,
        tree_ids_by_span={"span-a": (0, 1)},
    )

    editor.persist()

    arrays = load_final_danger_points(tmp_path, result.analysis_run_id)
    assert arrays is not None
    original_index, tree_id, xyz = arrays
    assert original_index.tolist() == [10]
    assert tree_id.tolist() == [0]
    assert xyz.tolist() == [[0.0, 0.0, 2.5]]


def test_tree_edit_invalidates_clearance_when_wire_set_cannot_be_recomputed(tmp_path) -> None:
    clearance = _clearance()
    result = _result(clearance)
    persist_final_danger_points(tmp_path, result.analysis_run_id, clearance)
    assert load_final_danger_points(tmp_path, result.analysis_run_id) is not None

    editor = BatchTreeEditor(
        tmp_path,
        result,
        tree_ids_by_span={"span-a": (0, 1)},
    )
    editor.delete("span-a", [0])
    stages: list[str] = []
    finalized = editor.finalize(progress=stages.append)

    assert finalized.final_clearance is None
    assert stages == ["APPLYING_EDITS", "VALIDATING_CLEARANCE"]
    assert finalized.global_reconciled.diagnostics["clearance_state"] == (
        "INVALIDATED_AFTER_TREE_EDIT"
    )
    assert finalized.global_reconciled.diagnostics["clearance_reason"] == (
        "INCOMPLETE_WIRE_RESULTS"
    )
    assert load_final_danger_points(tmp_path, result.analysis_run_id) is None


def test_no_tree_edit_keeps_existing_clearance_when_runtime_wires_are_unavailable(
    tmp_path,
) -> None:
    clearance = _clearance()
    result = _result(clearance)
    editor = BatchTreeEditor(
        tmp_path,
        result,
        tree_ids_by_span={"span-a": (0, 1)},
    )
    stages: list[str] = []

    finalized = editor.finalize(progress=stages.append)

    assert finalized.final_clearance is clearance
    assert stages == ["APPLYING_EDITS", "VALIDATING_CLEARANCE"]
    assert "clearance_state" not in finalized.global_reconciled.diagnostics
