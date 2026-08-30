"""Order-independent multi-Span clearance publication tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from algorithms.spans.project_adapter import SpanScienceResult
from algorithms.spans.reconcile import GlobalTreeRecord, TreeReconciliationResult
from algorithms.spans.wire import SpanWireResult
from core.config.fast_baseline import default_fast_baseline_config
from core.stage_a.risk import RiskPolicy
from core.stage_b.batch import (
    BatchAnalysisResult,
    BatchAnalysisScheduler,
    BatchSpanStatus,
    BatchTelemetry,
)
from core.stage_b.editor import BatchTreeEditor

_CONFIG = default_fast_baseline_config()


def _wire(y: float) -> SpanWireResult:
    curve = np.asarray(
        [[0.0, y, 10.0], [1.0, y, 10.0], [2.0, y, 10.0]],
        dtype=np.float32,
    )
    arrays = {
        "wire_id": np.asarray([0], dtype=np.int32),
        "wire_status": np.asarray([0], dtype=np.uint8),
        "wire_method": np.asarray([1], dtype=np.uint8),
        "wire_quality_flags": np.asarray([0], dtype=np.uint16),
        "wire_has_distance_ready_segments": np.asarray([True], dtype=np.bool_),
        "wire_continuous_distance_ready": np.asarray([True], dtype=np.bool_),
        "wire_segment_offsets": np.asarray([0, 1], dtype=np.int64),
        "segment_wire_id": np.asarray([0], dtype=np.int32),
        "segment_index": np.asarray([0], dtype=np.int32),
        "segment_method": np.asarray([1], dtype=np.uint8),
        "segment_quality_flags": np.asarray([0], dtype=np.uint16),
        "segment_distance_ready": np.asarray([True], dtype=np.bool_),
        "segment_parameter_domain": np.asarray([[0.0, 1.0]], dtype=np.float64),
        "sampled_curve_xyz": curve,
        "sampled_curve_parameter": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        "sampled_curve_offsets": np.asarray([0, 3], dtype=np.int64),
    }
    return SpanWireResult(
        strands=(),
        w1_arrays=arrays,
        candidate_mask=np.empty(0, dtype=np.bool_),
        cluster_id_by_rep=np.empty(0, dtype=np.int32),
        diagnostics={},
    )


def _reconciled() -> TreeReconciliationResult:
    record = GlobalTreeRecord(
        global_tree_id=0,
        line_id=0,
        span_id=0,
        span_name="S001",
        external_name="Tree-001",
        primary_span_tree_id=0,
        apex_xy_local_m=(1.0, 0.0, 10.0),
        apex_height_m=10.0,
        full_resolution_point_count=1,
        core_point_count=1,
        halo_point_count=0,
        core_fraction=1.0,
        review=False,
        merged_span_tree_keys=((0, 0),),
    )
    return TreeReconciliationResult(
        global_tree_records=(record,),
        final_original_index=np.asarray([0], dtype=np.int64),
        final_xyz_local_m=np.asarray([[1.0, 0.0, 10.0]], dtype=np.float64),
        final_tree_id=np.asarray([0], dtype=np.int32),
        review_tree_ids=(),
        span_tree_to_global={(0, 0): 0},
        diagnostics={"duplicate_final_original_index": 0},
    )


def _scheduler(tmp_path: Path, reconciled: TreeReconciliationResult) -> BatchAnalysisScheduler:
    """Build only the Stage-B publication seam; no source or science runs."""

    scheduler = object.__new__(BatchAnalysisScheduler)
    scheduler.project_root = tmp_path
    scheduler.project = SimpleNamespace(project_id="project-fixture")
    scheduler.bridge = SimpleNamespace(scan=SimpleNamespace(scan_id="scan-fixture"))
    scheduler.risk_policy = RiskPolicy()
    scheduler.config = _CONFIG
    scheduler._global_reconcile = lambda _bindings, _results: reconciled
    scheduler._persist_run = lambda _run: None
    scheduler.open_editor = lambda _result: SimpleNamespace(persist=lambda: None)
    return scheduler


def _span_results() -> dict[str, SpanScienceResult]:
    # The canonical bridge order is S001=span-z and S002=span-a.  Their wires
    # are equidistant from the tree point, but have different nearest XYZ so a
    # request-order-dependent wire 0 is observable.
    return {
        "span-z": SpanScienceResult(
            tree_count=1,
            minimum_clearance_m=1.0,
            risk_threshold_m=0.5,
            risk_counts={},
            wire_result=_wire(-1.0),
        ),
        "span-a": SpanScienceResult(
            tree_count=1,
            minimum_clearance_m=1.0,
            risk_threshold_m=0.5,
            risk_counts={},
            wire_result=_wire(1.0),
        ),
    }


def _finish(
    tmp_path: Path,
    selected: tuple[str, ...],
    results: dict[str, SpanScienceResult],
) -> BatchAnalysisResult:
    scheduler = _scheduler(tmp_path, _reconciled())
    bindings = {
        "span-z": SimpleNamespace(internal_span_id="S001"),
        "span-a": SimpleNamespace(internal_span_id="S002"),
    }
    statuses = {span_id: BatchSpanStatus.AUTO_ANALYZED for span_id in selected}
    return scheduler._finish(
        "analysis-fixture",
        selected,
        statuses,
        {span_id: results[span_id] for span_id in selected},
        BatchTelemetry(span_count=len(selected)),
        bindings,
    )


def test_forward_reverse_batch_order_has_identical_clearance_and_wire_tie(
    tmp_path: Path,
) -> None:
    results = _span_results()
    forward = _finish(
        tmp_path / "forward",
        ("span-z", "span-a"),
        results,
    )
    reverse = _finish(
        tmp_path / "reverse",
        ("span-a", "span-z"),
        results,
    )

    assert forward.final_clearance is not None
    assert reverse.final_clearance is not None
    assert forward.final_clearance.status == reverse.final_clearance.status == "VALID"
    assert forward.final_clearance.edge_count == reverse.final_clearance.edge_count == 4
    assert forward.final_clearance.global_minimum_clearance_m == 1.0
    assert reverse.final_clearance.global_minimum_clearance_m == 1.0
    assert forward.final_clearance.diagnostics["distance_ready_segments"] == 2
    assert reverse.final_clearance.diagnostics["distance_ready_segments"] == 2
    left = forward.final_clearance.records[0]
    right = reverse.final_clearance.records[0]
    assert left.minimum_clearance_m == right.minimum_clearance_m == 1.0
    assert left.nearest_wire_id == right.nearest_wire_id == 0
    assert left.nearest_wire_xyz_local_m == right.nearest_wire_xyz_local_m == (
        1.0,
        -1.0,
        10.0,
    )
    assert left.to_dict() == right.to_dict()
    assert forward.duplicate_final_original_index == reverse.duplicate_final_original_index == 0
    assert forward.batch_diagnostics["canonical_span_order"] == ["span-z", "span-a"]
    assert reverse.batch_diagnostics["canonical_span_order"] == ["span-z", "span-a"]
    assert list(forward.span_results) == list(reverse.span_results) == ["span-z", "span-a"]


def test_canonical_span_order_survives_json_mapping_round_trip(tmp_path: Path) -> None:
    result = _finish(
        tmp_path / "hydration",
        ("span-a", "span-z"),
        _span_results(),
    )
    manifest = result.to_dict()
    serialized = json.loads(json.dumps(manifest, sort_keys=True))
    # JSON serialization sorts product IDs, so deliberately reconstruct the
    # runtime mapping in that serialized order before BatchAnalysisResult sees
    # the explicit topology-derived order metadata.
    raw_results = {
        span_id: result.span_results[span_id]
        for span_id in serialized["span_results"]
    }
    hydrated = BatchAnalysisResult(
        analysis_run_id=result.analysis_run_id,
        project_id=result.project_id,
        scan_id=result.scan_id,
        status=result.status,
        span_statuses=result.span_statuses,
        span_results=raw_results,
        telemetry=result.telemetry,
        risk_policy_snapshot=result.risk_policy_snapshot,
        global_reconciled=result.global_reconciled,
        final_clearance=result.final_clearance,
        batch_diagnostics=serialized["batch_diagnostics"],
    )
    assert list(hydrated.span_results) == ["span-z", "span-a"]


def test_canonical_order_allows_selected_span_without_completed_result() -> None:
    result = BatchAnalysisResult(
        analysis_run_id="analysis-fixture",
        project_id="project-fixture",
        scan_id="scan-fixture",
        status="PARTIAL_FAILURE",
        span_statuses={"span-z": BatchSpanStatus.FAILED, "span-a": BatchSpanStatus.AUTO_ANALYZED},
        span_results={"span-a": _span_results()["span-a"]},
        telemetry=BatchTelemetry(span_count=2),
        risk_policy_snapshot={"warning_clearance_threshold_m": 0.5},
        batch_diagnostics={"canonical_span_order": ["span-z", "span-a"]},
    )
    assert list(result.span_results) == ["span-a"]


def test_invalid_canonical_order_metadata_fails_closed() -> None:
    kwargs = dict(
        analysis_run_id="analysis-fixture",
        project_id="project-fixture",
        scan_id="scan-fixture",
        status="COMPLETE",
        span_statuses={},
        telemetry=BatchTelemetry(),
        risk_policy_snapshot={},
    )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        BatchAnalysisResult(
            **kwargs,
            span_results={},
            batch_diagnostics={"canonical_span_order": ["span-z", "span-z"]},
        )
    with pytest.raises(ValueError, match="absent from canonical_span_order"):
        BatchAnalysisResult(
            **kwargs,
            span_results={"span-a": _span_results()["span-a"]},
            batch_diagnostics={"canonical_span_order": ["span-z"]},
        )


def test_legacy_result_without_order_metadata_preserves_mapping_order() -> None:
    result = BatchAnalysisResult(
        analysis_run_id="analysis-fixture",
        project_id="project-fixture",
        scan_id="scan-fixture",
        status="COMPLETE",
        span_statuses={},
        span_results={
            "span-a": _span_results()["span-a"],
            "span-z": _span_results()["span-z"],
        },
        telemetry=BatchTelemetry(),
        risk_policy_snapshot={},
        batch_diagnostics={},
    )
    assert list(result.span_results) == ["span-a", "span-z"]


def test_editor_finalize_uses_hydrated_canonical_wire_sequence(tmp_path: Path) -> None:
    initial = _finish(tmp_path / "initial", ("span-a", "span-z"), _span_results())
    hydrated = BatchAnalysisResult(
        analysis_run_id=initial.analysis_run_id,
        project_id=initial.project_id,
        scan_id=initial.scan_id,
        status=initial.status,
        span_statuses={
            "span-a": BatchSpanStatus.USER_REVIEWED,
            "span-z": BatchSpanStatus.USER_REVIEWED,
        },
        span_results={
            "span-a": initial.span_results["span-a"],
            "span-z": initial.span_results["span-z"],
        },
        telemetry=initial.telemetry,
        risk_policy_snapshot=initial.risk_policy_snapshot,
        global_reconciled=initial.global_reconciled,
        final_clearance=initial.final_clearance,
        batch_diagnostics=initial.batch_diagnostics,
    )
    assert list(hydrated.span_results) == ["span-z", "span-a"]
    editor = BatchTreeEditor(
        tmp_path / "editor",
        hydrated,
        tree_ids_by_span={"span-z": (0,), "span-a": (0,)},
    )
    finalized = editor.finalize()
    assert finalized.final_clearance is not None
    assert finalized.final_clearance.records[0].nearest_wire_id == 0
    assert finalized.final_clearance.records[0].nearest_wire_xyz_local_m == (
        1.0,
        -1.0,
        10.0,
    )
    assert list(finalized.span_results) == ["span-z", "span-a"]


def test_dataclass_replace_retains_canonical_span_order() -> None:
    result = BatchAnalysisResult(
        analysis_run_id="analysis-fixture",
        project_id="project-fixture",
        scan_id="scan-fixture",
        status="COMPLETE",
        span_statuses={},
        span_results={
            "span-a": _span_results()["span-a"],
            "span-z": _span_results()["span-z"],
        },
        telemetry=BatchTelemetry(),
        risk_policy_snapshot={},
        batch_diagnostics={"canonical_span_order": ["span-z", "span-a"]},
    )
    replaced = replace(result, status="FINALIZED")
    assert list(replaced.span_results) == ["span-z", "span-a"]
