"""Bounded Stage-B restart tests for exact per-Span artifacts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from algorithms.spans.canonical_executor import _span_record
from algorithms.spans.clearance import SpanClearanceResult, SpanDangerPoint, SpanTreeWireRecord
from algorithms.spans.project_adapter import (
    ScienceAdapterError,
    SourceSharedPreparation,
    SpanScienceResult,
)
from algorithms.spans.reconcile import reconcile_trees
from algorithms.spans.tree_runner import SpanTreeOutcome
from algorithms.spans.wire import SpanWireResult
from core.config.fast_baseline import FastBaselineConfig, canonical_digest
from core.stage_a import (
    CoordinateDeclaration,
    Line,
    Project,
    Scan,
    SourceDescriptor,
    Span,
    Tower,
    save_project,
)
from core.stage_a.models import stable_id
from core.stage_b.batch import BatchAnalysisScheduler, BatchSpanStatus


def _project(tmp_path: Path, *, span_count: int = 2) -> tuple[Project, str, tuple[str, ...]]:
    source = tmp_path / "scene.las"
    source.write_bytes(b"fixture-source")
    source_descriptor = SourceDescriptor(
        "source-fixture", str(source), "LAS", source.stat().st_size
    )
    scan = Scan(
        stable_id("scan"),
        source_descriptor,
        str(source),
        {"point_count": 24},
        coverage={"point_count": 24, "origin_xyz": [0.0, 0.0, 0.0]},
    )
    towers = tuple(
        Tower(
            f"tower-{index}",
            f"asset-{index}",
            f"Tower {index}",
            (float(index * 10), 0.0, 0.0),
            CoordinateDeclaration.local_metric(),
            "MANUAL_PICK",
            {"height_m": 12.0},
        )
        for index in range(span_count + 1)
    )
    line = Line(stable_id("line"), "Line A", tuple(tower.tower_id for tower in towers))
    spans = tuple(
        Span(
            f"span-{index}",
            line.line_id,
            towers[index].tower_id,
            towers[index + 1].tower_id,
        )
        for index in range(span_count)
    )
    project = replace(
        Project.new("Stage B Exact Cache Fixture"),
        lines=(line,),
        towers=towers,
        spans=spans,
        scans=(scan,),
    )
    save_project(project, tmp_path)
    return project, scan.scan_id, tuple(span.span_id for span in spans)


def _preparation(root: Path, input_data) -> SourceSharedPreparation:
    source_digest = canonical_digest(dict(input_data.source_identity))
    relative = Path("artifacts/project-runs/source-shared/preparation.json")
    preparation = SourceSharedPreparation(
        source_binding_digest=source_digest,
        configuration_digest=input_data.science_configuration_digest,
        manifest_reference=relative.as_posix(),
        ground_preparation={"coarse_ground_reference": "ground.npy"},
        telemetry={"source_prep_count": 1},
    )
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = preparation.to_dict()
    # Match the existing source-shared writer, which binds this file by its
    # path but predates a redundant manifest_reference field.
    manifest_payload.pop("manifest_reference")
    path.write_text(
        json.dumps(
            {
                "schema": "test-source-shared-preparation-v1",
                **manifest_payload,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return preparation


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _exact_result(
    input_data, output_root: Path, config: FastBaselineConfig
) -> SpanScienceResult:
    span = _span_record(input_data, config)
    internal_number = int(input_data.span_id[1:])
    x = 5.0 + 10.0 * (internal_number - 1)
    tree_outcome = SpanTreeOutcome(
        span=span,
        tree_records=(
            {
                "tree_id": 0,
                "apex_xy_local_m": [x, 0.0, 10.0],
                "apex_height_m": 10.0,
            },
        ),
        raw_original_index=np.asarray([internal_number], dtype=np.int64),
        raw_xyz_local_m=np.asarray([[x, 0.0, 10.0]], dtype=np.float32),
        raw_tree_id=np.asarray([0], dtype=np.int32),
        in_core=np.asarray([True], dtype=bool),
        raw_local_maxima=1,
        accepted_tree_tops=1,
        assigned_raw_point_count=1,
        artifact_root=output_root / "tree",
    )
    tree_root = output_root / "tree"
    tree_root.mkdir(parents=True, exist_ok=True)
    np.savez(
        tree_root / "tree_members.npz",
        original_index=tree_outcome.raw_original_index,
        xyz_local_m=tree_outcome.raw_xyz_local_m,
        tree_id=tree_outcome.raw_tree_id,
        in_core=tree_outcome.in_core,
    )
    _write_json(tree_root / "span_tree_manifest.json", tree_outcome.to_manifest())

    # Keep two W1 segments so restart tests exercise the persisted array
    # contract.  The loader's strand display summaries are not authoritative;
    # clearance consumes these exact W1 arrays.
    start_x = float(span.start_xyz_absolute_m[0])
    end_x = float(span.end_xyz_absolute_m[0])
    midpoint = (start_x + end_x) / 2.0
    wire_arrays = {
        "wire_id": np.asarray([0], dtype=np.int32),
        "wire_status": np.asarray([0], dtype=np.uint8),
        "wire_method": np.asarray([1], dtype=np.uint8),
        "wire_quality_flags": np.asarray([0], dtype=np.uint16),
        "wire_has_distance_ready_segments": np.asarray([True], dtype=bool),
        "wire_continuous_distance_ready": np.asarray([True], dtype=bool),
        "wire_segment_offsets": np.asarray([0, 2], dtype=np.int64),
        "segment_wire_id": np.asarray([0, 0], dtype=np.int32),
        "segment_index": np.asarray([0, 1], dtype=np.int32),
        "segment_method": np.asarray([1, 1], dtype=np.uint8),
        "segment_quality_flags": np.asarray([0, 0], dtype=np.uint16),
        "segment_distance_ready": np.asarray([True, True], dtype=bool),
        "segment_parameter_domain": np.asarray([[0.0, 0.5], [0.5, 1.0]], dtype=float),
        "sampled_curve_xyz": np.asarray(
            [
                [start_x, 0.0, 12.0],
                [midpoint, 0.0, 11.5],
                [midpoint, 0.0, 11.5],
                [end_x, 0.0, 12.0],
            ],
            dtype=np.float32,
        ),
        "sampled_curve_parameter": np.asarray([0.0, 0.5, 0.5, 1.0], dtype=np.float32),
        "sampled_curve_offsets": np.asarray([0, 2, 4], dtype=np.int64),
    }
    wire_root = output_root / "wire"
    wire_root.mkdir(parents=True, exist_ok=True)
    np.savez(wire_root / "w1_arrays.npz", **wire_arrays)
    _write_json(
        wire_root / "span_wire_result.json",
        {
            "schema": "fast-baseline-span-wire-v1",
            "marker": "FAST_MATURE_SPAN_WIRE",
            "strands": [],
            "diagnostics": {"fixture": "two-segment-w1"},
        },
    )
    wire_result = SpanWireResult(
        strands=(),
        w1_arrays=wire_arrays,
        candidate_mask=np.empty(0, dtype=bool),
        cluster_id_by_rep=np.empty(0, dtype=np.int32),
        diagnostics={"fixture": "two-segment-w1"},
    )
    reconciled = reconcile_trees([tree_outcome], config, origin_xyz=(0.0, 0.0, 0.0))
    return SpanScienceResult(
        tree_count=reconciled.final_tree_count,
        minimum_clearance_m=None,
        risk_threshold_m=0.5,
        risk_counts={},
        artifact_files={
            "tree": str(tree_root / "span_tree_manifest.json"),
            "wire": str(wire_root / "span_wire_result.json"),
        },
        diagnostics={"fixture": "exact-cache"},
        tree_outcome=tree_outcome,
        reconciled=reconciled,
        wire_result=wire_result,
    )


def _runner(calls: list[str], config: FastBaselineConfig, *, fail_span: str | None = None):
    def run(input_data, _preparation, output_root):
        calls.append(input_data.span_id)
        if input_data.span_id == fail_span:
            raise RuntimeError("fixture span failure")
        return _exact_result(input_data, output_root, config)

    return run


def _near_wire_clearance(
    global_reconciled,
    _wire_results,
    _config,
    *,
    danger_threshold_m,
    acceleration_mode="AUTO",
) -> SpanClearanceResult:
    del danger_threshold_m, acceleration_mode
    source_record = global_reconciled.global_tree_records[0]
    clearance_record = SpanTreeWireRecord(
        global_tree_id=source_record.global_tree_id,
        external_name=source_record.external_name,
        persistent_tree_id=source_record.persistent_tree_id,
        line_id=source_record.line_id,
        span_id=source_record.span_id,
        span_name=source_record.span_name,
        nearest_wire_id=0,
        minimum_clearance_m=0.1,
        nearest_tree_original_index=1,
        nearest_tree_xyz_local_m=(5.0, 0.0, 10.0),
        nearest_wire_xyz_local_m=(5.0, 0.0, 10.1),
        tree_point_count=1,
        evaluated_edge_count=1,
        status="DANGER",
        danger_point_count=1,
    )
    return SpanClearanceResult(
        records=(clearance_record,),
        status="VALID",
        global_minimum_clearance_m=0.1,
        edge_count=1,
        diagnostics={"edge_count": 1, "fixture": "near-wire"},
        danger_points=(
            SpanDangerPoint(
                global_tree_id=source_record.global_tree_id,
                persistent_tree_id=source_record.persistent_tree_id,
                original_index=1,
                xyz_local_m=(5.0, 0.0, 10.0),
            ),
        ),
    )


def _scheduler(
    root: Path,
    project: Project,
    scan_id: str,
    calls: list[str],
    *,
    config: FastBaselineConfig | None = None,
    fail_span: str | None = None,
    source_preparer=None,
) -> BatchAnalysisScheduler:
    tracked_config = config or FastBaselineConfig()
    return BatchAnalysisScheduler(
        root,
        project,
        scan_id,
        config=tracked_config,
        science_runner=_runner(calls, tracked_config, fail_span=fail_span),
        source_preparer=source_preparer or _preparation,
        source_access_enabled=False,
    )


def _batch_root(root: Path, run_id: str) -> Path:
    return root / "artifacts" / "project-runs" / "batches" / run_id


def _file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _project_snapshot(root: Path) -> dict[str, tuple[int, bytes]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_mtime_ns, path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


def _assert_batch_results_equal(left, right, span_ids: tuple[str, ...]) -> None:
    assert left.global_reconciled is not None
    assert right.global_reconciled is not None
    np.testing.assert_array_equal(
        left.global_reconciled.final_original_index,
        right.global_reconciled.final_original_index,
    )
    np.testing.assert_array_equal(
        left.global_reconciled.final_xyz_local_m,
        right.global_reconciled.final_xyz_local_m,
    )
    np.testing.assert_array_equal(
        left.global_reconciled.final_tree_id,
        right.global_reconciled.final_tree_id,
    )
    assert [record.to_dict() for record in left.global_reconciled.global_tree_records] == [
        record.to_dict() for record in right.global_reconciled.global_tree_records
    ]
    assert left.final_clearance is not None
    assert right.final_clearance is not None
    left_clearance_diagnostics = {
        key: value
        for key, value in left.final_clearance.diagnostics.items()
        if not key.endswith("_seconds")
    }
    right_clearance_diagnostics = {
        key: value
        for key, value in right.final_clearance.diagnostics.items()
        if not key.endswith("_seconds")
    }
    assert (
        left.final_clearance.status,
        left.final_clearance.global_minimum_clearance_m,
        left.final_clearance.edge_count,
        [record.to_dict() for record in left.final_clearance.records],
        left_clearance_diagnostics,
    ) == (
        right.final_clearance.status,
        right.final_clearance.global_minimum_clearance_m,
        right.final_clearance.edge_count,
        [record.to_dict() for record in right.final_clearance.records],
        right_clearance_diagnostics,
    )
    for span_id in span_ids:
        left_tree = left.span_results[span_id].tree_outcome
        right_tree = right.span_results[span_id].tree_outcome
        assert left_tree is not None and right_tree is not None
        np.testing.assert_array_equal(left_tree.raw_original_index, right_tree.raw_original_index)
        np.testing.assert_array_equal(left_tree.raw_xyz_local_m, right_tree.raw_xyz_local_m)
        left_wire = left.span_results[span_id].wire_result
        right_wire = right.span_results[span_id].wire_result
        assert left_wire is not None and right_wire is not None
        for key, left_array in left_wire.w1_arrays.items():
            np.testing.assert_array_equal(left_array, right_wire.w1_arrays[key])


@pytest.mark.parametrize("span_count", (2, 3))
def test_retry_rehydrates_exact_success_and_unions_rerun_span(
    tmp_path: Path, span_count: int
) -> None:
    project, scan_id, span_ids = _project(tmp_path, span_count=span_count)
    # Exercise the caller-facing reverse request order; reconciliation must
    # still use the topology-derived canonical order.
    span_count = int(span_count)
    requested = tuple(reversed(span_ids))
    failed_internal = f"S{span_count:03d}"
    run_id = f"analysis-exact-retry-{span_count}"
    calls: list[str] = []
    first = _scheduler(
        tmp_path,
        project,
        scan_id,
        calls,
        fail_span=failed_internal,
    ).run(requested, analysis_run_id=run_id)
    assert first.status == "PARTIAL_FAILURE"
    assert calls == list(
        reversed([f"S{index:03d}" for index in range(1, span_count + 1)])
    )

    calls.clear()
    second = _scheduler(tmp_path, project, scan_id, calls).run(
        requested, analysis_run_id=run_id
    )

    assert second.status == "COMPLETE"
    assert calls == [failed_internal]
    assert second.telemetry.cache_hit_count >= 1
    assert set(second.span_results) == set(requested)
    assert all(result.tree_outcome is not None for result in second.span_results.values())
    assert all(result.wire_result is not None for result in second.span_results.values())
    expected_original = np.arange(1, span_count + 1, dtype=np.int64)
    assert second.global_reconciled is not None
    np.testing.assert_array_equal(
        second.global_reconciled.final_original_index, expected_original
    )

    control_calls: list[str] = []
    control = _scheduler(tmp_path, project, scan_id, control_calls).run(
        span_ids, analysis_run_id=f"analysis-exact-control-{span_count}"
    )
    assert control.status == "COMPLETE"
    assert control_calls == [f"S{index:03d}" for index in range(1, span_count + 1)]
    _assert_batch_results_equal(second, control, span_ids)


def test_complete_restart_hydrates_exact_geometry_without_writes(tmp_path: Path) -> None:
    project, scan_id, span_ids = _project(tmp_path, span_count=2)
    run_id = "analysis-exact-restart"
    initial_calls: list[str] = []
    first = _scheduler(tmp_path, project, scan_id, initial_calls).run(
        span_ids, analysis_run_id=run_id
    )
    assert first.status == "COMPLETE"
    batch_root = _batch_root(tmp_path, run_id)
    before = _file_snapshot(batch_root)

    restarted_calls: list[str] = []

    def should_not_prepare(*_args):
        raise AssertionError("clean exact restart must not prepare source")

    second = _scheduler(
        tmp_path,
        project,
        scan_id,
        restarted_calls,
        source_preparer=should_not_prepare,
    ).run(span_ids, analysis_run_id=run_id)

    assert second.status == "COMPLETE"
    assert restarted_calls == []
    assert second.global_reconciled is not None
    np.testing.assert_array_equal(
        second.global_reconciled.final_original_index,
        first.global_reconciled.final_original_index,
    )
    np.testing.assert_array_equal(
        second.global_reconciled.final_tree_id,
        first.global_reconciled.final_tree_id,
    )
    assert second.final_clearance is not None
    assert first.final_clearance is not None
    assert (
        second.final_clearance.status,
        second.final_clearance.global_minimum_clearance_m,
        second.final_clearance.edge_count,
        [record.to_dict() for record in second.final_clearance.records],
        dict(second.final_clearance.diagnostics),
    ) == (
        first.final_clearance.status,
        first.final_clearance.global_minimum_clearance_m,
        first.final_clearance.edge_count,
        [record.to_dict() for record in first.final_clearance.records],
        dict(first.final_clearance.diagnostics),
    )
    assert [
        (
            record.persistent_tree_id,
            record.merged_span_tree_keys,
            record.identity_state,
        )
        for record in second.global_reconciled.global_tree_records
    ] == [
        (
            record.persistent_tree_id,
            record.merged_span_tree_keys,
            record.identity_state,
        )
        for record in first.global_reconciled.global_tree_records
    ]
    for span_id in span_ids:
        cached_tree = second.span_results[span_id].tree_outcome
        initial_tree = first.span_results[span_id].tree_outcome
        assert cached_tree is not None and initial_tree is not None
        np.testing.assert_array_equal(
            cached_tree.raw_original_index, initial_tree.raw_original_index
        )
        np.testing.assert_array_equal(
            cached_tree.raw_xyz_local_m, initial_tree.raw_xyz_local_m
        )
        cached_wire = second.span_results[span_id].wire_result
        initial_wire = first.span_results[span_id].wire_result
        assert cached_wire is not None and initial_wire is not None
        for key, initial_array in initial_wire.w1_arrays.items():
            np.testing.assert_array_equal(cached_wire.w1_arrays[key], initial_array)
    assert _file_snapshot(batch_root) == before


def test_complete_restart_hydrates_persisted_near_wire_danger_points_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, scan_id, span_ids = _project(tmp_path, span_count=1)
    run_id = "analysis-exact-danger-restart"

    import algorithms.spans.clearance as clearance_module

    monkeypatch.setattr(
        clearance_module,
        "run_span_tree_wire_clearance",
        _near_wire_clearance,
    )
    first_calls: list[str] = []
    first = _scheduler(tmp_path, project, scan_id, first_calls).run(
        span_ids, analysis_run_id=run_id
    )
    assert first.status == "COMPLETE"
    assert first.final_clearance is not None
    assert len(first.final_clearance.danger_points) == 1
    before = _project_snapshot(tmp_path)

    restarted_calls: list[str] = []

    def should_not_prepare(*_args):
        raise AssertionError("clean exact restart must not prepare source")

    second = _scheduler(
        tmp_path,
        project,
        scan_id,
        restarted_calls,
        source_preparer=should_not_prepare,
    ).run(span_ids, analysis_run_id=run_id)

    assert second.status == "COMPLETE"
    assert restarted_calls == []
    assert second.final_clearance is not None
    assert [point.to_dict() for point in second.final_clearance.danger_points] == [
        point.to_dict() for point in first.final_clearance.danger_points
    ]
    assert _project_snapshot(tmp_path) == before


@pytest.mark.parametrize("mutation", ("missing_npz", "missing_json", "tamper_npz", "tamper_json"))
def test_invalid_persisted_danger_sidecar_is_a_cache_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    project, scan_id, span_ids = _project(tmp_path, span_count=1)
    run_id = f"analysis-exact-danger-miss-{mutation}"

    import algorithms.spans.clearance as clearance_module

    monkeypatch.setattr(
        clearance_module,
        "run_span_tree_wire_clearance",
        _near_wire_clearance,
    )
    first_calls: list[str] = []
    first = _scheduler(tmp_path, project, scan_id, first_calls).run(
        span_ids, analysis_run_id=run_id
    )
    assert first.status == "COMPLETE"
    batch_root = _batch_root(tmp_path, run_id)
    npz_path = batch_root / "danger_points_final.npz"
    json_path = batch_root / "danger_points_final.json"
    if mutation == "missing_npz":
        npz_path.unlink()
    elif mutation == "missing_json":
        json_path.unlink()
    elif mutation == "tamper_npz":
        npz_path.write_bytes(b"tampered")
    else:
        json_path.write_text("{}", encoding="utf-8")

    rerun_calls: list[str] = []
    second = _scheduler(tmp_path, project, scan_id, rerun_calls).run(
        span_ids, analysis_run_id=run_id
    )

    assert second.status == "COMPLETE"
    assert rerun_calls == []
    assert second.final_clearance is not None
    assert len(second.final_clearance.danger_points) == 1
    assert npz_path.is_file() and json_path.is_file()


def test_tampered_run_clearance_proof_forces_exact_span_recompute_path(
    tmp_path: Path,
) -> None:
    project, scan_id, span_ids = _project(tmp_path, span_count=1)
    run_id = "analysis-tampered-run-proof"
    first_calls: list[str] = []
    first = _scheduler(tmp_path, project, scan_id, first_calls).run(
        span_ids, analysis_run_id=run_id
    )
    assert first.status == "COMPLETE"
    clearance_path = _batch_root(tmp_path, run_id) / "batch-clearance.json"
    assert clearance_path.is_file()
    clearance_path.write_text("{}", encoding="utf-8")

    prep_calls: list[str] = []

    def preparing(root, input_data):
        prep_calls.append(input_data.span_id)
        return _preparation(root, input_data)

    rerun_calls: list[str] = []
    second = _scheduler(
        tmp_path,
        project,
        scan_id,
        rerun_calls,
        source_preparer=preparing,
    ).run(span_ids, analysis_run_id=run_id)

    assert second.status == "COMPLETE"
    assert prep_calls == ["S001"]
    # The detached run proof is invalid, but the exact Span artifact remains
    # usable, so the normal path rehydrates that Span without science rerun.
    assert rerun_calls == []
    assert second.telemetry.cache_hit_count >= 1


def test_source_change_records_cache_miss_reason_after_recompute(tmp_path: Path) -> None:
    project, scan_id, span_ids = _project(tmp_path, span_count=1)
    run_id = "analysis-source-change"
    first_calls: list[str] = []
    _scheduler(tmp_path, project, scan_id, first_calls).run(
        span_ids, analysis_run_id=run_id
    )
    source = Path(project.scans[0].point_cloud_source)
    source.write_bytes(source.read_bytes() + b"-changed")

    rerun_calls: list[str] = []
    result = _scheduler(tmp_path, project, scan_id, rerun_calls).run(
        span_ids, analysis_run_id=run_id
    )

    assert result.status == "COMPLETE"
    assert rerun_calls == ["S001"]
    cache_diagnostics = result.span_results[span_ids[0]].diagnostics["span_cache"]
    assert cache_diagnostics == {
        "status": "MISS_RECOMPUTED",
        "reason": "SPAN_CACHE_SOURCE_FINGERPRINT_MISMATCH",
        "authoritative": False,
    }


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    (
        ("missing_index", "SPAN_CACHE_INDEX_MISSING"),
        ("tree_members", "SPAN_CACHE_ARTIFACT_HASH_MISMATCH:tree_members"),
        ("wire_arrays", "SPAN_CACHE_ARTIFACT_HASH_MISMATCH:wire_arrays"),
        ("missing_required", "SPAN_CACHE_ARTIFACT_MISSING:tree_members"),
        ("configuration", "SPAN_CACHE_CONFIGURATION_MISMATCH"),
        ("binding", "SPAN_CACHE_BINDING_MISMATCH"),
        ("preparation", "SPAN_CACHE_SOURCE_DIGEST_MISMATCH"),
    ),
)
def test_one_invalid_span_cache_reruns_only_that_span(
    tmp_path: Path, mutation: str, expected_reason: str
) -> None:
    project, scan_id, span_ids = _project(tmp_path, span_count=2)
    run_id = "analysis-one-span-cache-miss-" + mutation
    first_calls: list[str] = []
    first = _scheduler(tmp_path, project, scan_id, first_calls).run(
        span_ids, analysis_run_id=run_id
    )
    assert first.status == "COMPLETE"
    bad_span = span_ids[0]
    span_root = _batch_root(tmp_path, run_id) / bad_span
    index_path = span_root / "span-cache-index.json"
    if mutation == "missing_index":
        index_path.unlink()
    elif mutation == "tree_members":
        path = span_root / "tree" / "tree_members.npz"
        path.write_bytes(path.read_bytes() + b"tampered")
    elif mutation == "wire_arrays":
        path = span_root / "wire" / "w1_arrays.npz"
        path.write_bytes(path.read_bytes() + b"tampered")
    elif mutation == "missing_required":
        (span_root / "tree" / "tree_members.npz").unlink()
    else:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if mutation == "configuration":
            payload["configuration"]["fast_baseline_digest"] = "0" * 64
        elif mutation == "binding":
            payload["binding"]["identity"]["span_id"] = "tampered-span"
        else:
            payload["source_preparation"]["source_binding_digest"] = "0" * 64
        _write_json(index_path, payload)

    rerun_calls: list[str] = []
    result = _scheduler(tmp_path, project, scan_id, rerun_calls).run(
        span_ids, analysis_run_id=run_id
    )

    assert result.status == "COMPLETE"
    assert rerun_calls == ["S001"]
    _assert_batch_results_equal(result, first, span_ids)
    assert result.span_results[bad_span].diagnostics["span_cache"] == {
        "status": "MISS_RECOMPUTED",
        "reason": expected_reason,
        "authoritative": False,
    }


@pytest.mark.parametrize("guard", ("journal", "finalized", "malformed_journal", "malformed_batch"))
def test_immutable_or_finalized_run_fails_before_source_or_runner(
    tmp_path: Path, guard: str
) -> None:
    project, scan_id, span_ids = _project(tmp_path, span_count=1)
    run_id = "analysis-immutable-guard-" + guard
    initial_calls: list[str] = []
    _scheduler(tmp_path, project, scan_id, initial_calls).run(
        span_ids, analysis_run_id=run_id
    )
    batch_root = _batch_root(tmp_path, run_id)
    manifest_path = batch_root / "batch.json"
    if guard == "journal":
        _write_json(
            batch_root / "tree_edits.json",
            {
                "schema": "folder-to-excel-tree-edits-v1",
                "analysis_run_id": run_id,
                "operations": [{"op": "DELETE", "tree_ids": [0]}],
            },
        )
    elif guard == "finalized":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["span_statuses"][span_ids[0]] = BatchSpanStatus.FINALIZED.value
        _write_json(manifest_path, manifest)
    elif guard == "malformed_journal":
        (batch_root / "tree_edits.json").write_text("{", encoding="utf-8")
    else:
        manifest_path.write_text("{", encoding="utf-8")

    before = _project_snapshot(tmp_path)

    source_calls: list[str] = []
    runner_calls: list[str] = []

    def forbidden_preparer(*_args):
        source_calls.append("source")
        raise AssertionError("immutable run must fail before source preparation")

    with pytest.raises(ScienceAdapterError) as raised:
        _scheduler(
            tmp_path,
            project,
            scan_id,
            runner_calls,
            source_preparer=forbidden_preparer,
        ).run(span_ids, analysis_run_id=run_id)
    assert source_calls == []
    assert runner_calls == []
    if guard == "journal":
        assert str(raised.value) == (
            "BATCH_RUN_IMMUTABLE_EDITED_REQUIRES_WORKFLOW_REHYDRATION"
        )
    else:
        expected = (
            "BATCH_RUN_IMMUTABLE_FINALIZED_REQUIRES_WORKFLOW_REHYDRATION"
            if guard == "finalized"
            else (
                "BATCH_RUN_IMMUTABLE_EDIT_JOURNAL_INVALID"
                if guard == "malformed_journal"
                else "BATCH_RUN_MANIFEST_INVALID"
            )
        )
        assert str(raised.value) == expected
    assert _project_snapshot(tmp_path) == before


def test_invalid_existing_batch_manifest_fails_closed_before_writes(tmp_path: Path) -> None:
    project, scan_id, span_ids = _project(tmp_path, span_count=1)
    run_id = "analysis-invalid-manifest"
    batch_root = _batch_root(tmp_path, run_id)
    batch_root.mkdir(parents=True, exist_ok=True)
    (batch_root / "batch.json").write_text('{"status":"COMPLETE"}', encoding="utf-8")

    source_calls: list[str] = []
    with pytest.raises(ScienceAdapterError, match="BATCH_RUN_MANIFEST_INVALID"):
        _scheduler(
            tmp_path,
            project,
            scan_id,
            [],
            source_preparer=lambda *_args: source_calls.append("source"),
        ).run(span_ids, analysis_run_id=run_id)
    assert source_calls == []
