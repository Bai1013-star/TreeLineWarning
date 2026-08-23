from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from algorithms.spans.project_adapter import SourceSharedPreparation, SpanScienceResult
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
from core.stage_b import (
    BatchAnalysisResult,
    BatchAnalysisScheduler,
    BatchSpanStatus,
    BatchTelemetry,
    BatchTreeEditor,
    ProjectScienceBridge,
    ensure_ground_preparation,
    export_batch_prediction,
)
from core.stage_b.failures import structured_failure


def _project(tmp_path: Path, *, span_count: int = 2) -> tuple[Project, str, tuple[str, ...]]:
    source = tmp_path / "scene.las"
    source.write_bytes(b"fixture")
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
        Project.new("Stage B Fixture"),
        lines=(line,),
        towers=towers,
        spans=spans,
        scans=(scan,),
    )
    save_project(project, tmp_path)
    return project, scan.scan_id, tuple(span.span_id for span in spans)


def _preparation(_root: Path, input_data) -> SourceSharedPreparation:
    return SourceSharedPreparation(
        source_binding_digest="a" * 64,
        configuration_digest=input_data.science_configuration_digest,
        manifest_reference="artifacts/project-runs/source-shared/preparation.json",
        ground_preparation={"coarse_ground_reference": "ground.npy"},
        telemetry={"source_prep_count": 1, "cache_miss_count": 1, "ground_prep_count": 1},
    )


def test_project_science_bridge_preserves_product_identity_and_policy(tmp_path: Path) -> None:
    project, scan_id, span_ids = _project(tmp_path, span_count=1)
    bridge = ProjectScienceBridge(project, scan_id)
    binding = bridge.bind_span(span_ids[0], analysis_run_id="analysis-fixture")

    assert binding.identity.project_id == project.project_id
    assert binding.identity.scan_id == scan_id
    assert binding.identity.line_id == binding.product_span.line_id
    assert binding.identity.span_id == span_ids[0]
    assert binding.identity.analysis_run_id == "analysis-fixture"
    assert binding.science_input.risk_policy_snapshot["warning_clearance_threshold_m"] == 0.5
    assert binding.science_input.product_identity["source_identity"]["source_id"] == (
        project.scans[0].source_identity.source_id
    )
    assert binding.product_span.source_partition["full_source_copy"] is False


def test_project_science_bridge_uses_line_topology_for_internal_span_order(
    tmp_path: Path,
) -> None:
    project, scan_id, _span_ids = _project(tmp_path, span_count=2)
    topology_spans = (
        replace(project.spans[0], span_id="span-z-first"),
        replace(project.spans[1], span_id="span-a-second"),
    )
    project = replace(project, spans=topology_spans)
    bridge = ProjectScienceBridge(project, scan_id)
    assert bridge.bind_span("span-z-first").science_input.span_id == "S001"
    assert bridge.bind_span("span-a-second").science_input.span_id == "S002"


def test_ground_preparation_is_source_and_config_bound_and_reused(tmp_path: Path) -> None:
    source = tmp_path / "scene.las"
    source.write_bytes(b"fixture")
    identity = {
        "source_id": "source-fixture",
        "resolved_access_path": str(source),
        "source_file_sha256": "a" * 64,
        "origin_context": [1.0, 2.0, 3.0],
    }
    calls: list[int] = []

    def producer(_source: Path, output_root: Path):
        calls.append(1)
        output_root.mkdir(parents=True, exist_ok=True)
        ground = output_root / "ground_hag_result.json"
        ground.write_text("{}", encoding="utf-8")
        return {
            "ground_hag_reference": ground,
            "coarse_ground_xyz": np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
            "origin_xyz": (1.0, 2.0, 3.0),
        }

    first = ensure_ground_preparation(tmp_path, identity, producer=producer)
    second = ensure_ground_preparation(tmp_path, identity, producer=producer)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls == [1]
    assert first.coarse_ground_reference
    assert first.source_binding_digest == second.source_binding_digest

    changed = ensure_ground_preparation(
        tmp_path, identity, configuration={"changed": True}, producer=producer
    )
    assert changed.cache_hit is False
    assert calls == [1, 1]


def test_batch_scheduler_prepares_source_once_and_keeps_partial_failures_isolated(
    tmp_path: Path,
) -> None:
    project, scan_id, span_ids = _project(tmp_path, span_count=2)
    calls: list[str] = []

    def runner(input_data, _preparation, _output_root):
        calls.append(input_data.span_id)
        if input_data.span_id.endswith("002"):
            raise RuntimeError("fixture failure")
        return SpanScienceResult(
            tree_count=1,
            minimum_clearance_m=0.2,
            risk_threshold_m=0.5,
            risk_counts={"warning": 1},
            diagnostics={"tree_backend": "crown_stem_hybrid"},
        )

    scheduler = BatchAnalysisScheduler(
        tmp_path,
        project,
        scan_id,
        science_runner=runner,
        source_preparer=_preparation,
    )
    result = scheduler.run(span_ids)

    assert result.status == "PARTIAL_FAILURE"
    assert result.span_statuses[span_ids[0]] is BatchSpanStatus.AUTO_ANALYZED
    assert result.span_statuses[span_ids[1]] is BatchSpanStatus.FAILED
    assert len(calls) == 2
    assert result.telemetry.source_prep_count == 1
    assert result.telemetry.span_count == 2
    assert result.telemetry.source_full_pass_count == 0
    assert result.risk_policy_snapshot["warning_clearance_threshold_m"] == 0.5
    assert (tmp_path / "artifacts/project-runs/batches").is_dir()


def test_batch_scheduler_persists_structured_ground_runtime_failure(tmp_path: Path) -> None:
    project, scan_id, span_ids = _project(tmp_path, span_count=1)

    def unavailable_ground(_root, _input_data):
        raise RuntimeError("FAST_GROUND_BACKEND_UNAVAILABLE: Python PDAL binding missing")

    scheduler = BatchAnalysisScheduler(
        tmp_path,
        project,
        scan_id,
        source_preparer=unavailable_ground,
        source_access_enabled=False,
    )
    result = scheduler.run(span_ids, analysis_run_id="analysis-ground-runtime-failure")
    diagnostics = result.span_results[span_ids[0]].diagnostics

    assert result.status == "FAILED"
    assert diagnostics["failure_stage"] == "GROUND"
    assert diagnostics["failure_code"] == "FAST_GROUND_BACKEND_UNAVAILABLE"
    assert diagnostics["user_message"] == "地面处理组件未正确安装，当前档距尚未进入科学分析。"
    assert diagnostics["recoverable"] is True
    assert "Python PDAL binding missing" in diagnostics["technical_message"]

    manifest = json.loads(
        (
            tmp_path
            / "artifacts/project-runs/batches/analysis-ground-runtime-failure/batch.json"
        ).read_text(encoding="utf-8")
    )
    persisted = manifest["span_results"][span_ids[0]]["diagnostics"]
    assert persisted["failure_stage"] == "GROUND"
    assert persisted["recommended_action"].startswith("请安装")


def test_ground_bad_allocation_has_an_actionable_chinese_reason() -> None:
    failure = structured_failure(
        RuntimeError("FAST_GROUND_FAILED: bad allocation"), phase="SOURCE_PREPARATION"
    )

    assert failure["failure_stage"] == "GROUND"
    assert failure["failure_code"] == "FAST_GROUND_FAILED"
    assert failure["user_message"] == "点云规模超过当前可用内存，地面处理未能完成。"
    assert "关闭其他占用内存的程序" in failure["recommended_action"]
    assert failure["recoverable"] is True


def test_batch_scheduler_reuses_completed_span_on_restart(tmp_path: Path) -> None:
    project, scan_id, span_ids = _project(tmp_path, span_count=1)
    calls: list[str] = []

    def first_runner(input_data, _preparation, _output_root):
        calls.append(input_data.span_id)
        return SpanScienceResult(
            tree_count=1,
            minimum_clearance_m=1.2,
            risk_threshold_m=0.5,
            risk_counts={},
            diagnostics={"tree_backend": "crown_stem_hybrid"},
        )

    first = BatchAnalysisScheduler(
        tmp_path,
        project,
        scan_id,
        science_runner=first_runner,
        source_preparer=_preparation,
        source_access_enabled=False,
    )
    first.run(span_ids, analysis_run_id="analysis-restart-fixture")

    def restarted_runner(*_args):
        raise AssertionError("completed Span science must not rerun on restart")

    second = BatchAnalysisScheduler(
        tmp_path,
        project,
        scan_id,
        science_runner=restarted_runner,
        source_preparer=_preparation,
        source_access_enabled=False,
    )
    resumed = second.run(span_ids, analysis_run_id="analysis-restart-fixture")
    assert calls == ["S001"]
    assert resumed.span_statuses[span_ids[0]] is BatchSpanStatus.AUTO_ANALYZED


def test_reconciled_original_index_is_exclusive_and_order_independent() -> None:
    from algorithms.spans.reconcile import reconcile_trees
    from core.config.fast_baseline import FastBaselineConfig

    def outcome(span_id: int, core: tuple[bool, ...], indices: tuple[int, ...]):
        span = SimpleNamespace(
            line_id=1,
            line_name="Line",
            span_id=span_id,
            span_name=f"S{span_id + 1:03d}",
            start_xyz_absolute_m=(float(span_id * 10), 0.0, 0.0),
            end_xyz_absolute_m=(float((span_id + 1) * 10), 0.0, 0.0),
            length_m=10.0,
            half_width_m=9.0,
        )
        return SimpleNamespace(
            span=span,
            raw_original_index=np.asarray(indices, dtype=np.int64),
            raw_xyz_local_m=np.asarray([[float(item), 0.0, 10.0] for item in indices]),
            raw_tree_id=np.zeros(len(indices), dtype=np.int32),
            in_core=np.asarray(core, dtype=bool),
            tree_records=(
                {
                    "tree_id": 0,
                    "apex_xy_local_m": [9.5, 0.0, 10.0],
                    "apex_height_m": 10.0,
                },
            ),
        )

    left = outcome(0, (True, True, True), (10, 11, 12))
    right = outcome(1, (False, False, False), (10, 11, 12))
    result = reconcile_trees([left, right], FastBaselineConfig(), origin_xyz=(0.0, 0.0, 0.0))
    reverse = reconcile_trees([right, left], FastBaselineConfig(), origin_xyz=(0.0, 0.0, 0.0))
    assert result.diagnostics["duplicate_input_original_index_count"] == 3
    assert result.diagnostics["duplicate_final_original_index"] == 0
    assert len(result.final_original_index) == 3
    assert len(set(result.global_tree_records[0].merged_span_tree_keys)) == 2
    assert [item.apex_xy_local_m for item in result.global_tree_records] == [
        item.apex_xy_local_m for item in reverse.global_tree_records
    ]


def test_single_span_global_reconciliation_reuses_canonical_result(
    tmp_path: Path, monkeypatch,
) -> None:
    from algorithms.spans.reconcile import reconcile_trees
    from core.config.fast_baseline import FastBaselineConfig

    project, scan_id, _span_ids = _project(tmp_path, span_count=1)
    scheduler = BatchAnalysisScheduler(tmp_path, project, scan_id)
    span = SimpleNamespace(
        line_id=1,
        line_name="Line",
        span_id=0,
        span_name="S001",
        start_xyz_absolute_m=(0.0, 0.0, 0.0),
        end_xyz_absolute_m=(10.0, 0.0, 0.0),
        length_m=10.0,
        half_width_m=9.0,
    )
    outcome = SimpleNamespace(
        span=span,
        raw_original_index=np.asarray((10, 11, 12), dtype=np.int64),
        raw_xyz_local_m=np.asarray(
            ((1.0, 0.0, 3.0), (2.0, 0.0, 4.0), (3.0, 0.0, 5.0))
        ),
        raw_tree_id=np.zeros(3, dtype=np.int32),
        in_core=np.ones(3, dtype=bool),
        tree_records=(
            {
                "tree_id": 0,
                "apex_xy_local_m": [2.0, 0.0, 4.0],
                "apex_height_m": 4.0,
            },
        ),
    )
    canonical = reconcile_trees(
        [outcome], FastBaselineConfig(), origin_xyz=(0.0, 0.0, 0.0)
    )
    science = SpanScienceResult(
        tree_count=1,
        minimum_clearance_m=1.0,
        risk_threshold_m=0.5,
        risk_counts={},
        tree_outcome=outcome,
        reconciled=canonical,
    )

    monkeypatch.setattr(
        "core.stage_b.batch.reconcile_trees",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("single-Span membership must not be reconciled twice")
        ),
    )
    reused = scheduler._global_reconcile({}, {"span-0": science})

    assert reused is not None
    assert np.array_equal(reused.final_original_index, canonical.final_original_index)
    assert reused.diagnostics["duplicate_final_original_index"] == 0


def test_batch_tree_editor_isolates_span_delete_and_restores_with_undo(tmp_path: Path) -> None:
    from algorithms.spans.reconcile import reconcile_trees
    from core.config.fast_baseline import FastBaselineConfig

    def one_tree(span_id: int, indices: tuple[int, ...]):
        span = SimpleNamespace(
            line_id=1,
            line_name="Line",
            span_id=span_id,
            span_name=f"S{span_id + 1:03d}",
            start_xyz_absolute_m=(float(span_id * 10), 0.0, 0.0),
            end_xyz_absolute_m=(float((span_id + 1) * 10), 0.0, 0.0),
            length_m=10.0,
            half_width_m=9.0,
        )
        return SimpleNamespace(
            span=span,
            raw_original_index=np.asarray(indices, dtype=np.int64),
            raw_xyz_local_m=np.asarray([[float(item), 0.0, 10.0] for item in indices]),
            raw_tree_id=np.zeros(len(indices), dtype=np.int32),
            in_core=np.ones(len(indices), dtype=bool),
            tree_records=(
                {
                    "tree_id": 0,
                    # Production reconciliation may persist apex XY without a
                    # Z element; editor geometry refresh must use membership
                    # XYZ rather than indexing a nonexistent apex[2].
                    "apex_xy_local_m": [float(span_id * 10 + 5), 0.0],
                    "apex_height_m": 10.0,
                },
            ),
        )

    reconciled = reconcile_trees(
        [one_tree(0, (1, 2, 3)), one_tree(1, (11, 12, 13))],
        FastBaselineConfig(),
        origin_xyz=(0.0, 0.0, 0.0),
    )
    result = BatchAnalysisResult(
        analysis_run_id="analysis-editor-fixture",
        project_id="project-editor-fixture",
        span_statuses={
            "span-a": BatchSpanStatus.AUTO_ANALYZED,
            "span-b": BatchSpanStatus.AUTO_ANALYZED,
        },
        scan_id="scan-editor-fixture",
        status="COMPLETE",
        span_results={},
        telemetry=BatchTelemetry(span_count=2),
        risk_policy_snapshot={
            "risk_policy_version": "risk-policy-v1",
            "warning_clearance_threshold_m": 0.5,
        },
        global_reconciled=reconciled,
    )
    editor = BatchTreeEditor(
        tmp_path,
        result,
        tree_ids_by_span={"span-a": (0,), "span-b": (1,)},
    )

    editor.delete("span-a", [0])
    assert editor.session.current_tree_ids() == [1]
    assert np.array_equal(editor.original.members_of(1), np.asarray([11, 12, 13]))
    editor.undo()
    assert editor.session.current_tree_ids() == [0, 1]
    editor.persist()
    edit_path = (
        tmp_path
        / "artifacts/project-runs/batches/analysis-editor-fixture/tree_edits.json"
    )
    final_path = (
        tmp_path
        / "artifacts/project-runs/batches/analysis-editor-fixture/tree_segmentation_final.npz"
    )
    assert edit_path.is_file()
    assert final_path.is_file()
    editor.mark_reviewed()
    finalized = editor.finalize()
    assert finalized.span_statuses["span-a"] is BatchSpanStatus.FINALIZED
    assert finalized.global_reconciled.diagnostics["editor_schema"] == (
        "treeline-stage-b-batch-editor-v1"
    )
    output = export_batch_prediction(
        finalized,
        tmp_path / "Prediction_Input.xlsx",
        final_reconciled=finalized.global_reconciled,
    )
    assert output.is_file()
