"""Integration checks for the current bounded thread based batch scheduler."""

from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError
from functools import partial

import pytest
from core.config.fast_baseline import FastBaselineConfig
from core.stage_b import batch
from core.stage_b.batch import BatchAnalysisScheduler, BatchSpanStatus
from core.stage_b.export import export_batch_prediction

from tests.stage_b.spawn_batch_fixtures import fixture_executor, prepared_fixture
from tests.stage_b.test_batch_exact_cache import (
    _assert_batch_results_equal,
    _project,
    _project_snapshot,
)


def _scheduler(root, project, scan_id, *, mode="PARALLEL", runner=None):
    return BatchAnalysisScheduler(
        root,
        project,
        scan_id,
        execution_mode=mode,
        science_runner=runner or partial(fixture_executor, config=FastBaselineConfig()),
        source_preparer=prepared_fixture,
    )


def _enable_workers(monkeypatch, count=4):
    monkeypatch.setattr(batch, "_available_memory_bytes", lambda: 4 * 1024**3)
    monkeypatch.setattr(batch.os, "cpu_count", lambda: count)


def _export_rows(scheduler, result, output):
    from openpyxl import load_workbook

    editor = scheduler.open_editor(result)
    editor.mark_reviewed()
    export_batch_prediction(editor.finalize(), output)
    book = load_workbook(output, read_only=True, data_only=False)
    try:
        rows = {sheet.title: list(sheet.values) for sheet in book.worksheets}
        column = rows["Trees"][0].index("science_run_id")
        for index, row in enumerate(rows["Trees"][1:], start=1):
            assert row[column] == result.analysis_run_id
            values = list(row)
            values[column] = "<control-run>"
            rows["Trees"][index] = tuple(values)
        for index, row in enumerate(rows["Run_Metadata"]):
            if row[0] == "science_run_id":
                assert row[1] == result.analysis_run_id
                rows["Run_Metadata"][index] = (row[0], "<control-run>", *row[2:])
        return rows
    finally:
        book.close()


@pytest.mark.parametrize("span_count", [2, 3])
def test_multiworker_overlap_order_independent_restart_and_export(
    tmp_path, monkeypatch, span_count
):
    _enable_workers(monkeypatch)
    project, scan_id, span_ids = _project(tmp_path, span_count=span_count)
    serial_scheduler = _scheduler(tmp_path, project, scan_id, mode="SERIAL")
    serial = serial_scheduler.run(span_ids, analysis_run_id="serial-control")
    assert serial.status == "COMPLETE"

    barrier = threading.Barrier(span_count)
    completed = []
    worker_ids = set()
    lock = threading.Lock()

    def runner(input_data, preparation, output_root):
        with lock:
            worker_ids.add(threading.get_ident())
        barrier.wait(timeout=10)
        time.sleep(0.12 * (span_count - int(input_data.span_id[1:])))
        result = fixture_executor(
            input_data, preparation, output_root, config=FastBaselineConfig()
        )
        with lock:
            completed.append(input_data.span_id)
        return result

    scheduler = _scheduler(tmp_path, project, scan_id, runner=runner)
    result = scheduler.run(span_ids, analysis_run_id="thread-control")
    assert result.status == "COMPLETE", result.to_dict()
    plan = scheduler._last_execution_plan
    assert plan["execution_effective"] == "PARALLEL"
    assert plan["cpu_producer_workers"] == span_count
    assert len(worker_ids) == span_count
    assert completed == [f"S{index:03d}" for index in range(span_count, 0, -1)]
    assert tuple(result.span_results) == span_ids
    _assert_batch_results_equal(serial, result, span_ids)

    before = _project_snapshot(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("complete restart must not rerun science")

    resumed = _scheduler(tmp_path, project, scan_id, runner=forbidden).run(
        tuple(reversed(span_ids)), analysis_run_id="thread-control"
    )
    assert resumed.status == "COMPLETE"
    assert _project_snapshot(tmp_path) == before
    _assert_batch_results_equal(result, resumed, span_ids)
    # Synthetic cache fixtures contain no registered-tower raw region.
    monkeypatch.setattr(
        "core.stage_b.editor.ensure_editor_tower_exclusion", lambda *_args: None
    )
    serial_rows = _export_rows(serial_scheduler, serial, tmp_path / "serial.xlsx")
    parallel_rows = _export_rows(scheduler, result, tmp_path / "parallel.xlsx")
    assert serial_rows["Trees"] == parallel_rows["Trees"]


def test_worker_plan_bounds_cpu_memory_and_metadata(monkeypatch):
    monkeypatch.setattr(batch.os, "cpu_count", lambda: 8)
    gib = 1024**3
    assert batch._parallel_worker_plan(3, 100, 128 * 1024**2)["workers"] == 1
    assert batch._parallel_worker_plan(3, 100, 4 * gib)["workers"] == 3
    plan = batch._parallel_worker_plan(
        3, 100, 4 * gib, maximum_planned_role_rows=100_000_000
    )
    assert plan["workers"] == 1
    assert plan["estimate_basis"] == "MAXIMUM_METADATA_PLANNED_ROLE_ROWS"
    monkeypatch.setattr(batch.os, "cpu_count", lambda: 2)
    assert batch._parallel_worker_plan(3, 100, 4 * gib)["workers"] == 2


@pytest.mark.parametrize("error_type", [RuntimeError, CancelledError])
def test_parallel_failure_or_cancel_preserves_siblings_and_retry(
    tmp_path, monkeypatch, error_type
):
    _enable_workers(monkeypatch)
    project, scan_id, span_ids = _project(tmp_path, span_count=3)
    barrier = threading.Barrier(3)

    def fail_one(input_data, preparation, output_root):
        barrier.wait(timeout=10)
        if input_data.span_id == "S002":
            raise error_type("FIXTURE_TREE_FAILURE: original science error")
        return fixture_executor(
            input_data, preparation, output_root, config=FastBaselineConfig()
        )

    first = _scheduler(tmp_path, project, scan_id, runner=fail_one).run(
        span_ids, analysis_run_id="mixed"
    )
    assert first.status == "PARTIAL_FAILURE"
    assert first.span_statuses[span_ids[1]] is BatchSpanStatus.FAILED
    failure = first.span_results[span_ids[1]].diagnostics
    assert "original science error" in failure["technical_message"]
    batch_root = tmp_path / "artifacts/project-runs/batches/mixed"
    preserved = {
        span: _project_snapshot(batch_root / span) for span in (span_ids[0], span_ids[2])
    }
    retry_calls = []

    def retry(input_data, preparation, output_root):
        retry_calls.append(input_data.span_id)
        return fixture_executor(
            input_data, preparation, output_root, config=FastBaselineConfig()
        )

    second = _scheduler(tmp_path, project, scan_id, runner=retry).run(
        span_ids, analysis_run_id="mixed"
    )
    assert second.status == "COMPLETE"
    assert retry_calls == ["S002"]
    for span, snapshot in preserved.items():
        assert _project_snapshot(batch_root / span) == snapshot


def test_parallel_publication_is_serial_and_ordered(tmp_path, monkeypatch):
    _enable_workers(monkeypatch)
    project, scan_id, span_ids = _project(tmp_path, span_count=3)
    main_thread = threading.get_ident()
    publish_threads = []
    real_finish = BatchAnalysisScheduler._finish

    def observe_finish(self, *args, **kwargs):
        publish_threads.append(threading.get_ident())
        return real_finish(self, *args, **kwargs)

    monkeypatch.setattr(BatchAnalysisScheduler, "_finish", observe_finish)
    result = _scheduler(tmp_path, project, scan_id).run(
        tuple(reversed(span_ids)), analysis_run_id="parent-write"
    )
    assert result.status == "COMPLETE"
    assert publish_threads == [main_thread]
    assert tuple(result.span_results) == span_ids
    assert (tmp_path / "artifacts/project-runs/batches/parent-write").is_dir()
