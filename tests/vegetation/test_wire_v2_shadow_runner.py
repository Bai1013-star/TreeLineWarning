"""Publication safety tests for non-authoritative V2 wire shadow execution."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_wire_v2_project_shadow as project_shadow
from scripts import run_wire_v2_shadow as shadow
from tests.vegetation.test_fast_baseline_wire import _span


def test_failed_shadow_execution_never_publishes_final_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_dir = tmp_path / "S001"
    monkeypatch.setattr(shadow, "_load_wire_region", lambda _: object())

    def fail_pipeline(*_: object, **__: object) -> object:
        raise ValueError("intentional pipeline failure")

    monkeypatch.setattr(shadow, "run_span_wire_pipeline", fail_pipeline)
    with pytest.raises(shadow.ShadowExecutionError, match="intentional pipeline failure") as raised:
        shadow.execute_shadow_run(
            run_root=tmp_path / "source",
            final_output_dir=final_dir,
            span=_span(),
            origin_xyz=(0.0, 0.0, 0.0),
            authoritative_source_point_count=100,
        )
    assert not final_dir.exists()
    failure = raised.value.temporary_dir / "execution_failure.json"
    assert failure.is_file()
    assert "intentional pipeline failure" in failure.read_text(encoding="utf-8")


def test_complete_shadow_execution_is_published_only_after_required_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_dir = tmp_path / "S001"
    monkeypatch.setattr(shadow, "_load_wire_region", lambda _: object())

    def fake_pipeline(*_: object, output_dir: Path, **__: object) -> object:
        (output_dir / "span_wire_result.json").write_text("{}", encoding="utf-8")
        (output_dir / "fine_representatives.npz").write_bytes(b"representatives")
        (output_dir / "w1_arrays.npz").write_bytes(b"w1")
        (output_dir / "wire_v2_track_evidence.json").write_text("{}", encoding="utf-8")
        (output_dir / "wire_v2_track_evidence.npz").write_bytes(b"evidence")
        np.savez(
            output_dir / "span_wire_diagnostics.npz",
            confirmed_member_source_index=np.asarray([3, 7], dtype=np.int64),
        )
        return SimpleNamespace(
            strand_count=0,
            distance_ready_segments=1,
            diagnostics={
                "v2_tracks": [],
                "diagnostics_npz_path": "staging/span_wire_diagnostics.npz",
                "fine_representatives_npz_path": "staging/fine_representatives.npz",
            },
        )

    monkeypatch.setattr(shadow, "run_span_wire_pipeline", fake_pipeline)
    monkeypatch.setattr(
        shadow,
        "_exact_shadow_clearance",
        lambda *_: {"status": "VALID", "distance_ready_edges": 69, "records": []},
    )
    summary = shadow.execute_shadow_run(
        run_root=tmp_path / "source",
        final_output_dir=final_dir,
        span=_span(),
        origin_xyz=(0.0, 0.0, 0.0),
        authoritative_source_point_count=100,
    )
    assert summary["execution_status"] == "COMPLETE"
    assert summary["distance_ready_segment_count"] == 1
    assert summary["distance_ready_polyline_edge_count"] == 69
    assert summary["diagnostics"]["diagnostics_npz_path"] == str(
        final_dir / "span_wire_diagnostics.npz"
    )
    assert (final_dir / "shadow_summary.json").is_file()
    assert not list(tmp_path.glob(".S001.incomplete-*"))


def test_batch_records_a_span_failure_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary = tmp_path / ".S001.incomplete-test"
    temporary.mkdir()
    (tmp_path / "clearance.json").write_text(
        '{"diagnostics": {"distance_ready_segments": 0}, "global_minimum_clearance_m": null}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        project_shadow, "_span_context", lambda *_: (tmp_path, _span(), (0.0, 0.0, 0.0))
    )
    monkeypatch.setattr(project_shadow, "_source_point_count", lambda _: 100)
    monkeypatch.setattr(
        project_shadow,
        "execute_shadow_run",
        lambda **_: (_ for _ in ()).throw(
            shadow.ShadowExecutionError("S001", temporary, ValueError("intentional batch failure"))
        ),
    )
    record = {"span_id": "S001", "active_run_id": "not-read"}
    outcome = project_shadow._shadow_span(tmp_path, tmp_path / "out", "G001", record)
    assert outcome["execution_status"] == "FAILED"
    assert outcome["failure"]["exception_type"] == "ValueError"
    assert "intentional batch failure" in outcome["failure"]["exception_message"]


def test_project_progress_is_not_reported_complete_with_pending_spans() -> None:
    assert (
        project_shadow._project_summary_status(
            [
                {"span_id": "S001", "execution_status": "COMPLETE"},
                {"span_id": "S002", "execution_status": "PENDING"},
            ]
        )
        == "RUNNING"
    )
    assert (
        project_shadow._project_summary_status(
            [
                {"span_id": "S001", "execution_status": "COMPLETE"},
                {"span_id": "S002", "execution_status": "FAILED"},
            ]
        )
        == "FAILED"
    )
