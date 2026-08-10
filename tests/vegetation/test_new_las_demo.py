"""Focused safety tests for the Windows one-click new-LAS route."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import laspy
import numpy as np
import pytest
from algorithms.vegetation.fast_weighted_preprocess import (
    backmap_full_working_set_tree_ids,
    build_weighted_working_set,
)
from algorithms.vegetation.new_las_demo import (
    blocked_clearance_result,
    choose_representative_voxel_size,
    create_isolated_run_dir,
    discover_demo_inputs,
    new_source_id,
    write_summary_html,
)
from scripts.run_new_las_demo import (
    _NO_INPUT_MESSAGE,
    configure_windows_console,
    open_latest_results,
    parse_args,
    run_cli,
    select_input,
)


def _write_las(path: Path, xyz: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.asarray((0.001, 0.001, 0.001))
    header.offsets = np.zeros(3, dtype=np.float64)
    data = laspy.LasData(header)
    data.x, data.y, data.z = xyz.T
    data.classification = np.zeros(len(xyz), dtype=np.uint8)
    data.write(path)


def test_input_discovery_handles_zero_one_and_multiple_without_arbitrary_choice(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input_las"
    input_dir.mkdir()
    assert discover_demo_inputs(input_dir) == []
    assert select_input(input_dir, interactive=False) is None
    assert (tmp_path / "unused").exists() is False

    first = input_dir / "B.LAS"
    first.write_bytes(b"placeholder")
    assert discover_demo_inputs(input_dir) == [first]
    assert select_input(input_dir, interactive=False) == first

    second = input_dir / "a.laz"
    second.write_bytes(b"placeholder")
    assert discover_demo_inputs(input_dir) == [second, first]
    with pytest.raises(ValueError, match="多个 LAS/LAZ"):
        select_input(input_dir, interactive=False)


def test_input_discovery_interactively_uses_the_owner_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input_las"
    input_dir.mkdir()
    first = input_dir / "first.las"
    second = input_dir / "second.laz"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")
    assert select_input(input_dir) == second


def test_launcher_environment_overrides_are_for_controlled_smoke_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input with spaces"
    runs_root = tmp_path / "runs with spaces"
    monkeypatch.setenv("TREE_LINE_DEMO_INPUT_DIR", str(input_dir))
    monkeypatch.setenv("TREE_LINE_DEMO_RUNS_ROOT", str(runs_root))
    args = parse_args([])
    assert args.input_dir == input_dir
    assert args.runs_root == runs_root


def test_no_input_cli_prints_owner_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(["--input-dir", str(tmp_path), "--no-open-results"]) == 2
    assert _NO_INPUT_MESSAGE in capsys.readouterr().out


def test_multiple_input_cli_reports_a_clean_actionable_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "first.las").write_bytes(b"first")
    (tmp_path / "second.laz").write_bytes(b"second")
    assert run_cli(["--input-dir", str(tmp_path), "--non-interactive", "--no-open-results"]) == 1
    output = capsys.readouterr().out
    assert "多个 LAS/LAZ" in output
    assert "Traceback" not in output


def test_failure_reporting_keeps_a_traceback_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "empty.las"
    laspy.LasData(laspy.LasHeader(point_format=7, version="1.4")).write(source)
    runs_root = tmp_path / "runs"
    assert (
        run_cli(
            [
                "--input",
                str(source),
                "--runs-root",
                str(runs_root),
                "--no-open-results",
            ]
        )
        == 1
    )
    text = capsys.readouterr().out
    assert "处理失败" in text
    assert "失败阶段" in text
    logs = list(runs_root.glob("*/run.log"))
    assert len(logs) == 1
    assert "NEW_LAS_DEMO_EMPTY_SOURCE" in logs[0].read_text("utf-8")


def test_output_isolation_and_new_source_identity(tmp_path: Path) -> None:
    stamp = datetime(2026, 8, 10, 23, 59, 58)
    source = tmp_path / "新树.las"
    source.write_bytes(b"new source")
    first = create_isolated_run_dir(tmp_path / "runs", source, timestamp=stamp)
    second = create_isolated_run_dir(tmp_path / "runs", source, timestamp=stamp)
    assert first.name == "新树_20260810_235958"
    assert second.name == "新树_20260810_235958_02"
    assert new_source_id("a" * 64) == "new_las_aaaaaaaaaaaaaaaa"
    assert new_source_id("a" * 64) != "scan_small"
    with pytest.raises(Exception, match="SOURCE_SHA"):
        new_source_id("not-a-sha")


def test_representative_voxel_size_adapts_without_losing_the_normal_default() -> None:
    assert choose_representative_voxel_size(1) == 0.25
    assert choose_representative_voxel_size(5_000_000) == 0.25
    assert choose_representative_voxel_size(5_000_001) == 0.35
    assert choose_representative_voxel_size(20_000_001) == 0.5
    assert choose_representative_voxel_size(50_000_001) == 0.75


def test_full_resolution_backmap_preserves_every_source_row(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    _write_las(
        source,
        np.asarray(
            ((0.01, 0.01, 0.0), (0.10, 0.10, 0.1), (0.60, 0.1, 0.2),
             (0.61, 0.1, 0.3), (1.10, 0.1, 0.4)),
            dtype=np.float64,
        ),
    )
    _status, working = build_weighted_working_set(source, tmp_path / "working", voxel_size_m=0.5)
    result = backmap_full_working_set_tree_ids(
        working,
        np.asarray((0, 2, 4), dtype=np.int64),
        np.asarray((7, -1, 8), dtype=np.int32),
        tmp_path / "backmap" / "representative_backmap.npz",
    )
    raw_tree = np.load(result["full_resolution_tree_id_by_source_row_path"], mmap_mode="r")
    assert raw_tree.tolist() == [7, 7, -1, -1, 8]
    assert result["assigned_full_resolution_points"] == 3
    with np.load(result["representative_mapping_path"], allow_pickle=False) as stream:
        assert stream["tree_id_by_representative"].tolist() == [7, -1, 8]


def test_blocked_clearance_never_reads_an_old_w1_and_summary_is_written(tmp_path: Path) -> None:
    old_w1 = tmp_path / "old_scan_small_w1.npz"
    old_w1.write_bytes(b"stale")
    clearance = blocked_clearance_result(tmp_path / "clearance", reason="没有新源 W1")
    assert clearance["TREE_WIRE_STATUS"] == "BLOCKED_NO_VALID_W1"
    clearance_json = tmp_path / "clearance" / "tree_wire_clearance.json"
    assert old_w1.name not in clearance_json.read_text("utf-8")
    summary = {
        "TREE_WIRE_STATUS": clearance["TREE_WIRE_STATUS"],
        "TREE_COUNT": 0,
        "NON_CERTIFIED": True,
    }
    output_html = tmp_path / "report" / "RUN_SUMMARY.html"
    write_summary_html(summary, tmp_path, output_html)
    assert output_html.is_file()
    assert "BLOCKED_NO_VALID_W1" in output_html.read_text("utf-8")


def test_batch_launchers_are_ascii_adapters_with_windows_paths_quoted() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "RUN_NEW_LAS_DEMO.bat").read_text("utf-8")
    latest = (root / "OPEN_LATEST_RESULTS.bat").read_text("utf-8")
    assert launcher.isascii()
    assert latest.isascii()
    assert 'set "REPO_ROOT=%~dp0"' in launcher
    assert '"%PYTHON_EXE%" "%REPO_ROOT%scripts\\run_new_las_demo.py"' in launcher
    latest_command = '"%PYTHON_EXE%" "%REPO_ROOT%scripts\\run_new_las_demo.py"'
    assert f"{latest_command} --open-latest-results" in latest
    assert "(" not in launcher
    assert "(" not in latest


def test_open_latest_results_uses_only_completed_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    completed = tmp_path / "complete"
    summary = completed / "07_report" / "RUN_SUMMARY.html"
    summary.parent.mkdir(parents=True)
    summary.write_text("complete", encoding="utf-8")
    monkeypatch.setenv("TREE_LINE_DEMO_NO_OPEN_RESULTS", "1")
    assert open_latest_results(tmp_path) == 0
    assert str(summary) in capsys.readouterr().out


def test_console_configuration_requests_utf8_when_supported() -> None:
    configure_windows_console()
