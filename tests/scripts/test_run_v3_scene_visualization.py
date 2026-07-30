from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scripts.run_v3_scene_visualization as runner
from algorithms.vegetation.tree_segmentation import build_tree_segmentation_result
from pytest import CaptureFixture, MonkeyPatch

from tests.vegetation.test_tree_segmentation import _pipeline


def _scene() -> tuple[object, object, object, object, object]:
    tree_input, model, candidates, segmentation = _pipeline(
        xy=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        hag=[5.0, 4.0, 4.0, 4.0],
        minimum_peak_hag_m=0.0,
    )
    result = build_tree_segmentation_result(tree_input, model, candidates, segmentation)
    return tree_input, model, candidates, segmentation, result


def _arguments(output_dir: Path, maximum: str = "2") -> list[str]:
    return [
        "--v2-npz",
        "v2.npz",
        "--v2-json",
        "v2.json",
        "--a2a-npz",
        "a2a.npz",
        "--a2a-json",
        "a2a.json",
        "--a2b-npz",
        "a2b.npz",
        "--a2b-json",
        "a2b.json",
        "--output-dir",
        str(output_dir),
        "--max-3d-points",
        maximum,
    ]


def _patch_pipeline(monkeypatch: MonkeyPatch, result: object | None = None) -> None:
    tree_input, model, candidates, segmentation, default_result = _scene()
    monkeypatch.setattr(runner, "load_tree_segmentation_input_set", lambda *args: tree_input)
    monkeypatch.setattr(runner, "build_canopy_height_model", lambda value: model)
    monkeypatch.setattr(runner, "detect_treetop_candidates", lambda value: candidates)
    monkeypatch.setattr(runner, "segment_canopy_regions", lambda *args: segmentation)
    monkeypatch.setattr(
        runner,
        "build_tree_segmentation_result",
        lambda *args: default_result if result is None else result,
    )


def test_parse_args_accepts_required_visualization_arguments(tmp_path: Path) -> None:
    args = runner.parse_args(_arguments(tmp_path, "15"))
    assert args.max_3d_points == 15
    assert args.output_dir == tmp_path


@pytest.mark.parametrize("maximum", ["-1", "0", "1.5", "200001"])
def test_cli_parser_rejects_invalid_hard_cap(tmp_path: Path, maximum: str) -> None:
    with pytest.raises(runner.VisualizationCliError, match="max-3d-points"):
        runner.parse_args(_arguments(tmp_path, maximum))


@pytest.mark.parametrize("maximum", [True, np.bool_(True), 0, -1, 1.5, 200001])
def test_cli_run_rejects_non_cli_cap_types(tmp_path: Path, maximum: object) -> None:
    args = runner.parse_args(_arguments(tmp_path))
    args.max_3d_points = maximum
    with pytest.raises(runner.VisualizationCliError, match="max-3d-points"):
        runner.run(args)


def test_main_reports_missing_artifact_concisely(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    assert runner.main(_arguments(tmp_path)) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("V3 visualization failed:")
    assert "Traceback" not in captured.err


def test_main_reports_incompatible_input_concisely(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    tree_input, _model, _candidates, _segmentation, result = _scene()
    _patch_pipeline(
        monkeypatch, replace(result, source_id=f"{tree_input.identity.source_id}-other")
    )
    assert runner.main(_arguments(tmp_path)) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("V3 visualization failed:")
    assert "source_id" in captured.err
    assert "Traceback" not in captured.err


def test_main_rejects_unsafe_output_target_concisely(capsys: CaptureFixture[str]) -> None:
    unsafe = Path("data") / "raw" / "forbidden-visualization"
    assert runner.main(_arguments(unsafe)) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("V3 visualization failed:")
    assert "output-dir" in captured.err
    assert "Traceback" not in captured.err


def test_main_returns_zero_and_writes_all_outputs(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    _patch_pipeline(monkeypatch)
    assert runner.main(_arguments(tmp_path)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "topdown_png:" in captured.out
    assert {
        "tree_segmentation_topdown.png",
        "tree_segmentation_diagnostic.png",
        "tree_segmentation_3d.html",
        "tree_segmentation_summary.json",
    } == {path.name for path in tmp_path.iterdir()}
