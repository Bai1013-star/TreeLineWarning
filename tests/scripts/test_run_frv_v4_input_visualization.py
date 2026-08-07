"""FRV-VIS-A1 CLI tests (PLAN-FROZEN)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.run_frv_v4_input_visualization as runner
from algorithms.vegetation import build_v4_input_handoff

from tests.vegetation.test_frv_membership import run_membership_ready


@pytest.fixture(scope="module")
def ready(tmp_path_factory: pytest.TempPathFactory):
    root = Path(tmp_path_factory.mktemp("vis_cli"))
    packed = run_membership_ready(root)
    result, las_path, _col, _ground, reference, source_result, configuration, *_ = packed
    v4_root = root / "v4out"
    build_v4_input_handoff(
        reference,
        source_result,
        result,
        frv_configuration=configuration,
        output_dir=v4_root,
        source_chunk_size=40,
    )
    return result, las_path, v4_root


def _argv(ready, tmp_path: Path, **kwargs) -> list[str]:
    result, las_path, v4_root = ready
    argv = [
        "--source",
        str(las_path),
        "--frv-artifact-root",
        str(result.output_reference.root_path),
        "--v4-input-root",
        str(v4_root),
        "--output",
        str(tmp_path / "out.html"),
        "--summary-json",
        str(tmp_path / "out.json"),
        "--review-checklist",
        str(tmp_path / "out.md"),
    ]
    for key, value in kwargs.items():
        if value is None:
            continue
        dashed = key.replace("_", "-")
        if key in ("show_original_background", "show_excluded", "show_review"):
            if value:
                argv.append(f"--{dashed}")
            continue
        argv.extend([f"--{dashed}", str(value)])
    return argv


def test_parse_args_accepts_all_arguments(ready, tmp_path: Path) -> None:
    result, las_path, v4_root = ready
    args = runner.parse_args(
        [
            "--source",
            str(las_path),
            "--frv-artifact-root",
            str(result.output_reference.root_path),
            "--v4-input-root",
            str(v4_root),
            "--output",
            str(tmp_path / "out.html"),
            "--max-total-points",
            "300000",
            "--max-points-per-layer",
            "90000",
            "--max-background-points",
            "40000",
            "--sampling-seed",
            "7",
            "--show-original-background",
            "--show-excluded",
        ]
    )
    assert args.source == las_path
    assert args.budget.max_total_points == 300000
    assert args.budget.max_points_per_layer == 90000
    assert args.budget.max_background_points == 40000
    assert args.budget.sampling_seed == 7
    assert args.show_original_background is True
    assert args.show_excluded is True
    assert args.show_review is False


def test_scene_simple_alias_resolves_source() -> None:
    args = runner.parse_args(
        [
            "--scene-simple",
            "--frv-artifact-root",
            "f",
            "--v4-input-root",
            "v",
            "--output",
            "o.html",
        ]
    )
    assert args.source == runner._SCENE_SIMPLE_SOURCE
    assert runner._SCENE_SIMPLE_SOURCE.name == "scene_simple.las"
    assert args.budget.max_total_points == 250000


def test_source_and_scene_simple_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as excinfo:
        runner.parse_args(
            [
                "--source",
                "s.las",
                "--scene-simple",
                "--frv-artifact-root",
                "f",
                "--v4-input-root",
                "v",
                "--output",
                "o.html",
            ]
        )
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--max-total-points", "0"], "max_total_points"),
        (["--max-total-points", "500001"], "max_total_points"),
        (["--max-points-per-layer", "0"], "max_points_per_layer"),
        (["--max-points-per-layer", "100001"], "max_points_per_layer"),
        (["--max-background-points", "0"], "max_background_points"),
        (["--max-background-points", "100001"], "max_background_points"),
        (["--sampling-seed", "-1"], "sampling_seed"),
        (["--sampling-seed", "abc"], "numeric"),
    ],
)
def test_parse_args_rejects_invalid_budget(
    ready, tmp_path: Path, extra, match
) -> None:
    result, las_path, v4_root = ready
    base = [
        "--source",
        str(las_path),
        "--frv-artifact-root",
        str(result.output_reference.root_path),
        "--v4-input-root",
        str(v4_root),
        "--output",
        str(tmp_path / "out.html"),
    ]
    with pytest.raises(runner.FrvV4CliError, match=match):
        runner.parse_args(base + extra)


def test_main_returns_zero_and_writes_all_outputs(
    ready, tmp_path: Path, capsys
) -> None:
    code = runner.main(_argv(ready, tmp_path))
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert "html:" in captured.out
    assert "summary_json:" in captured.out
    assert "review_checklist:" in captured.out
    assert (tmp_path / "out.html").is_file()
    assert (tmp_path / "out.json").is_file()
    assert (tmp_path / "out.md").is_file()
    summary = json.loads((tmp_path / "out.json").read_text("utf-8"))
    assert summary["mode"] == "ready"


def test_main_reports_missing_artifact_concisely(
    ready, tmp_path: Path, capsys
) -> None:
    result, las_path, v4_root = ready
    argv = [
        "--source",
        str(las_path),
        "--frv-artifact-root",
        str(tmp_path / "missing_frv"),
        "--v4-input-root",
        str(v4_root),
        "--output",
        str(tmp_path / "out.html"),
    ]
    code = runner.main(argv)
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.startswith("FRV-VIS-A1 visualization failed:")
    assert "Traceback" not in captured.err


def test_main_rejects_existing_output(ready, tmp_path: Path, capsys) -> None:
    (tmp_path / "out.html").write_text("existing", "utf-8")
    code = runner.main(_argv(ready, tmp_path))
    captured = capsys.readouterr()
    assert code == 2
    assert "refusing to overwrite" in captured.err


def test_main_rejects_invalid_budget(ready, tmp_path: Path, capsys) -> None:
    code = runner.main(_argv(ready, tmp_path, max_total_points="0"))
    captured = capsys.readouterr()
    assert code == 2
    assert "max_total_points" in captured.err
    assert not (tmp_path / "out.html").exists()


def test_main_scene_simple_fails_cleanly_without_roots(
    tmp_path: Path, capsys
) -> None:
    code = runner.main(
        [
            "--scene-simple",
            "--frv-artifact-root",
            str(tmp_path / "missing_frv"),
            "--v4-input-root",
            str(tmp_path / "missing_v4"),
            "--output",
            str(tmp_path / "out.html"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.startswith("FRV-VIS-A1 visualization failed:")
    assert "Traceback" not in captured.err
