"""CLI tests for the V4 tree-segmentation review visualization script."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.run_v4_tree_visualization as cli

from tests.vegetation.test_v4_tree_visualization import _make_real_artifact


def _run(*argv: str) -> int:
    return cli.main(list(argv))


def _has_plotly(path: Path) -> bool:
    return "plotly" in path.read_text(encoding="utf-8")[:100_000].lower()


def test_success_publishes_html_and_summary(tmp_path: Path) -> None:
    v4_root = _make_real_artifact(tmp_path)
    out_html = tmp_path / "out.html"
    out_json = tmp_path / "out.json"
    code = _run(
        "--v4-root", str(v4_root),
        "--output-html", str(out_html),
        "--summary-json", str(out_json),
        "--max-points-per-tree", "100",
    )
    assert code == 0
    assert out_html.is_file()
    assert out_json.is_file()
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["tool"] == "v4-tree-visualization"
    assert _has_plotly(out_html)


def test_success_without_summary(tmp_path: Path) -> None:
    v4_root = _make_real_artifact(tmp_path)
    out_html = tmp_path / "out.html"
    code = _run("--v4-root", str(v4_root), "--output-html", str(out_html))
    assert code == 0
    assert out_html.is_file()


def test_missing_v4_root_fails_closed(tmp_path: Path) -> None:
    code = _run(
        "--v4-root", str(tmp_path / "missing"),
        "--output-html", str(tmp_path / "out.html"),
    )
    assert code == 1
    assert not (tmp_path / "out.html").exists()


def test_output_exists_refused_without_overwrite(tmp_path: Path) -> None:
    v4_root = _make_real_artifact(tmp_path)
    out_html = tmp_path / "out.html"
    out_html.write_text("old", encoding="utf-8")
    code = _run("--v4-root", str(v4_root), "--output-html", str(out_html))
    assert code == 1
    assert out_html.read_text(encoding="utf-8") == "old"
    code = _run("--v4-root", str(v4_root), "--output-html", str(out_html), "--overwrite")
    assert code == 0
    assert _has_plotly(out_html)


def test_invalid_budget_fails_closed(tmp_path: Path) -> None:
    v4_root = _make_real_artifact(tmp_path)
    code = _run(
        "--v4-root", str(v4_root),
        "--output-html", str(tmp_path / "out.html"),
        "--max-points-per-tree", "0",
    )
    assert code == 1
    assert not (tmp_path / "out.html").exists()


def test_usage_error_on_missing_required(tmp_path: Path) -> None:
    try:
        _run("--output-html", str(tmp_path / "out.html"))
        raised = False
    except SystemExit:
        raised = True
    assert raised
