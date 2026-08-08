"""CLI tests for the tree-wire clearance review visualization script."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.run_tree_wire_clearance as clearance_cli
import scripts.run_tree_wire_clearance_visualization as vis_cli

from tests.vegetation.test_tree_wire_clearance import (
    _make_v4_artifact,
    _wire_segment,
    _write_a4_w1,
)


def _setup(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [
            (0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0))),
            (0, 1, _wire_segment((5.0, 2.0, 3.0), (13.0, 2.0, 3.0))),
        ],
    )
    clearance_json = tmp_path / "clearance.json"
    code = clearance_cli.main(
        [
            "--v4-root", str(v4_root),
            "--w1-npz", str(w1_npz),
            "--w1-json", str(w1_json),
            "--project-root", str(project_root),
            "--output-json", str(clearance_json),
        ]
    )
    assert code == 0
    return clearance_json, v4_root, w1_npz, w1_json


def _has_plotly(path: Path) -> bool:
    return "plotly" in path.read_text(encoding="utf-8")[:100_000].lower()


def test_success_publishes_html_and_summary(tmp_path: Path) -> None:
    clearance_json, v4_root, w1_npz, w1_json = _setup(tmp_path)
    out_html = tmp_path / "out.html"
    out_json = tmp_path / "out.json"
    code = vis_cli.main(
        [
            "--clearance-json", str(clearance_json),
            "--v4-root", str(v4_root),
            "--w1-npz", str(w1_npz),
            "--w1-json", str(w1_json),
            "--output-html", str(out_html),
            "--summary-json", str(out_json),
        ]
    )
    assert code == 0
    assert out_html.is_file()
    assert out_json.is_file()
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["tool"] == "tree-wire-clearance-visualization"
    assert summary["status"] == "VALID"
    assert _has_plotly(out_html)


def test_invalid_clearance_json_fails_closed(tmp_path: Path) -> None:
    _clearance_json, v4_root, w1_npz, w1_json = _setup(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    code = vis_cli.main(
        [
            "--clearance-json", str(bad),
            "--v4-root", str(v4_root),
            "--w1-npz", str(w1_npz),
            "--w1-json", str(w1_json),
            "--output-html", str(tmp_path / "out.html"),
        ]
    )
    assert code == 1
    assert not (tmp_path / "out.html").exists()


def test_v4_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    clearance_json, _v4_root, w1_npz, w1_json = _setup(tmp_path)
    from algorithms.vegetation.v4_tree_artifacts import publish_v4_tree_segmentation_artifact
    from algorithms.vegetation.v4_tree_segmentation import segment_v4_trees

    from tests.vegetation.test_v4_tree_segmentation import (
        make_config,
        make_handoff,
        make_tree_points,
    )

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_v4 = other_dir / "v4"
    handoff = make_handoff(make_tree_points(5.0, 5.0))
    publish_v4_tree_segmentation_artifact(
        segment_v4_trees(handoff, make_config()), other_v4, scratch_dir=other_dir
    )
    code = vis_cli.main(
        [
            "--clearance-json", str(clearance_json),
            "--v4-root", str(other_v4),
            "--w1-npz", str(w1_npz),
            "--w1-json", str(w1_json),
            "--output-html", str(tmp_path / "out.html"),
        ]
    )
    assert code == 1
    assert not (tmp_path / "out.html").exists()
