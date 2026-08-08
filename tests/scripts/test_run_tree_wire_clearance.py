"""CLI tests for the tree-wire clearance computation script."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.run_tree_wire_clearance as cli
from core.contracts.tree_wire_clearance import (
    TreeWireClearanceResult,
    default_tree_wire_clearance_config,
)

from tests.vegetation.test_tree_wire_clearance import (
    _make_v4_artifact,
    _wire_segment,
    _write_a4_w1,
)


def _run(*argv: str) -> int:
    return cli.main(list(argv))


def _setup(tmp_path: Path, **kwargs):
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [
            (0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0))),
            (0, 1, _wire_segment((5.0, 2.0, 3.0), (13.0, 2.0, 3.0))),
        ],
        **kwargs,
    )
    return v4_root, w1_npz, w1_json, project_root


def test_success_publishes_result_json(tmp_path: Path) -> None:
    v4_root, w1_npz, w1_json, project_root = _setup(tmp_path)
    out_json = tmp_path / "clearance.json"
    code = _run(
        "--v4-root", str(v4_root),
        "--w1-npz", str(w1_npz),
        "--w1-json", str(w1_json),
        "--project-root", str(project_root),
        "--output-json", str(out_json),
    )
    assert code == 0
    assert out_json.is_file()
    result = TreeWireClearanceResult.from_dict(
        json.loads(out_json.read_text(encoding="utf-8"))
    )
    assert result.status.name == "VALID"
    assert len(result.records) == 2
    assert result.v4_result_digest


def test_success_with_reviewed_config(tmp_path: Path) -> None:
    v4_root, w1_npz, w1_json, project_root = _setup(tmp_path)
    config_json = tmp_path / "config.json"
    config_json.write_text(
        json.dumps(default_tree_wire_clearance_config(bvh_leaf_cap=4).to_dict()),
        encoding="utf-8",
    )
    out_json = tmp_path / "clearance.json"
    code = _run(
        "--v4-root", str(v4_root),
        "--w1-npz", str(w1_npz),
        "--w1-json", str(w1_json),
        "--project-root", str(project_root),
        "--output-json", str(out_json),
        "--config-json", str(config_json),
    )
    assert code == 0
    result = TreeWireClearanceResult.from_dict(
        json.loads(out_json.read_text(encoding="utf-8"))
    )
    assert result.config_digest == default_tree_wire_clearance_config(bvh_leaf_cap=4).config_digest


def test_origin_mismatch_fails_closed(tmp_path: Path) -> None:
    v4_root, w1_npz, w1_json, _ = _setup(tmp_path, origin=(500_000.0, 1.0, 1.0))
    out_json = tmp_path / "clearance.json"
    code = _run(
        "--v4-root", str(v4_root),
        "--w1-npz", str(w1_npz),
        "--w1-json", str(w1_json),
        "--project-root", str(tmp_path / "project"),
        "--output-json", str(out_json),
    )
    assert code == 1
    assert not out_json.exists()


def test_invalid_config_fails_closed(tmp_path: Path) -> None:
    v4_root, w1_npz, w1_json, project_root = _setup(tmp_path)
    bad_config = tmp_path / "bad.json"
    bad_config.write_text("{", encoding="utf-8")
    out_json = tmp_path / "clearance.json"
    code = _run(
        "--v4-root", str(v4_root),
        "--w1-npz", str(w1_npz),
        "--w1-json", str(w1_json),
        "--project-root", str(project_root),
        "--output-json", str(out_json),
        "--config-json", str(bad_config),
    )
    assert code == 1
    assert not out_json.exists()


def test_output_exists_refused_without_overwrite(tmp_path: Path) -> None:
    v4_root, w1_npz, w1_json, project_root = _setup(tmp_path)
    out_json = tmp_path / "clearance.json"
    out_json.write_text("old", encoding="utf-8")
    code = _run(
        "--v4-root", str(v4_root),
        "--w1-npz", str(w1_npz),
        "--w1-json", str(w1_json),
        "--project-root", str(project_root),
        "--output-json", str(out_json),
    )
    assert code == 1
    assert out_json.read_text(encoding="utf-8") == "old"
    code = _run(
        "--v4-root", str(v4_root),
        "--w1-npz", str(w1_npz),
        "--w1-json", str(w1_json),
        "--project-root", str(project_root),
        "--output-json", str(out_json),
        "--overwrite",
    )
    assert code == 0


def test_missing_project_root_fails_closed(tmp_path: Path) -> None:
    v4_root, w1_npz, w1_json, _ = _setup(tmp_path)
    out_json = tmp_path / "clearance.json"
    code = _run(
        "--v4-root", str(v4_root),
        "--w1-npz", str(w1_npz),
        "--w1-json", str(w1_json),
        "--project-root", str(tmp_path / "missing"),
        "--output-json", str(out_json),
    )
    assert code == 1
    assert not out_json.exists()
