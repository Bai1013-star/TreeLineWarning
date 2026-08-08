"""Tests for the read-only tree-wire clearance review visualization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from algorithms.vegetation.tree_wire_clearance import run_tree_wire_clearance
from algorithms.vegetation.tree_wire_clearance_visualization import (
    TOOL_NAME,
    TreeWireClearanceVisualizationError,
    TreeWireClearanceVisualizationOutputError,
    build_summary,
    load_tree_wire_clearance_result,
    render_html,
    run_tree_wire_clearance_visualization,
    validate_display_budget,
)
from core.contracts.tree_wire_clearance import (
    TreeWireClearanceRecordStatus,
    TreeWireClearanceResult,
    TreeWireClearanceStatus,
)

from tests.vegetation.test_tree_wire_clearance import (
    _make_v4_artifact,
    _wire_segment,
    _write_a4_w1,
)


def _run_and_write_clearance(
    tmp_path: Path,
    segments: list[tuple[int, int, list[tuple[float, float, float]]]],
    *,
    ready: list[bool] | None = None,
) -> tuple[Path, Path, Path, TreeWireClearanceResult]:
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(tmp_path, segments, ready=ready)
    clearance = run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    clearance_json = tmp_path / "clearance.json"
    clearance_json.write_text(json.dumps(clearance.to_dict()), encoding="utf-8")
    return v4_root, w1_npz, w1_json, clearance


def _standard_segments() -> list[tuple[int, int, list[tuple[float, float, float]]]]:
    return [
        (0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0))),
        (0, 1, _wire_segment((5.0, 2.0, 3.0), (13.0, 2.0, 3.0))),
    ]


def test_clearance_visualization_end_to_end(tmp_path: Path) -> None:
    v4_root, w1_npz, w1_json, clearance = _run_and_write_clearance(
        tmp_path, _standard_segments()
    )
    out_html = tmp_path / "out.html"
    out_json = tmp_path / "out.json"
    output = run_tree_wire_clearance_visualization(
        tmp_path / "clearance.json", v4_root, w1_npz, w1_json, out_html, out_json
    )
    assert out_html.is_file()
    assert out_json.is_file()
    html = out_html.read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    assert "clearance segments" in html
    assert "tree 0" in html
    assert "tree 1" in html
    summary = output.summary
    assert summary["tool"] == TOOL_NAME
    assert summary["status"] == TreeWireClearanceStatus.VALID.name
    assert summary["clearance_result_digest"] == clearance.result_digest
    assert summary["displayed_counts"]["clearance_segments"] == 2


def test_summary_uses_exact_stored_values_no_recomputation(tmp_path: Path) -> None:
    v4_root, w1_npz, w1_json, clearance = _run_and_write_clearance(
        tmp_path, _standard_segments()
    )
    output = run_tree_wire_clearance_visualization(
        tmp_path / "clearance.json", v4_root, w1_npz, w1_json, tmp_path / "out.html",
        tmp_path / "out.json",
    )
    summary = json.loads(output.summary_path.read_text(encoding="utf-8"))
    # Summary tree entries must equal the validated record values verbatim.
    records_by_id = {record.tree_id: record for record in clearance.records}
    for tree in summary["trees"]:
        record = records_by_id[tree["tree_id"]]
        assert tree["nearest_wire_id"] == record.nearest_wire_id
        assert tree["minimum_clearance_m"] == record.minimum_clearance_m
        assert tree["nearest_tree_xyz_local_m"] == list(record.nearest_tree_xyz_local_m)
        assert tree["nearest_wire_xyz_local_m"] == list(record.nearest_wire_xyz_local_m)
        assert tree["w1_segment_index"] == record.w1_segment_index
        assert tree["polyline_edge_index"] == record.polyline_edge_index
        assert tree["edge_interpolation_t"] == record.edge_interpolation_t


def test_render_html_contains_clearance_and_wire_layers(tmp_path: Path) -> None:
    v4_root, w1_npz, w1_json, clearance = _run_and_write_clearance(
        tmp_path, _standard_segments()
    )
    html = render_html(
        clearance,
        __import__(
            "algorithms.vegetation.v4_tree_artifacts",
            fromlist=["load_v4_tree_segmentation_artifact"],
        ).load_v4_tree_segmentation_artifact(v4_root).result,
        __import__(
            "algorithms.wires.reconstruct", fromlist=["load_wire_models"]
        ).load_wire_models(w1_npz, w1_json)[1],
        validate_display_budget(5000),
    )
    assert "plotly" in html.lower()
    assert "clearance segments" in html
    assert "wire distance-ready segments" in html


def test_visualization_is_deterministic(tmp_path: Path) -> None:
    v4_root, w1_npz, w1_json, clearance = _run_and_write_clearance(
        tmp_path, _standard_segments()
    )
    out_html = tmp_path / "out.html"
    out_json = tmp_path / "out.json"
    first = run_tree_wire_clearance_visualization(
        tmp_path / "clearance.json", v4_root, w1_npz, w1_json, out_html, out_json
    )
    second = run_tree_wire_clearance_visualization(
        tmp_path / "clearance.json", v4_root, w1_npz, w1_json, out_html, out_json,
        overwrite=True,
    )
    assert first.html_path.read_bytes() == second.html_path.read_bytes()
    assert first.summary == second.summary


def test_no_distance_ready_result_displays(tmp_path: Path) -> None:
    v4_root, w1_npz, w1_json, clearance = _run_and_write_clearance(
        tmp_path, _standard_segments(), ready=[False, False]
    )
    assert clearance.status is TreeWireClearanceStatus.NO_DISTANCE_READY_EDGE
    output = run_tree_wire_clearance_visualization(
        tmp_path / "clearance.json", v4_root, w1_npz, w1_json, tmp_path / "out.html",
        tmp_path / "out.json",
    )
    assert output.summary["status"] == TreeWireClearanceStatus.NO_DISTANCE_READY_EDGE.name
    assert output.summary["displayed_counts"]["clearance_segments"] == 0
    assert all(
        tree["status"] == TreeWireClearanceRecordStatus.NO_DISTANCE_READY_EDGE.name
        for tree in output.summary["trees"]
    )


def test_invalid_clearance_result_refused(tmp_path: Path) -> None:
    v4_root, w1_npz, w1_json, _clearance = _run_and_write_clearance(
        tmp_path, _standard_segments()
    )
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    with pytest.raises(TreeWireClearanceVisualizationError):
        run_tree_wire_clearance_visualization(
            bad_json, v4_root, w1_npz, w1_json, tmp_path / "out.html"
        )
    # Tampered digest must be refused.
    tampered = tmp_path / "tampered.json"
    tampered.write_text(
        json.dumps({**json.loads((tmp_path / "clearance.json").read_text("utf-8")),
                    "result_digest": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(TreeWireClearanceVisualizationError):
        run_tree_wire_clearance_visualization(
            tampered, v4_root, w1_npz, w1_json, tmp_path / "out2.html"
        )


def test_v4_identity_mismatch_refused(tmp_path: Path) -> None:
    _v4_root, w1_npz, w1_json, _clearance = _run_and_write_clearance(
        tmp_path, _standard_segments()
    )
    # A different, unrelated V4 artifact must be refused (single-tree scene).
    from algorithms.vegetation.v4_tree_artifacts import publish_v4_tree_segmentation_artifact
    from algorithms.vegetation.v4_tree_segmentation import segment_v4_trees

    from tests.vegetation.test_v4_tree_segmentation import (
        make_config,
        make_handoff,
        make_tree_points,
    )

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    handoff = make_handoff(make_tree_points(5.0, 5.0))
    other_result = segment_v4_trees(handoff, make_config())
    other_v4 = other_dir / "v4"
    publish_v4_tree_segmentation_artifact(other_result, other_v4, scratch_dir=other_dir)
    with pytest.raises(TreeWireClearanceVisualizationError):
        run_tree_wire_clearance_visualization(
            tmp_path / "clearance.json", other_v4, w1_npz, w1_json, tmp_path / "out.html"
        )


def test_output_overwrite_refusal(tmp_path: Path) -> None:
    v4_root, w1_npz, w1_json, _clearance = _run_and_write_clearance(
        tmp_path, _standard_segments()
    )
    out = tmp_path / "out.html"
    out.write_text("old", encoding="utf-8")
    with pytest.raises(TreeWireClearanceVisualizationOutputError):
        run_tree_wire_clearance_visualization(
            tmp_path / "clearance.json", v4_root, w1_npz, w1_json, out
        )
    assert out.read_text(encoding="utf-8") == "old"


def test_budget_validation() -> None:
    with pytest.raises(TreeWireClearanceVisualizationError):
        validate_display_budget(0)
    with pytest.raises(TreeWireClearanceVisualizationError):
        validate_display_budget(1_000_000)
    assert validate_display_budget(500).max_points_per_tree == 500


def test_load_tree_wire_clearance_result_validates(tmp_path: Path) -> None:
    _v4_root, _w1_npz, _w1_json, clearance = _run_and_write_clearance(
        tmp_path, _standard_segments()
    )
    loaded = load_tree_wire_clearance_result(tmp_path / "clearance.json")
    assert loaded == clearance


def test_build_summary_matches_result(tmp_path: Path) -> None:
    _v4_root, _w1_npz, _w1_json, clearance = _run_and_write_clearance(
        tmp_path, _standard_segments()
    )
    budget = validate_display_budget(5000)
    summary = build_summary(
        clearance=clearance,
        displayed_counts={"trees": 2, "valid_trees": 2, "clearance_segments": 2,
                          "assigned_points": 100, "wire_segments": 2},
        budget=budget,
        output_paths={"html": "a.html", "summary_json": "a.json"},
    )
    assert summary["clearance_result_digest"] == clearance.result_digest
    assert len(summary["trees"]) == 2
    assert summary["trees"][0]["tree_id"] == clearance.records[0].tree_id
