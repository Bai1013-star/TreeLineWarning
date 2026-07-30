from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from algorithms.vegetation.tree_segmentation import build_tree_segmentation_result
from algorithms.vegetation.tree_visualization import (
    MAXIMUM_3D_POINTS,
    UNASSIGNED_RGB,
    build_summary,
    deterministic_display_indices,
    render_3d_html,
    render_diagnostic_png,
    render_topdown_png,
    segment_color,
    topdown_pixel_statistics,
    validate_input_result_alignment,
    write_summary_json,
)
from PIL import Image

from tests.vegetation.test_tree_segmentation import _pipeline


def _scene() -> tuple[object, object, object, object, object]:
    tree_input, model, candidates, segmentation = _pipeline(
        xy=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        hag=[5.0, 4.0, 4.0, 4.0],
        minimum_peak_hag_m=0.0,
    )
    return (
        tree_input,
        model,
        candidates,
        segmentation,
        build_tree_segmentation_result(tree_input, model, candidates, segmentation),
    )


def _pixel_bundle(
    reference_input: object, xy: list[tuple[float, float]], segments: list[int]
) -> tuple[object, object]:
    count = len(segments)
    identity = replace(reference_input.identity, point_count=count)
    tree_input = SimpleNamespace(
        identity=identity,
        vegetation_points=SimpleNamespace(
            xyz_local=np.asarray([[x, y, 0.0] for x, y in xy], dtype=np.float32)
        ),
    )
    result = SimpleNamespace(
        statistics=SimpleNamespace(input_point_count=count),
        source_id=identity.source_id,
        manifest_source_id=identity.manifest_source_id,
        origin_xyz=identity.origin_xyz,
        input_provenance={"source_identity": identity.to_dict()},
        segment_id_by_point=np.asarray(segments, dtype=np.int32),
        tree_id_by_point=np.full(count, -1, dtype=np.int32),
        tree_distance_ready_by_point=np.zeros(count, dtype=np.bool_),
        records=(),
    )
    return tree_input, result


def _pixel(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as image:
        return image.convert("RGB").getpixel((0, 0))


def test_segment_colours_are_deterministic_and_unassigned_is_gray() -> None:
    assert segment_color(12) == segment_color(12)
    assert segment_color(12) != segment_color(13)
    assert segment_color(-1) == UNASSIGNED_RGB


def test_display_sampling_is_deterministic_and_bounded() -> None:
    first = deterministic_display_indices(101, 10)
    second = deterministic_display_indices(101, 10)
    assert np.array_equal(first, second)
    assert len(first) == 10
    assert int(first[0]) == 0
    assert int(first[-1]) == 100
    assert np.array_equal(deterministic_display_indices(3, 10), np.array([0, 1, 2]))


@pytest.mark.parametrize(
    ("segments", "expected", "conflicts"),
    [
        ([-1], UNASSIGNED_RGB, 0),
        ([0], segment_color(0), 0),
        ([0, -1], segment_color(0), 0),
        ([0, 1], (0, 0, 0), 1),
        ([0, 0], segment_color(0), 0),
    ],
)
def test_topdown_pixel_policy(
    tmp_path: Path, segments: list[int], expected: tuple[int, int, int], conflicts: int
) -> None:
    reference_input, _model, _candidates, _segmentation, _result = _scene()
    tree_input, result = _pixel_bundle(reference_input, [(0.0, 0.0)] * len(segments), segments)
    output = render_topdown_png(tree_input, result, tmp_path / "pixel.png")
    assert _pixel(output) == expected
    assert topdown_pixel_statistics(tree_input, result)["conflict_pixel_count"] == conflicts


def test_topdown_is_byte_identical_when_source_rows_are_reversed(tmp_path: Path) -> None:
    reference_input, _model, _candidates, _segmentation, _result = _scene()
    forward_input, forward_result = _pixel_bundle(
        reference_input, [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)], [0, 1, -1]
    )
    reverse_input, reverse_result = _pixel_bundle(
        reference_input, [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)], [-1, 1, 0]
    )
    forward = render_topdown_png(forward_input, forward_result, tmp_path / "forward.png")
    reverse = render_topdown_png(reverse_input, reverse_result, tmp_path / "reverse.png")
    assert forward.read_bytes() == reverse.read_bytes()


def test_renderers_reject_equal_length_but_differently_identified_result(tmp_path: Path) -> None:
    tree_input, _model, _candidates, _segmentation, result = _scene()
    misidentified = replace(result, source_id="different-source")
    with pytest.raises(ValueError, match="source_id"):
        render_topdown_png(tree_input, misidentified, tmp_path / "topdown.png")
    with pytest.raises(ValueError, match="source_id"):
        render_3d_html(tree_input, misidentified, tmp_path / "tree.html", maximum_points=2)


@pytest.mark.parametrize(
    "maximum_points", [-1, 0, True, np.bool_(False), 1.5, MAXIMUM_3D_POINTS + 1]
)
def test_direct_3d_renderer_enforces_hard_cap(tmp_path: Path, maximum_points: object) -> None:
    tree_input, _model, _candidates, _segmentation, result = _scene()
    with pytest.raises(ValueError, match="1 through 200000"):
        render_3d_html(tree_input, result, tmp_path / "tree.html", maximum_points=maximum_points)


def test_direct_3d_renderer_accepts_hard_cap(tmp_path: Path) -> None:
    tree_input, _model, _candidates, _segmentation, result = _scene()
    html, displayed = render_3d_html(
        tree_input, result, tmp_path / "tree.html", maximum_points=MAXIMUM_3D_POINTS
    )
    assert html.is_file()
    assert displayed == len(result.segment_id_by_point)


def test_renderers_create_outputs_without_mutating_formal_arrays(tmp_path: Path) -> None:
    tree_input, model, candidates, segmentation, result = _scene()
    original_segments = result.segment_id_by_point.copy()
    topdown = render_topdown_png(tree_input, result, tmp_path / "topdown.png")
    diagnostic = render_diagnostic_png(model, candidates, segmentation, tmp_path / "diagnostic.png")
    html, displayed = render_3d_html(tree_input, result, tmp_path / "tree.html", maximum_points=2)
    assert topdown.is_file() and topdown.stat().st_size > 0
    assert diagnostic.is_file() and diagnostic.stat().st_size > 0
    assert html.is_file() and html.stat().st_size > 0
    assert displayed == 2
    assert np.array_equal(result.segment_id_by_point, original_segments)


def test_summary_json_reports_verified_counts(tmp_path: Path) -> None:
    tree_input, model, candidates, segmentation, result = _scene()
    summary = build_summary(
        scene_name="synthetic",
        source_artifact_paths={"v2_npz": "v2.npz"},
        tree_input=tree_input,
        canopy_model=model,
        candidates=candidates,
        segmentation=segmentation,
        result=result,
        maximum_3d_points=20,
        displayed_3d_points=4,
        output_paths={"summary_json": tmp_path / "summary.json"},
        topdown_statistics=topdown_pixel_statistics(tree_input, result),
    )
    path = write_summary_json(summary, tmp_path / "summary.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert (
        loaded["assigned_point_count"] + loaded["unassigned_point_count"]
        == loaded["source_point_count"]
    )
    assert loaded["html_displayed_point_count"] == 4
    assert loaded["candidate_count"] == len(candidates.candidate_id)
    assert loaded["segment_record_count"] == len(result.records)
    assert loaded["chm_shape"] == list(model.canopy_height_m.shape)
    assert loaded["ambiguous_cell_count"] == segmentation.ambiguous_cell_count
    assert "conflict_pixel_count" in loaded
    assert "segment_id_by_point" not in loaded


def test_alignment_validator_accepts_the_formal_pipeline_pair() -> None:
    tree_input, _model, _candidates, _segmentation, result = _scene()
    assert validate_input_result_alignment(tree_input, result) == 4
