from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import scripts.crop_selector as crop_selector
from scripts.crop_selector import (
    DEFAULT_COLOR_MODE,
    crop_bounds_json,
    enhance_rgb,
    figure_for_selection,
    filter_preview_points,
    load_preview_data,
    projection_coordinates,
    projection_figure,
    save_selection_record,
    selection_record,
    selection_warning,
    validate_xyz_bounds,
)


def write_preview(tmp_path: Path) -> tuple[Path, Path]:
    npz_path = tmp_path / "preview.npz"
    json_path = tmp_path / "preview.json"
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=np.float64,
    )
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    np.savez_compressed(
        npz_path,
        xyz=xyz,
        local_xyz=xyz - origin,
        original_index=np.array([0, 10, 20, 30], dtype=np.int64),
        rgb=np.array(
            [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]],
            dtype=np.uint16,
        ),
        intensity=np.array([1, 2, 3, 4], dtype=np.uint16),
        classification=np.array([0, 1, 2, 3], dtype=np.uint8),
    )
    json_path.write_text(
        json.dumps(
            {
                "source_id": "preview_source",
                "source_path": str(tmp_path / "not_read.las"),
                "original_point_count": 100,
                "preview_point_count": 4,
                "original_xyz_range": {"min": [0.0, 0.0, 0.0], "max": [3.0, 3.0, 3.0]},
            }
        ),
        encoding="utf-8",
    )
    return npz_path, json_path


def test_load_preview_npz_and_json_without_original_las(tmp_path: Path) -> None:
    npz_path, json_path = write_preview(tmp_path)

    preview = load_preview_data(npz_path, json_path)

    assert preview.metadata["source_id"] == "preview_source"
    assert preview.metadata["source_path"].endswith("not_read.las")
    assert preview.xyz.shape == (4, 3)
    assert preview.rgb is not None
    assert preview.intensity is not None
    assert preview.classification is not None


def test_min_max_validation() -> None:
    validate_xyz_bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="X min"):
        validate_xyz_bounds([1.0, 0.0, 0.0], [1.0, 1.0, 1.0])


def test_range_filter_count_and_boundary_inclusion(tmp_path: Path) -> None:
    npz_path, json_path = write_preview(tmp_path)
    preview = load_preview_data(npz_path, json_path)

    mask = filter_preview_points(preview.xyz, [1.0, 1.0, 1.0], [2.0, 2.0, 2.0])

    assert mask.tolist() == [False, True, True, False]
    assert int(np.count_nonzero(mask)) == 2


def test_empty_selection_mask_is_explicit(tmp_path: Path) -> None:
    npz_path, json_path = write_preview(tmp_path)
    preview = load_preview_data(npz_path, json_path)

    mask = filter_preview_points(preview.xyz, [9.0, 9.0, 9.0], [10.0, 10.0, 10.0])

    assert int(np.count_nonzero(mask)) == 0


def test_crop_bounds_json_format() -> None:
    payload = crop_bounds_json([0.0, 1.0, 2.0], [3.0, 4.0, 5.0])

    assert payload == {
        "min_xyz": [0.0, 1.0, 2.0],
        "max_xyz": [3.0, 4.0, 5.0],
        "coordinate_mode": "source",
        "label": "scene_simple",
    }


def test_save_path_only_under_outputs_selections_and_overwrite_must_be_explicit(
    tmp_path: Path,
) -> None:
    npz_path, json_path = write_preview(tmp_path)
    preview = load_preview_data(npz_path, json_path)
    record = selection_record(
        preview,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        2,
        note="candidate",
    )
    output = tmp_path / "outputs" / "selections" / "scene_simple_bounds.json"

    save_selection_record(output, record, project_root=tmp_path)
    assert json.loads(output.read_text(encoding="utf-8"))["selected_preview_point_count"] == 2

    with pytest.raises(FileExistsError):
        save_selection_record(output, record, project_root=tmp_path)

    replacement = dict(record)
    replacement["selected_preview_point_count"] = 3
    save_selection_record(output, replacement, project_root=tmp_path, overwrite=True)
    assert json.loads(output.read_text(encoding="utf-8"))["selected_preview_point_count"] == 3

    with pytest.raises(ValueError, match="outputs/selections"):
        save_selection_record(tmp_path / "data" / "raw" / "bad.json", record, project_root=tmp_path)


def test_uint8_rgb_enhancement_outputs_plotly_rgb_range() -> None:
    rgb = np.array(
        [[0, 10, 20], [40, 80, 120], [160, 180, 200], [255, 240, 230]],
        dtype=np.uint8,
    )

    enhanced = enhance_rgb(rgb, gamma=0.8)

    assert enhanced.colors is not None
    assert enhanced.stats["valid"] is True
    assert enhanced.stats["dtype"] == "uint8"
    assert enhanced.stats["output_min"] == [0, 0, 0]
    assert enhanced.stats["output_max"] == [255, 255, 255]
    assert all(color.startswith("rgb(") for color in enhanced.colors)


def test_uint16_rgb_enhancement_uses_percentile_stretch() -> None:
    rgb = np.array(
        [[1000, 2000, 3000], [20_000, 30_000, 40_000], [45_000, 50_000, 55_000]],
        dtype=np.uint16,
    )

    enhanced = enhance_rgb(rgb, gamma=0.8)

    assert enhanced.colors is not None
    assert enhanced.stats["valid"] is True
    assert enhanced.stats["dtype"] == "uint16"
    assert enhanced.stats["method"] == "per_channel_1_99_percentile_gamma"
    assert min(enhanced.stats["output_min"]) >= 0
    assert max(enhanced.stats["output_max"]) <= 255


def test_invalid_rgb_falls_back_to_height_mode(tmp_path: Path) -> None:
    npz_path, json_path = write_preview(tmp_path)
    preview = load_preview_data(npz_path, json_path)
    preview = crop_selector.PreviewData(
        xyz=preview.xyz,
        local_xyz=preview.local_xyz,
        original_index=preview.original_index,
        metadata=preview.metadata,
        rgb=np.full((len(preview.xyz), 3), 65535, dtype=np.uint16),
        intensity=preview.intensity,
        classification=preview.classification,
    )
    mask = np.ones(len(preview.xyz), dtype=bool)

    enhanced = enhance_rgb(preview.rgb)
    figure = figure_for_selection(preview, mask, "增强 RGB")

    assert enhanced.colors is None
    assert enhanced.stats["reason"] == "no_rgb_contrast"
    assert list(figure.data[0].marker.color) == preview.xyz[:, 2].tolist()
    assert figure.data[0].marker.colorbar.title.text == "Z (m)"


def test_height_mode_is_default_and_marker_controls_work(tmp_path: Path) -> None:
    npz_path, json_path = write_preview(tmp_path)
    preview = load_preview_data(npz_path, json_path)
    mask = np.ones(len(preview.xyz), dtype=bool)

    figure = figure_for_selection(preview, mask, DEFAULT_COLOR_MODE, point_size=4.0, opacity=0.4)

    assert DEFAULT_COLOR_MODE == "高度"
    assert figure.data[0].marker.size == 4.0
    assert figure.data[0].marker.opacity == 0.4
    assert figure.data[0].marker.colorbar.title.text == "Z (m)"


def test_projection_coordinates_and_figures_use_filtered_points(tmp_path: Path) -> None:
    npz_path, json_path = write_preview(tmp_path)
    preview = load_preview_data(npz_path, json_path)
    mask = filter_preview_points(preview.xyz, [1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    selected_xyz = preview.xyz[mask]

    assert projection_coordinates(selected_xyz, "XY").tolist() == [[1.0, 1.0], [2.0, 2.0]]
    assert projection_coordinates(selected_xyz, "XZ").tolist() == [[1.0, 1.0], [2.0, 2.0]]
    assert projection_coordinates(selected_xyz, "YZ").tolist() == [[1.0, 1.0], [2.0, 2.0]]

    figure = projection_figure(
        preview,
        mask,
        "XY",
        "高度",
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    )

    assert len(figure.data) == 2
    assert list(figure.data[0].x) == [1.0, 2.0]
    assert list(figure.data[0].y) == [1.0, 2.0]
    assert figure.data[1].name == "选择范围"


def test_whole_scene_ratio_warning_threshold() -> None:
    assert selection_warning(0.8) is None
    assert selection_warning(0.81) == "当前范围接近整个场景，不建议作为 scene_simple"


def test_streamlit_page_figure_smoke(tmp_path: Path) -> None:
    npz_path, json_path = write_preview(tmp_path)
    preview = load_preview_data(npz_path, json_path)
    mask = filter_preview_points(preview.xyz, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0])

    figure = figure_for_selection(preview, mask, "高度", [0.0, 0.0, 0.0], [2.0, 2.0, 2.0])

    assert len(figure.data) == 2
    assert figure.data[0].name == "Selected preview points"
    assert figure.data[1].name == "裁剪包围盒"


def test_streamlit_page_entrypoint_smoke() -> None:
    assert importlib.util.find_spec("streamlit") is not None
    assert callable(crop_selector.main)
