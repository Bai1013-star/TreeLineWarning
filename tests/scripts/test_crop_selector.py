from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import scripts.crop_selector as crop_selector
from scripts.crop_selector import (
    crop_bounds_json,
    figure_for_selection,
    filter_preview_points,
    load_preview_data,
    save_selection_record,
    selection_record,
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


def test_save_path_only_under_outputs_selections_and_no_overwrite(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="outputs/selections"):
        save_selection_record(tmp_path / "data" / "raw" / "bad.json", record, project_root=tmp_path)


def test_streamlit_page_figure_smoke(tmp_path: Path) -> None:
    npz_path, json_path = write_preview(tmp_path)
    preview = load_preview_data(npz_path, json_path)
    mask = filter_preview_points(preview.xyz, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0])

    figure = figure_for_selection(preview, mask, "RGB", [0.0, 0.0, 0.0], [2.0, 2.0, 2.0])

    assert len(figure.data) == 2
    assert figure.data[0].name == "Selected preview points"
    assert figure.data[1].name == "Crop bounds"


def test_streamlit_page_entrypoint_smoke() -> None:
    assert importlib.util.find_spec("streamlit") is not None
    assert callable(crop_selector.main)
