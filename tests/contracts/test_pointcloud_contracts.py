from __future__ import annotations

import json
from pathlib import Path

import pytest
from core.contracts import (
    CropBounds,
    CropResult,
    LasHeaderSummary,
    PointCloudChunk,
    PointCloudSource,
)


class FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def make_header(**overrides: object) -> LasHeaderSummary:
    values: dict[str, object] = {
        "source_id": "scene_01",
        "source_path": Path("data/raw/scene_01_small/source.las"),
        "file_size_bytes": 456,
        "las_version": "1.4",
        "point_format_id": 7,
        "point_record_length_bytes": 34,
        "point_count": 1000,
        "min_xyz": (1.0, 2.0, 3.0),
        "max_xyz": (11.0, 12.0, 13.0),
        "scale": (0.001, 0.001, 0.001),
        "offset": (100.0, 200.0, 0.0),
        "dimensions": ("X", "Y", "Z", "intensity", "classification"),
        "extra_dimensions": ("semantic_label",),
        "has_rgb": True,
        "has_intensity": True,
        "has_classification": True,
        "estimated_record_memory_bytes": 34000,
    }
    values.update(overrides)
    return LasHeaderSummary(**values)  # type: ignore[arg-type]


def test_point_cloud_source_raw_requires_readonly() -> None:
    source = PointCloudSource(
        source_id="scene_01_raw",
        path=Path("data/raw/scene_01_small/source.las"),
        role="raw",
    )

    assert source.readonly_required is True

    with pytest.raises(ValueError, match="readonly_required must be True"):
        PointCloudSource(
            source_id="scene_01_raw",
            path=Path("data/raw/scene_01_small/source.las"),
            role="raw",
            readonly_required=False,
        )


def test_point_cloud_source_serializes_and_restores_path() -> None:
    source = PointCloudSource(
        source_id="scene_01_sample",
        path=Path("data/samples/scene_simple.las"),
        role="sample",
        size_bytes=123,
        readonly_required=False,
        crs_wkt="LOCAL_CS[]",
        metadata={"unit": "m", "tags": ["sample"]},
    )

    payload = source.to_dict()
    json.dumps(payload)
    restored = PointCloudSource.from_dict(payload)

    assert isinstance(restored.path, Path)
    assert restored.path == Path("data/samples/scene_simple.las")
    assert restored.metadata["tags"] == ["sample"]


def test_las_header_summary_serializes_header_only_metadata() -> None:
    header = make_header()

    payload = header.to_dict()
    json.dumps(payload)
    restored = LasHeaderSummary.from_dict(payload)

    assert restored.source_id == "scene_01"
    assert restored.source_path == Path("data/raw/scene_01_small/source.las")
    assert restored.point_record_length_bytes == 34
    assert restored.min_xyz == (1.0, 2.0, 3.0)
    assert restored.max_xyz == (11.0, 12.0, 13.0)
    assert restored.estimated_record_memory_bytes == 34000


def test_las_header_summary_rejects_invalid_counts_record_length_and_bounds() -> None:
    with pytest.raises(ValueError, match="point_count must be a non-negative integer"):
        make_header(point_count=-1)

    with pytest.raises(ValueError, match="point_record_length_bytes must be a positive integer"):
        make_header(point_record_length_bytes=0)

    with pytest.raises(ValueError, match="min_xyz cannot exceed max_xyz"):
        make_header(min_xyz=(2.0, 0.0, 0.0), max_xyz=(1.0, 1.0, 1.0))


def test_point_cloud_chunk_validates_xyz_shape_and_point_count() -> None:
    chunk = PointCloudChunk(
        source_id="scene_01",
        chunk_id=0,
        start_index=0,
        point_count=2,
        xyz=FakeArray((2, 3)),
        original_index=FakeArray((2,)),
        is_last=True,
    )

    assert chunk.summary_dict()["has_original_index"] is True

    with pytest.raises(ValueError, match="shape \\(N, 3\\)"):
        PointCloudChunk(
            source_id="scene_01",
            chunk_id=0,
            start_index=0,
            point_count=2,
            xyz=FakeArray((2, 2)),
        )

    with pytest.raises(ValueError, match="point_count must match xyz length"):
        PointCloudChunk(
            source_id="scene_01",
            chunk_id=0,
            start_index=0,
            point_count=3,
            xyz=FakeArray((2, 3)),
        )


def test_point_cloud_chunk_original_index_length_must_match_point_count() -> None:
    with pytest.raises(ValueError, match="original_index length must match point_count"):
        PointCloudChunk(
            source_id="scene_01",
            chunk_id=0,
            start_index=0,
            point_count=2,
            xyz=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
            original_index=[10],
        )


def test_crop_bounds_supports_source_and_local_modes() -> None:
    source_bounds = CropBounds(min_xyz=(0.0, 0.0, 0.0), max_xyz=(1.0, 1.0, 1.0))
    local_bounds = CropBounds(
        min_xyz=(0.0, 0.0, 0.0),
        max_xyz=(1.0, 1.0, 1.0),
        coordinate_mode="local",
        origin_xyz=(100.0, 200.0, 0.0),
    )

    assert source_bounds.coordinate_mode == "source"
    assert CropBounds.from_dict(local_bounds.to_dict()).origin_xyz == (100.0, 200.0, 0.0)

    with pytest.raises(ValueError, match="origin_xyz is required"):
        CropBounds(
            min_xyz=(0.0, 0.0, 0.0),
            max_xyz=(1.0, 1.0, 1.0),
            coordinate_mode="local",
        )


def test_crop_bounds_requires_min_less_than_max_on_each_axis() -> None:
    with pytest.raises(ValueError, match="strictly less than max_xyz"):
        CropBounds(min_xyz=(0.0, 0.0, 0.0), max_xyz=(1.0, 0.0, 1.0))


def test_crop_result_rejects_output_count_above_input_count() -> None:
    bounds = CropBounds(min_xyz=(0.0, 0.0, 0.0), max_xyz=(1.0, 1.0, 1.0))
    result = CropResult(
        source_id="scene_01",
        bounds=bounds,
        input_point_count=100,
        output_point_count=20,
        output_path=Path("data/samples/scene_simple.las"),
        header_preserved=True,
        dimension_names=("X", "Y", "Z"),
        warnings=("small sample",),
    )

    payload = result.to_dict()
    json.dumps(payload)
    assert CropResult.from_dict(payload).output_path == Path("data/samples/scene_simple.las")

    with pytest.raises(ValueError, match="output_point_count cannot exceed input_point_count"):
        CropResult(
            source_id="scene_01",
            bounds=bounds,
            input_point_count=10,
            output_point_count=11,
            output_path=Path("data/samples/scene_simple.las"),
            header_preserved=True,
            dimension_names=("X", "Y", "Z"),
        )


def test_unknown_field_has_clear_error() -> None:
    payload = {
        "source_id": "scene_01",
        "path": "data/raw/scene_01_small/source.las",
        "role": "raw",
        "unexpected": "value",
    }

    with pytest.raises(ValueError, match="unknown field"):
        PointCloudSource.from_dict(payload)


def test_non_json_metadata_has_clear_error() -> None:
    with pytest.raises(ValueError, match="JSON-serializable"):
        PointCloudSource(
            source_id="scene_01",
            path=Path("source.las"),
            role="sample",
            metadata={"bad": {1, 2, 3}},
        )
