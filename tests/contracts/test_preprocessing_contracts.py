from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from core.contracts import (
    DownsampledPointCloud,
    PreprocessingConfig,
    PreprocessingResult,
    ProcessingStatistics,
)


def make_statistics(**overrides: object) -> ProcessingStatistics:
    values: dict[str, object] = {
        "input_point_count": 10,
        "processed_input_point_count": 10,
        "valid_point_count": 8,
        "nan_inf_removed_count": 1,
        "abnormal_coord_removed_count": 1,
        "outlier_removed_count": 0,
        "downsampled_point_count": 3,
        "downsample_ratio": 3 / 8,
        "chunk_size": 4,
        "voxel_size_m": 0.05,
        "elapsed_seconds": 1.25,
        "rss_start_bytes": 100,
        "rss_peak_bytes": 200,
        "rss_end_bytes": 150,
        "sor_skipped_reason": "disabled_by_default_to_preserve_thin_wires",
        "warnings": ("test warning",),
    }
    values.update(overrides)
    return ProcessingStatistics(**values)  # type: ignore[arg-type]


def test_preprocessing_config_serializes_paths_and_defaults() -> None:
    config = PreprocessingConfig(source_id="scene_simple")

    payload = config.to_dict()
    json.dumps(payload)
    restored = PreprocessingConfig.from_dict(payload)

    assert restored.source_id == "scene_simple"
    assert restored.chunk_size == 250_000
    assert restored.coordinate_dtype == "float32"
    assert restored.enable_statistical_outlier_filter is False
    assert restored.voxel_size_m == 0.05


def test_preprocessing_config_validates_explicit_origin_and_limits() -> None:
    config = PreprocessingConfig(origin_mode="explicit", origin_xyz=(1.0, 2.0, 3.0))

    assert config.origin_xyz == (1.0, 2.0, 3.0)

    with pytest.raises(ValueError, match="origin_xyz is required"):
        PreprocessingConfig(origin_mode="explicit")

    with pytest.raises(ValueError, match="voxel_size_m"):
        PreprocessingConfig(voxel_size_m=0.0)


def test_processing_statistics_serializes_and_rejects_bad_counts() -> None:
    stats = make_statistics()

    payload = stats.to_dict()
    json.dumps(payload)
    restored = ProcessingStatistics.from_dict(payload)

    assert restored.downsampled_point_count == 3
    assert restored.warnings == ("test warning",)

    with pytest.raises(ValueError, match="input_point_count"):
        make_statistics(input_point_count=-1)


def test_downsampled_point_cloud_validates_shapes_and_attribute_lengths() -> None:
    cloud = DownsampledPointCloud(
        source_id="scene_simple",
        xyz_local=np.zeros((2, 3), dtype=np.float32),
        origin_xyz=np.zeros(3, dtype=np.float64),
        original_index=np.array([10, 20], dtype=np.int64),
        sample_index=np.array([1, 2], dtype=np.int64),
        voxel_key=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
        voxel_point_count=np.array([3, 4], dtype=np.int32),
        attributes={"intensity": np.array([1, 2], dtype=np.uint16)},
    )

    assert cloud.point_count == 2
    assert cloud.array_metadata()["xyz_local"]["shape"] == [2, 3]

    with pytest.raises(ValueError, match="attributes.intensity length"):
        DownsampledPointCloud(
            source_id="scene_simple",
            xyz_local=np.zeros((2, 3), dtype=np.float32),
            origin_xyz=np.zeros(3, dtype=np.float64),
            original_index=np.array([10, 20], dtype=np.int64),
            sample_index=np.array([1, 2], dtype=np.int64),
            voxel_key=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
            voxel_point_count=np.array([3, 4], dtype=np.int32),
            attributes={"intensity": np.array([1], dtype=np.uint16)},
        )


def test_preprocessing_result_json_roundtrip_restores_paths() -> None:
    result = PreprocessingResult(
        source_id="scene_simple",
        input_path=Path("data/samples/scene_simple.las"),
        index_map_path=Path("data/samples/scene_simple_index_map.npz"),
        manifest_path=Path("data/samples/scene_simple_manifest.json"),
        output_npz_path=Path("outputs/pointcloud/scene_simple_preprocessed.npz"),
        output_json_path=Path("outputs/pointcloud/scene_simple_preprocessed.json"),
        config=PreprocessingConfig(),
        statistics=make_statistics(),
        manifest_source_id="raw_source",
        origin_xyz=(1.0, 2.0, 3.0),
        arrays={"xyz_local": {"shape": [3, 3], "dtype": "float32"}},
        warnings=("warning",),
        created_at="2026-07-21T00:00:00+00:00",
    )

    payload = result.to_dict()
    json.dumps(payload)
    restored = PreprocessingResult.from_dict(payload)

    assert restored.input_path == Path("data/samples/scene_simple.las")
    assert restored.output_npz_path == Path("outputs/pointcloud/scene_simple_preprocessed.npz")
    assert restored.statistics.downsampled_point_count == 3
