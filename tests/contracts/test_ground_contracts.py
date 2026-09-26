from __future__ import annotations

import numpy as np
import pytest
from core.contracts import (
    GroundEstimationConfig,
    GroundEstimationResult,
    GroundEstimationStatistics,
    GroundQualityFlag,
    GroundSurfaceModel,
    HeightAboveGroundSet,
)


def test_ground_config_defaults_round_trip_and_fixed_bits() -> None:
    config = GroundEstimationConfig()

    assert config.grid_size_m == 0.5
    assert config.ground_quantile == 0.05
    assert config.quantile_method == "linear"
    assert GroundEstimationConfig.from_dict(config.to_dict()) == config
    assert GroundQualityFlag.SPARSE_GROUND_CELL == 1
    assert GroundQualityFlag.INVALID_GROUND_ESTIMATE == 128
    assert GroundQualityFlag.SOURCE_CLASSIFICATION_DISAGREES == 256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid_size_m": 0.0},
        {"ground_quantile": 0.0},
        {"quantile_method": "nearest"},
        {"min_points_per_cell": 0},
        {"interpolation_method": "nearest"},
        {"idw_neighbors": 0},
        {"smoothing_radius_cells": -1},
    ],
)
def test_ground_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GroundEstimationConfig(**kwargs)


def test_surface_and_hag_validate_shapes_and_legal_missing_values() -> None:
    surface = GroundSurfaceModel(
        grid_origin_xy=np.array([0.0, 0.0], dtype=np.float64),
        grid_size_m=1.0,
        grid_shape=(1, 2),
        cell_point_count=np.array([[4, 0]], dtype=np.int32),
        cell_ground_z_m=np.array([[1.0, np.nan]], dtype=np.float32),
        cell_control_mask=np.array([[True, False]], dtype=bool),
        cell_quality_flags=np.array([[0, 2]], dtype=np.uint16),
        control_cell_xy=np.array([[0.5, 0.5]], dtype=np.float32),
        control_ground_z_m=np.array([1.0], dtype=np.float32),
    )
    hag = HeightAboveGroundSet(
        source_id="scene_simple",
        manifest_source_id="raw_scene",
        original_index=np.array([10, 11], dtype=np.int64),
        sample_index=np.array([0, 1], dtype=np.int64),
        ground_z_m=np.array([1.0, np.nan], dtype=np.float32),
        raw_height_agl_m=np.array([0.0, np.nan], dtype=np.float32),
        height_agl_m=np.array([0.0, np.nan], dtype=np.float32),
        ground_cell_id=np.array([0, 1], dtype=np.int32),
        ground_distance_m=np.array([0.0, np.nan], dtype=np.float32),
        quality_flags=np.array([0, 128], dtype=np.uint16),
    )

    assert surface.control_cell_count == 1
    assert hag.point_count == 2
    assert hag.array_metadata()["height_agl_m"]["unit"] == "m"


def test_hag_rejects_unflagged_missing_value() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        HeightAboveGroundSet(
            source_id="scene_simple",
            manifest_source_id="raw_scene",
            original_index=np.array([10], dtype=np.int64),
            sample_index=np.array([0], dtype=np.int64),
            ground_z_m=np.array([np.nan], dtype=np.float32),
            raw_height_agl_m=np.array([np.nan], dtype=np.float32),
            height_agl_m=np.array([np.nan], dtype=np.float32),
            ground_cell_id=np.array([0], dtype=np.int32),
            ground_distance_m=np.array([np.nan], dtype=np.float32),
            quality_flags=np.array([0], dtype=np.uint16),
        )


def test_result_json_round_trip() -> None:
    statistics = GroundEstimationStatistics(
        input_point_count=2,
        output_point_count=2,
        total_cell_count=2,
        nonempty_cell_count=1,
        direct_control_cell_count=1,
        sparse_cell_count=0,
        empty_cell_count=1,
        interpolated_cell_count=0,
        extrapolation_too_far_cell_count=0,
        valid_ground_point_count=2,
        invalid_ground_point_count=0,
        high_gradient_cell_count=0,
        high_gradient_point_count=0,
        raw_negative_hag_count=0,
        below_ground_tolerance_count=0,
        extreme_hag_count=0,
        classification_ground_count=0,
        classification_disagreement_count=0,
        elapsed_seconds=0.1,
    )
    result = GroundEstimationResult(
        source_id="scene_simple",
        manifest_source_id="raw_scene",
        input_a1_npz_path="a1.npz",
        input_a1_json_path="a1.json",
        input_a1_npz_sha256="a" * 64,
        input_a1_json_sha256="b" * 64,
        input_a2a_npz_path="a2.npz",
        input_a2a_json_path="a2.json",
        input_a2a_npz_sha256="c" * 64,
        input_a2a_json_sha256="d" * 64,
        output_npz_path="out.npz",
        output_json_path="out.json",
        output_npz_sha256="e" * 64,
        output_npz_bytes=12,
        origin_xyz=(1.0, 2.0, 3.0),
        config=GroundEstimationConfig(),
        statistics=statistics,
        grid={"grid_shape": [1, 2]},
        arrays={},
        quality_flag_counts={"NONE": 2},
        distributions={},
        formulas={},
        classification_validation={},
        algorithm_version="test",
        created_at="2026-01-01T00:00:00+00:00",
    )

    restored = GroundEstimationResult.from_dict(result.to_dict())

    assert restored.config == result.config
    assert restored.statistics == statistics
    assert restored.output_npz_sha256 == "e" * 64
