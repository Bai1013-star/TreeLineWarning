from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from core.contracts import (
    FeatureQualityFlag,
    GeometryFeatureConfig,
    GeometryFeatureResult,
    GeometryFeatureStatistics,
    PointFeatureSet,
)


def make_feature_set(point_count: int = 2) -> PointFeatureSet:
    return PointFeatureSet(
        source_id="scene_simple",
        manifest_source_id="raw_scene",
        xyz_local=np.zeros((point_count, 3), dtype=np.float32),
        origin_xyz=np.zeros(3, dtype=np.float64),
        original_index=np.arange(point_count, dtype=np.int64),
        sample_index=np.arange(point_count, dtype=np.int64),
        linearity=np.zeros(point_count, dtype=np.float32),
        planarity=np.zeros(point_count, dtype=np.float32),
        scattering=np.zeros(point_count, dtype=np.float32),
        verticality=np.zeros(point_count, dtype=np.float32),
        local_density=np.zeros(point_count, dtype=np.float32),
        height_m=np.zeros(point_count, dtype=np.float32),
        neighbor_count=np.zeros(point_count, dtype=np.uint16),
        eigenvalues=np.zeros((point_count, 3), dtype=np.float32),
        principal_direction=np.zeros((point_count, 3), dtype=np.float32),
        normal=np.zeros((point_count, 3), dtype=np.float32),
        quality_flags=np.zeros(point_count, dtype=np.uint16),
    )


def make_statistics() -> GeometryFeatureStatistics:
    return GeometryFeatureStatistics(
        input_point_count=2,
        output_point_count=2,
        pca_computed_count=2,
        insufficient_neighbor_count=0,
        degenerate_covariance_count=0,
        duplicate_neighbor_coordinate_count=0,
        negative_eigenvalue_clipped_count=0,
        normal_direction_ambiguous_count=0,
        principal_direction_ambiguous_count=0,
        density_undefined_count=0,
        numeric_invalid_count=0,
        requested_k=2,
        query_batch_size=12_000,
        elapsed_seconds=0.5,
    )


def test_geometry_feature_config_defaults_and_round_trip() -> None:
    config = GeometryFeatureConfig()

    assert config.k_neighbors == 32
    assert config.min_neighbors == 8
    assert config.query_batch_size == 12_000
    assert config.workers == 1
    assert config.direction_gap_relative_tolerance == 1e-3
    assert GeometryFeatureConfig.from_dict(config.to_dict()) == config


def test_geometry_feature_config_rejects_invalid_neighbor_bounds() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        GeometryFeatureConfig(k_neighbors=8, min_neighbors=9)


def test_feature_quality_bits_are_stable_uint16_values() -> None:
    assert int(FeatureQualityFlag.NONE) == 0
    assert int(FeatureQualityFlag.INSUFFICIENT_NEIGHBORS) == 1
    assert int(FeatureQualityFlag.DEGENERATE_COVARIANCE) == 2
    assert int(FeatureQualityFlag.DUPLICATE_NEIGHBOR_COORDINATES) == 4
    assert int(FeatureQualityFlag.NEGATIVE_EIGENVALUE_CLIPPED) == 8
    assert int(FeatureQualityFlag.NORMAL_DIRECTION_AMBIGUOUS) == 16
    assert int(FeatureQualityFlag.PRINCIPAL_DIRECTION_AMBIGUOUS) == 32
    assert int(FeatureQualityFlag.NUMERIC_INVALID) == 64
    assert int(FeatureQualityFlag.DENSITY_UNDEFINED) == 128


def test_point_feature_set_requires_exact_shapes_and_dtypes() -> None:
    features = make_feature_set()
    assert features.point_count == 2
    assert features.array_metadata()["local_density"]["unit"] == "points/m^3"

    with pytest.raises(ValueError, match="dtype float32"):
        PointFeatureSet(**{**features.__dict__, "linearity": np.zeros(2, dtype=np.float64)})


def test_geometry_feature_result_round_trip_restores_paths() -> None:
    result = GeometryFeatureResult(
        source_id="scene_simple",
        manifest_source_id="raw_scene",
        input_a1_npz_path=Path("input.npz"),
        input_a1_json_path=Path("input.json"),
        input_a1_npz_sha256="a" * 64,
        input_a1_json_sha256="b" * 64,
        output_npz_path=Path("output.npz"),
        output_json_path=Path("output.json"),
        output_npz_sha256="c" * 64,
        output_npz_bytes=123,
        origin_xyz=(0.0, 1.0, 2.0),
        config=GeometryFeatureConfig(),
        statistics=make_statistics(),
        arrays=make_feature_set().array_metadata(),
        quality_flag_counts={"NONE": 2},
        formulas={"height_m": "xyz_local[:, 2]"},
        algorithm_version="a2a-local-pca-v1",
        created_at="2026-07-21T00:00:00+00:00",
    )

    restored = GeometryFeatureResult.from_dict(result.to_dict())

    assert restored.input_a1_npz_path == Path("input.npz")
    assert restored.statistics.output_point_count == 2
