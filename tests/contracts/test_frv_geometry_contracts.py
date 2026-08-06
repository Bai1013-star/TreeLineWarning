"""Focused FRV-A2 P-GEOMETRY contract tests (PLAN-FROZEN)."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import FrozenInstanceError

import core.contracts as contracts
import core.contracts.frv as frv
import core.contracts.frv_geometry as geometry
import numpy as np
import pytest
from core.contracts.frv_a2 import (
    FrvA2ArtifactComponentBinding,
    FrvA2ArtifactComponentKind,
    FrvA2DomainKind,
    FrvA2FailurePhase,
    FrvA2FailureRecord,
    FrvA2ProviderRole,
)
from core.contracts.frv_geometry import (
    FRV_GEOMETRY_CACHE_BYTES,
    FRV_GEOMETRY_PROVIDER_KEY,
    FrvGeometryChunk,
    FrvGeometryEvidenceFlag,
    FrvGeometryEvidenceState,
    FrvGeometryLoadError,
    FrvGeometryProviderConfig,
    FrvGeometryProviderProof,
    FrvGeometryProviderResult,
    derive_frv_geometry_provider_config_digest,
    validate_frv_geometry_chunk,
    validate_frv_geometry_provider_config,
)

SHA = "a" * 64

_RADII = (0.30, 0.75, 1.50)
_VOLUMES = [(4.0 / 3.0) * math.pi * radius**3 for radius in _RADII]


def _config(**overrides) -> FrvGeometryProviderConfig:
    defaults: dict = {
        "schema_version": geometry.FRV_GEOMETRY_CONFIG_SCHEMA_VERSION,
        "provider_key": FRV_GEOMETRY_PROVIDER_KEY,
        "provider_role": FrvA2ProviderRole.GEOMETRY,
        "interface_version": geometry.FRV_GEOMETRY_PROVIDER_INTERFACE_VERSION,
        "algorithm_version": geometry.FRV_GEOMETRY_ALGORITHM_VERSION,
        "radii_m": _RADII,
        "min_neighbor_counts": (8, 12, 20),
        "max_neighbors_per_scale": 64,
        "index_cell_size_m": 0.75,
        "support_neighbor_radius_m": 0.30,
        "support_neighbor_cap": 16,
        "direction_cosine_min": 0.75,
        "eigenvalue_floor_m2": 1e-12,
        "canopy_scattering_min": 0.08,
        "canopy_linearity_max": 0.90,
        "canopy_density_min_per_m3": 2.0,
        "woody_linearity_min": 0.70,
        "woody_verticality_max": 0.35,
        "woody_vertical_span_min_m": 0.75,
        "vertical_cylinder_radius_m": 0.30,
        "vertical_half_window_m": 1.50,
        "vertical_slice_height_m": 0.25,
        "min_vertical_occupied_slices": 3,
        "ground_planarity_min": 0.70,
        "ground_normal_z_min": 0.85,
        "max_index_cell_points": 1000000,
    }
    explicit = overrides.pop("provider_config_digest", None)
    defaults.update(overrides)
    if explicit is not None:
        defaults["provider_config_digest"] = explicit
    else:
        defaults["provider_config_digest"] = derive_frv_geometry_provider_config_digest(
            schema_version=defaults["schema_version"],
            provider_key=defaults["provider_key"],
            provider_role=defaults["provider_role"],
            interface_version=defaults["interface_version"],
            algorithm_version=defaults["algorithm_version"],
            radii_m=defaults["radii_m"],
            min_neighbor_counts=defaults["min_neighbor_counts"],
            max_neighbors_per_scale=defaults["max_neighbors_per_scale"],
            index_cell_size_m=defaults["index_cell_size_m"],
            support_neighbor_radius_m=defaults["support_neighbor_radius_m"],
            support_neighbor_cap=defaults["support_neighbor_cap"],
            direction_cosine_min=defaults["direction_cosine_min"],
            eigenvalue_floor_m2=defaults["eigenvalue_floor_m2"],
            canopy_scattering_min=defaults["canopy_scattering_min"],
            canopy_linearity_max=defaults["canopy_linearity_max"],
            canopy_density_min_per_m3=defaults["canopy_density_min_per_m3"],
            woody_linearity_min=defaults["woody_linearity_min"],
            woody_verticality_max=defaults["woody_verticality_max"],
            woody_vertical_span_min_m=defaults["woody_vertical_span_min_m"],
            vertical_cylinder_radius_m=defaults["vertical_cylinder_radius_m"],
            vertical_half_window_m=defaults["vertical_half_window_m"],
            vertical_slice_height_m=defaults["vertical_slice_height_m"],
            min_vertical_occupied_slices=defaults["min_vertical_occupied_slices"],
            ground_planarity_min=defaults["ground_planarity_min"],
            ground_normal_z_min=defaults["ground_normal_z_min"],
            max_index_cell_points=defaults["max_index_cell_points"],
        )
    return FrvGeometryProviderConfig(**defaults)


def _ro(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _density_row(raw_counts: list[int]) -> np.ndarray:
    values = [raw_counts[scale] / _VOLUMES[scale] for scale in range(3)]
    return np.asarray(values, dtype="<f4")


def _chunk(**overrides) -> FrvGeometryChunk:
    point_count = overrides.get("point_count", 3)
    defaults: dict = {
        "source_id": "scene-1",
        "chunk_id": 0,
        "start_index": 0,
        "point_count": point_count,
        "original_index": _ro(np.arange(point_count, dtype="<i8")),
        "raw_neighbor_count": _ro(np.tile([8, 12, 20], (point_count, 1)).astype("<u4")),
        "neighbor_count": _ro(np.tile([8, 12, 20], (point_count, 1)).astype("<u2")),
        "density_per_m3": _ro(np.tile(_density_row([8, 12, 20]), (point_count, 1))),
        "local_spacing_m": _ro(np.full((point_count, 3), 0.1, dtype="<f4")),
        "eigenvalues_m2": _ro(
            np.tile([0.1, 0.3, 0.6], (point_count, 3, 1)).astype("<f8")
        ),
        "principal_direction": _ro(
            np.tile([1.0, 0.0, 0.0], (point_count, 3, 1)).astype("<f4")
        ),
        "surface_normal": _ro(
            np.tile([0.0, 0.0, 1.0], (point_count, 3, 1)).astype("<f4")
        ),
        "linearity": _ro(np.full((point_count, 3), 0.5, dtype="<f4")),
        "planarity": _ro(np.full((point_count, 3), 1.0 / 3.0, dtype="<f4")),
        "scattering": _ro(np.full((point_count, 3), 1.0 / 6.0, dtype="<f4")),
        "verticality": _ro(np.full((point_count, 3), 1.0, dtype="<f4")),
        "vertical_span_m": _ro(np.full(point_count, 1.0, dtype="<f4")),
        "vertical_occupied_slices": _ro(np.full(point_count, 3, dtype="<u2")),
        "above_count": _ro(np.ones(point_count, dtype="<u2")),
        "below_count": _ro(np.ones(point_count, dtype="<u2")),
        "support_neighbor_count": _ro(np.zeros(point_count, dtype="|u1")),
        "support_neighbor_original_index": _ro(
            np.full((point_count, 16), -1, dtype="<i8")
        ),
        "geometry_evidence_flags": _ro(np.full(point_count, 33, dtype="<u4")),
        "geometry_evidence_state": _ro(
            np.full(point_count, int(FrvGeometryEvidenceState.VALID), dtype="|u1")
        ),
        "reason_flags": _ro(np.zeros(point_count, dtype="<u4")),
        "quality_flags": _ro(np.zeros(point_count, dtype="<u4")),
        "is_last": True,
    }
    defaults.update(overrides)
    return FrvGeometryChunk(**defaults)


def test_enum_exact_values() -> None:
    assert int(FrvGeometryEvidenceFlag.NEIGHBORHOOD_SUFFICIENT) == 1
    assert int(FrvGeometryEvidenceFlag.DENSITY_SUPPORTED) == 2
    assert int(FrvGeometryEvidenceFlag.CANOPY_BRANCHING_COMPATIBLE) == 4
    assert int(FrvGeometryEvidenceFlag.WOODY_COLUMN_COMPATIBLE) == 8
    assert int(FrvGeometryEvidenceFlag.GROUND_SURFACE_COMPATIBLE) == 16
    assert int(FrvGeometryEvidenceFlag.VERTICAL_CONTINUITY) == 32
    assert int(FrvGeometryEvidenceFlag.NONE) == 0
    assert int(FrvGeometryEvidenceState.VALID) == 1
    assert int(FrvGeometryEvidenceState.INSUFFICIENT) == 2
    assert int(FrvGeometryEvidenceState.DEGENERATE) == 3
    assert int(geometry.FrvGeometryReasonFlag.INSUFFICIENT_NEIGHBORS) == 1
    assert int(geometry.FrvGeometryReasonFlag.DEGENERATE_COVARIANCE) == 2
    assert int(geometry.FrvGeometryQualityFlag.NEIGHBOR_CAP_REACHED) == 1
    assert int(geometry.FrvGeometryQualityFlag.SUPPORT_NEIGHBOR_CAP_REACHED) == 2


def test_config_defaults_and_exact_digest() -> None:
    config = _config()
    assert config.schema_version == geometry.FRV_GEOMETRY_CONFIG_SCHEMA_VERSION
    assert config.provider_key == FRV_GEOMETRY_PROVIDER_KEY
    assert config.provider_role is FrvA2ProviderRole.GEOMETRY
    assert config.radii_m == (0.30, 0.75, 1.50)
    assert config.min_neighbor_counts == (8, 12, 20)
    assert config.max_neighbors_per_scale == 64
    assert config.index_cell_size_m == 0.75
    assert config.support_neighbor_radius_m == 0.30
    assert config.support_neighbor_cap == 16
    assert config.direction_cosine_min == 0.75
    assert config.eigenvalue_floor_m2 == 1e-12
    assert config.canopy_scattering_min == 0.08
    assert config.canopy_linearity_max == 0.90
    assert config.canopy_density_min_per_m3 == 2.0
    assert config.woody_linearity_min == 0.70
    assert config.woody_verticality_max == 0.35
    assert config.woody_vertical_span_min_m == 0.75
    assert config.vertical_cylinder_radius_m == 0.30
    assert config.vertical_half_window_m == 1.50
    assert config.vertical_slice_height_m == 0.25
    assert config.min_vertical_occupied_slices == 3
    assert config.ground_planarity_min == 0.70
    assert config.ground_normal_z_min == 0.85
    assert config.max_index_cell_points == 1000000
    validate_frv_geometry_provider_config(config)
    assert config.provider_config_digest == derive_frv_geometry_provider_config_digest(
        schema_version=config.schema_version,
        provider_key=config.provider_key,
        provider_role=config.provider_role,
        interface_version=config.interface_version,
        algorithm_version=config.algorithm_version,
        radii_m=config.radii_m,
        min_neighbor_counts=config.min_neighbor_counts,
        max_neighbors_per_scale=config.max_neighbors_per_scale,
        index_cell_size_m=config.index_cell_size_m,
        support_neighbor_radius_m=config.support_neighbor_radius_m,
        support_neighbor_cap=config.support_neighbor_cap,
        direction_cosine_min=config.direction_cosine_min,
        eigenvalue_floor_m2=config.eigenvalue_floor_m2,
        canopy_scattering_min=config.canopy_scattering_min,
        canopy_linearity_max=config.canopy_linearity_max,
        canopy_density_min_per_m3=config.canopy_density_min_per_m3,
        woody_linearity_min=config.woody_linearity_min,
        woody_verticality_max=config.woody_verticality_max,
        woody_vertical_span_min_m=config.woody_vertical_span_min_m,
        vertical_cylinder_radius_m=config.vertical_cylinder_radius_m,
        vertical_half_window_m=config.vertical_half_window_m,
        vertical_slice_height_m=config.vertical_slice_height_m,
        min_vertical_occupied_slices=config.min_vertical_occupied_slices,
        ground_planarity_min=config.ground_planarity_min,
        ground_normal_z_min=config.ground_normal_z_min,
        max_index_cell_points=config.max_index_cell_points,
    )


def test_config_exact_field_order() -> None:
    assert [field.name for field in dataclasses.fields(FrvGeometryProviderConfig)] == [
        "schema_version",
        "provider_key",
        "provider_role",
        "interface_version",
        "algorithm_version",
        "radii_m",
        "min_neighbor_counts",
        "max_neighbors_per_scale",
        "index_cell_size_m",
        "support_neighbor_radius_m",
        "support_neighbor_cap",
        "direction_cosine_min",
        "eigenvalue_floor_m2",
        "canopy_scattering_min",
        "canopy_linearity_max",
        "canopy_density_min_per_m3",
        "woody_linearity_min",
        "woody_verticality_max",
        "woody_vertical_span_min_m",
        "vertical_cylinder_radius_m",
        "vertical_half_window_m",
        "vertical_slice_height_m",
        "min_vertical_occupied_slices",
        "ground_planarity_min",
        "ground_normal_z_min",
        "max_index_cell_points",
        "provider_config_digest",
    ]


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        _config(index_cell_size_m=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _config(index_cell_size_m=1)  # type: ignore[arg-type]  # int not float
    with pytest.raises(ValueError):
        _config(min_neighbor_counts=(8, 12, True))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _config(min_neighbor_counts=(8, 12, np.int64(20)))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _config(max_index_cell_points=np.int64(5))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _config(radii_m=(0.75, 0.30, 1.50))  # not strictly increasing
    _config(radii_m=(0.30, 0.75, 1.50))  # sanity: the frozen defaults are legal
    with pytest.raises(ValueError):
        _config(min_neighbor_counts=(0, 12, 20))  # nonpositive
    with pytest.raises(ValueError):
        _config(min_neighbor_counts=(8, 12, 65))  # exceeds 64
    with pytest.raises(ValueError):
        _config(direction_cosine_min=1.5)
    with pytest.raises(ValueError):
        _config(canopy_linearity_max=1.1)
    with pytest.raises(ValueError):
        _config(support_neighbor_radius_m=0.5)  # != radii_m[0]
    with pytest.raises(ValueError):
        _config(vertical_half_window_m=1.25)  # != radii_m[2]
    with pytest.raises(ValueError):
        _config(vertical_slice_height_m=0.5)  # 2*hw/slice != 12
    with pytest.raises(ValueError):
        _config(eigenvalue_floor_m2=0.0)
    with pytest.raises(ValueError):
        _config(provider_role=FrvA2ProviderRole.SOURCE_SCAN)
    with pytest.raises(ValueError):
        _config(provider_config_digest="not-a-sha")
    with pytest.raises(ValueError):
        _config(max_neighbors_per_scale=0)


def test_config_frozen() -> None:
    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.index_cell_size_m = 2.0  # type: ignore[misc]


def test_config_from_dict_exact_keys() -> None:
    config = _config()
    restored = FrvGeometryProviderConfig.from_dict(config.to_dict())
    assert restored == config
    payload = config.to_dict()
    del payload["provider_key"]
    with pytest.raises(ValueError, match="missing"):
        FrvGeometryProviderConfig.from_dict(payload)
    payload = config.to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        FrvGeometryProviderConfig.from_dict(payload)


def test_chunk_exact_field_order() -> None:
    assert [field.name for field in dataclasses.fields(FrvGeometryChunk)] == [
        "source_id",
        "chunk_id",
        "start_index",
        "point_count",
        "original_index",
        "raw_neighbor_count",
        "neighbor_count",
        "density_per_m3",
        "local_spacing_m",
        "eigenvalues_m2",
        "principal_direction",
        "surface_normal",
        "linearity",
        "planarity",
        "scattering",
        "verticality",
        "vertical_span_m",
        "vertical_occupied_slices",
        "above_count",
        "below_count",
        "support_neighbor_count",
        "support_neighbor_original_index",
        "geometry_evidence_flags",
        "geometry_evidence_state",
        "reason_flags",
        "quality_flags",
        "is_last",
    ]


def test_chunk_normal_construction_and_read_only() -> None:
    chunk = _chunk()
    assert chunk.point_count == 3
    assert chunk.is_last is True
    assert not chunk.original_index.flags.writeable
    assert chunk.original_index.dtype == np.dtype("<i8")
    assert chunk.density_per_m3.dtype == np.dtype("<f4")
    assert chunk.eigenvalues_m2.dtype == np.dtype("<f8")
    assert chunk.eigenvalues_m2.shape == (3, 3, 3)
    assert chunk.support_neighbor_original_index.shape == (3, 16)
    assert chunk.support_neighbor_original_index.dtype == np.dtype("<i8")
    assert chunk.geometry_evidence_state.dtype == np.dtype("|u1")
    assert chunk.geometry_evidence_flags.dtype == np.dtype("<u4")
    with pytest.raises(FrozenInstanceError):
        chunk.source_id = "other"  # type: ignore[misc]
    # the default chunk is a legal VALID chunk
    validate_frv_geometry_chunk(chunk, provider_config=_config())


def test_chunk_rejects_empty_and_bad_arrays() -> None:
    with pytest.raises(ValueError):
        _chunk(point_count=0)
    with pytest.raises(ValueError):
        _chunk(point_count=-1)
    with pytest.raises(ValueError):
        _chunk(density_per_m3=_ro(np.full((3, 3), float("nan"), dtype="<f4")))
    with pytest.raises(ValueError):
        _chunk(local_spacing_m=_ro(np.full((3, 3), float("inf"), dtype="<f4")))
    with pytest.raises(ValueError):
        _chunk(density_per_m3=np.zeros((3, 3), dtype="<f4"))  # writable
    with pytest.raises(ValueError):
        _chunk(geometry_evidence_flags=_ro(np.zeros((3, 3), dtype="<u4")))  # bad shape
    # non-increasing original_index is rejected by the chunk validator
    non_increasing = _chunk(original_index=_ro(np.zeros(3, dtype="<i8")))
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_frv_geometry_chunk(non_increasing, provider_config=_config())


def test_chunk_rejects_negative_zero() -> None:
    negative_zero = np.zeros((3, 3), dtype="<f4")
    negative_zero[0, 0] = -0.0
    negative_zero.setflags(write=False)
    with pytest.raises(ValueError, match="negative zero"):
        _chunk(density_per_m3=negative_zero)
    with pytest.raises(ValueError, match="negative zero"):
        _chunk(linearity=negative_zero)
    eigen_zero = np.zeros((3, 3, 3), dtype="<f8")
    eigen_zero[0, 0, 0] = -0.0
    eigen_zero.setflags(write=False)
    with pytest.raises(ValueError, match="negative zero"):
        _chunk(eigenvalues_m2=eigen_zero)


def test_chunk_validator_state_and_reason_flags() -> None:
    config = _config()

    def single(**overrides):
        return _chunk(
            point_count=1,
            original_index=_ro(np.zeros(1, dtype="<i8")),
            is_last=True,
            **overrides,
        )

    # INSUFFICIENT: medium below its minimum, so the whole scale-1 block is +0.0
    insufficient = single(
        raw_neighbor_count=_ro(np.array([[8, 11, 20]], dtype="<u4")),
        neighbor_count=_ro(np.array([[8, 11, 20]], dtype="<u2")),
        density_per_m3=_ro(np.array([_density_row([8, 11, 20])], dtype="<f4")),
        eigenvalues_m2=_ro(
            np.array([[[0.1, 0.3, 0.6], [0.0, 0.0, 0.0], [0.1, 0.3, 0.6]]], dtype="<f8")
        ),
        linearity=_ro(np.array([[0.5, 0.0, 0.5]], dtype="<f4")),
        planarity=_ro(np.array([[1.0 / 3.0, 0.0, 1.0 / 3.0]], dtype="<f4")),
        scattering=_ro(np.array([[1.0 / 6.0, 0.0, 1.0 / 6.0]], dtype="<f4")),
        verticality=_ro(np.array([[1.0, 0.0, 1.0]], dtype="<f4")),
        principal_direction=_ro(
            np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype="<f4")
        ),
        surface_normal=_ro(
            np.array([[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype="<f4")
        ),
        geometry_evidence_state=_ro(
            np.array([int(FrvGeometryEvidenceState.INSUFFICIENT)], dtype="|u1")
        ),
        reason_flags=_ro(
            np.array(
                [int(geometry.FrvGeometryReasonFlag.INSUFFICIENT_NEIGHBORS)], dtype="<u4"
            )
        ),
        geometry_evidence_flags=_ro(np.array([32], dtype="<u4")),  # vertical only
    )
    validate_frv_geometry_chunk(insufficient, provider_config=config)

    # DEGENERATE: counts OK but medium lambda2 below the floor -> +0.0 medium block
    degenerate = single(
        eigenvalues_m2=_ro(
            np.array(
                [
                    [
                        [0.1, 0.3, 0.6],
                        [0.0, 0.0, 0.0],
                        [0.1, 0.3, 0.6],
                    ]
                ],
                dtype="<f8",
            )
        ),
        linearity=_ro(np.array([[0.5, 0.0, 0.5]], dtype="<f4")),
        planarity=_ro(np.array([[1.0 / 3.0, 0.0, 1.0 / 3.0]], dtype="<f4")),
        scattering=_ro(np.array([[1.0 / 6.0, 0.0, 1.0 / 6.0]], dtype="<f4")),
        verticality=_ro(np.array([[1.0, 0.0, 1.0]], dtype="<f4")),
        principal_direction=_ro(
            np.array(
                [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
                dtype="<f4",
            )
        ),
        surface_normal=_ro(
            np.array(
                [[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
                dtype="<f4",
            )
        ),
        geometry_evidence_state=_ro(
            np.array([int(FrvGeometryEvidenceState.DEGENERATE)], dtype="|u1")
        ),
        reason_flags=_ro(
            np.array(
                [int(geometry.FrvGeometryReasonFlag.DEGENERATE_COVARIANCE)], dtype="<u4"
            )
        ),
        geometry_evidence_flags=_ro(np.array([32], dtype="<u4")),  # vertical only
    )
    validate_frv_geometry_chunk(degenerate, provider_config=config)

    # reason flags that do not match the state are rejected
    bad_reason = dataclasses.replace(
        degenerate,
        reason_flags=_ro(
            np.array([int(geometry.FrvGeometryReasonFlag.INSUFFICIENT_NEIGHBORS)], dtype="<u4")
        ),
    )
    with pytest.raises(ValueError, match="reason_flags"):
        validate_frv_geometry_chunk(bad_reason, provider_config=config)


def test_chunk_validator_recomputes_evidence_flags() -> None:
    config = _config()

    def single(**overrides):
        return _chunk(
            point_count=1,
            original_index=_ro(np.zeros(1, dtype="<i8")),
            is_last=True,
            **overrides,
        )

    # Raise large raw count so large density crosses the 2.0 threshold and the
    # large scattering/linearity enter the canopy band.
    base = single(
        raw_neighbor_count=_ro(np.array([[8, 12, 40]], dtype="<u4")),
        neighbor_count=_ro(np.array([[8, 12, 40]], dtype="<u2")),
        density_per_m3=_ro(np.array([_density_row([8, 12, 40])], dtype="<f4")),
        eigenvalues_m2=_ro(
            np.array([[[0.1, 0.3, 0.6], [0.2, 0.4, 0.6], [0.2, 0.4, 0.6]]], dtype="<f8")
        ),
        linearity=_ro(np.array([[0.5, 1.0 / 3.0, 1.0 / 3.0]], dtype="<f4")),
        planarity=_ro(np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype="<f4")),
        scattering=_ro(np.array([[1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0]], dtype="<f4")),
        verticality=_ro(np.array([[1.0, 1.0, 1.0]], dtype="<f4")),
        principal_direction=_ro(
            np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype="<f4")
        ),
        surface_normal=_ro(
            np.array([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype="<f4")
        ),
    )
    # 40 / (4/3*pi*1.5^3) = 2.829 >= 2.0, scattering 1/3 >= 0.08,
    # linearity 1/3 <= 0.90 -> CANOPY; vertical continuity holds -> 1|2|4|32
    expected = (
        int(FrvGeometryEvidenceFlag.NEIGHBORHOOD_SUFFICIENT)
        | int(FrvGeometryEvidenceFlag.DENSITY_SUPPORTED)
        | int(FrvGeometryEvidenceFlag.CANOPY_BRANCHING_COMPATIBLE)
        | int(FrvGeometryEvidenceFlag.VERTICAL_CONTINUITY)
    )
    valid = dataclasses.replace(
        base,
        geometry_evidence_flags=_ro(np.array([expected], dtype="<u4")),
        geometry_evidence_state=_ro(
            np.array([int(FrvGeometryEvidenceState.VALID)], dtype="|u1")
        ),
    )
    validate_frv_geometry_chunk(valid, provider_config=config)
    # a wrong flag mask is rejected
    wrong = dataclasses.replace(
        valid,
        geometry_evidence_flags=_ro(np.array([expected - 2], dtype="<u4")),
    )
    with pytest.raises(ValueError, match="geometry_evidence_flags"):
        validate_frv_geometry_chunk(wrong, provider_config=config)


def test_chunk_validator_support_structure() -> None:
    config = _config()

    def single(**overrides):
        return _chunk(
            point_count=1,
            original_index=_ro(np.zeros(1, dtype="<i8")),
            is_last=True,
            **overrides,
        )

    support = single(
        support_neighbor_count=_ro(np.array([1], dtype="|u1")),
        support_neighbor_original_index=_ro(
            np.array([[5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype="<i8")
        ),
    )
    validate_frv_geometry_chunk(support, provider_config=config)
    # self-support is rejected
    self_support = single(
        support_neighbor_count=_ro(np.array([1], dtype="|u1")),
        support_neighbor_original_index=_ro(
            np.array([[0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype="<i8")
        ),
    )
    with pytest.raises(ValueError, match="non-self"):
        validate_frv_geometry_chunk(self_support, provider_config=config)
    # used slots beyond the declared count must be -1
    overrun = single(
        support_neighbor_count=_ro(np.array([1], dtype="|u1")),
        support_neighbor_original_index=_ro(
            np.array([[5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype="<i8")
        ),
    )
    with pytest.raises(ValueError, match="unused support"):
        validate_frv_geometry_chunk(overrun, provider_config=config)


def test_point_content_digest_oracle() -> None:
    chunk = _chunk(point_count=2, is_last=True)
    digest = geometry.frv_geometry_point_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=SHA,
        provider_config_digest=SHA,
        source_point_count=2,
        chunks=[chunk],
    )
    assert len(digest) == 64
    with pytest.raises(ValueError):
        geometry.frv_geometry_point_content_digest(
            source_content_identity_digest=SHA,
            source_identity_record_digest=SHA,
            provider_config_digest=SHA,
            source_point_count=3,  # chunks only cover 0..1
            chunks=[chunk],
        )


def test_result_no_public_constructor() -> None:
    with pytest.raises(TypeError):
        FrvGeometryProviderResult()  # type: ignore[call-arg]


def test_result_factory_and_capability() -> None:
    from tests.contracts.test_frv_source_contracts import _source_identity

    source_identity = _source_identity(path=r"C:\__geom__\s.las")
    proof = _proof_for_identity(source_identity)
    result = geometry._create_finalized_frv_geometry_provider_result(
        proof, source_identity, capability=geometry._FRV_GEOMETRY_FINALIZED_CAPABILITY
    )
    geometry.validate_finalized_frv_geometry_provider_result(result)
    assert "capability" not in repr(result)
    with pytest.raises(ValueError, match="capability"):
        geometry._create_finalized_frv_geometry_provider_result(
            proof, source_identity, capability="wrong"
        )
    forged = object.__new__(FrvGeometryProviderResult)
    object.__setattr__(forged, "proof", proof)
    object.__setattr__(forged, "source_identity", source_identity)
    object.__setattr__(forged, "_finalized_capability", object())
    with pytest.raises(ValueError, match="capability"):
        geometry.validate_finalized_frv_geometry_provider_result(forged)


def test_load_error_construction() -> None:
    record = FrvA2FailureRecord(
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        FrvA2ProviderRole.GEOMETRY,
        FRV_GEOMETRY_PROVIDER_KEY,
        "source_result",
        frv.FrvErrorCode.FRV_PROVIDER_INVALID,
        int(frv.FrvReasonFlag.PROVIDER_UNCERTAINTY),
        "structure_integrity:geometry:geometry_provider:source_result:FRV_PROVIDER_INVALID:1024",
    )
    error = FrvGeometryLoadError(
        failure_record=record,
        quality_flags=int(frv.FrvQualityFlag.PROVIDER_INVALID),
    )
    assert str(error) == record.detail
    assert error.args == (record.detail,)
    with pytest.raises(TypeError):
        FrvGeometryLoadError(failure_record="x", quality_flags=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FrvGeometryLoadError(failure_record=record, quality_flags=-1)


def _valid_proof() -> FrvGeometryProviderProof:
    from tests.contracts.test_frv_source_contracts import _source_identity

    return _proof_for_identity(_source_identity(path=r"C:\__geom__\s.las"))


def _proof_for_identity(source_identity) -> FrvGeometryProviderProof:
    source_content = source_identity.source_content_identity_digest
    source_record = source_identity.source_identity_record_digest
    count = int(source_identity.source_point_count)
    proof = FrvGeometryProviderProof.__new__(FrvGeometryProviderProof)
    fields = {
        "schema_version": geometry.FRV_GEOMETRY_PROOF_SCHEMA_VERSION,
        "provider_key": FRV_GEOMETRY_PROVIDER_KEY,
        "provider_role": FrvA2ProviderRole.GEOMETRY,
        "interface_version": geometry.FRV_GEOMETRY_PROVIDER_INTERFACE_VERSION,
        "algorithm_version": geometry.FRV_GEOMETRY_ALGORITHM_VERSION,
        "configuration_digest": SHA,
        "provider_config_digest": _config().provider_config_digest,
        "source_artifact_reference_id": (
            f"frv-a2-ref-v1|source_scan|source_scan|source_file,source_file,{SHA},123"
        ),
        "artifact_reference_id": (
            f"frv-a2-ref-v1|geometry|geometry_provider|source_file,source_file,{SHA},123"
        ),
        "artifact_components": (
            FrvA2ArtifactComponentBinding(
                component_key="source_file",
                component_kind=FrvA2ArtifactComponentKind.SOURCE_FILE,
                sha256=source_identity.source_file_sha256,
                size_bytes=123,
            ),
        ),
        "source_content_identity_digest": source_content,
        "source_identity_record_digest": source_record,
        "source_proof_digest": SHA,
        "source_binding_digest": SHA,
        "source_point_count": count,
        "domain_kind": FrvA2DomainKind.FULL_SOURCE,
        "domain_count": count,
        "original_index_domain_digest": contracts.frv_a2_index_domain_digest(
            "ORIGINAL_INDEX", source_content, FrvA2DomainKind.FULL_SOURCE, tuple(range(count))
        ),
        "geometry_point_content_digest": SHA,
        "valid_count": count,
        "insufficient_count": 0,
        "degenerate_count": 0,
        "neighborhood_sufficient_count": count,
        "density_supported_count": 0,
        "canopy_branching_compatible_count": 0,
        "woody_column_compatible_count": 0,
        "ground_surface_compatible_count": 0,
        "vertical_continuity_count": 0,
        "neighbor_cap_reached_count": 0,
        "support_neighbor_cap_reached_count": 0,
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "origin_xyz": (0.0, 0.0, 0.0),
        "transform_declaration": "identity",
        "quality_flags": 0,
        "reason_flags": 0,
        "error_codes": (),
    }
    for name, value in fields.items():
        object.__setattr__(proof, name, value)
    object.__setattr__(proof, "proof_digest", proof.computed_digest())
    return proof


def _forged_proof(**tamper) -> FrvGeometryProviderProof:
    from tests.contracts.test_frv_source_contracts import _source_identity

    identity = _source_identity(path=r"C:\__geom__\s.las")
    proof = _proof_for_identity(identity)
    for name, value in tamper.items():
        object.__setattr__(proof, name, value)
    object.__setattr__(proof, "proof_digest", proof.computed_digest())
    return proof


def test_forged_rehashed_proof_rejected() -> None:
    from core.contracts.frv_geometry import validate_frv_geometry_provider_proof

    cases = [
        {"provider_role": FrvA2ProviderRole.SOURCE_SCAN},
        {"provider_key": "other"},
        {"schema_version": "wrong"},
        {"domain_kind": FrvA2DomainKind.DECLARED_SUBSET},
        {"coordinate_frame": "OTHER"},
        {"units": "ft"},
        {"transform_declaration": "non_identity"},
        {"source_point_count": 999},  # breaks conservation with counts
        {"valid_count": 1, "degenerate_count": 1},  # state sum mismatch
        {"neighborhood_sufficient_count": 0},  # != valid_count
        {"quality_flags": int(frv.FrvQualityFlag.PROVIDER_INVALID)},
        {"error_codes": (frv.FrvErrorCode.FRV_PROVIDER_INVALID,)},
        {
            "artifact_components": (
                FrvA2ArtifactComponentBinding(
                    component_key="source_file",
                    component_kind=FrvA2ArtifactComponentKind.SOURCE_FILE,
                    sha256=SHA,
                    size_bytes=999,
                ),
            ),
        },
    ]
    for tamper in cases:
        forged = _forged_proof(**tamper)
        with pytest.raises(ValueError):
            validate_frv_geometry_provider_proof(forged)


def test_bnd1_adapter_for_finalized_result() -> None:
    from tests.contracts.test_frv_source_contracts import _source_identity

    source_identity = _source_identity(path=r"C:\__geom__\s.las")
    proof = _proof_for_identity(source_identity)
    result = geometry._create_finalized_frv_geometry_provider_result(
        proof, source_identity, capability=geometry._FRV_GEOMETRY_FINALIZED_CAPABILITY
    )
    evidence = geometry.to_frv_a2_geometry_provider_binding_evidence(result)
    contracts.validate_frv_a2_provider_binding_evidence(evidence)
    assert evidence.provider_key == FRV_GEOMETRY_PROVIDER_KEY
    assert evidence.provider_role is FrvA2ProviderRole.GEOMETRY
    assert evidence.provider_readiness.value == "validated"
    assert evidence.public_module == "algorithms.vegetation.geometry_provider"
    assert evidence.public_loader_name == "load_frv_geometry"
    # upstream digests are the sorted unique geometry/source binding triple
    expected_upstream = tuple(
        sorted(
            {
                proof.geometry_point_content_digest,
                proof.source_binding_digest,
                proof.source_proof_digest,
            }
        )
    )
    assert evidence.upstream_binding_digests == expected_upstream
    assert evidence.binding_digest == contracts.frv_a2_provider_binding_digest(evidence)


def test_cache_bytes_constant() -> None:
    assert FRV_GEOMETRY_CACHE_BYTES == 536870912
