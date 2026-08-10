"""Focused FAST-only tests for V3 positive admission and exclusions."""

from __future__ import annotations

import algorithms.vegetation.fast_vegetation_v3 as fast_v3
import numpy as np
from algorithms.vegetation.fast_vegetation_v3 import (
    FastV3DiagnosticState,
    FastVegetationV3Config,
    _assert_v3_selection,
    _dense_grid_component_labels_reference,
    _diagnostic_states,
    _grid_component_labels,
    _positive_v1_admission,
    _rigid_linear_exclusion,
    _rigid_surface_exclusion,
    _W1Metrics,
    _wire_geometry_exclusion,
)


def _wire_metrics(count: int, *, point_axis: str = "x") -> _W1Metrics:
    coordinate = np.arange(count, dtype=np.float64) * 0.025
    tangent = np.zeros((count, 3), dtype=np.float64)
    tangent[:, 0] = 1.0
    return _W1Metrics(
        distance_m=np.full(count, 0.04, dtype=np.float64),
        nearest_segment=np.zeros(count, dtype=np.int32),
        wire_id=np.zeros(count, dtype=np.int32),
        tangent_xyz=tangent,
        along_wire_m=coordinate,
    )


def _wire_points(count: int, *, axis: str = "x") -> np.ndarray:
    points = np.zeros((count, 3), dtype=np.float64)
    points[:, 0 if axis == "x" else 1] = np.arange(count, dtype=np.float64) * 0.025
    points[:, 2] = 7.5
    return points


def test_v3_geometric_wire_continuation_is_excluded_deterministically() -> None:
    count = 24
    kwargs = {
        "hag": np.full(count, 7.5, dtype=np.float32),
        "linearity": np.full(count, 0.8, dtype=np.float32),
        "scattering": np.full(count, 0.05, dtype=np.float32),
        "positive_v1_support": np.zeros(count, dtype=bool),
        "w1": _wire_metrics(count),
        "config": FastVegetationV3Config(),
    }
    first = _wire_geometry_exclusion(_wire_points(count), np.ones(count, dtype=bool), **kwargs)
    second = _wire_geometry_exclusion(_wire_points(count), np.ones(count, dtype=bool), **kwargs)
    assert np.array_equal(first[0], np.ones(count, dtype=bool))
    assert np.array_equal(first[0], second[0])
    assert first[2][0]["max_continuous_w1_bins"] >= 6


def test_v3_near_positive_vegetation_is_not_excluded_by_distance_alone() -> None:
    count = 24
    excluded, _component, summaries, _alignment = _wire_geometry_exclusion(
        _wire_points(count),
        np.ones(count, dtype=bool),
        hag=np.full(count, 7.5, dtype=np.float32),
        linearity=np.full(count, 0.8, dtype=np.float32),
        scattering=np.full(count, 0.05, dtype=np.float32),
        positive_v1_support=np.ones(count, dtype=bool),
        w1=_wire_metrics(count),
        config=FastVegetationV3Config(),
    )
    assert not np.any(excluded)
    assert summaries == []


def test_v3_tangent_alignment_discriminates_crossing_structure() -> None:
    count = 24
    excluded, _component, summaries, _alignment = _wire_geometry_exclusion(
        _wire_points(count, axis="y"),
        np.ones(count, dtype=bool),
        hag=np.full(count, 7.5, dtype=np.float32),
        linearity=np.full(count, 0.8, dtype=np.float32),
        scattering=np.full(count, 0.05, dtype=np.float32),
        positive_v1_support=np.zeros(count, dtype=bool),
        w1=_wire_metrics(count),
        config=FastVegetationV3Config(),
    )
    assert not np.any(excluded)
    assert summaries == []


def test_v3_positive_admission_preserves_index_order_and_excludes_forbidden_states() -> None:
    original = np.asarray((2, 5, 9, 12), dtype=np.int64)
    ground = np.asarray((True, False, False, False))
    exact_wire = np.asarray((False, True, False, False))
    wire_geometry = np.asarray((False, False, True, False))
    rigid_linear = np.zeros(4, dtype=bool)
    rigid_surface = np.zeros(4, dtype=bool)
    retained = np.asarray((False, False, False, True))
    state = _diagnostic_states(
        ground_mask=ground,
        exact_wire=exact_wire,
        wire_geometry=wire_geometry,
        rigid_linear=rigid_linear,
        rigid_surface=rigid_surface,
        retained=retained,
    )
    _assert_v3_selection(
        state=state,
        retained=retained,
        ground_mask=ground,
        exact_wire=exact_wire,
        wire_geometry=wire_geometry,
        rigid_linear=rigid_linear,
        rigid_surface=rigid_surface,
    )
    assert np.array_equal(original[retained], np.asarray((12,), dtype=np.int64))
    assert state.tolist() == [
        int(FastV3DiagnosticState.GROUND_EXCLUDED),
        int(FastV3DiagnosticState.EXACT_WIRE_EXCLUDED),
        int(FastV3DiagnosticState.WIRE_GEOMETRY_EXCLUDED),
        int(FastV3DiagnosticState.VEGETATION_RETAINED),
    ]


def test_v3_positive_v1_admission_and_excluded_states_cannot_reach_v4() -> None:
    v2_retained = np.ones(8, dtype=bool)
    positive_v1_support = np.asarray((True, False, True, True, True, True, True, True))
    ground = np.asarray((False, False, True, False, False, False, False, False))
    exact_wire = np.asarray((False, False, False, True, False, False, False, False))
    wire_geometry = np.asarray((False, False, False, False, True, False, False, False))
    rigid_linear = np.asarray((False, False, False, False, False, True, False, False))
    rigid_surface = np.asarray((False, False, False, False, False, False, True, False))
    retained = _positive_v1_admission(
        v2_retained=v2_retained,
        positive_v1_support=positive_v1_support,
        ground_mask=ground,
        exact_wire=exact_wire,
        wire_geometry=wire_geometry,
        rigid_linear=rigid_linear,
        rigid_surface=rigid_surface,
    )
    state = _diagnostic_states(
        ground_mask=ground,
        exact_wire=exact_wire,
        wire_geometry=wire_geometry,
        rigid_linear=rigid_linear,
        rigid_surface=rigid_surface,
        retained=retained,
    )
    assert retained.tolist() == [True, False, False, False, False, False, False, True]
    assert state[1] == int(FastV3DiagnosticState.NO_POSITIVE_VEGETATION_EVIDENCE)
    forbidden_states = np.asarray(
        (
            int(FastV3DiagnosticState.GROUND_EXCLUDED),
            int(FastV3DiagnosticState.EXACT_WIRE_EXCLUDED),
            int(FastV3DiagnosticState.WIRE_GEOMETRY_EXCLUDED),
            int(FastV3DiagnosticState.RIGID_LINEAR_EXCLUDED),
            int(FastV3DiagnosticState.RIGID_SURFACE_EXCLUDED),
        ),
        dtype=np.uint8,
    )
    assert not np.any(np.isin(state[retained], forbidden_states))


def test_v3_component_rejection_identifies_linear_and_surface_objects() -> None:
    linear_count = 40
    linear_xyz = np.zeros((linear_count, 3), dtype=np.float64)
    linear_xyz[:, 0] = np.linspace(0.0, 1.2, linear_count)
    linear_xyz[:, 2] = 0.4
    linear_excluded, _linear_id, linear_summaries = _rigid_linear_exclusion(
        linear_xyz,
        np.ones(linear_count, dtype=bool),
        hag=np.full(linear_count, 0.4, dtype=np.float32),
        linearity=np.full(linear_count, 0.7, dtype=np.float32),
        scattering=np.full(linear_count, 0.08, dtype=np.float32),
        config=FastVegetationV3Config(),
    )
    assert np.all(linear_excluded)
    assert linear_summaries[0]["rigid_linear_excluded"] is True

    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 1.4, 15), np.linspace(0.0, 1.4, 15))
    surface_xyz = np.column_stack(
        (grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, 0.8, dtype=np.float64))
    )
    surface_excluded, _surface_id, surface_summaries = _rigid_surface_exclusion(
        surface_xyz,
        np.ones(len(surface_xyz), dtype=bool),
        hag=np.full(len(surface_xyz), 0.8, dtype=np.float32),
        planarity=np.full(len(surface_xyz), 0.8, dtype=np.float32),
        scattering=np.full(len(surface_xyz), 0.08, dtype=np.float32),
        config=FastVegetationV3Config(),
    )
    assert np.all(surface_excluded)
    assert surface_summaries[0]["rigid_surface_excluded"] is True


def test_v3_invalid_configuration_is_rejected() -> None:
    try:
        FastVegetationV3Config(v1_support_radius_m=0.0)
    except Exception as exc:
        assert "FAST_V3_CONFIG_INVALID" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("invalid V3 configuration must fail")


def _assert_sparse_dense_component_equivalence(xyz: np.ndarray, seed: np.ndarray) -> None:
    sparse = _grid_component_labels(xyz, seed, 1.0)
    dense = _dense_grid_component_labels_reference(xyz, seed, 1.0)
    for sparse_value, dense_value in zip(sparse, dense, strict=True):
        assert np.array_equal(sparse_value, dense_value)


def test_sparse_components_exactly_match_dense_26_connected_reference() -> None:
    compact = np.asarray(((0.1, 0.1, 0.1), (0.9, 0.9, 0.9), (1.1, 0.1, 0.1)))
    disconnected = np.asarray(((0.1, 0.1, 0.1), (4.1, 0.1, 0.1), (4.8, 0.1, 0.1)))
    diagonal = np.asarray(((0.1, 0.1, 0.1), (1.1, 1.1, 1.1)))
    elongated = np.column_stack((np.arange(12, dtype=np.float64), np.zeros(12), np.zeros(12)))
    grid_x, grid_y = np.meshgrid(np.arange(4, dtype=np.float64), np.arange(4, dtype=np.float64))
    planar = np.column_stack((grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)))
    volumetric = np.asarray(
        [(x, y, z) for x in range(3) for y in range(3) for z in range(3)], dtype=np.float64
    )
    for xyz in (compact, disconnected, diagonal, elongated, planar, volumetric):
        _assert_sparse_dense_component_equivalence(xyz, np.ones(len(xyz), dtype=bool))
    diagonal_labels, _cell, _origin = _grid_component_labels(
        diagonal, np.ones(len(diagonal), dtype=bool), 1.0
    )
    assert diagonal_labels.tolist() == [1, 1]


def test_sparse_components_preserve_memberships_and_order_for_unsorted_points() -> None:
    xyz = np.asarray(
        ((4.1, 0.1, 0.1), (0.1, 0.1, 0.1), (4.8, 0.1, 0.1), (0.2, 0.2, 0.2)),
        dtype=np.float64,
    )
    seed = np.asarray((True, True, False, True))
    sparse_labels, sparse_cell, sparse_origin = _grid_component_labels(xyz, seed, 1.0)
    dense_labels, dense_cell, dense_origin = _dense_grid_component_labels_reference(xyz, seed, 1.0)
    assert sparse_labels.tolist() == [2, 1, 0, 1]
    assert np.array_equal(sparse_labels, dense_labels)
    assert np.array_equal(sparse_cell, dense_cell)
    assert np.array_equal(sparse_origin, dense_origin)


def test_sparse_components_handle_huge_empty_extent_without_dense_allocation() -> None:
    xyz = np.asarray(((0.0, 0.0, 0.0), (10_000.0, 10_000.0, 10_000.0)), dtype=np.float64)
    seed = np.ones(len(xyz), dtype=bool)
    diagnostics: dict[str, object] = {}
    labels, _cell, _origin = _grid_component_labels(xyz, seed, 0.1, diagnostics=diagnostics)
    assert labels.tolist() == [1, 2]
    assert diagnostics["occupied_cell_count"] == 2
    assert diagnostics["dense_label_bytes_reference_only"] > 256 * 1024 * 1024
    try:
        _dense_grid_component_labels_reference(xyz, seed, 0.1)
    except Exception as exc:
        assert "DENSE_COMPONENT_GRID_FORBIDDEN" in str(exc)
    else:  # pragma: no cover - the guarded reference must reject this extent
        raise AssertionError("large dense reference allocation must be rejected")


def test_sparse_rigid_exclusions_match_guarded_dense_reference(
    monkeypatch,
) -> None:
    linear_count = 40
    linear_xyz = np.zeros((linear_count, 3), dtype=np.float64)
    linear_xyz[:, 0] = np.linspace(0.0, 1.2, linear_count)
    linear_xyz[:, 2] = 0.4
    linear_kwargs = {
        "hag": np.full(linear_count, 0.4, dtype=np.float32),
        "linearity": np.full(linear_count, 0.7, dtype=np.float32),
        "scattering": np.full(linear_count, 0.08, dtype=np.float32),
        "config": FastVegetationV3Config(),
    }
    sparse_linear = _rigid_linear_exclusion(
        linear_xyz, np.ones(linear_count, dtype=bool), **linear_kwargs
    )
    monkeypatch.setattr(fast_v3, "_grid_component_labels", _dense_grid_component_labels_reference)
    dense_linear = _rigid_linear_exclusion(
        linear_xyz, np.ones(linear_count, dtype=bool), **linear_kwargs
    )
    assert np.array_equal(sparse_linear[0], dense_linear[0])
    assert np.array_equal(sparse_linear[1], dense_linear[1])
    assert sparse_linear[2] == dense_linear[2]

    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 1.4, 15), np.linspace(0.0, 1.4, 15))
    surface_xyz = np.column_stack(
        (grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, 0.8, dtype=np.float64))
    )
    surface_kwargs = {
        "hag": np.full(len(surface_xyz), 0.8, dtype=np.float32),
        "planarity": np.full(len(surface_xyz), 0.8, dtype=np.float32),
        "scattering": np.full(len(surface_xyz), 0.08, dtype=np.float32),
        "config": FastVegetationV3Config(),
    }
    dense_surface = _rigid_surface_exclusion(
        surface_xyz, np.ones(len(surface_xyz), dtype=bool), **surface_kwargs
    )
    monkeypatch.setattr(fast_v3, "_grid_component_labels", _grid_component_labels)
    sparse_surface = _rigid_surface_exclusion(
        surface_xyz, np.ones(len(surface_xyz), dtype=bool), **surface_kwargs
    )
    assert np.array_equal(sparse_surface[0], dense_surface[0])
    assert np.array_equal(sparse_surface[1], dense_surface[1])
    assert sparse_surface[2] == dense_surface[2]
