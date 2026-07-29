from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import algorithms.vegetation as vegetation
import algorithms.vegetation.treetop_candidates as treetops
import numpy as np
import pytest
from algorithms.vegetation.canopy_height_model import (
    CanopyHeightModelConfig,
    CanopyHeightModelResult,
)
from algorithms.vegetation.treetop_candidates import (
    TreetopCandidateResult,
    TreetopDetectionConfig,
    TreetopDetectionError,
    TreetopDetectionErrorCode,
    detect_treetop_candidates,
)
from core.contracts.tree_inputs import TreeSegmentationInputIdentity


def _readonly(value: Any, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _identity(point_count: int) -> TreeSegmentationInputIdentity:
    values: dict[str, Any] = {
        "source_id": "synthetic",
        "manifest_source_id": "synthetic-manifest",
        "origin_xyz": (0.0, 0.0, 0.0),
        "point_count": point_count,
        "v2_algorithm_version": "v2",
        "a2a_algorithm_version": "a2a",
        "a2b_algorithm_version": "a2b",
    }
    values.update(
        {
            name: "a" * 64
            for name in TreeSegmentationInputIdentity.__dataclass_fields__
            if name.endswith("_sha256")
        }
    )
    return TreeSegmentationInputIdentity(**values)


def _model(
    height: Any,
    *,
    counts: Any | None = None,
    support: Any | None = None,
    peak_support: Any | None = None,
    cell_size_m: float = 1.0,
    origin: tuple[float, float] | None = (0.0, 0.0),
) -> CanopyHeightModelResult:
    height_array = np.array(height, dtype=np.float32, order="C", copy=True)
    if height_array.shape == (0,):
        height_array = np.empty((0, 0), dtype=np.float32)
    occupied = np.isfinite(height_array)
    if counts is None:
        count_array = occupied.astype(np.int32)
    else:
        count_array = np.array(counts, dtype=np.int32, order="C", copy=True)
    if support is None:
        support_array = occupied.astype(np.uint8)
    else:
        support_array = np.array(support, dtype=np.uint8, order="C", copy=True)
    if peak_support is None:
        peak_array = support_array.copy()
    else:
        peak_array = np.array(peak_support, dtype=np.uint8, order="C", copy=True)
    for value in (height_array, occupied, count_array, support_array, peak_array):
        value.setflags(write=False)
    included = int(np.sum(count_array, dtype=np.int64))
    return CanopyHeightModelResult(
        source_identity=_identity(included),
        config=CanopyHeightModelConfig(
            cell_size_m=cell_size_m,
            minimum_hag_m=0.0,
        ),
        grid_origin_xy=None if height_array.shape == (0, 0) else origin,
        canopy_height_m=height_array,
        occupied_mask=occupied,
        point_count_by_cell=count_array,
        support_flags_by_cell=support_array,
        peak_support_flags_by_cell=peak_array,
        included_point_count=included,
        excluded_point_count=0,
    )


def _config(**changes: Any) -> TreetopDetectionConfig:
    values = {
        "minimum_peak_hag_m": 0.3,
        "local_max_radius_m": 1.1,
        "prominence_radius_m": 2.1,
    }
    values.update(changes)
    return TreetopDetectionConfig(**values)


def _result_arrays(result: TreetopCandidateResult) -> dict[str, np.ndarray]:
    return {
        name: getattr(result, name)
        for name in result.__dataclass_fields__
        if isinstance(getattr(result, name), np.ndarray)
    }


def test_public_api_exports_signature_defaults_and_frozen_contracts() -> None:
    expected = {
        "TreetopDetectionConfig": TreetopDetectionConfig,
        "TreetopDetectionErrorCode": TreetopDetectionErrorCode,
        "TreetopDetectionError": TreetopDetectionError,
        "TreetopCandidateResult": TreetopCandidateResult,
        "detect_treetop_candidates": detect_treetop_candidates,
    }
    for name, value in expected.items():
        assert getattr(vegetation, name) is value
    assert tuple(inspect.signature(detect_treetop_candidates).parameters) == (
        "canopy_model",
        "config",
    )
    config = TreetopDetectionConfig()
    assert config.minimum_separation_m == 0.0
    assert config.minimum_neighbour_occupied_cells == 0
    assert config.minimum_isolated_peak_point_count == 1
    assert config.maximum_candidate_neighbour_evaluations == 100_000_000
    assert [value.value for value in TreetopDetectionErrorCode] == [
        "resource_limit",
        "allocation_failed",
        "internal_invariant",
    ]
    with pytest.raises(FrozenInstanceError):
        config.minimum_peak_hag_m = 2.0  # type: ignore[misc]
    result = detect_treetop_candidates(_model([[2.0]]))
    with pytest.raises(FrozenInstanceError):
        result.source_grid_shape = (2, 2)  # type: ignore[misc]
    for name in (
        "write_treetop_candidates",
        "publish_treetop_candidates",
        "generate_treetop_preview",
        "watershed_treetops",
    ):
        assert not hasattr(vegetation, name)


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_peak_hag_m": True},
        {"minimum_peak_hag_m": -1.0},
        {"minimum_peak_hag_m": np.inf},
        {"local_max_radius_m": np.nan},
        {"prominence_radius_m": 0.5, "local_max_radius_m": 0.5},
        {"minimum_prominence_m": -0.1},
        {"minimum_separation_m": False},
        {"minimum_neighbour_occupied_cells": True},
        {"minimum_neighbour_occupied_cells": -1},
        {"minimum_isolated_peak_point_count": 0},
        {"maximum_neighbourhood_cells": 0},
        {"maximum_temporary_bytes": np.bool_(True)},
        {"maximum_candidate_count": np.iinfo(np.int32).max + 1},
        {"maximum_candidate_neighbour_evaluations": 0},
        {"maximum_candidate_neighbour_evaluations": 1.5},
    ],
)
def test_config_rejects_invalid_values(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TreetopDetectionConfig(**changes)
    normalized = TreetopDetectionConfig(
        minimum_peak_hag_m=np.float32(0.4),
        maximum_candidate_neighbour_evaluations=np.int64(10),
    )
    assert type(normalized.minimum_peak_hag_m) is float
    assert type(normalized.maximum_candidate_neighbour_evaluations) is int


def test_wrong_public_types_and_error_contract() -> None:
    with pytest.raises(TypeError):
        detect_treetop_candidates(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        detect_treetop_candidates(_model([[1.0]]), object())  # type: ignore[arg-type]
    error = TreetopDetectionError(
        TreetopDetectionErrorCode.RESOURCE_LIMIT,
        "limit",
        row_count=1,
        column_count=2,
        total_cell_count=2,
        local_maximum_offset_count=3,
        prominence_offset_count=4,
        estimated_temporary_bytes=5,
        candidate_count=6,
        candidate_neighbour_evaluations=24,
    )
    assert error.code is TreetopDetectionErrorCode.RESOURCE_LIMIT
    assert error.candidate_neighbour_evaluations == 24
    with pytest.raises(ValueError):
        TreetopDetectionError("bad", "message")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TreetopDetectionError(TreetopDetectionErrorCode.RESOURCE_LIMIT, "")
    with pytest.raises(ValueError):
        TreetopDetectionError(
            TreetopDetectionErrorCode.RESOURCE_LIMIT,
            "bad",
            row_count=np.int64(1),  # type: ignore[arg-type]
        )


def test_empty_no_occupied_and_below_threshold_results() -> None:
    inputs = (
        _model(np.empty((0, 0), dtype=np.float32), origin=None),
        _model(np.full((2, 2), np.nan, dtype=np.float32)),
        _model([[0.1, 0.2]]),
    )
    for model in inputs:
        result = detect_treetop_candidates(model)
        assert result.candidate_count == 0
        assert result.local_xy.shape == (0, 2)
        assert result.neighbourhood_support_fraction.shape == (0, 3)
        assert result.local_maximum_plateau_count == 0
        assert result.evidence_rejected_count == 0
        assert result.suppressed_candidate_count == 0
        for value in _result_arrays(result).values():
            assert value.flags.owndata and value.flags.c_contiguous
            assert not value.flags.writeable


def test_one_point_isolated_candidate_default_and_configured_filtering() -> None:
    model = _model([[4.0]])
    result = detect_treetop_candidates(model)
    assert result.candidate_count == 1
    assert result.neighbour_occupied_cell_count.tolist() == [0]
    assert result.neighbourhood_occupied_fraction.tolist() == [0.0]
    assert result.source_cell_point_count.tolist() == [1]
    assert result.prominence_available.tolist() == [False]
    assert result.prominence_m.tolist() == [0.0]
    assert not hasattr(result, "tree_id")

    rejected = detect_treetop_candidates(
        model,
        TreetopDetectionConfig(
            minimum_neighbour_occupied_cells=1,
            minimum_isolated_peak_point_count=2,
        ),
    )
    assert rejected.candidate_count == 0
    assert rejected.local_maximum_plateau_count == 1
    assert rejected.evidence_rejected_count == 1

    repeated = detect_treetop_candidates(
        _model([[4.0]], counts=[[2]]),
        TreetopDetectionConfig(
            minimum_neighbour_occupied_cells=1,
            minimum_isolated_peak_point_count=2,
        ),
    )
    assert repeated.candidate_count == 1
    neighbour_branch = detect_treetop_candidates(
        _model([[4.0, 2.0]]),
        _config(
            minimum_neighbour_occupied_cells=1,
            minimum_isolated_peak_point_count=2,
        ),
    )
    assert neighbour_branch.candidate_count == 1


def test_local_maximum_inside_outside_empty_edge_and_disk_rules() -> None:
    inside = detect_treetop_candidates(_model([[4.0, 5.0]]), _config())
    assert list(zip(inside.row_index, inside.column_index, strict=True)) == [(0, 1)]

    outside = detect_treetop_candidates(
        _model([[4.0, np.nan, 5.0]]),
        _config(local_max_radius_m=1.1),
    )
    assert list(zip(outside.row_index, outside.column_index, strict=True)) == [(0, 2), (0, 0)]

    below_cell = detect_treetop_candidates(
        _model([[4.0, 5.0]]),
        _config(local_max_radius_m=0.5),
    )
    assert below_cell.candidate_count == 2

    circular = detect_treetop_candidates(
        _model([[4.0, np.nan], [np.nan, 5.0]]),
        _config(local_max_radius_m=1.1),
    )
    assert circular.candidate_count == 2

    centre = detect_treetop_candidates(
        _model([[1.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 1.0]]),
        _config(local_max_radius_m=1.5),
    )
    assert (centre.row_index.tolist(), centre.column_index.tolist()) == ([1], [1])

    for model in (_model([[1.0, 4.0, 1.0]]), _model([[1.0], [4.0], [1.0]])):
        result = detect_treetop_candidates(model, _config())
        assert result.candidate_count == 1
        assert result.peak_height_m.tolist() == [4.0]


def test_edge_limited_uses_exact_circular_axis_extent() -> None:
    def candidate_at(
        shape: tuple[int, int],
        position: tuple[int, int],
        prominence_radius_m: float,
    ) -> TreetopCandidateResult:
        height = np.full(shape, np.nan, dtype=np.float32)
        height[position] = 5.0
        return detect_treetop_candidates(
            _model(height),
            _config(
                local_max_radius_m=0.0,
                prominence_radius_m=prominence_radius_m,
            ),
        )

    one_cell_inside = candidate_at((3, 3), (1, 1), 1.5)
    on_edge = candidate_at((3, 3), (0, 1), 1.5)
    subcell_radius = candidate_at((1, 1), (0, 0), 0.5)
    one_cell_from_edge = candidate_at((5, 5), (1, 2), 2.0)
    two_cells_inside = candidate_at((5, 5), (2, 2), 2.0)

    assert one_cell_inside.edge_limited.tolist() == [False]
    assert on_edge.edge_limited.tolist() == [True]
    assert subcell_radius.edge_limited.tolist() == [False]
    assert one_cell_from_edge.edge_limited.tolist() == [True]
    assert two_cells_inside.edge_limited.tolist() == [False]


def test_radius_zero_keeps_unequal_adjacent_cells_as_separate_proposals() -> None:
    result = detect_treetop_candidates(
        _model([[2.0, 3.0]]),
        _config(local_max_radius_m=0.0),
    )
    assert result.candidate_count == 2
    assert result.peak_height_m.tolist() == [3.0, 2.0]


@pytest.mark.parametrize(
    ("height", "expected_count"),
    [
        ([[5.0]], 1),
        ([[5.0, 5.0, 5.0]], 1),
        ([[5.0], [5.0], [5.0]], 1),
        ([[5.0, 5.0], [5.0, 5.0]], 1),
        ([[5.0, 5.0], [5.0, np.nan]], 1),
        ([[5.0, np.nan], [np.nan, 5.0]], 1),
        ([[5.0, 5.0, np.nan, 5.0, 5.0]], 2),
    ],
)
def test_exact_eight_connected_plateau_shapes(height: Any, expected_count: int) -> None:
    result = detect_treetop_candidates(_model(height), _config(local_max_radius_m=0.5))
    assert result.candidate_count == expected_count
    assert result.local_maximum_plateau_count == expected_count


def test_plateau_representative_centroid_point_count_and_row_column_ties() -> None:
    square = detect_treetop_candidates(
        _model([[5.0, 5.0], [5.0, 5.0]]),
        _config(local_max_radius_m=0.5),
    )
    assert (square.row_index[0], square.column_index[0]) == (0, 0)
    assert square.plateau_cell_count.tolist() == [4]

    point_tie = detect_treetop_candidates(
        _model([[5.0, 5.0]], counts=[[1, 3]]),
        _config(local_max_radius_m=0.5),
    )
    assert (point_tie.row_index[0], point_tie.column_index[0]) == (0, 1)

    first = detect_treetop_candidates(
        _model([[5.0, 5.0], [np.nan, 5.0]]),
        _config(local_max_radius_m=0.5),
    )
    second = detect_treetop_candidates(
        _model(np.array([[5.0, 5.0], [np.nan, 5.0]], dtype=np.float32, order="F")),
        _config(local_max_radius_m=0.5),
    )
    assert (first.row_index[0], first.column_index[0]) == (
        second.row_index[0],
        second.column_index[0],
    )


def test_singleton_plateau_representatives_use_vectorized_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height = np.full((9, 9), np.nan, dtype=np.float32)
    height[::2, ::2] = 5.0
    expected = [(row, column) for row in range(0, 9, 2) for column in range(0, 9, 2)]

    def forbidden(*args: Any, **kwargs: Any) -> tuple[int, int]:
        raise AssertionError("singleton plateaus must not use multi-cell ranking")

    monkeypatch.setattr(treetops, "_select_multicell_representative", forbidden)
    first = detect_treetop_candidates(
        _model(height),
        _config(local_max_radius_m=0.5, prominence_radius_m=1.5),
    )
    second = detect_treetop_candidates(
        _model(height),
        _config(local_max_radius_m=0.5, prominence_radius_m=1.5),
    )

    actual = list(zip(first.row_index, first.column_index, strict=True))
    assert actual == expected
    assert first.plateau_cell_count.tolist() == [1] * len(expected)
    assert np.array_equal(first.row_index, second.row_index)
    assert np.array_equal(first.column_index, second.column_index)


def test_run_label_uses_scalar_intervals_for_large_disconnected_rows() -> None:
    helper_source = inspect.getsource(treetops._label_plateau_runs)
    assert "previous_ids" not in helper_source
    assert "current_ids" not in helper_source
    assert "range(current_start_id, current_end_id)" in helper_source

    height = np.full((5, 401), np.nan, dtype=np.float32)
    height[::2, ::2] = 5.0
    expected = [(row, column) for row in range(0, 5, 2) for column in range(0, 401, 2)]
    first = detect_treetop_candidates(
        _model(height),
        _config(local_max_radius_m=0.0, prominence_radius_m=0.5),
    )
    second = detect_treetop_candidates(
        _model(height),
        _config(local_max_radius_m=0.0, prominence_radius_m=0.5),
    )

    assert list(zip(first.row_index, first.column_index, strict=True)) == expected
    assert first.local_maximum_plateau_count == len(expected)
    assert first.plateau_cell_count.tolist() == [1] * len(expected)
    assert np.array_equal(first.row_index, second.row_index)
    assert np.array_equal(first.column_index, second.column_index)


def test_near_equal_values_are_not_one_plateau() -> None:
    value = np.float32(5.0)
    nearby = np.nextafter(value, np.float32(np.inf), dtype=np.float32)
    result = detect_treetop_candidates(
        _model([[value, nearby]]),
        _config(local_max_radius_m=0.0),
    )
    assert result.local_maximum_plateau_count == 2
    assert result.candidate_count == 2


def test_prominence_positive_zero_unavailable_sparse_and_local() -> None:
    positive_model = _model([[2.0, 2.0, 2.0], [2.0, 5.0, 2.0], [2.0, 2.0, 2.0]])
    positive = detect_treetop_candidates(
        positive_model,
        _config(local_max_radius_m=0.5, prominence_radius_m=1.5),
    )
    peak_position = int(np.flatnonzero(positive.peak_height_m == 5.0)[0])
    assert positive.prominence_available[peak_position]
    assert positive.prominence_m[peak_position] == 3.0

    unavailable = detect_treetop_candidates(_model([[5.0]]))
    assert unavailable.prominence_available.tolist() == [False]
    assert unavailable.prominence_m.tolist() == [0.0]

    sparse = detect_treetop_candidates(
        _model([[2.0, np.nan, 4.0, np.nan, 8.0]]),
        _config(local_max_radius_m=0.0, prominence_radius_m=2.1),
    )
    peak_position = int(np.flatnonzero(sparse.peak_height_m == 4.0)[0])
    assert sparse.prominence_available[peak_position]
    assert sparse.prominence_m[peak_position] == 0.0

    distant_changed = detect_treetop_candidates(
        _model(
            [
                [2.0, 2.0, 2.0, np.nan, 100.0],
                [2.0, 5.0, 2.0, np.nan, np.nan],
                [2.0, 2.0, 2.0, np.nan, np.nan],
            ]
        ),
        _config(local_max_radius_m=0.5, prominence_radius_m=1.5),
    )
    peak_position = int(np.flatnonzero(distant_changed.peak_height_m == 5.0)[0])
    assert distant_changed.prominence_m[peak_position] == 3.0


def test_same_plateau_is_excluded_from_prominence_reference() -> None:
    model = _model(
        [
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [2.0, 2.0, 2.0],
        ]
    )
    result = detect_treetop_candidates(
        model,
        _config(local_max_radius_m=1.1, prominence_radius_m=2.5),
    )
    assert result.candidate_count == 1
    assert result.prominence_available.tolist() == [True]
    assert result.prominence_m.tolist() == [3.0]


def test_positive_prominence_threshold_rejects_unavailable_and_insufficient() -> None:
    unavailable = detect_treetop_candidates(
        _model([[5.0]]),
        TreetopDetectionConfig(minimum_prominence_m=0.1),
    )
    assert unavailable.candidate_count == 0
    low = detect_treetop_candidates(
        _model([[4.9, 5.0, 4.9]]),
        _config(
            local_max_radius_m=0.5,
            prominence_radius_m=1.5,
            minimum_prominence_m=0.2,
        ),
    )
    assert low.candidate_count == 0


def test_neighbourhood_evidence_excludes_centre_and_aggregates_support() -> None:
    support = np.array(
        [
            [1, 3, 2],
            [4, 1, 6],
            [1, 5, 4],
        ],
        dtype=np.uint8,
    )
    model = _model(
        [[2.0, 2.0, 2.0], [2.0, 5.0, 2.0], [2.0, 2.0, 2.0]],
        support=support,
        peak_support=support,
    )
    result = detect_treetop_candidates(
        model,
        _config(local_max_radius_m=0.5, prominence_radius_m=1.5),
    )
    peak_position = int(np.flatnonzero(result.peak_height_m == 5.0)[0])
    assert result.neighbour_occupied_cell_count[peak_position] == 8
    assert result.neighbourhood_occupied_fraction[peak_position] == 1.0
    assert result.neighbourhood_support_flags[peak_position] == 7
    expected = np.array([[4 / 8, 3 / 8, 4 / 8]], dtype=np.float32)
    assert np.array_equal(result.neighbourhood_support_fraction[[peak_position]], expected)
    assert float(result.neighbourhood_support_fraction[peak_position].sum()) > 1.0


@pytest.mark.parametrize("flag", [1, 2, 4, 7])
def test_support_status_changes_evidence_not_geometry(flag: int) -> None:
    height = [[3.0, 1.0, 3.0]]
    support = np.full((1, 3), flag, dtype=np.uint8)
    result = detect_treetop_candidates(
        _model(height, support=support, peak_support=support),
        _config(local_max_radius_m=1.1),
    )
    assert result.row_index.tolist() == [0, 0]
    assert result.column_index.tolist() == [0, 2]
    assert result.candidate_id.tolist() == [0, 1]
    assert result.source_support_flags.tolist() == [flag, flag]


def test_peak_support_is_copied_exactly() -> None:
    result = detect_treetop_candidates(
        _model([[5.0, 2.0]], support=[[7, 2]], peak_support=[[4, 2]]),
        _config(),
    )
    assert result.source_support_flags.tolist() == [7]
    assert result.source_peak_support_flags.tolist() == [4]


def test_default_nms_retains_close_trees_and_skips_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> np.ndarray:
        raise AssertionError("zero separation must skip the spatial-index helper")

    monkeypatch.setattr(treetops, "_apply_nms", forbidden)
    result = detect_treetop_candidates(
        _model([[5.0, np.nan, 4.0]]),
        _config(local_max_radius_m=0.5),
    )
    assert result.candidate_count == 2
    assert result.suppressed_candidate_count == 0


@pytest.mark.parametrize(
    ("separation", "expected_count"),
    [(1.1, 1), (1.0, 2), (0.9, 2)],
)
def test_positive_nms_exact_inside_equal_outside_boundary(
    separation: float,
    expected_count: int,
) -> None:
    result = detect_treetop_candidates(
        _model([[5.0, 4.0]]),
        _config(local_max_radius_m=0.0, minimum_separation_m=separation),
    )
    assert result.candidate_count == expected_count


def test_nms_ranking_and_status_neutrality() -> None:
    stronger = detect_treetop_candidates(
        _model([[4.0, 5.0]]),
        _config(local_max_radius_m=0.0, minimum_separation_m=2.0),
    )
    assert stronger.column_index.tolist() == [1]

    left_confirmed = detect_treetop_candidates(
        _model([[5.0, 5.0]], support=[[1, 4]], peak_support=[[1, 4]]),
        _config(local_max_radius_m=0.0, minimum_separation_m=2.0),
    )
    left_provisional = detect_treetop_candidates(
        _model([[5.0, 5.0]], support=[[4, 1]], peak_support=[[4, 1]]),
        _config(local_max_radius_m=0.0, minimum_separation_m=2.0),
    )
    assert left_confirmed.column_index.tolist() == [0]
    assert left_provisional.column_index.tolist() == [0]


def test_translation_invariance_including_large_and_negative_origins() -> None:
    origins = (
        (10.0, 20.0),
        (-10.0, -20.0),
        (1_000_000_000.0, -1_000_000_000.0),
    )
    results = [
        detect_treetop_candidates(
            _model([[2.0, 5.0, 2.0]], origin=origin),
            _config(),
        )
        for origin in origins
    ]
    for result in results[1:]:
        assert np.array_equal(result.row_index, results[0].row_index)
        assert np.array_equal(result.column_index, results[0].column_index)
        assert np.array_equal(result.candidate_id, results[0].candidate_id)
        assert np.array_equal(result.prominence_m, results[0].prominence_m)
    for origin, result in zip(origins, results, strict=True):
        assert result.local_xy[0, 0] == origin[0] + 1.5
        assert result.local_xy[0, 1] == origin[1] + 0.5


@pytest.mark.parametrize("cell_size", [0.10, 0.20, 0.25, 0.50])
def test_cell_size_compatibility_uses_physical_cell_centres(cell_size: float) -> None:
    origin = (-2.5 * cell_size, -2.5 * cell_size)
    height = np.full((5, 5), np.nan, dtype=np.float32)
    height[2, 2] = 5.0
    result = detect_treetop_candidates(
        _model(height, cell_size_m=cell_size, origin=origin),
        TreetopDetectionConfig(),
    )
    assert result.candidate_count == 1
    assert result.local_xy.tolist() == [[0.0, 0.0]]


@pytest.mark.parametrize("cell_size_m", [0.10, 0.20, 0.25, 0.50])
def test_physical_local_maximum_radius_is_resolution_invariant(
    cell_size_m: float,
) -> None:
    def detect_arrangement(distance_m: float) -> TreetopCandidateResult:
        cell_distance = int(distance_m / cell_size_m)
        height = np.ones((1, cell_distance + 1), dtype=np.float32)
        height[0, 0] = 4.0
        height[0, cell_distance] = 5.0
        return detect_treetop_candidates(
            _model(
                height,
                cell_size_m=cell_size_m,
                origin=(-0.5 * cell_size_m, -0.5 * cell_size_m),
            ),
            _config(local_max_radius_m=1.1, prominence_radius_m=1.5),
        )

    one_metre = detect_arrangement(1.0)
    assert one_metre.candidate_count == 1
    assert one_metre.peak_height_m.tolist() == [5.0]
    assert np.allclose(one_metre.local_xy, [[1.0, 0.0]], rtol=0.0, atol=1e-12)

    two_metres = detect_arrangement(2.0)
    assert two_metres.candidate_count == 2
    positions = two_metres.local_xy[np.argsort(two_metres.local_xy[:, 0])]
    assert np.allclose(positions, [[0.0, 0.0], [2.0, 0.0]], rtol=0.0, atol=1e-12)


def test_density_and_crown_scale_variation() -> None:
    height = [[1.0, 3.0, 5.0, 3.0, 1.0]]
    low = detect_treetop_candidates(_model(height), _config())
    high = detect_treetop_candidates(
        _model(height, counts=[[20, 20, 20, 20, 20]]),
        _config(),
    )
    assert low.column_index.tolist() == high.column_index.tolist() == [2]
    assert low.source_cell_point_count.tolist() == [1]
    assert high.source_cell_point_count.tolist() == [20]
    for crown in (
        [[1.0, 5.0, 1.0]],
        [[1.0, 3.0, 5.0, 3.0, 1.0]],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]],
        [[0.4, 0.5, 0.4]],
        [[2.0, 10.0, 2.0]],
    ):
        result = detect_treetop_candidates(_model(crown), _config())
        assert result.candidate_count == 1


def test_determinism_cwd_no_mutation_and_independent_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(
        [[2.0, 5.0, 2.0, np.nan, 5.0]],
        support=[[1, 2, 4, 0, 7]],
        peak_support=[[1, 2, 4, 0, 4]],
    )
    snapshots = {
        name: (getattr(model, name).copy(), getattr(model, name).flags.writeable)
        for name in (
            "canopy_height_m",
            "occupied_mask",
            "point_count_by_cell",
            "support_flags_by_cell",
            "peak_support_flags_by_cell",
        )
    }
    first = detect_treetop_candidates(model, _config())
    monkeypatch.chdir(tmp_path)
    second = detect_treetop_candidates(model, _config())
    copied = _model(
        model.canopy_height_m.copy(),
        counts=model.point_count_by_cell.copy(),
        support=model.support_flags_by_cell.copy(),
        peak_support=model.peak_support_flags_by_cell.copy(),
    )
    third = detect_treetop_candidates(copied, _config())
    for name, first_value in _result_arrays(first).items():
        second_value = getattr(second, name)
        third_value = getattr(third, name)
        assert np.array_equal(first_value, second_value)
        assert np.array_equal(first_value, third_value)
        assert not np.shares_memory(first_value, second_value)
    assert (
        first.local_maximum_plateau_count,
        first.evidence_rejected_count,
        first.suppressed_candidate_count,
    ) == (
        second.local_maximum_plateau_count,
        second.evidence_rejected_count,
        second.suppressed_candidate_count,
    )
    for name, (snapshot, writeable) in snapshots.items():
        value = getattr(model, name)
        assert np.array_equal(value, snapshot, equal_nan=True)
        assert value.flags.writeable is writeable


def test_runtime_input_revalidation_rejects_corruption() -> None:
    model = _model([[2.0]])
    model.canopy_height_m.setflags(write=True)
    try:
        with pytest.raises(ValueError, match="read-only"):
            detect_treetop_candidates(model)
    finally:
        model.canopy_height_m.setflags(write=False)


def test_all_result_array_contracts() -> None:
    first = detect_treetop_candidates(_model([[2.0, 5.0, 2.0]]), _config())
    second = detect_treetop_candidates(_model([[2.0, 5.0, 2.0]]), _config())
    expected = {
        "candidate_id": (np.dtype(np.int32), (1,)),
        "row_index": (np.dtype(np.int64), (1,)),
        "column_index": (np.dtype(np.int64), (1,)),
        "local_xy": (np.dtype(np.float64), (1, 2)),
        "peak_height_m": (np.dtype(np.float32), (1,)),
        "prominence_m": (np.dtype(np.float32), (1,)),
        "prominence_available": (np.dtype(np.bool_), (1,)),
        "plateau_cell_count": (np.dtype(np.int32), (1,)),
        "source_cell_point_count": (np.dtype(np.int32), (1,)),
        "source_support_flags": (np.dtype(np.uint8), (1,)),
        "source_peak_support_flags": (np.dtype(np.uint8), (1,)),
        "neighbour_occupied_cell_count": (np.dtype(np.int32), (1,)),
        "neighbourhood_occupied_fraction": (np.dtype(np.float32), (1,)),
        "neighbourhood_support_flags": (np.dtype(np.uint8), (1,)),
        "neighbourhood_support_fraction": (np.dtype(np.float32), (1, 3)),
        "edge_limited": (np.dtype(np.bool_), (1,)),
    }
    for name, (dtype, shape) in expected.items():
        value = getattr(first, name)
        assert value.dtype == dtype and value.shape == shape
        assert value.flags.owndata and value.flags.c_contiguous
        assert not value.flags.writeable
        assert not np.shares_memory(value, getattr(second, name))
        for input_value in (
            first.source_identity,
            first.source_chm_config,
        ):
            assert value is not input_value
        with pytest.raises(ValueError):
            value.flat[0] = value.flat[0]


def test_result_direct_construction_rejects_array_storage_and_cross_fields() -> None:
    valid = detect_treetop_candidates(_model([[5.0]]))
    with pytest.raises(ValueError, match="dtype"):
        replace(valid, candidate_id=_readonly([0], np.int64))
    writable = np.array([0], dtype=np.int32)
    with pytest.raises(ValueError, match="read-only"):
        replace(valid, candidate_id=writable)
    base = np.array([0], dtype=np.int32)
    view = base[:]
    view.setflags(write=False)
    with pytest.raises(ValueError, match="own"):
        replace(valid, candidate_id=view)
    fortran = np.asfortranarray(np.array([[0.5, 0.5]], dtype=np.float64))
    fortran.setflags(write=False)
    assert fortran.flags.c_contiguous  # one-row arrays are both C- and F-contiguous
    with pytest.raises(ValueError):
        replace(valid, candidate_id=_readonly([1], np.int32))
    with pytest.raises(ValueError):
        replace(valid, row_index=_readonly([2], np.int64))
    with pytest.raises(ValueError):
        replace(valid, local_xy=_readonly([[9.0, 9.0]], np.float64))
    with pytest.raises(ValueError):
        replace(
            valid,
            prominence_available=_readonly([False], np.bool_),
            prominence_m=_readonly([1]),
        )
    with pytest.raises(ValueError):
        replace(valid, plateau_cell_count=_readonly([0], np.int32))
    with pytest.raises(ValueError):
        replace(valid, source_support_flags=_readonly([8], np.uint8))
    with pytest.raises(ValueError):
        replace(valid, source_peak_support_flags=_readonly([2], np.uint8))
    with pytest.raises(ValueError):
        replace(valid, neighbourhood_occupied_fraction=_readonly([1.1], np.float32))
    with pytest.raises(ValueError):
        replace(valid, evidence_rejected_count=1)


def test_result_direct_construction_validates_edge_geometry() -> None:
    height = np.full((3, 3), np.nan, dtype=np.float32)
    height[1, 1] = 5.0
    valid = detect_treetop_candidates(
        _model(height),
        _config(local_max_radius_m=0.0, prominence_radius_m=1.5),
    )
    assert valid.edge_limited.tolist() == [False]
    accepted = replace(valid, edge_limited=_readonly([False], np.bool_))
    assert accepted.edge_limited.tolist() == [False]
    with pytest.raises(ValueError, match="edge_limited"):
        replace(valid, edge_limited=_readonly([True], np.bool_))


def test_result_direct_construction_rejects_inconsistent_neighbour_evidence() -> None:
    isolated = detect_treetop_candidates(_model([[5.0]]))
    with pytest.raises(ValueError, match="zero neighbour count"):
        replace(
            isolated,
            neighbourhood_occupied_fraction=_readonly([0.5], np.float32),
        )
    with pytest.raises(ValueError, match="zero neighbour count"):
        replace(
            isolated,
            neighbourhood_support_flags=_readonly([1], np.uint8),
        )
    with pytest.raises(ValueError, match="zero neighbour count"):
        replace(
            isolated,
            neighbourhood_support_fraction=_readonly([[1.0, 0.0, 0.0]], np.float32),
        )

    supported = detect_treetop_candidates(
        _model([[5.0, 2.0]]),
        _config(local_max_radius_m=1.1, prominence_radius_m=1.5),
    )
    assert supported.neighbour_occupied_cell_count.tolist() == [1]
    accepted = replace(supported)
    assert accepted.neighbourhood_support_flags.tolist() == [1]
    assert accepted.neighbourhood_support_fraction.tolist() == [[1.0, 0.0, 0.0]]
    with pytest.raises(ValueError, match="positive neighbour count"):
        replace(
            supported,
            neighbourhood_occupied_fraction=_readonly([0.0], np.float32),
        )
    with pytest.raises(ValueError, match="nonzero neighbourhood support"):
        replace(
            supported,
            neighbourhood_support_flags=_readonly([0], np.uint8),
            neighbourhood_support_fraction=_readonly([[0.0, 0.0, 0.0]], np.float32),
        )
    with pytest.raises(ValueError, match="positive support fraction"):
        replace(
            supported,
            neighbourhood_support_fraction=_readonly([[0.0, 0.0, 0.0]], np.float32),
        )
    with pytest.raises(ValueError, match="support bits"):
        replace(
            supported,
            neighbourhood_support_flags=_readonly([2], np.uint8),
        )


def test_result_rejects_duplicate_pairs_empty_grid_candidates_and_bad_scalars() -> None:
    multiple = detect_treetop_candidates(
        _model([[5.0, np.nan, 4.0]]),
        _config(local_max_radius_m=0.5),
    )
    with pytest.raises(ValueError, match="unique"):
        replace(
            multiple,
            row_index=_readonly([0, 0], np.int64),
            column_index=_readonly([0, 0], np.int64),
            local_xy=_readonly([[0.5, 0.5], [0.5, 0.5]], np.float64),
        )
    with pytest.raises(ValueError):
        replace(multiple, source_grid_shape=(0, 0), source_grid_origin_xy=None)
    with pytest.raises(ValueError):
        replace(multiple, source_grid_shape=(0, 2))
    with pytest.raises(ValueError):
        replace(multiple, source_grid_origin_xy=None)
    with pytest.raises(ValueError):
        replace(multiple, local_maximum_plateau_count=np.int64(2))  # type: ignore[arg-type]


def test_offset_and_temporary_resource_limits() -> None:
    model = _model(np.ones((5, 5), dtype=np.float32))
    with pytest.raises(TreetopDetectionError) as local:
        detect_treetop_candidates(
            model,
            TreetopDetectionConfig(
                local_max_radius_m=1.5,
                prominence_radius_m=2.5,
                maximum_neighbourhood_cells=1,
            ),
        )
    assert local.value.code is TreetopDetectionErrorCode.RESOURCE_LIMIT
    assert local.value.local_maximum_offset_count is not None

    with pytest.raises(TreetopDetectionError) as prominence:
        detect_treetop_candidates(
            model,
            TreetopDetectionConfig(
                local_max_radius_m=0.0,
                prominence_radius_m=1.5,
                maximum_neighbourhood_cells=1,
            ),
        )
    assert prominence.value.prominence_offset_count is not None

    with pytest.raises(TreetopDetectionError) as temporary:
        detect_treetop_candidates(
            _model([[5.0]]),
            TreetopDetectionConfig(maximum_temporary_bytes=1),
        )
    assert temporary.value.estimated_temporary_bytes == 3


def test_phase_resource_preflights_precede_dangerous_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(np.ones((5, 5), dtype=np.float32))

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("phase allocation must not begin after failed preflight")

    with monkeypatch.context() as scoped:
        scoped.setattr(treetops, "_allocate_local_maximum_work_arrays", forbidden)
        with pytest.raises(TreetopDetectionError) as local:
            detect_treetop_candidates(
                model,
                _config(
                    local_max_radius_m=0.5,
                    prominence_radius_m=1.5,
                    maximum_temporary_bytes=1,
                ),
            )
    assert local.value.code is TreetopDetectionErrorCode.RESOURCE_LIMIT
    assert local.value.estimated_temporary_bytes is not None
    assert local.value.estimated_temporary_bytes > 1

    with monkeypatch.context() as scoped:
        scoped.setattr(treetops, "_allocate_plateau_labels", forbidden)
        with pytest.raises(TreetopDetectionError) as run_label:
            detect_treetop_candidates(
                _model(
                    np.array(
                        [
                            [5.0, 4.0, 5.0, 4.0, 5.0],
                            [4.0, 5.0, 4.0, 5.0, 4.0],
                            [5.0, 4.0, 5.0, 4.0, 5.0],
                            [4.0, 5.0, 4.0, 5.0, 4.0],
                            [5.0, 4.0, 5.0, 4.0, 5.0],
                        ],
                        dtype=np.float32,
                    )
                ),
                _config(
                    local_max_radius_m=0.0,
                    prominence_radius_m=1.5,
                    maximum_temporary_bytes=500,
                ),
            )
    assert run_label.value.code is TreetopDetectionErrorCode.RESOURCE_LIMIT
    assert run_label.value.estimated_temporary_bytes is not None
    assert run_label.value.estimated_temporary_bytes > 500

    with monkeypatch.context() as scoped:
        scoped.setattr(treetops, "_represent_plateaus", forbidden)
        with pytest.raises(TreetopDetectionError) as representative:
            detect_treetop_candidates(
                model,
                _config(
                    local_max_radius_m=0.5,
                    prominence_radius_m=1.5,
                    maximum_temporary_bytes=500,
                ),
            )
    assert representative.value.code is TreetopDetectionErrorCode.RESOURCE_LIMIT
    assert representative.value.estimated_temporary_bytes is not None
    assert representative.value.estimated_temporary_bytes > 500


def test_candidate_count_and_neighbour_evaluation_resource_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model([[5.0, np.nan, 4.0]])
    with pytest.raises(TreetopDetectionError) as candidates:
        detect_treetop_candidates(
            model,
            _config(
                local_max_radius_m=0.0,
                maximum_candidate_count=1,
            ),
        )
    assert candidates.value.candidate_count == 2

    metric_called = False

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        nonlocal metric_called
        metric_called = True
        raise AssertionError("metric evaluation must not begin")

    monkeypatch.setattr(treetops, "_evaluate_candidate_metrics", forbidden)
    with pytest.raises(TreetopDetectionError) as evaluations:
        detect_treetop_candidates(
            _model([[np.nan, 5.0, np.nan], [6.0, np.nan, 7.0], [np.nan, 8.0, np.nan]]),
            _config(
                local_max_radius_m=0.0,
                prominence_radius_m=1.5,
                maximum_candidate_neighbour_evaluations=31,
            ),
        )
    assert evaluations.value.code is TreetopDetectionErrorCode.RESOURCE_LIMIT
    assert evaluations.value.candidate_count == 4
    assert evaluations.value.prominence_offset_count == 8
    assert evaluations.value.candidate_neighbour_evaluations == 32
    assert not metric_called


def test_candidate_neighbour_evaluations_equal_limit_are_accepted() -> None:
    model = _model([[np.nan, 5.0, np.nan], [6.0, np.nan, 7.0], [np.nan, 8.0, np.nan]])
    result = detect_treetop_candidates(
        model,
        _config(
            local_max_radius_m=0.0,
            prominence_radius_m=1.5,
            maximum_candidate_neighbour_evaluations=32,
        ),
    )
    assert result.candidate_count == 4


def test_memory_error_is_chained(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_shape: tuple[int, int]) -> np.ndarray:
        raise MemoryError("simulated plateau allocation")

    monkeypatch.setattr(treetops, "_allocate_plateau_labels", fail)
    with pytest.raises(TreetopDetectionError) as raised:
        detect_treetop_candidates(_model([[5.0]]))
    assert raised.value.code is TreetopDetectionErrorCode.ALLOCATION_FAILED
    assert isinstance(raised.value.__cause__, MemoryError)
