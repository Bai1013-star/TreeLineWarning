from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import algorithms.vegetation as vegetation
import algorithms.vegetation.canopy_segmentation as segmentation
import numpy as np
import pytest
from algorithms.vegetation.canopy_height_model import (
    CanopyHeightModelConfig,
    CanopyHeightModelResult,
)
from algorithms.vegetation.canopy_segmentation import (
    CanopySegmentationConfig,
    CanopySegmentationError,
    CanopySegmentationErrorCode,
    CanopySegmentationResult,
    segment_canopy_regions,
)
from algorithms.vegetation.treetop_candidates import (
    TreetopCandidateResult,
    TreetopDetectionConfig,
)
from core.contracts.tree_inputs import TreeSegmentationInputIdentity


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _identity(point_count: int, source_id: str = "scene") -> TreeSegmentationInputIdentity:
    digest = "a" * 64
    return TreeSegmentationInputIdentity(
        source_id=source_id,
        manifest_source_id="manifest",
        origin_xyz=(0.0, 0.0, 0.0),
        point_count=point_count,
        v2_algorithm_version="v2",
        a2a_algorithm_version="a2a",
        a2b_algorithm_version="a2b",
        v2_npz_sha256=digest,
        v2_json_sha256=digest,
        a2a_npz_sha256=digest,
        a2a_json_sha256=digest,
        a2b_npz_sha256=digest,
        a2b_json_sha256=digest,
        a2a_input_a1_npz_sha256=digest,
        a2a_input_a1_json_sha256=digest,
        a2b_input_a1_npz_sha256=digest,
        a2b_input_a1_json_sha256=digest,
        a2b_input_a2a_npz_sha256=digest,
        a2b_input_a2a_json_sha256=digest,
        v2_input_v1_npz_sha256=digest,
        v2_input_v1_json_sha256=digest,
        v2_input_a2a_npz_sha256=digest,
        v2_input_a2a_json_sha256=digest,
        v2_input_a2b_npz_sha256=digest,
        v2_input_a2b_json_sha256=digest,
    )


def _model(
    height: Any,
    *,
    origin: tuple[float, float] | None = (0.0, 0.0),
    cell_size: float = 1.0,
    support: Any | None = None,
    identity: TreeSegmentationInputIdentity | None = None,
) -> CanopyHeightModelResult:
    array = np.array(height, dtype=np.float32, order="C", copy=True)
    if array.shape == (0,):
        array = np.empty((0, 0), dtype=np.float32)
    occupied = np.isfinite(array)
    counts = occupied.astype(np.int32)
    support_array = (
        np.where(occupied, 1, 0).astype(np.uint8)
        if support is None
        else np.array(support, dtype=np.uint8, order="C", copy=True)
    )
    peak_support = support_array.copy()
    point_count = int(np.sum(counts, dtype=np.int64))
    for value in (array, occupied, counts, support_array, peak_support):
        value.setflags(write=False)
    return CanopyHeightModelResult(
        source_identity=identity or _identity(point_count),
        config=CanopyHeightModelConfig(
            cell_size_m=cell_size,
            minimum_hag_m=0.0,
            maximum_grid_cells=10_000_000,
        ),
        grid_origin_xy=None if array.shape == (0, 0) else origin,
        canopy_height_m=array,
        occupied_mask=occupied,
        point_count_by_cell=counts,
        support_flags_by_cell=support_array,
        peak_support_flags_by_cell=peak_support,
        included_point_count=point_count,
        excluded_point_count=0,
    )


def _candidates(
    model: CanopyHeightModelResult,
    cells: list[tuple[int, int]],
    *,
    config: TreetopDetectionConfig | None = None,
) -> TreetopCandidateResult:
    cfg = config or TreetopDetectionConfig(
        minimum_peak_hag_m=0.0,
        local_max_radius_m=0.0,
        prominence_radius_m=1.5,
    )
    count = len(cells)
    rows = np.asarray([row for row, _ in cells], dtype=np.int64)
    columns = np.asarray([column for _, column in cells], dtype=np.int64)
    local_xy = np.empty((count, 2), dtype=np.float64)
    if count:
        assert model.grid_origin_xy is not None
        local_xy[:, 0] = (
            model.grid_origin_xy[0] + (columns.astype(np.float64) + 0.5) * model.config.cell_size_m
        )
        local_xy[:, 1] = (
            model.grid_origin_xy[1] + (rows.astype(np.float64) + 0.5) * model.config.cell_size_m
        )
        peak = model.canopy_height_m[rows, columns]
        point_count = model.point_count_by_cell[rows, columns]
        source_support = model.support_flags_by_cell[rows, columns]
        peak_support = model.peak_support_flags_by_cell[rows, columns]
    else:
        peak = np.empty(0, dtype=np.float32)
        point_count = np.empty(0, dtype=np.int32)
        source_support = np.empty(0, dtype=np.uint8)
        peak_support = np.empty(0, dtype=np.uint8)
    extent = (
        int(np.floor(np.float64(cfg.prominence_radius_m) / model.config.cell_size_m))
        if model.canopy_height_m.shape != (0, 0)
        else 0
    )
    edge = np.zeros(count, dtype=np.bool_)
    if count and extent:
        shape = model.canopy_height_m.shape
        edge = (
            (rows < extent)
            | (rows >= shape[0] - extent)
            | (columns < extent)
            | (columns >= shape[1] - extent)
        )
    return TreetopCandidateResult(
        source_identity=model.source_identity,
        source_chm_config=model.config,
        source_grid_origin_xy=model.grid_origin_xy,
        source_grid_shape=model.canopy_height_m.shape,
        config=cfg,
        local_maximum_plateau_count=count,
        evidence_rejected_count=0,
        suppressed_candidate_count=0,
        candidate_id=_readonly(np.arange(count), np.int32),
        row_index=_readonly(rows, np.int64),
        column_index=_readonly(columns, np.int64),
        local_xy=_readonly(local_xy, np.float64),
        peak_height_m=_readonly(peak, np.float32),
        prominence_m=_readonly(np.zeros(count), np.float32),
        prominence_available=_readonly(np.zeros(count), np.bool_),
        plateau_cell_count=_readonly(np.ones(count), np.int32),
        source_cell_point_count=_readonly(point_count, np.int32),
        source_support_flags=_readonly(source_support, np.uint8),
        source_peak_support_flags=_readonly(peak_support, np.uint8),
        neighbour_occupied_cell_count=_readonly(np.zeros(count), np.int32),
        neighbourhood_occupied_fraction=_readonly(np.zeros(count), np.float32),
        neighbourhood_support_flags=_readonly(np.zeros(count), np.uint8),
        neighbourhood_support_fraction=_readonly(np.zeros((count, 3)), np.float32),
        edge_limited=_readonly(edge, np.bool_),
    )


def _segment(
    height: Any,
    cells: list[tuple[int, int]],
    **config_changes: Any,
) -> CanopySegmentationResult:
    model = _model(height)
    return segment_canopy_regions(
        model,
        _candidates(model, cells),
        CanopySegmentationConfig(**config_changes),
    )


def _arrays(result: CanopySegmentationResult) -> tuple[np.ndarray, ...]:
    return (
        result.region_label_by_cell,
        result.segmentation_domain_mask,
        result.ambiguity_mask,
        result.marker_flat_cell_index_by_candidate,
        result.region_cell_count_by_candidate,
        result.ambiguous_cell_count_by_candidate,
        result.region_touches_grid_edge_by_candidate,
    )


def test_public_api_signature_exports_and_frozen_contracts() -> None:
    expected = {
        "CanopySegmentationConfig": CanopySegmentationConfig,
        "CanopySegmentationErrorCode": CanopySegmentationErrorCode,
        "CanopySegmentationError": CanopySegmentationError,
        "CanopySegmentationResult": CanopySegmentationResult,
        "segment_canopy_regions": segment_canopy_regions,
    }
    for name, value in expected.items():
        assert getattr(vegetation, name) is value
    assert tuple(inspect.signature(segment_canopy_regions).parameters) == (
        "canopy_model",
        "treetop_candidates",
        "config",
    )
    assert [value.value for value in CanopySegmentationErrorCode] == [
        "resource_limit",
        "allocation_failed",
        "input_pair_mismatch",
        "internal_invariant",
    ]
    assert "candidate_count" not in CanopySegmentationResult.__dataclass_fields__
    with pytest.raises(FrozenInstanceError):
        CanopySegmentationConfig().connectivity = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_segmentation_hag_m": True},
        {"minimum_segmentation_hag_m": -0.1},
        {"minimum_segmentation_hag_m": np.inf},
        {"connectivity": True},
        {"connectivity": 6},
        {"maximum_grid_cells": 0},
        {"maximum_grid_cells": np.iinfo(np.int32).max + 1},
        {"maximum_candidate_count": np.iinfo(np.int32).max},
        {"maximum_component_count": 0},
        {"maximum_temporary_bytes": 0},
        {"maximum_queue_entries": np.iinfo(np.int32).max + 1},
        {"maximum_edge_evaluations": 0},
    ],
)
def test_config_rejects_invalid_values(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CanopySegmentationConfig(**changes)
    assert isinstance(
        CanopySegmentationConfig(maximum_grid_cells=np.int64(10)).maximum_grid_cells,
        int,
    )
    assert (
        CanopySegmentationConfig(
            minimum_segmentation_hag_m=np.float32(0.5)
        ).minimum_segmentation_hag_m
        == 0.5
    )


def test_error_contract_and_wrong_public_types() -> None:
    with pytest.raises(TypeError):
        segment_canopy_regions(object(), object())  # type: ignore[arg-type]
    model = _model([[1.0]])
    candidates = _candidates(model, [(0, 0)])
    with pytest.raises(TypeError):
        segment_canopy_regions(model, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        segment_canopy_regions(model, candidates, object())  # type: ignore[arg-type]
    error = CanopySegmentationError(
        CanopySegmentationErrorCode.RESOURCE_LIMIT,
        "limit",
        row_count=1,
        edge_evaluation_count=2,
    )
    assert error.code is CanopySegmentationErrorCode.RESOURCE_LIMIT
    assert error.edge_evaluation_count == 2
    with pytest.raises(ValueError):
        CanopySegmentationError("bad", "message")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CanopySegmentationError(CanopySegmentationErrorCode.RESOURCE_LIMIT, "")
    with pytest.raises(ValueError):
        CanopySegmentationError(
            CanopySegmentationErrorCode.RESOURCE_LIMIT,
            "bad",
            row_count=np.int64(1),  # type: ignore[arg-type]
        )


def test_empty_no_domain_and_unseeded_results() -> None:
    empty = _model(np.empty((0, 0), dtype=np.float32), origin=None)
    empty_result = segment_canopy_regions(empty, _candidates(empty, []))
    assert empty_result.source_grid_shape == (0, 0)
    assert empty_result.candidate_count == empty_result.domain_component_count == 0
    assert all(value.size == 0 for value in _arrays(empty_result))

    no_domain = _model([[np.nan]])
    no_domain_result = segment_canopy_regions(no_domain, _candidates(no_domain, []))
    assert no_domain_result.domain_component_count == 0
    assert no_domain_result.region_label_by_cell.tolist() == [[0]]

    unseeded = _segment([[1.0, np.nan, 2.0]], [])
    assert unseeded.unseeded_component_count == 2
    assert unseeded.seeded_component_count == 0
    assert unseeded.unassigned_domain_cell_count == 2
    assert not np.any(unseeded.ambiguity_mask)


def test_one_marker_components_are_direct_and_nonambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model([[1.0, 2.0, 3.0]])
    candidates = _candidates(model, [(0, 1)])

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("one-marker component must not use propagation")

    monkeypatch.setattr(segmentation, "_propagate", forbidden)
    result = segment_canopy_regions(model, candidates)
    assert result.region_label_by_cell.tolist() == [[1, 1, 1]]
    assert result.region_cell_count_by_candidate.tolist() == [3]
    assert result.seeded_component_count == 1
    assert result.unseeded_component_count == 0
    assert not np.any(result.ambiguity_mask)


def test_disconnected_component_policies_and_candidate_preservation() -> None:
    result = _segment(
        [[5.0, 4.0, np.nan, 3.0, np.nan, 6.0]],
        [(0, 0), (0, 5)],
        connectivity=4,
    )
    assert result.region_label_by_cell.tolist() == [[1, 1, 0, 0, 0, 2]]
    assert result.seeded_component_count == 2
    assert result.unseeded_component_count == 1
    assert result.region_cell_count_by_candidate.tolist() == [2, 1]


def test_flat_row_tie_is_deterministic_and_ambiguous() -> None:
    result = _segment([[5.0, 5.0, 5.0, 5.0, 5.0]], [(0, 0), (0, 4)])
    assert result.region_label_by_cell.tolist() == [[1, 1, 1, 2, 2]]
    assert result.ambiguity_mask.tolist() == [[False, False, True, False, False]]
    assert result.ambiguous_cell_count_by_candidate.tolist() == [1, 0]
    assert result.region_cell_count_by_candidate.tolist() == [3, 2]


def test_ambiguity_propagates_through_common_optimal_tail() -> None:
    height = np.array(
        [
            [5.0, np.nan, 5.0],
            [np.nan, 5.0, np.nan],
            [np.nan, 5.0, np.nan],
            [np.nan, 5.0, np.nan],
        ],
        dtype=np.float32,
    )
    result = _segment(height, [(0, 0), (0, 2)], connectivity=8)
    assert result.region_label_by_cell[1:, 1].tolist() == [1, 1, 1]
    assert result.ambiguity_mask[1:, 1].tolist() == [True, True, True]
    assert not result.ambiguity_mask[0, 0]
    assert not result.ambiguity_mask[0, 2]


def test_widest_path_beats_nearer_low_saddle() -> None:
    height = np.array(
        [
            [9.0, 2.0, 8.0],
            [9.0, 9.0, 9.0],
        ],
        dtype=np.float32,
    )
    result = _segment(height, [(0, 0), (0, 2)], connectivity=4)
    assert result.region_label_by_cell[0, 1] == 1
    assert result.region_label_by_cell[1, 1] == 1
    assert result.region_label_by_cell[0, 2] == 2


def test_cardinal_and_diagonal_connectivity_are_explicit() -> None:
    height = [[5.0, np.nan], [np.nan, 4.0]]
    four = _segment(height, [(0, 0)], connectivity=4)
    eight = _segment(height, [(0, 0)], connectivity=8)
    assert four.region_label_by_cell.tolist() == [[1, 0], [0, 0]]
    assert four.unseeded_component_count == 1
    assert eight.region_label_by_cell.tolist() == [[1, 0], [0, 1]]
    assert eight.unseeded_component_count == 0


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ((0, 0), (0, 0), 0),
        ((1, 0), (0, 1), -1),
        ((2, 0), (0, 1), 1),
        ((1, 1), (2, 0), 1),
        ((100_000, 20_000), (99_999, 20_001), -1),
    ],
)
def test_exact_weighted_distance_comparator(
    left: tuple[int, int],
    right: tuple[int, int],
    expected: int,
) -> None:
    actual = segmentation._compare_weighted_distance(*left, *right)
    assert actual == expected
    assert segmentation._compare_weighted_distance(*right, *left) == -expected


def test_weighted_distance_comparator_transitivity() -> None:
    values = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    ordered = sorted(
        values,
        key=lambda value: value[0] + value[1] * np.sqrt(2.0),
    )
    for first, second, third in zip(ordered, ordered[1:], ordered[2:], strict=False):
        assert segmentation._compare_weighted_distance(*first, *second) < 0
        assert segmentation._compare_weighted_distance(*second, *third) < 0
        assert segmentation._compare_weighted_distance(*first, *third) < 0


def test_indexed_heap_priority_updates_and_int32_storage() -> None:
    bottleneck = np.asarray([2.0, 3.0, 3.0, 3.0], dtype=np.float32)
    cardinal = np.asarray([0, 2, 1, 1], dtype=np.int32)
    diagonal = np.zeros(4, dtype=np.int32)
    owner = np.asarray([1, 1, 2, 1], dtype=np.int32)
    config = CanopySegmentationConfig(maximum_queue_entries=4)
    heap = segmentation._IndexedMaxHeap(
        4,
        4,
        bottleneck,
        cardinal,
        diagonal,
        owner,
        config,
        (1, 4),
        4,
        2,
    )
    for cell in (0, 1, 2, 3):
        heap.push_or_update(cell)
    assert heap.cells.dtype == np.int32
    assert heap.positions.dtype == np.int32
    assert [heap.pop(), heap.pop(), heap.pop(), heap.pop()] == [3, 2, 1, 0]

    heap = segmentation._IndexedMaxHeap(
        2,
        4,
        bottleneck,
        cardinal,
        diagonal,
        owner,
        CanopySegmentationConfig(maximum_queue_entries=2),
        (1, 4),
        2,
        2,
    )
    heap.push_or_update(0)
    bottleneck[0] = 4.0
    heap.push_or_update(0)
    assert heap.size == 1 and heap.pop() == 0


def test_edge_budget_preflight_and_cumulative_counter() -> None:
    equal = CanopySegmentationConfig(
        connectivity=8,
        maximum_edge_evaluations=240,
    )
    budget = segmentation._preflight_edge_budget(10, equal, (2, 5), 2)
    budget.consume(100)
    budget.consume(140)
    assert budget.value == 240
    with pytest.raises(CanopySegmentationError) as raised:
        budget.consume()
    assert raised.value.code is CanopySegmentationErrorCode.RESOURCE_LIMIT
    assert raised.value.edge_evaluation_count == 241
    with pytest.raises(CanopySegmentationError):
        segmentation._preflight_edge_budget(11, equal, (1, 11), 0)
    four = segmentation._preflight_edge_budget(
        10,
        CanopySegmentationConfig(connectivity=4, maximum_edge_evaluations=120),
        (2, 5),
        0,
    )
    assert four.maximum == 120


def test_internal_allocation_dtypes() -> None:
    assert segmentation._allocate_component_labels((2, 2)).dtype == np.int32
    assert segmentation._allocate_component_stack(4).dtype == np.int32
    assert segmentation._allocate_owner((2, 2)).dtype == np.int32
    assert segmentation._allocate_heap_cells(4).dtype == np.int32
    assert segmentation._allocate_heap_positions(4).dtype == np.int32
    bottleneck, cardinal, diagonal, settled = segmentation._allocate_score_arrays((2, 2))
    assert bottleneck.dtype == np.float32
    assert cardinal.dtype == diagonal.dtype == np.int32
    assert settled.dtype == np.bool_
    assert segmentation._allocate_ambiguity_queue(4).dtype == np.int32


def test_threshold_excluding_marker_is_value_error() -> None:
    model = _model([[1.0, 5.0]])
    candidates = _candidates(model, [(0, 0)])
    with pytest.raises(ValueError, match="excludes"):
        segment_canopy_regions(
            model,
            candidates,
            CanopySegmentationConfig(minimum_segmentation_hag_m=2.0),
        )


def test_true_pair_mismatches_are_typed() -> None:
    model = _model([[5.0]])
    candidates = _candidates(model, [(0, 0)])
    other_identity = _identity(1, "other")
    mismatched = replace(candidates, source_identity=other_identity)
    with pytest.raises(CanopySegmentationError) as identity:
        segment_canopy_regions(model, mismatched)
    assert identity.value.code is CanopySegmentationErrorCode.INPUT_PAIR_MISMATCH

    wrong_peak = replace(candidates, peak_height_m=_readonly([4.0], np.float32))
    with pytest.raises(CanopySegmentationError) as peak:
        segment_canopy_regions(model, wrong_peak)
    assert peak.value.code is CanopySegmentationErrorCode.INPUT_PAIR_MISMATCH


def test_all_independently_valid_pair_mismatch_categories_are_typed() -> None:
    identity = _identity(2)
    model = _model([[5.0, 4.0]], identity=identity)
    cases = [
        _candidates(
            _model([[5.0, 4.0]], origin=(1.0, 0.0), identity=identity),
            [(0, 0)],
        ),
        _candidates(
            _model([[5.0, 4.0]], cell_size=0.5, identity=identity),
            [(0, 0)],
        ),
        _candidates(
            _model([[5.0], [4.0]], identity=identity),
            [(0, 0)],
        ),
        replace(
            _candidates(model, [(0, 0)]),
            source_cell_point_count=_readonly([2], np.int32),
        ),
        replace(
            _candidates(model, [(0, 0)]),
            source_support_flags=_readonly([2], np.uint8),
            source_peak_support_flags=_readonly([2], np.uint8),
        ),
    ]
    for candidates in cases:
        with pytest.raises(CanopySegmentationError) as raised:
            segment_canopy_regions(model, candidates)
        assert raised.value.code is CanopySegmentationErrorCode.INPUT_PAIR_MISMATCH


def test_runtime_revalidation_rejects_mutated_writeability() -> None:
    model = _model([[5.0]])
    candidates = _candidates(model, [(0, 0)])
    model.canopy_height_m.setflags(write=True)
    try:
        with pytest.raises(ValueError, match="read-only"):
            segment_canopy_regions(model, candidates)
    finally:
        model.canopy_height_m.setflags(write=False)


def test_support_status_is_geometry_neutral() -> None:
    height = [[5.0, 4.0, 5.0]]
    results = []
    for support in ([[1, 1, 1]], [[2, 2, 2]], [[4, 4, 4]], [[1, 2, 4]]):
        model = _model(height, support=support)
        results.append(segment_canopy_regions(model, _candidates(model, [(0, 0), (0, 2)])))
    for result in results[1:]:
        assert np.array_equal(
            result.segmentation_domain_mask,
            results[0].segmentation_domain_mask,
        )
        assert np.array_equal(result.region_label_by_cell, results[0].region_label_by_cell)
        assert np.array_equal(result.ambiguity_mask, results[0].ambiguity_mask)
        assert result.seeded_component_count == results[0].seeded_component_count
        assert result.unseeded_component_count == results[0].unseeded_component_count
        assert np.array_equal(
            result.region_cell_count_by_candidate,
            results[0].region_cell_count_by_candidate,
        )
        assert np.array_equal(
            result.region_touches_grid_edge_by_candidate,
            results[0].region_touches_grid_edge_by_candidate,
        )


@pytest.mark.parametrize(
    "origin",
    [(10.0, 20.0), (-10.0, -20.0), (1.0e12, -1.0e12)],
)
def test_translation_does_not_change_segmentation(origin: tuple[float, float]) -> None:
    base = _model([[5.0, 5.0, 5.0]], origin=(0.0, 0.0))
    translated = _model([[5.0, 5.0, 5.0]], origin=origin)
    first = segment_canopy_regions(base, _candidates(base, [(0, 0), (0, 2)]))
    second = segment_canopy_regions(
        translated,
        _candidates(translated, [(0, 0), (0, 2)]),
    )
    assert np.array_equal(first.region_label_by_cell, second.region_label_by_cell)
    assert np.array_equal(first.ambiguity_mask, second.ambiguity_mask)


def test_determinism_cwd_no_input_mutation_and_independent_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model([[5.0, 4.0, 5.0]])
    candidates = _candidates(model, [(0, 0), (0, 2)])
    snapshots = [
        (value.copy(), value.flags.writeable)
        for value in (
            model.canopy_height_m,
            model.occupied_mask,
            candidates.candidate_id,
            candidates.local_xy,
        )
    ]
    first = segment_canopy_regions(model, candidates)
    monkeypatch.chdir(tmp_path)
    second = segment_canopy_regions(model, candidates)
    for left, right in zip(_arrays(first), _arrays(second), strict=True):
        assert np.array_equal(left, right)
        assert not np.shares_memory(left, right)
    assert (
        first.seeded_component_count,
        first.unseeded_component_count,
        first.assigned_cell_count,
        first.ambiguous_cell_count,
    ) == (
        second.seeded_component_count,
        second.unseeded_component_count,
        second.assigned_cell_count,
        second.ambiguous_cell_count,
    )
    for value, (snapshot, writeable) in zip(
        (
            model.canopy_height_m,
            model.occupied_mask,
            candidates.candidate_id,
            candidates.local_xy,
        ),
        snapshots,
        strict=True,
    ):
        assert np.array_equal(value, snapshot, equal_nan=True)
        assert value.flags.writeable is writeable


def test_public_result_arrays_are_detached_owning_c_and_readonly() -> None:
    model = _model([[5.0, 4.0, 5.0]])
    candidates = _candidates(model, [(0, 0), (0, 2)])
    result = segment_canopy_regions(model, candidates)
    expected = (
        (np.dtype(np.int32), (1, 3)),
        (np.dtype(np.bool_), (1, 3)),
        (np.dtype(np.bool_), (1, 3)),
        (np.dtype(np.int64), (2,)),
        (np.dtype(np.int64), (2,)),
        (np.dtype(np.int64), (2,)),
        (np.dtype(np.bool_), (2,)),
    )
    inputs = (
        model.canopy_height_m,
        model.occupied_mask,
        model.point_count_by_cell,
        candidates.candidate_id,
        candidates.row_index,
        candidates.local_xy,
    )
    for value, (dtype, shape) in zip(_arrays(result), expected, strict=True):
        assert value.dtype == dtype and value.shape == shape
        assert value.flags.owndata and value.flags.c_contiguous
        assert not value.flags.writeable
        assert not any(np.shares_memory(value, source) for source in inputs)
        with pytest.raises(ValueError):
            value.flat[0] = value.flat[0]


def test_result_constructor_does_not_run_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _segment([[5.0, 5.0, 5.0]], [(0, 0), (0, 2)])

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("result construction must not rerun components")

    monkeypatch.setattr(segmentation, "_label_components", forbidden)
    rebuilt = replace(result)
    assert rebuilt.domain_component_count == result.domain_component_count


@pytest.mark.parametrize(
    "change",
    [
        {"seeded_component_count": -1},
        {"seeded_component_count": 3},
        {"assigned_cell_count": 2},
        {"unassigned_domain_cell_count": 1},
        {"ambiguous_cell_count": 0},
        {"region_cell_count_by_candidate": _readonly([1, 1], np.int64)},
        {"ambiguous_cell_count_by_candidate": _readonly([0, 0], np.int64)},
        {"region_touches_grid_edge_by_candidate": _readonly([False, False], np.bool_)},
    ],
)
def test_result_constructor_rejects_audit_inconsistency(change: dict[str, Any]) -> None:
    result = _segment([[5.0, 5.0, 5.0]], [(0, 0), (0, 2)])
    with pytest.raises(ValueError):
        replace(result, **change)


def test_result_constructor_rejects_storage_and_label_failures() -> None:
    result = _segment([[5.0, 5.0, 5.0]], [(0, 0), (0, 2)])
    writable = result.region_label_by_cell.copy()
    with pytest.raises(ValueError, match="read-only"):
        replace(result, region_label_by_cell=writable)
    base = result.region_label_by_cell.copy()
    view = base[:, :]
    view.setflags(write=False)
    with pytest.raises(ValueError, match="own"):
        replace(result, region_label_by_cell=view)
    bad = result.region_label_by_cell.copy()
    bad[0, 0] = 0
    bad.setflags(write=False)
    with pytest.raises(ValueError, match="marker"):
        replace(result, region_label_by_cell=bad)
    duplicate = _readonly([0, 0], np.int64)
    with pytest.raises(ValueError, match="unique"):
        replace(result, marker_flat_cell_index_by_candidate=duplicate)


def test_resource_preflight_and_memory_error_chaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model([[5.0]])
    candidates = _candidates(model, [(0, 0)])
    with pytest.raises(CanopySegmentationError) as grid:
        segment_canopy_regions(
            model,
            candidates,
            CanopySegmentationConfig(maximum_grid_cells=1, maximum_temporary_bytes=1),
        )
    assert grid.value.code is CanopySegmentationErrorCode.RESOURCE_LIMIT
    assert grid.value.estimated_temporary_bytes is not None

    def fail(_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        raise MemoryError("simulated domain allocation")

    monkeypatch.setattr(segmentation, "_allocate_domain", fail)
    with pytest.raises(CanopySegmentationError) as allocation:
        segment_canopy_regions(model, candidates)
    assert allocation.value.code is CanopySegmentationErrorCode.ALLOCATION_FAILED
    assert isinstance(allocation.value.__cause__, MemoryError)


def test_candidate_and_component_properties_are_readonly() -> None:
    result = _segment([[5.0]], [(0, 0)])
    assert result.candidate_count == 1
    assert result.domain_component_count == 1
    with pytest.raises(FrozenInstanceError):
        result.seeded_component_count = 2  # type: ignore[misc]


def test_synthetic_tests_only_demonstrate_code_level_generality() -> None:
    """Synthetic tests demonstrate code-level generality, not botanical generalization."""

    for cell_size in (0.10, 0.20, 0.25, 0.50):
        model = _model([[5.0, 4.0, 5.0]], cell_size=cell_size)
        result = segment_canopy_regions(model, _candidates(model, [(0, 0), (0, 2)]))
        assert result.region_label_by_cell.tolist() == [[1, 1, 2]]


def test_runtime_revalidation_rejects_noncanonical_frozen_scalars() -> None:
    model = _model([[5.0]])
    candidates = _candidates(model, [(0, 0)])
    object.__setattr__(model.config, "maximum_grid_cells", np.int64(10_000_000))
    try:
        with pytest.raises(ValueError, match="canonical"):
            segment_canopy_regions(model, candidates)
    finally:
        object.__setattr__(model.config, "maximum_grid_cells", 10_000_000)

    object.__setattr__(model.source_identity, "point_count", np.int64(1))
    try:
        with pytest.raises(ValueError, match="canonical"):
            segment_canopy_regions(model, candidates)
    finally:
        object.__setattr__(model.source_identity, "point_count", 1)

    result = segment_canopy_regions(model, candidates)
    object.__setattr__(result.config, "maximum_queue_entries", np.int64(10_000_000))
    try:
        with pytest.raises(ValueError, match="canonical"):
            replace(result)
    finally:
        object.__setattr__(result.config, "maximum_queue_entries", 10_000_000)


def test_domain_threshold_is_inclusive_and_preserves_unseeded_cells() -> None:
    model = _model([[1.0, 2.0, np.nan, 3.0]])
    candidates = _candidates(model, [(0, 3)])
    result = segment_canopy_regions(
        model,
        candidates,
        CanopySegmentationConfig(
            minimum_segmentation_hag_m=np.float32(2.0),
            connectivity=4,
        ),
    )
    assert result.segmentation_domain_mask.tolist() == [[False, True, False, True]]
    assert result.region_label_by_cell.tolist() == [[0, 0, 0, 1]]
    assert result.unseeded_component_count == 1
    assert result.unassigned_domain_cell_count == 1


@pytest.mark.parametrize(
    ("height", "marker"),
    [
        ([[5.0]], (0, 0)),
        ([[5.0, 4.0, 3.0, 2.0]], (0, 2)),
        ([[5.0], [4.0], [3.0], [2.0]], (2, 0)),
        (
            [
                [5.0, 5.0, 5.0],
                [5.0, np.nan, 5.0],
                [5.0, 5.0, 5.0],
            ],
            (0, 0),
        ),
        ([[5.0, 1.0, 5.0]], (0, 0)),
    ],
)
def test_one_marker_geometry_is_direct_nonambiguous_and_complete(
    height: Any,
    marker: tuple[int, int],
) -> None:
    result = _segment(height, [marker], connectivity=4)
    assert np.all(result.region_label_by_cell[result.segmentation_domain_mask] == 1)
    assert not np.any(result.ambiguity_mask)
    assert result.region_cell_count_by_candidate[0] == np.count_nonzero(
        result.segmentation_domain_mask
    )
    assert result.region_label_by_cell[marker] == 1


def test_component_labels_start_row_major_and_component_limit_is_hard() -> None:
    domain = np.asarray(
        [[True, False, True], [False, False, False], [True, True, False]],
        dtype=np.bool_,
    )
    owner = np.zeros(domain.shape, dtype=np.int32)
    budget = segmentation._EdgeBudget(100, domain.shape, 4, 0)
    labels, marker_counts, sole_labels, count, seeded, unseeded = segmentation._label_components(
        domain,
        owner,
        CanopySegmentationConfig(connectivity=4),
        budget,
    )
    assert labels.tolist() == [[0, -1, 1], [-1, -1, -1], [2, 2, -1]]
    assert marker_counts[:count].tolist() == [0, 0, 0]
    assert sole_labels[:count].tolist() == [0, 0, 0]
    assert (count, seeded, unseeded) == (3, 0, 3)

    with pytest.raises(CanopySegmentationError) as raised:
        _segment(
            [[1.0, np.nan, 1.0, np.nan, 1.0]],
            [],
            connectivity=4,
            maximum_component_count=2,
        )
    assert raised.value.code is CanopySegmentationErrorCode.RESOURCE_LIMIT
    assert raised.value.component_count == 3


@pytest.mark.parametrize(
    ("height", "markers"),
    [
        ([[9.0, 8.0, 7.0, 8.0, 9.0]], [(0, 0), (0, 4)]),
        ([[9.0, 7.0, 8.0]], [(0, 0), (0, 2)]),
        (
            [[9.0, 8.0, 7.0], [8.0, 7.0, 8.0], [7.0, 8.0, 9.0]],
            [(0, 0), (2, 2)],
        ),
        ([[9.0, 6.0, 8.0, 6.0, 7.0]], [(0, 0), (0, 2), (0, 4)]),
        ([[9.0, 2.0, 3.0, 8.0]], [(0, 0), (0, 2), (0, 3)]),
    ],
)
def test_multi_marker_topographies_preserve_every_marker(
    height: Any,
    markers: list[tuple[int, int]],
) -> None:
    result = _segment(height, markers, connectivity=8)
    for candidate_id, marker in enumerate(markers):
        assert result.region_label_by_cell[marker] == candidate_id + 1
        assert result.region_cell_count_by_candidate[candidate_id] > 0
    assert result.assigned_cell_count == np.count_nonzero(result.segmentation_domain_mask)


def test_flat_two_dimensional_and_three_marker_ties_are_deterministic() -> None:
    two = _segment(np.full((3, 3), 5.0), [(1, 0), (1, 2)], connectivity=4)
    assert two.region_label_by_cell[:, 1].tolist() == [1, 1, 1]
    assert two.ambiguity_mask[:, 1].tolist() == [True, True, True]
    assert not two.ambiguity_mask[1, 0] and not two.ambiguity_mask[1, 2]

    three = _segment([[5.0] * 7], [(0, 0), (0, 3), (0, 6)], connectivity=4)
    assert three.region_label_by_cell.tolist() == [[1, 1, 2, 2, 2, 3, 3]]
    assert three.ambiguity_mask.tolist() == [[False, False, False, False, False, False, False]]
    assert np.all(three.region_cell_count_by_candidate > 0)


def test_candidate_order_only_changes_final_id_tie_break() -> None:
    model = _model([[5.0, 5.0, 5.0, 5.0, 5.0]])
    forward = segment_canopy_regions(model, _candidates(model, [(0, 0), (0, 4)]))
    reverse = segment_canopy_regions(model, _candidates(model, [(0, 4), (0, 0)]))
    assert np.array_equal(
        forward.segmentation_domain_mask,
        reverse.segmentation_domain_mask,
    )
    assert np.array_equal(forward.ambiguity_mask, reverse.ambiguity_mask)
    assert forward.ambiguity_mask[0, 2]
    assert forward.region_label_by_cell[0, 2] == 1
    assert reverse.region_label_by_cell[0, 2] == 1


def test_nonoptimal_edges_do_not_propagate_ambiguity() -> None:
    height = np.asarray([5.0, 1.0], dtype=np.float32)
    bottleneck = np.asarray([5.0, 5.0], dtype=np.float32)
    cardinal = np.asarray([0, 0], dtype=np.int32)
    diagonal = np.asarray([0, 0], dtype=np.int32)
    assert not segmentation._is_optimal_predecessor(
        0,
        1,
        False,
        height,
        bottleneck,
        cardinal,
        diagonal,
    )


def test_exact_edge_accounting_is_cumulative_across_all_phases() -> None:
    height = np.asarray([[5.0, 5.0, 5.0]], dtype=np.float32)
    active = np.ones((1, 3), dtype=np.bool_)
    owner = np.asarray([[1, 0, 2]], dtype=np.int32)
    marker_flat = np.asarray([0, 2], dtype=np.int64)
    config = CanopySegmentationConfig(
        connectivity=4,
        maximum_edge_evaluations=100,
    )
    budget = segmentation._EdgeBudget(100, (1, 3), 3, 2)
    component_outputs = segmentation._label_components(
        active,
        owner,
        config,
        budget,
    )
    assert budget.value == 2
    del component_outputs
    bottleneck, cardinal, diagonal, _settled = segmentation._propagate(
        height,
        active,
        owner,
        marker_flat,
        config,
        budget,
    )
    assert budget.value == 6
    ambiguity = segmentation._ambiguity_fixed_point(
        height,
        active,
        owner,
        bottleneck,
        cardinal,
        diagonal,
        config,
        budget,
    )
    assert ambiguity.tolist() == [[False, True, False]]
    assert budget.value == 10


def test_heap_capacity_overflow_empty_pop_and_repeat_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bottleneck = np.full(3, 5.0, dtype=np.float32)
    cardinal = np.zeros(3, dtype=np.int32)
    diagonal = np.zeros(3, dtype=np.int32)
    owner = np.ones(3, dtype=np.int32)
    config = CanopySegmentationConfig(maximum_queue_entries=3)

    def make_heap(capacity: int = 3) -> Any:
        return segmentation._IndexedMaxHeap(
            capacity,
            3,
            bottleneck,
            cardinal,
            diagonal,
            owner,
            config,
            (1, 3),
            3,
            1,
        )

    orders = []
    for _ in range(2):
        heap = make_heap()
        for cell in (2, 0, 1):
            heap.push_or_update(cell)
        orders.append([heap.pop(), heap.pop(), heap.pop()])
    assert orders == [[0, 1, 2], [0, 1, 2]]

    empty = make_heap()
    with pytest.raises(CanopySegmentationError) as raised:
        empty.pop()
    assert raised.value.code is CanopySegmentationErrorCode.INTERNAL_INVARIANT

    overflow = make_heap(1)
    overflow.push_or_update(0)
    with pytest.raises(CanopySegmentationError) as capacity:
        overflow.push_or_update(1)
    assert capacity.value.queue_entry_count == 2

    def fail(_capacity: int) -> np.ndarray:
        raise MemoryError("simulated heap allocation")

    monkeypatch.setattr(segmentation, "_allocate_heap_cells", fail)
    with pytest.raises(MemoryError, match="simulated"):
        make_heap()


def test_valid_result_direct_construction_variants() -> None:
    values = (
        segment_canopy_regions(
            _model(np.empty((0, 0), dtype=np.float32), origin=None),
            _candidates(_model(np.empty((0, 0), dtype=np.float32), origin=None), []),
        ),
        _segment([[1.0, np.nan, 2.0]], []),
        _segment([[5.0]], [(0, 0)]),
        _segment([[5.0, 5.0, 5.0]], [(0, 0), (0, 2)]),
    )
    for value in values:
        assert replace(value) == value


def test_result_constructor_rejects_extended_grid_and_array_corruption() -> None:
    result = _segment([[5.0, 5.0, 5.0]], [(0, 0), (0, 2)])

    invalid_changes: list[dict[str, Any]] = [
        {"source_grid_shape": (1, np.int64(3))},
        {"source_grid_origin_xy": (np.float64(0.0), 0.0)},
        {"region_label_by_cell": _readonly([[1, 1, 2]], np.int64)},
        {"region_label_by_cell": _readonly([1, 1, 2], np.int32)},
        {"region_label_by_cell": _readonly([[1, 1], [2, 2]], np.int32)},
        {"ambiguity_mask": _readonly([[False, False, False]], np.int32)},
        {"marker_flat_cell_index_by_candidate": _readonly([0, 3], np.int64)},
        {"marker_flat_cell_index_by_candidate": _readonly([0, 1], np.int64)},
        {"region_cell_count_by_candidate": _readonly([2], np.int64)},
        {"ambiguous_cell_count_by_candidate": _readonly([1], np.int64)},
        {"region_touches_grid_edge_by_candidate": _readonly([True], np.bool_)},
        {"assigned_cell_count": np.int64(result.assigned_cell_count)},
        {"source_identity": object()},
        {"config": object()},
    ]
    for changes in invalid_changes:
        with pytest.raises(ValueError):
            replace(result, **changes)

    empty_model = _model(np.empty((0, 0), dtype=np.float32), origin=None)
    empty = segment_canopy_regions(empty_model, _candidates(empty_model, []))
    with pytest.raises(ValueError):
        replace(empty, seeded_component_count=1)

    negative = result.region_label_by_cell.copy()
    negative[0, 1] = -1
    negative.setflags(write=False)
    excessive = result.region_label_by_cell.copy()
    excessive[0, 1] = 3
    excessive.setflags(write=False)
    outside_domain = result.segmentation_domain_mask.copy()
    outside_domain[0, 1] = False
    outside_domain.setflags(write=False)
    ambiguity_zero = result.ambiguity_mask.copy()
    ambiguity_zero[0, 1] = True
    ambiguity_zero.setflags(write=False)
    zero_label = result.region_label_by_cell.copy()
    zero_label[0, 1] = 0
    zero_label.setflags(write=False)
    for changes in (
        {"region_label_by_cell": negative},
        {"region_label_by_cell": excessive},
        {"segmentation_domain_mask": outside_domain},
        {
            "region_label_by_cell": zero_label,
            "ambiguity_mask": ambiguity_zero,
        },
    ):
        with pytest.raises(ValueError):
            replace(result, **changes)


@pytest.mark.parametrize(
    ("helper_name", "height", "markers"),
    [
        ("_allocate_domain", [[5.0]], [(0, 0)]),
        ("_allocate_component_labels", [[5.0]], [(0, 0)]),
        ("_allocate_component_stack", [[5.0]], [(0, 0)]),
        ("_allocate_component_metadata", [[5.0]], [(0, 0)]),
        ("_allocate_score_arrays", [[5.0, 5.0, 5.0]], [(0, 0), (0, 2)]),
        ("_allocate_heap_cells", [[5.0, 5.0, 5.0]], [(0, 0), (0, 2)]),
        ("_allocate_heap_positions", [[5.0, 5.0, 5.0]], [(0, 0), (0, 2)]),
        ("_allocate_ambiguity_queue", [[5.0, 5.0, 5.0]], [(0, 0), (0, 2)]),
        ("_freeze_array", [[5.0]], [(0, 0)]),
    ],
)
def test_all_guarded_allocation_failures_chain_memory_error(
    helper_name: str,
    height: Any,
    markers: list[tuple[int, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(height)
    candidates = _candidates(model, markers)
    snapshots = tuple(value.copy() for value in (model.canopy_height_m, candidates.candidate_id))

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise MemoryError(f"simulated {helper_name}")

    monkeypatch.setattr(segmentation, helper_name, fail)
    with pytest.raises(CanopySegmentationError) as raised:
        segment_canopy_regions(model, candidates)
    assert raised.value.code is CanopySegmentationErrorCode.ALLOCATION_FAILED
    assert isinstance(raised.value.__cause__, MemoryError)
    assert np.array_equal(model.canopy_height_m, snapshots[0], equal_nan=True)
    assert np.array_equal(candidates.candidate_id, snapshots[1])


def test_resource_limits_cover_candidates_components_queue_and_freeze_peak() -> None:
    model = _model([[5.0, 5.0, 5.0]])
    candidates = _candidates(model, [(0, 0), (0, 2)])
    with pytest.raises(CanopySegmentationError) as candidate_limit:
        segment_canopy_regions(
            model,
            candidates,
            CanopySegmentationConfig(maximum_candidate_count=1),
        )
    assert candidate_limit.value.candidate_count == 2

    with pytest.raises(CanopySegmentationError) as queue_limit:
        segment_canopy_regions(
            model,
            candidates,
            CanopySegmentationConfig(maximum_queue_entries=2),
        )
    assert queue_limit.value.queue_entry_count == 3

    one = _model([[5.0]])
    with pytest.raises(CanopySegmentationError) as component_peak:
        segment_canopy_regions(
            one,
            _candidates(one, [(0, 0)]),
            CanopySegmentationConfig(maximum_temporary_bytes=29),
        )
    assert component_peak.value.estimated_temporary_bytes == 30

    with pytest.raises(CanopySegmentationError) as freeze_peak:
        segment_canopy_regions(
            one,
            _candidates(one, [(0, 0)]),
            CanopySegmentationConfig(maximum_temporary_bytes=49),
        )
    assert freeze_peak.value.estimated_temporary_bytes == 124

    with pytest.raises(CanopySegmentationError) as propagation_peak:
        segment_canopy_regions(
            model,
            candidates,
            CanopySegmentationConfig(maximum_temporary_bytes=82),
        )
    assert propagation_peak.value.estimated_temporary_bytes == 103

    phase_exclusive = segment_canopy_regions(
        model,
        candidates,
        CanopySegmentationConfig(maximum_temporary_bytes=260),
    )
    assert phase_exclusive.assigned_cell_count == 3


def test_empty_result_arrays_are_owning_c_contiguous_and_readonly() -> None:
    model = _model(np.empty((0, 0), dtype=np.float32), origin=None)
    result = segment_canopy_regions(model, _candidates(model, []))
    for value in _arrays(result):
        assert value.flags.owndata and value.flags.c_contiguous
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.shape = (1,)


@pytest.mark.parametrize(
    "height",
    [
        [[5.0, 4.0, 5.0]],
        [[5.0], [4.0], [5.0]],
        [[5.0, np.nan, 4.0], [4.0, 3.0, 5.0]],
        np.full((4, 7), 5.0),
    ],
)
def test_shape_generalization_has_no_square_grid_assumption(height: Any) -> None:
    array = np.asarray(height, dtype=np.float32)
    finite = np.argwhere(np.isfinite(array))
    markers = [tuple(finite[0]), tuple(finite[-1])]
    result = _segment(array, markers, connectivity=8)
    assert result.region_label_by_cell.shape == array.shape
    assert np.all(result.region_cell_count_by_candidate > 0)


def test_resolution_generalization_places_flat_boundary_at_same_physical_midline() -> None:
    for cell_size in (0.10, 0.20, 0.25, 0.50):
        column_count = round(1.0 / cell_size)
        model = _model(
            np.full((1, column_count), 5.0, dtype=np.float32),
            cell_size=cell_size,
        )
        result = segment_canopy_regions(
            model,
            _candidates(model, [(0, 0), (0, column_count - 1)]),
            CanopySegmentationConfig(connectivity=4),
        )
        centres = (np.arange(column_count, dtype=np.float64) + 0.5) * cell_size
        assert np.all(result.region_label_by_cell[0, centres < 0.5] == 1)
        assert np.all(result.region_label_by_cell[0, centres > 0.5] == 2)
        midpoint = np.isclose(centres, 0.5, rtol=0.0, atol=1e-12)
        assert np.array_equal(result.ambiguity_mask[0], midpoint)


@pytest.mark.parametrize(("connectivity", "expected_edges"), [(4, 4), (8, 6)])
def test_component_dfs_counts_each_undirected_edge_once(
    connectivity: int,
    expected_edges: int,
) -> None:
    domain = np.ones((2, 2), dtype=np.bool_)
    owner = np.zeros((2, 2), dtype=np.int32)
    budget = segmentation._EdgeBudget(100, domain.shape, 4, 0)
    labels, _counts, _sole, component_count, seeded, unseeded = segmentation._label_components(
        domain,
        owner,
        CanopySegmentationConfig(connectivity=connectivity),
        budget,
    )
    assert budget.value == expected_edges
    assert labels.tolist() == [[0, 0], [0, 0]]
    assert (component_count, seeded, unseeded) == (1, 0, 1)
    assert not hasattr(segmentation, "_count_undirected_edges")


def test_corrected_component_preflight_includes_active_and_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total = domain_count = component_capacity = candidate_count = 1
    old_estimate = total * 9 + domain_count * 4 + component_capacity * 8
    corrected = segmentation._component_temporary_estimate(
        total,
        domain_count,
        component_capacity,
        candidate_count,
    )
    assert corrected == old_estimate + total + candidate_count * 8 == 30

    def forbidden(_shape: tuple[int, int]) -> np.ndarray:
        raise AssertionError("component allocation occurred after failed preflight")

    monkeypatch.setattr(segmentation, "_allocate_owner", forbidden)
    model = _model([[5.0]])
    with pytest.raises(CanopySegmentationError) as raised:
        segment_canopy_regions(
            model,
            _candidates(model, [(0, 0)]),
            CanopySegmentationConfig(maximum_temporary_bytes=29),
        )
    assert raised.value.code is CanopySegmentationErrorCode.RESOURCE_LIMIT
    assert raised.value.estimated_temporary_bytes == corrected


def test_corrected_ambiguity_preflight_and_settled_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total = active_count = 3
    candidate_count = 2
    old_estimate = total * 19 + active_count * 4
    corrected = segmentation._ambiguity_temporary_estimate(
        total,
        active_count,
        candidate_count,
    )
    assert corrected == old_estimate + candidate_count * 8 == 85

    model = _model([[5.0, 5.0, 5.0]])
    candidates = _candidates(model, [(0, 0), (0, 2)])
    original_preflight = segmentation._preflight_temporary

    with monkeypatch.context() as context:

        def ambiguity_only(
            phase: str,
            estimated_bytes: int,
            config: CanopySegmentationConfig,
            shape: tuple[int, int],
            domain_count: int,
            candidate_count: int,
        ) -> None:
            if phase == "ambiguity":
                original_preflight(
                    phase,
                    estimated_bytes,
                    config,
                    shape,
                    domain_count,
                    candidate_count,
                )

        def forbidden(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("ambiguity allocation occurred after failed preflight")

        context.setattr(segmentation, "_preflight_temporary", ambiguity_only)
        context.setattr(segmentation, "_ambiguity_fixed_point", forbidden)
        with pytest.raises(CanopySegmentationError) as raised:
            segment_canopy_regions(
                model,
                candidates,
                CanopySegmentationConfig(maximum_temporary_bytes=80),
            )
        assert raised.value.estimated_temporary_bytes == corrected

    observed = False

    def inspect_lifetime(
        phase: str,
        estimated_bytes: int,
        config: CanopySegmentationConfig,
        shape: tuple[int, int],
        domain_count: int,
        candidate_count: int,
    ) -> None:
        nonlocal observed
        if phase == "ambiguity":
            caller = inspect.currentframe()
            assert caller is not None and caller.f_back is not None
            assert "settled" not in caller.f_back.f_locals
            observed = True
        original_preflight(
            phase,
            estimated_bytes,
            config,
            shape,
            domain_count,
            candidate_count,
        )

    monkeypatch.setattr(segmentation, "_preflight_temporary", inspect_lifetime)
    segment_canopy_regions(model, candidates)
    assert observed


def test_corrected_freeze_preflight_precedes_detached_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corrected = segmentation._freeze_temporary_estimate(1, 1)
    minimum_without_constructor_workspace = 12 + 50
    assert corrected == 124
    assert corrected > minimum_without_constructor_workspace

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("freeze copy occurred after failed preflight")

    monkeypatch.setattr(segmentation, "_freeze_array", forbidden)
    model = _model([[5.0]])
    with pytest.raises(CanopySegmentationError) as raised:
        segment_canopy_regions(
            model,
            _candidates(model, [(0, 0)]),
            CanopySegmentationConfig(maximum_temporary_bytes=49),
        )
    assert raised.value.code is CanopySegmentationErrorCode.RESOURCE_LIMIT
    assert raised.value.estimated_temporary_bytes == corrected


def test_validation_limits_precede_content_pairing_and_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model([[5.0, 5.0]])
    no_candidates = _candidates(model, [])

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("guarded validation work ran after failed preflight")

    with monkeypatch.context() as context:
        context.setattr(segmentation, "_validate_chm_content", forbidden)
        with pytest.raises(CanopySegmentationError) as oversized:
            segment_canopy_regions(
                model,
                no_candidates,
                CanopySegmentationConfig(maximum_grid_cells=1),
            )
        assert oversized.value.total_cell_count == 2

    candidates = _candidates(model, [(0, 0), (0, 1)])
    with monkeypatch.context() as context:
        context.setattr(segmentation, "_validate_pair_content", forbidden)
        with pytest.raises(CanopySegmentationError) as candidate_limit:
            segment_canopy_regions(
                model,
                candidates,
                CanopySegmentationConfig(maximum_candidate_count=1),
            )
        assert candidate_limit.value.candidate_count == 2

    one = _model([[5.0]])
    one_candidate = _candidates(one, [(0, 0)])
    assert segmentation._validation_temporary_estimate(1, 1) == 24
    with monkeypatch.context() as context:
        context.setattr(segmentation, "_validate_chm_content", forbidden)
        context.setattr(segmentation, "_allocate_validation_workspace", forbidden)
        with pytest.raises(CanopySegmentationError) as temporary_limit:
            segment_canopy_regions(
                one,
                one_candidate,
                CanopySegmentationConfig(maximum_temporary_bytes=23),
            )
        assert temporary_limit.value.estimated_temporary_bytes == 24


@pytest.mark.parametrize(
    "helper_name",
    [
        "_validate_chm_structure",
        "_validate_candidate_structure",
        "_allocate_validation_workspace",
        "_validate_chm_content",
        "_validate_candidate_content",
        "_validate_pair_content",
    ],
)
def test_validation_memory_errors_are_typed_and_chained(
    helper_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model([[5.0]])
    candidates = _candidates(model, [(0, 0)])

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise MemoryError(f"simulated {helper_name}")

    monkeypatch.setattr(segmentation, helper_name, fail)
    with pytest.raises(CanopySegmentationError) as raised:
        segment_canopy_regions(model, candidates)
    assert raised.value.code is CanopySegmentationErrorCode.ALLOCATION_FAILED
    assert isinstance(raised.value.__cause__, MemoryError)
    if helper_name != "_validate_chm_structure":
        assert raised.value.row_count == 1
        assert raised.value.column_count == 1
        assert raised.value.total_cell_count == 1


@pytest.mark.parametrize(
    ("field_name", "corrupted_value"),
    [
        ("connectivity", 6),
        ("connectivity", np.int64(8)),
        ("maximum_grid_cells", np.int64(10_000_000)),
        ("maximum_temporary_bytes", -1),
        ("maximum_edge_evaluations", 0),
        ("minimum_segmentation_hag_m", np.float32(0.0)),
        ("minimum_segmentation_hag_m", float("nan")),
    ],
)
def test_supplied_config_is_runtime_revalidated_before_all_helpers(
    field_name: str,
    corrupted_value: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model([[5.0]])
    candidates = _candidates(model, [(0, 0)])
    config = CanopySegmentationConfig()
    original = getattr(config, field_name)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("runtime helper ran before supplied-config validation")

    for helper_name in (
        "_validate_chm_structure",
        "_preflight_temporary",
        "_allocate_domain",
        "_label_components",
        "_neighbour_offsets",
    ):
        monkeypatch.setattr(segmentation, helper_name, forbidden)
    object.__setattr__(config, field_name, corrupted_value)
    try:
        with pytest.raises(ValueError):
            segment_canopy_regions(model, candidates, config)
    finally:
        object.__setattr__(config, field_name, original)


@pytest.mark.parametrize(
    "corrupted_origin",
    [
        (0, 0),
        (np.float64(0.0), np.float64(0.0)),
    ],
)
def test_chm_runtime_origin_requires_exact_python_floats(
    corrupted_origin: tuple[Any, Any],
) -> None:
    model = _model([[5.0]])
    candidates = _candidates(model, [(0, 0)])
    original = model.grid_origin_xy
    object.__setattr__(model, "grid_origin_xy", corrupted_origin)
    try:
        with pytest.raises(ValueError, match="Python floats"):
            segment_canopy_regions(model, candidates)
    finally:
        object.__setattr__(model, "grid_origin_xy", original)


@pytest.mark.parametrize(
    "corrupted_origin",
    [
        (0, 0),
        (np.float64(0.0), np.float64(0.0)),
    ],
)
def test_candidate_runtime_origin_requires_exact_python_floats(
    corrupted_origin: tuple[Any, Any],
) -> None:
    model = _model([[5.0]])
    candidates = _candidates(model, [(0, 0)])
    original = candidates.source_grid_origin_xy
    object.__setattr__(candidates, "source_grid_origin_xy", corrupted_origin)
    try:
        with pytest.raises(ValueError, match="Python floats"):
            segment_canopy_regions(model, candidates)
    finally:
        object.__setattr__(candidates, "source_grid_origin_xy", original)


@pytest.mark.parametrize(
    "origin",
    [
        (0.0, 0.0),
        (-100.0, -200.0),
        (1.0e12, -1.0e12),
    ],
)
def test_canonical_python_float_origins_and_config_remain_accepted(
    origin: tuple[float, float],
) -> None:
    model = _model([[5.0]], origin=origin)
    candidates = _candidates(model, [(0, 0)])
    config = CanopySegmentationConfig()
    result = segment_canopy_regions(model, candidates, config)
    assert result.source_grid_origin_xy == origin
    assert all(type(value) is float for value in result.source_grid_origin_xy)
    assert result.config == config
    assert result.config is not config


def test_bounded_propagation_completion_accepts_inactive_unsettled_cells() -> None:
    active = np.asarray([True, False, True], dtype=np.bool_)
    settled = np.asarray([True, False, True], dtype=np.bool_)
    owner = np.asarray([1, 0, 2], dtype=np.int32)
    snapshots = tuple(value.copy() for value in (active, settled, owner))
    segmentation._validate_propagation_completion(
        active,
        settled,
        owner,
        chunk_size=2,
    )
    for value, snapshot in zip((active, settled, owner), snapshots, strict=True):
        assert np.array_equal(value, snapshot)


@pytest.mark.parametrize(
    ("settled", "owner"),
    [
        ([True, False], [1, 2]),
        ([True, True], [1, 0]),
    ],
)
def test_bounded_propagation_completion_rejects_incomplete_active_cells(
    settled: list[bool],
    owner: list[int],
) -> None:
    active = np.asarray([True, True], dtype=np.bool_)
    with pytest.raises(CanopySegmentationError) as raised:
        segmentation._validate_propagation_completion(
            active,
            np.asarray(settled, dtype=np.bool_),
            np.asarray(owner, dtype=np.int32),
            chunk_size=1,
        )
    assert raised.value.code is CanopySegmentationErrorCode.INTERNAL_INVARIANT
    assert "left an active cell unassigned" in str(raised.value)


def test_propagation_completion_workspace_is_bounded_and_old_expression_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacities: list[int] = []

    def allocate(capacity: int) -> np.ndarray:
        capacities.append(capacity)
        return np.empty(capacity, dtype=np.bool_)

    monkeypatch.setattr(
        segmentation,
        "_allocate_propagation_completion_workspace",
        allocate,
    )
    active = np.ones(5, dtype=np.bool_)
    settled = np.ones(5, dtype=np.bool_)
    owner = np.ones(5, dtype=np.int32)
    segmentation._validate_propagation_completion(
        active,
        settled,
        owner,
        chunk_size=2,
    )
    assert capacities == [2]
    production_source = inspect.getsource(segmentation._propagate)
    assert "active &" not in production_source
    assert "~settled" not in production_source
    assert "owner == 0" not in production_source


def test_propagation_estimate_includes_bounded_completion_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_estimate = 10 * 24 + 5 * 4 + 2 * 8
    assert segmentation._propagation_temporary_estimate(10, 5, 2) == old_estimate + 10
    assert (
        segmentation._propagation_temporary_estimate(
            segmentation._VALIDATION_CHUNK_SIZE + 1,
            1,
            1,
        )
        == (segmentation._VALIDATION_CHUNK_SIZE + 1) * 24
        + 4
        + 8
        + segmentation._VALIDATION_CHUNK_SIZE
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("propagation allocation occurred after failed preflight")

    monkeypatch.setattr(segmentation, "_allocate_score_arrays", forbidden)
    monkeypatch.setattr(segmentation, "_allocate_heap_cells", forbidden)
    model = _model([[5.0, 5.0, 5.0]])
    with pytest.raises(CanopySegmentationError) as raised:
        segment_canopy_regions(
            model,
            _candidates(model, [(0, 0), (0, 2)]),
            CanopySegmentationConfig(maximum_temporary_bytes=100),
        )
    assert raised.value.code is CanopySegmentationErrorCode.RESOURCE_LIMIT
    assert raised.value.estimated_temporary_bytes == 103
