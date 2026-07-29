from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import algorithms.vegetation as vegetation
import algorithms.vegetation.canopy_height_model as chm
import numpy as np
import pytest
from algorithms.vegetation.canopy_height_model import (
    CanopyHeightModelConfig,
    CanopyHeightModelError,
    CanopyHeightModelErrorCode,
    CanopyHeightModelResult,
    CanopySupportFlag,
    build_canopy_height_model,
)
from core.contracts import (
    GroundQualityFlag,
    HeightAboveGroundSet,
    PointFeatureSet,
    TreeSegmentationInputClusterDisposition,
    TreeSegmentationInputDisposition,
    TreeSegmentationInputIdentity,
    TreeSegmentationInputSet,
    VegetationClusterStatus,
    VegetationInstancePointSet,
)


def _freeze(*arrays: np.ndarray) -> None:
    for array in arrays:
        array.setflags(write=False)


def _identity(point_count: int) -> TreeSegmentationInputIdentity:
    values: dict[str, Any] = {
        "source_id": "scene",
        "manifest_source_id": "manifest",
        "origin_xyz": (10.0, 20.0, 30.0),
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


def _tree_input(
    *,
    xy: list[tuple[float, float]],
    hag: list[float],
    statuses: list[VegetationClusterStatus] | None = None,
    cluster_ids: list[int] | None = None,
    wire_ids: list[int] | None = None,
    ground_usable: list[bool] | None = None,
    local_z: list[float] | None = None,
    original_index: list[int] | None = None,
) -> TreeSegmentationInputSet:
    count = len(xy)
    assert len(hag) == count
    statuses = statuses or [VegetationClusterStatus.CONFIRMED] * count
    cluster_ids = cluster_ids or list(range(count))
    wire_ids = wire_ids or [-1] * count
    ground_usable = ground_usable or [True] * count
    local_z = local_z or [100.0] * count
    original_index = original_index or list(range(100, 100 + count))

    xyz = np.asarray(
        [(x, y, z) for (x, y), z in zip(xy, local_z, strict=True)],
        dtype=np.float32,
    ).reshape((count, 3))
    original = np.asarray(original_index, dtype=np.int64)
    sample = np.arange(count, dtype=np.int64)
    status_array = np.asarray([int(value) for value in statuses], dtype=np.uint8)
    cluster_by_point = np.asarray(cluster_ids, dtype=np.int32)
    wire = np.asarray(wire_ids, dtype=np.int32)
    vegetation_ids = np.full(count, -1, dtype=np.int32)
    distance_ready = np.zeros(count, dtype=np.bool_)

    formal_clusters = sorted(set(value for value in cluster_ids if value >= 0))
    vegetation_id_by_cluster: dict[int, int] = {}
    for cluster in formal_clusters:
        rows = [index for index, value in enumerate(cluster_ids) if value == cluster]
        cluster_statuses = {statuses[index] for index in rows}
        assert len(cluster_statuses) == 1
        status = cluster_statuses.pop()
        assert status is not VegetationClusterStatus.UNASSIGNED
        if status is VegetationClusterStatus.CONFIRMED:
            vegetation_id_by_cluster[cluster] = len(vegetation_id_by_cluster)
            vegetation_ids[rows] = vegetation_id_by_cluster[cluster]
            distance_ready[rows] = True
    for index, cluster in enumerate(cluster_ids):
        if cluster < 0:
            assert statuses[index] is VegetationClusterStatus.UNASSIGNED

    zeros8 = np.zeros(count, dtype=np.uint8)
    zeros16 = np.zeros(count, dtype=np.uint16)
    zeros32 = np.zeros(count, dtype=np.float32)
    vegetation_points = VegetationInstancePointSet(
        original,
        sample,
        xyz,
        wire,
        np.full(count, 0.8, dtype=np.float32),
        np.full(count, 0.8, dtype=np.float32),
        zeros8.copy(),
        zeros8.copy(),
        zeros8.copy(),
        zeros32.copy(),
        zeros32.copy(),
        zeros8.copy(),
        cluster_by_point,
        vegetation_ids,
        status_array,
        distance_ready,
        zeros32.copy(),
        zeros16.copy(),
        zeros16.copy(),
        zeros16.copy(),
    )

    feature_arrays = (
        xyz.copy(),
        np.zeros(3, dtype=np.float64),
        original.copy(),
        sample.copy(),
        *(np.zeros(count, dtype=np.float32) for _ in range(6)),
        np.zeros(count, dtype=np.uint16),
        np.zeros((count, 3), dtype=np.float32),
        np.zeros((count, 3), dtype=np.float32),
        np.zeros((count, 3), dtype=np.float32),
        np.zeros(count, dtype=np.uint16),
    )
    geometry = PointFeatureSet("scene", "manifest", *feature_arrays)

    usable = np.asarray(ground_usable, dtype=np.bool_)
    hag_array = np.asarray(hag, dtype=np.float32)
    quality = np.zeros(count, dtype=np.uint16)
    quality[~usable] |= int(GroundQualityFlag.INVALID_GROUND_ESTIMATE)
    ground_z = np.zeros(count, dtype=np.float32)
    raw_hag = hag_array.copy()
    ground_distance = np.zeros(count, dtype=np.float32)
    ground_z[~usable] = np.nan
    raw_hag[~usable] = hag_array[~usable]
    ground_distance[~usable] = np.nan
    heights = HeightAboveGroundSet(
        "scene",
        "manifest",
        original.copy(),
        sample.copy(),
        ground_z,
        raw_hag,
        hag_array,
        np.zeros(count, dtype=np.int32),
        ground_distance,
        quality,
    )

    cluster_status_values: list[int] = []
    cluster_vegetation_values: list[int] = []
    offsets = [0]
    member_rows: list[int] = []
    eligible: list[int] = []
    review: list[int] = []
    excluded: list[int] = []
    dispositions: list[TreeSegmentationInputClusterDisposition] = []
    usable_counts: list[int] = []
    for cluster in formal_clusters:
        rows = [index for index, value in enumerate(cluster_ids) if value == cluster]
        status = statuses[rows[0]]
        cluster_status_values.append(int(status))
        cluster_vegetation_values.append(vegetation_id_by_cluster.get(cluster, -1))
        member_rows.extend(rows)
        offsets.append(len(member_rows))
        usable_counts.append(sum(ground_usable[index] for index in rows))
        if status is VegetationClusterStatus.CONFIRMED:
            disposition = TreeSegmentationInputDisposition.ELIGIBLE
            eligible.append(cluster)
        elif status in {
            VegetationClusterStatus.REVIEW_REQUIRED,
            VegetationClusterStatus.PROVISIONAL,
        }:
            disposition = TreeSegmentationInputDisposition.REVIEW_ONLY
            review.append(cluster)
        else:
            assert status is VegetationClusterStatus.REJECTED_AS_INSTANCE
            disposition = TreeSegmentationInputDisposition.EXCLUDED
            excluded.append(cluster)
        dispositions.append(
            TreeSegmentationInputClusterDisposition(cluster, disposition, status.name.lower())
        )

    top_arrays = (
        np.asarray(formal_clusters, dtype=np.int32),
        np.asarray(cluster_status_values, dtype=np.uint8),
        np.asarray(cluster_vegetation_values, dtype=np.int32),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(member_rows, dtype=np.int64),
        usable,
        np.asarray(usable_counts, dtype=np.int64),
    )
    nested_arrays = [
        *(
            getattr(vegetation_points, name)
            for name in vegetation_points.__dataclass_fields__
            if isinstance(getattr(vegetation_points, name), np.ndarray)
        ),
        *(
            getattr(geometry, name)
            for name in geometry.__dataclass_fields__
            if isinstance(getattr(geometry, name), np.ndarray)
        ),
        *(
            getattr(heights, name)
            for name in heights.__dataclass_fields__
            if isinstance(getattr(heights, name), np.ndarray)
        ),
        *top_arrays,
    ]
    _freeze(*nested_arrays)
    return TreeSegmentationInputSet(
        vegetation_points=vegetation_points,
        geometry_features=geometry,
        heights_above_ground=heights,
        cluster_id=top_arrays[0],
        cluster_status=top_arrays[1],
        cluster_vegetation_id=top_arrays[2],
        cluster_member_offsets=top_arrays[3],
        ordered_cluster_member_row_index=top_arrays[4],
        identity=_identity(count),
        eligible_cluster_ids=tuple(eligible),
        review_only_cluster_ids=tuple(review),
        excluded_cluster_ids=tuple(excluded),
        cluster_dispositions=tuple(dispositions),
        ground_usable_by_point=top_arrays[5],
        usable_ground_member_count_by_cluster=top_arrays[6],
    )


def _readonly(value: np.ndarray, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _valid_direct_result() -> CanopyHeightModelResult:
    return CanopyHeightModelResult(
        source_identity=_identity(4),
        config=CanopyHeightModelConfig(),
        grid_origin_xy=(0.0, 0.0),
        canopy_height_m=_readonly(np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)),
        occupied_mask=_readonly(np.ones((2, 2), np.bool_)),
        point_count_by_cell=_readonly(np.ones((2, 2), np.int32)),
        support_flags_by_cell=_readonly(np.ones((2, 2), np.uint8)),
        peak_support_flags_by_cell=_readonly(np.ones((2, 2), np.uint8)),
        included_point_count=4,
        excluded_point_count=0,
    )


def _valid_partial_result() -> CanopyHeightModelResult:
    return CanopyHeightModelResult(
        source_identity=_identity(4),
        config=CanopyHeightModelConfig(),
        grid_origin_xy=(0.0, 0.0),
        canopy_height_m=_readonly(np.array([[1.0, np.nan], [3.0, 4.0]], np.float32)),
        occupied_mask=_readonly(np.array([[True, False], [True, True]], np.bool_)),
        point_count_by_cell=_readonly(np.array([[1, 0], [1, 1]], np.int32)),
        support_flags_by_cell=_readonly(np.array([[1, 0], [1, 1]], np.uint8)),
        peak_support_flags_by_cell=_readonly(np.array([[1, 0], [1, 1]], np.uint8)),
        included_point_count=3,
        excluded_point_count=1,
    )


def test_public_api_exports_values_signature_and_frozen_contracts() -> None:
    expected = {
        "CanopySupportFlag": CanopySupportFlag,
        "CanopyHeightModelConfig": CanopyHeightModelConfig,
        "CanopyHeightModelErrorCode": CanopyHeightModelErrorCode,
        "CanopyHeightModelError": CanopyHeightModelError,
        "CanopyHeightModelResult": CanopyHeightModelResult,
        "build_canopy_height_model": build_canopy_height_model,
    }
    for name, value in expected.items():
        assert getattr(vegetation, name) is value
    assert [int(value) for value in CanopySupportFlag] == [1, 2, 4]
    assert CanopySupportFlag.NONE == 0
    assert [value.value for value in CanopyHeightModelErrorCode] == [
        "resource_limit",
        "allocation_failed",
        "numerical_index_invalid",
    ]
    assert tuple(inspect.signature(build_canopy_height_model).parameters) == (
        "tree_input",
        "config",
    )
    assert not hasattr(vegetation, "write_canopy_height_model")
    assert not hasattr(vegetation, "publish_canopy_height_model")
    with pytest.raises(FrozenInstanceError):
        CanopyHeightModelConfig().cell_size_m = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        _valid_direct_result().grid_origin_xy = None  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"cell_size_m": True},
        {"cell_size_m": 0.0},
        {"cell_size_m": np.inf},
        {"cell_size_m": "bad"},
        {"minimum_hag_m": False},
        {"minimum_hag_m": -0.1},
        {"minimum_hag_m": np.nan},
        {"maximum_grid_cells": True},
        {"maximum_grid_cells": 0},
        {"maximum_grid_cells": 1.5},
    ],
)
def test_config_rejects_invalid_values(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CanopyHeightModelConfig(**changes)
    assert isinstance(CanopyHeightModelConfig(cell_size_m=np.float32(0.2)).cell_size_m, float)
    assert isinstance(
        CanopyHeightModelConfig(maximum_grid_cells=np.int64(10)).maximum_grid_cells,
        int,
    )


def test_wrong_input_and_config_types_and_error_contract() -> None:
    with pytest.raises(TypeError):
        build_canopy_height_model(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_canopy_height_model(
            _tree_input(xy=[(0.0, 0.0)], hag=[1.0]),
            object(),  # type: ignore[arg-type]
        )
    error = CanopyHeightModelError(
        CanopyHeightModelErrorCode.RESOURCE_LIMIT,
        "limit",
        row_count=1,
        column_count=2,
        total_cell_count=2,
        estimated_output_bytes=22,
        estimated_peak_dense_bytes=38,
    )
    assert error.code is CanopyHeightModelErrorCode.RESOURCE_LIMIT
    assert error.total_cell_count == 2
    with pytest.raises(ValueError):
        CanopyHeightModelError("bad", "message")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CanopyHeightModelError(CanopyHeightModelErrorCode.RESOURCE_LIMIT, "")
    with pytest.raises(ValueError):
        CanopyHeightModelError(
            CanopyHeightModelErrorCode.RESOURCE_LIMIT,
            "bad scalar",
            row_count=np.int64(1),  # type: ignore[arg-type]
        )


def test_confirmed_review_and_provisional_are_included_without_distance_gate() -> None:
    source = _tree_input(
        xy=[(0.01, 0.01), (0.02, 0.02), (0.03, 0.03)],
        hag=[1.0, 2.0, 3.0],
        statuses=[
            VegetationClusterStatus.CONFIRMED,
            VegetationClusterStatus.REVIEW_REQUIRED,
            VegetationClusterStatus.PROVISIONAL,
        ],
        cluster_ids=[10, 20, 30],
    )
    assert source.vegetation_points.vegetation_id_by_point.tolist() == [0, -1, -1]
    assert source.vegetation_points.distance_ready_by_point.tolist() == [True, False, False]
    result = build_canopy_height_model(source)
    assert result.included_point_count == 3
    assert result.canopy_height_m[0, 0] == np.float32(3.0)
    assert result.support_flags_by_cell[0, 0] == 7
    assert result.peak_support_flags_by_cell[0, 0] == 4


def test_rejected_unassigned_wire_unusable_and_low_hag_are_excluded() -> None:
    source = _tree_input(
        xy=[(0.0, 0.0)] * 6,
        hag=[2.0, 2.0, 9.0, np.nan, 0.29, 1.5],
        statuses=[
            VegetationClusterStatus.REJECTED_AS_INSTANCE,
            VegetationClusterStatus.UNASSIGNED,
            VegetationClusterStatus.CONFIRMED,
            VegetationClusterStatus.REVIEW_REQUIRED,
            VegetationClusterStatus.PROVISIONAL,
            VegetationClusterStatus.CONFIRMED,
        ],
        cluster_ids=[10, -1, 30, 40, 50, 60],
        wire_ids=[-1, -1, 7, -1, -1, -1],
        ground_usable=[True, True, True, False, True, True],
    )
    result = build_canopy_height_model(source)
    assert result.included_point_count == 1
    assert result.excluded_point_count == 5
    assert result.canopy_height_m.tolist() == [[1.5]]
    assert result.support_flags_by_cell.tolist() == [[1]]


def test_maximum_counts_empty_cells_zero_height_duplicates_and_ties() -> None:
    source = _tree_input(
        xy=[(0.01, 0.01), (0.01, 0.01), (0.41, 0.01), (0.41, 0.01)],
        hag=[0.0, 2.0, 3.0, 3.0],
        statuses=[
            VegetationClusterStatus.CONFIRMED,
            VegetationClusterStatus.REVIEW_REQUIRED,
            VegetationClusterStatus.PROVISIONAL,
            VegetationClusterStatus.CONFIRMED,
        ],
        cluster_ids=[10, 20, 30, 40],
    )
    result = build_canopy_height_model(
        source,
        CanopyHeightModelConfig(minimum_hag_m=0.0),
    )
    assert result.canopy_height_m.shape == (1, 3)
    assert result.canopy_height_m[0, 0] == 2.0
    assert np.isnan(result.canopy_height_m[0, 1])
    assert result.canopy_height_m[0, 2] == 3.0
    assert result.point_count_by_cell.tolist() == [[2, 0, 2]]
    assert result.occupied_mask.tolist() == [[True, False, True]]
    assert result.support_flags_by_cell.tolist() == [[3, 0, 5]]
    assert result.peak_support_flags_by_cell.tolist() == [[2, 0, 5]]

    zero = build_canopy_height_model(
        _tree_input(xy=[(0.0, 0.0), (0.4, 0.0)], hag=[0.0, 1.0]),
        CanopyHeightModelConfig(minimum_hag_m=0.0),
    )
    assert zero.canopy_height_m[0, 0] == 0.0
    assert zero.occupied_mask[0, 0]
    assert np.isnan(zero.canopy_height_m[0, 1])


@pytest.mark.parametrize(
    ("statuses", "heights", "support", "peak"),
    [
        ([VegetationClusterStatus.CONFIRMED], [2.0], 1, 1),
        ([VegetationClusterStatus.REVIEW_REQUIRED], [2.0], 2, 2),
        ([VegetationClusterStatus.PROVISIONAL], [2.0], 4, 4),
        (
            [VegetationClusterStatus.CONFIRMED, VegetationClusterStatus.REVIEW_REQUIRED],
            [2.0, 1.0],
            3,
            1,
        ),
        (
            [VegetationClusterStatus.CONFIRMED, VegetationClusterStatus.PROVISIONAL],
            [1.0, 2.0],
            5,
            4,
        ),
        (
            [VegetationClusterStatus.CONFIRMED, VegetationClusterStatus.PROVISIONAL],
            [2.0, 1.0],
            5,
            1,
        ),
        (
            [VegetationClusterStatus.CONFIRMED, VegetationClusterStatus.REVIEW_REQUIRED],
            [2.0, 2.0],
            3,
            3,
        ),
        (
            [
                VegetationClusterStatus.CONFIRMED,
                VegetationClusterStatus.REVIEW_REQUIRED,
                VegetationClusterStatus.PROVISIONAL,
            ],
            [2.0, 1.0, 0.5],
            7,
            1,
        ),
    ],
)
def test_support_and_peak_support_combinations(
    statuses: list[VegetationClusterStatus],
    heights: list[float],
    support: int,
    peak: int,
) -> None:
    result = build_canopy_height_model(
        _tree_input(
            xy=[(0.0, 0.0)] * len(statuses),
            hag=heights,
            statuses=statuses,
            cluster_ids=list(range(10, 10 + len(statuses))),
        )
    )
    assert result.support_flags_by_cell[0, 0] == support
    assert result.peak_support_flags_by_cell[0, 0] == peak
    assert not (int(result.peak_support_flags_by_cell[0, 0]) & ~support)


def test_grid_geometry_negative_boundaries_orientation_and_centers() -> None:
    source = _tree_input(
        xy=[(-0.4, -0.2), (-0.2, -0.2), (0.0, 0.2), (0.4, 0.6)],
        hag=[1.0, 2.0, 3.0, 4.0],
    )
    result = build_canopy_height_model(source)
    source_xy = source.vegetation_points.xyz_local[:, :2].astype(np.float64)
    expected_origin = np.floor(np.min(source_xy, axis=0) / 0.2) * 0.2
    assert result.grid_origin_xy == pytest.approx(tuple(expected_origin))
    assert result.canopy_height_m.shape == (6, 6)
    assert result.canopy_height_m[0, 0] == 1.0
    assert result.canopy_height_m[0, 1] == 2.0
    assert result.canopy_height_m[3, 3] == 3.0
    assert result.canopy_height_m[5, 5] == 4.0
    center_x = result.grid_origin_xy[0] + (5 + 0.5) * result.config.cell_size_m
    center_y = result.grid_origin_xy[1] + (5 + 0.5) * result.config.cell_size_m
    assert (center_x, center_y) == pytest.approx((0.5, 0.7))

    one_column = build_canopy_height_model(
        _tree_input(xy=[(-0.1, 0.0), (-0.1, 0.4)], hag=[1.0, 2.0])
    )
    assert one_column.canopy_height_m.shape == (3, 1)
    one_row = build_canopy_height_model(_tree_input(xy=[(0.0, -0.1), (0.4, -0.1)], hag=[1.0, 2.0]))
    assert one_row.canopy_height_m.shape == (1, 3)


def test_nextafter_boundary_and_configurable_cell_size() -> None:
    boundary = np.float32(0.2)
    below = np.nextafter(boundary, np.float32(-np.inf), dtype=np.float32)
    above = np.nextafter(boundary, np.float32(np.inf), dtype=np.float32)
    result = build_canopy_height_model(
        _tree_input(
            xy=[(0.0, 0.0), (float(below), 0.0), (float(boundary), 0.0), (float(above), 0.0)],
            hag=[1.0, 2.0, 3.0, 4.0],
        )
    )
    assert result.point_count_by_cell.tolist() == [[2, 2]]
    resized = build_canopy_height_model(
        _tree_input(xy=[(0.0, 0.0), (0.4, 0.0)], hag=[1.0, 2.0]),
        CanopyHeightModelConfig(cell_size_m=0.5),
    )
    assert resized.canopy_height_m.shape == (1, 1)


def test_only_included_points_determine_extent_and_resource_limit() -> None:
    source = _tree_input(
        xy=[(0.0, 0.0), (1_000_000.0, 1_000_000.0), (-1_000_000.0, -1_000_000.0)],
        hag=[1.0, 2.0, 3.0],
        statuses=[
            VegetationClusterStatus.CONFIRMED,
            VegetationClusterStatus.REJECTED_AS_INSTANCE,
            VegetationClusterStatus.CONFIRMED,
        ],
        cluster_ids=[10, 20, 30],
        wire_ids=[-1, -1, 9],
    )
    result = build_canopy_height_model(source, CanopyHeightModelConfig(maximum_grid_cells=1))
    assert result.canopy_height_m.shape == (1, 1)
    assert result.included_point_count == 1

    huge = _tree_input(xy=[(0.0, 0.0), (1_000_000.0, 1_000_000.0)], hag=[1.0, 2.0])
    with pytest.raises(CanopyHeightModelError) as raised:
        build_canopy_height_model(huge)
    assert raised.value.code is CanopyHeightModelErrorCode.RESOURCE_LIMIT


def test_canonical_empty_results_and_all_below_threshold() -> None:
    for source in (
        _tree_input(xy=[], hag=[]),
        _tree_input(
            xy=[(0.0, 0.0)],
            hag=[2.0],
            statuses=[VegetationClusterStatus.REJECTED_AS_INSTANCE],
            cluster_ids=[10],
        ),
        _tree_input(xy=[(0.0, 0.0), (1.0, 1.0)], hag=[0.1, 0.2]),
    ):
        result = build_canopy_height_model(source)
        assert result.grid_origin_xy is None
        assert result.included_point_count == 0
        assert result.excluded_point_count == source.identity.point_count
        for name, dtype in (
            ("canopy_height_m", np.float32),
            ("occupied_mask", np.bool_),
            ("point_count_by_cell", np.int32),
            ("support_flags_by_cell", np.uint8),
            ("peak_support_flags_by_cell", np.uint8),
        ):
            value = getattr(result, name)
            assert value.shape == (0, 0)
            assert value.dtype == np.dtype(dtype)
            assert value.flags.owndata and value.flags.c_contiguous
            assert not value.flags.writeable


def test_determinism_shuffle_repetition_cwd_and_input_immutability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        ((0.0, 0.0), 2.0, VegetationClusterStatus.CONFIRMED, 10, 100),
        ((0.0, 0.0), 2.0, VegetationClusterStatus.REVIEW_REQUIRED, 20, 101),
        ((0.4, 0.2), 3.0, VegetationClusterStatus.PROVISIONAL, 30, 102),
    ]
    first_input = _tree_input(
        xy=[row[0] for row in rows],
        hag=[row[1] for row in rows],
        statuses=[row[2] for row in rows],
        cluster_ids=[row[3] for row in rows],
        original_index=[row[4] for row in rows],
    )
    shuffled_rows = [rows[index] for index in (2, 0, 1)]
    shuffled_input = _tree_input(
        xy=[row[0] for row in shuffled_rows],
        hag=[row[1] for row in shuffled_rows],
        statuses=[row[2] for row in shuffled_rows],
        cluster_ids=[row[3] for row in shuffled_rows],
        original_index=[row[4] for row in shuffled_rows],
    )
    arrays = [
        value
        for contract in (
            first_input.vegetation_points,
            first_input.geometry_features,
            first_input.heights_above_ground,
        )
        for value in contract.__dict__.values()
        if isinstance(value, np.ndarray)
    ]
    snapshots = [(value.copy(), value.flags.writeable) for value in arrays]
    first = build_canopy_height_model(first_input)
    monkeypatch.chdir(tmp_path)
    repeated = build_canopy_height_model(first_input)
    shuffled = build_canopy_height_model(shuffled_input)
    for name in (
        "canopy_height_m",
        "occupied_mask",
        "point_count_by_cell",
        "support_flags_by_cell",
        "peak_support_flags_by_cell",
    ):
        assert np.array_equal(
            getattr(first, name),
            getattr(repeated, name),
            equal_nan=True,
        )
        assert np.array_equal(
            getattr(first, name),
            getattr(shuffled, name),
            equal_nan=True,
        )
        assert getattr(first, name) is not getattr(repeated, name)
        assert not np.shares_memory(getattr(first, name), first_input.vegetation_points.xyz_local)
    assert first.grid_origin_xy == repeated.grid_origin_xy == shuffled.grid_origin_xy
    for value, (snapshot, writeable) in zip(arrays, snapshots, strict=True):
        assert np.array_equal(value, snapshot, equal_nan=True)
        assert value.flags.writeable is writeable


def test_hag_not_local_z_threshold_and_usable_corruption() -> None:
    low_z = _tree_input(xy=[(0.0, 0.0)], hag=[1.5], local_z=[-100.0])
    high_z = _tree_input(xy=[(0.0, 0.0)], hag=[1.5], local_z=[10_000.0])
    changed_hag = _tree_input(xy=[(0.0, 0.0)], hag=[2.5], local_z=[-100.0])
    assert np.array_equal(
        build_canopy_height_model(low_z).canopy_height_m,
        build_canopy_height_model(high_z).canopy_height_m,
    )
    assert build_canopy_height_model(changed_hag).canopy_height_m[0, 0] == 2.5
    threshold = CanopyHeightModelConfig(minimum_hag_m=0.3)
    assert (
        build_canopy_height_model(
            _tree_input(xy=[(0.0, 0.0)], hag=[0.3]),
            threshold,
        ).included_point_count
        == 1
    )
    assert (
        build_canopy_height_model(
            _tree_input(xy=[(0.0, 0.0)], hag=[float(np.nextafter(np.float32(0.3), -np.inf))]),
            threshold,
        ).included_point_count
        == 0
    )

    for replacement in (np.nan, -1.0):
        corrupted = _tree_input(xy=[(0.0, 0.0)], hag=[1.0])
        values = corrupted.heights_above_ground.height_agl_m
        values.setflags(write=True)
        values[0] = replacement
        values.setflags(write=False)
        with pytest.raises(ValueError, match="ground-usable"):
            build_canopy_height_model(corrupted)


def test_nonfinite_xy_corruption_is_rejected() -> None:
    corrupted = _tree_input(xy=[(0.0, 0.0)], hag=[1.0])
    xyz = corrupted.vegetation_points.xyz_local
    xyz.setflags(write=True)
    xyz[0, 0] = np.nan
    xyz.setflags(write=False)
    with pytest.raises(ValueError, match="nonfinite XY"):
        build_canopy_height_model(corrupted)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("canopy_height_m", _readonly(np.ones((2, 2), np.float64))),
        ("occupied_mask", _readonly(np.ones((2, 2), np.uint8))),
        ("point_count_by_cell", _readonly(np.ones((2, 2), np.int64))),
        ("support_flags_by_cell", _readonly(np.ones((2, 2), np.int16))),
        ("peak_support_flags_by_cell", _readonly(np.ones((2, 2), np.int16))),
    ],
)
def test_result_rejects_wrong_array_dtype(field: str, replacement: np.ndarray) -> None:
    with pytest.raises(ValueError, match="dtype"):
        replace(_valid_direct_result(), **{field: replacement})


def test_result_rejects_array_storage_and_shape_failures() -> None:
    valid = _valid_direct_result()
    writeable = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="read-only"):
        replace(valid, canopy_height_m=writeable)
    base = np.ones((2, 2), dtype=np.float32)
    view = base[:, :]
    view.setflags(write=False)
    with pytest.raises(ValueError, match="own"):
        replace(valid, canopy_height_m=view)
    fortran = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order="F")
    fortran.setflags(write=False)
    with pytest.raises(ValueError, match="C-contiguous"):
        replace(valid, canopy_height_m=fortran)
    with pytest.raises(ValueError, match="identical"):
        replace(valid, occupied_mask=_readonly(np.ones((1, 1), np.bool_)))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "canopy_height_m",
            _readonly(np.array([[1.0, 0.0], [3.0, 4.0]], np.float32)),
        ),
        (
            "support_flags_by_cell",
            _readonly(np.array([[1, 1], [1, 1]], np.uint8)),
        ),
        (
            "peak_support_flags_by_cell",
            _readonly(np.array([[1, 1], [1, 1]], np.uint8)),
        ),
    ],
)
def test_result_rejects_nonempty_values_in_empty_cells(
    field: str,
    replacement: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        replace(_valid_partial_result(), **{field: replacement})


@pytest.mark.parametrize(
    "changes",
    [
        {"occupied_mask": _readonly(np.zeros((2, 2), np.bool_))},
        {"canopy_height_m": _readonly(np.array([[1.0, np.nan], [3.0, 4.0]], np.float32))},
        {"canopy_height_m": _readonly(np.array([[1.0, -2.0], [3.0, 4.0]], np.float32))},
        {"support_flags_by_cell": _readonly(np.zeros((2, 2), np.uint8))},
        {"peak_support_flags_by_cell": _readonly(np.zeros((2, 2), np.uint8))},
        {"support_flags_by_cell": _readonly(np.full((2, 2), 8, np.uint8))},
        {
            "support_flags_by_cell": _readonly(np.ones((2, 2), np.uint8)),
            "peak_support_flags_by_cell": _readonly(np.full((2, 2), 2, np.uint8)),
        },
        {"included_point_count": 3, "excluded_point_count": 1},
        {"included_point_count": 4, "excluded_point_count": 1},
        {"grid_origin_xy": None},
    ],
)
def test_result_rejects_cross_field_inconsistency(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(_valid_direct_result(), **changes)


def test_result_rejects_identity_config_and_empty_origin_inconsistency() -> None:
    valid = _valid_direct_result()
    with pytest.raises(ValueError, match="source_identity"):
        replace(valid, source_identity=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="config"):
        replace(valid, config=object())  # type: ignore[arg-type]
    empty = build_canopy_height_model(_tree_input(xy=[], hag=[]))
    with pytest.raises(ValueError, match="grid_origin"):
        replace(empty, grid_origin_xy=(0.0, 0.0))


def test_valid_direct_empty_and_nonempty_results() -> None:
    assert _valid_direct_result().canopy_height_m.shape == (2, 2)
    empty = build_canopy_height_model(_tree_input(xy=[], hag=[]))
    recreated = CanopyHeightModelResult(**empty.__dict__)
    assert recreated.canopy_height_m.shape == (0, 0)


def test_resource_helpers_limits_overflow_numerical_and_memory_chaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted = build_canopy_height_model(
        _tree_input(xy=[(0.0, 0.0), (0.4, 0.0)], hag=[1.0, 2.0]),
        CanopyHeightModelConfig(maximum_grid_cells=3),
    )
    assert accepted.canopy_height_m.size == 3
    with pytest.raises(CanopyHeightModelError) as over:
        build_canopy_height_model(
            _tree_input(xy=[(0.0, 0.0), (0.4, 0.0)], hag=[1.0, 2.0]),
            CanopyHeightModelConfig(maximum_grid_cells=2),
        )
    assert over.value.code is CanopyHeightModelErrorCode.RESOURCE_LIMIT
    assert over.value.estimated_output_bytes == 33
    assert over.value.estimated_peak_dense_bytes == 57

    with pytest.raises(CanopyHeightModelError) as numerical:
        chm._indices_from_relative(
            np.array([np.inf], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        )
    assert numerical.value.code is CanopyHeightModelErrorCode.NUMERICAL_INDEX_INVALID
    with pytest.raises(CanopyHeightModelError) as overflow:
        chm._validate_point_counts(
            np.array([np.iinfo(np.int32).max + 1], dtype=np.int64),
            np.iinfo(np.int32).max + 1,
        )
    assert overflow.value.code is CanopyHeightModelErrorCode.NUMERICAL_INDEX_INVALID

    def fail_allocation(
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise MemoryError(f"simulated {shape}")

    monkeypatch.setattr(chm, "_allocate_dense_arrays", fail_allocation)
    with pytest.raises(CanopyHeightModelError) as allocation:
        build_canopy_height_model(_tree_input(xy=[(0.0, 0.0)], hag=[1.0]))
    assert allocation.value.code is CanopyHeightModelErrorCode.ALLOCATION_FAILED
    assert isinstance(allocation.value.__cause__, MemoryError)


def test_all_output_arrays_are_detached_immutable_and_independent() -> None:
    source = _tree_input(xy=[(0.0, 0.0), (0.4, 0.2)], hag=[1.0, 2.0])
    first = build_canopy_height_model(source)
    second = build_canopy_height_model(source)
    for name, dtype in (
        ("canopy_height_m", np.float32),
        ("occupied_mask", np.bool_),
        ("point_count_by_cell", np.int32),
        ("support_flags_by_cell", np.uint8),
        ("peak_support_flags_by_cell", np.uint8),
    ):
        value = getattr(first, name)
        assert value.dtype == np.dtype(dtype)
        assert value.shape == first.canopy_height_m.shape
        assert value.flags.owndata and value.flags.c_contiguous
        assert not value.flags.writeable
        assert value is not getattr(second, name)
        assert not np.shares_memory(value, getattr(second, name))
        with pytest.raises(ValueError):
            value.flat[0] = value.flat[0]
