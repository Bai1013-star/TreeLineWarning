"""Deterministic CPU treetop-candidate detection from an in-memory V3 canopy model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from core.contracts.tree_inputs import TreeSegmentationInputIdentity

from algorithms.vegetation.canopy_height_model import (
    CanopyHeightModelConfig,
    CanopyHeightModelResult,
    CanopySupportFlag,
)

_ALLOWED_SUPPORT_BITS = np.uint8(
    int(
        CanopySupportFlag.CONFIRMED
        | CanopySupportFlag.REVIEW_REQUIRED
        | CanopySupportFlag.PROVISIONAL
    )
)
_UNKNOWN_SUPPORT_BITS = np.uint8(0xFF ^ int(_ALLOWED_SUPPORT_BITS))


def _finite_non_negative_float(value: Any, name: str) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise ValueError(f"{name} must be a real numeric value")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _integer(value: Any, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        if maximum is None:
            raise ValueError(f"{name} must be at least {minimum}")
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _non_negative_python_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative Python integer")
    return value


@dataclass(frozen=True)
class TreetopDetectionConfig:
    """Physical and resource configuration for deterministic treetop detection."""

    minimum_peak_hag_m: float = 0.30
    local_max_radius_m: float = 0.50
    prominence_radius_m: float = 1.50
    minimum_prominence_m: float = 0.0
    minimum_separation_m: float = 0.0
    minimum_neighbour_occupied_cells: int = 0
    minimum_isolated_peak_point_count: int = 1
    maximum_neighbourhood_cells: int = 4096
    maximum_temporary_bytes: int = 536_870_912
    maximum_candidate_count: int = 1_000_000
    maximum_candidate_neighbour_evaluations: int = 100_000_000

    def __post_init__(self) -> None:
        for name in (
            "minimum_peak_hag_m",
            "local_max_radius_m",
            "prominence_radius_m",
            "minimum_prominence_m",
            "minimum_separation_m",
        ):
            object.__setattr__(
                self,
                name,
                _finite_non_negative_float(getattr(self, name), name),
            )
        if self.prominence_radius_m <= self.local_max_radius_m:
            raise ValueError("prominence_radius_m must be greater than local_max_radius_m")
        object.__setattr__(
            self,
            "minimum_neighbour_occupied_cells",
            _integer(
                self.minimum_neighbour_occupied_cells,
                "minimum_neighbour_occupied_cells",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "minimum_isolated_peak_point_count",
            _integer(
                self.minimum_isolated_peak_point_count,
                "minimum_isolated_peak_point_count",
                minimum=1,
            ),
        )
        for name in (
            "maximum_neighbourhood_cells",
            "maximum_temporary_bytes",
            "maximum_candidate_neighbour_evaluations",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, minimum=1),
            )
        object.__setattr__(
            self,
            "maximum_candidate_count",
            _integer(
                self.maximum_candidate_count,
                "maximum_candidate_count",
                minimum=1,
                maximum=np.iinfo(np.int32).max,
            ),
        )


class TreetopDetectionErrorCode(str, Enum):
    """Stable runtime failure categories for treetop detection."""

    RESOURCE_LIMIT = "resource_limit"
    ALLOCATION_FAILED = "allocation_failed"
    INTERNAL_INVARIANT = "internal_invariant"


class TreetopDetectionError(RuntimeError):
    """A compact treetop-detection failure containing fixed resource scalars."""

    def __init__(
        self,
        code: TreetopDetectionErrorCode,
        message: str,
        *,
        row_count: int | None = None,
        column_count: int | None = None,
        total_cell_count: int | None = None,
        local_maximum_offset_count: int | None = None,
        prominence_offset_count: int | None = None,
        estimated_temporary_bytes: int | None = None,
        candidate_count: int | None = None,
        candidate_neighbour_evaluations: int | None = None,
    ) -> None:
        if type(code) is not TreetopDetectionErrorCode:
            raise ValueError("code must be a TreetopDetectionErrorCode")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        values = (
            ("row_count", row_count),
            ("column_count", column_count),
            ("total_cell_count", total_cell_count),
            ("local_maximum_offset_count", local_maximum_offset_count),
            ("prominence_offset_count", prominence_offset_count),
            ("estimated_temporary_bytes", estimated_temporary_bytes),
            ("candidate_count", candidate_count),
            ("candidate_neighbour_evaluations", candidate_neighbour_evaluations),
        )
        for name, value in values:
            if value is not None:
                _non_negative_python_int(value, name)
            setattr(self, name, value)
        self.code = code
        super().__init__(message)


def _result_array(
    value: Any,
    dtype: Any,
    shape: tuple[int | None, ...],
    name: str,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise ValueError(f"{name} must be a numpy ndarray")
    if value.dtype != np.dtype(dtype) or value.ndim != len(shape):
        raise ValueError(f"{name} has invalid dtype or rank")
    if any(
        expected is not None and expected != actual
        for expected, actual in zip(shape, value.shape, strict=True)
    ):
        raise ValueError(f"{name} has invalid shape")
    if not value.flags.owndata:
        raise ValueError(f"{name} must own its data")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if value.flags.writeable:
        raise ValueError(f"{name} must be read-only")
    return value


def _source_grid_shape(value: Any) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not int or item < 0 for item in value)
        or (value != (0, 0) and (value[0] == 0 or value[1] == 0))
    ):
        raise ValueError("source_grid_shape must be (0, 0) or contain two positive integers")
    return value


def _source_origin(
    value: Any,
    shape: tuple[int, int],
) -> tuple[float, float] | None:
    if shape == (0, 0):
        if value is not None:
            raise ValueError("an empty source grid requires source_grid_origin_xy=None")
        return None
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(
            isinstance(item, bool | np.bool_)
            or not isinstance(item, int | float | np.integer | np.floating)
            for item in value
        )
    ):
        raise ValueError("a non-empty source grid requires a numeric origin tuple")
    try:
        finite = all(math.isfinite(float(item)) for item in value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("source_grid_origin_xy must be finite") from exc
    if not finite:
        raise ValueError("source_grid_origin_xy must be finite")
    return value


@dataclass(frozen=True)
class TreetopCandidateResult:
    """Immutable sparse treetop markers and local evidence for future V3B3."""

    source_identity: TreeSegmentationInputIdentity
    source_chm_config: CanopyHeightModelConfig
    source_grid_origin_xy: tuple[float, float] | None
    source_grid_shape: tuple[int, int]
    config: TreetopDetectionConfig
    local_maximum_plateau_count: int
    evidence_rejected_count: int
    suppressed_candidate_count: int
    candidate_id: np.ndarray
    row_index: np.ndarray
    column_index: np.ndarray
    local_xy: np.ndarray
    peak_height_m: np.ndarray
    prominence_m: np.ndarray
    prominence_available: np.ndarray
    plateau_cell_count: np.ndarray
    source_cell_point_count: np.ndarray
    source_support_flags: np.ndarray
    source_peak_support_flags: np.ndarray
    neighbour_occupied_cell_count: np.ndarray
    neighbourhood_occupied_fraction: np.ndarray
    neighbourhood_support_flags: np.ndarray
    neighbourhood_support_fraction: np.ndarray
    edge_limited: np.ndarray

    def __post_init__(self) -> None:
        if type(self.source_identity) is not TreeSegmentationInputIdentity:
            raise ValueError("source_identity must be a TreeSegmentationInputIdentity")
        if type(self.source_chm_config) is not CanopyHeightModelConfig:
            raise ValueError("source_chm_config must be a CanopyHeightModelConfig")
        if type(self.config) is not TreetopDetectionConfig:
            raise ValueError("config must be a TreetopDetectionConfig")
        shape = _source_grid_shape(self.source_grid_shape)
        origin = _source_origin(self.source_grid_origin_xy, shape)
        plateau_total = _non_negative_python_int(
            self.local_maximum_plateau_count,
            "local_maximum_plateau_count",
        )
        evidence_rejected = _non_negative_python_int(
            self.evidence_rejected_count,
            "evidence_rejected_count",
        )
        suppressed = _non_negative_python_int(
            self.suppressed_candidate_count,
            "suppressed_candidate_count",
        )

        candidate_id = _result_array(self.candidate_id, np.int32, (None,), "candidate_id")
        count = len(candidate_id)
        row = _result_array(self.row_index, np.int64, (count,), "row_index")
        column = _result_array(self.column_index, np.int64, (count,), "column_index")
        local_xy = _result_array(self.local_xy, np.float64, (count, 2), "local_xy")
        height = _result_array(self.peak_height_m, np.float32, (count,), "peak_height_m")
        prominence = _result_array(self.prominence_m, np.float32, (count,), "prominence_m")
        prominence_available = _result_array(
            self.prominence_available,
            np.bool_,
            (count,),
            "prominence_available",
        )
        plateau_count = _result_array(
            self.plateau_cell_count,
            np.int32,
            (count,),
            "plateau_cell_count",
        )
        point_count = _result_array(
            self.source_cell_point_count,
            np.int32,
            (count,),
            "source_cell_point_count",
        )
        source_support = _result_array(
            self.source_support_flags,
            np.uint8,
            (count,),
            "source_support_flags",
        )
        peak_support = _result_array(
            self.source_peak_support_flags,
            np.uint8,
            (count,),
            "source_peak_support_flags",
        )
        neighbour_count = _result_array(
            self.neighbour_occupied_cell_count,
            np.int32,
            (count,),
            "neighbour_occupied_cell_count",
        )
        occupied_fraction = _result_array(
            self.neighbourhood_occupied_fraction,
            np.float32,
            (count,),
            "neighbourhood_occupied_fraction",
        )
        neighbourhood_support = _result_array(
            self.neighbourhood_support_flags,
            np.uint8,
            (count,),
            "neighbourhood_support_flags",
        )
        support_fraction = _result_array(
            self.neighbourhood_support_fraction,
            np.float32,
            (count, 3),
            "neighbourhood_support_fraction",
        )
        edge_limited = _result_array(
            self.edge_limited,
            np.bool_,
            (count,),
            "edge_limited",
        )

        if not np.array_equal(candidate_id, np.arange(count, dtype=np.int32)):
            raise ValueError("candidate_id must be contiguous and start at zero")
        if plateau_total != evidence_rejected + suppressed + count:
            raise ValueError("candidate audit counts are inconsistent")
        if shape == (0, 0) and count:
            raise ValueError("an empty source grid cannot contain candidates")
        if count:
            if (
                np.any(row < 0)
                or np.any(row >= shape[0])
                or np.any(column < 0)
                or np.any(column >= shape[1])
            ):
                raise ValueError("candidate row or column is outside source_grid_shape")
            pairs = np.column_stack((row, column))
            if len(np.unique(pairs, axis=0)) != count:
                raise ValueError("candidate row and column pairs must be unique")
            if not np.isfinite(local_xy).all():
                raise ValueError("local_xy must be finite")
            assert origin is not None
            expected_xy = np.empty((count, 2), dtype=np.float64)
            expected_xy[:, 0] = (
                float(origin[0])
                + (column.astype(np.float64) + 0.5) * self.source_chm_config.cell_size_m
            )
            expected_xy[:, 1] = (
                float(origin[1])
                + (row.astype(np.float64) + 0.5) * self.source_chm_config.cell_size_m
            )
            if not np.array_equal(local_xy, expected_xy):
                raise ValueError("local_xy is inconsistent with source grid geometry")
            axis_extent = _prominence_axis_extent_cells(
                self.config.prominence_radius_m,
                self.source_chm_config.cell_size_m,
            )
            expected_edge_limited = np.zeros(count, dtype=np.bool_)
            if axis_extent > 0:
                expected_edge_limited = (
                    (row < axis_extent)
                    | (row >= shape[0] - axis_extent)
                    | (column < axis_extent)
                    | (column >= shape[1] - axis_extent)
                )
            if not np.array_equal(edge_limited, expected_edge_limited):
                raise ValueError("edge_limited is inconsistent with source grid geometry")
        if (
            not np.isfinite(height).all()
            or np.any(height < 0.0)
            or np.any(height < np.float64(self.config.minimum_peak_hag_m))
        ):
            raise ValueError("peak_height_m is invalid")
        if not np.isfinite(prominence).all() or np.any(prominence < 0.0):
            raise ValueError("prominence_m must be finite and non-negative")
        if np.any(~prominence_available & (prominence != 0.0)):
            raise ValueError("unavailable prominence must be zero")
        if np.any(plateau_count <= 0):
            raise ValueError("plateau_cell_count must be positive")
        if np.any(point_count <= 0):
            raise ValueError("source_cell_point_count must be positive")
        if np.any(neighbour_count < 0):
            raise ValueError("neighbour_occupied_cell_count must be non-negative")
        for value, name, allow_zero in (
            (source_support, "source_support_flags", False),
            (peak_support, "source_peak_support_flags", False),
            (neighbourhood_support, "neighbourhood_support_flags", True),
        ):
            if np.any(value & _UNKNOWN_SUPPORT_BITS) or (not allow_zero and np.any(value == 0)):
                raise ValueError(f"{name} contains invalid support bits")
        if np.any(peak_support & np.bitwise_not(source_support)):
            raise ValueError("source_peak_support_flags must be a subset of source_support_flags")
        for value, name in (
            (occupied_fraction, "neighbourhood_occupied_fraction"),
            (support_fraction, "neighbourhood_support_fraction"),
        ):
            if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
                raise ValueError(f"{name} must contain finite fractions in [0, 1]")
        no_neighbours = neighbour_count == 0
        if np.any(occupied_fraction[no_neighbours] != 0.0):
            raise ValueError("zero neighbour count requires zero occupied fraction")
        if np.any(neighbourhood_support[no_neighbours] != 0):
            raise ValueError("zero neighbour count requires zero neighbourhood support")
        if np.any(support_fraction[no_neighbours] != 0.0):
            raise ValueError("zero neighbour count requires zero support fractions")
        positive_neighbours = ~no_neighbours
        if np.any(occupied_fraction[positive_neighbours] <= 0.0):
            raise ValueError("positive neighbour count requires positive occupied fraction")
        if np.any(neighbourhood_support[positive_neighbours] == 0):
            raise ValueError("positive neighbour count requires nonzero neighbourhood support")
        if np.any(~np.any(support_fraction[positive_neighbours] > 0.0, axis=1)):
            raise ValueError("positive neighbour count requires a positive support fraction")
        for column_index, support_bit in enumerate((1, 2, 4)):
            fraction_positive = support_fraction[:, column_index] > 0.0
            bit_set = (neighbourhood_support & support_bit) != 0
            if not np.array_equal(fraction_positive, bit_set):
                raise ValueError(
                    "neighbourhood support bits and support fractions are inconsistent"
                )

    @property
    def candidate_count(self) -> int:
        """Return the number of retained candidate markers."""

        return len(self.candidate_id)


def _validate_runtime_input(canopy_model: CanopyHeightModelResult) -> tuple[int, int]:
    if type(canopy_model.source_identity) is not TreeSegmentationInputIdentity:
        raise ValueError("source_identity must be a TreeSegmentationInputIdentity")
    if type(canopy_model.config) is not CanopyHeightModelConfig:
        raise ValueError("config must be a CanopyHeightModelConfig")
    arrays = (
        (canopy_model.canopy_height_m, np.float32, "canopy_height_m"),
        (canopy_model.occupied_mask, np.bool_, "occupied_mask"),
        (canopy_model.point_count_by_cell, np.int32, "point_count_by_cell"),
        (canopy_model.support_flags_by_cell, np.uint8, "support_flags_by_cell"),
        (canopy_model.peak_support_flags_by_cell, np.uint8, "peak_support_flags_by_cell"),
    )
    shape: tuple[int, int] | None = None
    for value, dtype, name in arrays:
        checked = _result_array(value, dtype, (None, None), name)
        if shape is None:
            shape = checked.shape
        elif checked.shape != shape:
            raise ValueError("all CHM arrays must have identical shapes")
    assert shape is not None
    if shape != (0, 0) and (shape[0] == 0 or shape[1] == 0):
        raise ValueError("an empty CHM must have shape (0, 0)")
    _source_origin(canopy_model.grid_origin_xy, shape)

    occupied = canopy_model.occupied_mask
    counts = canopy_model.point_count_by_cell
    height = canopy_model.canopy_height_m
    support = canopy_model.support_flags_by_cell
    peak_support = canopy_model.peak_support_flags_by_cell
    if np.any(counts < 0) or not np.array_equal(occupied, counts > 0):
        raise ValueError("occupied_mask must equal positive point counts")
    if not np.all(np.isnan(height[~occupied])):
        raise ValueError("empty CHM cells must contain NaN height")
    if np.any(~np.isfinite(height[occupied])) or np.any(height[occupied] < 0.0):
        raise ValueError("occupied CHM cells must contain finite non-negative height")
    if (
        np.any(support & _UNKNOWN_SUPPORT_BITS)
        or np.any(peak_support & _UNKNOWN_SUPPORT_BITS)
        or np.any(support[~occupied] != 0)
        or np.any(support[occupied] == 0)
        or np.any(peak_support[~occupied] != 0)
        or np.any(peak_support[occupied] == 0)
        or np.any(peak_support & np.bitwise_not(support))
    ):
        raise ValueError("CHM support flags are inconsistent")
    included = _non_negative_python_int(canopy_model.included_point_count, "included_point_count")
    excluded = _non_negative_python_int(canopy_model.excluded_point_count, "excluded_point_count")
    if int(np.sum(counts, dtype=np.int64)) != included:
        raise ValueError("point-count sum must equal included_point_count")
    if included + excluded != canopy_model.source_identity.point_count:
        raise ValueError("included and excluded counts must equal source point_count")
    return shape


def _resource_error(
    message: str,
    *,
    shape: tuple[int, int],
    local_offset_count: int | None = None,
    prominence_offset_count: int | None = None,
    estimated_temporary_bytes: int | None = None,
    candidate_count: int | None = None,
    candidate_neighbour_evaluations: int | None = None,
) -> TreetopDetectionError:
    return TreetopDetectionError(
        TreetopDetectionErrorCode.RESOURCE_LIMIT,
        message,
        row_count=shape[0],
        column_count=shape[1],
        total_cell_count=shape[0] * shape[1],
        local_maximum_offset_count=local_offset_count,
        prominence_offset_count=prominence_offset_count,
        estimated_temporary_bytes=estimated_temporary_bytes,
        candidate_count=candidate_count,
        candidate_neighbour_evaluations=candidate_neighbour_evaluations,
    )


def _generate_offsets(
    radius_m: float,
    cell_size_m: float,
    shape: tuple[int, int],
    config: TreetopDetectionConfig,
    *,
    local_maximum: bool,
) -> list[tuple[int, int]]:
    if radius_m == 0.0 or shape == (0, 0):
        return []
    bounding = math.ceil(radius_m / cell_size_m)
    row_extent = min(bounding, shape[0] - 1)
    column_extent = min(bounding, shape[1] - 1)
    guaranteed = math.floor(radius_m / (cell_size_m * math.sqrt(2.0)))
    guaranteed_rows = min(guaranteed, shape[0] - 1)
    guaranteed_columns = min(guaranteed, shape[1] - 1)
    guaranteed_count = (2 * guaranteed_rows + 1) * (2 * guaranteed_columns + 1) - 1
    if guaranteed_count > config.maximum_neighbourhood_cells:
        kwargs = (
            {"local_offset_count": guaranteed_count}
            if local_maximum
            else {"prominence_offset_count": guaranteed_count}
        )
        raise _resource_error(
            "physical neighbourhood exceeds maximum_neighbourhood_cells",
            shape=shape,
            **kwargs,
        )

    radius_squared = radius_m * radius_m
    offsets: list[tuple[int, int]] = []
    for dr in range(-row_extent, row_extent + 1):
        dr_squared = (dr * cell_size_m) ** 2
        for dc in range(-column_extent, column_extent + 1):
            if (dr == 0 and dc == 0) or dr_squared + (dc * cell_size_m) ** 2 > radius_squared:
                continue
            offsets.append((dr, dc))
            if len(offsets) > config.maximum_neighbourhood_cells:
                kwargs = (
                    {"local_offset_count": len(offsets)}
                    if local_maximum
                    else {"prominence_offset_count": len(offsets)}
                )
                raise _resource_error(
                    "physical neighbourhood exceeds maximum_neighbourhood_cells",
                    shape=shape,
                    **kwargs,
                )
    return offsets


def _prominence_axis_extent_cells(
    prominence_radius_m: float,
    cell_size_m: float,
) -> int:
    return math.floor(np.float64(prominence_radius_m) / np.float64(cell_size_m))


def _local_maximum_resource_preflight(
    shape: tuple[int, int],
    config: TreetopDetectionConfig,
    local_offset_count: int,
    prominence_offset_count: int,
) -> int:
    total = int(shape[0]) * int(shape[1])
    boolean_item_size = int(np.dtype(np.bool_).itemsize)
    estimated = total * boolean_item_size * 3
    if estimated > config.maximum_temporary_bytes:
        raise _resource_error(
            "local-maximum work arrays exceed maximum_temporary_bytes",
            shape=shape,
            local_offset_count=local_offset_count,
            prominence_offset_count=prominence_offset_count,
            estimated_temporary_bytes=estimated,
        )
    return estimated


def _overlap_slices(
    shape: tuple[int, int],
    dr: int,
    dc: int,
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    center_rows = slice(max(0, -dr), min(shape[0], shape[0] - dr))
    center_columns = slice(max(0, -dc), min(shape[1], shape[1] - dc))
    neighbour_rows = slice(max(0, dr), min(shape[0], shape[0] + dr))
    neighbour_columns = slice(max(0, dc), min(shape[1], shape[1] + dc))
    return (center_rows, center_columns), (neighbour_rows, neighbour_columns)


def _allocate_local_maximum_work_arrays(
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty(shape, dtype=np.bool_),
        np.empty(shape, dtype=np.bool_),
        np.empty(shape, dtype=np.bool_),
    )


def _local_maximum_mask(
    canopy_model: CanopyHeightModelResult,
    config: TreetopDetectionConfig,
    offsets: list[tuple[int, int]],
) -> np.ndarray:
    height = canopy_model.canopy_height_m
    occupied = canopy_model.occupied_mask
    height_valid, local_maximum, comparison_workspace = _allocate_local_maximum_work_arrays(
        height.shape
    )
    np.greater_equal(
        height,
        np.float64(config.minimum_peak_hag_m),
        out=height_valid,
    )
    np.logical_and(height_valid, occupied, out=height_valid)
    np.copyto(local_maximum, height_valid)
    for dr, dc in offsets:
        center_slice, neighbour_slice = _overlap_slices(height.shape, dr, dc)
        center = height[center_slice]
        neighbour = height[neighbour_slice]
        workspace = comparison_workspace[center_slice]
        np.greater(neighbour, center, out=workspace)
        np.logical_and(workspace, occupied[neighbour_slice], out=workspace)
        np.logical_not(workspace, out=workspace)
        np.logical_and(local_maximum[center_slice], workspace, out=local_maximum[center_slice])
    return local_maximum


def _row_run_count(mask: np.ndarray, height: np.ndarray, row: int) -> int:
    columns = np.flatnonzero(mask[row])
    if not len(columns):
        return 0
    starts = np.ones(len(columns), dtype=np.bool_)
    starts[1:] = (np.diff(columns) != 1) | (height[row, columns[1:]] != height[row, columns[:-1]])
    return int(np.count_nonzero(starts))


def _allocate_run_arrays(
    run_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty(run_count, dtype=np.int32),
        np.empty(run_count, dtype=np.int32),
        np.empty(run_count, dtype=np.int32),
        np.empty(run_count, dtype=np.float32),
        np.arange(run_count, dtype=np.int32),
    )


def _allocate_plateau_labels(shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, -1, dtype=np.int32)


def _plateau_run_label_resource_preflight(
    shape: tuple[int, int],
    run_count: int,
    config: TreetopDetectionConfig,
    local_offset_count: int,
    prominence_offset_count: int,
) -> int:
    total = int(shape[0]) * int(shape[1])
    run_total = int(run_count)
    local_maximum_bytes = total * int(np.dtype(np.bool_).itemsize)
    label_bytes = total * int(np.dtype(np.int32).itemsize)
    int32_item_size = int(np.dtype(np.int32).itemsize)
    int64_item_size = int(np.dtype(np.int64).itemsize)
    float32_item_size = int(np.dtype(np.float32).itemsize)
    run_row_start_end_bytes = run_total * int32_item_size * 3
    run_height_bytes = run_total * float32_item_size
    parent_bytes = run_total * int32_item_size
    roots_bytes = run_total * int32_item_size
    unique_roots_bytes = run_total * int32_item_size
    compact_search_bytes = run_total * int64_item_size
    compact_mapping_bytes = run_total * int32_item_size
    root_ordering_workspace_bytes = run_total * int32_item_size * 2
    estimated = (
        local_maximum_bytes
        + label_bytes
        + run_row_start_end_bytes
        + run_height_bytes
        + parent_bytes
        + roots_bytes
        + unique_roots_bytes
        + compact_search_bytes
        + compact_mapping_bytes
        + root_ordering_workspace_bytes
    )
    if estimated > config.maximum_temporary_bytes:
        raise _resource_error(
            "plateau run and label arrays exceed maximum_temporary_bytes",
            shape=shape,
            local_offset_count=local_offset_count,
            prominence_offset_count=prominence_offset_count,
            estimated_temporary_bytes=estimated,
        )
    return estimated


def _plateau_run_count_resource_preflight(
    shape: tuple[int, int],
    config: TreetopDetectionConfig,
    local_offset_count: int,
    prominence_offset_count: int,
) -> int:
    total = int(shape[0]) * int(shape[1])
    maximum_row_width = int(shape[1])
    local_maximum_bytes = total * int(np.dtype(np.bool_).itemsize)
    maximum_row_counting_workspace_bytes = maximum_row_width * 64
    estimated = local_maximum_bytes + maximum_row_counting_workspace_bytes
    if estimated > config.maximum_temporary_bytes:
        raise _resource_error(
            "plateau run-count workspace exceeds maximum_temporary_bytes",
            shape=shape,
            local_offset_count=local_offset_count,
            prominence_offset_count=prominence_offset_count,
            estimated_temporary_bytes=estimated,
        )
    return estimated


def _plateau_representative_resource_preflight(
    shape: tuple[int, int],
    local_maximum_cell_count: int,
    component_count: int,
    config: TreetopDetectionConfig,
    local_offset_count: int,
    prominence_offset_count: int,
) -> int:
    total = int(shape[0]) * int(shape[1])
    member_total = int(local_maximum_cell_count)
    component_total = int(component_count)
    int32_item_size = int(np.dtype(np.int32).itemsize)
    int64_item_size = int(np.dtype(np.int64).itemsize)
    float64_item_size = int(np.dtype(np.float64).itemsize)
    label_bytes = total * int32_item_size
    local_maximum_rows_columns_bytes = member_total * int64_item_size * 2
    component_ids_bytes = member_total * int32_item_size
    component_count_bytes = component_total * (int64_item_size + int32_item_size)
    row_column_sums_bytes = component_total * float64_item_size * 2
    stable_order_bytes = member_total * int64_item_size
    boundaries_bytes = (component_total + 1) * int64_item_size
    representative_rows_columns_bytes = component_total * int64_item_size * 2
    component_selection_workspace_bytes = component_total * int64_item_size * 2
    deterministic_ranking_workspace_bytes = member_total * 64
    estimated = (
        label_bytes
        + local_maximum_rows_columns_bytes
        + component_ids_bytes
        + component_count_bytes
        + row_column_sums_bytes
        + stable_order_bytes
        + boundaries_bytes
        + representative_rows_columns_bytes
        + component_selection_workspace_bytes
        + deterministic_ranking_workspace_bytes
    )
    if estimated > config.maximum_temporary_bytes:
        raise _resource_error(
            "plateau representative arrays exceed maximum_temporary_bytes",
            shape=shape,
            local_offset_count=local_offset_count,
            prominence_offset_count=prominence_offset_count,
            estimated_temporary_bytes=estimated,
            candidate_count=component_count,
        )
    return estimated


def _select_multicell_representative(
    member_rows: np.ndarray,
    member_columns: np.ndarray,
    point_count: np.ndarray,
    centroid_row: float,
    centroid_column: float,
) -> tuple[int, int]:
    distance = (member_rows - centroid_row) ** 2 + (member_columns - centroid_column) ** 2
    member_points = point_count[member_rows, member_columns]
    ranked = np.lexsort(
        (
            member_columns,
            member_rows,
            -member_points.astype(np.int64),
            distance,
        )
    )
    chosen = int(ranked[0])
    return int(member_rows[chosen]), int(member_columns[chosen])


def _represent_plateaus(
    local_maximum: np.ndarray,
    labels: np.ndarray,
    point_count: np.ndarray,
    component_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = np.nonzero(local_maximum)
    component = labels[rows, columns]
    if np.any(component < 0):
        shape = local_maximum.shape
        raise TreetopDetectionError(
            TreetopDetectionErrorCode.INTERNAL_INVARIANT,
            "local-maximum cell lacks a plateau component",
            row_count=shape[0],
            column_count=shape[1],
            total_cell_count=shape[0] * shape[1],
        )
    counts = np.bincount(component, minlength=component_count).astype(np.int32)
    row_sum = np.bincount(component, weights=rows, minlength=component_count)
    column_sum = np.bincount(component, weights=columns, minlength=component_count)
    order = np.argsort(component, kind="stable")
    boundaries = np.zeros(component_count + 1, dtype=np.int64)
    boundaries[1:] = np.cumsum(counts, dtype=np.int64)
    representative_rows = np.empty(component_count, dtype=np.int64)
    representative_columns = np.empty(component_count, dtype=np.int64)

    singleton_components = np.flatnonzero(counts == 1)
    singleton_members = order[boundaries[singleton_components]]
    representative_rows[singleton_components] = rows[singleton_members]
    representative_columns[singleton_components] = columns[singleton_members]

    for component_id in np.flatnonzero(counts > 1):
        members = order[boundaries[component_id] : boundaries[component_id + 1]]
        member_rows = rows[members]
        member_columns = columns[members]
        centroid_row = row_sum[component_id] / counts[component_id]
        centroid_column = column_sum[component_id] / counts[component_id]
        selected_row, selected_column = _select_multicell_representative(
            member_rows,
            member_columns,
            point_count,
            centroid_row,
            centroid_column,
        )
        representative_rows[component_id] = selected_row
        representative_columns[component_id] = selected_column
    return representative_rows, representative_columns, counts


def _label_plateau_runs(
    local_maximum: np.ndarray,
    height: np.ndarray,
    run_count: int,
    config: TreetopDetectionConfig,
    estimated_temporary_bytes: int,
    local_offset_count: int,
    prominence_offset_count: int,
) -> tuple[np.ndarray, int]:
    shape = local_maximum.shape
    run_row, run_start, run_end, run_height, parent = _allocate_run_arrays(run_count)

    def find(value: int) -> int:
        while int(parent[value]) != value:
            parent[value] = parent[int(parent[value])]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        parent[larger] = smaller

    position = 0
    previous_start_id = 0
    previous_end_id = 0
    for row in range(shape[0]):
        columns = np.flatnonzero(local_maximum[row])
        current_start_id = position
        if len(columns):
            starts_mask = np.ones(len(columns), dtype=np.bool_)
            starts_mask[1:] = (np.diff(columns) != 1) | (
                height[row, columns[1:]] != height[row, columns[:-1]]
            )
            starts = np.flatnonzero(starts_mask)
            ends = np.concatenate((starts[1:] - 1, np.array([len(columns) - 1])))
            for start_position, end_position in zip(starts, ends, strict=True):
                run_row[position] = row
                run_start[position] = columns[start_position]
                run_end[position] = columns[end_position]
                run_height[position] = height[row, columns[start_position]]
                position += 1
        current_end_id = position

        previous_cursor = previous_start_id
        for current in range(current_start_id, current_end_id):
            current_start = int(run_start[current])
            current_end = int(run_end[current])
            while (
                previous_cursor < previous_end_id
                and int(run_end[previous_cursor]) < current_start - 1
            ):
                previous_cursor += 1
            scan = previous_cursor
            while scan < previous_end_id and int(run_start[scan]) <= current_end + 1:
                if run_height[scan] == run_height[current]:
                    union(scan, current)
                scan += 1
        previous_start_id = current_start_id
        previous_end_id = current_end_id
    if position != run_count:
        raise TreetopDetectionError(
            TreetopDetectionErrorCode.INTERNAL_INVARIANT,
            "plateau run count changed during deterministic labeling",
            row_count=shape[0],
            column_count=shape[1],
            total_cell_count=shape[0] * shape[1],
        )

    roots = np.fromiter((find(index) for index in range(run_count)), dtype=np.int32)
    unique_roots = np.unique(roots)
    component_count = len(unique_roots)
    if component_count > config.maximum_candidate_count:
        raise _resource_error(
            "plateau proposal count exceeds maximum_candidate_count",
            shape=shape,
            local_offset_count=local_offset_count,
            prominence_offset_count=prominence_offset_count,
            estimated_temporary_bytes=estimated_temporary_bytes,
            candidate_count=component_count,
        )
    compact = np.searchsorted(unique_roots, roots).astype(np.int32, copy=False)
    labels = _allocate_plateau_labels(shape)
    for run_id in range(run_count):
        labels[
            int(run_row[run_id]),
            int(run_start[run_id]) : int(run_end[run_id]) + 1,
        ] = compact[run_id]
    return labels, component_count


def _plateau_components(
    local_maximum: np.ndarray,
    height: np.ndarray,
    point_count: np.ndarray,
    config: TreetopDetectionConfig,
    local_offset_count: int,
    prominence_offset_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = local_maximum.shape
    _plateau_run_count_resource_preflight(
        shape,
        config,
        local_offset_count,
        prominence_offset_count,
    )
    run_count = sum(_row_run_count(local_maximum, height, row) for row in range(shape[0]))
    estimated = _plateau_run_label_resource_preflight(
        shape,
        run_count,
        config,
        local_offset_count,
        prominence_offset_count,
    )
    if run_count == 0:
        labels = _allocate_plateau_labels(shape)
        _plateau_representative_resource_preflight(
            shape,
            0,
            0,
            config,
            local_offset_count,
            prominence_offset_count,
        )
        return (
            labels,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
        )

    labels, component_count = _label_plateau_runs(
        local_maximum,
        height,
        run_count,
        config,
        estimated,
        local_offset_count,
        prominence_offset_count,
    )
    local_maximum_cell_count = int(np.count_nonzero(local_maximum))
    _plateau_representative_resource_preflight(
        shape,
        local_maximum_cell_count,
        component_count,
        config,
        local_offset_count,
        prominence_offset_count,
    )
    representative_rows, representative_columns, counts = _represent_plateaus(
        local_maximum,
        labels,
        point_count,
        component_count,
    )
    return labels, representative_rows, representative_columns, counts


@dataclass(frozen=True)
class _CandidateMetrics:
    prominence_m: np.ndarray
    prominence_available: np.ndarray
    neighbour_count: np.ndarray
    occupied_fraction: np.ndarray
    neighbourhood_support: np.ndarray
    support_fraction: np.ndarray
    edge_limited: np.ndarray


def _metric_resource_preflight(
    shape: tuple[int, int],
    proposal_count: int,
    prominence_offset_count: int,
    annulus_offset_count: int,
    config: TreetopDetectionConfig,
    dense_estimate: int,
) -> int:
    evaluations = int(proposal_count) * int(prominence_offset_count)
    if evaluations > config.maximum_candidate_neighbour_evaluations:
        raise _resource_error(
            "candidate-neighbour evaluations exceed configured maximum",
            shape=shape,
            prominence_offset_count=prominence_offset_count,
            candidate_count=proposal_count,
            candidate_neighbour_evaluations=evaluations,
        )
    fixed_candidate_bytes = proposal_count * 64
    remaining = config.maximum_temporary_bytes - dense_estimate - fixed_candidate_bytes
    bytes_per_candidate = max(1, annulus_offset_count * 8 + 64)
    if remaining < bytes_per_candidate:
        estimated = dense_estimate + fixed_candidate_bytes + bytes_per_candidate
        raise _resource_error(
            "candidate metric workspace exceeds maximum_temporary_bytes",
            shape=shape,
            prominence_offset_count=prominence_offset_count,
            estimated_temporary_bytes=estimated,
            candidate_count=proposal_count,
            candidate_neighbour_evaluations=evaluations,
        )
    return max(1, min(proposal_count, remaining // bytes_per_candidate))


def _evaluate_candidate_metrics(
    canopy_model: CanopyHeightModelResult,
    labels: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    offsets: list[tuple[int, int]],
    annulus_offsets: list[tuple[int, int]],
    config: TreetopDetectionConfig,
    chunk_size: int,
) -> _CandidateMetrics:
    count = len(rows)
    prominence = np.zeros(count, dtype=np.float32)
    prominence_available = np.zeros(count, dtype=np.bool_)
    neighbour_count = np.zeros(count, dtype=np.int32)
    occupied_fraction = np.zeros(count, dtype=np.float32)
    neighbourhood_support = np.zeros(count, dtype=np.uint8)
    support_fraction = np.zeros((count, 3), dtype=np.float32)
    edge_limited = np.zeros(count, dtype=np.bool_)
    annulus_positions = {offset: index for index, offset in enumerate(annulus_offsets)}
    height = canopy_model.canopy_height_m
    occupied = canopy_model.occupied_mask
    support = canopy_model.support_flags_by_cell
    shape = height.shape
    axis_extent = _prominence_axis_extent_cells(
        config.prominence_radius_m,
        canopy_model.config.cell_size_m,
    )

    for start in range(0, count, chunk_size):
        end = min(count, start + chunk_size)
        chunk_rows = rows[start:end]
        chunk_columns = columns[start:end]
        chunk_components = labels[chunk_rows, chunk_columns]
        chunk_count = end - start
        denominators = np.zeros(chunk_count, dtype=np.int32)
        bit_counts = np.zeros((chunk_count, 3), dtype=np.int32)
        ring = np.full((chunk_count, len(annulus_offsets)), np.nan, dtype=np.float64)
        if axis_extent > 0:
            edge_limited[start:end] = (
                (chunk_rows < axis_extent)
                | (chunk_rows >= shape[0] - axis_extent)
                | (chunk_columns < axis_extent)
                | (chunk_columns >= shape[1] - axis_extent)
            )
        for offset in offsets:
            dr, dc = offset
            neighbour_rows = chunk_rows + dr
            neighbour_columns = chunk_columns + dc
            in_bounds = (
                (neighbour_rows >= 0)
                & (neighbour_rows < shape[0])
                & (neighbour_columns >= 0)
                & (neighbour_columns < shape[1])
            )
            denominators += in_bounds
            if not np.any(in_bounds):
                continue
            positions = np.flatnonzero(in_bounds)
            selected_rows = neighbour_rows[positions]
            selected_columns = neighbour_columns[positions]
            selected_occupied = occupied[selected_rows, selected_columns]
            occupied_positions = positions[selected_occupied]
            if not len(occupied_positions):
                continue
            occupied_rows = neighbour_rows[occupied_positions]
            occupied_columns = neighbour_columns[occupied_positions]
            selected_support = support[occupied_rows, occupied_columns]
            neighbour_count[start + occupied_positions] += 1
            neighbourhood_support[start + occupied_positions] |= selected_support
            for bit_index, flag in enumerate(
                (
                    CanopySupportFlag.CONFIRMED,
                    CanopySupportFlag.REVIEW_REQUIRED,
                    CanopySupportFlag.PROVISIONAL,
                )
            ):
                bit_counts[occupied_positions, bit_index] += (selected_support & int(flag)) != 0
            ring_position = annulus_positions.get(offset)
            if ring_position is not None:
                different_plateau = (
                    labels[occupied_rows, occupied_columns] != chunk_components[occupied_positions]
                )
                eligible = occupied_positions[different_plateau]
                ring[eligible, ring_position] = height[
                    neighbour_rows[eligible],
                    neighbour_columns[eligible],
                ].astype(np.float64)
        occupied_values = neighbour_count[start:end]
        valid_denominator = denominators > 0
        occupied_fraction[start:end][valid_denominator] = (
            occupied_values[valid_denominator] / denominators[valid_denominator]
        ).astype(np.float32)
        occupied_nonzero = occupied_values > 0
        support_fraction[start:end][occupied_nonzero] = (
            bit_counts[occupied_nonzero] / occupied_values[occupied_nonzero, None]
        ).astype(np.float32)
        for local_index in range(chunk_count):
            values = ring[local_index]
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            reference = float(np.median(values.astype(np.float64, copy=False)))
            peak = float(height[chunk_rows[local_index], chunk_columns[local_index]])
            prominence[start + local_index] = np.float32(max(0.0, peak - reference))
            prominence_available[start + local_index] = True
    return _CandidateMetrics(
        prominence,
        prominence_available,
        neighbour_count,
        occupied_fraction,
        neighbourhood_support,
        support_fraction,
        edge_limited,
    )


def _candidate_order(
    peak_height: np.ndarray,
    prominence_available: np.ndarray,
    prominence: np.ndarray,
    occupied_fraction: np.ndarray,
    point_count: np.ndarray,
    plateau_count: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    return np.lexsort(
        (
            columns,
            rows,
            -plateau_count.astype(np.int64),
            -point_count.astype(np.int64),
            -occupied_fraction.astype(np.float64),
            -prominence.astype(np.float64),
            (~prominence_available).astype(np.uint8),
            -peak_height.astype(np.float64),
        )
    )


def _apply_nms(
    rows: np.ndarray,
    columns: np.ndarray,
    cell_size_m: float,
    minimum_separation_m: float,
) -> np.ndarray:
    if minimum_separation_m == 0.0:
        return np.arange(len(rows), dtype=np.int64)
    separation_squared = minimum_separation_m**2
    bins: dict[tuple[int, int], list[int]] = {}
    retained: list[int] = []
    for index in range(len(rows)):
        bin_row = math.floor(int(rows[index]) * cell_size_m / minimum_separation_m)
        bin_column = math.floor(int(columns[index]) * cell_size_m / minimum_separation_m)
        suppress = False
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                for other in bins.get(
                    (bin_row + row_delta, bin_column + column_delta),
                    (),
                ):
                    delta_row = int(rows[index]) - int(rows[other])
                    delta_column = int(columns[index]) - int(columns[other])
                    squared_distance = (delta_row * delta_row + delta_column * delta_column) * (
                        cell_size_m**2
                    )
                    if squared_distance < separation_squared:
                        suppress = True
                        break
                if suppress:
                    break
            if suppress:
                break
        if suppress:
            continue
        retained.append(index)
        bins.setdefault((bin_row, bin_column), []).append(index)
    return np.asarray(retained, dtype=np.int64)


def _freeze_array(value: Any, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if result.size:
        result[result == 0] = 0
    result.setflags(write=False)
    return result


def _empty_array(shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    result = np.empty(shape, dtype=dtype, order="C")
    result.setflags(write=False)
    return result


def _empty_result(
    canopy_model: CanopyHeightModelResult,
    config: TreetopDetectionConfig,
    *,
    plateau_count: int = 0,
    evidence_rejected_count: int = 0,
) -> TreetopCandidateResult:
    return TreetopCandidateResult(
        source_identity=canopy_model.source_identity,
        source_chm_config=canopy_model.config,
        source_grid_origin_xy=canopy_model.grid_origin_xy,
        source_grid_shape=canopy_model.canopy_height_m.shape,
        config=config,
        local_maximum_plateau_count=plateau_count,
        evidence_rejected_count=evidence_rejected_count,
        suppressed_candidate_count=0,
        candidate_id=_empty_array((0,), np.int32),
        row_index=_empty_array((0,), np.int64),
        column_index=_empty_array((0,), np.int64),
        local_xy=_empty_array((0, 2), np.float64),
        peak_height_m=_empty_array((0,), np.float32),
        prominence_m=_empty_array((0,), np.float32),
        prominence_available=_empty_array((0,), np.bool_),
        plateau_cell_count=_empty_array((0,), np.int32),
        source_cell_point_count=_empty_array((0,), np.int32),
        source_support_flags=_empty_array((0,), np.uint8),
        source_peak_support_flags=_empty_array((0,), np.uint8),
        neighbour_occupied_cell_count=_empty_array((0,), np.int32),
        neighbourhood_occupied_fraction=_empty_array((0,), np.float32),
        neighbourhood_support_flags=_empty_array((0,), np.uint8),
        neighbourhood_support_fraction=_empty_array((0, 3), np.float32),
        edge_limited=_empty_array((0,), np.bool_),
    )


def detect_treetop_candidates(
    canopy_model: CanopyHeightModelResult,
    config: TreetopDetectionConfig | None = None,
) -> TreetopCandidateResult:
    """Detect deterministic sparse treetop candidates from a validated in-memory CHM."""

    if type(canopy_model) is not CanopyHeightModelResult:
        raise TypeError("canopy_model must be a CanopyHeightModelResult")
    if config is None:
        effective_config = TreetopDetectionConfig()
    elif type(config) is TreetopDetectionConfig:
        effective_config = config
    else:
        raise TypeError("config must be a TreetopDetectionConfig or None")
    shape = _validate_runtime_input(canopy_model)
    if shape == (0, 0):
        return _empty_result(canopy_model, effective_config)

    local_offsets: list[tuple[int, int]] = []
    prominence_offsets: list[tuple[int, int]] = []
    dense_estimate = 0
    try:
        local_offsets = _generate_offsets(
            effective_config.local_max_radius_m,
            canopy_model.config.cell_size_m,
            shape,
            effective_config,
            local_maximum=True,
        )
        prominence_offsets = _generate_offsets(
            effective_config.prominence_radius_m,
            canopy_model.config.cell_size_m,
            shape,
            effective_config,
            local_maximum=False,
        )
        dense_estimate = _local_maximum_resource_preflight(
            shape,
            effective_config,
            len(local_offsets),
            len(prominence_offsets),
        )
        local_maximum = _local_maximum_mask(
            canopy_model,
            effective_config,
            local_offsets,
        )
        if not np.any(local_maximum):
            return _empty_result(canopy_model, effective_config)
        labels, rows, columns, plateau_count = _plateau_components(
            local_maximum,
            canopy_model.canopy_height_m,
            canopy_model.point_count_by_cell,
            effective_config,
            len(local_offsets),
            len(prominence_offsets),
        )
        del local_maximum
        proposal_count = len(rows)
        if proposal_count == 0:
            return _empty_result(canopy_model, effective_config)
        local_radius_squared = effective_config.local_max_radius_m**2
        cell_size_squared = canopy_model.config.cell_size_m**2
        annulus_offsets = [
            (dr, dc)
            for dr, dc in prominence_offsets
            if (dr * dr + dc * dc) * cell_size_squared > local_radius_squared
        ]
        metric_base_estimate = (
            int(shape[0]) * int(shape[1]) * int(np.dtype(np.int32).itemsize)
            + int(proposal_count) * 20
        )
        dense_estimate = max(dense_estimate, metric_base_estimate)
        chunk_size = _metric_resource_preflight(
            shape,
            proposal_count,
            len(prominence_offsets),
            len(annulus_offsets),
            effective_config,
            metric_base_estimate,
        )
        metrics = _evaluate_candidate_metrics(
            canopy_model,
            labels,
            rows,
            columns,
            prominence_offsets,
            annulus_offsets,
            effective_config,
            chunk_size,
        )

        peak_height = canopy_model.canopy_height_m[rows, columns]
        source_point_count = canopy_model.point_count_by_cell[rows, columns]
        source_support = canopy_model.support_flags_by_cell[rows, columns]
        source_peak_support = canopy_model.peak_support_flags_by_cell[rows, columns]
        evidence = (
            metrics.neighbour_count >= effective_config.minimum_neighbour_occupied_cells
        ) | (source_point_count >= effective_config.minimum_isolated_peak_point_count)
        if effective_config.minimum_prominence_m > 0.0:
            evidence &= metrics.prominence_available & (
                metrics.prominence_m.astype(np.float64) >= effective_config.minimum_prominence_m
            )
        evidence_rejected_count = int(proposal_count - np.count_nonzero(evidence))
        if not np.any(evidence):
            return _empty_result(
                canopy_model,
                effective_config,
                plateau_count=proposal_count,
                evidence_rejected_count=evidence_rejected_count,
            )

        rows = rows[evidence]
        columns = columns[evidence]
        plateau_count = plateau_count[evidence]
        peak_height = peak_height[evidence]
        source_point_count = source_point_count[evidence]
        source_support = source_support[evidence]
        source_peak_support = source_peak_support[evidence]
        prominence = metrics.prominence_m[evidence]
        prominence_available = metrics.prominence_available[evidence]
        neighbour_count = metrics.neighbour_count[evidence]
        occupied_fraction = metrics.occupied_fraction[evidence]
        neighbourhood_support = metrics.neighbourhood_support[evidence]
        support_fraction = metrics.support_fraction[evidence]
        edge_limited = metrics.edge_limited[evidence]

        order = _candidate_order(
            peak_height,
            prominence_available,
            prominence,
            occupied_fraction,
            source_point_count,
            plateau_count,
            rows,
            columns,
        )
        rows = rows[order]
        columns = columns[order]
        plateau_count = plateau_count[order]
        peak_height = peak_height[order]
        source_point_count = source_point_count[order]
        source_support = source_support[order]
        source_peak_support = source_peak_support[order]
        prominence = prominence[order]
        prominence_available = prominence_available[order]
        neighbour_count = neighbour_count[order]
        occupied_fraction = occupied_fraction[order]
        neighbourhood_support = neighbourhood_support[order]
        support_fraction = support_fraction[order]
        edge_limited = edge_limited[order]

        if effective_config.minimum_separation_m == 0.0:
            retained = np.arange(len(rows), dtype=np.int64)
        else:
            retained = _apply_nms(
                rows,
                columns,
                canopy_model.config.cell_size_m,
                effective_config.minimum_separation_m,
            )
        suppressed_count = len(rows) - len(retained)
        rows = rows[retained]
        columns = columns[retained]
        plateau_count = plateau_count[retained]
        peak_height = peak_height[retained]
        source_point_count = source_point_count[retained]
        source_support = source_support[retained]
        source_peak_support = source_peak_support[retained]
        prominence = prominence[retained]
        prominence_available = prominence_available[retained]
        neighbour_count = neighbour_count[retained]
        occupied_fraction = occupied_fraction[retained]
        neighbourhood_support = neighbourhood_support[retained]
        support_fraction = support_fraction[retained]
        edge_limited = edge_limited[retained]

        candidate_count = len(rows)
        if candidate_count > np.iinfo(np.int32).max:
            raise _resource_error(
                "candidate IDs cannot fit int32",
                shape=shape,
                local_offset_count=len(local_offsets),
                prominence_offset_count=len(prominence_offsets),
                candidate_count=candidate_count,
            )
        origin = canopy_model.grid_origin_xy
        assert origin is not None
        local_xy = np.empty((candidate_count, 2), dtype=np.float64)
        local_xy[:, 0] = (
            float(origin[0]) + (columns.astype(np.float64) + 0.5) * canopy_model.config.cell_size_m
        )
        local_xy[:, 1] = (
            float(origin[1]) + (rows.astype(np.float64) + 0.5) * canopy_model.config.cell_size_m
        )
        return TreetopCandidateResult(
            source_identity=canopy_model.source_identity,
            source_chm_config=canopy_model.config,
            source_grid_origin_xy=canopy_model.grid_origin_xy,
            source_grid_shape=shape,
            config=effective_config,
            local_maximum_plateau_count=proposal_count,
            evidence_rejected_count=evidence_rejected_count,
            suppressed_candidate_count=suppressed_count,
            candidate_id=_freeze_array(np.arange(candidate_count), np.int32),
            row_index=_freeze_array(rows, np.int64),
            column_index=_freeze_array(columns, np.int64),
            local_xy=_freeze_array(local_xy, np.float64),
            peak_height_m=_freeze_array(peak_height, np.float32),
            prominence_m=_freeze_array(prominence, np.float32),
            prominence_available=_freeze_array(prominence_available, np.bool_),
            plateau_cell_count=_freeze_array(plateau_count, np.int32),
            source_cell_point_count=_freeze_array(source_point_count, np.int32),
            source_support_flags=_freeze_array(source_support, np.uint8),
            source_peak_support_flags=_freeze_array(source_peak_support, np.uint8),
            neighbour_occupied_cell_count=_freeze_array(neighbour_count, np.int32),
            neighbourhood_occupied_fraction=_freeze_array(
                occupied_fraction,
                np.float32,
            ),
            neighbourhood_support_flags=_freeze_array(
                neighbourhood_support,
                np.uint8,
            ),
            neighbourhood_support_fraction=_freeze_array(
                support_fraction,
                np.float32,
            ),
            edge_limited=_freeze_array(edge_limited, np.bool_),
        )
    except MemoryError as exc:
        raise TreetopDetectionError(
            TreetopDetectionErrorCode.ALLOCATION_FAILED,
            "treetop detection allocation failed",
            row_count=shape[0],
            column_count=shape[1],
            total_cell_count=shape[0] * shape[1],
            local_maximum_offset_count=len(local_offsets),
            prominence_offset_count=len(prominence_offsets),
            estimated_temporary_bytes=dense_estimate or None,
        ) from exc
