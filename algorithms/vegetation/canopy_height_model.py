"""Deterministic CPU canopy-height rasterization for validated V3 tree inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, IntFlag
from typing import Any

import numpy as np
from core.contracts.tree_inputs import TreeSegmentationInputIdentity, TreeSegmentationInputSet
from core.contracts.vegetation_instances import VegetationClusterStatus


class CanopySupportFlag(IntFlag):
    """Vegetation-cluster support states retained in each occupied CHM cell."""

    NONE = 0
    CONFIRMED = 1
    REVIEW_REQUIRED = 2
    PROVISIONAL = 4


class CanopyHeightModelErrorCode(str, Enum):
    """Stable runtime failure categories for canopy-height construction."""

    RESOURCE_LIMIT = "resource_limit"
    ALLOCATION_FAILED = "allocation_failed"
    NUMERICAL_INDEX_INVALID = "numerical_index_invalid"


def _finite_float(value: Any, name: str, *, minimum: float, strict: bool) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise ValueError(f"{name} must be a real numeric value")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if (strict and result <= minimum) or (not strict and result < minimum):
        comparison = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be {comparison} {minimum}")
    return result


def _non_negative_python_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative Python integer")
    return value


@dataclass(frozen=True)
class CanopyHeightModelConfig:
    """Configuration that affects only raw canopy-height rasterization."""

    cell_size_m: float = 0.20
    minimum_hag_m: float = 0.30
    maximum_grid_cells: int = 10_000_000

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cell_size_m",
            _finite_float(self.cell_size_m, "cell_size_m", minimum=0.0, strict=True),
        )
        object.__setattr__(
            self,
            "minimum_hag_m",
            _finite_float(self.minimum_hag_m, "minimum_hag_m", minimum=0.0, strict=False),
        )
        value = self.maximum_grid_cells
        if (
            isinstance(value, bool | np.bool_)
            or not isinstance(value, int | np.integer)
            or int(value) <= 0
        ):
            raise ValueError("maximum_grid_cells must be a positive integer")
        object.__setattr__(self, "maximum_grid_cells", int(value))


class CanopyHeightModelError(RuntimeError):
    """A compact CHM runtime failure containing only fixed resource scalars."""

    def __init__(
        self,
        code: CanopyHeightModelErrorCode,
        message: str,
        *,
        row_count: int | None = None,
        column_count: int | None = None,
        total_cell_count: int | None = None,
        estimated_output_bytes: int | None = None,
        estimated_peak_dense_bytes: int | None = None,
    ) -> None:
        if type(code) is not CanopyHeightModelErrorCode:
            raise ValueError("code must be a CanopyHeightModelErrorCode")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        for name, value in (
            ("row_count", row_count),
            ("column_count", column_count),
            ("total_cell_count", total_cell_count),
            ("estimated_output_bytes", estimated_output_bytes),
            ("estimated_peak_dense_bytes", estimated_peak_dense_bytes),
        ):
            if value is not None:
                _non_negative_python_int(value, name)
        self.code = code
        self.row_count = row_count
        self.column_count = column_count
        self.total_cell_count = total_cell_count
        self.estimated_output_bytes = estimated_output_bytes
        self.estimated_peak_dense_bytes = estimated_peak_dense_bytes
        super().__init__(message)


def _result_array(
    value: Any,
    dtype: Any,
    name: str,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise ValueError(f"{name} must be a numpy ndarray")
    if value.dtype != np.dtype(dtype) or value.ndim != 2:
        raise ValueError(f"{name} has invalid dtype or rank")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not value.flags.owndata:
        raise ValueError(f"{name} must own its data")
    if value.flags.writeable:
        raise ValueError(f"{name} must be read-only")
    return value


@dataclass(frozen=True)
class CanopyHeightModelResult:
    """Immutable in-memory CHM and cell-level status evidence for future V3B2."""

    source_identity: TreeSegmentationInputIdentity
    config: CanopyHeightModelConfig
    grid_origin_xy: tuple[float, float] | None
    canopy_height_m: np.ndarray
    occupied_mask: np.ndarray
    point_count_by_cell: np.ndarray
    support_flags_by_cell: np.ndarray
    peak_support_flags_by_cell: np.ndarray
    included_point_count: int
    excluded_point_count: int

    def __post_init__(self) -> None:
        if type(self.source_identity) is not TreeSegmentationInputIdentity:
            raise ValueError("source_identity must be a TreeSegmentationInputIdentity")
        if type(self.config) is not CanopyHeightModelConfig:
            raise ValueError("config must be a CanopyHeightModelConfig")
        included = _non_negative_python_int(self.included_point_count, "included_point_count")
        excluded = _non_negative_python_int(self.excluded_point_count, "excluded_point_count")

        height = _result_array(self.canopy_height_m, np.float32, "canopy_height_m")
        occupied = _result_array(self.occupied_mask, np.bool_, "occupied_mask")
        counts = _result_array(self.point_count_by_cell, np.int32, "point_count_by_cell")
        support = _result_array(self.support_flags_by_cell, np.uint8, "support_flags_by_cell")
        peak_support = _result_array(
            self.peak_support_flags_by_cell,
            np.uint8,
            "peak_support_flags_by_cell",
        )
        shape = height.shape
        if any(value.shape != shape for value in (occupied, counts, support, peak_support)):
            raise ValueError("all CHM arrays must have identical shapes")
        if shape != (0, 0) and (shape[0] == 0 or shape[1] == 0):
            raise ValueError("an empty CHM must have shape (0, 0)")

        if shape == (0, 0):
            if self.grid_origin_xy is not None:
                raise ValueError("an empty CHM requires grid_origin_xy=None")
        else:
            origin = self.grid_origin_xy
            if type(origin) is not tuple or len(origin) != 2:
                raise ValueError("a non-empty CHM requires a two-value grid_origin_xy tuple")
            normalized_origin = (
                _finite_float(origin[0], "grid_origin_xy[0]", minimum=-math.inf, strict=False),
                _finite_float(origin[1], "grid_origin_xy[1]", minimum=-math.inf, strict=False),
            )
            object.__setattr__(self, "grid_origin_xy", normalized_origin)

        if np.any(counts < 0):
            raise ValueError("point_count_by_cell must be non-negative")
        expected_occupied = counts > 0
        if not np.array_equal(occupied, expected_occupied):
            raise ValueError("occupied_mask must equal point_count_by_cell > 0")
        if not np.all(np.isnan(height[~occupied])):
            raise ValueError("empty cells must contain NaN height")
        if np.any(~np.isfinite(height[occupied])) or np.any(height[occupied] < 0.0):
            raise ValueError("occupied cells must contain finite non-negative height")

        allowed_bits = np.uint8(
            int(
                CanopySupportFlag.CONFIRMED
                | CanopySupportFlag.REVIEW_REQUIRED
                | CanopySupportFlag.PROVISIONAL
            )
        )
        unknown_mask = np.uint8(0xFF ^ int(allowed_bits))
        if np.any(support & unknown_mask):
            raise ValueError("support_flags_by_cell contains unknown support bits")
        if np.any(peak_support & unknown_mask):
            raise ValueError("peak_support_flags_by_cell contains unknown support bits")
        if np.any(support[~occupied] != 0) or np.any(support[occupied] == 0):
            raise ValueError("support flags must be nonzero exactly in occupied cells")
        if np.any(peak_support[~occupied] != 0) or np.any(peak_support[occupied] == 0):
            raise ValueError("peak support flags must be nonzero exactly in occupied cells")
        if np.any(peak_support & np.bitwise_not(support)):
            raise ValueError("peak support flags must be a subset of support flags")

        if int(np.sum(counts, dtype=np.int64)) != included:
            raise ValueError("point-count sum must equal included_point_count")
        if included + excluded != self.source_identity.point_count:
            raise ValueError("included and excluded counts must equal source point_count")


_STATUS_SUPPORT = {
    VegetationClusterStatus.CONFIRMED: CanopySupportFlag.CONFIRMED,
    VegetationClusterStatus.REVIEW_REQUIRED: CanopySupportFlag.REVIEW_REQUIRED,
    VegetationClusterStatus.PROVISIONAL: CanopySupportFlag.PROVISIONAL,
}


def _status_support_bits(status: np.ndarray) -> np.ndarray:
    bits = np.zeros(len(status), dtype=np.uint8)
    known_values = {int(value) for value in VegetationClusterStatus}
    if not np.isin(status, np.fromiter(known_values, dtype=np.uint8)).all():
        raise ValueError("cluster_status_by_point contains an unknown status")
    for state, flag in _STATUS_SUPPORT.items():
        bits[status == int(state)] = np.uint8(int(flag))
    return bits


def _require_input_array(
    value: Any,
    dtype: Any,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError(f"{name} has invalid type, dtype, or shape")
    return value


def _validate_runtime_input(tree_input: TreeSegmentationInputSet) -> int:
    count = tree_input.identity.point_count
    points = tree_input.vegetation_points
    _require_input_array(points.xyz_local, np.float32, (count, 3), "xyz_local")
    _require_input_array(
        points.cluster_id_by_point,
        np.int32,
        (count,),
        "cluster_id_by_point",
    )
    _require_input_array(
        points.cluster_status_by_point,
        np.uint8,
        (count,),
        "cluster_status_by_point",
    )
    _require_input_array(points.wire_id_by_point, np.int32, (count,), "wire_id_by_point")
    _require_input_array(
        points.vegetation_id_by_point,
        np.int32,
        (count,),
        "vegetation_id_by_point",
    )
    _require_input_array(
        points.distance_ready_by_point,
        np.bool_,
        (count,),
        "distance_ready_by_point",
    )
    hag = _require_input_array(
        tree_input.heights_above_ground.height_agl_m,
        np.float32,
        (count,),
        "height_agl_m",
    )
    usable = _require_input_array(
        tree_input.ground_usable_by_point,
        np.bool_,
        (count,),
        "ground_usable_by_point",
    )
    xy = points.xyz_local[:, :2]
    if not np.isfinite(xy).all():
        raise ValueError("xyz_local contains nonfinite XY coordinates")
    if np.any(~np.isfinite(hag[usable])) or np.any(hag[usable] < 0.0):
        raise ValueError("ground-usable rows must contain finite non-negative HAG")
    return count


def _numerical_error(message: str) -> CanopyHeightModelError:
    return CanopyHeightModelError(
        CanopyHeightModelErrorCode.NUMERICAL_INDEX_INVALID,
        message,
    )


def _indices_from_relative(
    relative_x: np.ndarray,
    relative_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        type(relative_x) is not np.ndarray
        or type(relative_y) is not np.ndarray
        or relative_x.dtype != np.dtype(np.float64)
        or relative_y.dtype != np.dtype(np.float64)
        or relative_x.ndim != 1
        or relative_y.shape != relative_x.shape
        or not np.isfinite(relative_x).all()
        or not np.isfinite(relative_y).all()
        or np.any(relative_x < 0.0)
        or np.any(relative_y < 0.0)
    ):
        raise _numerical_error("relative cell coordinates are invalid")
    int64_upper = float(2**63)
    if np.any(relative_x >= int64_upper) or np.any(relative_y >= int64_upper):
        raise _numerical_error("cell coordinates cannot be represented as int64")
    columns = np.floor(relative_x).astype(np.int64)
    rows = np.floor(relative_y).astype(np.int64)
    return rows, columns


def _resource_preflight(
    row_count: int,
    column_count: int,
    config: CanopyHeightModelConfig,
) -> tuple[int, int, int]:
    total = row_count * column_count
    output_bytes = total * 11
    peak_dense_bytes = total * 19
    if total > config.maximum_grid_cells:
        raise CanopyHeightModelError(
            CanopyHeightModelErrorCode.RESOURCE_LIMIT,
            "canopy-height grid exceeds maximum_grid_cells",
            row_count=row_count,
            column_count=column_count,
            total_cell_count=total,
            estimated_output_bytes=output_bytes,
            estimated_peak_dense_bytes=peak_dense_bytes,
        )
    return total, output_bytes, peak_dense_bytes


def _validate_point_counts(counts: np.ndarray, included_point_count: int) -> None:
    if (
        type(counts) is not np.ndarray
        or counts.dtype != np.dtype(np.int64)
        or counts.ndim != 1
        or np.any(counts < 0)
        or int(np.sum(counts, dtype=np.int64)) != included_point_count
        or (counts.size and int(np.max(counts)) > np.iinfo(np.int32).max)
    ):
        raise _numerical_error("per-cell point counts cannot be represented as int32")


def _allocate_dense_arrays(
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height = np.full(shape, -np.inf, dtype=np.float32)
    occupied = np.empty(shape, dtype=np.bool_)
    counts = np.empty(shape, dtype=np.int32)
    support = np.zeros(shape, dtype=np.uint8)
    peak_support = np.zeros(shape, dtype=np.uint8)
    return height, occupied, counts, support, peak_support


def _empty_array(dtype: Any) -> np.ndarray:
    value = np.empty((0, 0), dtype=dtype)
    value.setflags(write=False)
    return value


def _empty_result(
    tree_input: TreeSegmentationInputSet,
    config: CanopyHeightModelConfig,
) -> CanopyHeightModelResult:
    return CanopyHeightModelResult(
        source_identity=tree_input.identity,
        config=config,
        grid_origin_xy=None,
        canopy_height_m=_empty_array(np.float32),
        occupied_mask=_empty_array(np.bool_),
        point_count_by_cell=_empty_array(np.int32),
        support_flags_by_cell=_empty_array(np.uint8),
        peak_support_flags_by_cell=_empty_array(np.uint8),
        included_point_count=0,
        excluded_point_count=tree_input.identity.point_count,
    )


def build_canopy_height_model(
    tree_input: TreeSegmentationInputSet,
    config: CanopyHeightModelConfig | None = None,
) -> CanopyHeightModelResult:
    """Rasterize validated segmentation-candidate points into a raw in-memory CHM."""

    if type(tree_input) is not TreeSegmentationInputSet:
        raise TypeError("tree_input must be a TreeSegmentationInputSet")
    if config is None:
        effective_config = CanopyHeightModelConfig()
    elif type(config) is CanopyHeightModelConfig:
        effective_config = config
    else:
        raise TypeError("config must be a CanopyHeightModelConfig or None")

    point_count = _validate_runtime_input(tree_input)
    points = tree_input.vegetation_points
    hag = tree_input.heights_above_ground.height_agl_m
    status_bits = _status_support_bits(points.cluster_status_by_point)
    segmentation_clusters = np.asarray(
        (*tree_input.eligible_cluster_ids, *tree_input.review_only_cluster_ids),
        dtype=np.int32,
    )
    assigned_candidate = (points.cluster_id_by_point >= 0) & np.isin(
        points.cluster_id_by_point,
        segmentation_clusters,
    )
    included = (
        assigned_candidate
        & (status_bits != int(CanopySupportFlag.NONE))
        & (points.wire_id_by_point < 0)
        & tree_input.ground_usable_by_point
        & (hag.astype(np.float64, copy=False) >= np.float64(effective_config.minimum_hag_m))
    )
    included_point_count = int(np.count_nonzero(included))
    if included_point_count == 0:
        return _empty_result(tree_input, effective_config)

    try:
        xy = np.array(points.xyz_local[included, :2], dtype=np.float64, order="C", copy=True)
        selected_hag = np.array(hag[included], dtype=np.float32, order="C", copy=True)
        selected_support = np.array(
            status_bits[included],
            dtype=np.uint8,
            order="C",
            copy=True,
        )
    except MemoryError as exc:
        raise CanopyHeightModelError(
            CanopyHeightModelErrorCode.ALLOCATION_FAILED,
            "canopy-height point work-array allocation failed",
        ) from exc
    xy[xy == 0.0] = 0.0
    selected_hag[selected_hag == 0.0] = 0.0

    cell_size = np.float64(effective_config.cell_size_m)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        origin = np.floor(np.min(xy, axis=0) / cell_size) * cell_size
        relative_x = np.maximum((xy[:, 0] - origin[0]) / cell_size, 0.0)
        relative_y = np.maximum((xy[:, 1] - origin[1]) / cell_size, 0.0)
    if not np.isfinite(origin).all():
        raise _numerical_error("grid origin is nonfinite")
    rows, columns = _indices_from_relative(relative_x, relative_y)
    row_count = int(np.max(rows)) + 1
    column_count = int(np.max(columns)) + 1
    total_cells, output_bytes, peak_dense_bytes = _resource_preflight(
        row_count,
        column_count,
        effective_config,
    )
    cell_ids = rows * np.int64(column_count) + columns
    if np.any(cell_ids < 0) or np.any(cell_ids >= total_cells):
        raise CanopyHeightModelError(
            CanopyHeightModelErrorCode.NUMERICAL_INDEX_INVALID,
            "flat canopy-height cell ID is outside the computed grid",
            row_count=row_count,
            column_count=column_count,
            total_cell_count=total_cells,
            estimated_output_bytes=output_bytes,
            estimated_peak_dense_bytes=peak_dense_bytes,
        )

    shape = (row_count, column_count)
    try:
        counts64 = np.bincount(cell_ids, minlength=total_cells)
        _validate_point_counts(counts64, included_point_count)
        height, occupied, counts, support, peak_support = _allocate_dense_arrays(shape)
        counts.ravel()[:] = counts64
        del counts64
        np.maximum.at(height.ravel(), cell_ids, selected_hag)
        occupied[:] = counts > 0
        np.bitwise_or.at(support.ravel(), cell_ids, selected_support)
        height_flat = height.ravel()
        peak_rows = selected_hag == height_flat[cell_ids]
        np.bitwise_or.at(
            peak_support.ravel(),
            cell_ids[peak_rows],
            selected_support[peak_rows],
        )
        height[~occupied] = np.nan
    except MemoryError as exc:
        raise CanopyHeightModelError(
            CanopyHeightModelErrorCode.ALLOCATION_FAILED,
            "canopy-height dense allocation failed",
            row_count=row_count,
            column_count=column_count,
            total_cell_count=total_cells,
            estimated_output_bytes=output_bytes,
            estimated_peak_dense_bytes=peak_dense_bytes,
        ) from exc

    for value in (height, occupied, counts, support, peak_support):
        value.setflags(write=False)
    return CanopyHeightModelResult(
        source_identity=tree_input.identity,
        config=effective_config,
        grid_origin_xy=(float(origin[0]), float(origin[1])),
        canopy_height_m=height,
        occupied_mask=occupied,
        point_count_by_cell=counts,
        support_flags_by_cell=support,
        peak_support_flags_by_cell=peak_support,
        included_point_count=included_point_count,
        excluded_point_count=point_count - included_point_count,
    )
