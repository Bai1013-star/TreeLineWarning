"""Deterministic CPU marker-controlled canopy-region segmentation."""

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
from algorithms.vegetation.treetop_candidates import (
    TreetopCandidateResult,
    TreetopDetectionConfig,
)

_INT32_MAX = int(np.iinfo(np.int32).max)
_ALLOWED_SUPPORT_BITS = np.uint8(
    int(
        CanopySupportFlag.CONFIRMED
        | CanopySupportFlag.REVIEW_REQUIRED
        | CanopySupportFlag.PROVISIONAL
    )
)
_UNKNOWN_SUPPORT_BITS = np.uint8(0xFF ^ int(_ALLOWED_SUPPORT_BITS))
_OFFSETS_4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
_OFFSETS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
_FORWARD_OFFSETS_4 = ((0, 1), (1, 0))
_FORWARD_OFFSETS_8 = ((0, 1), (1, -1), (1, 0), (1, 1))
_VALIDATION_CHUNK_SIZE = 65_536


def _finite_optional_height(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise ValueError(f"{name} must be None or a real numeric value")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
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


def _matches_canonical_dataclass(value: Any, canonical: Any) -> bool:
    return value == canonical and all(
        type(getattr(value, name)) is type(getattr(canonical, name))
        for name in canonical.__dataclass_fields__
    )


def _validate_identity_runtime(identity: TreeSegmentationInputIdentity) -> None:
    if type(identity) is not TreeSegmentationInputIdentity:
        raise ValueError("source_identity must be a TreeSegmentationInputIdentity")
    canonical = TreeSegmentationInputIdentity(
        **{
            name: getattr(identity, name)
            for name in TreeSegmentationInputIdentity.__dataclass_fields__
        }
    )
    if (
        not _matches_canonical_dataclass(identity, canonical)
        or type(identity.origin_xyz) is not tuple
        or any(type(value) is not float for value in identity.origin_xyz)
    ):
        raise ValueError("source_identity is not canonical")


@dataclass(frozen=True)
class CanopySegmentationConfig:
    """Physical-domain and hard resource controls for canopy segmentation."""

    minimum_segmentation_hag_m: float | None = None
    connectivity: int = 8
    maximum_grid_cells: int = 10_000_000
    maximum_candidate_count: int = 1_000_000
    maximum_component_count: int = 10_000_000
    maximum_temporary_bytes: int = 805_306_368
    maximum_queue_entries: int = 10_000_000
    maximum_edge_evaluations: int = 240_000_000

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_segmentation_hag_m",
            _finite_optional_height(
                self.minimum_segmentation_hag_m,
                "minimum_segmentation_hag_m",
            ),
        )
        object.__setattr__(
            self,
            "connectivity",
            _integer(self.connectivity, "connectivity", minimum=4, maximum=8),
        )
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        limits = (
            ("maximum_grid_cells", _INT32_MAX),
            ("maximum_candidate_count", _INT32_MAX - 1),
            ("maximum_component_count", _INT32_MAX),
            ("maximum_temporary_bytes", None),
            ("maximum_queue_entries", _INT32_MAX),
            ("maximum_edge_evaluations", None),
        )
        for name, maximum in limits:
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, minimum=1, maximum=maximum),
            )


class CanopySegmentationErrorCode(str, Enum):
    """Stable runtime failure categories for canopy segmentation."""

    RESOURCE_LIMIT = "resource_limit"
    ALLOCATION_FAILED = "allocation_failed"
    INPUT_PAIR_MISMATCH = "input_pair_mismatch"
    INTERNAL_INVARIANT = "internal_invariant"


class CanopySegmentationError(RuntimeError):
    """A compact segmentation failure containing fixed resource scalars."""

    def __init__(
        self,
        code: CanopySegmentationErrorCode,
        message: str,
        *,
        row_count: int | None = None,
        column_count: int | None = None,
        total_cell_count: int | None = None,
        domain_cell_count: int | None = None,
        candidate_count: int | None = None,
        component_count: int | None = None,
        estimated_temporary_bytes: int | None = None,
        queue_entry_count: int | None = None,
        edge_evaluation_count: int | None = None,
    ) -> None:
        if type(code) is not CanopySegmentationErrorCode:
            raise ValueError("code must be a CanopySegmentationErrorCode")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        values = (
            ("row_count", row_count),
            ("column_count", column_count),
            ("total_cell_count", total_cell_count),
            ("domain_cell_count", domain_cell_count),
            ("candidate_count", candidate_count),
            ("component_count", component_count),
            ("estimated_temporary_bytes", estimated_temporary_bytes),
            ("queue_entry_count", queue_entry_count),
            ("edge_evaluation_count", edge_evaluation_count),
        )
        for name, value in values:
            if value is not None:
                _non_negative_python_int(value, name)
            setattr(self, name, value)
        self.code = code
        super().__init__(message)


def _source_grid_shape(value: Any) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not int or item < 0 for item in value)
        or (value != (0, 0) and (value[0] == 0 or value[1] == 0))
    ):
        raise ValueError("source_grid_shape must be (0, 0) or contain two positive Python integers")
    return value


def _source_origin(
    value: Any,
    shape: tuple[int, int],
    *,
    exact_float: bool,
) -> tuple[float, float] | None:
    if shape == (0, 0):
        if value is not None:
            raise ValueError("an empty source grid requires source_grid_origin_xy=None")
        return None
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("a non-empty source grid requires a two-value origin tuple")
    if exact_float and any(type(item) is not float for item in value):
        raise ValueError("source_grid_origin_xy must contain Python floats")
    if any(
        isinstance(item, bool | np.bool_)
        or not isinstance(item, int | float | np.integer | np.floating)
        for item in value
    ):
        raise ValueError("source_grid_origin_xy must contain finite numeric values")
    try:
        result = (float(value[0]), float(value[1]))
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("source_grid_origin_xy must contain finite numeric values") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError("source_grid_origin_xy must contain finite numeric values")
    return result


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


def _edge_flags(labels: np.ndarray, candidate_count: int) -> np.ndarray:
    flags = np.zeros(candidate_count, dtype=np.bool_)
    if not candidate_count or labels.shape == (0, 0):
        return flags
    boundaries = (labels[0], labels[-1], labels[:, 0], labels[:, -1])
    for boundary in boundaries:
        positive = boundary[boundary > 0]
        if len(positive):
            flags[np.unique(positive) - 1] = True
    return flags


@dataclass(frozen=True)
class CanopySegmentationResult:
    """Immutable candidate-linked canopy-region labels for future V3B4 mapping."""

    source_identity: TreeSegmentationInputIdentity
    source_chm_config: CanopyHeightModelConfig
    source_grid_origin_xy: tuple[float, float] | None
    source_grid_shape: tuple[int, int]
    source_treetop_config: TreetopDetectionConfig
    config: CanopySegmentationConfig
    seeded_component_count: int
    unseeded_component_count: int
    assigned_cell_count: int
    unassigned_domain_cell_count: int
    ambiguous_cell_count: int
    region_label_by_cell: np.ndarray
    segmentation_domain_mask: np.ndarray
    ambiguity_mask: np.ndarray
    marker_flat_cell_index_by_candidate: np.ndarray
    region_cell_count_by_candidate: np.ndarray
    ambiguous_cell_count_by_candidate: np.ndarray
    region_touches_grid_edge_by_candidate: np.ndarray

    def __post_init__(self) -> None:
        _validate_identity_runtime(self.source_identity)
        if type(self.source_chm_config) is not CanopyHeightModelConfig:
            raise ValueError("source_chm_config must be a CanopyHeightModelConfig")
        if type(self.source_treetop_config) is not TreetopDetectionConfig:
            raise ValueError("source_treetop_config must be a TreetopDetectionConfig")
        if type(self.config) is not CanopySegmentationConfig:
            raise ValueError("config must be a CanopySegmentationConfig")
        canonical_chm_config = CanopyHeightModelConfig(
            **{
                name: getattr(self.source_chm_config, name)
                for name in CanopyHeightModelConfig.__dataclass_fields__
            }
        )
        canonical_treetop_config = TreetopDetectionConfig(
            **{
                name: getattr(self.source_treetop_config, name)
                for name in TreetopDetectionConfig.__dataclass_fields__
            }
        )
        canonical_config = CanopySegmentationConfig(
            **{
                name: getattr(self.config, name)
                for name in CanopySegmentationConfig.__dataclass_fields__
            }
        )
        if (
            not _matches_canonical_dataclass(self.source_chm_config, canonical_chm_config)
            or not _matches_canonical_dataclass(
                self.source_treetop_config,
                canonical_treetop_config,
            )
            or not _matches_canonical_dataclass(self.config, canonical_config)
        ):
            raise ValueError("source and segmentation configurations must be canonical")
        shape = _source_grid_shape(self.source_grid_shape)
        _source_origin(self.source_grid_origin_xy, shape, exact_float=True)

        for name in (
            "seeded_component_count",
            "unseeded_component_count",
            "assigned_cell_count",
            "unassigned_domain_cell_count",
            "ambiguous_cell_count",
        ):
            _non_negative_python_int(getattr(self, name), name)

        labels = _result_array(
            self.region_label_by_cell,
            np.int32,
            shape,
            "region_label_by_cell",
        )
        domain = _result_array(
            self.segmentation_domain_mask,
            np.bool_,
            shape,
            "segmentation_domain_mask",
        )
        ambiguity = _result_array(
            self.ambiguity_mask,
            np.bool_,
            shape,
            "ambiguity_mask",
        )
        marker_flat = _result_array(
            self.marker_flat_cell_index_by_candidate,
            np.int64,
            (None,),
            "marker_flat_cell_index_by_candidate",
        )
        candidate_count = len(marker_flat)
        region_counts = _result_array(
            self.region_cell_count_by_candidate,
            np.int64,
            (candidate_count,),
            "region_cell_count_by_candidate",
        )
        ambiguous_counts = _result_array(
            self.ambiguous_cell_count_by_candidate,
            np.int64,
            (candidate_count,),
            "ambiguous_cell_count_by_candidate",
        )
        edge_flags = _result_array(
            self.region_touches_grid_edge_by_candidate,
            np.bool_,
            (candidate_count,),
            "region_touches_grid_edge_by_candidate",
        )

        total = shape[0] * shape[1]
        if np.any(labels < 0) or np.any(labels > candidate_count):
            raise ValueError("region labels must be between zero and candidate_count")
        if np.any(labels[~domain] != 0):
            raise ValueError("cells outside the segmentation domain must have label zero")
        if np.any(ambiguity & (~domain | (labels == 0))):
            raise ValueError("ambiguity must be a subset of assigned domain cells")
        if np.any(marker_flat < 0) or np.any(marker_flat >= total):
            raise ValueError("candidate marker flat index is outside the source grid")
        if len(np.unique(marker_flat)) != candidate_count:
            raise ValueError("candidate marker flat indices must be unique")
        if candidate_count:
            expected_labels = np.arange(1, candidate_count + 1, dtype=np.int32)
            flat_labels = labels.ravel()
            if not np.array_equal(flat_labels[marker_flat], expected_labels):
                raise ValueError("every marker cell must retain its candidate label")
            if not np.all(domain.ravel()[marker_flat]):
                raise ValueError("every candidate marker must be in the segmentation domain")
            if np.any(region_counts <= 0):
                raise ValueError("every candidate-linked region must be nonempty")
        if self.seeded_component_count > candidate_count:
            raise ValueError("seeded_component_count must not exceed candidate_count")
        if candidate_count == 0 and self.seeded_component_count != 0:
            raise ValueError("zero candidates require zero seeded components")
        if candidate_count > 0 and self.seeded_component_count < 1:
            raise ValueError("positive candidates require at least one seeded component")

        raster_counts = np.bincount(labels.ravel(), minlength=candidate_count + 1)[1:]
        if not np.array_equal(raster_counts, region_counts):
            raise ValueError("region cell counts do not match the label raster")
        ambiguous_raster_counts = np.bincount(
            labels.ravel()[ambiguity.ravel()],
            minlength=candidate_count + 1,
        )[1:]
        if not np.array_equal(ambiguous_raster_counts, ambiguous_counts):
            raise ValueError("ambiguous cell counts do not match the ambiguity raster")
        assigned = int(np.count_nonzero(domain & (labels > 0)))
        unassigned = int(np.count_nonzero(domain & (labels == 0)))
        ambiguous = int(np.count_nonzero(ambiguity))
        if assigned != self.assigned_cell_count:
            raise ValueError("assigned_cell_count is inconsistent")
        if unassigned != self.unassigned_domain_cell_count:
            raise ValueError("unassigned_domain_cell_count is inconsistent")
        if assigned + unassigned != int(np.count_nonzero(domain)):
            raise ValueError("domain audit counts are inconsistent")
        if ambiguous != self.ambiguous_cell_count:
            raise ValueError("ambiguous_cell_count is inconsistent")
        if int(np.sum(region_counts, dtype=np.int64)) != assigned:
            raise ValueError("region counts do not sum to assigned_cell_count")
        if int(np.sum(ambiguous_counts, dtype=np.int64)) != ambiguous:
            raise ValueError("ambiguous counts do not sum to ambiguous_cell_count")
        if not np.array_equal(_edge_flags(labels, candidate_count), edge_flags):
            raise ValueError("region edge-touch flags are inconsistent")

    @property
    def candidate_count(self) -> int:
        """Return the number of candidate-linked region labels."""

        return len(self.marker_flat_cell_index_by_candidate)

    @property
    def domain_component_count(self) -> int:
        """Return the audited number of segmentation-domain components."""

        return self.seeded_component_count + self.unseeded_component_count


def _validate_chm_structure(canopy_model: CanopyHeightModelResult) -> tuple[int, int]:
    _validate_identity_runtime(canopy_model.source_identity)
    if type(canopy_model.config) is not CanopyHeightModelConfig:
        raise ValueError("config must be a CanopyHeightModelConfig")
    validated_config = CanopyHeightModelConfig(
        cell_size_m=canopy_model.config.cell_size_m,
        minimum_hag_m=canopy_model.config.minimum_hag_m,
        maximum_grid_cells=canopy_model.config.maximum_grid_cells,
    )
    if not _matches_canonical_dataclass(canopy_model.config, validated_config):
        raise ValueError("CHM config is not canonical")
    _non_negative_python_int(canopy_model.included_point_count, "included_point_count")
    _non_negative_python_int(canopy_model.excluded_point_count, "excluded_point_count")
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
    _source_grid_shape(shape)
    _source_origin(canopy_model.grid_origin_xy, shape, exact_float=True)
    return shape


def _validate_candidate_structure(
    candidates: TreetopCandidateResult,
) -> tuple[int, tuple[int, int]]:
    _validate_identity_runtime(candidates.source_identity)
    if type(candidates.source_chm_config) is not CanopyHeightModelConfig:
        raise ValueError("candidate source_chm_config has invalid type")
    if type(candidates.config) is not TreetopDetectionConfig:
        raise ValueError("candidate config has invalid type")
    validated_source_config = CanopyHeightModelConfig(
        cell_size_m=candidates.source_chm_config.cell_size_m,
        minimum_hag_m=candidates.source_chm_config.minimum_hag_m,
        maximum_grid_cells=candidates.source_chm_config.maximum_grid_cells,
    )
    validated_detection_config = TreetopDetectionConfig(
        **{
            name: getattr(candidates.config, name)
            for name in TreetopDetectionConfig.__dataclass_fields__
        }
    )
    if not _matches_canonical_dataclass(
        candidates.source_chm_config,
        validated_source_config,
    ) or not _matches_canonical_dataclass(
        candidates.config,
        validated_detection_config,
    ):
        raise ValueError("candidate configuration is not canonical")
    shape = _source_grid_shape(candidates.source_grid_shape)
    _source_origin(candidates.source_grid_origin_xy, shape, exact_float=True)
    plateau_total = _non_negative_python_int(
        candidates.local_maximum_plateau_count,
        "local_maximum_plateau_count",
    )
    rejected = _non_negative_python_int(
        candidates.evidence_rejected_count,
        "evidence_rejected_count",
    )
    suppressed = _non_negative_python_int(
        candidates.suppressed_candidate_count,
        "suppressed_candidate_count",
    )
    candidate_id = _result_array(candidates.candidate_id, np.int32, (None,), "candidate_id")
    count = len(candidate_id)
    specifications = (
        ("row_index", np.int64, (count,)),
        ("column_index", np.int64, (count,)),
        ("local_xy", np.float64, (count, 2)),
        ("peak_height_m", np.float32, (count,)),
        ("prominence_m", np.float32, (count,)),
        ("prominence_available", np.bool_, (count,)),
        ("plateau_cell_count", np.int32, (count,)),
        ("source_cell_point_count", np.int32, (count,)),
        ("source_support_flags", np.uint8, (count,)),
        ("source_peak_support_flags", np.uint8, (count,)),
        ("neighbour_occupied_cell_count", np.int32, (count,)),
        ("neighbourhood_occupied_fraction", np.float32, (count,)),
        ("neighbourhood_support_flags", np.uint8, (count,)),
        ("neighbourhood_support_fraction", np.float32, (count, 3)),
        ("edge_limited", np.bool_, (count,)),
    )
    for name, dtype, expected_shape in specifications:
        _result_array(getattr(candidates, name), dtype, expected_shape, name)
    if plateau_total != rejected + suppressed + count:
        raise ValueError("candidate audit counts are inconsistent")
    return count, shape


def _validation_capacity(total: int, candidate_count: int) -> int:
    return min(max(total, candidate_count), _VALIDATION_CHUNK_SIZE)


def _validation_temporary_estimate(total: int, candidate_count: int) -> int:
    capacity = _validation_capacity(total, candidate_count)
    return int(candidate_count) * 8 + int(capacity) * 16


def _allocate_validation_workspace(
    candidate_count: int,
    capacity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty(candidate_count, dtype=np.int64),
        np.empty(capacity, dtype=np.float64),
        np.empty(capacity, dtype=np.bool_),
        np.empty(capacity, dtype=np.bool_),
    )


def _validation_chunks(length: int, capacity: int):
    if length and capacity < 1:
        raise ValueError("positive validation length requires positive chunk capacity")
    for start in range(0, length, capacity or 1):
        end = min(start + capacity, length)
        yield start, end


def _validate_chm_content(
    canopy_model: CanopyHeightModelResult,
    capacity: int,
    boolean_a: np.ndarray,
    boolean_b: np.ndarray,
) -> None:
    height = canopy_model.canopy_height_m.ravel()
    occupied = canopy_model.occupied_mask.ravel()
    counts = canopy_model.point_count_by_cell.ravel()
    support = canopy_model.support_flags_by_cell.ravel()
    peak_support = canopy_model.peak_support_flags_by_cell.ravel()
    included_sum = 0
    for start, end in _validation_chunks(len(height), capacity):
        size = end - start
        first = boolean_a[:size]
        second = boolean_b[:size]
        byte_work = second.view(np.uint8)
        height_chunk = height[start:end]
        occupied_chunk = occupied[start:end]
        count_chunk = counts[start:end]
        support_chunk = support[start:end]
        peak_chunk = peak_support[start:end]

        np.less(count_chunk, 0, out=first)
        if np.any(first):
            raise ValueError("point_count_by_cell must be non-negative")
        np.greater(count_chunk, 0, out=first)
        if not np.array_equal(occupied_chunk, first):
            raise ValueError("occupied_mask must equal positive point counts")
        included_sum += int(np.sum(count_chunk, dtype=np.int64))

        np.isnan(height_chunk, out=first)
        np.logical_or(occupied_chunk, first, out=first)
        if not np.all(first):
            raise ValueError("empty CHM cells must contain NaN height")
        np.isfinite(height_chunk, out=first)
        np.logical_not(occupied_chunk, out=second)
        np.logical_or(first, second, out=first)
        if not np.all(first):
            raise ValueError("occupied CHM cells must contain finite height")
        np.less(height_chunk, 0.0, out=first)
        np.logical_and(first, occupied_chunk, out=first)
        if np.any(first):
            raise ValueError("occupied CHM cells must contain non-negative height")

        for value in (support_chunk, peak_chunk):
            np.bitwise_and(value, _UNKNOWN_SUPPORT_BITS, out=byte_work)
            if np.any(byte_work):
                raise ValueError("CHM support flags contain unknown bits")
            np.equal(value, 0, out=first)
            np.logical_not(occupied_chunk, out=second)
            if not np.array_equal(first, second):
                raise ValueError("CHM support flags are inconsistent with occupancy")
        np.bitwise_not(support_chunk, out=byte_work)
        np.bitwise_and(peak_chunk, byte_work, out=byte_work)
        if np.any(byte_work):
            raise ValueError("CHM peak support must be a subset of support")

    if included_sum != canopy_model.included_point_count:
        raise ValueError("point-count sum must equal included_point_count")
    if (
        canopy_model.included_point_count + canopy_model.excluded_point_count
        != canopy_model.source_identity.point_count
    ):
        raise ValueError("included and excluded counts must equal source point_count")


def _validate_candidate_content(
    candidates: TreetopCandidateResult,
    capacity: int,
    flat_workspace: np.ndarray,
    numeric_workspace: np.ndarray,
    boolean_a: np.ndarray,
    boolean_b: np.ndarray,
) -> None:
    count = len(candidates.candidate_id)
    shape = candidates.source_grid_shape
    origin = candidates.source_grid_origin_xy
    if count and shape == (0, 0):
        raise ValueError("positive candidates require a nonempty source grid")
    axis_extent = math.floor(
        np.float64(candidates.config.prominence_radius_m)
        / np.float64(candidates.source_chm_config.cell_size_m)
    )
    for start, end in _validation_chunks(count, capacity):
        size = end - start
        first = boolean_a[:size]
        second = boolean_b[:size]
        byte_work = second.view(np.uint8)
        numeric = numeric_workspace[:size]
        flat = flat_workspace[start:end]
        row = candidates.row_index[start:end]
        column = candidates.column_index[start:end]

        for expected, actual in enumerate(candidates.candidate_id[start:end], start):
            if int(actual) != expected:
                raise ValueError("candidate IDs must be contiguous and start at zero")
        np.less(row, 0, out=first)
        np.greater_equal(row, shape[0], out=second)
        np.logical_or(first, second, out=first)
        np.less(column, 0, out=second)
        np.logical_or(first, second, out=first)
        np.greater_equal(column, shape[1], out=second)
        np.logical_or(first, second, out=first)
        if np.any(first):
            raise ValueError("candidate row or column is outside source_grid_shape")
        np.multiply(row, shape[1], out=flat)
        np.add(flat, column, out=flat)

        assert origin is not None
        for coordinate, indices, origin_value in (
            (candidates.local_xy[start:end, 0], column, origin[0]),
            (candidates.local_xy[start:end, 1], row, origin[1]),
        ):
            np.copyto(numeric, indices, casting="unsafe")
            np.add(numeric, 0.5, out=numeric)
            np.multiply(numeric, candidates.source_chm_config.cell_size_m, out=numeric)
            np.add(numeric, origin_value, out=numeric)
            np.equal(coordinate, numeric, out=first)
            if not np.all(first):
                raise ValueError("candidate local_xy is inconsistent with copied grid geometry")

        peak = candidates.peak_height_m[start:end]
        np.isfinite(peak, out=first)
        np.greater_equal(peak, candidates.config.minimum_peak_hag_m, out=second)
        np.logical_and(first, second, out=first)
        if not np.all(first):
            raise ValueError("candidate peak heights are invalid")
        prominence = candidates.prominence_m[start:end]
        np.isfinite(prominence, out=first)
        np.greater_equal(prominence, 0.0, out=second)
        np.logical_and(first, second, out=first)
        np.equal(prominence, 0.0, out=second)
        np.logical_or(candidates.prominence_available[start:end], second, out=second)
        np.logical_and(first, second, out=first)
        if not np.all(first):
            raise ValueError("candidate prominence fields are invalid")
        for value, positive in (
            (candidates.plateau_cell_count[start:end], True),
            (candidates.source_cell_point_count[start:end], True),
            (candidates.neighbour_occupied_cell_count[start:end], False),
        ):
            comparison = np.greater if positive else np.greater_equal
            comparison(value, 0, out=first)
            if not np.all(first):
                raise ValueError("candidate count fields are invalid")

        for value, allow_zero in (
            (candidates.source_support_flags[start:end], False),
            (candidates.source_peak_support_flags[start:end], False),
            (candidates.neighbourhood_support_flags[start:end], True),
        ):
            np.bitwise_and(value, _UNKNOWN_SUPPORT_BITS, out=byte_work)
            if np.any(byte_work):
                raise ValueError("candidate support fields contain unknown bits")
            if not allow_zero and np.any(value == 0):
                raise ValueError("candidate support fields must be nonzero")
        np.bitwise_not(candidates.source_support_flags[start:end], out=byte_work)
        np.bitwise_and(
            candidates.source_peak_support_flags[start:end],
            byte_work,
            out=byte_work,
        )
        if np.any(byte_work):
            raise ValueError("candidate peak support must be a subset of source support")

        for value in (
            candidates.neighbourhood_occupied_fraction[start:end],
            candidates.neighbourhood_support_fraction[start:end, 0],
            candidates.neighbourhood_support_fraction[start:end, 1],
            candidates.neighbourhood_support_fraction[start:end, 2],
        ):
            np.isfinite(value, out=first)
            np.greater_equal(value, 0.0, out=second)
            np.logical_and(first, second, out=first)
            np.less_equal(value, 1.0, out=second)
            np.logical_and(first, second, out=first)
            if not np.all(first):
                raise ValueError("candidate neighbourhood fractions are invalid")

        for offset in range(size):
            index = start + offset
            neighbour_count = int(candidates.neighbour_occupied_cell_count[index])
            occupied_fraction = float(candidates.neighbourhood_occupied_fraction[index])
            flags = int(candidates.neighbourhood_support_flags[index])
            fractions = candidates.neighbourhood_support_fraction[index]
            if neighbour_count == 0:
                if occupied_fraction != 0.0 or flags != 0 or np.any(fractions != 0.0):
                    raise ValueError("zero-neighbour candidate evidence is inconsistent")
            elif occupied_fraction <= 0.0 or flags == 0 or not np.any(fractions > 0.0):
                raise ValueError("positive-neighbour candidate evidence is inconsistent")
            for support_column, support_bit in enumerate((1, 2, 4)):
                if bool(fractions[support_column] > 0.0) != bool(flags & support_bit):
                    raise ValueError("candidate neighbourhood support evidence is inconsistent")
            expected_edge = axis_extent > 0 and (
                int(row[offset]) < axis_extent
                or int(row[offset]) >= shape[0] - axis_extent
                or int(column[offset]) < axis_extent
                or int(column[offset]) >= shape[1] - axis_extent
            )
            if bool(candidates.edge_limited[index]) != expected_edge:
                raise ValueError("candidate edge_limited is inconsistent with source geometry")

    flat_workspace.sort(kind="quicksort")
    for start, end in _validation_chunks(max(count - 1, 0), capacity):
        size = end - start
        np.equal(
            flat_workspace[start:end],
            flat_workspace[start + 1 : end + 1],
            out=boolean_a[:size],
        )
        if np.any(boolean_a[:size]):
            raise ValueError("candidate row and column pairs must be unique")


def _pair_error(
    message: str,
    shape: tuple[int, int],
    candidate_count: int,
) -> CanopySegmentationError:
    return CanopySegmentationError(
        CanopySegmentationErrorCode.INPUT_PAIR_MISMATCH,
        message,
        row_count=shape[0],
        column_count=shape[1],
        total_cell_count=shape[0] * shape[1],
        candidate_count=candidate_count,
    )


def _validate_pair_content(
    canopy_model: CanopyHeightModelResult,
    candidates: TreetopCandidateResult,
    shape: tuple[int, int],
    candidate_count: int,
    capacity: int,
    flat_workspace: np.ndarray,
    numeric_workspace: np.ndarray,
    boolean_workspace: np.ndarray,
) -> np.ndarray:
    if canopy_model.source_identity != candidates.source_identity:
        raise _pair_error(
            "CHM and candidates have different source identities", shape, candidate_count
        )
    if canopy_model.config != candidates.source_chm_config:
        raise _pair_error(
            "CHM and candidates have different CHM configurations", shape, candidate_count
        )
    if canopy_model.grid_origin_xy != candidates.source_grid_origin_xy:
        raise _pair_error("CHM and candidates have different grid origins", shape, candidate_count)
    if shape != candidates.source_grid_shape:
        raise _pair_error("CHM and candidates have different grid shapes", shape, candidate_count)
    if canopy_model.config.cell_size_m != candidates.source_chm_config.cell_size_m:
        raise _pair_error("CHM and candidates have different cell sizes", shape, candidate_count)
    if not candidate_count:
        return flat_workspace
    origin = canopy_model.grid_origin_xy
    assert origin is not None
    height = canopy_model.canopy_height_m.ravel()
    occupied = canopy_model.occupied_mask.ravel()
    counts = canopy_model.point_count_by_cell.ravel()
    support = canopy_model.support_flags_by_cell.ravel()
    peak_support = canopy_model.peak_support_flags_by_cell.ravel()
    for start, end in _validation_chunks(candidate_count, capacity):
        size = end - start
        flat = flat_workspace[start:end]
        numeric = numeric_workspace[:size]
        comparison = boolean_workspace[:size]
        row = candidates.row_index[start:end]
        column = candidates.column_index[start:end]
        np.multiply(row, shape[1], out=flat)
        np.add(flat, column, out=flat)
        np.take(occupied, flat, out=comparison)
        if not np.all(comparison):
            raise _pair_error(
                "a candidate cell is not occupied in the CHM",
                shape,
                candidate_count,
            )
        for source, expected in (
            (height, candidates.peak_height_m[start:end]),
            (counts, candidates.source_cell_point_count[start:end]),
            (support, candidates.source_support_flags[start:end]),
            (peak_support, candidates.source_peak_support_flags[start:end]),
        ):
            sampled = np.take(source, flat)
            np.equal(expected, sampled, out=comparison)
            if not np.all(comparison):
                raise _pair_error(
                    "candidate source-cell values do not match the supplied CHM",
                    shape,
                    candidate_count,
                )
            del sampled
        for coordinate, indices, origin_value in (
            (candidates.local_xy[start:end, 0], column, origin[0]),
            (candidates.local_xy[start:end, 1], row, origin[1]),
        ):
            np.copyto(numeric, indices, casting="unsafe")
            np.add(numeric, 0.5, out=numeric)
            np.multiply(numeric, canopy_model.config.cell_size_m, out=numeric)
            np.add(numeric, origin_value, out=numeric)
            np.equal(coordinate, numeric, out=comparison)
            if not np.all(comparison):
                raise _pair_error(
                    "candidate local XY does not match the supplied CHM",
                    shape,
                    candidate_count,
                )
    return flat_workspace


def _resource_error(
    message: str,
    *,
    shape: tuple[int, int],
    domain_cell_count: int | None = None,
    candidate_count: int | None = None,
    component_count: int | None = None,
    estimated_temporary_bytes: int | None = None,
    queue_entry_count: int | None = None,
    edge_evaluation_count: int | None = None,
) -> CanopySegmentationError:
    return CanopySegmentationError(
        CanopySegmentationErrorCode.RESOURCE_LIMIT,
        message,
        row_count=shape[0],
        column_count=shape[1],
        total_cell_count=shape[0] * shape[1],
        domain_cell_count=domain_cell_count,
        candidate_count=candidate_count,
        component_count=component_count,
        estimated_temporary_bytes=estimated_temporary_bytes,
        queue_entry_count=queue_entry_count,
        edge_evaluation_count=edge_evaluation_count,
    )


def _preflight_temporary(
    phase: str,
    estimated_bytes: int,
    config: CanopySegmentationConfig,
    shape: tuple[int, int],
    domain_count: int,
    candidate_count: int,
) -> None:
    if estimated_bytes > config.maximum_temporary_bytes:
        raise _resource_error(
            f"{phase} arrays exceed maximum_temporary_bytes",
            shape=shape,
            domain_cell_count=domain_count,
            candidate_count=candidate_count,
            estimated_temporary_bytes=estimated_bytes,
        )


def _component_temporary_estimate(
    total: int,
    domain_count: int,
    component_capacity: int,
    candidate_count: int,
) -> int:
    return (
        int(total) * 10
        + int(domain_count) * 4
        + int(component_capacity) * 8
        + int(candidate_count) * 8
    )


def _domain_temporary_estimate(total: int, candidate_count: int) -> int:
    return int(total) * 2 + int(candidate_count) * 9


def _ambiguity_temporary_estimate(
    total: int,
    active_count: int,
    candidate_count: int,
) -> int:
    return int(total) * 19 + int(active_count) * 4 + int(candidate_count) * 8


def _propagation_temporary_estimate(
    total: int,
    active_count: int,
    candidate_count: int,
) -> int:
    completion_capacity = min(int(total), _VALIDATION_CHUNK_SIZE)
    return int(total) * 24 + int(active_count) * 4 + int(candidate_count) * 8 + completion_capacity


def _freeze_temporary_estimate(total: int, candidate_count: int) -> int:
    # Includes mutable plus detached public arrays and conservative constructor
    # validation workspace; this is a bounded allocation estimate, not RSS.
    return int(total) * 20 + int(candidate_count) * 96 + 8


@dataclass
class _EdgeBudget:
    maximum: int
    shape: tuple[int, int]
    domain_count: int
    candidate_count: int
    value: int = 0

    def consume(self, count: int = 1) -> None:
        increment = int(count)
        if increment < 0:
            raise ValueError("edge-evaluation increment must be non-negative")
        next_value = self.value + increment
        if next_value > self.maximum:
            raise _resource_error(
                "cumulative edge evaluations exceed maximum_edge_evaluations",
                shape=self.shape,
                domain_cell_count=self.domain_count,
                candidate_count=self.candidate_count,
                edge_evaluation_count=next_value,
            )
        self.value = next_value


def _preflight_edge_budget(
    domain_count: int,
    config: CanopySegmentationConfig,
    shape: tuple[int, int],
    candidate_count: int,
) -> _EdgeBudget:
    factor = 12 if config.connectivity == 4 else 24
    estimate = int(factor) * int(domain_count)
    if estimate > config.maximum_edge_evaluations:
        raise _resource_error(
            "estimated whole-run edge evaluations exceed maximum_edge_evaluations",
            shape=shape,
            domain_cell_count=domain_count,
            candidate_count=candidate_count,
            edge_evaluation_count=estimate,
        )
    return _EdgeBudget(
        config.maximum_edge_evaluations,
        shape,
        domain_count,
        candidate_count,
    )


def _allocate_domain(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    return np.empty(shape, dtype=np.bool_), np.empty(shape, dtype=np.bool_)


def _allocate_owner(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.int32)


def _allocate_component_labels(shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, -1, dtype=np.int32)


def _allocate_component_stack(capacity: int) -> np.ndarray:
    return np.empty(capacity, dtype=np.int32)


def _allocate_component_metadata(capacity: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(capacity, dtype=np.int32), np.zeros(capacity, dtype=np.int32)


def _allocate_score_arrays(
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.full(shape, -np.inf, dtype=np.float32),
        np.full(shape, _INT32_MAX, dtype=np.int32),
        np.full(shape, _INT32_MAX, dtype=np.int32),
        np.zeros(shape, dtype=np.bool_),
    )


def _allocate_heap_cells(capacity: int) -> np.ndarray:
    return np.empty(capacity, dtype=np.int32)


def _allocate_heap_positions(total_cells: int) -> np.ndarray:
    return np.full(total_cells, -1, dtype=np.int32)


def _allocate_ambiguity_queue(capacity: int) -> np.ndarray:
    return np.empty(capacity, dtype=np.int32)


def _freeze_array(value: Any, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if result.size and np.issubdtype(result.dtype, np.floating):
        result[result == 0] = 0
    result.setflags(write=False)
    return result


def _neighbour_offsets(connectivity: int) -> tuple[tuple[int, int], ...]:
    return _OFFSETS_4 if connectivity == 4 else _OFFSETS_8


def _forward_offsets(connectivity: int) -> tuple[tuple[int, int], ...]:
    return _FORWARD_OFFSETS_4 if connectivity == 4 else _FORWARD_OFFSETS_8


def _iter_neighbours(
    flat: int,
    shape: tuple[int, int],
    offsets: tuple[tuple[int, int], ...],
):
    row, column = divmod(flat, shape[1])
    for dr, dc in offsets:
        other_row = row + dr
        other_column = column + dc
        if 0 <= other_row < shape[0] and 0 <= other_column < shape[1]:
            yield other_row * shape[1] + other_column, dr != 0 and dc != 0


def _label_components(
    domain: np.ndarray,
    owner: np.ndarray,
    config: CanopySegmentationConfig,
    budget: _EdgeBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    shape = domain.shape
    total = shape[0] * shape[1]
    domain_count = int(np.count_nonzero(domain))
    component_capacity = min(domain_count, config.maximum_component_count)
    component_labels = _allocate_component_labels(shape)
    stack = _allocate_component_stack(domain_count)
    marker_counts, sole_labels = _allocate_component_metadata(component_capacity)
    domain_flat = domain.ravel()
    owner_flat = owner.ravel()
    component_flat = component_labels.ravel()
    offsets = _neighbour_offsets(config.connectivity)
    component_count = 0
    for start in range(total):
        if not domain_flat[start] or component_flat[start] >= 0:
            continue
        if component_count >= config.maximum_component_count:
            raise _resource_error(
                "domain component count exceeds maximum_component_count",
                shape=shape,
                domain_cell_count=domain_count,
                component_count=component_count + 1,
            )
        top = 1
        stack[0] = start
        component_flat[start] = component_count
        marker_count = 0
        sole_label = 0
        while top:
            top -= 1
            cell = int(stack[top])
            marker_label = int(owner_flat[cell])
            if marker_label > 0:
                marker_count += 1
                sole_label = marker_label
            for other, _diagonal in _iter_neighbours(cell, shape, offsets):
                if not domain_flat[other]:
                    continue
                if other > cell:
                    budget.consume()
                if component_flat[other] < 0:
                    component_flat[other] = component_count
                    stack[top] = other
                    top += 1
        marker_counts[component_count] = marker_count
        sole_labels[component_count] = sole_label if marker_count == 1 else 0
        component_count += 1
    seeded = int(np.count_nonzero(marker_counts[:component_count]))
    unseeded = component_count - seeded
    return component_labels, marker_counts, sole_labels, component_count, seeded, unseeded


def _compare_weighted_distance(
    left_cardinal: int,
    left_diagonal: int,
    right_cardinal: int,
    right_diagonal: int,
) -> int:
    """Return -1, 0, or 1 when the left exact weighted distance is shorter/equal/longer."""

    p = int(left_cardinal) - int(right_cardinal)
    q = int(left_diagonal) - int(right_diagonal)
    if p == 0:
        return (q > 0) - (q < 0)
    if q == 0:
        return (p > 0) - (p < 0)
    if (p > 0) == (q > 0):
        return 1 if p > 0 else -1
    p_squared = p * p
    twice_q_squared = 2 * q * q
    if p > 0:
        return 1 if p_squared > twice_q_squared else -1
    return 1 if twice_q_squared > p_squared else -1


def _cell_precedes(
    left: int,
    right: int,
    bottleneck: np.ndarray,
    cardinal: np.ndarray,
    diagonal: np.ndarray,
    owner: np.ndarray,
) -> bool:
    left_height = bottleneck[left]
    right_height = bottleneck[right]
    if left_height != right_height:
        return bool(left_height > right_height)
    distance_order = _compare_weighted_distance(
        int(cardinal[left]),
        int(diagonal[left]),
        int(cardinal[right]),
        int(diagonal[right]),
    )
    if distance_order:
        return distance_order < 0
    if owner[left] != owner[right]:
        return bool(owner[left] < owner[right])
    return left < right


class _IndexedMaxHeap:
    def __init__(
        self,
        capacity: int,
        total_cells: int,
        bottleneck: np.ndarray,
        cardinal: np.ndarray,
        diagonal: np.ndarray,
        owner: np.ndarray,
        config: CanopySegmentationConfig,
        shape: tuple[int, int],
        domain_count: int,
        candidate_count: int,
    ) -> None:
        if capacity > config.maximum_queue_entries:
            raise _resource_error(
                "active heap capacity exceeds maximum_queue_entries",
                shape=shape,
                domain_cell_count=domain_count,
                candidate_count=candidate_count,
                queue_entry_count=capacity,
            )
        self.cells = _allocate_heap_cells(capacity)
        self.positions = _allocate_heap_positions(total_cells)
        self.size = 0
        self._bottleneck = bottleneck
        self._cardinal = cardinal
        self._diagonal = diagonal
        self._owner = owner
        self._config = config
        self._shape = shape
        self._domain_count = domain_count
        self._candidate_count = candidate_count

    def __bool__(self) -> bool:
        return self.size > 0

    def _precedes(self, left: int, right: int) -> bool:
        return _cell_precedes(
            left,
            right,
            self._bottleneck,
            self._cardinal,
            self._diagonal,
            self._owner,
        )

    def _swap(self, left_position: int, right_position: int) -> None:
        left = int(self.cells[left_position])
        right = int(self.cells[right_position])
        self.cells[left_position] = right
        self.cells[right_position] = left
        self.positions[left] = right_position
        self.positions[right] = left_position

    def _sift_up(self, position: int) -> int:
        while position:
            parent = (position - 1) // 2
            if not self._precedes(int(self.cells[position]), int(self.cells[parent])):
                break
            self._swap(position, parent)
            position = parent
        return position

    def _sift_down(self, position: int) -> None:
        while True:
            left = position * 2 + 1
            if left >= self.size:
                return
            right = left + 1
            best = left
            if right < self.size and self._precedes(
                int(self.cells[right]),
                int(self.cells[left]),
            ):
                best = right
            if not self._precedes(int(self.cells[best]), int(self.cells[position])):
                return
            self._swap(position, best)
            position = best

    def push_or_update(self, cell: int) -> None:
        position = int(self.positions[cell])
        if position >= 0:
            position = self._sift_up(position)
            self._sift_down(position)
            return
        if self.size >= len(self.cells):
            raise _resource_error(
                "indexed heap capacity was exceeded",
                shape=self._shape,
                domain_cell_count=self._domain_count,
                candidate_count=self._candidate_count,
                queue_entry_count=self.size + 1,
            )
        position = self.size
        self.size += 1
        self.cells[position] = cell
        self.positions[cell] = position
        self._sift_up(position)

    def pop(self) -> int:
        if not self.size:
            raise CanopySegmentationError(
                CanopySegmentationErrorCode.INTERNAL_INVARIANT,
                "cannot pop an empty indexed heap",
            )
        root = int(self.cells[0])
        self.size -= 1
        self.positions[root] = -1
        if self.size:
            replacement = int(self.cells[self.size])
            self.cells[0] = replacement
            self.positions[replacement] = 0
            self._sift_down(0)
        return root


def _candidate_score_better(
    candidate_bottleneck: np.float32,
    candidate_cardinal: int,
    candidate_diagonal: int,
    candidate_owner: int,
    current_bottleneck: np.float32,
    current_cardinal: int,
    current_diagonal: int,
    current_owner: int,
) -> bool:
    if current_owner == 0:
        return True
    if candidate_bottleneck != current_bottleneck:
        return bool(candidate_bottleneck > current_bottleneck)
    distance_order = _compare_weighted_distance(
        candidate_cardinal,
        candidate_diagonal,
        current_cardinal,
        current_diagonal,
    )
    if distance_order:
        return distance_order < 0
    return candidate_owner < current_owner


def _allocate_propagation_completion_workspace(capacity: int) -> np.ndarray:
    return np.empty(capacity, dtype=np.bool_)


def _validate_propagation_completion(
    active: np.ndarray,
    settled: np.ndarray,
    owner: np.ndarray,
    *,
    chunk_size: int = _VALIDATION_CHUNK_SIZE,
) -> None:
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive Python integer")
    total = active.size
    capacity = min(total, chunk_size)
    workspace = _allocate_propagation_completion_workspace(capacity)
    active_flat = active.ravel()
    settled_flat = settled.ravel()
    owner_flat = owner.ravel()
    for start, end in _validation_chunks(total, capacity):
        work = workspace[: end - start]
        active_chunk = active_flat[start:end]
        np.logical_not(settled_flat[start:end], out=work)
        np.logical_and(active_chunk, work, out=work)
        if np.any(work):
            raise CanopySegmentationError(
                CanopySegmentationErrorCode.INTERNAL_INVARIANT,
                "multi-marker component propagation left an active cell unassigned",
            )
        np.equal(owner_flat[start:end], 0, out=work)
        np.logical_and(active_chunk, work, out=work)
        if np.any(work):
            raise CanopySegmentationError(
                CanopySegmentationErrorCode.INTERNAL_INVARIANT,
                "multi-marker component propagation left an active cell unassigned",
            )


def _propagate(
    height: np.ndarray,
    active: np.ndarray,
    owner: np.ndarray,
    marker_flat: np.ndarray,
    config: CanopySegmentationConfig,
    budget: _EdgeBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = height.shape
    total = shape[0] * shape[1]
    active_count = int(np.count_nonzero(active))
    bottleneck, cardinal, diagonal, settled = _allocate_score_arrays(shape)
    bottleneck_flat = bottleneck.ravel()
    cardinal_flat = cardinal.ravel()
    diagonal_flat = diagonal.ravel()
    settled_flat = settled.ravel()
    owner_flat = owner.ravel()
    active_flat = active.ravel()
    height_flat = height.ravel()
    heap = _IndexedMaxHeap(
        active_count,
        total,
        bottleneck_flat,
        cardinal_flat,
        diagonal_flat,
        owner_flat,
        config,
        shape,
        budget.domain_count,
        budget.candidate_count,
    )
    for flat_value in marker_flat:
        flat = int(flat_value)
        if not active_flat[flat]:
            continue
        bottleneck_flat[flat] = height_flat[flat]
        cardinal_flat[flat] = 0
        diagonal_flat[flat] = 0
        heap.push_or_update(flat)
    offsets = _neighbour_offsets(config.connectivity)
    while heap:
        cell = heap.pop()
        if settled_flat[cell]:
            raise CanopySegmentationError(
                CanopySegmentationErrorCode.INTERNAL_INVARIANT,
                "an active cell was extracted from the heap twice",
            )
        settled_flat[cell] = True
        for other, is_diagonal in _iter_neighbours(cell, shape, offsets):
            if not active_flat[other]:
                continue
            budget.consume()
            candidate_cardinal = int(cardinal_flat[cell]) + (0 if is_diagonal else 1)
            candidate_diagonal = int(diagonal_flat[cell]) + (1 if is_diagonal else 0)
            if (
                candidate_cardinal > total - 1
                or candidate_diagonal > total - 1
                or candidate_cardinal > _INT32_MAX
                or candidate_diagonal > _INT32_MAX
            ):
                raise _resource_error(
                    "winning path step count cannot fit int32",
                    shape=shape,
                    domain_cell_count=budget.domain_count,
                    candidate_count=budget.candidate_count,
                )
            candidate_bottleneck = np.float32(
                min(float(bottleneck_flat[cell]), float(height_flat[other]))
            )
            candidate_owner = int(owner_flat[cell])
            better = _candidate_score_better(
                candidate_bottleneck,
                candidate_cardinal,
                candidate_diagonal,
                candidate_owner,
                bottleneck_flat[other],
                int(cardinal_flat[other]),
                int(diagonal_flat[other]),
                int(owner_flat[other]),
            )
            if not better:
                continue
            if settled_flat[other]:
                raise CanopySegmentationError(
                    CanopySegmentationErrorCode.INTERNAL_INVARIANT,
                    "a strictly better score reached an already settled cell",
                )
            bottleneck_flat[other] = candidate_bottleneck
            cardinal_flat[other] = candidate_cardinal
            diagonal_flat[other] = candidate_diagonal
            owner_flat[other] = candidate_owner
            heap.push_or_update(other)
    _validate_propagation_completion(active, settled, owner)
    return bottleneck, cardinal, diagonal, settled


def _is_optimal_predecessor(
    source: int,
    target: int,
    is_diagonal: bool,
    height: np.ndarray,
    bottleneck: np.ndarray,
    cardinal: np.ndarray,
    diagonal: np.ndarray,
) -> bool:
    extended_bottleneck = np.float32(min(float(bottleneck[source]), float(height[target])))
    if extended_bottleneck != bottleneck[target]:
        return False
    extended_cardinal = int(cardinal[source]) + (0 if is_diagonal else 1)
    extended_diagonal = int(diagonal[source]) + (1 if is_diagonal else 0)
    return (
        _compare_weighted_distance(
            extended_cardinal,
            extended_diagonal,
            int(cardinal[target]),
            int(diagonal[target]),
        )
        == 0
    )


def _ambiguity_fixed_point(
    height: np.ndarray,
    active: np.ndarray,
    owner: np.ndarray,
    bottleneck: np.ndarray,
    cardinal: np.ndarray,
    diagonal: np.ndarray,
    config: CanopySegmentationConfig,
    budget: _EdgeBudget,
) -> np.ndarray:
    shape = height.shape
    total = shape[0] * shape[1]
    active_count = int(np.count_nonzero(active))
    if active_count > config.maximum_queue_entries:
        raise _resource_error(
            "ambiguity queue capacity exceeds maximum_queue_entries",
            shape=shape,
            domain_cell_count=budget.domain_count,
            candidate_count=budget.candidate_count,
            queue_entry_count=active_count,
        )
    ambiguity = np.zeros(shape, dtype=np.bool_)
    active_flat = active.ravel()
    ambiguity_flat = ambiguity.ravel()
    owner_flat = owner.ravel()
    height_flat = height.ravel()
    bottleneck_flat = bottleneck.ravel()
    cardinal_flat = cardinal.ravel()
    diagonal_flat = diagonal.ravel()
    forward = _forward_offsets(config.connectivity)
    for cell in range(total):
        if not active_flat[cell]:
            continue
        for other, is_diagonal in _iter_neighbours(cell, shape, forward):
            if not active_flat[other]:
                continue
            budget.consume()
            if owner_flat[cell] != owner_flat[other]:
                if _is_optimal_predecessor(
                    cell,
                    other,
                    is_diagonal,
                    height_flat,
                    bottleneck_flat,
                    cardinal_flat,
                    diagonal_flat,
                ):
                    ambiguity_flat[other] = True
                if _is_optimal_predecessor(
                    other,
                    cell,
                    is_diagonal,
                    height_flat,
                    bottleneck_flat,
                    cardinal_flat,
                    diagonal_flat,
                ):
                    ambiguity_flat[cell] = True
    queue = _allocate_ambiguity_queue(active_count)
    tail = 0
    for cell in range(total):
        if ambiguity_flat[cell]:
            queue[tail] = cell
            tail += 1
    head = 0
    offsets = _neighbour_offsets(config.connectivity)
    while head < tail:
        cell = int(queue[head])
        head += 1
        for other, is_diagonal in _iter_neighbours(cell, shape, offsets):
            if not active_flat[other]:
                continue
            budget.consume()
            if ambiguity_flat[other] or not _is_optimal_predecessor(
                cell,
                other,
                is_diagonal,
                height_flat,
                bottleneck_flat,
                cardinal_flat,
                diagonal_flat,
            ):
                continue
            ambiguity_flat[other] = True
            if tail >= len(queue):
                raise _resource_error(
                    "ambiguity queue capacity was exceeded",
                    shape=shape,
                    domain_cell_count=budget.domain_count,
                    candidate_count=budget.candidate_count,
                    queue_entry_count=tail + 1,
                )
            queue[tail] = other
            tail += 1
    return ambiguity


def _metadata(
    labels: np.ndarray,
    domain: np.ndarray,
    ambiguity: np.ndarray,
    marker_flat: np.ndarray,
    candidate_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    labels_flat = labels.ravel()
    region_counts = np.bincount(labels_flat, minlength=candidate_count + 1)[1:].astype(
        np.int64,
        copy=False,
    )
    ambiguous_counts = np.bincount(
        labels_flat[ambiguity.ravel()],
        minlength=candidate_count + 1,
    )[1:].astype(np.int64, copy=False)
    edge_flags = _edge_flags(labels, candidate_count)
    assigned = int(np.count_nonzero(domain & (labels > 0)))
    unassigned = int(np.count_nonzero(domain & (labels == 0)))
    ambiguous = int(np.count_nonzero(ambiguity))
    return region_counts, ambiguous_counts, edge_flags, assigned, unassigned, ambiguous


def segment_canopy_regions(
    canopy_model: CanopyHeightModelResult,
    treetop_candidates: TreetopCandidateResult,
    config: CanopySegmentationConfig | None = None,
) -> CanopySegmentationResult:
    """Partition CHM domain cells into deterministic candidate-linked canopy regions."""

    if type(canopy_model) is not CanopyHeightModelResult:
        raise TypeError("canopy_model must be a CanopyHeightModelResult")
    if type(treetop_candidates) is not TreetopCandidateResult:
        raise TypeError("treetop_candidates must be a TreetopCandidateResult")
    if config is None:
        effective_config = CanopySegmentationConfig()
    elif type(config) is CanopySegmentationConfig:
        canonical_config = CanopySegmentationConfig(
            **{
                name: getattr(config, name)
                for name in CanopySegmentationConfig.__dataclass_fields__
            }
        )
        if not _matches_canonical_dataclass(config, canonical_config):
            raise ValueError("config must be canonical")
        effective_config = canonical_config
    else:
        raise TypeError("config must be a CanopySegmentationConfig or None")

    shape = (0, 0)
    total = 0
    candidate_count = 0
    domain_count = 0
    component_count = 0
    estimated_peak = 0
    try:
        shape = _validate_chm_structure(canopy_model)
        total = shape[0] * shape[1]
        candidate_count, candidate_shape = _validate_candidate_structure(treetop_candidates)
        if total > effective_config.maximum_grid_cells:
            raise _resource_error(
                "source grid exceeds maximum_grid_cells",
                shape=shape,
                candidate_count=candidate_count,
            )
        if total > _INT32_MAX:
            raise _resource_error(
                "source grid cell IDs cannot fit int32",
                shape=shape,
                candidate_count=candidate_count,
            )
        if candidate_count > effective_config.maximum_candidate_count:
            raise _resource_error(
                "candidate count exceeds maximum_candidate_count",
                shape=shape,
                candidate_count=candidate_count,
            )
        if candidate_count > _INT32_MAX - 1:
            raise _resource_error(
                "positive candidate labels cannot fit int32",
                shape=shape,
                candidate_count=candidate_count,
            )

        validation_capacity = _validation_capacity(total, candidate_count)
        validation_estimate = _validation_temporary_estimate(total, candidate_count)
        estimated_peak = max(estimated_peak, validation_estimate)
        _preflight_temporary(
            "validation",
            validation_estimate,
            effective_config,
            shape,
            0,
            candidate_count,
        )
        marker_flat, validation_numeric, validation_a, validation_b = (
            _allocate_validation_workspace(candidate_count, validation_capacity)
        )
        _validate_chm_content(
            canopy_model,
            validation_capacity,
            validation_a,
            validation_b,
        )
        _validate_candidate_content(
            treetop_candidates,
            validation_capacity,
            marker_flat,
            validation_numeric,
            validation_a,
            validation_b,
        )
        if candidate_shape != shape:
            raise _pair_error(
                "CHM and candidates have different grid shapes",
                shape,
                candidate_count,
            )
        marker_flat = _validate_pair_content(
            canopy_model,
            treetop_candidates,
            shape,
            candidate_count,
            validation_capacity,
            marker_flat,
            validation_numeric,
            validation_a,
        )
        del validation_numeric, validation_a, validation_b

        domain_estimate = _domain_temporary_estimate(total, candidate_count)
        estimated_peak = max(estimated_peak, domain_estimate)
        _preflight_temporary(
            "domain construction",
            domain_estimate,
            effective_config,
            shape,
            0,
            candidate_count,
        )
        domain, domain_workspace = _allocate_domain(shape)
        np.isfinite(canopy_model.canopy_height_m, out=domain)
        np.logical_and(domain, canopy_model.occupied_mask, out=domain)
        threshold = (
            canopy_model.config.minimum_hag_m
            if effective_config.minimum_segmentation_hag_m is None
            else effective_config.minimum_segmentation_hag_m
        )
        np.greater_equal(
            canopy_model.canopy_height_m,
            np.float64(threshold),
            out=domain_workspace,
        )
        np.logical_and(domain, domain_workspace, out=domain)
        del domain_workspace
        domain_count = int(np.count_nonzero(domain))
        if candidate_count and not np.all(domain.ravel()[marker_flat]):
            if effective_config.minimum_segmentation_hag_m is not None:
                raise ValueError("minimum_segmentation_hag_m excludes a candidate marker cell")
            raise CanopySegmentationError(
                CanopySegmentationErrorCode.INTERNAL_INVARIANT,
                "a default-domain candidate marker is unexpectedly excluded",
                row_count=shape[0],
                column_count=shape[1],
                total_cell_count=total,
                domain_cell_count=domain_count,
                candidate_count=candidate_count,
            )
        budget = _preflight_edge_budget(
            domain_count,
            effective_config,
            shape,
            candidate_count,
        )
        component_capacity = min(domain_count, effective_config.maximum_component_count)
        component_estimate = _component_temporary_estimate(
            total,
            domain_count,
            component_capacity,
            candidate_count,
        )
        estimated_peak = max(estimated_peak, component_estimate)
        _preflight_temporary(
            "component",
            component_estimate,
            effective_config,
            shape,
            domain_count,
            candidate_count,
        )
        owner = _allocate_owner(shape)
        owner_flat = owner.ravel()
        for candidate_id, flat in enumerate(marker_flat):
            owner_flat[int(flat)] = candidate_id + 1
        (
            component_labels,
            marker_counts,
            sole_labels,
            component_count,
            seeded_components,
            unseeded_components,
        ) = _label_components(domain, owner, effective_config, budget)
        active = np.zeros(shape, dtype=np.bool_)
        component_flat = component_labels.ravel()
        active_flat = active.ravel()
        domain_flat = domain.ravel()
        for flat in range(total):
            if not domain_flat[flat]:
                continue
            component = int(component_flat[flat])
            count = int(marker_counts[component])
            if count == 1:
                owner_flat[flat] = sole_labels[component]
            elif count > 1:
                active_flat[flat] = True
        del component_labels, marker_counts, sole_labels

        active_count = int(np.count_nonzero(active))
        if active_count > effective_config.maximum_queue_entries:
            raise _resource_error(
                "active multi-marker domain exceeds maximum_queue_entries",
                shape=shape,
                domain_cell_count=domain_count,
                candidate_count=candidate_count,
                component_count=component_count,
                queue_entry_count=active_count,
            )
        if active_count:
            propagation_estimate = _propagation_temporary_estimate(
                total,
                active_count,
                candidate_count,
            )
            estimated_peak = max(estimated_peak, propagation_estimate)
            _preflight_temporary(
                "multi-marker propagation",
                propagation_estimate,
                effective_config,
                shape,
                domain_count,
                candidate_count,
            )
            bottleneck, cardinal, diagonal, settled = _propagate(
                canopy_model.canopy_height_m,
                active,
                owner,
                marker_flat,
                effective_config,
                budget,
            )
            del settled
            ambiguity_estimate = _ambiguity_temporary_estimate(
                total,
                active_count,
                candidate_count,
            )
            estimated_peak = max(estimated_peak, ambiguity_estimate)
            _preflight_temporary(
                "ambiguity",
                ambiguity_estimate,
                effective_config,
                shape,
                domain_count,
                candidate_count,
            )
            ambiguity = _ambiguity_fixed_point(
                canopy_model.canopy_height_m,
                active,
                owner,
                bottleneck,
                cardinal,
                diagonal,
                effective_config,
                budget,
            )
            del bottleneck, cardinal, diagonal
        else:
            ambiguity = np.zeros(shape, dtype=np.bool_)
        del active

        freeze_estimate = _freeze_temporary_estimate(total, candidate_count)
        estimated_peak = max(estimated_peak, freeze_estimate)
        _preflight_temporary(
            "result freezing",
            freeze_estimate,
            effective_config,
            shape,
            domain_count,
            candidate_count,
        )
        (
            region_counts,
            ambiguous_counts,
            edge_flags,
            assigned_count,
            unassigned_count,
            ambiguous_count,
        ) = _metadata(owner, domain, ambiguity, marker_flat, candidate_count)
        result = CanopySegmentationResult(
            source_identity=canopy_model.source_identity,
            source_chm_config=canopy_model.config,
            source_grid_origin_xy=canopy_model.grid_origin_xy,
            source_grid_shape=shape,
            source_treetop_config=treetop_candidates.config,
            config=effective_config,
            seeded_component_count=seeded_components,
            unseeded_component_count=unseeded_components,
            assigned_cell_count=assigned_count,
            unassigned_domain_cell_count=unassigned_count,
            ambiguous_cell_count=ambiguous_count,
            region_label_by_cell=_freeze_array(owner, np.int32),
            segmentation_domain_mask=_freeze_array(domain, np.bool_),
            ambiguity_mask=_freeze_array(ambiguity, np.bool_),
            marker_flat_cell_index_by_candidate=_freeze_array(marker_flat, np.int64),
            region_cell_count_by_candidate=_freeze_array(region_counts, np.int64),
            ambiguous_cell_count_by_candidate=_freeze_array(ambiguous_counts, np.int64),
            region_touches_grid_edge_by_candidate=_freeze_array(edge_flags, np.bool_),
        )
        return result
    except MemoryError as exc:
        raise CanopySegmentationError(
            CanopySegmentationErrorCode.ALLOCATION_FAILED,
            "canopy segmentation allocation failed",
            row_count=shape[0],
            column_count=shape[1],
            total_cell_count=total,
            domain_cell_count=domain_count,
            candidate_count=candidate_count,
            component_count=component_count,
            estimated_temporary_bytes=estimated_peak or None,
        ) from exc
