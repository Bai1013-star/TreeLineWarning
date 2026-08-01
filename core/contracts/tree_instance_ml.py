"""Immutable ML-only predictions aligned to one tree-segmentation input.

This contract deliberately contains no model invocation, training format,
standard-result adaptation, tree readiness, or wire/clearance decision data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any

import numpy as np

from core.contracts._json import json_safe_mapping
from core.contracts.tree_inputs import TreeSegmentationInputSet

TREE_INSTANCE_ML_SCHEMA_VERSION = "tree-instance-ml-v1"


class TreeSemanticClass(IntEnum):
    """Stable semantic classes for one point-aligned ML prediction."""

    NON_TREE = 0
    SHRUB = 1
    TREE = 2


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int | np.integer) or isinstance(value, bool) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _origin_xyz(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError("origin_xyz must be a three-item numeric sequence")
    if len(value) != 3:
        raise ValueError("origin_xyz must contain exactly three values")
    result: list[float] = []
    for coordinate in value:
        if (
            not isinstance(coordinate, int | float | np.integer | np.floating)
            or isinstance(coordinate, bool | np.bool_)
        ):
            raise ValueError("origin_xyz must contain only finite numeric values")
        try:
            numeric = float(coordinate)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("origin_xyz must contain only finite numeric values") from exc
        if not math.isfinite(numeric):
            raise ValueError("origin_xyz must contain only finite numeric values")
        result.append(numeric)
    return (result[0], result[1], result[2])


def _immutable_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _immutable_json_value(value[key]) for key in sorted(value)}
        )
    if isinstance(value, list | tuple):
        return tuple(_immutable_json_value(item) for item in value)
    return value


def _immutable_json_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a non-empty mapping")
    safe = json_safe_mapping(value, name)
    if not safe:
        raise ValueError(f"{name} must be a non-empty mapping")
    return _immutable_json_value(safe)


def _require_ndarray(value: Any, name: str) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise ValueError(f"{name} must be a numpy ndarray")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if value.dtype == object:
        raise ValueError(f"{name} must not use object dtype")
    return value


def _integer_array(
    value: Any,
    dtype: Any,
    length: int | None,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> np.ndarray:
    array = _require_ndarray(value, name)
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must use an integer dtype")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} must have shape ({length},)")
    if minimum is not None and np.any(array < minimum):
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and np.any(array > maximum):
        raise ValueError(f"{name} must not exceed {maximum}")
    return _freeze_array(np.array(array, dtype=dtype, order="C", copy=True, subok=False))


def _confidence_array(value: Any, length: int | None, name: str) -> np.ndarray:
    array = _require_ndarray(value, name)
    if not (
        np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)
    ) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must use a real numeric dtype")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} must have shape ({length},)")
    normalized = np.array(array, dtype=np.float32, order="C", copy=True, subok=False)
    if not np.isfinite(normalized).all() or np.any((normalized < 0.0) | (normalized > 1.0)):
        raise ValueError(f"{name} must contain finite values in [0, 1]")
    return _freeze_array(normalized)


def _freeze_array(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, eq=False)
class TreeInstanceMlPrediction:
    """One immutable ML sidecar aligned exactly to input point rows.

    ``segment_id_by_point == -1`` means only that a point has no retained
    instance membership.  Every point still has a semantic class and semantic
    confidence.  This type intentionally has no tree ID or readiness fields.
    """

    schema_version: str
    source_id: str
    manifest_source_id: str
    point_count: int
    original_index: Any
    sample_index: Any
    origin_xyz: tuple[float, float, float]
    model_version: str
    model_sha256: str
    preprocessing_signature: Mapping[str, Any]
    semantic_class_by_point: Any
    semantic_confidence_by_point: Any
    segment_id_by_point: Any
    instance_confidence_by_point: Any
    segment_ids: Any
    member_offsets: Any
    member_rows: Any
    instance_confidence: Any

    def __post_init__(self) -> None:
        if self.schema_version != TREE_INSTANCE_ML_SCHEMA_VERSION:
            raise ValueError("unsupported tree instance ML schema")
        for name in ("source_id", "manifest_source_id", "model_version"):
            object.__setattr__(self, name, _non_empty_string(getattr(self, name), name))
        point_count = _non_negative_int(self.point_count, "point_count")
        object.__setattr__(self, "point_count", point_count)
        object.__setattr__(self, "origin_xyz", _origin_xyz(self.origin_xyz))
        object.__setattr__(self, "model_sha256", _sha256(self.model_sha256, "model_sha256"))
        object.__setattr__(
            self,
            "preprocessing_signature",
            _immutable_json_mapping(self.preprocessing_signature, "preprocessing_signature"),
        )

        maximum_index = np.iinfo(np.int64).max
        original_index = _integer_array(
            self.original_index,
            np.int64,
            point_count,
            "original_index",
            minimum=0,
            maximum=maximum_index,
        )
        sample_index = _integer_array(
            self.sample_index,
            np.int64,
            point_count,
            "sample_index",
            minimum=0,
            maximum=maximum_index,
        )
        if len(np.unique(original_index)) != point_count:
            raise ValueError("original_index must be unique")
        if len(np.unique(sample_index)) != point_count:
            raise ValueError("sample_index must be unique")

        semantic_class = _integer_array(
            self.semantic_class_by_point,
            np.uint8,
            point_count,
            "semantic_class_by_point",
            minimum=int(TreeSemanticClass.NON_TREE),
            maximum=int(TreeSemanticClass.TREE),
        )
        semantic_confidence = _confidence_array(
            self.semantic_confidence_by_point,
            point_count,
            "semantic_confidence_by_point",
        )
        segment_ids = _integer_array(
            self.segment_ids,
            np.int32,
            None,
            "segment_ids",
            minimum=0,
            maximum=np.iinfo(np.int32).max,
        )
        segment_count = len(segment_ids)
        if len(segment_ids) > 1 and np.any(segment_ids[1:] <= segment_ids[:-1]):
            raise ValueError("segment_ids must be unique and strictly ascending")
        segment_id_by_point = _integer_array(
            self.segment_id_by_point,
            np.int32,
            point_count,
            "segment_id_by_point",
            minimum=-1,
        )
        retained_segment_ids = segment_id_by_point[segment_id_by_point >= 0]
        if len(retained_segment_ids) and not np.isin(retained_segment_ids, segment_ids).all():
            raise ValueError("segment_id_by_point must reference a retained segment_id")
        instance_confidence_by_point = _confidence_array(
            self.instance_confidence_by_point,
            point_count,
            "instance_confidence_by_point",
        )
        if np.any(instance_confidence_by_point[segment_id_by_point == -1] != 0.0):
            raise ValueError("unassigned points must have zero instance_confidence_by_point")
        instance_confidence = _confidence_array(
            self.instance_confidence,
            segment_count,
            "instance_confidence",
        )
        member_offsets = _integer_array(
            self.member_offsets,
            np.int64,
            segment_count + 1,
            "member_offsets",
            minimum=0,
            maximum=np.iinfo(np.int64).max,
        )
        member_rows = _integer_array(
            self.member_rows,
            np.int64,
            None,
            "member_rows",
            minimum=0,
            maximum=max(point_count - 1, -1),
        )
        if member_offsets[0] != 0 or member_offsets[-1] != len(member_rows):
            raise ValueError("member_offsets must begin at zero and end at len(member_rows)")
        if len(member_offsets) > 1 and np.any(member_offsets[1:] < member_offsets[:-1]):
            raise ValueError("member_offsets must be monotonic")

        seen_rows = np.zeros(point_count, dtype=np.bool_)
        for segment_position, segment_id in enumerate(segment_ids):
            start = int(member_offsets[segment_position])
            end = int(member_offsets[segment_position + 1])
            rows = member_rows[start:end]
            if len(rows) > 1 and np.any(rows[1:] <= rows[:-1]):
                raise ValueError("member_rows must be strictly increasing within a segment")
            if len(rows) and np.any(seen_rows[rows]):
                raise ValueError("member_rows must not contain duplicate source rows")
            if len(rows):
                seen_rows[rows] = True
            if len(rows) and np.any(segment_id_by_point[rows] == -1):
                raise ValueError("unassigned rows must not appear in CSR membership")
            expected_rows = np.flatnonzero(segment_id_by_point == segment_id).astype(
                np.int64, copy=False
            )
            if not len(expected_rows):
                raise ValueError("each segment_id must be represented by pointwise membership")
            if not np.array_equal(rows, expected_rows):
                raise ValueError("CSR member rows must exactly match segment_id_by_point")
        if not np.array_equal(seen_rows, segment_id_by_point >= 0):
            raise ValueError("CSR membership must cover every retained point exactly once")

        object.__setattr__(self, "original_index", original_index)
        object.__setattr__(self, "sample_index", sample_index)
        object.__setattr__(self, "semantic_class_by_point", semantic_class)
        object.__setattr__(self, "semantic_confidence_by_point", semantic_confidence)
        object.__setattr__(self, "segment_id_by_point", segment_id_by_point)
        object.__setattr__(self, "instance_confidence_by_point", instance_confidence_by_point)
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "member_offsets", member_offsets)
        object.__setattr__(self, "member_rows", member_rows)
        object.__setattr__(self, "instance_confidence", instance_confidence)


def validate_tree_instance_ml_prediction_alignment(
    tree_input: TreeSegmentationInputSet,
    prediction: TreeInstanceMlPrediction,
) -> int:
    """Reject a prediction that is not exactly identified and aligned to its input.

    This validator deliberately compares the stable identity and point-row
    traceability fields only.  In particular, XYZ values are never used to
    reconstruct row alignment.
    """
    if not isinstance(tree_input, TreeSegmentationInputSet):
        raise ValueError("tree_input must be a TreeSegmentationInputSet")
    if not isinstance(prediction, TreeInstanceMlPrediction):
        raise ValueError("prediction must be a TreeInstanceMlPrediction")
    identity = tree_input.identity
    original_index = tree_input.vegetation_points.original_index
    sample_index = tree_input.vegetation_points.sample_index
    point_count = identity.point_count
    if len(original_index) != point_count or len(sample_index) != point_count:
        raise ValueError("tree input index counts do not match tree input identity")
    if prediction.source_id != identity.source_id:
        raise ValueError("prediction source_id does not match tree input identity")
    if prediction.manifest_source_id != identity.manifest_source_id:
        raise ValueError("prediction manifest_source_id does not match tree input identity")
    if prediction.point_count != point_count:
        raise ValueError("prediction point_count does not match tree input identity")
    if prediction.origin_xyz != identity.origin_xyz:
        raise ValueError("prediction origin_xyz does not exactly match tree input identity")
    if not np.array_equal(prediction.original_index, original_index):
        raise ValueError("prediction original_index does not exactly match tree input row order")
    if not np.array_equal(prediction.sample_index, sample_index):
        raise ValueError("prediction sample_index does not exactly match tree input row order")

    segment_ids = prediction.segment_ids
    segment_id_by_point = prediction.segment_id_by_point
    member_offsets = prediction.member_offsets
    member_rows = prediction.member_rows
    if len(segment_id_by_point) != point_count:
        raise ValueError("prediction segment_id_by_point does not match tree input point_count")
    retained_segment_ids = segment_id_by_point[segment_id_by_point >= 0]
    if len(retained_segment_ids) and not np.isin(retained_segment_ids, segment_ids).all():
        raise ValueError("prediction segment_id_by_point must reference a retained segment_id")
    if len(member_offsets) != len(segment_ids) + 1:
        raise ValueError("prediction member_offsets length must equal segment_ids length plus one")
    if member_offsets[0] != 0 or member_offsets[-1] != len(member_rows):
        raise ValueError("prediction member_offsets must begin at zero and end at len(member_rows)")
    if len(member_offsets) > 1 and np.any(member_offsets[1:] < member_offsets[:-1]):
        raise ValueError("prediction member_offsets must be monotonic")
    if len(member_rows) and np.any((member_rows < 0) | (member_rows >= point_count)):
        raise ValueError("prediction member_rows must contain valid tree input row positions")

    seen_rows = np.zeros(point_count, dtype=np.bool_)
    for segment_position, segment_id in enumerate(segment_ids):
        start = int(member_offsets[segment_position])
        end = int(member_offsets[segment_position + 1])
        rows = member_rows[start:end]
        if len(rows) > 1 and np.any(rows[1:] <= rows[:-1]):
            raise ValueError("prediction member_rows must be strictly increasing within a segment")
        if len(rows) and np.any(seen_rows[rows]):
            raise ValueError("prediction member_rows must not contain duplicate source rows")
        if len(rows) and np.any(segment_id_by_point[rows] == -1):
            raise ValueError("unassigned rows must not appear in prediction CSR membership")
        if len(rows) and np.any(segment_id_by_point[rows] != segment_id):
            raise ValueError("prediction CSR member rows must exactly match segment_id_by_point")
        if len(rows):
            seen_rows[rows] = True
        if not np.any(segment_id_by_point == segment_id):
            raise ValueError(
                "each prediction segment_id must be represented by pointwise membership"
            )
    if not np.array_equal(seen_rows, segment_id_by_point >= 0):
        raise ValueError("prediction CSR membership must cover every retained point exactly once")
    return point_count


__all__ = [
    "TREE_INSTANCE_ML_SCHEMA_VERSION",
    "TreeInstanceMlPrediction",
    "TreeSemanticClass",
    "validate_tree_instance_ml_prediction_alignment",
]
