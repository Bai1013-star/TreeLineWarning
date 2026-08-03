"""Typed, read-only V3 inputs after sibling artifacts have been validated.

These contracts deliberately contain no file loading or cross-artifact identity
validation.  A future V3 input loader is responsible for producing these
already frozen, structurally aligned runtime views.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from core.contracts._json import reject_unknown_fields, require_fields
from core.contracts.features import PointFeatureSet
from core.contracts.ground import HeightAboveGroundSet
from core.contracts.vegetation_instances import VegetationInstancePointSet

_SHA256_FIELDS = (
    "v2_npz_sha256",
    "v2_json_sha256",
    "a2a_npz_sha256",
    "a2a_json_sha256",
    "a2b_npz_sha256",
    "a2b_json_sha256",
    "a2a_input_a1_npz_sha256",
    "a2a_input_a1_json_sha256",
    "a2b_input_a1_npz_sha256",
    "a2b_input_a1_json_sha256",
    "a2b_input_a2a_npz_sha256",
    "a2b_input_a2a_json_sha256",
    "v2_input_v1_npz_sha256",
    "v2_input_v1_json_sha256",
    "v2_input_a2a_npz_sha256",
    "v2_input_a2a_json_sha256",
    "v2_input_a2b_npz_sha256",
    "v2_input_a2b_json_sha256",
)


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _origin_xyz(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError("origin_xyz must be a three-item numeric sequence")
    if len(value) != 3:
        raise ValueError("origin_xyz must contain exactly three values")
    origin: list[float] = []
    for coordinate in value:
        if (
            not isinstance(coordinate, int | float | np.integer | np.floating)
            or isinstance(coordinate, bool | np.bool_)
        ):
            raise ValueError("origin_xyz must contain only finite numeric values")
        try:
            numeric = float(coordinate)
        except (OverflowError, ValueError) as exc:
            raise ValueError("origin_xyz must contain only finite numeric values") from exc
        if not math.isfinite(numeric):
            raise ValueError("origin_xyz must contain only finite numeric values")
        origin.append(numeric)
    return (origin[0], origin[1], origin[2])


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int | np.integer) or isinstance(value, bool) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _readonly_array(value: Any, dtype: Any, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a numpy ndarray")
    if value.dtype != np.dtype(dtype) or value.ndim != len(shape):
        raise ValueError(f"{name} has invalid dtype or rank")
    if any(
        expected is not None and expected != actual
        for expected, actual in zip(shape, value.shape, strict=True)
    ):
        raise ValueError(f"{name} has invalid shape")
    if value.flags.writeable:
        raise ValueError(f"{name} must be read-only")
    return value


def _readonly_contract_arrays(value: Any, name: str) -> None:
    for field_name in value.__dataclass_fields__:
        field_value = getattr(value, field_name)
        if isinstance(field_value, np.ndarray) and field_value.flags.writeable:
            raise ValueError(f"{name}.{field_name} must be read-only")


def _partition_ids(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple of integer IDs")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value):
        raise ValueError(f"{name} must contain non-negative integer IDs")
    if tuple(sorted(value)) != value or len(value) != len(set(value)):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return value


@dataclass(frozen=True)
class TreeSegmentationInputIdentity:
    """JSON-safe artifact audit identity for the three V3 inputs."""

    source_id: str
    manifest_source_id: str
    origin_xyz: tuple[float, float, float]
    point_count: int
    v2_algorithm_version: str
    a2a_algorithm_version: str
    a2b_algorithm_version: str
    v2_npz_sha256: str
    v2_json_sha256: str
    a2a_npz_sha256: str
    a2a_json_sha256: str
    a2b_npz_sha256: str
    a2b_json_sha256: str
    a2a_input_a1_npz_sha256: str
    a2a_input_a1_json_sha256: str
    a2b_input_a1_npz_sha256: str
    a2b_input_a1_json_sha256: str
    a2b_input_a2a_npz_sha256: str
    a2b_input_a2a_json_sha256: str
    v2_input_v1_npz_sha256: str
    v2_input_v1_json_sha256: str
    v2_input_a2a_npz_sha256: str
    v2_input_a2a_json_sha256: str
    v2_input_a2b_npz_sha256: str
    v2_input_a2b_json_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "manifest_source_id",
            "v2_algorithm_version",
            "a2a_algorithm_version",
            "a2b_algorithm_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "origin_xyz", _origin_xyz(self.origin_xyz))
        object.__setattr__(self, "point_count", _non_negative_int(self.point_count, "point_count"))
        for name in _SHA256_FIELDS:
            _sha256(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        data = {name: getattr(self, name) for name in self.__dataclass_fields__}
        data["origin_xyz"] = list(self.origin_xyz)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeSegmentationInputIdentity:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


class TreeSegmentationInputDisposition(str, Enum):
    """The only legal, non-fatal V2 cluster partitions for future V3 work."""

    ELIGIBLE = "eligible"
    REVIEW_ONLY = "review_only"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class TreeSegmentationInputClusterDisposition:
    """One stable reasoned partition record, ordered by ``cluster_id`` in an input set."""

    cluster_id: int
    disposition: TreeSegmentationInputDisposition
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cluster_id", _non_negative_int(self.cluster_id, "cluster_id"))
        if not isinstance(self.disposition, TreeSegmentationInputDisposition):
            raise ValueError("disposition must be a TreeSegmentationInputDisposition")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "disposition": self.disposition.value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeSegmentationInputClusterDisposition:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values = dict(data)
        try:
            values["disposition"] = TreeSegmentationInputDisposition(values["disposition"])
        except (TypeError, ValueError) as exc:
            raise ValueError("disposition is unsupported") from exc
        return cls(**values)


@dataclass(frozen=True)
class TreeSegmentationInputSet:
    """Read-only runtime view with structural, but not cross-artifact, validation.

    The future path loader must freeze every array before constructing this
    contract.  This class preserves those array references without copying,
    sorting, coercing, or deriving V3 segmentation decisions.
    """

    vegetation_points: VegetationInstancePointSet
    geometry_features: PointFeatureSet
    heights_above_ground: HeightAboveGroundSet
    cluster_id: np.ndarray
    cluster_status: np.ndarray
    cluster_vegetation_id: np.ndarray
    cluster_member_offsets: np.ndarray
    ordered_cluster_member_row_index: np.ndarray
    identity: TreeSegmentationInputIdentity
    eligible_cluster_ids: tuple[int, ...]
    review_only_cluster_ids: tuple[int, ...]
    excluded_cluster_ids: tuple[int, ...]
    cluster_dispositions: tuple[TreeSegmentationInputClusterDisposition, ...]
    ground_usable_by_point: np.ndarray
    usable_ground_member_count_by_cluster: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.vegetation_points, VegetationInstancePointSet):
            raise ValueError("vegetation_points must be a VegetationInstancePointSet")
        if not isinstance(self.geometry_features, PointFeatureSet):
            raise ValueError("geometry_features must be a PointFeatureSet")
        if not isinstance(self.heights_above_ground, HeightAboveGroundSet):
            raise ValueError("heights_above_ground must be a HeightAboveGroundSet")
        if not isinstance(self.identity, TreeSegmentationInputIdentity):
            raise ValueError("identity must be a TreeSegmentationInputIdentity")

        _readonly_contract_arrays(self.vegetation_points, "vegetation_points")
        _readonly_contract_arrays(self.geometry_features, "geometry_features")
        _readonly_contract_arrays(self.heights_above_ground, "heights_above_ground")

        point_count = len(self.vegetation_points.original_index)
        if (
            self.geometry_features.point_count != point_count
            or self.heights_above_ground.point_count != point_count
            or self.identity.point_count != point_count
        ):
            raise ValueError("point sets and identity must have the same point_count")

        cluster_id = _readonly_array(self.cluster_id, np.int32, (None,), "cluster_id")
        cluster_count = len(cluster_id)
        if np.any(cluster_id < 0):
            raise ValueError("cluster_id must be non-negative")
        if np.unique(cluster_id).size != cluster_count:
            raise ValueError("cluster_id must be unique")
        _readonly_array(self.cluster_status, np.uint8, (cluster_count,), "cluster_status")
        _readonly_array(
            self.cluster_vegetation_id,
            np.int32,
            (cluster_count,),
            "cluster_vegetation_id",
        )
        _readonly_array(
            self.cluster_member_offsets,
            np.int64,
            (cluster_count + 1,),
            "cluster_member_offsets",
        )
        _readonly_array(
            self.ordered_cluster_member_row_index,
            np.int64,
            (None,),
            "ordered_cluster_member_row_index",
        )
        _readonly_array(
            self.ground_usable_by_point,
            np.bool_,
            (point_count,),
            "ground_usable_by_point",
        )
        usable_counts = _readonly_array(
            self.usable_ground_member_count_by_cluster,
            np.int64,
            (cluster_count,),
            "usable_ground_member_count_by_cluster",
        )
        if np.any(usable_counts < 0):
            raise ValueError("usable_ground_member_count_by_cluster must be non-negative")

        eligible = _partition_ids(self.eligible_cluster_ids, "eligible_cluster_ids")
        review_only = _partition_ids(self.review_only_cluster_ids, "review_only_cluster_ids")
        excluded = _partition_ids(self.excluded_cluster_ids, "excluded_cluster_ids")
        partition_ids = (*eligible, *review_only, *excluded)
        if len(partition_ids) != len(set(partition_ids)):
            raise ValueError("cluster partitions must not overlap")
        if set(partition_ids) != {int(value) for value in cluster_id}:
            raise ValueError("cluster partitions must contain every cluster_id exactly once")

        if not isinstance(self.cluster_dispositions, tuple) or any(
            not isinstance(record, TreeSegmentationInputClusterDisposition)
            for record in self.cluster_dispositions
        ):
            raise ValueError("cluster_dispositions must be a tuple of disposition records")
        disposition_ids = tuple(record.cluster_id for record in self.cluster_dispositions)
        if disposition_ids != tuple(sorted(disposition_ids)) or len(disposition_ids) != len(
            set(disposition_ids)
        ):
            raise ValueError("cluster_dispositions must be sorted by unique cluster_id")
        expected_dispositions = tuple(
            sorted(
                (
                    *((cluster, TreeSegmentationInputDisposition.ELIGIBLE) for cluster in eligible),
                    *(
                        (cluster, TreeSegmentationInputDisposition.REVIEW_ONLY)
                        for cluster in review_only
                    ),
                    *((cluster, TreeSegmentationInputDisposition.EXCLUDED) for cluster in excluded),
                )
            )
        )
        actual_dispositions = tuple(
            (record.cluster_id, record.disposition) for record in self.cluster_dispositions
        )
        if actual_dispositions != expected_dispositions:
            raise ValueError("cluster_dispositions must exactly match cluster partitions")
