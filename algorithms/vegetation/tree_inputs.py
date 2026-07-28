"""Typed identity validation and V2 membership construction for V3 inputs.

This module accepts only already-loaded public contracts.  It provides pure
cross-artifact identity validation and pure V2 membership/InputSet construction
without I/O, artifact hashing, or segmentation execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from core.contracts.features import GeometryFeatureResult, PointFeatureSet
from core.contracts.ground import GroundEstimationResult, GroundQualityFlag, HeightAboveGroundSet
from core.contracts.tree_inputs import (
    TreeSegmentationInputClusterDisposition,
    TreeSegmentationInputDisposition,
    TreeSegmentationInputIdentity,
    TreeSegmentationInputSet,
)
from core.contracts.vegetation_instances import (
    VegetationClusterStatus,
    VegetationInstancePointSet,
    VegetationInstanceResult,
    VegetationInstanceStatistics,
)
from core.contracts.vegetation_trees import TreeSegmentationErrorCode

_V2_PROVENANCE_PATHS = {
    "v1_npz": ("v1_npz", "sha256"),
    "v1_json": ("v1_json", "sha256"),
    "a2a_npz": ("a2a_npz", "sha256"),
    "a2a_json": ("a2a_json", "sha256"),
    "a2b_npz": ("a2b_npz", "sha256"),
    "a2b_json": ("a2b_json", "sha256"),
}

_POINT_ARRAY_DTYPES = {
    "original_index": np.int64,
    "sample_index": np.int64,
    "xyz_local": np.float32,
    "wire_id_by_point": np.int32,
    "vegetation_rule_score": np.float32,
    "vegetation_confidence": np.float32,
    "automatic_vegetation_level": np.uint8,
    "vegetation_level": np.uint8,
    "hag_state": np.uint8,
    "local_scale_raw_m": np.float32,
    "local_scale_m": np.float32,
    "local_scale_state": np.uint8,
    "cluster_id_by_point": np.int32,
    "vegetation_id_by_point": np.int32,
    "cluster_status_by_point": np.uint8,
    "distance_ready_by_point": np.bool_,
    "assignment_confidence": np.float32,
    "assignment_reason_flags": np.uint16,
    "assignment_quality_flags": np.uint16,
    "manual_provenance_flags": np.uint16,
}
_POINT_ARRAY_FIELDS = tuple(_POINT_ARRAY_DTYPES)
_CLUSTER_ARRAY_DTYPES = {
    "cluster_id": np.int32,
    "cluster_status": np.uint8,
    "cluster_vegetation_id": np.int32,
    "cluster_member_offsets": np.int64,
    "ordered_cluster_member_row_index": np.int64,
}
_STATUS_DISPOSITIONS = {
    VegetationClusterStatus.CONFIRMED: (
        TreeSegmentationInputDisposition.ELIGIBLE,
        "confirmed_distance_ready",
    ),
    VegetationClusterStatus.REVIEW_REQUIRED: (
        TreeSegmentationInputDisposition.REVIEW_ONLY,
        "review_required",
    ),
    VegetationClusterStatus.PROVISIONAL: (
        TreeSegmentationInputDisposition.REVIEW_ONLY,
        "provisional",
    ),
    VegetationClusterStatus.REJECTED_AS_INSTANCE: (
        TreeSegmentationInputDisposition.EXCLUDED,
        "rejected_as_instance",
    ),
}


class TreeSegmentationInputError(RuntimeError):
    """A stable, compact failure from typed V3 identity or membership validation."""

    def __init__(
        self,
        code: TreeSegmentationErrorCode,
        artifact_role: str,
        field: str,
        expected: Any,
        actual: Any,
        *,
        source_id: str | None = None,
    ) -> None:
        self.code = code
        self.artifact_role = artifact_role
        self.field = field
        self.expected = expected
        self.actual = actual
        self.source_id = source_id
        message = f"{artifact_role}.{field} is invalid: expected {expected!r}, got {actual!r}"
        if source_id is not None:
            message = f"{message} (source_id={source_id!r})"
        super().__init__(message)


def _raise(
    code: TreeSegmentationErrorCode,
    artifact_role: str,
    field: str,
    expected: Any,
    actual: Any,
    *,
    source_id: str | None = None,
) -> None:
    raise TreeSegmentationInputError(
        code,
        artifact_role,
        field,
        expected,
        actual,
        source_id=source_id,
    )


def _require_type(value: Any, expected_type: type[Any], artifact_role: str) -> None:
    if not isinstance(value, expected_type):
        _raise(
            TreeSegmentationErrorCode.UNSUPPORTED_STATUS,
            artifact_role,
            "contract_type",
            expected_type.__name__,
            type(value).__name__,
        )


def _require_sha256(value: Any, artifact_role: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _raise(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            artifact_role,
            field,
            "lowercase SHA-256 hexadecimal string",
            _value_summary(value),
        )
    return value


def _value_summary(value: Any) -> str:
    """Describe invalid values without embedding an artifact array in an error."""
    if isinstance(value, np.ndarray):
        return f"ndarray[dtype={value.dtype}, shape={value.shape}]"
    return f"{type(value).__name__}"


def _require_v2_origin(value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        _raise(
            TreeSegmentationErrorCode.NONFINITE_REQUIRED_SOURCE_GEOMETRY,
            "v2",
            "origin_xyz",
            "ndarray[dtype=float64, shape=(3,)]",
            _value_summary(value),
        )
    if value.dtype != np.dtype(np.float64) or value.ndim != 1 or value.shape != (3,):
        _raise(
            TreeSegmentationErrorCode.NONFINITE_REQUIRED_SOURCE_GEOMETRY,
            "v2",
            "origin_xyz",
            "ndarray[dtype=float64, shape=(3,)]",
            _value_summary(value),
        )
    if not np.isfinite(value).all():
        _raise(
            TreeSegmentationErrorCode.NONFINITE_REQUIRED_SOURCE_GEOMETRY,
            "v2",
            "origin_xyz",
            "three finite float64 values",
            _value_summary(value),
        )
    return value


def _require_equal(
    left: Any,
    right: Any,
    *,
    artifact_role: str,
    field: str,
    code: TreeSegmentationErrorCode,
    source_id: str | None,
) -> None:
    if left != right:
        _raise(code, artifact_role, field, left, right, source_id=source_id)


def _require_array_equal(
    left: np.ndarray,
    right: Any,
    *,
    artifact_role: str,
    field: str,
    code: TreeSegmentationErrorCode,
    source_id: str,
) -> None:
    if not np.array_equal(left, right):
        _raise(
            code,
            artifact_role,
            field,
            _value_summary(left),
            _value_summary(right),
            source_id=source_id,
        )


def _require_count(
    expected: int,
    actual: Any,
    *,
    artifact_role: str,
    field: str,
    source_id: str,
) -> None:
    if actual != expected:
        _raise(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            artifact_role,
            field,
            expected,
            actual,
            source_id=source_id,
        )


def _v2_provenance_hash(
    result: VegetationInstanceResult,
    name: str,
    source_id: str,
) -> str:
    path = _V2_PROVENANCE_PATHS[name]
    provenance = result.input_provenance
    if not isinstance(provenance, Mapping):
        _raise(
            TreeSegmentationErrorCode.MISSING_REQUIRED_FIELD,
            "v2",
            "input_provenance",
            "mapping containing required provenance hashes",
            _value_summary(provenance),
            source_id=source_id,
        )
    group = provenance.get(path[0])
    if not isinstance(group, Mapping) or path[1] not in group:
        _raise(
            TreeSegmentationErrorCode.MISSING_REQUIRED_FIELD,
            "v2",
            f"input_provenance.{path[0]}.{path[1]}",
            "present lowercase SHA-256",
            "missing",
            source_id=source_id,
        )
    return _require_sha256(
        group[path[1]],
        "v2",
        f"input_provenance.{path[0]}.{path[1]}",
    )


def validate_tree_input_identity(
    v2_result: VegetationInstanceResult,
    v2_points: VegetationInstancePointSet,
    v2_origin_xyz: np.ndarray,
    a2a_result: GeometryFeatureResult,
    a2a_points: PointFeatureSet,
    a2b_result: GroundEstimationResult,
    a2b_points: HeightAboveGroundSet,
    *,
    v2_json_sha256: str,
    a2a_json_sha256: str,
    a2b_json_sha256: str,
) -> TreeSegmentationInputIdentity:
    """Validate V2/A2A/A2B identity and return its typed audit record.

    No file paths are accepted or read.  Comparisons are exact and preserve all
    supplied array objects, values, ordering, and writeability flags.
    """
    _require_type(v2_result, VegetationInstanceResult, "v2")
    _require_type(v2_points, VegetationInstancePointSet, "v2_points")
    _require_type(a2a_result, GeometryFeatureResult, "a2a")
    _require_type(a2a_points, PointFeatureSet, "a2a_points")
    _require_type(a2b_result, GroundEstimationResult, "a2b")
    _require_type(a2b_points, HeightAboveGroundSet, "a2b_points")
    v2_origin = _require_v2_origin(v2_origin_xyz)

    supplied_hashes = {
        "v2_json_sha256": _require_sha256(v2_json_sha256, "v2", "json_sha256"),
        "a2a_json_sha256": _require_sha256(a2a_json_sha256, "a2a", "json_sha256"),
        "a2b_json_sha256": _require_sha256(a2b_json_sha256, "a2b", "json_sha256"),
    }
    source_id = v2_result.source_id
    _require_equal(
        source_id,
        a2a_result.source_id,
        artifact_role="a2a",
        field="source_id",
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        source_id=source_id,
    )
    _require_equal(
        source_id,
        a2b_result.source_id,
        artifact_role="a2b",
        field="source_id",
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        source_id=source_id,
    )
    for artifact_role, point_set in (("a2a_points", a2a_points), ("a2b_points", a2b_points)):
        _require_equal(
            source_id,
            point_set.source_id,
            artifact_role=artifact_role,
            field="source_id",
            code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            source_id=source_id,
        )

    manifest_source_id = v2_result.manifest_source_id
    for artifact_role, value in (
        ("a2a", a2a_result.manifest_source_id),
        ("a2b", a2b_result.manifest_source_id),
        ("a2a_points", a2a_points.manifest_source_id),
        ("a2b_points", a2b_points.manifest_source_id),
    ):
        _require_equal(
            manifest_source_id,
            value,
            artifact_role=artifact_role,
            field="manifest_source_id",
            code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            source_id=source_id,
        )

    _require_array_equal(
        v2_origin,
        v2_result.origin_xyz,
        artifact_role="v2",
        field="origin_xyz",
        code=TreeSegmentationErrorCode.COORDINATE_MISMATCH,
        source_id=source_id,
    )
    _require_array_equal(
        v2_origin,
        a2a_result.origin_xyz,
        artifact_role="a2a",
        field="origin_xyz",
        code=TreeSegmentationErrorCode.COORDINATE_MISMATCH,
        source_id=source_id,
    )
    _require_array_equal(
        v2_origin,
        a2a_points.origin_xyz,
        artifact_role="a2a_points",
        field="origin_xyz",
        code=TreeSegmentationErrorCode.COORDINATE_MISMATCH,
        source_id=source_id,
    )
    _require_array_equal(
        v2_origin,
        a2b_result.origin_xyz,
        artifact_role="a2b",
        field="origin_xyz",
        code=TreeSegmentationErrorCode.COORDINATE_MISMATCH,
        source_id=source_id,
    )

    point_count = len(v2_points.original_index)
    _require_count(
        point_count,
        a2a_points.point_count,
        artifact_role="a2a_points",
        field="point_count",
        source_id=source_id,
    )
    _require_count(
        point_count,
        a2b_points.point_count,
        artifact_role="a2b_points",
        field="point_count",
        source_id=source_id,
    )
    for artifact_role, value in (
        ("v2.statistics.input_point_count", v2_result.statistics.input_point_count),
        ("a2a.statistics.input_point_count", a2a_result.statistics.input_point_count),
        ("a2a.statistics.output_point_count", a2a_result.statistics.output_point_count),
        ("a2b.statistics.input_point_count", a2b_result.statistics.input_point_count),
        ("a2b.statistics.output_point_count", a2b_result.statistics.output_point_count),
    ):
        role, field = artifact_role.split(".", maxsplit=1)
        _require_count(point_count, value, artifact_role=role, field=field, source_id=source_id)

    for field in ("original_index", "sample_index"):
        _require_array_equal(
            getattr(v2_points, field),
            getattr(a2a_points, field),
            artifact_role="a2a_points",
            field=field,
            code=TreeSegmentationErrorCode.INVALID_INDEXES,
            source_id=source_id,
        )
        _require_array_equal(
            getattr(v2_points, field),
            getattr(a2b_points, field),
            artifact_role="a2b_points",
            field=field,
            code=TreeSegmentationErrorCode.INVALID_INDEXES,
            source_id=source_id,
        )
    _require_array_equal(
        v2_points.xyz_local,
        a2a_points.xyz_local,
        artifact_role="a2a_points",
        field="xyz_local",
        code=TreeSegmentationErrorCode.COORDINATE_MISMATCH,
        source_id=source_id,
    )

    a2a_a1_npz = _require_sha256(a2a_result.input_a1_npz_sha256, "a2a", "input_a1_npz_sha256")
    a2a_a1_json = _require_sha256(a2a_result.input_a1_json_sha256, "a2a", "input_a1_json_sha256")
    a2b_a1_npz = _require_sha256(a2b_result.input_a1_npz_sha256, "a2b", "input_a1_npz_sha256")
    a2b_a1_json = _require_sha256(a2b_result.input_a1_json_sha256, "a2b", "input_a1_json_sha256")
    a2b_a2a_npz = _require_sha256(a2b_result.input_a2a_npz_sha256, "a2b", "input_a2a_npz_sha256")
    a2b_a2a_json = _require_sha256(a2b_result.input_a2a_json_sha256, "a2b", "input_a2a_json_sha256")
    a2a_npz = _require_sha256(a2a_result.output_npz_sha256, "a2a", "output_npz_sha256")
    a2b_npz = _require_sha256(a2b_result.output_npz_sha256, "a2b", "output_npz_sha256")
    v2_npz = _require_sha256(v2_result.output_npz_sha256, "v2", "output_npz_sha256")
    _require_equal(
        a2a_a1_npz,
        a2b_a1_npz,
        artifact_role="a2b",
        field="input_a1_npz_sha256",
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        source_id=source_id,
    )
    _require_equal(
        a2a_a1_json,
        a2b_a1_json,
        artifact_role="a2b",
        field="input_a1_json_sha256",
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        source_id=source_id,
    )
    _require_equal(
        a2a_npz,
        a2b_a2a_npz,
        artifact_role="a2b",
        field="input_a2a_npz_sha256",
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        source_id=source_id,
    )
    _require_equal(
        supplied_hashes["a2a_json_sha256"],
        a2b_a2a_json,
        artifact_role="a2b",
        field="input_a2a_json_sha256",
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        source_id=source_id,
    )

    v2_v1_npz = _v2_provenance_hash(v2_result, "v1_npz", source_id)
    v2_v1_json = _v2_provenance_hash(v2_result, "v1_json", source_id)
    v2_a2a_npz = _v2_provenance_hash(v2_result, "a2a_npz", source_id)
    v2_a2a_json = _v2_provenance_hash(v2_result, "a2a_json", source_id)
    v2_a2b_npz = _v2_provenance_hash(v2_result, "a2b_npz", source_id)
    v2_a2b_json = _v2_provenance_hash(v2_result, "a2b_json", source_id)
    _require_equal(
        a2a_npz,
        v2_a2a_npz,
        artifact_role="v2",
        field="input_provenance.a2a_npz.sha256",
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        source_id=source_id,
    )
    _require_equal(
        supplied_hashes["a2a_json_sha256"],
        v2_a2a_json,
        artifact_role="v2",
        field="input_provenance.a2a_json.sha256",
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        source_id=source_id,
    )
    _require_equal(
        a2b_npz,
        v2_a2b_npz,
        artifact_role="v2",
        field="input_provenance.a2b_npz.sha256",
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        source_id=source_id,
    )
    _require_equal(
        supplied_hashes["a2b_json_sha256"],
        v2_a2b_json,
        artifact_role="v2",
        field="input_provenance.a2b_json.sha256",
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        source_id=source_id,
    )

    return TreeSegmentationInputIdentity(
        source_id=source_id,
        manifest_source_id=manifest_source_id,
        origin_xyz=(float(v2_origin[0]), float(v2_origin[1]), float(v2_origin[2])),
        point_count=point_count,
        v2_algorithm_version=v2_result.algorithm_version,
        a2a_algorithm_version=a2a_result.algorithm_version,
        a2b_algorithm_version=a2b_result.algorithm_version,
        v2_npz_sha256=v2_npz,
        v2_json_sha256=supplied_hashes["v2_json_sha256"],
        a2a_npz_sha256=a2a_npz,
        a2a_json_sha256=supplied_hashes["a2a_json_sha256"],
        a2b_npz_sha256=a2b_npz,
        a2b_json_sha256=supplied_hashes["a2b_json_sha256"],
        a2a_input_a1_npz_sha256=a2a_a1_npz,
        a2a_input_a1_json_sha256=a2a_a1_json,
        a2b_input_a1_npz_sha256=a2b_a1_npz,
        a2b_input_a1_json_sha256=a2b_a1_json,
        a2b_input_a2a_npz_sha256=a2b_a2a_npz,
        a2b_input_a2a_json_sha256=a2b_a2a_json,
        v2_input_v1_npz_sha256=v2_v1_npz,
        v2_input_v1_json_sha256=v2_v1_json,
        v2_input_a2a_npz_sha256=v2_a2a_npz,
        v2_input_a2a_json_sha256=v2_a2a_json,
        v2_input_a2b_npz_sha256=v2_a2b_npz,
        v2_input_a2b_json_sha256=v2_a2b_json,
    )


def _require_mapping(value: Any) -> Mapping[str, np.ndarray]:
    if not isinstance(value, Mapping):
        _raise(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "v2",
            "arrays",
            "Mapping[str, numpy.ndarray]",
            _value_summary(value),
        )
    return value


def _require_array(
    arrays: Mapping[str, Any],
    name: str,
    dtype: Any,
    shape: tuple[int | None, ...],
    *,
    artifact_role: str = "v2",
) -> np.ndarray:
    if name not in arrays:
        _raise(
            TreeSegmentationErrorCode.MISSING_REQUIRED_FIELD,
            artifact_role,
            name,
            "required ndarray field",
            "missing",
        )
    value = arrays[name]
    if not isinstance(value, np.ndarray):
        _raise(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            artifact_role,
            name,
            f"ndarray[dtype={np.dtype(dtype)}, shape={shape}]",
            _value_summary(value),
        )
    if (
        value.dtype != np.dtype(dtype)
        or value.ndim != len(shape)
        or any(
            expected is not None and expected != actual
            for expected, actual in zip(shape, value.shape, strict=True)
        )
    ):
        _raise(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            artifact_role,
            name,
            f"ndarray[dtype={np.dtype(dtype)}, shape={shape}]",
            _value_summary(value),
        )
    return value


def _require_status(
    value: int, artifact_role: str, field: str, source_id: str
) -> VegetationClusterStatus:
    try:
        return VegetationClusterStatus(int(value))
    except ValueError:
        _raise(
            TreeSegmentationErrorCode.UNSUPPORTED_STATUS,
            artifact_role,
            field,
            "VegetationClusterStatus value",
            int(value),
            source_id=source_id,
        )
    raise AssertionError("unreachable")


def _require_readonly_contract_arrays(value: Any, artifact_role: str, source_id: str) -> None:
    for name in value.__dataclass_fields__:
        array = getattr(value, name)
        if isinstance(array, np.ndarray) and array.flags.writeable:
            _raise(
                TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
                artifact_role,
                name,
                "read-only ndarray",
                "writeable ndarray",
                source_id=source_id,
            )


def _validate_offsets(offsets: np.ndarray, member_count: int, source_id: str) -> None:
    if (
        offsets[0] != 0
        or offsets[-1] != member_count
        or np.any(offsets < 0)
        or np.any(offsets > member_count)
        or np.any(np.diff(offsets) < 0)
    ):
        _raise(
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "v2",
            "cluster_member_offsets",
            "monotonic int64 offsets spanning ordered_cluster_member_row_index",
            _value_summary(offsets),
            source_id=source_id,
        )


def _validate_membership(
    point_cluster_ids: np.ndarray,
    point_status: np.ndarray,
    point_vegetation_ids: np.ndarray,
    point_distance_ready: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_status: np.ndarray,
    cluster_vegetation_ids: np.ndarray,
    offsets: np.ndarray,
    member_rows: np.ndarray,
    source_id: str,
) -> tuple[VegetationClusterStatus, ...]:
    point_count = len(point_cluster_ids)
    if np.any(member_rows < 0) or np.any(member_rows >= point_count):
        _raise(
            TreeSegmentationErrorCode.INVALID_INDEXES,
            "v2",
            "ordered_cluster_member_row_index",
            f"int64 rows in [0, {point_count})",
            _value_summary(member_rows),
            source_id=source_id,
        )
    member_counts = np.zeros(point_count, dtype=np.int64)
    np.add.at(member_counts, member_rows, 1)
    if np.any(member_counts > 1):
        _raise(
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "v2",
            "ordered_cluster_member_row_index",
            "each point row appears in the member table at most once",
            "duplicate member row",
            source_id=source_id,
        )
    assigned = point_cluster_ids >= 0
    if np.any(member_counts[assigned] != 1) or np.any(member_counts[~assigned] != 0):
        _raise(
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "v2",
            "ordered_cluster_member_row_index",
            "complete coverage of assigned rows only",
            "member table does not exactly cover assigned rows",
            source_id=source_id,
        )

    statuses = tuple(
        _require_status(value, "v2", "cluster_status", source_id) for value in cluster_status
    )
    for value in point_status:
        _require_status(value, "v2", "cluster_status_by_point", source_id)
    if VegetationClusterStatus.UNASSIGNED in statuses:
        _raise(
            TreeSegmentationErrorCode.UNSUPPORTED_STATUS,
            "v2",
            "cluster_status",
            "formal cluster status",
            VegetationClusterStatus.UNASSIGNED.name,
            source_id=source_id,
        )

    unassigned = point_cluster_ids < 0
    if (
        np.any(point_vegetation_ids[unassigned] != -1)
        or np.any(point_distance_ready[unassigned])
        or np.any(point_status[unassigned] != int(VegetationClusterStatus.UNASSIGNED))
    ):
        _raise(
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "v2",
            "unassigned_points",
            "UNASSIGNED status, vegetation_id=-1, and distance_ready=False",
            "inconsistent point-level unassigned state",
            source_id=source_id,
        )

    for position, (cluster_id, status) in enumerate(zip(cluster_ids, statuses, strict=True)):
        start, end = offsets[position : position + 2]
        rows = member_rows[start:end]
        if not len(rows):
            _raise(
                TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
                "v2",
                "cluster_member_offsets",
                "every formal cluster has at least one member row",
                f"cluster_id={int(cluster_id)} has no members",
                source_id=source_id,
            )
        if np.any(point_cluster_ids[rows] != cluster_id) or np.any(
            point_status[rows] != int(status)
        ):
            _raise(
                TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
                "v2",
                "point_cluster_membership",
                "member rows agree with cluster_id and status",
                f"cluster_id={int(cluster_id)} has inconsistent member metadata",
                source_id=source_id,
            )
        if status is VegetationClusterStatus.CONFIRMED:
            if cluster_vegetation_ids[position] < 0:
                _raise(
                    TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
                    "v2",
                    "cluster_vegetation_id",
                    "non-negative vegetation ID for CONFIRMED cluster",
                    int(cluster_vegetation_ids[position]),
                    source_id=source_id,
                )
            if np.any(point_vegetation_ids[rows] != cluster_vegetation_ids[position]) or not np.all(
                point_distance_ready[rows]
            ):
                _raise(
                    TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
                    "v2",
                    "confirmed_member_assignment",
                    "matching vegetation ID and distance_ready=True",
                    f"cluster_id={int(cluster_id)} has inconsistent confirmed members",
                    source_id=source_id,
                )
        elif (
            cluster_vegetation_ids[position] != -1
            or np.any(point_vegetation_ids[rows] != -1)
            or np.any(point_distance_ready[rows])
        ):
            _raise(
                TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
                "v2",
                "nonconfirmed_member_assignment",
                "vegetation_id=-1 and distance_ready=False",
                f"cluster_id={int(cluster_id)} has formal assignment outside CONFIRMED",
                source_id=source_id,
            )

    return statuses


def _require_summary_scalar(
    value: Any,
    expected_type: type[Any],
    field: str,
    source_id: str,
    *,
    non_negative: bool = False,
) -> Any:
    if type(value) is not expected_type or (non_negative and value < 0):
        expected = expected_type.__name__
        if non_negative:
            expected = f"non-negative {expected}"
        _raise(
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "v2",
            f"clusters[].{field}",
            expected,
            _value_summary(value),
            source_id=source_id,
        )
    return value


def _validate_summaries(
    summaries: Any,
    cluster_ids: np.ndarray,
    cluster_status: tuple[VegetationClusterStatus, ...],
    cluster_vegetation_ids: np.ndarray,
    offsets: np.ndarray,
    source_id: str,
) -> None:
    if not isinstance(summaries, tuple) or len(summaries) != len(cluster_ids):
        _raise(
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "v2",
            "clusters",
            f"tuple of {len(cluster_ids)} cluster summaries",
            _value_summary(summaries),
            source_id=source_id,
        )
    expected = {int(value): position for position, value in enumerate(cluster_ids)}
    seen: set[int] = set()
    for summary in summaries:
        if not isinstance(summary, Mapping):
            _raise(
                TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
                "v2",
                "clusters[]",
                "mapping cluster summary",
                _value_summary(summary),
                source_id=source_id,
            )
        for field in ("cluster_id", "point_count", "status", "vegetation_id", "requires_review"):
            if field not in summary:
                _raise(
                    TreeSegmentationErrorCode.MISSING_REQUIRED_FIELD,
                    "v2",
                    f"clusters[].{field}",
                    "required summary field",
                    "missing",
                    source_id=source_id,
                )
        cluster_id = _require_summary_scalar(summary["cluster_id"], int, "cluster_id", source_id)
        point_count = _require_summary_scalar(
            summary["point_count"], int, "point_count", source_id, non_negative=True
        )
        status = _require_summary_scalar(summary["status"], str, "status", source_id)
        vegetation_id = _require_summary_scalar(
            summary["vegetation_id"], int, "vegetation_id", source_id
        )
        requires_review = _require_summary_scalar(
            summary["requires_review"], bool, "requires_review", source_id
        )
        if cluster_id not in expected:
            _raise(
                TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
                "v2",
                "clusters[].cluster_id",
                "unique cluster ID present in cluster_id array",
                _value_summary(cluster_id),
                source_id=source_id,
            )
        if cluster_id in seen:
            _raise(
                TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
                "v2",
                "clusters[].cluster_id",
                "unique summary cluster ID",
                cluster_id,
                source_id=source_id,
            )
        seen.add(cluster_id)
        position = expected[cluster_id]
        expected_count = int(offsets[position + 1] - offsets[position])
        if point_count != expected_count:
            _raise(
                TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
                "v2",
                "clusters[].point_count",
                expected_count,
                point_count,
                source_id=source_id,
            )
        if status != cluster_status[position].name:
            _raise(
                TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
                "v2",
                "clusters[].status",
                cluster_status[position].name,
                status,
                source_id=source_id,
            )
        if vegetation_id != int(cluster_vegetation_ids[position]):
            _raise(
                TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
                "v2",
                "clusters[].vegetation_id",
                int(cluster_vegetation_ids[position]),
                vegetation_id,
                source_id=source_id,
            )
        expected_review = cluster_status[position] is VegetationClusterStatus.REVIEW_REQUIRED
        if requires_review is not expected_review:
            _raise(
                TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
                "v2",
                "clusters[].requires_review",
                expected_review,
                requires_review,
                source_id=source_id,
            )


def _validate_statistics(
    statistics: Any,
    point_count: int,
    cluster_ids: np.ndarray,
    statuses: tuple[VegetationClusterStatus, ...],
    point_cluster_ids: np.ndarray,
    source_id: str,
) -> None:
    if not isinstance(statistics, VegetationInstanceStatistics):
        _raise(
            TreeSegmentationErrorCode.UNSUPPORTED_STATUS,
            "v2",
            "statistics",
            "VegetationInstanceStatistics",
            _value_summary(statistics),
            source_id=source_id,
        )
    expected = {
        "input_point_count": point_count,
        "cluster_count": len(cluster_ids),
        "confirmed_count": statuses.count(VegetationClusterStatus.CONFIRMED),
        "review_count": statuses.count(VegetationClusterStatus.REVIEW_REQUIRED),
        "provisional_count": statuses.count(VegetationClusterStatus.PROVISIONAL),
        "rejected_count": statuses.count(VegetationClusterStatus.REJECTED_AS_INSTANCE),
        "assigned_candidate_count": int(np.count_nonzero(point_cluster_ids >= 0)),
    }
    for field, value in expected.items():
        if getattr(statistics, field) != value:
            _raise(
                TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
                "v2",
                f"statistics.{field}",
                value,
                getattr(statistics, field),
                source_id=source_id,
            )


def build_tree_segmentation_input_set(
    v2_result: VegetationInstanceResult,
    v2_arrays: Mapping[str, np.ndarray],
    geometry_features: PointFeatureSet,
    heights_above_ground: HeightAboveGroundSet,
    identity: TreeSegmentationInputIdentity,
) -> TreeSegmentationInputSet:
    """Validate loaded V2 membership and build the read-only V3 runtime input.

    This pure builder performs no path, provenance, or artifact work.  It
    preserves loaded V2 array objects and freezes only the V2 arrays retained
    by the returned input set after every structural and semantic check passes.
    """
    _require_type(v2_result, VegetationInstanceResult, "v2")
    _require_type(geometry_features, PointFeatureSet, "a2a_points")
    _require_type(heights_above_ground, HeightAboveGroundSet, "a2b_points")
    _require_type(identity, TreeSegmentationInputIdentity, "identity")
    arrays = _require_mapping(v2_arrays)
    source_id = v2_result.source_id

    point_arrays: dict[str, np.ndarray] = {}
    for name in _POINT_ARRAY_FIELDS:
        shape = (None,) if name != "xyz_local" else (None, 3)
        point_arrays[name] = _require_array(arrays, name, _POINT_ARRAY_DTYPES[name], shape)
    point_count = len(point_arrays["original_index"])
    for name, array in point_arrays.items():
        expected = (point_count, 3) if name == "xyz_local" else (point_count,)
        if array.shape != expected:
            _raise(
                TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
                "v2",
                name,
                f"ndarray[dtype={array.dtype}, shape={expected}]",
                _value_summary(array),
                source_id=source_id,
            )

    cluster_id = _require_array(arrays, "cluster_id", _CLUSTER_ARRAY_DTYPES["cluster_id"], (None,))
    cluster_count = len(cluster_id)
    cluster_status = _require_array(
        arrays, "cluster_status", _CLUSTER_ARRAY_DTYPES["cluster_status"], (cluster_count,)
    )
    cluster_vegetation_id = _require_array(
        arrays,
        "cluster_vegetation_id",
        _CLUSTER_ARRAY_DTYPES["cluster_vegetation_id"],
        (cluster_count,),
    )
    offsets = _require_array(
        arrays,
        "cluster_member_offsets",
        _CLUSTER_ARRAY_DTYPES["cluster_member_offsets"],
        (cluster_count + 1,),
    )
    member_rows = _require_array(
        arrays,
        "ordered_cluster_member_row_index",
        _CLUSTER_ARRAY_DTYPES["ordered_cluster_member_row_index"],
        (None,),
    )
    if np.any(cluster_id < 0) or len(np.unique(cluster_id)) != cluster_count:
        _raise(
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "v2",
            "cluster_id",
            "unique non-negative int32 cluster IDs",
            _value_summary(cluster_id),
            source_id=source_id,
        )
    _validate_offsets(offsets, len(member_rows), source_id)

    if (
        geometry_features.point_count != point_count
        or heights_above_ground.point_count != point_count
        or identity.point_count != point_count
    ):
        _raise(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "v3_inputs",
            "point_count",
            point_count,
            {
                "a2a": geometry_features.point_count,
                "a2b": heights_above_ground.point_count,
                "identity": identity.point_count,
            },
            source_id=source_id,
        )
    _require_readonly_contract_arrays(geometry_features, "a2a_points", source_id)
    _require_readonly_contract_arrays(heights_above_ground, "a2b_points", source_id)

    statuses = _validate_membership(
        point_arrays["cluster_id_by_point"],
        point_arrays["cluster_status_by_point"],
        point_arrays["vegetation_id_by_point"],
        point_arrays["distance_ready_by_point"],
        cluster_id,
        cluster_status,
        cluster_vegetation_id,
        offsets,
        member_rows,
        source_id,
    )
    try:
        vegetation_points = VegetationInstancePointSet(**point_arrays)
    except ValueError as exc:
        _raise(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "v2",
            "point_arrays",
            "valid VegetationInstancePointSet arrays",
            str(exc),
            source_id=source_id,
        )
    _validate_summaries(
        v2_result.clusters,
        cluster_id,
        statuses,
        cluster_vegetation_id,
        offsets,
        source_id,
    )
    _validate_statistics(
        v2_result.statistics,
        point_count,
        cluster_id,
        statuses,
        point_arrays["cluster_id_by_point"],
        source_id,
    )

    disposition_rows = []
    for position, status in enumerate(statuses):
        disposition, reason = _STATUS_DISPOSITIONS[status]
        disposition_rows.append((int(cluster_id[position]), disposition, reason))
    disposition_rows.sort(key=lambda row: row[0])
    dispositions = tuple(
        TreeSegmentationInputClusterDisposition(cluster, disposition, reason)
        for cluster, disposition, reason in disposition_rows
    )
    eligible = tuple(
        cluster
        for cluster, disposition, _reason in disposition_rows
        if disposition is TreeSegmentationInputDisposition.ELIGIBLE
    )
    review_only = tuple(
        cluster
        for cluster, disposition, _reason in disposition_rows
        if disposition is TreeSegmentationInputDisposition.REVIEW_ONLY
    )
    excluded = tuple(
        cluster
        for cluster, disposition, _reason in disposition_rows
        if disposition is TreeSegmentationInputDisposition.EXCLUDED
    )
    ground_usable = (
        heights_above_ground.quality_flags & int(GroundQualityFlag.INVALID_GROUND_ESTIMATE)
    ) == 0
    ground_usable = ground_usable.astype(np.bool_, copy=False)
    usable_counts = np.zeros(cluster_count, dtype=np.int64)
    for position in range(cluster_count):
        start, end = offsets[position : position + 2]
        usable_counts[position] = np.count_nonzero(ground_usable[member_rows[start:end]])

    retained = (
        *point_arrays.values(),
        cluster_id,
        cluster_status,
        cluster_vegetation_id,
        offsets,
        member_rows,
    )
    writeable_before = tuple(array.flags.writeable for array in retained)
    try:
        for array in retained:
            array.setflags(write=False)
        ground_usable.setflags(write=False)
        usable_counts.setflags(write=False)
        return TreeSegmentationInputSet(
            vegetation_points=vegetation_points,
            geometry_features=geometry_features,
            heights_above_ground=heights_above_ground,
            cluster_id=cluster_id,
            cluster_status=cluster_status,
            cluster_vegetation_id=cluster_vegetation_id,
            cluster_member_offsets=offsets,
            ordered_cluster_member_row_index=member_rows,
            identity=identity,
            eligible_cluster_ids=eligible,
            review_only_cluster_ids=review_only,
            excluded_cluster_ids=excluded,
            cluster_dispositions=dispositions,
            ground_usable_by_point=ground_usable,
            usable_ground_member_count_by_cluster=usable_counts,
        )
    except ValueError as exc:
        for array, writeable in zip(retained, writeable_before, strict=True):
            if writeable:
                array.setflags(write=True)
        _raise(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "v3_inputs",
            "TreeSegmentationInputSet",
            "valid read-only input set",
            str(exc),
            source_id=source_id,
        )
