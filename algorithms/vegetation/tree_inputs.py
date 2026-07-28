"""Pure cross-artifact identity validation for typed V3 tree inputs.

This module accepts only already-loaded public contracts.  It neither performs
I/O nor constructs the broader V3 input set; callers remain responsible for
loading and freezing their artifact arrays before invoking the validator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from core.contracts.features import GeometryFeatureResult, PointFeatureSet
from core.contracts.ground import GroundEstimationResult, HeightAboveGroundSet
from core.contracts.tree_inputs import TreeSegmentationInputIdentity
from core.contracts.vegetation_instances import (
    VegetationInstancePointSet,
    VegetationInstanceResult,
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


class TreeSegmentationInputError(RuntimeError):
    """A stable, compact failure from typed V3 input identity validation."""

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
