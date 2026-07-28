from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from pathlib import Path
from typing import get_type_hints

import algorithms.vegetation.tree_inputs as tree_inputs
import numpy as np
import pytest
from algorithms.vegetation.tree_inputs import (
    TreeSegmentationInputError,
    validate_tree_input_identity,
)
from core.contracts.features import (
    GeometryFeatureConfig,
    GeometryFeatureResult,
    GeometryFeatureStatistics,
    PointFeatureSet,
)
from core.contracts.ground import (
    GroundEstimationConfig,
    GroundEstimationResult,
    GroundEstimationStatistics,
    HeightAboveGroundSet,
)
from core.contracts.tree_inputs import TreeSegmentationInputIdentity
from core.contracts.vegetation_instances import (
    VegetationInstanceConfig,
    VegetationInstancePointSet,
    VegetationInstanceResult,
    VegetationInstanceStatistics,
    VegetationManualLog,
)
from core.contracts.vegetation_trees import TreeSegmentationErrorCode


def _digest(character: str) -> str:
    return character * 64


def _bundle(count: int = 3):
    source_id = "scene-001"
    manifest_source_id = "manifest-001"
    origin = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    original_index = np.arange(10, 10 + count, dtype=np.int64)
    sample_index = np.arange(count, dtype=np.int64)
    xyz = np.arange(count * 3, dtype=np.float32).reshape((count, 3))
    floats = np.ones(count, dtype=np.float32)
    uint8 = np.zeros(count, dtype=np.uint8)
    uint16 = np.zeros(count, dtype=np.uint16)
    int32 = np.full(count, -1, dtype=np.int32)

    v2_points = VegetationInstancePointSet(
        original_index=original_index,
        sample_index=sample_index,
        xyz_local=xyz,
        wire_id_by_point=int32.copy(),
        vegetation_rule_score=floats.copy(),
        vegetation_confidence=floats.copy(),
        automatic_vegetation_level=uint8.copy(),
        vegetation_level=uint8.copy(),
        hag_state=uint8.copy(),
        local_scale_raw_m=floats.copy(),
        local_scale_m=floats.copy(),
        local_scale_state=uint8.copy(),
        cluster_id_by_point=int32.copy(),
        vegetation_id_by_point=int32.copy(),
        cluster_status_by_point=uint8.copy(),
        distance_ready_by_point=np.zeros(count, dtype=np.bool_),
        assignment_confidence=floats.copy(),
        assignment_reason_flags=uint16.copy(),
        assignment_quality_flags=uint16.copy(),
        manual_provenance_flags=uint16.copy(),
    )
    a2a_points = PointFeatureSet(
        source_id=source_id,
        manifest_source_id=manifest_source_id,
        xyz_local=xyz.copy(),
        origin_xyz=origin.copy(),
        original_index=original_index.copy(),
        sample_index=sample_index.copy(),
        linearity=floats.copy(),
        planarity=floats.copy(),
        scattering=floats.copy(),
        verticality=floats.copy(),
        local_density=floats.copy(),
        height_m=floats.copy(),
        neighbor_count=np.ones(count, dtype=np.uint16),
        eigenvalues=np.ones((count, 3), dtype=np.float32),
        principal_direction=np.ones((count, 3), dtype=np.float32),
        normal=np.ones((count, 3), dtype=np.float32),
        quality_flags=uint16.copy(),
    )
    a2b_points = HeightAboveGroundSet(
        source_id=source_id,
        manifest_source_id=manifest_source_id,
        original_index=original_index.copy(),
        sample_index=sample_index.copy(),
        ground_z_m=floats.copy(),
        raw_height_agl_m=floats.copy(),
        height_agl_m=floats.copy(),
        ground_cell_id=np.zeros(count, dtype=np.int32),
        ground_distance_m=floats.copy(),
        quality_flags=uint16.copy(),
    )
    v2_stats = VegetationInstanceStatistics(
        input_point_count=count,
        candidate_count=count,
        core_point_count=count,
        cluster_count=0,
        confirmed_count=0,
        review_count=0,
        provisional_count=0,
        rejected_count=0,
        assigned_candidate_count=0,
        unassigned_candidate_count=count,
        elapsed_seconds=0.0,
    )
    a2a_stats = GeometryFeatureStatistics(
        input_point_count=count,
        output_point_count=count,
        pca_computed_count=0,
        insufficient_neighbor_count=0,
        degenerate_covariance_count=0,
        duplicate_neighbor_coordinate_count=0,
        negative_eigenvalue_clipped_count=0,
        normal_direction_ambiguous_count=0,
        principal_direction_ambiguous_count=0,
        density_undefined_count=0,
        numeric_invalid_count=0,
        requested_k=1,
        query_batch_size=1,
        elapsed_seconds=0.0,
    )
    a2b_stats = GroundEstimationStatistics(
        input_point_count=count,
        output_point_count=count,
        total_cell_count=0,
        nonempty_cell_count=0,
        direct_control_cell_count=0,
        sparse_cell_count=0,
        empty_cell_count=0,
        interpolated_cell_count=0,
        extrapolation_too_far_cell_count=0,
        valid_ground_point_count=count,
        invalid_ground_point_count=0,
        high_gradient_cell_count=0,
        high_gradient_point_count=0,
        raw_negative_hag_count=0,
        below_ground_tolerance_count=0,
        extreme_hag_count=0,
        classification_ground_count=0,
        classification_disagreement_count=0,
        elapsed_seconds=0.0,
    )
    a1_npz, a1_json = _digest("1"), _digest("2")
    a2a_npz, a2a_json = _digest("3"), _digest("4")
    a2b_npz, a2b_json = _digest("5"), _digest("6")
    v1_npz, v1_json = _digest("7"), _digest("8")
    v2_result = VegetationInstanceResult(
        schema_version="vegetation-instances-v2",
        source_id=source_id,
        manifest_source_id=manifest_source_id,
        origin_xyz=tuple(origin),
        output_npz_filename="v2.npz",
        output_npz_sha256=_digest("9"),
        output_npz_bytes=1,
        config=VegetationInstanceConfig(source_id=source_id),
        manual_log=VegetationManualLog(),
        input_provenance={
            "v1_npz": {"sha256": v1_npz},
            "v1_json": {"sha256": v1_json},
            "a2a_npz": {"sha256": a2a_npz},
            "a2a_json": {"sha256": a2a_json},
            "a2b_npz": {"sha256": a2b_npz},
            "a2b_json": {"sha256": a2b_json},
        },
        statistics=v2_stats,
        arrays={},
        clusters=(),
        algorithm_version="v2-generalized-morphology-safe-vegetation-instances-v2",
        created_at="2026-01-01T00:00:00Z",
    )
    a2a_result = GeometryFeatureResult(
        source_id=source_id,
        manifest_source_id=manifest_source_id,
        input_a1_npz_path=Path("a1.npz"),
        input_a1_json_path=Path("a1.json"),
        input_a1_npz_sha256=a1_npz,
        input_a1_json_sha256=a1_json,
        output_npz_path=Path("a2a.npz"),
        output_json_path=Path("a2a.json"),
        output_npz_sha256=a2a_npz,
        output_npz_bytes=1,
        origin_xyz=tuple(origin),
        config=GeometryFeatureConfig(source_id=source_id),
        statistics=a2a_stats,
        arrays={},
        quality_flag_counts={},
        formulas={},
        algorithm_version="a2a-local-pca-v1",
        created_at="2026-01-01T00:00:00Z",
    )
    a2b_result = GroundEstimationResult(
        source_id=source_id,
        manifest_source_id=manifest_source_id,
        input_a1_npz_path=Path("a1.npz"),
        input_a1_json_path=Path("a1.json"),
        input_a1_npz_sha256=a1_npz,
        input_a1_json_sha256=a1_json,
        input_a2a_npz_path=Path("a2a.npz"),
        input_a2a_json_path=Path("a2a.json"),
        input_a2a_npz_sha256=a2a_npz,
        input_a2a_json_sha256=a2a_json,
        output_npz_path=Path("a2b.npz"),
        output_json_path=Path("a2b.json"),
        output_npz_sha256=a2b_npz,
        output_npz_bytes=1,
        origin_xyz=tuple(origin),
        config=GroundEstimationConfig(source_id=source_id),
        statistics=a2b_stats,
        grid={},
        arrays={},
        quality_flag_counts={},
        distributions={},
        formulas={},
        classification_validation={},
        algorithm_version="a2b-grid-quantile-idw-v1",
        created_at="2026-01-01T00:00:00Z",
    )
    return {
        "v2_result": v2_result,
        "v2_points": v2_points,
        "v2_origin_xyz": origin,
        "a2a_result": a2a_result,
        "a2a_points": a2a_points,
        "a2b_result": a2b_result,
        "a2b_points": a2b_points,
        "v2_json_sha256": _digest("a"),
        "a2a_json_sha256": a2a_json,
        "a2b_json_sha256": a2b_json,
    }


def _validate(bundle):
    return validate_tree_input_identity(**bundle)


def _assert_error(
    bundle,
    *,
    code: TreeSegmentationErrorCode,
    artifact_role: str,
    field: str,
) -> TreeSegmentationInputError:
    with pytest.raises(TreeSegmentationInputError) as raised:
        _validate(bundle)
    error = raised.value
    assert error.code is code
    assert error.artifact_role == artifact_role
    assert error.field == field
    assert not isinstance(error.expected, np.ndarray)
    assert not isinstance(error.actual, np.ndarray)
    assert "ndarray([" not in str(error)
    assert "KeyError" not in str(error)
    assert "AttributeError" not in str(error)
    assert "OverflowError" not in str(error)
    return error


def _set_result_field(bundle, result_name: str, field: str, value: object) -> None:
    """Bypass construction checks to test validator handling of loaded contracts."""
    object.__setattr__(bundle[result_name], field, value)


def test_valid_inputs_return_deterministic_identity_and_preserve_arrays() -> None:
    bundle = _bundle()
    arrays = [
        value
        for contract in (bundle["v2_points"], bundle["a2a_points"], bundle["a2b_points"])
        for value in (getattr(contract, name) for name in contract.__dataclass_fields__)
        if isinstance(value, np.ndarray)
    ] + [bundle["v2_origin_xyz"]]
    before = [(value, value.copy(), value.flags.writeable) for value in arrays]

    identity = _validate(bundle)

    expected = {
        "source_id": "scene-001",
        "manifest_source_id": "manifest-001",
        "origin_xyz": [100.0, 200.0, 300.0],
        "point_count": 3,
        "v2_algorithm_version": "v2-generalized-morphology-safe-vegetation-instances-v2",
        "a2a_algorithm_version": "a2a-local-pca-v1",
        "a2b_algorithm_version": "a2b-grid-quantile-idw-v1",
        "v2_npz_sha256": _digest("9"),
        "v2_json_sha256": _digest("a"),
        "a2a_npz_sha256": _digest("3"),
        "a2a_json_sha256": _digest("4"),
        "a2b_npz_sha256": _digest("5"),
        "a2b_json_sha256": _digest("6"),
        "a2a_input_a1_npz_sha256": _digest("1"),
        "a2a_input_a1_json_sha256": _digest("2"),
        "a2b_input_a1_npz_sha256": _digest("1"),
        "a2b_input_a1_json_sha256": _digest("2"),
        "a2b_input_a2a_npz_sha256": _digest("3"),
        "a2b_input_a2a_json_sha256": _digest("4"),
        "v2_input_v1_npz_sha256": _digest("7"),
        "v2_input_v1_json_sha256": _digest("8"),
        "v2_input_a2a_npz_sha256": _digest("3"),
        "v2_input_a2a_json_sha256": _digest("4"),
        "v2_input_a2b_npz_sha256": _digest("5"),
        "v2_input_a2b_json_sha256": _digest("6"),
    }
    assert identity.to_dict() == expected
    assert identity == _validate(bundle)
    for value, contents, writeable in before:
        assert value.flags.writeable is writeable
        assert np.array_equal(value, contents)


def test_empty_aligned_point_sets_are_valid() -> None:
    assert _validate(_bundle(0)).point_count == 0


@pytest.mark.parametrize(
    ("name", "value", "artifact_role"),
    [
        ("v2_result", object(), "v2"),
        ("v2_points", object(), "v2_points"),
        ("a2a_result", object(), "a2a"),
        ("a2a_points", object(), "a2a_points"),
        ("a2b_result", object(), "a2b"),
        ("a2b_points", object(), "a2b_points"),
    ],
)
def test_rejects_non_public_contract_types(
    name: str,
    value: object,
    artifact_role: str,
) -> None:
    bundle = _bundle()
    bundle[name] = value
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.UNSUPPORTED_STATUS,
        artifact_role=artifact_role,
        field="contract_type",
    )


@pytest.mark.parametrize(
    "origin",
    [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        np.array([1.0, np.nan, 3.0], dtype=np.float64),
    ],
)
def test_rejects_invalid_v2_origin(origin: np.ndarray) -> None:
    bundle = _bundle()
    bundle["v2_origin_xyz"] = origin
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.NONFINITE_REQUIRED_SOURCE_GEOMETRY,
        artifact_role="v2",
        field="origin_xyz",
    )


@pytest.mark.parametrize("name", ["v2_json_sha256", "a2a_json_sha256", "a2b_json_sha256"])
@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "g" * 64, 123])
def test_rejects_noncanonical_supplied_audit_digest(name: str, digest: object) -> None:
    bundle = _bundle()
    bundle[name] = digest
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        artifact_role=name.removesuffix("_json_sha256"),
        field="json_sha256",
    )


@pytest.mark.parametrize(
    ("target", "field", "value", "artifact_role"),
    [
        ("a2a_result", "source_id", "Scene-001", "a2a"),
        ("a2b_result", "source_id", " scene-001 ", "a2b"),
        ("a2a_points", "source_id", "scene-001 ", "a2a_points"),
        ("a2b_points", "source_id", "SCENE-001", "a2b_points"),
        ("a2a_result", "manifest_source_id", "Manifest-001", "a2a"),
        ("a2b_result", "manifest_source_id", " manifest-001 ", "a2b"),
        ("a2a_points", "manifest_source_id", "manifest-001 ", "a2a_points"),
        ("a2b_points", "manifest_source_id", "MANIFEST-001", "a2b_points"),
    ],
)
def test_rejects_exact_source_and_manifest_mismatch(
    target: str,
    field: str,
    value: str,
    artifact_role: str,
) -> None:
    bundle = _bundle()
    bundle[target] = replace(bundle[target], **{field: value})
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        artifact_role=artifact_role,
        field=field,
    )


@pytest.mark.parametrize(
    ("target", "artifact_role"),
    [
        ("v2_result", "v2"),
        ("a2a_result", "a2a"),
        ("a2a_points", "a2a_points"),
        ("a2b_result", "a2b"),
    ],
)
def test_rejects_exact_origin_mismatch_without_tolerance(
    target: str,
    artifact_role: str,
) -> None:
    bundle = _bundle()
    origin = bundle[target].origin_xyz
    changed = origin.copy() if isinstance(origin, np.ndarray) else list(origin)
    changed[-1] = np.nextafter(changed[-1], np.inf)
    replacement = changed if isinstance(changed, np.ndarray) else tuple(changed)
    bundle[target] = replace(bundle[target], origin_xyz=replacement)
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.COORDINATE_MISMATCH,
        artifact_role=artifact_role,
        field="origin_xyz",
    )


@pytest.mark.parametrize(
    ("point_name", "result_name", "artifact_role"),
    [
        ("a2a_points", "a2a_result", "a2a_points"),
        ("a2b_points", "a2b_result", "a2b_points"),
    ],
)
def test_rejects_actual_point_set_count_mismatch(
    point_name: str,
    result_name: str,
    artifact_role: str,
) -> None:
    bundle = _bundle()
    smaller = _bundle(2)
    bundle[point_name] = smaller[point_name]
    bundle[result_name] = replace(bundle[result_name], statistics=smaller[result_name].statistics)
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        artifact_role=artifact_role,
        field="point_count",
    )


@pytest.mark.parametrize(
    ("result_name", "artifact_role", "statistics_field"),
    [
        ("v2_result", "v2", "input_point_count"),
        ("a2a_result", "a2a", "input_point_count"),
        ("a2b_result", "a2b", "input_point_count"),
        ("a2a_result", "a2a", "output_point_count"),
        ("a2b_result", "a2b", "output_point_count"),
    ],
)
def test_rejects_statistics_count_mismatch(
    result_name: str,
    artifact_role: str,
    statistics_field: str,
) -> None:
    bundle = _bundle()
    statistics = replace(bundle[result_name].statistics, **{statistics_field: 4})
    bundle[result_name] = replace(bundle[result_name], statistics=statistics)
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        artifact_role=artifact_role,
        field=f"statistics.{statistics_field}",
    )


@pytest.mark.parametrize("field", ["original_index", "sample_index"])
@pytest.mark.parametrize(
    ("target", "artifact_role"), [("a2a_points", "a2a_points"), ("a2b_points", "a2b_points")]
)
def test_rejects_row_order_permutation(field: str, target: str, artifact_role: str) -> None:
    bundle = _bundle()
    values = getattr(bundle[target], field)
    before = values.copy()
    values[:2] = values[1::-1]
    assert not np.array_equal(values, before)
    assert np.array_equal(np.sort(values), np.sort(before))
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.INVALID_INDEXES,
        artifact_role=artifact_role,
        field=field,
    )


def test_rejects_v2_a2a_coordinate_alignment_change() -> None:
    bundle = _bundle()
    bundle["a2a_points"].xyz_local[0, 0] = np.nextafter(np.float32(0.0), np.float32(1.0))
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.COORDINATE_MISMATCH,
        artifact_role="a2a_points",
        field="xyz_local",
    )


@pytest.mark.parametrize(
    ("result_name", "field"),
    [
        ("a2b_result", "input_a1_npz_sha256"),
        ("a2b_result", "input_a1_json_sha256"),
        ("a2b_result", "input_a2a_npz_sha256"),
        ("a2b_result", "input_a2a_json_sha256"),
    ],
)
def test_rejects_a2b_provenance_mismatch(result_name: str, field: str) -> None:
    bundle = _bundle()
    bundle[result_name] = replace(bundle[result_name], **{field: _digest("b")})
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        artifact_role="a2b",
        field=field,
    )


@pytest.mark.parametrize("name", ["a2a_npz", "a2a_json", "a2b_npz", "a2b_json"])
def test_rejects_v2_provenance_mismatch(name: str) -> None:
    bundle = _bundle()
    provenance = dict(bundle["v2_result"].input_provenance)
    provenance[name] = {"sha256": _digest("b")}
    bundle["v2_result"] = replace(bundle["v2_result"], input_provenance=provenance)
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        artifact_role="v2",
        field=f"input_provenance.{name}.sha256",
    )


def test_rejects_missing_v2_provenance_field() -> None:
    bundle = _bundle()
    provenance = dict(bundle["v2_result"].input_provenance)
    provenance.pop("v1_npz")
    bundle["v2_result"] = replace(bundle["v2_result"], input_provenance=provenance)
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.MISSING_REQUIRED_FIELD,
        artifact_role="v2",
        field="input_provenance.v1_npz.sha256",
    )


def test_rejects_non_mapping_v2_provenance_container() -> None:
    bundle = _bundle()
    malformed = object()
    _set_result_field(bundle, "v2_result", "input_provenance", malformed)
    error = _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.MISSING_REQUIRED_FIELD,
        artifact_role="v2",
        field="input_provenance",
    )
    assert repr(malformed) not in str(error)


def test_rejects_non_mapping_v2_provenance_group() -> None:
    bundle = _bundle()
    bundle["v2_result"].input_provenance["v1_npz"] = object()
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.MISSING_REQUIRED_FIELD,
        artifact_role="v2",
        field="input_provenance.v1_npz.sha256",
    )


@pytest.mark.parametrize(
    ("result_name", "artifact_role"),
    [
        ("v2_result", "v2"),
        ("a2a_result", "a2a"),
        ("a2b_result", "a2b"),
    ],
)
@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "g" * 64, 123])
def test_rejects_invalid_output_npz_digest(
    result_name: str,
    artifact_role: str,
    digest: object,
) -> None:
    bundle = _bundle()
    _set_result_field(bundle, result_name, "output_npz_sha256", digest)
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        artifact_role=artifact_role,
        field="output_npz_sha256",
    )


@pytest.mark.parametrize(
    ("result_name", "field", "artifact_role", "digest"),
    [
        ("a2a_result", "input_a1_npz_sha256", "a2a", "A" * 64),
        ("a2b_result", "input_a1_json_sha256", "a2b", "g" * 64),
        ("a2b_result", "input_a2a_npz_sha256", "a2b", 123),
    ],
)
def test_rejects_invalid_upstream_result_provenance_digest(
    result_name: str,
    field: str,
    artifact_role: str,
    digest: object,
) -> None:
    bundle = _bundle()
    _set_result_field(bundle, result_name, field, digest)
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        artifact_role=artifact_role,
        field=field,
    )


@pytest.mark.parametrize("digest", ["A" * 64, "g" * 64, 123])
def test_rejects_invalid_v2_provenance_digest(digest: object) -> None:
    bundle = _bundle()
    bundle["v2_result"].input_provenance["v1_npz"]["sha256"] = digest
    _assert_error(
        bundle,
        code=TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        artifact_role="v2",
        field="input_provenance.v1_npz.sha256",
    )


def test_identity_validator_has_no_loading_or_membership_scope() -> None:
    validator = tree_inputs.validate_tree_input_identity
    source = inspect.getsource(validator)
    parsed = ast.parse(source)
    calls = [node for node in ast.walk(parsed) if isinstance(node, ast.Call)]
    call_names = {node.func.id for node in calls if isinstance(node.func, ast.Name)} | {
        node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
    }

    assert not any(
        isinstance(node.func, ast.Name) and node.func.id in {"open", "hash"} for node in calls
    )
    assert not any(
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr == "load"
        for node in calls
    )
    assert not any(
        isinstance(node.func, ast.Attribute) and node.func.attr in {"read_text", "read_bytes"}
        for node in calls
    )
    assert "TreeSegmentationInputSet" not in call_names
    assert not {"cluster", "member", "eligibility"} & call_names

    assert get_type_hints(validator)["return"] is TreeSegmentationInputIdentity
    parameters = inspect.signature(validator).parameters
    assert not any("path" in name for name in parameters)
    assert not {"reporter", "progress", "cancellation_token", "cancel"} & set(parameters)
    assert not {"cluster", "member", "eligibility"} & set(parameters)
