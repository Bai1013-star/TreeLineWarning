from __future__ import annotations

import ast
import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

import algorithms.vegetation.tree_inputs as tree_inputs
import numpy as np
import pytest
from algorithms.features.geometry import GeometryFeatureError
from algorithms.features.ground import GroundEstimationError
from algorithms.vegetation.instances import VegetationInstanceError
from algorithms.vegetation.tree_inputs import (
    TreeSegmentationInputError,
    load_tree_segmentation_input_set,
)
from core.contracts.errors import UnifiedError
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
    GroundQualityFlag,
    HeightAboveGroundSet,
)
from core.contracts.tree_inputs import TreeSegmentationInputSet
from core.contracts.vegetation_instances import (
    VegetationClusterStatus,
    VegetationInstanceConfig,
    VegetationInstanceResult,
    VegetationInstanceStatistics,
    VegetationManualLog,
)
from core.contracts.vegetation_trees import TreeSegmentationErrorCode


def _digest(character: str) -> str:
    return character * 64


def _freeze_contract_arrays(contract: object) -> None:
    for name in contract.__dataclass_fields__:
        value = getattr(contract, name)
        if isinstance(value, np.ndarray):
            value.setflags(write=False)


def _paths(tmp_path: Path, *, uppercase: bool = False) -> tuple[Path, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    suffixes = (".NPZ", ".JSON") if uppercase else (".npz", ".json")
    paths = (
        tmp_path / f"v2{suffixes[0]}",
        tmp_path / f"v2{suffixes[1]}",
        tmp_path / f"a2a{suffixes[0]}",
        tmp_path / f"a2a{suffixes[1]}",
        tmp_path / f"a2b{suffixes[0]}",
        tmp_path / f"a2b{suffixes[1]}",
    )
    for index, path in enumerate(paths):
        path.write_bytes(
            f'{{"artifact": {index}}}'.encode() if index % 2 else f"artifact-{index}".encode()
        )
    return paths


def _contracts(paths: tuple[Path, ...]) -> dict[str, object]:
    count = 3
    source_id = "scene-001"
    manifest_source_id = "manifest-001"
    origin = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    v2_json_sha256 = hashlib.sha256(paths[1].read_bytes()).hexdigest()
    a2a_json_sha256 = hashlib.sha256(paths[3].read_bytes()).hexdigest()
    a2b_json_sha256 = hashlib.sha256(paths[5].read_bytes()).hexdigest()
    original = np.arange(100, 100 + count, dtype=np.int64)
    sample = np.arange(count, dtype=np.int64)
    xyz = np.arange(count * 3, dtype=np.float32).reshape((count, 3))
    floats = np.ones(count, dtype=np.float32)
    zero8 = np.zeros(count, dtype=np.uint8)
    point_status = np.array(
        [
            VegetationClusterStatus.CONFIRMED,
            VegetationClusterStatus.CONFIRMED,
            VegetationClusterStatus.UNASSIGNED,
        ],
        dtype=np.uint8,
    )
    arrays = {
        "original_index": original,
        "sample_index": sample,
        "xyz_local": xyz,
        "wire_id_by_point": np.full(count, -1, dtype=np.int32),
        "vegetation_rule_score": floats.copy(),
        "vegetation_confidence": floats.copy(),
        "automatic_vegetation_level": zero8.copy(),
        "vegetation_level": zero8.copy(),
        "hag_state": zero8.copy(),
        "local_scale_raw_m": floats.copy(),
        "local_scale_m": floats.copy(),
        "local_scale_state": zero8.copy(),
        "cluster_id_by_point": np.array([7, 7, -1], dtype=np.int32),
        "vegetation_id_by_point": np.array([0, 0, -1], dtype=np.int32),
        "cluster_status_by_point": point_status,
        "distance_ready_by_point": np.array([True, True, False], dtype=np.bool_),
        "assignment_confidence": floats.copy(),
        "assignment_reason_flags": np.zeros(count, dtype=np.uint16),
        "assignment_quality_flags": np.zeros(count, dtype=np.uint16),
        "manual_provenance_flags": np.zeros(count, dtype=np.uint16),
        "origin_xyz": origin,
        "cluster_id": np.array([7], dtype=np.int32),
        "cluster_status": np.array([VegetationClusterStatus.CONFIRMED], dtype=np.uint8),
        "cluster_vegetation_id": np.array([0], dtype=np.int32),
        "cluster_member_offsets": np.array([0, 2], dtype=np.int64),
        "ordered_cluster_member_row_index": np.array([0, 1], dtype=np.int64),
    }
    v2_statistics = VegetationInstanceStatistics(
        input_point_count=count,
        candidate_count=count,
        core_point_count=count,
        cluster_count=1,
        confirmed_count=1,
        review_count=0,
        provisional_count=0,
        rejected_count=0,
        assigned_candidate_count=2,
        unassigned_candidate_count=1,
        elapsed_seconds=0.0,
    )
    v2_result = VegetationInstanceResult(
        schema_version="vegetation-instances-v2",
        source_id=source_id,
        manifest_source_id=manifest_source_id,
        origin_xyz=tuple(origin),
        output_npz_filename="v2.npz",
        output_npz_sha256=_digest("b"),
        output_npz_bytes=1,
        config=VegetationInstanceConfig(source_id=source_id),
        manual_log=VegetationManualLog(),
        input_provenance={
            "v1_npz": {"sha256": _digest("e")},
            "v1_json": {"sha256": _digest("f")},
            "a2a_npz": {"sha256": _digest("c")},
            "a2a_json": {"sha256": a2a_json_sha256},
            "a2b_npz": {"sha256": _digest("d")},
            "a2b_json": {"sha256": a2b_json_sha256},
        },
        statistics=v2_statistics,
        arrays={},
        clusters=(
            {
                "cluster_id": 7,
                "point_count": 2,
                "status": "CONFIRMED",
                "vegetation_id": 0,
                "requires_review": False,
            },
        ),
        algorithm_version="v2-generalized-morphology-safe-vegetation-instances-v2",
        created_at="2026-01-01T00:00:00Z",
    )
    geometry = PointFeatureSet(
        source_id=source_id,
        manifest_source_id=manifest_source_id,
        xyz_local=xyz.copy(),
        origin_xyz=origin.copy(),
        original_index=original.copy(),
        sample_index=sample.copy(),
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
        quality_flags=np.zeros(count, dtype=np.uint16),
    )
    heights = HeightAboveGroundSet(
        source_id=source_id,
        manifest_source_id=manifest_source_id,
        original_index=original.copy(),
        sample_index=sample.copy(),
        ground_z_m=floats.copy(),
        raw_height_agl_m=floats.copy(),
        height_agl_m=floats.copy(),
        ground_cell_id=np.zeros(count, dtype=np.int32),
        ground_distance_m=floats.copy(),
        quality_flags=np.array(
            [GroundQualityFlag.INVALID_GROUND_ESTIMATE, 0, 0], dtype=np.uint16
        ),
    )
    _freeze_contract_arrays(geometry)
    _freeze_contract_arrays(heights)
    geometry_statistics = GeometryFeatureStatistics(
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
    ground_statistics = GroundEstimationStatistics(
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
    a2a_result = GeometryFeatureResult(
        source_id=source_id,
        manifest_source_id=manifest_source_id,
        input_a1_npz_path=Path("a1.npz"),
        input_a1_json_path=Path("a1.json"),
        input_a1_npz_sha256=_digest("e"),
        input_a1_json_sha256=_digest("f"),
        output_npz_path=Path("a2a.npz"),
        output_json_path=Path("a2a.json"),
        output_npz_sha256=_digest("c"),
        output_npz_bytes=1,
        origin_xyz=tuple(origin),
        config=GeometryFeatureConfig(source_id=source_id),
        statistics=geometry_statistics,
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
        input_a1_npz_sha256=_digest("e"),
        input_a1_json_sha256=_digest("f"),
        input_a2a_npz_path=Path("a2a.npz"),
        input_a2a_json_path=Path("a2a.json"),
        input_a2a_npz_sha256=_digest("c"),
        input_a2a_json_sha256=a2a_json_sha256,
        output_npz_path=Path("a2b.npz"),
        output_json_path=Path("a2b.json"),
        output_npz_sha256=_digest("d"),
        output_npz_bytes=1,
        origin_xyz=tuple(origin),
        config=GroundEstimationConfig(source_id=source_id),
        statistics=ground_statistics,
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
        "arrays": arrays,
        "a2a_result": a2a_result,
        "geometry": geometry,
        "a2b_result": a2b_result,
        "heights": heights,
        "v2_json_sha256": v2_json_sha256,
        "a2a_json_sha256": a2a_json_sha256,
        "a2b_json_sha256": a2b_json_sha256,
    }


def _patch_loaders(
    monkeypatch: pytest.MonkeyPatch,
    contracts: dict[str, object],
    calls: list[tuple[str, Path, Path]],
) -> None:
    def load_v2(
        npz_path: Path, json_path: Path
    ) -> tuple[VegetationInstanceResult, dict[str, np.ndarray]]:
        calls.append(("v2", npz_path, json_path))
        return contracts["v2_result"], contracts["arrays"]

    def load_a2a(
        npz_path: Path, json_path: Path
    ) -> tuple[GeometryFeatureResult, PointFeatureSet]:
        calls.append(("a2a", npz_path, json_path))
        return contracts["a2a_result"], contracts["geometry"]

    def load_a2b(
        npz_path: Path, json_path: Path
    ) -> tuple[GroundEstimationResult, object, HeightAboveGroundSet]:
        calls.append(("a2b", npz_path, json_path))
        return contracts["a2b_result"], object(), contracts["heights"]

    monkeypatch.setattr(tree_inputs, "load_vegetation_instances", load_v2)
    monkeypatch.setattr(tree_inputs, "load_geometry_features", load_a2a)
    monkeypatch.setattr(tree_inputs, "load_ground_estimation", load_a2b)


@pytest.mark.parametrize("as_strings", [False, True])
def test_loads_explicit_paths_hashes_json_and_calls_b2_b3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, as_strings: bool
) -> None:
    paths = _paths(tmp_path)
    contracts = _contracts(paths)
    calls: list[tuple[str, Path, Path]] = []
    _patch_loaders(monkeypatch, contracts, calls)
    identity_calls = 0
    builder_calls = 0
    original_identity = tree_inputs.validate_tree_input_identity
    original_builder = tree_inputs.build_tree_segmentation_input_set

    def validate(*args: object, **kwargs: str):
        nonlocal identity_calls
        identity_calls += 1
        return original_identity(*args, **kwargs)

    def build(*args: object):
        nonlocal builder_calls
        builder_calls += 1
        return original_builder(*args)

    monkeypatch.setattr(tree_inputs, "validate_tree_input_identity", validate)
    monkeypatch.setattr(tree_inputs, "build_tree_segmentation_input_set", build)
    arguments = tuple(str(path) for path in paths) if as_strings else paths
    result = load_tree_segmentation_input_set(*arguments)

    assert isinstance(result, TreeSegmentationInputSet)
    assert calls == [
        ("v2", paths[0], paths[1]),
        ("a2a", paths[2], paths[3]),
        ("a2b", paths[4], paths[5]),
    ]
    assert identity_calls == builder_calls == 1
    assert result.eligible_cluster_ids == (7,)
    assert result.usable_ground_member_count_by_cluster.tolist() == [1]
    assert result.identity.v2_json_sha256 == contracts["v2_json_sha256"]
    assert result.identity.a2a_json_sha256 == contracts["a2a_json_sha256"]
    assert result.identity.a2b_json_sha256 == contracts["a2b_json_sha256"]


@pytest.mark.parametrize(
    ("filename", "suffix", "field"),
    [
        ("artifact.NPZ", ".npz", "npz"),
        ("artifact.JSON", ".json", "json"),
    ],
)
def test_artifact_path_suffix_check_is_case_insensitive(
    tmp_path: Path,
    filename: str,
    suffix: str,
    field: str,
) -> None:
    path = tmp_path / filename
    path.write_text("{}", encoding="utf-8")

    assert tree_inputs._artifact_path(path, suffix, "v2", field) == path


@pytest.mark.parametrize("index", range(6))
def test_rejects_wrong_suffix_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int
) -> None:
    paths = list(_paths(tmp_path))
    wrong = tmp_path / f"wrong-{index}.txt"
    wrong.write_text("x", encoding="utf-8")
    paths[index] = wrong
    calls: list[tuple[str, Path, Path]] = []
    _patch_loaders(monkeypatch, _contracts(_paths(tmp_path / "contracts")), calls)

    with pytest.raises(TreeSegmentationInputError) as raised:
        load_tree_segmentation_input_set(*paths)

    assert raised.value.code is TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT
    artifact_roles = ("v2", "v2", "a2a", "a2a", "a2b", "a2b")
    assert raised.value.artifact_role == artifact_roles[index]
    assert raised.value.field == (
        "npz_suffix",
        "json_suffix",
        "npz_suffix",
        "json_suffix",
        "npz_suffix",
        "json_suffix",
    )[index]
    assert calls == []


@pytest.mark.parametrize("index", range(6))
def test_rejects_missing_explicit_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index: int,
) -> None:
    paths = list(_paths(tmp_path))
    paths[index].unlink()
    calls: list[tuple[str, Path, Path]] = []
    _patch_loaders(monkeypatch, _contracts(_paths(tmp_path / "contracts")), calls)

    with pytest.raises(TreeSegmentationInputError) as raised:
        load_tree_segmentation_input_set(*paths)

    assert raised.value.code is TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT
    artifact_roles = ("v2", "v2", "a2a", "a2a", "a2b", "a2b")
    assert raised.value.artifact_role == artifact_roles[index]
    assert raised.value.field == (
        "npz_path",
        "json_path",
        "npz_path",
        "json_path",
        "npz_path",
        "json_path",
    )[index]
    assert not isinstance(raised.value, FileNotFoundError)
    assert calls == []


@pytest.mark.parametrize(
    ("role", "exception_type"),
    [
        ("v2", VegetationInstanceError),
        ("a2a", GeometryFeatureError),
        ("a2b", GroundEstimationError),
    ],
)
def test_wraps_exact_public_loader_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    exception_type: type[Exception],
) -> None:
    paths = _paths(tmp_path)
    contracts = _contracts(paths)
    calls: list[tuple[str, Path, Path]] = []
    _patch_loaders(monkeypatch, contracts, calls)
    failure = exception_type(
        UnifiedError(
            code="UPSTREAM_BAD",
            message="compact failure",
            severity="error",
            module="tests",
            recoverable=True,
            details={},
        )
    )

    def fail(*_args: object):
        raise failure

    loader_names = {
        "v2": "load_vegetation_instances",
        "a2a": "load_geometry_features",
        "a2b": "load_ground_estimation",
    }
    monkeypatch.setattr(tree_inputs, loader_names[role], fail)
    with pytest.raises(TreeSegmentationInputError) as raised:
        load_tree_segmentation_input_set(*paths)

    assert raised.value.code is TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT
    assert raised.value.artifact_role == role
    assert raised.value.field == "load"
    assert raised.value.actual == {"type": exception_type.__name__, "code": "UPSTREAM_BAD"}


def test_wraps_json_hash_io_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    calls: list[tuple[str, Path, Path]] = []
    _patch_loaders(monkeypatch, _contracts(paths), calls)

    def fail_hash(path: Path) -> str:
        if path == paths[3]:
            raise OSError("simulated")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(tree_inputs, "_sha256_file", fail_hash)
    with pytest.raises(TreeSegmentationInputError) as raised:
        load_tree_segmentation_input_set(*paths)

    assert raised.value.code is TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT
    assert raised.value.artifact_role == "a2a"
    assert raised.value.field == "json_sha256"
    assert not isinstance(raised.value, OSError)
    assert [role for role, _npz, _json in calls] == ["v2", "a2a", "a2b"]


def test_identity_and_membership_errors_propagate_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    contracts = _contracts(paths)
    calls: list[tuple[str, Path, Path]] = []
    _patch_loaders(monkeypatch, contracts, calls)

    def load_mismatched_a2a(
        *_args: object,
    ) -> tuple[GeometryFeatureResult, PointFeatureSet]:
        result = replace(contracts["a2a_result"], source_id="other")
        return result, contracts["geometry"]

    monkeypatch.setattr(
        tree_inputs,
        "load_geometry_features",
        load_mismatched_a2a,
    )
    with pytest.raises(TreeSegmentationInputError) as identity_error:
        load_tree_segmentation_input_set(*paths)
    assert identity_error.value.artifact_role == "a2a"
    assert identity_error.value.field == "source_id"

    contracts = _contracts(paths)
    contracts["arrays"]["ordered_cluster_member_row_index"][1] = 0
    calls.clear()
    _patch_loaders(monkeypatch, contracts, calls)
    with pytest.raises(TreeSegmentationInputError) as membership_error:
        load_tree_segmentation_input_set(*paths)
    assert (
        membership_error.value.code
        is TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP
    )
    assert membership_error.value.field == "ordered_cluster_member_row_index"


def test_success_preserves_v2_array_objects_and_only_freezes_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    contracts = _contracts(paths)
    calls: list[tuple[str, Path, Path]] = []
    _patch_loaders(monkeypatch, contracts, calls)
    geometry_flags = [
        value.flags.writeable
        for value in contracts["geometry"].__dict__.values()
        if isinstance(value, np.ndarray)
    ]
    height_flags = [
        value.flags.writeable
        for value in contracts["heights"].__dict__.values()
        if isinstance(value, np.ndarray)
    ]
    arrays = contracts["arrays"]
    result = load_tree_segmentation_input_set(*paths)

    assert result.vegetation_points.xyz_local is arrays["xyz_local"]
    assert result.cluster_id is arrays["cluster_id"]
    assert all(
        not value.flags.writeable for name, value in arrays.items() if name != "origin_xyz"
    )
    assert arrays["origin_xyz"].flags.writeable
    assert [
        value.flags.writeable
        for value in contracts["geometry"].__dict__.values()
        if isinstance(value, np.ndarray)
    ] == geometry_flags
    assert [
        value.flags.writeable
        for value in contracts["heights"].__dict__.values()
        if isinstance(value, np.ndarray)
    ] == height_flags


def test_path_loader_does_not_use_broad_exception_handler() -> None:
    module = ast.parse(inspect.getsource(tree_inputs))
    loader = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "load_tree_segmentation_input_set"
    )
    handlers = [
        node
        for node in ast.walk(loader)
        if isinstance(node, ast.ExceptHandler)
    ]
    assert not any(
        handler.type is None
        or (
            isinstance(handler.type, ast.Name)
            and handler.type.id == "Exception"
        )
        for handler in handlers
    )
