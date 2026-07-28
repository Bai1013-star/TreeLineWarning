from __future__ import annotations

from dataclasses import replace
from typing import Any

import algorithms.vegetation.tree_inputs as tree_inputs
import numpy as np
import pytest
from algorithms.vegetation.tree_inputs import (
    TreeSegmentationInputError,
    build_tree_segmentation_input_set,
)
from core.contracts.features import PointFeatureSet
from core.contracts.ground import GroundQualityFlag, HeightAboveGroundSet
from core.contracts.tree_inputs import (
    TreeSegmentationInputDisposition,
    TreeSegmentationInputIdentity,
    TreeSegmentationInputSet,
)
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


def _freeze_contract_arrays(contract: Any) -> None:
    for name in contract.__dataclass_fields__:
        value = getattr(contract, name)
        if isinstance(value, np.ndarray):
            value.setflags(write=False)


def _identity(point_count: int) -> TreeSegmentationInputIdentity:
    return TreeSegmentationInputIdentity(
        source_id="scene-001",
        manifest_source_id="manifest-001",
        origin_xyz=(1.0, 2.0, 3.0),
        point_count=point_count,
        v2_algorithm_version="v2",
        a2a_algorithm_version="a2a",
        a2b_algorithm_version="a2b",
        **{
            name: _digest("a")
            for name in TreeSegmentationInputIdentity.__dataclass_fields__
            if name.endswith("sha256")
        },
    )


def _points(count: int) -> tuple[PointFeatureSet, HeightAboveGroundSet]:
    original = np.arange(100, 100 + count, dtype=np.int64)
    sample = np.arange(count, dtype=np.int64)
    floats = np.ones(count, dtype=np.float32)
    geometry = PointFeatureSet(
        source_id="scene-001",
        manifest_source_id="manifest-001",
        xyz_local=np.arange(count * 3, dtype=np.float32).reshape((-1, 3)),
        origin_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float64),
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
    quality = np.zeros(count, dtype=np.uint16)
    if count:
        quality[0] = int(GroundQualityFlag.INVALID_GROUND_ESTIMATE)
    heights = HeightAboveGroundSet(
        source_id="scene-001",
        manifest_source_id="manifest-001",
        original_index=original.copy(),
        sample_index=sample.copy(),
        ground_z_m=floats.copy(),
        raw_height_agl_m=floats.copy(),
        height_agl_m=floats.copy(),
        ground_cell_id=np.zeros(count, dtype=np.int32),
        ground_distance_m=floats.copy(),
        quality_flags=quality,
    )
    _freeze_contract_arrays(geometry)
    _freeze_contract_arrays(heights)
    return geometry, heights


def _bundle() -> tuple[
    VegetationInstanceResult,
    dict[str, np.ndarray],
    PointFeatureSet,
    HeightAboveGroundSet,
    TreeSegmentationInputIdentity,
]:
    count = 6
    statuses = np.array(
        [
            VegetationClusterStatus.CONFIRMED,
            VegetationClusterStatus.REVIEW_REQUIRED,
            VegetationClusterStatus.PROVISIONAL,
            VegetationClusterStatus.REJECTED_AS_INSTANCE,
        ],
        dtype=np.uint8,
    )
    cluster_ids = np.array([9, 2, 7, 4], dtype=np.int32)
    members = np.array([2, 0, 1, 3, 4], dtype=np.int64)
    offsets = np.array([0, 2, 3, 4, 5], dtype=np.int64)
    point_cluster = np.array([9, 2, 9, 7, 4, -1], dtype=np.int32)
    point_status = np.array(
        [
            VegetationClusterStatus.CONFIRMED,
            VegetationClusterStatus.REVIEW_REQUIRED,
            VegetationClusterStatus.CONFIRMED,
            VegetationClusterStatus.PROVISIONAL,
            VegetationClusterStatus.REJECTED_AS_INSTANCE,
            VegetationClusterStatus.UNASSIGNED,
        ],
        dtype=np.uint8,
    )
    vegetation = np.array([0, -1, 0, -1, -1, -1], dtype=np.int32)
    floats = np.ones(count, dtype=np.float32)
    zeros8 = np.zeros(count, dtype=np.uint8)
    arrays = {
        "original_index": np.arange(100, 100 + count, dtype=np.int64),
        "sample_index": np.arange(count, dtype=np.int64),
        "xyz_local": np.arange(count * 3, dtype=np.float32).reshape((-1, 3)),
        "wire_id_by_point": np.full(count, -1, dtype=np.int32),
        "vegetation_rule_score": floats.copy(),
        "vegetation_confidence": floats.copy(),
        "automatic_vegetation_level": zeros8.copy(),
        "vegetation_level": zeros8.copy(),
        "hag_state": zeros8.copy(),
        "local_scale_raw_m": floats.copy(),
        "local_scale_m": floats.copy(),
        "local_scale_state": zeros8.copy(),
        "cluster_id_by_point": point_cluster,
        "vegetation_id_by_point": vegetation,
        "cluster_status_by_point": point_status,
        "distance_ready_by_point": vegetation >= 0,
        "assignment_confidence": floats.copy(),
        "assignment_reason_flags": np.zeros(count, dtype=np.uint16),
        "assignment_quality_flags": np.zeros(count, dtype=np.uint16),
        "manual_provenance_flags": np.zeros(count, dtype=np.uint16),
        "cluster_id": cluster_ids,
        "cluster_status": statuses,
        "cluster_vegetation_id": np.array([0, -1, -1, -1], dtype=np.int32),
        "cluster_member_offsets": offsets,
        "ordered_cluster_member_row_index": members,
    }
    summaries = tuple(
        {
            "cluster_id": int(cluster_ids[position]),
            "point_count": int(offsets[position + 1] - offsets[position]),
            "status": VegetationClusterStatus(int(statuses[position])).name,
            "vegetation_id": int(arrays["cluster_vegetation_id"][position]),
            "requires_review": bool(statuses[position] == VegetationClusterStatus.REVIEW_REQUIRED),
        }
        for position in range(len(cluster_ids))
    )
    statistics = VegetationInstanceStatistics(
        input_point_count=count,
        candidate_count=count,
        core_point_count=count,
        cluster_count=4,
        confirmed_count=1,
        review_count=1,
        provisional_count=1,
        rejected_count=1,
        assigned_candidate_count=5,
        unassigned_candidate_count=1,
        elapsed_seconds=0.0,
    )
    result = VegetationInstanceResult(
        schema_version="vegetation-instances-v2",
        source_id="scene-001",
        manifest_source_id="manifest-001",
        origin_xyz=(1.0, 2.0, 3.0),
        output_npz_filename="v2.npz",
        output_npz_sha256=_digest("b"),
        output_npz_bytes=1,
        config=VegetationInstanceConfig(source_id="scene-001"),
        manual_log=VegetationManualLog(),
        input_provenance={},
        statistics=statistics,
        arrays={},
        clusters=summaries,
        algorithm_version="v2-generalized-morphology-safe-vegetation-instances-v2",
        created_at="2026-01-01T00:00:00Z",
    )
    geometry, heights = _points(count)
    return result, arrays, geometry, heights, _identity(count)


def _build(bundle=None) -> TreeSegmentationInputSet:
    return build_tree_segmentation_input_set(*(bundle or _bundle()))


def _assert_error(
    bundle,
    code: TreeSegmentationErrorCode,
    field: str,
    artifact_role: str = "v2",
) -> None:
    with pytest.raises(TreeSegmentationInputError) as raised:
        _build(bundle)
    assert raised.value.code is code
    assert raised.value.artifact_role == artifact_role
    assert raised.value.field == field
    assert not isinstance(raised.value.expected, np.ndarray)
    assert not isinstance(raised.value.actual, np.ndarray)
    assert "ndarray([" not in str(raised.value)
    assert not any(
        name in str(raised.value)
        for name in (
            "KeyError",
            "IndexError",
            "AttributeError",
            "TypeError",
            "ValueError",
            "OverflowError",
        )
    )


def test_builds_sorted_partitions_preserves_member_order_and_freezes_v2() -> None:
    result, arrays, geometry, heights, identity = _bundle()
    retained = list(arrays.values())
    identities = {name: id(value) for name, value in arrays.items()}
    geometry_flags = [
        value.flags.writeable
        for value in geometry.__dict__.values()
        if isinstance(value, np.ndarray)
    ]
    height_flags = [
        value.flags.writeable
        for value in heights.__dict__.values()
        if isinstance(value, np.ndarray)
    ]

    output = _build((result, arrays, geometry, heights, identity))

    assert isinstance(output, TreeSegmentationInputSet)
    assert output.eligible_cluster_ids == (9,)
    assert output.review_only_cluster_ids == (2, 7)
    assert output.excluded_cluster_ids == (4,)
    assert [record.cluster_id for record in output.cluster_dispositions] == [2, 4, 7, 9]
    assert [record.disposition for record in output.cluster_dispositions] == [
        TreeSegmentationInputDisposition.REVIEW_ONLY,
        TreeSegmentationInputDisposition.EXCLUDED,
        TreeSegmentationInputDisposition.REVIEW_ONLY,
        TreeSegmentationInputDisposition.ELIGIBLE,
    ]
    assert output.ordered_cluster_member_row_index.tolist() == [2, 0, 1, 3, 4]
    assert output.ground_usable_by_point.tolist() == [False, True, True, True, True, True]
    assert output.usable_ground_member_count_by_cluster.tolist() == [1, 1, 1, 1]
    for name in output.vegetation_points.__dataclass_fields__:
        assert getattr(output.vegetation_points, name) is arrays[name]
    for name in (
        "cluster_id",
        "cluster_status",
        "cluster_vegetation_id",
        "cluster_member_offsets",
        "ordered_cluster_member_row_index",
    ):
        assert getattr(output, name) is arrays[name]
    assert output.geometry_features is geometry
    assert output.heights_above_ground is heights
    assert all(not value.flags.writeable for value in retained)
    assert all(id(arrays[name]) == value for name, value in identities.items())
    assert [
        value.flags.writeable
        for value in geometry.__dict__.values()
        if isinstance(value, np.ndarray)
    ] == geometry_flags
    assert [
        value.flags.writeable
        for value in heights.__dict__.values()
        if isinstance(value, np.ndarray)
    ] == height_flags
    assert not output.ground_usable_by_point.flags.writeable
    assert not output.usable_ground_member_count_by_cluster.flags.writeable


@pytest.mark.parametrize(
    "field",
    [
        "original_index",
        "xyz_local",
        "assignment_quality_flags",
        "cluster_id",
        "cluster_status",
        "cluster_vegetation_id",
        "cluster_member_offsets",
        "ordered_cluster_member_row_index",
    ],
)
def test_missing_required_arrays_are_stable(field: str) -> None:
    bundle = list(_bundle())
    bundle[1].pop(field)
    _assert_error(tuple(bundle), TreeSegmentationErrorCode.MISSING_REQUIRED_FIELD, field)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_index", [0, 1]),
        ("vegetation_confidence", np.ones(6, dtype=np.float64)),
        ("xyz_local", np.ones(18, dtype=np.float32)),
        ("cluster_member_offsets", np.array([0, 2, 3, 4, 5], dtype=np.int32)),
    ],
)
def test_array_dtype_rank_and_type_are_not_coerced(field: str, value: object) -> None:
    bundle = list(_bundle())
    bundle[1][field] = value
    _assert_error(tuple(bundle), TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT, field)


def test_non_mapping_v2_arrays_are_rejected() -> None:
    result, _arrays, geometry, heights, identity = _bundle()
    _assert_error(
        (result, object(), geometry, heights, identity),
        TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        "arrays",
    )


@pytest.mark.parametrize(
    ("mutation", "code", "field"),
    [
        (
            lambda arrays: arrays["cluster_id"].__setitem__(0, -1),
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "cluster_id",
        ),
        (
            lambda arrays: arrays["cluster_id"].__setitem__(1, 9),
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "cluster_id",
        ),
        (
            lambda arrays: arrays["cluster_member_offsets"].__setitem__(0, 1),
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "cluster_member_offsets",
        ),
        (
            lambda arrays: arrays["cluster_member_offsets"].__setitem__(2, 1),
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "cluster_member_offsets",
        ),
        (
            lambda arrays: arrays["ordered_cluster_member_row_index"].__setitem__(0, -1),
            TreeSegmentationErrorCode.INVALID_INDEXES,
            "ordered_cluster_member_row_index",
        ),
        (
            lambda arrays: arrays["ordered_cluster_member_row_index"].__setitem__(0, 5),
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "ordered_cluster_member_row_index",
        ),
        (
            lambda arrays: arrays["cluster_member_offsets"].__setitem__(1, 0),
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "cluster_member_offsets",
        ),
    ],
)
def test_cluster_and_member_integrity_is_fatal(
    mutation, code: TreeSegmentationErrorCode, field: str
) -> None:
    bundle = list(_bundle())
    mutation(bundle[1])
    _assert_error(tuple(bundle), code, field)


@pytest.mark.parametrize(
    ("members", "offsets"),
    [
        ([2, 0, 2, 1, 3, 4], [0, 3, 4, 5, 6]),
        ([2, 0, 2, 1, 3, 4], [0, 2, 4, 5, 6]),
    ],
)
def test_same_and_cross_cluster_duplicate_members_are_rejected(
    members: list[int], offsets: list[int]
) -> None:
    bundle = list(_bundle())
    arrays = bundle[1]
    arrays["ordered_cluster_member_row_index"] = np.asarray(members, dtype=np.int64)
    arrays["cluster_member_offsets"] = np.asarray(offsets, dtype=np.int64)
    summaries = list(bundle[0].clusters)
    for position, summary in enumerate(summaries):
        summary["point_count"] = offsets[position + 1] - offsets[position]
    bundle[0] = replace(bundle[0], clusters=tuple(summaries))
    _assert_error(
        tuple(bundle),
        TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
        "ordered_cluster_member_row_index",
    )


@pytest.mark.parametrize(
    ("members", "offsets"),
    [
        ([0, 1, 3, 4], [0, 1, 2, 3, 4]),
        ([2, 0, 1, 3, 5], [0, 2, 3, 4, 5]),
    ],
)
def test_member_coverage_rejects_omitted_or_unassigned_rows(
    members: list[int], offsets: list[int]
) -> None:
    bundle = list(_bundle())
    bundle[1]["ordered_cluster_member_row_index"] = np.asarray(members, dtype=np.int64)
    bundle[1]["cluster_member_offsets"] = np.asarray(offsets, dtype=np.int64)
    _assert_error(
        tuple(bundle),
        TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
        "ordered_cluster_member_row_index",
    )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("cluster_status", 99, TreeSegmentationErrorCode.UNSUPPORTED_STATUS),
        ("cluster_status_by_point", 99, TreeSegmentationErrorCode.UNSUPPORTED_STATUS),
        ("cluster_vegetation_id", -1, TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP),
        ("vegetation_id_by_point", -1, TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP),
        (
            "distance_ready_by_point",
            False,
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
        ),
    ],
)
def test_status_and_assignment_semantics_are_verified(
    field: str, value: object, code: TreeSegmentationErrorCode
) -> None:
    bundle = list(_bundle())
    index = 0 if field.startswith("cluster_") else 0
    bundle[1][field][index] = value
    expected_field = (
        "cluster_status"
        if field == "cluster_status"
        else "cluster_status_by_point"
        if field == "cluster_status_by_point"
        else "cluster_vegetation_id"
        if field == "cluster_vegetation_id"
        else "confirmed_member_assignment"
    )
    _assert_error(tuple(bundle), code, expected_field)


def test_unassigned_rows_and_summary_and_statistics_mismatches_are_fatal() -> None:
    bundle = list(_bundle())
    bundle[1]["distance_ready_by_point"][5] = True
    _assert_error(
        tuple(bundle),
        TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
        "unassigned_points",
    )

    bundle = list(_bundle())
    summaries = list(bundle[0].clusters)
    summaries[0] = {**summaries[0], "point_count": 1}
    bundle[0] = replace(bundle[0], clusters=tuple(summaries))
    _assert_error(
        tuple(bundle),
        TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
        "clusters[].point_count",
    )

    bundle = list(_bundle())
    bundle[0] = replace(bundle[0], statistics=replace(bundle[0].statistics, rejected_count=0))
    _assert_error(
        tuple(bundle),
        TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
        "statistics.rejected_count",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("point_count", True),
        ("point_count", 1.0),
        ("point_count", np.array([1])),
        ("point_count", np.array([1, 2])),
        ("status", np.array(["CONFIRMED"])),
        ("vegetation_id", False),
        ("vegetation_id", 0.0),
        ("vegetation_id", np.array([0])),
        ("requires_review", np.bool_(True)),
        ("requires_review", 1),
    ],
)
def test_summary_scalars_require_strict_json_types(field: str, value: object) -> None:
    bundle = list(_bundle())
    bundle[0].clusters[0][field] = value
    _assert_error(
        tuple(bundle),
        TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
        f"clusters[].{field}",
    )


@pytest.mark.parametrize(
    ("mutation", "code", "field"),
    [
        (
            lambda result: object.__setattr__(result, "clusters", result.clusters[:-1]),
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "clusters",
        ),
        (
            lambda result: object.__setattr__(result, "clusters", (object(), *result.clusters[1:])),
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "clusters[]",
        ),
        (
            lambda result: result.clusters[0].pop("status"),
            TreeSegmentationErrorCode.MISSING_REQUIRED_FIELD,
            "clusters[].status",
        ),
        (
            lambda result: result.clusters[1].__setitem__(
                "cluster_id", result.clusters[0]["cluster_id"]
            ),
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "clusters[].cluster_id",
        ),
        (
            lambda result: result.clusters[0].__setitem__("status", "PROVISIONAL"),
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "clusters[].status",
        ),
        (
            lambda result: result.clusters[0].__setitem__("vegetation_id", -1),
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "clusters[].vegetation_id",
        ),
        (
            lambda result: result.clusters[0].__setitem__("requires_review", True),
            TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
            "clusters[].requires_review",
        ),
    ],
)
def test_summary_structure_and_content_mismatches_are_fatal(
    mutation, code: TreeSegmentationErrorCode, field: str
) -> None:
    bundle = list(_bundle())
    mutation(bundle[0])
    _assert_error(
        tuple(bundle),
        code,
        field,
    )


def test_empty_and_all_unassigned_inputs_are_legal() -> None:
    result, arrays, _geometry, _heights, _identity_value = _bundle()
    for name, value in list(arrays.items()):
        if name == "xyz_local":
            arrays[name] = np.empty((0, 3), dtype=value.dtype)
        elif name.startswith("cluster_") or name == "ordered_cluster_member_row_index":
            arrays[name] = np.empty(0, dtype=value.dtype)
        else:
            arrays[name] = np.empty(0, dtype=value.dtype)
    arrays["cluster_member_offsets"] = np.array([0], dtype=np.int64)
    empty_stats = replace(
        result.statistics,
        input_point_count=0,
        candidate_count=0,
        core_point_count=0,
        cluster_count=0,
        confirmed_count=0,
        review_count=0,
        provisional_count=0,
        rejected_count=0,
        assigned_candidate_count=0,
        unassigned_candidate_count=0,
    )
    geometry, heights = _points(0)
    output = _build(
        (
            replace(result, statistics=empty_stats, clusters=()),
            arrays,
            geometry,
            heights,
            _identity(0),
        )
    )
    assert output.eligible_cluster_ids == ()
    assert output.ground_usable_by_point.shape == (0,)
    assert output.usable_ground_member_count_by_cluster.shape == (0,)

    result, arrays, geometry, heights, identity = _bundle()
    for name in ("cluster_id", "cluster_status", "cluster_vegetation_id"):
        arrays[name] = np.empty(0, dtype=arrays[name].dtype)
    arrays["cluster_member_offsets"] = np.array([0], dtype=np.int64)
    arrays["ordered_cluster_member_row_index"] = np.empty(0, dtype=np.int64)
    arrays["cluster_id_by_point"].fill(-1)
    arrays["cluster_status_by_point"].fill(int(VegetationClusterStatus.UNASSIGNED))
    arrays["vegetation_id_by_point"].fill(-1)
    arrays["distance_ready_by_point"].fill(False)
    statistics = replace(
        result.statistics,
        cluster_count=0,
        confirmed_count=0,
        review_count=0,
        provisional_count=0,
        rejected_count=0,
        assigned_candidate_count=0,
        unassigned_candidate_count=6,
    )
    output = _build(
        (replace(result, statistics=statistics, clusters=()), arrays, geometry, heights, identity)
    )
    assert output.eligible_cluster_ids == ()
    assert output.review_only_cluster_ids == ()
    assert output.excluded_cluster_ids == ()


def test_early_failure_is_atomic() -> None:
    bundle = list(_bundle())
    bundle[1]["cluster_member_offsets"][0] = 1
    before = {name: (value.copy(), value.flags.writeable) for name, value in bundle[1].items()}
    _assert_error(
        tuple(bundle),
        TreeSegmentationErrorCode.INCONSISTENT_CLUSTER_MEMBERSHIP,
        "cluster_member_offsets",
    )
    for name, (contents, writeable) in before.items():
        assert np.array_equal(bundle[1][name], contents)
        assert bundle[1][name].flags.writeable is writeable


def test_late_inputset_failure_restores_v2_writeability(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = list(_bundle())
    arrays = bundle[1]
    arrays["cluster_id"].setflags(write=False)
    before = {name: (value.copy(), value.flags.writeable) for name, value in arrays.items()}

    def fail_input_set(**_kwargs: object) -> TreeSegmentationInputSet:
        raise ValueError("forced construction failure")

    monkeypatch.setattr(tree_inputs, "TreeSegmentationInputSet", fail_input_set)
    _assert_error(
        tuple(bundle),
        TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
        "TreeSegmentationInputSet",
        artifact_role="v3_inputs",
    )
    for name, (contents, writeable) in before.items():
        assert np.array_equal(arrays[name], contents)
        assert arrays[name].flags.writeable is writeable
