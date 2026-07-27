from __future__ import annotations

import numpy as np
import pytest
from core.contracts import (
    TREE_SEGMENTATION_ALGORITHM_VERSION,
    TREE_SEGMENTATION_SCHEMA_VERSION,
    TreeInstanceRecord,
    TreeSegmentationConfig,
    TreeSegmentationErrorCode,
    TreeSegmentationQualityFlag,
    TreeSegmentationResult,
    TreeSegmentationStatistics,
    TreeSegmentationStatus,
    VegetationClusterStatus,
)


def _record(
    tree_id: int = -1,
    status: TreeSegmentationStatus = TreeSegmentationStatus.REVIEW_REQUIRED,
    source_ids: tuple[int, ...] = (3,),
    source_statuses: tuple[VegetationClusterStatus, ...] = (VegetationClusterStatus.CONFIRMED,),
    source_ready: tuple[bool, ...] = (True,),
    tree_ready: bool = False,
    confidence: float = 0.5,
    quality_flags: TreeSegmentationQualityFlag = TreeSegmentationQualityFlag.NONE,
) -> TreeInstanceRecord:
    return TreeInstanceRecord(
        tree_id,
        status,
        source_ids,
        source_statuses,
        source_ready,
        tree_ready,
        confidence,
        quality_flags,
    )


def _statistics(
    records: tuple[TreeInstanceRecord, ...], point_count: int
) -> TreeSegmentationStatistics:
    return TreeSegmentationStatistics(
        point_count,
        len(records),
        sum(record.status is TreeSegmentationStatus.CONFIRMED_TREE for record in records),
        sum(record.status is TreeSegmentationStatus.REVIEW_REQUIRED for record in records),
        sum(record.status is TreeSegmentationStatus.PROVISIONAL for record in records),
        sum(record.status is TreeSegmentationStatus.REJECTED for record in records),
        sum(record.status is TreeSegmentationStatus.UNRESOLVED for record in records),
        0.0,
    )


def _result(
    records: tuple[TreeInstanceRecord, ...] = (),
    ids: np.ndarray | None = None,
    ready: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    members: np.ndarray | None = None,
    *,
    config: TreeSegmentationConfig | None = None,
    statistics: TreeSegmentationStatistics | None = None,
    source_id: object = "synthetic",
    manifest_source_id: object = "manifest",
    origin_xyz: object = (0.0, 0.0, 0.0),
    warnings: tuple[object, ...] = (),
    limitations: tuple[object, ...] = (),
) -> TreeSegmentationResult:
    if ids is None:
        ids = np.full(4, -1, np.int32)
    if ready is None:
        ready = np.zeros(len(ids), np.bool_)
    if offsets is None:
        offsets = np.zeros(len(records) + 1, np.int64)
    if members is None:
        members = np.empty(0, np.int64)
    return TreeSegmentationResult(
        TREE_SEGMENTATION_SCHEMA_VERSION,
        TREE_SEGMENTATION_ALGORITHM_VERSION,
        source_id,
        manifest_source_id,
        origin_xyz,
        config or TreeSegmentationConfig(),
        statistics or _statistics(records, len(ids)),
        records,
        ids,
        ready,
        offsets,
        members,
        {"v2": {"sha256": "a" * 64}},
        {"frame": "LOCAL_CARTESIAN_XYZ", "unit": "m"},
        warnings,
        limitations,
    )


def test_public_enums_and_exports_are_frozen() -> None:
    assert [int(value) for value in TreeSegmentationStatus] == [0, 1, 2, 3, 4]
    assert TreeSegmentationErrorCode.CANCELLED.value == "V3_CANCELLED"
    assert TreeSegmentationQualityFlag.CROSS_CLUSTER_REVIEW == 64


def test_empty_result_and_nonconfirmed_roundtrip() -> None:
    empty = _result(ids=np.empty(0, np.int32), ready=np.empty(0, np.bool_))
    assert TreeSegmentationResult.from_dict(empty.to_dict()).to_dict() == empty.to_dict()
    records = (
        _record(status=TreeSegmentationStatus.REVIEW_REQUIRED),
        _record(status=TreeSegmentationStatus.PROVISIONAL, source_ids=(4,)),
        _record(status=TreeSegmentationStatus.REJECTED, source_ids=(5,)),
        _record(status=TreeSegmentationStatus.UNRESOLVED, source_ids=(6,)),
    )
    result = _result(
        records,
        offsets=np.array([0, 1, 2, 3, 4], np.int64),
        members=np.arange(4, dtype=np.int64),
    )
    assert TreeSegmentationResult.from_dict(result.to_dict()).to_dict() == result.to_dict()


def test_confirmed_tree_ids_start_at_zero_and_are_continuous() -> None:
    first = _record(0, TreeSegmentationStatus.CONFIRMED_TREE, tree_ready=True)
    second = _record(1, TreeSegmentationStatus.CONFIRMED_TREE, source_ids=(4,), tree_ready=True)
    result = _result(
        (second, first),
        np.array([0, 0, 1, 1], np.int32),
        np.ones(4, np.bool_),
        np.array([0, 2, 4], np.int64),
        np.array([2, 3, 0, 1], np.int64),
    )
    assert [record.tree_id for record in result.records] == [0, 1]
    assert np.array_equal(result.ordered_tree_member_row_index, np.array([0, 1, 2, 3], np.int64))
    canonical = _result(
        (first, second),
        np.array([0, 0, 1, 1], np.int32),
        np.ones(4, np.bool_),
        np.array([0, 2, 4], np.int64),
        np.array([0, 1, 2, 3], np.int64),
    )
    assert result.to_dict() == canonical.to_dict()
    with pytest.raises(ValueError, match="continuous"):
        _result(
            (_record(1, TreeSegmentationStatus.CONFIRMED_TREE),),
            np.array([-1, 1, -1, -1], np.int32),
            offsets=np.array([0, 1], np.int64),
            members=np.array([1], np.int64),
        )


def test_statistics_must_exactly_match_records() -> None:
    review = _record()
    with pytest.raises(ValueError, match="record_count"):
        _result(
            (review,),
            statistics=TreeSegmentationStatistics(4, 0, 0, 0, 0, 0, 0, 0.0),
        )
    with pytest.raises(ValueError, match="review_required_count"):
        _result(
            (review,),
            statistics=TreeSegmentationStatistics(4, 1, 0, 0, 1, 0, 0, 0.0),
        )
    assert _result((review,)).statistics.record_count == 1


@pytest.mark.parametrize(
    "status",
    (
        TreeSegmentationStatus.REVIEW_REQUIRED,
        TreeSegmentationStatus.PROVISIONAL,
        TreeSegmentationStatus.REJECTED,
        TreeSegmentationStatus.UNRESOLVED,
    ),
)
def test_nonconfirmed_status_cannot_receive_tree_identity_or_readiness(
    status: TreeSegmentationStatus,
) -> None:
    with pytest.raises(ValueError, match="only confirmed"):
        _record(0, status)
    with pytest.raises(ValueError, match="requires CONFIRMED"):
        _record(-1, status, tree_ready=True)


def test_confirmed_sources_and_cross_cluster_policy_are_frozen() -> None:
    with pytest.raises(ValueError, match="V2 confirmed"):
        _record(
            0,
            TreeSegmentationStatus.CONFIRMED_TREE,
            source_statuses=(VegetationClusterStatus.REVIEW_REQUIRED,),
        )
    with pytest.raises(ValueError, match="distance-ready"):
        _record(0, TreeSegmentationStatus.CONFIRMED_TREE, source_ready=(False,))
    cross = _record(
        0,
        TreeSegmentationStatus.CONFIRMED_TREE,
        source_ids=(3, 4),
        source_statuses=(VegetationClusterStatus.CONFIRMED, VegetationClusterStatus.CONFIRMED),
        source_ready=(True, True),
    )
    kwargs = {
        "records": (cross,),
        "ids": np.array([-1, 0, -1, -1], np.int32),
        "offsets": np.array([0, 1], np.int64),
        "members": np.array([1], np.int64),
    }
    with pytest.raises(ValueError, match="cross-cluster"):
        _result(**kwargs)
    assert _result(
        **kwargs,
        config=TreeSegmentationConfig(allow_cross_cluster_confirmation=True),
    ).records == (cross,)


def test_record_rejects_invalid_source_order_and_nonfinite_confidence() -> None:
    with pytest.raises(ValueError, match="sorted"):
        _record(
            source_ids=(4, 3),
            source_statuses=(VegetationClusterStatus.CONFIRMED,) * 2,
            source_ready=(True,) * 2,
        )
    with pytest.raises(ValueError, match="finite"):
        _record(confidence=np.nan)
    with pytest.raises(ValueError, match="align"):
        _record(
            source_statuses=(
                VegetationClusterStatus.CONFIRMED,
                VegetationClusterStatus.CONFIRMED,
            )
        )


def test_record_enum_and_flag_values_reject_bool_before_conversion() -> None:
    with pytest.raises(ValueError, match="status"):
        _record(status=False)
    with pytest.raises(ValueError, match="source_cluster_statuses"):
        _record(source_statuses=(False,))
    with pytest.raises(ValueError, match="quality flags"):
        _record(quality_flags=False)
    payload = _record().to_dict()
    for field, value in (
        ("status", False),
        ("source_cluster_statuses", [False]),
        ("quality_flags", False),
    ):
        invalid = dict(payload)
        invalid[field] = value
        with pytest.raises(ValueError):
            TreeInstanceRecord.from_dict(invalid)
    valid_json = _record().to_dict()
    assert valid_json["status"] == int(TreeSegmentationStatus.REVIEW_REQUIRED)
    assert valid_json["source_cluster_statuses"] == [int(VegetationClusterStatus.CONFIRMED)]
    assert valid_json["quality_flags"] == int(TreeSegmentationQualityFlag.NONE)
    assert TreeInstanceRecord.from_dict(valid_json) == _record()


@pytest.mark.parametrize(
    "origin_xyz",
    (
        (True, 0, 0),
        ("0", 0, 0),
        "000",
        ((0, 0), 0, 0),
        (np.nan, 0, 0),
        (np.inf, 0, 0),
        (0, 0),
    ),
)
def test_origin_xyz_is_strictly_numeric_finite_and_three_dimensional(origin_xyz: object) -> None:
    with pytest.raises(ValueError, match="origin_xyz"):
        _result(origin_xyz=origin_xyz)
    assert _result(origin_xyz=(1, 2.5, np.float32(3))).origin_xyz == (1.0, 2.5, 3.0)


@pytest.mark.parametrize("offsets,members,error", [
    (np.array([0, 2, 1], np.int64), np.array([0], np.int64), "monotonic"),
    (np.array([0, 1], np.int64), np.array([4], np.int64), "outside"),
])
def test_member_offsets_and_indexes_are_strict(
    offsets: np.ndarray, members: np.ndarray, error: str
) -> None:
    records = (_record(),) if len(offsets) == 2 else (_record(), _record(source_ids=(4,)))
    with pytest.raises(ValueError, match=error):
        _result(records, offsets=offsets, members=members)


def test_confirmed_members_cannot_overlap_or_disagree_with_point_arrays() -> None:
    first = _record(0, TreeSegmentationStatus.CONFIRMED_TREE)
    second = _record(1, TreeSegmentationStatus.CONFIRMED_TREE, source_ids=(4,))
    with pytest.raises(ValueError, match="share"):
        _result(
            (first, second),
            np.array([0, 1, -1, -1], np.int32),
            offsets=np.array([0, 1, 3], np.int64),
            members=np.array([0, 0, 1], np.int64),
        )
    with pytest.raises(ValueError, match="exactly match"):
        _result(
            (first,),
            np.array([0, -1, -1, -1], np.int32),
            offsets=np.array([0, 2], np.int64),
            members=np.array([0, 1], np.int64),
        )


def test_confirmed_member_readiness_must_match_each_member_row() -> None:
    not_ready = _record(0, TreeSegmentationStatus.CONFIRMED_TREE, tree_ready=False)
    ready = _record(0, TreeSegmentationStatus.CONFIRMED_TREE, tree_ready=True)
    kwargs = {
        "ids": np.array([0, 0, -1, -1], np.int32),
        "offsets": np.array([0, 2], np.int64),
        "members": np.array([0, 1], np.int64),
    }
    assert not _result(
        (not_ready,), ready=np.zeros(4, np.bool_), **kwargs
    ).tree_distance_ready_by_point.any()
    assert _result(
        (ready,), ready=np.array([True, True, False, False], np.bool_), **kwargs
    ).tree_distance_ready_by_point[:2].all()
    with pytest.raises(ValueError, match="readiness"):
        _result(
            (not_ready,), ready=np.array([True, False, False, False], np.bool_), **kwargs
        )
    with pytest.raises(ValueError, match="readiness"):
        _result(
            (ready,), ready=np.array([True, False, False, False], np.bool_), **kwargs
        )


def test_point_alignment_dtype_nonfinite_and_input_immutability() -> None:
    ids = np.array([-1, -1, -1, -1], np.int32)
    ready = np.zeros(4, np.bool_)
    offsets = np.array([0], np.int64)
    members = np.empty(0, np.int64)
    before = (ids.copy(), ready.copy(), offsets.copy(), members.copy())
    _result(ids=ids, ready=ready, offsets=offsets, members=members)
    assert all(
        np.array_equal(value, snapshot)
        for value, snapshot in zip((ids, ready, offsets, members), before, strict=True)
    )
    with pytest.raises(ValueError, match="dtype"):
        _result(ids=np.array([-1] * 4, object))
    with pytest.raises(ValueError, match="finite"):
        _record(confidence=float("inf"))
    with pytest.raises(ValueError, match="shape"):
        _result(ids=np.full(4, -1, np.int32), ready=np.zeros(3, np.bool_))


def test_config_roundtrip_and_illegal_values() -> None:
    config = TreeSegmentationConfig()
    assert TreeSegmentationConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="positive integer"):
        TreeSegmentationConfig(maximum_seeds=0)
    with pytest.raises(ValueError, match="positive"):
        TreeSegmentationConfig(deterministic_quantization_m=0.0)
    with pytest.raises(ValueError, match="bool"):
        TreeSegmentationConfig(allow_cross_cluster_confirmation=1)


@pytest.mark.parametrize(
    "field,value,error",
    (
        ("ordered_tree_member_row_index", [1.5], "integer"),
        ("tree_id_by_point", ["0", -1, -1, -1], "integer"),
        ("tree_distance_ready_by_point", [0, 0, 0, 0], "bool"),
        ("tree_id_by_point", [True, -1, -1, -1], "integer"),
        ("tree_member_offsets", [True], "integer"),
        ("tree_id_by_point", [[-1], [-1], [-1], [-1]], "integer"),
    ),
)
def test_from_dict_rejects_non_json_array_types(
    field: str, value: object, error: str
) -> None:
    payload = _result().to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match=error):
        TreeSegmentationResult.from_dict(payload)


@pytest.mark.parametrize(
    "field,value",
    (
        ("source_id", 1),
        ("manifest_source_id", True),
        ("source_id", None),
        ("manifest_source_id", ""),
    ),
)
def test_source_identifiers_and_messages_are_strict(field: str, value: object) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match="source identifiers"):
        _result(**kwargs)
    with pytest.raises(ValueError, match="strings"):
        _result(warnings=(1,))
    with pytest.raises(ValueError, match="strings"):
        _result(limitations=(object(),))


def test_messages_require_nonstring_sequences_of_strings() -> None:
    for field, value in (
        ("warnings", "abc"),
        ("limitations", "abc"),
        ("warnings", b"abc"),
    ):
        with pytest.raises(ValueError, match="non-string sequence"):
            _result(**{field: value})
    result = _result(warnings=["warning"])
    assert result.warnings == ("warning",)
