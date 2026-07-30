from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType

import core.contracts.vegetation_trees as vegetation_trees
import numpy as np
import pytest
from core.contracts import (
    TREE_SEGMENTATION_ALGORITHM_VERSION,
    TREE_SEGMENTATION_SCHEMA_VERSION,
    TreeCrownSummary,
    TreeGeometrySummary,
    TreeInstanceRecord,
    TreeSegmentationConfig,
    TreeSegmentationErrorCode,
    TreeSegmentationQualityFlag,
    TreeSegmentationResult,
    TreeSegmentationStatistics,
    TreeSegmentationStatus,
    VegetationClusterStatus,
)


def _crown(
    *,
    crown_xy_bounds_local: object = (0.0, 0.0, 4.0, 4.0),
    crown_area_m2: object = 4.0,
    crown_equivalent_radius_m: object = math.sqrt(4.0 / math.pi),
) -> TreeCrownSummary:
    return TreeCrownSummary(
        crown_xy_bounds_local=crown_xy_bounds_local,
        crown_area_m2=crown_area_m2,
        crown_equivalent_radius_m=crown_equivalent_radius_m,
    )


def _geometry(
    *,
    apex_xyz_local: object = (1.0, 2.0, 12.0),
    maximum_hag_m: object = 10.0,
    robust_height_hag_m: object = 9.5,
    vertical_extent_m: object = 9.0,
) -> TreeGeometrySummary:
    return TreeGeometrySummary(
        apex_xyz_local=apex_xyz_local,
        maximum_hag_m=maximum_hag_m,
        robust_height_hag_m=robust_height_hag_m,
        vertical_extent_m=vertical_extent_m,
    )


_DEFAULT_CROWN = _crown()
_DEFAULT_GEOMETRY = _geometry()


def _frozen_array(value: object, dtype: object | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True, subok=False)
    array.setflags(write=False)
    return array


def _result_array(value: np.ndarray) -> np.ndarray:
    if (
        type(value) is np.ndarray
        and value.flags.owndata
        and value.flags.c_contiguous
        and not value.flags.writeable
    ):
        return value
    return _frozen_array(value)


_RESULT_ARRAY_FIELDS = (
    "segment_id_by_point",
    "tree_id_by_point",
    "tree_distance_ready_by_point",
    "tree_member_offsets",
    "ordered_tree_member_row_index",
)


def _record(
    tree_id: int = -1,
    status: TreeSegmentationStatus = TreeSegmentationStatus.REVIEW_REQUIRED,
    segment_id: int = 0,
    source_ids: tuple[int, ...] = (3,),
    source_statuses: tuple[VegetationClusterStatus, ...] = (VegetationClusterStatus.CONFIRMED,),
    source_ready: tuple[bool, ...] = (True,),
    tree_ready: bool = False,
    confidence: float = 0.5,
    geometry_summary: TreeGeometrySummary | None = _DEFAULT_GEOMETRY,
    crown_summary: TreeCrownSummary | None = _DEFAULT_CROWN,
    quality_flags: TreeSegmentationQualityFlag = TreeSegmentationQualityFlag.NONE,
) -> TreeInstanceRecord:
    return TreeInstanceRecord(
        tree_id=tree_id,
        segment_id=segment_id,
        status=status,
        source_cluster_ids=source_ids,
        source_cluster_statuses=source_statuses,
        source_cluster_distance_ready=source_ready,
        tree_distance_ready=tree_ready,
        confidence=confidence,
        geometry_summary=geometry_summary,
        crown_summary=crown_summary,
        quality_flags=quality_flags,
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
    segment_ids: np.ndarray | None = None,
    config: TreeSegmentationConfig | None = None,
    statistics: TreeSegmentationStatistics | None = None,
    source_id: object = "synthetic",
    manifest_source_id: object = "manifest",
    origin_xyz: object = (0.0, 0.0, 0.0),
    warnings: tuple[object, ...] = (),
    limitations: tuple[object, ...] = (),
    freeze_arrays: bool = True,
) -> TreeSegmentationResult:
    if ids is None:
        ids = np.full(4, -1, np.int32)
    if ready is None:
        ready = np.zeros(len(ids), np.bool_)
    if offsets is None:
        offsets = np.zeros(len(records) + 1, np.int64)
    if members is None:
        members = np.empty(0, np.int64)
    if segment_ids is None:
        segment_ids = np.full(len(ids), -1, np.int32)
        for index, record in enumerate(records):
            if index + 1 >= len(offsets):
                break
            start, end = offsets[index : index + 2]
            rows = members[start:end]
            valid_rows = rows[(rows >= 0) & (rows < len(segment_ids))]
            segment_ids[valid_rows] = record.segment_id
    if freeze_arrays:
        segment_ids, ids, ready, offsets, members = (
            _result_array(value) for value in (segment_ids, ids, ready, offsets, members)
        )
    return TreeSegmentationResult(
        TREE_SEGMENTATION_SCHEMA_VERSION,
        TREE_SEGMENTATION_ALGORITHM_VERSION,
        source_id,
        manifest_source_id,
        origin_xyz,
        config or TreeSegmentationConfig(),
        statistics or _statistics(records, len(ids)),
        records,
        segment_ids,
        ids,
        ready,
        offsets,
        members,
        {"v2": {"sha256": "a" * 64}},
        {"frame": "LOCAL_CARTESIAN_XYZ", "unit": "m"},
        warnings,
        limitations,
    )


def _confirmed_result(
    record: TreeInstanceRecord,
    *,
    config: TreeSegmentationConfig | None = None,
) -> TreeSegmentationResult:
    return _result(
        (record,),
        ids=np.array([record.tree_id, -1, -1, -1], np.int32),
        ready=np.array([record.tree_distance_ready, False, False, False], np.bool_),
        offsets=np.array([0, 1], np.int64),
        members=np.array([0], np.int64),
        config=config,
    )


def test_public_enums_and_exports_are_frozen() -> None:
    assert [int(value) for value in TreeSegmentationStatus] == [0, 1, 2, 3, 4]
    assert TreeSegmentationErrorCode.CANCELLED.value == "V3_CANCELLED"
    assert TreeSegmentationQualityFlag.CROSS_CLUSTER_REVIEW == 64
    assert TreeCrownSummary.__name__ == "TreeCrownSummary"


def test_crown_summary_is_frozen_canonical_hashable_and_json_roundtrippable() -> None:
    bounds = [np.int32(1), 2.5, np.float64(5), 6]
    area = 8.0
    radius = math.sqrt(area / math.pi)
    summary = _crown(
        crown_xy_bounds_local=bounds,
        crown_area_m2=np.int64(area),
        crown_equivalent_radius_m=np.float64(radius),
    )
    expected = TreeCrownSummary((1.0, 2.5, 5.0, 6.0), 8.0, radius)
    assert summary == expected
    assert hash(summary) == hash(expected)
    assert summary.crown_xy_bounds_local == (1.0, 2.5, 5.0, 6.0)
    assert all(type(value) is float for value in summary.crown_xy_bounds_local)
    assert type(summary.crown_area_m2) is float
    assert type(summary.crown_equivalent_radius_m) is float
    assert summary.to_dict() == {
        "crown_xy_bounds_local": [1.0, 2.5, 5.0, 6.0],
        "crown_area_m2": 8.0,
        "crown_equivalent_radius_m": radius,
    }
    payload = json.loads(json.dumps(summary.to_dict()))
    assert TreeCrownSummary.from_dict(payload) == summary
    bounds[0] = 99
    assert summary.crown_xy_bounds_local == (1.0, 2.5, 5.0, 6.0)
    with pytest.raises(FrozenInstanceError):
        summary.crown_area_m2 = 1.0


@pytest.mark.parametrize("numeric", (0, 0.0, np.int32(0), np.float32(0.0)))
def test_crown_summary_accepts_python_and_numpy_numeric_scalars(numeric: object) -> None:
    summary = _crown(
        crown_xy_bounds_local=(numeric, numeric, numeric, numeric),
        crown_area_m2=numeric,
        crown_equivalent_radius_m=numeric,
    )
    assert summary.crown_xy_bounds_local == (0.0, 0.0, 0.0, 0.0)
    assert type(summary.crown_area_m2) is float
    assert type(summary.crown_equivalent_radius_m) is float


@pytest.mark.parametrize(
    "field,value",
    (
        ("crown_area_m2", True),
        ("crown_area_m2", np.bool_(False)),
        ("crown_area_m2", "1"),
        ("crown_area_m2", [1]),
        ("crown_area_m2", np.nan),
        ("crown_area_m2", np.inf),
        ("crown_area_m2", -np.inf),
        ("crown_equivalent_radius_m", True),
        ("crown_equivalent_radius_m", np.bool_(False)),
        ("crown_equivalent_radius_m", "1"),
        ("crown_equivalent_radius_m", {"value": 1}),
        ("crown_equivalent_radius_m", np.nan),
        ("crown_equivalent_radius_m", np.inf),
        ("crown_equivalent_radius_m", -np.inf),
    ),
)
def test_crown_summary_rejects_invalid_scalars(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _crown(**{field: value})


@pytest.mark.parametrize(
    "bounds",
    (
        (1, 2, 3),
        (1, 2, 3, 4, 5),
        "1234",
        b"1234",
        {"min_x": 1, "min_y": 2, "max_x": 3, "max_y": 4},
        ((1,), 2, 3, 4),
        (True, 2, 3, 4),
        (np.bool_(False), 2, 3, 4),
        (np.nan, 2, 3, 4),
        (np.inf, 2, 3, 4),
        (-np.inf, 2, 3, 4),
        np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64),
    ),
)
def test_crown_summary_rejects_invalid_bounds(bounds: object) -> None:
    with pytest.raises(ValueError, match="crown_xy_bounds_local"):
        _crown(crown_xy_bounds_local=bounds)


def test_crown_bounds_normalize_sequences_and_allow_degenerate_extents() -> None:
    as_list = [0, 1, 0, 5]
    list_summary = _crown(
        crown_xy_bounds_local=as_list,
        crown_area_m2=0,
        crown_equivalent_radius_m=0,
    )
    assert list_summary.crown_xy_bounds_local == (0.0, 1.0, 0.0, 5.0)
    as_array = np.array([2, 3, 7, 3], dtype=np.int32)
    array_summary = _crown(
        crown_xy_bounds_local=as_array,
        crown_area_m2=0,
        crown_equivalent_radius_m=0,
    )
    assert array_summary.crown_xy_bounds_local == (2.0, 3.0, 7.0, 3.0)
    point = _crown(
        crown_xy_bounds_local=(4, 5, 4, 5),
        crown_area_m2=0,
        crown_equivalent_radius_m=0,
    )
    assert point.crown_xy_bounds_local == (4.0, 5.0, 4.0, 5.0)
    as_list[0] = -100
    as_array[0] = -100
    assert list_summary.crown_xy_bounds_local[0] == 0.0
    assert array_summary.crown_xy_bounds_local[0] == 2.0


@pytest.mark.parametrize(
    "bounds",
    (
        (2, 0, 1, 1),
        (0, 2, 1, 1),
    ),
)
def test_crown_summary_rejects_reversed_bounds(bounds: object) -> None:
    with pytest.raises(ValueError, match="minima"):
        _crown(crown_xy_bounds_local=bounds)


def test_crown_area_must_fit_within_bounds_with_documented_tolerance() -> None:
    equal = _crown(
        crown_area_m2=16.0,
        crown_equivalent_radius_m=math.sqrt(16.0 / math.pi),
    )
    assert equal.crown_area_m2 == 16.0
    below = _crown(
        crown_area_m2=3.0,
        crown_equivalent_radius_m=math.sqrt(3.0 / math.pi),
    )
    assert below.crown_area_m2 == 3.0
    within_tolerance = 16.0 + 1e-9
    tolerated = _crown(
        crown_area_m2=within_tolerance,
        crown_equivalent_radius_m=math.sqrt(within_tolerance / math.pi),
    )
    assert tolerated.crown_area_m2 == within_tolerance
    materially_greater = 16.0 + 1e-6
    with pytest.raises(ValueError, match="bounding-box area"):
        _crown(
            crown_area_m2=materially_greater,
            crown_equivalent_radius_m=math.sqrt(materially_greater / math.pi),
        )
    with pytest.raises(ValueError, match="at least"):
        _crown(crown_area_m2=-1, crown_equivalent_radius_m=0)
    with pytest.raises(ValueError, match="bounding-box area"):
        _crown(
            crown_xy_bounds_local=(0, 0, 0, 1),
            crown_area_m2=1,
            crown_equivalent_radius_m=math.sqrt(1 / math.pi),
        )


@pytest.mark.parametrize(
    "bounds",
    (
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 2.0, 1.0, 2.0),
    ),
)
def test_degenerate_crown_bounds_require_exactly_zero_area(
    bounds: tuple[float, float, float, float],
) -> None:
    tiny_area = 1e-13
    with pytest.raises(ValueError, match="bounding-box area"):
        _crown(
            crown_xy_bounds_local=bounds,
            crown_area_m2=tiny_area,
            crown_equivalent_radius_m=math.sqrt(tiny_area / math.pi),
        )
    zero = _crown(
        crown_xy_bounds_local=bounds,
        crown_area_m2=0.0,
        crown_equivalent_radius_m=0.0,
    )
    assert zero.crown_area_m2 == 0.0
    assert zero.crown_equivalent_radius_m == 0.0


def test_crown_equivalent_radius_is_strictly_consistent_with_area() -> None:
    area = 5.0
    expected = math.sqrt(area / math.pi)
    assert (
        _crown(
            crown_area_m2=area,
            crown_equivalent_radius_m=expected,
        ).crown_equivalent_radius_m
        == expected
    )
    tolerated = expected + 5e-13
    assert (
        _crown(
            crown_area_m2=area,
            crown_equivalent_radius_m=tolerated,
        ).crown_equivalent_radius_m
        == tolerated
    )
    with pytest.raises(ValueError, match="inconsistent"):
        _crown(crown_area_m2=area, crown_equivalent_radius_m=expected + 1e-6)
    with pytest.raises(ValueError, match="at least"):
        _crown(crown_equivalent_radius_m=-1)
    zero = _crown(
        crown_xy_bounds_local=(0, 0, 0, 0),
        crown_area_m2=0,
        crown_equivalent_radius_m=0,
    )
    assert zero.crown_equivalent_radius_m == 0.0
    with pytest.raises(ValueError, match="must be zero"):
        _crown(
            crown_xy_bounds_local=(0, 0, 0, 0),
            crown_area_m2=0,
            crown_equivalent_radius_m=1e-13,
        )


def test_crown_summary_json_fields_are_exact_and_mapping_is_required() -> None:
    payload = _crown().to_dict()
    for field in payload:
        missing = dict(payload)
        missing.pop(field)
        with pytest.raises(ValueError, match="missing"):
            TreeCrownSummary.from_dict(missing)
    unknown = dict(payload)
    unknown["crown_vertical_range_m"] = 1.0
    with pytest.raises(ValueError, match="unknown"):
        TreeCrownSummary.from_dict(unknown)
    for value in (None, [], "crown", 1, True):
        with pytest.raises(ValueError, match="mapping"):
            TreeCrownSummary.from_dict(value)


def test_record_crown_summary_is_complete_or_absent_and_strictly_serialized() -> None:
    record = _record(crown_summary=_crown())
    payload = json.loads(json.dumps(record.to_dict()))
    assert payload["crown_summary"] == _crown().to_dict()
    assert TreeInstanceRecord.from_dict(payload) == record
    absent = _record(crown_summary=None)
    absent_payload = absent.to_dict()
    assert absent_payload["crown_summary"] is None
    assert TreeInstanceRecord.from_dict(absent_payload) == absent

    missing = dict(payload)
    missing.pop("crown_summary")
    with pytest.raises(ValueError, match="missing"):
        TreeInstanceRecord.from_dict(missing)
    unknown = dict(payload)
    unknown["crown"] = payload["crown_summary"]
    with pytest.raises(ValueError, match="unknown"):
        TreeInstanceRecord.from_dict(unknown)
    for value in ([], "crown", 1, True):
        invalid = dict(payload)
        invalid["crown_summary"] = value
        with pytest.raises(ValueError, match="mapping"):
            TreeInstanceRecord.from_dict(invalid)


@pytest.mark.parametrize(
    "value",
    (
        (0.0, 0.0, 1.0, 1.0),
        [0.0, 0.0, 1.0, 1.0],
        {
            "crown_xy_bounds_local": [0.0, 0.0, 1.0, 1.0],
            "crown_area_m2": 1.0,
            "crown_equivalent_radius_m": math.sqrt(1.0 / math.pi),
        },
        _geometry(),
    ),
)
def test_record_rejects_partial_or_untyped_crown_values(value: object) -> None:
    with pytest.raises(ValueError, match="crown_summary"):
        _record(crown_summary=value)


@pytest.mark.parametrize(
    "status",
    (
        TreeSegmentationStatus.REVIEW_REQUIRED,
        TreeSegmentationStatus.PROVISIONAL,
        TreeSegmentationStatus.REJECTED,
        TreeSegmentationStatus.UNRESOLVED,
    ),
)
@pytest.mark.parametrize("crown_summary", (_DEFAULT_CROWN, None))
def test_nonconfirmed_statuses_retain_complete_or_absent_crown(
    status: TreeSegmentationStatus,
    crown_summary: TreeCrownSummary | None,
) -> None:
    record = _record(status=status, crown_summary=crown_summary)
    assert record.crown_summary is crown_summary
    assert record.tree_id == -1
    assert record.tree_distance_ready is False


def test_default_confirmed_tree_requires_positive_crown_summary() -> None:
    missing = _record(
        tree_id=0,
        status=TreeSegmentationStatus.CONFIRMED_TREE,
        crown_summary=None,
    )
    with pytest.raises(ValueError, match="complete crown summary"):
        _confirmed_result(missing)

    zero_crown = _crown(
        crown_xy_bounds_local=(0, 0, 0, 0),
        crown_area_m2=0,
        crown_equivalent_radius_m=0,
    )
    degenerate = _record(
        tree_id=0,
        status=TreeSegmentationStatus.CONFIRMED_TREE,
        crown_summary=zero_crown,
    )
    with pytest.raises(ValueError, match="positive"):
        _confirmed_result(degenerate)
    with pytest.raises(ValueError, match="positive"):
        _confirmed_result(
            degenerate,
            config=TreeSegmentationConfig(allow_trunk_only_confirmation=True),
        )

    confirmed = _record(
        tree_id=0,
        status=TreeSegmentationStatus.CONFIRMED_TREE,
        crown_summary=_crown(),
    )
    assert _confirmed_result(confirmed).records == (confirmed,)


def test_trunk_only_confirmation_exception_preserves_existing_readiness_semantics() -> None:
    trunk_only = _record(
        tree_id=0,
        status=TreeSegmentationStatus.CONFIRMED_TREE,
        tree_ready=True,
        crown_summary=None,
        quality_flags=TreeSegmentationQualityFlag.MISSING_CROWN_EVIDENCE,
    )
    result = _confirmed_result(
        trunk_only,
        config=TreeSegmentationConfig(allow_trunk_only_confirmation=True),
    )
    assert result.records == (trunk_only,)
    assert result.records[0].tree_distance_ready is True
    assert result.tree_distance_ready_by_point[0]


def test_tree_quality_flag_bitmask_behavior_remains_strict_and_deterministic() -> None:
    combined = (
        TreeSegmentationQualityFlag.INSUFFICIENT_SUPPORT
        | TreeSegmentationQualityFlag.PARTIAL_GEOMETRY
    )
    record = _record(quality_flags=np.int32(int(combined)))
    assert record.quality_flags == combined
    assert TreeInstanceRecord.from_dict(record.to_dict()) == record
    with pytest.raises(ValueError, match="quality flags"):
        _record(quality_flags=np.bool_(False))
    with pytest.raises(ValueError, match="quality flags"):
        _record(quality_flags=-1)
    unknown_bit = 1 << max(int(flag).bit_length() for flag in TreeSegmentationQualityFlag)
    with pytest.raises(ValueError, match="quality flags"):
        _record(quality_flags=unknown_bit)


def test_geometry_summary_is_frozen_canonical_and_json_roundtrippable() -> None:
    apex = [np.int32(1), 2.5, np.float64(3)]
    summary = _geometry(
        apex_xyz_local=apex,
        maximum_hag_m=np.int64(12),
        robust_height_hag_m=np.float32(11.5),
        vertical_extent_m=np.float64(10),
    )
    assert summary.apex_xyz_local == (1.0, 2.5, 3.0)
    assert all(type(value) is float for value in summary.apex_xyz_local)
    assert type(summary.maximum_hag_m) is float
    assert type(summary.robust_height_hag_m) is float
    assert type(summary.vertical_extent_m) is float
    assert summary.to_dict() == {
        "apex_xyz_local": [1.0, 2.5, 3.0],
        "maximum_hag_m": 12.0,
        "robust_height_hag_m": 11.5,
        "vertical_extent_m": 10.0,
    }
    payload = json.loads(json.dumps(summary.to_dict()))
    assert TreeGeometrySummary.from_dict(payload) == summary
    assert TreeGeometrySummary.from_dict(payload).to_dict() == payload
    apex[0] = 99
    assert summary.apex_xyz_local == (1.0, 2.5, 3.0)
    with pytest.raises(FrozenInstanceError):
        summary.vertical_extent_m = 1.0


@pytest.mark.parametrize("numeric", (1, 1.25, np.int32(2), np.float32(2.5)))
def test_geometry_summary_accepts_python_and_numpy_numeric_values(numeric: object) -> None:
    summary = _geometry(
        apex_xyz_local=(numeric, numeric, numeric),
        maximum_hag_m=numeric,
        robust_height_hag_m=numeric,
        vertical_extent_m=numeric,
    )
    assert all(type(value) is float for value in summary.apex_xyz_local)
    assert type(summary.maximum_hag_m) is float
    assert type(summary.robust_height_hag_m) is float
    assert type(summary.vertical_extent_m) is float


@pytest.mark.parametrize(
    "apex",
    (
        (1, 2),
        (1, 2, 3, 4),
        "123",
        b"123",
        [[1], 2, 3],
        None,
        (True, 2, 3),
        (np.bool_(False), 2, 3),
        (np.nan, 2, 3),
        (np.inf, 2, 3),
        (-np.inf, 2, 3),
        np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    ),
)
def test_geometry_summary_rejects_invalid_apex(apex: object) -> None:
    with pytest.raises(ValueError, match="apex_xyz_local"):
        _geometry(apex_xyz_local=apex)


@pytest.mark.parametrize(
    "field,value",
    (
        ("maximum_hag_m", True),
        ("maximum_hag_m", np.bool_(False)),
        ("maximum_hag_m", "10"),
        ("maximum_hag_m", [10]),
        ("maximum_hag_m", np.nan),
        ("maximum_hag_m", np.inf),
        ("maximum_hag_m", -np.inf),
        ("maximum_hag_m", -1),
        ("robust_height_hag_m", True),
        ("robust_height_hag_m", np.bool_(False)),
        ("robust_height_hag_m", "9"),
        ("robust_height_hag_m", {"value": 9}),
        ("robust_height_hag_m", np.nan),
        ("robust_height_hag_m", np.inf),
        ("robust_height_hag_m", -np.inf),
        ("robust_height_hag_m", -1),
        ("vertical_extent_m", True),
        ("vertical_extent_m", np.bool_(False)),
        ("vertical_extent_m", "8"),
        ("vertical_extent_m", (8,)),
        ("vertical_extent_m", np.nan),
        ("vertical_extent_m", np.inf),
        ("vertical_extent_m", -np.inf),
        ("vertical_extent_m", -1),
    ),
)
def test_geometry_summary_rejects_invalid_scalars(field: str, value: object) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=field):
        _geometry(**kwargs)


def test_geometry_summary_hag_nullability_and_cross_field_rules() -> None:
    local_only = _geometry(maximum_hag_m=None, robust_height_hag_m=None)
    assert local_only.maximum_hag_m is None
    assert local_only.robust_height_hag_m is None
    with pytest.raises(ValueError, match="both"):
        _geometry(maximum_hag_m=None, robust_height_hag_m=0.0)
    with pytest.raises(ValueError, match="both"):
        _geometry(maximum_hag_m=0.0, robust_height_hag_m=None)
    with pytest.raises(ValueError, match="must not exceed"):
        _geometry(maximum_hag_m=5.0, robust_height_hag_m=5.1)

    zero = _geometry(
        maximum_hag_m=0,
        robust_height_hag_m=0,
        vertical_extent_m=0,
    )
    assert zero.maximum_hag_m == 0.0
    assert zero.robust_height_hag_m == 0.0
    assert zero.vertical_extent_m == 0.0
    assert (
        _geometry(
            maximum_hag_m=2.0,
            robust_height_hag_m=2.0,
            vertical_extent_m=5.0,
        ).vertical_extent_m
        == 5.0
    )


def test_geometry_summary_json_fields_are_exact_and_mapping_is_required() -> None:
    payload = _geometry().to_dict()
    for field in payload:
        missing = dict(payload)
        missing.pop(field)
        with pytest.raises(ValueError, match="missing"):
            TreeGeometrySummary.from_dict(missing)
    unknown = dict(payload)
    unknown["crown_area_m2"] = 1.0
    with pytest.raises(ValueError, match="unknown"):
        TreeGeometrySummary.from_dict(unknown)
    for value in (None, [], "geometry", 1):
        with pytest.raises(ValueError, match="mapping"):
            TreeGeometrySummary.from_dict(value)


def test_status_dependent_geometry_rules_are_strict() -> None:
    local_only = _geometry(maximum_hag_m=None, robust_height_hag_m=None)
    with pytest.raises(ValueError, match="complete geometry"):
        _record(
            tree_id=0,
            status=TreeSegmentationStatus.CONFIRMED_TREE,
            geometry_summary=None,
        )
    with pytest.raises(ValueError, match="complete geometry"):
        _record(
            tree_id=0,
            status=TreeSegmentationStatus.CONFIRMED_TREE,
            geometry_summary=local_only,
        )
    assert (
        _record(
            tree_id=0,
            status=TreeSegmentationStatus.CONFIRMED_TREE,
            geometry_summary=_geometry(),
        ).geometry_summary
        == _geometry()
    )

    for status in (
        TreeSegmentationStatus.REVIEW_REQUIRED,
        TreeSegmentationStatus.PROVISIONAL,
    ):
        for geometry in (_geometry(), local_only, None):
            assert _record(status=status, geometry_summary=geometry).geometry_summary is geometry

    for status in (
        TreeSegmentationStatus.REJECTED,
        TreeSegmentationStatus.UNRESOLVED,
    ):
        for geometry in (_geometry(), None):
            assert _record(status=status, geometry_summary=geometry).geometry_summary is geometry


def test_record_geometry_serialization_is_strict() -> None:
    record = _record()
    payload = json.loads(json.dumps(record.to_dict()))
    assert TreeInstanceRecord.from_dict(payload) == record
    missing = dict(payload)
    missing.pop("geometry_summary")
    with pytest.raises(ValueError, match="missing"):
        TreeInstanceRecord.from_dict(missing)
    unknown = dict(payload)
    unknown["apex"] = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="unknown"):
        TreeInstanceRecord.from_dict(unknown)
    for value in ([], "geometry", 1, True):
        invalid = dict(payload)
        invalid["geometry_summary"] = value
        with pytest.raises(ValueError, match="mapping"):
            TreeInstanceRecord.from_dict(invalid)
    absent = _record(geometry_summary=None)
    assert TreeInstanceRecord.from_dict(absent.to_dict()) == absent


def test_every_record_has_segment_identity_independent_of_tree_identity() -> None:
    review = _record(segment_id=0)
    confirmed = _record(
        0,
        TreeSegmentationStatus.CONFIRMED_TREE,
        segment_id=1,
        tree_ready=True,
    )
    result = _result(
        (review, confirmed),
        ids=np.array([-1, 0, 0, -1], np.int32),
        ready=np.array([False, True, True, False], np.bool_),
        offsets=np.array([0, 1, 3], np.int64),
        members=np.array([0, 1, 2], np.int64),
        segment_ids=np.array([0, 1, 1, -1], np.int32),
    )

    assert [(record.segment_id, record.tree_id) for record in result.records] == [
        (0, -1),
        (1, 0),
    ]
    assert result.records[0].tree_distance_ready is False
    assert result.records[1].tree_distance_ready is True


def test_record_segment_identity_is_strict_and_non_negative() -> None:
    with pytest.raises(ValueError, match="segment_id"):
        _record(segment_id=-1)
    with pytest.raises(ValueError, match="segment_id"):
        _record(segment_id=False)
    payload = _record().to_dict()
    assert payload["segment_id"] == 0
    for value in (True, -1, "0"):
        invalid = dict(payload)
        invalid["segment_id"] = value
        with pytest.raises(ValueError, match="segment_id"):
            TreeInstanceRecord.from_dict(invalid)


def test_segment_ids_are_unique_continuous_and_canonical() -> None:
    with pytest.raises(ValueError, match="unique"):
        _result((_record(), _record(source_ids=(4,))))
    with pytest.raises(ValueError, match="continuous"):
        _result((_record(segment_id=1),))


def test_empty_result_and_nonconfirmed_roundtrip() -> None:
    empty = _result(ids=np.empty(0, np.int32), ready=np.empty(0, np.bool_))
    assert TreeSegmentationResult.from_dict(empty.to_dict()).to_dict() == empty.to_dict()
    records = (
        _record(status=TreeSegmentationStatus.REVIEW_REQUIRED),
        _record(
            status=TreeSegmentationStatus.PROVISIONAL,
            segment_id=1,
            source_ids=(4,),
            geometry_summary=_geometry(maximum_hag_m=None, robust_height_hag_m=None),
        ),
        _record(
            status=TreeSegmentationStatus.REJECTED,
            segment_id=2,
            source_ids=(5,),
            geometry_summary=None,
        ),
        _record(
            status=TreeSegmentationStatus.UNRESOLVED,
            segment_id=3,
            source_ids=(6,),
        ),
    )
    result = _result(
        records,
        offsets=np.array([0, 1, 2, 3, 4], np.int64),
        members=np.arange(4, dtype=np.int64),
    )
    payload = json.loads(json.dumps(result.to_dict()))
    assert TreeSegmentationResult.from_dict(payload).to_dict() == payload


def test_caller_record_order_does_not_change_canonical_segment_output() -> None:
    review = _record(segment_id=0)
    confirmed = _record(
        0,
        TreeSegmentationStatus.CONFIRMED_TREE,
        segment_id=1,
        source_ids=(4,),
        tree_ready=True,
    )
    segment_ids = np.array([0, 1, 1, -1], np.int32)
    tree_ids = np.array([-1, 0, 0, -1], np.int32)
    ready = np.array([False, True, True, False], np.bool_)
    canonical = _result(
        (review, confirmed),
        tree_ids,
        ready,
        np.array([0, 1, 3], np.int64),
        np.array([0, 1, 2], np.int64),
        segment_ids=segment_ids,
    )
    shuffled = _result(
        (confirmed, review),
        tree_ids,
        ready,
        np.array([0, 2, 3], np.int64),
        np.array([1, 2, 0], np.int64),
        segment_ids=segment_ids,
    )

    assert shuffled.to_dict() == canonical.to_dict()
    assert [record.segment_id for record in shuffled.records] == [0, 1]
    assert [record.tree_id for record in shuffled.records] == [-1, 0]


def test_confirmed_tree_ids_start_at_zero_and_are_continuous() -> None:
    first = _record(0, TreeSegmentationStatus.CONFIRMED_TREE, tree_ready=True)
    second = _record(
        1,
        TreeSegmentationStatus.CONFIRMED_TREE,
        segment_id=1,
        source_ids=(4,),
        tree_ready=True,
    )
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


@pytest.mark.parametrize(
    "offsets,members,error",
    [
        (np.array([0, 2, 1], np.int64), np.array([0], np.int64), "monotonic"),
        (np.array([0, 1], np.int64), np.array([4], np.int64), "outside"),
    ],
)
def test_member_offsets_and_indexes_are_strict(
    offsets: np.ndarray, members: np.ndarray, error: str
) -> None:
    records = (
        (_record(),) if len(offsets) == 2 else (_record(), _record(segment_id=1, source_ids=(4,)))
    )
    with pytest.raises(ValueError, match=error):
        _result(records, offsets=offsets, members=members)


def test_segment_point_array_dtype_rank_length_and_ids_are_strict() -> None:
    with pytest.raises(ValueError, match="segment_id_by_point"):
        _result(segment_ids=np.full(4, -1, np.int64))
    with pytest.raises(ValueError, match="segment_id_by_point"):
        _result(segment_ids=np.full((4, 1), -1, np.int32))
    with pytest.raises(ValueError, match="segment_id_by_point"):
        _result(segment_ids=np.full(3, -1, np.int32))
    with pytest.raises(ValueError, match="unknown"):
        _result(segment_ids=np.array([-1, 0, -1, -1], np.int32))


def test_segment_members_must_exactly_match_point_identity() -> None:
    review = _record()
    with pytest.raises(ValueError, match="segment member rows"):
        _result(
            (review,),
            offsets=np.array([0, 1], np.int64),
            members=np.array([0], np.int64),
            segment_ids=np.array([-1, -1, -1, -1], np.int32),
        )
    with pytest.raises(ValueError, match="segment member rows"):
        _result(
            (review,),
            offsets=np.array([0, 1], np.int64),
            members=np.array([0], np.int64),
            segment_ids=np.array([0, 0, -1, -1], np.int32),
        )


def test_no_point_may_belong_to_two_retained_segments() -> None:
    with pytest.raises(ValueError, match="share member rows"):
        _result(
            (_record(), _record(segment_id=1, source_ids=(4,))),
            offsets=np.array([0, 1, 2], np.int64),
            members=np.array([0, 0], np.int64),
            segment_ids=np.array([0, -1, -1, -1], np.int32),
        )


def test_nonconfirmed_segment_members_cannot_receive_tree_identity_or_readiness() -> None:
    review = _record()
    confirmed = _record(
        0,
        TreeSegmentationStatus.CONFIRMED_TREE,
        segment_id=1,
        source_ids=(4,),
    )
    with pytest.raises(ValueError, match="non-confirmed member rows"):
        _result(
            (review, confirmed),
            ids=np.array([0, 0, -1, -1], np.int32),
            offsets=np.array([0, 1, 2], np.int64),
            members=np.array([0, 1], np.int64),
            segment_ids=np.array([0, 1, -1, -1], np.int32),
        )
    with pytest.raises(ValueError, match="confirmed tree_id"):
        _result(
            (review,),
            ready=np.array([True, False, False, False], np.bool_),
            offsets=np.array([0, 1], np.int64),
            members=np.array([0], np.int64),
            segment_ids=np.array([0, -1, -1, -1], np.int32),
        )


def test_confirmed_members_cannot_overlap_or_disagree_with_point_arrays() -> None:
    first = _record(0, TreeSegmentationStatus.CONFIRMED_TREE)
    second = _record(
        1,
        TreeSegmentationStatus.CONFIRMED_TREE,
        segment_id=1,
        source_ids=(4,),
    )
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
    assert (
        _result((ready,), ready=np.array([True, True, False, False], np.bool_), **kwargs)
        .tree_distance_ready_by_point[:2]
        .all()
    )
    with pytest.raises(ValueError, match="readiness"):
        _result((not_ready,), ready=np.array([True, False, False, False], np.bool_), **kwargs)
    with pytest.raises(ValueError, match="readiness"):
        _result((ready,), ready=np.array([True, False, False, False], np.bool_), **kwargs)


def test_point_alignment_dtype_nonfinite_and_input_immutability() -> None:
    segment_ids = np.array([-1, -1, -1, -1], np.int32)
    ids = np.array([-1, -1, -1, -1], np.int32)
    ready = np.zeros(4, np.bool_)
    offsets = np.array([0], np.int64)
    members = np.empty(0, np.int64)
    before = (
        segment_ids.copy(),
        ids.copy(),
        ready.copy(),
        offsets.copy(),
        members.copy(),
    )
    _result(
        ids=ids,
        ready=ready,
        offsets=offsets,
        members=members,
        segment_ids=segment_ids,
    )
    assert all(
        np.array_equal(value, snapshot)
        for value, snapshot in zip(
            (segment_ids, ids, ready, offsets, members),
            before,
            strict=True,
        )
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
        ("segment_id_by_point", ["0", -1, -1, -1], "integer"),
        ("segment_id_by_point", [True, -1, -1, -1], "integer"),
        ("segment_id_by_point", [[-1], [-1], [-1], [-1]], "integer"),
        ("tree_id_by_point", ["0", -1, -1, -1], "integer"),
        ("tree_distance_ready_by_point", [0, 0, 0, 0], "bool"),
        ("tree_id_by_point", [True, -1, -1, -1], "integer"),
        ("tree_member_offsets", [True], "integer"),
        ("tree_id_by_point", [[-1], [-1], [-1], [-1]], "integer"),
    ),
)
def test_from_dict_rejects_non_json_array_types(field: str, value: object, error: str) -> None:
    payload = _result().to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match=error):
        TreeSegmentationResult.from_dict(payload)


def test_segment_json_fields_are_required_and_unknown_fields_are_rejected() -> None:
    record_payload = _record().to_dict()
    missing_record = dict(record_payload)
    missing_record.pop("segment_id")
    with pytest.raises(ValueError, match="missing"):
        TreeInstanceRecord.from_dict(missing_record)
    unknown_record = dict(record_payload)
    unknown_record["segment"] = 0
    with pytest.raises(ValueError, match="unknown"):
        TreeInstanceRecord.from_dict(unknown_record)

    result_payload = _result().to_dict()
    missing_result = dict(result_payload)
    missing_result.pop("segment_id_by_point")
    with pytest.raises(ValueError, match="missing"):
        TreeSegmentationResult.from_dict(missing_result)
    unknown_result = dict(result_payload)
    unknown_result["segments"] = []
    with pytest.raises(ValueError, match="unknown"):
        TreeSegmentationResult.from_dict(unknown_result)


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


def test_canonical_arrays_are_retained_read_only_and_replace_is_zero_copy() -> None:
    arrays = (
        _frozen_array([-1, -1, -1, -1], np.int32),
        _frozen_array([-1, -1, -1, -1], np.int32),
        _frozen_array([False, False, False, False], np.bool_),
        _frozen_array([0], np.int64),
        _frozen_array([], np.int64),
    )
    result = _result(
        ids=arrays[1],
        ready=arrays[2],
        offsets=arrays[3],
        members=arrays[4],
        segment_ids=arrays[0],
    )
    for name, value in zip(
        (
            "segment_id_by_point",
            "tree_id_by_point",
            "tree_distance_ready_by_point",
            "tree_member_offsets",
            "ordered_tree_member_row_index",
        ),
        arrays,
        strict=True,
    ):
        assert getattr(result, name) is value
        assert value.flags.owndata
        assert value.flags.c_contiguous
        assert not value.flags.writeable
        assert getattr(replace(result), name) is value
    with pytest.raises(ValueError, match="ownership"):
        replace(result, segment_id_by_point=result.segment_id_by_point.copy())


class _ArraySubclass(np.ndarray):
    pass


def _storage_policy_base_result() -> TreeSegmentationResult:
    return _result(
        (_record(),),
        ids=np.array([-1, -1, -1, -1], np.int32),
        ready=np.zeros(4, np.bool_),
        offsets=np.array([0, 2], np.int64),
        members=np.array([0, 1], np.int64),
        segment_ids=np.array([0, 0, -1, -1], np.int32),
    )


def _invalid_storage_array(
    valid: np.ndarray,
    defect: str,
    path: Path,
) -> np.ndarray:
    if defect == "writable":
        return valid.copy()
    if defect in {"view", "readonly_view"}:
        backing = np.empty(len(valid) + 1, dtype=valid.dtype)
        backing[: len(valid)] = valid
        view = backing[: len(valid)]
        if defect == "readonly_view":
            view.setflags(write=False)
        return view
    if defect == "subclass":
        return valid.view(_ArraySubclass)
    if defect == "memmap":
        value = np.memmap(path, dtype=valid.dtype, mode="w+", shape=valid.shape)
        value[:] = valid
        value.flush()
        value.setflags(write=False)
        return value
    if defect == "wrong_dtype":
        dtype = np.int32 if valid.dtype in {np.dtype(np.int64), np.dtype(np.bool_)} else np.int64
        return _frozen_array(valid, dtype)
    if defect == "wrong_rank":
        return _frozen_array(valid.reshape((-1, 1)), valid.dtype)
    if defect == "non_c":
        backing = np.empty(len(valid) * 2, dtype=valid.dtype)
        backing[::2] = valid
        view = backing[::2]
        view.setflags(write=False)
        assert not view.flags.c_contiguous
        return view
    raise AssertionError(f"unknown storage defect: {defect}")


@pytest.mark.parametrize("field", _RESULT_ARRAY_FIELDS)
@pytest.mark.parametrize(
    "defect",
    (
        "writable",
        "view",
        "readonly_view",
        "subclass",
        "memmap",
        "wrong_dtype",
        "wrong_rank",
        "non_c",
    ),
)
def test_each_result_array_rejects_every_applicable_storage_defect(
    field: str,
    defect: str,
    tmp_path: Path,
) -> None:
    result = _storage_policy_base_result()
    valid = getattr(result, field)
    replacement = _invalid_storage_array(
        valid,
        defect,
        tmp_path / f"{field}-{defect}.bin",
    )
    assert replacement.shape == valid.shape or defect == "wrong_rank"
    with pytest.raises(ValueError, match=field):
        replace(result, **{field: replacement})


def test_from_dict_creates_one_canonical_frozen_array_per_field() -> None:
    restored = TreeSegmentationResult.from_dict(_result().to_dict())
    for name in (
        "segment_id_by_point",
        "tree_id_by_point",
        "tree_distance_ready_by_point",
        "tree_member_offsets",
        "ordered_tree_member_row_index",
    ):
        value = getattr(restored, name)
        assert type(value) is np.ndarray
        assert value.flags.owndata
        assert value.flags.c_contiguous
        assert not value.flags.writeable


def test_linear_membership_accepts_exact_and_rejects_count_and_label_mismatches() -> None:
    records = (_record(segment_id=0), _record(segment_id=1, source_ids=(4,)))
    valid = {
        "records": records,
        "offsets": np.array([0, 2, 3], np.int64),
        "members": np.array([0, 2, 3], np.int64),
        "segment_ids": np.array([0, -1, 0, 1], np.int32),
    }
    assert np.array_equal(_result(**valid).ordered_tree_member_row_index, [0, 2, 3])
    with pytest.raises(ValueError, match="assigned count"):
        _result(
            records,
            offsets=np.array([0, 1, 2], np.int64),
            members=np.array([0, 3], np.int64),
            segment_ids=valid["segment_ids"],
        )
    with pytest.raises(ValueError, match="exactly match"):
        _result(
            records,
            offsets=np.array([0, 2, 3], np.int64),
            members=np.array([0, 1, 3], np.int64),
            segment_ids=valid["segment_ids"],
        )
    with pytest.raises(ValueError, match="exactly match"):
        _result(
            records,
            offsets=np.array([0, 2, 3], np.int64),
            members=np.array([0, 2, 3], np.int64),
            segment_ids=np.array([1, -1, 0, 1], np.int32),
        )


@pytest.mark.parametrize(
    "members,error",
    (
        ([-1], "outside"),
        ([4], "outside"),
        ([0, 0], "strictly increasing"),
    ),
)
def test_linear_member_bounds_and_duplicates_are_rejected(
    members: list[int],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _result(
            (_record(),),
            offsets=np.array([0, len(members)], np.int64),
            members=np.array(members, np.int64),
            segment_ids=np.array([0, -1, -1, -1], np.int32),
        )


def test_duplicate_member_across_segment_blocks_is_rejected() -> None:
    with pytest.raises(ValueError, match="share member rows"):
        _result(
            (_record(), _record(segment_id=1, source_ids=(4,))),
            offsets=np.array([0, 1, 2], np.int64),
            members=np.array([0, 0], np.int64),
            segment_ids=np.array([0, 1, -1, -1], np.int32),
        )


def test_strict_member_order_is_checked_across_validation_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vegetation_trees, "_TREE_RESULT_VALIDATION_CHUNK_SIZE", 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        _result(
            (_record(),),
            offsets=np.array([0, 3], np.int64),
            members=np.array([0, 2, 1], np.int64),
            segment_ids=np.array([0, 0, 0, -1], np.int32),
        )


def test_noncanonical_records_reorder_whole_blocks_without_point_array_copies() -> None:
    records = tuple(_record(segment_id=index, source_ids=(index + 10,)) for index in range(3))
    point_segments = _frozen_array([0, 0, 1, 2, 2, -1], np.int32)
    point_ids = _frozen_array([-1] * 6, np.int32)
    point_ready = _frozen_array([False] * 6, np.bool_)
    offsets = _frozen_array([0, 1, 3, 5], np.int64)
    members = _frozen_array([2, 3, 4, 0, 1], np.int64)
    result = _result(
        (records[1], records[2], records[0]),
        ids=point_ids,
        ready=point_ready,
        offsets=offsets,
        members=members,
        segment_ids=point_segments,
    )
    assert result.records == records
    assert np.array_equal(result.tree_member_offsets, [0, 2, 3, 5])
    assert np.array_equal(result.ordered_tree_member_row_index, [0, 1, 2, 3, 4])
    assert result.tree_member_offsets is not offsets
    assert result.ordered_tree_member_row_index is not members
    assert result.segment_id_by_point is point_segments
    assert result.tree_id_by_point is point_ids
    assert result.tree_distance_ready_by_point is point_ready
    for index in range(3):
        start, end = result.tree_member_offsets[index : index + 2]
        assert np.all(np.diff(result.ordered_tree_member_row_index[start:end]) > 0)


def _instrument_result_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[str, tuple[int, ...]]], list[int]]:
    allocations: list[tuple[str, tuple[int, ...]]] = []
    iterated_lengths: list[int] = []
    real_empty = vegetation_trees.np.empty
    real_full = vegetation_trees.np.full
    real_zeros = vegetation_trees.np.zeros
    real_iter_chunks = vegetation_trees._iter_chunks

    def shape_tuple(shape: object) -> tuple[int, ...]:
        return (shape,) if isinstance(shape, int) else tuple(shape)

    def empty_spy(shape: object, *args: object, **kwargs: object) -> np.ndarray:
        allocations.append(("empty", shape_tuple(shape)))
        return real_empty(shape, *args, **kwargs)

    def full_spy(shape: object, *args: object, **kwargs: object) -> np.ndarray:
        allocations.append(("full", shape_tuple(shape)))
        return real_full(shape, *args, **kwargs)

    def zeros_spy(shape: object, *args: object, **kwargs: object) -> np.ndarray:
        allocations.append(("zeros", shape_tuple(shape)))
        return real_zeros(shape, *args, **kwargs)

    def iter_chunks_spy(length: int) -> range:
        iterated_lengths.append(length)
        return real_iter_chunks(length)

    monkeypatch.setattr(vegetation_trees.np, "empty", empty_spy)
    monkeypatch.setattr(vegetation_trees.np, "full", full_spy)
    monkeypatch.setattr(vegetation_trees.np, "zeros", zeros_spy)
    monkeypatch.setattr(vegetation_trees, "_iter_chunks", iter_chunks_spy)
    monkeypatch.setattr(
        vegetation_trees.np,
        "sort",
        lambda *_args, **_kwargs: pytest.fail("global sorting is forbidden"),
    )
    monkeypatch.setattr(
        vegetation_trees.np,
        "unique",
        lambda *_args, **_kwargs: pytest.fail("global uniqueness reconstruction is forbidden"),
    )
    monkeypatch.setattr(
        vegetation_trees.np,
        "flatnonzero",
        lambda *_args, **_kwargs: pytest.fail("per-record full point scans are forbidden"),
    )
    monkeypatch.setattr(
        vegetation_trees.np,
        "concatenate",
        lambda *_args, **_kwargs: pytest.fail("global validation copies are forbidden"),
    )
    return allocations, iterated_lengths


def _scaling_case(record_count: int, point_count: int) -> dict[str, object]:
    records = tuple(
        _record(segment_id=index, source_ids=(index + 10,)) for index in range(record_count)
    )
    segment_ids = np.full(point_count, -1, np.int32)
    segment_ids[:record_count] = np.arange(record_count, dtype=np.int32)
    return {
        "records": records,
        "ids": _frozen_array(np.full(point_count, -1, np.int32)),
        "ready": _frozen_array(np.zeros(point_count, np.bool_)),
        "offsets": _frozen_array(np.arange(record_count + 1, dtype=np.int64)),
        "members": _frozen_array(np.arange(record_count, dtype=np.int64)),
        "segment_ids": _frozen_array(segment_ids),
    }


@pytest.mark.parametrize("record_count", (1, 7))
def test_complete_point_scan_count_is_independent_of_record_count(
    record_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point_count = 37
    case = _scaling_case(record_count, point_count)
    arrays = tuple(case[name] for name in ("segment_ids", "ids", "ready", "offsets", "members"))
    allocations, iterated_lengths = _instrument_result_validation(monkeypatch)
    result = _result(**case)
    assert iterated_lengths.count(point_count) == 2
    assert all(shape != (point_count,) for _kind, shape in allocations)
    for name, value in zip(_RESULT_ARRAY_FIELDS, arrays, strict=True):
        assert getattr(result, name) is value


def test_noncanonical_validation_allocates_only_required_membership_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point_count = 11
    records = tuple(_record(segment_id=index, source_ids=(index + 10,)) for index in range(3))
    point_segments = _frozen_array([0, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1], np.int32)
    point_ids = _frozen_array([-1] * point_count, np.int32)
    point_ready = _frozen_array([False] * point_count, np.bool_)
    offsets = _frozen_array([0, 1, 3, 5], np.int64)
    members = _frozen_array([2, 3, 4, 0, 1], np.int64)
    allocations, iterated_lengths = _instrument_result_validation(monkeypatch)
    result = _result(
        (records[1], records[2], records[0]),
        ids=point_ids,
        ready=point_ready,
        offsets=offsets,
        members=members,
        segment_ids=point_segments,
    )
    assert iterated_lengths.count(point_count) == 2
    assert [shape for kind, shape in allocations if kind == "empty"].count((5,)) == 1
    assert [shape for kind, shape in allocations if kind == "empty"].count((4,)) == 1
    assert all(shape != (point_count,) for _kind, shape in allocations)
    assert result.segment_id_by_point is point_segments
    assert result.tree_id_by_point is point_ids
    assert result.tree_distance_ready_by_point is point_ready
    assert result.tree_member_offsets is not offsets
    assert result.ordered_tree_member_row_index is not members


def test_large_member_block_validation_is_bounded_by_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_size = vegetation_trees._TREE_RESULT_VALIDATION_CHUNK_SIZE
    point_count = chunk_size + 3
    case = {
        "records": (_record(),),
        "ids": _frozen_array(np.full(point_count, -1, np.int32)),
        "ready": _frozen_array(np.zeros(point_count, np.bool_)),
        "offsets": _frozen_array([0, point_count], np.int64),
        "members": _frozen_array(np.arange(point_count, dtype=np.int64)),
        "segment_ids": _frozen_array(np.zeros(point_count, np.int32)),
    }
    real_any = vegetation_trees.np.any
    observed_sizes: list[int] = []

    def any_spy(value: object, *args: object, **kwargs: object) -> np.bool_:
        observed_sizes.append(np.asarray(value).size)
        return real_any(value, *args, **kwargs)

    monkeypatch.setattr(vegetation_trees.np, "any", any_spy)
    result = _result(**case)
    assert result.ordered_tree_member_row_index is case["members"]
    assert max(observed_sizes) <= chunk_size
    assert chunk_size in observed_sizes
    assert 3 in observed_sizes


def test_canonical_membership_arrays_are_retained_by_identity() -> None:
    record = _record()
    offsets = _frozen_array([0, 2], np.int64)
    members = _frozen_array([0, 2], np.int64)
    result = _result(
        (record,),
        offsets=offsets,
        members=members,
        segment_ids=np.array([0, -1, 0, -1], np.int32),
    )
    assert result.records is not None
    assert result.tree_member_offsets is offsets
    assert result.ordered_tree_member_row_index is members


def test_tree_id_inventory_and_point_readiness_rules_are_linear() -> None:
    first = _record(0, TreeSegmentationStatus.CONFIRMED_TREE, segment_id=0)
    second = _record(
        1,
        TreeSegmentationStatus.CONFIRMED_TREE,
        segment_id=1,
        source_ids=(4,),
        tree_ready=True,
    )
    result = _result(
        (first, second),
        ids=np.array([0, 0, 1, 1], np.int32),
        ready=np.array([False, False, True, True], np.bool_),
        offsets=np.array([0, 2, 4], np.int64),
        members=np.array([0, 1, 2, 3], np.int64),
        segment_ids=np.array([0, 0, 1, 1], np.int32),
    )
    assert [record.tree_id for record in result.records] == [0, 1]
    with pytest.raises(ValueError, match="continuous"):
        _result(
            (
                first,
                replace(second, tree_id=0),
            ),
            ids=np.array([0, 0, 0, 0], np.int32),
            ready=np.zeros(4, np.bool_),
            offsets=np.array([0, 2, 4], np.int64),
            members=np.array([0, 1, 2, 3], np.int64),
            segment_ids=np.array([0, 0, 1, 1], np.int32),
        )
    with pytest.raises(ValueError, match="formal tree_id"):
        _result(ids=np.array([0, -1, -1, -1], np.int32))
    with pytest.raises(ValueError, match="confirmed tree_id"):
        _result(ready=np.array([True, False, False, False], np.bool_))


def test_nested_result_metadata_is_detached_sorted_and_deeply_immutable() -> None:
    timings = {"validate": 1, "load": 0.25}
    statistics = TreeSegmentationStatistics(0, 0, 0, 0, 0, 0, 0, 1.25, timings)
    provenance = {"z": [{"b": 2}], "a": {"items": [1, {"x": True}]}}
    context = {"unit": "m", "nested": {"axis": ["x", "y", "z"]}}
    result = TreeSegmentationResult(
        TREE_SEGMENTATION_SCHEMA_VERSION,
        TREE_SEGMENTATION_ALGORITHM_VERSION,
        "synthetic",
        "manifest",
        (0.0, 0.0, 0.0),
        TreeSegmentationConfig(),
        statistics,
        (),
        _frozen_array([], np.int32),
        _frozen_array([], np.int32),
        _frozen_array([], np.bool_),
        _frozen_array([0], np.int64),
        _frozen_array([], np.int64),
        provenance,
        context,
    )
    timings["load"] = 99
    provenance["z"][0]["b"] = 99
    context["nested"]["axis"].append("w")
    assert isinstance(result.input_provenance, MappingProxyType)
    assert isinstance(result.input_provenance["a"], MappingProxyType)
    assert isinstance(result.input_provenance["a"]["items"], tuple)
    assert list(result.input_provenance) == ["a", "z"]
    assert list(result.statistics.phase_timings_seconds) == ["load", "validate"]
    assert result.statistics.phase_timings_seconds["load"] == 0.25
    assert result.input_provenance["z"][0]["b"] == 2
    assert result.coordinate_context["nested"]["axis"] == ("x", "y", "z")
    with pytest.raises(TypeError):
        result.input_provenance["new"] = 1
    with pytest.raises(TypeError):
        result.input_provenance["a"]["new"] = 1
    with pytest.raises(TypeError):
        result.coordinate_context["nested"]["axis"][0] = "q"

    payload = result.to_dict()
    assert type(payload["input_provenance"]) is dict
    assert type(payload["input_provenance"]["a"]["items"]) is list
    payload["input_provenance"]["a"]["items"][0] = 99
    assert result.input_provenance["a"]["items"][0] == 1
    restored = TreeSegmentationResult.from_dict(result.to_dict())
    assert isinstance(restored.input_provenance, MappingProxyType)
    assert isinstance(restored.input_provenance["a"]["items"], tuple)


def test_validation_executes_without_global_membership_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        vegetation_trees.np,
        "flatnonzero",
        lambda *_args, **_kwargs: pytest.fail("np.flatnonzero must not be used"),
    )
    monkeypatch.setattr(
        vegetation_trees.np,
        "unique",
        lambda *_args, **_kwargs: pytest.fail("np.unique must not be used"),
    )
    assert _result((_record(),), offsets=np.array([0, 1]), members=np.array([0])).records
