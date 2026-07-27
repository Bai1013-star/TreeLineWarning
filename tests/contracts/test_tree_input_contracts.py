from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
from typing import get_type_hints

import numpy as np
import pytest

import core.contracts.tree_inputs as tree_inputs
from core.contracts import (
    HeightAboveGroundSet,
    PointFeatureSet,
    TreeSegmentationInputClusterDisposition,
    TreeSegmentationInputDisposition,
    TreeSegmentationInputIdentity,
    TreeSegmentationInputSet,
    VegetationInstancePointSet,
)


def _freeze(*arrays: np.ndarray) -> None:
    for array in arrays:
        array.setflags(write=False)


def _identity(**changes: object) -> TreeSegmentationInputIdentity:
    values: dict[str, object] = {
        "source_id": "scene",
        "manifest_source_id": "manifest",
        "origin_xyz": (1.0, 2.0, 3.0),
        "point_count": 3,
        "v2_algorithm_version": "v2-version",
        "a2a_algorithm_version": "a2a-version",
        "a2b_algorithm_version": "a2b-version",
    }
    values.update({name: "a" * 64 for name in tree_inputs._SHA256_FIELDS})
    values.update(changes)
    return TreeSegmentationInputIdentity(**values)


def _vegetation_points(*, frozen: bool) -> VegetationInstancePointSet:
    count = 3
    arrays = (
        np.arange(10, 10 + count, dtype=np.int64),
        np.arange(count, dtype=np.int64),
        np.zeros((count, 3), np.float32),
        np.full(count, -1, np.int32),
        np.full(count, 0.8, np.float32),
        np.full(count, 0.8, np.float32),
        np.zeros(count, np.uint8),
        np.zeros(count, np.uint8),
        np.zeros(count, np.uint8),
        np.zeros(count, np.float32),
        np.zeros(count, np.float32),
        np.zeros(count, np.uint8),
        np.full(count, -1, np.int32),
        np.full(count, -1, np.int32),
        np.zeros(count, np.uint8),
        np.zeros(count, np.bool_),
        np.zeros(count, np.float32),
        np.zeros(count, np.uint16),
        np.zeros(count, np.uint16),
        np.zeros(count, np.uint16),
    )
    if frozen:
        _freeze(*arrays)
    return VegetationInstancePointSet(*arrays)


def _geometry_features(*, frozen: bool) -> PointFeatureSet:
    count = 3
    arrays = (
        np.zeros((count, 3), np.float32),
        np.zeros(3, np.float64),
        np.arange(10, 10 + count, dtype=np.int64),
        np.arange(count, dtype=np.int64),
        np.zeros(count, np.float32),
        np.zeros(count, np.float32),
        np.zeros(count, np.float32),
        np.zeros(count, np.float32),
        np.zeros(count, np.float32),
        np.zeros(count, np.float32),
        np.zeros(count, np.uint16),
        np.zeros((count, 3), np.float32),
        np.zeros((count, 3), np.float32),
        np.zeros((count, 3), np.float32),
        np.zeros(count, np.uint16),
    )
    if frozen:
        _freeze(*arrays)
    return PointFeatureSet("scene", "manifest", *arrays)


def _heights_above_ground(*, frozen: bool) -> HeightAboveGroundSet:
    count = 3
    arrays = (
        np.arange(10, 10 + count, dtype=np.int64),
        np.arange(count, dtype=np.int64),
        np.zeros(count, np.float32),
        np.zeros(count, np.float32),
        np.zeros(count, np.float32),
        np.zeros(count, np.int32),
        np.zeros(count, np.float32),
        np.zeros(count, np.uint16),
    )
    if frozen:
        _freeze(*arrays)
    return HeightAboveGroundSet("scene", "manifest", *arrays)


def _input_set(*, frozen: bool = True, **changes: object) -> TreeSegmentationInputSet:
    arrays: dict[str, object] = {
        "cluster_id": np.array([10, 20, 30], np.int32),
        "cluster_status": np.array([1, 2, 4], np.uint8),
        "cluster_vegetation_id": np.array([0, -1, -1], np.int32),
        "cluster_member_offsets": np.array([0, 1, 2, 3], np.int64),
        "ordered_cluster_member_row_index": np.array([0, 1, 2], np.int64),
        "ground_usable_by_point": np.ones(3, np.bool_),
        "usable_ground_member_count_by_cluster": np.ones(3, np.int64),
    }
    if frozen:
        _freeze(*(value for value in arrays.values() if isinstance(value, np.ndarray)))
    values: dict[str, object] = {
        "vegetation_points": _vegetation_points(frozen=frozen),
        "geometry_features": _geometry_features(frozen=frozen),
        "heights_above_ground": _heights_above_ground(frozen=frozen),
        **arrays,
        "identity": _identity(),
        "eligible_cluster_ids": (10,),
        "review_only_cluster_ids": (20,),
        "excluded_cluster_ids": (30,),
        "cluster_dispositions": (
            TreeSegmentationInputClusterDisposition(
                10, TreeSegmentationInputDisposition.ELIGIBLE, "confirmed"
            ),
            TreeSegmentationInputClusterDisposition(
                20, TreeSegmentationInputDisposition.REVIEW_ONLY, "review"
            ),
            TreeSegmentationInputClusterDisposition(
                30, TreeSegmentationInputDisposition.EXCLUDED, "excluded"
            ),
        ),
    }
    values.update(changes)
    return TreeSegmentationInputSet(**values)


def test_identity_round_trip_and_frozen() -> None:
    identity = _identity()
    assert TreeSegmentationInputIdentity.from_dict(identity.to_dict()) == identity
    with pytest.raises(FrozenInstanceError):
        identity.source_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_id": ""}, "source_id"),
        ({"manifest_source_id": ""}, "manifest_source_id"),
        ({"origin_xyz": (0.0, 1.0)}, "exactly three"),
        ({"origin_xyz": (False, 1.0, 2.0)}, "finite numeric"),
        ({"origin_xyz": (np.nan, 1.0, 2.0)}, "finite numeric"),
        ({"origin_xyz": (np.inf, 1.0, 2.0)}, "finite numeric"),
        ({"origin_xyz": (10**400, 1.0, 2.0)}, "finite numeric"),
        ({"point_count": -1}, "point_count"),
        ({"v2_npz_sha256": "a" * 63}, "SHA"),
        ({"a2a_npz_sha256": "g" * 64}, "SHA"),
        ({"a2b_npz_sha256": "A" * 64}, "SHA"),
    ],
)
def test_identity_rejects_invalid_values(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _identity(**changes)


def test_identity_requires_exact_json_fields() -> None:
    payload = _identity().to_dict()
    payload.pop("v2_npz_sha256")
    with pytest.raises(ValueError):
        TreeSegmentationInputIdentity.from_dict(payload)

    payload = _identity().to_dict()
    payload["origin_xyz"][0] = 10**400
    with pytest.raises(ValueError, match="finite numeric"):
        TreeSegmentationInputIdentity.from_dict(payload)

    payload = _identity().to_dict()
    payload["unknown"] = True
    with pytest.raises(ValueError):
        TreeSegmentationInputIdentity.from_dict(payload)


def test_dispositions_round_trip_and_frozen() -> None:
    for disposition in TreeSegmentationInputDisposition:
        record = TreeSegmentationInputClusterDisposition(4, disposition, "documented")
        assert TreeSegmentationInputClusterDisposition.from_dict(record.to_dict()) == record
    record = TreeSegmentationInputClusterDisposition(
        4, TreeSegmentationInputDisposition.ELIGIBLE, "documented"
    )
    with pytest.raises(FrozenInstanceError):
        record.reason = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("cluster_id", "reason"),
    [(-1, "documented"), (1, "")],
)
def test_disposition_rejects_invalid_values(cluster_id: int, reason: str) -> None:
    with pytest.raises(ValueError):
        TreeSegmentationInputClusterDisposition(
            cluster_id, TreeSegmentationInputDisposition.ELIGIBLE, reason
        )


def test_input_set_is_frozen_read_only_and_zero_copy() -> None:
    vegetation_points = _vegetation_points(frozen=True)
    geometry_features = _geometry_features(frozen=True)
    heights_above_ground = _heights_above_ground(frozen=True)
    cluster_id = np.array([10, 20, 30], np.int32)
    cluster_id.setflags(write=False)
    top_level_arrays = {
        "cluster_id": cluster_id,
        "cluster_status": np.array([1, 2, 4], np.uint8),
        "cluster_vegetation_id": np.array([0, -1, -1], np.int32),
        "cluster_member_offsets": np.array([0, 1, 2, 3], np.int64),
        "ordered_cluster_member_row_index": np.array([0, 1, 2], np.int64),
        "ground_usable_by_point": np.ones(3, np.bool_),
        "usable_ground_member_count_by_cluster": np.ones(3, np.int64),
    }
    _freeze(*(value for value in top_level_arrays.values() if isinstance(value, np.ndarray)))
    input_set = _input_set(
        vegetation_points=vegetation_points,
        geometry_features=geometry_features,
        heights_above_ground=heights_above_ground,
        **top_level_arrays,
    )
    assert input_set.identity.point_count == 3
    assert not input_set.cluster_id.flags.writeable
    assert not input_set.vegetation_points.xyz_local.flags.writeable
    assert not input_set.geometry_features.eigenvalues.flags.writeable
    assert not input_set.heights_above_ground.height_agl_m.flags.writeable
    assert input_set.vegetation_points is vegetation_points
    assert input_set.geometry_features is geometry_features
    assert input_set.heights_above_ground is heights_above_ground
    for name, value in top_level_arrays.items():
        assert getattr(input_set, name) is value
    with pytest.raises(FrozenInstanceError):
        input_set.identity = _identity()  # type: ignore[misc]


def test_input_set_rejects_writable_top_level_and_nested_arrays() -> None:
    writable_cluster_id = np.array([10, 20, 30], np.int32)
    with pytest.raises(ValueError, match="cluster_id must be read-only"):
        _input_set(cluster_id=writable_cluster_id)

    input_set = _input_set()
    input_set.vegetation_points.xyz_local.setflags(write=True)
    with pytest.raises(ValueError, match="vegetation_points.xyz_local"):
        _input_set(vegetation_points=input_set.vegetation_points)


@pytest.mark.parametrize(
    ("cluster_id", "message"),
    [
        (np.array([10, 10, 30], np.int32), "unique"),
        (np.array([10, -1, 30], np.int32), "non-negative"),
    ],
)
def test_input_set_rejects_duplicate_or_negative_cluster_ids(
    cluster_id: np.ndarray, message: str
) -> None:
    cluster_id.setflags(write=False)
    with pytest.raises(ValueError, match=message):
        _input_set(cluster_id=cluster_id)


def test_input_set_public_array_annotations_are_ndarray() -> None:
    annotations = get_type_hints(TreeSegmentationInputSet)
    for name in (
        "cluster_id",
        "cluster_status",
        "cluster_vegetation_id",
        "cluster_member_offsets",
        "ordered_cluster_member_row_index",
        "ground_usable_by_point",
        "usable_ground_member_count_by_cluster",
    ):
        assert annotations[name] is np.ndarray


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"identity": _identity(point_count=2)}, "point_count"),
        ({"cluster_status": np.array([1, 2], np.uint8)}, "cluster_status"),
        ({"cluster_member_offsets": np.array([0, 1, 2, 3], np.int32)}, "offsets"),
        ({"cluster_member_offsets": np.array([[0, 1, 2, 3]], np.int64)}, "offsets"),
        ({"ordered_cluster_member_row_index": np.array([0, 1, 2], np.int32)}, "member"),
        ({"ground_usable_by_point": np.ones(3, np.uint8)}, "ground_usable"),
        ({"ground_usable_by_point": np.ones(2, np.bool_)}, "ground_usable"),
        ({"usable_ground_member_count_by_cluster": np.ones(3, np.int32)}, "usable_ground"),
        ({"usable_ground_member_count_by_cluster": np.array([1, -1, 1], np.int64)}, "non-negative"),
    ],
)
def test_input_set_rejects_invalid_array_structure(
    changes: dict[str, object], message: str
) -> None:
    for value in changes.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    with pytest.raises(ValueError, match=message):
        _input_set(**changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"eligible_cluster_ids": (20, 10)}, "strictly increasing"),
        ({"eligible_cluster_ids": (10, 10)}, "strictly increasing"),
        (
            {"review_only_cluster_ids": (10, 20)},
            "overlap",
        ),
        (
            {
                "cluster_dispositions": (
                    TreeSegmentationInputClusterDisposition(
                        10, TreeSegmentationInputDisposition.REVIEW_ONLY, "wrong"
                    ),
                    TreeSegmentationInputClusterDisposition(
                        20, TreeSegmentationInputDisposition.REVIEW_ONLY, "review"
                    ),
                    TreeSegmentationInputClusterDisposition(
                        30, TreeSegmentationInputDisposition.EXCLUDED, "excluded"
                    ),
                )
            },
            "exactly match",
        ),
    ],
)
def test_input_set_rejects_invalid_partitions(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _input_set(**changes)


def test_input_set_allows_a_legal_empty_structure() -> None:
    empty_v2 = VegetationInstancePointSet(
        *(
            np.empty((0, 3), np.float32)
            if name == "xyz_local"
            else np.empty(0, dtype=dtype)
            for name, dtype in (
                ("original_index", np.int64),
                ("sample_index", np.int64),
                ("xyz_local", np.float32),
                ("wire_id_by_point", np.int32),
                ("vegetation_rule_score", np.float32),
                ("vegetation_confidence", np.float32),
                ("automatic_vegetation_level", np.uint8),
                ("vegetation_level", np.uint8),
                ("hag_state", np.uint8),
                ("local_scale_raw_m", np.float32),
                ("local_scale_m", np.float32),
                ("local_scale_state", np.uint8),
                ("cluster_id_by_point", np.int32),
                ("vegetation_id_by_point", np.int32),
                ("cluster_status_by_point", np.uint8),
                ("distance_ready_by_point", np.bool_),
                ("assignment_confidence", np.float32),
                ("assignment_reason_flags", np.uint16),
                ("assignment_quality_flags", np.uint16),
                ("manual_provenance_flags", np.uint16),
            )
        )
    )
    empty_features = PointFeatureSet(
        "scene",
        "manifest",
        np.empty((0, 3), np.float32),
        np.zeros(3, np.float64),
        np.empty(0, np.int64),
        np.empty(0, np.int64),
        *(
            np.empty(0, np.float32)
            for _ in range(6)
        ),
        np.empty(0, np.uint16),
        np.empty((0, 3), np.float32),
        np.empty((0, 3), np.float32),
        np.empty((0, 3), np.float32),
        np.empty(0, np.uint16),
    )
    empty_hag = HeightAboveGroundSet(
        "scene",
        "manifest",
        np.empty(0, np.int64),
        np.empty(0, np.int64),
        np.empty(0, np.float32),
        np.empty(0, np.float32),
        np.empty(0, np.float32),
        np.empty(0, np.int32),
        np.empty(0, np.float32),
        np.empty(0, np.uint16),
    )
    arrays = [
        *(
            getattr(empty_v2, name)
            for name in empty_v2.__dataclass_fields__
            if isinstance(getattr(empty_v2, name), np.ndarray)
        ),
        *(
            getattr(empty_features, name)
            for name in empty_features.__dataclass_fields__
            if isinstance(getattr(empty_features, name), np.ndarray)
        ),
        *(
            getattr(empty_hag, name)
            for name in empty_hag.__dataclass_fields__
            if isinstance(getattr(empty_hag, name), np.ndarray)
        ),
    ]
    _freeze(*arrays)
    cluster_id = np.empty(0, np.int32)
    cluster_status = np.empty(0, np.uint8)
    cluster_vegetation_id = np.empty(0, np.int32)
    cluster_member_offsets = np.array([0], np.int64)
    member_rows = np.empty(0, np.int64)
    ground_usable = np.empty(0, np.bool_)
    usable_counts = np.empty(0, np.int64)
    _freeze(
        cluster_id,
        cluster_status,
        cluster_vegetation_id,
        cluster_member_offsets,
        member_rows,
        ground_usable,
        usable_counts,
    )
    empty = TreeSegmentationInputSet(
        empty_v2,
        empty_features,
        empty_hag,
        cluster_id,
        cluster_status,
        cluster_vegetation_id,
        cluster_member_offsets,
        member_rows,
        _identity(point_count=0),
        (),
        (),
        (),
        (),
        ground_usable,
        usable_counts,
    )
    assert empty.identity.point_count == 0
    assert empty.cluster_member_offsets.tolist() == [0]


def test_input_set_does_not_mutate_input_array_contents_or_use_python_hash() -> None:
    input_set = _input_set()
    before = input_set.cluster_member_offsets.copy()
    assert np.array_equal(input_set.cluster_member_offsets, before)
    assert "hash(" not in inspect.getsource(tree_inputs)
