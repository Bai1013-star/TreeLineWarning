from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from core.contracts import (
    TREE_INSTANCE_ML_SCHEMA_VERSION,
    TreeInstanceMlPrediction,
    TreeSemanticClass,
    validate_tree_instance_ml_prediction_alignment,
)

from tests.contracts.test_tree_input_contracts import _input_set


def _prediction(**changes: object) -> TreeInstanceMlPrediction:
    values: dict[str, object] = {
        "schema_version": TREE_INSTANCE_ML_SCHEMA_VERSION,
        "source_id": "scene",
        "manifest_source_id": "manifest",
        "point_count": 5,
        "original_index": np.array([101, 102, 103, 104, 105], np.int64),
        "sample_index": np.array([0, 1, 2, 3, 4], np.int64),
        "origin_xyz": (1.0, 2.0, 3.0),
        "model_version": "ml0-test",
        "model_sha256": "a" * 64,
        "preprocessing_signature": {"voxel_m": 0.2, "axes": ["x", "y", "z"]},
        "semantic_class_by_point": np.array(
            [TreeSemanticClass.TREE, TreeSemanticClass.SHRUB, TreeSemanticClass.NON_TREE,
             TreeSemanticClass.TREE, TreeSemanticClass.NON_TREE],
            np.uint8,
        ),
        "semantic_confidence_by_point": np.array([0.9, 0.8, 0.7, 0.95, 0.6], np.float32),
        "segment_id_by_point": np.array([0, -1, 1, 0, -1], np.int32),
        "instance_confidence_by_point": np.array([0.8, 0.0, 0.7, 0.8, 0.0], np.float32),
        "segment_ids": np.array([0, 1], np.int32),
        "member_offsets": np.array([0, 2, 3], np.int64),
        "member_rows": np.array([0, 3, 2], np.int64),
        "instance_confidence": np.array([0.8, 0.7], np.float32),
    }
    values.update(changes)
    return TreeInstanceMlPrediction(**values)


def test_valid_multi_instance_prediction_is_immutable_and_deterministic() -> None:
    prediction = _prediction()
    assert prediction.semantic_class_by_point.tolist() == [2, 1, 0, 2, 0]
    assert prediction.segment_id_by_point.tolist() == [0, -1, 1, 0, -1]
    assert prediction.member_offsets.tolist() == [0, 2, 3]
    assert prediction.member_rows.tolist() == [0, 3, 2]
    assert isinstance(prediction.preprocessing_signature, MappingProxyType)
    assert prediction.preprocessing_signature["axes"] == ("x", "y", "z")
    assert list(prediction.preprocessing_signature) == ["axes", "voxel_m"]
    assert repr(prediction) == repr(_prediction())
    with pytest.raises(FrozenInstanceError):
        prediction.source_id = "other"  # type: ignore[misc]


def test_valid_gapful_segment_ids_are_preserved_without_renumbering() -> None:
    prediction = _prediction(
        segment_id_by_point=np.array([2, 5, 9, 2, -1], np.int32),
        instance_confidence_by_point=np.array([0.8, 0.7, 0.6, 0.8, 0.0], np.float32),
        segment_ids=np.array([2, 5, 9], np.int32),
        member_offsets=np.array([0, 2, 3, 4], np.int64),
        member_rows=np.array([0, 3, 1, 2], np.int64),
        instance_confidence=np.array([0.8, 0.7, 0.6], np.float32),
    )

    assert prediction.segment_ids.tolist() == [2, 5, 9]
    assert prediction.segment_id_by_point.tolist() == [2, 5, 9, 2, -1]
    assert prediction.member_offsets.tolist() == [0, 2, 3, 4]
    assert prediction.member_rows.tolist() == [0, 3, 1, 2]
    for name in (
        "segment_id_by_point",
        "segment_ids",
        "member_offsets",
        "member_rows",
        "instance_confidence",
    ):
        array = getattr(prediction, name)
        assert array.flags.owndata
        assert array.flags.c_contiguous
        assert not array.flags.writeable


def test_zero_instance_prediction_is_valid() -> None:
    prediction = _prediction(
        point_count=3,
        original_index=np.array([1, 2, 3], np.int64),
        sample_index=np.array([10, 11, 12], np.int64),
        semantic_class_by_point=np.array([0, 1, 2], np.uint8),
        semantic_confidence_by_point=np.array([0.4, 0.5, 0.6], np.float32),
        segment_id_by_point=np.full(3, -1, np.int32),
        instance_confidence_by_point=np.zeros(3, np.float32),
        segment_ids=np.empty(0, np.int32),
        member_offsets=np.array([0], np.int64),
        member_rows=np.empty(0, np.int64),
        instance_confidence=np.empty(0, np.float32),
    )
    assert prediction.point_count == 3
    assert prediction.segment_ids.tolist() == []
    assert prediction.member_rows.tolist() == []


def test_empty_prediction_is_valid() -> None:
    prediction = _prediction(
        point_count=0,
        original_index=np.empty(0, np.int64),
        sample_index=np.empty(0, np.int64),
        semantic_class_by_point=np.empty(0, np.uint8),
        semantic_confidence_by_point=np.empty(0, np.float32),
        segment_id_by_point=np.empty(0, np.int32),
        instance_confidence_by_point=np.empty(0, np.float32),
        segment_ids=np.empty(0, np.int32),
        member_offsets=np.array([0], np.int64),
        member_rows=np.empty(0, np.int64),
        instance_confidence=np.empty(0, np.float32),
    )
    assert prediction.point_count == 0
    assert prediction.semantic_class_by_point.tolist() == []


def test_arrays_are_normalized_owned_contiguous_read_only_and_detached() -> None:
    original = np.array([101, 102, 103, 104, 105], np.int32)
    sample = np.array([0, 1, 2, 3, 4], np.uint32)
    semantic = np.array([2, 1, 0, 2, 0], np.int16)
    semantic_confidence = np.array([0.9, 0.8, 0.7, 0.95, 0.6], np.float64)
    segments = np.array([0, -1, 1, 0, -1], np.int64)
    point_confidence = np.array([0.8, 0.0, 0.7, 0.8, 0.0], np.float64)
    segment_ids = np.array([0, 1], np.int64)
    offsets = np.array([0, 2, 3], np.int32)
    members = np.array([0, 3, 2], np.int32)
    instance_confidence = np.array([0.8, 0.7], np.float64)
    prediction = _prediction(
        original_index=original,
        sample_index=sample,
        semantic_class_by_point=semantic,
        semantic_confidence_by_point=semantic_confidence,
        segment_id_by_point=segments,
        instance_confidence_by_point=point_confidence,
        segment_ids=segment_ids,
        member_offsets=offsets,
        member_rows=members,
        instance_confidence=instance_confidence,
    )
    expected = {
        "original_index": np.int64,
        "sample_index": np.int64,
        "semantic_class_by_point": np.uint8,
        "semantic_confidence_by_point": np.float32,
        "segment_id_by_point": np.int32,
        "instance_confidence_by_point": np.float32,
        "segment_ids": np.int32,
        "member_offsets": np.int64,
        "member_rows": np.int64,
        "instance_confidence": np.float32,
    }
    for name, dtype in expected.items():
        array = getattr(prediction, name)
        assert array.dtype == np.dtype(dtype)
        assert type(array) is np.ndarray
        assert array.flags.owndata
        assert array.flags.c_contiguous
        assert not array.flags.writeable
    original[0] = 999
    semantic_confidence[0] = 0.0
    assert prediction.original_index[0] == 101
    assert prediction.semantic_confidence_by_point[0] == pytest.approx(0.9)


def test_non_contiguous_input_view_is_normalized_and_detached_from_its_base() -> None:
    base = np.array([101, -1, 102, -1, 103, -1, 104, -1, 105, -1], np.int64)
    original_index = base[::2]
    prediction = _prediction(original_index=original_index)

    assert prediction.original_index.tolist() == [101, 102, 103, 104, 105]
    assert prediction.original_index.dtype == np.dtype(np.int64)
    assert prediction.original_index.flags.owndata
    assert prediction.original_index.flags.c_contiguous
    assert not prediction.original_index.flags.writeable
    base[0] = 999
    assert prediction.original_index[0] == 101


def test_foreign_endian_input_is_normalized_and_detached() -> None:
    foreign_byteorder = ">" if np.little_endian else "<"
    original_index = np.array([101, 102, 103, 104, 105], dtype=f"{foreign_byteorder}i8")
    prediction = _prediction(original_index=original_index)

    assert original_index.dtype.byteorder == foreign_byteorder
    assert prediction.original_index.tolist() == [101, 102, 103, 104, 105]
    assert prediction.original_index.dtype == np.dtype(np.int64)
    assert prediction.original_index.flags.owndata
    assert prediction.original_index.flags.c_contiguous
    assert not prediction.original_index.flags.writeable
    original_index[0] = 999
    assert prediction.original_index[0] == 101


def test_nested_preprocessing_signature_is_immutable_and_detached() -> None:
    signature = {
        "tiling": {"size": [32, 32], "options": {"overlap": 4}},
        "features": ["height", "linearity"],
    }
    prediction = _prediction(preprocessing_signature=signature)

    signature["tiling"]["size"][0] = 64
    signature["tiling"]["options"]["overlap"] = 8
    signature["features"].append("planarity")

    assert prediction.preprocessing_signature["tiling"]["size"] == (32, 32)
    assert prediction.preprocessing_signature["tiling"]["options"]["overlap"] == 4
    assert prediction.preprocessing_signature["features"] == ("height", "linearity")
    assert isinstance(prediction.preprocessing_signature["tiling"], MappingProxyType)
    assert isinstance(prediction.preprocessing_signature["tiling"]["options"], MappingProxyType)
    with pytest.raises(TypeError):
        prediction.preprocessing_signature["tiling"]["options"]["overlap"] = 8


@pytest.mark.parametrize(
    "value",
    [np.array([0, 1, 3, 2, 0], np.int32), np.array([0, 1, -1, 2, 0], np.int32)],
)
def test_invalid_semantic_values_are_rejected(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="semantic_class_by_point"):
        _prediction(semantic_class_by_point=value)


@pytest.mark.parametrize(
    "field",
    ["semantic_confidence_by_point", "instance_confidence_by_point", "instance_confidence"],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_confidence_values_are_rejected(field: str, value: float) -> None:
    source = getattr(_prediction(), field)
    invalid = source.copy()
    invalid[0] = value
    with pytest.raises(ValueError, match=field):
        _prediction(**{field: invalid})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("semantic_class_by_point", np.array([0, 1], np.uint8)),
        ("semantic_confidence_by_point", np.array([0.5, 0.6], np.float32)),
        ("segment_id_by_point", np.array([0, -1], np.int32)),
        ("instance_confidence_by_point", np.array([0.5, 0.0], np.float32)),
    ],
)
def test_point_arrays_require_exact_n_rows(field: str, value: np.ndarray) -> None:
    with pytest.raises(ValueError, match=field):
        _prediction(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("original_index", np.array([[1], [2], [3], [4], [5]], np.int64), "one-dimensional"),
        ("sample_index", np.array([0.0, 1.0, 2.0, 3.0, 4.0]), "integer dtype"),
        ("original_index", np.array([1, 1, 3, 4, 5], np.int64), "unique"),
        ("sample_index", np.array([-1, 1, 2, 3, 4], np.int64), "at least"),
    ],
)
def test_original_and_sample_index_validation(field: str, value: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _prediction(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("segment_id_by_point", np.array([0, -2, 1, 0, -1], np.int32), "segment_id_by_point"),
        ("segment_id_by_point", np.array([0, 2, 1, 0, -1], np.int32), "segment_id_by_point"),
        ("segment_ids", np.array([1, 0], np.int32), "segment_ids"),
        ("segment_ids", np.array([0, 0], np.int32), "segment_ids"),
        ("segment_ids", np.array([-1, 1], np.int32), "segment_ids"),
    ],
)
def test_invalid_segment_ids_are_rejected(field: str, value: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _prediction(**{field: value})


@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        (np.array([1, 2, 3], np.int64), "begin at zero"),
        (np.array([0, 4, 3], np.int64), "monotonic"),
        (np.array([0, 2, 4], np.int64), "end at len"),
    ],
)
def test_malformed_csr_offsets_are_rejected(offsets: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _prediction(member_offsets=offsets)


@pytest.mark.parametrize("rows", [np.array([-1, 3, 2], np.int64), np.array([0, 3, 5], np.int64)])
def test_out_of_range_member_rows_are_rejected(rows: np.ndarray) -> None:
    with pytest.raises(ValueError, match="member_rows"):
        _prediction(member_rows=rows)


def test_duplicate_member_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _prediction(member_rows=np.array([0, 3, 0], np.int64))


def test_point_and_csr_membership_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        _prediction(member_rows=np.array([0, 2, 3], np.int64))


def test_missing_retained_segment_is_rejected() -> None:
    with pytest.raises(ValueError, match="represented"):
        _prediction(
            segment_id_by_point=np.array([0, -1, -1, 0, -1], np.int32),
            instance_confidence_by_point=np.array([0.8, 0.0, 0.0, 0.8, 0.0], np.float32),
            member_offsets=np.array([0, 2, 2], np.int64),
            member_rows=np.array([0, 3], np.int64),
        )


def test_unassigned_csr_row_is_rejected() -> None:
    with pytest.raises(ValueError, match="unassigned"):
        _prediction(member_rows=np.array([0, 1, 2], np.int64))


def test_unassigned_points_require_zero_instance_confidence() -> None:
    with pytest.raises(ValueError, match="unassigned"):
        _prediction(instance_confidence_by_point=np.array([0.8, 0.1, 0.7, 0.8, 0.0], np.float32))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "v3-single-tree-segmentation-v1", "schema"),
        ("source_id", "", "source_id"),
        ("manifest_source_id", "", "manifest_source_id"),
        ("model_version", "", "model_version"),
        ("model_sha256", "a" * 63, "SHA"),
        ("model_sha256", "A" * 64, "SHA"),
        ("preprocessing_signature", [], "mapping"),
        ("preprocessing_signature", {}, "mapping"),
        ("origin_xyz", (10**400, 0.0, 0.0), "finite numeric"),
    ],
)
def test_provenance_and_model_metadata_are_validated(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _prediction(**{field: value})


def test_contract_has_no_standard_result_or_safety_decision_fields() -> None:
    prediction = _prediction()
    for name in (
        "tree_id",
        "tree_distance_ready",
        "tree_distance_ready_by_point",
        "wire_id_by_point",
        "clearance",
        "growth",
        "risk",
        "warning",
        "algorithm_version",
    ):
        assert not hasattr(prediction, name)


def _aligned_prediction(**changes: object) -> TreeInstanceMlPrediction:
    tree_input = _input_set()
    values: dict[str, object] = {
        "point_count": tree_input.identity.point_count,
        "original_index": tree_input.vegetation_points.original_index,
        "sample_index": tree_input.vegetation_points.sample_index,
        "origin_xyz": tree_input.identity.origin_xyz,
        "semantic_class_by_point": np.array([2, 1, 0], np.uint8),
        "semantic_confidence_by_point": np.array([0.9, 0.8, 0.7], np.float32),
        "segment_id_by_point": np.array([0, -1, 0], np.int32),
        "instance_confidence_by_point": np.array([0.8, 0.0, 0.8], np.float32),
        "segment_ids": np.array([0], np.int32),
        "member_offsets": np.array([0, 2], np.int64),
        "member_rows": np.array([0, 2], np.int64),
        "instance_confidence": np.array([0.8], np.float32),
    }
    values.update(changes)
    return _prediction(**values)


def _set_prediction_array(
    prediction: TreeInstanceMlPrediction, name: str, value: np.ndarray
) -> None:
    value.setflags(write=False)
    object.__setattr__(prediction, name, value)


def _empty_input_set():
    tree_input = _input_set()
    point_count = tree_input.identity.point_count

    def empty_point_arrays(value: object) -> object:
        changes = {}
        for name in value.__dataclass_fields__:
            array = getattr(value, name)
            if (
                isinstance(array, np.ndarray)
                and name != "origin_xyz"
                and array.ndim
                and array.shape[0] == point_count
            ):
                empty = np.empty((0, *array.shape[1:]), dtype=array.dtype)
                empty.setflags(write=False)
                changes[name] = empty
        return replace(value, **changes)

    def empty_array(dtype: np.dtype) -> np.ndarray:
        return np.empty(0, dtype=dtype)

    arrays = [
        empty_array(np.int32),
        empty_array(np.uint8),
        empty_array(np.int32),
        np.array([0], np.int64),
        empty_array(np.int64),
        empty_array(np.bool_),
        empty_array(np.int64),
    ]
    for array in arrays:
        array.setflags(write=False)
    return replace(
        tree_input,
        vegetation_points=empty_point_arrays(tree_input.vegetation_points),
        geometry_features=empty_point_arrays(tree_input.geometry_features),
        heights_above_ground=empty_point_arrays(tree_input.heights_above_ground),
        cluster_id=arrays[0],
        cluster_status=arrays[1],
        cluster_vegetation_id=arrays[2],
        cluster_member_offsets=arrays[3],
        ordered_cluster_member_row_index=arrays[4],
        identity=replace(tree_input.identity, point_count=0),
        eligible_cluster_ids=(),
        review_only_cluster_ids=(),
        excluded_cluster_ids=(),
        cluster_dispositions=(),
        ground_usable_by_point=arrays[5],
        usable_ground_member_count_by_cluster=arrays[6],
    )


def test_alignment_accepts_exact_rows_gapful_segments_and_zero_retained_instances() -> None:
    tree_input = _input_set()
    prediction = _aligned_prediction(
        segment_id_by_point=np.array([2, 5, 9], np.int32),
        instance_confidence_by_point=np.array([0.8, 0.7, 0.6], np.float32),
        segment_ids=np.array([2, 5, 9], np.int32),
        member_offsets=np.array([0, 1, 2, 3], np.int64),
        member_rows=np.array([0, 1, 2], np.int64),
        instance_confidence=np.array([0.8, 0.7, 0.6], np.float32),
    )
    assert validate_tree_instance_ml_prediction_alignment(tree_input, prediction) == 3

    no_instances = _aligned_prediction(
        segment_id_by_point=np.full(3, -1, np.int32),
        instance_confidence_by_point=np.zeros(3, np.float32),
        segment_ids=np.empty(0, np.int32),
        member_offsets=np.array([0], np.int64),
        member_rows=np.empty(0, np.int64),
        instance_confidence=np.empty(0, np.float32),
    )
    assert validate_tree_instance_ml_prediction_alignment(tree_input, no_instances) == 3


def test_alignment_accepts_zero_point_input_and_prediction() -> None:
    tree_input = _empty_input_set()
    prediction = _aligned_prediction(
        point_count=0,
        original_index=np.empty(0, np.int64),
        sample_index=np.empty(0, np.int64),
        origin_xyz=tree_input.identity.origin_xyz,
        semantic_class_by_point=np.empty(0, np.uint8),
        semantic_confidence_by_point=np.empty(0, np.float32),
        segment_id_by_point=np.empty(0, np.int32),
        instance_confidence_by_point=np.empty(0, np.float32),
        segment_ids=np.empty(0, np.int32),
        member_offsets=np.array([0], np.int64),
        member_rows=np.empty(0, np.int64),
        instance_confidence=np.empty(0, np.float32),
    )
    assert validate_tree_instance_ml_prediction_alignment(tree_input, prediction) == 0


def test_alignment_requires_actual_public_contract_types() -> None:
    tree_input = _input_set()
    prediction = _aligned_prediction()
    fake_input = SimpleNamespace(
        identity=tree_input.identity,
        vegetation_points=SimpleNamespace(
            original_index=tree_input.vegetation_points.original_index,
            sample_index=tree_input.vegetation_points.sample_index,
        ),
    )
    fake_prediction = SimpleNamespace(
        **{name: getattr(prediction, name) for name in prediction.__dataclass_fields__}
    )

    assert validate_tree_instance_ml_prediction_alignment(tree_input, prediction) == 3
    for invalid_input in (fake_input, {}, object()):
        with pytest.raises(ValueError, match="tree_input must be a TreeSegmentationInputSet"):
            validate_tree_instance_ml_prediction_alignment(invalid_input, prediction)
    with pytest.raises(ValueError, match="prediction must be a TreeInstanceMlPrediction"):
        validate_tree_instance_ml_prediction_alignment(tree_input, fake_prediction)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_id": "other"}, "source_id"),
        ({"manifest_source_id": "other"}, "manifest_source_id"),
        ({"origin_xyz": (9.0, 2.0, 3.0)}, "origin_xyz"),
    ],
)
def test_alignment_rejects_identity_mismatches(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_tree_instance_ml_prediction_alignment(_input_set(), _aligned_prediction(**changes))


def test_alignment_rejects_point_count_mismatch() -> None:
    prediction = _aligned_prediction()
    object.__setattr__(prediction, "point_count", 4)
    with pytest.raises(ValueError, match="point_count"):
        validate_tree_instance_ml_prediction_alignment(_input_set(), prediction)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("original_index", np.array([11, 10, 12], np.int64), "original_index"),
        ("sample_index", np.array([1, 0, 2], np.int64), "sample_index"),
        ("original_index", np.array([10, 10, 12], np.int64), "original_index"),
        ("original_index", np.array([10, 11, 99], np.int64), "original_index"),
        ("sample_index", np.array([0, 0, 2], np.int64), "sample_index"),
        ("sample_index", np.array([0, 1, 99], np.int64), "sample_index"),
    ],
)
def test_alignment_rejects_row_identity_permutations_duplicates_and_substitutions(
    name: str, value: np.ndarray, message: str
) -> None:
    prediction = _aligned_prediction()
    _set_prediction_array(prediction, name, value)
    with pytest.raises(ValueError, match=message):
        validate_tree_instance_ml_prediction_alignment(_input_set(), prediction)


def test_alignment_does_not_use_xyz_to_reconstruct_identity_or_row_order() -> None:
    tree_input = _input_set()
    prediction = _aligned_prediction()
    _set_prediction_array(prediction, "original_index", np.array([11, 10, 12], np.int64))

    assert np.array_equal(tree_input.vegetation_points.xyz_local, np.zeros((3, 3), np.float32))
    with pytest.raises(ValueError, match="original_index"):
        validate_tree_instance_ml_prediction_alignment(tree_input, prediction)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("segment_id_by_point", np.array([0, -1, 3], np.int32), "retained segment_id"),
        ("member_rows", np.array([0, 2, 3], np.int64), "valid tree input row"),
        ("member_rows", np.array([0, 1], np.int64), "unassigned"),
        ("member_rows", np.array([0], np.int64), "cover every retained"),
    ],
)
def test_alignment_defensively_rejects_invalid_csr_membership(
    name: str, value: np.ndarray, message: str
) -> None:
    prediction = _aligned_prediction()
    _set_prediction_array(prediction, name, value)
    if name == "member_rows":
        _set_prediction_array(prediction, "member_offsets", np.array([0, len(value)], np.int64))
    with pytest.raises(ValueError, match=message):
        validate_tree_instance_ml_prediction_alignment(_input_set(), prediction)


def test_alignment_defensively_rejects_decreasing_offsets_and_duplicate_csr_rows() -> None:
    decreasing = _aligned_prediction()
    _set_prediction_array(decreasing, "segment_ids", np.array([0, 1], np.int32))
    _set_prediction_array(decreasing, "member_offsets", np.array([0, 1, 0], np.int64))
    _set_prediction_array(decreasing, "member_rows", np.empty(0, np.int64))
    with pytest.raises(ValueError, match="monotonic"):
        validate_tree_instance_ml_prediction_alignment(_input_set(), decreasing)

    duplicate = _aligned_prediction()
    _set_prediction_array(duplicate, "segment_id_by_point", np.array([0, -1, 1], np.int32))
    _set_prediction_array(duplicate, "segment_ids", np.array([0, 1], np.int32))
    _set_prediction_array(duplicate, "member_offsets", np.array([0, 1, 2], np.int64))
    _set_prediction_array(duplicate, "member_rows", np.array([0, 0], np.int64))
    with pytest.raises(ValueError, match="duplicate"):
        validate_tree_instance_ml_prediction_alignment(_input_set(), duplicate)


def test_alignment_defensively_rejects_metadata_only_segment_id() -> None:
    prediction = _aligned_prediction()
    _set_prediction_array(prediction, "segment_ids", np.array([0, 1], np.int32))
    _set_prediction_array(prediction, "member_offsets", np.array([0, 2, 2], np.int64))
    with pytest.raises(ValueError, match="represented by pointwise membership"):
        validate_tree_instance_ml_prediction_alignment(_input_set(), prediction)


def test_alignment_rejects_csr_members_assigned_to_the_wrong_segment() -> None:
    prediction = _aligned_prediction()
    _set_prediction_array(prediction, "segment_id_by_point", np.array([0, -1, 1], np.int32))
    _set_prediction_array(prediction, "segment_ids", np.array([0, 1], np.int32))
    _set_prediction_array(prediction, "member_offsets", np.array([0, 1, 2], np.int64))
    _set_prediction_array(prediction, "member_rows", np.array([2, 0], np.int64))
    with pytest.raises(ValueError, match="exactly match"):
        validate_tree_instance_ml_prediction_alignment(_input_set(), prediction)


def test_alignment_does_not_mutate_input_or_prediction_arrays_after_success_or_failure() -> None:
    tree_input = _input_set()
    prediction = _aligned_prediction()
    snapshots = {
        "input_original": tree_input.vegetation_points.original_index.copy(),
        "input_sample": tree_input.vegetation_points.sample_index.copy(),
        "input_xyz": tree_input.vegetation_points.xyz_local.copy(),
        "prediction_segments": prediction.segment_id_by_point.copy(),
        "prediction_offsets": prediction.member_offsets.copy(),
        "prediction_rows": prediction.member_rows.copy(),
    }
    validate_tree_instance_ml_prediction_alignment(tree_input, prediction)
    for name, expected in snapshots.items():
        actual = {
            "input_original": tree_input.vegetation_points.original_index,
            "input_sample": tree_input.vegetation_points.sample_index,
            "input_xyz": tree_input.vegetation_points.xyz_local,
            "prediction_segments": prediction.segment_id_by_point,
            "prediction_offsets": prediction.member_offsets,
            "prediction_rows": prediction.member_rows,
        }[name]
        assert np.array_equal(actual, expected)

    failing_prediction = _aligned_prediction()
    _set_prediction_array(failing_prediction, "member_rows", np.array([0, 1], np.int64))
    input_arrays = {
        "vegetation_original_index": tree_input.vegetation_points.original_index,
        "vegetation_sample_index": tree_input.vegetation_points.sample_index,
    }
    prediction_arrays = {
        name: getattr(failing_prediction, name)
        for name in (
            "original_index",
            "sample_index",
            "semantic_class_by_point",
            "semantic_confidence_by_point",
            "segment_id_by_point",
            "instance_confidence_by_point",
            "segment_ids",
            "member_offsets",
            "member_rows",
            "instance_confidence",
        )
    }
    array_snapshots = {
        name: (array, array.copy(), array.flags.writeable)
        for name, array in {**input_arrays, **prediction_arrays}.items()
    }
    input_origin = tree_input.identity.origin_xyz
    prediction_origin = failing_prediction.origin_xyz
    with pytest.raises(ValueError, match="unassigned"):
        validate_tree_instance_ml_prediction_alignment(tree_input, failing_prediction)
    assert tree_input.identity.origin_xyz == input_origin
    assert failing_prediction.origin_xyz == prediction_origin
    for name, (array, expected, writeable) in array_snapshots.items():
        actual = {**input_arrays, **prediction_arrays}[name]
        assert actual is array
        assert np.array_equal(actual, expected)
        assert actual.flags.writeable is writeable
