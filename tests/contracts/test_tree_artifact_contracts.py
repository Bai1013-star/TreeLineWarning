from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from core import contracts
from core.contracts import (
    TREE_SEGMENTATION_ALGORITHM_VERSION,
    TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
    TREE_SEGMENTATION_MANIFEST_FILENAME,
    TREE_SEGMENTATION_METADATA_FILENAME,
    TREE_SEGMENTATION_NPZ_FILENAME,
    TREE_SEGMENTATION_SCHEMA_VERSION,
    TreeCrownSummary,
    TreeGeometrySummary,
    TreeInstanceRecord,
    TreeSegmentationArtifactArrayInventory,
    TreeSegmentationArtifactArraySpec,
    TreeSegmentationArtifactManifest,
    TreeSegmentationArtifactMetadata,
    TreeSegmentationConfig,
    TreeSegmentationInputIdentity,
    TreeSegmentationStatistics,
    TreeSegmentationStatus,
    VegetationClusterStatus,
)

_DEFAULT = object()


def _digest(character: str = "a") -> str:
    return character * 64


def _identity(point_count: object = 4) -> TreeSegmentationInputIdentity:
    return TreeSegmentationInputIdentity(
        source_id="synthetic-source",
        manifest_source_id="synthetic-manifest",
        origin_xyz=(1.0, 2.0, 3.0),
        point_count=point_count,
        v2_algorithm_version="synthetic-v2",
        a2a_algorithm_version="synthetic-a2a",
        a2b_algorithm_version="synthetic-a2b",
        v2_npz_sha256=_digest("0"),
        v2_json_sha256=_digest("1"),
        a2a_npz_sha256=_digest("2"),
        a2a_json_sha256=_digest("3"),
        a2b_npz_sha256=_digest("4"),
        a2b_json_sha256=_digest("5"),
        a2a_input_a1_npz_sha256=_digest("6"),
        a2a_input_a1_json_sha256=_digest("7"),
        a2b_input_a1_npz_sha256=_digest("8"),
        a2b_input_a1_json_sha256=_digest("9"),
        a2b_input_a2a_npz_sha256=_digest("a"),
        a2b_input_a2a_json_sha256=_digest("b"),
        v2_input_v1_npz_sha256=_digest("c"),
        v2_input_v1_json_sha256=_digest("d"),
        v2_input_a2a_npz_sha256=_digest("e"),
        v2_input_a2a_json_sha256=_digest("f"),
        v2_input_a2b_npz_sha256=_digest("0"),
        v2_input_a2b_json_sha256=_digest("1"),
    )


def _config() -> TreeSegmentationConfig:
    return TreeSegmentationConfig()


def _record(
    segment_id: int = 0,
    *,
    status: TreeSegmentationStatus = TreeSegmentationStatus.REVIEW_REQUIRED,
    geometry: bool = True,
    crown: bool = True,
) -> TreeInstanceRecord:
    confirmed = status is TreeSegmentationStatus.CONFIRMED_TREE
    geometry_summary = (
        TreeGeometrySummary(
            apex_xyz_local=(1.0, 2.0, 8.0),
            maximum_hag_m=8.0 if confirmed else None,
            robust_height_hag_m=7.0 if confirmed else None,
            vertical_extent_m=6.0,
        )
        if geometry
        else None
    )
    crown_summary = (
        TreeCrownSummary(
            crown_xy_bounds_local=(0.0, 0.0, 2.0, 2.0),
            crown_area_m2=math.pi,
            crown_equivalent_radius_m=1.0,
        )
        if crown
        else None
    )
    return TreeInstanceRecord(
        tree_id=segment_id if confirmed else -1,
        segment_id=segment_id,
        status=status,
        source_cluster_ids=(segment_id + 10,),
        source_cluster_statuses=(VegetationClusterStatus.CONFIRMED,),
        source_cluster_distance_ready=(True,),
        tree_distance_ready=confirmed,
        confidence=0.5,
        geometry_summary=geometry_summary,
        crown_summary=crown_summary,
    )


def _statistics(
    point_count: int,
    records: tuple[TreeInstanceRecord, ...],
) -> TreeSegmentationStatistics:
    return TreeSegmentationStatistics(
        input_point_count=point_count,
        record_count=len(records),
        confirmed_tree_count=sum(
            record.status is TreeSegmentationStatus.CONFIRMED_TREE for record in records
        ),
        review_required_count=sum(
            record.status is TreeSegmentationStatus.REVIEW_REQUIRED for record in records
        ),
        provisional_count=sum(
            record.status is TreeSegmentationStatus.PROVISIONAL for record in records
        ),
        rejected_count=sum(record.status is TreeSegmentationStatus.REJECTED for record in records),
        unresolved_count=sum(
            record.status is TreeSegmentationStatus.UNRESOLVED for record in records
        ),
        elapsed_seconds=0.0,
    )


def _inventory(
    point_count: int = 4,
    record_count: int = 0,
    member_count: int = 0,
) -> TreeSegmentationArtifactArrayInventory:
    return TreeSegmentationArtifactArrayInventory(
        segment_id_by_point=TreeSegmentationArtifactArraySpec("<i4", (point_count,)),
        tree_id_by_point=TreeSegmentationArtifactArraySpec("<i4", (point_count,)),
        tree_distance_ready_by_point=TreeSegmentationArtifactArraySpec("|b1", (point_count,)),
        tree_member_offsets=TreeSegmentationArtifactArraySpec("<i8", (record_count + 1,)),
        ordered_tree_member_row_index=TreeSegmentationArtifactArraySpec("<i8", (member_count,)),
    )


def _metadata(
    *,
    point_count: int = 4,
    records: tuple[TreeInstanceRecord, ...] = (),
    member_count: int = 0,
    identity: TreeSegmentationInputIdentity | None = None,
    statistics: TreeSegmentationStatistics | None = None,
    arrays: TreeSegmentationArtifactArrayInventory | None = None,
    input_provenance: object = _DEFAULT,
    coordinate_context: object = _DEFAULT,
    warnings: object = (),
    limitations: object = (),
) -> TreeSegmentationArtifactMetadata:
    return TreeSegmentationArtifactMetadata(
        artifact_schema_version=TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
        result_schema_version=TREE_SEGMENTATION_SCHEMA_VERSION,
        algorithm_version=TREE_SEGMENTATION_ALGORITHM_VERSION,
        source_identity=identity or _identity(point_count),
        config=_config(),
        statistics=statistics or _statistics(point_count, records),
        records=records,
        input_provenance=(
            {"v2": {"sha256": _digest()}} if input_provenance is _DEFAULT else input_provenance
        ),
        coordinate_context=(
            {"frame": "LOCAL_CARTESIAN_XYZ", "axis_order": ["x", "y", "z"]}
            if coordinate_context is _DEFAULT
            else coordinate_context
        ),
        warnings=warnings,
        limitations=limitations,
        arrays=arrays or _inventory(point_count, len(records), member_count),
    )


def test_constants_and_public_exports_are_exact() -> None:
    assert TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION == "tree-segmentation-artifact-v1"
    assert TREE_SEGMENTATION_NPZ_FILENAME == "tree_segmentation.npz"
    assert TREE_SEGMENTATION_METADATA_FILENAME == "tree_segmentation.json"
    assert TREE_SEGMENTATION_MANIFEST_FILENAME == "tree_segmentation_manifest.json"
    for name in (
        "TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION",
        "TREE_SEGMENTATION_NPZ_FILENAME",
        "TREE_SEGMENTATION_METADATA_FILENAME",
        "TREE_SEGMENTATION_MANIFEST_FILENAME",
        "TreeSegmentationArtifactArraySpec",
        "TreeSegmentationArtifactArrayInventory",
        "TreeSegmentationArtifactManifest",
        "TreeSegmentationArtifactMetadata",
        "TreeSegmentationResult",
        "TreeSegmentationInputIdentity",
    ):
        assert hasattr(contracts, name)


def test_array_spec_normalizes_and_copies_shape_and_is_frozen() -> None:
    shape = [np.int64(3)]
    spec = TreeSegmentationArtifactArraySpec("<i4", shape)
    shape[0] = 99
    assert spec.shape == (3,)
    assert type(spec.shape[0]) is int
    assert spec.to_dict() == {"dtype": "<i4", "shape": [3]}
    with pytest.raises(FrozenInstanceError):
        spec.dtype = "<i8"


@pytest.mark.parametrize("dimension", [0, 7, np.int32(5), np.int64(9)])
def test_array_spec_accepts_non_negative_python_and_numpy_integers(
    dimension: object,
) -> None:
    spec = TreeSegmentationArtifactArraySpec("<i4", (dimension,))
    assert spec.shape == (int(dimension),)
    assert type(spec.shape[0]) is int


@pytest.mark.parametrize(
    "shape",
    [
        (True,),
        (np.bool_(False),),
        (1.0,),
        (-1,),
        (),
        (1, 2),
        ((1,),),
        "1",
        {"dimension": 1},
        1,
    ],
)
def test_array_spec_rejects_invalid_shapes(shape: object) -> None:
    with pytest.raises(ValueError):
        TreeSegmentationArtifactArraySpec("<i4", shape)


@pytest.mark.parametrize("dtype", ["", " <i4", "<i4 ", 4, None])
def test_array_spec_rejects_invalid_dtype(dtype: object) -> None:
    with pytest.raises(ValueError):
        TreeSegmentationArtifactArraySpec(dtype, (1,))


def test_array_spec_strict_json_round_trip_and_list_normalization() -> None:
    data = {"dtype": "<i4", "shape": [2]}
    parsed = TreeSegmentationArtifactArraySpec.from_dict(data)
    assert parsed.shape == (2,)
    assert parsed.to_dict() == data
    assert TreeSegmentationArtifactArraySpec.from_dict(parsed.to_dict()) == parsed


@pytest.mark.parametrize(
    "data",
    [
        {"dtype": "<i4"},
        {"shape": [1]},
        {"dtype": "<i4", "shape": [1], "rank": 1},
        None,
        {"dtype": "<i4", "shape": None},
        {"dtype": "<i4", "shape": {"dimension": 1}},
    ],
)
def test_array_spec_rejects_invalid_json(data: object) -> None:
    with pytest.raises(ValueError):
        TreeSegmentationArtifactArraySpec.from_dict(data)


def test_array_inventory_has_exact_dtypes_and_deterministic_keys() -> None:
    inventory = _inventory(4, 2, 3)
    data = inventory.to_dict()
    assert list(data) == [
        "segment_id_by_point",
        "tree_id_by_point",
        "tree_distance_ready_by_point",
        "tree_member_offsets",
        "ordered_tree_member_row_index",
    ]
    assert {name: value["dtype"] for name, value in data.items()} == {
        "segment_id_by_point": "<i4",
        "tree_id_by_point": "<i4",
        "tree_distance_ready_by_point": "|b1",
        "tree_member_offsets": "<i8",
        "ordered_tree_member_row_index": "<i8",
    }
    assert set(data) == set(TreeSegmentationArtifactArrayInventory.__dataclass_fields__)


@pytest.mark.parametrize(
    ("point_count", "record_count", "member_count"),
    [(0, 0, 0), (8, 3, 5)],
)
def test_array_inventory_strict_nested_round_trip(
    point_count: int,
    record_count: int,
    member_count: int,
) -> None:
    inventory = _inventory(point_count, record_count, member_count)
    assert TreeSegmentationArtifactArrayInventory.from_dict(inventory.to_dict()) == inventory


@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        ("segment_id_by_point", "int32"),
        ("tree_id_by_point", "=i4"),
        ("tree_distance_ready_by_point", "bool"),
        ("tree_member_offsets", "int64"),
        ("ordered_tree_member_row_index", ">i8"),
        ("segment_id_by_point", "|i4"),
        ("tree_member_offsets", "object"),
        ("tree_distance_ready_by_point", " |b1"),
    ],
)
def test_array_inventory_rejects_wrong_member_dtype(name: str, dtype: str) -> None:
    inventory = _inventory()
    with pytest.raises(ValueError):
        replace(
            inventory,
            **{name: TreeSegmentationArtifactArraySpec(dtype, getattr(inventory, name).shape)},
        )


def test_array_inventory_direct_constructor_requires_specs() -> None:
    inventory = _inventory()
    with pytest.raises(ValueError):
        replace(inventory, segment_id_by_point={"dtype": "<i4", "shape": [4]})


@pytest.mark.parametrize("mutation", ["missing", "unknown", "wrong_nested"])
def test_array_inventory_rejects_invalid_json(mutation: str) -> None:
    data = _inventory().to_dict()
    if mutation == "missing":
        data.pop("segment_id_by_point")
    elif mutation == "unknown":
        data["source_cluster_id_by_point"] = {"dtype": "<i4", "shape": [4]}
    else:
        data["segment_id_by_point"] = None
    with pytest.raises(ValueError):
        TreeSegmentationArtifactArrayInventory.from_dict(data)


def test_manifest_accepts_and_normalizes_counts_and_is_frozen() -> None:
    manifest = TreeSegmentationArtifactManifest(
        TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
        _digest("a"),
        np.int64(0),
        _digest("b"),
        np.int32(12),
    )
    assert manifest.metadata_bytes == 0
    assert manifest.npz_bytes == 12
    assert type(manifest.metadata_bytes) is int
    assert type(manifest.npz_bytes) is int
    assert set(manifest.to_dict()) == set(manifest.__dataclass_fields__)
    assert "manifest_sha256" not in manifest.to_dict()
    with pytest.raises(FrozenInstanceError):
        manifest.npz_bytes = 1


@pytest.mark.parametrize(
    "digest",
    [
        "A" * 64,
        "a" * 63,
        "a" * 65,
        "g" * 64,
        f" {_digest()}",
        f"{_digest()} ",
        f"sha256:{_digest()}",
        _digest().encode(),
        1,
    ],
)
def test_manifest_rejects_noncanonical_hashes(digest: object) -> None:
    with pytest.raises(ValueError):
        TreeSegmentationArtifactManifest(
            TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
            digest,
            1,
            _digest("b"),
            2,
        )


@pytest.mark.parametrize("count", [-1, True, np.bool_(False), 1.5, "1", [], {}])
def test_manifest_rejects_invalid_byte_counts(count: object) -> None:
    with pytest.raises(ValueError):
        TreeSegmentationArtifactManifest(
            TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
            _digest(),
            count,
            _digest("b"),
            1,
        )


def test_manifest_rejects_unsupported_version() -> None:
    with pytest.raises(ValueError):
        TreeSegmentationArtifactManifest(
            "tree-segmentation-artifact-v2", _digest(), 1, _digest(), 1
        )


def test_manifest_strict_json_round_trip() -> None:
    manifest = TreeSegmentationArtifactManifest(
        TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
        _digest("a"),
        1,
        _digest("b"),
        2,
    )
    assert TreeSegmentationArtifactManifest.from_dict(manifest.to_dict()) == manifest
    for changed in (
        {key: value for key, value in manifest.to_dict().items() if key != "npz_bytes"},
        {**manifest.to_dict(), "bundle_id": "not-allowed"},
        {**manifest.to_dict(), "metadata_bytes": "1"},
        None,
    ):
        with pytest.raises(ValueError):
            TreeSegmentationArtifactManifest.from_dict(changed)


def test_metadata_accepts_valid_empty_and_non_empty_contracts() -> None:
    empty = _metadata(point_count=0)
    records = (_record(0), _record(1, geometry=False, crown=False))
    populated = _metadata(point_count=4, records=records, member_count=4)
    assert empty.records == ()
    assert empty.arrays.tree_member_offsets.shape == (1,)
    assert populated.records == records
    assert populated.arrays.ordered_tree_member_row_index.shape == (4,)
    assert populated.source_identity is not None
    assert populated.config is not None
    assert populated.statistics is not None
    assert populated.arrays is not None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_schema_version", "tree-segmentation-artifact-v2"),
        ("result_schema_version", "tree-segmentation-v2"),
        ("algorithm_version", "unknown-algorithm"),
        ("source_identity", {}),
        ("config", {}),
        ("statistics", {}),
        ("arrays", {}),
    ],
)
def test_metadata_requires_exact_versions_and_nested_types(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(_metadata(), **{field: value})


def test_metadata_copies_records_and_rejects_invalid_record_containers() -> None:
    record_list = [_record(0)]
    metadata = _metadata(records=record_list)
    record_list.clear()
    assert len(metadata.records) == 1
    assert isinstance(metadata.records, tuple)
    for value in ("records", {"record": _record(0)}, 1, [_record(0), object()]):
        with pytest.raises(ValueError):
            _metadata(
                records=value,
                statistics=_statistics(4, ()),
                arrays=_inventory(4, 0, 0),
            )


@pytest.mark.parametrize("case", ["out_of_order", "duplicate", "skipped", "negative"])
def test_metadata_requires_canonical_record_segment_ids(case: str) -> None:
    first = _record(0)
    second = _record(1)
    if case == "out_of_order":
        records = (second, first)
    elif case == "duplicate":
        object.__setattr__(second, "segment_id", 0)
        records = (first, second)
    elif case == "skipped":
        object.__setattr__(second, "segment_id", 2)
        records = (first, second)
    else:
        object.__setattr__(first, "segment_id", -1)
        records = (first,)
    with pytest.raises(ValueError):
        _metadata(records=records)


def test_metadata_normalizes_string_sequences_without_trimming_or_deduplication() -> None:
    warnings = [" repeated ", " repeated "]
    limitations = ["limit"]
    metadata = _metadata(warnings=warnings, limitations=limitations)
    warnings.append("later")
    limitations.clear()
    assert metadata.warnings == (" repeated ", " repeated ")
    assert metadata.limitations == ("limit",)
    assert _metadata(warnings=(), limitations=()).warnings == ()
    for field, value in (
        ("warnings", "warning"),
        ("limitations", b"limit"),
        ("warnings", ["ok", None]),
        ("limitations", [["nested"]]),
    ):
        with pytest.raises(ValueError):
            _metadata(**{field: value})


def test_metadata_deep_copies_json_safe_mappings() -> None:
    provenance = {"source": {"hashes": [_digest()]}}
    context = {"frame": {"name": "local"}, "nullable": None, "valid": True}
    metadata = _metadata(input_provenance=provenance, coordinate_context=context)
    provenance["source"]["hashes"][0] = _digest("b")
    context["frame"]["name"] = "changed"
    assert metadata.input_provenance["source"]["hashes"] == [_digest()]
    assert metadata.coordinate_context["frame"]["name"] == "local"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_provenance", {1: "bad"}),
        ("coordinate_context", {"nested": {2: "bad"}}),
        ("input_provenance", {"value": float("nan")}),
        ("coordinate_context", {"value": float("inf")}),
        ("coordinate_context", {"value": float("-inf")}),
        ("input_provenance", {"value": np.array([1])}),
        ("coordinate_context", {"value": object()}),
        ("input_provenance", {"value": {1, 2}}),
        ("coordinate_context", {"value": b"bytes"}),
        ("input_provenance", None),
        ("coordinate_context", None),
    ],
)
def test_metadata_rejects_non_json_safe_mappings(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _metadata(**{field: value})


def test_metadata_cross_field_counts_must_match() -> None:
    with pytest.raises(ValueError, match="input_point_count"):
        _metadata(point_count=4, statistics=_statistics(3, ()))
    with pytest.raises(ValueError, match="record_count"):
        _metadata(
            point_count=4,
            records=(_record(0),),
            statistics=_statistics(4, ()),
        )


def test_metadata_accepts_valid_mixed_status_statistics() -> None:
    records = tuple(
        _record(segment_id, status=status)
        for segment_id, status in enumerate(TreeSegmentationStatus)
    )
    metadata = _metadata(
        point_count=len(records),
        records=records,
        statistics=_statistics(len(records), records),
    )
    assert metadata.statistics.confirmed_tree_count == 1
    assert metadata.statistics.review_required_count == 1
    assert metadata.statistics.provisional_count == 1
    assert metadata.statistics.rejected_count == 1
    assert metadata.statistics.unresolved_count == 1


@pytest.mark.parametrize(
    ("field_name", "balancing_field"),
    [
        ("confirmed_tree_count", "review_required_count"),
        ("review_required_count", "provisional_count"),
        ("provisional_count", "rejected_count"),
        ("rejected_count", "unresolved_count"),
        ("unresolved_count", "confirmed_tree_count"),
    ],
)
def test_metadata_rejects_each_status_count_mismatch(
    field_name: str,
    balancing_field: str,
) -> None:
    records = tuple(
        _record(segment_id, status=status)
        for segment_id, status in enumerate(TreeSegmentationStatus)
    )
    statistics = replace(
        _statistics(len(records), records),
        **{field_name: 0, balancing_field: 2},
    )
    assert statistics.record_count == len(records)
    assert sum(
        getattr(statistics, name)
        for name in (
            "confirmed_tree_count",
            "review_required_count",
            "provisional_count",
            "rejected_count",
            "unresolved_count",
        )
    ) == len(records)
    with pytest.raises(
        ValueError,
        match=rf"statistics\.{field_name} must match records",
    ):
        _metadata(
            point_count=len(records),
            records=records,
            statistics=statistics,
        )


@pytest.mark.parametrize(
    "name",
    [
        "segment_id_by_point",
        "tree_id_by_point",
        "tree_distance_ready_by_point",
    ],
)
def test_metadata_each_point_array_shape_must_match_point_count(name: str) -> None:
    arrays = _inventory(4, 0, 0)
    wrong = replace(getattr(arrays, name), shape=(3,))
    with pytest.raises(ValueError, match=name):
        _metadata(arrays=replace(arrays, **{name: wrong}))


def test_metadata_offsets_shape_must_match_record_count_plus_one() -> None:
    records = (_record(0),)
    arrays = _inventory(4, 1, 0)
    arrays = replace(
        arrays,
        tree_member_offsets=TreeSegmentationArtifactArraySpec("<i8", (1,)),
    )
    with pytest.raises(ValueError, match="tree_member_offsets"):
        _metadata(records=records, arrays=arrays)


@pytest.mark.parametrize("member_count", [0, 4])
def test_metadata_member_count_may_be_zero_or_point_count(member_count: int) -> None:
    metadata = _metadata(point_count=4, member_count=member_count)
    assert metadata.arrays.ordered_tree_member_row_index.shape == (member_count,)


def test_metadata_rejects_member_count_greater_than_point_count() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        _metadata(point_count=4, member_count=5)


def test_metadata_json_has_exact_structure_and_round_trips_nested_contracts() -> None:
    records = (_record(0), _record(1, geometry=False, crown=False))
    metadata = _metadata(
        records=records,
        member_count=2,
        warnings=("warning",),
        limitations=("limitation",),
    )
    data = metadata.to_dict()
    assert list(data) == [
        "artifact_schema_version",
        "result_schema_version",
        "algorithm_version",
        "source_identity",
        "config",
        "statistics",
        "records",
        "input_provenance",
        "coordinate_context",
        "warnings",
        "limitations",
        "arrays",
    ]
    assert data["records"][0]["geometry_summary"] is not None
    assert data["records"][0]["crown_summary"] is not None
    assert data["records"][1]["geometry_summary"] is None
    assert data["records"][1]["crown_summary"] is None
    assert TreeSegmentationArtifactMetadata.from_dict(data).to_dict() == data
    assert TreeSegmentationArtifactMetadata.from_dict(data).warnings == ("warning",)
    assert TreeSegmentationArtifactMetadata.from_dict(data).limitations == ("limitation",)
    assert not {
        "segment_id_by_point",
        "tree_id_by_point",
        "tree_distance_ready_by_point",
        "tree_member_offsets",
        "ordered_tree_member_row_index",
        "metadata_sha256",
        "npz_sha256",
        "metadata_filename",
        "npz_filename",
        "path",
    } & set(data)


@pytest.mark.parametrize("case", ["missing", "unknown", "null", "wrong_nested"])
def test_metadata_json_rejects_invalid_root_and_nested_structures(case: str) -> None:
    data = _metadata().to_dict()
    if case == "missing":
        data.pop("algorithm_version")
    elif case == "unknown":
        data["notes"] = "not allowed"
    elif case == "null":
        data["source_identity"] = None
    else:
        data["arrays"] = "not a mapping"
    with pytest.raises(ValueError):
        TreeSegmentationArtifactMetadata.from_dict(data)


@pytest.mark.parametrize("case", ["nested_missing", "nested_unknown", "wrong_record"])
def test_metadata_json_preserves_nested_strictness(case: str) -> None:
    data = _metadata(records=(_record(0),)).to_dict()
    if case == "nested_missing":
        data["source_identity"].pop("point_count")
    elif case == "nested_unknown":
        data["config"]["future_option"] = True
    else:
        data["records"][0] = None
    with pytest.raises(ValueError):
        TreeSegmentationArtifactMetadata.from_dict(data)
