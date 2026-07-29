from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import shutil
import zipfile
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import algorithms.vegetation as vegetation
import algorithms.vegetation.tree_artifacts as tree_artifacts
import numpy as np
import pytest
from algorithms.vegetation.tree_artifacts import (
    LoadedTreeSegmentationArtifact,
    TreeSegmentationArtifactError,
    load_tree_segmentation_artifact,
)
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
    TreeSegmentationErrorCode,
    TreeSegmentationInputIdentity,
    TreeSegmentationResult,
    TreeSegmentationStatistics,
    TreeSegmentationStatus,
)
from core.contracts.vegetation_instances import VegetationClusterStatus

ARRAY_NAMES = (
    "segment_id_by_point",
    "tree_id_by_point",
    "tree_distance_ready_by_point",
    "tree_member_offsets",
    "ordered_tree_member_row_index",
)
DISK_DTYPES = {
    "segment_id_by_point": np.dtype("<i4"),
    "tree_id_by_point": np.dtype("<i4"),
    "tree_distance_ready_by_point": np.dtype("|b1"),
    "tree_member_offsets": np.dtype("<i8"),
    "ordered_tree_member_row_index": np.dtype("<i8"),
}


def _canonical(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _digest(character: str) -> str:
    return character * 64


def _identity(point_count: int) -> TreeSegmentationInputIdentity:
    return TreeSegmentationInputIdentity(
        source_id="scene-001",
        manifest_source_id="manifest-001",
        origin_xyz=(1.0, 2.0, 3.0),
        point_count=point_count,
        v2_algorithm_version="v2",
        a2a_algorithm_version="a2a",
        a2b_algorithm_version="a2b",
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


def _record(
    segment_id: int,
    *,
    confirmed: bool,
) -> TreeInstanceRecord:
    if confirmed:
        return TreeInstanceRecord(
            tree_id=0,
            segment_id=segment_id,
            status=TreeSegmentationStatus.CONFIRMED_TREE,
            source_cluster_ids=(7,),
            source_cluster_statuses=(VegetationClusterStatus.CONFIRMED,),
            source_cluster_distance_ready=(True,),
            tree_distance_ready=True,
            confidence=0.95,
            geometry_summary=TreeGeometrySummary(
                apex_xyz_local=(2.0, 3.0, 10.0),
                maximum_hag_m=10.0,
                robust_height_hag_m=9.0,
                vertical_extent_m=9.5,
            ),
            crown_summary=TreeCrownSummary(
                crown_xy_bounds_local=(0.0, 0.0, 2.0, 2.0),
                crown_area_m2=math.pi,
                crown_equivalent_radius_m=1.0,
            ),
        )
    return TreeInstanceRecord(
        tree_id=-1,
        segment_id=segment_id,
        status=TreeSegmentationStatus.REVIEW_REQUIRED,
        source_cluster_ids=(5,),
        source_cluster_statuses=(VegetationClusterStatus.REVIEW_REQUIRED,),
        source_cluster_distance_ready=(False,),
        tree_distance_ready=False,
        confidence=0.5,
        geometry_summary=None,
        crown_summary=None,
    )


def _result(kind: str = "mixed") -> TreeSegmentationResult:
    if kind == "empty":
        records: tuple[TreeInstanceRecord, ...] = ()
        segment = np.empty(0, np.int32)
        tree = np.empty(0, np.int32)
        ready = np.empty(0, np.bool_)
        offsets = np.array([0], np.int64)
        members = np.empty(0, np.int64)
    elif kind == "confirmed":
        records = (_record(0, confirmed=True),)
        segment = np.array([0, 0], np.int32)
        tree = np.array([0, 0], np.int32)
        ready = np.array([True, True], np.bool_)
        offsets = np.array([0, 2], np.int64)
        members = np.array([0, 1], np.int64)
    else:
        records = (_record(0, confirmed=False), _record(1, confirmed=True))
        segment = np.array([0, 0, 1, 1], np.int32)
        tree = np.array([-1, -1, 0, 0], np.int32)
        ready = np.array([False, False, True, True], np.bool_)
        offsets = np.array([0, 2, 4], np.int64)
        members = np.array([0, 1, 2, 3], np.int64)
    status_counts = {
        status: sum(record.status is status for record in records)
        for status in TreeSegmentationStatus
    }
    statistics = TreeSegmentationStatistics(
        input_point_count=len(segment),
        record_count=len(records),
        confirmed_tree_count=status_counts[TreeSegmentationStatus.CONFIRMED_TREE],
        review_required_count=status_counts[TreeSegmentationStatus.REVIEW_REQUIRED],
        provisional_count=status_counts[TreeSegmentationStatus.PROVISIONAL],
        rejected_count=status_counts[TreeSegmentationStatus.REJECTED],
        unresolved_count=status_counts[TreeSegmentationStatus.UNRESOLVED],
        elapsed_seconds=1.25,
        phase_timings_seconds={"load": 0.25},
    )
    return TreeSegmentationResult(
        schema_version=TREE_SEGMENTATION_SCHEMA_VERSION,
        algorithm_version=TREE_SEGMENTATION_ALGORITHM_VERSION,
        source_id="scene-001",
        manifest_source_id="manifest-001",
        origin_xyz=(1.0, 2.0, 3.0),
        config=TreeSegmentationConfig(),
        statistics=statistics,
        records=records,
        segment_id_by_point=segment,
        tree_id_by_point=tree,
        tree_distance_ready_by_point=ready,
        tree_member_offsets=offsets,
        ordered_tree_member_row_index=members,
        input_provenance={"input": {"kind": "synthetic", "index": 1}},
        coordinate_context={"frame": "local", "unit": "m"},
        warnings=("synthetic warning",),
        limitations=("synthetic limitation",),
    )


def _metadata(result: TreeSegmentationResult) -> TreeSegmentationArtifactMetadata:
    arrays = TreeSegmentationArtifactArrayInventory(
        **{
            name: TreeSegmentationArtifactArraySpec(
                DISK_DTYPES[name].str,
                tuple(getattr(result, name).shape),
            )
            for name in ARRAY_NAMES
        }
    )
    return TreeSegmentationArtifactMetadata(
        artifact_schema_version=TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
        result_schema_version=result.schema_version,
        algorithm_version=result.algorithm_version,
        source_identity=_identity(result.statistics.input_point_count),
        config=result.config,
        statistics=result.statistics,
        records=result.records,
        input_provenance=result.input_provenance,
        coordinate_context=result.coordinate_context,
        warnings=result.warnings,
        limitations=result.limitations,
        arrays=arrays,
    )


def _array_values(result: TreeSegmentationResult) -> dict[str, np.ndarray]:
    return {
        name: np.array(getattr(result, name), dtype=DISK_DTYPES[name], copy=True)
        for name in ARRAY_NAMES
    }


def _npy_bytes(array: np.ndarray, *, allow_pickle: bool = False) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=allow_pickle)
    return stream.getvalue()


def _write_npz_members(
    path: Path,
    members: list[tuple[str, bytes]],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, data in members:
            archive.writestr(name, data)


def _refresh_manifest(bundle: Path) -> None:
    metadata = (bundle / TREE_SEGMENTATION_METADATA_FILENAME).read_bytes()
    npz = (bundle / TREE_SEGMENTATION_NPZ_FILENAME).read_bytes()
    manifest = TreeSegmentationArtifactManifest(
        artifact_schema_version=TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
        metadata_sha256=hashlib.sha256(metadata).hexdigest(),
        metadata_bytes=len(metadata),
        npz_sha256=hashlib.sha256(npz).hexdigest(),
        npz_bytes=len(npz),
    )
    (bundle / TREE_SEGMENTATION_MANIFEST_FILENAME).write_bytes(_canonical(manifest.to_dict()))


def _write_bundle(tmp_path: Path, kind: str = "mixed") -> tuple[Path, TreeSegmentationResult]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bundle = tmp_path / f"bundle-{kind}"
    bundle.mkdir()
    result = _result(kind)
    np.savez_compressed(
        bundle / TREE_SEGMENTATION_NPZ_FILENAME,
        **_array_values(result),
    )
    (bundle / TREE_SEGMENTATION_METADATA_FILENAME).write_bytes(
        _canonical(_metadata(result).to_dict())
    )
    _refresh_manifest(bundle)
    return bundle, result


def _replace_npz(
    bundle: Path,
    arrays: dict[str, np.ndarray],
    *,
    names: list[str] | None = None,
    compression: int = zipfile.ZIP_DEFLATED,
    duplicate: str | None = None,
    raw_member: tuple[str, bytes] | None = None,
) -> None:
    member_names = names or list(ARRAY_NAMES)
    members = [(f"{name}.npy", _npy_bytes(arrays[name])) for name in member_names]
    if duplicate is not None:
        members.append((f"{duplicate}.npy", _npy_bytes(arrays[duplicate])))
    if raw_member is not None:
        members.append(raw_member)
    _write_npz_members(
        bundle / TREE_SEGMENTATION_NPZ_FILENAME,
        members,
        compression=compression,
    )
    _refresh_manifest(bundle)


def _rewrite_metadata(bundle: Path, data: bytes) -> None:
    (bundle / TREE_SEGMENTATION_METADATA_FILENAME).write_bytes(data)
    _refresh_manifest(bundle)


def _rewrite_manifest(bundle: Path, mutation: dict[str, Any]) -> None:
    path = bundle / TREE_SEGMENTATION_MANIFEST_FILENAME
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(mutation)
    path.write_bytes(_canonical(value))


def _assert_corrupt(bundle: Path, *, role: str | None = None, field: str | None = None) -> None:
    with pytest.raises(TreeSegmentationArtifactError) as raised:
        load_tree_segmentation_artifact(bundle)
    assert raised.value.code is TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT
    if role is not None:
        assert raised.value.artifact_role == role
    if field is not None:
        assert raised.value.field == field


def test_public_exports_and_loader_only_signature() -> None:
    assert vegetation.LoadedTreeSegmentationArtifact is LoadedTreeSegmentationArtifact
    assert vegetation.TreeSegmentationArtifactError is TreeSegmentationArtifactError
    assert vegetation.load_tree_segmentation_artifact is load_tree_segmentation_artifact
    assert not hasattr(vegetation, "publish_tree_segmentation_artifact")
    assert tuple(inspect.signature(load_tree_segmentation_artifact).parameters) == ("bundle_path",)


def test_loaded_runtime_object_is_frozen_and_validates_direct_inputs(tmp_path: Path) -> None:
    bundle, _ = _write_bundle(tmp_path, "empty")
    loaded = load_tree_segmentation_artifact(bundle)
    with pytest.raises(FrozenInstanceError):
        loaded.bundle_path = tmp_path
    with pytest.raises(ValueError, match="absolute"):
        LoadedTreeSegmentationArtifact(Path("relative"), loaded.result, loaded.source_identity)
    with pytest.raises(ValueError, match="TreeSegmentationResult"):
        LoadedTreeSegmentationArtifact(bundle, object(), loaded.source_identity)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="TreeSegmentationInputIdentity"):
        LoadedTreeSegmentationArtifact(bundle, loaded.result, object())  # type: ignore[arg-type]


def test_direct_constructor_rejects_identity_and_writeability_mismatch(tmp_path: Path) -> None:
    bundle, _ = _write_bundle(tmp_path, "confirmed")
    loaded = load_tree_segmentation_artifact(bundle)
    with pytest.raises(ValueError, match="source_id"):
        LoadedTreeSegmentationArtifact(
            bundle,
            loaded.result,
            replace(loaded.source_identity, source_id="different"),
        )
    loaded.result.segment_id_by_point.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        LoadedTreeSegmentationArtifact(bundle, loaded.result, loaded.source_identity)


@pytest.mark.parametrize("kind", ["empty", "confirmed", "mixed"])
@pytest.mark.parametrize("as_string", [False, True])
def test_valid_round_trip_is_complete_detached_and_read_only(
    tmp_path: Path, kind: str, as_string: bool
) -> None:
    bundle, expected = _write_bundle(tmp_path, kind)
    loaded = load_tree_segmentation_artifact(str(bundle) if as_string else bundle)
    assert loaded.bundle_path == bundle
    assert loaded.source_identity == _identity(expected.statistics.input_point_count)
    assert loaded.result.config == expected.config
    assert loaded.result.statistics == expected.statistics
    assert loaded.result.records == expected.records
    assert loaded.result.input_provenance == expected.input_provenance
    assert loaded.result.coordinate_context == expected.coordinate_context
    assert loaded.result.warnings == expected.warnings
    assert loaded.result.limitations == expected.limitations
    for name in ARRAY_NAMES:
        actual = getattr(loaded.result, name)
        assert np.array_equal(actual, getattr(expected, name))
        assert actual.dtype == getattr(expected, name).dtype
        assert actual.flags.c_contiguous
        assert actual.flags.owndata
        assert not actual.flags.writeable
        with pytest.raises(ValueError):
            actual[0:1] = actual[0:1]
    shutil.rmtree(bundle)
    assert loaded.result.statistics.input_point_count == len(loaded.result.segment_id_by_point)


def test_repeated_loads_do_not_share_arrays(tmp_path: Path) -> None:
    bundle, _ = _write_bundle(tmp_path)
    first = load_tree_segmentation_artifact(bundle)
    second = load_tree_segmentation_artifact(bundle)
    for name in ARRAY_NAMES:
        assert getattr(first.result, name) is not getattr(second.result, name)


@pytest.mark.parametrize(
    "path_factory",
    [
        lambda tmp: Path("relative"),
        lambda tmp: tmp / "missing",
        lambda tmp: tmp / "file.txt",
        lambda tmp: tmp / "empty",
    ],
)
def test_rejects_invalid_bundle_roots(tmp_path: Path, path_factory: Any) -> None:
    path = path_factory(tmp_path)
    if path.name == "file.txt":
        path.write_text("x", encoding="utf-8")
    elif path.name == "empty":
        path.mkdir()
    with pytest.raises(TreeSegmentationArtifactError) as raised:
        load_tree_segmentation_artifact(path)
    assert raised.value.code is TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT
    assert not isinstance(raised.value, FileNotFoundError)


@pytest.mark.parametrize(
    "mutation",
    ["missing_manifest", "missing_metadata", "missing_npz", "extra", "nested", "alternate_case"],
)
def test_exact_directory_inventory(tmp_path: Path, mutation: str) -> None:
    bundle, _ = _write_bundle(tmp_path)
    targets = {
        "missing_manifest": TREE_SEGMENTATION_MANIFEST_FILENAME,
        "missing_metadata": TREE_SEGMENTATION_METADATA_FILENAME,
        "missing_npz": TREE_SEGMENTATION_NPZ_FILENAME,
    }
    if mutation in targets:
        (bundle / targets[mutation]).unlink()
    elif mutation == "extra":
        (bundle / "README.txt").write_text("extra", encoding="utf-8")
    elif mutation == "nested":
        (bundle / "nested").mkdir()
    else:
        path = bundle / TREE_SEGMENTATION_METADATA_FILENAME
        path.rename(bundle / TREE_SEGMENTATION_METADATA_FILENAME.upper())
    _assert_corrupt(bundle, role="bundle", field="inventory")


def test_rejects_root_and_child_symlinks_when_supported(tmp_path: Path) -> None:
    bundle, _ = _write_bundle(tmp_path)
    root_link = tmp_path / "root-link"
    try:
        root_link.symlink_to(bundle, target_is_directory=True)
    except OSError:
        pytest.skip("filesystem does not permit symlink creation")
    _assert_corrupt(root_link, role="bundle", field="path")
    metadata = bundle / TREE_SEGMENTATION_METADATA_FILENAME
    target = tmp_path / "metadata-target"
    metadata.rename(target)
    metadata.symlink_to(target)
    _assert_corrupt(bundle)


@pytest.mark.parametrize(
    "transform",
    [
        lambda data: data[:-1],
        lambda data: data[:-1] + b"\r\n",
        lambda data: data + b"\n",
        lambda data: data[:-1] + b" \n",
        lambda data: b"\xef\xbb\xbf" + data,
        lambda data: data.replace(b",", b", ", 1),
    ],
)
@pytest.mark.parametrize("role", ["manifest", "metadata"])
def test_rejects_noncanonical_json_bytes(tmp_path: Path, transform: Any, role: str) -> None:
    bundle, _ = _write_bundle(tmp_path)
    filename = (
        TREE_SEGMENTATION_MANIFEST_FILENAME
        if role == "manifest"
        else TREE_SEGMENTATION_METADATA_FILENAME
    )
    path = bundle / filename
    changed = transform(path.read_bytes())
    if role == "manifest":
        path.write_bytes(changed)
    else:
        _rewrite_metadata(bundle, changed)
    _assert_corrupt(bundle, role=role)


@pytest.mark.parametrize("role", ["manifest", "metadata"])
@pytest.mark.parametrize(
    "case",
    [
        "invalid_utf8",
        "malformed",
        "non_object",
        "duplicate",
        "duplicate_nested",
        "nan",
        "infinity",
        "negative_infinity",
    ],
)
def test_strict_json_parser_rejects_invalid_inputs(tmp_path: Path, role: str, case: str) -> None:
    bundle, _ = _write_bundle(tmp_path)
    filename = (
        TREE_SEGMENTATION_MANIFEST_FILENAME
        if role == "manifest"
        else TREE_SEGMENTATION_METADATA_FILENAME
    )
    path = bundle / filename
    original = path.read_bytes()
    if case == "invalid_utf8":
        changed = b"\xff\n"
    elif case == "malformed":
        changed = b"{\n"
    elif case == "non_object":
        changed = b"[]\n"
    elif case == "duplicate":
        changed = b'{"x":1,"x":2}\n'
    elif case == "duplicate_nested":
        changed = b'{"outer":{"x":1,"x":2}}\n'
    elif case == "nan":
        changed = b'{"x":NaN}\n'
    elif case == "infinity":
        changed = b'{"x":Infinity}\n'
    else:
        changed = b'{"x":-Infinity}\n'
    if role == "manifest":
        path.write_bytes(changed)
    else:
        _rewrite_metadata(bundle, changed)
    _assert_corrupt(bundle, role=role)
    assert original != changed


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("artifact_schema_version", "other", TreeSegmentationErrorCode.SCHEMA_MISMATCH),
        ("result_schema_version", "other", TreeSegmentationErrorCode.SCHEMA_MISMATCH),
        (
            "algorithm_version",
            "other",
            TreeSegmentationErrorCode.ALGORITHM_VERSION_MISMATCH,
        ),
    ],
)
def test_metadata_version_errors_are_mapped(
    tmp_path: Path, field: str, value: str, code: TreeSegmentationErrorCode
) -> None:
    bundle, _ = _write_bundle(tmp_path)
    path = bundle / TREE_SEGMENTATION_METADATA_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw[field] = value
    _rewrite_metadata(bundle, _canonical(raw))
    with pytest.raises(TreeSegmentationArtifactError) as raised:
        load_tree_segmentation_artifact(bundle)
    assert raised.value.code is code
    assert raised.value.field == field


def test_manifest_schema_error_is_mapped(tmp_path: Path) -> None:
    bundle, _ = _write_bundle(tmp_path)
    _rewrite_manifest(bundle, {"artifact_schema_version": "other"})
    with pytest.raises(TreeSegmentationArtifactError) as raised:
        load_tree_segmentation_artifact(bundle)
    assert raised.value.code is TreeSegmentationErrorCode.SCHEMA_MISMATCH


@pytest.mark.parametrize("role", ["manifest", "metadata"])
@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_json_contract_rejects_missing_and_unknown_fields(
    tmp_path: Path, role: str, mutation: str
) -> None:
    bundle, _ = _write_bundle(tmp_path)
    filename = (
        TREE_SEGMENTATION_MANIFEST_FILENAME
        if role == "manifest"
        else TREE_SEGMENTATION_METADATA_FILENAME
    )
    path = bundle / filename
    raw = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing":
        raw.pop("artifact_schema_version")
    else:
        raw["unknown"] = True
    if role == "manifest":
        path.write_bytes(_canonical(raw))
    else:
        _rewrite_metadata(bundle, _canonical(raw))
    _assert_corrupt(bundle, role=role, field="contract")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metadata_bytes", 0),
        ("npz_bytes", 0),
        ("metadata_bytes", 1),
        ("npz_bytes", 1),
        ("metadata_sha256", "0" * 64),
        ("npz_sha256", "0" * 64),
    ],
)
def test_manifest_size_and_hash_failures_identify_field(
    tmp_path: Path, field: str, value: Any
) -> None:
    bundle, _ = _write_bundle(tmp_path)
    _rewrite_manifest(bundle, {field: value})
    with pytest.raises(TreeSegmentationArtifactError) as raised:
        load_tree_segmentation_artifact(bundle)
    assert raised.value.code is TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT
    expected_field = field if value == 0 else field.split("_", maxsplit=1)[1]
    assert raised.value.field == expected_field
    assert raised.value.artifact_role in {"manifest", "metadata", "npz"}


def test_manifest_size_ceiling(tmp_path: Path) -> None:
    bundle, _ = _write_bundle(tmp_path)
    (bundle / TREE_SEGMENTATION_MANIFEST_FILENAME).write_bytes(b" " * (64 * 1024 + 1))
    _assert_corrupt(bundle, role="manifest", field="bytes")


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "nested", "case", "stored"])
def test_npz_inventory_is_exact(tmp_path: Path, mutation: str) -> None:
    bundle, result = _write_bundle(tmp_path)
    arrays = _array_values(result)
    if mutation == "missing":
        _replace_npz(bundle, arrays, names=list(ARRAY_NAMES[:-1]))
    elif mutation == "extra":
        _replace_npz(bundle, arrays, raw_member=("extra.npy", _npy_bytes(np.array([1]))))
    elif mutation == "duplicate":
        _replace_npz(bundle, arrays, duplicate=ARRAY_NAMES[0])
    elif mutation == "nested":
        _replace_npz(
            bundle,
            arrays,
            names=list(ARRAY_NAMES[1:]),
            raw_member=(f"nested/{ARRAY_NAMES[0]}.npy", _npy_bytes(arrays[ARRAY_NAMES[0]])),
        )
    elif mutation == "case":
        _replace_npz(
            bundle,
            arrays,
            names=list(ARRAY_NAMES[1:]),
            raw_member=(
                f"{ARRAY_NAMES[0].upper()}.npy",
                _npy_bytes(arrays[ARRAY_NAMES[0]]),
            ),
        )
    else:
        _replace_npz(bundle, arrays, compression=zipfile.ZIP_STORED)
    _assert_corrupt(bundle, role="npz", field="archive")


@pytest.mark.parametrize("missing_name", ARRAY_NAMES)
def test_npz_rejects_each_missing_member(tmp_path: Path, missing_name: str) -> None:
    bundle, result = _write_bundle(tmp_path)
    arrays = _array_values(result)
    _replace_npz(
        bundle,
        arrays,
        names=[name for name in ARRAY_NAMES if name != missing_name],
    )
    _assert_corrupt(bundle, role="npz", field="archive")


def test_reordered_npz_members_are_accepted(tmp_path: Path) -> None:
    bundle, result = _write_bundle(tmp_path)
    arrays = _array_values(result)
    _replace_npz(bundle, arrays, names=list(reversed(ARRAY_NAMES)))
    loaded = load_tree_segmentation_artifact(bundle)
    assert np.array_equal(loaded.result.segment_id_by_point, result.segment_id_by_point)


def test_npz_replacement_after_initial_integrity_check_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, result = _write_bundle(tmp_path)
    npz_path = bundle / TREE_SEGMENTATION_NPZ_FILENAME
    original_bytes = npz_path.read_bytes()
    arrays = _array_values(result)
    replacement_path = tmp_path / "replacement.npz"
    _write_npz_members(
        replacement_path,
        [(f"{name}.npy", _npy_bytes(arrays[name])) for name in reversed(ARRAY_NAMES)],
    )
    replacement_bytes = replacement_path.read_bytes()
    assert hashlib.sha256(replacement_bytes).digest() != hashlib.sha256(original_bytes).digest()

    original_verify = tree_artifacts._verify_file
    npz_verification_attempts = 0

    def replace_after_first_npz_verification(
        path: Path,
        *,
        role: str,
        expected_size: int,
        expected_sha256: str,
        bundle_path: Path,
    ) -> None:
        nonlocal npz_verification_attempts
        if role == "npz":
            npz_verification_attempts += 1
        original_verify(
            path,
            role=role,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            bundle_path=bundle_path,
        )
        if role == "npz" and npz_verification_attempts == 1:
            path.write_bytes(replacement_bytes)

    monkeypatch.setattr(
        tree_artifacts,
        "_verify_file",
        replace_after_first_npz_verification,
    )

    with pytest.raises(TreeSegmentationArtifactError) as raised:
        load_tree_segmentation_artifact(bundle)

    assert npz_verification_attempts == 2
    assert raised.value.code is TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT
    assert raised.value.artifact_role == "npz"
    assert raised.value.field in {"bytes", "sha256"}


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("segment_id_by_point", lambda value: value.astype(np.int64)),
        ("tree_id_by_point", lambda value: value.astype(np.uint32)),
        ("tree_distance_ready_by_point", lambda value: value.astype(np.int8)),
        ("tree_member_offsets", lambda value: value.astype(np.int32)),
        ("ordered_tree_member_row_index", lambda value: value.astype(">i8")),
        ("segment_id_by_point", lambda value: value.reshape((2, 2))),
        ("tree_id_by_point", lambda value: np.array(["x"] * len(value))),
    ],
)
def test_rejects_wrong_npz_dtype_or_shape(tmp_path: Path, name: str, replacement: Any) -> None:
    bundle, result = _write_bundle(tmp_path)
    arrays = _array_values(result)
    arrays[name] = replacement(arrays[name])
    _replace_npz(bundle, arrays)
    _assert_corrupt(bundle, role="npz", field="archive")


def test_rejects_object_structured_and_subarray_headers(tmp_path: Path) -> None:
    for index, invalid in enumerate(
        (
            np.array([object()], dtype=object),
            np.array([(1,)], dtype=[("field", "<i4")]),
            np.zeros(1, dtype=np.dtype(("<i4", (2,)))),
        )
    ):
        bundle, result = _write_bundle(tmp_path / str(index))
        arrays = _array_values(result)
        members = [(f"{name}.npy", _npy_bytes(arrays[name])) for name in ARRAY_NAMES[1:]]
        members.append(
            (
                f"{ARRAY_NAMES[0]}.npy",
                _npy_bytes(invalid, allow_pickle=invalid.dtype.hasobject),
            )
        )
        _write_npz_members(bundle / TREE_SEGMENTATION_NPZ_FILENAME, members)
        _refresh_manifest(bundle)
        _assert_corrupt(bundle, role="npz", field="archive")


@pytest.mark.parametrize(
    "mutation",
    ["malformed", "unsupported_version", "uncompressed_size", "fortran_order"],
)
def test_rejects_invalid_npy_header_forms(tmp_path: Path, mutation: str) -> None:
    bundle, result = _write_bundle(tmp_path)
    arrays = _array_values(result)
    target = ARRAY_NAMES[0]
    original = _npy_bytes(arrays[target])
    if mutation == "malformed":
        changed = b"not-an-npy"
    elif mutation == "unsupported_version":
        changed = original[:6] + b"\x03\x00" + original[8:]
    elif mutation == "uncompressed_size":
        changed = original + b"x"
    else:
        changed = original.replace(b"False", b"True ", 1)
    members = [
        (
            f"{name}.npy",
            changed if name == target else _npy_bytes(arrays[name]),
        )
        for name in ARRAY_NAMES
    ]
    _write_npz_members(bundle / TREE_SEGMENTATION_NPZ_FILENAME, members)
    _refresh_manifest(bundle)
    _assert_corrupt(bundle, role="npz", field="archive")


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("segment_id_by_point", np.array([0, 0, 0, 1], np.int32)),
        ("tree_id_by_point", np.array([-1, -1, -1, 0], np.int32)),
        ("tree_distance_ready_by_point", np.array([False, False, False, True], np.bool_)),
        ("tree_member_offsets", np.array([0, 3, 2], np.int64)),
        ("ordered_tree_member_row_index", np.array([0, 0, 2, 3], np.int64)),
        ("ordered_tree_member_row_index", np.array([0, 1, 2, 8], np.int64)),
    ],
)
def test_result_invariant_failures_are_wrapped_with_cause(
    tmp_path: Path, name: str, replacement: np.ndarray
) -> None:
    bundle, result = _write_bundle(tmp_path)
    arrays = _array_values(result)
    arrays[name] = replacement
    _replace_npz(bundle, arrays)
    with pytest.raises(TreeSegmentationArtifactError) as raised:
        load_tree_segmentation_artifact(bundle)
    assert raised.value.code is TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT
    assert raised.value.artifact_role == "result"
    assert raised.value.__cause__ is not None


def test_truncated_and_malformed_npz_are_rejected_before_result(tmp_path: Path) -> None:
    bundle, _ = _write_bundle(tmp_path)
    path = bundle / TREE_SEGMENTATION_NPZ_FILENAME
    path.write_bytes(path.read_bytes()[:20])
    _refresh_manifest(bundle)
    _assert_corrupt(bundle, role="npz", field="archive")


def test_current_working_directory_does_not_affect_absolute_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _ = _write_bundle(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    assert load_tree_segmentation_artifact(bundle).bundle_path == bundle
