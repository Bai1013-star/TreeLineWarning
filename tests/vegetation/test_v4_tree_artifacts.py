"""V4 tree-segmentation artifact I/O tests (PLAN-FROZEN consolidated plan)."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from algorithms.vegetation.v4_tree_artifacts import (
    load_v4_tree_segmentation_artifact,
    publish_v4_tree_segmentation_artifact,
)
from algorithms.vegetation.v4_tree_segmentation import segment_v4_trees
from core.contracts.v4_tree_segmentation import (
    V4_TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
    V4_TREE_SEGMENTATION_JSON_FILENAME,
    V4_TREE_SEGMENTATION_MANIFEST_FILENAME,
    V4_TREE_SEGMENTATION_NPZ_FILENAME,
    V4TreeHypothesisRecord,
    V4TreeInstanceRecord,
    V4TreeSegmentationArtifactManifest,
    V4TreeSegmentationError,
    V4TreeSegmentationResult,
    v4_member_index_digest,
    v4_tree_segmentation_result_json,
)

from tests.vegetation.test_v4_tree_segmentation import (
    make_config,
    make_handoff,
    two_trees_scene,
)

_NPZ_FIELDS = (
    "original_index",
    "xyz_local_m",
    "hag_m",
    "assignment_state_by_point",
    "tree_id_by_point",
    "assignment_confidence_by_point",
    "assignment_reason_flags_by_point",
    "tree_member_offsets",
    "ordered_tree_member_original_index",
    "crown_boundary_vertices",
    "crown_boundary_offsets",
)


def _clone_result(result: V4TreeSegmentationResult, **mutations) -> V4TreeSegmentationResult:
    clone = V4TreeSegmentationResult.__new__(V4TreeSegmentationResult)
    for name in V4TreeSegmentationResult.__dataclass_fields__:
        if name == "result_digest":
            continue
        object.__setattr__(clone, name, mutations.get(name, getattr(result, name)))
    object.__setattr__(clone, "result_digest", clone.computed_result_digest())
    return clone


def _write_corrupt_artifact(result: V4TreeSegmentationResult, root: Path) -> Path:
    """Write a semantically corrupted (but rehashed) artifact without validation."""
    root.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(getattr(result, name)) for name in _NPZ_FIELDS}
    npz_buffer = io.BytesIO()
    np.savez(npz_buffer, **arrays)
    npz_bytes = npz_buffer.getvalue()
    metadata_bytes = json.dumps(
        v4_tree_segmentation_result_json(result),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    inventory = {
        name: {
            "dtype": str(np.asarray(getattr(result, name)).dtype),
            "shape": [int(dimension) for dimension in np.asarray(getattr(result, name)).shape],
        }
        for name in _NPZ_FIELDS
    }
    manifest = V4TreeSegmentationArtifactManifest(
        artifact_schema_version=V4_TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
        result_digest=result.result_digest,
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        metadata_bytes=len(metadata_bytes),
        npz_sha256=hashlib.sha256(npz_bytes).hexdigest(),
        npz_bytes=len(npz_bytes),
        array_inventory=inventory,
    )
    (root / V4_TREE_SEGMENTATION_JSON_FILENAME).write_bytes(metadata_bytes)
    (root / V4_TREE_SEGMENTATION_NPZ_FILENAME).write_bytes(npz_bytes)
    (root / V4_TREE_SEGMENTATION_MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_dict()), "utf-8"
    )
    return root


@pytest.fixture()
def result():
    handoff = make_handoff(two_trees_scene())
    return segment_v4_trees(handoff, make_config())


def _publish(result, tmp_path: Path, name: str = "out"):
    root = tmp_path / name
    reference = publish_v4_tree_segmentation_artifact(
        result, root, scratch_dir=tmp_path
    )
    return root, reference


def test_artifact_round_trip(result, tmp_path: Path) -> None:
    root, reference = _publish(result, tmp_path)
    assert sorted(path.name for path in root.iterdir()) == sorted(
        [
            V4_TREE_SEGMENTATION_JSON_FILENAME,
            V4_TREE_SEGMENTATION_MANIFEST_FILENAME,
            V4_TREE_SEGMENTATION_NPZ_FILENAME,
        ]
    )
    loaded = load_v4_tree_segmentation_artifact(root)
    assert loaded.result.result_digest == result.result_digest
    assert loaded.result.config == result.config
    assert (loaded.result.original_index == result.original_index).all()
    assert np.allclose(loaded.result.xyz_local_m, result.xyz_local_m)
    assert (loaded.result.assignment_state_by_point == result.assignment_state_by_point).all()
    assert loaded.manifest.result_digest == result.result_digest


def test_artifact_rejects_corrupt_npz(result, tmp_path: Path) -> None:
    root, _reference = _publish(result, tmp_path)
    npz_path = root / V4_TREE_SEGMENTATION_NPZ_FILENAME
    data = bytearray(npz_path.read_bytes())
    data[-1] ^= 0xFF
    npz_path.write_bytes(bytes(data))
    with pytest.raises(V4TreeSegmentationError):
        load_v4_tree_segmentation_artifact(root)


def test_artifact_rejects_corrupt_metadata(result, tmp_path: Path) -> None:
    root, _reference = _publish(result, tmp_path)
    metadata_path = root / V4_TREE_SEGMENTATION_JSON_FILENAME
    payload = json.loads(metadata_path.read_text("utf-8"))
    payload["result_digest"] = "0" * 64
    metadata_path.write_text(json.dumps(payload), "utf-8")
    with pytest.raises(V4TreeSegmentationError):
        load_v4_tree_segmentation_artifact(root)


def test_artifact_rejects_missing_file(result, tmp_path: Path) -> None:
    root, _reference = _publish(result, tmp_path)
    (root / V4_TREE_SEGMENTATION_NPZ_FILENAME).unlink()
    with pytest.raises(V4TreeSegmentationError):
        load_v4_tree_segmentation_artifact(root)


def test_artifact_manifest_inventory(result, tmp_path: Path) -> None:
    root, _reference = _publish(result, tmp_path)
    manifest_path = root / V4_TREE_SEGMENTATION_MANIFEST_FILENAME
    manifest = V4TreeSegmentationArtifactManifest.from_dict(
        json.loads(manifest_path.read_text("utf-8"))
    )
    expected_arrays = {
        "original_index",
        "xyz_local_m",
        "hag_m",
        "assignment_state_by_point",
        "tree_id_by_point",
        "assignment_confidence_by_point",
        "assignment_reason_flags_by_point",
        "tree_member_offsets",
        "ordered_tree_member_original_index",
        "crown_boundary_vertices",
        "crown_boundary_offsets",
    }
    assert set(manifest.array_inventory) == expected_arrays
    assert manifest.result_digest == result.result_digest


def test_artifact_rejects_overwrite(result, tmp_path: Path) -> None:
    root, _reference = _publish(result, tmp_path)
    with pytest.raises(ValueError):
        publish_v4_tree_segmentation_artifact(result, root, scratch_dir=tmp_path)


def test_artifact_rejects_extra_npz_array(result, tmp_path: Path) -> None:
    root, _reference = _publish(result, tmp_path)
    npz_path = root / V4_TREE_SEGMENTATION_NPZ_FILENAME
    arrays = dict(np.load(npz_path, allow_pickle=False))
    arrays["extra_array"] = np.zeros(3, dtype=np.float64)
    import io

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    npz_path.write_bytes(buffer.getvalue())
    manifest_path = root / V4_TREE_SEGMENTATION_MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text("utf-8"))
    payload["npz_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), "utf-8")
    with pytest.raises(V4TreeSegmentationError):
        load_v4_tree_segmentation_artifact(root)


def test_artifact_no_partial_output_on_failure(result, tmp_path: Path) -> None:
    target = tmp_path / "partial"
    with pytest.raises(TypeError):
        publish_v4_tree_segmentation_artifact(
            object(), target, scratch_dir=tmp_path  # type: ignore[arg-type]
        )
    assert not target.exists()
    assert list(tmp_path.glob(".frv_publish_*")) == []


def test_artifact_empty_result_round_trip(tmp_path: Path) -> None:
    handoff = make_handoff([])
    result = segment_v4_trees(handoff, make_config())
    root, _reference = _publish(result, tmp_path)
    loaded = load_v4_tree_segmentation_artifact(root)
    assert loaded.result.result_digest == result.result_digest
    assert len(loaded.result.tree_records) == 0


def _rehash_and_load_rejects(result: V4TreeSegmentationResult, tmp_path: Path) -> None:
    """Write a rehashed corrupted artifact and require the loader to reject it."""
    root = _write_corrupt_artifact(result, tmp_path / "corrupt")
    with pytest.raises(V4TreeSegmentationError):
        load_v4_tree_segmentation_artifact(root)


def test_rehashed_out_of_range_tree_id_rejected(result, tmp_path: Path) -> None:
    """V4A-IR-012/015 #18: out-of-range Tree ID is rejected even when every
    outer/semantic digest is recomputed consistently."""
    if len(result.tree_records) == 0:
        pytest.skip("scene produced no tree records")
    corrupted = _clone_result(
        result,
        tree_id_by_point=_ro(np.full(
            result.tree_id_by_point.shape,
            len(result.tree_records) + 5,
            dtype="<i4",
        )),
    )
    _rehash_and_load_rejects(corrupted, tmp_path)


def test_rehashed_bad_member_digest_rejected(result, tmp_path: Path) -> None:
    """V4A-IR-012/015 #19: a wrong tree member digest is rejected after rehash."""
    if not result.tree_records:
        pytest.skip("scene produced no tree records")
    tree_records = []
    for tree in result.tree_records:
        clone = V4TreeInstanceRecord.__new__(V4TreeInstanceRecord)
        for name in V4TreeInstanceRecord.__dataclass_fields__:
            value = getattr(tree, name)
            if name == "member_original_index_digest":
                value = "0" * 64
            object.__setattr__(clone, name, value)
        tree_records.append(clone)
    corrupted = _clone_result(result, tree_records=tuple(tree_records))
    _rehash_and_load_rejects(corrupted, tmp_path)


def test_rehashed_dangling_apex_ref_rejected(result, tmp_path: Path) -> None:
    """V4A-IR-012/015 #20: a dangling tree apex reference is rejected after rehash."""
    if not result.tree_records:
        pytest.skip("scene produced no tree records")
    tree_records = []
    for tree in result.tree_records:
        clone = V4TreeInstanceRecord.__new__(V4TreeInstanceRecord)
        for name in V4TreeInstanceRecord.__dataclass_fields__:
            value = getattr(tree, name)
            if name == "primary_apex_candidate_id":
                value = "apex-does-not-exist"
            object.__setattr__(clone, name, value)
        tree_records.append(clone)
    corrupted = _clone_result(result, tree_records=tuple(tree_records))
    _rehash_and_load_rejects(corrupted, tmp_path)


def test_rehashed_dangling_fragment_ref_rejected(result, tmp_path: Path) -> None:
    """V4A-IR-012: a fragment ref cannot resolve when no formal fragment
    inventory is persisted, even after the complete artifact is rehashed."""

    if not result.hypothesis_records:
        pytest.skip("scene produced no hypothesis records")
    records = []
    for position, record in enumerate(result.hypothesis_records):
        clone = V4TreeHypothesisRecord.__new__(V4TreeHypothesisRecord)
        for name in V4TreeHypothesisRecord.__dataclass_fields__:
            value = getattr(record, name)
            if position == 0 and name == "fragment_refs":
                value = ("frag-does-not-exist",)
            object.__setattr__(clone, name, value)
        records.append(clone)
    corrupted = _clone_result(result, hypothesis_records=tuple(records))
    _rehash_and_load_rejects(corrupted, tmp_path)


def test_rehashed_non_increasing_member_block_rejected(result, tmp_path: Path) -> None:
    """V4A-IR-013/015 #21: interleaved-but-not-increasing member block rejected."""
    if not result.tree_records:
        pytest.skip("scene produced no tree records")
    members = np.asarray(result.ordered_tree_member_original_index, dtype="<i8").copy()
    if members.shape[0] < 2:
        pytest.skip("not enough members to reorder")
    members[0], members[1] = members[1], members[0]
    corrupted = _clone_result(
        result, ordered_tree_member_original_index=_ro(members)
    )
    _rehash_and_load_rejects(corrupted, tmp_path)


def test_rehashed_dense_ragged_tree_disagreement_rejected(result, tmp_path: Path) -> None:
    """V4A-IR-012: individually valid ragged blocks cannot be rebound to the
    wrong dense tree IDs, even when offsets/counts/member digests are rehashed."""

    if len(result.tree_records) < 2:
        pytest.skip("scene produced fewer than two trees")
    offsets = np.asarray(result.tree_member_offsets, dtype=np.int64)
    members = np.asarray(result.ordered_tree_member_original_index, dtype=np.int64)
    blocks = [
        members[int(offsets[index]) : int(offsets[index + 1])].copy()
        for index in range(len(result.tree_records))
    ]
    blocks[0], blocks[1] = blocks[1], blocks[0]
    changed_members = _ro(np.concatenate(blocks).astype("<i8", copy=False))
    changed_offsets = [0]
    for block in blocks:
        changed_offsets.append(changed_offsets[-1] + len(block))
    changed_records = []
    for record, block in zip(result.tree_records, blocks, strict=True):
        clone = V4TreeInstanceRecord.__new__(V4TreeInstanceRecord)
        for name in V4TreeInstanceRecord.__dataclass_fields__:
            value = getattr(record, name)
            if name == "member_count":
                value = len(block)
            elif name == "member_original_index_digest":
                value = v4_member_index_digest(block)
            object.__setattr__(clone, name, value)
        changed_records.append(clone)
    corrupted = _clone_result(
        result,
        tree_member_offsets=_ro(np.asarray(changed_offsets, dtype="<i8")),
        ordered_tree_member_original_index=changed_members,
        tree_records=tuple(changed_records),
    )
    _rehash_and_load_rejects(corrupted, tmp_path)


def _ro(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array
