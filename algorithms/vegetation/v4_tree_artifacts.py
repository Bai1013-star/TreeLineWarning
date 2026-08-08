"""V4 tree-segmentation artifact publication and loading (PLAN-FROZEN).

``publish_v4_tree_segmentation_artifact`` writes the JSON metadata, the NPZ
arrays and a digest-bound manifest through a sibling staging directory with
re-read validation and a single atomic rename.  ``load_v4_tree_segmentation_artifact``
re-validates every byte, schema, array, count, identity and digest before
reconstructing the ``V4TreeSegmentationResult``.  Any failure removes the
staging directory and never leaves a partial output root.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts.v4_tree_segmentation import (
    V4_TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
    V4_TREE_SEGMENTATION_JSON_FILENAME,
    V4_TREE_SEGMENTATION_MANIFEST_FILENAME,
    V4_TREE_SEGMENTATION_NPZ_FILENAME,
    LoadedV4TreeSegmentationArtifact,
    V4TreeSegmentationArtifactManifest,
    V4TreeSegmentationError,
    V4TreeSegmentationErrorCode,
    V4TreeSegmentationOutputReference,
    V4TreeSegmentationResult,
    v4_tree_segmentation_result_json,
    validate_v4_tree_segmentation_result,
)

from algorithms.vegetation.frv_artifacts import publish_directory, write_canonical_json

# The exact dense/ragged arrays persisted in the NPZ, in canonical order.
_NPZ_ARRAY_FIELDS = (
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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_spec(array: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(array.dtype),
        "shape": [int(dimension) for dimension in array.shape],
    }


def _result_arrays(result: V4TreeSegmentationResult) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(result, name)) for name in _NPZ_ARRAY_FIELDS
    }


def publish_v4_tree_segmentation_artifact(
    result: V4TreeSegmentationResult,
    output_root: Path,
    *,
    scratch_dir: Path | None = None,
) -> V4TreeSegmentationOutputReference:
    """Publish a validated V4 result artifact with atomic directory semantics."""

    if type(result) is not V4TreeSegmentationResult:
        raise TypeError("result must be a V4TreeSegmentationResult")
    validate_v4_tree_segmentation_result(result)
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a pathlib.Path")
    if scratch_dir is None:
        scratch_dir = output_root.parent
    if not isinstance(scratch_dir, Path):
        raise TypeError("scratch_dir must be a pathlib.Path")

    arrays = _result_arrays(result)
    npz_buffer = io.BytesIO()
    np.savez(npz_buffer, **arrays)
    npz_bytes = npz_buffer.getvalue()
    npz_sha = _sha256_bytes(npz_bytes)
    metadata_payload = v4_tree_segmentation_result_json(result)
    metadata_bytes = json.dumps(
        metadata_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    metadata_sha = _sha256_bytes(metadata_bytes)
    inventory = {
        name: _array_spec(np.asarray(arrays[name])) for name in _NPZ_ARRAY_FIELDS
    }
    manifest = V4TreeSegmentationArtifactManifest(
        artifact_schema_version=V4_TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
        result_digest=result.result_digest,
        metadata_sha256=metadata_sha,
        metadata_bytes=len(metadata_bytes),
        npz_sha256=npz_sha,
        npz_bytes=len(npz_bytes),
        array_inventory=inventory,
    )

    def publish(staging: Path) -> None:
        (staging / V4_TREE_SEGMENTATION_JSON_FILENAME).write_bytes(metadata_bytes)
        (staging / V4_TREE_SEGMENTATION_NPZ_FILENAME).write_bytes(npz_bytes)
        write_canonical_json(staging / V4_TREE_SEGMENTATION_MANIFEST_FILENAME, manifest.to_dict())

    publish_directory(scratch_dir=scratch_dir, target_dir=output_root, publish=publish)
    return V4TreeSegmentationOutputReference(
        root_path=output_root,
        metadata_json_path=output_root / V4_TREE_SEGMENTATION_JSON_FILENAME,
        npz_path=output_root / V4_TREE_SEGMENTATION_NPZ_FILENAME,
        manifest_path=output_root / V4_TREE_SEGMENTATION_MANIFEST_FILENAME,
    )


def _fail(code: V4TreeSegmentationErrorCode, field_name: str) -> V4TreeSegmentationError:
    return V4TreeSegmentationError(code=code, field_name=field_name)


def load_v4_tree_segmentation_artifact(
    root_path: Path,
) -> LoadedV4TreeSegmentationArtifact:
    """Load and fully re-validate a published V4 result artifact."""

    if not isinstance(root_path, Path) or not root_path.is_dir():
        raise _fail(V4TreeSegmentationErrorCode.INPUT_ROOT_INVALID, "root")
    metadata_path = root_path / V4_TREE_SEGMENTATION_JSON_FILENAME
    npz_path = root_path / V4_TREE_SEGMENTATION_NPZ_FILENAME
    manifest_path = root_path / V4_TREE_SEGMENTATION_MANIFEST_FILENAME
    for path in (metadata_path, npz_path, manifest_path):
        if not path.is_file():
            raise _fail(V4TreeSegmentationErrorCode.INPUT_ROOT_INVALID, path.name)
    try:
        manifest_payload = json.loads(manifest_path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise _fail(V4TreeSegmentationErrorCode.ARTIFACT_HASH_FAILED, "manifest") from exc
    try:
        manifest = V4TreeSegmentationArtifactManifest.from_dict(manifest_payload)
    except ValueError as exc:
        raise _fail(V4TreeSegmentationErrorCode.ARTIFACT_HASH_FAILED, "manifest") from exc

    metadata_bytes = metadata_path.read_bytes()
    npz_bytes = npz_path.read_bytes()
    if _sha256_bytes(metadata_bytes) != manifest.metadata_sha256:
        raise _fail(V4TreeSegmentationErrorCode.ARTIFACT_HASH_FAILED, "metadata_sha256")
    if _sha256_bytes(npz_bytes) != manifest.npz_sha256:
        raise _fail(V4TreeSegmentationErrorCode.ARTIFACT_HASH_FAILED, "npz_sha256")
    if len(metadata_bytes) != manifest.metadata_bytes:
        raise _fail(V4TreeSegmentationErrorCode.ARTIFACT_HASH_FAILED, "metadata_bytes")
    if len(npz_bytes) != manifest.npz_bytes:
        raise _fail(V4TreeSegmentationErrorCode.ARTIFACT_HASH_FAILED, "npz_bytes")

    try:
        metadata_payload = json.loads(metadata_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _fail(V4TreeSegmentationErrorCode.ARTIFACT_SERIALIZATION_FAILED, "metadata") from exc
    expected_fields = {
        "schema_version",
        "algorithm_version",
        "source_identity",
        "config",
        "handoff_digest",
        "validated_view_digest",
        "coverage_report_digest",
        "apex_records",
        "trunk_records",
        "trunk_apex_association_records",
        "structural_path_records",
        "rotating_profile_records",
        "hypothesis_records",
        "slice_records",
        "relation_records",
        "tree_records",
        "audit",
        "result_digest",
        "array_fields",
    }
    if set(metadata_payload) != expected_fields:
        raise _fail(V4TreeSegmentationErrorCode.ARTIFACT_SERIALIZATION_FAILED, "metadata_fields")

    arrays: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as npz:
        for name in _NPZ_ARRAY_FIELDS:
            if name not in npz.files:
                raise _fail(V4TreeSegmentationErrorCode.ARTIFACT_SERIALIZATION_FAILED, name)
            array = np.asarray(npz[name])
            spec = manifest.array_inventory[name]
            if str(array.dtype) != spec["dtype"]:
                raise _fail(V4TreeSegmentationErrorCode.ARTIFACT_SERIALIZATION_FAILED, name)
            if list(array.shape) != list(spec["shape"]):
                raise _fail(V4TreeSegmentationErrorCode.ARTIFACT_SERIALIZATION_FAILED, name)
            arrays[name] = array
    if len(np.load(io.BytesIO(npz_bytes), allow_pickle=False).files) != len(_NPZ_ARRAY_FIELDS):
        raise _fail(V4TreeSegmentationErrorCode.ARTIFACT_SERIALIZATION_FAILED, "npz_extra")

    payload = dict(metadata_payload)
    payload.pop("array_fields", None)
    for name in _NPZ_ARRAY_FIELDS:
        payload[name] = arrays[name]
    try:
        result = V4TreeSegmentationResult.from_dict(payload)
        validate_v4_tree_segmentation_result(result)
    except (ValueError, TypeError) as exc:
        raise _fail(V4TreeSegmentationErrorCode.ARTIFACT_SERIALIZATION_FAILED, "result") from exc
    if result.result_digest != manifest.result_digest:
        raise _fail(V4TreeSegmentationErrorCode.ARTIFACT_HASH_FAILED, "result_digest")
    reference = V4TreeSegmentationOutputReference(
        root_path=root_path,
        metadata_json_path=metadata_path,
        npz_path=npz_path,
        manifest_path=manifest_path,
    )
    return LoadedV4TreeSegmentationArtifact(
        result=result, output_reference=reference, manifest=manifest
    )


__all__ = [
    "publish_v4_tree_segmentation_artifact",
    "load_v4_tree_segmentation_artifact",
]
