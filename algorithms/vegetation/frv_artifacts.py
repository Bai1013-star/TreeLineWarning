"""FRV membership artifact publication helpers (PLAN-FROZEN).

This module implements the frozen FRVBLK1 block framing, the FRVART1 artifact
digest domain, the exact payload schema inventory, canonical JSON file writing,
and the atomic temp-directory validate/rename publication discipline.  It owns
no membership decisions, no scope construction, and no provider access; it only
turns already-finalized in-memory arrays and headers into verifiable bytes and
published directories.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts.frv import (
    FRV_ARTIFACT_PAYLOAD_SCHEMA_VERSION,
    REQUIRED_FRV_ARRAY_SPECS,
    FRVArtifactArrayDescriptor,
    FRVArtifactBlockDescriptor,
    FRVArtifactPayloadSchema,
    FullResolutionVegetationMembershipArtifact,
    canonical_frv_json_bytes,
    frv_digest,
)

# ---------------------------------------------------------------------------
# Frozen framing constants
# ---------------------------------------------------------------------------

FRV_BLOCK_MAGIC = b"FRVBLK1\0"
FRV_EVIDENCE_BLOCK_MAGIC = b"FRVEVDBLK1\0"
V4_NORMAL_BLOCK_MAGIC = b"V4NRMBLK1\0"
V4_REVIEW_BLOCK_MAGIC = b"V4REVBLK1\0"
FRV_ART1_MAGIC = "FRVART1"
FRV_ART1_BYTE_MAGIC = b"FRVART1\0"

_KNOWN_BLOCK_MAGICS = (
    FRV_BLOCK_MAGIC,
    FRV_EVIDENCE_BLOCK_MAGIC,
    V4_NORMAL_BLOCK_MAGIC,
    V4_REVIEW_BLOCK_MAGIC,
)

# Exact required array specs (field_name, dtype_code, shape_rule, enum_version).
_REQUIRED_ARRAY_SPECS = REQUIRED_FRV_ARRAY_SPECS

# Frozen optional arrays always published by Session B (candidate_confidence is
# deliberately never published).
_OPTIONAL_ARRAY_SPECS = (
    ("raw_hag", "<f4", "[N]", None),
    ("clamped_hag", "<f4", "[N]", None),
    ("hag_quality_flags", "<u4", "[N]", "qf-v1"),
    ("hag_reason_flags", "<u4", "[N]", "rf-v1"),
)

_SCHEMA_LITERALS = {
    "payload_encoding_id": "FRV-BLOCK-PAYLOAD-1",
    "payload_encoder_version": "1",
    "payload_compression_algorithm": "NONE",
    "payload_compression_version": "1",
    "membership_state_schema_version": "ms-v1",
    "reason_flag_schema_version": "rf-v1",
    "quality_flag_schema_version": "qf-v1",
    "hag_evidence_schema_version": "he-v1",
    "segmentation_hag_readiness_schema_version": "shr-v1",
    "block_schema_version": "frv-artifact-block-v1",
}

# Block-header exact keys (FRVBLK1 / FRVEVDBLK1 canonical headers).
_BLOCK_HEADER_KEYS = (
    "artifact_id",
    "block_id",
    "payload_schema_digest",
    "record_count",
    "original_index_start",
    "original_index_stop",
)

# FRVART1 declared-header exact keys (validated by the A1 validator).
_ART1_HEADER_KEYS = (
    "magic",
    "artifact_schema_version",
    "algorithm_version",
    "decision_rule_version",
    "artifact_id",
    "source_content_identity_digest",
    "scope_union_id",
    "configuration_digest",
    "payload_schema_digest",
    "block_schema_version",
    "block_count",
    "ordered_block_ids",
)


# ---------------------------------------------------------------------------
# Canonical JSON helpers
# ---------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical FRV JSON bytes for *value* (UTF-8, sorted keys)."""

    return canonical_frv_json_bytes(value)


def write_canonical_json(path: Path, value: Any) -> None:
    """Write canonical JSON (UTF-8, no BOM, trailing single LF) to *path*."""

    data = canonical_frv_json_bytes(value)
    path.write_bytes(data + b"\n")


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 of the file at *path*."""

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _normalize_zero(value: float) -> float:
    return value + 0.0 if value == 0.0 else value


# ---------------------------------------------------------------------------
# Payload schema construction
# ---------------------------------------------------------------------------


def _array_descriptor(
    *,
    field_name: str,
    dtype_code: str,
    shape_rule: str,
    required: bool,
    enum_version: str | None,
    storage_key: str,
) -> FRVArtifactArrayDescriptor:
    return FRVArtifactArrayDescriptor(
        field_name=field_name,
        dtype_code=dtype_code,
        shape_rule=shape_rule,
        required=required,
        nullable=False,
        finite_required=dtype_code in ("<f4", "<f8"),
        enum_or_flag_schema_version=enum_version,
        storage_key=storage_key,
    )


def build_frv_payload_schema(
    *,
    scope_union_id: str,
    primary_scope_ids: tuple[str, ...],
    supporting_region_strings: tuple[str, ...],
    manual_override_digests: tuple[str, ...] | None,
) -> FRVArtifactPayloadSchema:
    """Build the exact Session B payload schema with the four optional arrays.

    The required arrays follow the frozen inventory order; the optional arrays
    (``raw_hag``, ``clamped_hag``, ``hag_quality_flags``, ``hag_reason_flags``)
    are always appended in that order.  ``candidate_confidence`` is never
    published.  Reference tables are sorted and unique; the manual table is
    null when no override is effective.
    """

    if type(scope_union_id) is not str or not scope_union_id:
        raise ValueError("scope_union_id must be a non-empty str")
    if type(primary_scope_ids) is not tuple or not primary_scope_ids:
        raise ValueError("primary_scope_ids must be a non-empty tuple")
    if type(supporting_region_strings) is not tuple or not supporting_region_strings:
        raise ValueError("supporting_region_strings must be a non-empty tuple")

    arrays: list[FRVArtifactArrayDescriptor] = []
    for field_name, dtype_code, shape_rule, enum_version in _REQUIRED_ARRAY_SPECS:
        arrays.append(
            _array_descriptor(
                field_name=field_name,
                dtype_code=dtype_code,
                shape_rule=shape_rule,
                required=True,
                enum_version=enum_version,
                storage_key=field_name,
            )
        )
    for field_name, dtype_code, shape_rule, enum_version in _OPTIONAL_ARRAY_SPECS:
        arrays.append(
            _array_descriptor(
                field_name=field_name,
                dtype_code=dtype_code,
                shape_rule=shape_rule,
                required=False,
                enum_version=enum_version,
                storage_key=field_name,
            )
        )

    scope_union_table = tuple(sorted(set((scope_union_id,))))
    primary_table = tuple(sorted(set(primary_scope_ids)))
    supporting_table = tuple(sorted(set(supporting_region_strings)))
    if manual_override_digests is None:
        manual_table = None
        manual_digest = None
    else:
        manual_table = tuple(sorted(set(manual_override_digests)))
        manual_digest = frv_digest(manual_table)

    payload = {
        "payload_schema_version": FRV_ARTIFACT_PAYLOAD_SCHEMA_VERSION,
        "block_schema_version": _SCHEMA_LITERALS["block_schema_version"],
        "membership_state_schema_version": _SCHEMA_LITERALS["membership_state_schema_version"],
        "reason_flag_schema_version": _SCHEMA_LITERALS["reason_flag_schema_version"],
        "quality_flag_schema_version": _SCHEMA_LITERALS["quality_flag_schema_version"],
        "hag_evidence_schema_version": _SCHEMA_LITERALS["hag_evidence_schema_version"],
        "segmentation_hag_readiness_schema_version": _SCHEMA_LITERALS[
            "segmentation_hag_readiness_schema_version"
        ],
        "payload_encoding_id": _SCHEMA_LITERALS["payload_encoding_id"],
        "payload_encoder_version": _SCHEMA_LITERALS["payload_encoder_version"],
        "payload_compression_algorithm": _SCHEMA_LITERALS["payload_compression_algorithm"],
        "payload_compression_version": _SCHEMA_LITERALS["payload_compression_version"],
        "arrays": [array.to_dict() for array in arrays],
        "scope_union_reference_table": list(scope_union_table),
        "scope_union_reference_table_digest": frv_digest(scope_union_table),
        "primary_scope_reference_table": list(primary_table),
        "primary_scope_reference_table_digest": frv_digest(primary_table),
        "supporting_region_reference_table": list(supporting_table),
        "supporting_region_reference_table_digest": frv_digest(supporting_table),
        "manual_override_reference_table": (
            list(manual_table) if manual_table is not None else None
        ),
        "manual_override_reference_table_digest": manual_digest,
    }
    payload["payload_schema_digest"] = frv_digest(payload)
    return FRVArtifactPayloadSchema.from_dict(payload)


# ---------------------------------------------------------------------------
# FRVBLK1 / FRVEVDBLK1 block encoding and decoding
# ---------------------------------------------------------------------------


def _encode_block_header(
    *,
    artifact_id: str,
    block_id: str,
    payload_schema_digest: str,
    record_count: int,
    original_index_start: int,
    original_index_stop: int,
) -> bytes:
    header = {
        "artifact_id": artifact_id,
        "block_id": block_id,
        "payload_schema_digest": payload_schema_digest,
        "record_count": record_count,
        "original_index_start": original_index_start,
        "original_index_stop": original_index_stop,
    }
    body = canonical_frv_json_bytes(header)
    return struct.pack("<Q", len(body)) + body


def encode_frv_block(
    *,
    magic: bytes,
    artifact_id: str,
    block_id: str,
    payload_schema_digest: str,
    arrays: Mapping[str, np.ndarray],
    schema_order: tuple[str, ...],
) -> bytes:
    """Encode one FRVBLK1/FRVEVDBLK1 block over *arrays* in schema order."""

    if magic not in _KNOWN_BLOCK_MAGICS:
        raise ValueError("magic must be FRVBLK1, FRVEVDBLK1, V4NRMBLK1 or V4REVBLK1")
    if type(record_count := _record_count(arrays)) is not int:
        raise ValueError("record_count must be a built-in int")
    indexes = arrays.get("original_index")
    if indexes is None:
        raise ValueError("arrays must contain original_index")
    first = int(indexes[0])
    last = int(indexes[-1])
    header = _encode_block_header(
        artifact_id=artifact_id,
        block_id=block_id,
        payload_schema_digest=payload_schema_digest,
        record_count=record_count,
        original_index_start=first,
        original_index_stop=last + 1,
    )
    parts = [magic, header]
    for field_name in schema_order:
        array = arrays[field_name]
        name_bytes = field_name.encode("utf-8")
        if len(name_bytes) > 0xFFFF:
            raise ValueError("field name is too long for the block framing")
        raw = array.tobytes(order="C")
        parts.append(struct.pack("<H", len(name_bytes)))
        parts.append(name_bytes)
        parts.append(struct.pack("<Q", len(raw)))
        parts.append(raw)
    return b"".join(parts)


def _record_count(arrays: Mapping[str, np.ndarray]) -> int:
    indexes = arrays.get("original_index")
    if indexes is None:
        raise ValueError("arrays must contain original_index")
    count = int(indexes.shape[0])
    for field_name, array in arrays.items():
        if int(array.shape[0]) != count:
            raise ValueError(f"array {field_name} length does not match original_index")
    return count


def _read_u64(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 8 > len(data):
        raise ValueError("truncated block header length")
    return struct.unpack_from("<Q", data, offset)[0], offset + 8


def _read_u16(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 2 > len(data):
        raise ValueError("truncated block field-name length")
    return struct.unpack_from("<H", data, offset)[0], offset + 2


def decode_frv_block(
    data: bytes,
    *,
    magic: bytes,
    schema_order: tuple[str, ...],
    dtypes: Mapping[str, np.dtype],
    row_shapes: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Decode one FRVBLK1/FRVEVDBLK1 block and return (header, arrays).

    ``row_shapes`` maps each field to the shape of a single row (``()`` for a
    scalar field, ``(3,)`` for a per-point vector).  The full array shape is
    ``(record_count,) + row_shape``.
    """

    if magic not in _KNOWN_BLOCK_MAGICS:
        raise ValueError("magic must be FRVBLK1, FRVEVDBLK1, V4NRMBLK1 or V4REVBLK1")
    if not data.startswith(magic):
        raise ValueError("block magic mismatch")
    offset = len(magic)
    header_length, offset = _read_u64(data, offset)
    if offset + header_length > len(data):
        raise ValueError("truncated block header")
    header_bytes = data[offset : offset + header_length]
    offset += header_length
    header = json.loads(header_bytes.decode("utf-8"))
    expected_keys = set(_BLOCK_HEADER_KEYS)
    if set(header) != expected_keys:
        raise ValueError("block header field mismatch")
    record_count = int(header["record_count"])

    arrays: dict[str, np.ndarray] = {}
    while offset < len(data):
        name_length, offset = _read_u16(data, offset)
        if offset + name_length > len(data):
            raise ValueError("truncated block field name")
        field_name = data[offset : offset + name_length].decode("utf-8")
        offset += name_length
        raw_length, offset = _read_u64(data, offset)
        if offset + raw_length > len(data):
            raise ValueError(f"truncated block field {field_name}")
        raw = data[offset : offset + raw_length]
        offset += raw_length
        if field_name not in schema_order:
            raise ValueError(f"unknown block field {field_name!r}")
        dtype = dtypes[field_name]
        row_shape = row_shapes[field_name]
        shape = (record_count,) + row_shape
        array = np.frombuffer(raw, dtype=dtype)
        expected_count = int(np.prod(shape)) if shape else 1
        if int(array.size) != expected_count:
            raise ValueError(f"block field {field_name} has an unexpected element count")
        arrays[field_name] = array.reshape(shape).copy()
        arrays[field_name].flags.writeable = False
    if offset != len(data):
        raise ValueError("trailing bytes in block")
    if set(arrays) != set(schema_order):
        raise ValueError("block is missing payload fields")
    if set(header) != expected_keys:
        raise ValueError("block header field mismatch")
    return header, arrays


# ---------------------------------------------------------------------------
# FRVART1 artifact digest
# ---------------------------------------------------------------------------


def build_art1_declared_header(
    artifact: FullResolutionVegetationMembershipArtifact,
) -> dict[str, Any]:
    """Return the exact FRVART1 declared header for *artifact*."""

    return {
        "magic": FRV_ART1_MAGIC,
        "artifact_schema_version": artifact.artifact_schema_version,
        "algorithm_version": artifact.algorithm_version,
        "decision_rule_version": artifact.decision_rule_version,
        "artifact_id": artifact.artifact_id,
        "source_content_identity_digest": artifact.source_content_identity_digest,
        "scope_union_id": artifact.scope_union_id,
        "configuration_digest": artifact.configuration_digest,
        "payload_schema_digest": artifact.payload_schema_digest,
        "block_schema_version": artifact.block_schema_version,
        "block_count": len(artifact.block_descriptors),
        "ordered_block_ids": [block.block_id for block in artifact.block_descriptors],
    }


def compute_art1_artifact_sha256(artifact: FullResolutionVegetationMembershipArtifact) -> str:
    """Return the FRVART1 artifact_sha256 for the finalized artifact metadata."""

    declared_header = build_art1_declared_header(artifact)
    header_bytes = canonical_frv_json_bytes(declared_header)
    hasher = hashlib.sha256()
    hasher.update(FRV_ART1_BYTE_MAGIC)
    hasher.update(struct.pack("<Q", len(header_bytes)))
    hasher.update(header_bytes)
    for block in artifact.block_descriptors:
        block_id_bytes = block.block_id.encode("utf-8")
        if len(block_id_bytes) > 0xFFFF:
            raise ValueError("block_id is too long for the FRVART1 framing")
        hasher.update(struct.pack("<H", len(block_id_bytes)))
        hasher.update(block_id_bytes)
        hasher.update(bytes.fromhex(block.block_digest))
        hasher.update(bytes.fromhex(block.block_descriptor_digest))
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Atomic directory publication
# ---------------------------------------------------------------------------


def publish_directory(
    *,
    scratch_dir: Path,
    target_dir: Path,
    publish: Any,
) -> None:
    """Write files into a sibling temp directory, re-read and validate all
    bytes, then publish with a single directory rename.

    *publish* is a callable ``(staging_dir) -> None`` that writes every file.
    The target directory must not already exist; overwrite is not supported.
    On any failure the temp directory is removed and the target stays absent.
    """

    if not isinstance(scratch_dir, Path):
        raise TypeError("scratch_dir must be a pathlib.Path")
    if not isinstance(target_dir, Path):
        raise TypeError("target_dir must be a pathlib.Path")
    if target_dir.exists():
        raise ValueError("output_dir already exists; overwrite is not supported")
    staging = scratch_dir / f".frv_publish_{os.getpid()}"
    try:
        staging.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        raise ValueError("a stale publish staging directory exists") from None
    try:
        publish(staging)
        _validate_published_files(staging)
        # Single atomic rename.
        os.replace(staging, target_dir)
    except Exception:
        _remove_tree(staging)
        raise
    if staging.exists():
        raise ValueError("publication did not atomically move the staging directory")


def _validate_published_files(staging: Path) -> None:
    """Ensure every written file is re-readable and re-hashable."""

    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        path.read_bytes()
        if path.stat().st_size == 0:
            raise ValueError(f"published file {path.name} is empty")


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.iterdir(), key=lambda item: item.name, reverse=True):
        if child.is_dir():
            _remove_tree(child)
        else:
            try:
                child.unlink()
            except OSError:
                pass
    try:
        path.rmdir()
    except OSError:
        pass


def block_storage_path(block_id: str) -> str:
    """Return the descriptor storage reference for a membership block."""

    return f"blocks/{block_id}.frvblk"


def evidence_block_storage_path(block_id: str) -> str:
    """Return the descriptor storage reference for a decision-evidence block."""

    return f"evidence_blocks/{block_id}.frvblk"


def block_id_for_ordinal(ordinal: int, *, evidence: bool = False) -> str:
    if evidence:
        return f"frv-evidence-block-{ordinal:08d}"
    return f"frv-block-{ordinal:08d}"


def descriptor_digest(block: FRVArtifactBlockDescriptor) -> str:
    """Return the canonical block-descriptor digest (self-computed)."""

    return block.computed_digest()


__all__ = [
    "FRV_BLOCK_MAGIC",
    "FRV_EVIDENCE_BLOCK_MAGIC",
    "V4_NORMAL_BLOCK_MAGIC",
    "V4_REVIEW_BLOCK_MAGIC",
    "FRV_ART1_MAGIC",
    "FRV_ART1_BYTE_MAGIC",
    "canonical_json_bytes",
    "write_canonical_json",
    "sha256_file",
    "build_frv_payload_schema",
    "encode_frv_block",
    "decode_frv_block",
    "build_art1_declared_header",
    "compute_art1_artifact_sha256",
    "publish_directory",
    "block_storage_path",
    "evidence_block_storage_path",
    "block_id_for_ordinal",
    "descriptor_digest",
]
