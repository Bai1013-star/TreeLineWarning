"""FRV membership artifact publication helpers tests (PLAN-FROZEN)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from algorithms.vegetation.frv_artifacts import (
    FRV_ART1_MAGIC,
    FRV_BLOCK_MAGIC,
    FRV_EVIDENCE_BLOCK_MAGIC,
    build_art1_declared_header,
    build_frv_payload_schema,
    compute_art1_artifact_sha256,
    decode_frv_block,
    encode_frv_block,
    publish_directory,
    sha256_file,
    write_canonical_json,
)
from core.contracts.frv import (
    FRV_ARTIFACT_BLOCK_SCHEMA_VERSION,
    FRVArtifactBlockDescriptor,
    frv_digest,
    validate_artifact_payload_digest_domain,
    validate_payload_schema,
)


def _canon(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _ro(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _payload_arrays(count: int = 3) -> dict[str, np.ndarray]:
    return {
        "original_index": _ro(np.asarray(list(range(count)), dtype="<i8")),
        "membership_state": _ro(np.zeros(count, dtype="|u1")),
        "reason_flags": _ro(np.zeros(count, dtype="<u4")),
        "input_quality_flags": _ro(np.zeros(count, dtype="<u4")),
        "hag_evidence_state": _ro(np.zeros(count, dtype="|u1")),
        "segmentation_hag_readiness": _ro(np.zeros(count, dtype="|u1")),
        "scope_union_id_reference": _ro(np.zeros(count, dtype="<u4")),
        "primary_scope_id_reference": _ro(np.zeros(count, dtype="<u4")),
        "supporting_region_offset": _ro(np.zeros(count, dtype="<u8")),
        "supporting_region_count": _ro(np.ones(count, dtype="<u4")),
        "manual_override_reference": _ro(
            np.full(count, 0xFFFFFFFF, dtype="<u4")
        ),
        "raw_hag": _ro(np.zeros(count, dtype="<f4")),
        "clamped_hag": _ro(np.zeros(count, dtype="<f4")),
        "hag_quality_flags": _ro(np.zeros(count, dtype="<u4")),
        "hag_reason_flags": _ro(np.zeros(count, dtype="<u4")),
    }


def _schema() -> None:
    return build_frv_payload_schema(
        scope_union_id="frv-scope-union-abc",
        primary_scope_ids=("scope-b", "scope-a"),
        supporting_region_strings=(
            '["scope-b", "scope-a"]',
            '["scope-a"]',
        ),
        manual_override_digests=None,
    )


def test_payload_schema_is_valid() -> None:
    schema = _schema()
    validate_payload_schema(schema)
    names = [array.field_name for array in schema.arrays]
    assert names[0] == "original_index"
    assert "raw_hag" in names
    assert "clamped_hag" in names
    assert "hag_quality_flags" in names
    assert "hag_reason_flags" in names
    assert "candidate_confidence" not in names
    assert schema.scope_union_reference_table == ("frv-scope-union-abc",)
    assert schema.primary_scope_reference_table == ("scope-a", "scope-b")
    assert schema.manual_override_reference_table is None


def test_payload_schema_manual_table_sorted_unique() -> None:
    schema = build_frv_payload_schema(
        scope_union_id="u1",
        primary_scope_ids=("s1",),
        supporting_region_strings=('["s1"]',),
        manual_override_digests=(
            "b" * 64,
            "a" * 64,
            "a" * 64,
        ),
    )
    validate_payload_schema(schema)
    assert schema.manual_override_reference_table == ("a" * 64, "b" * 64)
    assert schema.manual_override_reference_table_digest == hashlib.sha256(
        _canon(["a" * 64, "b" * 64])
    ).hexdigest()


def test_block_round_trip(tmp_path: Path) -> None:
    schema = _schema()
    schema_order = tuple(array.field_name for array in schema.arrays)
    dtypes = {array.field_name: np.dtype(array.dtype_code) for array in schema.arrays}
    payload = _payload_arrays(4)
    payload["original_index"] = _ro(np.asarray([10, 11, 12, 13], dtype="<i8"))
    data = encode_frv_block(
        magic=FRV_BLOCK_MAGIC,
        artifact_id="frv-membership-" + "a" * 64,
        block_id="frv-block-00000000",
        payload_schema_digest=schema.payload_schema_digest,
        arrays=payload,
        schema_order=schema_order,
    )
    assert data.startswith(b"FRVBLK1\0")
    header, arrays = decode_frv_block(
        data,
        magic=FRV_BLOCK_MAGIC,
        schema_order=schema_order,
        dtypes=dtypes,
        row_shapes={name: () for name in schema_order},
    )
    assert header["block_id"] == "frv-block-00000000"
    assert header["record_count"] == 4
    assert header["original_index_start"] == 10
    assert header["original_index_stop"] == 14
    np.testing.assert_array_equal(arrays["original_index"], payload["original_index"])
    np.testing.assert_array_equal(arrays["clamped_hag"], payload["clamped_hag"])
    assert not arrays["original_index"].flags.writeable


def test_block_fixed_vector() -> None:
    # Minimal evidence block framing: magic + header + field framing.
    payload = {
        "original_index": _ro(np.asarray([7], dtype="<i8")),
        "geometry_evidence_state": _ro(np.asarray([1], dtype="|u1")),
    }
    data = encode_frv_block(
        magic=FRV_EVIDENCE_BLOCK_MAGIC,
        artifact_id="frv-membership-" + "a" * 64,
        block_id="frv-evidence-block-00000000",
        payload_schema_digest="b" * 64,
        arrays=payload,
        schema_order=("original_index", "geometry_evidence_state"),
    )
    assert data[: len(b"FRVEVDBLK1\0")] == b"FRVEVDBLK1\0"
    # header length uint64
    header_len = int.from_bytes(data[11:19], "little")
    offset = 11 + 8 + header_len
    # field-name u16
    name_len = int.from_bytes(data[offset : offset + 2], "little")
    assert data[offset + 2 : offset + 2 + name_len] == b"original_index"
    offset += 2 + name_len
    raw_len = int.from_bytes(data[offset : offset + 8], "little")
    assert raw_len == 8
    assert data[offset + 8 : offset + 8 + 8] == (7).to_bytes(8, "little")


def test_art1_sha256_matches_oracle() -> None:
    from core.contracts.frv import (
        FRV_ARTIFACT_BLOCK_SCHEMA_VERSION,
        FRV_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION,
        FullResolutionVegetationMembershipArtifact,
        frv_digest,
    )

    block = _block_descriptor("frv-block-00000000", "1" * 64)
    artifact = FullResolutionVegetationMembershipArtifact.__new__(
        FullResolutionVegetationMembershipArtifact
    )
    fields = {
        "artifact_schema_version": FRV_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION,
        "algorithm_version": "frv-membership-synchronous-v1",
        "decision_rule_version": "frv-membership-decision-v1",
        "artifact_id": "",
        "source_content_identity_digest": "2" * 64,
        "source_identity_record_digest": "3" * 64,
        "source_file_sha256": "4" * 64,
        "canonical_point_content_digest": None,
        "canonical_point_content_spec_digest": None,
        "scope_coverage_manifest_digest": "5" * 64,
        "scope_union_id": "frv-scope-union-oracle",
        "scope_construction_version": "frv-scope-construction-v1",
        "scope_union_geometry_digest": "6" * 64,
        "canonical_inventory_digest": "7" * 64,
        "required_inventory_digest": "8" * 64,
        "configuration_digest": "9" * 64,
        "payload_schema_digest": "a" * 64,
        "block_schema_version": FRV_ARTIFACT_BLOCK_SCHEMA_VERSION,
        "artifact_sha256": "",
        "block_descriptors": (block,),
        "source_point_count": 3,
        "unique_in_scope_point_count": 3,
        "evaluated_point_count": 3,
        "verified_vegetation_candidate_count": 3,
        "excluded_non_candidate_count": 0,
        "review_required_membership_count": 0,
        "manual_include_count": 0,
        "manual_exclude_count": 0,
        "unasserted_in_scope_count": 0,
        "canonical_inventory_count": 1,
        "required_inventory_count": 1,
        "approved_hard_rejection_count": 0,
        "scope_created_or_merged_count": 1,
        "coverage_blocked_count": 0,
        "canonical_inventory_complete": True,
        "required_inventory_complete": True,
        "approved_hard_rejection_complete": True,
        "scope_coverage_complete": True,
        "routing_complete": True,
        "membership_decision_complete": True,
        "provider_digest_references": ("c" * 64,),
        "override_digest_references": (),
        "block_digest_references": ("1" * 64,),
        "warning_digest": frv_digest(()),
        "limitation_digest": frv_digest(()),
        "provenance_digest": "d" * 64,
        "manifest_digest": "",
    }
    for name, value in fields.items():
        object.__setattr__(artifact, name, value)
    object.__setattr__(artifact, "artifact_id", artifact.computed_artifact_id())
    computed_sha = compute_art1_artifact_sha256(artifact)

    declared = build_art1_declared_header(artifact)
    header_bytes = _canon(declared)
    hasher = hashlib.sha256()
    hasher.update(b"FRVART1\0")
    hasher.update(len(header_bytes).to_bytes(8, "little"))
    hasher.update(header_bytes)
    for block_item in artifact.block_descriptors:
        block_id = block_item.block_id.encode("utf-8")
        hasher.update(len(block_id).to_bytes(2, "little"))
        hasher.update(block_id)
        hasher.update(bytes.fromhex(block_item.block_digest))
        hasher.update(bytes.fromhex(block_item.block_descriptor_digest))
    assert computed_sha == hasher.hexdigest()
    assert declared["magic"] == FRV_ART1_MAGIC
    validate_artifact_payload_digest_domain(
        artifact, declared_header=declared
    )


def _block_descriptor(block_id: str, block_digest: str) -> FRVArtifactBlockDescriptor:
    descriptor = FRVArtifactBlockDescriptor.__new__(FRVArtifactBlockDescriptor)
    object.__setattr__(
        descriptor, "block_schema_version", FRV_ARTIFACT_BLOCK_SCHEMA_VERSION
    )
    object.__setattr__(descriptor, "payload_schema_digest", "a" * 64)
    object.__setattr__(descriptor, "block_id", block_id)
    object.__setattr__(descriptor, "block_digest", block_digest)
    object.__setattr__(descriptor, "original_index_start", 0)
    object.__setattr__(descriptor, "original_index_stop", 3)
    object.__setattr__(descriptor, "record_count", 3)
    object.__setattr__(descriptor, "first_original_index", 0)
    object.__setattr__(descriptor, "last_original_index", 2)
    object.__setattr__(
        descriptor, "storage_reference", f"blocks/{block_id}.frvblk"
    )
    object.__setattr__(
        descriptor,
        "payload_schema_version",
        "frv-artifact-payload-v1",
    )
    object.__setattr__(descriptor, "array_descriptor_digest", frv_digest([]))
    object.__setattr__(descriptor, "block_descriptor_digest", "")
    object.__setattr__(descriptor, "block_descriptor_digest", descriptor.computed_digest())
    return descriptor


def test_publish_directory_atomic(tmp_path: Path) -> None:
    target = tmp_path / "out"
    writes: list[Path] = []

    def publish(staging: Path) -> None:
        write_canonical_json(staging / "a.json", {"x": 1})
        writes.append(staging)

    publish_directory(scratch_dir=tmp_path, target_dir=target, publish=publish)
    assert target.is_dir()
    assert (target / "a.json").is_file()
    assert json.loads((target / "a.json").read_text("utf-8")) == {"x": 1}
    assert not list(tmp_path.glob(".frv_publish_*"))
    assert writes[0].exists() is False  # staging was moved, not copied


def test_publish_directory_rejects_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "out"
    target.mkdir()

    def publish(staging: Path) -> None:
        write_canonical_json(staging / "a.json", {"x": 1})

    with pytest.raises(ValueError):
        publish_directory(scratch_dir=tmp_path, target_dir=target, publish=publish)
    assert (target / "a.json").exists() is False


def test_publish_directory_cleanup_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "out"

    def publish(staging: Path) -> None:
        write_canonical_json(staging / "a.json", {"x": 1})
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        publish_directory(scratch_dir=tmp_path, target_dir=target, publish=publish)
    assert target.exists() is False
    assert not list(tmp_path.glob(".frv_publish_*"))


def test_write_canonical_json_trailing_lf(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    write_canonical_json(path, {"b": 2, "a": [1, 2]})
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert raw.decode("utf-8").endswith('{"a":[1,2],"b":2}\n')
    assert sha256_file(path) == hashlib.sha256(raw).hexdigest()
