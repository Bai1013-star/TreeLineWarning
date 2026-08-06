"""Focused FRV-A2 P-WIRE contract tests (PLAN-FROZEN)."""

from __future__ import annotations

import hashlib
import json
import struct

import core.contracts.frv_wire as wire
import numpy as np
import pytest
from core.contracts.frv import FrvErrorCode
from core.contracts.frv_a2 import (
    FrvA2ArtifactComponentBinding,
    FrvA2ArtifactComponentKind,
    FrvA2DomainKind,
    FrvA2FailurePhase,
    FrvA2FailureRecord,
    FrvA2ProviderRole,
)
from core.contracts.frv_wire import (
    FRV_WIRE_AUTHORITY_DIGEST_DOMAIN,
    FRV_WIRE_AUTHORITY_SCHEMA_VERSION,
    FRV_WIRE_CONFIG_SCHEMA_VERSION,
    FRV_WIRE_DEFAULT_BODY_POSSIBLE_RADIUS_M,
    FRV_WIRE_DEFAULT_CONFLICT_RADIUS_M,
    FRV_WIRE_DEFAULT_MAX_EDGES_PER_INDEX_CELL,
    FRV_WIRE_DEFAULT_PROXIMITY_RADIUS_M,
    FRV_WIRE_DEFAULT_SPATIAL_INDEX_CELL_SIZE_M,
    FRV_WIRE_GEOMETRY_DIGEST_DOMAIN,
    FRV_WIRE_PROVIDER_KEY,
    FrvWireAuthorityManifest,
    FrvWireAuthorityMode,
    FrvWireChunk,
    FrvWireLoadError,
    FrvWireProviderConfig,
    FrvWireProviderProof,
    FrvWireProviderResult,
    derive_frv_wire_authority_approval_digest,
    derive_frv_wire_authority_manifest_digest,
    derive_frv_wire_provider_config_digest,
    frv_wire_geometry_content_digest,
    frv_wire_point_content_digest,
    validate_frv_wire_authority_manifest,
    validate_frv_wire_chunk,
    validate_frv_wire_provider_config,
    validate_frv_wire_provider_proof,
)

SHA = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


# ---------------------------------------------------------------------------
# Independent canonical-JSON oracle (never calls production digest functions)
# ---------------------------------------------------------------------------


def _canon(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canon_sha(value) -> str:
    return hashlib.sha256(_canon(value)).hexdigest()


def _component(key: str, kind: FrvA2ArtifactComponentKind) -> FrvA2ArtifactComponentBinding:
    return FrvA2ArtifactComponentBinding(
        component_key=key, component_kind=kind, sha256=SHA, size_bytes=64
    )


_W1_PAIR = (
    _component("manifest_json", FrvA2ArtifactComponentKind.MANIFEST_JSON),
    _component("payload_npz", FrvA2ArtifactComponentKind.PAYLOAD_NPZ),
)
_PROOF_COMPONENTS = (
    _component("authority_manifest_json", FrvA2ArtifactComponentKind.MANIFEST_JSON),
) + _W1_PAIR


def _index_domain_expected(source_content: str, count: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"FRVA2IDX1\0")
    hasher.update(b"ORIGINAL_INDEX\0")
    hasher.update(source_content.encode("utf-8") + b"\0")
    hasher.update(b"full_source\0")
    hasher.update(struct.pack("<Q", count))
    for index in range(count):
        hasher.update(struct.pack("<Q", index))
    return hasher.hexdigest()


def _ref_id(components: tuple[FrvA2ArtifactComponentBinding, ...]) -> str:
    ordered = sorted(
        components,
        key=lambda c: (
            {
                member: index
                for index, member in enumerate(FrvA2ArtifactComponentKind)
            }[c.component_kind],
            c.component_key,
        ),
    )
    sequence = ";".join(
        ",".join(
            (c.component_kind.value, c.component_key, c.sha256, str(c.size_bytes))
        )
        for c in ordered
    )
    return f"frv-a2-ref-v1|wire|wire_provider|{sequence}"


def _manifest_fields(**overrides) -> dict:
    defaults: dict = {
        "schema_version": FRV_WIRE_AUTHORITY_SCHEMA_VERSION,
        "authority_mode": FrvWireAuthorityMode.W1_MODEL_SET,
        "source_content_identity_digest": SHA,
        "source_identity_record_digest": _SHA_B,
        "source_id": "scene_simple",
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "origin_xyz": (0.0, 0.0, 0.0),
        "transform_declaration": "identity",
        "inventory_complete": True,
        "approved_wire_ids": (0, 1),
        "w1_artifact_components": _W1_PAIR,
        "geometry_content_digest": _SHA_C,
        "authority_rule_id": "wire-authority-rule",
        "authority_rule_version": "rule-v1",
        "approver": "approver",
        "approval_timestamp": "2026-01-01T00:00:00Z",
        "approval_digest": None,
        "manifest_digest": None,
    }
    defaults.update(overrides)
    return defaults


def _manifest(**overrides) -> FrvWireAuthorityManifest:
    fields = _manifest_fields(**overrides)
    explicit_approval = fields.pop("approval_digest", None)
    explicit_manifest = fields.pop("manifest_digest", None)
    approval_payload = _manifest_payload_json(fields)
    approval = (
        explicit_approval
        if explicit_approval is not None
        else hashlib.sha256(_canon(approval_payload)).hexdigest()
    )
    manifest_payload = {"digest_domain": FRV_WIRE_AUTHORITY_DIGEST_DOMAIN, **approval_payload}
    manifest_payload["approval_digest"] = approval
    manifest_digest = (
        explicit_manifest
        if explicit_manifest is not None
        else hashlib.sha256(_canon(manifest_payload)).hexdigest()
    )
    return FrvWireAuthorityManifest(
        **{
            **fields,
            "approved_wire_ids": tuple(fields["approved_wire_ids"]),
            "w1_artifact_components": tuple(fields["w1_artifact_components"]),
            "origin_xyz": tuple(fields["origin_xyz"]),
            "approval_digest": approval,
            "manifest_digest": manifest_digest,
        }
    )


def _manifest_payload_json(fields: dict) -> dict:
    result = dict(fields)
    result["authority_mode"] = result["authority_mode"].value
    result["approved_wire_ids"] = list(result["approved_wire_ids"])
    result["w1_artifact_components"] = [
        component.to_dict() for component in result["w1_artifact_components"]
    ]
    result["origin_xyz"] = list(result["origin_xyz"])
    return result


# ---------------------------------------------------------------------------
# Independent byte-framing oracles
# ---------------------------------------------------------------------------

_PCD_FIELDS = (
    ("original_index", "<i8", 1),
    ("provider_state", "|u1", 1),
    ("relation_state", "|u1", 1),
    ("nearest_wire_id", "<i4", 1),
    ("nearest_segment_index", "<i4", 1),
    ("nearest_edge_index", "<i4", 1),
    ("centerline_distance_m", "<f8", 1),
    ("reason_flags", "<u4", 1),
    ("quality_flags", "<u4", 1),
)


def _cfg_expected(**fields) -> str:
    payload = {"digest_domain": "FRVWIRCFG1", **fields}
    return _canon_sha(payload)


def _geo_expected(source_content, source_record, origin, edges) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"FRVWIRGEO1\0")
    hasher.update(source_content.encode("utf-8") + b"\0")
    hasher.update(source_record.encode("utf-8") + b"\0")
    hasher.update(struct.pack("<ddd", *origin))
    hasher.update(struct.pack("<Q", len(edges)))
    for edge in sorted(edges):
        hasher.update(struct.pack("<iii6d", *edge))
    return hasher.hexdigest()


def _pcd_expected(headers, rows) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"FRVWIRPCD1\0")
    for header in headers:
        hasher.update(header.encode("utf-8") + b"\0")
    hasher.update(struct.pack("<Q", len(rows)))
    for name, dtype_str, count in _PCD_FIELDS:
        name_bytes = name.encode("utf-8")
        dtype = np.dtype(dtype_str)
        hasher.update(struct.pack("<H", len(name_bytes)))
        hasher.update(name_bytes)
        hasher.update(struct.pack("<I", dtype.itemsize * count))
        for row in rows:
            array = row[name]
            if dtype_str == "<f8":
                hasher.update((array + np.zeros(1, dtype=dtype)).tobytes())
            else:
                hasher.update(array.tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _config(**overrides) -> FrvWireProviderConfig:
    defaults: dict = {
        "schema_version": FRV_WIRE_CONFIG_SCHEMA_VERSION,
        "provider_key": FRV_WIRE_PROVIDER_KEY,
        "provider_role": FrvA2ProviderRole.WIRE,
        "interface_version": wire.FRV_WIRE_PROVIDER_INTERFACE_VERSION,
        "algorithm_version": wire.FRV_WIRE_ALGORITHM_VERSION,
        "wire_body_possible_radius_m": FRV_WIRE_DEFAULT_BODY_POSSIBLE_RADIUS_M,
        "vegetation_conflict_radius_m": FRV_WIRE_DEFAULT_CONFLICT_RADIUS_M,
        "proximity_radius_m": FRV_WIRE_DEFAULT_PROXIMITY_RADIUS_M,
        "spatial_index_cell_size_m": FRV_WIRE_DEFAULT_SPATIAL_INDEX_CELL_SIZE_M,
        "max_edges_per_index_cell": FRV_WIRE_DEFAULT_MAX_EDGES_PER_INDEX_CELL,
    }
    explicit = overrides.pop("provider_config_digest", None)
    defaults.update(overrides)
    if explicit is not None:
        defaults["provider_config_digest"] = explicit
    else:
        defaults["provider_config_digest"] = derive_frv_wire_provider_config_digest(
            schema_version=defaults["schema_version"],
            provider_key=defaults["provider_key"],
            provider_role=defaults["provider_role"],
            interface_version=defaults["interface_version"],
            algorithm_version=defaults["algorithm_version"],
            wire_body_possible_radius_m=defaults["wire_body_possible_radius_m"],
            vegetation_conflict_radius_m=defaults["vegetation_conflict_radius_m"],
            proximity_radius_m=defaults["proximity_radius_m"],
            spatial_index_cell_size_m=defaults["spatial_index_cell_size_m"],
            max_edges_per_index_cell=defaults["max_edges_per_index_cell"],
        )
    return FrvWireProviderConfig(**defaults)


def _ro(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _chunk(**overrides) -> FrvWireChunk:
    rows = overrides.pop("rows", None)
    if rows is None:
        rows = [
            # index, provider_state, relation, wire, seg, edge, distance
            (0, 1, 0, -1, -1, -1, 0.0),
            (1, 1, 2, 0, 0, 0, 0.05),
            (2, 1, 4, 0, 0, 1, 0.25),
            (3, 1, 3, 1, 1, 2, 1.0),
        ]
    point_count = len(rows)
    defaults: dict = {
        "source_id": "scene_simple",
        "chunk_id": 0,
        "start_index": 0,
        "point_count": point_count,
        "original_index": _ro(np.asarray([row[0] for row in rows], dtype="<i8")),
        "provider_state": _ro(np.asarray([row[1] for row in rows], dtype="|u1")),
        "relation_state": _ro(np.asarray([row[2] for row in rows], dtype="|u1")),
        "nearest_wire_id": _ro(np.asarray([row[3] for row in rows], dtype="<i4")),
        "nearest_segment_index": _ro(np.asarray([row[4] for row in rows], dtype="<i4")),
        "nearest_edge_index": _ro(np.asarray([row[5] for row in rows], dtype="<i4")),
        "centerline_distance_m": _ro(np.asarray([row[6] for row in rows], dtype="<f8")),
        "reason_flags": _ro(np.zeros(point_count, dtype="<u4")),
        "quality_flags": _ro(np.zeros(point_count, dtype="<u4")),
        "is_last": True,
    }
    defaults.update(overrides)
    return FrvWireChunk(**defaults)


def _proof_payload(**overrides) -> dict:
    payload: dict = {
        "digest_domain": "FRVWIRPRF1",
        "schema_version": wire.FRV_WIRE_PROOF_SCHEMA_VERSION,
        "provider_key": FRV_WIRE_PROVIDER_KEY,
        "provider_role": "wire",
        "interface_version": wire.FRV_WIRE_PROVIDER_INTERFACE_VERSION,
        "algorithm_version": wire.FRV_WIRE_ALGORITHM_VERSION,
        "configuration_digest": SHA,
        "provider_config_digest": SHA,
        "source_artifact_reference_id": "frv-a2-ref-v1|source_scan|source_scan|a",
        "artifact_reference_id": _ref_id(_PROOF_COMPONENTS),
        "artifact_components": [
            component.to_dict() for component in _PROOF_COMPONENTS
        ],
        "source_content_identity_digest": SHA,
        "source_identity_record_digest": _SHA_B,
        "source_proof_digest": SHA,
        "source_binding_digest": SHA,
        "source_point_count": 4,
        "domain_kind": "full_source",
        "domain_count": 4,
        "original_index_domain_digest": _index_domain_expected(SHA, 4),
        "authority_mode": "w1_model_set",
        "authority_manifest_digest": SHA,
        "wire_geometry_content_digest": _SHA_C,
        "wire_point_content_digest": SHA,
        "approved_wire_count": 2,
        "wire_relation_valid_count": 4,
        "wire_not_applicable_proven_count": 0,
        "outside_influence_domain_count": 1,
        "wire_body_possible_count": 1,
        "wire_proximity_only_count": 1,
        "wire_vegetation_conflict_count": 1,
        "minimum_centerline_distance_m": 0.05,
        "maximum_centerline_distance_m": 1.0,
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "origin_xyz": [0.0, 0.0, 0.0],
        "transform_declaration": "identity",
        "quality_flags": 0,
        "reason_flags": 0,
        "error_codes": [],
    }
    payload.update(overrides)
    return payload


def _proof(**overrides) -> FrvWireProviderProof:
    payload = _proof_payload(**overrides)
    proof = FrvWireProviderProof.__new__(FrvWireProviderProof)
    for name, value in payload.items():
        if name == "digest_domain":
            continue
        if name == "artifact_components":
            value = tuple(
                FrvA2ArtifactComponentBinding.from_dict(item) for item in value
            )
        elif name == "error_codes":
            value = tuple(value)
        elif name == "origin_xyz":
            value = tuple(value)
        elif name == "authority_mode":
            value = FrvWireAuthorityMode(value)
        elif name == "provider_role":
            value = FrvA2ProviderRole(value)
        elif name == "domain_kind":
            value = FrvA2DomainKind(value)
        object.__setattr__(proof, name, value)
    object.__setattr__(proof, "proof_digest", proof.computed_digest())
    return proof


# ---------------------------------------------------------------------------
# Authority manifest tests
# ---------------------------------------------------------------------------


def test_authority_manifest_w1_builds_and_hashes() -> None:
    manifest = _manifest()
    assert manifest.schema_version == FRV_WIRE_AUTHORITY_SCHEMA_VERSION
    assert manifest.authority_mode is FrvWireAuthorityMode.W1_MODEL_SET
    assert manifest.inventory_complete is True
    assert manifest.approved_wire_ids == (0, 1)
    assert len(manifest.w1_artifact_components) == 2
    validate_frv_wire_authority_manifest(manifest)
    assert manifest.manifest_digest == derive_frv_wire_authority_manifest_digest(manifest)
    assert manifest.approval_digest == derive_frv_wire_authority_approval_digest(manifest)


def test_authority_manifest_approval_digest_oracle() -> None:
    fields = _manifest_fields()
    fields.pop("approval_digest")
    fields.pop("manifest_digest")
    payload = _manifest_payload_json(fields)
    expected_approval = hashlib.sha256(_canon(payload)).hexdigest()
    manifest = _manifest()
    assert manifest.approval_digest == expected_approval


def test_authority_manifest_manifest_digest_oracle() -> None:
    fields = _manifest_fields()
    fields.pop("approval_digest")
    fields.pop("manifest_digest")
    payload = {"digest_domain": FRV_WIRE_AUTHORITY_DIGEST_DOMAIN, **_manifest_payload_json(fields)}
    payload["approval_digest"] = hashlib.sha256(
        _canon(_manifest_payload_json(fields))
    ).hexdigest()
    expected_manifest = hashlib.sha256(_canon(payload)).hexdigest()
    manifest = _manifest()
    assert manifest.manifest_digest == expected_manifest


def test_authority_manifest_approval_digest_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        _manifest(approval_digest="0" * 64)


def test_authority_manifest_manifest_digest_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        _manifest(manifest_digest="0" * 64)


def test_authority_manifest_empty_inventory() -> None:
    manifest = _manifest(
        authority_mode=FrvWireAuthorityMode.EMPTY_INVENTORY,
        approved_wire_ids=(),
        w1_artifact_components=(),
    )
    validate_frv_wire_authority_manifest(manifest)
    assert manifest.approved_wire_ids == ()


def test_authority_manifest_empty_forbids_components() -> None:
    with pytest.raises(ValueError):
        _manifest(
            authority_mode=FrvWireAuthorityMode.EMPTY_INVENTORY,
            approved_wire_ids=(),
            w1_artifact_components=_W1_PAIR,
        )


def test_authority_manifest_w1_requires_pair() -> None:
    with pytest.raises(ValueError):
        _manifest(w1_artifact_components=(_W1_PAIR[0],))


def test_authority_manifest_inventory_must_be_complete() -> None:
    with pytest.raises(ValueError):
        _manifest(inventory_complete=False)


def test_authority_manifest_requires_increasing_ids() -> None:
    with pytest.raises(ValueError):
        _manifest(approved_wire_ids=(1, 0))


def test_authority_manifest_w1_requires_nonempty_ids() -> None:
    with pytest.raises(ValueError):
        _manifest(approved_wire_ids=())


def test_authority_manifest_rejects_bad_frame() -> None:
    with pytest.raises(ValueError):
        _manifest(coordinate_frame="WGS84")


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_config_builds_with_expected_digest() -> None:
    config = _config()
    assert config.provider_key == FRV_WIRE_PROVIDER_KEY
    assert config.provider_role is FrvA2ProviderRole.WIRE
    assert config.wire_body_possible_radius_m == 0.10
    assert config.vegetation_conflict_radius_m == 0.50
    assert config.proximity_radius_m == 1.50
    assert config.spatial_index_cell_size_m == 1.50
    assert config.max_edges_per_index_cell == 100000
    validate_frv_wire_provider_config(config)


def test_config_fixed_vector_oracle() -> None:
    fields = {
        "schema_version": FRV_WIRE_CONFIG_SCHEMA_VERSION,
        "provider_key": FRV_WIRE_PROVIDER_KEY,
        "provider_role": "wire",
        "interface_version": wire.FRV_WIRE_PROVIDER_INTERFACE_VERSION,
        "algorithm_version": wire.FRV_WIRE_ALGORITHM_VERSION,
        "wire_body_possible_radius_m": 0.10,
        "vegetation_conflict_radius_m": 0.50,
        "proximity_radius_m": 1.50,
        "spatial_index_cell_size_m": 1.50,
        "max_edges_per_index_cell": 100000,
    }
    expected = _cfg_expected(**fields)
    config = _config()
    assert config.provider_config_digest == expected


def test_config_rejects_bool_float() -> None:
    with pytest.raises(ValueError):
        _config(wire_body_possible_radius_m=True)


def test_config_rejects_numpy_float_scalar() -> None:
    with pytest.raises(ValueError):
        _config(wire_body_possible_radius_m=np.float64(0.10))


def test_config_rejects_non_increasing_radii() -> None:
    with pytest.raises(ValueError):
        _config(vegetation_conflict_radius_m=0.10)


def test_config_rejects_nonpositive_cell() -> None:
    with pytest.raises(ValueError):
        _config(spatial_index_cell_size_m=0.0)


def test_config_digest_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        _config(provider_config_digest="0" * 64)


def test_config_from_dict_rejects_unknown_field() -> None:
    data = _config().to_dict()
    data["extra"] = 1
    with pytest.raises(ValueError):
        FrvWireProviderConfig.from_dict(data)


def test_config_from_dict_rejects_missing_field() -> None:
    data = _config().to_dict()
    data.pop("max_edges_per_index_cell")
    with pytest.raises(ValueError):
        FrvWireProviderConfig.from_dict(data)


def test_config_round_trip() -> None:
    config = _config()
    restored = FrvWireProviderConfig.from_dict(config.to_dict())
    assert restored == config


# ---------------------------------------------------------------------------
# Chunk tests
# ---------------------------------------------------------------------------


def test_chunk_valid() -> None:
    chunk = _chunk()
    validate_frv_wire_chunk(chunk, provider_config=_config())


def test_chunk_relation_boundaries() -> None:
    # body at exactly r0, conflict at exactly r1, proximity at exactly r2.
    rows = [
        (0, 1, 2, 0, 0, 0, 0.10),
        (1, 1, 4, 0, 0, 0, 0.50),
        (2, 1, 3, 0, 0, 0, 1.50),
        (3, 1, 0, -1, -1, -1, 0.0),
    ]
    validate_frv_wire_chunk(_chunk(rows=rows), provider_config=_config())


def test_chunk_rejects_body_above_radius() -> None:
    rows = [(0, 1, 2, 0, 0, 0, 0.1001)]
    with pytest.raises(ValueError):
        validate_frv_wire_chunk(_chunk(rows=rows), provider_config=_config())


def test_chunk_rejects_conflict_outside_band() -> None:
    rows = [(0, 1, 4, 0, 0, 0, 0.1000)]  # <= r0 is body, not conflict
    with pytest.raises(ValueError):
        validate_frv_wire_chunk(_chunk(rows=rows), provider_config=_config())


def test_chunk_rejects_proximity_outside_band() -> None:
    rows = [(0, 1, 3, 0, 0, 0, 1.5001)]
    with pytest.raises(ValueError):
        validate_frv_wire_chunk(_chunk(rows=rows), provider_config=_config())


def test_chunk_rejects_in_domain_with_negative_id() -> None:
    rows = [(0, 1, 2, -1, 0, 0, 0.05)]
    with pytest.raises(ValueError):
        validate_frv_wire_chunk(_chunk(rows=rows), provider_config=_config())


def test_chunk_rejects_outside_with_nonzero_distance() -> None:
    rows = [(0, 1, 0, -1, -1, -1, 0.2)]
    with pytest.raises(ValueError):
        validate_frv_wire_chunk(_chunk(rows=rows), provider_config=_config())


def test_chunk_rejects_not_applicable_with_in_domain_relation() -> None:
    rows = [(0, 2, 2, 0, 0, 0, 0.05)]
    with pytest.raises(ValueError):
        validate_frv_wire_chunk(_chunk(rows=rows), provider_config=_config())


def test_chunk_not_applicable_valid() -> None:
    rows = [(0, 2, 0, -1, -1, -1, 0.0)]
    validate_frv_wire_chunk(_chunk(rows=rows), provider_config=_config())


def test_chunk_rejects_unknown_provider_state() -> None:
    rows = [(0, 9, 0, -1, -1, -1, 0.0)]
    with pytest.raises(ValueError):
        validate_frv_wire_chunk(_chunk(rows=rows), provider_config=_config())


def test_chunk_rejects_unknown_relation_state() -> None:
    rows = [(0, 1, 1, 0, 0, 0, 0.05)]  # 1 is the reserved confirmed value
    with pytest.raises(ValueError):
        validate_frv_wire_chunk(_chunk(rows=rows), provider_config=_config())


def test_chunk_rejects_non_increasing_index() -> None:
    rows = [
        (1, 1, 0, -1, -1, -1, 0.0),
        (1, 1, 0, -1, -1, -1, 0.0),
    ]
    with pytest.raises(ValueError):
        validate_frv_wire_chunk(_chunk(rows=rows), provider_config=_config())


def test_chunk_rejects_writable_array() -> None:
    chunk = _chunk()
    mutable = chunk.original_index.copy()
    with pytest.raises(ValueError):
        validate_frv_wire_chunk(
            _chunk(original_index=mutable), provider_config=_config()
        )


def test_chunk_requires_positive_point_count() -> None:
    with pytest.raises(ValueError):
        _chunk(point_count=0)


def test_chunk_rejects_nonfinite_distance() -> None:
    rows = [(0, 1, 3, 0, 0, 0, float("nan"))]
    with pytest.raises(ValueError):
        validate_frv_wire_chunk(_chunk(rows=rows), provider_config=_config())


# ---------------------------------------------------------------------------
# Geometry digest oracle
# ---------------------------------------------------------------------------


def test_wire_geometry_digest_oracle() -> None:
    edges = [
        (0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0),
        (0, 0, 1, 0.0, 0.0, 5.0, 1.0, 0.0, 5.0),
        (1, 0, 0, 2.0, 2.0, 0.0, 2.0, 2.0, 4.0),
    ]
    actual = frv_wire_geometry_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=_SHA_B,
        origin_xyz=(0.0, 0.0, 0.0),
        edges=edges,
    )
    expected = _geo_expected(SHA, _SHA_B, (0.0, 0.0, 0.0), edges)
    assert actual == expected


def test_wire_geometry_digest_origin_binding() -> None:
    edges = [(0, 0, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
    origin = (100.0, 200.0, 300.0)
    actual = frv_wire_geometry_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=_SHA_B,
        origin_xyz=origin,
        edges=edges,
    )
    expected = _geo_expected(SHA, _SHA_B, origin, edges)
    assert actual == expected


def test_wire_geometry_digest_empty() -> None:
    actual = frv_wire_geometry_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=_SHA_B,
        origin_xyz=(0.0, 0.0, 0.0),
        edges=[],
    )
    expected = _geo_expected(SHA, _SHA_B, (0.0, 0.0, 0.0), [])
    assert actual == expected


def test_wire_geometry_digest_rejects_unsorted_edges() -> None:
    edges = [
        (0, 0, 1, 0.0, 0.0, 5.0, 1.0, 0.0, 5.0),
        (0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0),
    ]
    with pytest.raises(ValueError):
        frv_wire_geometry_content_digest(
            source_content_identity_digest=SHA,
            source_identity_record_digest=_SHA_B,
            origin_xyz=(0.0, 0.0, 0.0),
            edges=edges,
        )


# ---------------------------------------------------------------------------
# Point-content digest oracle
# ---------------------------------------------------------------------------


def _digest_rows() -> list[dict[str, np.ndarray]]:
    rows = []
    for index, provider, relation, wire_id, segment, edge, distance in [
        (0, 1, 0, -1, -1, -1, 0.0),
        (1, 1, 2, 0, 0, 0, 0.05),
        (2, 1, 4, 0, 0, 1, 0.25),
        (3, 1, 3, 1, 1, 2, 1.0),
    ]:
        rows.append(
            {
                "original_index": np.asarray([index], dtype="<i8"),
                "provider_state": np.asarray([provider], dtype="|u1"),
                "relation_state": np.asarray([relation], dtype="|u1"),
                "nearest_wire_id": np.asarray([wire_id], dtype="<i4"),
                "nearest_segment_index": np.asarray([segment], dtype="<i4"),
                "nearest_edge_index": np.asarray([edge], dtype="<i4"),
                "centerline_distance_m": np.asarray([distance], dtype="<f8"),
                "reason_flags": np.asarray([0], dtype="<u4"),
                "quality_flags": np.asarray([0], dtype="<u4"),
            }
        )
    return rows


def test_wire_point_content_digest_oracle() -> None:
    chunk = _chunk()
    actual = frv_wire_point_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=_SHA_B,
        authority_manifest_digest=SHA,
        wire_geometry_content_digest=_SHA_C,
        provider_config_digest=SHA,
        source_point_count=4,
        chunks=[chunk],
    )
    expected = _pcd_expected([SHA, _SHA_B, SHA, _SHA_C, SHA], _digest_rows())
    assert actual == expected


def test_wire_point_content_digest_rejects_gap() -> None:
    rows = [
        (0, 1, 0, -1, -1, -1, 0.0),
        (2, 1, 0, -1, -1, -1, 0.0),
    ]
    with pytest.raises(ValueError):
        frv_wire_point_content_digest(
            source_content_identity_digest=SHA,
            source_identity_record_digest=_SHA_B,
            authority_manifest_digest=SHA,
            wire_geometry_content_digest=_SHA_C,
            provider_config_digest=SHA,
            source_point_count=4,
            chunks=[_chunk(rows=rows)],
        )


def test_wire_point_content_digest_chunk_boundary_invariant() -> None:
    single = frv_wire_point_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=_SHA_B,
        authority_manifest_digest=SHA,
        wire_geometry_content_digest=_SHA_C,
        provider_config_digest=SHA,
        source_point_count=4,
        chunks=[_chunk()],
    )
    first = _chunk(rows=[(0, 1, 0, -1, -1, -1, 0.0), (1, 1, 2, 0, 0, 0, 0.05)])
    second = _chunk(
        chunk_id=1,
        start_index=2,
        is_last=True,
        rows=[(2, 1, 4, 0, 0, 1, 0.25), (3, 1, 3, 1, 1, 2, 1.0)],
    )
    split = frv_wire_point_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=_SHA_B,
        authority_manifest_digest=SHA,
        wire_geometry_content_digest=_SHA_C,
        provider_config_digest=SHA,
        source_point_count=4,
        chunks=[first, second],
    )
    assert single == split


# ---------------------------------------------------------------------------
# Proof tests
# ---------------------------------------------------------------------------


def test_proof_valid_non_empty() -> None:
    proof = _proof()
    validate_frv_wire_provider_proof(proof)


def test_proof_valid_empty() -> None:
    proof = _proof(
        authority_mode="empty_inventory",
        approved_wire_count=0,
        wire_relation_valid_count=0,
        wire_not_applicable_proven_count=4,
        outside_influence_domain_count=4,
        wire_body_possible_count=0,
        wire_proximity_only_count=0,
        wire_vegetation_conflict_count=0,
        minimum_centerline_distance_m=0.0,
        maximum_centerline_distance_m=0.0,
    )
    validate_frv_wire_provider_proof(proof)


def test_proof_state_counts_conservation() -> None:
    with pytest.raises(ValueError):
        validate_frv_wire_provider_proof(
            _proof(wire_relation_valid_count=3, wire_not_applicable_proven_count=0)
        )


def test_proof_relation_counts_conservation() -> None:
    with pytest.raises(ValueError):
        validate_frv_wire_provider_proof(
            _proof(outside_influence_domain_count=2)
        )


def test_proof_empty_forbids_in_domain_counts() -> None:
    with pytest.raises(ValueError):
        validate_frv_wire_provider_proof(
            _proof(
                authority_mode="empty_inventory",
                approved_wire_count=0,
                wire_relation_valid_count=0,
                wire_not_applicable_proven_count=4,
                outside_influence_domain_count=3,
                wire_body_possible_count=1,
                wire_proximity_only_count=0,
                wire_vegetation_conflict_count=0,
                minimum_centerline_distance_m=0.05,
                maximum_centerline_distance_m=0.05,
            )
        )


def test_proof_non_empty_forbids_not_applicable() -> None:
    with pytest.raises(ValueError):
        validate_frv_wire_provider_proof(
            _proof(wire_not_applicable_proven_count=1)
        )


def test_proof_no_in_domain_requires_zero_aggregates() -> None:
    with pytest.raises(ValueError):
        validate_frv_wire_provider_proof(
            _proof(
                wire_body_possible_count=0,
                wire_proximity_only_count=0,
                wire_vegetation_conflict_count=0,
                outside_influence_domain_count=4,
                minimum_centerline_distance_m=0.5,
                maximum_centerline_distance_m=0.5,
            )
        )


def test_proof_aggregates_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        validate_frv_wire_provider_proof(
            _proof(minimum_centerline_distance_m=1.0, maximum_centerline_distance_m=0.05)
        )


def test_proof_proof_digest_oracle() -> None:
    proof = _proof()
    payload = _proof_payload()
    payload["artifact_reference_id"] = proof.artifact_reference_id
    expected = hashlib.sha256(_canon(payload)).hexdigest()
    assert proof.proof_digest == expected


def test_proof_rejects_bad_domain() -> None:
    with pytest.raises(ValueError):
        validate_frv_wire_provider_proof(_proof(domain_kind="declared_subset"))


def test_proof_rejects_nonzero_quality() -> None:
    with pytest.raises(ValueError):
        validate_frv_wire_provider_proof(_proof(quality_flags=8))


def test_proof_round_trip() -> None:
    proof = _proof()
    restored = FrvWireProviderProof.from_dict(proof.to_dict())
    assert restored == proof


# ---------------------------------------------------------------------------
# Error and result capability
# ---------------------------------------------------------------------------


def test_load_error_fields() -> None:
    record = FrvA2FailureRecord(
        phase=FrvA2FailurePhase.SEMANTIC_READINESS,
        provider_role=FrvA2ProviderRole.WIRE,
        provider_key=FRV_WIRE_PROVIDER_KEY,
        field_name="wire_geometry",
        error_code=FrvErrorCode.FRV_WIRE_GEOMETRY_UNAVAILABLE,
        reason_flags=1024,
        detail=(
            "semantic_readiness:wire:wire_provider:wire_geometry:"
            "FRV_WIRE_GEOMETRY_UNAVAILABLE:1024"
        ),
    )
    error = FrvWireLoadError(failure_record=record, quality_flags=128)
    assert error.failure_record is record
    assert error.quality_flags == 128
    assert str(error) == record.detail


def test_load_error_rejects_unknown_quality_flag() -> None:
    record = FrvA2FailureRecord(
        phase=FrvA2FailurePhase.SEMANTIC_READINESS,
        provider_role=FrvA2ProviderRole.WIRE,
        provider_key=FRV_WIRE_PROVIDER_KEY,
        field_name="wire_geometry",
        error_code=FrvErrorCode.FRV_WIRE_GEOMETRY_UNAVAILABLE,
        reason_flags=1024,
        detail=(
            "semantic_readiness:wire:wire_provider:wire_geometry:"
            "FRV_WIRE_GEOMETRY_UNAVAILABLE:1024"
        ),
    )
    with pytest.raises(ValueError):
        FrvWireLoadError(failure_record=record, quality_flags=0x80000000)


def test_result_has_no_public_constructor() -> None:
    with pytest.raises(TypeError):
        FrvWireProviderResult()  # type: ignore[call-arg]


def test_authority_digest_domains_frozen() -> None:
    assert FRV_WIRE_AUTHORITY_DIGEST_DOMAIN == "FRVWIRAUTH1"
    assert FRV_WIRE_GEOMETRY_DIGEST_DOMAIN == "FRVWIRGEO1"
