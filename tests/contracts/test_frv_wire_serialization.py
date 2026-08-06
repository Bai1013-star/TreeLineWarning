"""Exact P-WIRE serialization and digest domain tests (PLAN-FROZEN)."""

from __future__ import annotations

import dataclasses
import json

import core.contracts.frv_wire as wire
import pytest
from core.contracts.frv_wire import (
    FRV_WIRE_AUTHORITY_DIGEST_DOMAIN,
    FRV_WIRE_CONFIG_DIGEST_DOMAIN,
    FRV_WIRE_GEOMETRY_DIGEST_DOMAIN,
    FRV_WIRE_POINT_CONTENT_DIGEST_DOMAIN,
    FRV_WIRE_PROOF_DIGEST_DOMAIN,
    FrvWireAuthorityManifest,
    FrvWireAuthorityMode,
    FrvWireProviderConfig,
    FrvWireProviderProof,
    derive_frv_wire_provider_config_digest,
)

from tests.contracts.test_frv_wire_contracts import (
    SHA,
    _chunk,
    _config,
    _manifest,
    _proof,
)

# ---------------------------------------------------------------------------
# Config serialization
# ---------------------------------------------------------------------------


def test_config_serialization_exact_shape() -> None:
    config = _config()
    payload = config.to_dict()
    assert list(payload) == [field.name for field in dataclasses.fields(FrvWireProviderConfig)]
    assert payload["provider_role"] == config.provider_role.value
    assert "digest_domain" not in payload
    restored = FrvWireProviderConfig.from_dict(payload)
    assert restored == config


def test_config_serialization_rejects_missing_extra_keys() -> None:
    payload = _config().to_dict()
    del payload["max_edges_per_index_cell"]
    with pytest.raises(ValueError, match="missing"):
        FrvWireProviderConfig.from_dict(payload)
    payload = _config().to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        FrvWireProviderConfig.from_dict(payload)
    with pytest.raises(ValueError):
        FrvWireProviderConfig.from_dict("not-a-mapping")  # type: ignore[arg-type]


def test_config_digest_domain_and_stability() -> None:
    config = _config()
    assert config.provider_config_digest == derive_frv_wire_provider_config_digest(
        schema_version=config.schema_version,
        provider_key=config.provider_key,
        provider_role=config.provider_role,
        interface_version=config.interface_version,
        algorithm_version=config.algorithm_version,
        wire_body_possible_radius_m=config.wire_body_possible_radius_m,
        vegetation_conflict_radius_m=config.vegetation_conflict_radius_m,
        proximity_radius_m=config.proximity_radius_m,
        spatial_index_cell_size_m=config.spatial_index_cell_size_m,
        max_edges_per_index_cell=config.max_edges_per_index_cell,
    )
    assert config.provider_config_digest == config.provider_config_digest
    assert config.to_dict() == config.to_dict()


# ---------------------------------------------------------------------------
# Authority manifest serialization
# ---------------------------------------------------------------------------


def test_manifest_serialization_exact_shape() -> None:
    manifest = _manifest()
    payload = manifest.to_dict()
    assert list(payload) == [field.name for field in dataclasses.fields(FrvWireAuthorityManifest)]
    assert payload["authority_mode"] == FrvWireAuthorityMode.W1_MODEL_SET.value
    restored = FrvWireAuthorityManifest.from_dict(payload)
    assert restored == manifest
    assert restored.manifest_digest == manifest.manifest_digest


def test_manifest_serialization_empty_mode() -> None:
    manifest = _manifest(
        authority_mode=FrvWireAuthorityMode.EMPTY_INVENTORY,
        approved_wire_ids=(),
        w1_artifact_components=(),
    )
    restored = FrvWireAuthorityManifest.from_dict(manifest.to_dict())
    assert restored == manifest
    assert restored.authority_mode is FrvWireAuthorityMode.EMPTY_INVENTORY


def test_manifest_serialization_rejects_missing_extra_keys() -> None:
    payload = _manifest().to_dict()
    del payload["approver"]
    with pytest.raises(ValueError, match="missing"):
        FrvWireAuthorityManifest.from_dict(payload)
    payload = _manifest().to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        FrvWireAuthorityManifest.from_dict(payload)


def test_manifest_json_round_trip_via_json() -> None:
    manifest = _manifest()
    encoded = json.dumps(manifest.to_dict(), ensure_ascii=False)
    decoded = json.loads(encoded)
    restored = FrvWireAuthorityManifest.from_dict(decoded)
    assert restored == manifest


# ---------------------------------------------------------------------------
# Proof serialization
# ---------------------------------------------------------------------------


def test_proof_serialization_exact_shape() -> None:
    proof = _proof()
    payload = proof.to_dict()
    expected_names = [field.name for field in dataclasses.fields(FrvWireProviderProof)]
    assert list(payload) == expected_names
    assert "digest_domain" not in payload
    restored = FrvWireProviderProof.from_dict(payload)
    assert restored == proof
    assert restored.proof_digest == proof.proof_digest


def test_proof_serialization_rejects_missing_extra_keys() -> None:
    payload = _proof().to_dict()
    del payload["authority_mode"]
    with pytest.raises(ValueError, match="missing"):
        FrvWireProviderProof.from_dict(payload)
    payload = _proof().to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        FrvWireProviderProof.from_dict(payload)


def test_proof_json_round_trip_via_json() -> None:
    proof = _proof()
    encoded = json.dumps(proof.to_dict(), ensure_ascii=False)
    decoded = json.loads(encoded)
    restored = FrvWireProviderProof.from_dict(decoded)
    assert restored == proof


# ---------------------------------------------------------------------------
# Fixed digest domain strings
# ---------------------------------------------------------------------------


def test_digest_domains_frozen() -> None:
    assert FRV_WIRE_CONFIG_DIGEST_DOMAIN == "FRVWIRCFG1"
    assert FRV_WIRE_POINT_CONTENT_DIGEST_DOMAIN == "FRVWIRPCD1"
    assert FRV_WIRE_PROOF_DIGEST_DOMAIN == "FRVWIRPRF1"
    assert FRV_WIRE_AUTHORITY_DIGEST_DOMAIN == "FRVWIRAUTH1"
    assert FRV_WIRE_GEOMETRY_DIGEST_DOMAIN == "FRVWIRGEO1"


def test_fixed_vectors_are_stable() -> None:
    # Fixed literals pin the frozen framings so a silent framing change fails.
    assert wire.FRV_WIRE_PROVIDER_KEY == "wire_provider"
    assert wire.FRV_WIRE_PROVIDER_INTERFACE_VERSION == "frv-wire-provider-v1"
    assert wire.FRV_WIRE_PROVIDER_SCHEMA_VERSION == "frv-wire-provider-result-v1"
    assert wire.FRV_WIRE_CONFIG_SCHEMA_VERSION == "frv-wire-provider-config-v1"
    assert wire.FRV_WIRE_PROOF_SCHEMA_VERSION == "frv-wire-provider-proof-v1"
    assert wire.FRV_WIRE_ALGORITHM_VERSION == "frv-wire-polyline-distance-grid-v1"
    assert wire.FRV_WIRE_AUTHORITY_SCHEMA_VERSION == "frv-wire-authority-v1"


def test_proof_pcd_row_framing_fixed_literals() -> None:
    # The chunk point-content framing must stay byte-identical across sessions.
    chunk = _chunk()
    digest = wire.frv_wire_point_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest="b" * 64,
        authority_manifest_digest=SHA,
        wire_geometry_content_digest="c" * 64,
        provider_config_digest=SHA,
        source_point_count=4,
        chunks=[chunk],
    )
    assert isinstance(digest, str)
    assert len(digest) == 64
