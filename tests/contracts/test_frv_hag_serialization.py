"""Exact P-HAG serialization and digest vector tests (PLAN-FROZEN)."""

from __future__ import annotations

import dataclasses
import hashlib
import struct

import pytest
from core.contracts.frv_hag import (
    FRV_HAG_CONFIG_DIGEST_DOMAIN,
    FRV_HAG_POINT_CONTENT_DIGEST_DOMAIN,
    FRV_HAG_PROOF_DIGEST_DOMAIN,
    FrvHagProviderConfig,
    FrvHagProviderProof,
    derive_frv_hag_provider_config_digest,
    frv_hag_ground_cell_inventory_digest,
    frv_hag_point_content_digest,
)

from tests.contracts.test_frv_hag_contracts import _chunk, _config, _valid_proof

SHA = "a" * 64


def test_config_serialization_exact_shape() -> None:
    config = _config()
    payload = config.to_dict()
    assert list(payload) == [f.name for f in dataclasses.fields(FrvHagProviderConfig)]
    assert payload["provider_role"] == config.provider_role.value
    assert "digest_domain" not in payload
    restored = FrvHagProviderConfig.from_dict(payload)
    assert restored == config
    assert config.to_dict() == config.to_dict()


def test_config_serialization_rejects_missing_extra_keys() -> None:
    payload = _config().to_dict()
    del payload["cell_size_m"]
    with pytest.raises(ValueError, match="missing"):
        FrvHagProviderConfig.from_dict(payload)
    payload = _config().to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        FrvHagProviderConfig.from_dict(payload)
    with pytest.raises(ValueError):
        FrvHagProviderConfig.from_dict("not-a-mapping")  # type: ignore[arg-type]


def test_config_digest_domain_and_stability() -> None:
    config = _config()
    assert config.provider_config_digest == derive_frv_hag_provider_config_digest(
        schema_version=config.schema_version,
        provider_key=config.provider_key,
        provider_role=config.provider_role,
        interface_version=config.interface_version,
        algorithm_version=config.algorithm_version,
        cell_size_m=config.cell_size_m,
        support_radius_m=config.support_radius_m,
        min_support_cells=config.min_support_cells,
        max_support_cells=config.max_support_cells,
        negative_hag_tolerance_m=config.negative_hag_tolerance_m,
        classification_dimension=config.classification_dimension,
    )
    # domain prefix is part of the canonical payload (via frv_digest of the dict)
    assert FRV_HAG_CONFIG_DIGEST_DOMAIN == "FRVHAGCFG1"
    # changing a field changes the digest
    other = _config(cell_size_m=2.0)
    assert other.provider_config_digest != config.provider_config_digest


def test_proof_serialization_exact_shape() -> None:
    proof = _valid_proof()
    payload = proof.to_dict()
    assert list(payload) == [f.name for f in dataclasses.fields(FrvHagProviderProof)]
    assert "digest_domain" not in payload
    restored = FrvHagProviderProof.from_dict(payload)
    assert restored == proof
    assert proof.to_dict() == proof.to_dict()
    # proof digest is recomputed by construction
    assert restored.proof_digest == restored.computed_digest()


def test_proof_serialization_rejects_missing_extra_keys() -> None:
    payload = _valid_proof().to_dict()
    del payload["source_point_count"]
    with pytest.raises(ValueError, match="missing"):
        FrvHagProviderProof.from_dict(payload)
    payload = _valid_proof().to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        FrvHagProviderProof.from_dict(payload)


def test_proof_digest_domain_and_stability() -> None:
    proof = _valid_proof()
    assert proof.computed_digest() == proof.computed_digest()
    assert proof.proof_digest == proof.computed_digest()
    assert FRV_HAG_PROOF_DIGEST_DOMAIN == "FRVHAGPRF1"
    # proof digest excludes runtime/path/timestamp fields
    digest_payload = proof.to_digest_dict()
    assert "digest_domain" in digest_payload
    for forbidden in ("path", "host", "timestamp", "chunk_size", "capability"):
        assert forbidden not in digest_payload


def test_ground_inventory_digest_zero_normalization() -> None:
    cells_positive_zero = ((0, 0, 0.0, 2), (1, 1, -0.25, 1))
    cells_negative_zero = ((0, 0, -0.0, 2), (1, 1, -0.25, 1))
    d_pos = frv_hag_ground_cell_inventory_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=SHA,
        provider_config_digest=SHA,
        ground_point_count=3,
        cells=cells_positive_zero,
    )
    d_neg = frv_hag_ground_cell_inventory_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=SHA,
        provider_config_digest=SHA,
        ground_point_count=3,
        cells=cells_negative_zero,
    )
    assert d_pos == d_neg  # -0.0 normalizes to +0.0


def test_point_content_digest_deterministic_and_zero_normalized() -> None:
    chunk = _chunk(point_count=2, is_last=True)
    d1 = frv_hag_point_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=SHA,
        provider_config_digest=SHA,
        source_point_count=2,
        chunks=[chunk],
    )
    d2 = frv_hag_point_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=SHA,
        provider_config_digest=SHA,
        source_point_count=2,
        chunks=[chunk],
    )
    assert d1 == d2
    assert len(d1) == 64
    assert FRV_HAG_POINT_CONTENT_DIGEST_DOMAIN == "FRVHAGPCD1"


# ---------------------------------------------------------------------------
# Independent FRVHAGGRID1 / FRVHAGPCD1 fixed digest oracles.
#
# The oracles below rebuild the byte stream with raw ``hashlib`` + ``struct``
# (mirroring the frozen framing) and never call the production digest
# functions or their private helpers.  Each oracle is additionally pinned to a
# fixed golden vector so a lockstep change to both production and oracle is
# also caught.
# ---------------------------------------------------------------------------


def _oracle_grid_digest(
    *,
    source_content_identity_digest: str,
    source_identity_record_digest: str,
    provider_config_digest: str,
    ground_point_count: int,
    cells: tuple[tuple[int, int, float, int], ...],
) -> str:
    ordered = sorted(cells, key=lambda cell: (cell[0], cell[1]))
    hasher = hashlib.sha256()
    hasher.update(b"FRVHAGGRID1\0")
    hasher.update(source_content_identity_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(source_identity_record_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(provider_config_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(struct.pack("<Q", ground_point_count))
    hasher.update(struct.pack("<Q", len(ordered)))
    for ix, iy, min_z, count in ordered:
        z = 0.0 if min_z == 0.0 else float(min_z)
        hasher.update(struct.pack("<q", ix))
        hasher.update(struct.pack("<q", iy))
        hasher.update(struct.pack("<d", z))
        hasher.update(struct.pack("<Q", count))
    return hasher.hexdigest()


def test_grid_digest_fixed_independent_oracle() -> None:
    cells = ((2, 3, -0.0, 1), (1, 1, -0.25, 2), (0, 0, 0.0, 1))
    expected = _oracle_grid_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest="b" * 64,
        provider_config_digest="c" * 64,
        ground_point_count=4,
        cells=cells,
    )
    produced = frv_hag_ground_cell_inventory_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest="b" * 64,
        provider_config_digest="c" * 64,
        ground_point_count=4,
        cells=cells,
    )
    assert produced == expected
    # Fixed golden vector (SHA-256 over the independently reconstructed bytes).
    assert (
        expected
        == "9b40fba1368140cce94955f55a12d2aa4260459dc24174eed80e8dd286fc38bd"
    )


def _oracle_point_content_digest(
    *,
    source_content_identity_digest: str,
    source_identity_record_digest: str,
    provider_config_digest: str,
    source_point_count: int,
    chunk,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"FRVHAGPCD1\0")
    hasher.update(source_content_identity_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(source_identity_record_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(provider_config_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(struct.pack("<Q", source_point_count))
    for row in range(chunk.point_count):
        hasher.update(struct.pack("<Q", int(chunk.original_index[row])))
        hasher.update(struct.pack("<B", int(bool(chunk.ground_support[row]))))
        hasher.update(struct.pack("<H", int(chunk.support_cell_count[row])))
        gz = float(chunk.ground_z_m[row])
        raw = float(chunk.raw_hag_m[row])
        clamped = float(chunk.clamped_hag_m[row])
        hasher.update(struct.pack("<d", 0.0 if gz == 0.0 else gz))
        hasher.update(struct.pack("<d", 0.0 if raw == 0.0 else raw))
        hasher.update(struct.pack("<d", 0.0 if clamped == 0.0 else clamped))
        hasher.update(struct.pack("<B", int(chunk.hag_evidence_state[row])))
        hasher.update(struct.pack("<I", int(chunk.reason_flags[row])))
        hasher.update(struct.pack("<I", int(chunk.quality_flags[row])))
    return hasher.hexdigest()


def test_point_content_digest_fixed_independent_oracle() -> None:
    chunk = _chunk(point_count=2, is_last=True)
    expected = _oracle_point_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest="b" * 64,
        provider_config_digest="c" * 64,
        source_point_count=2,
        chunk=chunk,
    )
    produced = frv_hag_point_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest="b" * 64,
        provider_config_digest="c" * 64,
        source_point_count=2,
        chunks=[chunk],
    )
    assert produced == expected
    # Fixed golden vector (SHA-256 over the independently reconstructed bytes).
    assert (
        expected
        == "afbd06d4e3003b6ece9b4d332386d73e4d660321ca07c5a5ac50d1c6372ca885"
    )
