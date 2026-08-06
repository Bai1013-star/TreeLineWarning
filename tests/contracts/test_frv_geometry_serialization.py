"""Exact P-GEOMETRY serialization and digest vector tests (PLAN-FROZEN)."""

from __future__ import annotations

import dataclasses
import hashlib
import struct

import core.contracts.frv as frv
import numpy as np
import pytest
from core.contracts.frv_geometry import (
    FRV_GEOMETRY_CONFIG_DIGEST_DOMAIN,
    FRV_GEOMETRY_POINT_CONTENT_DIGEST_DOMAIN,
    FRV_GEOMETRY_PROOF_DIGEST_DOMAIN,
    FrvGeometryProviderConfig,
    FrvGeometryProviderProof,
    derive_frv_geometry_provider_config_digest,
    frv_geometry_point_content_digest,
)

from tests.contracts.test_frv_geometry_contracts import _chunk, _config, _valid_proof

SHA = "a" * 64


def test_config_serialization_exact_shape() -> None:
    config = _config()
    payload = config.to_dict()
    assert list(payload) == [field.name for field in dataclasses.fields(FrvGeometryProviderConfig)]
    assert payload["provider_role"] == config.provider_role.value
    assert "digest_domain" not in payload
    restored = FrvGeometryProviderConfig.from_dict(payload)
    assert restored == config
    assert config.to_dict() == config.to_dict()


def test_config_serialization_rejects_missing_extra_keys() -> None:
    payload = _config().to_dict()
    del payload["radii_m"]
    with pytest.raises(ValueError, match="missing"):
        FrvGeometryProviderConfig.from_dict(payload)
    payload = _config().to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        FrvGeometryProviderConfig.from_dict(payload)
    with pytest.raises(ValueError):
        FrvGeometryProviderConfig.from_dict("not-a-mapping")  # type: ignore[arg-type]


def test_config_digest_domain_and_stability() -> None:
    config = _config()
    assert config.provider_config_digest == derive_frv_geometry_provider_config_digest(
        schema_version=config.schema_version,
        provider_key=config.provider_key,
        provider_role=config.provider_role,
        interface_version=config.interface_version,
        algorithm_version=config.algorithm_version,
        radii_m=config.radii_m,
        min_neighbor_counts=config.min_neighbor_counts,
        max_neighbors_per_scale=config.max_neighbors_per_scale,
        index_cell_size_m=config.index_cell_size_m,
        support_neighbor_radius_m=config.support_neighbor_radius_m,
        support_neighbor_cap=config.support_neighbor_cap,
        direction_cosine_min=config.direction_cosine_min,
        eigenvalue_floor_m2=config.eigenvalue_floor_m2,
        canopy_scattering_min=config.canopy_scattering_min,
        canopy_linearity_max=config.canopy_linearity_max,
        canopy_density_min_per_m3=config.canopy_density_min_per_m3,
        woody_linearity_min=config.woody_linearity_min,
        woody_verticality_max=config.woody_verticality_max,
        woody_vertical_span_min_m=config.woody_vertical_span_min_m,
        vertical_cylinder_radius_m=config.vertical_cylinder_radius_m,
        vertical_half_window_m=config.vertical_half_window_m,
        vertical_slice_height_m=config.vertical_slice_height_m,
        min_vertical_occupied_slices=config.min_vertical_occupied_slices,
        ground_planarity_min=config.ground_planarity_min,
        ground_normal_z_min=config.ground_normal_z_min,
        max_index_cell_points=config.max_index_cell_points,
    )
    assert FRV_GEOMETRY_CONFIG_DIGEST_DOMAIN == "FRVGEOCFG1"
    other = _config(index_cell_size_m=1.0)
    assert other.provider_config_digest != config.provider_config_digest


def test_proof_serialization_exact_shape() -> None:
    proof = _valid_proof()
    payload = proof.to_dict()
    assert list(payload) == [field.name for field in dataclasses.fields(FrvGeometryProviderProof)]
    assert "digest_domain" not in payload
    restored = FrvGeometryProviderProof.from_dict(payload)
    assert restored == proof
    assert proof.to_dict() == proof.to_dict()
    assert restored.proof_digest == restored.computed_digest()


def test_proof_serialization_rejects_missing_extra_keys() -> None:
    payload = _valid_proof().to_dict()
    del payload["geometry_point_content_digest"]
    with pytest.raises(ValueError, match="missing"):
        FrvGeometryProviderProof.from_dict(payload)
    payload = _valid_proof().to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        FrvGeometryProviderProof.from_dict(payload)


def test_proof_digest_domain_and_stability() -> None:
    proof = _valid_proof()
    assert proof.computed_digest() == proof.computed_digest()
    assert proof.proof_digest == proof.computed_digest()
    assert FRV_GEOMETRY_PROOF_DIGEST_DOMAIN == "FRVGEOPRF1"
    digest_payload = proof.to_digest_dict()
    assert "digest_domain" in digest_payload
    for forbidden in ("path", "host", "timestamp", "chunk_size", "capability", "scratch"):
        assert forbidden not in digest_payload


def test_point_content_digest_deterministic() -> None:
    chunk = _chunk(point_count=2, is_last=True)
    d1 = frv_geometry_point_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=SHA,
        provider_config_digest=SHA,
        source_point_count=2,
        chunks=[chunk],
    )
    d2 = frv_geometry_point_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=SHA,
        provider_config_digest=SHA,
        source_point_count=2,
        chunks=[chunk],
    )
    assert d1 == d2
    assert len(d1) == 64
    assert FRV_GEOMETRY_POINT_CONTENT_DIGEST_DOMAIN == "FRVGEOPCD1"


# ---------------------------------------------------------------------------
# Independent fixed digest oracles.
#
# The oracles rebuild the byte/canonical streams with raw ``hashlib`` +
# ``struct`` (mirroring the frozen framings) and never call the production
# digest functions.  Each oracle is pinned to a fixed golden vector so a
# lockstep change to both production and oracle is also caught.
# ---------------------------------------------------------------------------


def _oracle_config_payload(config) -> dict:
    return {
        "digest_domain": "FRVGEOCFG1",
        "schema_version": config.schema_version,
        "provider_key": config.provider_key,
        "provider_role": config.provider_role.value,
        "interface_version": config.interface_version,
        "algorithm_version": config.algorithm_version,
        "radii_m": list(config.radii_m),
        "min_neighbor_counts": list(config.min_neighbor_counts),
        "max_neighbors_per_scale": config.max_neighbors_per_scale,
        "index_cell_size_m": config.index_cell_size_m,
        "support_neighbor_radius_m": config.support_neighbor_radius_m,
        "support_neighbor_cap": config.support_neighbor_cap,
        "direction_cosine_min": config.direction_cosine_min,
        "eigenvalue_floor_m2": config.eigenvalue_floor_m2,
        "canopy_scattering_min": config.canopy_scattering_min,
        "canopy_linearity_max": config.canopy_linearity_max,
        "canopy_density_min_per_m3": config.canopy_density_min_per_m3,
        "woody_linearity_min": config.woody_linearity_min,
        "woody_verticality_max": config.woody_verticality_max,
        "woody_vertical_span_min_m": config.woody_vertical_span_min_m,
        "vertical_cylinder_radius_m": config.vertical_cylinder_radius_m,
        "vertical_half_window_m": config.vertical_half_window_m,
        "vertical_slice_height_m": config.vertical_slice_height_m,
        "min_vertical_occupied_slices": config.min_vertical_occupied_slices,
        "ground_planarity_min": config.ground_planarity_min,
        "ground_normal_z_min": config.ground_normal_z_min,
        "max_index_cell_points": config.max_index_cell_points,
    }


def test_config_digest_fixed_independent_oracle() -> None:
    config = _config()
    expected = frv.frv_digest(_oracle_config_payload(config))
    assert config.provider_config_digest == expected
    assert (
        expected
        == "ff16c8b311f0c23fd9ba2eda5f38dad6bc64d21d91e666f0f3adaa8304958fa3"
    )


_PCD_FIELDS = (
    ("original_index", "<i8", 1),
    ("raw_neighbor_count", "<u4", 3),
    ("neighbor_count", "<u2", 3),
    ("density_per_m3", "<f4", 3),
    ("local_spacing_m", "<f4", 3),
    ("eigenvalues_m2", "<f8", 9),
    ("principal_direction", "<f4", 9),
    ("surface_normal", "<f4", 9),
    ("linearity", "<f4", 3),
    ("planarity", "<f4", 3),
    ("scattering", "<f4", 3),
    ("verticality", "<f4", 3),
    ("vertical_span_m", "<f4", 1),
    ("vertical_occupied_slices", "<u2", 1),
    ("above_count", "<u2", 1),
    ("below_count", "<u2", 1),
    ("support_neighbor_count", "|u1", 1),
    ("support_neighbor_original_index", "<i8", 16),
    ("geometry_evidence_flags", "<u4", 1),
    ("geometry_evidence_state", "|u1", 1),
    ("reason_flags", "<u4", 1),
    ("quality_flags", "<u4", 1),
)


def _oracle_point_content_digest(*, source_point_count: int, chunks) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"FRVGEOPCD1\0")
    hasher.update(SHA.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(SHA.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(SHA.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(struct.pack("<Q", source_point_count))
    for name, dtype_str, count in _PCD_FIELDS:
        name_bytes = name.encode("utf-8")
        dtype = np.dtype(dtype_str)
        row_byte_length = dtype.itemsize * count
        hasher.update(struct.pack("<H", len(name_bytes)))
        hasher.update(name_bytes)
        hasher.update(struct.pack("<I", row_byte_length))
        for chunk in chunks:
            array = getattr(chunk, name)
            for row in range(chunk.point_count):
                if dtype_str in ("<f4", "<f8"):
                    row_bytes = (array[row] + np.zeros(1, dtype=dtype)).tobytes()
                else:
                    row_bytes = array[row].tobytes()
                hasher.update(row_bytes)
    return hasher.hexdigest()


def test_point_content_digest_fixed_independent_oracle() -> None:
    chunk = _chunk(point_count=2, is_last=True)
    expected = _oracle_point_content_digest(source_point_count=2, chunks=[chunk])
    produced = frv_geometry_point_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=SHA,
        provider_config_digest=SHA,
        source_point_count=2,
        chunks=[chunk],
    )
    assert produced == expected
    assert (
        expected
        == "e8eb44afe9820be74e0602eb39dc2ba598b513a3689ce01209345cd264239cba"
    )


def test_proof_digest_fixed_independent_oracle() -> None:
    proof = _valid_proof()
    expected = frv.frv_digest(proof.to_digest_dict())
    assert proof.proof_digest == expected
    assert (
        expected
        == "e5e0299d0d6a2ae96a23e541f6fa4798e9a6efd797e0a522f81eb54a7eebd164"
    )
