"""Canonical finite serialization and digest tests for FGC contracts.

Authority: docs/full-resolution-ground-classification-design.md sections 3,
11, 12, and the frozen Bundle A plan test gate (schema/digest/finite JSON,
P-HAG digest decoupling, completion-manifest discovery/path traversal).
"""

from __future__ import annotations

import json

import pytest
from core.contracts.frv import frv_digest
from core.contracts.frv_hag import FrvHagProviderConfig
from core.contracts.ground_classification import (
    FGC_COMPLETION_MANIFEST_SCHEMA_VERSION,
    FGC_DECISION_MANIFEST_SCHEMA_VERSION,
    FGC_PUBLICATION_PROOF_SCHEMA_VERSION,
    FGC_SPOOL_SCHEMA_VERSION,
    FgcCompletionManifest,
    FgcCoverageComponent,
    FgcCoverageResult,
    FgcDecisionChunkRecord,
    FgcDecisionManifest,
    FgcPublicationProof,
    FgcSpoolManifest,
    FgcSpoolTileSummary,
    GroundClassificationPolicyConfig,
    GroundDecision,
    fgc_decision_digest,
    map_derivative_class,
    pack_coverage_record,
    unpack_coverage_record,
)

from tests.ground._identity import make_source_identity


def _chunk_records() -> tuple[FgcDecisionChunkRecord, ...]:
    return (
        FgcDecisionChunkRecord(
            start_index=0,
            end_index=3,
            relative_path="decisions/chunk_000000.bin",
            sha256="b" * 64,
            size_bytes=3,
            state_counts={"non_ground": 1, "confirmed_ground": 1, "review_ground": 1},
        ),
        FgcDecisionChunkRecord(
            start_index=3,
            end_index=7,
            relative_path="decisions/chunk_000001.bin",
            sha256="c" * 64,
            size_bytes=4,
            state_counts={"non_ground": 2, "confirmed_ground": 2, "review_ground": 0},
        ),
    )


def test_policy_serialization_roundtrip() -> None:
    policy = GroundClassificationPolicyConfig()
    restored = GroundClassificationPolicyConfig.from_dict(json.loads(json.dumps(policy.to_dict())))
    assert restored == policy
    assert restored.policy_digest == policy.policy_digest


def test_spool_manifest_roundtrip() -> None:
    identity = make_source_identity(count=7)
    manifest = FgcSpoolManifest(
        source_identity_record_digest=identity.source_identity_record_digest,
        source_point_count=7,
        tile_layout_digest="d" * 64,
        tiles=(
            FgcSpoolTileSummary(tile_id_ty=0, tile_id_tx=0, row_count=4, sha256="e" * 64),
            FgcSpoolTileSummary(tile_id_ty=0, tile_id_tx=1, row_count=3, sha256="f" * 64),
        ),
    )
    assert manifest.schema_version == FGC_SPOOL_SCHEMA_VERSION
    restored = FgcSpoolManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    assert restored == manifest
    assert restored.manifest_digest == manifest.manifest_digest


def test_spool_manifest_rejects_unordered_tiles() -> None:
    identity = make_source_identity(count=7)
    with pytest.raises(ValueError, match="ordered"):
        FgcSpoolManifest(
            source_identity_record_digest=identity.source_identity_record_digest,
            source_point_count=7,
            tile_layout_digest="d" * 64,
            tiles=(
                FgcSpoolTileSummary(tile_id_ty=0, tile_id_tx=1, row_count=3, sha256="f" * 64),
                FgcSpoolTileSummary(tile_id_ty=0, tile_id_tx=0, row_count=4, sha256="e" * 64),
            ),
        )


def test_decision_manifest_digest_and_policy_binding() -> None:
    identity = make_source_identity(count=7)
    policy = GroundClassificationPolicyConfig()
    chunks = _chunk_records()
    digest = fgc_decision_digest(
        source_identity=identity,
        policy_digest=policy.policy_digest,
        source_point_count=7,
        chunks=chunks,
    )
    manifest = FgcDecisionManifest(
        source_identity=identity,
        raw_source_sha256="a" * 64,
        source_point_count=7,
        policy_digest=policy.policy_digest,
        chunks=chunks,
        total_state_counts={"non_ground": 3, "confirmed_ground": 3, "review_ground": 1},
        decision_digest=digest,
    )
    assert manifest.schema_version == FGC_DECISION_MANIFEST_SCHEMA_VERSION
    # Cross-bound digest validation.
    manifest.validated_with_policy(policy)
    # Changing policy digest changes the decision digest.
    other_policy = GroundClassificationPolicyConfig.from_dict(
        {**policy.to_dict(), "tile_side_m": 512.0, "policy_digest": ""}
    )
    other_digest = fgc_decision_digest(
        source_identity=identity,
        policy_digest=other_policy.policy_digest,
        source_point_count=7,
        chunks=chunks,
    )
    assert other_digest != digest
    restored = FgcDecisionManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    assert restored == manifest


def test_decision_digest_rejects_reordered_chunks() -> None:
    identity = make_source_identity(count=7)
    policy = GroundClassificationPolicyConfig()
    chunks = _chunk_records()
    fgc_decision_digest(
        source_identity=identity,
        policy_digest=policy.policy_digest,
        source_point_count=7,
        chunks=chunks,
    )
    # Reordering chunks violates ordered contiguous ranges and is rejected,
    # satisfying the design requirement that reordering is detectable.
    reordered = (chunks[1], chunks[0])
    with pytest.raises(ValueError, match="gap or overlap"):
        fgc_decision_digest(
            source_identity=identity,
            policy_digest=policy.policy_digest,
            source_point_count=7,
            chunks=reordered,
        )


def test_p_hag_digest_is_decoupled_from_policy_and_decisions() -> None:
    identity = make_source_identity(count=7)
    policy = GroundClassificationPolicyConfig()
    chunks = _chunk_records()
    digest = fgc_decision_digest(
        source_identity=identity,
        policy_digest=policy.policy_digest,
        source_point_count=7,
        chunks=chunks,
    )

    def _hag_config(cell_size_m: float) -> FrvHagProviderConfig:
        from core.contracts.frv_a2 import FrvA2ProviderRole
        from core.contracts.frv_hag import (
            FRV_HAG_ALGORITHM_VERSION,
            FRV_HAG_CONFIG_SCHEMA_VERSION,
            FRV_HAG_PROVIDER_INTERFACE_VERSION,
            FRV_HAG_PROVIDER_KEY,
            derive_frv_hag_provider_config_digest,
        )

        return FrvHagProviderConfig(
            schema_version=FRV_HAG_CONFIG_SCHEMA_VERSION,
            provider_key=FRV_HAG_PROVIDER_KEY,
            provider_role=FrvA2ProviderRole.HAG,
            interface_version=FRV_HAG_PROVIDER_INTERFACE_VERSION,
            algorithm_version=FRV_HAG_ALGORITHM_VERSION,
            cell_size_m=cell_size_m,
            support_radius_m=3.0,
            min_support_cells=1,
            max_support_cells=16,
            negative_hag_tolerance_m=0.1,
            classification_dimension="classification",
            provider_config_digest=derive_frv_hag_provider_config_digest(
                schema_version=FRV_HAG_CONFIG_SCHEMA_VERSION,
                provider_key=FRV_HAG_PROVIDER_KEY,
                provider_role=FrvA2ProviderRole.HAG,
                interface_version=FRV_HAG_PROVIDER_INTERFACE_VERSION,
                algorithm_version=FRV_HAG_ALGORITHM_VERSION,
                cell_size_m=cell_size_m,
                support_radius_m=3.0,
                min_support_cells=1,
                max_support_cells=16,
                negative_hag_tolerance_m=0.1,
                classification_dimension="classification",
            ),
        )

    # Changing P-HAG numerical config must NOT change policy or decision digests.
    config_a = _hag_config(1.0)
    config_b = _hag_config(2.0)
    assert config_a.provider_config_digest != config_b.provider_config_digest
    assert policy.policy_digest == GroundClassificationPolicyConfig().policy_digest
    same_digest = fgc_decision_digest(
        source_identity=identity,
        policy_digest=policy.policy_digest,
        source_point_count=7,
        chunks=chunks,
    )
    assert same_digest == digest
    # Policy schema never contains a P-HAG field.
    assert "hag" not in policy.to_dict()


def test_publication_proof_roundtrip_and_digest() -> None:
    raw_identity = make_source_identity(count=7, source_id="raw-scan")
    classified_identity = make_source_identity(
        count=7, source_id="raw-scan--fgc-v1--0123456789abcdef"
    )
    proof = FgcPublicationProof(
        raw_source_identity=raw_identity,
        raw_source_sha256="a" * 64,
        source_point_count=7,
        policy_digest=GroundClassificationPolicyConfig().policy_digest,
        decision_digest="b" * 64,
        decision_manifest_digest="c" * 64,
        classification_mapping_counts={"preserved": 4, "to_2": 2, "to_1": 1},
        classified_source_identity=classified_identity,
        classified_source_sha256="d" * 64,
        p_source_replay={"status": "valid"},
        p_hag_compatibility={"config_digest": "e" * 64, "result": "ok"},
    )
    assert proof.schema_version == FGC_PUBLICATION_PROOF_SCHEMA_VERSION
    restored = FgcPublicationProof.from_dict(json.loads(json.dumps(proof.to_dict())))
    assert restored == proof
    assert restored.proof_digest == proof.proof_digest
    # Tampering with the P-HAG compatibility block changes only the proof.
    altered = FgcPublicationProof.from_dict(
        {
            **proof.to_dict(),
            "p_hag_compatibility": {"config_digest": "f" * 64, "result": "ok"},
            "proof_digest": "",
        }
    )
    assert altered.proof_digest != proof.proof_digest
    assert altered.policy_digest == proof.policy_digest


def test_completion_manifest_digest_omits_itself_and_binds_components() -> None:
    raw_identity = make_source_identity(count=7, source_id="raw-scan")
    classified_identity = make_source_identity(
        count=7, source_id="raw-scan--fgc-v1--0123456789abcdef"
    )
    chunks = _chunk_records()
    manifest = FgcCompletionManifest(
        policy_digest=GroundClassificationPolicyConfig().policy_digest,
        raw_source_identity=raw_identity,
        classified_source_identity=classified_identity,
        decision_manifest_sha256="b" * 64,
        decision_manifest_size_bytes=120,
        decision_manifest_digest="c" * 64,
        publication_proof_sha256="d" * 64,
        publication_proof_size_bytes=200,
        publication_proof_digest="e" * 64,
        classified_las_path="classified/scan.fgc-v1.las",
        classified_las_sha256="f" * 64,
        classified_las_size_bytes=1000,
        classified_las_point_count=7,
        decision_chunks=chunks,
    )
    assert manifest.schema_version == FGC_COMPLETION_MANIFEST_SCHEMA_VERSION
    # The completion digest is SHA-256 of the canonical payload without itself.
    payload = manifest.to_digest_dict()
    assert "completion_digest" not in payload
    assert manifest.completion_digest == frv_digest(payload)
    restored = FgcCompletionManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    assert restored == manifest
    # Tampering with a bound component changes the completion digest.
    altered = FgcCompletionManifest.from_dict(
        {
            **manifest.to_dict(),
            "classified_las_sha256": "0" * 64,
            "completion_digest": "",
        }
    )
    assert altered.completion_digest != manifest.completion_digest


def test_completion_manifest_rejects_unsafe_paths() -> None:
    raw_identity = make_source_identity(count=7, source_id="raw-scan")
    classified_identity = make_source_identity(count=7, source_id="raw-scan--fgc-v1--x")
    kwargs = dict(
        policy_digest=GroundClassificationPolicyConfig().policy_digest,
        raw_source_identity=raw_identity,
        classified_source_identity=classified_identity,
        decision_manifest_sha256="b" * 64,
        decision_manifest_size_bytes=120,
        decision_manifest_digest="c" * 64,
        publication_proof_sha256="d" * 64,
        publication_proof_size_bytes=200,
        publication_proof_digest="e" * 64,
        classified_las_sha256="f" * 64,
        classified_las_size_bytes=1000,
        classified_las_point_count=7,
    )
    with pytest.raises(ValueError, match="relative"):
        FgcCompletionManifest(classified_las_path="C:/absolute/scan.las", **kwargs)
    with pytest.raises(ValueError, match="relative"):
        FgcCompletionManifest(classified_las_path="classified/../../scan.las", **kwargs)
    with pytest.raises(ValueError):
        FgcCompletionManifest(classified_las_path="", **kwargs)


def test_coverage_record_pack_roundtrip() -> None:
    record = pack_coverage_record(12, -5, 0x03)
    assert len(record) == 17
    ix, iy, flags = unpack_coverage_record(record)
    assert (ix, iy, flags) == (12, -5, 0x03)
    with pytest.raises(ValueError):
        pack_coverage_record(0, 0, 300)
    with pytest.raises(ValueError):
        unpack_coverage_record(b"x" * 16)


def test_coverage_component_width() -> None:
    component = FgcCoverageComponent(min_ix=0, max_ix=24, min_iy=0, max_iy=0, is_edge=False)
    assert component.width_cells() == 25
    assert component.width_m(coverage_cell_m=1.0) == 25.0
    assert component.width_m(coverage_cell_m=0.5) == 12.5


def test_coverage_result_occupancy_gate() -> None:
    policy = GroundClassificationPolicyConfig()
    result = FgcCoverageResult(
        observed_cell_count=100,
        confirmed_occupied_cell_count=20,
        occupancy_fraction=0.20,
        unsupported_components=(),
    )
    assert result.passes_occupancy(policy=policy)
    below = FgcCoverageResult(
        observed_cell_count=100,
        confirmed_occupied_cell_count=19,
        occupancy_fraction=0.19,
        unsupported_components=(),
    )
    assert not below.passes_occupancy(policy=policy)


def test_map_derivative_class_json_safe() -> None:
    mapping = {
        "to_2": sum(
            map_derivative_class(c, GroundDecision.CONFIRMED_GROUND) == 2 for c in (0, 1, 2, 3, 5)
        ),
        "preserved": sum(
            map_derivative_class(c, GroundDecision.REVIEW_GROUND) == c for c in (0, 1, 3, 5)
        ),
    }
    assert mapping["to_2"] == 5
    assert mapping["preserved"] == 4
