"""FGC + P-SOURCE replay + P-HAG compatibility integration tests.

Authority: docs/full-resolution-ground-classification-design.md sections 3, 14
and the frozen Bundle C plan test gate (source replay mismatch, P-HAG
compatibility, P-HAG digest decoupling, classified parent lineage).
"""

from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
from algorithms.ground.classification import (
    classify_ground,
)
from core.contracts.ground_classification import (
    FGC_PUBLICATION_PROOF_PATH,
    GroundClassificationPolicyConfig,
)

from tests.ground._identity import make_hag_config
from tests.ground._scene import load_scene, write_synthetic_las


def _policy(**overrides) -> GroundClassificationPolicyConfig:
    payload = {**GroundClassificationPolicyConfig().to_dict(), **overrides}
    payload["policy_digest"] = ""
    return GroundClassificationPolicyConfig.from_dict(payload)


def _flat_scene(tmp_path: Path):
    gx, gy = np.meshgrid(np.arange(0.0, 8.0, 0.5), np.arange(0.0, 8.0, 0.5))
    return load_scene(
        write_synthetic_las(
            tmp_path / "flat.las",
            x=100.0 + gx.ravel(),
            y=200.0 + gy.ravel(),
            z=np.zeros(gx.size),
        )
    )


def test_classified_las_replays_through_unchanged_p_source(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    out = classify_ground(
        scene.reference,
        scene.result,
        scene.configuration,
        make_hag_config(),
        tmp_path / "out",
        _policy(),
    )
    proof = out["publication_proof"]
    assert proof.p_source_replay["status"] == "valid"
    assert proof.classified_source_identity.source_point_count == scene.count
    # The classified LAS must load through unchanged P-SOURCE.
    classified_path = out["classified_las_path"]
    with laspy.open(classified_path) as reader:
        assert int(reader.header.point_count) == scene.count
    # Raw source must be untouched.
    with laspy.open(scene.path) as reader:
        assert int(reader.header.point_count) == scene.count


def test_p_hag_compatibility_recorded_in_proof_only(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    out = classify_ground(
        scene.reference,
        scene.result,
        scene.configuration,
        make_hag_config(cell_size_m=1.0),
        tmp_path / "out",
        _policy(),
    )
    proof = out["publication_proof"]
    hag = proof.p_hag_compatibility
    assert hag["config_digest"] == make_hag_config(cell_size_m=1.0).provider_config_digest
    assert hag["ground_point_count"] >= 1
    # P-HAG config must never enter the decision/policy digest.
    assert "hag" not in proof.policy_digest
    assert proof.policy_digest == _policy().policy_digest


def test_p_hag_config_change_requires_new_proof_but_not_new_decisions(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    out_a = classify_ground(
        scene.reference,
        scene.result,
        scene.configuration,
        make_hag_config(cell_size_m=1.0),
        tmp_path / "out_a",
        _policy(),
    )
    out_b = classify_ground(
        scene.reference,
        scene.result,
        scene.configuration,
        make_hag_config(cell_size_m=3.0),
        tmp_path / "out_b",
        _policy(),
    )
    assert out_a["decision_manifest"].decision_digest == out_b["decision_manifest"].decision_digest
    assert (
        out_a["publication_proof"].p_hag_compatibility["config_digest"]
        != out_b["publication_proof"].p_hag_compatibility["config_digest"]
    )


def test_decision_manifest_binds_source_identity_and_policy(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    out = classify_ground(
        scene.reference,
        scene.result,
        scene.configuration,
        make_hag_config(),
        tmp_path / "out",
        _policy(),
    )
    manifest = out["decision_manifest"]
    manifest.validated_with_policy(_policy())
    assert manifest.source_identity.source_id == scene.result.source_identity.source_id
    assert manifest.decision_digest == out["publication_proof"].decision_digest


def test_raw_source_immutability_and_derivative_distinct_identity(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    raw_bytes = scene.path.read_bytes()
    raw_identity = scene.result.source_identity
    out = classify_ground(
        scene.reference,
        scene.result,
        scene.configuration,
        make_hag_config(),
        tmp_path / "out",
        _policy(),
    )
    assert scene.path.read_bytes() == raw_bytes  # raw never mutated
    classified_identity = out["classified_source_identity"]
    assert classified_identity.source_id != raw_identity.source_id
    assert classified_identity.source_file_sha256 != raw_identity.source_file_sha256
    assert out["completion_manifest"].raw_source_identity.source_id == raw_identity.source_id
    assert (
        out["completion_manifest"].classified_source_identity.source_id
        == classified_identity.source_id
    )


def test_publication_proof_roundtrip_and_p_source_binding(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    out = classify_ground(
        scene.reference,
        scene.result,
        scene.configuration,
        make_hag_config(),
        tmp_path / "out",
        _policy(),
    )
    root = (
        tmp_path
        / "out"
        / "ground-classification"
        / scene.result.source_identity.source_id
        / scene.result.source_identity.source_file_sha256[:16]
    )
    proof_payload = json.loads((root / FGC_PUBLICATION_PROOF_PATH).read_text(encoding="utf-8"))
    from core.contracts.ground_classification import FgcPublicationProof

    proof = FgcPublicationProof.from_dict(proof_payload)
    assert proof == out["publication_proof"]
    assert proof.p_source_replay["classified_source_file_sha256"] == proof.classified_source_sha256
