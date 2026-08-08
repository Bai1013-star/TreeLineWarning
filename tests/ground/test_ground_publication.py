"""FGC atomic publication, completion-manifest discovery, and collision tests.

Authority: docs/full-resolution-ground-classification-design.md sections 12
and 16, and the frozen Bundle C plan test gate (atomic failure, corrupt
completion manifest, orphan LAS non-authority, publication collision).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from algorithms.ground.classification import (
    FgcExecutionError,
    classify_ground,
)
from core.contracts.ground_classification import (
    FGC_COMPLETION_MANIFEST_PATH,
    FGC_DECISION_MANIFEST_PATH,
    FGC_PUBLICATION_PROOF_PATH,
    FgcCompletionManifest,
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


def _transaction_root(output_root: Path, scene) -> Path:
    identity = scene.result.source_identity
    return (
        output_root
        / "ground-classification"
        / identity.source_id
        / identity.source_file_sha256[:16]
    )


def test_publication_tree_and_completion_last(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    classify_ground(
        scene.reference,
        scene.result,
        scene.configuration,
        make_hag_config(),
        tmp_path / "out",
        _policy(),
    )
    root = _transaction_root(tmp_path / "out", scene)
    assert (root / FGC_COMPLETION_MANIFEST_PATH).exists()
    assert (root / FGC_DECISION_MANIFEST_PATH).exists()
    assert (root / FGC_PUBLICATION_PROOF_PATH).exists()
    assert (root / "classified").is_dir()
    classified_files = list((root / "classified").glob("*.las"))
    assert len(classified_files) == 1
    # Scratch must not leak into the published tree.
    assert not (root / "_scratch").exists()
    assert (root / "decisions").is_dir()


def test_completion_manifest_roundtrip_and_bindings(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    classify_ground(
        scene.reference,
        scene.result,
        scene.configuration,
        make_hag_config(),
        tmp_path / "out",
        _policy(),
    )
    root = _transaction_root(tmp_path / "out", scene)
    manifest_file = root / FGC_COMPLETION_MANIFEST_PATH
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest = FgcCompletionManifest.from_dict(payload)
    # Completion digest must equal the canonical payload digest without itself.
    digest_payload = manifest.to_digest_dict()
    from core.contracts.frv import frv_digest

    assert manifest.completion_digest == frv_digest(digest_payload)
    # Bound component hashes must match the files on disk.
    assert manifest.classified_las_sha256 == _file_sha(root / manifest.classified_las_path)
    assert manifest.decision_manifest_sha256 == _file_sha(root / manifest.decision_manifest_path)
    assert manifest.publication_proof_sha256 == _file_sha(root / manifest.publication_proof_path)


def test_completion_manifest_verify_bindings(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    classify_ground(
        scene.reference,
        scene.result,
        scene.configuration,
        make_hag_config(),
        tmp_path / "out",
        _policy(),
    )
    root = _transaction_root(tmp_path / "out", scene)
    manifest_file = root / FGC_COMPLETION_MANIFEST_PATH
    manifest = FgcCompletionManifest.from_dict(
        json.loads(manifest_file.read_text(encoding="utf-8"))
    )
    manifest.verify_bindings(root)  # no exception


def test_completion_manifest_verify_bindings_detects_tamper(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    classify_ground(
        scene.reference,
        scene.result,
        scene.configuration,
        make_hag_config(),
        tmp_path / "out",
        _policy(),
    )
    root = _transaction_root(tmp_path / "out", scene)
    manifest = FgcCompletionManifest.from_dict(
        json.loads((root / FGC_COMPLETION_MANIFEST_PATH).read_text(encoding="utf-8"))
    )
    # Mutate a bound decision chunk byte; bindings verification must fail.
    chunk = root / "decisions" / "chunk_000000.bin"
    chunk.write_bytes(chunk.read_bytes()[:-1] + b"\x00")
    with pytest.raises(ValueError, match="SHA-256"):
        manifest.verify_bindings(root)


def test_corrupt_completion_manifest_rejected(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    classify_ground(
        scene.reference,
        scene.result,
        scene.configuration,
        make_hag_config(),
        tmp_path / "out",
        _policy(),
    )
    root = _transaction_root(tmp_path / "out", scene)
    manifest_file = root / FGC_COMPLETION_MANIFEST_PATH
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    payload["completion_digest"] = "0" * 64
    manifest_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="completion_digest"):
        FgcCompletionManifest.from_dict(payload)


def test_orphan_las_is_nonauthoritative_without_manifest(tmp_path: Path) -> None:
    # A classified LAS without a valid completion manifest is non-authoritative.
    scene = _flat_scene(tmp_path)
    root_out = tmp_path / "orphan"
    classify_ground(
        scene.reference, scene.result, scene.configuration, make_hag_config(), root_out, _policy()
    )
    root = _transaction_root(root_out, scene)
    # Remove the completion manifest; the LAS remains but is non-authoritative.
    (root / FGC_COMPLETION_MANIFEST_PATH).unlink()
    assert (root / "classified").exists()
    assert not (root / FGC_COMPLETION_MANIFEST_PATH).exists()


def test_publication_collision_rejected(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    out_root = tmp_path / "out"
    classify_ground(
        scene.reference, scene.result, scene.configuration, make_hag_config(), out_root, _policy()
    )
    # A second run over the same namespace must fail closed on collision.
    with pytest.raises(FgcExecutionError, match="already contains published components"):
        classify_ground(
            scene.reference,
            scene.result,
            scene.configuration,
            make_hag_config(),
            out_root,
            _policy(),
        )


def test_p_hag_config_change_does_not_change_decisions(tmp_path: Path) -> None:
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
        make_hag_config(cell_size_m=2.0),
        tmp_path / "out_b",
        _policy(),
    )
    assert out_a["decision_manifest"].decision_digest == out_b["decision_manifest"].decision_digest
    assert (
        out_a["publication_proof"].p_hag_compatibility["config_digest"]
        != out_b["publication_proof"].p_hag_compatibility["config_digest"]
    )


def _file_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
