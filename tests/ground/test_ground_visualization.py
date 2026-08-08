"""Bundle D ground visualization review tests (PLAN-FROZEN section 6).

Covers valid-artifact visualization, completion-manifest refusal, corrupt
component/hash refusal, deterministic sampling, per-state caps, finite
diagnostics, no-authority-recomputation / no-modification guarantees, and the
Ground Human Gate machine-status contract (never auto-approved).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from algorithms.ground.classification import classify_ground
from algorithms.ground.visualization import (
    FgcReviewSamplingConfig,
    FgcVisualizationError,
    build_ground_review_summary,
    generate_ground_human_gate_material,
    sample_review_points,
    validate_fgc_publication,
)
from core.contracts.ground_classification import (
    FGC_COMPLETION_MANIFEST_PATH,
    FGC_DECISIONS_DIR,
    GroundClassificationPolicyConfig,
)

from tests.ground._identity import make_hag_config
from tests.ground._scene import load_scene, write_synthetic_las


def _policy() -> GroundClassificationPolicyConfig:
    payload = {**GroundClassificationPolicyConfig().to_dict(), "policy_digest": ""}
    return GroundClassificationPolicyConfig.from_dict(payload)


def _flat_scene(tmp_path: Path):
    gx, gy = np.meshgrid(np.arange(0.0, 12.0, 0.5), np.arange(0.0, 8.0, 0.5))
    return load_scene(
        write_synthetic_las(
            tmp_path / "flat.las",
            x=100.0 + gx.ravel(),
            y=200.0 + gy.ravel(),
            z=np.zeros(gx.size),
        )
    )


def _mixed_scene(tmp_path: Path):
    """A dense plane plus a vertical pole (review) and an isolated tall point
    (non-ground), yielding all three terminal states."""
    gx, gy = np.meshgrid(np.arange(0.0, 8.0, 0.5), np.arange(0.0, 4.0, 0.5))
    xs = list(100.0 + gx.ravel())
    ys = list(200.0 + gy.ravel())
    zs = list(np.zeros(gx.size))
    for level in range(12):
        xs.append(104.0)
        ys.append(202.0)
        zs.append(0.2 * level + 0.5)
    xs.append(109.5)
    ys.append(202.0)
    zs.append(5.0)
    return load_scene(
        write_synthetic_las(
            tmp_path / "mix.las",
            x=np.asarray(xs),
            y=np.asarray(ys),
            z=np.asarray(zs),
        )
    )


def _classify(scene, tmp_path: Path, *, sub: str = "out"):
    return classify_ground(
        scene.reference,
        scene.result,
        scene.configuration,
        make_hag_config(),
        tmp_path / sub,
        _policy(),
    )


def _transaction_root(scene, tmp_path: Path, *, sub: str = "out") -> Path:
    identity = scene.result.source_identity
    return (
        tmp_path
        / sub
        / "ground-classification"
        / identity.source_id
        / identity.source_file_sha256[:16]
    )


def _classify_root(scene, tmp_path: Path, *, sub: str = "out") -> Path:
    """Classify *scene* and return its published transaction root."""
    _classify(scene, tmp_path, sub=sub)
    return _transaction_root(scene, tmp_path, sub=sub)


def _source_state_counts(scene, tmp_path: Path, *, sub: str = "out") -> dict[str, int]:
    out = _classify(scene, tmp_path, sub=sub)
    total = {"non_ground": 0, "confirmed_ground": 0, "review_ground": 0}
    for chunk in out["chunks"]:
        for key in total:
            total[key] += chunk.state_counts[key]
    return total


def _decision_bytes(root: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for chunk_path in sorted((root / FGC_DECISIONS_DIR).iterdir()):
        if chunk_path.is_file():
            result[chunk_path.name] = chunk_path.read_bytes()
    return result


def _classified_las_bytes(root: Path, scene) -> bytes:
    stem = scene.path.stem
    return (root / "classified" / f"{stem}.fgc-v1.las").read_bytes()


# ---------------------------------------------------------------------------
# Validated publication and review material
# ---------------------------------------------------------------------------


def test_valid_publication_validation_passes(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    publication = validate_fgc_publication(root)
    assert publication.source_point_count == scene.count
    assert publication.policy_is_frozen_reviewed is True
    assert publication.classified_las_path.is_file()
    assert len(publication.decision_chunk_paths) >= 1


def test_flat_scene_summary_counts(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    publication = validate_fgc_publication(root)
    sample = sample_review_points(publication)
    summary = build_ground_review_summary(publication, sample)
    counts = summary["counts"]
    assert counts["total_points"] == scene.count
    assert counts["confirmed_ground_count"] == scene.count
    assert counts["confirmed_fraction"] == pytest.approx(1.0)
    assert counts["non_ground_count"] == 0
    assert counts["review_ground_count"] == 0
    assert counts["class_2_published_count"] == scene.count
    assert summary["coverage"]["occupancy_fraction"] == pytest.approx(1.0)


def test_generate_human_gate_material_writes_artifacts(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    review_dir = root / "diagnostics"
    summary = generate_ground_human_gate_material(root, review_dir)
    assert (review_dir / "fgc_ground_review.html").is_file()
    assert (review_dir / "fgc_ground_review_summary.json").is_file()
    assert summary["status"] == "READY_FOR_OWNER_HUMAN_GATE_REVIEW"
    assert summary["machine_validity"] == "PASS"
    assert summary["owner_decision"] == "PENDING"
    payload = json.loads(
        (review_dir / "fgc_ground_review_summary.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "fgc-ground-review-summary-v1"


def test_review_artifacts_contain_expected_paths(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    summary = generate_ground_human_gate_material(root, root / "diagnostics")
    artifacts = summary["review_artifacts"]
    assert artifacts["html_path"].endswith("fgc_ground_review.html")
    assert artifacts["summary_path"].endswith("fgc_ground_review_summary.json")
    assert Path(artifacts["html_path"]).is_file()


# ---------------------------------------------------------------------------
# Refusal: completion manifest and corrupt components
# ---------------------------------------------------------------------------


def test_missing_completion_manifest_refused(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    (root / FGC_COMPLETION_MANIFEST_PATH).unlink()
    with pytest.raises(FgcVisualizationError) as exc:
        validate_fgc_publication(root)
    assert exc.value.code == "COMPLETION_MANIFEST_MISSING"


def test_invalid_completion_manifest_refused(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    (root / FGC_COMPLETION_MANIFEST_PATH).write_text("{ not json", encoding="utf-8")
    with pytest.raises(FgcVisualizationError) as exc:
        validate_fgc_publication(root)
    assert exc.value.code == "COMPLETION_MANIFEST_INVALID"


def test_corrupt_completion_digest_refused(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    manifest_path = root / FGC_COMPLETION_MANIFEST_PATH
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["classified_las_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FgcVisualizationError) as exc:
        validate_fgc_publication(root)
    assert exc.value.code == "COMPLETION_MANIFEST_INVALID"


def test_corrupt_decision_chunk_hash_refused(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    chunk_path = next((root / FGC_DECISIONS_DIR).iterdir())
    data = bytearray(chunk_path.read_bytes())
    data[0] ^= 0xFF
    chunk_path.write_bytes(bytes(data))
    with pytest.raises(FgcVisualizationError) as exc:
        validate_fgc_publication(root)
    assert exc.value.code == "PUBLICATION_HASH_MISMATCH"


def test_corrupt_classified_las_hash_refused(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    classified = _classified_las_bytes(root, scene)
    data = bytearray(classified)
    data[100] ^= 0xFF
    (root / "classified" / f"{scene.path.stem}.fgc-v1.las").write_bytes(bytes(data))
    with pytest.raises(FgcVisualizationError) as exc:
        validate_fgc_publication(root)
    assert exc.value.code == "PUBLICATION_HASH_MISMATCH"


def test_classified_las_alone_is_nonauthoritative(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    only_las = tmp_path / "orphan"
    only_las.mkdir()
    (only_las / "classified").mkdir()
    (only_las / "classified" / f"{scene.path.stem}.fgc-v1.las").write_bytes(
        _classified_las_bytes(root, scene)
    )
    with pytest.raises(FgcVisualizationError) as exc:
        validate_fgc_publication(only_las)
    assert exc.value.code == "COMPLETION_MANIFEST_MISSING"


def test_cross_binding_mismatch_refused(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    manifest_path = root / FGC_COMPLETION_MANIFEST_PATH
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Tamper the decision-manifest digest binding and let the completion digest
    # be recomputed so the manifest itself stays schema-valid.
    payload["decision_manifest_digest"] = "f" * 64
    payload["completion_digest"] = ""
    from core.contracts.ground_classification import FgcCompletionManifest

    tampered = FgcCompletionManifest.from_dict(payload)
    manifest_path.write_text(json.dumps(tampered.to_dict()), encoding="utf-8")
    with pytest.raises(FgcVisualizationError) as exc:
        validate_fgc_publication(root)
    assert exc.value.code == "PUBLICATION_CROSS_BINDING_INVALID"


def test_decision_manifest_structural_invalid_refused(tmp_path: Path) -> None:
    import hashlib

    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    decision_path = root / "fgc_decision_manifest.json"
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    # Remove a required field so the manifest is structurally invalid, then
    # re-bind its SHA in the completion manifest so hash verification passes
    # and the schema refusal is what fires.
    payload.pop("decision_digest", None)
    decision_path.write_text(json.dumps(payload), encoding="utf-8")

    completion_path = root / FGC_COMPLETION_MANIFEST_PATH
    completion_payload = json.loads(completion_path.read_text(encoding="utf-8"))
    completion_payload["decision_manifest_sha256"] = hashlib.sha256(
        decision_path.read_bytes()
    ).hexdigest()
    completion_payload["decision_manifest_size_bytes"] = decision_path.stat().st_size
    completion_payload["completion_digest"] = ""
    from core.contracts.ground_classification import FgcCompletionManifest

    tampered = FgcCompletionManifest.from_dict(completion_payload)
    completion_path.write_text(json.dumps(tampered.to_dict()), encoding="utf-8")
    with pytest.raises(FgcVisualizationError) as exc:
        validate_fgc_publication(root)
    assert exc.value.code == "DECISION_MANIFEST_INVALID"


# ---------------------------------------------------------------------------
# Deterministic bounded sampling
# ---------------------------------------------------------------------------


def test_sampling_is_deterministic(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    publication = validate_fgc_publication(root)
    config = FgcReviewSamplingConfig()
    sample_a = sample_review_points(publication, config)
    sample_b = sample_review_points(publication, config)
    assert [p.original_index for p in sample_a.points] == [
        p.original_index for p in sample_b.points
    ]
    assert sample_a.displayed_total == sample_b.displayed_total


def test_sampling_reports_stride_rule(tmp_path: Path) -> None:
    scene = _mixed_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    publication = validate_fgc_publication(root)
    sample = sample_review_points(publication)
    assert sample.sampling_rule == "stride"
    assert sample.source_state_counts["non_ground"] == 1
    assert sample.displayed_state_counts["non_ground"] == 1
    assert sample.strides["non_ground"] == 1


def test_sampling_respects_per_state_caps(tmp_path: Path) -> None:
    scene = _mixed_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    publication = validate_fgc_publication(root)
    config = FgcReviewSamplingConfig(
        sample_cap_per_state=10,
        non_ground_sample_cap=5,
    )
    sample = sample_review_points(publication, config)
    assert sample.displayed_state_counts["confirmed_ground"] <= 10
    assert sample.displayed_state_counts["review_ground"] <= 10
    assert sample.displayed_state_counts["non_ground"] <= 5
    assert sample.displayed_total <= config.max_displayed_points


def test_sampling_stride_matches_documented_rule(tmp_path: Path) -> None:
    scene = _mixed_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    publication = validate_fgc_publication(root)
    config = FgcReviewSamplingConfig(
        sample_cap_per_state=7,
        non_ground_sample_cap=4,
    )
    sample = sample_review_points(publication, config)
    for state in ("confirmed_ground", "review_ground", "non_ground"):
        source_count = sample.source_state_counts[state]
        cap = (
            config.non_ground_sample_cap
            if state == "non_ground"
            else config.sample_cap_per_state
        )
        stride = max(1, math.ceil(source_count / cap))
        expected = math.ceil(source_count / stride)
        assert sample.strides[state] == stride
        assert sample.displayed_state_counts[state] == expected


def test_sampling_all_states_present_and_sorted(tmp_path: Path) -> None:
    scene = _mixed_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    publication = validate_fgc_publication(root)
    sample = sample_review_points(publication)
    for state in ("confirmed_ground", "review_ground", "non_ground"):
        points = sample.points_for(state)
        assert points, f"expected displayed points for {state}"
        indices = [point.original_index for point in points]
        assert indices == sorted(indices)
    # Every point belongs to exactly one state and carries a finite position.
    assert all(
        math.isfinite(point.x) and math.isfinite(point.z)
        for point in sample.points
    )


def test_flat_scene_displays_all_points(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    publication = validate_fgc_publication(root)
    sample = sample_review_points(publication)
    # 576 < cap, so stride 1 displays every confirmed point.
    assert sample.displayed_total == scene.count
    assert sample.source_state_counts["confirmed_ground"] == scene.count


def test_sampling_config_rejects_invalid(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        FgcReviewSamplingConfig(sample_cap_per_state=0)
    with pytest.raises(ValueError):
        FgcReviewSamplingConfig(
            sample_cap_per_state=40_000,
            non_ground_sample_cap=30_000,
            max_displayed_points=60_000,
        )
    with pytest.raises(ValueError):
        FgcReviewSamplingConfig(sampling_rule="random")
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    publication = validate_fgc_publication(root)
    with pytest.raises(FgcVisualizationError) as exc:
        sample_review_points(publication, config="not-a-config")
    assert exc.value.code == "VISUALIZATION_CONFIG_INVALID"


# ---------------------------------------------------------------------------
# Finite diagnostics and no-modification guarantees
# ---------------------------------------------------------------------------


def test_diagnostics_are_finite_and_json_safe(tmp_path: Path) -> None:
    scene = _mixed_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    publication = validate_fgc_publication(root)
    sample = sample_review_points(publication)
    summary = build_ground_review_summary(publication, sample)
    encoded = json.dumps(summary, allow_nan=False)
    payload = json.loads(encoded)
    for count in payload["counts"].values():
        if isinstance(count, float):
            assert math.isfinite(count)
    assert payload["sampling"]["source_count"] == scene.count
    assert payload["identity"]["classified_source_sha256"] == (
        publication.completion_manifest.classified_las_sha256
    )


def test_review_does_not_modify_any_published_component(tmp_path: Path) -> None:
    scene = _mixed_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    decision_before = _decision_bytes(root)
    las_before = _classified_las_bytes(root, scene)
    completion_before = (root / FGC_COMPLETION_MANIFEST_PATH).read_bytes()
    manifest_before = (root / "fgc_decision_manifest.json").read_bytes()
    proof_before = (root / "fgc_publication_proof.json").read_bytes()

    generate_ground_human_gate_material(root, root / "diagnostics")

    assert _decision_bytes(root) == decision_before
    assert _classified_las_bytes(root, scene) == las_before
    assert (root / FGC_COMPLETION_MANIFEST_PATH).read_bytes() == completion_before
    assert (root / "fgc_decision_manifest.json").read_bytes() == manifest_before
    assert (root / "fgc_publication_proof.json").read_bytes() == proof_before


def test_review_only_writes_into_review_dir(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    review_dir = tmp_path / "human_gate"
    generate_ground_human_gate_material(root, review_dir)
    written = {p.name for p in review_dir.iterdir()}
    assert written == {"fgc_ground_review.html", "fgc_ground_review_summary.json"}


def test_human_gate_not_auto_approved(tmp_path: Path) -> None:
    scene = _flat_scene(tmp_path)
    root = _classify_root(scene, tmp_path)
    summary = generate_ground_human_gate_material(root, root / "diagnostics")
    assert summary["owner_decision"] == "PENDING"
    assert summary["status"] == "READY_FOR_OWNER_HUMAN_GATE_REVIEW"
    assert "PASS_READY_FOR_FRV_RUNTIME" not in json.dumps(summary)
    assert summary["publication"]["p_source_replay_status"] == "valid"
