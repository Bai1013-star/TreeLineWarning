# ruff: noqa: E501
from __future__ import annotations

import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from algorithms.vegetation import candidates
from core.contracts import (
    CancellationToken,
    FeatureQualityFlag,
    GroundQualityFlag,
    VegetationCandidateConfig,
    VegetationCandidateLevel,
    VegetationEvidenceFlag,
    VegetationHagState,
    VegetationManualAction,
    VegetationManualProvenanceFlag,
    VegetationManualSelectionConfig,
    VegetationReasonFlag,
)


def _data(count: int = 8) -> dict[str, dict[str, np.ndarray]]:
    xyz = np.column_stack(
        (
            np.arange(count, dtype=np.float32),
            np.zeros(count, np.float32),
            np.linspace(1, 8, count, dtype=np.float32),
        )
    )
    indexes = np.arange(100, 100 + count, dtype=np.int64)
    return {
        "a1": {
            "source_id": np.asarray("scene_simple"),
            "manifest_source_id": np.asarray("raw"),
            "xyz_local": xyz,
            "origin_xyz": np.zeros(3, np.float64),
            "original_index": indexes,
            "sample_index": np.arange(count, dtype=np.int64),
        },
        "a2a": {
            "linearity": np.full(count, 0.1, np.float32),
            "planarity": np.full(count, 0.1, np.float32),
            "scattering": np.full(count, 0.2, np.float32),
            "verticality": np.zeros(count, np.float32),
            "local_density": np.linspace(10, 100, count, dtype=np.float32),
            "height_m": xyz[:, 2].copy(),
            "neighbor_count": np.full(count, 32, np.uint16),
            "quality_flags": np.zeros(count, np.uint16),
        },
        "a2b": {
            "height_agl_m": np.full(count, 4.0, np.float32),
            "raw_height_agl_m": np.full(count, 4.0, np.float32),
            "ground_distance_m": np.zeros(count, np.float32),
            "quality_flags": np.zeros(count, np.uint16),
        },
        "a4": {
            "wire_id_by_point": np.full(count, -1, np.int32),
            "track_id_by_point": np.full(count, -1, np.int32),
        },
    }


def _score(
    data: dict[str, dict[str, np.ndarray]],
    manual: VegetationManualSelectionConfig | None = None,
    config: VegetationCandidateConfig | None = None,
):
    return candidates._score(
        data,
        config or VegetationCandidateConfig(),
        manual or VegetationManualSelectionConfig(),
        0.0,
        None,
    )


def test_reliable_near_ground_and_confirmed_wire_are_hard_excluded() -> None:
    data = _data()
    data["a2b"]["height_agl_m"][0] = 0.1
    data["a2b"]["raw_height_agl_m"][0] = 0.1
    data["a4"]["wire_id_by_point"][1] = 4
    arrays, _points, _stats, _counts, _formulas = _score(data)
    assert arrays["vegetation_level"][0] == VegetationCandidateLevel.REJECTED
    assert (
        arrays["vegetation_reason_flags"][0]
        & VegetationReasonFlag.RELIABLE_NEAR_GROUND_HARD_EXCLUSION
    )
    assert arrays["vegetation_level"][1] == VegetationCandidateLevel.REJECTED
    assert arrays["vegetation_reason_flags"][1] & VegetationReasonFlag.CONFIRMED_WIRE_HARD_EXCLUSION


def test_uncertain_hag_lowers_confidence_not_likelihood_or_hard_excludes() -> None:
    data = _data()
    data["a2a"]["local_density"][:] = 50.0
    data["a2b"]["quality_flags"][0] = int(GroundQualityFlag.HIGH_LOCAL_GROUND_VARIATION)
    arrays, _points, _stats, _counts, _formulas = _score(data)
    assert arrays["hag_state"][0] == VegetationHagState.UNCERTAIN
    assert arrays["vegetation_confidence"][0] < arrays["vegetation_confidence"][1]
    assert arrays["vegetation_rule_score"][0] == pytest.approx(arrays["vegetation_rule_score"][1])
    assert arrays["vegetation_mask"][0]


def test_invalid_hag_uses_bounded_height_fallback_and_strong_negative_does_not_exclude() -> None:
    data = _data()
    data["a2b"]["quality_flags"][0] = int(GroundQualityFlag.INVALID_GROUND_ESTIMATE)
    data["a2b"]["height_agl_m"][0] = np.nan
    data["a2b"]["raw_height_agl_m"][0] = np.nan
    data["a2b"]["raw_height_agl_m"][1] = -1.0
    arrays, _points, _stats, _counts, _formulas = _score(data)
    assert arrays["hag_state"][0] == VegetationHagState.INVALID_OR_UNAVAILABLE
    assert arrays["vegetation_evidence_flags"][0] & VegetationEvidenceFlag.HEIGHT_FALLBACK_USED
    assert arrays["hag_state"][1] == VegetationHagState.STRONGLY_NEGATIVE
    assert (
        not arrays["vegetation_reason_flags"][1]
        & VegetationReasonFlag.RELIABLE_NEAR_GROUND_HARD_EXCLUSION
    )


def test_manual_conflict_and_wire_blocked_preserve_rule_arrays() -> None:
    data = _data()
    data["a4"]["wire_id_by_point"][0] = 3
    before, *_ = _score(data)
    manual = VegetationManualSelectionConfig(
        include_original_index=(100, 101), exclude_original_index=(101,)
    )
    arrays, *_ = _score(data, manual)
    assert arrays["manual_action"][0] == VegetationManualAction.WIRE_BLOCKED
    assert arrays["manual_provenance_flags"][0] & VegetationManualProvenanceFlag.MANUAL_WIRE_BLOCKED
    assert arrays["manual_action"][1] == VegetationManualAction.EXCLUDE
    assert arrays["manual_provenance_flags"][1] & VegetationManualProvenanceFlag.MANUAL_CONFLICT
    assert arrays["vegetation_rule_score"][0] == before["vegetation_rule_score"][0]


def test_score_monotonicity_penalty_and_optional_attribute_neutrality() -> None:
    low = _data()
    high = _data()
    low["a2a"]["scattering"][:] = 0.02
    high["a2a"]["scattering"][:] = 0.20
    low_arrays, *_ = _score(low)
    high_arrays, *_ = _score(high)
    assert np.all(high_arrays["vegetation_rule_score"] >= low_arrays["vegetation_rule_score"])
    planar = _data()
    planar["a2a"]["planarity"][:] = 0.9
    planar_arrays, *_ = _score(planar)
    base_arrays, *_ = _score(_data())
    assert np.all(planar_arrays["vegetation_rule_score"] <= base_arrays["vegetation_rule_score"])
    rgb = _data()
    rgb["a1"]["rgb"] = np.full((8, 3), 65000, np.uint16)
    rgb["a1"]["intensity"] = np.full(8, 255, np.uint16)
    rgb_arrays, *_ = _score(rgb)
    assert np.array_equal(rgb_arrays["vegetation_level"], base_arrays["vegetation_level"])


def test_density_invalid_and_cancellation() -> None:
    data = _data()
    data["a2a"]["quality_flags"][0] = int(FeatureQualityFlag.DENSITY_UNDEFINED)
    arrays, *_ = _score(data)
    assert arrays["density_contribution"][0] == 0
    token = CancellationToken()
    token.cancel()
    with pytest.raises(candidates.VegetationCandidateError):
        candidates._check_cancelled(token, "test")


def test_component_boundaries_and_confidence_level_caps() -> None:
    cfg = VegetationCandidateConfig()
    values = candidates._clip(np.array([-1.0, 0.0, 0.5, 2.0]))
    assert values.tolist() == [0.0, 0.0, 0.5, 1.0]
    score = np.array([cfg.high_score, cfg.high_score, cfg.medium_score], np.float32)
    confidence = np.array([0.8, 0.6, 0.2], np.float32)
    levels = candidates._levels(score, confidence, cfg)
    assert levels.tolist() == [
        VegetationCandidateLevel.HIGH,
        VegetationCandidateLevel.MEDIUM,
        VegetationCandidateLevel.LOW,
    ]


def test_relocated_standalone_output_loads(tmp_path: Path) -> None:
    data = _data()
    arrays, points, stats, counts, formulas = _score(data)
    source = tmp_path / "project"
    output = source / "outputs" / "vegetation_candidates"
    output.mkdir(parents=True)
    npz, js = (
        output / "scene_simple_vegetation_candidates.npz",
        output / "scene_simple_vegetation_candidates.json",
    )
    paths = {
        name: source / f"{name}.bin"
        for name in (
            "a1_npz",
            "a1_json",
            "a2a_npz",
            "a2a_json",
            "a2b_npz",
            "a2b_json",
            "a4_npz",
            "a4_json",
        )
    }
    hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in paths}
    candidates._publish(
        arrays,
        points,
        npz,
        js,
        VegetationCandidateConfig(),
        VegetationManualSelectionConfig(),
        paths,
        hashes,
        stats,
        counts,
        formulas,
        source,
    )
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    shutil.copy2(npz, relocated / npz.name)
    shutil.copy2(js, relocated / js.name)
    result, loaded = candidates.load_vegetation_candidates(relocated / js.name)
    assert result.output_npz_filename == npz.name
    assert len(loaded["vegetation_mask"]) == len(data["a1"]["original_index"])


def test_invalid_geometry_is_hard_excluded_and_include_box_is_deterministic() -> None:
    data = _data()
    data["a2a"]["quality_flags"][0] = int(FeatureQualityFlag.NUMERIC_INVALID)
    manual = VegetationManualSelectionConfig(include_boxes=(((1.0, -1.0, 0.0), (1.0, 1.0, 10.0)),))
    arrays, *_ = _score(data, manual)
    assert arrays["vegetation_level"][0] == VegetationCandidateLevel.REJECTED
    assert arrays["manual_action"][1] == VegetationManualAction.INCLUDE
    assert arrays["vegetation_mask"][1]


def test_spatially_scaled_equivalent_hag_configuration_is_stable() -> None:
    baseline, *_ = _score(_data())
    for scale in (0.5, 2.0):
        data = {
            stage: {name: value.copy() for name, value in values.items()}
            for stage, values in _data().items()
        }
        data["a1"]["xyz_local"] *= scale
        data["a2a"]["height_m"] *= scale
        for name in ("height_agl_m", "raw_height_agl_m", "ground_distance_m"):
            data["a2b"][name] *= scale
        config = replace(
            VegetationCandidateConfig(),
            near_ground_hag_m=0.30 * scale,
            height_full_hag_m=3.00 * scale,
            strong_negative_raw_hag_m=0.30 * scale,
            ground_support_distance_m=1.50 * scale,
        )
        scaled, *_ = _score(data, config=config)
        assert np.allclose(scaled["vegetation_rule_score"], baseline["vegetation_rule_score"])
        assert np.allclose(scaled["vegetation_confidence"], baseline["vegetation_confidence"])
