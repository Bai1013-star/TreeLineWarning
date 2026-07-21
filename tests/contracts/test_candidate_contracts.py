from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from core.contracts import (
    CandidateEvidenceSet,
    CandidateLevel,
    CandidatePointSet,
    CandidateReasonFlag,
    CandidateSource,
    ManualSelectionConfig,
    WireCandidateConfig,
    WireCandidateQualityFlag,
    WireCandidateResult,
    WireCandidateStatistics,
)


def _points() -> CandidatePointSet:
    count = 2
    return CandidatePointSet(
        source_id="scene_simple",
        manifest_source_id="raw_scene",
        xyz_local=np.zeros((count, 3), dtype=np.float32),
        origin_xyz=np.zeros(3, dtype=np.float64),
        original_index=np.array([10, 11], dtype=np.int64),
        sample_index=np.array([0, 1], dtype=np.int64),
        principal_direction=np.tile(np.array([[1, 0, 0]], dtype=np.float32), (count, 1)),
        rule_score=np.array([0.8, 0.0], dtype=np.float32),
        candidate_score=np.array([0.8, 0.0], dtype=np.float32),
        candidate_mask=np.array([True, False]),
        candidate_level=np.array([CandidateLevel.HIGH, CandidateLevel.NONE], dtype=np.uint8),
        candidate_source=np.array([CandidateSource.RULE, CandidateSource.NONE], dtype=np.uint8),
        evidence_flags=np.zeros(count, dtype=np.uint16),
        quality_flags=np.zeros(count, dtype=np.uint16),
        geometry_quality_flags=np.zeros(count, dtype=np.uint16),
        ground_quality_flags=np.zeros(count, dtype=np.uint16),
    )


def test_candidate_enums_are_stable() -> None:
    assert int(CandidateReasonFlag.HEIGHT_FALLBACK_USED) == 128
    assert int(WireCandidateQualityFlag.INVALID_INPUT_FEATURE) == 128
    assert [int(value) for value in CandidateLevel] == [0, 1, 2, 3]
    assert [int(value) for value in CandidateSource] == [0, 1, 2, 3]


def test_config_and_manual_json_round_trip_and_validation() -> None:
    config = WireCandidateConfig()
    assert WireCandidateConfig.from_dict(config.to_dict()) == config
    manual = ManualSelectionConfig(
        include_original_index=(3,),
        exclude_original_index=(4,),
        include_boxes=(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),),
    )
    assert ManualSelectionConfig.from_dict(manual.to_dict()) == manual
    with pytest.raises(ValueError, match="unique"):
        ManualSelectionConfig(include_original_index=(1, 1))
    with pytest.raises(ValueError, match="threshold ordering"):
        WireCandidateConfig(linearity_guard=0.9, linearity_full=0.8)


def test_evidence_and_point_contracts_require_aligned_finite_arrays() -> None:
    points = _points()
    assert points.point_count == 2
    assert points.array_metadata()["candidate_score"]["unit"] == "unitless"
    evidence = CandidateEvidenceSet(
        linearity_score=np.ones(2, dtype=np.float32),
        scattering_score=np.ones(2, dtype=np.float32),
        planarity_score=np.ones(2, dtype=np.float32),
        height_score=np.ones(2, dtype=np.float32),
        density_score=np.ones(2, dtype=np.float32),
        quality_penalty=np.zeros(2, dtype=np.float32),
        geometry_guard_mask=np.ones(2, dtype=bool),
    )
    assert evidence.point_count == 2
    with pytest.raises(ValueError, match="unique"):
        CandidatePointSet(
            **{
                **points.__dict__,
                "sample_index": np.array([0, 0], dtype=np.int64),
            }
        )


def test_result_json_round_trip_converts_paths_safely() -> None:
    statistics = WireCandidateStatistics(
        input_point_count=2,
        candidate_count=1,
        low_count=0,
        medium_count=0,
        high_count=1,
        rule_count=1,
        manual_count=0,
        hybrid_count=0,
        hag_normal_count=2,
        hag_mixed_count=0,
        hag_fallback_count=0,
        elapsed_seconds=0.0,
    )
    result = WireCandidateResult(
        source_id="scene_simple",
        manifest_source_id="raw_scene",
        input_paths={"a1_npz": Path("in.npz")},
        input_sha256={"a1_npz": "a" * 64},
        output_npz_path=Path("out.npz"),
        output_json_path=Path("out.json"),
        output_npz_sha256="b" * 64,
        output_npz_bytes=1,
        origin_xyz=(0.0, 0.0, 0.0),
        config=WireCandidateConfig(),
        manual_selection=ManualSelectionConfig(),
        mode="rule",
        statistics=statistics,
        arrays={},
        evidence_counts={},
        quality_counts={},
        formulas={},
        warnings=(),
        algorithm_version="test",
        created_at="2026-01-01T00:00:00+00:00",
    )
    restored = WireCandidateResult.from_dict(result.to_dict())
    assert restored.output_npz_path == Path("out.npz")
    assert restored.statistics.candidate_count == 1
