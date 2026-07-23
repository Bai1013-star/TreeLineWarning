# ruff: noqa: E501
from __future__ import annotations

import numpy as np
import pytest
from core.contracts import (
    VegetationCandidateConfig,
    VegetationCandidateLevel,
    VegetationCandidatePointSet,
    VegetationHagState,
    VegetationManualAction,
    VegetationManualSelectionConfig,
)


def _points() -> VegetationCandidatePointSet:
    count = 2
    return VegetationCandidatePointSet(
        "scene",
        "raw",
        np.zeros((count, 3), np.float32),
        np.zeros(3, np.float64),
        np.array([4, 9], np.int64),
        np.array([3, 8], np.int64),
        np.array([-1, -1], np.int32),
        np.array([VegetationHagState.RELIABLE] * count, np.uint8),
        np.array([0.2, 0.8], np.float32),
        np.array([1.0, 0.7], np.float32),
        np.array([0, 2], np.uint8),
        np.array([0, 2], np.uint8),
        np.array([False, True]),
        np.zeros(count, np.uint16),
        np.zeros(count, np.uint16),
        np.zeros(count, np.uint16),
        np.zeros(count, np.uint8),
        np.array([VegetationManualAction.NONE] * count, np.uint8),
        np.zeros(count, np.uint16),
        np.zeros(count, np.uint16),
    )


def test_v1_enums_and_full_m_contract() -> None:
    points = _points()
    assert points.point_count == 2
    assert points.array_metadata()["xyz_local"]["unit"] == "m"
    assert [int(value) for value in VegetationCandidateLevel] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="mask"):
        VegetationCandidatePointSet(
            **{**points.__dict__, "vegetation_mask": np.array([True, True])}
        )


def test_config_threshold_weight_and_manual_validation() -> None:
    config = VegetationCandidateConfig()
    assert VegetationCandidateConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="score thresholds"):
        VegetationCandidateConfig(low_score=0.8, medium_score=0.6)
    with pytest.raises(ValueError, match="weights"):
        VegetationCandidateConfig(height_weight=0, volumetric_weight=0, density_weight=0)
    with pytest.raises(ValueError, match="confidence weights"):
        VegetationCandidateConfig(geometry_confidence_weight=0.4)
    manual = VegetationManualSelectionConfig(
        include_original_index=(3,), exclude_boxes=(((0, 0, 0), (1, 1, 1)),)
    )
    assert VegetationManualSelectionConfig.from_dict(manual.to_dict()) == manual
    with pytest.raises(ValueError, match="unique"):
        VegetationManualSelectionConfig(include_original_index=(1, 1))
    with pytest.raises(ValueError, match="boxes"):
        VegetationManualSelectionConfig(include_boxes=(((2, 0, 0), (1, 0, 0)),))
