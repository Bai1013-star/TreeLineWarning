from __future__ import annotations

import numpy as np
import pytest
from core.contracts import (
    LocalScaleState,
    VegetationClusterStatus,
    VegetationInstanceConfig,
    VegetationInstancePointSet,
    VegetationManualActionRecord,
    VegetationManualLog,
)


def _points() -> VegetationInstancePointSet:
    n = 2
    return VegetationInstancePointSet(
        np.array([4, 9], np.int64),
        np.array([3, 8], np.int64),
        np.zeros((n, 3), np.float32),
        np.full(n, -1, np.int32),
        np.array([0.7, 0.8], np.float32),
        np.array([0.8, 0.8], np.float32),
        np.array([2, 3], np.uint8),
        np.array([2, 3], np.uint8),
        np.zeros(n, np.uint8),
        np.array([0.04, 0.05], np.float32),
        np.array([0.04, 0.05], np.float32),
        np.full(n, LocalScaleState.LOCAL_SCALE_VALID, np.uint8),
        np.array([0, 0], np.int32),
        np.array([0, 0], np.int32),
        np.full(n, VegetationClusterStatus.CONFIRMED, np.uint8),
        np.ones(n, bool),
        np.ones(n, np.float32),
        np.zeros(n, np.uint16),
        np.zeros(n, np.uint16),
        np.zeros(n, np.uint16),
    )


def test_full_m_semantics_and_confirmed_readiness() -> None:
    points = _points()
    assert points.xyz_local.shape == (2, 3)
    assert points.distance_ready_by_point.all()
    with pytest.raises(ValueError, match="only confirmed"):
        VegetationInstancePointSet(
            **{
                **points.__dict__,
                "cluster_status_by_point": np.full(
                    2, VegetationClusterStatus.REVIEW_REQUIRED, np.uint8
                ),
            }
        )


def test_config_support_radius_and_validation() -> None:
    cfg = VegetationInstanceConfig()
    assert cfg.local_scale_support_radius_m == pytest.approx(0.3)
    assert VegetationInstanceConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="local scale"):
        VegetationInstanceConfig(local_scale_lower_m=0.2, local_scale_upper_m=0.1)


def test_manual_log_requires_stable_identity_when_used() -> None:
    action = VegetationManualActionRecord("confirm_cluster", cluster_ids=(1,))
    with pytest.raises(ValueError, match="SHA"):
        VegetationManualLog(source_id="scene", manifest_source_id="raw", actions=(action,))
    digest = "a" * 64
    log = VegetationManualLog(digest, digest, digest, "scene", "raw", (action,))
    assert VegetationManualLog.from_dict(log.to_dict()) == log
