from __future__ import annotations

import numpy as np
import pytest
from core.contracts import (
    WireModelMethod,
    WireModelSet,
    WireModelStatus,
    WireModelSummary,
    WireReconstructionConfig,
)


def _set(
    *, failed: bool = False, invalid_failed_ready: bool = False, wire_ids: np.ndarray | None = None
) -> WireModelSet:
    wire_ids = np.array([3, 9], dtype=np.int32) if wire_ids is None else wire_ids
    status = np.array(
        [
            int(WireModelStatus.CONFIRMED_MODEL),
            int(WireModelStatus.FAILED if failed else WireModelStatus.FALLBACK_MODEL),
        ],
        dtype=np.uint8,
    )
    return WireModelSet(
        wire_ids,
        np.array([4, 10], dtype=np.int32),
        status,
        np.array([0, 4], dtype=np.uint8),
        np.zeros(2, dtype=np.uint16),
        np.array([True, False], dtype=bool),
        np.array([True, False], dtype=bool),
        np.array([0, 1, 2], dtype=np.int64),
        wire_ids.copy(),
        np.array([0, 0], dtype=np.int32),
        np.array([0, 3], dtype=np.uint8),
        np.zeros(2, dtype=np.uint16),
        np.array([True, invalid_failed_ready or not failed], dtype=bool),
        np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32),
        np.array([0, 1, 2, 3], dtype=np.float32),
        np.array([0, 2, 4], dtype=np.int64),
    )


def test_model_set_accepts_preserved_non_contiguous_wire_ids() -> None:
    values = _set()
    assert values.wire_id.tolist() == [3, 9]


def test_failed_wire_cannot_contain_ready_segment() -> None:
    with pytest.raises(ValueError, match="FAILED"):
        _set(failed=True, invalid_failed_ready=True)


def test_duplicate_or_unsorted_wire_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        _set(wire_ids=np.array([3, 3], dtype=np.int32))
    with pytest.raises(ValueError, match="ordered"):
        _set(wire_ids=np.array([9, 3], dtype=np.int32))


def test_summary_and_config_round_trip() -> None:
    config = WireReconstructionConfig()
    assert WireReconstructionConfig.from_dict(config.to_dict()) == config
    summary = WireModelSummary(
        7,
        8,
        WireModelStatus.CONFIRMED_MODEL,
        WireModelMethod.CUBIC_BSPLINE,
        0,
        True,
        True,
        1,
        10,
        8,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        (0, 0, 0),
        (1, 0, 0),
        (1, 0, 0),
    )
    assert WireModelSummary.from_dict(summary.to_dict()) == summary


def test_config_rejects_invalid_anti_oscillation_thresholds() -> None:
    with pytest.raises(ValueError, match="maximum_turn_angle_rad"):
        WireReconstructionConfig(maximum_turn_angle_rad=0.0)
    with pytest.raises(ValueError, match="maximum_sample_step_multiplier"):
        WireReconstructionConfig(maximum_sample_step_multiplier=0.0)
    with pytest.raises(ValueError, match="maximum_arc_length_inflation_ratio"):
        WireReconstructionConfig(maximum_arc_length_inflation_ratio=0.0)
    with pytest.raises(ValueError, match="unique and increasing"):
        WireReconstructionConfig(smoothing_candidates=(0.002, 0.0, 0.01))
