"""Focused tests for deterministic crown-likelihood ROI ranking."""

from __future__ import annotations

import numpy as np
from algorithms.vegetation.fast_multi_tree_roi import rank_multi_tree_roi_candidates


def _candidate_arrays() -> dict[str, np.ndarray]:
    generator = np.random.default_rng(20260810)
    centers = np.asarray(((2.0, 2.0), (5.0, 5.0), (8.0, 7.0), (3.0, 11.0), (8.0, 14.0)))
    xy_parts: list[np.ndarray] = []
    hag_parts: list[np.ndarray] = []
    for center in centers:
        xy = generator.normal(center, 0.35, size=(90, 2))
        distance = np.hypot(xy[:, 0] - center[0], xy[:, 1] - center[1])
        xy_parts.append(xy)
        hag_parts.append(8.0 - 4.0 * distance + generator.normal(0.0, 0.03, size=len(xy)))
    xy = np.vstack(xy_parts)
    hag = np.concatenate(hag_parts).astype(np.float32)
    return {
        "xyz_absolute": np.column_stack((xy, hag)).astype(np.float64),
        "hag_m": hag,
        "raw_weight": np.ones(len(xy), dtype=np.uint32),
        "classification": np.ones(len(xy), dtype=np.uint8),
        "linearity": np.full(len(xy), 0.20, dtype=np.float32),
        "planarity": np.full(len(xy), 0.20, dtype=np.float32),
        "scattering": np.full(len(xy), 0.45, dtype=np.float32),
    }


def test_ranked_candidate_prefers_multiple_compact_crowns() -> None:
    candidates, evaluated = rank_multi_tree_roi_candidates(
        **_candidate_arrays(),
        window_width_m=12.0,
        window_height_m=18.0,
        stride_m=2.0,
    )
    assert evaluated > 0
    assert candidates
    candidate = candidates[0]
    assert candidate.automatic_gate_pass
    assert candidate.crown_peak_count >= 4
    assert candidate.rigid_geometry_fraction == 0.0
    assert candidate.elongation_ratio < 3.5


def test_linear_low_scattering_geometry_does_not_produce_candidate() -> None:
    x = np.linspace(0.0, 24.0, 300)
    arrays = {
        "xyz_absolute": np.column_stack((x, np.zeros_like(x), np.full_like(x, 6.0))),
        "hag_m": np.full(len(x), 6.0, dtype=np.float32),
        "raw_weight": np.ones(len(x), dtype=np.uint32),
        "classification": np.ones(len(x), dtype=np.uint8),
        "linearity": np.full(len(x), 0.95, dtype=np.float32),
        "planarity": np.full(len(x), 0.90, dtype=np.float32),
        "scattering": np.full(len(x), 0.01, dtype=np.float32),
    }
    candidates, evaluated = rank_multi_tree_roi_candidates(**arrays)
    assert evaluated == 0
    assert candidates == []
