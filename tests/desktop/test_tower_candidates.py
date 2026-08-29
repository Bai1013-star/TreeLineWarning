from __future__ import annotations

import numpy as np
import pytest
from desktop.tower_candidates import rank_tower_candidates


def _tower_cloud(
    center: tuple[float, float],
    *,
    height: float,
    count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = np.linspace(0.0, height, count)
    x = center[0] + rng.normal(0.0, 0.45, count)
    y = center[1] + rng.normal(0.0, 0.45, count)
    return np.column_stack((x, y, z))


def test_nearby_tall_compact_structure_ranks_first() -> None:
    near = _tower_cloud((1.5, 0.5), height=24.0, count=480, seed=1)
    far = _tower_cloud((12.0, 8.0), height=21.0, count=420, seed=2)
    ground_x = np.linspace(-15.0, 15.0, 300)
    ground = np.column_stack((ground_x, np.sin(ground_x), np.zeros_like(ground_x)))
    candidates = rank_tower_candidates(
        np.vstack((near, far, ground)),
        (0.0, 0.0),
        maximum_candidates=3,
    )

    assert 1 <= len(candidates) <= 3
    assert candidates[0].distance_to_seed_m < 4.0
    assert candidates[0].height_m > 20.0
    assert candidates[0].vertical_continuity >= 0.75
    assert candidates[0].authority == "PRESENTATION_ASSISTANCE_ONLY"


def test_candidate_ranking_deduplicates_nearby_grid_peaks() -> None:
    cloud = _tower_cloud((0.0, 0.0), height=25.0, count=800, seed=3)
    candidates = rank_tower_candidates(
        cloud,
        (0.0, 0.0),
        maximum_candidates=3,
        grid_size_m=1.0,
        merge_distance_m=4.0,
    )
    assert len(candidates) == 1
    assert candidates[0].candidate_id == "candidate-1"


def test_flat_or_tiny_cloud_returns_no_candidates() -> None:
    flat = np.column_stack(
        (
            np.linspace(-5.0, 5.0, 100),
            np.zeros(100),
            np.linspace(0.0, 1.0, 100),
        )
    )
    assert rank_tower_candidates(flat, (0.0, 0.0)) == ()
    assert rank_tower_candidates([[0.0, 0.0, 0.0]] * 4, (0.0, 0.0)) == ()


def test_candidate_ranking_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="maximum_candidates"):
        rank_tower_candidates([[0.0, 0.0, 0.0]] * 8, (0.0, 0.0), maximum_candidates=0)
    with pytest.raises(ValueError, match="seed_xy"):
        rank_tower_candidates([[0.0, 0.0, 0.0]] * 8, (0.0,))
    with pytest.raises(ValueError, match="N×3"):
        rank_tower_candidates([[0.0, 0.0]] * 8, (0.0, 0.0))
