"""Focused tests for the pole-first span-based central configuration."""

from __future__ import annotations

import pytest
from core.config.fast_baseline import (
    FastBaselineConfig,
    default_fast_baseline_config,
    height_to_suppression_radius_m,
)


def test_default_config_is_frozen_and_deterministic() -> None:
    cfg = default_fast_baseline_config()
    assert isinstance(cfg, FastBaselineConfig)
    assert len(cfg.computed_digest()) == 64
    assert cfg.computed_digest() == default_fast_baseline_config().computed_digest()
    with pytest.raises(AttributeError):
        cfg.chm_resolution_m = 1.0


def test_height_to_suppression_radius_is_monotonic_and_capped() -> None:
    cfg = default_fast_baseline_config()
    low = height_to_suppression_radius_m(cfg, 3.0)
    mid = height_to_suppression_radius_m(cfg, 15.0)
    high = height_to_suppression_radius_m(cfg, 100.0)
    assert low < mid
    assert high == cfg.tree_top_radius_cap_m


def test_config_digest_changes_with_any_parameter() -> None:
    base = default_fast_baseline_config()
    changed = default_fast_baseline_config(pole_minimum_height_m=6.0)
    assert base.computed_digest() != changed.computed_digest()


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        FastBaselineConfig(chm_resolution_m=0.0)
    with pytest.raises(ValueError):
        FastBaselineConfig(tree_top_radius_cap_m=0.1)
    with pytest.raises(ValueError):
        FastBaselineConfig(topology_direction_tolerance_rad=-0.5)
    with pytest.raises(ValueError):
        FastBaselineConfig(tree_top_min_prominence_fraction=1.5)


def test_round_trip_preserves_all_fields() -> None:
    cfg = default_fast_baseline_config()
    restored = FastBaselineConfig.from_dict(cfg.to_dict())
    assert restored == cfg
    assert restored.computed_digest() == cfg.computed_digest()


def test_coarse_voxel_size_keeps_reviewed_scale_for_compact_scenes() -> None:
    from algorithms.vegetation.fast_baseline_pipeline import (
        choose_representative_voxel_size,
    )

    # A compact scene keeps 0.25 m even with a huge point count.
    assert choose_representative_voxel_size(105_000_000, xy_area_m2=87_000) == 0.25
    # A large scene coarsens only when its area demands it.
    assert choose_representative_voxel_size(105_000_000, xy_area_m2=4_000_000) == 0.75
    assert choose_representative_voxel_size(1, xy_area_m2=100) == 0.25
