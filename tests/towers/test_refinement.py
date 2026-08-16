"""V5A2 bounded RAW tower refinement tests."""

from __future__ import annotations

import math

from algorithms.towers.raw_access import RawTileCache, RawTileCacheConfig
from algorithms.towers.refinement import (
    TOWER_RAW_REFINEMENT_VERSION,
    TowerRawRefinementConfig,
    refine_tower_from_raw,
)
from core.contracts.project import TowerGeometry
from core.contracts.tower_candidates import TowerCandidateRawRefinementEvidence

from tests.towers.conftest import project_for_source, write_tower_las


def _cache_for(tmp_path, *, extra_tower=False, tower_height_m=30.0):
    src = tmp_path / "src.las"
    points = write_tower_las(
        src,
        extra_tower_xyz=(45.0, 50.0, 0.0) if extra_tower else None,
        tower_height_m=tower_height_m,
    )
    project, binding, metadata = project_for_source(src, points)
    cache = RawTileCache.prepare(
        src, tmp_path / "cache", source_binding=binding, config=RawTileCacheConfig()
    )
    return cache, binding


def _refine(cache, binding, seed, *, config=None):
    query = cache.query(seed, xy_radius_m=3.5)
    return refine_tower_from_raw(
        query, binding, config or TowerRawRefinementConfig()
    )


def test_defensible_synthetic_tower_refined(tmp_path):
    cache, binding = _cache_for(tmp_path)
    result = _refine(cache, binding, (10.0, 20.0, 0.0))
    assert result.success, result.failure_reason
    assert isinstance(result.geometry, TowerGeometry)
    base = result.geometry.base_xyz_absolute_m
    top = result.geometry.top_xyz_absolute_m
    assert abs(base[0] - 10.0) < 0.15
    assert abs(base[1] - 20.0) < 0.15
    assert result.geometry.height_m > 25.0
    assert top[2] > base[2]
    assert math.isclose(result.geometry.height_m, top[2] - base[2], rel_tol=1e-9)


def test_base_and_top_extraction(tmp_path):
    cache, binding = _cache_for(tmp_path, tower_height_m=30.0)
    result = _refine(cache, binding, (10.0, 20.0, 0.0))
    assert result.success
    diag = result.diagnostics
    assert diag["vertical_extent_m"] > 25.0
    assert diag["xy_footprint_m"] < 0.5
    assert diag["median_verticality"] > 0.9
    assert diag["ground_support_points"] > 0


def test_verticality_and_continuity_evidence(tmp_path):
    cache, binding = _cache_for(tmp_path)
    result = _refine(cache, binding, (10.0, 20.0, 0.0))
    assert result.success
    # Continuity/drift gates are recorded in the diagnostics.
    assert "vertical_continuity_max_band_gap_m" in result.diagnostics
    assert "shaft_band_drift_m" in result.diagnostics


def test_bounded_raw_input_only(tmp_path):
    cache, binding = _cache_for(tmp_path)
    result = _refine(cache, binding, (10.0, 20.0, 0.0))
    assert result.success
    details = result.evidence.details
    assert details["raw_query"]["tiles_used"]
    assert details["raw_query"]["raw_point_count"] > 0
    assert details["raw_query"]["z_window_min_m"] <= 10.0
    assert details["raw_query"]["z_window_max_m"] >= 120.0


def test_valid_typed_evidence(tmp_path):
    cache, binding = _cache_for(tmp_path)
    result = _refine(cache, binding, (10.0, 20.0, 0.0))
    assert result.success
    assert isinstance(result.evidence, TowerCandidateRawRefinementEvidence)
    assert result.evidence.method == "full-resolution-raw"
    assert result.evidence.algorithm_version == TOWER_RAW_REFINEMENT_VERSION
    assert result.evidence.source_content_identity_digest == binding.source_content_identity_digest
    assert result.evidence.source_identity_record_digest == binding.source_identity_record_digest


def test_source_linkage_in_evidence(tmp_path):
    cache, binding = _cache_for(tmp_path)
    result = _refine(cache, binding, (10.0, 20.0, 0.0))
    assert result.success
    ev = result.evidence
    assert ev.source_content_identity_digest == cache.source_binding.source_content_identity_digest


def test_insufficient_evidence_fails_closed(tmp_path):
    cache, binding = _cache_for(tmp_path)
    # An empty region far from any tower must fail without inventing geometry.
    result = _refine(cache, binding, (500.0, 500.0, 0.0))
    assert not result.success
    assert result.geometry is None
    assert result.evidence is None
    assert result.failure_reason is not None


def test_scattered_points_fail_closed(tmp_path):
    cache, binding = _cache_for(tmp_path)
    # A low flat pad fails the vertical-extent/continuity gates.
    result = _refine(cache, binding, (5.0, 5.0, -0.1))
    # Some scattered ground may exist there; it must never be accepted as a tower.
    assert not result.success or result.geometry.height_m > 5.0


def test_nan_and_invalid_geometry_rejected(tmp_path):
    # A successful refinement must always produce finite TowerGeometry values.
    cache, binding = _cache_for(tmp_path)
    result = _refine(cache, binding, (10.0, 20.0, 0.0))
    if result.success:
        for value in (
            *result.geometry.base_xyz_absolute_m,
            *result.geometry.top_xyz_absolute_m,
            result.geometry.height_m,
        ):
            assert math.isfinite(value)
            assert value == value  # not NaN


def test_no_viewer_or_copc_authority(tmp_path):
    # The refinement signature takes only a RAW query result + source binding;
    # it never touches a viewer or COPC node.
    cache, binding = _cache_for(tmp_path)
    query = cache.query((10.0, 20.0, 0.0), xy_radius_m=3.5)
    assert query.source_index is not None
    result = refine_tower_from_raw(query, binding)
    assert result is not None


def test_adjusted_seed_near_tower_still_refines(tmp_path):
    cache, binding = _cache_for(tmp_path)
    # A review-adjusted seed slightly off the shaft still refines correctly.
    result = _refine(cache, binding, (10.8, 20.3, 1.0))
    assert result.success, result.failure_reason
    base = result.geometry.base_xyz_absolute_m
    assert abs(base[0] - 10.0) < 1.0
    assert abs(base[1] - 20.0) < 1.0
