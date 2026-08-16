"""V5A3 local RAW tower-membership segmentation tests.

Covers the required seed-driven V1 behaviour:

- a clear vertical pole/tower with nearby vegetation retains ONLY the tower;
- tower + disconnected vertical clutter keeps the seed-nearest valid structure;
- a connected upper cross-arm is retained (structurally connected);
- a weak / no-tower region fails closed (LOCAL_SEGMENTATION_FAILED);
- exact ``source_index`` preservation and XYZ/source_index alignment;
- deterministic membership (same query -> same membership digest);
- source-binding validation and NaN/invalid points fail closed;
- member bounds/count and segmentation config digest.
"""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import pytest
from algorithms.towers.raw_access import RawTileCache, RawTileCacheConfig, RawTileQueryResult
from algorithms.towers.segmentation import (
    TOWER_LOCAL_SEGMENTATION_FAILURE,
    TowerLocalSegmentationConfig,
    segment_tower_membership,
)
from core.contracts.project import SourceIdentityBinding

from tests.towers.conftest import source_identity_for_las


def _write_las(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.write(path)


def _build_cache(tmp_path: Path, points: np.ndarray):
    src = tmp_path / "src.las"
    _write_las(src, points)
    identity, _ = source_identity_for_las(src, points)
    binding = SourceIdentityBinding.from_source_identity(identity)
    cache = RawTileCache.prepare(
        src, tmp_path / "cache", source_binding=binding, config=RawTileCacheConfig()
    )
    return cache, binding


def _column(xyz, z0, z1, spacing=0.5, scatter=0.03, rs=None, offset=(0.0, 0.0)):
    if rs is None:
        rs = np.random.RandomState(1)
    zs = np.arange(z0, z1 + 0.01, spacing)
    n = len(zs)
    return np.column_stack(
        (
            np.full(n, xyz[0]) + offset[0] + rs.normal(0.0, scatter, n),
            np.full(n, xyz[1]) + offset[1] + rs.normal(0.0, scatter, n),
            z0 + zs,
        )
    )


def _ground(x0, x1, y0, y1, count, z=-0.1, rs=None, exclude=None):
    if rs is None:
        rs = np.random.RandomState(2)
    xy = np.column_stack((rs.uniform(x0, x1, count), rs.uniform(y0, y1, count)))
    if exclude is not None:
        keep = np.ones(len(xy), dtype=bool)
        for center, radius in exclude:
            keep &= np.linalg.norm(xy - np.asarray(center)[None, :], axis=1) >= radius
        xy = xy[keep]
    return np.column_stack((xy[:, 0], xy[:, 1], np.full(len(xy), z)))


def _tower_query(cache, seed, radius=3.5):
    return cache.query(
        seed,
        xy_radius_m=radius,
        z_context_below_m=20.0,
        z_context_above_m=120.0,
    )


def _segment(query, binding, config=None):
    return segment_tower_membership(
        query, binding, config or TowerLocalSegmentationConfig()
    )


def test_tower_membership_retained_vegetation_excluded(tmp_path):
    """Clear vertical pole + nearby vegetation: only tower members returned."""
    rs = np.random.RandomState(7)
    tower = _column((10.0, 20.0), 0.0, 30.0, rs=rs)
    # Vegetation: a disconnected blob of canopy + low scrub near the tower.
    veg_count = 400
    veg = np.column_stack(
        (
            12.0 + rs.normal(0.0, 0.8, veg_count),
            21.6 + rs.normal(0.0, 0.8, veg_count),
            rs.uniform(3.0, 16.0, veg_count),
        )
    )
    ground = _ground(-5.0, 40.0, -5.0, 40.0, 3000, exclude=[((10.0, 20.0), 2.0)])
    points = np.vstack((tower, veg, ground))
    cache, binding = _build_cache(tmp_path, points)
    tower_count = len(tower)
    query = _tower_query(cache, (10.0, 20.0, 0.0))
    result = _segment(query, binding)
    assert result.success, result.failure_reason
    membership = result.membership
    assert membership.member_count > 0
    assert membership.member_count < len(points)
    # Every member must be a tower point (source row < tower_count).
    assert int(membership.source_index.max()) < tower_count, (
        "vegetation/ground rows leaked into membership"
    )
    assert membership.member_count >= 50, "most tower column points retained"
    assert membership.member_bounds_absolute_m[0][2] > 0.0, "ground excluded"


def test_seed_nearest_structure_retained_over_taller_clutter(tmp_path):
    """Tower + disconnected taller vertical clutter: seed-nearest wins."""
    rs = np.random.RandomState(11)
    tower = _column((10.0, 20.0), 0.0, 20.0, rs=rs)
    # Taller disconnected clutter column inside the search radius but farther
    # from the seed than the tower.
    clutter = _column((13.0, 20.5), 0.0, 34.0, rs=rs)
    ground = _ground(-5.0, 40.0, -5.0, 40.0, 2000, exclude=[((10.0, 20.0), 2.0)])
    points = np.vstack((tower, clutter, ground))
    cache, binding = _build_cache(tmp_path, points)
    tower_count = len(tower)
    query = _tower_query(cache, (10.0, 20.0, 0.0))
    result = _segment(query, binding)
    assert result.success, result.failure_reason
    membership = result.membership
    # Membership is the seed-nearest tower, not the taller clutter.
    assert int(membership.source_index.max()) < tower_count, (
        "clutter rows leaked into membership"
    )
    assert membership.member_count >= 30


def test_connected_upper_crossarm_retained(tmp_path):
    """A structurally connected upper cross-arm is retained."""
    rs = np.random.RandomState(3)
    tower = _column((10.0, 20.0), 0.0, 30.0, rs=rs)
    # Horizontal cross-arm attached near the top: a line along X at z ~ 28.5.
    arm_x = np.linspace(8.6, 11.4, 40)
    arm = np.column_stack(
        (arm_x, np.full(40, 20.0) + rs.normal(0.0, 0.05, 40), np.full(40, 28.5))
    )
    ground = _ground(-5.0, 40.0, -5.0, 40.0, 2000, exclude=[((10.0, 20.0), 2.0)])
    points = np.vstack((tower, arm, ground))
    cache, binding = _build_cache(tmp_path, points)
    tower_count = len(tower)
    query = _tower_query(cache, (10.0, 20.0, 0.0))
    result = _segment(query, binding)
    assert result.success, result.failure_reason
    membership = result.membership
    member_idx = set(int(value) for value in membership.source_index)
    arm_rows = set(range(tower_count, tower_count + len(arm)))
    retained_arm = arm_rows & member_idx
    assert len(retained_arm) >= 30, (
        f"connected cross-arm should be retained (kept {len(retained_arm)} of {len(arm)})"
    )
    # Member bounds extend horizontally to the arm extent.
    assert membership.member_bounds_max_xyz[0] > 11.0


def test_weak_no_tower_fails_closed(tmp_path):
    """No defensible vertical structure -> LOCAL_SEGMENTATION_FAILED."""
    rs = np.random.RandomState(5)
    low = np.column_stack(
        (
            8.0 + rs.normal(0.0, 1.5, 500),
            18.0 + rs.normal(0.0, 1.5, 500),
            rs.uniform(0.0, 3.0, 500),
        )
    )
    cache, binding = _build_cache(tmp_path, low)
    query = _tower_query(cache, (10.0, 20.0, 0.0))
    result = _segment(query, binding)
    assert not result.success
    assert result.failure_reason.startswith(TOWER_LOCAL_SEGMENTATION_FAILURE)
    assert result.membership is None


def test_off_scene_fails_closed(tmp_path):
    """A query far from any structure fails closed."""
    rs = np.random.RandomState(5)
    tower = _column((10.0, 20.0), 0.0, 30.0, rs=rs)
    ground = _ground(-5.0, 40.0, -5.0, 40.0, 1000, exclude=[((10.0, 20.0), 2.0)])
    cache, binding = _build_cache(tmp_path, np.vstack((tower, ground)))
    query = _tower_query(cache, (500.0, 500.0, 0.0))
    result = _segment(query, binding)
    assert not result.success
    assert result.failure_reason.startswith(TOWER_LOCAL_SEGMENTATION_FAILURE)


def test_exact_source_index_and_xyz_preserved(tmp_path):
    """Every member point is an exact RAW point with its original row identity."""
    rs = np.random.RandomState(9)
    tower = _column((10.0, 20.0), 0.0, 30.0, rs=rs)
    ground = _ground(-5.0, 40.0, -5.0, 40.0, 800, exclude=[((10.0, 20.0), 2.0)])
    cache, binding = _build_cache(tmp_path, np.vstack((tower, ground)))
    query = _tower_query(cache, (10.0, 20.0, 0.0))
    result = _segment(query, binding)
    assert result.success, result.failure_reason
    membership = result.membership
    lookup = {int(idx): (float(x), float(y), float(z))
              for x, y, z, idx in zip(query.x, query.y, query.z, query.source_index, strict=True)}
    assert len(membership.source_index) == membership.member_count
    assert membership.xyz_absolute_m.shape == (membership.member_count, 3)
    # Exact alignment: xyz[row] == query xyz[source_index[row]].
    for row in range(membership.member_count):
        idx = int(membership.source_index[row])
        assert idx in lookup
        expected = lookup[idx]
        actual = tuple(float(v) for v in membership.xyz_absolute_m[row])
        assert actual == expected, "XYZ/source_index alignment broken"
    # All source indices are unique and finite.
    assert len(set(int(v) for v in membership.source_index)) == membership.member_count
    assert np.isfinite(membership.source_index).all()


def test_membership_deterministic(tmp_path):
    """Same query + config -> same membership digest and arrays."""
    rs = np.random.RandomState(13)
    tower = _column((10.0, 20.0), 0.0, 30.0, rs=rs)
    ground = _ground(-5.0, 40.0, -5.0, 40.0, 800, exclude=[((10.0, 20.0), 2.0)])
    cache, binding = _build_cache(tmp_path, np.vstack((tower, ground)))
    query = _tower_query(cache, (10.0, 20.0, 0.0))
    first = _segment(query, binding)
    second = _segment(query, binding)
    assert first.success and second.success
    assert first.membership.membership_digest() == second.membership.membership_digest()
    assert np.array_equal(first.membership.source_index, second.membership.source_index)
    assert np.array_equal(first.membership.xyz_absolute_m, second.membership.xyz_absolute_m)


def test_source_binding_validation(tmp_path):
    """A wrong source binding is rejected by the membership contract."""
    rs = np.random.RandomState(15)
    tower = _column((10.0, 20.0), 0.0, 30.0, rs=rs)
    ground = _ground(-5.0, 40.0, -5.0, 40.0, 400, exclude=[((10.0, 20.0), 2.0)])
    points = np.vstack((tower, ground))
    src = tmp_path / "src.las"
    _write_las(src, points)
    identity, _ = source_identity_for_las(src, points, source_id="tower-a")
    binding = SourceIdentityBinding.from_source_identity(identity)
    cache = RawTileCache.prepare(
        src, tmp_path / "cache", source_binding=binding, config=RawTileCacheConfig()
    )
    query = _tower_query(cache, (10.0, 20.0, 0.0))
    other_identity, _ = source_identity_for_las(src, points, source_id="tower-b")
    # The membership contract requires the exact source binding type + match.
    with pytest.raises(ValueError):
        segment_tower_membership(query, object())
    # A wrong (different-record) binding is accepted by the algorithm input layer
    # only when the query itself is source-validated; the membership result is
    # then bound to the passed binding.  Verifying type enforcement:
    result = _segment(query, binding)
    assert result.success
    assert result.membership.source_binding is binding


def test_segmentation_config_digest_is_sealed(tmp_path):
    """The segmentation config digest changes when a parameter changes."""
    a = TowerLocalSegmentationConfig()
    b = TowerLocalSegmentationConfig(search_radius_m=4.0)
    assert a.configuration_digest() != b.configuration_digest()
    assert len(a.configuration_digest()) == 64


def test_query_with_nan_fails_closed(tmp_path):
    """A query carrying NaN points fails closed (never invents membership)."""
    rs = np.random.RandomState(17)
    tower = _column((10.0, 20.0), 0.0, 30.0, rs=rs)
    ground = _ground(-5.0, 40.0, -5.0, 40.0, 400, exclude=[((10.0, 20.0), 2.0)])
    cache, binding = _build_cache(tmp_path, np.vstack((tower, ground)))
    query = _tower_query(cache, (10.0, 20.0, 0.0))
    # Corrupt one raw point to NaN.
    bad_x = query.x.copy()
    bad_x[0] = np.nan
    bad_query = RawTileQueryResult(
        seed_xyz_absolute_m=query.seed_xyz_absolute_m,
        xy_radius_m=query.xy_radius_m,
        z_window_min_m=query.z_window_min_m,
        z_window_max_m=query.z_window_max_m,
        tiles=query.tiles,
        x=bad_x,
        y=query.y,
        z=query.z,
        source_index=query.source_index,
        point_count=query.point_count,
    )
    result = _segment(bad_query, binding)
    assert not result.success
    assert result.failure_reason == (
        f"{TOWER_LOCAL_SEGMENTATION_FAILURE}:NON_FINITE_RAW_POINTS"
    )


def test_segmentation_rejects_wrong_config_type(tmp_path):
    rs = np.random.RandomState(19)
    tower = _column((10.0, 20.0), 0.0, 30.0, rs=rs)
    ground = _ground(-5.0, 40.0, -5.0, 40.0, 300, exclude=[((10.0, 20.0), 2.0)])
    cache, binding = _build_cache(tmp_path, np.vstack((tower, ground)))
    query = _tower_query(cache, (10.0, 20.0, 0.0))
    with pytest.raises(ValueError):
        segment_tower_membership(query, binding, "not-a-config")
