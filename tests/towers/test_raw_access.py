"""V5A2 RAW spatial tile cache tests."""

from __future__ import annotations

import json

import pytest
from algorithms.towers.raw_access import (
    RawTileCache,
    RawTileCacheConfig,
    RawTileCacheError,
    RawTileCacheIntegrityError,
    RawTileCacheSourceMismatchError,
)

from tests.towers.conftest import project_for_source, write_tower_las


def _prepare(tmp_path, *, config=None, second_tower=False):
    src = tmp_path / "src.las"
    points = write_tower_las(src, extra_tower_xyz=(45.0, 50.0, 0.0) if second_tower else None)
    project, binding, metadata = project_for_source(src, points)
    cache = RawTileCache.prepare(
        src,
        tmp_path / "cache",
        source_binding=binding,
        config=config or RawTileCacheConfig(),
    )
    return cache, binding, src


def test_tile_creation_round_trip(tmp_path):
    cache, binding, _ = _prepare(tmp_path)
    assert cache.point_count > 0
    assert cache.tile_count >= 1
    assert cache.byte_size > 0
    reloaded = RawTileCache.load(
        tmp_path / "cache",
        source_binding=binding,
        config=RawTileCacheConfig(),
    )
    assert reloaded.point_count == cache.point_count
    assert reloaded.tile_count == cache.tile_count
    assert reloaded.byte_size == cache.byte_size
    assert reloaded.cache_id == cache.cache_id


def test_tile_grid_buckets_points_by_xy(tmp_path):
    cache, binding, _ = _prepare(tmp_path, second_tower=True)
    # The two towers are far apart, so they land in distinct grid tiles.
    q_a = cache.query((10.0, 20.0, 0.0), xy_radius_m=3.0)
    q_b = cache.query((45.0, 50.0, 0.0), xy_radius_m=3.0)
    assert q_a.point_count > 0
    assert q_b.point_count > 0
    # The tiles used by each query must differ (towers are > cell apart).
    assert set(q_a.tiles).isdisjoint(set(q_b.tiles))


def test_bounded_query_returns_local_points_only(tmp_path):
    cache, binding, _ = _prepare(tmp_path)
    q = cache.query((10.0, 20.0, 0.0), xy_radius_m=3.0)
    assert q.point_count > 0
    x, y, _ = q.seed_xyz_absolute_m
    import numpy as np

    assert np.max(np.abs(q.x - x)) <= 3.0 + 1e-9
    assert np.max(np.abs(q.y - y)) <= 3.0 + 1e-9
    # Z window is bounded around the seed.
    assert float(q.z.max()) <= q.z_window_max_m + 1e-9
    assert float(q.z.min()) >= q.z_window_min_m - 1e-9


def test_query_returns_unrelated_tiles_only_when_needed(tmp_path):
    cache, binding, _ = _prepare(tmp_path, second_tower=True)
    q_a = cache.query((10.0, 20.0, 0.0), xy_radius_m=3.0)
    for tile in q_a.tiles:
        # The queried tile must contain the seed XY region.
        assert tile in cache.index
    q_far = cache.query((5.0, 5.0, 0.0), xy_radius_m=1.0)
    # A point far from any tower may legitimately return zero RAW points.
    assert isinstance(q_far.point_count, int)


def test_z_window_filters_out_of_window_tiles(tmp_path):
    cache, binding, _ = _prepare(tmp_path)
    # The tower is ~0..30 m; a window high above the seed returns nothing.
    q = cache.query(
        (10.0, 20.0, 500.0), xy_radius_m=3.0, z_context_below_m=10, z_context_above_m=10
    )
    assert q.point_count == 0


def test_source_original_index_preserved(tmp_path):
    cache, binding, _ = _prepare(tmp_path)
    q = cache.query((10.0, 20.0, 0.0), xy_radius_m=3.0)
    import numpy as np

    assert q.source_index.dtype == np.int64
    assert np.all(q.source_index >= 0)
    # Original identity must be unique within the returned set.
    assert len(set(q.source_index.tolist())) == len(q.source_index)


def test_source_binding_round_trip(tmp_path):
    cache, binding, _ = _prepare(tmp_path)
    assert cache.source_binding == binding
    assert cache.config.configuration_digest() == cache.metadata["configuration_digest"]


def test_wrong_source_fails(tmp_path):
    _, binding, _ = _prepare(tmp_path)
    # A different source binding must fail closed on reload.
    other = tmp_path / "other.las"
    other_points = write_tower_las(other, extra_tower_xyz=None, ground_count=0)
    project, other_binding, _ = project_for_source(other, other_points)
    assert other_binding != binding
    with pytest.raises(RawTileCacheSourceMismatchError):
        RawTileCache.load(
            tmp_path / "cache",
            source_binding=other_binding,
            config=RawTileCacheConfig(),
        )


def test_corrupt_metadata_fails(tmp_path):
    _, binding, _ = _prepare(tmp_path)
    meta_path = tmp_path / "cache" / "cache.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["totals"]["tile_count"] = 999
    meta_path.write_text(json.dumps(meta, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(RawTileCacheIntegrityError):
        RawTileCache.load(
            tmp_path / "cache",
            source_binding=binding,
            config=RawTileCacheConfig(),
        )


def test_missing_cache_fails(tmp_path):
    with pytest.raises(RawTileCacheIntegrityError):
        RawTileCache.load(
            tmp_path / "nope",
            source_binding=None,  # type: ignore[arg-type]
            config=RawTileCacheConfig(),
        )


def test_refuse_overwrite_existing_cache(tmp_path):
    _, binding, _ = _prepare(tmp_path)
    from tests.towers.conftest import project_for_source

    other = tmp_path / "other2.las"
    points = write_tower_las(other, extra_tower_xyz=None, ground_count=0)
    project, binding2, _ = project_for_source(other, points)
    with pytest.raises(RawTileCacheError):
        RawTileCache.prepare(
            other,
            tmp_path / "cache",
            source_binding=binding2,
            config=RawTileCacheConfig(),
        )
