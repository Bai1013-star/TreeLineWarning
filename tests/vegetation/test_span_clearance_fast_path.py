"""Reference-equivalence and Span-local routing tests for the FAST path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from algorithms.spans.clearance import (
    DIRECT_EDGE_CAP,
    _direct_nearest_edges,
    run_span_tree_wire_clearance,
)
from algorithms.spans.region import (
    SpanLocalFrame,
    SpanRegionSpec,
    subset_raw_tile_cache_oriented,
)
from algorithms.spans.role_tiles import (
    ROLE_TILE_CELL_SIZE_M,
    RoleSpecificTileCache,
    RoleTileCacheError,
    role_tile_configuration_digest,
)
from algorithms.spans.wire import SpanWireResult
from algorithms.vegetation.coarse_to_fine import RawRegion
from algorithms.vegetation.tree_wire_clearance import SegmentEdgeBvh, WireEdgeBundle
from core.config.fast_baseline import FastBaselineConfig


def _edges() -> WireEdgeBundle:
    return WireEdgeBundle(
        a_xyz=np.asarray(((0.0, 0.0, 0.0), (0.0, 2.0, 0.0)), dtype=np.float64),
        b_xyz=np.asarray(((10.0, 0.0, 0.0), (10.0, 2.0, 0.0)), dtype=np.float64),
        wire_id=np.asarray((0, 1), dtype=np.int32),
        segment_index=np.asarray((0, 0), dtype=np.int32),
        polyline_edge_index=np.asarray((0, 0), dtype=np.int32),
        wire_status=np.asarray((0, 0), dtype=np.uint8),
        segment_quality_flags=np.asarray((0, 0), dtype=np.uint16),
        edge_count=2,
    )


def _w1() -> SpanWireResult:
    return SpanWireResult(
        strands=(),
        candidate_mask=np.empty(0, dtype=bool),
        cluster_id_by_rep=np.empty(0, dtype=np.int32),
        diagnostics={},
        w1_arrays={
            "wire_id": np.asarray((0, 1), dtype=np.int32),
            "wire_status": np.asarray((0, 0), dtype=np.uint8),
            "wire_method": np.asarray((0, 0), dtype=np.uint8),
            "wire_quality_flags": np.asarray((0, 0), dtype=np.uint16),
            "wire_has_distance_ready_segments": np.asarray((True, True), dtype=bool),
            "wire_continuous_distance_ready": np.asarray((True, True), dtype=bool),
            "wire_segment_offsets": np.asarray((0, 1, 2), dtype=np.int64),
            "segment_wire_id": np.asarray((0, 1), dtype=np.int32),
            "segment_index": np.asarray((0, 0), dtype=np.int32),
            "segment_method": np.asarray((0, 0), dtype=np.uint8),
            "segment_quality_flags": np.asarray((0, 0), dtype=np.uint16),
            "segment_distance_ready": np.asarray((True, True), dtype=bool),
            "segment_parameter_domain": np.asarray(((0.0, 1.0), (0.0, 1.0))),
            "sampled_curve_xyz": np.asarray(
                ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 2.0, 0.0), (10.0, 2.0, 0.0)),
                dtype=np.float32,
            ),
            "sampled_curve_parameter": np.asarray((0.0, 1.0, 0.0, 1.0), dtype=np.float64),
            "sampled_curve_offsets": np.asarray((0, 2, 4), dtype=np.int64),
        },
    )


def test_direct_backend_matches_reference_bvh_including_ties_and_degenerate_edges():
    edges = _edges()
    points = np.asarray(
        ((1.0, 1.0, 0.0), (5.0, 0.0, 0.0), (10.0, 2.0, 0.0)), dtype=np.float64
    )
    direct = _direct_nearest_edges(points, edges, 1e-9, chunk_size=2)
    bvh = SegmentEdgeBvh(edges, leaf_cap=1)
    reference = [bvh.nearest_edge(point, 1e-9) for point in points]
    np.testing.assert_array_equal(direct[0], [0, 0, 1])
    np.testing.assert_allclose(direct[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(direct[2], [item[2] for item in reference])
    np.testing.assert_allclose(direct[1], [item[1] for item in reference])


def test_direct_backend_matches_reference_for_random_large_and_near_degenerate_geometry():
    rng = np.random.default_rng(1234)
    a = rng.normal(1.0e6, 50.0, size=(8, 3))
    b = a + rng.normal(0.0, 20.0, size=(8, 3))
    b[0] = a[0]  # exact degenerate edge is part of the reference contract
    b[1] = a[1] + np.asarray((1.0e-10, -1.0e-10, 1.0e-10))
    edges = WireEdgeBundle(
        a_xyz=a,
        b_xyz=b,
        wire_id=np.asarray((7, 2, 4, 1, 9, 0, 3, 5), dtype=np.int32),
        segment_index=np.asarray((3, 0, 2, 0, 1, 4, 2, 1), dtype=np.int32),
        polyline_edge_index=np.arange(8, dtype=np.int32),
        wire_status=np.zeros(8, dtype=np.uint8),
        segment_quality_flags=np.zeros(8, dtype=np.uint16),
        edge_count=8,
    )
    points = rng.normal(1.0e6, 50.0, size=(100, 3))
    direct = _direct_nearest_edges(points, edges, 1e-9, chunk_size=17)
    bvh = SegmentEdgeBvh(edges, leaf_cap=2)
    reference = [bvh.nearest_edge(point, 1e-9) for point in points]
    np.testing.assert_array_equal(direct[0], [item[0] for item in reference])
    np.testing.assert_allclose(direct[1], [item[1] for item in reference], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(direct[2], [item[2] for item in reference], rtol=0.0, atol=1e-10)


def test_bounded_direct_backend_matches_reference_for_realistic_119_edge_bundle():
    rng = np.random.default_rng(4321)
    a = np.cumsum(rng.normal(0.0, 0.5, size=(119, 3)), axis=0)
    b = a + rng.normal(0.0, 0.25, size=(119, 3))
    edges = WireEdgeBundle(
        a_xyz=a,
        b_xyz=b,
        wire_id=np.arange(119, dtype=np.int32),
        segment_index=np.zeros(119, dtype=np.int32),
        polyline_edge_index=np.arange(119, dtype=np.int32),
        wire_status=np.zeros(119, dtype=np.uint8),
        segment_quality_flags=np.zeros(119, dtype=np.uint16),
        edge_count=119,
    )
    points = rng.normal(0.0, 3.0, size=(257, 3))
    direct = _direct_nearest_edges(points, edges, 1e-9, chunk_size=31)
    bvh = SegmentEdgeBvh(edges, leaf_cap=4)
    reference = [bvh.nearest_edge(point, 1e-9) for point in points]
    assert DIRECT_EDGE_CAP >= 119
    np.testing.assert_array_equal(direct[0], [item[0] for item in reference])
    np.testing.assert_allclose(direct[1], [item[1] for item in reference], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(direct[2], [item[2] for item in reference], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(direct[3], [item[3] for item in reference], rtol=0.0, atol=1e-12)


def test_tree_clearance_result_matches_reference_backend():
    reconciled = SimpleNamespace(
        global_tree_records=(
            SimpleNamespace(
                global_tree_id=7,
                external_name="T001",
                line_id=1,
                span_id=1,
                span_name="S002",
                full_resolution_point_count=3,
            ),
        ),
        final_tree_id=np.asarray((7, 7, 7), dtype=np.int32),
        final_xyz_local_m=np.asarray(((1.0, 1.0, 0.0), (5.0, 0.0, 0.0), (10.0, 2.0, 0.0))),
        final_original_index=np.asarray((20, 21, 22), dtype=np.int64),
    )
    direct = run_span_tree_wire_clearance(
        reconciled, [_w1()], FastBaselineConfig(), backend="NUMPY_DIRECT"
    )
    reference = run_span_tree_wire_clearance(
        reconciled, [_w1()], FastBaselineConfig(), backend="PYTHON_BVH"
    )
    assert direct.diagnostics["backend"] == "NUMPY_DIRECT"
    assert reference.diagnostics["backend"] == "PYTHON_BVH"
    assert direct.records[0].nearest_wire_id == reference.records[0].nearest_wire_id
    assert (
        direct.records[0].nearest_tree_original_index
        == reference.records[0].nearest_tree_original_index
    )
    np.testing.assert_allclose(
        direct.records[0].minimum_clearance_m, reference.records[0].minimum_clearance_m
    )
    np.testing.assert_allclose(
        direct.records[0].nearest_wire_xyz_local_m,
        reference.records[0].nearest_wire_xyz_local_m,
    )


class _TileCache:
    def __init__(self, points: np.ndarray):
        self.points = points
        self.calls = 0

    def query(self, seed, *, xy_radius_m, z_context_below_m, z_context_above_m):
        self.calls += 1
        return SimpleNamespace(
            x=self.points[:, 0],
            y=self.points[:, 1],
            z=self.points[:, 2],
            source_index=np.arange(len(self.points), dtype=np.int64),
            point_count=len(self.points),
            tiles=("t000000_000000",),
        )


def test_rotated_span_filters_aabb_corner_points_after_tile_query(tmp_path: Path):
    frame = SpanLocalFrame.from_endpoints((0.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 0.0, 0.0))
    spec = SpanRegionSpec("vegetation-search", 1.0, 0.5, 0.5, 0.0, 2.0)
    points = np.asarray(
        (
            (5.0, 5.0, 1.0),  # inside the oriented corridor
            (0.0, 1.5, 1.0),  # inside coarse AABB, outside cross-track width
            (6.0, 4.0, 1.0),  # inside coarse AABB, outside cross-track width
            (100.0, 100.0, 1.0),  # distant source point
        ),
        dtype=np.float64,
    )
    cache = _TileCache(points)
    result = subset_raw_tile_cache_oriented(cache, frame, spec, tmp_path / "region")
    assert cache.calls == 1
    assert result.aabb_query_points == len(points)
    assert result.oriented_region_points == 1
    assert result.region.count == 1
    assert result.tiles == ("t000000_000000",)


def test_role_specific_wire_and_vegetation_indexes_preserve_attrs_and_tile_telemetry(
    tmp_path: Path,
):
    points = np.asarray(
        (
            (5.0, 5.0, 1.0),
            (5.5, 5.5, 2.0),
            (0.0, 1.5, 1.0),
            (100.0, 100.0, 1.0),
        ),
        dtype=np.float32,
    )
    xyz_path = tmp_path / "source" / "xyz.bin"
    index_path = tmp_path / "source" / "index.bin"
    class_path = tmp_path / "source" / "classification.bin"
    xyz_path.parent.mkdir()
    points.tofile(xyz_path)
    np.asarray((20, 21, 22, 23), dtype=np.int64).tofile(index_path)
    np.asarray((5, 6, 7, 8), dtype=np.uint8).tofile(class_path)
    region = RawRegion(
        str(xyz_path),
        str(index_path),
        str(class_path),
        len(points),
        (0.0, 1.5, 1.0),
        (100.0, 100.0, 2.0),
        (0.0, 0.0, 0.0),
    )
    source_identity = {
        "source_content_identity_digest": "a" * 64,
        "source_identity_record_digest": "b" * 64,
    }
    frame = SpanLocalFrame.from_endpoints((0.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 0.0, 0.0))
    spec = SpanRegionSpec("vegetation-search", 1.0, 0.5, 0.5, 0.0, 2.0)
    for role in ("wire", "vegetation"):
        cache = RoleSpecificTileCache.prepare_from_raw_region(
            region,
            tmp_path / f"{role}-role-tile-index",
            role=role,
            source_identity=source_identity,
            config_digest=f"{role}-config",
            source_point_count=4,
        )
        loaded = RoleSpecificTileCache.load(
            cache.root,
            role=role,
            source_identity=source_identity,
            config_digest=f"{role}-config",
            source_point_count=4,
        )
        result = subset_raw_tile_cache_oriented(
            loaded, frame, spec, tmp_path / f"{role}-region"
        )
        assert result.total_tiles == len(cache.index)
        assert result.intersecting_aabb_tiles == result.opened_tiles == 1
        assert result.points_read == 2
        assert result.oriented_region_points == 2
        np.testing.assert_array_equal(result.region.classification(), [5, 6])


def test_role_specific_index_fails_closed_on_role_source_or_config_mismatch(tmp_path: Path):
    points = np.asarray(((0.0, 0.0, 1.0),), dtype=np.float32)
    root = tmp_path / "source"
    root.mkdir()
    xyz = root / "xyz.bin"
    idx = root / "index.bin"
    cls = root / "classification.bin"
    points.tofile(xyz)
    np.asarray((1,), dtype=np.int64).tofile(idx)
    np.asarray((5,), dtype=np.uint8).tofile(cls)
    region = RawRegion(
        str(xyz),
        str(idx),
        str(cls),
        1,
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
    )
    source = {
        "source_content_identity_digest": "c" * 64,
        "source_identity_record_digest": "d" * 64,
    }
    cache = RoleSpecificTileCache.prepare_from_raw_region(
        region,
        tmp_path / "wire-role-tile-index",
        role="wire",
        source_identity=source,
        config_digest="wire-config",
        source_point_count=1,
    )
    with pytest.raises(RoleTileCacheError, match="ROLE_TILE_CACHE_ROLE_MISMATCH"):
        RoleSpecificTileCache.load(
            cache.root,
            role="vegetation",
            source_identity=source,
            config_digest="wire-config",
            source_point_count=1,
        )
    with pytest.raises(RoleTileCacheError, match="ROLE_TILE_CACHE_CONFIGURATION_MISMATCH"):
        RoleSpecificTileCache.load(
            cache.root,
            role="wire",
            source_identity=source,
            config_digest="wrong",
            source_point_count=1,
        )


def _role_region(tmp_path: Path, name: str, points, source_indices) -> RawRegion:
    root = tmp_path / name
    root.mkdir()
    xyz = root / "xyz.bin"
    index = root / "index.bin"
    classification = root / "classification.bin"
    points = np.asarray(points, dtype=np.float32)
    points.tofile(xyz)
    np.asarray(source_indices, dtype=np.int64).tofile(index)
    np.zeros(len(points), dtype=np.uint8).tofile(classification)
    return RawRegion(
        str(xyz),
        str(index),
        str(classification),
        len(points),
        tuple(np.min(points, axis=0)),
        tuple(np.max(points, axis=0)),
        (0.0, 0.0, 0.0),
    )


def _role_identity():
    return {
        "source_content_identity_digest": "a" * 64,
        "source_identity_record_digest": "b" * 64,
    }


def test_canonical_role_tile_configuration_is_five_metres_and_digest_bound():
    assert ROLE_TILE_CELL_SIZE_M == 5.0
    assert role_tile_configuration_digest(5.0) != role_tile_configuration_digest(40.0)


def test_five_metre_index_binds_and_historical_forty_metre_index_is_rejected(tmp_path: Path):
    region = _role_region(tmp_path, "source", ((0.0, 0.0, 1.0),), (1,))
    five = RoleSpecificTileCache.prepare_from_raw_region(
        region,
        tmp_path / "five-metre",
        role="wire",
        source_identity=_role_identity(),
        config_digest="wire-config",
        source_point_count=4,
    )
    forty = RoleSpecificTileCache.prepare_from_raw_region(
        region,
        tmp_path / "historical-forty-metre",
        role="wire",
        source_identity=_role_identity(),
        config_digest="wire-config",
        source_point_count=4,
        cell_size_m=40.0,
    )
    loaded = RoleSpecificTileCache.load(
        five.root,
        role="wire",
        source_identity=_role_identity(),
        config_digest="wire-config",
        source_point_count=4,
        expected_cell_size_m=5.0,
    )
    assert loaded.metadata["tile_grid"]["cell_size_m"] == 5.0
    assert five.root.is_dir() and forty.root.is_dir()
    with pytest.raises(RoleTileCacheError, match="EXPECTED_TILE_SIZE_MISMATCH"):
        RoleSpecificTileCache.load(
            forty.root,
            role="wire",
            source_identity=_role_identity(),
            config_digest="wire-config",
            source_point_count=4,
            expected_cell_size_m=5.0,
        )


def test_duplicate_source_index_within_one_tile_fails_before_publication(tmp_path: Path):
    region = _role_region(
        tmp_path,
        "duplicate-within-tile",
        ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
        (7, 7),
    )
    target = tmp_path / "duplicate-index"
    with pytest.raises(RoleTileCacheError, match="SOURCE_INDEX_DUPLICATE"):
        RoleSpecificTileCache.prepare_from_raw_region(
            region,
            target,
            role="wire",
            source_identity=_role_identity(),
            config_digest="wire-config",
            source_point_count=8,
        )
    assert not target.exists()


def test_duplicate_source_index_across_candidate_tiles_fails_before_publication(
    tmp_path: Path,
):
    region = _role_region(
        tmp_path,
        "duplicate-cross-tile",
        ((0.0, 0.0, 1.0), (5.0, 0.0, 1.0)),
        (9, 9),
    )
    target = tmp_path / "duplicate-cross-tile-index"
    with pytest.raises(RoleTileCacheError, match="SOURCE_INDEX_DUPLICATE"):
        RoleSpecificTileCache.prepare_from_raw_region(
            region,
            target,
            role="vegetation",
            source_identity=_role_identity(),
            config_digest="vegetation-config",
            source_point_count=10,
        )
    assert not target.exists()


def test_unique_source_indices_conserve_records_and_preserve_half_open_boundary(
    tmp_path: Path,
):
    region = _role_region(
        tmp_path,
        "unique-boundary",
        ((4.999, 0.0, 1.0), (5.0, 0.0, 1.0)),
        (10, 11),
    )
    cache = RoleSpecificTileCache.prepare_from_raw_region(
        region,
        tmp_path / "unique-index",
        role="wire",
        source_identity=_role_identity(),
        config_digest="wire-config",
        source_point_count=12,
    )
    assert tuple(cache.index) == ("t0_0", "t1_0")
    assert cache.metadata["role_record_count"] == 2
    assert cache.metadata["unique_source_index_count"] == 2
    assert cache.metadata["totals"]["point_count"] == 2


def test_vegetation_role_index_excludes_exact_wire_source_indices(tmp_path: Path):
    points = np.asarray(
        ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0), (3.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    source = tmp_path / "source"
    source.mkdir()
    xyz = source / "xyz.bin"
    idx = source / "index.bin"
    cls = source / "classification.bin"
    points.tofile(xyz)
    np.asarray((20, 21, 22, 23), dtype=np.int64).tofile(idx)
    np.asarray((0, 0, 0, 0), dtype=np.uint8).tofile(cls)
    region = RawRegion(
        str(xyz), str(idx), str(cls), 4, (0.0, 0.0, 1.0), (3.0, 0.0, 1.0), (0.0, 0.0, 0.0)
    )
    identity = {
        "source_content_identity_digest": "1" * 64,
        "source_identity_record_digest": "2" * 64,
    }
    cache = RoleSpecificTileCache.prepare_from_raw_region(
        region,
        tmp_path / "vegetation-role-tile-index",
        role="vegetation",
        source_identity=identity,
        config_digest="vegetation-config",
        source_point_count=4,
        cell_size_m=5.0,
        excluded_source_indices=np.asarray((21, 23), dtype=np.int64),
        exclusion_role="wire",
        exclusion_source_index_digest="3" * 64,
    )
    loaded = RoleSpecificTileCache.load(
        cache.root,
        role="vegetation",
        source_identity=identity,
        config_digest="vegetation-config",
        source_point_count=4,
        expected_source_index_exclusion={
            "role": "wire",
            "source_index_digest": "3" * 64,
        },
    )
    result = loaded.query(
        (1.0, 0.0, 1.0),
        xy_radius_m=5.0,
        z_context_below_m=1.0,
        z_context_above_m=1.0,
    )
    np.testing.assert_array_equal(result.source_index, [20, 22])
    assert loaded.metadata["totals"]["point_count"] == 2


def test_long_line_role_indexes_route_wire_and_vegetation_locally(tmp_path: Path):
    along = np.linspace(0.0, 1000.0, 101)
    points = np.column_stack((along, along, np.full_like(along, 4.0))).astype(np.float32)
    root = tmp_path / "long-source"
    root.mkdir()
    xyz = root / "xyz.bin"
    idx = root / "index.bin"
    cls = root / "classification.bin"
    points.tofile(xyz)
    np.arange(len(points), dtype=np.int64).tofile(idx)
    np.full(len(points), 5, dtype=np.uint8).tofile(cls)
    region = RawRegion(
        str(xyz),
        str(idx),
        str(cls),
        len(points),
        (0.0, 0.0, 4.0),
        (1000.0, 1000.0, 4.0),
        (0.0, 0.0, 0.0),
    )
    source = {
        "source_content_identity_digest": "e" * 64,
        "source_identity_record_digest": "f" * 64,
    }
    frame = SpanLocalFrame.from_endpoints(
        (0.0, 0.0, 0.0), (1000.0, 1000.0, 0.0), (0.0, 0.0, 0.0)
    )
    spec = SpanRegionSpec("role", 2.0, 10.0, 10.0, 0.0, 5.0)
    for role in ("wire", "vegetation"):
        cache = RoleSpecificTileCache.prepare_from_raw_region(
            region,
            tmp_path / f"long-{role}-role-tile-index",
            role=role,
            source_identity=source,
            config_digest=f"{role}-config",
            source_point_count=len(points),
        )
        result = subset_raw_tile_cache_oriented(
            cache, frame, spec, tmp_path / f"long-{role}-region"
        )
        assert result.total_tiles == len(cache.index)
        assert result.opened_tiles == result.intersecting_aabb_tiles
        assert result.points_read == len(points)
        assert result.oriented_region_points == len(points)
