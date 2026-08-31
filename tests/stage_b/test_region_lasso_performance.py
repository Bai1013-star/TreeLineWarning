from __future__ import annotations

import time

import numpy as np
import pytest
from algorithms.folder_pipeline.tree_editor import OriginalTree, OriginalTrees, _points_in_polygon
from core.stage_b.region_edit_session import RegionTreeEditSession


@pytest.mark.parametrize("seed", range(5))
def test_indexed_edge_ranges_match_existing_even_odd_rule(seed):
    rng = np.random.default_rng(seed)
    polygon = rng.uniform(-1, 1, (73, 2))  # also self-intersecting polygons
    points = np.vstack((rng.uniform(-1.5, 1.5, (4000, 2)), polygon))
    expected = np.flatnonzero(_points_in_polygon(points, polygon))
    actual = np.sort(RegionTreeEditSession._polygon_rows(points, polygon))
    np.testing.assert_array_equal(actual, expected)


def test_million_member_lasso_bounds_candidates_and_never_uses_raw_or_per_tree_copies(monkeypatch):
    import laspy

    monkeypatch.setattr(laspy, "open", lambda *a, **k: pytest.fail("raw source read"))
    n, trees, vertices = 1_000_000, 100, 512
    xyz = np.random.default_rng(210831).uniform(-1, 1, (n, 3))
    original = OriginalTrees(
        tuple(OriginalTree(i, str(i), n // trees, None, None) for i in range(trees)),
        np.arange(n, dtype=np.int64) * 11 + 123,
        np.repeat(np.arange(trees, dtype=np.int32), n // trees),
        xyz,
    )
    session = RegionTreeEditSession(original)
    monkeypatch.setattr(
        session, "current_membership", lambda *a: pytest.fail("per-tree full membership copy")
    )
    angle = np.arange(vertices) * 2 * np.pi / vertices
    polygon = np.column_stack((0.04 * np.cos(angle), 0.04 * np.sin(angle)))
    started = time.perf_counter()
    preview = session.preview_trim_lasso(-1, np.eye(4), polygon, binding="large")
    elapsed = time.perf_counter() - started
    assert preview["point_count"] == 1207  # measured original implementation, same seed/input
    diagnostics = preview["timings_seconds"]
    assert diagnostics["raw_source_reads"] == 0
    assert diagnostics["projected_points"] == n
    assert diagnostics["polygon_candidates"] < 2000
    assert elapsed < 8  # generous bound; structural assertions catch O(N * edges) regressions
    selected = session.preview_indices(preview["preview_id"]).copy()
    # Independent original predicate on bounding-box candidates verifies exact identities.
    candidate_rows = np.flatnonzero(np.all(np.abs(xyz[:, :2]) <= 0.04, axis=1))
    expected_rows = candidate_rows[_points_in_polygon(xyz[candidate_rows, :2], polygon)]
    np.testing.assert_array_equal(selected, original.original_index[expected_rows])
    session.commit_trim(preview["preview_id"], binding="large", minimum_remaining_points=1)
    final_index, final_ids, final_xyz, _ = session.build_final()
    keep = ~np.isin(original.original_index, selected)
    np.testing.assert_array_equal(final_index, original.original_index[keep])
    np.testing.assert_array_equal(final_ids, original.tree_id[keep])
    np.testing.assert_array_equal(final_xyz, xyz[keep])
    session.undo()
    np.testing.assert_array_equal(session.build_final()[0], original.original_index)
