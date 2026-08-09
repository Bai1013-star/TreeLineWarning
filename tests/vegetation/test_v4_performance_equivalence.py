"""Deterministic equivalence oracle for the V4 exact-semantics performance fix.

The optimized V4 segmentation must reproduce the frozen baseline outputs
exactly: same result digest, same dense arrays (original index, assignment
state, tree id, reason flags, ragged members, XYZ/HAG/confidence) and the same
per-tree record summaries.  The oracle file ``golden_v4_optimization_baseline.json``
was captured from the unoptimized baseline implementation on deterministic
synthetic scenes, so it doubles as the "old logic as oracle" reference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from algorithms.vegetation.v4_tree_segmentation import (
    _group_assigned_positions,
    _positions_in_cell,
    segment_v4_trees,
)

from tests.vegetation.test_v4_tree_segmentation import (
    make_config,
    make_handoff,
    make_tree_points,
    two_trees_scene,
)

_GOLDEN_PATH = Path(__file__).with_name("golden_v4_optimization_baseline.json")
_TILE_SPACING = 6.0


def _tiled_scene(rows: int, cols: int) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for r in range(rows):
        for c in range(cols):
            points.extend(make_tree_points(2.0 + c * _TILE_SPACING, 2.0 + r * _TILE_SPACING))
    return points


def _scenes() -> dict[str, list[tuple[float, float, float]]]:
    return {
        "single_tree": make_tree_points(2.0, 2.0),
        "two_trees": two_trees_scene(),
        "tile_2x2": _tiled_scene(2, 2),
        "tile_3x3": _tiled_scene(3, 3),
        "tile_4x4": _tiled_scene(4, 4),
        "tile_5x5": _tiled_scene(5, 5),
        "tile_6x6": _tiled_scene(6, 6),
        "tile_7x7": _tiled_scene(7, 7),
    }


def _golden() -> dict[str, dict]:
    return json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


@pytest.mark.parametrize("scene_name", sorted(_scenes()))
def test_optimized_result_matches_frozen_baseline(scene_name: str) -> None:
    golden = _golden()[scene_name]
    result = segment_v4_trees(make_handoff(_scenes()[scene_name]), make_config())

    assert result.result_digest == golden["result_digest"]
    assert _sha(result.original_index) == golden["original_index_sha"]
    assert _sha(result.assignment_state_by_point) == golden["state_sha"]
    assert _sha(result.tree_id_by_point) == golden["tree_id_sha"]
    assert _sha(result.assignment_reason_flags_by_point) == golden["reason_sha"]
    assert _sha(result.ordered_tree_member_original_index) == golden["members_sha"]
    assert _sha(result.xyz_local_m) == golden["xyz_sha"]
    assert _sha(result.hag_m) == golden["hag_sha"]
    assert _sha(result.assignment_confidence_by_point) == golden["confidence_sha"]

    assert len(result.tree_records) == golden["tree_count"]
    for tree, expected in zip(result.tree_records, golden["trees"], strict=True):
        assert tree.tree_id == expected["tree_id"]
        assert tree.member_count == expected["member_count"]
        assert tree.primary_apex_candidate_id == expected["primary_apex_candidate_id"]
        assert tree.primary_trunk_candidate_id == expected["primary_trunk_candidate_id"]
        assert tree.confidence == expected["confidence"]
        assert tree.geometry_summary.to_dict() == expected["geometry_summary"]


def test_golden_has_expected_scene_digests() -> None:
    golden = _golden()
    expected_digests = {
        "single_tree": "d591575c4576",
        "two_trees": "7c0744bed5b6",
        "tile_7x7": "1afc9b027735",
    }
    for scene_name, prefix in expected_digests.items():
        assert golden[scene_name]["result_digest"].startswith(prefix)


def test_positions_in_cell_matches_brute_force_scan() -> None:
    """P0 oracle: the persistent cell lookup equals the full-population scan."""
    from algorithms.vegetation.v4_tree_segmentation import (
        _build_grid,
        _canonical_population,
        derive_algorithm_scales,
    )

    population = _canonical_population(make_handoff(two_trees_scene()))
    grid = _build_grid(population, derive_algorithm_scales(make_config()), make_config())
    for cell_id in grid.occupied_cell_ids:
        expected = [
            position
            for position in range(int(grid.point_cell_ids.shape[0]))
            if int(grid.point_cell_ids[position]) == cell_id
        ]
        assert _positions_in_cell(grid, cell_id) == expected
    # An unoccupied cell yields an empty list.
    assert _positions_in_cell(grid, -1) == []


def test_group_assigned_positions_matches_brute_force_scan() -> None:
    """P1 oracle: the O(N) grouping equals the per-tree full-N scans."""
    from algorithms.vegetation.v4_tree_segmentation import _canonical_population

    population = _canonical_population(make_handoff(two_trees_scene()))
    result = segment_v4_trees(make_handoff(two_trees_scene()), make_config())
    grouped = _group_assigned_positions(
        population, result.tree_id_by_point, result.assignment_state_by_point
    )
    assigned = 1
    for tree_id in range(len(result.tree_records)):
        expected = np.flatnonzero(
            (result.tree_id_by_point == tree_id)
            & (result.assignment_state_by_point == assigned)
        ).tolist()
        assert grouped.get(tree_id, []) == expected
