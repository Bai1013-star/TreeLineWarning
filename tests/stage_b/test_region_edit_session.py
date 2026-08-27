from __future__ import annotations

import numpy as np
import pytest
from algorithms.folder_pipeline.tree_editor import OriginalTree, OriginalTrees, TreeEditorError

from core.stage_b.region_edit_session import RegionTreeEditSession


def _session() -> RegionTreeEditSession:
    xyz = np.asarray(
        [
            [-0.50, 0.00, 0.00],
            [0.20, 0.00, 0.00],
            [0.85, 0.00, 0.00],
            [-0.45, 0.40, 0.00],
            [0.10, 0.40, 0.00],
            [0.90, 0.40, 0.00],
        ],
        dtype=np.float64,
    )
    original = OriginalTrees(
        records=(
            OriginalTree(1, "Tree001", 3, (0.85, 0.0, 0.0), 5.0),
            OriginalTree(2, "Tree002", 3, (0.90, 0.4, 0.0), 5.5),
        ),
        original_index=np.arange(100, 106, dtype=np.int64),
        tree_id=np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32),
        xyz_local_m=xyz,
    )
    return RegionTreeEditSession(original)


def _identity() -> list[list[float]]:
    return np.eye(4, dtype=np.float64).tolist()


def test_region_lasso_removes_exact_points_across_multiple_trees_and_undoes_once() -> None:
    session = _session()
    session.select([1, 2])
    polygon = [[-0.60, -0.20], [0.30, -0.20], [0.30, 0.60], [-0.60, 0.60]]

    preview = session.preview_trim_lasso(-1, _identity(), polygon, binding="b1")

    assert preview["method"] == "REGION_LASSO"
    assert preview["point_count"] == 4
    assert preview["affected_tree_count"] == 2
    assert preview["per_tree_removed_counts"] == {"1": 2, "2": 2}
    assert preview["fully_removed_tree_ids"] == []
    operation = session.commit_trim(
        preview["preview_id"],
        binding="b1",
        minimum_remaining_points=1,
    )
    assert operation["type"] == "REMOVE_POINTS"
    assert operation["tree_ids"] == [1, 2]
    assert operation["point_count"] == 4
    assert len(session.operations) == 1
    assert {tree["tree_id"]: tree["point_count"] for tree in session.snapshot()["trees"]} == {
        1: 1,
        2: 1,
    }

    undone = session.undo()
    assert undone is not None
    assert len(session.operations) == 0
    assert {tree["tree_id"]: tree["point_count"] for tree in session.snapshot()["trees"]} == {
        1: 3,
        2: 3,
    }


def test_region_lasso_scope_is_the_current_selection() -> None:
    session = _session()
    session.select([2])
    polygon = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]

    preview = session.preview_trim_lasso(-1, _identity(), polygon, binding="scope")

    assert preview["tree_ids"] == [2]
    assert preview["point_count"] == 3
    assert preview["fully_removed_tree_ids"] == [2]
    session.commit_trim(preview["preview_id"], binding="scope", minimum_remaining_points=2)
    snapshot = session.snapshot()
    assert [tree["tree_id"] for tree in snapshot["trees"]] == [1]
    assert session.current_tree_ids() == [1]


def test_region_lasso_rejects_tiny_partial_tree_remnants_but_allows_zero() -> None:
    session = _session()
    session.select([1, 2])
    polygon = [[-1.0, -1.0], [0.30, -1.0], [0.30, 1.0], [-1.0, 1.0]]
    preview = session.preview_trim_lasso(-1, _identity(), polygon, binding="tiny")

    with pytest.raises(TreeEditorError, match="不足以形成有效单木"):
        session.commit_trim(
            preview["preview_id"],
            binding="tiny",
            minimum_remaining_points=2,
        )
    assert session.operations == []


def test_single_tree_lasso_contract_remains_unchanged() -> None:
    session = _session()
    polygon = [[-0.60, -0.20], [0.30, -0.20], [0.30, 0.20], [-0.60, 0.20]]

    preview = session.preview_trim_lasso(1, _identity(), polygon, binding="single")

    assert preview["tree_id"] == 1
    assert preview["method"] == "LASSO_TRIM"
    assert preview["point_count"] == 2
