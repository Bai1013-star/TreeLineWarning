"""Tree editor semantics: DELETE / MERGE / UNDO, union, no source mutation."""

from __future__ import annotations

import numpy as np
import pytest
from algorithms.folder_pipeline.server import _stable_color
from algorithms.folder_pipeline.tree_editor import (
    OriginalTrees,
    TreeEditorError,
    TreeEditSession,
    load_edits_payload,
)


def _members_of(original_index: np.ndarray, tree_id: np.ndarray, tid: int) -> np.ndarray:
    return original_index[tree_id == tid]


def test_original_membership_is_exclusive_and_consumed(original_trees: OriginalTrees) -> None:
    total = len(original_trees.original_index)
    assert len(np.unique(original_trees.original_index)) == total
    assert set(np.unique(original_trees.tree_id)) == {t.tree_id for t in original_trees.records}


def test_delete_removes_tree_from_final_membership(original_trees: OriginalTrees) -> None:
    session = TreeEditSession(original_trees)
    session.select([1])
    session.delete_selected()
    final_index, final_tree, _xyz, _merged = session.build_final()
    assert 1 not in set(final_tree)
    # The deleted tree's points became unassigned vegetation, not another tree.
    removed = _members_of(original_trees.original_index, original_trees.tree_id, 1)
    assert not set(removed).intersection(final_index)


def test_delete_does_not_mutate_raw_source(original_trees: OriginalTrees) -> None:
    session = TreeEditSession(original_trees)
    original_xyz = original_trees.xyz_local_m.copy()
    original_index = original_trees.original_index.copy()
    session.select([2])
    session.delete_selected()
    session.build_final()
    assert np.array_equal(original_trees.xyz_local_m, original_xyz)
    assert np.array_equal(original_trees.original_index, original_index)
    # Raw source points are never dropped: total original_index is conserved.
    assert len(original_trees.original_index) == len(original_xyz)


def test_delete_keeps_algorithm_artifact_unchanged(original_trees: OriginalTrees) -> None:
    session = TreeEditSession(original_trees)
    snapshot_before = original_trees.original_index.copy()
    session.select([0])
    session.delete_selected()
    session.build_final()
    # The immutable original is untouched by edits.
    assert np.array_equal(original_trees.original_index, snapshot_before)
    assert original_trees.count == 4


def test_merge_is_exact_union_no_duplicates(original_trees: OriginalTrees) -> None:
    session = TreeEditSession(original_trees)
    session.select([0, 1])
    merged_id = session.merge_selected()
    final_index, final_tree, _xyz, merged = session.build_final()
    union = np.union1d(
        _members_of(original_trees.original_index, original_trees.tree_id, 0),
        _members_of(original_trees.original_index, original_trees.tree_id, 1),
    )
    # The merged tree's OWN membership is the exact union of the two sources.
    merged_members = final_index[final_tree == merged_id]
    assert set(merged_members) == set(union)
    # MERGE_DUPLICATE_ORIGINAL_INDEX == 0 across the whole final set.
    assert len(final_index) == len(np.unique(final_index))
    # Trees 0 and 1 are replaced by exactly ONE final entity; 2 and 3 remain.
    assert 0 not in set(final_tree)
    assert 1 not in set(final_tree)
    assert merged_id in set(final_tree)
    assert {int(v) for v in set(final_tree)} == {merged_id, 2, 3}
    assert merged[merged_id]["source_tree_ids"] == [0, 1]
    assert merged[merged_id]["union_point_count"] == 100 + 120


def test_merge_no_selected_point_lost(original_trees: OriginalTrees) -> None:
    session = TreeEditSession(original_trees)
    session.select([1, 2])
    session.merge_selected()
    final_index, _tree, _xyz, _merged = session.build_final()
    joined = np.union1d(
        _members_of(original_trees.original_index, original_trees.tree_id, 1),
        _members_of(original_trees.original_index, original_trees.tree_id, 2),
    )
    assert set(joined).issubset(final_index)
    # XYZ unchanged (membership-only relabel).
    merged_xyz = original_trees.xyz_local_m[
        np.isin(original_trees.original_index, joined)
    ]
    assert len(merged_xyz) == len(joined)


def test_merge_requires_two_selected(original_trees: OriginalTrees) -> None:
    session = TreeEditSession(original_trees)
    session.select([0])
    with pytest.raises(TreeEditorError):
        session.merge_selected()
    session.selected_ids = []
    with pytest.raises(TreeEditorError):
        session.delete_selected()


def test_undo_restores_previous_edit_state(original_trees: OriginalTrees) -> None:
    session = TreeEditSession(original_trees)
    session.select([1])
    session.delete_selected()
    assert 1 not in set(session.current_tree_ids())
    session.select([2])
    session.delete_selected()
    assert 2 not in set(session.current_tree_ids())
    session.undo()  # restores tree 2
    assert 2 in set(session.current_tree_ids())
    assert 1 not in set(session.current_tree_ids())
    session.undo()  # restores tree 1
    assert set(session.current_tree_ids()) == {0, 1, 2, 3}


def test_undo_merge_restores_sources(original_trees: OriginalTrees) -> None:
    session = TreeEditSession(original_trees)
    session.select([0, 1])
    merged_id = session.merge_selected()
    assert merged_id in set(session.current_tree_ids())
    session.undo()
    assert 0 in set(session.current_tree_ids())
    assert 1 in set(session.current_tree_ids())
    assert merged_id not in set(session.current_tree_ids())


def test_undo_empty_returns_none(original_trees: OriginalTrees) -> None:
    session = TreeEditSession(original_trees)
    assert session.undo() is None


def test_merge_and_delete_sequence(original_trees: OriginalTrees) -> None:
    session = TreeEditSession(original_trees)
    session.select([0, 1])
    merged_id = session.merge_selected()
    session.select([merged_id])
    session.delete_selected()
    final_ids = set(session.current_tree_ids())
    assert 0 not in final_ids and 1 not in final_ids and merged_id not in final_ids
    assert set(final_ids) == {2, 3}


def test_edit_log_roundtrip(original_trees: OriginalTrees) -> None:
    session = TreeEditSession(original_trees)
    session.select([0])
    session.delete_selected()
    session.select([1, 2])
    session.merge_selected()
    operations = session.edits_payload()["operations"]
    restored = load_edits_payload(operations)
    assert restored == operations
    session2 = TreeEditSession(original_trees)
    session2.operations = restored
    assert session2.current_tree_ids() == session.current_tree_ids()


def test_stable_deterministic_tree_colors(original_trees: OriginalTrees) -> None:
    # Same tree id -> same colour every call; different ids differ.
    assert _stable_color(3) == _stable_color(3)
    colors = {_stable_color(tid) for tid in range(10)}
    assert len(colors) == 10


def test_display_decimation_does_not_alter_science_membership(
    original_trees: OriginalTrees,
) -> None:
    from algorithms.folder_pipeline.server import _even_rows

    full = original_trees.original_index
    sampled = full[_even_rows(len(full), 50)]
    assert len(sampled) <= 50
    assert set(sampled).issubset(set(full))
    # The sampled display never changes membership: it is a subset projection.
    assert len(set(sampled)) == len(sampled)


def test_selection_validates_unknown_tree(original_trees: OriginalTrees) -> None:
    session = TreeEditSession(original_trees)
    with pytest.raises(TreeEditorError):
        session.select([999])
