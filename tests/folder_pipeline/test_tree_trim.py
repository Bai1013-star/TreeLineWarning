"""Full-resolution local tree trim selection, commit, persistence, and composition."""

from __future__ import annotations

import numpy as np
import pytest
from algorithms.folder_pipeline.tree_editor import (
    OriginalTree,
    OriginalTrees,
    TreeEditorError,
    TreeEditSession,
    load_edits_payload,
)


def _separated_tree() -> OriginalTrees:
    rng = np.random.default_rng(81)
    main = rng.normal((0.0, 0.0, 0.0), 0.035, (240, 3))
    bad = rng.normal((2.0, 0.0, 0.0), 0.025, (24, 3))
    other = rng.normal((4.0, 0.0, 0.0), 0.04, (40, 3))
    xyz = np.vstack((main, bad, other))
    return OriginalTrees(
        records=(
            OriginalTree(1, "Tree-1", 264, (0.0, 0.0, 1.0), 1.0),
            OriginalTree(2, "Tree-2", 40, (4.0, 0.0, 1.0), 1.0),
        ),
        original_index=np.arange(10_000, 10_304, dtype=np.int64),
        tree_id=np.concatenate((np.full(264, 1, np.int32), np.full(40, 2, np.int32))),
        xyz_local_m=xyz,
    )


def test_component_preview_commit_cancel_and_exact_undo() -> None:
    session = TreeEditSession(_separated_tree())
    preview = session.preview_trim_component(1, [2.0, 0.0, 0.0], binding="project/run")
    assert preview["point_count"] == 24
    assert preview["tree_point_count"] == 264
    before = session.build_final()[0].copy()
    assert session.cancel_trim(preview["preview_id"])
    assert np.array_equal(session.build_final()[0], before)
    assert session.operations == []

    preview = session.preview_trim_component(1, [2.0, 0.0, 0.0], binding="project/run")
    removed = session.preview_indices(preview["preview_id"]).copy()
    session.commit_trim(preview["preview_id"], binding="project/run")
    final_index, final_tree, _xyz, _summary = session.build_final()
    assert not set(removed).intersection(final_index)
    assert np.count_nonzero(final_tree == 1) == 240
    assert np.count_nonzero(final_tree == 2) == 40
    session.undo()
    assert np.array_equal(session.build_final()[0], before)


def test_component_main_and_tightly_connected_fail_safe() -> None:
    session = TreeEditSession(_separated_tree())
    with pytest.raises(TreeEditorError, match="套索剔除"):
        session.preview_trim_component(1, [0.0, 0.0, 0.0])


def test_lasso_is_full_resolution_current_tree_only_and_selects_depth_volume() -> None:
    original = _separated_tree()
    session = TreeEditSession(original)
    # Identity clip transform: x/y define the screen position; every z depth
    # along the view direction remains selected, but Tree 2 is never eligible.
    polygon = [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]]
    preview = session.preview_trim_lasso(1, np.eye(4), polygon, binding="p")
    selected = session.preview_indices(preview["preview_id"])
    assert len(selected) == 240
    assert set(selected).issubset(set(original.original_index[original.tree_id == 1]))
    assert not set(selected).intersection(original.original_index[original.tree_id == 2])


def test_stale_preview_and_wrong_binding_are_rejected() -> None:
    session = TreeEditSession(_separated_tree())
    preview = session.preview_trim_component(1, [2.0, 0.0, 0.0], binding="run-a")
    with pytest.raises(TreeEditorError, match="过期"):
        session.commit_trim(preview["preview_id"], binding="run-b")

    preview = session.preview_trim_component(1, [2.0, 0.0, 0.0], binding="run-a")
    session.select([2])
    session.delete_selected()
    with pytest.raises(TreeEditorError, match="过期"):
        session.commit_trim(preview["preview_id"], binding="run-a")


def test_trim_persistence_and_ordered_edit_composition() -> None:
    original = _separated_tree()
    session = TreeEditSession(original)
    preview = session.preview_trim_component(1, [2.0, 0.0, 0.0])
    removed = session.preview_indices(preview["preview_id"]).copy()
    session.commit_trim(preview["preview_id"])
    session.select([1, 2])
    merged_id = session.merge_selected()
    merged_index, merged_tree, _xyz, _summary = session.build_final()
    assert not set(removed).intersection(merged_index)
    assert np.count_nonzero(merged_tree == merged_id) == 280
    session.undo()
    assert set(session.current_tree_ids()) == {1, 2}
    assert not set(removed).intersection(session.build_final()[0])
    session.undo()
    assert len(session.build_final()[0]) == len(original.original_index)

    restored = TreeEditSession(original)
    restored.operations = load_edits_payload(session.edits_payload()["operations"])
    assert np.array_equal(restored.build_final()[0], session.build_final()[0])


def test_trim_log_roundtrip_keeps_exact_indices_out_of_ui_snapshot() -> None:
    session = TreeEditSession(_separated_tree())
    preview = session.preview_trim_component(1, [2.0, 0.0, 0.0])
    exact = session.preview_indices(preview["preview_id"]).copy()
    session.commit_trim(preview["preview_id"])
    persisted = session.edits_payload()["operations"]
    assert "removed_indices_b64" in persisted[0]
    assert "removed_indices_b64" not in session.snapshot()["operations"][0]
    restored = TreeEditSession(_separated_tree())
    restored.operations = load_edits_payload(persisted)
    assert not set(exact).intersection(restored.build_final()[0])


def test_existing_tree_validity_minimum_blocks_commit_without_mutation() -> None:
    session = TreeEditSession(_separated_tree())
    polygon = [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]]
    preview = session.preview_trim_lasso(1, np.eye(4), polygon)
    before = session.build_final()[0].copy()
    with pytest.raises(TreeEditorError, match="有效结果"):
        session.commit_trim(preview["preview_id"], minimum_remaining_points=25)
    assert session.operations == []
    assert np.array_equal(session.build_final()[0], before)


def test_step2_exposes_both_trim_modes_and_non_mutating_preview_controls() -> None:
    from algorithms.folder_pipeline.html import step2_html

    html = step2_html()
    assert "剔除误分团块" in html
    assert "套索剔除" in html
    assert "确认剔除" in html
    assert "取消" in html
    assert 'e.key==="Escape"' in html
    assert "/api/trim/component-preview" in html
    assert "/api/trim/lasso-preview" in html
    assert "cameraParams" in html
    assert 'name:"待剔除"' in html


def test_preview_path_does_not_rebuild_global_final_membership(monkeypatch) -> None:
    session = TreeEditSession(_separated_tree())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("preview must stay on current-tree cached membership")

    monkeypatch.setattr(session, "build_final", forbidden)
    preview = session.preview_trim_component(1, [2.0, 0.0, 0.0])
    assert preview["point_count"] == 24
    assert len(session.preview_xyz(preview["preview_id"])) == 24
