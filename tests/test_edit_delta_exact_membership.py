from types import SimpleNamespace

import numpy as np
import pytest

from algorithms.folder_pipeline.tree_editor import OriginalTree, OriginalTrees
from core.stage_b.region_edit_session import RegionTreeEditSession
from desktop.workflow import DesktopWorkflow


@pytest.fixture
def setup_delta(monkeypatch):
    monkeypatch.setattr('desktop.tree_display_numbering_extensions._current_display_identities', lambda *args: {})
    monkeypatch.setattr('desktop.review_extensions._result_matches_editor_revision', lambda *args: False)
    # A connected source with two groups; deliberately non-row original indices.
    rng = np.random.default_rng(53)
    left = rng.normal([-.5, 0, 0], [.04, .04, .15], (300, 3))
    right = rng.normal([.5, 0, 0], [.04, .04, .15], (300, 3))
    source = OriginalTrees(
        records=(OriginalTree(1, 'tree', 600, None, None),),
        original_index=np.arange(10000, 10600, dtype=np.int64),
        tree_id=np.ones(600, dtype=np.int32), xyz_local_m=np.vstack((left, right)),
    )
    session = RegionTreeEditSession(source)
    editor = SimpleNamespace(session=session, result=None)
    workflow = object.__new__(DesktopWorkflow)
    basis = workflow._editor_display_basis(editor, 600, None)
    def delta():
        snapshot = {'trees': [{'tree_id': tree_id} for tree_id in session.current_tree_ids()],
                    'operations': list(session.operations)}
        return workflow._editor_display_delta(editor, snapshot, from_revision=max(0, session.revision-1))
    return session, basis, delta


def test_region_delete_delta_hides_exact_points_and_undo_restores(setup_delta):
    session, basis, delta = setup_delta
    session.select([1])
    preview = session.preview_trim_lasso(-1, np.eye(4), [[-1,-1],[0,-1],[0,1],[-1,1]], binding='test')
    removed = session.preview_indices(preview['preview_id'])
    session.commit_trim(preview['preview_id'], binding='test')
    update = delta()
    expected = np.isin(basis['original_index'], removed)
    assert np.any(expected)
    np.testing.assert_array_equal(update['visible_by_index'], ~expected)
    np.testing.assert_array_equal(np.asarray(update['tree_id_by_index'])[expected], -1)
    np.testing.assert_array_equal(update['removed_original_index'], basis['original_index'][expected])
    session.undo()
    assert all(delta()['visible_by_index'])
    assert delta()['removed_original_index'] == []


def test_split_delta_has_exact_children_not_one_source_mapping(setup_delta):
    session, basis, delta = setup_delta
    session.select([1])
    children = session.split_selected([[-.5,0,0],[.5,0,0]])
    update = delta()
    assert set(update['tree_id_by_index']) == set(children)
    assert all(update['visible_by_index'])
    for child in children:
        indices, _ = session.current_membership(child)
        np.testing.assert_array_equal(
            np.asarray(update['tree_id_by_index']) == child,
            np.isin(basis['original_index'], indices),
        )
    session.undo()
    assert set(delta()['tree_id_by_index']) == {1}


def test_delete_entire_object_has_no_visible_display_points(setup_delta):
    session, _, delta = setup_delta
    session.select([1])
    session.delete_selected()
    update = delta()
    assert not any(update['visible_by_index'])
    assert len(update['removed_original_index']) == 600
    session.undo()
    assert all(delta()['visible_by_index'])
