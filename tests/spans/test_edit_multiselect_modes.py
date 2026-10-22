import numpy as np
import pytest

from algorithms.folder_pipeline.tree_editor import OriginalTree, OriginalTrees, TreeEditSession, TreeEditorError
from core.stage_b.region_edit_session import RegionTreeEditSession


def source(clouds):
    return OriginalTrees(
        records=tuple(OriginalTree(i + 1, str(i + 1), len(cloud), None, None) for i, cloud in enumerate(clouds)),
        original_index=np.arange(sum(map(len, clouds)), dtype=np.int64),
        tree_id=np.concatenate([np.full(len(cloud), i + 1, dtype=np.int32) for i, cloud in enumerate(clouds)]),
        xyz_local_m=np.vstack(clouds),
    )


def test_manual_split_centers_preserve_trunks_and_exact_membership():
    rng = np.random.default_rng(31)
    left = rng.normal([-1, 0, 3], [.10, .10, 1], (300, 3))
    right = rng.normal([1, 0, 3], [.10, .10, 1], (300, 3))
    original = source([np.vstack([left, right])])
    session = TreeEditSession(original)
    session.select([1])
    first, second = session.split_selected([left[0].tolist(), right[0].tolist()])
    a, _ = session.current_membership(first)
    b, _ = session.current_membership(second)
    np.testing.assert_array_equal(a, np.arange(300))
    np.testing.assert_array_equal(b, np.arange(300, 600))
    assert session.operations[-1]['selection_method'] == 'MANUAL_TWO_SEEDS'
    session.undo()
    np.testing.assert_array_equal(session.current_membership(1)[0], original.original_index)


def test_split_rejects_missed_or_duplicate_seeds_without_mutation():
    rng = np.random.default_rng(31)
    xyz = rng.normal([0, 0, 3], [.2, .2, 1], (600, 3))
    session = TreeEditSession(source([xyz]))
    session.select([1])
    for seeds in ([xyz[0].tolist(), xyz[0].tolist()], [xyz[0].tolist(), [100, 100, 3]], [[1, 2]]):
        with pytest.raises(TreeEditorError):
            session.split_selected(seeds)
        assert not session.operations


def component_source():
    rng = np.random.default_rng(2)
    clouds = []
    for offset in (0., 10.):
        main = rng.uniform(-.015, .015, (200, 3)) + [offset, 0, 2]
        one = rng.uniform(-.015, .015, (40, 3)) + [offset + 2, 0, 2]
        two = rng.uniform(-.015, .015, (40, 3)) + [offset + 4, 0, 2]
        clouds.append(np.vstack([main, one, two]))
    return source(clouds)


def test_multiple_components_deduplicated_commit_and_undo_once():
    original = component_source()
    session = RegionTreeEditSession(original)
    picks = [{'tree_id': 1, 'seed_xyz': original.xyz_local_m[205].tolist()},
             {'tree_id': 1, 'seed_xyz': original.xyz_local_m[210].tolist()},
             {'tree_id': 1, 'seed_xyz': original.xyz_local_m[245].tolist()},
             {'tree_id': 2, 'seed_xyz': original.xyz_local_m[485].tolist()}]
    preview = session.preview_trim_components(picks, binding='test')
    assert preview['point_count'] == 120
    assert preview['affected_tree_count'] == 2
    assert len(session.preview_xyz(preview['preview_id'])) == 120
    assert not session.operations
    session.commit_trim(preview['preview_id'], binding='test', minimum_remaining_points=100)
    assert len(session.operations) == 1
    assert len(session.current_membership(1)[0]) == 200
    assert len(session.current_membership(2)[0]) == 240
    session.undo()
    assert len(session.current_membership(1)[0]) == 280
    assert len(session.current_membership(2)[0]) == 280


def test_bad_second_component_keeps_prior_preview_and_no_commit():
    original = component_source()
    session = RegionTreeEditSession(original)
    pick = {'tree_id': 1, 'seed_xyz': original.xyz_local_m[205].tolist()}
    previous = session.preview_trim_components([pick], binding='test')
    with pytest.raises(TreeEditorError):
        session.preview_trim_components([pick, {'tree_id': 2, 'seed_xyz': [100, 100, 100]}], binding='test')
    assert not session.operations
    assert len(session.preview_xyz(previous['preview_id'])) == 40


def test_component_batch_stale_binding_and_small_remnant_rejected():
    original = component_source()
    session = RegionTreeEditSession(original)
    pick = {'tree_id': 1, 'seed_xyz': original.xyz_local_m[205].tolist()}
    preview = session.preview_trim_components([pick], binding='test')
    with pytest.raises(TreeEditorError):
        session.commit_trim(preview['preview_id'], binding='other')
    preview = session.preview_trim_components([pick], binding='test')
    with pytest.raises(TreeEditorError):
        session.commit_trim(preview['preview_id'], binding='test', minimum_remaining_points=250)
    assert not session.operations


@pytest.mark.parametrize('asynchronous', [False, True])
def test_desktop_routes_forward_multi_component_and_split_seeds(monkeypatch, asynchronous):
    from io import BytesIO
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock
    from desktop import server

    workflow = SimpleNamespace(split_selected=Mock(return_value={'ok': True}),
                               preview_trim_component=Mock(return_value={'ok': True}))
    tasks = SimpleNamespace(assert_available=lambda: None,
                            submit=lambda kind, callback, request_id: callback())
    monkeypatch.setattr(server, 'editor_tasks', lambda _: tasks)
    for path, payload in [('/api/split', {'seed_xyz': [[1, 2, 3], [4, 5, 6]]}),
                          ('/api/trim/component-preview', {'picks': [{'tree_id': 1, 'seed_xyz': [1, 2, 3]}]})]:
        handler = object.__new__(server._DesktopHandler)
        handler.application = SimpleNamespace(workflow=workflow)
        payload['async'] = asynchronous
        body = json.dumps(payload).encode()
        handler.headers = {'Content-Length': str(len(body))}
        handler.rfile = BytesIO(body)
        handler._send_json = Mock()
        handler._handle_post(path)
        assert handler._send_json.called
    workflow.split_selected.assert_called_once_with([[1, 2, 3], [4, 5, 6]])
    workflow.preview_trim_component.assert_called_once_with(-1, [], [{'tree_id': 1, 'seed_xyz': [1, 2, 3]}])


def test_risk_overlay_preserves_pointer_modifier_event():
    from desktop.review_risk_frontend import TREE_REVIEW_RISK_EXTENSION_JS
    assert 'viewer.options.onPointPick = (hit, event)' in TREE_REVIEW_RISK_EXTENSION_JS
    assert 'originalPointPick(hit, event)' in TREE_REVIEW_RISK_EXTENSION_JS


def test_region_lasso_ignores_selected_tree_and_false_positive_classification():
    # Second object represents a railing/vehicle falsely assigned a tree ID.
    # Region selection must reach it even if the first tree is selected.
    original = source([np.array([[-.8, 0, 0], [-.7, 0, 1]]),
                       np.array([[.2, 0, 0], [.3, 0, 1], [.8, 0, 2]])])
    session = RegionTreeEditSession(original)
    session.select([1])
    polygon = [[.1, -.2], [.5, -.2], [.5, .2], [.1, .2]]
    preview = session.preview_trim_lasso(-1, np.eye(4), polygon, binding='test')
    assert preview['point_count'] == 2
    assert preview['tree_ids'] == [2]
    session.commit_trim(preview['preview_id'], binding='test', minimum_remaining_points=100)
    np.testing.assert_array_equal(session.current_membership(1)[0], [0, 1])
    np.testing.assert_array_equal(session.current_membership(2)[0], [4])
    # Exact persisted identity, not merely the rendered sample, survives reload.
    from algorithms.folder_pipeline.tree_editor import load_edits_payload
    reopened = RegionTreeEditSession(original)
    reopened.operations = load_edits_payload(session.edits_payload()['operations'])
    np.testing.assert_array_equal(reopened.current_membership(2)[0], [4])
    reopened.undo()
    np.testing.assert_array_equal(reopened.current_membership(2)[0], [2, 3, 4])
    np.testing.assert_array_equal(original.original_index, np.arange(5))


@pytest.mark.parametrize('scope', [[1], [2], []])
def test_region_lasso_can_remove_entire_false_positive_object_and_undo(scope):
    original = source([np.array([[-.8, 0, 0], [-.7, 0, 1]]),
                       np.array([[.2, 0, 0], [.3, 0, 1]])])
    session = RegionTreeEditSession(original)
    session.select(scope)
    preview = session.preview_trim_lasso(
        -1, np.eye(4), [[.1, -.2], [.5, -.2], [.5, .2], [.1, .2]], binding='test')
    assert preview['fully_removed_tree_ids'] == [2]
    session.commit_trim(preview['preview_id'], binding='test', minimum_remaining_points=100)
    assert session.current_tree_ids() == [1]
    assert len(session.operations) == 1
    session.undo()
    assert session.current_tree_ids() == [1, 2]
    np.testing.assert_array_equal(session.current_membership(2)[0], [2, 3])


def test_region_lasso_binding_still_protects_against_stale_delete():
    session = RegionTreeEditSession(source([np.array([[0., 0, 0], [.1, 0, 1]])]))
    preview = session.preview_trim_lasso(
        -1, np.eye(4), [[-.2, -.2], [.5, -.2], [.5, .2], [-.2, .2]], binding='test')
    with pytest.raises(TreeEditorError):
        session.commit_trim(preview['preview_id'], binding='wrong', minimum_remaining_points=100)
    assert not session.operations


def test_merge_preserves_full_membership_and_replays_then_undoes():
    from algorithms.folder_pipeline.tree_editor import load_edits_payload
    original = component_source()
    session = RegionTreeEditSession(original)
    session.select([1, 2])
    merged = session.merge_selected()
    assert session.current_tree_ids() == [merged]
    np.testing.assert_array_equal(session.current_membership(merged)[0], original.original_index)
    reopened = RegionTreeEditSession(original)
    reopened.operations = load_edits_payload(session.edits_payload()['operations'])
    np.testing.assert_array_equal(reopened.current_membership(merged)[0], original.original_index)
    reopened.undo()
    assert reopened.current_tree_ids() == [1, 2]


def test_merge_button_calls_commit_without_second_click_toggling_off():
    from pathlib import Path
    script = (Path(__file__).resolve().parents[2] / 'viewer_poc/giro3d/tree_review_app.js').read_text(encoding='utf8')
    assert "byId('btn-merge').onclick = () => void mergeSelection();" in script
    assert "await chooseOperationMode('MERGE', false);" in script
