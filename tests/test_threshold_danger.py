from types import SimpleNamespace as NS

import numpy as np
import pytest

from desktop.threshold_danger import build_threshold_danger_display, threshold_membership


def fixture_result(threshold=.5):
    # Three different actual conductors, with a deliberately unsupported gap.
    curves = np.array([[0, 0, 5], [2, 0, 5], [8, 0, 5], [10, 0, 5],
                       [0, 2, 5], [10, 2, 5], [0, 4, 5], [10, 4, 5]], dtype=np.float32)
    arrays = dict(wire_id=np.array([0, 1, 2]), wire_status=np.ones(3, dtype=np.uint8),
                  segment_wire_id=np.array([0, 0, 1, 2]), segment_index=np.array([0, 1, 0, 0]),
                  segment_distance_ready=np.ones(4, bool), segment_quality_flags=np.zeros(4, np.uint16),
                  sampled_curve_xyz=curves, sampled_curve_offsets=np.array([0, 2, 4, 6, 8]))
    xyz = np.array([[1, 0, 5.1], [1, 0, 5.5], [1, 0, 6], [5, 0, 5], [1, 2, 5.1], [1, 4, 5.1]])
    reconciled = NS(final_xyz_local_m=xyz, final_tree_id=np.array([0, 0, 0, 0, 1, 2]),
                    final_original_index=np.arange(6) + 10)
    return NS(global_reconciled=reconciled, final_clearance=NS(records=[]),
              risk_policy_snapshot=dict(warning_clearance_threshold_m=threshold, configuration_digest='extra'),
              span_results={'span': NS(wire_result=NS(w1_arrays=arrays))})


def test_threshold_updates_exact_points_and_includes_boundary_without_bridging_gap():
    result, cache = fixture_result(), {}
    display = build_threshold_danger_display(result, visible_tree_ids=[0, 1, 2], cache=cache)
    assert display['original_index'] == [10, 11, 14, 15]
    assert display['full_point_count'] == 4
    assert display['authority'] == 'EXACT_CURRENT_THRESHOLD_FULL_RESOLUTION'
    actual_distance = np.linalg.norm(np.array(display['xyz']) - display['wire_xyz'], axis=1)
    np.testing.assert_allclose(actual_distance, display['distance_m'], atol=1e-12)
    np.testing.assert_allclose(display['wire_xyz'], [[1, 0, 5], [1, 0, 5], [1, 2, 5], [1, 4, 5]])
    result.risk_policy_snapshot['warning_clearance_threshold_m'] = .2
    assert build_threshold_danger_display(result, visible_tree_ids=[0, 1, 2], cache=cache)['original_index'] == [10, 14, 15]
    result.risk_policy_snapshot['warning_clearance_threshold_m'] = 1
    assert build_threshold_danger_display(result, visible_tree_ids=[0], cache=cache)['original_index'] == [10, 11, 12]
    assert result.global_reconciled.final_original_index.tolist() == list(range(10, 16))


def test_sample_budget_does_not_decide_membership():
    display = build_threshold_danger_display(fixture_result(), visible_tree_ids=[0, 1, 2], display_budget=1)
    assert display['full_point_count'] == 4 and display['display_point_count'] == 1
    assert display['display_only_sampling']


def test_cache_reuses_geometry_but_invalidates_changed_geometry(monkeypatch):
    import desktop.threshold_danger as module
    result, cache = fixture_result(), {}
    build_threshold_danger_display(result, visible_tree_ids=[0], cache=cache)
    original = module.threshold_membership
    monkeypatch.setattr(module, 'threshold_membership', lambda *a: pytest.fail('same geometry recalculated'))
    assert build_threshold_danger_display(result, visible_tree_ids=[2], cache=cache)['original_index'] == [15]
    monkeypatch.setattr(module, 'threshold_membership', original)
    result.global_reconciled = NS(**vars(result.global_reconciled))
    result.global_reconciled.final_xyz_local_m = result.global_reconciled.final_xyz_local_m + [0, 0, 10]
    assert build_threshold_danger_display(result, visible_tree_ids=[0, 1, 2], cache=cache)['full_point_count'] == 0


def test_no_ready_wire_is_unavailable_not_safe():
    result = fixture_result()
    result.span_results['span'].wire_result.w1_arrays['segment_distance_ready'][:] = False
    assert build_threshold_danger_display(result, visible_tree_ids=[0])['authority'] == 'NO_DISTANCE_READY_WIRE'


def test_current_editor_reconciled_override_does_not_require_final_clearance():
    result = fixture_result()
    result.final_clearance = None
    edited = NS(
        final_xyz_local_m=np.array([[1, 0, 5.1], [1, 0, 7.0]]),
        final_tree_id=np.array([9, 9]),
        final_original_index=np.array([100, 101]),
    )
    display = build_threshold_danger_display(
        result,
        visible_tree_ids=[9],
        reconciled_override=edited,
    )
    assert display['authority'] == 'EXACT_CURRENT_THRESHOLD_FULL_RESOLUTION'
    assert display['original_index'] == [100]


def test_midpoint_screen_does_not_miss_long_edge_endpoint_or_degenerate_edge():
    rows, distance = threshold_membership(np.array([[-.2, 0, 0], [5, .5, 0], [10.2, 0, 0], [20, 0, .1]]),
                                        np.array([[0, 0, 0], [20, 0, 0]]), np.array([[10, 0, 0], [20, 0, 0]]), .5)
    assert rows.tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(distance, [.2, .5, .2, .1])


def test_judge_matches_brute_force_for_random_unequal_edges():
    from algorithms.vegetation.tree_wire_clearance import exact_point_to_segment_distance
    rng = np.random.default_rng(912)
    xyz, a, b = rng.normal(size=(170, 3)), rng.normal(size=(11, 3)), rng.normal(size=(11, 3))
    minimum = np.array([min(exact_point_to_segment_distance(p, s, e)[0] for s, e in zip(a, b)) for p in xyz])
    rows, distance = threshold_membership(xyz, a, b, .6)
    np.testing.assert_array_equal(rows, np.flatnonzero(minimum <= .6))
    np.testing.assert_allclose(distance, minimum[rows], atol=1e-14)


@pytest.mark.parametrize('threshold', [0, -1, float('nan'), float('inf')])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        threshold_membership(np.zeros((1, 3)), np.zeros((1, 3)), np.ones((1, 3)), threshold)
