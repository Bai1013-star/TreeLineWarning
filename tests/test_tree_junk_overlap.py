"""Overlapping infrastructure must not become a tree merely by touching it."""
import numpy as np
from algorithms.spans.raw_stem_first import segment_from_raw_stems
from algorithms.spans.stem_guided_hybrid import StemEvidence
from tests.test_raw_stem_first import cylinder, crown


def test_connected_vertical_panel_is_not_crown_or_trunk():
    tree = np.vstack((cylinder(0, 0), crown(0, 0)))
    # Tall railing/panel touching both the shaft and the crown; class 1,
    # therefore LAS classification alone cannot distinguish it.
    panel = np.mgrid[-2.:2.01:.05, .18:.181:.05, .3:5.51:.05].reshape(3, -1).T
    xyz = np.vstack((tree, panel))
    stem = StemEvidence(0, (0., 0.), 3., 7, 0., 1., .12, 100, 0., 1., .9)
    result = segment_from_raw_stems(xyz, xyz[:, 2], (stem,))
    assert len(result.candidates) == 1
    assert np.all(result.fine_tree_id[:len(cylinder(0, 0))] == 0)
    # Intersection points cannot be identified physically; all clearly
    # separated panel returns must be quarantined, not labelled as a tree.
    separated = np.abs(panel[:, 0]) > .45
    assert np.all(result.fine_tree_id[len(tree):][separated] == -1)


def test_las_building_returns_touching_crown_are_not_tree():
    tree = np.vstack((cylinder(0, 0), crown(0, 0)))
    building = np.mgrid[-2.:2.01:.08, .5:.51:.08, 2.6:5.51:.08].reshape(3, -1).T
    xyz = np.vstack((tree, building))
    cls = np.ones(len(xyz), dtype=np.uint8); cls[len(tree):] = 6
    stem = StemEvidence(0, (0., 0.), 3., 7, 0., 1., .12, 100, 0., 1., .9)
    result = segment_from_raw_stems(xyz, xyz[:, 2], (stem,), classification=cls)
    assert len(result.candidates) == 1
    assert np.all(result.fine_tree_id[len(tree):] == -1)


def test_raw_backmapping_and_trunk_restore_cannot_reintroduce_panel(tmp_path):
    from algorithms.spans.tree_runner import _restore_validated_trunk_membership
    from tests.test_raw_stem_first import raw_region
    tree = np.vstack((cylinder(0, 0), crown(0, 0)))
    panel = np.mgrid[-2.:2.01:.05, .18:.181:.05, .3:5.51:.05].reshape(3, -1).T
    xyz = np.vstack((tree, panel))
    stem = StemEvidence(0, (0., 0.), 3., 7, 0., 1., .12, 100, 0., 1., .9)
    hybrid = segment_from_raw_stems(xyz, xyz[:, 2], (stem,))
    region = raw_region(tmp_path, xyz)
    ground = np.column_stack((np.mgrid[-5:6:1., -5:6:1.].reshape(2, -1).T, np.zeros(121)))
    # Worst case: coarse-to-fine cells have assigned ALL infrastructure rows.
    mapped = np.zeros(len(xyz), dtype=np.int32)
    clean, proof = _restore_validated_trunk_membership(region, mapped, hybrid, ground, (0., 0., 0.))
    assert np.all(clean[:len(cylinder(0, 0))] == 0)
    assert np.all(clean[len(tree):][np.abs(panel[:, 0]) > .45] == -1)
    assert proof['quarantined_raw_object_points'] > 0


def test_sparse_branches_and_curved_crown_are_not_large_planar_junk():
    from algorithms.spans.raw_stem_first import detect_structural_planes
    xyz = np.vstack((cylinder(0, 0), crown(0, 0)))
    assert detect_structural_planes(xyz, xyz[:, 2], ()) == []


def test_filter_revision_invalidates_previous_science_checkpoints():
    from dataclasses import replace
    from core.config.fast_baseline import FastBaselineConfig
    config = FastBaselineConfig()
    assert config.computed_digest() != replace(config, tree_structural_filter_version='previous-filter').computed_digest()
