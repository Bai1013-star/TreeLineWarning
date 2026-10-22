import numpy as np
from algorithms.spans.raw_stem_first import detect_raw_stem_axes, segment_from_raw_stems
from algorithms.vegetation.coarse_to_fine import RawRegion


def cylinder(x, y, radius=.12, height=4.):
    angle, z = np.meshgrid(np.linspace(0, 2*np.pi, 48, endpoint=False), np.arange(.1, height, .05))
    return np.column_stack((x + radius*np.cos(angle.ravel()), y + radius*np.sin(angle.ravel()), z.ravel()))


def crown(x, y):
    # Real volumetric branching/foliage, not a planar disk or rescaled shaft.
    grid = np.mgrid[-.8:.81:.12, -.8:.81:.12, 3.:5.81:.12].reshape(3, -1).T
    keep = (grid[:, 0]**2 + grid[:, 1]**2)/.8**2 + ((grid[:, 2]-4.4)/1.4)**2 < 1
    grid = grid[keep]; grid[:, :2] += (x, y)
    return grid


def raw_region(tmp_path, xyz):
    xyz.astype('<f4').tofile(tmp_path/'xyz.bin')
    np.arange(len(xyz), dtype='<i8').tofile(tmp_path/'index.bin')
    np.ones(len(xyz), dtype='u1').tofile(tmp_path/'class.bin')
    return RawRegion(str(tmp_path/'xyz.bin'), str(tmp_path/'index.bin'), str(tmp_path/'class.bin'),
        len(xyz), tuple(xyz.min(0)), tuple(xyz.max(0)), (0., 0., 0.))


def test_raw_stems_split_touching_crowns_and_exclude_rail_car_post(tmp_path):
    tree1 = np.vstack((cylinder(0, 0), crown(0, 0)))
    tree2 = np.vstack((cylinder(1.2, 0), crown(1.2, 0)))
    bare_post = cylinder(4, 0, height=6.)
    rail = np.mgrid[-2:3:.08, 1.1:1.2:.08, .4:2.1:.08].reshape(3, -1).T
    car = np.mgrid[-1:1:.08, -2.5:-1.3:.08, 1.4:1.5:.08].reshape(3, -1).T
    xyz = np.vstack((tree1, tree2, bare_post, rail, car))
    region = raw_region(tmp_path, xyz)
    ground = np.column_stack((np.mgrid[-5:6:1., -5:6:1.].reshape(2, -1).T, np.zeros(121)))
    stems, proof = detect_raw_stem_axes(region, ground, (0., 0., 0.))
    assert len(stems) == 3, proof
    result = segment_from_raw_stems(xyz, xyz[:, 2], stems)
    assert len(result.candidates) == 2, result.diagnostics
    # Overlapping crown returns have indistinguishable geometry; assert each
    # unambiguous shaft and its own-side crown, not impossible source labels.
    shaft_count = len(cylinder(0, 0))
    assert len(set(result.fine_tree_id[:shaft_count])) == 1
    assert result.fine_tree_id[0] != result.fine_tree_id[len(tree1)]
    assert np.all(result.fine_tree_id[:len(cylinder(0, 0))] >= 0)  # trunk to 10 cm
    assert np.all(result.fine_tree_id[len(tree1)+len(tree2):] == -1)
    assert proof['axis_sections_are_not_dbh']  # proposals never publish fake DBH


def test_no_stem_does_not_invent_axis_or_dbh():
    xyz = crown(0, 0)
    result = segment_from_raw_stems(xyz, xyz[:, 2], ())
    assert not result.candidates
    assert np.all(result.fine_tree_id == -1)


def test_large_plane_not_a_tree():
    from algorithms.spans.stem_guided_hybrid import StemEvidence
    xyz = np.vstack((cylinder(0, 0), np.mgrid[-1:1:.1, 0:.01:.1, 3:6:.1].reshape(3, -1).T))
    stem = StemEvidence(0, (0., 0.), 3., 7, 0., 1., .12, 100, 0., 1., .9)
    result = segment_from_raw_stems(xyz, xyz[:, 2], (stem,))
    assert not result.candidates


def test_display_lod_keeps_sparse_lower_trunk_without_changing_members():
    from algorithms.folder_pipeline.tree_editor import OriginalTrees, OriginalTree, TreeEditSession
    shaft = cylinder(0, 0)
    dense = np.repeat(crown(0, 0), 30, axis=0)
    xyz = np.vstack((shaft, dense))
    ids = np.zeros(len(xyz), dtype=np.int32)
    source = np.arange(len(xyz), dtype=np.int64)
    record = OriginalTree(0, 'Tree0', len(xyz), (0., 0., 5.8), 5.8)
    session = TreeEditSession(OriginalTrees((record,), source, ids, xyz))
    before = source.copy()
    idx, owners, view = session.display_basis(1000)
    assert len(idx) == len(np.unique(idx)) == 1000
    assert np.count_nonzero(view[:, 2] <= 2.5) > 100
    assert np.array_equal(session.original.original_index, before)
    assert np.array_equal(view, xyz[idx])
    assert np.all(owners == 0)


def test_confirmed_pole_shaft_excluded_without_deleting_adjacent_tree():
    from algorithms.spans.stem_guided_hybrid import StemEvidence
    tree = np.vstack((cylinder(0, 0), crown(0, 0)))
    pole = cylinder(1., 0., height=7.)
    xyz = np.vstack((tree, pole))
    stem = StemEvidence(0, (0., 0.), 3., 7, 0., 1., .12, 100, 0., 1., .9)
    pole_axis = StemEvidence(1, (1., 0.), 3., 7, 0., 1., .12, 100, 0., 1., .9)
    result = segment_from_raw_stems(xyz, xyz[:, 2], (stem,), excluded_pole_axes=(pole_axis,))
    assert len(result.candidates) == 1
    assert np.all(result.fine_tree_id[:len(cylinder(0, 0))] == 0)
    assert np.all(result.fine_tree_id[len(tree):] == -1)
