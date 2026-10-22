"""Protection is an exact identity rule, independent of journal history."""
import itertools
from pathlib import Path
import numpy as np
import pytest
from algorithms.folder_pipeline.tree_editor import OriginalTree, OriginalTrees, load_edits_payload, TreeEditorError
from core.stage_b.region_edit_session import RegionTreeEditSession
from core.stage_b.infrastructure import fit_selected_pole
from core.stage_b.dbh_evidence import calibrated_breast_height


def session():
    rng = np.random.default_rng(54)
    xyz = np.vstack([rng.normal([x,0,0],[.04,.04,.2],(300,3)) for x in [-.5,.5]])
    source = OriginalTrees((OriginalTree(1,'one',300,None,None),OriginalTree(2,'two',300,None,None)),
                           np.arange(1000,1600), np.repeat([1,2],300).astype(np.int32), xyz)
    s = RegionTreeEditSession(source)
    s.protect_source_rows(np.r_[np.arange(1000,1020),np.arange(1300,1320)])
    return s


@pytest.mark.parametrize('sequence', list(itertools.product(['delete','merge','split','lasso','undo'], repeat=3)))
def test_bounded_action_sequences_never_unlock_infrastructure(sequence):
    s = session()
    original_xyz = s.original.xyz_local_m.copy()
    for action in sequence:
        ids = s.current_tree_ids()
        if action=='undo':
            s.undo()
        elif action=='delete' and ids:
            s.select([ids[0]]); s.delete_selected()
        elif action=='merge' and len(ids)>=2:
            s.select(ids[:2]); s.merge_selected()
        elif action=='split' and ids:
            # Split only a merged two-centre tree, not an arbitrary single stem.
            choices = [tid for tid in ids if np.ptp(s.current_membership(tid)[1][:,0])>.8]
            if choices:
                s.select([choices[0]]); s.split_selected([[-.5,0,0],[.5,0,0]])
        elif action=='lasso' and ids:
            count = len(s.operations)
            try:
                p = s.preview_trim_lasso(-1,np.eye(4),[[-1,-1],[0,-1],[0,1],[-1,1]],binding='fixed')
            except TreeEditorError as error:
                # Repeating an already deleted region is a safe empty selection,
                # not permission to delete locked points or mutate the journal.
                assert '套索内没有可编辑点云' in str(error)
                assert len(s.operations)==count
            else:
                s.commit_trim(p['preview_id'],binding='fixed')
        final, _, _, _ = s.build_final()
        assert not np.intersect1d(final,s.protected_original_index).size
        np.testing.assert_array_equal(s.original.xyz_local_m,original_xyz)
        # Replay as after close/reopen, including empty trees and older splits.
        reopened = session()
        reopened.operations = load_edits_payload(s.operations)
        np.testing.assert_array_equal(reopened.build_final()[0],final)


def test_legacy_split_can_be_protected_then_merged_and_split_again():
    s = session()
    s.protect_source_rows([])
    s.select([1,2]); merged=s.merge_selected()
    s.select([merged]); s.split_selected([[-.5,0,0],[.5,0,0]])
    s.protect_source_rows(np.arange(1000,1020))
    s.select(s.current_tree_ids()); merged=s.merge_selected()
    s.select([merged]); s.split_selected([[-.5,0,0],[.5,0,0]])
    assert len(s.build_final()[0])==580
    while s.operations:
        s.undo()
        assert not np.intersect1d(s.build_final()[0],s.protected_original_index).size


def test_dbh_exclusions_after_split_delete_only_the_deleted_child(tmp_path):
    from core.stage_b.dbh_raw_recovery import current_exclusions
    s = session();s.protect_source_rows([])
    original = [{'global_tree_id':i,'merged_span_tree_keys':[[1,i]]} for i in (1,2)]
    np.savez(tmp_path/'tree_segmentation_original.npz',original_index=s.original.original_index,tree_id=s.original.tree_id)
    s.select([1,2]);merged=s.merge_selected()
    s.select([merged]);children=s.split_selected([[-.5,0,0],[.5,0,0]])
    deleted=s.current_membership(children[0])[0].copy()
    s.select([children[0]]);s.delete_selected()
    remaining=s.current_membership(children[1])[0]
    excluded=current_exclusions(tmp_path,tmp_path,original,s.operations)
    np.testing.assert_array_equal(excluded,np.sort(deleted))
    assert not np.intersect1d(excluded,remaining).size
    s.undo();assert not len(current_exclusions(tmp_path,tmp_path,original,s.operations))


def reference(slope=0.,z=10.):
    return dict(raw_ground_plane_coefficients=[np.tan(np.deg2rad(slope)),0,z],
                raw_ground_plane_xy_origin_m=[0,0], processed_ground_prior_z_absolute_m=z,
                ground_reference_method='ORIGINAL_RAW_LOCAL_GROUND_MODE',raw_ground_support_point_count=20)


@pytest.mark.parametrize('slope,xy,ground,shared',[(0,[0,0],10,True),(4,[0,0],10,True),
                                               (5,[0,0],10,False),(4,[20,0],10,False),
                                               (0,[0,0],10.02,False)])
def test_shared_height_never_changes_this_trees_physical_ground(slope,xy,ground,shared):
    calibration = dict(dbh_cm=20,global_tree_id=1,dbh_ground_z_absolute_m=ground,
                       dbh_measurement_z_absolute_m=ground+1.3,dbh_height_reference='ORIGINAL_RAW_GROUND_PLUS_1P30M')
    value = calibrated_breast_height(reference(slope),xy,[calibration])
    assert value['common_height_applied'] is shared
    expected_ground = 10+xy[0]*np.tan(np.deg2rad(slope))
    assert value['ground_z_absolute_m']==pytest.approx(expected_ground)
    assert value['breast_height_z_absolute_m']-expected_ground==pytest.approx(1.3)


def test_pole_fit_requires_a_real_continuous_shaft():
    z = np.repeat(np.linspace(.3,9.8,60),40)
    angle = np.tile(np.linspace(0,2*np.pi,40,endpoint=False),60)
    raw = np.column_stack((.14*np.cos(angle),.14*np.sin(angle),z))
    keep, fit = fit_selected_pole(raw,np.arange(len(raw)),[0,0,0],10)
    assert keep.sum()>len(raw)*.8 and fit['support_sections']>=20
    with pytest.raises(ValueError,match='INSUFFICIENT'):
        fit_selected_pole(raw[raw[:,2]>9],np.arange((raw[:,2]>9).sum()),[0,0,0],10)


def test_shared_height_accepts_valid_numeric_strings_without_moving_ground():
    row = dict(dbh_cm='20', dbh_ground_z_absolute_m='10',
               dbh_measurement_z_absolute_m='11.3', global_tree_id=1,
               dbh_height_reference='ORIGINAL_RAW_GROUND_PLUS_1P30M')
    value = calibrated_breast_height(reference(), [0, 0], [row])
    assert value['common_height_applied']
    assert value['breast_height_z_absolute_m'] == pytest.approx(11.3)
