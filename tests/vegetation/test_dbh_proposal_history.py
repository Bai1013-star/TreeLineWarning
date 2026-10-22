import json
from types import SimpleNamespace
import numpy as np
from core.stage_b.dbh_identity_recovery import _matching_proposal_history
from core.stage_b.dbh_evidence import member_binding


def snapshot(tmp_path, binding):
    br = tmp_path / 'run'
    path = br / 'dbh_v1' / 'snapshot' / 'finalized_geometry.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'binding': binding, 'trees': [{
        'global_tree_id': 3, 'dbh_cm': 22.9,
        'v3_queries': [{'seed': {'xy': [1., 2.], 'source': 'BASELINE_ACCEPTED_STEM_AXIS', 'support': 12}}]
    }]}), encoding='utf-8')
    return br


def binding():
    return dict(project_id='p', analysis_run_id='r', source_identity={'sha256': 's'},
                preparation_sha256='prep', ground_sha256='ground', original_lineage_sha256='l',
                membership={'original_index': 'members', 'tree_id': 'owners'}, editor_revision='old')


def test_undo_same_members_retains_only_seed_not_diameter(tmp_path):
    old = binding()
    br = snapshot(tmp_path, old)
    new = dict(old, editor_revision='new')
    found = _matching_proposal_history(tmp_path, br, new)
    assert found[3]['seed']['xy'] == [1., 2.]
    assert 'dbh_cm' not in found[3]
    assert len(found[3]['sha256']) == 64


def test_changed_members_or_source_cannot_reuse_proposals(tmp_path):
    old = binding()
    br = snapshot(tmp_path, old)
    assert _matching_proposal_history(tmp_path, br, dict(old, membership={'original_index': 'different'})) == {}
    assert _matching_proposal_history(tmp_path, br, dict(old, source_identity={'sha256': 'different'})) == {}


def test_membership_proof_is_row_order_independent_but_owner_sensitive():
    def members(order, owner=None):
        xyz = np.array([[1., 2., 3.], [4., 5., 6.]])
        return SimpleNamespace(global_tree_records=[], final_original_index=np.array([3, 1])[order],
                               final_xyz_local_m=xyz[order], final_tree_id=np.array(owner or [0, 1])[order])
    assert member_binding(members([0, 1])) == member_binding(members([1, 0]))
    assert member_binding(members([0, 1])) != member_binding(members([0, 1], [1, 0]))


def test_legacy_seed_is_proposal_only_and_requires_matching_identity(tmp_path):
    old = binding()
    old['membership']['identities'] = [{'global_tree_id': 3, 'point_count': 100}]
    br = snapshot(tmp_path, old)
    new = dict(old, membership=dict(old['membership'], original_index='canonical', ordering='ORIGINAL_INDEX_ASCENDING_V1'))
    assert _matching_proposal_history(tmp_path, br, new)[3]['legacy_ordering_proposal_only'] is True
    new['membership']['identities'] = [{'global_tree_id': 4, 'point_count': 100}]
    assert _matching_proposal_history(tmp_path, br, new) == {}


def test_structural_filter_revision_keeps_original_raw_shaft_search_proposal():
    from pathlib import Path
    from algorithms.spans.raw_stem_first import RAW_STEM_FIRST_VERSION
    source = (Path(__file__).parents[2] / 'core/stage_b/dbh_identity_recovery.py').read_text(encoding='utf8')
    proposal_guard = source.split('for tree_root in sorted(br.glob', 1)[1].split('span_proof =', 1)[0]
    assert RAW_STEM_FIRST_VERSION in proposal_guard
    replay = (Path(__file__).parents[2] / 'core/stage_b/dbh_recovery.py').read_text(encoding='utf8')
    assert RAW_STEM_FIRST_VERSION in replay.split('raw_backend =', 1)[1].split('if raw_backend:', 1)[0]
