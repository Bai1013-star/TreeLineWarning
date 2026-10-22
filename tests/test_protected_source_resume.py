"""Original-source count proof and protected publication survive reopening."""
import json
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pytest
from algorithms.spans.role_tiles import ROLE_TILE_RECORD_DTYPE
from core.stage_b.dbh_evidence import local_raw_query, sha256
from desktop.workflow import DesktopWorkflow


@pytest.mark.parametrize('embedded_count,top_count,valid', [(None,2,True),(2,2,True),(3,2,False),(None,3,False)])
def test_role_count_is_verified_in_its_actual_publication(tmp_path, embedded_count, top_count, valid):
    source = dict(sha256='a'*64, file_size_bytes=300, source_point_count=2)
    tile = tmp_path/'tiles'/'one.bin'
    tile.parent.mkdir()
    records = np.zeros(2, dtype=ROLE_TILE_RECORD_DTYPE)
    records['source_index'] = [0,1]
    records['z'] = [1.29,1.31]
    records.tofile(tile)
    index = tmp_path/'index.json'
    index.write_text(json.dumps(dict(schema='treeline-span-role-tile-index-v1', tiles={
        'one':dict(file=tile.name,point_count=2,min_xyz=[0,0,1.29],max_xyz=[0,0,1.31],sha256=sha256(tile))})), encoding='utf8')
    metadata = dict(schema='treeline-span-role-tile-cache-v1',role='vegetation',
        admission_semantics='HIGH_RECALL',source_index_exclusion=None,
        source_identity={**source,'source_point_count':embedded_count},source_point_count=top_count,
        record_dtype=[list(v) for v in ROLE_TILE_RECORD_DTYPE.descr],
        role_record_count=2,unique_source_index_count=2,min_source_index=0,max_source_index=1,
        totals={'point_count':2},source_index_validation='UNIQUE_SOURCE_INDEXES_V1',
        artifact_references={'tile_index':{'relative_path':index.name,'sha256':sha256(index)}})
    (tmp_path/'role-tile-cache.json').write_text(json.dumps(metadata),encoding='utf8')
    if valid:
        _, indices, proof = local_raw_query(tmp_path,source,[0,0],(1.2,1.4))
        assert indices.tolist() == [0,1]
        assert proof['source_identity'] == source
        assert proof['source_coverage'] == 'ALL_IMPORTED_ORIGINAL_POINTS_ONE_TO_ONE'
    else:
        with pytest.raises(ValueError,match='PROOF_INVALID'):
            local_raw_query(tmp_path,source,[0,0],(1.2,1.4))


def test_reopen_hydrates_protected_digest_not_just_operation_count(tmp_path, monkeypatch):
    import desktop.workflow as module
    @dataclass
    class Reconciled:
        diagnostics: dict
    class Scheduler:
        def __init__(self,*args,**kwargs): pass
        def _load_persisted_global(self,run_id): return Reconciled({'original':True})
    monkeypatch.setattr(module,'BatchAnalysisScheduler',Scheduler)
    workflow = object.__new__(DesktopWorkflow)
    workflow._root = tmp_path
    workflow._stage_project = lambda: SimpleNamespace()
    workflow._scan = lambda project: SimpleNamespace(scan_id='scan')
    workflow._risk_policy = lambda: None
    marker = 'b'*64
    manifest = dict(analysis_run_id='run',project_id='project',scan_id='scan',status='COMPLETE',
        global_reconciliation={'diagnostics':dict(editor_schema='treeline-stage-b-batch-editor-v1',
            editor_operation_count=0,protected_infrastructure_index_sha256=marker)})
    result = workflow._batch_result_from_manifest(manifest)
    assert result is not None
    assert result.global_reconciled.diagnostics['protected_infrastructure_index_sha256'] == marker
    assert result.global_reconciled.diagnostics['original']
