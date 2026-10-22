from types import SimpleNamespace
import pytest
from desktop.review_measurement_cache import review_rows


class Workflow:
    def __init__(self, root):
        self.project_root = root
        self.batch = root/'batch.json'
        self.batch.write_text('{}')
        self._workflow = {'last_batch_id': 'run-1'}
        self._editor = SimpleNamespace(session=SimpleNamespace(operations=[]))
        self._result = object()
        self.count = 0
        self.stale = False
        self.during = None

    def _latest_batch_path(self): return self.batch
    def _editor_revision_key(self, editor):
        return (id(editor), *(id(op) for op in editor.session.operations))
    def final_projection(self, **kwargs):
        self.count += 1
        if self.during:self.during()
        return dict(trees=[dict(height_m=self.count,diameter_cm=None)],
                    metadata=dict(tree_metrics_source='STALE_AFTER_TREE_EDIT' if self.stale
                                  else 'CANONICAL_FULLRES_POST_EDITOR'))


def test_cached_raw_rows_detached_and_mutation_undo_invalidate(tmp_path):
    w=Workflow(tmp_path)
    first=review_rows(w)
    first[0]['diameter_cm']=27
    assert review_rows(w)[0]['diameter_cm'] is None
    assert w.count==1
    w._editor.session.operations.append({})
    assert review_rows(w)[0]['height_m']==2
    w._editor.session.operations.pop()
    assert review_rows(w)[0]['height_m']==3


@pytest.mark.parametrize('change',['result','run','batch','dbh','edits','workflow','project'])
def test_scientific_and_project_revisions_invalidate(tmp_path,change):
    w=Workflow(tmp_path);review_rows(w)
    if change=='result':w._result=object()
    elif change=='run':w._workflow['last_batch_id']='run-2'
    elif change=='batch':w.batch.write_text('{"changed":true}')
    elif change=='project':
        w.project_root=tmp_path/'another';w.project_root.mkdir()
    else:
        names={'dbh':'dbh_final.json','edits':'tree_edits.json','workflow':'desktop-workflow.json'}
        (tmp_path/names[change]).write_text('{}')
    review_rows(w);assert w.count==2


@pytest.mark.parametrize('busy',['_thread','_finalize_thread','_edit_tasks'])
def test_inflight_never_reuses_cached_rows(tmp_path,busy):
    w=Workflow(tmp_path);review_rows(w)
    active=SimpleNamespace(is_alive=lambda:True) if busy!='_edit_tasks' else SimpleNamespace(snapshot=lambda:dict(state='RUNNING'))
    setattr(w,busy,active)
    review_rows(w);review_rows(w);assert w.count==3


def test_stale_and_mid_computation_edits_not_cached(tmp_path):
    w=Workflow(tmp_path);w.stale=True
    review_rows(w);review_rows(w);assert w.count==2
    w.stale=False
    w.during=lambda:w._editor.session.operations.append({})
    review_rows(w);review_rows(w);assert w.count==4


def test_without_stable_editor_not_cached(tmp_path):
    w=Workflow(tmp_path);w._editor=None
    review_rows(w);review_rows(w);assert w.count==2
