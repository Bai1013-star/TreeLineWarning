"""Process-local review rows; never publish or cache an in-flight revision."""
from copy import deepcopy
from desktop.integrated_cache import revision_fingerprint


def revision(workflow):
    editor = getattr(workflow, '_editor', None)
    fingerprint = revision_fingerprint(workflow)
    if editor is None or fingerprint is None:
        return None
    for name in ('_thread', '_finalize_thread'):
        thread = getattr(workflow, name, None)
        if thread is not None and thread.is_alive():
            return None
    tasks = getattr(workflow, '_edit_tasks', None)
    if tasks is not None and tasks.snapshot().get('state') == 'RUNNING':
        return None
    try:
        editor_key = workflow._editor_revision_key(editor)
    except (AttributeError, TypeError):
        return None
    return fingerprint, editor_key, id(getattr(workflow, '_result', None))


def review_rows(workflow):
    before = revision(workflow)
    cached = getattr(workflow, '_integrated_review_rows_cache', None)
    if before is not None and cached is not None and cached[0] == before:
        return deepcopy(cached[1])
    projection = workflow.final_projection(persist_display_evidence=False)
    rows = projection.get('trees') or []
    after = revision(workflow)
    if (before is not None and before == after
            and projection.get('metadata', {}).get('tree_metrics_source') == 'CANONICAL_FULLRES_POST_EDITOR'):
        workflow._integrated_review_rows_cache = (after, deepcopy(rows))
    else:
        workflow._integrated_review_rows_cache = None
    return rows
