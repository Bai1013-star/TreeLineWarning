"""Isolated real-LAS acceptance; never mutate the operator's projects or source."""
import json
import os
import time
from pathlib import Path
from urllib.parse import quote

from desktop.app import BackendProcess, WarningBackend
from desktop.pdal_runtime import _configure_bundled_runtime
from desktop.project_management import create_empty_project, import_pointcloud

SOURCE_ROOT = Path(os.environ.get('AUDIT_SOURCE_ROOT', str(Path(__file__).resolve().parents[2])))
ACCEPTANCE_ROOT = Path(__file__).resolve().parents[3] / 'verification_real_20260930_weststation'
SOURCE_LAS = Path('E:/treeline/9.20北京西站南路-1_colorized.las')
SEED = Path('D:/TreeLineWarning_一体化软件_20260926/TreeLineWarning一体化软件2.0_20260926/项目数据/2026.09.30-05/stage-a-project.json')
PROJECT = ACCEPTANCE_ROOT / os.environ.get('AUDIT_PROJECT_NAME', '北京西站南路_独立实测验收')
if PROJECT.resolve().parent != ACCEPTANCE_ROOT.resolve():
    raise ValueError('Audit project must stay within the isolated acceptance directory')


def emit(event, **details):
    print(json.dumps({'event': event, **details}, ensure_ascii=False), flush=True)


def main():
    ACCEPTANCE_ROOT.mkdir(exist_ok=True)
    _configure_bundled_runtime()
    if not PROJECT.exists():
        create_empty_project(ACCEPTANCE_ROOT, PROJECT.name)
        emit('import_start', source=str(SOURCE_LAS))
        import_pointcloud(PROJECT, SOURCE_LAS)
        emit('import_complete')
    backend = BackendProcess(PROJECT, SOURCE_ROOT)
    processing_url = backend.start()
    warning = WarningBackend(PROJECT, SOURCE_ROOT, backend)
    warning_url = warning.start()
    workflow = backend.local_app.workflow
    if not workflow._stage_project().spans:
        workflow.confirm_unit('METRE')
        source_project = json.loads(SEED.read_text(encoding='utf-8'))['project']
        rows = [dict(display_name=t['display_name'], x=t['position'][0],
                     y=t['position'][1], z=t['position'][2],
                     height_m=t['attributes']['height_m']) for t in source_project['towers']]
        workflow.save_manual_towers(dict(line_name='主线路', line_role='MAIN', towers=rows))
        emit('topology', basis='Operator-confirmed tower positions from project 2026.09.30-05', towers=rows)
    url = processing_url + '?warning_url=' + quote(warning_url, safe='')
    (ACCEPTANCE_ROOT / 'servers.json').write_text(json.dumps(dict(
        desktop=url, processing=processing_url, warning=warning_url, project=str(PROJECT)
    ), ensure_ascii=False, indent=2), encoding='utf-8')
    emit('servers', desktop=url, project=str(PROJECT))
    batch = workflow.batch_projection()
    if batch.get('status') not in {'COMPLETE', 'RUNNING'}:
        span_ids = [span.span_id for span in workflow._stage_project().spans]
        workflow.start_batch(span_ids, mode='REAL', execution_mode='SERIAL')
    marker = None
    try:
        while True:
            batch = workflow.batch_projection()
            current = (batch.get('status'), tuple((s.get('stage'), s.get('status')) for s in batch.get('spans', [])))
            if marker != current:
                emit('batch', status=batch.get('status'), stages=current[1], error=batch.get('error'))
                marker = current
            time.sleep(2)
    finally:
        warning.stop()
        backend.stop()


if __name__ == '__main__':
    main()
