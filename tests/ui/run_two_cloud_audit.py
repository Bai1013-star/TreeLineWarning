"""Fresh independent real projects, never alter existing operator projects."""
import argparse
import json
import os
import time
from pathlib import Path

from desktop.app import BackendProcess, WarningBackend
from desktop.pdal_runtime import _configure_bundled_runtime
from desktop.project_management import create_empty_project, import_pointcloud
from desktop.workflow import DesktopWorkflow

APP = Path('D:/TreeLineWarning_一体化软件_20260926/TreeLineWarning一体化软件2.0_20260926')
OUT = Path('D:/TreeLineWarning/双点云全功能复测')
CASES = {
    'A': (APP / '原始点云/2026-08-07_16没54-10_PointCloud_colorized.las',
          APP / '项目数据/2026.09.30-04/stage-a-project.json'),
    'B': (Path('E:/treeline/9.20北京西站南路-1_colorized.las'),
          APP / '项目数据/2026.09.30-05/stage-a-project.json'),
}

def emit(event, **values):
    print(json.dumps(dict(event=event, **values), ensure_ascii=False), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--seed-project', type=Path, help='Existing operator-confirmed topology, read only')
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--project-name', required=True)
    parser.add_argument('--project-parent', type=Path, default=APP/'项目数据')
    parser.add_argument('--output-root', type=Path, default=OUT)
    parser.add_argument('--resource-root', type=Path, default=APP/'程序文件')
    args = parser.parse_args()
    source, seed_path = CASES[args.case]
    if args.seed_project is not None:
        seed_path = args.seed_project / 'stage-a-project.json'
    project = args.project_parent / args.project_name
    if project.resolve().parent != args.project_parent.resolve():
        raise ValueError('Project must be a direct child of designated audit parent')
    if not args.project_name.startswith('双点云复测_'):
        raise ValueError('Only explicitly prefixed independent audit projects allowed')
    output = args.output_root / args.case
    output.mkdir(parents=True, exist_ok=True)
    resource = args.resource_root
    _configure_bundled_runtime()
    if not project.exists():
        create_empty_project(project.parent, project.name)
        emit('import_start', case=args.case, source=str(source))
        import_pointcloud(project, source)
        emit('import_complete')
    workflow = DesktopWorkflow(project, resource_root=resource)
    if not workflow._stage_project().spans:
        workflow.confirm_unit('METRE')
        seed = json.loads(seed_path.read_text(encoding='utf-8'))['project']
        towers = [dict(display_name=t['display_name'], x=t['position'][0],
                       y=t['position'][1], z=t['position'][2],
                       height_m=t['attributes']['height_m']) for t in seed['towers']]
        workflow.save_manual_towers(dict(line_name='主线路', line_role='MAIN', towers=towers))
        emit('topology', basis='Prior operator-confirmed towers; not new field survey', towers=towers)
    if args.serve:
        from urllib.parse import quote
        processing = BackendProcess(project, resource)
        processing_url = processing.start()
        warning = WarningBackend(project, resource, processing)
        warning_url = warning.start()
        servers = dict(desktop=processing_url+'?warning_url='+quote(warning_url, safe=''),
                       processing=processing_url, warning=warning_url, project=str(project))
        (output/'servers.json').write_text(json.dumps(servers, ensure_ascii=False, indent=2), encoding='utf-8')
        emit('servers', **servers)
        try:
            while True:
                time.sleep(2)
        finally:
            warning.stop()
            processing.stop()
        return
    started = time.monotonic()
    workflow.start_batch([s.span_id for s in workflow._stage_project().spans], mode='REAL', execution_mode='SERIAL')
    marker = None
    while True:
        batch = workflow.batch_projection()
        compact = (batch['status'], batch.get('source_preparation'),
                   tuple((s.get('stage'), s.get('status')) for s in batch.get('spans', [])))
        if compact != marker:
            emit('analysis', case=args.case, elapsed=round(time.monotonic()-started),
                 status=batch['status'], source_preparation=batch.get('source_preparation'),
                 stages=compact[-1], error=batch.get('error'))
            marker = compact
        if batch['status'] in {'COMPLETE', 'FAILED', 'PARTIAL'}:
            break
        if time.monotonic()-started > 3600:
            raise TimeoutError('Audit exceeds one hour, see persisted batch state')
        time.sleep(3)
    (output/'batch.json').write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding='utf-8')
    assert batch['status'] == 'COMPLETE', batch.get('error')
    emit('finalize_start')
    finalized = workflow.finalize()
    projection = workflow.final_projection()
    export = workflow.export_prediction()
    report = dict(case=args.case, source=str(source), project=str(project),
                  elapsed_s=round(time.monotonic()-started), batch=batch,
                  finalized=finalized, trees=projection.get('trees', []), export=str(export))
    (output/'pipeline.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    emit('complete', trees=len(report['trees']), measured_dbh=sum(t.get('dbh_cm') is not None for t in report['trees']),
         seconds=report['elapsed_s'], export=str(export))

if __name__ == '__main__':
    main()
