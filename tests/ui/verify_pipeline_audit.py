"""Fresh real-LAS run in isolated project; originals and user projects untouched."""
import json
import os
import time
from pathlib import Path
from desktop.project_management import create_empty_project, import_pointcloud
from desktop.workflow import DesktopWorkflow
from desktop.pdal_runtime import _configure_bundled_runtime

source=Path('E:/treeline/9.20北京西站南路-1_colorized.las')
out=Path('D:/TreeLineWarning/verification_real_20260930_weststation')
root=out/os.environ.get('AUDIT_PROJECT_NAME', '20261001_从原始点云重新分析')
resource=Path(__file__).resolve().parents[2]
_configure_bundled_runtime()
if not root.exists():
    create_empty_project(out,root.name)
    import_pointcloud(root,source)
    print('Fresh project created and original LAS inspected',flush=True)
w=DesktopWorkflow(root,resource_root=resource)
if not w._stage_project().spans:
    w.confirm_unit('METRE')
    seed=json.loads((out/'北京西站南路_独立实测验收'/'stage-a-project.json').read_text(encoding='utf-8'))['project']
    rows=[dict(display_name=t['display_name'],x=t['position'][0],y=t['position'][1],z=t['position'][2],height_m=t['attributes']['height_m']) for t in seed['towers']]
    w.save_manual_towers(dict(line_name='主线路',line_role='MAIN',towers=rows))
span_ids=[s.span_id for s in w._stage_project().spans]
started=time.monotonic()
w.start_batch(span_ids,mode='REAL',execution_mode='SERIAL')
last=None
while True:
    batch=w.batch_projection()
    marker=(batch['status'],str(batch.get('spans',[])))
    if marker!=last:
        print(json.dumps(dict(status=batch['status'],elapsed_s=round(time.monotonic()-started),spans=[dict(id=s['span_id'],stage=s.get('stage'),status=s.get('status'),failure=s.get('failure')) for s in batch.get('spans',[])],error=batch.get('error')),ensure_ascii=False),flush=True)
        last=marker
    if batch['status'] in {'COMPLETE','FAILED','PARTIAL'}:break
    if time.monotonic()-started>1800:raise TimeoutError('Pipeline audit exceeded 30 minutes')
    time.sleep(5)
assert batch['status']=='COMPLETE',batch
finalized=w.finalize()
projection=w.final_projection()
export=w.export_prediction()
audit=dict(source=str(source),project=str(root),elapsed_s=round(time.monotonic()-started),batch=batch,finalized=finalized,export=str(export),trees=projection.get('trees'),topology_basis='Prior operator-confirmed tower coordinates; no new field survey')
(out/os.environ.get('AUDIT_REPORT_NAME', 'fresh_pipeline_audit_20261001.json')).write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(passed=True,trees=len(projection.get('trees',[])),export=str(export)),ensure_ascii=False),flush=True)
