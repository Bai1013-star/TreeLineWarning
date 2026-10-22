"""Read-only raw pole diagnostics; writes no project artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
from desktop.workflow import DesktopWorkflow
from core.stage_b.dbh_recovery import _context
from core.stage_b.dbh_evidence import original_raw_query, confined
from core.stage_b.infrastructure import protected_infrastructure, original_pole_query

parser = argparse.ArgumentParser()
parser.add_argument('project', type=Path)
args = parser.parse_args()
class ReadOnlyWorkflow(DesktopWorkflow):
    def _log(self, *args, **kwargs):
        pass

wf = ReadOnlyWorkflow(args.project)
result, project = wf._current_result(), wf._stage_project()
_, prep, _, origin = _context(args.project, result)
for tower in project.towers:
    base = np.asarray(tower.position)
    height = float(tower.attributes['height_m'])
    raw, idx = original_pole_query(args.project, confined(args.project, prep['vegetation_role_tile_index_reference']),
                                 prep['ground_preparation']['source_identity'], base, height)
    near = (np.linalg.norm(raw[:,:2]-base[:2],axis=1)<=1.2) & (raw[:,2]>=base[2]+.25)
    candidate = raw[near]
    cells = np.floor(candidate[:,:2]/.15).astype(np.int64)
    levels = np.floor((candidate[:,2]-base[2])/.4).astype(np.int64)
    unique = np.unique(np.column_stack((cells,levels)),axis=0)
    columns, counts = np.unique(unique[:,:2],axis=0,return_counts=True)
    ranked = np.lexsort((np.linalg.norm((columns+.5)*.15-base[:2],axis=1),-counts))
    center = (columns[ranked[0]]+.5)*.15
    rows = raw[np.linalg.norm(raw[:,:2]-center,axis=1)<=.35]
    sections = []
    for low in np.arange(base[2]+.25,base[2]+height,.4):
        band = rows[(rows[:,2]>=low)&(rows[:,2]<low+.4)]
        if len(band)>=8:
            sections.append([round(float(np.median(band[:,2])),3),
                             *np.round(np.median(band[:,:2],axis=0),3).tolist(),len(band)])
    print(json.dumps({'pole':tower.display_name,'base':base.tolist(),'height':height,
                      'column_seed':center.tolist(),'sections':sections},ensure_ascii=False))
data = protected_infrastructure(args.project,result,project)
print(json.dumps({k:v for k,v in data.items() if k not in {'xyz','original_index','pole_segments'}},ensure_ascii=False))
print(json.dumps([{k:v for k,v in pole.items() if k!='xyz'} for pole in data['pole_segments']],ensure_ascii=False))
