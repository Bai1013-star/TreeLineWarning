"""Re-fit real source representatives and compare top conductors to lower cables."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
product=Path('D:/TreeLineWarning/交付20261021/TreeLineWarning_一体化_20261021/程序文件')
sys.path.insert(1,str(product/'packages'))
import numpy as np
from algorithms.spans.wire_v2 import reconstruct_slice_tracks,CONFIRMED
from algorithms.spans.wire import _parallel_family_interior_rescue,_strand_approaches_pole
from core.config.fast_baseline import FastBaselineConfig

parser=argparse.ArgumentParser()
parser.add_argument('--project',type=Path,required=True)
parser.add_argument('--batch',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
project=json.loads((args.project/'stage-a-project.json').read_text('utf8'))['project']
towers=project['towers'][:2]
length=float(np.linalg.norm(np.array(towers[1]['position'][:2])-towers[0]['position'][:2]))
span=SimpleNamespace(start_pole_top_z_m=sum([towers[0]['position'][2],towers[0]['attributes']['height_m']]),
                     end_pole_top_z_m=sum([towers[1]['position'][2],towers[1]['attributes']['height_m']]))
path=next((args.project/'artifacts/project-runs/batches'/args.batch).rglob('span_wire_diagnostics.npz'))
arrays=np.load(path)
s,u,z=arrays['s_m'],arrays['u_m'],arrays['z_m']
config=FastBaselineConfig()
tracks=reconstruct_slice_tracks(np.flatnonzero(arrays['candidate_mask']),s,u,z,arrays['raw_count'],length,config)
published=[]
for index,track in enumerate(tracks):
    rows=track.inlier_rows
    if not len(rows) or track.status!=CONFIRMED:continue
    start=_strand_approaches_pole(s[rows],u[rows],z[rows],span,0.,config)
    end=_strand_approaches_pole(s[rows],u[rows],z[rows],span,length,config)
    family=_parallel_family_interior_rescue(track,tracks,config)
    if not(start or end or family):continue
    if not(start and end) and track.coverage_ratio<config.wire_confirmed_strong_interior_coverage_ratio:continue
    published.append(dict(track=index,median_u_m=track.median_u_m,median_z_m=float(np.median(z[rows])),
                          p95_residual_m=track.p95_residual_m,coverage=track.coverage_ratio))
assert len(published)==3,published
assert np.ptp([item['median_z_m'] for item in published])<.75,published
assert min(item['median_z_m'] for item in published)>10,published
result=dict(passed=True,published_conductors=published,lower_cable_not_used_as_phase=True,
            evidence=str(path),height_spread_m=float(np.ptp([item['median_z_m'] for item in published])))
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf8')
print(json.dumps(result,ensure_ascii=False))
