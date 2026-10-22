"""Read-only pole neighbourhood inspection; output is outside project data."""
import argparse
from pathlib import Path
import numpy as np
from desktop.workflow import DesktopWorkflow
from core.stage_b.dbh_recovery import _context
from core.stage_b.dbh_evidence import confined
from core.stage_b.infrastructure import original_pole_query

parser = argparse.ArgumentParser()
parser.add_argument('project', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
class ReadOnlyWorkflow(DesktopWorkflow):
    def _log(self, *a, **kw):
        pass
workflow = ReadOnlyWorkflow(args.project)
result, project = workflow._current_result(), workflow._stage_project()
_, prep, _, origin = _context(args.project, result)
tower = project.towers[1]
base = np.asarray(tower.position)
height = float(tower.attributes['height_m'])
xyz, ids = original_pole_query(args.project,
    confined(args.project, prep['vegetation_role_tile_index_reference']),
    prep['ground_preparation']['source_identity'], base, height)
args.output.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.output/'pole_T002_original_inspection.npz', xyz=xyz, ids=ids, base=base, height=height)
# Optional numeric-evidence rendering is isolated in plot_pole_evidence.py;
# the application itself does not need a plotting dependency.
print({'original_points': len(xyz), 'base': base.tolist(), 'height': height}, flush=True)
