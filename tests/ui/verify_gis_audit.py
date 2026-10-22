"""Canonical GIS import smoke test in an isolated synthetic project."""
import json
import os
from pathlib import Path

from desktop.project_management import create_empty_project, import_gis_file
from core.stage_a import load_project

out = Path(os.environ.get('REAL_OUTPUT','D:/TreeLineWarning/verification_real_20260930_weststation'))
root = out / '20261001_GIS合成导入验收'
if not root.exists():
    create_empty_project(out, root.name)
source = out / 'audit_gis_towers.csv'
import_gis_file(root, source, crs='EPSG:32650', linear_unit='m')
first = load_project(root)
assert len(first.towers) == 2
assert len(first.spans) == 1
first_ids = [t.tower_id for t in first.towers]
import_gis_file(root, source, crs='EPSG:32650', linear_unit='m')
second = load_project(root)
assert [t.tower_id for t in second.towers] == first_ids
assert len(second.spans) == 1
assert all(t.position_crs.horizontal_crs == 'EPSG:32650' for t in second.towers)
audit = dict(passed=True, project=str(root), source=str(source), towers=2,
             lines=len(second.lines), spans=1, repeated_import_duplicates=False,
             crs='EPSG:32650', scope='Synthetic CSV actual import; not a field GIS/native dialog accuracy test')
(out / 'gis_import_audit_20261001.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(audit, ensure_ascii=False))
