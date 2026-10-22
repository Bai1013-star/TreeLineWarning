"""Compare actual native Save As workbook and deployed repaired source."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from openpyxl import load_workbook

audit = Path('D:/TreeLineWarning/双点云全功能复测')
installed = Path('D:/TreeLineWarning_一体化软件_20260926/TreeLineWarning一体化软件2.0_20260926/程序文件')
staged = Path('D:/TreeLineWarning/release_src_20260930')

def workbook_cells(path):
    workbook = load_workbook(path, read_only=True, data_only=False)
    result = {sheet.title: list(sheet.values) for sheet in workbook}
    workbook.close()
    return result

native = audit / 'B/native_processed_after_fix.xlsx'
expected = audit / 'B/processed.xlsx'
actual_cells, expected_cells = workbook_cells(native), workbook_cells(expected)
assert actual_cells == expected_cells, 'Native Save As values changed'
files = ['desktop/wind_contact.py', 'api/prediction_api.py', 'desktop/warning_bridge.py',
         'web/templates/index.html', 'desktop/server.py', 'algorithms/prediction/cma_weather.py']
hashes = []
for name in files:
    origin = hashlib.sha256((staged / name).read_bytes()).hexdigest()
    deployed = hashlib.sha256((installed / name).read_bytes()).hexdigest()
    assert origin == deployed, name
    hashes.append(dict(file=name, sha256=deployed, matches=True))
env = os.environ.copy()
env['PYTHONPATH'] = str(installed)
env['PYTHONDONTWRITEBYTECODE'] = '1'
env['PYTHONUTF8'] = '1'
health = subprocess.run([str(installed / '.venv/Scripts/python.exe'), '-m', 'desktop.app',
                         '--dependency-health'], cwd=installed, env=env,
                        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
assert health.returncode == 0, health.stdout + health.stderr
report = dict(native_save=dict(path=str(native), bytes=native.stat().st_size,
                 all_workbook_cells_match=True,
                 scope='Real Windows Save As, held open beyond five minutes; UI confirmed saved without timeout'),
              deployed_source=hashes,
              dependency_health=dict(exit_code=health.returncode, stdout=health.stdout, stderr=health.stderr))
(audit / 'final_delivery_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
print(json.dumps(report, ensure_ascii=False), flush=True)
