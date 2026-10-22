"""Check the actual distributable ZIP after a fresh Chinese/space relocation."""
from pathlib import Path
import json
import subprocess
from archive_portable_20261007 import verify, digest

archive = Path('D:/TreeLineWarning/交付20261007/TreeLineWarning_20261007_便携.zip')
target = Path('D:/TreeLineWarning/验收20261007补填便携/最终包 换目录 20261008')
verify(archive, target)
root = target/'TreeLineWarning_一体化_20261007'
exe = root/'TreeLineWarning_20261007.exe'
result = subprocess.run([str(exe), '--self-test'], cwd=root, timeout=180,
                        capture_output=True, text=True, encoding='utf8')
assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
manifest = json.loads((root/'SHA256文件清单.json').read_text('utf8'))
assert '程序文件/desktop/review_measurement_cache.py' in manifest
assert 'expected_analysis_run_id' in (root/'程序文件/web/static/manual_dbh.js').read_text('utf8')
assert 'return integratedContext' in (root/'程序文件/web/static/manual_dbh.js').read_text('utf8')
assert not list((root/'项目数据').rglob('*'))
report = dict(zip=str(archive), zip_bytes=archive.stat().st_size, zip_sha256=digest(archive),
              relocated_root=str(root), verified_files=len(manifest), exe_self_test_exit=0,
              passed=True, release_status='CANDIDATE_NOT_ALL_REQUIREMENTS_ACCEPTED',
              remaining=['original first LAS second pole fitting', 'native Windows work-order Save As',
                         'live CMA authorization and field-calibrated species/branch models'])
path = Path('D:/TreeLineWarning/验收20261007补填便携/最终包验证20261008.json')
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(report, ensure_ascii=False), flush=True)
