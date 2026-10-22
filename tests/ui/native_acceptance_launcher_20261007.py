"""Local GUI acceptance launcher, NOT the final distributable software.

Uses the installed runtime with the exact staged source and the independent
acceptance project; it never overwrites the user's installed application.
"""
from pathlib import Path
import json
import os
import subprocess


def main():
    stage = Path('D:/TreeLineWarning/release_src_20260930')
    output = Path('D:/TreeLineWarning/双点云全功能复测/人工胸径及工单另存为20261007')
    pythonw = Path('D:/TreeLineWarning_一体化软件_20260926/TreeLineWarning一体化软件2.0_20260926/程序文件/.venv/Scripts/pythonw.exe')
    project = Path('D:/TreeLineWarning/双点云全功能复测/杆线胸径返回修复20261006/新建项目/双点云复测_20261006B完整导入')
    if not pythonw.is_file() or not (stage/'desktop/app.py').is_file():
        raise RuntimeError('Acceptance runtime or staged source unavailable')
    output.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(stage), PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PROJ_NETWORK='OFF')
    for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
        environment.pop(key, None)
    environment['NO_PROXY'] = '*'
    with (output/'native_gui_acceptance.log').open('ab') as log:
        process = subprocess.Popen([str(pythonw), '-m', 'desktop.app', '--project-root', str(project)],
            cwd=stage, env=environment, stdout=log, stderr=log, close_fds=True)
    (output/'native_gui_process.json').write_text(json.dumps(dict(pid=process.pid, source=str(stage),
        project=str(project), status='GUI_LAUNCHED_NOT_ACCEPTED')), encoding='utf8')


if __name__ == '__main__':
    main()
