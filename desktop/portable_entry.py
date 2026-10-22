"""Relative-path executable entry for the self-contained integrated release."""
import ctypes
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(sys.executable).resolve().parent
    program = root / '程序文件'
    runtime = program / 'runtime'
    python_root = program / 'python-runtime'
    python = python_root / ('python.exe' if '--self-test' in sys.argv else 'pythonw.exe')
    if not python.is_file() or not (program / 'portable_start.py').is_file():
        ctypes.windll.user32.MessageBoxW(None, '请先完整解压压缩包，不要单独移动 EXE。', 'TreeLineWarning', 16)
        return 1
    env = os.environ.copy()
    env.update(PYTHONHOME=str(python_root), PYTHONPATH=os.pathsep.join([str(program), str(program/'packages'), str(runtime/'Lib/site-packages')]),
               PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PROJ_NETWORK='OFF',
               TREELINE_PDAL_RUNTIME_ENV=str(runtime), TREELINE_PDAL_PROJ_DATA_ROOT=str(runtime/'Library/share/proj'))
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    env['NO_PROXY'] = 'localhost,127.0.0.1,::1'
    # Existing user PATH is not needed for Python or the native scientific runtime.
    env['PATH'] = os.pathsep.join([str(python_root), str(runtime/'Library/bin'),
                                str(Path(os.environ['SystemRoot'])/'System32'), os.environ['SystemRoot']])
    logs = root/'运行日志';logs.mkdir(exist_ok=True)
    with (logs/'portable-launch.log').open('ab') as log:
        child = subprocess.Popen([str(python), str(program/'portable_start.py'), *sys.argv[1:]],
                                 cwd=root, env=env, stdout=log, stderr=log, close_fds=True)
    if '--self-test' in sys.argv:
        return child.wait()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
