"""Snapshot the tested integrated source and a complete relative CPython runtime.

Build into an absent destination only. Never deletes or overwrites user releases.
"""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--installed-program',type=Path,required=True)
    parser.add_argument('--python-base',type=Path,required=True)
    parser.add_argument('--version-tag',default='20261011')
    args=parser.parse_args();dst=args.destination.resolve();installed=args.installed_program.resolve()
    dst.relative_to(Path('D:/TreeLineWarning').resolve())
    dst.mkdir(parents=True,exist_ok=False);program=dst/'程序文件';program.mkdir()
    ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache','node_modules','.git')
    for name in ('algorithms','api','core','desktop','web','viewer_poc','models','data','scripts'):
        shutil.copytree(ROOT/name,program/name,ignore=ignore)
    for name in ('launch.py','portable_start.py','desktop_launcher.py','pyproject.toml'):
        shutil.copy2(ROOT/name,program/name)
    shutil.copy2(ROOT/'tests/ui/serve_offline_audit.py',program/'scripts/portable_audit_server.py')
    print('Source snapshot complete',flush=True)
    shutil.copytree(installed/'.pdal-runtime',program/'runtime',ignore=ignore)
    print('Relative CPython / PDAL snapshot complete',flush=True)
    python_root=program/'python-runtime';python_root.mkdir()
    for item in args.python_base.iterdir():
        if item.is_file():shutil.copy2(item,python_root/item.name)
    for name in ('Lib','DLLs'):
        shutil.copytree(args.python_base/name,python_root/name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','site-packages'))
    shutil.copytree(installed/'.venv/Lib/site-packages',program/'packages',ignore=ignore)
    print('Tested wheels copied',flush=True)
    for name in ('项目数据','导出结果','运行日志','使用说明'):(dst/name).mkdir()
    build=dst.parent/(dst.name+'_构建');build.mkdir(exist_ok=False)
    executable_name=f'TreeLineWarning_{args.version_tag}'
    subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm','--onefile','--windowed',
                    '--name',executable_name,'--distpath',str(dst),'--workpath',str(build/'work'),
                    '--specpath',str(build),str(ROOT/'desktop/portable_entry.py')],check=True)
    print('Launcher compiled: '+str(dst/(executable_name+'.exe')),flush=True)


if __name__=='__main__':main()
