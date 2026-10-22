"""Package only product files; validate every ZIP member after relocation."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def package(root, archive):
    root = root.resolve()
    root.relative_to(Path('D:/TreeLineWarning').resolve())
    files = sorted(p for p in root.rglob('*') if p.is_file()
        and p.relative_to(root).parts[0] not in ('项目数据', '运行日志', '导出结果', '用户配置', '用户设置', '运行缓存')
        and '__pycache__' not in p.parts and p.suffix != '.pyc'
        and p.name != 'SHA256文件清单.json'
        and p.name != 'MicrosoftEdgeWebView2RuntimeInstallerX64.exe')
    manifest = {p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size, sha256=digest(p)) for p in files}
    mp = root/'SHA256文件清单.json'
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    files.append(mp)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for index, p in enumerate(files):
            z.write(p, root.name+'/'+p.relative_to(root).as_posix())
            if index % 2000 == 0:
                print(f'Archived {index}/{len(files)}', flush=True)
        for directory in ('项目数据', '运行日志', '导出结果'):
            z.writestr(root.name+'/'+directory+'/', '')
    print(json.dumps(dict(files=len(files), size=archive.stat().st_size, sha256=digest(archive)),ensure_ascii=False),flush=True)


def verify(archive, target):
    target = target.resolve()
    target.relative_to(Path('D:/TreeLineWarning').resolve())
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            (target/member.filename).resolve().relative_to(target)
        z.extractall(target)
    roots=list(target.iterdir())
    assert len(roots)==1 and roots[0].is_dir()
    root=roots[0]
    manifest=json.loads((root/'SHA256文件清单.json').read_text(encoding='utf8'))
    for index,(name,expected) in enumerate(manifest.items()):
        path=root/name
        assert path.stat().st_size==expected['bytes'] and digest(path)==expected['sha256'], name
        if index%2000==0: print(f'Verified {index}/{len(manifest)}',flush=True)
    assert not list((root/'项目数据').rglob('*'))
    assert not list((root/'运行日志').rglob('*'))
    print(json.dumps(dict(passed=True,root=str(root),verifiedFiles=len(manifest)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--extract',type=Path)
    args=parser.parse_args()
    if args.source: package(args.source,args.archive)
    if args.extract: verify(args.archive,args.extract)
