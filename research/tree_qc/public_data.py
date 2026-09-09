"""Pinned official two-site download; bounded ZIP inventory/extraction. No training."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
import urllib.request
import zipfile

ROOT = Path('D:/TreeQCData')
RESERVE = 35 * 1024**3
FILES = {
    'litchfield': (2331915728, '8e374d940d7e651484a84d7d63f18636'),
    'wytham': (3166118961, '8d7918ed1270d603efc83aae5a8777c1'),
}


def identity(site):
    size, md5 = FILES[site]
    return dict(site=site, record=16875688, name=site+'.zip', size=size,
                md5=md5, url=f'https://zenodo.org/api/records/16875688/files/{site}.zip/content')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def budget(additional):
    free = shutil.disk_usage(ROOT.anchor).free
    if free - additional < RESERVE:
        raise RuntimeError(f'DISK_GATE: free={free}, additional={additional}, reserve={RESERVE}')
    return free


def approved(path, action):
    if not path:
        raise RuntimeError('Supervisor --approval JSON required')
    evidence = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if evidence.get('gate_a') != 'PASS' or evidence.get('gate_b') != 'PASS':
        raise RuntimeError('Supervisor Gate A and B must both PASS')
    if evidence.get('record_id') != 16875688 or evidence.get('sites') != ['litchfield', 'wytham']:
        raise RuntimeError('Approval must pin record 16875688 and exactly ordered two sites')
    if evidence.get('allow_'+action) is not True:
        raise RuntimeError(f'Supervisor has not approved {action}')
    budget(sum(x[0] for x in FILES.values()) + evidence.get('max_total_expanded_bytes', 0)
           + evidence.get('reserve_training_bytes', 0))
    return evidence


def digests(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(4*1024**2):
            md5.update(chunk)
            sha.update(chunk)
    return dict(md5=md5.hexdigest(), sha256=sha.hexdigest(), bytes=path.stat().st_size)


def verify(path, spec):
    actual = digests(path)
    if actual['bytes'] != spec['size'] or actual['md5'] != spec['md5']:
        raise RuntimeError(f'INTEGRITY_FAILED: {path}: {actual}')
    save(path.with_suffix('.verified.json'), dict(identity=spec, actual=actual))
    return actual


def validate_response(status, content_range, offset, size):
    if status == 206:
        cr = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
        if not cr or tuple(map(int, cr.groups())) != (offset, size-1, size):
            raise RuntimeError('Invalid Content-Range; refusing append')
    elif status != 200 or offset:
        raise RuntimeError('Server ignored Range; refusing duplicate restart')


def expansion_cap(combined, approval):
    if combined > approval['max_total_expanded_bytes']:
        raise RuntimeError(f'EXPANSION_BUDGET_FAILED: {combined} bytes across both sites')


def download(site):
    spec = identity(site)
    dest = ROOT/'archives'/spec['name']
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix('.zip.part')
    provenance = dest.with_suffix('.source.json')
    if dest.exists():
        verify(dest, spec)
        return dest
    if part.exists():
        if not provenance.exists() or json.loads(provenance.read_text()) != spec:
            raise RuntimeError('Untrusted partial provenance; retained, not appended or installed')
    else:
        save(provenance, spec)
    for attempt in range(1, 7):
        offset = part.stat().st_size if part.exists() else 0
        if offset == spec['size']:
            verify(part, spec)
            os.replace(part, dest)
            return dest
        if offset > spec['size']:
            raise RuntimeError('Oversized partial retained for inspection')
        budget(spec['size']-offset)
        headers = {'User-Agent': 'TreeQC-research/1.0', 'Accept-Encoding': 'identity',
                   'Range': f'bytes={offset}-'}
        start, last, transferred = time.monotonic(), time.monotonic(), 0
        try:
            # Same pinned Zenodo file via its two official endpoints.
            url = (spec['url'] if attempt % 2 else
                   f'https://zenodo.org/records/16875688/files/{site}.zip?download=1')
            print(json.dumps(dict(site=site, attempt=attempt, url=url, resume_bytes=offset)), flush=True)
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as r:
                if not r.url.startswith('https://zenodo.org/'):
                    raise RuntimeError('Unexpected non-pinned redirect')
                validate_response(r.status, r.headers.get('Content-Range', ''), offset, spec['size'])
                if r.headers.get('Content-Encoding', 'identity') != 'identity':
                    raise RuntimeError('Unexpected content encoding')
                length = r.headers.get('Content-Length')
                if length is not None and int(length) != spec['size']-offset:
                    raise RuntimeError('Content-Length inconsistent with pinned identity')
                with part.open('ab') as f:
                    while chunk := r.read(1024**2):
                        if f.tell()+len(chunk) > spec['size']:
                            raise RuntimeError('Response exceeds pinned size')
                        f.write(chunk)
                        transferred += len(chunk)
                        if time.monotonic()-last >= 10:
                            f.flush()
                            event = dict(site=site, attempt=attempt, bytes=f.tell(), total=spec['size'],
                                         bytes_per_second=transferred/(time.monotonic()-start))
                            print(json.dumps(event), flush=True)
                            save(dest.with_suffix('.progress.json'), event)
                            budget(spec['size']-f.tell())
                            last = time.monotonic()
            if part.stat().st_size != spec['size']:
                raise OSError('Premature EOF')
        except (OSError, TimeoutError) as exc:
            print(json.dumps(dict(site=site, attempt=attempt, error=str(exc))), flush=True)
            if attempt == 6:
                raise
            time.sleep(min(5*attempt, 20))
    verify(part, spec)
    os.replace(part, dest)
    return dest


def read_ply_header(src, name):
    """Bounded helper; callers need separate approval to access file payloads."""
    header = bytearray()
    while len(header) < 65536:
        line = src.readline(min(4096, 65536-len(header)))
        if not line:
            raise RuntimeError(f'PLY header ended before end_header: {name}')
        header.extend(line)
        if line.strip() == b'end_header':
            break
    else:
        raise RuntimeError(f'PLY header too large: {name}')
    decoded = header.decode('ascii', errors='strict')
    if decoded.splitlines()[0] != 'ply':
        raise RuntimeError(f'Invalid PLY header: {name}')
    return decoded


def inventory(path, include_headers=False):
    rows, names, total, schemas = [], set(), 0, []
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename.replace('\\', '/')
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or ':' in name or '\x00' in name:
                raise RuntimeError(f'ZIP unsafe path: {name!r}')
            if any(x.endswith(('.', ' ')) or x.split('.')[0].upper() in
                   {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}
                   for x in p.parts):
                raise RuntimeError(f'ZIP Windows unsafe path: {name!r}')
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise RuntimeError(f'ZIP special file: {name}')
            key = str(p).casefold()
            if key in names or info.flag_bits & 1:
                raise RuntimeError(f'ZIP duplicate or encrypted entry: {name}')
            names.add(key)
            if info.file_size > max(info.compress_size, 1)*1000:
                raise RuntimeError(f'ZIP expansion ratio exceeds 1000: {name}')
            total += info.file_size
            rows.append(dict(name=name, bytes=info.file_size, compressed=info.compress_size,
                             crc32=f'{info.CRC:08x}', directory=info.is_dir()))
            if include_headers and name.lower().endswith('.ply'):
                # Read headers only, capped at 64 KiB; never load plot geometry.
                with z.open(info) as src:
                    decoded = read_ply_header(src, name)
                schemas.append(dict(name=name, header=decoded))
    result = dict(archive=str(path), entry_count=len(rows), extracted_bytes=total, entries=rows,
                  free_bytes=shutil.disk_usage(ROOT.anchor).free)
    save(path.with_suffix('.inventory.json'), result)
    if include_headers:
        save(path.with_suffix('.ply_headers.json'), dict(archive=str(path), schemas=schemas,
             coordinate_status='Header schema only; world-coordinate consistency and overlap require Gate C audit'))
    return result


def extract(path, listing):
    target = ROOT/'extracted'/path.stem
    marker = target/'.extraction_complete.json'
    if marker.exists():
        return target
    # Conservative on resume: reserve the full size even if some entries exist.
    budget(listing['extracted_bytes'])
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for row, info in zip(listing['entries'], z.infolist()):
            dest = target/row['name']
            if not dest.resolve().is_relative_to(target.resolve()):
                raise RuntimeError('Extraction target escaped root')
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            if dest.exists():
                raise RuntimeError(f'Partial extraction retained; manual review needed: {dest}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix(dest.suffix+'.extracting')
            with z.open(info) as src, temp.open('wb') as out:
                shutil.copyfileobj(src, out, 1024**2)
            if temp.stat().st_size != info.file_size:
                raise RuntimeError('Extracted size mismatch')
            os.replace(temp, dest)
    save(marker, dict(archive=str(path), extracted_bytes=listing['extracted_bytes']))
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inspect', 'download', 'extract'], nargs='?', default='inspect')
    parser.add_argument('--approval')
    parser.add_argument('--site', choices=list(FILES))
    args = parser.parse_args()
    if args.action != 'inspect':
        approval = approved(args.approval, args.action)
    if args.action == 'extract':
        # Inspect all approved archives before mutating extraction directories.
        combined = 0
        for site in FILES:
            archive = ROOT/'archives'/(site+'.zip')
            verify(archive, identity(site))
            combined += inventory(archive)['extracted_bytes']
        expansion_cap(combined, approval)
        budget(combined + approval['reserve_training_bytes'])
    sites = [args.site] if args.site else list(FILES)
    for site in sites:
        spec = identity(site)
        archive = ROOT/'archives'/spec['name']
        if args.action == 'download':
            archive = download(site)
        if archive.exists():
            verify(archive, spec)
            listing = inventory(archive)
            print(json.dumps({k:v for k,v in listing.items() if k != 'entries'}), flush=True)
            if args.action == 'extract':
                print(extract(archive, listing), flush=True)
        else:
            print(json.dumps(dict(identity=spec, archive_exists=False)), flush=True)


if __name__ == '__main__':
    main()
