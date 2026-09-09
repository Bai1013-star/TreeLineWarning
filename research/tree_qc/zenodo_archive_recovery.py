"""Bounded official archive resume and offline integrity verification; never extracts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

from public_data import FILES, RESERVE, digests, inventory, save

DEST = Path('D:/TreeQCData/downloads')
LOG = Path('D:/TreeQCTraining/logs/zenodo_recovery_20260909')

ZIP_METHOD_NAMES = {0: 'Stored', 1: 'Shrink', 2: 'Reduce factor 1', 3: 'Reduce factor 2',
                    4: 'Reduce factor 3', 5: 'Reduce factor 4', 6: 'Implode',
                    8: 'Deflate', 9: 'Deflate64', 12: 'BZIP2', 14: 'LZMA',
                    20: 'Zstandard (deprecated ID)', 93: 'Zstandard', 95: 'XZ',
                    98: 'PPMd'}


def stdlib_supports(method):
    # This checks the running interpreter and optional decoder modules, not
    # merely a hard-coded version number. No archive payload is read here.
    try:
        zipfile._get_decompressor(method)
        return True
    except (NotImplementedError, RuntimeError):
        return False


def external_verifier():
    """Find an actual executable, never infer availability from associations."""
    for name in ('7z.exe', '7zz.exe', 'WinRAR.exe', 'rar.exe'):
        candidates = [shutil.which(name)]
        for variable in ('ProgramFiles', 'ProgramFiles(x86)', 'LOCALAPPDATA'):
            base = os.environ.get(variable)
            if base:
                for folder in ('7-Zip', 'WinRAR', 'Programs/7-Zip', 'Programs/WinRAR'):
                    candidates.append(str(Path(base)/folder/name))
        if os.name == 'nt':
            import winreg
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                    try:
                        with winreg.OpenKey(hive, 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\'+name,
                                            0, winreg.KEY_READ | view) as key:
                            candidates.append(winreg.QueryValueEx(key, None)[0].strip('"'))
                    except OSError:
                        pass
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return dict(executable=str(Path(candidate).resolve()),
                            kind='7z' if name.lower().startswith('7z') else
                            'winrar' if name.lower() == 'winrar.exe' else 'rar')
    return None


def external_crc_test(path, tool):
    """Full test command only: no output directory, extraction or modification."""
    LOG.mkdir(parents=True, exist_ok=True)
    stem = LOG/(path.name+'.crc.'+str(time.time_ns()))
    if tool['kind'] == 'winrar':
        quoted = tool['executable'].replace("'", "''")
        version_args = ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                        "[System.Diagnostics.FileVersionInfo]::GetVersionInfo('"+quoted+"').FileVersion"]
    else:
        version_args = [tool['executable'], 'i'] if tool['kind'] == '7z' else [tool['executable'], '-?']
    version = subprocess.run(version_args, capture_output=True, text=True, errors='replace', timeout=30)
    Path(str(stem)+'.version.stdout.txt').write_text(version.stdout, encoding='utf-8')
    Path(str(stem)+'.version.stderr.txt').write_text(version.stderr, encoding='utf-8')
    # Empty password disables interactive password prompts; encrypted entries
    # have already been rejected by central-directory inventory.
    args = ([tool['executable'], 't', '-bd', '-y', '-p', '--', str(path.resolve())]
            if tool['kind'] == '7z' else
            [tool['executable'], 't', '-cfg-', '-y', '-p-',
             *(['-ibck', '-ilog'+str(stem)+'.errors.txt'] if tool['kind'] == 'winrar' else []),
             str(path.resolve())])
    startup = None
    if os.name == 'nt':
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
    started = time.monotonic()
    environment = {key: value for key, value in os.environ.items() if key.upper() != 'RAR'}
    with Path(str(stem)+'.stdout.txt').open('w', encoding='utf-8') as stdout, \
            Path(str(stem)+'.stderr.txt').open('w', encoding='utf-8') as stderr:
        result = subprocess.run(args, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                timeout=7200, startupinfo=startup, env=environment)
    evidence = dict(**tool, command=args, exit_code=result.returncode,
                    seconds=time.monotonic()-started, version_output=version.stdout+version.stderr,
                    version_exit_code=version.returncode, log_prefix=str(stem),
                    inherited_rar_environment_removed=True,
                    config_disabled=tool['kind'] in ('winrar', 'rar'))
    save(Path(str(stem)+'.json'), evidence)
    if result.returncode != 0:
        raise RuntimeError(f'STOP_ARCHIVE_INTEGRITY_FAILED: external CRC test exit={result.returncode}; {stem}')
    return evidence


def refresh_metadata():
    """Refresh online transfer identities; offline verify never calls this."""
    LOG.mkdir(parents=True, exist_ok=True)
    snapshots = Path(__file__).resolve().parent/'reports'
    fetched = []
    url = 'https://zenodo.org/api/records/16875688'
    for name in ('metadata', 'files'):
        stamp = time.time_ns()
        target = LOG/f'{name}.refresh.{stamp}.json'
        command = ['curl.exe', '--location', '--proto', '=https', '--proto-redir', '=https',
                   '--fail', '--silent', '--show-error', '--retry', '2', '--retry-all-errors',
                   '--retry-max-time', '60', '--connect-timeout', '10', '--max-time', '20',
                   '--user-agent', 'TreeQC-research/1.0', '--dump-header', str(target)+'.headers',
                   '--output', str(target), url]
        response = subprocess.run(command, capture_output=True, text=True, timeout=90)
        Path(str(target)+'.log').write_text(response.stdout+response.stderr)
        if response.returncode:
            raise RuntimeError('ZENODO_METADATA_REFRESH_BLOCKED; retained all existing bytes')
        value = json.loads(target.read_text())
        fetched.append((name, target, value))
        if name == 'metadata':
            if value['id'] != 16875688:
                raise RuntimeError('RECORD_IDENTITY_CHANGED')
            url = value['links']['files']
            if url != 'https://zenodo.org/api/records/16875688/files':
                raise RuntimeError('OFFICIAL_ENDPOINT_CHANGED: Supervisor review required')
    record, files = fetched[0][2], fetched[1][2]
    for site, (size, md5) in FILES.items():
        for collection in (record['files'], files['entries']):
            item = next(x for x in collection if x['key'] == site+'.zip')
            if item['size'] != size or item['checksum'] != 'md5:'+md5:
                raise RuntimeError('ARCHIVE_IDENTITY_CHANGED: Supervisor review required')
    for name, target, _ in fetched:
        shutil.copyfile(target, snapshots/f'zenodo_archive_recovery_current_{name}.json')


def final_range_valid(header, start, end, total):
    """Only the last HTTP response may authorize payload append."""
    blocks = re.split(r'(?mi)^HTTP/', header)
    if len(blocks) < 2:
        return False
    final = 'HTTP/'+blocks[-1]
    if not re.match(r'HTTP/\S+ 206(?:\s|$)', final, re.I):
        return False
    ranges = re.findall(r'(?mi)^content-range:\s*bytes (\d+)-(\d+)/(\d+)\s*$', final)
    return len(ranges) == 1 and tuple(map(int, ranges[0])) == (start, end, total)


def bind_source(path, expected, existing_data=False):
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise RuntimeError(f'PROVENANCE_MISMATCH: {path}; retained')
    elif existing_data:
        raise RuntimeError(f'PROVENANCE_MISSING: {path}; retained')
    else:
        save(path, expected)


def require_litchfield(spec):
    if spec['name'] == 'wytham.zip':
        first = next(x for x in specifications() if x['name'] == 'litchfield.zip')
        path = DEST/first['name']
        if not path.is_file():
            raise RuntimeError('LITCHFIELD_VERIFICATION_REQUIRED_BEFORE_WYTHAM')
        checked(path, first)


def finish_if_complete(part, final, spec):
    if part.exists() and part.stat().st_size == spec['size']:
        result = checked(part, spec)
        os.replace(part, final)
        return result
    return None


def specifications():
    # Frozen snapshots from this recovery survive external log relocation.
    snapshots = Path(__file__).resolve().parent/'reports'
    record = json.loads((snapshots/'zenodo_archive_recovery_current_metadata.json').read_text())
    files = json.loads((snapshots/'zenodo_archive_recovery_current_files.json').read_text())
    assert record['id'] == 16875688
    result = []
    for site, (size, md5) in FILES.items():
        name = site+'.zip'
        entry = next(x for x in files['entries'] if x['key'] == name)
        original = next(x for x in record['files'] if x['key'] == name)
        assert entry['size'] == original['size'] == size
        assert entry['checksum'] == original['checksum'] == 'md5:'+md5
        url = entry['links']['content']
        assert url == f'https://zenodo.org/api/records/16875688/files/{name}/content'
        result.append(dict(name=name, size=size, md5=md5, url=url,
                           file_link=entry['links']['self'], record=record['id']))
    return result


def checked(path, spec):
    if path.stat().st_size != spec['size']:
        raise RuntimeError(f'STOP_ARCHIVE_INTEGRITY_FAILED: WRONG_SIZE: {path}: {path.stat().st_size}/{spec["size"]}')
    actual = digests(path)
    if actual['md5'] != spec['md5']:
        raise RuntimeError(f'STOP_ARCHIVE_INTEGRITY_FAILED: MD5_MISMATCH: {path}; retained; extraction prohibited')
    listing = inventory(path)
    with zipfile.ZipFile(path) as archive:
        methods = sorted({info.compress_type for info in archive.infolist()})
        supported = all(stdlib_supports(method) for method in methods)
        preflight = dict(identity=spec, actual_bytes=path.stat().st_size, actual=actual,
                         central_directory='PASS', compression_method_ids=methods,
                         compression_method_names={str(m): ZIP_METHOD_NAMES.get(m, 'Unknown') for m in methods},
                         python_stdlib_supported=supported, file_count=listing['entry_count'])
        save(DEST/(spec['name']+'.preflight.json'), preflight)
        print(json.dumps(preflight), flush=True)
        if supported:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f'STOP_ARCHIVE_INTEGRITY_FAILED: ZIP_CRC_FAILED: {bad}')
            crc_evidence = dict(implementation='python zipfile.testzip', exit_code=0)
        else:
            tool = external_verifier()
            if tool is None:
                raise RuntimeError('PYTHON_STDLIB_LIMITATION: no external full CRC verifier available; archive not accepted or declared corrupt')
            crc_evidence = external_crc_test(path, tool)
    report = dict(identity=spec, actual=actual, zip_crc='PASS',
                  actual_bytes=path.stat().st_size, size='PASS', official_md5='PASS', central_directory='PASS',
                  compression_method_ids=methods, compression_method_names=preflight['compression_method_names'],
                  python_stdlib_supported=supported,
                  python_stdlib_status='SUPPORTED' if supported else 'PYTHON_STDLIB_LIMITATION',
                  full_decompression_crc_test='PASS', crc_evidence=crc_evidence,
                  expanded_bytes=listing['extracted_bytes'],
                  compressed_bytes=sum(x['compressed'] for x in listing['entries']),
                  file_count=sum(not x['directory'] for x in listing['entries']))
    save(DEST/(spec['name']+'.verified.json'), report)
    print(json.dumps(report), flush=True)
    return report


def transfer(spec):
    require_litchfield(spec)
    final, part = DEST/spec['name'], DEST/(spec['name']+'.part')
    source = DEST/(spec['name']+'.source.json')
    if final.exists():
        return checked(final, spec)
    if part.exists() and (not source.exists() or json.loads(source.read_text()) != spec):
        raise RuntimeError('Untrusted partial retained; cannot append')
    save(source, spec)
    # Probe was freshly requested from metadata content URL; promote only exact 206 prefix.
    probe = LOG/(spec['name'].removesuffix('.zip')+'.probe')
    headers = LOG/(spec['name'].removesuffix('.zip')+'_range.headers')
    if not part.exists() and probe.exists() and headers.exists():
        if final_range_valid(headers.read_text(), 0, 1048575, spec['size']) and probe.stat().st_size == 1048576:
            shutil.copyfile(probe, part)
    for attempt in range(1, 13):
        offset = part.stat().st_size if part.exists() else 0
        if offset == spec['size']:
            return finish_if_complete(part, final, spec)
        if offset > spec['size']:
            raise RuntimeError('Oversized partial retained')
        if shutil.disk_usage(DEST.anchor).free - (spec['size']-offset) < RESERVE:
            raise RuntimeError('DISK_GUARD_FAILED')
        cmd = ['curl.exe', '--location', '--proto', '=https', '--proto-redir', '=https',
               '--fail', '--silent', '--show-error', '--continue-at', '-',
               '--retry', '2', '--retry-all-errors', '--retry-max-time', '320',
               '--connect-timeout', '15', '--max-time', '300',
               '--speed-time', '40', '--speed-limit', '1024',
               '--user-agent', 'TreeQC-research/1.0', '--header', 'Accept-Encoding: identity',
               '--dump-header', str(LOG/f'{spec["name"]}.attempt{attempt}.headers'),
               '--output', str(part), '--write-out',
               'HTTP=%{http_code} bytes=%{size_download} speed=%{speed_download} seconds=%{time_total} error=%{errormsg}', spec['url']]
        start = time.monotonic()
        with (LOG/f'{spec["name"]}.attempt{attempt}.log').open('w') as out:
            process = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    time.sleep(10)
                    now = part.stat().st_size if part.exists() else 0
                    print(json.dumps(dict(file=spec['name'], attempt=attempt, bytes=now,
                                          total=spec['size'], speed=(now-offset)/(time.monotonic()-start))), flush=True)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
        now = part.stat().st_size if part.exists() else 0
        complete = finish_if_complete(part, final, spec)
        if complete is not None:
            return complete
        if now == offset:
            raise RuntimeError('ZENODO_ARCHIVE_TRANSFER_BLOCKED: no progress in bounded curl retries')
        if now != spec['size']:
            time.sleep(min(2**attempt, 30))
    raise RuntimeError('ZENODO_ARCHIVE_TRANSFER_BLOCKED: bounded transfer budget exhausted')


def range_piece(spec, start, end):
    folder = DEST/(spec['name']+'.ranges')
    folder.mkdir(exist_ok=True)
    piece = folder/f'{start}-{end}.part'
    expected = dict(identity=spec, start=start, end=end)
    bind_source(Path(str(piece)+'.source.json'), expected, piece.exists())
    for attempt in range(3):
        # A prior interrupted nonce payload can be consumed only with its own
        # matching source record and a final 206 for the exact remaining range.
        for old in sorted(folder.glob(f'{start}-{end}.attempt*.payload.*')):
            source_path = Path(str(old)+'.source.json')
            if old.name.endswith('.source.json') or not source_path.exists():
                continue
            evidence = json.loads(source_path.read_text())
            current = start+(piece.stat().st_size if piece.exists() else 0)
            if evidence.get('identity') != spec or evidence.get('start') != current or evidence.get('end') != end:
                continue
            old_head = Path(evidence['header'])
            n = old.stat().st_size
            if old_head.exists() and 0 < n <= end-current+1 and final_range_valid(old_head.read_text(), current, end, spec['size']):
                with piece.open('ab') as out, old.open('rb') as src:
                    shutil.copyfileobj(src, out, 1024**2)
                    out.flush()
                    os.fsync(out.fileno())
                old.unlink()
        have = piece.stat().st_size if piece.exists() else 0
        if have == end-start+1:
            return piece
        if have > end-start+1:
            raise RuntimeError('Oversized range piece')
        actual_start = start+have
        stem = folder/f'{start}-{end}.attempt{attempt}'
        payload, head = Path(str(stem)+'.payload'), Path(str(stem)+'.headers')
        # Never overwrite prior interrupted bytes or evidence.
        nonce = time.time_ns()
        payload, head = Path(str(payload)+f'.{nonce}'), Path(str(head)+f'.{nonce}')
        save(Path(str(payload)+'.source.json'), dict(identity=spec, start=actual_start, end=end, header=str(head)))
        cmd = ['curl.exe', '--location', '--proto', '=https', '--proto-redir', '=https',
               '--fail', '--silent', '--show-error', '--range', f'{actual_start}-{end}',
               '--max-filesize', str(end-actual_start+1),
               '--connect-timeout', '15', '--max-time', '120',
               '--speed-time', '40', '--speed-limit', '1024',
               '--user-agent', 'TreeQC-research/1.0', '--header', 'Accept-Encoding: identity',
               '--dump-header', str(head), '--output', str(payload), '--write-out',
               'HTTP=%{http_code} bytes=%{size_download} speed=%{speed_download} seconds=%{time_total} error=%{errormsg}', spec['url']]
        before = time.monotonic()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            output, _ = process.communicate(timeout=140)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        Path(str(head)+'.log').write_text(output)
        header = head.read_text().lower() if head.exists() else ''
        n = payload.stat().st_size if payload.exists() else 0
        valid = final_range_valid(header, actual_start, end, spec['size']) and n <= end-actual_start+1
        if valid and n:
            with piece.open('ab') as out, payload.open('rb') as src:
                shutil.copyfileobj(src, out, 1024**2)
                out.flush()
                os.fsync(out.fileno())
            # Delete only this invocation's validated scratch after durable append.
            payload.unlink()
        print(json.dumps(dict(file=spec['name'], range=[actual_start,end], valid=bool(valid),
                              received=n, seconds=time.monotonic()-before, curl=output.strip())), flush=True)
        if piece.exists() and piece.stat().st_size == end-start+1:
            return piece
        time.sleep(2**(attempt+1))
    raise RuntimeError(f'ZENODO_ARCHIVE_TRANSFER_BLOCKED: bounded range attempts failed {start}-{end}')


def range_transfer(spec, chunk_mib=16):
    require_litchfield(spec)
    final, part = DEST/spec['name'], DEST/(spec['name']+'.part')
    source = DEST/(spec['name']+'.source.json')
    if final.exists():
        return checked(final, spec)
    ranges_dir = DEST/(spec['name']+'.ranges')
    old_ranges = ranges_dir.exists() and any(ranges_dir.iterdir())
    bind_source(source, spec, part.exists() or old_ranges)
    while (offset := part.stat().st_size if part.exists() else 0) < spec['size']:
        if shutil.disk_usage(DEST.anchor).free - (spec['size']-offset) - 64*1024**2 < RESERVE:
            raise RuntimeError('DISK_GUARD_FAILED')
        chunk_bytes = chunk_mib*1024**2
        ranges = [(start, min(start+chunk_bytes, spec['size'])-1)
                  for start in range(offset, min(offset+4*chunk_bytes, spec['size']), chunk_bytes)]
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            pieces = list(pool.map(lambda bounds: range_piece(spec, *bounds), ranges))
        for (start, end), piece in zip(ranges, pieces):
            assert (part.stat().st_size if part.exists() else 0) == start
            assert piece.stat().st_size == end-start+1
            with part.open('ab') as out, piece.open('rb') as src:
                shutil.copyfileobj(src, out, 1024**2)
                out.flush()
                os.fsync(out.fileno())
            piece.unlink()
        print(json.dumps(dict(file=spec['name'], contiguous_bytes=part.stat().st_size,
                              batch_speed=(part.stat().st_size-offset)/(time.monotonic()-started))), flush=True)
    report = checked(part, spec)
    os.replace(part, final)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['download', 'ranges', 'verify', 'refresh'])
    parser.add_argument('--site', choices=list(FILES))
    parser.add_argument('--chunk-mib', type=int, choices=(1, 4, 8, 16), default=16)
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if args.action != 'verify':
        refresh_metadata()
    if args.action == 'refresh':
        return
    reports = []
    for spec in specifications():
        if args.site and spec['name'] != args.site+'.zip':
            continue
        reports.append(transfer(spec) if args.action == 'download' else
                       range_transfer(spec, args.chunk_mib) if args.action == 'ranges' else checked(DEST/spec['name'], spec))
    if len(reports) == 2:
        free = shutil.disk_usage(DEST.anchor).free
        expanded = sum(x['expanded_bytes'] for x in reports)
        result = dict(status='BOTH_ARCHIVES_VERIFIED', archives=reports, free_bytes=free,
                      expanded_total=expanded, final_free_bytes=free-expanded,
                      disk_guard_pass=free-expanded >= RESERVE,
                      allow_extract=False, reason='Supervisor must review actual budget before extraction')
        save(LOG/'verified_summary.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
