import concurrent.futures, hashlib, json, pathlib, time, urllib.request, shutil, zipfile
root=pathlib.Path(r'D:\TreeQCTraining\gate_a_recovery_20260909'); chunks=root/'chunks';chunks.mkdir(exist_ok=True)
url='https://download.pytorch.org/whl/cu130/torch-2.14.0%2Bcu130-cp311-cp311-win_amd64.whl'
total=1990587822; prefix=43735040; expected='00fb5440e7560b3b21e5e24567c6096fec4ef65b50a7ef1a105ebed5755eef7e'
original=pathlib.Path(r'D:\TreeQCTraining\tmp\pip-unpack-97lmycbe\torch-2.14.0+cu130-cp311-cp311-win_amd64.whl')
log=open(root/'download.jsonl','a',buffering=1)
def event(**kw):
    kw['time']=time.time();line=json.dumps(kw);log.write(line+'\n');print(line,flush=True)
assert original.is_file() and original.stat().st_size==prefix, 'Original partial missing or wrong size'
assert hashlib.file_digest(original.open('rb'),'sha256').hexdigest()=='c8f0e139d9d40f9ef8aa541d213847a1e7fe0e4cffeeacea043d8709e855d463', 'Original partial changed'
manifest={'url':url,'sha256':expected,'total':total,'retained_path':str(original),'retained_bytes':prefix,'historical_remote_verification':'residual_verification.json on 2026-09-09: first 1MiB and tail 64KiB matched live official ranges; this invocation verifies retained full partial SHA256','chunk_size':16777216,'workers':4}
(root/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
def download(bounds):
    start,end=bounds; p=chunks/f'{start}-{end}.part'; begin=time.time(); retries=0; etag=None
    assert not p.exists() or p.stat().st_size<=end-start+1, 'Existing chunk exceeds expected size'
    while (p.stat().st_size if p.exists() else 0)<end-start+1:
        n=p.stat().st_size if p.exists() else 0; offset=start+n
        try:
            req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-{end}','User-Agent':'TreeQC-GateA-Recovery/1.0','Accept-Encoding':'identity'})
            with urllib.request.urlopen(req,timeout=30) as r:
                required=f'bytes {offset}-{end}/{total}'
                if r.status!=206 or r.headers.get('Content-Range')!=required: raise RuntimeError(f'Invalid range: {r.status} {r.headers}')
                etag=r.headers.get('ETag');
                if etag != '"70ad4f0aedcda136411002e2f71020fe"': raise RuntimeError('Official artifact ETag changed; stop and reverify')
                with p.open('ab') as f:
                    while True:
                        b=r.read(262144)
                        if not b:break
                        if f.tell()+len(b)>end-start+1:raise RuntimeError('range overflow')
                        f.write(b)
            if p.stat().st_size!=end-start+1:raise RuntimeError('incomplete response')
        except Exception as e:
            retries+=1;event(chunk=start,error=str(e),retry=retries,bytes=p.stat().st_size if p.exists() else 0)
            if retries>=4:raise
    assert p.stat().st_size==end-start+1, 'Existing chunk size mismatch'
    event(chunk=start,bytes=p.stat().st_size,seconds=time.time()-begin,etag=etag)
    return p
remaining_chunks=max(0,total-prefix-sum(p.stat().st_size for p in chunks.glob('*.part')))
assert shutil.disk_usage('D:\\').free - remaining_chunks - total >= 35*1024**3, 'STOP_DISK_GUARD'
start_time=time.time();ranges=[(i,min(i+16777216-1,total-1)) for i in range(prefix,total,16777216)]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    futures=[ex.submit(download,b) for b in ranges]
    for future in concurrent.futures.as_completed(futures):future.result()
out=root/'torch-2.14.0+cu130-cp311-cp311-win_amd64.whl'
with out.open('wb') as f:
    with original.open('rb') as src:shutil.copyfileobj(src,f)
    for a,b in ranges:
        with (chunks/f'{a}-{b}.part').open('rb') as src:shutil.copyfileobj(src,f)
h=hashlib.file_digest(out.open('rb'),'sha256').hexdigest();event(final_size=out.stat().st_size,sha256=h,download_seconds=time.time()-start_time)
assert out.stat().st_size==total and h==expected,'INTEGRITY FAILURE'
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    wheel=z.read('torch-2.14.0+cu130.dist-info/WHEEL').decode()
    assert 'Tag: cp311-cp311-win_amd64' in wheel, 'Incompatible wheel tag'
    import sys
    from email.parser import Parser
    from pip._vendor.packaging.specifiers import SpecifierSet
    metadata=Parser().parsestr(z.read('torch-2.14.0+cu130.dist-info/METADATA').decode())
    assert '.'.join(map(str,sys.version_info[:3])) in SpecifierSet(metadata['Requires-Python']), 'Incompatible Python requirement'
    event(wheel_metadata=wheel,requires_python=metadata['Requires-Python'])
(root/'wheel_verified.json').write_text(json.dumps({'path':str(out),'sha256':h,'size':total,'zip_crc':'PASS','wheel_metadata':wheel},indent=2))
