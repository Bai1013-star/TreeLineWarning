"""Standard-library preflight preserves honest failure before torch imports."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    source = Path(__file__).parent
    base = {'status':'ENVIRONMENT_NOT_READY','gate_a':'NOT_RUN','mini_overfit':'NOT_RUN','public_download_allowed':False,'python_executable':sys.executable,'python_version':sys.version,'d_free_bytes':shutil.disk_usage('D:\\').free,'time':time.time(),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')},'frozen_config_sha256':hashlib.sha256((source/'mini_gate_config.json').read_bytes()).hexdigest()}
    try:
        if base['d_free_bytes'] < 35*1024**3:
            raise RuntimeError('STOP_DISK_GUARD')
        import torch
        base['torch'] = torch.__version__
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
    except Exception:
        base['error'] = traceback.format_exc()
        (args.output/'mini_gate.json').write_text(json.dumps(base,indent=2),encoding='utf-8')
        print(json.dumps(base),flush=True)
        return 2
    with (args.output/'tests.raw.log').open('w',encoding='utf-8') as out:
        tests = subprocess.run([sys.executable,str(source/'test_models.py')],stdout=out,stderr=subprocess.STDOUT)
    if tests.returncode:
        base.update(status='MODEL_TEST_FAILED',gate_a='FAIL',error='See tests.raw.log')
        (args.output/'mini_gate.json').write_text(json.dumps(base,indent=2),encoding='utf-8')
        print(json.dumps(base),flush=True)
        return tests.returncode
    return subprocess.call([sys.executable,str(source/'mini_gate.py'),'--output',str(args.output)])


if __name__ == '__main__':
    raise SystemExit(main())
