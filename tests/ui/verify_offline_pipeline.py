"""Re-run the installed real pipeline behind the offline audit guard."""
import json
import sys
import time
import urllib.request
from pathlib import Path

output = Path(sys.argv[1])
urls = json.loads((output / 'servers.json').read_text(encoding='utf8'))
client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def call(path, data=None):
    request = urllib.request.Request(
        urls['processing'] + path.lstrip('/'),
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with client.open(request, timeout=900) as response:
        return json.load(response)

workspace = call('/api/desktop/workspace')
span_ids = [span['span_id'] for span in workspace['batch']['spans']]
started = time.monotonic()
initial = call('/api/desktop/batch/start', dict(span_ids=span_ids, mode='REAL', execution_mode='SERIAL', acceleration_mode='CPU'))
print(json.dumps({key: initial.get(key) for key in ['status','batch_id','acceptance_mode','source_cache_hint']}, ensure_ascii=False), flush=True)
while True:
    result = call('/api/desktop/batch')
    if result['status'] != 'RUNNING':
        break
    if time.monotonic() - started > 1200:
        raise TimeoutError('offline real pipeline exceeded 20 minutes')
    time.sleep(5)
elapsed = time.monotonic() - started
assert result['status'] == 'COMPLETE', result
assert result['acceptance_mode'] == 'REAL', result
denied = json.loads((output/'server_outbound_denied.json').read_text(encoding='utf8'))
assert denied == [], denied
record = dict(passed=True, elapsed_seconds=elapsed, batch=result, backend_outbound=denied)
(output/'real_pipeline_offline.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(dict(passed=True,elapsed_seconds=elapsed,status=result['status'],trees=result['final_tree_count'],backend_outbound=denied), ensure_ascii=False), flush=True)
