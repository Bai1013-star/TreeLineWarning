"""Measure three real context reads and ensure the cache retains raw measurements."""
import json
import sys
import time
import urllib.request
from pathlib import Path

servers=json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
times=[];rows=[]
for index in range(3):
    start=time.perf_counter()
    with urllib.request.urlopen(servers['warning']+'integrated/context',timeout=180) as response:
        data=json.load(response)
    times.append(time.perf_counter()-start)
    assert data['status']=='READY'
    rows.append(data['trees'])
    print(json.dumps(dict(read=index+1,seconds=times[-1],trees=len(rows[-1])),ensure_ascii=False),flush=True)
assert rows[0]==rows[1]==rows[2]
assert times[2]<2, 'Warm context should not recompute the full point-cloud metrics'
result=dict(passed=True,seconds=times,rows_unchanged=True,count=len(rows[0]),finalized=data['finalized'])
Path(sys.argv[2]).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
