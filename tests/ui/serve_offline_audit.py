"""Deny every non-loopback application connection while testing local UI."""
import argparse
import ipaddress
import json
import os
import shutil
import socket
import time
from pathlib import Path
from urllib.parse import quote
from desktop.app import BackendProcess, WarningBackend
from desktop.pdal_runtime import _configure_bundled_runtime

APP = Path('D:/TreeLineWarning_一体化软件_20260926/TreeLineWarning一体化软件2.0_20260926')
OUT = Path('D:/TreeLineWarning/双点云全功能复测/离线版本20261002')
parser = argparse.ArgumentParser()
parser.add_argument('case', choices=['A','B','C'])
parser.add_argument('--tower-only', action='store_true')
parser.add_argument('--output-root', type=Path, default=OUT)
parser.add_argument('--resource-root', type=Path, default=APP/'程序文件')
parser.add_argument('--project', type=Path, help='Dedicated acceptance project below D:/TreeLineWarning')
args = parser.parse_args()
OUT = args.output_root
output = OUT / (args.case + ('_杆塔' if args.tower_only else ''))
output.mkdir(parents=True, exist_ok=True)
projects = dict(A=APP/'项目数据/双点云复测_A_105百万点', B=APP/'项目数据/双点云复测_B_北京西站南路')
if args.case == 'C':
    project = OUT/'C_不完整LAS诊断项目'
    project.mkdir(exist_ok=True)
    for name in ['desktop-project.json','desktop-workflow.json','stage-a-project.json']:
        shutil.copy2(APP/'项目数据/2026.10.02-01'/name, project/name)
else:
    project = projects[args.case]
if args.project is not None:
    project = args.project.resolve()
    project.relative_to(Path('D:/TreeLineWarning').resolve())
    if not project.name.startswith('双点云复测_') or not (project/'stage-a-project.json').is_file():
        raise ValueError('Only independent audit projects may be overridden')
if args.tower_only:
    seed = project
    project = OUT / (args.case + '_杆塔显示验收')
    project.mkdir(exist_ok=True)
    for name in ['desktop-project.json','desktop-workflow.json','stage-a-project.json']:
        document = json.loads((seed/name).read_text(encoding='utf8'))
        if name == 'stage-a-project.json':
            for field in ['lines','towers','spans','routes','analysis_runs']:
                document['project'][field] = []
            document['project']['name'] += '_杆塔显示验收'
        elif name == 'desktop-workflow.json':
            document.update(last_batch_id=None,last_batch_status=None,selected_span_ids=[])
        (project/name).write_text(json.dumps(document,ensure_ascii=False),encoding='utf8')
for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
    os.environ.pop(key, None)
os.environ['NO_PROXY'] = '*'
denied = []
original_connect = socket.socket.connect
original_getaddrinfo = socket.getaddrinfo
def local_host(host):
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
def reject(host):
    denied.append(str(host))
    (output/'server_outbound_denied.json').write_text(json.dumps(denied), encoding='utf8')
    raise OSError('OFFLINE_AUDIT_BLOCKED_NON_LOOPBACK:' + str(host))
def guarded_connect(client, address):
    if isinstance(address, tuple) and not local_host(str(address[0])):
        reject(address[0])
    return original_connect(client, address)
def guarded_getaddrinfo(host, *values, **options):
    if host is not None and not local_host(str(host)):
        reject(host)
    return original_getaddrinfo(host, *values, **options)
socket.socket.connect = guarded_connect
socket.getaddrinfo = guarded_getaddrinfo
(output/'server_outbound_denied.json').write_text('[]', encoding='utf8')
_configure_bundled_runtime()
processing = BackendProcess(project, args.resource_root)
processing_url = processing.start()
warning = WarningBackend(project, args.resource_root, processing)
warning_url = warning.start()
urls = dict(processing=processing_url,warning=warning_url,project=str(project),
            desktop=processing_url+'?warning_url='+quote(warning_url,safe=''),offline_guard=True)
(output/'servers.json').write_text(json.dumps(urls,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(urls,ensure_ascii=False),flush=True)
try:
    while True:
        time.sleep(2)
finally:
    warning.stop()
    processing.stop()
