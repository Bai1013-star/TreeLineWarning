"""Serve an existing independent acceptance project, not the user's project."""
import argparse
import json
import socket
import sys
import time
from pathlib import Path
from urllib.parse import quote

# Running this file directly otherwise puts only ``scripts`` on sys.path and
# can silently import the older checkout from D:\TreeLineWarning.  Pin every
# acceptance server to the source tree that owns this script.
source_root = Path(__file__).resolve().parents[1]
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from desktop.app import BackendProcess, WarningBackend

parser = argparse.ArgumentParser()
parser.add_argument('project', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
args.project.resolve().relative_to(Path('D:/TreeLineWarning/双点云全功能复测').resolve())
if not args.project.name.startswith('双点云复测_'):
    raise ValueError('Acceptance project required')
args.output.mkdir(parents=True, exist_ok=True)
connect = socket.socket.connect
def offline_connect(client, address):
    if isinstance(address, tuple) and address[0] not in ('127.0.0.1', '::1', 'localhost'):
        raise OSError('OFFLINE_ACCEPTANCE_ONLY_LOOPBACK')
    return connect(client, address)
socket.socket.connect = offline_connect
root = source_root
processing = BackendProcess(args.project, root)
warning = WarningBackend(args.project, root, processing)
try:
    urls = dict(processing=processing.start(), warning=warning.start(), project=str(args.project), offline_guard=True)
    urls['desktop'] = urls['processing'] + '?warning_url=' + quote(urls['warning'], safe='')
    (args.output/'servers.json').write_text(json.dumps(urls, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(urls, ensure_ascii=False), flush=True)
    while True:
        time.sleep(1)
finally:
    warning.stop()
    processing.stop()
