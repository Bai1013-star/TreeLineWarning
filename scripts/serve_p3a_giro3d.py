"""Serve the P3A Giro3D + COPC viewer (Candidate B) over loopback only.

Mirrors the P3A Potree server's safety posture: strictly loopback
(``127.0.0.1``, no ``--host`` override), Windows-first exclusive bind
(``SO_EXCLUSIVEADDRUSE``) so an occupied port fails deterministically, and only
virtual roots that cannot escape to the protected source.

Virtual routes::

    /vendor/        -> viewer_poc/giro3d/vendor/            (Giro3D bundle)
    /laz-perf/      -> viewer_poc/giro3d/vendor/laz-perf/   (lazperf WASM)
    /pointclouds/   -> cache/p3a_giro3d/dataset/            (the derived COPC)
    /               -> viewer_poc/giro3d/                   (viewer page)
    /p3a_giro3d_healthz -> JSON status (active COPC presence)

Usage::

    python scripts/serve_p3a_giro3d.py
    python scripts/serve_p3a_giro3d.py --port 8091 --no-open
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import socket
import sys
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VIEWER_ROOT = _REPOSITORY_ROOT / "viewer_poc" / "giro3d"
VENDOR_ROOT = VIEWER_ROOT / "vendor"
LAZPERF_ROOT = VENDOR_ROOT / "laz-perf"
DATASET_ROOT = _REPOSITORY_ROOT / "cache" / "p3a_giro3d" / "dataset"
# Viewer config served at /poc_config.json (ROI POC or FULL-SOURCE gate).
_CONFIG_FILE = "poc_config.json"

# The single canonical P3A binding.  Loopback-only; never broaden.
P3A_BIND_HOST = "127.0.0.1"

def _routes() -> tuple[tuple[str, Path], ...]:
    """Virtual roots, computed at request time so the dataset root (ROI POC or
    FULL-SOURCE gate) can be switched per launcher via --dataset-root."""
    return (
        ("/vendor/", VENDOR_ROOT),
        ("/laz-perf/", LAZPERF_ROOT),
        ("/pointclouds/", DATASET_ROOT),
        ("/", VIEWER_ROOT),
    )

_EXTRA_MIME = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".json": "application/json",
    ".wasm": "application/wasm",
    ".laz": "application/octet-stream",
    ".glsl": "text/plain",
}


def _guess_type(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0]
    if mime:
        return mime
    return _EXTRA_MIME.get(path.suffix.lower(), "application/octet-stream")


def active_copc_name() -> str | None:
    """Derive the active COPC filename from the served viewer config's copcUrl."""
    try:
        cfg = json.loads((VIEWER_ROOT / _CONFIG_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    url = cfg.get("copcUrl") if isinstance(cfg, dict) else None
    if isinstance(url, str) and url.endswith(".laz"):
        return url.rsplit("/", 1)[-1]
    return None


class _ExclusiveBindServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class _P3aHandler(BaseHTTPRequestHandler):
    server_version = "P3aGiro3dPOC/0.1"

    def _resolve(self, path: str) -> Path | None:
        path = urllib.parse.unquote(path.split("?")[0].split("#")[0])
        # Collapse duplicate slashes (e.g. the "/laz-perf//laz-perf.wasm" URL that
        # Giro3D produces when a trailing-slash lazPerfPath is combined with its
        # own "/laz-perf.wasm").  The path still stays within the virtual roots.
        path = re.sub(r"//+", "/", path)
        # The viewer always fetches poc_config.json; serve whichever config this
        # launcher was started with (ROI POC or FULL-SOURCE gate).
        if path == "/poc_config.json":
            target = (VIEWER_ROOT / _CONFIG_FILE).resolve()
            if target.is_file():
                return target
            return None
        for prefix, root in _routes():
            if path == prefix or path.startswith(prefix):
                relative = path[len(prefix):]
                if not relative:
                    relative = "index.html"
                target = (root / relative).resolve()
                try:
                    target.relative_to(root.resolve())
                except ValueError:
                    return None
                return target if target.is_file() else None
        return None

    def _serve_file(self, target: Path) -> None:
        """Serve a file, honouring HTTP Range requests (copc.js reads the COPC
        via byte-range fetches for the header, hierarchy pages and chunks)."""
        data = target.read_bytes()
        total = len(data)
        start, end = 0, total - 1

        range_header = self.headers.get("Range")
        if range_header:
            m = range_header.strip().lower()
            if m.startswith("bytes="):
                spec = m[len("bytes="):].split(",", 1)[0].strip()
                if spec:
                    parts = spec.split("-", 1)
                    try:
                        if parts[0] == "" and parts[1] != "":
                            # suffix range: last N bytes
                            n = int(parts[1])
                            start = max(0, total - n)
                        elif parts[0] != "":
                            start = int(parts[0])
                            if parts[1] != "":
                                end = min(int(parts[1]), total - 1)
                        if 0 <= start <= end < total:
                            self.send_response(HTTPStatus.PARTIAL_CONTENT)
                            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                            self.send_header("Accept-Ranges", "bytes")
                            self.send_header("Content-Type", _guess_type(target))
                            self.send_header("Content-Length", str(end - start + 1))
                            self.send_header("Cache-Control", "no-store")
                            self.end_headers()
                            self.wfile.write(data[start : end + 1])
                            return
                    except ValueError:
                        pass
        self.send_response(HTTPStatus.OK)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", _guess_type(target))
        self.send_header("Content-Length", str(total))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_healthz(self) -> None:
        # The active COPC is derived from the served viewer config's copcUrl so
        # the health check reflects whatever dataset this launcher is serving
        # (ROI POC or FULL-SOURCE gate) — never a stale hard-coded name.
        active_copc = active_copc_name()
        copc = DATASET_ROOT / active_copc if active_copc else None
        body = json.dumps(
            {
                "status": "ok",
                "viewer_root": str(VIEWER_ROOT),
                "active_copc": active_copc,
                "dataset_ready": bool(copc and copc.is_file()),
                "bundle_ready": (VENDOR_ROOT / "giro3d.bundle.js").is_file(),
                "lazperf_ready": (LAZPERF_ROOT / "laz-perf.wasm").is_file(),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path.split("?")[0] == "/p3a_giro3d_healthz":
            self._serve_healthz()
            return
        target = self._resolve(self.path)
        if target is None:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        self._serve_file(target)

    do_HEAD = do_GET  # noqa: N815 - http.server API

    def log_message(self, fmt: str, *args) -> None:  # noqa: N802
        sys.stdout.write(f"[p3a-giro3d] {self.address_string()} {fmt % args}\n")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    global DATASET_ROOT, _CONFIG_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8091,
                        help="loopback port to bind (default: 8091, 0 = auto-assign)")
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="dataset cache root to serve at /pointclouds/ (default: ROI POC cache)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="viewer config served at /poc_config.json (default: poc_config.json = ROI POC)",
    )
    args = parser.parse_args(argv)

    if args.dataset_root is not None:
        DATASET_ROOT = args.dataset_root
    if args.config is not None:
        _CONFIG_FILE = args.config

    for label, root in (("viewer", VIEWER_ROOT), ("vendor", VENDOR_ROOT), ("dataset", DATASET_ROOT)):
        if not root.is_dir():
            print(f"[p3a-giro3d] ERROR: {label} root missing: {root}", file=sys.stderr)
            return 2

    try:
        httpd = _ExclusiveBindServer((P3A_BIND_HOST, args.port), _P3aHandler)
    except OSError as exc:
        print(
            f"[p3a-giro3d] ERROR: cannot bind loopback {P3A_BIND_HOST}:{args.port} "
            f"(port in use or address unavailable): {exc}",
            file=sys.stderr,
        )
        return 3

    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"[p3a-giro3d] serving P3A Giro3D viewer at {url}")
    print(f"[p3a-giro3d] dataset root: {DATASET_ROOT}")
    print(f"[p3a-giro3d] viewer config: {_CONFIG_FILE}")
    copc = active_copc_name()
    if copc:
        copc_path = DATASET_ROOT / copc
        print(f"[p3a-giro3d] COPC: {url}pointclouds/{copc}"
              + (" (ready)" if copc_path.is_file() else " (MISSING — run the converter)"))
    if not args.no_open:
        import webbrowser

        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[p3a-giro3d] shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
