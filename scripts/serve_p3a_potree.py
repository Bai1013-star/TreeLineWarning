"""Serve the P3A Potree POC over a loopback-only HTTP server (P3A-I6).

The server is strictly local: the bind address is fixed to ``127.0.0.1`` and
there is no ``--host`` override.  Binding is Windows-first exclusive
(``SO_EXCLUSIVEADDRUSE`` before ``bind``, ``allow_reuse_address`` off), so an
already-occupied loopback port fails deterministically with a clear error and a
nonzero exit instead of silently sharing the port.

Only three virtual roots are exposed; everything else returns 404 and no path
may escape its root, so the protected source under ``input_las/`` is never
reachable over HTTP.

Virtual routes::

    /potree/            -> cache/p3a_potree/dependencies/Potree_1.8.2/
    /pointclouds/       -> cache/p3a_potree/dataset/
    /                   -> viewer_poc/p3a/
    /p3a_healthz        -> JSON status used by the automated smoke tests

Usage::

    python scripts/serve_p3a_potree.py
    python scripts/serve_p3a_potree.py --port 8091 --no-open
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import socket
import sys
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POTREE_ROOT = _REPOSITORY_ROOT / "cache" / "p3a_potree" / "dependencies" / "Potree_1.8.2"
DATASET_ROOT = _REPOSITORY_ROOT / "cache" / "p3a_potree" / "dataset"
VIEWER_ROOT = _REPOSITORY_ROOT / "viewer_poc" / "p3a"

# The single canonical P3A binding.  Loopback-only; never broaden to remote
# access, never expose a --host override.
P3A_BIND_HOST = "127.0.0.1"


def active_dataset_name() -> str | None:
    """Derive the ACTIVE P3A dataset name from the authoritative viewer config.

    Reads ``viewer_poc/p3a/poc_config.json`` and returns the dataset name in its
    ``pointcloudUrl`` (e.g. ``"pointclouds/p3a_roi/metadata.json"`` →
    ``"p3a_roi"``).  Returns ``None`` when the config is missing/unparsable or
    the URL is not the expected Potree octree shape, so callers never silently
    fall back to an unrelated stale dataset name (P3A-VISUAL-004).
    """
    try:
        cfg = json.loads((VIEWER_ROOT / "poc_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    url = cfg.get("pointcloudUrl") if isinstance(cfg, dict) else None
    if not isinstance(url, str):
        return None
    parts = url.split("/")
    if len(parts) == 3 and parts[0] == "pointclouds" and parts[2] == "metadata.json":
        return parts[1]
    return None

_ROUTES = (
    ("/potree/", POTREE_ROOT),
    ("/pointclouds/", DATASET_ROOT),
    ("/", VIEWER_ROOT),
)

# Python's mimetypes may not know these; force correct types so the browser
# executes Potree workers / wasm decoders.
_EXTRA_MIME = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".json": "application/json",
    ".wasm": "application/wasm",
    ".bin": "application/octet-stream",
    ".glsl": "text/plain",
}


def _guess_type(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0]
    if mime:
        return mime
    return _EXTRA_MIME.get(path.suffix.lower(), "application/octet-stream")


class _ExclusiveBindServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with deterministic, exclusive address binding.

    ``allow_reuse_address`` is off so a held loopback port is never silently
    shared or ambiguously rebound.  On Windows, ``SO_EXCLUSIVEADDRUSE`` is set
    before ``bind`` (exclusive-address semantics); a conflicting bind then
    raises ``OSError`` instead of succeeding.
    """

    allow_reuse_address = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            # Must be set before bind(); makes the address exclusive on Windows.
            self.socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
            )
        super().server_bind()


class _P3aHandler(BaseHTTPRequestHandler):
    server_version = "P3aPotreePOC/0.1"

    def _resolve(self, path: str) -> Path | None:
        """Map a request path to a file inside one of the virtual roots.

        Returns ``None`` when the request is not servable (404/403).
        """
        path = urllib.parse.unquote(path.split("?")[0].split("#")[0])
        for prefix, root in _ROUTES:
            if path == prefix or path.startswith(prefix):
                relative = path[len(prefix) :]
                if not relative:
                    relative = "index.html"
                target = (root / relative).resolve()
                # Never allow a request to escape its root.
                try:
                    target.relative_to(root.resolve())
                except ValueError:
                    return None
                return target if target.is_file() else None
        return None

    def _serve_file(self, target: Path) -> None:
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", _guess_type(target))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_healthz(self) -> None:
        # The health check reports the ACTUAL configured active dataset (from the
        # authoritative viewer config), never a stale unrelated sample.  It is NOT
        # "ready" when that active dataset's metadata/octree are missing even if an
        # old sample octree still exists on disk.
        active = active_dataset_name()
        if active is None:
            body = json.dumps(
                {
                    "status": "error",
                    "viewer_root": str(VIEWER_ROOT),
                    "error": "cannot determine active P3A dataset from viewer config",
                    "active_dataset": None,
                    "dataset_ready": False,
                    "potree_root_exists": POTREE_ROOT.is_dir(),
                },
                ensure_ascii=False,
            ).encode("utf-8")
        else:
            metadata = DATASET_ROOT / active / "metadata.json"
            octree = DATASET_ROOT / active / "octree.bin"
            body = json.dumps(
                {
                    "status": "ok",
                    "viewer_root": str(VIEWER_ROOT),
                    "active_dataset": active,
                    "dataset_ready": metadata.is_file() and octree.is_file(),
                    "potree_root_exists": POTREE_ROOT.is_dir(),
                    "active_dataset_metadata": metadata.is_file(),
                },
                ensure_ascii=False,
            ).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path.split("?")[0] == "/p3a_healthz":
            self._serve_healthz()
            return
        target = self._resolve(self.path)
        if target is None:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        self._serve_file(target)

    do_HEAD = do_GET  # noqa: N815 - http.server API

    def log_message(self, fmt: str, *args) -> None:  # noqa: N802
        sys.stdout.write(f"[p3a] {self.address_string()} {fmt % args}\n")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="loopback port to bind (default: 8090, 0 = auto-assign)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="do not open the browser after the server starts",
    )
    args = parser.parse_args(argv)

    for label, root in (("potree", POTREE_ROOT), ("dataset", DATASET_ROOT), ("viewer", VIEWER_ROOT)):
        if not root.is_dir():
            print(f"[p3a] ERROR: {label} root missing: {root}", file=sys.stderr)
            return 2

    try:
        httpd = _ExclusiveBindServer((P3A_BIND_HOST, args.port), _P3aHandler)
    except OSError as exc:
        print(
            f"[p3a] ERROR: cannot bind loopback {P3A_BIND_HOST}:{args.port} "
            f"(port in use or address unavailable): {exc}",
            file=sys.stderr,
        )
        return 3

    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"[p3a] serving P3A Potree POC at {url}")
    active = active_dataset_name()
    if active:
        print(f"[p3a] active dataset: {url}pointclouds/{active}/metadata.json")
    else:
        print(
            "[p3a] WARNING: cannot determine active P3A dataset from viewer config",
            file=sys.stderr,
        )
    if not args.no_open:
        import webbrowser

        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[p3a] shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
