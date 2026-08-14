"""P3A Giro3D loopback-server tests.

The server must bind loopback only, reject a non-loopback host through the CLI,
never silently share an occupied loopback port, serve the Giro3D viewer bundle /
laz-perf WASM / the active COPC (with HTTP Range support for copc.js), expose a
health endpoint reporting the ACTIVE COPC, and never leak files outside its
virtual roots.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

import scripts.serve_p3a_giro3d as serve


@pytest.fixture(scope="module")
def server_addr() -> tuple[str, int]:
    httpd = serve._ExclusiveBindServer(("127.0.0.1", 0), serve._P3aHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    yield (host, port)
    httpd.shutdown()
    httpd.server_close()


def _get(server_addr: tuple[str, int], path: str):
    url = f"http://127.0.0.1:{server_addr[1]}{path}"
    with urlopen(url, timeout=15) as response:
        return response.status, response.read()


def _get_range(server_addr: tuple[str, int], path: str, start: int, end: int):
    url = f"http://127.0.0.1:{server_addr[1]}{path}"
    request = http.client.HTTPConnection("127.0.0.1", server_addr[1], timeout=15)
    request.request("GET", path, headers={"Range": f"bytes={start}-{end}"})
    response = request.getresponse()
    return response.status, response.read(), response.getheader("Content-Range")


# ---------------------------------------------------------------------------
# Loopback-only + exclusive bind
# ---------------------------------------------------------------------------


def test_bind_host_is_loopback_only() -> None:
    assert serve.P3A_BIND_HOST == "127.0.0.1"


def test_no_host_override_option() -> None:
    # The CLI must not offer a --host override (loopback-only posture): argparse
    # rejects the unknown option with a SystemExit before any server is created.
    with pytest.raises(SystemExit) as exc:
        serve.main(["--host", "0.0.0.0", "--port", "0"])
    assert exc.value.code != 0


def test_exclusive_bind_rejects_occupied_port(capsys) -> None:
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        rc = serve.main(["--no-open", "--port", str(port)])
        assert rc != 0
        err = capsys.readouterr().err
        assert "bind" in err.lower()
        assert str(port) in err
    finally:
        holder.close()


# ---------------------------------------------------------------------------
# Health + asset serving + Range
# ---------------------------------------------------------------------------


def test_healthz_reports_active_copc(server_addr: tuple[str, int]) -> None:
    status, body = _get(server_addr, "/p3a_giro3d_healthz")
    assert status == 200
    data = json.loads(body)
    assert data["status"] == "ok"
    assert data["active_copc"] == "p3a_roi.copc.laz"
    assert data["dataset_ready"] is True
    assert data["bundle_ready"] is True
    assert data["lazperf_ready"] is True


def test_healthz_not_ready_when_active_copc_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(serve, "DATASET_ROOT", tmp_path)  # active COPC absent
    httpd = serve._ExclusiveBindServer(("127.0.0.1", 0), serve._P3aHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _get(httpd.server_address, "/p3a_giro3d_healthz")
        data = json.loads(body)
        assert data["dataset_ready"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_viewer_page_and_bundle_served(server_addr: tuple[str, int]) -> None:
    status, body = _get(server_addr, "/")
    assert status == 200
    assert b"giro3d" in body.lower()
    status, body = _get(server_addr, "/vendor/giro3d.bundle.js")
    assert status == 200
    assert b"P3AGiro3D" in body
    status, body = _get(server_addr, "/laz-perf/laz-perf.wasm")
    assert status == 200
    assert body[:4] == b"\x00asm"  # WASM magic


def test_active_copc_served_with_range(server_addr: tuple[str, int]) -> None:
    # Whole-file GET works.
    status, body = _get(server_addr, "/pointclouds/p3a_roi.copc.laz")
    assert status == 200
    assert body[:4] == b"LASF"

    # copc.js-style Range GET returns 206 Partial Content with correct range.
    status, body, content_range = _get_range(server_addr, "/pointclouds/p3a_roi.copc.laz", 0, 99)
    assert status == 206
    assert content_range == "bytes 0-99/4849855"
    assert len(body) == 100
    assert body[:4] == b"LASF"


def test_laz_perf_wasm_url_served(server_addr: tuple[str, int]) -> None:
    """The EXACT URL the browser requests for LAZ decompression must return 200
    with the correct WASM identity and MIME type (regression: a trailing-slash
    lazPerfPath made Giro3D fetch "/laz-perf//laz-perf.wasm" which 404'd)."""
    import hashlib

    wasm = serve.LAZPERF_ROOT / "laz-perf.wasm"
    expected_sha = hashlib.sha256(wasm.read_bytes()).hexdigest()

    url = f"http://127.0.0.1:{server_addr[1]}/laz-perf/laz-perf.wasm"
    with urlopen(url, timeout=15) as response:
        assert response.status == 200
        assert response.headers.get_content_type() == "application/wasm"
        body = response.read()
    assert len(body) > 0
    assert body[:4] == b"\x00asm"  # WASM magic
    assert hashlib.sha256(body).hexdigest() == expected_sha

    # The double-slash URL Giro3D historically produced must now also resolve
    # (server collapses duplicate slashes within the virtual roots).
    url2 = f"http://127.0.0.1:{server_addr[1]}/laz-perf//laz-perf.wasm"
    with urlopen(url2, timeout=15) as response:
        assert response.status == 200
        assert hashlib.sha256(response.read()).hexdigest() == expected_sha


def test_traversal_cannot_escape_roots(server_addr: tuple[str, int]) -> None:
    for path in ("/../input_las/x.las", "/pointclouds/../p3a_roi.copc.laz", "/%2e%2e/etc/passwd"):
        with pytest.raises(HTTPError) as exc:
            _get(server_addr, path)
        assert exc.value.code in (404, 403)
