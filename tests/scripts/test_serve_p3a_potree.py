"""P3A-I6 loopback-server tests (incl. P3A-002 loopback + P3A-003 exclusive bind).

The server must bind loopback only, reject any non-loopback host through the
CLI, never silently share an occupied loopback port (Windows-exclusive bind),
serve the viewer/Potree assets and the converted octree, expose a health
endpoint, and never leak files outside its three virtual roots (the protected
105M-point source stays unreachable).
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

import scripts.serve_p3a_potree as serve

# The ACTIVE dataset is the one the authoritative viewer config requires.
_ACTIVE_DATASET = serve.active_dataset_name()
_OCTREE_METADATA = serve.DATASET_ROOT / (_ACTIVE_DATASET or "p3a_roi") / "metadata.json"


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


def _raw_status(server_addr: tuple[str, int], raw_path: str) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", server_addr[1], timeout=15)
    try:
        conn.putrequest("GET", raw_path, skip_host=True)
        conn.putheader("Host", f"127.0.0.1:{server_addr[1]}")
        conn.endheaders()
        response = conn.getresponse()
        return response.status
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# P3A-002: loopback only
# ---------------------------------------------------------------------------


def test_default_binding_is_loopback() -> None:
    assert serve.P3A_BIND_HOST == "127.0.0.1"


def test_binds_loopback_only(server_addr: tuple[str, int]) -> None:
    host, _ = server_addr
    assert host == "127.0.0.1"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::"])
def test_cli_rejects_non_loopback_host(host: str) -> None:
    # The CLI has no --host override; a remote-binding attempt is a parse
    # failure before any server is created.
    with pytest.raises(SystemExit) as exc:
        serve.main(["--host", host])
    assert exc.value.code != 0


# ---------------------------------------------------------------------------
# P3A-003: exclusive / deterministic Windows bind
# ---------------------------------------------------------------------------


def test_exclusive_bind_disables_reuse() -> None:
    assert serve._ExclusiveBindServer.allow_reuse_address is False


@pytest.mark.skipif(
    not serve.DATASET_ROOT.is_dir(), reason="dataset not converted"
)
def test_occupied_port_fails_clearly(capsys: pytest.CaptureFixture) -> None:
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        rc = serve.main(["--port", str(port), "--no-open"])
        assert rc != 0
        err = capsys.readouterr().err
        assert "bind" in err.lower()
        assert str(port) in err
    finally:
        holder.close()


# ---------------------------------------------------------------------------
# Serving behaviour (unchanged)
# ---------------------------------------------------------------------------


def test_healthz(server_addr: tuple[str, int]) -> None:
    status, body = _get(server_addr, "/p3a_healthz")
    assert status == 200
    data = json.loads(body)
    assert data["status"] == "ok"
    # The health check reports the ACTUAL configured active dataset (p3a_roi),
    # never a stale unrelated sample name.
    assert data["active_dataset"] == _ACTIVE_DATASET
    assert data["active_dataset"] == "p3a_roi"
    assert data["dataset_ready"] is True


def test_active_dataset_derived_from_authoritative_config() -> None:
    # P3A-VISUAL-004: the health check / startup derive the active dataset from
    # viewer_poc/p3a/poc_config.json pointcloudUrl, not a hard-coded name.
    name = serve.active_dataset_name()
    assert name == "p3a_roi"
    cfg = json.loads((serve.VIEWER_ROOT / "poc_config.json").read_text(encoding="utf-8"))
    assert cfg["pointcloudUrl"] == f"pointclouds/{name}/metadata.json"


def test_active_dataset_name_rejects_malformed_or_missing_config(
    monkeypatch, tmp_path: Path
) -> None:
    # A config whose pointcloudUrl is not the expected Potree octree shape yields
    # None so callers never fall back to an unrelated dataset name.
    (tmp_path / "poc_config.json").write_text(
        '{"pointcloudUrl": "pointclouds/p3a_sample/metadata.json/extra"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(serve, "VIEWER_ROOT", tmp_path)
    assert serve.active_dataset_name() is None

    (tmp_path / "poc_config.json").write_text('{"pointcloudUrl": "http://other"}', encoding="utf-8")
    assert serve.active_dataset_name() is None

    (tmp_path / "poc_config.json").unlink()
    assert serve.active_dataset_name() is None


def test_healthz_not_ready_when_active_dataset_missing(monkeypatch) -> None:
    # P3A-VISUAL-004: when the ACTIVE configured dataset is absent, the health
    # check must NOT report ready even if an unrelated old sample octree exists.
    assert (serve.DATASET_ROOT / "p3a_sample" / "metadata.json").is_file()
    monkeypatch.setattr(serve, "active_dataset_name", lambda: "p3a_missing")
    httpd = serve._ExclusiveBindServer(("127.0.0.1", 0), serve._P3aHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _get(httpd.server_address, "/p3a_healthz")
        data = json.loads(body)
        assert status == 200
        assert data["active_dataset"] == "p3a_missing"
        assert data["dataset_ready"] is False
        assert data["active_dataset_metadata"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_viewer_index_served(server_addr: tuple[str, int]) -> None:
    status, body = _get(server_addr, "/")
    assert status == 200
    assert b"potree" in body.lower()
    assert b"viewer.js" in body


def test_potree_js_served(server_addr: tuple[str, int]) -> None:
    status, body = _get(server_addr, "/potree/build/potree/potree.js")
    assert status == 200
    assert b"Potree" in body


@pytest.mark.skipif(not _OCTREE_METADATA.is_file(), reason="octree not converted")
def test_octree_served(server_addr: tuple[str, int]) -> None:
    status, body = _get(server_addr, f"/pointclouds/{_ACTIVE_DATASET}/metadata.json")
    assert status == 200
    metadata = json.loads(body)
    names = [a["name"] for a in metadata["attributes"]]
    assert "position" in names
    assert "rgb" in names

    status, octree = _get(server_addr, f"/pointclouds/{_ACTIVE_DATASET}/octree.bin")
    assert status == 200
    assert len(octree) > 1_000_000


def test_traversal_cannot_reach_protected_source(server_addr: tuple[str, int]) -> None:
    # Five levels up from dataset root reaches the repository root where the
    # protected 105M-point source lives; the server must reject it.
    status = _raw_status(
        server_addr,
        "/pointclouds/../../../input_las/2026-08-07_16%E6%B2%A154-10_PointCloud_colorized.las",
    )
    assert status == 404


def test_unknown_path_is_404(server_addr: tuple[str, int]) -> None:
    with pytest.raises(HTTPError) as exc:
        _get(server_addr, "/not-a-real-path.txt")
    assert exc.value.code == 404
