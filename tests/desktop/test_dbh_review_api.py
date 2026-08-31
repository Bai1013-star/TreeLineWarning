from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from desktop.server import _DesktopHandler
from tests.stage_b.test_dbh_identity_recovery import dense_query
from tests.stage_b.test_dbh_raw_recovery import current_project, v1_project  # noqa: F401

from desktop import dbh_review as adapter


@pytest.fixture
def service(current_project, monkeypatch):  # noqa: F811
    root, _, result, members, _ = current_project
    workflow = SimpleNamespace(project_root=root, _log=lambda *a, **k: None)
    monkeypatch.setattr(adapter, "_context", lambda w: (root, result, members, object()))
    monkeypatch.setattr(
        adapter, "_current_display_identities", lambda *a: {0: {"display_tree_id": "T-1-001"}}
    )
    monkeypatch.setattr(adapter.review, "local_raw_query", dense_query(current_project))
    app = SimpleNamespace(workflow=workflow, resource_root=Path(__file__).resolve().parents[2])
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_DesktopHandler, application=app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    def http(path, data=None, headers=None):
        req = urllib.request.Request(
            url + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    yield http, workflow, url
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def wait_job(http, request_id):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        _, record = http("/api/editor/mutation")
        assert record["request_id"] == request_id
        if record["state"] != "RUNNING":
            assert record["state"] != "FAILED", record
            return record["result"]
        time.sleep(0.02)
    pytest.fail("Bounded DBH API job timeout")


def test_http_async_prepare_confirm_and_read_only_list(service):
    http, _, _ = service
    status, data = http("/api/dbh-review/trees")
    assert status == 200 and data["trees"][0]["dbh_cm"] is None
    status, _ = http(
        "/api/dbh-review/prepare",
        {"persistent_tree_id": "tree-0", "async": True, "request_id": "prepare-1"},
    )
    assert status == 202
    data = wait_job(http, "prepare-1")["review"]
    assert data["display_tree_id"] == "T-1-001"
    # Selecting this fixture point is explicitly synthetic acceptance only.
    status, _ = http(
        "/api/dbh-review/confirm",
        {
            "review_id": data["review_id"],
            "identity_confirmed": True,
            "seed_original_index": data["display"]["original_index"][0],
            "async": True,
            "request_id": "confirm-1",
        },
    )
    assert status == 202
    result = wait_job(http, "confirm-1")
    assert result["dbh_method"] == "REVIEW_CONFIRMED_STEM"
    assert http("/api/dbh-review/trees")[1]["trees"][0]["dbh_cm"] == pytest.approx(30, abs=0.15)


def test_http_failure_and_cross_origin_leave_backend_alive(service):
    http, _, _ = service
    status, _ = http("/api/dbh-review/confirm", {"dbh_cm": 99})
    assert status == 409
    status, _ = http(
        "/api/dbh-review/prepare",
        {"persistent_tree_id": "tree-0"},
        {"Origin": "http://outside.invalid"},
    )
    assert status == 403
    assert http("/api/dbh-review/trees")[0] == 200


def test_current_finalize_gate_is_mandatory(monkeypatch):
    monkeypatch.setattr(adapter, "current_finalized_gate", lambda w: {"status": "LOCKED"})
    with pytest.raises(ValueError, match="REQUIRES_CURRENT_FINALIZE"):
        adapter._context(object())
