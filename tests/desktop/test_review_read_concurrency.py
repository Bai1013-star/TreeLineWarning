from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from desktop.server import _DesktopHandler


class _SlowReviewWorkflow:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counter_lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0

    def _log(self, _event: str, **_details: object) -> None:
        return None

    def editor_data(
        self, *, display_budget: int, span_id: str | None
    ) -> dict[str, object]:
        with self._counter_lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            time.sleep(0.05)
            return {
                "trees": [],
                "display_budget": display_budget,
                "review_span_id": span_id,
            }
        finally:
            with self._counter_lock:
                self.active -= 1


class _Application:
    def __init__(self, workflow: _SlowReviewWorkflow, resource_root: Path) -> None:
        self.workflow = workflow
        self.resource_root = resource_root
        self.ui_test_mode = False


def test_concurrent_editor_display_cache_fills_are_serialized() -> None:
    workflow = _SlowReviewWorkflow()
    application = _Application(workflow, Path(__file__).resolve().parents[2])
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_DesktopHandler, application=application)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    def request() -> dict[str, object]:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/editor_data?display_budget=50000&span_id=S001",
            timeout=5,
        ) as response:
            assert response.status == 200
            return json.loads(response.read().decode("utf-8"))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            payloads = list(executor.map(lambda _index: request(), range(2)))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert workflow.maximum_active == 1
    assert all(payload["ok"] is True for payload in payloads)
    assert all(payload["display_budget"] == 50000 for payload in payloads)
    assert all(payload["review_span_id"] == "S001" for payload in payloads)
