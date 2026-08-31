from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
from algorithms.folder_pipeline.html import step2_html
from desktop.server import DesktopApplication
from desktop.viewer_diagnostics import record_diagnostic, runtime_report
from tests.desktop.test_stage_c_workflow import _workflow


def test_runtime_records_actual_assets_and_redacts_unrelated_flags(tmp_path, monkeypatch):
    import desktop.packaged_diagnostics as diagnostics

    monkeypatch.setattr(
        diagnostics,
        "webview_loader_info",
        lambda root, module_origin: {
            "status": "PASS",
            "browser_version": "151.0.1.2",
            "browser_version_minimum": "86.0.622.0",
        },
    )
    monkeypatch.setenv(
        "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "--disable-gpu --token=private --use-angle=d3d11"
    )
    report = runtime_report(tmp_path)
    assert not report["assets"]["/giro3d.bundle.js"]["exists"]
    assert report["browser_arguments"]["selected_environment_flags"] == [
        "--disable-gpu",
        "--use-angle=d3d11",
    ]
    assert "private" not in json.dumps(report)
    assert report["browser_arguments"]["application_gpu_override"] is False
    assert report["three_expected"] == "180"


def test_live_html_has_diagnostic_bootstrap_before_assets():
    page = step2_html()
    assert page.count('id="viewer-init-failure"') == 1
    assert page.index("__treeViewerResourceFailures") < page.index(
        '<script src="/viewer-runtime.js"'
    )
    assert page.index('<script src="/viewer-runtime.js"') < page.index(
        '<script src="/giro3d.bundle.js"'
    )
    for element in ("viewer-init-details", "btn-viewer-retry", "btn-viewer-copy"):
        assert f'id="{element}"' in page
    assert "3D Viewer could not start on this device." in page
    assert "VIEWER_RESOURCE_MISSING" in page


def request(base, route, payload=None, origin=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = origin
    req = urllib.request.Request(base + route.lstrip("/"), data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as response:
        return response.code, json.load(response)


def test_backend_and_project_survive_viewer_failures(tmp_path, monkeypatch):
    workflow = _workflow(tmp_path)
    app = DesktopApplication.__new__(DesktopApplication)
    app.workflow, app.resource_root = workflow, tmp_path / "missing-resources"
    app._server = app._thread = None
    app.ui_test_mode = False
    manifest = workflow.project_root / "stage-a-project.json"
    before = manifest.read_bytes()
    base = app.start()
    try:
        codes = [
            "WEBGL_UNAVAILABLE",
            "WEBGL_CONTEXT_CREATION_FAILED",
            "THREE_RENDERER_FAILED",
            "VIEWER_RESOURCE_MISSING",
            "GPU_DRIVER_BLOCKED",
            "WEBVIEW2_RUNTIME_INCOMPATIBLE",
            "UNKNOWN_VIEWER_INIT_FAILURE",
        ]
        for code in codes:
            diagnostic = {
                "failure_code": code,
                "exception": {
                    "name": "Error",
                    "message": "original browser error",
                    "stack": "original-stack",
                },
            }
            status, response = request(base, "/api/viewer/diagnostic", diagnostic)
            assert status == 200 and response["logged"] and response["latest_saved"]
            assert request(base, "/state")[0] == 200
        saved = json.loads((workflow.project_root / "logs/viewer_init.latest.json").read_text())
        assert saved["exception"]["stack"] == "original-stack"
        assert request(base, "/tree-review-3d.js")[0] == 404
        assert request(base, "/api/desktop/project")[0] == 200
        assert manifest.read_bytes() == before
        assert request(base, "/api/viewer/diagnostic", {"message": "x" * 70_000})[0] == 413
        assert request(base, "/api/viewer/diagnostic", {}, "https://unrelated.invalid")[0] == 403
        assert request(base, "/state")[0] == 200
        assert not list((workflow.project_root / "logs").glob("viewer_init.*.tmp"))
    finally:
        app.stop()


def test_viewer_diagnostic_logging_failure_is_nonfatal(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("injected read-only log directory")

    workflow = SimpleNamespace(project_root=tmp_path, _log=fail)
    monkeypatch.setattr(Path, "replace", fail)
    result = record_diagnostic(workflow, {"failure_code": "THREE_RENDERER_FAILED"})
    assert result == {"ok": True, "logged": False, "latest_saved": False}
    assert not list(tmp_path.rglob("*.tmp"))
    with pytest.raises(ValueError, match="TOO_LARGE"):
        record_diagnostic(workflow, {"message": "x" * 70_000})
