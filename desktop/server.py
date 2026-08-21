"""Loopback-only HTTP service for the offline Stage-C desktop shell."""

from __future__ import annotations

import json
import math
import mimetypes
import threading
import urllib.parse
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from algorithms.folder_pipeline.html import step2_html, step3_html

from desktop.pdal_runtime import dependency_health
from desktop.project_management import create_empty_project, import_pointcloud
from desktop.workflow import DesktopWorkflow, ensure_desktop_manifest


def _json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    payload = json.loads(raw.decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _strict_json_value(value: Any) -> Any:
    """Normalize diagnostics to browser-parseable RFC JSON values."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    return value


class DesktopApplication:
    """Own the workflow and HTTP lifecycle separately from PyWebView."""

    def __init__(
        self,
        project_root: Path,
        resource_root: Path,
        *,
        desktop_home: bool = False,
        ui_test_mode: bool = False,
    ) -> None:
        self.resource_root = resource_root.resolve()
        self.ui_test_mode = ui_test_mode
        self.workflow = DesktopWorkflow(
            project_root.resolve(),
            resource_root=self.resource_root,
            desktop_home=desktop_home,
        )
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def project_root(self) -> Path:
        return self.workflow.project_root

    def configure_project(self, project_root: Path) -> dict[str, Any]:
        ensure_desktop_manifest(project_root.resolve())
        return self.workflow.configure_root(project_root)

    def start(self) -> str:
        if self._server is not None:
            return f"http://127.0.0.1:{self._server.server_address[1]}/"
        handler = partial(_DesktopHandler, application=self)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="treeline-desktop-http", daemon=True
        )
        self._thread.start()
        self.workflow._log("local_server_started", port=self._server.server_address[1])
        return f"http://127.0.0.1:{self._server.server_address[1]}/"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        self._server = None
        self._thread = None
        self.workflow._log("local_server_stopped")


class _DesktopHandler(BaseHTTPRequestHandler):
    server_version = "TreeLineWarningDesktop/0.1"

    def __init__(self, *args: Any, application: DesktopApplication, **kwargs: Any) -> None:
        self.application = application
        super().__init__(*args, **kwargs)

    @property
    def workflow(self) -> DesktopWorkflow:
        return self.application.workflow

    def _send_bytes(
        self,
        body: bytes,
        *,
        status: int = HTTPStatus.OK,
        content_type: str = "application/octet-stream",
        download_name: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], *, status: int = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(
                _strict_json_value(payload),
                ensure_ascii=False,
                default=str,
                allow_nan=False,
            ).encode("utf-8"),
            status=status,
            content_type="application/json; charset=utf-8",
        )

    def _error(self, message: str, *, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json(
            {"ok": False, "error": {"code": "DESKTOP_WORKFLOW_ERROR", "message": message}},
            status=status,
        )

    def _static(self, relative: str) -> None:
        root = self.application.resource_root.resolve()
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            self._error("静态资源路径无效", status=HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            self._error(f"本地资源不存在: {relative}", status=HTTPStatus.NOT_FOUND)
            return
        mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if target.suffix.lower() == ".js":
            mime = "text/javascript; charset=utf-8"
        self._send_bytes(target.read_bytes(), content_type=mime)

    def _handle_get(self, path: str) -> None:
        if path in {"/", "/desktop.html"}:
            self._static("viewer_poc/giro3d/desktop.html")
            return
        if path == "/desktop.css":
            self._static("viewer_poc/giro3d/desktop.css")
            return
        if path == "/desktop.js":
            self._static("viewer_poc/giro3d/desktop.js")
            return
        if path == "/step2":
            self._send_bytes(step2_html().encode("utf-8"), content_type="text/html; charset=utf-8")
            return
        if path == "/step3":
            self._send_bytes(step3_html().encode("utf-8"), content_type="text/html; charset=utf-8")
            return
        if path == "/plotly.js":
            try:
                from plotly.offline import get_plotlyjs

                self._send_bytes(
                    get_plotlyjs().encode("utf-8"), content_type="text/javascript; charset=utf-8"
                )
            except Exception as exc:  # dependency diagnostic is returned locally
                self._error(
                    f"本地 Plotly 资源不可用: {type(exc).__name__}: {exc}",
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return
        if path == "/api/desktop/project":
            self._send_json({"ok": True, "project": self.workflow.project_projection()})
            return
        if path == "/api/desktop/workspace":
            self._send_json(
                {
                    "ok": True,
                    **self.workflow.workspace_projection(),
                    "test_mode": self.application.ui_test_mode,
                }
            )
            return
        if path == "/api/desktop/capability":
            self._send_json({"ok": True, "capability": self.workflow.capability_projection()})
            return
        if path == "/api/desktop/topology":
            project = self.workflow.project_projection()
            self._send_json(
                {
                    "ok": True,
                    "lines": project.get("line_records", []),
                    "towers": project.get("tower_records", []),
                    "spans": project.get("span_records", []),
                }
            )
            return
        if path == "/api/desktop/tower-viewer/overview":
            self._send_json({"ok": True, **self.workflow.tower_viewer_overview()})
            return
        if path == "/api/desktop/batch":
            self._send_json(self.workflow.batch_projection())
            return
        if path == "/api/desktop/dependency-health":
            self._send_json({"ok": True, "dependency": dependency_health()})
            return
        if path in {"/api/desktop/review", "/api/editor_data"}:
            self._send_json({"ok": True, **self.workflow.editor_data()})
            return
        if path == "/state":
            self._send_json(self.workflow.editor_state())
            return
        if path == "/api/final":
            self._send_json(self.workflow.final_projection())
            return
        if path == "/download_prediction":
            try:
                output = self.workflow.export_prediction()
                self._send_bytes(
                    output.read_bytes(),
                    content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    download_name="Prediction_Input.xlsx",
                )
            except Exception as exc:
                self._error(f"导出失败: {type(exc).__name__}: {exc}", status=HTTPStatus.CONFLICT)
            return
        self._error("未找到本地资源", status=HTTPStatus.NOT_FOUND)

    def _handle_post(self, path: str) -> None:
        try:
            payload = _json_body(self)
            if path == "/api/desktop/create":
                parent = (
                    Path(str(payload.get("parent") or self.workflow.project_root.parent))
                    .expanduser()
                    .resolve()
                )
                project = create_empty_project(
                    parent, str(payload.get("project_name") or "TreeLineWarning Project")
                )
                self.application.configure_project(Path(project["root"]))
                self._send_json({"ok": True, "project": self.workflow.project_projection()})
                return
            if path == "/api/desktop/fixture-source":
                source = self._create_fixture_source()
                imported = import_pointcloud(self.workflow.project_root, source)
                self._send_json(
                    {
                        "ok": True,
                        "project": self.application.configure_project(self.workflow.project_root),
                        "imported": imported,
                    }
                )
                return
            if path == "/api/desktop/capability/confirm":
                self._send_json(
                    {
                        "ok": True,
                        "capability": self.workflow.confirm_unit(str(payload.get("unit", ""))),
                    }
                )
                return
            if path == "/api/desktop/risk-policy":
                self._send_json(
                    {
                        "ok": True,
                        "risk_policy": self.workflow.set_risk_policy(
                            float(payload.get("warning_clearance_threshold_m"))
                        ),
                    }
                )
                return
            if path == "/api/desktop/towers":
                self._send_json({"ok": True, "project": self.workflow.save_manual_towers(payload)})
                return
            if path == "/api/desktop/tower-viewer/refine":
                self._send_json({"ok": True, **self.workflow.refine_tower_pick(payload)})
                return
            if path == "/api/desktop/batch/start":
                self._send_json(
                    self.workflow.start_batch(
                        list(payload.get("span_ids") or []), mode=str(payload.get("mode", "REAL"))
                    )
                )
                return
            if path == "/api/desktop/batch/retry":
                self._send_json(self.workflow.retry_failed())
                return
            if path == "/api/desktop/batch/finalize" or path == "/api/confirm_trees":
                self._send_json(self.workflow.finalize())
                return
            if path == "/api/select":
                self._send_json(
                    self.workflow.select_editor_trees(
                        [int(item) for item in payload.get("tree_ids", [])]
                    )
                )
                return
            if path == "/api/delete":
                self._send_json(self.workflow.delete_selected())
                return
            if path == "/api/merge":
                self._send_json(self.workflow.merge_selected())
                return
            if path == "/api/undo":
                self._send_json(self.workflow.undo_editor())
                return
            self._error("未找到本地操作", status=HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            self.workflow._log("api_failure", path=path, error=f"{type(exc).__name__}: {exc}")
            self._error(f"{type(exc).__name__}: {exc}", status=HTTPStatus.CONFLICT)

    def _create_fixture_source(self) -> Path:
        root = self.workflow.project_root / "data" / "raw"
        root.mkdir(parents=True, exist_ok=True)
        path = root / "offline-ui-fixture.las"
        if not path.is_file():
            import laspy
            import numpy as np

            header = laspy.LasHeader(point_format=3, version="1.2")
            header.scales = np.asarray([0.01, 0.01, 0.01])
            cloud = laspy.LasData(header)
            x = np.linspace(0.0, 40.0, 300)
            cloud.x = x
            cloud.y = np.sin(x / 5.0)
            cloud.z = 2.0 + np.abs(np.cos(x / 4.0)) * 8.0
            cloud.red = np.full(len(x), 120, dtype=np.uint16)
            cloud.green = np.full(len(x), 180, dtype=np.uint16)
            cloud.blue = np.full(len(x), 80, dtype=np.uint16)
            cloud.write(path)
        return path

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        self._handle_get(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        self._handle_post(path)

    do_HEAD = do_GET

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
        self.workflow._log("http", method=self.command, path=self.path, message=fmt % args)


__all__ = ["DesktopApplication"]
