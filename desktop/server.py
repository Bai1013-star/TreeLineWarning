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

from algorithms.folder_pipeline.html import step2_html

from desktop.editor_tasks import EditorTaskError, editor_tasks
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

    def _error(
        self,
        message: str,
        *,
        status: int = HTTPStatus.BAD_REQUEST,
        details: str | None = None,
    ) -> None:
        self._send_json(
            {
                "ok": False,
                "error": {
                    "code": "DESKTOP_WORKFLOW_ERROR",
                    "message": message,
                    "details": details,
                },
            },
            status=status,
        )

    def _exception_error(self, exc: Exception, *, status: int = HTTPStatus.CONFLICT) -> None:
        details = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, EditorTaskError):
            code = str(exc).split(":", 1)[0]
            self._send_json(
                {
                    "ok": False,
                    "status": code,
                    "error": {"code": code, "message": str(exc), "details": details},
                },
                status=HTTPStatus.CONFLICT,
            )
            return
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
            self._error(
                f"磁盘剩余空间不足。请释放项目盘 {self.application.workflow.project_root.anchor} 空间后重试。",
                status=status,
                details=details,
            )
            return
        self._error(details, status=status)

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
        if path in {"/dbh-review", "/dbh_review.html"}:
            self._static("viewer_poc/giro3d/dbh_review.html")
            return
        if path in {"/dbh_review.js", "/dbh_review.css"}:
            self._static("viewer_poc/giro3d/" + path.lstrip("/"))
            return
        if path == "/api/dbh-review/trees":
            from desktop.dbh_review import list_trees

            self._send_json(list_trees(self.workflow))
            return
        if path == "/api/viewer/runtime":
            from desktop.viewer_diagnostics import runtime_report

            self._send_json(runtime_report(self.application.resource_root))
            return
        if path == "/viewer-runtime.js":
            self._static("viewer_poc/giro3d/viewer_runtime.js")
            return
        if path == "/api/editor/mutation":
            self._send_json(editor_tasks(self.workflow).snapshot())
            return
        if path in {
            "/api/editor_data",
            "/api/desktop/review",
            "/api/state",
            "/state",
            "/api/current-report",
            "/api/desktop/current-report",
            "/api/final",
            "/api/future-preview",
        }:
            try:
                editor_tasks(self.workflow).assert_available()
            except EditorTaskError as exc:
                self._exception_error(exc)
                return
        if path in {"/", "/desktop.html"}:
            self._static("viewer_poc/giro3d/desktop.html")
            return
        if path == "/desktop.css":
            self._static("viewer_poc/giro3d/desktop.css")
            return
        if path == "/desktop.js":
            self._static("viewer_poc/giro3d/desktop.js")
            return
        if path == "/tower-picker-3d.js":
            self._static("viewer_poc/giro3d/tower_picker_3d.js")
            return
        if path == "/giro3d.bundle.js":
            self._static("viewer_poc/giro3d/vendor/giro3d.bundle.js")
            return
        if path == "/tree-review-3d.js":
            self._static("viewer_poc/giro3d/tree_review_3d.js")
            return
        if path == "/tree-review-app.js":
            self._static("viewer_poc/giro3d/tree_review_app.js")
            return
        if path == "/current-report.html":
            self._static("viewer_poc/giro3d/current_report.html")
            return
        if path in {"/future-preview.html", "/future_preview.js", "/future_preview.css"}:
            name = "future_preview.html" if path.endswith(".html") else path.lstrip("/")
            self._static("viewer_poc/giro3d/" + name)
            return
        if path == "/current_report.css":
            self._static("viewer_poc/giro3d/current_report.css")
            return
        if path == "/current_report.js":
            self._static("viewer_poc/giro3d/current_report.js")
            return
        if path == "/step2":
            self._send_bytes(step2_html().encode("utf-8"), content_type="text/html; charset=utf-8")
            return
        if path == "/step3":
            self._static("viewer_poc/giro3d/current_report.html")
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
        if path in {"/api/current-report", "/api/desktop/current-report"}:
            self._send_json({"ok": True, **self.workflow.current_report()})
            return
        if path == "/api/future-preview/gate":
            from desktop.future_preview import preview_gate

            self._send_json(
                {"ok": True, "gate": preview_gate(self.workflow, hydrate_editor=False)}
            )
            return
        if path in {"/api/future-preview", "/api/desktop/future-preview"}:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                if "unit" in query or "value" in query:
                    units = query.get("unit", [])
                    values = query.get("value", [])
                    if len(units) != 1 or len(values) != 1 or units[0] not in {
                        "days", "months", "years"
                    }:
                        raise ValueError("unit/value must specify days, months, or years")
                    value = int(values[0])
                    limits = {"days": (1, 1826), "months": (1, 60), "years": (1, 5)}
                    low, high = limits[units[0]]
                    if not low <= value <= high:
                        raise ValueError(f"{units[0]} value must be in {low}..{high}")
                    result = self.workflow.future_preview(
                        horizon_value=value,
                        horizon_unit=units[0],
                    )
                else:
                    values = query.get("horizon", ["1"])
                    if len(values) != 1 or values[0] not in {"1", "3", "5"}:
                        raise ValueError("horizon must be 1, 3, or 5 years")
                    result = self.workflow.future_preview(int(values[0]))
                self._send_json({"ok": True, **result})
            except ValueError as exc:
                self._exception_error(exc, status=HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/desktop/dependency-health":
            self._send_json({"ok": True, "dependency": dependency_health()})
            return
        if path in {"/api/desktop/review", "/api/editor_data"}:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                requested_budget = int(query.get("display_budget", ["8000"])[0])
            except (TypeError, ValueError):
                requested_budget = 8_000
            requested_span = query.get("span_id", [None])[0]
            self._send_json(
                {
                    "ok": True,
                    **self.workflow.editor_data(
                        display_budget=requested_budget, span_id=requested_span
                    ),
                }
            )
            return
        if path == "/state":
            self._send_json(self.workflow.editor_state())
            return
        if path == "/api/final":
            self._send_json(self.workflow.final_projection())
            return
        if path in {"/download_prediction", "/download_excel"}:
            try:
                output = self.workflow.export_prediction()
                self._send_bytes(
                    output.read_bytes(),
                    content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    download_name="Prediction_Input.xlsx",
                )
            except Exception as exc:
                self._exception_error(exc)
            return
        self._error("未找到本地资源", status=HTTPStatus.NOT_FOUND)

    def _handle_post(self, path: str) -> None:
        try:
            if path in {"/api/dbh-review/prepare", "/api/dbh-review/confirm"}:
                from desktop.dbh_review import confirm, prepare

                length = int(self.headers.get("Content-Length", "0") or 0)
                origin = self.headers.get("Origin")
                if length < 0 or length > 4096:
                    self._error("DBH_REVIEW_REQUEST_TOO_LARGE", status=413)
                    return
                if origin and origin != "http://" + self.headers.get("Host", ""):
                    self._error("DBH_REVIEW_ORIGIN_MISMATCH", status=403)
                    return
                payload = _json_body(self)
                asynchronous = payload.pop("async", False)
                request_id = payload.pop("request_id", None)

                def callback():
                    return (prepare if path.endswith("prepare") else confirm)(
                        self.workflow, payload
                    )

                if asynchronous is True:
                    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
                        raise ValueError("DBH_REVIEW_REQUEST_ID_REQUIRED")
                    self._send_json(
                        editor_tasks(self.workflow).submit(
                            "DBH_REVIEW_PREVIEW"
                            if path.endswith("prepare")
                            else "DBH_REVIEW_CONFIRM",
                            callback,
                            request_id,
                        ),
                        status=HTTPStatus.ACCEPTED,
                    )
                else:
                    self._send_json(callback())
                return
            if path == "/api/viewer/diagnostic":
                from desktop.viewer_diagnostics import MAX_DIAGNOSTIC_BYTES, record_diagnostic

                length = int(self.headers.get("Content-Length", "0") or 0)
                if length < 0 or length > MAX_DIAGNOSTIC_BYTES:
                    self._error("VIEWER_DIAGNOSTIC_TOO_LARGE", status=413)
                    return
                origin = self.headers.get("Origin")
                if origin and origin != "http://" + self.headers.get("Host", ""):
                    self._error("VIEWER_DIAGNOSTIC_ORIGIN_MISMATCH", status=403)
                    return
                self._send_json(record_diagnostic(self.workflow, _json_body(self)))
                return
            payload = _json_body(self)
            editor_tasks(self.workflow).assert_available()
            actions = {
                "/api/delete": ("DELETE", lambda: self.workflow.delete_selected()),
                "/api/merge": ("MERGE", lambda: self.workflow.merge_selected()),
                "/api/split": ("SPLIT", lambda: self.workflow.split_selected(payload.get("seed_xyz"))),
                "/api/undo": ("UNDO", lambda: self.workflow.undo_editor()),
                "/api/trim/commit": (
                    "TRIM_COMMIT",
                    lambda: self.workflow.commit_trim(str(payload.get("preview_id") or "")),
                ),
                "/api/trim/lasso-preview": (
                    "LASSO_PREVIEW",
                    lambda: self.workflow.preview_trim_lasso(
                        int(payload.get("tree_id")),
                        list(payload.get("view_projection") or []),
                        list(payload.get("polygon_ndc") or []),
                    ),
                ),
                "/api/trim/component-preview": (
                    "COMPONENT_PREVIEW",
                    lambda: self.workflow.preview_trim_component(
                        int(payload.get("tree_id", -1)), list(payload.get("seed_xyz") or []), payload.get("picks")
                    ),
                ),
            }
            if payload.get("async") is True and path in actions:
                kind, callback = actions[path]
                request_id = str(payload.get("request_id") or "")[:128] or None
                self._send_json(
                    editor_tasks(self.workflow).submit(kind, callback, request_id),
                    status=HTTPStatus.ACCEPTED,
                )
                return
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
                        list(payload.get("span_ids") or []),
                        mode=str(payload.get("mode", "REAL")),
                        execution_mode=str(payload.get("execution_mode", "AUTO")),
                        acceleration_mode=str(payload.get("acceleration_mode", "AUTO")),
                    )
                )
                return
            if path == "/api/desktop/batch/retry":
                self._send_json(
                    self.workflow.retry_failed(execution_mode=payload.get("execution_mode"))
                )
                return
            if path == "/api/desktop/batch/finalize" or path == "/api/confirm_trees":
                self._send_json(self.workflow.start_finalize())
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
            if path == "/api/split":
                self._send_json(self.workflow.split_selected(payload.get("seed_xyz")))
                return
            if path == "/api/undo":
                self._send_json(self.workflow.undo_editor())
                return
            if path == "/api/trim/component-preview":
                self._send_json(
                    self.workflow.preview_trim_component(
                        int(payload.get("tree_id", -1)), list(payload.get("seed_xyz") or []), payload.get("picks")
                    )
                )
                return
            if path == "/api/trim/lasso-preview":
                self._send_json(
                    self.workflow.preview_trim_lasso(
                        int(payload.get("tree_id")),
                        list(payload.get("view_projection") or []),
                        list(payload.get("polygon_ndc") or []),
                    )
                )
                return
            if path == "/api/trim/commit":
                self._send_json(self.workflow.commit_trim(str(payload.get("preview_id") or "")))
                return
            if path == "/api/trim/cancel":
                self._send_json(self.workflow.cancel_trim(str(payload.get("preview_id") or "")))
                return
            self._error("未找到本地操作", status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            try:
                self.workflow._log("api_failure", path=path, error=f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
            self._exception_error(exc)

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
        try:
            self._handle_get(path)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # The client disconnected; do not terminate the local service.
        except Exception as exc:
            self._exception_error(exc)

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        self._handle_post(path)

    do_HEAD = do_GET

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
        try:
            self.workflow._log("http", method=self.command, path=self.path, message=fmt % args)
        except Exception:
            pass  # Logging failure must not abort send_response before JSON headers.


__all__ = ["DesktopApplication"]
