"""Local interactive application for the manual single-span workflow.

One local Python process serves a localhost browser UI.  The teacher sees the
real source cloud (deterministic sample, real coordinates), clicks Pole A and
Pole B approximately, presses 开始分析, later enters the 危险净空阈值 and
presses 生成最终结果.  The same process owns every stage — no separate
development handoffs, no cloud, no internet dependency.
"""

# ruff: noqa: E501  (long HTML/JS template content lines)
from __future__ import annotations

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from algorithms.manual_span.pipeline import (
    STAGE_LABELS,
    ManualSingleSpanRun,
    PoleRefinementError,
    _spatial_display_rows,
)


class WorkflowState:
    """Thread-safe live selection authority for the interactive application.

    ``WorkflowState`` is the only live source of truth for the selector.  The
    browser is a projection of ``snapshot()``; the pending JSON journal is an
    audit aid and is deliberately never auto-restored on server initialization.
    The confirmed ``manual_span_selection.json`` is written exactly once by the
    confirmation path and is invalidated by an explicit reset.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = "SELECTING"
        self.selection_mode = "IDLE"  # IDLE | SELECT_A | SELECT_B
        self.clicks: dict[str, dict[str, float]] = {"A": {}, "B": {}}
        self.poles: dict[str, dict[str, Any]] = {}
        self.stages: list[dict[str, Any]] = []
        self.current_stage = ""
        self.error: str | None = None
        self.summary: dict[str, Any] | None = None
        self.threshold_m: float | None = None

    def update(self, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, key, value)

    def append_stage(self, item: dict[str, Any]) -> None:
        with self._lock:
            self.stages = self.stages + [dict(item)]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            distance_xy = None
            if self.clicks["A"] and self.clicks["B"]:
                distance_xy = math.dist(
                    (self.clicks["A"]["x"], self.clicks["A"]["y"]),
                    (self.clicks["B"]["x"], self.clicks["B"]["y"]),
                )
            return {
                "state": self.state,
                "selection_mode": self.selection_mode,
                "clicks": json.loads(json.dumps(self.clicks)),
                "selection_distance_xy_m": distance_xy,
                "selection_complete": bool(self.clicks["A"] and self.clicks["B"]),
                "poles": json.loads(json.dumps(self.poles)),
                "stages": [dict(item) for item in self.stages],
                "current_stage": self.current_stage,
                "error": self.error,
                "summary": self.summary,
                "threshold_m": self.threshold_m,
                "stage_labels": STAGE_LABELS,
            }


def _make_progress_callback(state: WorkflowState):
    def _progress(stage: int, message: str, done: bool, elapsed: float | None) -> None:
        if done:
            state.append_stage(
                {
                    "stage": stage,
                    "message": message,
                    "done": True,
                    "elapsed": elapsed,
                }
            )
            state.update(current_stage="")
        else:
            state.update(current_stage=f"[{stage}/10] {message}")

    return _progress


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _pending_selection_path(run: Any) -> Path | None:
    run_root = getattr(run, "run_root", None)
    if run_root is None:
        return None
    return Path(run_root) / "03_selection" / "manual_selector_state.json"


def _persist_pending_selection(run: Any, state: WorkflowState) -> None:
    """Journal exact pending source-frame clicks; never use it for auto-restore."""

    path = _pending_selection_path(run)
    if path is None:
        return
    with state._lock:
        payload = {
            "schema_version": "manual-selector-state-v1",
            "status": "PENDING_UNCONFIRMED",
            "source_sha256": str(getattr(run, "source_sha256", "")),
            "source_identity": str(getattr(run, "source_id", "")),
            "coordinate_frame": "SOURCE_LAS_ABSOLUTE",
            "selection_mode": state.selection_mode,
            "clicks": json.loads(json.dumps(state.clicks)),
        }
    _write_json_atomic(path, payload)


def _clear_run_selection(run: Any) -> None:
    """Invalidate every selection artifact owned by this interactive run."""

    for name, value in (
        ("pole_a", None),
        ("pole_b", None),
        ("span", None),
        ("selection_payload", None),
        ("span_digest", ""),
    ):
        setattr(run, name, value)
    for path in (
        _pending_selection_path(run),
        (Path(run.run_root) / "03_selection" / "manual_span_selection.json")
        if getattr(run, "run_root", None) is not None
        else None,
        (Path(run.run_root) / "03_selection" / "selection.checkpoint.json")
        if getattr(run, "run_root", None) is not None
        else None,
    ):
        if path is not None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _delete_pending_selection(run: Any) -> None:
    path = _pending_selection_path(run)
    if path is not None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _source_bounds(run: Any, xyz: Any) -> tuple[float, float, float, float, float, float]:
    inspection = getattr(run, "inspection", None)
    if isinstance(inspection, dict):
        minimum = inspection.get("bounds_min_xyz")
        maximum = inspection.get("bounds_max_xyz")
        if minimum is not None and maximum is not None and len(minimum) == 3 and len(maximum) == 3:
            return tuple(float(value) for value in (*minimum, *maximum))  # type: ignore[return-value]
    import numpy as np

    array = np.asarray(xyz, dtype=np.float64)
    if len(array) == 0:
        return (math.inf, math.inf, math.inf, -math.inf, -math.inf, -math.inf)
    minimum = np.min(array, axis=0)
    maximum = np.max(array, axis=0)
    return tuple(float(value) for value in (*minimum, *maximum))  # type: ignore[return-value]


def _selection_is_valid(
    clicks: dict[str, dict[str, float]],
    bounds: tuple[float, float, float, float, float, float],
) -> bool:
    if not clicks.get("A") or not clicks.get("B"):
        return False
    minimum = bounds[:3]
    maximum = bounds[3:]
    values: list[tuple[float, float, float]] = []
    for label in ("A", "B"):
        click = clicks[label]
        try:
            xyz = tuple(float(click[axis]) for axis in ("x", "y", "z"))
        except (KeyError, TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in xyz):
            return False
        if not all(lo <= value <= hi for value, lo, hi in zip(xyz, minimum, maximum, strict=True)):
            return False
        values.append(xyz)
    return math.dist(values[0][:2], values[1][:2]) > 1e-6


def run_interactive(
    source: Path,
    run_root: Path,
    *,
    cfg: Any = None,
    manual: Any = None,
    ground_reuse_dir: Any = None,
    port: int = 0,
    open_browser: bool = True,
) -> dict[str, Any]:
    """Run the full interactive workflow and return the final summary."""

    state = WorkflowState()
    run = ManualSingleSpanRun(
        source,
        run_root,
        cfg=cfg,
        manual=manual,
        progress=_make_progress_callback(state),
        ground_reuse_dir=ground_reuse_dir,
    )
    run.initialize()
    handler_factory = _make_handler_factory(run, state)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler_factory)
    actual_port = httpd.server_address[1]
    server_url = f"http://127.0.0.1:{actual_port}/"
    if open_browser:
        import webbrowser

        try:
            webbrowser.open(server_url)
        except OSError:  # pragma: no cover - best effort
            pass

    def _serve() -> None:
        try:
            httpd.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:  # pragma: no cover
            pass

    thread = threading.Thread(target=_serve, name="manual-span-server", daemon=True)
    thread.start()

    try:
        while state.state not in {"DONE", "ERROR"}:
            if state.state == "ERROR" and state.error is not None:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
    summary = state.summary or {}
    summary.setdefault("RUN_STATUS", "INTERACTIVE_CLOSED")
    summary.setdefault("SERVER_URL", server_url)
    return summary


def _make_handler_factory(run: ManualSingleSpanRun, state: WorkflowState):
    import numpy as np

    _spatial_rows = _spatial_display_rows  # module global bound into factory scope
    sample = run.display_sample or {}
    xyz = sample.get("xyz_absolute", np.empty((0, 3), dtype=np.float64))
    rgb = sample.get("rgb", np.empty((0, 3), dtype=np.uint8))
    # Bounded deterministic display subset (browser-friendly; selection stays in
    # the real LAS frame and automatic refinement localizes the pole precisely).
    _MAX_DISPLAY = 80_000
    if len(xyz) > _MAX_DISPLAY:
        rows = _spatial_rows(xyz, _MAX_DISPLAY)
        xyz = xyz[rows]
        rgb = rgb[rows] if len(rgb) == sample.get("xyz_absolute", np.empty((0, 3))).shape[0] else rgb
    if len(rgb) != len(xyz):
        rgb = np.tile(np.asarray((160, 180, 210), dtype=np.uint8), (len(xyz), 1))
    rgb = np.asarray(rgb, dtype=np.uint8)
    bounds = _source_bounds(run, xyz)
    display_source = str(np.asarray(sample.get("display_source", ["UNKNOWN"])).ravel()[0])
    # The 3D view is deliberately secondary and smaller.  Keep vertical bands
    # from the same source-bound display representation without sending another
    # large copy to the browser.
    xyz3d = xyz
    rgb3d = rgb
    if len(xyz) > 30_000:
        rows3d = _spatial_rows(xyz, 30_000)
        xyz3d = xyz[rows3d]
        rgb3d = rgb[rows3d]
    points_payload = {
        "xyz": xyz.tolist(),
        "rgb": rgb.tolist(),
        "xyz3d": xyz3d.tolist(),
        "rgb3d": rgb3d.tolist(),
        "count": int(len(xyz)),
        "count3d": int(len(xyz3d)),
        "display_source": display_source,
        "source_bounds": [*bounds[:3], *bounds[3:]],
    }

    def _nearest_display_z(x: float, y: float) -> float | None:
        if len(xyz) == 0:
            return None
        delta = xyz[:, :2] - np.asarray((x, y), dtype=np.float64)[None, :]
        row = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
        return float(xyz[row, 2])

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

        def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self._send(200, _selector_html().encode("utf-8"), "text/html; charset=utf-8")
                return
            if self.path == "/plotly.js":
                try:
                    import plotly

                    body = plotly.offline.get_plotlyjs().encode("utf-8")
                    self._send(200, body, "application/javascript")
                except Exception:  # pragma: no cover
                    self._send(500, b"plotly unavailable", "text/plain")
                return
            if self.path == "/state":
                self._send_json(200, state.snapshot())
                return
            if self.path == "/points":
                self._send(200, json.dumps(points_payload).encode("utf-8"))
                return
            if self.path.startswith("/local_points"):
                self._handle_local_points()
                return
            self._send(404, b"not found", "text/plain")

        def _handle_local_points(self) -> None:
            from urllib.parse import parse_qs, urlparse

            query = parse_qs(urlparse(self.path).query)
            try:
                cx = float(query["cx"][0])
                cy = float(query["cy"][0])
                radius = float(query.get("r", ["6.0"])[0])
            except (KeyError, ValueError, IndexError):
                self._send_json(400, {"error": "需要 cx, cy, r 查询参数"})
                return
            try:
                data = run.local_raw_for_xy(cx, cy, radius=radius)
            except Exception as exc:  # pragma: no cover - surfaced to UI
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            xyz = data.get("xyz_absolute", np.empty((0, 3), dtype=np.float64))
            rgb = data.get("rgb", np.empty((0, 3), dtype=np.uint8))
            if len(xyz) > 60_000:
                rows = _spatial_rows(xyz, 60_000)
                xyz = xyz[rows]
                rgb = rgb[rows] if len(rgb) == 0 or len(rgb) == len(data.get("xyz_absolute", np.empty((0, 3)))) else rgb
            self._send(
                200,
                json.dumps(
                    {"xyz": xyz.tolist(), "rgb": np.asarray(rgb, dtype=np.uint8).tolist(),
                     "count": int(len(xyz))}
                ).encode("utf-8"),
            )

        def do_POST(self) -> None:  # noqa: N802
            try:
                self._route_post()
            except PoleRefinementError as exc:
                state.update(error=f"{exc}")
                self._send_json(
                    200,
                    {
                        **state.snapshot(),
                        "refinement_failure": {
                            "pole": exc.label,
                            "reason": exc.reason,
                            "refinement_diagnostics": dict(exc.refinement_diagnostics),
                        },
                    },
                )
            except Exception as exc:  # pragma: no cover - closed failure to UI
                state.update(error=f"{type(exc).__name__}: {exc}")
                self._send_json(500, state.snapshot())

        def _route_post(self) -> None:
            payload = self._read_json()
            path = self.path
            if path == "/mode":
                pole = str(payload.get("pole", ""))
                if pole not in {"A", "B"}:
                    self._send_json(400, {"error": "invalid pole mode"})
                    return
                with state._lock:
                    if state.state != "SELECTING":
                        self._send_json(400, {"error": "当前阶段不能重新选择"})
                        return
                    state.selection_mode = f"SELECT_{pole}"
                    state.error = None
                _persist_pending_selection(run, state)
                self._send_json(200, state.snapshot())
                return
            if path == "/click":
                pole = str(payload.get("pole", ""))
                if pole not in {"A", "B"} or not all(
                    isinstance(payload.get(axis), (int, float))
                    and math.isfinite(float(payload.get(axis)))
                    for axis in ("x", "y")
                ):
                    self._send_json(400, {"error": "invalid click"})
                    return
                x = float(payload["x"])
                y = float(payload["y"])
                if not (bounds[0] <= x <= bounds[3] and bounds[1] <= y <= bounds[4]):
                    self._send_json(400, {"error": "click outside source bounds"})
                    return
                with state._lock:
                    if state.state != "SELECTING":
                        self._send_json(400, {"error": "当前阶段不能点击"})
                        return
                    if state.selection_mode != f"SELECT_{pole}":
                        self._send_json(400, {"error": f"请先按选择杆 {pole}"})
                        return
                    supplied_z = payload.get("z")
                    z = (
                        float(supplied_z)
                        if isinstance(supplied_z, (int, float)) and math.isfinite(float(supplied_z))
                        else _nearest_display_z(x, y)
                    )
                    if z is None or not bounds[2] <= z <= bounds[5]:
                        self._send_json(400, {"error": "无法从显示点云推导有效高度"})
                        return
                    state.clicks[pole] = {
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                    state.selection_mode = "IDLE"
                    state.error = None
                _persist_pending_selection(run, state)
                # Pre-recover the RAW local box in the background so the dense
                # local view and raw-based refinement are ready promptly.
                local_recover = getattr(run, "local_raw_for_xy", None)
                if local_recover is not None:
                    threading.Thread(
                        target=local_recover,
                        args=(x, y, run.manual.refine_search_radius_m + 1.0),
                        daemon=True,
                    ).start()
                self._send_json(200, state.snapshot())
                return
            if path == "/refine":
                pole = str(payload.get("pole", ""))
                if pole not in {"A", "B"}:
                    self._send_json(400, {"error": "invalid pole"})
                    return
                with state._lock:
                    click = json.loads(json.dumps(state.clicks.get(pole) or {}))
                if not click:
                    self._send_json(400, {"error": f"尚未选择杆 {pole}"})
                    return
                try:
                    refined = run._refine_pole(
                        pole, (click["x"], click["y"], click["z"])
                    )
                except Exception as exc:  # pragma: no cover - surfaced to UI
                    self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
                    return
                self._send_json(
                    200,
                    {
                        "pole": pole,
                        "defensible": refined.defensible,
                        "reason": refined.reason,
                        "refinement_diagnostics": dict(refined.refinement_diagnostics),
                        "refined_pole": refined.to_dict(),
                    },
                )
                return
            if path == "/reselect":
                with state._lock:
                    if state.state in {"CONFIRMING", "ANALYZING", "FINALIZING"}:
                        self._send_json(400, {"error": "分析进行中，不能重置选择"})
                        return
                    state.clicks = {"A": {}, "B": {}}
                    state.selection_mode = "IDLE"
                    state.poles = {}
                    state.error = None
                    state.stages = []
                    state.current_stage = ""
                    state.summary = None
                    state.threshold_m = None
                    state.state = "SELECTING"
                _clear_run_selection(run)
                self._send_json(200, state.snapshot())
                return
            if path == "/confirm":
                with state._lock:
                    clicks = json.loads(json.dumps(state.clicks))
                    current_state = state.state
                if current_state != "SELECTING" or not _selection_is_valid(clicks, bounds):
                    self._send_json(400, {"error": "需要两个不同且位于源范围内的有效电杆选择"})
                    return
                # Reserve confirmation under the same lock used by the state
                # snapshot so concurrent browser requests cannot persist the
                # source-bound selection twice.
                with state._lock:
                    if state.state != "SELECTING":
                        self._send_json(400, {"error": "选择正在确认"})
                        return
                    state.state = "CONFIRMING"
                click_a = (clicks["A"]["x"], clicks["A"]["y"], clicks["A"]["z"])
                click_b = (clicks["B"]["x"], clicks["B"]["y"], clicks["B"]["z"])
                try:
                    run.refine_and_build_span(click_a, click_b)
                except Exception:
                    state.update(state="SELECTING")
                    raise
                _delete_pending_selection(run)
                state.update(
                    poles={
                        "A": run.pole_a.to_dict() if run.pole_a else {},
                        "B": run.pole_b.to_dict() if run.pole_b else {},
                    },
                    selection_mode="IDLE",
                    state="CONFIRMED",
                    error=None,
                )
                self._send_json(200, state.snapshot())
                return
            if path == "/analyze":
                if state.state != "CONFIRMED":
                    self._send_json(400, {"error": "尚未确认选择"})
                    return
                state.update(state="ANALYZING", error=None)

                def _work() -> None:
                    try:
                        run.run_analysis()
                        state.update(state="AWAIT_THRESHOLD", current_stage="")
                    except Exception as exc:  # pragma: no cover - surfaced to UI
                        state.update(state="ERROR", error=f"{type(exc).__name__}: {exc}")

                threading.Thread(target=_work, name="manual-span-analysis", daemon=True).start()
                self._send_json(200, state.snapshot())
                return
            if path == "/threshold":
                threshold = float(payload.get("threshold_m", 0.0))
                if not (
                    run.manual.risk_threshold_min_m
                    <= threshold
                    <= run.manual.risk_threshold_max_m
                ):
                    self._send_json(400, {"error": "阈值超出有效范围"})
                    return
                state.update(state="FINALIZING", error=None)

                def _finalize() -> None:
                    try:
                        summary = run.finalize(threshold, open_results=True)
                        state.update(state="DONE", summary=summary, threshold_m=threshold)
                    except Exception as exc:  # pragma: no cover - surfaced to UI
                        state.update(state="ERROR", error=f"{type(exc).__name__}: {exc}")

                threading.Thread(target=_finalize, name="manual-span-finalize", daemon=True).start()
                self._send_json(200, state.snapshot())
                return
            self._send(404, b"not found", "text/plain")

    return _Handler


def _selector_html() -> str:
    """Two-panel selector: clickable 2D top-down XY (primary) + 3D preview.

    The top-down XY view is the authoritative click surface and uses RAW
    pre-denoise / pre-ground source points.  A point click (Plotly) or a native
    click anywhere on the plot (converted to real-LAS XY via the axis
    transform) selects the approximate pole location; the click is persisted
    through the local server and refined against the RAW local points.  Color
    can be toggled between original RGB and elevation/height.  After Pole A/B
    is clicked the 3D panel shows the local RAW crop for confirmation.
    """
    return """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>人工单档距分析 - 选杆</title>
<script src="/plotly.js"></script>
<style>
body { font-family: "Microsoft YaHei", sans-serif; margin: 0; background: #f4f6fb; color: #222; }
#app { max-width: 1400px; margin: 0 auto; padding: 14px; }
h1 { color: #0b3d91; margin: 6px 0; }
#steps { background: #eef3fb; padding: 8px 14px; border-radius: 6px; margin-bottom: 10px; font-size: 14px; }
#panels { display: flex; gap: 12px; flex-wrap: wrap; }
.panel { flex: 1 1 540px; min-width: 480px; background: #fff; border-radius: 8px;
         box-shadow: 0 1px 4px rgba(0,0,0,.12); padding: 8px; }
.panel h2 { font-size: 15px; color: #0b3d91; margin: 4px 8px; }
#chart2d { width: 100%; height: 620px; }
#chart3d { width: 100%; height: 620px; }
#controls { margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
button { padding: 9px 18px; font-size: 15px; border: none; border-radius: 6px; cursor: pointer; }
button:disabled { opacity: .45; cursor: not-allowed; }
#btn-select-a, #btn-select-b { background: #1565c0; color: #fff; }
#btn-select-a.active, #btn-select-b.active { background: #ef6c00; box-shadow: 0 0 0 3px #ffcc80; }
#btn-confirm { background: #0b3d91; color: #fff; font-size: 17px; }
#btn-reselect { background: #9e9e9e; color: #fff; }
#btn-full { background: #546e7a; color: #fff; }
#btn-color { background: #eceff1; color: #222; }
#btn-color.active { background: #0b3d91; color: #fff; }
#btn-local { background: #6a1b9a; color: #fff; }
#statusbar { margin: 8px 0; font-size: 16px; font-weight: 600; color: #0b3d91; }
#poleinfo { font-size: 15px; margin: 4px 0; }
#distance { font-size: 15px; color: #0b3d91; }
#refinemessage { margin: 6px 0; padding: 8px 12px; border-radius: 6px; font-size: 14px; }
#refinemessage.ok { background: #e8f5e9; color: #1b5e20; }
#refinemessage.fail { background: #ffebee; color: #b71c1c; }
#progress { margin-top: 10px; background: #fff; padding: 12px 16px; border-radius: 8px; min-height: 40px; }
#threshold { margin-top: 10px; background: #fff3e0; padding: 12px 16px; border-radius: 8px; }
#threshold input { width: 120px; padding: 8px; font-size: 16px; border: 1px solid #bbb; border-radius: 6px; }
#btn-final { background: #d32f2f; color: #fff; font-size: 17px; }
#done { margin-top: 10px; background: #e8f5e9; padding: 14px 16px; border-radius: 8px; }
#error { color: #b71c1c; }
</style></head><body>
<div id="app">
<h1>人工单档距分析（MANUAL_SINGLE_SPAN）— 选杆</h1>
<div id="steps">在 <b>左上俯视 XY 图</b> 中：① 点击 <b>电杆A</b> → ② 点击 <b>电杆B</b> → ③ 点 <b>确认并开始分析</b> → ④ 输入<b>危险净空阈值</b> → ⑤ 生成最终结果</div>
<div id="statusbar">状态：等待选择。请先按“选择杆 A”</div>
<div id="poleinfo"></div>
<div id="distance"></div>
<div id="refinemessage"></div>
<div id="panels">
  <div class="panel"><h2>俯视 XY 点云（真实坐标 / 原始 RGB·RAW 源点 / 可点击选杆）</h2><div id="chart2d"></div></div>
  <div class="panel"><h2>3D 预览（点击杆后显示该点局部 RAW 原始点云确认）</h2><div id="chart3d"></div></div>
</div>
<div id="controls">
  <button id="btn-select-a">选择杆 A</button>
  <button id="btn-select-b">选择杆 B</button>
  <button id="btn-color">色彩：RGB</button>
  <button id="btn-local">加载局部原始点云</button>
  <button id="btn-full">全场</button>
  <button id="btn-reselect">重新选择</button>
  <button id="btn-confirm" disabled>确认并开始分析</button>
</div>
<div id="threshold" hidden>
  <label for="thr"><b>危险净空阈值（m）</b></label>
  <input id="thr" type="number" step="0.1" min="0.5" max="20" value="5.0">
  <button id="btn-final">生成最终结果</button>
</div>
<div id="progress"></div>
<div id="done" hidden></div>
</div>
<script>
const PLOT2D = document.getElementById("chart2d");
const PLOT3D = document.getElementById("chart3d");
  let g2d = null, g3d = null;
  let state = null;
  let POINTS = [];
  let clicks = { A: null, B: null };
  let selectionMode = "IDLE";
  let colorMode = "rgb";
  let local3dData = null;
  let local3dLabel = "";
  let pollId = null;

async function fetchJSON(url, body) {
  const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const res = await fetch(url, opt);
  return res.json();
}

async function loadPoints() {
  const res = await fetch("/points");
  const data = await res.json();
  POINTS = data.xyz || [];
  const rgb = data.rgb || [];
  const color = (rgb && rgb.length === POINTS.length)
    ? rgb.map(r => `rgb(${Math.max(0, Math.min(255, r[0]))},${Math.max(0, Math.min(255, r[1]))},${Math.max(0, Math.min(255, r[2]))})`)
    : "rgb(160,180,210)";
  const xyz3d = data.xyz3d || POINTS;
  const rgb3d = data.rgb3d || rgb;
  const color3d = (rgb3d && rgb3d.length === xyz3d.length)
    ? rgb3d.map(r => `rgb(${Math.max(0, Math.min(255, r[0]))},${Math.max(0, Math.min(255, r[1]))},${Math.max(0, Math.min(255, r[2]))})`)
    : "rgb(160,180,210)";
  const xs = POINTS.map(p => p[0]);
  const ys = POINTS.map(p => p[1]);
  const zs = POINTS.map(p => p[2]);
  function minmax(values) {
    if (!values.length) return [0, 1];
    return values.reduce((range, value) => [Math.min(range[0], value), Math.max(range[1], value)], [Infinity, -Infinity]);
  }
  function quantile(values, fraction) {
    if (!values.length) return 0;
    const ordered = values.slice().sort((a, b) => a - b);
    const position = (ordered.length - 1) * fraction;
    const lower = Math.floor(position), upper = Math.ceil(position);
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower);
  }
  function displayRange(values) {
    if (!values.length) return [0, 1];
    const full = minmax(values);
    const robust = [quantile(values, 0.005), quantile(values, 0.995)];
    const span = Math.max(robust[1] - robust[0], (full[1] - full[0]) * 0.01, 1e-6);
    const pad = span * 0.06;
    return [robust[0] - pad, robust[1] + pad];
  }
  window._fullBounds = data.source_bounds && data.source_bounds.length === 6
    ? { x: [data.source_bounds[0], data.source_bounds[3]], y: [data.source_bounds[1], data.source_bounds[4]] }
    : { x: minmax(xs), y: minmax(ys) };
  // Primary surface uses canvas-based ``scatter`` (not the WebGL variant) so
  // the RAW overview renders even when WebGL is unavailable in the browser.
  const trace2d = {
    type: "scatter", mode: "markers",
    x: xs, y: ys,
    marker: { size: 3, color: color, opacity: 0.85 },
    name: `RAW 源点云（${data.display_source || "source"}）`
  };
  const layout2d = {
    margin: { l: 50, r: 20, t: 20, b: 40 },
    // Locked 1:1 XY aspect ratio (y anchored to x) so poles are not distorted;
    // the full RAW scene fills the panel while keeping equal X/Y scale.
    xaxis: { title: "X (m)", range: displayRange(xs) },
    yaxis: { title: "Y (m)", range: displayRange(ys), scaleanchor: "x", scaleratio: 1 },
    showlegend: true,
    uirevision: "manual-selector-v1",
    dragmode: "pan"
  };
  window._rgbColors = color;
  window._cloudZs = zs;
  g2d = document.createElement("div");
  g2d.style.width = "100%";
  g2d.style.height = "620px";
  PLOT2D.appendChild(g2d);
  Plotly.newPlot(g2d, [trace2d], layout2d, { responsive: true, displaylogo: false, scrollZoom: true });
  // Point click selects (Plotly supplies coordinates after its own transforms).
  let plotlyClickTs = 0;
  g2d.on("plotly_click", (data) => {
    const pt = data.points && data.points[0];
    if (!pt) return;
    plotlyClickTs = Date.now();
    registerClick(pt.x, pt.y);
  });
  // Native click anywhere on the plot converts to real-LAS XY (zoom-safe) so
  // the owner can click NEAR a pole without hitting an exact point pixel.
  g2d.addEventListener("click", (evt) => {
    if (Date.now() - plotlyClickTs < 120) return;
    const rect = g2d.getBoundingClientRect();
    const xa = g2d._fullLayout.xaxis;
    const ya = g2d._fullLayout.yaxis;
    const x = xa.p2c(evt.clientX - rect.left - xa._offset);
    const y = ya.p2c(evt.clientY - rect.top - ya._offset);
    registerClick(x, y);
  });

  const trace3d = {
    type: "scatter3d", mode: "markers",
    x: xyz3d.map(p => p[0]), y: xyz3d.map(p => p[1]), z: xyz3d.map(p => p[2]),
    marker: { size: 2.5, color: color3d, opacity: 0.72 },
    name: "3D 预览（次要）"
  };
  g3d = document.createElement("div");
  g3d.style.width = "100%";
  g3d.style.height = "620px";
  PLOT3D.appendChild(g3d);
  Plotly.newPlot(g3d, [trace3d], {
    margin: { l: 0, r: 0, t: 20, b: 0 },
    scene: { aspectmode: "data", xaxis: { title: "X" }, yaxis: { title: "Y" }, zaxis: { title: "Z" } },
    showlegend: true
  }, { responsive: true, displaylogo: false, scrollZoom: true });
}

function registerClick(x, y) {
  if (!state || state.state !== "SELECTING" || !["SELECT_A", "SELECT_B"].includes(selectionMode)) return;
  const pole = selectionMode === "SELECT_A" ? "A" : "B";
  fetchJSON("/click", { pole: pole, x: Number(x), y: Number(y) }).then((next) => {
    if (next.error) {
      state = next;
      updateUI();
      return;
    }
    state = next;
    syncSelectionFromState();
    render2d();
    updateUI();
    if (clicks[pole]) {
      refreshPoleRefinement(pole);
      loadLocalForXY(clicks[pole].x, clicks[pole].y, pole);
    }
  });
}

function applyColorMode() {
  if (!window._cloudTrace || !g2d) return;
  const base = window._cloudTrace;
  const updated = { ...base };
  if (colorMode === "height") {
    const zs = window._cloudZs || POINTS.map(p => p[2]);
    updated.marker = {
      size: 4,
      color: zs,
      colorscale: "Viridis",
      cmin: Math.min(...zs),
      cmax: Math.max(...zs),
      opacity: 0.82,
      colorbar: { title: "Z (m)" }
    };
  } else {
    updated.marker = { size: 4, color: window._rgbColors || "rgb(160,180,210)", opacity: 0.82 };
  }
  Plotly.react(g2d, [updated], window._cloudLayout, { responsive: true, displaylogo: false });
}

function render2d() {
  if (!g2d) return;
  applyColorMode();
  const markers = [];
  for (const [label, click] of Object.entries(clicks)) {
    if (click) {
      markers.push({
        type: "scatter", mode: "markers+text",
        x: [click.x], y: [click.y],
        text: [label === "A" ? "Pole A" : "Pole B"], textposition: "top center",
         marker: { size: 22, color: label === "A" ? "#ff9800" : "#d32f2f", symbol: "circle", line: { color: "#ffffff", width: 3 } },
        name: `Pole ${label}`
      });
    }
  }
  if (clicks.A && clicks.B) {
    markers.push({
      type: "scatter", mode: "lines",
      x: [clicks.A.x, clicks.B.x], y: [clicks.A.y, clicks.B.y],
      line: { color: "#0b3d91", width: 2, dash: "dot" },
      name: "Pole A → Pole B"
    });
  }
  const traces = [window._cloudTrace, ...markers].filter(Boolean);
  Plotly.react(g2d, traces, window._cloudLayout, { responsive: true, displaylogo: false });
}

function showRefinementResult(result) {
  const box = document.getElementById("refinemessage");
  if (!box) return;
  if (result && result.defensible) {
    const pole = result.pole || "";
    const d = result.refinement_diagnostics || {};
    const xy = d.refined_xy_absolute_m || [];
    const ext = d.vertical_extent_m != null ? d.vertical_extent_m.toFixed(2) : "?";
    box.className = "ok";
    box.innerHTML = `✅ 杆 ${pole} 细化成功：垂直延伸 ${ext} m，细化XY ${xy.length ? xy.map(v => Number(v).toFixed(2)).join(", ") : "?"}。`;
  } else if (result) {
    box.className = "fail";
    const d = result.refinement_diagnostics || {};
    const ext = d.vertical_extent_m != null ? `，垂直延伸 ${d.vertical_extent_m.toFixed(2)} m` : "";
    box.innerHTML = `⚠ 杆 ${result.pole || ""} 细化失败：${result.reason || "未知原因"}${ext}。点击仍保留，可重新点击或按“重新选择”。`;
  } else {
    box.className = "";
    box.innerHTML = "";
  }
}

async function refreshPoleRefinement(pole) {
  const click = clicks[pole];
  if (!click) return;
  const result = await fetchJSON("/refine", { pole: pole });
  showRefinementResult(result);
  if (result && result.defensible && result.refined_pole) {
    // update the refined base marker hint
    const base = result.refined_pole.base_xyz_absolute_m;
    window._refinedHints = window._refinedHints || {};
    window._refinedHints[pole] = { x: base[0], y: base[1], z: base[2] };
  }
}

async function loadLocalForXY(x, y, label) {
  const status = document.getElementById("statusbar");
  const prev = status.innerHTML;
  status.innerHTML = `正在加载杆 ${label} 附近的局部 RAW 原始点云（首次约 1 分钟）…`;
  try {
    const res = await fetch(`/local_points?cx=${encodeURIComponent(x)}&cy=${encodeURIComponent(y)}&r=6`);
    const data = await res.json();
    if (data.error) { status.innerHTML = prev + `　<span id="error">局部加载失败：${data.error}</span>`; return; }
    local3dData = data;
    local3dLabel = label;
    renderLocal3D(data, label);
    status.innerHTML = prev + `　✅ 已显示杆 ${label} 的局部 RAW 点云（${data.count} 点）`;
  } catch (err) {
    status.innerHTML = prev + `　<span id="error">局部加载失败：${err}</span>`;
  }
}

function renderLocal3D(data, label) {
  if (!g3d) return;
  const xyz = data.xyz || [];
  const rgb = data.rgb || [];
  const color = (rgb && rgb.length === xyz.length)
    ? rgb.map(r => `rgb(${Math.max(0, Math.min(255, r[0]))},${Math.max(0, Math.min(255, r[1]))},${Math.max(0, Math.min(255, r[2]))})`)
    : "rgb(160,180,210)";
  const trace = {
    type: "scatter3d", mode: "markers",
    x: xyz.map(p => p[0]), y: xyz.map(p => p[1]), z: xyz.map(p => p[2]),
    marker: { size: 2.5, color: color, opacity: 0.85 },
    name: `杆 ${label} 局部 RAW 原始点云`
  };
  Plotly.react(g3d, [trace], {
    margin: { l: 0, r: 0, t: 20, b: 0 },
    scene: { aspectmode: "data", xaxis: { title: "X" }, yaxis: { title: "Y" }, zaxis: { title: "Z" } },
    showlegend: true
  }, { responsive: true, displaylogo: false, scrollZoom: true });
}

async function loadLocalForCurrentView() {
  if (!g2d) return;
  const xa = g2d._fullLayout.xaxis, ya = g2d._fullLayout.yaxis;
  const cx = (xa.range[0] + xa.range[1]) / 2;
  const cy = (ya.range[0] + ya.range[1]) / 2;
  const r = Math.max((xa.range[1] - xa.range[0]) / 2, (ya.range[1] - ya.range[0]) / 2);
  const status = document.getElementById("statusbar");
  const prev = status.innerHTML;
  status.innerHTML = "正在加载当前视图的局部 RAW 原始点云（首次约 1 分钟）…";
  try {
    const res = await fetch(`/local_points?cx=${encodeURIComponent(cx)}&cy=${encodeURIComponent(cy)}&r=${Math.min(r, 60)}`);
    const data = await res.json();
    if (data.error) { status.innerHTML = prev + `　<span id="error">局部加载失败：${data.error}</span>`; return; }
    local3dData = data; local3dLabel = "view";
    renderLocal3D(data, "view");
    // Denser overlay on the 2D view.
    const xyz = data.xyz || [];
    const rgb = data.rgb || [];
    const color = (rgb && rgb.length === xyz.length)
      ? rgb.map(r2 => `rgb(${Math.max(0, Math.min(255, r2[0]))},${Math.max(0, Math.min(255, r2[1]))},${Math.max(0, Math.min(255, r2[2]))})`)
      : "rgb(160,180,210)";
    const dense = {
      type: "scatter", mode: "markers",
      x: xyz.map(p => p[0]), y: xyz.map(p => p[1]),
      marker: { size: 2, color: color, opacity: 0.9 },
      name: `局部 RAW 点云（${data.count} 点）`
    };
    const traces = [window._cloudTrace, dense, ...Object.entries(clicks).filter(([,c]) => c).map(([l, c]) => ({
      type: "scatter", mode: "markers+text", x: [c.x], y: [c.y],
      text: [l === "A" ? "Pole A" : "Pole B"], textposition: "top center",
      marker: { size: 22, color: l === "A" ? "#ff9800" : "#d32f2f", symbol: "circle" }, name: `Pole ${l}`
    }))].filter(Boolean);
    Plotly.react(g2d, traces, window._cloudLayout, { responsive: true, displaylogo: false });
    status.innerHTML = prev + `　✅ 当前视图已叠加 ${data.count} 个局部 RAW 点`;
  } catch (err) {
    status.innerHTML = prev + `　<span id="error">局部加载失败：${err}</span>`;
  }
}

function updateUI() {
  const bar = document.getElementById("statusbar");
  const info = document.getElementById("poleinfo");
  const dist = document.getElementById("distance");
  const confirmBtn = document.getElementById("btn-confirm");
  if (selectionMode === "SELECT_A") {
    bar.innerHTML = "请选择 <b>杆 A</b>：请在俯视图中点击杆 A 附近";
  } else if (selectionMode === "SELECT_B") {
    bar.innerHTML = "请选择 <b>杆 B</b>：请在俯视图中点击杆 B 附近";
  } else if (clicks.A && clicks.B) {
    bar.innerHTML = "Pole A：已选择，Pole B：已选择 —— 可点击 <b>确认并开始分析</b>";
  } else if (clicks.A) {
    bar.innerHTML = "Pole A：已选择；Pole B：未选择。请按 <b>选择杆 B</b>";
  } else {
    bar.innerHTML = "Pole A：未选择；Pole B：未选择。请按 <b>选择杆 A</b>";
  }
  info.innerHTML = `Pole A：${clicks.A ? clicks.A.x.toFixed(2) + ", " + clicks.A.y.toFixed(2) : "未选择"}　|　Pole B：${clicks.B ? clicks.B.x.toFixed(2) + ", " + clicks.B.y.toFixed(2) : "未选择"}`;
  if (clicks.A && clicks.B) {
    const dx = clicks.B.x - clicks.A.x, dy = clicks.B.y - clicks.A.y;
    dist.innerHTML = `A→B 近似 XY 距离：${Math.sqrt(dx * dx + dy * dy).toFixed(2)} m`;
  } else { dist.innerHTML = ""; }
  const same = clicks.A && clicks.B && clicks.A.x === clicks.B.x && clicks.A.y === clicks.B.y;
  confirmBtn.disabled = !(clicks.A && clicks.B && !same && state && state.state === "SELECTING");
  document.getElementById("btn-select-a").classList.toggle("active", selectionMode === "SELECT_A");
  document.getElementById("btn-select-b").classList.toggle("active", selectionMode === "SELECT_B");
  if (state && state.error) info.innerHTML += `　<span id="error">⚠ ${state.error}</span>`;
  if (state && state.state === "AWAIT_THRESHOLD") document.getElementById("threshold").hidden = false;
}

function renderProgress() {
  const box = document.getElementById("progress");
  if (!state) return;
  let html = "";
  if (state.stages && state.stages.length) {
    html += "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'><tr><th>阶段</th><th>状态</th><th>耗时</th></tr>";
    for (const s of state.stages) {
      html += `<tr><td>${s.message}</td><td>完成</td><td>${s.elapsed ? s.elapsed.toFixed(1) + "s" : ""}</td></tr>`;
    }
    html += "</table>";
  }
  if (state.current_stage) html += `<div>${state.current_stage} …</div>`;
  if (state.state === "ERROR") html += `<div id="error">错误：${state.error}</div>`;
  if (state.state === "DONE") html += "<div>✅ 全部完成。</div>";
  box.innerHTML = html;
}

async function poll() {
  state = await fetchJSON("/state");
  syncSelectionFromState();
  updateUI();
  renderProgress();
  if (g2d) render2d();
  if (state.state === "DONE" && state.summary) {
    const done = document.getElementById("done");
    const s = state.summary;
    done.hidden = false;
    done.innerHTML = `<b>完成！</b> 档距 ${s.SPAN_LENGTH_M ? s.SPAN_LENGTH_M.toFixed(2) : "?"} m，树 ${s.FINAL_TREE_COUNT}，导线 ${s.WIRE_STRAND_COUNT}，危险树 ${s.DANGEROUS_TREE_COUNT}，危险点 ${s.DANGER_POINT_COUNT}，最小净空 ${s.GLOBAL_MINIMUM_CLEARANCE_M ? s.GLOBAL_MINIMUM_CLEARANCE_M.toFixed(3) : "?"} m。<br>结果目录：<code>${s.OUTPUT_DIRECTORY || ""}</code>`;
    clearInterval(pollId);
  }
  if (state.state === "ERROR") clearInterval(pollId);
}

function syncSelectionFromState() {
  clicks = {
    A: state && state.clicks && state.clicks.A && Object.keys(state.clicks.A).length ? state.clicks.A : null,
    B: state && state.clicks && state.clicks.B && Object.keys(state.clicks.B).length ? state.clicks.B : null
  };
  selectionMode = state && state.selection_mode ? state.selection_mode : "IDLE";
}

async function chooseMode(pole) {
  const next = await fetchJSON("/mode", { pole: pole });
  state = next;
  syncSelectionFromState();
  updateUI();
}

document.getElementById("btn-reselect").addEventListener("click", async () => {
  state = await fetchJSON("/reselect", {});
  syncSelectionFromState();
  render2d(); updateUI(); renderProgress();
});
document.getElementById("btn-select-a").addEventListener("click", () => chooseMode("A"));
document.getElementById("btn-select-b").addEventListener("click", () => chooseMode("B"));
document.getElementById("btn-color").addEventListener("click", () => {
  colorMode = colorMode === "rgb" ? "height" : "rgb";
  document.getElementById("btn-color").textContent = colorMode === "rgb" ? "色彩：RGB" : "色彩：高度";
  document.getElementById("btn-color").classList.toggle("active", colorMode === "height");
  render2d();
});
document.getElementById("btn-local").addEventListener("click", () => loadLocalForCurrentView());
document.getElementById("btn-full").addEventListener("click", () => {
  if (!g2d || !window._fullBounds) return;
  Plotly.relayout(g2d, {
    "xaxis.range": window._fullBounds.x,
    "yaxis.range": window._fullBounds.y
  });
});
document.getElementById("btn-confirm").addEventListener("click", async () => {
  state = await fetchJSON("/confirm", {});
  if (state.state === "CONFIRMED") {
    document.getElementById("statusbar").innerHTML = "✅ 已确认，开始自动分析…（完成后将询问阈值）";
    state = await fetchJSON("/analyze", {});
    pollId = setInterval(poll, 1200);
  } else {
    updateUI();
  }
  poll();
});
document.getElementById("btn-final").addEventListener("click", async () => {
  const value = parseFloat(document.getElementById("thr").value);
  state = await fetchJSON("/threshold", { threshold_m: value });
  pollId = setInterval(poll, 1200);
  poll();
});

window.addEventListener("load", async () => {
  try {
    await loadPoints();
    window._cloudTrace = g2d && g2d.data && g2d.data[0];
    window._cloudLayout = g2d && g2d.layout;
    if (!window._cloudTrace) {
      document.getElementById("statusbar").innerHTML +=
        '　<span id="error">点云渲染失败：Plotly 未创建轨迹</span>';
      return;
    }
    render2d();
  } catch (err) {
    document.getElementById("statusbar").innerHTML +=
      `　<span id="error">RAW 点云加载失败：${err && err.message ? err.message : err}</span>`;
    console.error("loadPoints error", err);
  }
  state = await fetchJSON("/state");
  syncSelectionFromState();
  updateUI(); renderProgress();
  pollId = setInterval(poll, 2000);
});
</script></body></html>
"""


__all__ = ["WorkflowState", "run_interactive", "_make_progress_callback"]
