"""Local three-step browser server for the Folder pipeline.

One ``ThreadingHTTPServer`` serves /step1 /step2 /step3 and the shared API.
STEP 1 reuses the proven manual selector interaction (Plotly top-down + 3D,
explicit pole modes, zoom-safe clicks); STEP 2 is the human tree editor
(DELETE / MERGE / UNDO); STEP 3 shows the final result and the Excel download.
After a span is confirmed the science runs on a background thread; the browser
polls /state and advances automatically.
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.folder_pipeline.html import step1_html, step2_html, step3_html
from algorithms.folder_pipeline.orchestrator import FolderOrchestrator, FolderPipelineError
from algorithms.folder_pipeline.state import (
    STEP_ERROR,
    STEP_SCIENCE_RUNNING,
    STEP_SELECTING_SPAN,
    STEP_TREE_EDITING,
    FolderWorkflowState,
)
from algorithms.folder_pipeline.workspace import FolderWorkspace

_MAX_DISPLAY_2D = 80_000
_MAX_DISPLAY_3D = 30_000
_MAX_TREE_DISPLAY = 60_000
_MAX_BACKGROUND_DISPLAY = 40_000
_MAX_WIRE_DISPLAY = 20_000
_MAX_DANGER_DISPLAY = 40_000


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", 0))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    payload = json.loads(raw.decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _even_rows(count: int, maximum: int) -> np.ndarray:
    if count <= 0 or maximum <= 0:
        return np.empty(0, dtype=np.int64)
    step = max(1, int(np.ceil(count / maximum)))
    return np.arange(0, count, step, dtype=np.int64)


class FolderServer:
    """Owns the workspace + orchestrator + workflow state; builds the handler."""

    def __init__(
        self,
        workspace: FolderWorkspace,
        *,
        state: FolderWorkflowState | None = None,
        threshold_m: float | None = None,
        headless_poles: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    ) -> None:
        self.workspace = workspace
        self.state = state if state is not None else FolderWorkflowState()
        # threshold_m=None -> the canonical project risk standard (never a new
        # Folder-pipeline default); the orchestrator resolves it.
        self.orchestrator = FolderOrchestrator(
            workspace, state=self.state, threshold_m=threshold_m
        )
        self._science_thread: threading.Thread | None = None
        self.headless_poles = headless_poles
        self._prepare()

    def _prepare(self) -> None:
        """Resume persisted checkpoints or start fresh."""

        try:
            self.orchestrator.initialize()
        except Exception as exc:  # noqa: BLE001 - surface a clear Chinese error
            self.state.set_error(
                "点云准备失败",
                _brief(exc)
                + "（提示：若本机缺少 PDAL 地面后端且该源点云没有历史地面缓存，"
                "将无法自动准备点云。请确认运行环境具备 PDAL，或先处理缓存问题。）",
            )
            return
        if self.headless_poles is not None:
            self._run_headless()
            return
        selection = self.orchestrator.load_selection()
        if selection is None:
            self.state.set_step(STEP_SELECTING_SPAN, "[1/3] 等待人工选择档距...")
            return
        if self.workspace.original_tree_npz.is_file():
            self.orchestrator.build_editor()
            self.state.set_step(STEP_TREE_EDITING, "等待人工审核单木...")
            return
        # Selection confirmed but science not finished -> (re)start science.
        self.state.set_step(STEP_SCIENCE_RUNNING, "[2/3] 正在执行科学分析...")
        self._start_science()

    # ------------------------------------------------------------- background
    def _start_science(self) -> None:
        def _work() -> None:
            try:
                self.orchestrator.run_science()
                self.orchestrator.build_editor()
                self.state.set_step(STEP_TREE_EDITING, "等待人工审核单木...")
            except Exception as exc:  # noqa: BLE001 - surface to the UI
                self.state.set_error("科学分析失败", _brief(exc))

        if self._science_thread is None or not self._science_thread.is_alive():
            self._science_thread = threading.Thread(target=_work, daemon=True)
            self._science_thread.start()

    def _run_headless(self) -> None:
        """Headless driver used by automated smoke runs (not the browser path)."""

        if self.headless_poles is None:
            return
        click_a, click_b = self.headless_poles

        def _auto() -> None:
            try:
                if not self.orchestrator.has_selection():
                    self.orchestrator.apply_selection(click_a, click_b)
                    self._start_science()
                # wait for science to build the editor
                for _ in range(7200):
                    if self.orchestrator.edit_session is not None:
                        break
                    if self.state.step == STEP_ERROR:
                        return
                    time.sleep(0.5)
                self.orchestrator.confirm_trees_and_finalize()
            except Exception as exc:  # noqa: BLE001
                self.state.set_error("headless 失败", _brief(exc))

        threading.Thread(target=_auto, daemon=True).start()

    # --------------------------------------------------------------- handlers
    def handler_factory(self):
        state_ref = self
        folder_server = self

        class _Handler(BaseHTTPRequestHandler):
            server_version = "TreeLineFolder/1.0"

            # -- helpers -----------------------------------------------------
            def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_text(self, body: str, content_type: str = "text/html; charset=utf-8") -> None:
                data = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_error(self, message: str, status: int = 400) -> None:
                self._send_json({"error": message, "ok": False}, status=status)

            def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
                pass

            # -- GET ----------------------------------------------------------
            def do_GET(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path
                try:
                    if path in ("/", "/step1"):
                        return self._send_text(step1_html())
                    if path == "/step2":
                        return self._send_text(step2_html())
                    if path == "/step3":
                        return self._send_text(step3_html())
                    if path == "/state":
                        return self._send_json(state_ref.state.snapshot())
                    if path == "/points":
                        return self._points()
                    if path == "/plotly.js":
                        try:
                            import plotly

                            js = plotly.offline.get_plotlyjs()
                            return self._send_text(js, "application/javascript; charset=utf-8")
                        except Exception:  # noqa: BLE001
                            return self._send_error("plotly.js 不可用")
                    if path == "/api/editor_data":
                        return self._editor_data()
                    if path == "/api/final":
                        return self._final_data()
                    if path == "/download_excel":
                        return self._download_excel()
                    return self._send_error(f"未知路径 {path}", status=404)
                except FolderPipelineError as exc:
                    state_ref.state.set_error("STEP 1 点云预览失败", str(exc))
                    return self._send_error(str(exc), status=500)
                except Exception as exc:  # noqa: BLE001 - never expose a raw traceback
                    state_ref.state.set_error("服务器处理失败", _brief(exc))
                    return self._send_error(f"服务器处理失败: {_brief(exc)}", status=500)

            # -- POST ---------------------------------------------------------
            def do_POST(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path
                payload = _read_json(self)
                try:
                    if path == "/mode":
                        state_ref.state.update(selection_mode=payload.get("pole", "IDLE"))
                        return self._send_json(state_ref.state.snapshot())
                    if path == "/click":
                        return self._click(payload)
                    if path == "/refine":
                        return self._refine(payload)
                    if path == "/reselect":
                        folder_server.orchestrator.reset_selection()
                        return self._send_json(state_ref.state.snapshot())
                    if path == "/confirm":
                        return self._confirm(payload)
                    if path == "/api/select":
                        return self._send_json(
                            folder_server.orchestrator.editor_select(
                                [int(v) for v in payload.get("tree_ids", [])]
                            )
                        )
                    if path == "/api/delete":
                        return self._send_json(folder_server.orchestrator.editor_delete())
                    if path == "/api/merge":
                        return self._send_json(folder_server.orchestrator.editor_merge())
                    if path == "/api/undo":
                        return self._send_json(folder_server.orchestrator.editor_undo())
                    if path == "/api/confirm_trees":
                        return self._confirm_trees()
                    if path == "/restart":
                        folder_server._restart()
                        return self._send_json({"ok": True})
                except FolderPipelineError as exc:
                    return self._send_error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    return self._send_error(f"服务器错误: {_brief(exc)}", status=500)
                return self._send_error(f"未知路径 {path}", status=404)

            # -- STEP 1 handlers ---------------------------------------------
            def _click(self, payload: dict[str, Any]) -> None:
                pole = payload.get("pole")
                if pole not in ("A", "B"):
                    return self._send_error("请先按“选择杆塔 A”或“选择杆塔 B”")
                try:
                    x = float(payload["x"])
                    y = float(payload["y"])
                except (KeyError, TypeError, ValueError):
                    return self._send_error("坐标无效")
                z = payload.get("z")
                if z is not None:
                    try:
                        z = float(z)
                    except (TypeError, ValueError):
                        z = None
                bounds = folder_server.orchestrator.run.inspection["bounds_min_xyz"]
                bmax = folder_server.orchestrator.run.inspection["bounds_max_xyz"]
                if not (bounds[0] <= x <= bmax[0] and bounds[1] <= y <= bmax[1]):
                    return self._send_error("点击位置在源点云范围之外，请放大后重试")
                state_ref.state.update(
                    clicks={
                        **state_ref.state.clicks,
                        pole: {"x": x, "y": y, "z": z},
                    }
                )
                return self._send_json(state_ref.state.snapshot())

            def _refine(self, payload: dict[str, Any]) -> None:
                pole = payload.get("pole")
                click = state_ref.state.clicks.get(pole)
                if click is None or "x" not in click or "y" not in click:
                    return self._send_error("请先记录该杆塔点击位置")
                z = click.get("z")
                if z is None:
                    z = _nearest_display_z(folder_server, click["x"], click["y"])
                refined = folder_server.orchestrator.refine_pole(
                    pole, (float(click["x"]), float(click["y"]), float(z))
                )
                return self._send_json(refined)

            def _confirm(self, payload: dict[str, Any]) -> None:
                del payload
                click_a = state_ref.state.clicks.get("A")
                click_b = state_ref.state.clicks.get("B")
                if not click_a or not click_b:
                    return self._send_error("需要先选择杆塔 A 和杆塔 B")
                if folder_server.orchestrator.has_selection():
                    return self._send_error("档距已确认，不能重复确认")
                try:
                    z_a = click_a.get("z") or _nearest_display_z(folder_server, click_a["x"], click_a["y"])
                    z_b = click_b.get("z") or _nearest_display_z(folder_server, click_b["x"], click_b["y"])
                    folder_server.orchestrator.apply_selection(
                        (float(click_a["x"]), float(click_a["y"]), float(z_a)),
                        (float(click_b["x"]), float(click_b["y"]), float(z_b)),
                    )
                except (FolderPipelineError, ValueError) as exc:
                    return self._send_error(str(exc))
                state_ref.state.set_step(STEP_SCIENCE_RUNNING, "[2/3] 正在执行科学分析...")
                folder_server._start_science()
                return self._send_json(state_ref.state.snapshot())

            # -- STEP 2 / 3 data ---------------------------------------------
            def _editor_data(self) -> None:
                folder_server.orchestrator.build_editor()
                payload = folder_server._build_editor_display()
                return self._send_json(payload)

            def _confirm_trees(self) -> None:
                state_ref.state.set_step(STEP_SCIENCE_RUNNING, "[3/3] 正在计算最终净空...")

                def _work() -> None:
                    try:
                        folder_server.orchestrator.confirm_trees_and_finalize()
                    except Exception as exc:  # noqa: BLE001
                        state_ref.state.set_error("最终计算失败", _brief(exc))

                threading.Thread(target=_work, daemon=True).start()
                return self._send_json({"ok": True})

            def _final_data(self) -> None:
                return self._send_json(folder_server._build_final_display())

            def _download_excel(self) -> None:
                excel_path = folder_server.workspace.excel_path
                if not excel_path.is_file():
                    return self._send_error("Excel 尚未生成，请先完成最终计算")
                data = excel_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                self.send_header("Content-Disposition", 'attachment; filename="Prediction_Input.xlsx"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            # -- STEP 1 points -----------------------------------------------
            def _points(self) -> None:
                display = folder_server.orchestrator.display_points()
                xyz = np.asarray(display["xyz"], dtype=np.float64)
                rgb = np.asarray(display["rgb"], dtype=np.uint8)
                rows = _even_rows(len(xyz), _MAX_DISPLAY_2D)
                return self._send_json(
                    {
                        "xyz": xyz[rows].tolist(),
                        "rgb": rgb[rows].tolist(),
                        "count": int(len(rows)),
                        "source_bounds": display["source_bounds"],
                    }
                )

        return _Handler

    # ------------------------------------------------------- display builders
    def _build_editor_display(self) -> dict[str, Any]:
        orch = self.orchestrator
        run = orch.run
        origin = np.asarray(run.origin_xyz, dtype=np.float64)
        session = orch.build_editor()
        snap = session.snapshot()
        trees_out = {"xyz": [], "color": [], "tree_id_by_index": []}

        if orch.original_trees is not None:
            original = orch.original_trees
            # original tree id -> display colour (lowest-order rule for merges)
            color_map: dict[int, str] = {}
            for tree in snap["trees"]:
                color_map[tree["tree_id"]] = _tree_color(tree)
            rows = _even_rows(len(original.xyz_local_m), _MAX_TREE_DISPLAY)
            xyz = np.asarray(original.xyz_local_m, dtype=np.float64)[rows] + origin
            tids = np.asarray(original.tree_id, dtype=np.int32)[rows]
            colors: list[str] = []
            for tid in tids:
                effective = _effective_tree_id(session, int(tid))
                colors.append(color_map.get(effective, "#9e9e9e"))
            trees_out = {
                "xyz": xyz.tolist(),
                "color": colors,
                "tree_id_by_index": [int(t) for t in tids],
            }

        return {
            "trees": snap["trees"],
            "trees3d": trees_out,
            "background": self._background_display(),
            "wire": self._wire_member_display(_MAX_WIRE_DISPLAY),
            "wires3d": self._wire_curves_display(),
            "tower": self._tower_display(),
        }

    def _build_final_display(self) -> dict[str, Any]:
        orch = self.orchestrator
        run = orch.run
        origin = np.asarray(run.origin_xyz, dtype=np.float64)
        trees_xyz: list[list[float]] = []
        trees_color: list[str] = []
        if orch.final_reconciled is not None:
            reconciled = orch.final_reconciled
            xyz = np.asarray(reconciled.final_xyz_local_m, dtype=np.float64) + origin
            tids = np.asarray(reconciled.final_tree_id, dtype=np.int32)
            rows = _even_rows(len(xyz), _MAX_TREE_DISPLAY)
            trees_xyz = xyz[rows].tolist()
            for tid in tids[rows]:
                trees_color.append(_stable_color(int(tid)))
        danger_xyz: list[list[float]] = []
        if orch.final_risk is not None and len(orch.final_risk.danger_xyz_local_m):
            danger = np.asarray(orch.final_risk.danger_xyz_local_m, dtype=np.float32) + origin
            rows = _even_rows(len(danger), _MAX_DANGER_DISPLAY)
            danger_xyz = danger[rows].tolist()
        return {
            "trees": orch.final_rows,
            "display": {
                "background": self._background_display(),
                "trees": trees_xyz,
                "tree_color": trees_color,
                "danger": danger_xyz,
                "wire3d": self._wire_curves_display(),
                "towers": self._tower_display(),
            },
            "metadata": orch.run_metadata,
        }

    def _background_display(self) -> dict[str, Any]:
        run = self.orchestrator.run
        display = run.display_sample
        xyz = np.asarray(display["xyz_absolute"], dtype=np.float64)
        rows = _even_rows(len(xyz), _MAX_BACKGROUND_DISPLAY)
        gray = np.full((len(rows), 3), 170, dtype=np.uint8)
        return {"xyz": xyz[rows].tolist(), "rgb": gray.tolist()}

    def _wire_member_display(self, maximum: int) -> dict[str, Any]:
        run = self.orchestrator.run
        try:
            from algorithms.vegetation.coarse_to_fine import load_raw_region

            sidecar = Path(run.wire_region_dir) / "region_sidecar.json"
            region = load_raw_region(sidecar) if sidecar.is_file() else None
            confirmed = np.asarray(run.wire_outcome.confirmed_member_source_index, dtype=np.int64)
            if region is None or len(confirmed) == 0:
                return {"xyz": []}
            idx = np.asarray(region.original_index(), dtype=np.int64)
            xyz = np.asarray(region.xyz(), dtype=np.float64)
            origin = np.asarray(run.origin_xyz, dtype=np.float64)
            keep = np.isin(idx, confirmed)
            if np.count_nonzero(keep) == 0:
                return {"xyz": []}
            rows = _even_rows(int(np.count_nonzero(keep)), maximum)
            return {"xyz": (xyz[keep] + origin)[rows].tolist()}
        except Exception:  # noqa: BLE001 - display only
            return {"xyz": []}

    def _wire_curves_display(self) -> dict[str, Any]:
        run = self.orchestrator.run
        origin = np.asarray(run.origin_xyz, dtype=np.float64)
        curves: list[list[float]] = []
        for strand in run.wire_outcome.strands:
            curve = np.asarray(strand.curve_xyz_local_m, dtype=np.float64) + origin
            curves.extend(curve.tolist())
        return {"xyz": curves}

    def _tower_display(self) -> dict[str, Any]:
        run = self.orchestrator.run
        out = {"xyz": [], "labels": []}
        for label, pole in (("A", run.pole_a), ("B", run.pole_b)):
            if pole is not None:
                out["xyz"].append(list(pole.base_xyz_absolute_m))
                out["labels"].append(label)
        return out

    def _restart(self) -> None:
        self.orchestrator = FolderOrchestrator(
            self.workspace, state=self.state, threshold_m=self.orchestrator.threshold_m
        )
        self.orchestrator.initialize()
        self.state.update(
            selection=None, clicks={"A": {}, "B": {}}, tree_snapshot=None,
            final_summary=None, error=None, stages=[], tree_edit_count=0,
        )
        self.state.set_step(STEP_SELECTING_SPAN, "[1/3] 等待人工选择档距...")


def _tree_color(tree: dict[str, Any]) -> str:
    source_ids = tree.get("source_tree_ids") or [tree["tree_id"]]
    colour = _stable_color(int(min(source_ids)))
    return colour


def _stable_color(tree_id: int) -> str:
    from algorithms.vegetation.tree_visualization import segment_color

    r, g, b = segment_color(tree_id)
    return f"rgb({r},{g},{b})"


def _effective_tree_id(session: Any, original_tree_id: int) -> int:
    """Map an original algorithm tree id to its current effective tree id."""

    snap = session.snapshot()
    for tree in snap["trees"]:
        if original_tree_id in (tree.get("source_tree_ids") or []):
            return int(tree["tree_id"])
    return original_tree_id


def _nearest_display_z(folder_server: FolderServer, x: float, y: float) -> float:
    display = folder_server.orchestrator.display_points()
    xyz = np.asarray(display["xyz"], dtype=np.float64)
    if len(xyz) == 0:
        return 0.0
    dist = (xyz[:, 0] - x) ** 2 + (xyz[:, 1] - y) ** 2
    return float(xyz[int(np.argmin(dist)), 2])


def _brief(exc: Exception) -> str:
    text = str(exc)
    if len(text) > 300:
        return text[:300] + "..."
    return text


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def run_folder_server(
    workspace: FolderWorkspace,
    *,
    port: int = 0,
    open_browser: bool = True,
    threshold_m: float | None = None,
    headless_poles: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
) -> tuple[ThreadingHTTPServer, str]:
    """Start the folder server and return (httpd, url)."""

    # threshold_m=None -> the canonical project risk standard.
    server = FolderServer(workspace, threshold_m=threshold_m, headless_poles=headless_poles)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), server.handler_factory())
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    if open_browser and headless_poles is None:
        import webbrowser

        try:
            webbrowser.open(url)
        except OSError:  # pragma: no cover - best effort
            pass
    return httpd, url


__all__ = [
    "FolderServer",
    "run_folder_server",
    "_stable_color",
    "_even_rows",
]
