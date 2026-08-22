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
from algorithms.folder_pipeline.potree_page import potree_step1_html
from algorithms.folder_pipeline.state import (
    STEP_ERROR,
    STEP_SCIENCE_RUNNING,
    STEP_SELECTING_SPAN,
    STEP_TREE_EDITING,
    FolderWorkflowState,
)
from algorithms.folder_pipeline.viewer_cache import (
    POTREE_RUNTIME_DIR,
    load_viewer_cache,
    viewer_cache_status,
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


_EXTRA_MIME = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".json": "application/json",
    ".wasm": "application/wasm",
    ".bin": "application/octet-stream",
    ".glsl": "text/plain",
}


def _mime_type(path: Path) -> str:
    import mimetypes

    mime = mimetypes.guess_type(str(path))[0]
    if mime:
        return mime
    return _EXTRA_MIME.get(path.suffix.lower(), "application/octet-stream")


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
        self._viewer_cache: Any = None
        self._prepare()

    def viewer_cache(self) -> Any:
        """Lazily load the valid workspace Potree viewer cache (or None)."""

        if self._viewer_cache is None:
            try:
                self._viewer_cache = load_viewer_cache(self.workspace)
            except Exception:  # noqa: BLE001 - a missing cache is not fatal
                self._viewer_cache = None
        return self._viewer_cache

    def _prepare(self) -> None:
        """Resume persisted checkpoints or start fresh."""

        # A completed Step-2 workspace must restart by RESTORING persisted
        # artifacts — never by re-running science or loading the heavy coarse
        # representative evidence (not needed for the editor display).
        completed = (
            self.workspace.original_tree_npz.is_file()
            and self.workspace.original_tree_manifest.is_file()
        )
        if completed:
            try:
                self.orchestrator.hydrate_completed_workspace()
            except Exception as exc:  # noqa: BLE001 - surface a clear error
                self.state.set_error("恢复 Step 2 失败", _brief(exc))
                return
            self.state.set_step(STEP_TREE_EDITING, "等待人工审核单木...")
            return
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
                        return self._step1_page()
                    if path == "/step1_plotly":
                        return self._send_text(step1_html())
                    if path == "/step2":
                        return self._send_text(step2_html())
                    if path == "/step3":
                        return self._send_text(step3_html())
                    if path == "/state":
                        return self._send_json(state_ref.state.snapshot())
                    if path == "/points":
                        return self._points()
                    if path == "/viewer_cache":
                        return self._send_json(viewer_cache_status(folder_server.workspace))
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
                    if path == "/potree" or path.startswith("/potree/"):
                        return self._serve_static_root(
                            path, POTREE_RUNTIME_DIR, "/potree"
                        )
                    if path == "/pointclouds" or path.startswith("/pointclouds/"):
                        return self._serve_pointcloud(path)
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
                    if path == "/api/pick":
                        return self._pick(payload)
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
                    if path == "/api/trim/component-preview":
                        return self._send_json(
                            folder_server.orchestrator.editor_preview_trim_component(
                                int(payload.get("tree_id")), list(payload.get("seed_xyz") or [])
                            )
                        )
                    if path == "/api/trim/lasso-preview":
                        return self._send_json(
                            folder_server.orchestrator.editor_preview_trim_lasso(
                                int(payload.get("tree_id")),
                                list(payload.get("view_projection") or []),
                                list(payload.get("polygon_ndc") or []),
                            )
                        )
                    if path == "/api/trim/commit":
                        return self._send_json(
                            folder_server.orchestrator.editor_commit_trim(
                                str(payload.get("preview_id") or "")
                            )
                        )
                    if path == "/api/trim/cancel":
                        return self._send_json(
                            folder_server.orchestrator.editor_cancel_trim(
                                str(payload.get("preview_id") or "")
                            )
                        )
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

            # -- STEP 1 primary page (Potree LOD, Plotly fallback) -------------
            def _step1_page(self) -> None:
                cache = folder_server.viewer_cache()
                if cache is not None:
                    bounds = folder_server.orchestrator.source_bounds()
                    return self._send_text(
                        potree_step1_html(cache, source_bounds=bounds)
                    )
                # No valid viewer cache -> Plotly legacy/debug fallback (visible).
                return self._send_text(step1_html())

            # -- coordinate bridge: viewer-local -> source-absolute ------------
            def _pick(self, payload: dict[str, Any]) -> None:
                cache = folder_server.viewer_cache()
                if cache is None:
                    return self._send_error("Potree 视图缓存尚未构建，无法转换拾取坐标")
                request_cache_id = payload.get("cache_id")
                if request_cache_id != cache.cache_id:
                    return self._send_error("拾取坐标来自过期的视图缓存，请刷新页面")
                try:
                    x = float(payload["x"])
                    y = float(payload["y"])
                    z = float(payload["z"])
                except (KeyError, TypeError, ValueError):
                    return self._send_error("拾取坐标无效")
                world = (x, y, z)
                offset = cache.offset_xyz
                # Potree 1.8.2 fact (pinned bundle): PointCloudOctree.pick()
                # applies `position.applyMatrix4(pc.matrixWorld)` and the octree
                # constructor sets `this.position.copy(geometry.offset)`, so
                # Potree.Utils.getMousePointCloudIntersection(...).location is
                # ALREADY world-transformed = source-absolute.  Adding the offset
                # here would DOUBLE-offset and push the pick out of the source
                # bounds.  source_absolute = world pick (no offset addition).
                source = (x, y, z)
                bounds = folder_server.orchestrator.source_bounds()
                bmin, bmax = bounds
                if not (
                    bmin[0] - 10 <= source[0] <= bmax[0] + 10
                    and bmin[1] - 10 <= source[1] <= bmax[1] + 10
                ):
                    return self._send_error("拾取位置在源点云范围之外，请放大后重试")
                return self._send_json(
                    {
                        "ok": True,
                        "viewer_world_xyz_m": list(world),
                        "offset_xyz": list(offset),
                        "source_xyz": list(source),
                        "pick_frame": "POTREE_WORLD_SOURCE_ABSOLUTE",
                        "cache_id": cache.cache_id,
                        "source_bounds_min": list(bmin),
                        "source_bounds_max": list(bmax),
                    }
                )

            # -- static Potree runtime assets (with Range support) -------------
            def _serve_static_root(self, path: str, root: Any, prefix: str) -> None:
                relative = path[len(prefix):].lstrip("/")
                if not relative:
                    relative = "index.html"
                target = (Path(root) / relative).resolve()
                try:
                    target.relative_to(Path(root).resolve())
                except ValueError:
                    return self._send_error("路径越界", status=403)
                if not target.is_file():
                    return self._send_error("文件不存在", status=404)
                return folder_server._serve_static_file(self, target)

            def _serve_pointcloud(self, path: str) -> None:
                cache = folder_server.viewer_cache()
                if cache is None:
                    return self._send_error("Potree 视图缓存尚未构建", status=404)
                # /pointclouds/<cache_id>/<relative>
                parts = path.split("/")
                if len(parts) < 3 or parts[1] != "pointclouds":
                    return self._send_error("未知路径", status=404)
                request_id = parts[2]
                if request_id != cache.cache_id:
                    return self._send_error("视图缓存不匹配", status=404)
                relative = "/".join(parts[3:])
                if not relative:
                    relative = "metadata.json"
                target = (cache.potree_dir / relative).resolve()
                try:
                    target.relative_to(cache.potree_dir.resolve())
                except ValueError:
                    return self._send_error("路径越界", status=403)
                if not target.is_file():
                    return self._send_error("文件不存在", status=404)
                return folder_server._serve_static_file(self, target)

        return _Handler

    # ------------------------------------------------------- display builders
    def _build_editor_display(self) -> dict[str, Any]:
        orch = self.orchestrator
        run = orch.run
        origin = np.asarray(run.origin_xyz, dtype=np.float64)
        session = orch.build_editor()
        snap = session.snapshot()
        trees_out = {"xyz": [], "color": [], "tree_id_by_index": [], "labels": []}

        if orch.original_trees is not None:
            final_index, final_tree, final_xyz, _summary = session.build_final()
            del final_index
            rows = _even_rows(len(final_xyz), _MAX_TREE_DISPLAY)
            # The view is sampled only after authoritative edit-log replay, so
            # removed points disappear visually while the edit itself remains
            # full-resolution and original_index based.
            xyz = final_xyz[rows] + origin
            tids = np.asarray(final_tree, dtype=np.int32)[rows]
            colors = [_stable_color(int(tid)) for tid in tids]
            labels = _tree_label_payload(snap["trees"], final_tree, final_xyz, origin)
            trees_out = {
                "xyz": xyz.tolist(),
                "color": colors,
                "tree_id_by_index": tids.tolist(),
                "labels": labels,
            }

        return {
            "trees": snap["trees"],
            "trees3d": trees_out,
            "background": self._background_display(),
            "wire": self._wire_member_display(_MAX_WIRE_DISPLAY),
            "wires3d": self._wire_curves_display(),
            "tower": self._tower_display(),
        }

    def _serve_static_file(self, handler: BaseHTTPRequestHandler, target: Path) -> None:
        """Serve a static file with HTTP Range support (Potree octree node reads)."""

        size = target.stat().st_size
        range_header = handler.headers.get("Range", "")
        content_type = _mime_type(target)
        if range_header.startswith("bytes="):
            spec = range_header[6:].split(",")[0].strip()
            start_text, _, end_text = spec.partition("-")
            try:
                start = int(start_text) if start_text else 0
                end = int(end_text) if end_text else size - 1
            except ValueError:
                start, end = 0, size - 1
            end = min(end, size - 1)
            if start < 0 or start >= size or start > end:
                handler.send_response(416)
                handler.send_header("Content-Range", f"bytes */{size}")
                handler.end_headers()
                return
            length = end - start + 1
            handler.send_response(206)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(length))
            handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            handler.send_header("Accept-Ranges", "bytes")
            handler.send_header("Cache-Control", "no-store")
            handler.end_headers()
            with target.open("rb") as stream:
                stream.seek(start)
                remaining = length
                while remaining > 0:
                    block = stream.read(min(1 << 20, remaining))
                    if not block:
                        break
                    handler.wfile.write(block)
                    handler.wfile.flush()
                    remaining -= len(block)
            return
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(size))
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        with target.open("rb") as stream:
            while True:
                block = stream.read(1 << 20)
                if not block:
                    break
                handler.wfile.write(block)
                handler.wfile.flush()

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
        self.orchestrator.hydrate_wire_runtime()
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
        self.orchestrator.hydrate_wire_runtime()
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


def _tree_label_payload(
    trees: list[dict[str, Any]],
    final_tree: np.ndarray,
    final_xyz: np.ndarray,
    origin: np.ndarray,
) -> list[dict[str, Any]]:
    """Build UI-only tree labels from the current editor projection.

    The label id comes from the same current snapshot that builds the table.
    Anchors use the persisted apex when it is valid, otherwise an upper
    centroid of the current edited members.  Neither path changes science
    membership or coordinates.
    """

    labels: list[dict[str, Any]] = []
    current_tree = np.asarray(final_tree, dtype=np.int32)
    current_xyz = np.asarray(final_xyz, dtype=np.float64)
    scene_origin = np.asarray(origin, dtype=np.float64)
    for tree in trees:
        tree_id = int(tree["tree_id"])
        anchor = np.asarray(tree.get("apex_xy_local_m"), dtype=np.float64)
        if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
            members = current_xyz[current_tree == tree_id]
            if len(members):
                z = members[:, 2]
                top_band = max(float(np.ptp(z)) * 0.15, np.finfo(np.float64).eps * 100)
                anchor = np.mean(members[z >= float(np.max(z)) - top_band], axis=0)
            else:
                anchor = np.zeros(3, dtype=np.float64)
        labels.append(
            {
                "tree_id": tree_id,
                "text": str(tree_id),
                "xyz": (anchor + scene_origin).tolist(),
            }
        )
    return labels


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
