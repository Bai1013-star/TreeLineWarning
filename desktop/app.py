"""Windows desktop launcher; it owns process lifecycle, not science logic."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from desktop.packaged_diagnostics import (
    PackagedPrerequisiteError,
    current_survey_smoke,
    dependency_report,
    finite_backend_smoke,
    require_edgechromium_renderer,
    require_webview2_prerequisites,
    write_json_exclusive,
)
from desktop.project_management import (
    create_empty_project,
    describe_project,
    import_gis_file,
    import_pointcloud,
)


def resource_root() -> Path:
    """Resolve source/PyInstaller resources without machine-specific paths."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))


def default_project_root() -> Path | None:
    candidates = (
        Path.cwd() / "cache" / "tower_ordered_project",
        resource_root() / "cache" / "tower_ordered_project",
    )
    return next((path for path in candidates if (path / "project.json").is_file()), None)


class BackendProcess:
    def __init__(
        self,
        project_root: Path,
        repository_root: Path,
        *,
        desktop_home: bool = False,
        ui_test_mode: bool = False,
    ) -> None:
        self.project_root = project_root.resolve()
        self.repository_root = repository_root
        self.desktop_home = desktop_home
        self.ui_test_mode = ui_test_mode
        self.httpd = None
        self.thread: threading.Thread | None = None
        self.local_app = None

    def _viewer_cache_paths(self) -> tuple[Path, Path]:
        # Runtime caches deliberately remain external to the EXE.  A prepared
        # project is normally under ``<workspace>/cache/tower_ordered_project``.
        # Deriving siblings from that selected project keeps packaged builds
        # portable across drives, Chinese paths, and workspace locations.
        candidates = (
            self.project_root.parent,
            Path.cwd() / "cache",
            resource_root() / "cache",
        )
        cache_root = next(
            (item for item in candidates if (item / "tower_raw_tiles").is_dir()), None
        )
        if cache_root is None:
            raise RuntimeError(
                "Required local viewer cache is unavailable. Select a prepared project workspace."
            )
        return cache_root / "tower_raw_tiles", cache_root / "giro3d_full_source" / "dataset"

    def configure_project(self, project_root: Path) -> dict:
        """Switch the local API to a verified project without carrying UI state."""
        self.project_root = project_root.resolve()
        self.desktop_home = False
        if self.local_app is not None:
            return self.local_app.configure_project(self.project_root)
        from scripts import serve_p3a_giro3d as server

        tile_cache, _dataset = self._viewer_cache_paths()
        server._P3aHandler.tower_review_api = server.TowerReviewApi(
            server.TowerReviewConfig(project_root=self.project_root, tile_cache=tile_cache)
        )
        return describe_project(self.project_root)

    def start(self, timeout_seconds: float = 25.0) -> str:
        try:
            if (self.project_root / "stage-a-project.json").is_file() or (
                self.project_root / "desktop-project.json"
            ).is_file():
                from desktop.server import DesktopApplication

                self.local_app = DesktopApplication(
                    self.project_root,
                    self.repository_root,
                    desktop_home=self.desktop_home,
                    ui_test_mode=self.ui_test_mode,
                )
                return self.local_app.start()
            from scripts import serve_p3a_giro3d as server

            tile_cache, dataset = self._viewer_cache_paths()
            server.DATASET_ROOT = dataset
            server._CONFIG_FILE = "poc_config_full.json"
            self.configure_project(self.project_root)
            self.httpd = server._ExclusiveBindServer((server.P3A_BIND_HOST, 0), server._P3aHandler)
            self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.thread.start()
            return f"http://127.0.0.1:{self.httpd.server_address[1]}/"
        except (ImportError, OSError, RuntimeError) as exc:
            self.stop()
            raise RuntimeError(f"Viewer backend failed to start: {exc}") from exc

    def stop(self) -> None:
        if self.local_app is not None:
            self.local_app.stop()
            self.local_app = None
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        self.httpd = None
        self.thread = None


class DesktopApi:
    """Native file dialogs plus narrow project lifecycle operations for the UI."""

    def __init__(self, backend: BackendProcess) -> None:
        self._backend = backend
        # pywebview recursively scans every public non-callable attribute on
        # ``js_api`` objects.  Keeping the native Window public makes the scan
        # descend into .NET objects and abort before any bridge function is
        # installed.  The leading underscore is therefore part of the bridge
        # contract, not merely an implementation style choice.
        self._window = None

    def _folder(self, title: str) -> str | None:
        if self._window is None:
            return None
        import webview

        selected = self._window.create_file_dialog(
            webview.FOLDER_DIALOG, directory="", allow_multiple=False
        )
        return str(selected[0]) if selected else None

    def choose_project_location(self) -> str | None:
        return self._folder("选择项目保存位置")

    def _configure_project(self, project_root: Path) -> dict[str, bool | str]:
        """Switch projects while keeping scientific projections off the native bridge."""
        resolved_root = project_root.resolve()
        self._backend.configure_project(resolved_root)
        return {"ok": True, "project_root": str(resolved_root)}

    def choose_existing_project(self) -> dict | None:
        selected = self._folder("打开 TreeLineWarning 项目")
        if not selected:
            return None
        return self._configure_project(Path(selected))

    def create_project(self, parent: str, project_name: str) -> dict:
        descriptor = create_empty_project(Path(parent), project_name)
        return self._configure_project(Path(descriptor["root"]))

    def import_point_cloud(self) -> dict | None:
        if self._window is None:
            return None
        import webview

        selected = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            directory="",
            allow_multiple=False,
            file_types=("Point cloud (*.las;*.laz)", "All files (*.*)"),
        )
        if not selected:
            return None
        descriptor = import_pointcloud(self._backend.project_root, Path(selected[0]))
        return self._configure_project(Path(descriptor["root"]))

    def import_gis(self) -> dict | None:
        if self._window is None:
            return None
        import webview

        selected = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            directory="",
            allow_multiple=False,
            file_types=(
                "GIS (*.kml;*.kmz;*.csv;*.xlsx;*.geojson;*.json;*.shp;*.zip)",
                "All files (*.*)",
            ),
        )
        if not selected:
            return None
        descriptor = import_gis_file(self._backend.project_root, Path(selected[0]))
        return self._configure_project(Path(descriptor["root"]))

    def _save_generated_file(
        self,
        source: Path,
        *,
        save_filename: str,
        file_types: tuple[str, ...],
    ) -> dict[str, Any]:
        """Use WebView2's native save dialog, then publish with atomic replace."""

        if self._window is None:
            return {"ok": False, "cancelled": True}
        import webview

        selected = self._window.create_file_dialog(
            webview.SAVE_DIALOG,
            directory="",
            allow_multiple=False,
            save_filename=save_filename,
            file_types=file_types,
        )
        if not selected:
            return {"ok": False, "cancelled": True}
        target = Path(selected[0])
        expected_suffix = Path(save_filename).suffix
        if expected_suffix and not target.suffix:
            target = target.with_suffix(expected_suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            temporary = target.with_name(target.name + "." + uuid4().hex + ".tmp")
            try:
                shutil.copyfile(source, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return {"ok": True, "path": str(target.resolve()), "bytes": target.stat().st_size}

    def export_excel(self) -> dict[str, Any]:
        if self._backend.local_app is None:
            raise RuntimeError("当前项目后端尚未就绪")
        source = self._backend.local_app.workflow.export_prediction()
        return self._save_generated_file(
            source,
            save_filename="Prediction_Input.xlsx",
            file_types=("Excel workbook (*.xlsx)",),
        )

    def export_html(self) -> dict[str, Any]:
        if self._backend.local_app is None:
            raise RuntimeError("当前项目后端尚未就绪")
        source = self._backend.local_app.workflow.export_html_report()
        return self._save_generated_file(
            source,
            save_filename="TreeLineWarning_Current_Report.html",
            file_types=("HTML report (*.html)",),
        )


def _write_diagnostic(path: Path | None, payload: dict) -> int:
    if path is None:
        print(json.dumps(payload, ensure_ascii=False, default=str))
        return 0
    try:
        target = write_json_exclusive(path, payload)
    except FileExistsError:
        print(f"Diagnostic output already exists; refusing to overwrite: {path}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Unable to write diagnostic output {path}: {exc}", file=sys.stderr)
        return 2
    print(f"Diagnostic written: {target}")
    return 0


def _diagnostic_failure(
    path: Path | None,
    *,
    mode: str,
    error: Exception | str,
    status: str = "FAIL",
) -> int:
    payload = {
        "schema": "treeline-desktop-diagnostic-error-v1",
        "status": status,
        "mode": mode,
        "error": str(error),
        "error_type": type(error).__name__ if isinstance(error, Exception) else None,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    output_status = _write_diagnostic(path, payload)
    return output_status if output_status != 0 else 5


def _show_gui_error(message: str) -> None:
    """Show a native error when a windowed executable has no console."""

    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, str(message), "TreeLineWarning", 0x10)
    except Exception:
        # The JSON diagnostic and stderr remain the machine-readable fallback.
        return


def launch(
    project_root: Path | None,
    *,
    ui_test_mode: bool = False,
    diagnostic_output: Path | None = None,
) -> int:
    prerequisite_report = None
    try:
        prerequisite_report = require_webview2_prerequisites(resource_root())
    except PackagedPrerequisiteError as exc:
        _diagnostic_failure(
            diagnostic_output,
            mode="gui-launch",
            error=exc,
            status="MISSING_PREREQUISITE",
        )
        _show_gui_error(str(exc))
        print(str(exc), file=sys.stderr)
        return 3
    try:
        import webview
    except ImportError:
        message = (
            "pywebview is unavailable. Install the desktop requirements; the native "
            "WebView2 runtime is also required and is never replaced with MSHTML."
        )
        _diagnostic_failure(diagnostic_output, mode="gui-launch", error=message)
        _show_gui_error(message)
        print(message, file=sys.stderr)
        return 2
    try:
        renderer_preflight = require_edgechromium_renderer(webview, resource_root())
    except PackagedPrerequisiteError as exc:
        _diagnostic_failure(
            diagnostic_output,
            mode="gui-launch",
            error=exc,
            status="MISSING_PREREQUISITE",
        )
        _show_gui_error(str(exc))
        print(str(exc), file=sys.stderr)
        return 3
    # UI acceptance must start from the same empty product home as a fresh
    # packaged install, even when the source checkout contains developer caches.
    project = project_root or (None if ui_test_mode else default_project_root())
    desktop_home = False
    if project is None:
        desktop_home = True
        home_parent = (Path.cwd() / "workspace").resolve()
        home_parent.mkdir(parents=True, exist_ok=True)
        home_root = home_parent / "TreeLineWarning Desktop Home"
        if not (home_root / "stage-a-project.json").is_file():
            descriptor = create_empty_project(home_parent, home_root.name)
            project = Path(descriptor["root"])
        else:
            project = home_root
    if not (
        (project / "project.json").is_file()
        or (project / "stage-a-project.json").is_file()
        or (project / "desktop-project.json").is_file()
    ):
        _diagnostic_failure(
            diagnostic_output,
            mode="gui-launch",
            error="Selected project has no supported local TreeLineWarning manifest.",
        )
        _show_gui_error("Selected project has no supported local TreeLineWarning manifest.")
        print("Selected project has no supported local TreeLineWarning manifest.", file=sys.stderr)
        return 2
    backend = BackendProcess(
        project.resolve(),
        resource_root(),
        desktop_home=desktop_home,
        ui_test_mode=ui_test_mode,
    )
    try:
        url = backend.start()
    except RuntimeError as exc:
        _diagnostic_failure(diagnostic_output, mode="gui-launch", error=exc)
        _show_gui_error(str(exc))
        print(str(exc), file=sys.stderr)
        return 3
    try:
        desktop_api = DesktopApi(backend)
        window = webview.create_window(
            "TreeLineWarning",
            # Each backend owns its landing route: the Stage-A desktop server
            # maps ``/`` to the dashboard, while a legacy project server maps
            # ``/`` to the full-source project viewer.
            url,
            width=1600,
            height=980,
            min_size=(1100, 700),
            js_api=desktop_api,
        )
        desktop_api._window = window
        webview.start(gui="edgechromium", debug=False)
        active_renderer = getattr(webview, "renderer", None)
        if active_renderer != "edgechromium":
            raise PackagedPrerequisiteError(
                "pywebview did not use the EdgeChromium renderer; refusing an MSHTML fallback."
            )
        diagnostic_status = _write_diagnostic(
            diagnostic_output,
            {
                "schema": "treeline-desktop-gui-launch-v1",
                "status": "PASS",
                "backend_url": url,
                "project_root": str(backend.project_root),
                "resource_root": str(resource_root()),
                "webview2_prerequisites": prerequisite_report,
                "renderer_preflight": renderer_preflight,
                "renderer": active_renderer,
                "lifecycle": "WINDOW_OPENED_AND_CLOSED",
                "visual_acceptance": "UNVERIFIED",
                "visual_acceptance_note": (
                    "The launcher window completed its lifecycle; visual/browser acceptance "
                    "requires a separate inspection."
                ),
            },
        )
        return diagnostic_status
    except PackagedPrerequisiteError as exc:
        _diagnostic_failure(
            diagnostic_output,
            mode="gui-launch",
            error=exc,
            status="MISSING_PREREQUISITE",
        )
        _show_gui_error(str(exc))
        print(str(exc), file=sys.stderr)
        return 3
    except Exception as exc:  # pywebview backend exceptions are implementation-specific.
        _diagnostic_failure(diagnostic_output, mode="gui-launch", error=exc)
        _show_gui_error(str(exc))
        print(f"Desktop window failed to open: {exc}", file=sys.stderr)
        return 4
    finally:
        backend.stop()


def prediction_smoke(project_root: Path | None) -> int:
    """Run packaged Formula and AI prediction without opening a GUI window."""
    project = project_root or default_project_root()
    if project is None or not (project / "project.json").is_file():
        print("Prediction smoke requires a prepared project.", file=sys.stderr)
        return 2
    backend = BackendProcess(project.resolve(), resource_root())
    try:
        url = backend.start()
        results = {}
        for mode in ("formula", "ai"):
            request = urllib.request.Request(
                url + "api/spans/S005/prediction",
                data=json.dumps({"prediction_mode": mode}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            result = payload["result"]
            if result["status"] != "COMPLETE" or payload["output_tree_count"] != 22:
                raise RuntimeError(f"{mode} prediction smoke failed: {result.get('diagnostic')}")
            g002_t007 = next(
                item for item in result["entries"] if item.get("persistent_tree_id") == "G002-T007"
            )
            day_14 = next(item for item in g002_t007["horizons"] if item["horizon"] == "14天")
            if day_14.get("future_height_m") is None or day_14.get("future_crown_width_m") is None:
                raise RuntimeError(f"{mode} geometry mapping smoke failed")
            results[mode] = {
                "trees": payload["output_tree_count"],
                "uses_temporal_model": bool(
                    payload["result"]["entries"][0].get("uses_temporal_model")
                ),
                "g002_t007_14d_height": day_14["future_height_m"],
                "g002_t007_14d_crown_width": day_14["future_crown_width_m"],
            }
        selected_id = payload["result"]["entries"][0]["persistent_tree_id"]
        work_order_request = urllib.request.Request(
            url + "api/spans/S005/work-order",
            data=json.dumps({"persistent_tree_id": selected_id, "prediction_mode": "ai"}).encode(
                "utf-8"
            ),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(work_order_request, timeout=60) as response:
            work_order = json.loads(response.read().decode("utf-8"))
        if "打印 / 保存为 PDF" not in work_order.get("html", ""):
            raise RuntimeError("work-order smoke failed")
        results["work_order"] = {"persistent_tree_id": selected_id, "printable": True}
        print(json.dumps(results, ensure_ascii=False))
        return 0
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Prediction smoke failed: {exc}", file=sys.stderr)
        return 5
    finally:
        backend.stop()


def dependency_health_smoke(diagnostic_output: Path | None = None) -> int:
    """Print or persist the native/frozen dependency report for release acceptance."""

    try:
        report = dependency_report(resource_root())
    except Exception as exc:
        return _diagnostic_failure(
            diagnostic_output,
            mode="dependency-health",
            error=exc,
        )
    output_status = _write_diagnostic(diagnostic_output, report)
    if output_status != 0:
        return output_status
    return 0 if report.get("overall_pass") else 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TreeLineWarning desktop viewer")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--prediction-smoke", action="store_true")
    parser.add_argument("--dependency-health", action="store_true")
    parser.add_argument(
        "--diagnostic-output",
        "--diagnostic-path",
        dest="diagnostic_output",
        type=Path,
        default=None,
        help="exclusive JSON path for windowed/package diagnostics",
    )
    parser.add_argument("--survey-smoke", action="store_true")
    parser.add_argument("--backend-smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stop-signal", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--smoke-timeout-seconds",
        type=float,
        default=120.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--ui-test-mode", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.dependency_health:
        return dependency_health_smoke(args.diagnostic_output)
    if args.survey_smoke:
        if args.project_root is None:
            return _diagnostic_failure(
                args.diagnostic_output,
                mode="survey-smoke",
                error="--survey-smoke requires --project-root",
            )
        try:
            report = current_survey_smoke(
                args.project_root,
                resource_root(),
                timeout_seconds=args.smoke_timeout_seconds,
            )
        except Exception as exc:
            return _diagnostic_failure(args.diagnostic_output, mode="survey-smoke", error=exc)
        output_status = _write_diagnostic(args.diagnostic_output, report)
        if output_status != 0:
            return output_status
        return 0 if report.get("status") == "PASS" else 5
    if args.backend_smoke:
        if args.project_root is None or args.diagnostic_output is None:
            print(
                "--backend-smoke requires --project-root and --diagnostic-output",
                file=sys.stderr,
            )
            return 2
        try:
            code, _report = finite_backend_smoke(
                args.project_root,
                resource_root(),
                args.diagnostic_output,
                stop_signal=args.stop_signal,
                timeout_seconds=args.smoke_timeout_seconds,
            )
            return code
        except Exception as exc:
            return _diagnostic_failure(args.diagnostic_output, mode="backend-smoke", error=exc)
    if args.prediction_smoke:
        return prediction_smoke(args.project_root)
    return launch(
        args.project_root,
        ui_test_mode=args.ui_test_mode,
        diagnostic_output=args.diagnostic_output,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
