"""Windows desktop launcher; it owns process lifecycle, not science logic."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.request
from pathlib import Path

from desktop.pdal_runtime import dependency_health
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
    def __init__(self, project_root: Path, repository_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.repository_root = repository_root
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

                self.local_app = DesktopApplication(self.project_root, self.repository_root)
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
        self.backend = backend
        self.window = None

    def _folder(self, title: str) -> str | None:
        if self.window is None:
            return None
        import webview

        selected = self.window.create_file_dialog(
            webview.FOLDER_DIALOG, directory="", allow_multiple=False
        )
        return str(selected[0]) if selected else None

    def choose_project_location(self) -> str | None:
        return self._folder("选择项目保存位置")

    def choose_existing_project(self) -> dict | None:
        selected = self._folder("打开 TreeLineWarning 项目")
        if not selected:
            return None
        return self.backend.configure_project(Path(selected))

    def create_project(self, parent: str, project_name: str) -> dict:
        descriptor = create_empty_project(Path(parent), project_name)
        return self.backend.configure_project(Path(descriptor["root"]))

    def import_point_cloud(self) -> dict | None:
        if self.window is None:
            return None
        import webview

        selected = self.window.create_file_dialog(
            webview.OPEN_DIALOG,
            directory="",
            allow_multiple=False,
            file_types=("Point cloud (*.las;*.laz)", "All files (*.*)"),
        )
        if not selected:
            return None
        descriptor = import_pointcloud(self.backend.project_root, Path(selected[0]))
        return self.backend.configure_project(Path(descriptor["root"]))

    def import_gis(self) -> dict | None:
        if self.window is None:
            return None
        import webview

        selected = self.window.create_file_dialog(
            webview.OPEN_DIALOG,
            directory="",
            allow_multiple=False,
            file_types=("GIS (*.kml;*.csv)", "All files (*.*)"),
        )
        if not selected:
            return None
        descriptor = import_gis_file(self.backend.project_root, Path(selected[0]))
        return self.backend.configure_project(Path(descriptor["root"]))


def launch(project_root: Path | None) -> int:
    try:
        import webview
    except ImportError:
        print(
            "WebView2/pywebview is unavailable. Install the desktop requirements, then retry.",
            file=sys.stderr,
        )
        return 2
    project = project_root or default_project_root()
    if project is None:
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
        print("Selected project has no supported local TreeLineWarning manifest.", file=sys.stderr)
        return 2
    backend = BackendProcess(project.resolve(), resource_root())
    try:
        url = backend.start()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    try:
        desktop_api = DesktopApi(backend)
        window = webview.create_window(
            "TreeLineWarning",
            url + "desktop.html",
            width=1600,
            height=980,
            min_size=(1100, 700),
            js_api=desktop_api,
        )
        desktop_api.window = window
        webview.start(gui="edgechromium", debug=False)
        return 0
    except Exception as exc:  # pywebview backend exceptions are implementation-specific.
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


def dependency_health_smoke() -> int:
    """Print the local PDAL health report for release acceptance tooling."""

    report = dependency_health()
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report.get("found") and report.get("minimal_operation_pass") else 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TreeLineWarning desktop viewer")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--prediction-smoke", action="store_true")
    parser.add_argument("--dependency-health", action="store_true")
    args = parser.parse_args(argv)
    if args.dependency_health:
        return dependency_health_smoke()
    if args.prediction_smoke:
        return prediction_smoke(args.project_root)
    return launch(args.project_root)


if __name__ == "__main__":
    raise SystemExit(main())
