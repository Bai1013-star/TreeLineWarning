"""Windows desktop launcher; it owns process lifecycle, not science logic."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.request
from pathlib import Path


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
        self.project_root = project_root
        self.repository_root = repository_root
        self.httpd = None
        self.thread: threading.Thread | None = None

    def start(self, timeout_seconds: float = 25.0) -> str:
        # Runtime caches deliberately remain external to the EXE.  A prepared
        # project is normally under ``<workspace>/cache/tower_ordered_project``.
        # Deriving siblings from that selected project keeps packaged builds
        # portable across drives, Chinese paths, and workspace locations.
        cache_root = self.project_root.parent
        tile_cache = cache_root / "tower_raw_tiles"
        dataset = cache_root / "giro3d_full_source" / "dataset"
        if not tile_cache.is_dir() or not dataset.is_dir():
            raise RuntimeError(
                "Required local viewer cache is unavailable. "
                "Select a prepared project workspace."
            )
        try:
            from scripts import serve_p3a_giro3d as server

            server.DATASET_ROOT = dataset
            server._CONFIG_FILE = "poc_config_full.json"
            server._P3aHandler.tower_review_api = server.TowerReviewApi(
                server.TowerReviewConfig(project_root=self.project_root, tile_cache=tile_cache)
            )
            self.httpd = server._ExclusiveBindServer((server.P3A_BIND_HOST, 0), server._P3aHandler)
            self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.thread.start()
            return f"http://127.0.0.1:{self.httpd.server_address[1]}/"
        except (ImportError, OSError, RuntimeError) as exc:
            self.stop()
            raise RuntimeError(f"Viewer backend failed to start: {exc}") from exc

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        self.httpd = None
        self.thread = None


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
    if project is None or not (project / "project.json").is_file():
        print(
            "No prepared TreeLineWarning project was found. Start from a workspace containing "
            "cache\\tower_ordered_project or use --project-root.",
            file=sys.stderr,
        )
        return 2
    backend = BackendProcess(project.resolve(), resource_root())
    try:
        url = backend.start()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    try:
        webview.create_window(
            "TreeLineWarning", url + "desktop.html", width=1600, height=980, min_size=(1100, 700),
        )
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
            results[mode] = {
                "trees": payload["output_tree_count"],
                "uses_temporal_model": bool(
                    payload["result"]["entries"][0].get("uses_temporal_model")
                ),
            }
        print(json.dumps(results, ensure_ascii=False))
        return 0
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Prediction smoke failed: {exc}", file=sys.stderr)
        return 5
    finally:
        backend.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TreeLineWarning desktop viewer")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--prediction-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.prediction_smoke:
        return prediction_smoke(args.project_root)
    return launch(args.project_root)


if __name__ == "__main__":
    raise SystemExit(main())
