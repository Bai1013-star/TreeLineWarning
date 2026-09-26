"""Windows desktop launcher for the TreeLineWarning local application."""

from __future__ import annotations

import ctypes
import os
import shutil
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from base64 import b64decode
from binascii import Error as Base64Error
from pathlib import Path
from typing import Any

import uvicorn
import webview
from api.prediction_api import create_app

# PyInstaller windowed applications do not have console streams. Uvicorn's
# logging setup still expects writable streams, so route them to the null device.
_stdout_fallback = None
_stderr_fallback = None
if sys.stdout is None:
    _stdout_fallback = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    sys.stdout = _stdout_fallback
if sys.stderr is None:
    _stderr_fallback = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    sys.stderr = _stderr_fallback

APP_NAME = "TreeLineWarning"
PREFERRED_PORT = 8010


class DesktopCsvTemplateBridge:
    """Expose a native Save dialog for template downloads in the desktop app."""

    def __init__(self, window: Any | None = None) -> None:
        self.window = window

    def save_csv_template(self, content_base64: str) -> dict[str, Any]:
        """Save a browser-generated CSV through the Windows Save dialog."""
        if self.window is None:
            raise RuntimeError("保存窗口尚未准备完成，请稍后重试。")
        try:
            content = b64decode(content_base64, validate=True)
        except (Base64Error, TypeError, ValueError) as exc:
            raise ValueError("CSV 示例模板内容无效。") from exc

        selected = self.window.create_file_dialog(
            webview.FileDialog.SAVE,
            save_filename="treeline_history_training_template.csv",
            file_types=("CSV 文件 (*.csv)",),
        )
        if not selected:
            return {"saved": False, "cancelled": True}
        destination = Path(selected[0] if isinstance(selected, (list, tuple)) else selected)
        if destination.suffix.lower() != ".csv":
            destination = destination.with_suffix(".csv")
        try:
            destination.write_bytes(content)
        except OSError as exc:
            raise RuntimeError(f"无法保存 CSV 示例模板：{exc}") from exc
        return {"saved": True, "path": str(destination)}


def bundled_path(*parts: str) -> Path:
    """Return a resource path in source checkout or a PyInstaller bundle."""
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return bundle_root.joinpath(*parts)


def user_data_dir() -> Path:
    """Return the writable per-user data directory used by the desktop build."""
    local_app_data = os.getenv("LOCALAPPDATA")
    root = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return root / APP_NAME


def available_port(preferred: int = PREFERRED_PORT) -> int:
    """Find a local port without stopping or replacing another process."""
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                candidate.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("本机 8010–8029 端口均被占用，无法启动软件。")


def prepare_runtime() -> dict[str, Path]:
    """Create writable runtime paths and seed the bundled default models."""
    data_dir = user_data_dir()
    model_dir = data_dir / "models"
    training_dir = model_dir / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    resources = {
        "temporal_model": bundled_path("models", "temporal_attention.npz"),
        "species_model": bundled_path("models", "species_growth.json"),
        "web_index": bundled_path("web", "templates", "index.html"),
    }
    for label, source in resources.items():
        if not source.is_file():
            raise RuntimeError(f"软件资源缺失：{label}")

    temporal_model = model_dir / "temporal_attention.npz"
    species_model = model_dir / "species_growth.json"
    if not temporal_model.exists():
        shutil.copy2(resources["temporal_model"], temporal_model)
    if not species_model.exists():
        shutil.copy2(resources["species_model"], species_model)

    return {
        "data_dir": data_dir,
        "database": data_dir / "prediction_history.db",
        "temporal_model": temporal_model,
        "species_model": species_model,
        "training_dir": training_dir,
        "web_index": resources["web_index"],
    }


def build_desktop_app(runtime: dict[str, Path], server_holder: dict[str, Any]):
    """Build the local app and add desktop-only status and shutdown controls."""
    application = create_app(
        database_path=runtime["database"],
        temporal_model_path=runtime["temporal_model"],
        species_growth_model_path=runtime["species_model"],
        training_output_dir=runtime["training_dir"],
        web_index_path=runtime["web_index"],
        access_password="",
    )

    @application.get("/desktop/status")
    def desktop_status() -> dict[str, Any]:
        return {"desktop": True, "data_directory": str(runtime["data_dir"])}

    @application.post("/desktop/shutdown")
    def desktop_shutdown() -> dict[str, str]:
        server = server_holder.get("server")

        def request_shutdown() -> None:
            time.sleep(0.35)
            if server is not None:
                server.should_exit = True

        threading.Thread(target=request_shutdown, daemon=True).start()
        return {"message": "TreeLineWarning 正在安全退出"}

    return application


def wait_for_server(url: str, server: uvicorn.Server) -> bool:
    """Wait until the local API is ready before creating the native window."""
    for _ in range(100):
        if server.should_exit:
            return False
        try:
            with urllib.request.urlopen(f"{url}health", timeout=0.5) as response:
                if response.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    return False


def open_native_window(url: str, server: uvicorn.Server) -> None:
    """Show the local interface inside a native Edge WebView window."""
    if not wait_for_server(url, server):
        raise RuntimeError("本地服务未能在 10 秒内启动。")

    csv_template_bridge = DesktopCsvTemplateBridge()
    window = webview.create_window(
        APP_NAME,
        url,
        width=1440,
        height=920,
        min_size=(1120, 720),
        confirm_close=True,
        js_api=csv_template_bridge,
    )
    csv_template_bridge.window = window

    def stop_server() -> None:
        server.should_exit = True

    window.events.closed += stop_server
    # On supported Windows systems pywebview uses the installed Edge WebView2
    # runtime. No external browser process is launched.
    webview.start(gui="edgechromium", debug=False)


def show_error(message: str) -> None:
    """Show a native Windows error without depending on Tcl/Tk."""
    if os.name == "nt":
        ctypes.windll.user32.MessageBoxW(0, message, APP_NAME, 0x10)
    else:
        print(message, file=sys.stderr)


def main() -> int:
    try:
        runtime = prepare_runtime()
        port = available_port()
        url = f"http://127.0.0.1:{port}/"
        server_holder: dict[str, Any] = {}
        application = build_desktop_app(runtime, server_holder)
        server = uvicorn.Server(
            uvicorn.Config(
                application,
                host="127.0.0.1",
                port=port,
                log_level="warning",
                access_log=False,
            )
        )
        server_holder["server"] = server
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        open_native_window(url, server)
        server_thread.join(timeout=5)
    except Exception as exc:  # noqa: BLE001 - desktop boundary must show friendly errors.
        show_error(f"软件无法启动：\n{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
