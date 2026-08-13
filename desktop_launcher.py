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
import webbrowser
from pathlib import Path
from typing import Any

import uvicorn
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


def open_browser_when_ready(url: str, server: uvicorn.Server) -> None:
    """Open the desktop UI only after its local health endpoint responds."""
    for _ in range(100):
        if server.should_exit:
            return
        try:
            with urllib.request.urlopen(f"{url}health", timeout=0.5) as response:
                if response.status == 200:
                    webbrowser.open(url, new=2)
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)


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
        if os.getenv("TREELINE_WARNING_DESKTOP_NO_BROWSER") != "1":
            threading.Thread(
                target=open_browser_when_ready,
                args=(url, server),
                daemon=True,
            ).start()
        server.run()
    except Exception as exc:  # noqa: BLE001 - desktop boundary must show friendly errors.
        show_error(f"软件无法启动：\n{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
