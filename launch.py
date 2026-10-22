"""Stable Windows launcher for the integrated TreeLineWarning application."""

from __future__ import annotations

import ctypes
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path


PROGRAM_ROOT = Path(__file__).resolve().parent
APP_ROOT = PROGRAM_ROOT.parent
LOG_ROOT = APP_ROOT / "运行日志"


def _message(text: str, *, error: bool = False) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(
            None,
            text,
            "TreeLineWarning 一体化软件",
            0x10 if error else 0x40,
        )
    except Exception:
        pass


def main() -> int:
    # A named mutex prevents repeated double-clicks from opening multiple servers.
    kernel32 = ctypes.windll.kernel32
    mutex = kernel32.CreateMutexW(None, False, "Local\\TreeLineWarningIntegratedAppV2")
    if kernel32.GetLastError() == 183:
        _message("软件已经打开，请在任务栏中找到 TreeLineWarning 窗口。")
        return 0

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"启动_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_stream = log_path.open("w", encoding="utf-8", buffering=1)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = log_stream
    sys.stderr = log_stream

    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PROJ_NETWORK", "OFF")
    os.chdir(APP_ROOT)
    sys.path.insert(0, str(PROGRAM_ROOT))

    try:
        from desktop.app import main as desktop_main

        exit_code = int(desktop_main())
        if exit_code:
            _message(
                f"软件没有正常启动（错误代码 {exit_code}）。\n\n"
                f"详细记录：{log_path}",
                error=True,
            )
        return exit_code
    except BaseException:
        traceback.print_exc()
        _message(
            "软件启动时发生错误。\n\n"
            f"错误记录已保存到：{log_path}",
            error=True,
        )
        return 1
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        log_stream.close()
        if mutex:
            kernel32.CloseHandle(mutex)


if __name__ == "__main__":
    raise SystemExit(main())
