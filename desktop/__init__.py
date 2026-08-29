"""PyWebView desktop shell for the existing local Giro3D viewer."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from desktop.corridor_extensions import install_corridor_extensions
from desktop.corridor_server_extensions import install_corridor_server_extensions
from desktop.corridor_topology_extensions import install_corridor_topology_extensions
from desktop.review_extensions import install_review_extensions
from desktop.shapefile_surface_extensions import install_shapefile_surface_extensions
from desktop.tree_display_numbering_extensions import (
    install_tree_display_numbering_extensions,
)


def _ensure_packaged_writable_cwd() -> None:
    """Move a packaged launch away from a protected inherited working directory.

    Windows can start the EXE with a current directory under a protected app or
    Program Files location.  The desktop home intentionally uses a relative
    ``workspace`` directory, so a protected inherited CWD used to fail before
    logging with WinError 5.  Source/developer runs keep their existing CWD;
    packaged runs only move when an actual write probe fails.
    """

    if getattr(sys, "_MEIPASS", None) is None:
        return
    try:
        current = Path.cwd()
        probe = current / f".treeline-write-probe-{os.getpid()}"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return
    except OSError:
        pass

    candidates: list[Path] = []
    for variable in ("LOCALAPPDATA", "APPDATA"):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value) / "TreeLineWarning")
    try:
        candidates.append(Path.home() / ".treelinewarning")
    except RuntimeError:
        pass
    candidates.append(Path(tempfile.gettempdir()) / "TreeLineWarning")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            os.chdir(candidate)
            return
        except OSError:
            continue


_ensure_packaged_writable_cwd()

install_review_extensions()
install_corridor_extensions()
install_corridor_topology_extensions()
install_shapefile_surface_extensions()
install_tree_display_numbering_extensions()
install_corridor_server_extensions()
