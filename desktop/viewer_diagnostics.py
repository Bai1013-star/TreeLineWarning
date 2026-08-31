"""Read-only renderer environment and bounded local viewer failure evidence."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

VIEWER_ASSETS = {
    "/viewer-runtime.js": "viewer_runtime.js",
    "/giro3d.bundle.js": "vendor/giro3d.bundle.js",
    "/tree-review-3d.js": "tree_review_3d.js",
    "/tree-review-app.js": "tree_review_app.js",
}
MAX_DIAGNOSTIC_BYTES = 65_536


def runtime_report(resource_root: Path) -> dict[str, Any]:
    # No PDAL import, project hydration, science computation, or installer execution.
    webview: dict[str, Any] = {"status": "UNKNOWN", "version": None}
    if sys.platform == "win32":
        try:
            from desktop.packaged_diagnostics import webview_loader_info

            webview_spec = importlib.util.find_spec("webview")
            loader = webview_loader_info(
                resource_root, webview_spec.origin if webview_spec else None
            )
            webview.update(
                status=loader.get("status", "UNKNOWN"),
                version=loader.get("browser_version"),
                minimum_version=loader.get("browser_version_minimum"),
                loader_version=loader.get("loader_file_version"),
                error=loader.get("error"),
            )
        except Exception as exc:
            webview["error"] = f"{type(exc).__name__}: {exc}"
    else:
        webview["status"] = "NOT_APPLICABLE"
    raw_flags = os.environ.get("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "")
    selected_flags = re.findall(
        r"--(?:disable-gpu|disable-webgl2?|disable-software-rasterizer|"
        r"ignore-gpu-blocklist|enable-unsafe-swiftshader|use-angle(?:=[\w-]+)?)(?=\s|$)",
        raw_flags,
    )
    root = resource_root / "viewer_poc" / "giro3d"
    assets = {}
    for url, relative in VIEWER_ASSETS.items():
        path = root / relative
        try:
            assets[url] = {"exists": path.is_file(), "bytes": path.stat().st_size}
        except OSError:
            assets[url] = {"exists": False, "bytes": None}
    return {
        "schema": "treeline-viewer-native-runtime-v1",
        "webview2": webview,
        "browser_arguments": {
            "selected_environment_flags": selected_flags,
            "additional_arguments_present": bool(raw_flags),
            "application_gpu_override": False,
            "pywebview_default": "--disable-features=ElasticOverscroll",
            "renderer_requested": "edgechromium",
            "debug": False,
            "note": (
                "Environment and launcher configuration; not all effective "
                "registry/policy flags are observable."
            ),
        },
        "assets": assets,
        "three_expected": "180",
    }


def record_diagnostic(workflow: Any, payload: dict[str, Any]) -> dict[str, bool]:
    """A diagnostic failure must never alter project/editor state or break the API."""
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_DIAGNOSTIC_BYTES:
        raise ValueError("VIEWER_DIAGNOSTIC_TOO_LARGE")
    logged = saved = False
    try:
        workflow._log("viewer_initialization", diagnostic=payload)
        logged = True
    except Exception:
        pass
    temporary = None
    try:
        directory = workflow.project_root / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "viewer_init.latest.json"
        temporary = directory / f"viewer_init.{uuid4().hex}.tmp"
        with temporary.open("xb") as stream:
            stream.write(encoded)
        temporary.replace(target)
        saved = True
    except OSError:
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return {"ok": True, "logged": logged, "latest_saved": saved}
