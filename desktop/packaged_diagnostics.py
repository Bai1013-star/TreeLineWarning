"""Bounded diagnostics for the frozen desktop launcher.

This module stays outside the science workflow.  It reports the runtime that a
PyInstaller process actually sees, checks the native desktop prerequisites
before pywebview can choose a fallback GUI, and provides read-only probes for
the modern Survey backend.  Diagnostic files are exclusive-create artifacts so
an old acceptance result can never be silently replaced.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WEBVIEW2_CLIENT_GUID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
DOTNET_MINIMUM_RELEASE = 394_802
WEBVIEW2_MINIMUM_VERSION = (86, 0, 622, 0)
WEBVIEW2_MINIMUM_VERSION_TEXT = ".".join(str(part) for part in WEBVIEW2_MINIMUM_VERSION)
WEBVIEW2_LOADER_RELATIVE = Path(
    "webview"
) / "lib" / "runtimes" / "win-x64" / "native" / "WebView2Loader.dll"
WEBVIEW2_INSTALLER_RELATIVE = Path(
    "webview2_offline_prerequisite"
) / "offline_verified" / "MicrosoftEdgeWebView2RuntimeInstallerX64.exe"
WEBVIEW2_EVIDENCE_INSTALLER_RELATIVE = Path(
    "acceptance_evidence"
) / "final_delivery_20260830" / WEBVIEW2_INSTALLER_RELATIVE
WEBVIEW2_INSTALLER_BYTES = 258_438_352
WEBVIEW2_INSTALLER_SHA256 = (
    "987a9d8b3107e84f9b53b4a077d28ae4814fc3d964d5a55c559e7334bbf24d61"
)
MAX_SMOKE_TIMEOUT_SECONDS = 600.0


class PackagedPrerequisiteError(RuntimeError):
    """Raised when the native GUI prerequisites are not safe to launch."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json_exclusive(path: Path | str, payload: Any) -> Path:
    """Write one UTF-8 JSON diagnostic and refuse to replace an existing file."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        stream.write("\n")
    return target


def _module_origin(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError) as exc:
        return {
            "found": False,
            "origin": None,
            "loader": None,
            "search_locations": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    if spec is None:
        return {
            "found": False,
            "origin": None,
            "loader": None,
            "search_locations": [],
            "error": "module spec not found",
        }
    locations = getattr(spec, "submodule_search_locations", None)
    return {
        "found": True,
        "origin": str(spec.origin) if spec.origin is not None else None,
        "loader": type(spec.loader).__name__ if spec.loader is not None else None,
        "search_locations": [str(value) for value in (locations or ())],
        "error": None,
    }


def module_origins() -> dict[str, dict[str, Any]]:
    """Return import origins for the frozen launcher and its native boundary."""

    names = (
        "desktop.app",
        "desktop.packaged_diagnostics",
        "desktop.server",
        "desktop.workflow",
        "desktop.pdal_runtime",
        "laspy",
        "lazrs",
        "numpy",
        "PIL",
        "pyproj",
        "pdal",
        "webview",
        "cupy",
        "cuda.pathfinder",
    )
    return {name: _module_origin(name) for name in names}


def runtime_origins(resource_root: Path | str) -> dict[str, Any]:
    """Describe process/resource origins without assuming a developer checkout."""

    root = Path(resource_root).expanduser().resolve()
    return {
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": str(Path(sys.executable).resolve()),
        "resource_root": str(root),
        "resource_root_exists": root.is_dir(),
        "cwd": str(Path.cwd().resolve()),
        "pid": os.getpid(),
        "python_version": platform_python_version(),
        "python_implementation": getattr(sys, "implementation", None).name,
        "platform": sys.platform,
    }


def platform_python_version() -> str:
    version = sys.version_info
    return f"{version.major}.{version.minor}.{version.micro}"


def _registry_rows(subkey: str, value_names: tuple[str, ...]) -> list[dict[str, Any]]:
    if sys.platform != "win32":
        return []
    try:
        import winreg
    except ImportError:
        return []

    hives = (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER)
    views = tuple(
        dict.fromkeys(
            [
                0,
                getattr(winreg, "KEY_WOW64_64KEY", 0),
                getattr(winreg, "KEY_WOW64_32KEY", 0),
            ]
        )
    )
    rows: list[dict[str, Any]] = []
    for hive in hives:
        for view in views:
            try:
                with winreg.OpenKey(hive, subkey, 0, winreg.KEY_READ | view) as key:
                    values: dict[str, Any] = {}
                    for name in value_names:
                        try:
                            values[name] = winreg.QueryValueEx(key, name)[0]
                        except OSError:
                            continue
                    if values:
                        rows.append(
                            {
                                "hive": str(hive),
                                "view": int(view),
                                "values": values,
                            }
                        )
            except OSError:
                continue
    return rows


def _pywebview_registry_rows(
    subkey: str,
    value_names: tuple[str, ...],
    *,
    include_hkcu: bool = True,
) -> list[dict[str, Any]]:
    """Read the exact default-view registry locations used by pywebview 5.4."""

    if sys.platform != "win32":
        return []
    try:
        import winreg
    except ImportError:
        return []

    hives = (
        (("HKEY_CURRENT_USER", winreg.HKEY_CURRENT_USER),) if include_hkcu else ()
    ) + (("HKEY_LOCAL_MACHINE", winreg.HKEY_LOCAL_MACHINE),)
    is_edge_key = subkey.startswith("SOFTWARE\\Microsoft\\EdgeUpdate\\")
    is_32_bit_python = platform.machine().lower() == "x86"
    rows: list[dict[str, Any]] = []
    for hive_name, hive in hives:
        registry_path = subkey
        if is_edge_key and hive == winreg.HKEY_LOCAL_MACHINE and not is_32_bit_python:
            suffix = subkey.removeprefix("SOFTWARE\\")
            registry_path = rf"SOFTWARE\WOW6432Node\{suffix}"
        try:
            with winreg.OpenKey(hive, registry_path) as key:
                values: dict[str, Any] = {}
                for name in value_names:
                    try:
                        values[name] = winreg.QueryValueEx(key, name)[0]
                    except OSError:
                        continue
                if values:
                    rows.append(
                        {
                            "hive": hive_name,
                            "view": "DEFAULT",
                            "registry_path": registry_path,
                            "values": values,
                            "pywebview_visible": True,
                        }
                    )
        except OSError:
            continue
    return rows


def _webview2_runtime() -> dict[str, Any]:
    rows = _registry_rows(
        f"SOFTWARE\\Microsoft\\EdgeUpdate\\Clients\\{WEBVIEW2_CLIENT_GUID}",
        ("pv", "version", "location", "name"),
    )
    visible_rows = _pywebview_registry_rows(
        f"SOFTWARE\\Microsoft\\EdgeUpdate\\Clients\\{WEBVIEW2_CLIENT_GUID}",
        ("pv", "version", "location", "name"),
    )
    selected = None
    rejected_versions: list[dict[str, Any]] = []
    for row in visible_rows:
        values = row["values"]
        raw_version = values.get("pv")
        version_field = "pv"
        if raw_version is None:
            rejected_versions.append(
                {
                    "value": values.get("version"),
                    "field": version_field,
                    "reason": "WebView2 registry row has no pv value",
                }
            )
            continue
        parsed = _parse_version(raw_version)
        if parsed is None or parsed < WEBVIEW2_MINIMUM_VERSION:
            rejected_versions.append(
                {
                    "value": raw_version,
                    "field": version_field,
                    "reason": (
                        "missing or malformed version"
                        if parsed is None
                        else f"below {WEBVIEW2_MINIMUM_VERSION_TEXT}"
                    ),
                }
            )
            continue
        selected = row
        break
    values = selected["values"] if selected else {}
    version = values.get("pv")
    return {
        "required": True,
        "found": selected is not None,
        "status": "PASS" if selected is not None else "MISSING",
        "version": str(version) if version is not None else None,
        "version_valid": selected is not None,
        "minimum_version": WEBVIEW2_MINIMUM_VERSION_TEXT,
        "install_root": values.get("location"),
        "registry_rows": rows,
        "pywebview_registry_rows": visible_rows,
        "rejected_versions": rejected_versions,
        "client_guid": WEBVIEW2_CLIENT_GUID,
    }


def _parse_version(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, str):
        return None
    parts = value.strip().split(".")
    if len(parts) != 4 or any(not part.isdigit() for part in parts):
        return None
    parsed = tuple(int(part) for part in parts)
    return parsed if all(part >= 0 for part in parsed) else None


def _parse_version_prefix(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, str):
        return None
    match = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def _dotnet_framework() -> dict[str, Any]:
    rows = _registry_rows(
        r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full",
        ("Release", "Version"),
    )
    visible_rows = _pywebview_registry_rows(
        r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full",
        ("Release", "Version"),
        include_hkcu=False,
    )
    release = None
    version = None
    for row in visible_rows:
        values = row["values"]
        candidate = values.get("Release")
        try:
            candidate = int(candidate) if candidate is not None else None
        except (TypeError, ValueError):
            candidate = None
        if candidate is not None and (release is None or candidate > release):
            release = candidate
            version = values.get("Version")
    found = release is not None and release >= DOTNET_MINIMUM_RELEASE
    return {
        "required": True,
        "found": found,
        "status": "PASS" if found else "MISSING",
        "minimum_release": DOTNET_MINIMUM_RELEASE,
        "release": release,
        "version": version,
        "registry_rows": rows,
        "pywebview_registry_rows": visible_rows,
    }


def _installer_candidates(resource_root: Path) -> tuple[Path, ...]:
    roots = (
        (resource_root.parent, resource_root)
        if resource_root.name.lower() == "_internal"
        else (resource_root,)
    )
    relative_paths = (WEBVIEW2_INSTALLER_RELATIVE, WEBVIEW2_EVIDENCE_INSTALLER_RELATIVE)
    return tuple(
        dict.fromkeys(
            root / relative
            for root in roots
            for relative in relative_paths
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _windows_file_version(path: Path) -> str | None:
    """Read a PE file version without importing pywin32 or spawning a shell."""

    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        version_dll = ctypes.WinDLL("version", use_last_error=True)
        size = version_dll.GetFileVersionInfoSizeW(str(path), None)
        if not size:
            return None
        buffer = ctypes.create_string_buffer(size)
        if not version_dll.GetFileVersionInfoW(str(path), 0, size, buffer):
            return None
        value_pointer = ctypes.c_void_p()
        value_length = wintypes.UINT()
        if not version_dll.VerQueryValueW(
            buffer,
            "\\",
            ctypes.byref(value_pointer),
            ctypes.byref(value_length),
        ):
            return None

        class FixedFileInfo(ctypes.Structure):
            _fields_ = [
                ("dwSignature", wintypes.DWORD),
                ("dwStrucVersion", wintypes.DWORD),
                ("dwFileVersionMS", wintypes.DWORD),
                ("dwFileVersionLS", wintypes.DWORD),
            ]

        info = ctypes.cast(value_pointer, ctypes.POINTER(FixedFileInfo)).contents
        if info.dwSignature != 0xFEEF04BD:
            return None
        return ".".join(
            str(value)
            for value in (
                info.dwFileVersionMS >> 16,
                info.dwFileVersionMS & 0xFFFF,
                info.dwFileVersionLS >> 16,
                info.dwFileVersionLS & 0xFFFF,
            )
        )
    except Exception:  # pragma: no cover - Windows API failure is host-specific
        return None


def _available_browser_version(path: Path) -> str | None:
    """Ask WebView2Loader for the installed Evergreen browser version."""

    if sys.platform != "win32":
        return None
    try:
        import ctypes

        loader = ctypes.WinDLL(str(path), use_last_error=True)
        get_version = loader.GetAvailableCoreWebView2BrowserVersionString
        version_pointer = ctypes.c_wchar_p()
        try:
            result = get_version(None, ctypes.byref(version_pointer))
            if result != 0 or not version_pointer.value:
                return None
            return str(version_pointer.value)
        finally:
            if version_pointer.value:
                ole32 = ctypes.WinDLL("ole32", use_last_error=True)
                ole32.CoTaskMemFree(ctypes.cast(version_pointer, ctypes.c_void_p))
    except Exception:  # pragma: no cover - Windows loader/API is host-specific
        return None


def _webview_loader_candidates(
    resource_root: Path,
    webview_module: Any,
) -> tuple[Path, ...]:
    roots = [resource_root]
    if resource_root.name.lower() == "_internal":
        roots.insert(0, resource_root.parent)
    module_file = (
        webview_module
        if isinstance(webview_module, (str, Path))
        else getattr(webview_module, "__file__", None)
    )
    if module_file:
        module_root = Path(module_file).expanduser().resolve().parent.parent
        roots.append(module_root)
    return tuple(
        dict.fromkeys(root / WEBVIEW2_LOADER_RELATIVE for root in roots)
    )


def webview_loader_info(
    resource_root: Path | str,
    webview_module: Any,
) -> dict[str, Any]:
    """Verify the bundled x64 WebView2Loader and report its PE version."""

    root = Path(resource_root).expanduser().resolve()
    candidates = _webview_loader_candidates(root, webview_module)
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    exists = path.is_file()
    size = None
    loader_file_version = None
    browser_version = None
    if exists:
        try:
            size = path.stat().st_size
            loader_file_version = _windows_file_version(path)
            browser_version = _available_browser_version(path)
        except OSError as exc:
            return {
                "required": sys.platform == "win32",
                "ok": False,
                "status": "MISSING",
                "path": str(path),
                "candidate_paths": [str(candidate) for candidate in candidates],
                "bytes": None,
                "loader_file_version": None,
                "browser_version": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    applicable = sys.platform == "win32"
    browser_parsed = _parse_version_prefix(browser_version)
    ok = not applicable or bool(
        exists
        and loader_file_version
        and loader_file_version != "0.0.0.0"
        and browser_parsed is not None
        and browser_parsed >= WEBVIEW2_MINIMUM_VERSION
    )
    if not exists:
        error = "WebView2Loader.dll is missing"
    elif not loader_file_version or loader_file_version == "0.0.0.0":
        error = "WebView2Loader.dll has no valid PE version"
    elif browser_parsed is None:
        error = (
            "GetAvailableCoreWebView2BrowserVersionString returned no valid "
            "installed browser version"
        )
    elif browser_parsed < WEBVIEW2_MINIMUM_VERSION:
        error = f"installed WebView2 browser is below {WEBVIEW2_MINIMUM_VERSION_TEXT}"
    else:
        error = None
    return {
        "required": applicable,
        "ok": ok,
        "status": "PASS" if ok else "MISSING",
        "path": str(path),
        "candidate_paths": [str(candidate) for candidate in candidates],
        "bytes": size,
        "loader_file_version": loader_file_version,
        "browser_version": browser_version,
        "browser_version_valid": browser_parsed is not None
        and browser_parsed >= WEBVIEW2_MINIMUM_VERSION,
        "browser_version_minimum": WEBVIEW2_MINIMUM_VERSION_TEXT,
        "browser_api": "GetAvailableCoreWebView2BrowserVersionString",
        "error": error,
    }


def offline_installer_info(
    resource_root: Path | str,
    *,
    verify_hash: bool = False,
) -> dict[str, Any]:
    """Locate and describe the approved offline WebView2 installer."""

    root = Path(resource_root).expanduser().resolve()
    candidates = _installer_candidates(root)
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    exists = path.is_file()
    actual_bytes = None
    actual_hash = None
    if exists:
        try:
            actual_bytes = path.stat().st_size
            if verify_hash:
                actual_hash = _sha256(path)
        except OSError as exc:
            exists = False
            return {
                "expected_relative_path": str(WEBVIEW2_INSTALLER_RELATIVE),
                "path": str(path),
                "candidate_paths": [str(candidate) for candidate in candidates],
                "exists": False,
                "bytes": None,
                "sha256": None,
                "expected_bytes": WEBVIEW2_INSTALLER_BYTES,
                "expected_sha256": WEBVIEW2_INSTALLER_SHA256,
                "size_matches": False,
                "hash_matches": False if verify_hash else None,
                "hash_verification": "CHECKED" if verify_hash else "NOT_CHECKED",
                "error": f"{type(exc).__name__}: {exc}",
            }
    size_matches = actual_bytes == WEBVIEW2_INSTALLER_BYTES
    hash_matches = (
        actual_hash == WEBVIEW2_INSTALLER_SHA256 if verify_hash else None
    )
    return {
        "expected_relative_path": str(WEBVIEW2_INSTALLER_RELATIVE),
        "candidate_paths": [str(candidate) for candidate in candidates],
        "path": str(path),
        "exists": exists,
        "bytes": actual_bytes,
        "sha256": actual_hash,
        "expected_bytes": WEBVIEW2_INSTALLER_BYTES,
        "expected_sha256": WEBVIEW2_INSTALLER_SHA256,
        "size_matches": size_matches,
        "hash_matches": hash_matches,
        "hash_verification": "CHECKED" if verify_hash else "NOT_CHECKED",
        "error": None,
    }


def require_edgechromium_renderer(
    webview_module: Any,
    resource_root: Path | str,
) -> dict[str, Any]:
    """Preflight the explicit EdgeChromium path without importing WinForms."""

    loader = webview_loader_info(resource_root, webview_module)
    if not loader["ok"]:
        raise PackagedPrerequisiteError(
            "The packaged pywebview EdgeChromium loader/browser preflight failed: "
            f"{loader.get('error') or 'unknown preflight error'} "
            f"({loader.get('path')}); refusing an MSHTML fallback."
        )
    return {"requested_renderer": "edgechromium", "webview_loader": loader}


def webview2_prerequisites(resource_root: Path | str) -> dict[str, Any]:
    """Check native GUI prerequisites without importing or starting pywebview."""

    installer = offline_installer_info(resource_root)
    if sys.platform != "win32":
        return {
            "status": "NOT_APPLICABLE",
            "ok": True,
            "message": "WebView2/.NET prerequisite check is Windows-only.",
            "webview2": {"required": False, "status": "NOT_APPLICABLE", "found": None},
            "dotnet": {"required": False, "status": "NOT_APPLICABLE", "found": None},
            "offline_installer": installer,
        }

    webview2 = _webview2_runtime()
    dotnet = _dotnet_framework()
    ok = bool(webview2["found"] and dotnet["found"])
    missing: list[str] = []
    if not webview2["found"]:
        missing.append("Microsoft Edge WebView2 Runtime")
    if not dotnet["found"]:
        missing.append(".NET Framework 4.6.2 or newer")
    if ok:
        message = "WebView2 EdgeChromium and .NET prerequisites are available."
    else:
        message = (
            "Missing native desktop prerequisite(s): "
            + ", ".join(missing)
            + ". Install the approved offline WebView2 runtime manually from "
            + str(installer["path"])
            + "; TreeLineWarning does not install prerequisites automatically or "
            + "fall back to MSHTML."
        )
    return {
        "status": "PASS" if ok else "MISSING",
        "ok": ok,
        "message": message,
        "webview2": webview2,
        "dotnet": dotnet,
        "offline_installer": installer,
    }


def require_webview2_prerequisites(resource_root: Path | str) -> dict[str, Any]:
    report = webview2_prerequisites(resource_root)
    if not report["ok"]:
        raise PackagedPrerequisiteError(str(report["message"]))
    return report


def _validate_smoke_timeout(timeout_seconds: float) -> None:
    if (
        not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or timeout_seconds > MAX_SMOKE_TIMEOUT_SECONDS
    ):
        raise ValueError(
            "timeout_seconds must be finite, positive, and at most "
            f"{MAX_SMOKE_TIMEOUT_SECONDS:g}"
        )


def dependency_report(resource_root: Path | str) -> dict[str, Any]:
    """Build the complete file-friendly native/frozen dependency report."""

    try:
        from desktop.pdal_runtime import dependency_health, pillow_health
    except Exception as exc:  # pragma: no cover - defensive release diagnostics
        import_error = {
            "found": False,
            "minimal_operation_pass": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        pillow = {**import_error, "component": "pillow"}
        pdal = {**import_error, "component": "pdal"}
    else:
        try:
            pillow = pillow_health()
        except Exception as exc:  # pragma: no cover - native dependency-specific details
            pillow = {
                "found": False,
                "minimal_operation_pass": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            pdal = dependency_health()
        except Exception as exc:  # pragma: no cover - native dependency-specific details
            pdal = {
                "found": False,
                "minimal_operation_pass": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    origins = module_origins()
    webview_origin = origins.get("webview", {}).get("origin")
    loader = webview_loader_info(resource_root, webview_origin)
    prerequisites = webview2_prerequisites(resource_root)
    try:
        from algorithms.spans.gpu_clearance import probe_gpu

        capability, _module = probe_gpu()
        gpu = {
            **capability.to_dict(),
            "optional": True,
            "cpu_fallback_available": True,
        }
    except Exception as exc:  # optional GPU diagnostics must not block CPU startup
        gpu = {
            "available": False,
            "reason": f"GPU_PROBE_FAILED:{type(exc).__name__}:{exc}",
            "optional": True,
            "cpu_fallback_available": True,
        }
    native_pass = all(
        bool(item.get("found") and item.get("minimal_operation_pass"))
        for item in (pillow, pdal)
    )
    overall_pass = native_pass and bool(prerequisites["ok"]) and bool(loader["ok"])
    return {
        "schema": "treeline-packaged-dependency-report-v1",
        "generated_at": _utc_now(),
        "runtime": runtime_origins(resource_root),
        "module_origins": origins,
        "native_operations": {"pillow": pillow, "pdal": pdal},
        "gpu_acceleration": gpu,
        "webview2_prerequisites": prerequisites,
        "webview_loader": loader,
        "native_operations_pass": native_pass,
        "overall_pass": overall_pass,
    }


def _read_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Cache-Control": "no-store"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("backend returned a non-object JSON payload")
    return payload


def _modern_project_root(project_root: Path | str) -> Path:
    root = Path(project_root).expanduser().resolve()
    if not any(
        (root / name).is_file()
        for name in ("stage-a-project.json", "desktop-project.json")
    ):
        raise ValueError("Survey smoke requires a Stage-A desktop project workspace")
    return root


def current_survey_smoke(
    project_root: Path | str,
    resource_root: Path | str,
    *,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """Open the modern backend and retrieve only the read-only current report."""

    _validate_smoke_timeout(timeout_seconds)
    root = _modern_project_root(project_root)
    from desktop.app import BackendProcess

    backend = BackendProcess(root, Path(resource_root).expanduser().resolve())
    started = time.monotonic()
    url = None
    try:
        url = backend.start()
        payload = _read_json(
            url.rstrip("/") + "/api/current-report",
            timeout_seconds=timeout_seconds,
        )
        gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
        wires = payload.get("wires") if isinstance(payload.get("wires"), dict) else {}
        segments = wires.get("segments") if isinstance(wires.get("segments"), list) else []
        canonical_rows = payload.get("trees")
        kpis = payload.get("kpis")
        checks = {
            "api_ok": payload.get("ok") is True,
            "current_report_finalized": payload.get("status") == "FINALIZED",
            "current_gate_pass": gate.get("status") == "PASS",
            "canonical_current_report_retrieved": (
                payload.get("schema") == "treeline-current-final-report-v1"
                and payload.get("current_only") is True
                and isinstance(canonical_rows, list)
                and isinstance(kpis, dict)
            ),
            "w1_curves_are_persisted": wires.get("source") == "PERSISTED_SPAN_WIRE_RESULT_W1",
            "w1_curves_are_separate_segments": bool(segments)
            and all(
                isinstance(segment, dict)
                and isinstance(segment.get("xyz"), list)
                and len(segment["xyz"]) >= 2
                for segment in segments
            ),
            "w1_geometry_is_fitted_current": (
                wires.get("geometry_kind") == "FITTED_CURRENT_GEOMETRY"
            ),
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        return {
            "schema": "treeline-current-survey-smoke-v1",
            "status": status,
            "generated_at": _utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "project_root": str(root),
            "backend_url": url,
            "checks": checks,
            "gate": gate,
            "current_wires": wires,
            "report": payload,
            "read_only_actions": {
                "analysis_started": False,
                "editor_operations": False,
                "finalize_started": False,
                "prediction_export_requested": False,
            },
        }
    finally:
        backend.stop()


def _completion_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.complete{output_path.suffix}")
    return output_path.with_name(output_path.name + ".complete.json")


def finite_backend_smoke(
    project_root: Path | str,
    resource_root: Path | str,
    output_path: Path | str,
    *,
    stop_signal: Path | str | None = None,
    timeout_seconds: float = 120.0,
) -> tuple[int, dict[str, Any]]:
    """Publish a backend URL, then serve it until a signal or finite timeout."""

    _validate_smoke_timeout(timeout_seconds)
    root = _modern_project_root(project_root)
    output = Path(output_path).expanduser().resolve()
    stop = (
        Path(stop_signal).expanduser().resolve()
        if stop_signal is not None
        else output.with_name(output.stem + ".stop")
    )
    completion = _completion_path(output)
    if output == stop or output == completion:
        raise ValueError("diagnostic output, completion, and stop paths must differ")
    if output.exists() or completion.exists():
        raise FileExistsError("backend smoke diagnostic paths must be new")

    from desktop.app import BackendProcess

    backend = BackendProcess(root, Path(resource_root).expanduser().resolve())
    started = time.monotonic()
    url = None
    result: dict[str, Any]
    exit_code = 5
    try:
        url = backend.start()
        started_payload = {
            "schema": "treeline-finite-backend-smoke-v1",
            "status": "RUNNING",
            "generated_at": _utc_now(),
            "project_root": str(root),
            "resource_root": str(Path(resource_root).expanduser().resolve()),
            "backend_url": url,
            "stop_signal": str(stop),
            "completion_output": str(completion),
            "timeout_seconds": timeout_seconds,
        }
        write_json_exclusive(output, started_payload)
        deadline = time.monotonic() + timeout_seconds
        stopped_by_signal = False
        while time.monotonic() < deadline:
            if stop.is_file():
                stopped_by_signal = True
                break
            time.sleep(0.2)
        result = {
            **started_payload,
            "status": "STOPPED" if stopped_by_signal else "TIMEOUT",
            "finished_at": _utc_now(),
            "elapsed_seconds": time.monotonic() - started,
        }
        exit_code = 0 if stopped_by_signal else 5
    except KeyboardInterrupt:
        result = {
            "schema": "treeline-finite-backend-smoke-v1",
            "status": "INTERRUPTED",
            "generated_at": _utc_now(),
            "project_root": str(root),
            "backend_url": url,
            "stop_signal": str(stop),
            "completion_output": str(completion),
            "error": "KeyboardInterrupt",
        }
        exit_code = 130
    except Exception as exc:
        result = {
            "schema": "treeline-finite-backend-smoke-v1",
            "status": "FAIL",
            "generated_at": _utc_now(),
            "project_root": str(root),
            "backend_url": url,
            "stop_signal": str(stop),
            "completion_output": str(completion),
            "error": f"{type(exc).__name__}: {exc}",
        }
        exit_code = 3
    finally:
        backend.stop()
    if not output.exists():
        try:
            write_json_exclusive(output, result)
        except FileExistsError:
            result["diagnostic_write"] = "SKIPPED_EXISTING_PATH"
    if output.exists() and result.get("status") != "RUNNING":
        try:
            write_json_exclusive(completion, result)
        except FileExistsError:
            result["completion_write"] = "SKIPPED_EXISTING_PATH"
    return exit_code, result


__all__ = [
    "DOTNET_MINIMUM_RELEASE",
    "MAX_SMOKE_TIMEOUT_SECONDS",
    "PackagedPrerequisiteError",
    "WEBVIEW2_EVIDENCE_INSTALLER_RELATIVE",
    "WEBVIEW2_INSTALLER_BYTES",
    "WEBVIEW2_INSTALLER_RELATIVE",
    "WEBVIEW2_INSTALLER_SHA256",
    "WEBVIEW2_MINIMUM_VERSION",
    "WEBVIEW2_MINIMUM_VERSION_TEXT",
    "current_survey_smoke",
    "dependency_report",
    "finite_backend_smoke",
    "module_origins",
    "offline_installer_info",
    "require_edgechromium_renderer",
    "require_webview2_prerequisites",
    "runtime_origins",
    "webview2_prerequisites",
    "write_json_exclusive",
]
