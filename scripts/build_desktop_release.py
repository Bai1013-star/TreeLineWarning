"""Prepare a reviewable Survey-only desktop package or run its PyInstaller build.

The default action is a metadata-only preparation.  ``--execute`` is required
to invoke PyInstaller.  Every run uses two new directories supplied by the
caller (or generated with a timestamp and nonce); an existing directory is a
hard error and is never deleted or reused.

This is intentionally a small release boundary for the current desktop
baseline.  It does not install Python, Conda, PDAL, WebView2, or any other
dependency, and it does not perform scientific or REAL-data acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import shutil
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NATIVE_RUNTIME = REPO_ROOT / ".build-pdal-runtime-final-20260830"
DEFAULT_PROJ_DATA = REPO_ROOT / ".build-pdal-proj-data-final-20260830-02"
DEFAULT_VENV_PYTHON = REPO_ROOT / ".venv311" / "Scripts" / "python.exe"
DEFAULT_SPEC = REPO_ROOT / "desktop_build.spec"
DEFAULT_GUIDE = REPO_ROOT / "docs" / "WINDOWS_USER_GUIDE.zh-CN.md"
DEFAULT_EVIDENCE = REPO_ROOT / "acceptance_evidence" / "final_delivery_20260830"
DEFAULT_NOTICE_TEXT = DEFAULT_EVIDENCE / "THIRD_PARTY_NOTICES.final_native03.txt"
DEFAULT_NOTICE_JSON = DEFAULT_EVIDENCE / "THIRD_PARTY_NOTICES.final_native03.json"
DEFAULT_WEBVIEW_ROOT = DEFAULT_EVIDENCE / "webview2_offline_prerequisite"
WEBVIEW2_INSTALLER_NAME = "MicrosoftEdgeWebView2RuntimeInstallerX64.exe"
WEBVIEW2_INSTALLER_RELATIVE = Path(
    "webview2_offline_prerequisite",
    "offline_verified",
    WEBVIEW2_INSTALLER_NAME,
)
WEBVIEW2_INSTALLER_BYTES = 258_438_352
WEBVIEW2_INSTALLER_SHA256 = (
    "987a9d8b3107e84f9b53b4a077d28ae4814fc3d964d5a55c559e7334bbf24d61"
)
WEBVIEW2_RECEIPTS = (
    "authenticode.json",
    "offline_download.json",
    "official_webview2_eula.json",
)
MAX_BUILD_TIMEOUT_SECONDS = 7_200.0
BUILD_SCHEMA = "treeline-desktop-release-receipt-v1"
MANIFEST_SCHEMA = "treeline-desktop-release-file-manifest-v1"
FINAL_PACKAGE_NAME = "TreeLineWarning-Engineering-Release-2026-Final"


class ReleaseError(RuntimeError):
    """Raised when release inputs or output boundaries are unsafe."""


@dataclass(frozen=True)
class ReleaseOptions:
    """Resolved inputs for one immutable prepare/build attempt."""

    dist_root: Path
    work_root: Path
    native_runtime: Path
    proj_data_root: Path | None
    python_executable: Path
    spec_path: Path
    guide_path: Path
    notice_text: Path
    notice_json: Path
    webview_root: Path
    execute: bool = False
    include_webview: bool = True
    timeout_seconds: float = 1_800.0


@dataclass(frozen=True)
class NativeRuntime:
    root: Path
    manifest: Path
    manifest_sha256: str
    file_count: int
    versions: dict[str, Any]


@dataclass(frozen=True)
class ProjData:
    root: Path
    manifest: Path
    manifest_sha256: str
    file_count: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repo_path(value: Path | str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & 0x400)


def _lexists(path: Path) -> bool:
    return os.path.lexists(str(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file() or _is_reparse(path):
        raise ReleaseError(f"{label} must be a regular file without a link/reparse point: {path}")
    return path


def _require_directory(path: Path, label: str) -> Path:
    if not path.is_dir() or _is_reparse(path):
        raise ReleaseError(f"{label} must be a real directory without a link/reparse point: {path}")
    return path


def _ensure_real_tree(root: Path, label: str) -> None:
    _require_directory(root, label)
    for path in root.rglob("*"):
        if _is_reparse(path):
            raise ReleaseError(f"{label} contains a link/reparse point: {path}")


def _ensure_new_roots(dist_root: Path, work_root: Path) -> None:
    if dist_root == work_root:
        raise ReleaseError("dist and work roots must be different paths")
    if dist_root.is_relative_to(work_root) or work_root.is_relative_to(dist_root):
        raise ReleaseError("dist and work roots must not contain one another")
    for label, path in (("dist root", dist_root), ("work root", work_root)):
        if _lexists(path):
            raise ReleaseError(f"refusing to reuse existing {label}: {path}")


def _default_roots() -> tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    nonce = secrets.token_hex(4)
    stem = f"TreeLineWarning-survey-only-{stamp}-{nonce}"
    return REPO_ROOT / "dist" / stem, REPO_ROOT / "build" / stem


def _git_capture(args: Sequence[str], *, binary: bool = False) -> bytes | str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            timeout=60,
            text=not binary,
            encoding=None if binary else "utf-8",
            errors=None if binary else "strict",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReleaseError(f"git snapshot failed ({' '.join(args)}): {exc}") from exc
    return result.stdout


def tracked_source_digest() -> dict[str, Any]:
    """Capture HEAD and a deterministic digest of tracked worktree bytes."""

    head = str(_git_capture(["rev-parse", "HEAD"])).strip()
    raw_paths = _git_capture(["ls-files", "-z"], binary=True)
    assert isinstance(raw_paths, bytes)
    digest = hashlib.sha256()
    file_count = 0
    for raw_path in raw_paths.split(b"\0"):
        if not raw_path:
            continue
        relative = os.fsdecode(raw_path)
        path = REPO_ROOT / Path(relative)
        if not path.is_file():
            raise ReleaseError(f"tracked source file is unavailable: {relative}")
        normalized = raw_path.replace(b"\\", b"/")
        size = path.stat().st_size
        digest.update(b"PATH\0")
        digest.update(len(normalized).to_bytes(8, "big"))
        digest.update(normalized)
        digest.update(b"SIZE\0")
        digest.update(size.to_bytes(8, "big"))
        digest.update(b"DATA\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        file_count += 1
    return {
        "head": head,
        "tracked_source_sha256": digest.hexdigest(),
        "tracked_file_count": file_count,
    }


def _project_version() -> str:
    try:
        payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        version = payload.get("project", {}).get("version")
    except (OSError, tomllib.TOMLDecodeError, AttributeError):
        version = None
    if not isinstance(version, str) or not version:
        raise ReleaseError("pyproject.toml has no project.version")
    return version


def _validate_python(executable: Path) -> dict[str, Any]:
    _require_file(executable, "CPython 3.11 executable")
    venv_root = executable.parent.parent
    config_path = _require_file(venv_root / "pyvenv.cfg", "CPython venv configuration")
    config = config_path.read_text(encoding="utf-8", errors="replace")
    version_line = next(
        (line.strip() for line in config.splitlines() if line.lower().startswith("version")),
        "",
    )
    if not version_line.lower().startswith("version = 3.11"):
        raise ReleaseError(
            "release builds require CPython 3.11 in the selected venv; "
            f"found {version_line or 'no version'} at {config_path}"
        )
    system_site = next(
        (
            line.strip().lower()
            for line in config.splitlines()
            if line.lower().startswith("include-system-site-packages")
        ),
        "",
    )
    if system_site and not system_site.endswith("false"):
        raise ReleaseError("selected CPython venv must exclude system site packages")
    site = venv_root / "Lib" / "site-packages"
    _require_directory(site, "CPython venv site-packages")
    _require_directory(site / "PyInstaller", "PyInstaller package")
    return {
        "path": str(executable),
        "venv_root": str(venv_root),
        "version_config": version_line,
        "include_system_site_packages": False,
    }


def _validate_native_runtime(root: Path) -> NativeRuntime:
    """Validate the real-file PDAL + compatibility snapshot and every hash."""

    _ensure_real_tree(root, "materialized native runtime")
    manifest_path = _require_file(root / "native_runtime_manifest.json", "native runtime manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"native runtime manifest is unreadable: {manifest_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise ReleaseError("native runtime manifest schema must be 1")
    policy = str(payload.get("snapshot_policy", ""))
    if "new_destination_only" not in policy or "no junctions or symlinks" not in policy:
        raise ReleaseError("native runtime manifest does not prove a real immutable snapshot")
    raw_files = payload.get("files", [])
    raw_compatibility = payload.get("compatibility_files", [])
    if not isinstance(raw_files, list) or not isinstance(raw_compatibility, list):
        raise ReleaseError("native runtime manifest payload sections must be lists")
    raw_rows = [*raw_files, *raw_compatibility]
    if not raw_rows:
        raise ReleaseError("native runtime manifest has no payload rows")
    seen: set[str] = set()
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ReleaseError("native runtime manifest contains an invalid file row")
        relative_value = row.get("destination_relative")
        expected_hash = row.get("sha256")
        expected_size = row.get("bytes")
        if not isinstance(relative_value, str) or not relative_value:
            raise ReleaseError("native runtime manifest contains an invalid destination path")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ReleaseError(f"native runtime manifest escapes its root: {relative_value}")
        normalized = relative.as_posix().lower()
        if normalized in seen:
            raise ReleaseError(f"native runtime manifest repeats {relative_value}")
        seen.add(normalized)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ReleaseError(f"native runtime manifest has no valid hash for {relative_value}")
        if type(expected_size) is not int or expected_size < 0:
            raise ReleaseError(f"native runtime manifest has no valid size for {relative_value}")
        file_path = _require_file(root / relative, "native runtime payload")
        if (
            file_path.stat().st_size != expected_size
            or _sha256(file_path).lower() != expected_hash.lower()
        ):
            raise ReleaseError(f"native runtime payload hash/size mismatch: {relative_value}")
    required = {
        "lib/site-packages/pdal/libpdalpython.cp311-win_amd64.pyd",
        "library/bin/pdalcpp.dll",
        "compat/openjp2.dll",
        "compat/zlib-ng2.dll",
    }
    if not required <= seen:
        raise ReleaseError("native runtime is missing the PDAL binding or required compat DLLs")
    actual = {
        path.relative_to(root).as_posix().lower()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != seen:
        extras = sorted(actual - seen)
        missing = sorted(seen - actual)
        raise ReleaseError(
            "native runtime manifest does not cover the real tree"
            f" (extra={extras[:3]}, missing={missing[:3]})"
        )
    versions = payload.get("versions")
    if not isinstance(versions, dict):
        versions = {}
    return NativeRuntime(
        root=root,
        manifest=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        file_count=len(raw_rows),
        versions=dict(versions),
    )


def _validate_proj_data(root: Path | None) -> ProjData | None:
    """Validate the optional explicit PROJ data overlay when selected."""

    if root is None:
        return None
    _ensure_real_tree(root, "materialized PROJ data overlay")
    manifests = sorted(root.glob("*manifest*.json"))
    if len(manifests) != 1:
        raise ReleaseError(
            "PROJ data overlay must contain exactly one top-level *manifest*.json "
            f"(found {len(manifests)})"
        )
    manifest_path = _require_file(manifests[0], "PROJ data manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"PROJ data manifest is unreadable: {manifest_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise ReleaseError("PROJ data manifest schema must be 1")
    payload_rows = []
    for key in ("files", "provenance_copies"):
        raw_rows = payload.get(key)
        if raw_rows is not None:
            if not isinstance(raw_rows, list):
                raise ReleaseError(f"PROJ data manifest {key} must be a list")
            if key == "provenance_copies":
                for row in raw_rows:
                    if not isinstance(row, dict):
                        raise ReleaseError(
                            "PROJ provenance copies must be explicitly matched to their source"
                        )
                    source_proof_row = (
                        row.get("matches_source") is None
                        and row.get("source") is None
                        and str(row.get("destination_relative", "")).replace("\\", "/")
                        == "provenance/source_proof.json"
                    )
                    if row.get("matches_source") is not True and not source_proof_row:
                        raise ReleaseError(
                            "PROJ provenance copies must be explicitly matched to their source"
                        )
            payload_rows.extend(raw_rows)
    if not payload_rows:
        legacy_rows = payload.get("artifacts")
        if isinstance(legacy_rows, list):
            payload_rows.extend(legacy_rows)
    if not payload_rows:
        raise ReleaseError("PROJ data manifest must list its exact payload and provenance files")
    seen: set[str] = set()
    for row in payload_rows:
        if not isinstance(row, dict):
            raise ReleaseError("PROJ data manifest contains an invalid file row")
        relative_value = row.get("destination_relative") or row.get("relative_path")
        expected_hash = row.get("sha256")
        expected_size = row.get("bytes", row.get("byte_size"))
        if not isinstance(relative_value, str) or not isinstance(expected_hash, str):
            raise ReleaseError("PROJ data manifest file rows need relative path and sha256")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ReleaseError(f"PROJ data manifest escapes its root: {relative_value}")
        normalized = relative.as_posix().lower()
        if normalized in seen:
            raise ReleaseError(f"PROJ data manifest repeats {relative_value}")
        path = _require_file(root / relative, "PROJ data payload")
        if type(expected_size) is not int or expected_size < 0:
            raise ReleaseError(f"PROJ data manifest has no valid size for {relative_value}")
        if path.stat().st_size != expected_size:
            raise ReleaseError(f"PROJ data size mismatch: {relative_value}")
        if len(expected_hash) != 64 or _sha256(path).lower() != expected_hash.lower():
            raise ReleaseError(f"PROJ data hash mismatch: {relative_value}")
        seen.add(normalized)
    if not any(Path(item).name.lower() == "proj.db" for item in seen):
        raise ReleaseError("PROJ data overlay manifest must include proj.db")
    actual = {
        path.relative_to(root).as_posix().lower()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != seen:
        raise ReleaseError("PROJ data manifest does not cover the real tree")
    return ProjData(
        root=root,
        manifest=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        file_count=len(payload_rows),
    )


def _validate_notices(text_path: Path, json_path: Path) -> dict[str, Any]:
    _require_file(text_path, "third-party notice text")
    _require_file(json_path, "third-party notice JSON")
    text = text_path.read_bytes()
    if b"ENGINEERING INVENTORY ONLY" not in text:
        raise ReleaseError("third-party notice text is not the reviewed engineering inventory")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"third-party notice JSON is unreadable: {json_path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("inventory_kind") != "desktop-third-party-notices"
    ):
        raise ReleaseError("third-party notice JSON has an unexpected inventory kind")
    unresolved = payload.get("unresolved_provenance")
    return {
        "text_path": str(text_path),
        "text_sha256": _sha256(text_path),
        "json_path": str(json_path),
        "json_sha256": _sha256(json_path),
        "unresolved_provenance_count": len(unresolved) if isinstance(unresolved, list) else None,
    }


def _validate_webview(root: Path) -> dict[str, Any]:
    installer = root / "offline_verified" / WEBVIEW2_INSTALLER_NAME
    auth_path = root / "authenticode.json"
    download_path = root / "offline_download.json"
    eula_path = root / "official_webview2_eula.json"
    _require_file(installer, "verified WebView2 standalone installer")
    for name, path in (
        ("WebView2 Authenticode receipt", auth_path),
        ("WebView2 download receipt", download_path),
        ("WebView2 EULA record", eula_path),
    ):
        _require_file(path, name)
    if installer.stat().st_size != WEBVIEW2_INSTALLER_BYTES:
        raise ReleaseError(
            "verified WebView2 installer size does not match the acceptance contract"
        )
    installer_hash = _sha256(installer)
    if installer_hash.lower() != WEBVIEW2_INSTALLER_SHA256:
        raise ReleaseError(
            "verified WebView2 installer hash does not match the acceptance contract"
        )
    try:
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
        download = json.loads(download_path.read_text(encoding="utf-8"))
        eula = json.loads(eula_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError("WebView2 verification receipt is unreadable") from exc
    if (
        not isinstance(auth, dict)
        or auth.get("status") != "Valid"
        or auth.get("executed") is not False
        or auth.get("length") != WEBVIEW2_INSTALLER_BYTES
    ):
        raise ReleaseError("WebView2 Authenticode receipt is not valid/download-only")
    auth_relative = str(auth.get("path", "")).replace("\\", "/")
    if not auth_relative.endswith(f"/offline_verified/{WEBVIEW2_INSTALLER_NAME}"):
        raise ReleaseError("WebView2 Authenticode receipt is not bound to the accepted installer")
    if not isinstance(download, dict):
        raise ReleaseError("WebView2 download receipt is not a JSON object")
    accepted_relative = (Path("offline_verified") / WEBVIEW2_INSTALLER_NAME).as_posix()
    recorded_relative = str(download.get("relative_path", "")).replace("\\", "/")
    if (
        download.get("action") != "download_only_not_executed"
        or download.get("bytes") != WEBVIEW2_INSTALLER_BYTES
        or str(download.get("sha256", "")).lower() != WEBVIEW2_INSTALLER_SHA256
        or recorded_relative != accepted_relative
    ):
        raise ReleaseError("WebView2 download receipt is not download-only")
    if not isinstance(eula, dict) or not eula.get("evergreenHtml"):
        raise ReleaseError("WebView2 official EULA record is missing unchanged HTML")
    return {
        "source_root": str(root),
        "installer_source_relative": str(Path("offline_verified") / WEBVIEW2_INSTALLER_NAME),
        "installer_sha256": installer_hash,
        "installer_bytes": installer.stat().st_size,
        "authenticode_sha256": _sha256(auth_path),
        "download_receipt_sha256": _sha256(download_path),
        "eula_record_sha256": _sha256(eula_path),
        "offline_verified": True,
        "executed": False,
        "clean_pc": "NOT_TESTED",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _copy_exact(source: Path, destination: Path) -> None:
    _require_file(source, "release input")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _render_version(
    *,
    version: str,
    status: str,
    source: dict[str, str],
    native: NativeRuntime,
    proj_data: ProjData | None,
) -> str:
    lines = [
        "TreeLineWarning Engineering Release 2026 Final",
        f"version = {version}",
        f"status = {status}",
        "baseline = Survey Core",
        "parallel = BOUNDED_DETERMINISTIC_INCLUDED",
        "gpu = OPTIONAL_CUPY_CUDA13_CERTIFIED_SCREEN_CPU_EXACT_AUTHORITY",
        "cuda_toolkit_required = NO",
        "preview = EXPERIMENTAL_NOT_ACCEPTED",
        f"source_head = {source['head']}",
        f"tracked_source_sha256 = {source['tracked_source_sha256']}",
        f"native_runtime_manifest_sha256 = {native.manifest_sha256}",
        f"pdal_proj_data = {'SELECTED' if proj_data is not None else 'NOT_SELECTED'}",
        f"built_utc = {_utc_now()}",
    ]
    return "\n".join(lines) + "\n"


def _render_contract(
    *,
    status: str,
    source: dict[str, str],
    native: NativeRuntime,
    proj_data: ProjData | None,
    notice: dict[str, Any],
    webview: dict[str, Any],
) -> str:
    unresolved = notice.get("unresolved_provenance_count")
    webview_path = str(WEBVIEW2_INSTALLER_RELATIVE).replace("\\", "/")
    webview_verified = bool(webview.get("offline_verified"))
    lines = [
        "# TreeLineWarning packaged acceptance contract",
        "",
        "This package targets Survey Core; local preview is experimental and not accepted.",
        "It is not a clean-PC acceptance result and carries no legal clearance claim.",
        "",
        f"- package_status: {status}",
        "- baseline: Survey Core",
        "- package_type: Windows x64 ONEDIR",
        "- parallel: BOUNDED_DETERMINISTIC_INCLUDED",
        "- gpu: OPTIONAL_CUPY_CUDA13_CERTIFIED_SCREEN_CPU_EXACT_AUTHORITY",
        "- cuda_toolkit_required: false (compatible NVIDIA driver required for GPU)",
        "- cpu_fallback: required and authoritative",
        "- preview: EXPERIMENTAL_NOT_ACCEPTED",
        f"- source_head: {source['head']}",
        f"- tracked_source_sha256: {source['tracked_source_sha256']}",
        f"- native_runtime_manifest_sha256: {native.manifest_sha256}",
        f"- native_payload_files: {native.file_count}",
        f"- native_versions: {json.dumps(native.versions, ensure_ascii=False, sort_keys=True)}",
        (
            f"- pdal_proj_data: {proj_data.manifest.name} sha256={proj_data.manifest_sha256}"
            if proj_data is not None
            else "- pdal_proj_data: NOT_SELECTED"
        ),
        f"- third_party_notice_text_sha256: {notice['text_sha256']}",
        f"- third_party_notice_json_sha256: {notice['json_sha256']}",
        f"- notice_unresolved_provenance_entries: {unresolved}",
        "- clean_pc: NOT_TESTED",
        f"- offline_verified: {str(webview['offline_verified']).lower()}",
        (
            f"- offline_installer_path: {webview_path}"
            if webview_verified
            else "- offline_installer_path: NOT_SELECTED"
        ),
        (
            f"- offline_installer_sha256: {webview.get('installer_sha256')}"
            if webview_verified
            else "- offline_installer_sha256: NOT_SELECTED"
        ),
        (
            f"- offline_installer_bytes: {webview.get('installer_bytes')}"
            if webview_verified
            else "- offline_installer_bytes: NOT_SELECTED"
        ),
        f"- offline_installer_included: {str(webview.get('included_in_package', False)).lower()}",
        "- offline_installer_executed: false",
        "",
        "The WebView2 installer is an optional manual prerequisite for a user.  The",
        "application never auto-installs it.  Retain the Authenticode/download receipts",
        "and official EULA beside the installer.  Only the offline_verified standalone",
        "installer is package input; bootstrapper or incomplete sibling downloads are excluded.",
        "",
        "The user may relocate the extracted directory, including Chinese characters and",
        "spaces, into a writable folder.  Python, Conda, PDAL, and developer paths are not",
        "required at runtime.  A clean-PC WebView2/.NET launch remains NOT_TESTED.",
    ]
    return "\n".join(lines) + "\n"


def _package_manifest(package_root: Path, *, status: str) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(package_root.rglob("*")):
        if _is_reparse(path):
            raise ReleaseError(f"package contains a link/reparse point: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(package_root).as_posix()
        if relative == "release_file_manifest.json":
            continue
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "package_status": status,
        "baseline": "Survey Core",
        "package_type": "Windows x64 ONEDIR",
        "parallel": "BOUNDED_DETERMINISTIC_INCLUDED",
        "gpu": "OPTIONAL_CUPY_CUDA13_CERTIFIED_SCREEN_CPU_EXACT_AUTHORITY",
        "preview": "EXPERIMENTAL_NOT_ACCEPTED",
        "self_entry": "EXCLUDED_AFTER_WRITE",
        "files": entries,
    }


def _run_pyinstaller(options: ReleaseOptions, work_root: Path) -> dict[str, Any]:
    log_path = work_root / "pyinstaller.log"
    command = [
        str(options.python_executable),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--distpath",
        str(options.dist_root),
        "--workpath",
        str(work_root),
        str(options.spec_path),
    ]
    env = os.environ.copy()
    env["TREELINE_PDAL_RUNTIME_ENV"] = str(options.native_runtime)
    env["TREELINE_COMPAT_DLL_ROOT"] = str(options.native_runtime / "compat")
    env["TREELINE_PACKAGE_DIR_NAME"] = FINAL_PACKAGE_NAME
    if options.proj_data_root is not None:
        env["TREELINE_PDAL_PROJ_DATA_ROOT"] = str(options.proj_data_root)
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_EXE",
        "CONDA_PYTHON_EXE",
    ):
        env.pop(name, None)
    started = time_monotonic()
    returncode: int | None = None
    timed_out = False
    error: str | None = None
    process: subprocess.Popen[str] | None = None
    reader: threading.Thread | None = None
    log_handle = log_path.open("x", encoding="utf-8", newline="\n")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None

        def stream_output() -> None:
            try:
                for line in process.stdout:
                    log_handle.write(line)
                    log_handle.flush()
            except (OSError, ValueError):
                # The parent closes the pipe after termination so the bounded
                # cleanup path cannot leave a writer behind the log handle.
                pass

        reader = threading.Thread(target=stream_output, name="pyinstaller-log", daemon=True)
        reader.start()
        while process.poll() is None:
            if time_monotonic() - started >= options.timeout_seconds:
                timed_out = True
                error = f"PyInstaller exceeded timeout {options.timeout_seconds:g}s"
                try:
                    process.kill()
                except OSError:
                    pass
                break
            import time

            time.sleep(0.1)
        returncode = process.wait(timeout=10)
    except OSError as exc:
        error = f"unable to start PyInstaller: {type(exc).__name__}: {exc}"
    except subprocess.TimeoutExpired:
        timed_out = True
        error = f"PyInstaller did not exit after timeout {options.timeout_seconds:g}s"
        if process is not None:
            try:
                process.kill()
            except (OSError, ValueError):
                pass
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                error = f"{error}; child did not exit after kill timeout"
    finally:
        if process is not None and process.stdout is not None:
            try:
                process.stdout.close()
            except (OSError, ValueError):
                pass
        if reader is not None:
            reader.join(timeout=10)
        log_handle.close()
    return {
        "command": command,
        "executed": True,
        "cwd": str(REPO_ROOT),
        "log_path": str(log_path),
        "log_sha256": _sha256(log_path),
        "elapsed_seconds": time_monotonic() - started,
        "returncode": returncode,
        "timed_out": timed_out,
        "error": error,
        "environment": {
            "TREELINE_PDAL_RUNTIME_ENV": str(options.native_runtime),
            "TREELINE_COMPAT_DLL_ROOT": str(options.native_runtime / "compat"),
            "TREELINE_PACKAGE_DIR_NAME": FINAL_PACKAGE_NAME,
            "TREELINE_PDAL_PROJ_DATA_ROOT": (
                str(options.proj_data_root) if options.proj_data_root is not None else None
            ),
            "removed_developer_variables": [
                "PYTHONHOME",
                "PYTHONPATH",
                "CONDA_PREFIX",
                "CONDA_DEFAULT_ENV",
                "CONDA_EXE",
                "CONDA_PYTHON_EXE",
            ],
        },
    }


def time_monotonic() -> float:
    # Kept as a tiny seam for tests without importing a release framework.
    import time

    return time.monotonic()


def _copy_package_metadata(
    package_root: Path,
    options: ReleaseOptions,
    *,
    status: str,
    source: dict[str, str],
    native: NativeRuntime,
    proj_data: ProjData | None,
    notice: dict[str, Any],
    webview: dict[str, Any],
    receipt: dict[str, Any],
    version: str,
) -> dict[str, Any]:
    package_root.mkdir(parents=True, exist_ok=True)
    _copy_exact(options.guide_path, package_root / options.guide_path.name)
    (package_root / "README.md").write_text(
        "# TreeLineWarning Engineering Release 2026 Final\n\n"
        "## 软件介绍\n\nTreeLineWarning 是 Windows x64 输电线路点云分析与单木复核工具。"
        "本版本支持 LAS/LAZ 导入、AUTO 分析、3D 复核、DELETE/MERGE/UNDO，"
        "以及 XLSX/HTML 最终报告导出。\n\n"
        "## 系统要求\n\n- Windows 10/11 x64\n- 建议 16 GB 以上内存和足够的本地磁盘空间\n"
        "- Microsoft Edge WebView2 Runtime；包内附离线安装程序\n"
        "- GPU 可选，CPU fallback 为权威路径\n\n"
        "## 安装运行方式\n\n完整解压 ZIP 到可写本地目录，保留 `_internal` 和所有相邻文件，"
        "双击 `TreeLineWarning.exe`。"
        "无需安装 Python、Conda 或 PDAL。\n\n"
        "## LAS/LAZ 导入方法\n\n新建或打开项目，点击“导入点云”，选择 `.las` 或 `.laz`；"
        "确认单位、杆塔顺序与档距。"
        "中文路径和含空格路径受支持。\n\n"
        "## 分析流程\n\n选择档距并执行 AUTO 分析；打开 3D 单木复核，使用 DELETE、MERGE、UNDO；"
        "确认后点击“完成复核”生成 final state。原始分割保持只读，修改仅记录在 edits 层。\n\n"
        "## 导出方法\n\n定稿后点击“导出预测输入”保存 `Prediction_Input.xlsx`，"
        "或点击“导出 HTML 报告”。"
        "两个导出均来自同一 final projection。\n\n"
        "## 已知限制\n\n- 首次大型点云分析可能较慢，速度取决于磁盘与内存。\n"
        "- GPU 仅用于经认证的净空候选筛选；精确距离判定仍由 CPU 完成。\n"
        "- 编辑后必须重新完成复核才能导出，以防旧报告泄漏。\n"
        "- 若系统缺少 WebView2，请由用户或 IT 手动运行包内离线安装程序。\n",
        encoding="utf-8",
        newline="\n",
    )
    (package_root / "QUICK_START.md").write_text(
        "# TreeLineWarning Quick Start\n\n"
        "1. Extract the entire folder to a writable local Windows x64 directory.\n"
        "2. Double-click TreeLineWarning.exe; retain _internal and all adjacent files.\n"
        "3. No Python, Conda, PDAL, PYTHONPATH or repository checkout is required.\n"
        "4. WebView2 and .NET Framework 4.6.2+ are Windows prerequisites. If needed,\n"
        "   ask IT to run webview2_offline_prerequisite/offline_verified/"
        "MicrosoftEdgeWebView2RuntimeInstallerX64.exe manually. No automatic install.\n"
        "5. New/Open Project > import LAS/LAZ > confirm units > towers/spans >\n"
        "   analyze (AUTO or SERIAL) > Tree Review > Finalize > current report/export.\n"
        "6. Close and restart, reopen the project, and verify finalized state/export.\n\n"
        "Parallel is unavailable in this release.\n"
        "Local future preview is experimental, not a validated forecast or safety result.\n"
        "See WINDOWS_USER_GUIDE.zh-CN.md and THIRD_PARTY_NOTICES.txt.\n",
        encoding="utf-8",
        newline="\n",
    )
    _copy_exact(options.notice_text, package_root / "THIRD_PARTY_NOTICES.txt")
    _copy_exact(options.notice_json, package_root / "THIRD_PARTY_NOTICES.json")
    _copy_exact(options.notice_text, package_root / "licenses" / "THIRD_PARTY_NOTICES.txt")
    _copy_exact(options.notice_json, package_root / "licenses" / "THIRD_PARTY_NOTICES.json")
    (package_root / "examples" / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (package_root / "examples" / "README.md").write_text(
        "# Examples\n\nUse your own LAS/LAZ data. This directory is intentionally "
        "data-free so the release stays compact.\n",
        encoding="utf-8",
        newline="\n",
    )
    (package_root / "configs" / "default.json").parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        package_root / "configs" / "default.json",
        {
            "schema": "treeline-engineering-release-config-v1",
            "execution_mode": "AUTO",
            "acceleration_mode": "AUTO",
            "scientific_configuration": "BUILT_IN_FROZEN_DEFAULTS",
        },
    )
    (package_root / "VERSION.txt").write_text(
        _render_version(
            version=version,
            status=status,
            source=source,
            native=native,
            proj_data=proj_data,
        ),
        encoding="utf-8",
        newline="\n",
    )
    webview_package = dict(webview)
    webview_package["included_in_package"] = False
    if options.include_webview:
        webview_source = options.webview_root / "offline_verified" / WEBVIEW2_INSTALLER_NAME
        _copy_exact(webview_source, package_root / WEBVIEW2_INSTALLER_RELATIVE)
        for name in WEBVIEW2_RECEIPTS:
            _copy_exact(
                options.webview_root / name,
                package_root / "webview2_offline_prerequisite" / name,
            )
        webview_package["included_in_package"] = True
    contract = _render_contract(
        status=status,
        source=source,
        native=native,
        proj_data=proj_data,
        notice=notice,
        webview=webview_package,
    )
    (package_root / "PACKAGED_ACCEPTANCE_CONTRACT.md").write_text(
        contract,
        encoding="utf-8",
        newline="\n",
    )
    _write_json(package_root / "build_receipt.json", receipt)
    manifest = _package_manifest(package_root, status=status)
    _write_json(package_root / "release_file_manifest.json", manifest)
    return {
        "package_root": str(package_root),
        "package_status": status,
        "webview": webview_package,
        "file_manifest": str(package_root / "release_file_manifest.json"),
    }


def run_release(options: ReleaseOptions) -> dict[str, Any]:
    """Run one safe prepare/build operation and return its receipt summary."""

    if not math.isfinite(options.timeout_seconds) or not (
        0 < options.timeout_seconds <= MAX_BUILD_TIMEOUT_SECONDS
    ):
        raise ReleaseError(
            f"timeout must be finite, positive, and <= {MAX_BUILD_TIMEOUT_SECONDS:g} seconds"
        )
    _ensure_new_roots(options.dist_root, options.work_root)
    if options.execute and options.proj_data_root is None:
        raise ReleaseError(
            "--execute requires an explicit --pdal-proj-data-root approved overlay"
        )
    if options.execute and not options.include_webview:
        raise ReleaseError(
            "--execute requires the verified WebView2 installer; remove --skip-webview2"
        )
    source_start = tracked_source_digest()
    native = _validate_native_runtime(options.native_runtime)
    proj_data = _validate_proj_data(options.proj_data_root)
    python_info = _validate_python(options.python_executable)
    _require_file(options.spec_path, "PyInstaller spec")
    _require_file(options.guide_path, "Chinese user guide")
    notice = _validate_notices(options.notice_text, options.notice_json)
    if options.include_webview:
        webview = _validate_webview(options.webview_root)
    else:
        webview = {
            "source_root": str(options.webview_root),
            "installer_source_relative": None,
            "installer_sha256": None,
            "installer_bytes": None,
            "authenticode_sha256": None,
            "download_receipt_sha256": None,
            "eula_record_sha256": None,
            "offline_verified": False,
            "executed": False,
            "clean_pc": "NOT_TESTED",
        }
    version = _project_version()

    started_at = _utc_now()
    options.work_root.mkdir(parents=True, exist_ok=False)
    process: dict[str, Any] = {
        "executed": False,
        "command": None,
        "log_path": None,
        "returncode": None,
        "timed_out": False,
        "elapsed_seconds": 0.0,
    }
    pyinstaller_package_root = options.dist_root / "TreeLineWarning"
    package_root = options.dist_root / FINAL_PACKAGE_NAME
    if options.execute:
        process = _run_pyinstaller(options, options.work_root)
        if (
            process.get("returncode") == 0
            and pyinstaller_package_root.is_dir()
            and not package_root.exists()
        ):
            pyinstaller_package_root.replace(package_root)
        source_end = tracked_source_digest()
        build_ok = (
            process.get("returncode") == 0
            and not process.get("timed_out")
            and process.get("error") is None
            and source_start == source_end
            and package_root.is_dir()
            and (package_root / "TreeLineWarning.exe").is_file()
        )
        status = "BUILT" if build_ok else "BUILD_FAILED"
    else:
        source_end = tracked_source_digest()
        status = "PREPARED_ONLY" if source_start == source_end else "SOURCE_CHANGED_DURING_PREPARE"
    receipt = {
        "schema": BUILD_SCHEMA,
        "package_status": status,
        "baseline": "Survey Core",
        "package_type": "Windows x64 ONEDIR",
        "project_version": version,
        "parallel": "BOUNDED_DETERMINISTIC_INCLUDED",
        "gpu": "OPTIONAL_CUPY_CUDA13_CERTIFIED_SCREEN_CPU_EXACT_AUTHORITY",
        "preview": "EXPERIMENTAL_NOT_ACCEPTED",
        "started_utc": started_at,
        "finished_utc": _utc_now(),
        "source_start": source_start,
        "source_end": source_end,
        "source_changed_during_run": source_start != source_end,
        "dist_root": str(options.dist_root),
        "work_root": str(options.work_root),
        "package_root": str(package_root),
        "python": python_info,
        "native_runtime": {
            "root": str(native.root),
            "manifest": str(native.manifest),
            "manifest_sha256": native.manifest_sha256,
            "file_count": native.file_count,
            "versions": native.versions,
        },
        "pdal_proj_data": (
            {
                "root": str(proj_data.root),
                "manifest": str(proj_data.manifest),
                "manifest_sha256": proj_data.manifest_sha256,
                "file_count": proj_data.file_count,
            }
            if proj_data is not None
            else {"status": "NOT_SELECTED"}
        ),
        "third_party_notices": notice,
        "webview2": webview,
        "webview2_included": options.include_webview,
        "process": process,
    }
    _write_json(options.work_root / "build_receipt.json", receipt)
    if status == "BUILD_FAILED":
        return receipt
    package_summary = _copy_package_metadata(
        package_root,
        options,
        status=status,
        source=source_end,
        native=native,
        proj_data=proj_data,
        notice=notice,
        webview=webview,
        receipt=receipt,
        version=version,
    )
    receipt["package"] = package_summary
    _write_json(options.work_root / "build_receipt.json", receipt)
    # Keep the package receipt in sync with the final package summary while the
    # file manifest remains the last package metadata file written.
    _write_json(package_root / "build_receipt.json", receipt)
    _write_json(
        package_root / "release_file_manifest.json",
        _package_manifest(package_root, status=status),
    )
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_dist, default_work = _default_roots()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-root", type=Path, default=default_dist)
    parser.add_argument("--work-root", type=Path, default=default_work)
    parser.add_argument("--native-runtime", type=Path, default=DEFAULT_NATIVE_RUNTIME)
    parser.add_argument(
        "--pdal-proj-data-root",
        type=Path,
        default=DEFAULT_PROJ_DATA if DEFAULT_PROJ_DATA.is_dir() else None,
        help="explicit approved PROJ data overlay; exported as TREELINE_PDAL_PROJ_DATA_ROOT",
    )
    parser.add_argument(
        "--python", dest="python_executable", type=Path, default=DEFAULT_VENV_PYTHON
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    parser.add_argument("--notice-text", type=Path, default=DEFAULT_NOTICE_TEXT)
    parser.add_argument("--notice-json", type=Path, default=DEFAULT_NOTICE_JSON)
    parser.add_argument("--webview-root", type=Path, default=DEFAULT_WEBVIEW_ROOT)
    parser.add_argument(
        "--skip-webview2",
        action="store_true",
        help="development-only metadata preparation",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run PyInstaller; preparation is the default",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dist_root = _repo_path(args.dist_root)
    work_root = _repo_path(args.work_root)
    options = ReleaseOptions(
        dist_root=dist_root,
        work_root=work_root,
        native_runtime=_repo_path(args.native_runtime),
        proj_data_root=(
            None
            if args.pdal_proj_data_root is None
            else _repo_path(args.pdal_proj_data_root)
        ),
        python_executable=_repo_path(args.python_executable),
        spec_path=_repo_path(args.spec),
        guide_path=_repo_path(args.guide),
        notice_text=_repo_path(args.notice_text),
        notice_json=_repo_path(args.notice_json),
        webview_root=_repo_path(args.webview_root),
        execute=bool(args.execute),
        include_webview=not bool(args.skip_webview2),
        timeout_seconds=float(args.timeout_seconds),
    )
    try:
        receipt = run_release(options)
    except ReleaseError as exc:
        print(f"release preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0 if receipt.get("package_status") in {"PREPARED_ONLY", "BUILT"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
