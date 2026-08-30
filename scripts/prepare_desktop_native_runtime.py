"""Materialize the approved native runtime used by the desktop release build.

The release build must receive a real, reviewable directory.  In particular,
the older development stage used junctions into a previous PyInstaller output,
which makes it unsafe as a build input.  This script copies only the Python
PDAL binding, its selected native runtime, and the explicitly required codec
DLLs.  Optional PDAL plugin DLLs are intentionally recorded as excluded; the
desktop workflow does not use their stages and the available copies have
incomplete native closures.

The destination is an immutable snapshot by convention: it is never replaced
in place.  Re-running the script against an existing destination fails before
touching it.  A manifest is written inside the temporary copy before the copy
is atomically moved into place.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = Path("dist/TreeLineWarning-rc-20260823f/TreeLineWarning/_internal")
DEFAULT_DESTINATION = Path(".build-pdal-runtime-final-20260830")
MANIFEST_NAME = "native_runtime_manifest.json"
_REPARSE_POINT = 0x400
_BINDING_VERSION = re.compile(r"__version__\s*=\s*['\"]([^'\"]+)")


def _is_reparse_point(path: Path) -> bool:
    """Return whether *path* is a symlink or Windows reparse point."""

    if path.is_symlink():
        return True
    try:
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & _REPARSE_POINT)


def _ensure_real_tree(root: Path) -> None:
    if not root.is_dir() or _is_reparse_point(root):
        raise SystemExit(f"native runtime source must be a real directory: {root}")
    for path in root.rglob("*"):
        if _is_reparse_point(path):
            raise SystemExit(f"native runtime source contains a link/reparse point: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pe_metadata(path: Path) -> dict[str, Any]:
    """Return small, optional PE metadata for the file manifest."""

    pe = None
    try:
        import pefile

        pe = pefile.PE(str(path), fast_load=True)
        machine_names = {
            0x014C: "I386",
            0x8664: "AMD64",
            0xAA64: "ARM64",
        }
        return {
            "format": "PE",
            "machine": machine_names.get(pe.FILE_HEADER.Machine, hex(pe.FILE_HEADER.Machine)),
        }
    except Exception:
        return {}
    finally:
        if pe is not None:
            pe.close()


def _copy_files(
    source: Path,
    destination: Path,
    source_root: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    """Copy one real source tree and return manifest rows for every file."""

    rows: list[dict[str, Any]] = []
    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source_root).as_posix()
        target = destination / source_path.relative_to(source)
        if source_path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not source_path.is_file() or _is_reparse_point(source_path):
            raise SystemExit(f"selected native payload is not a regular file: {source_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        rows.append(
            {
                "source_relative": relative,
                "destination_relative": target.relative_to(output_root).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
                **_pe_metadata(target),
            }
        )
    return rows


def _copy_one(
    source: Path,
    destination: Path,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if not source.is_file() or _is_reparse_point(source):
        raise SystemExit(f"required compatibility DLL is not a real file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "source_relative": source.relative_to(source_root).as_posix(),
        "destination_relative": destination.relative_to(output_root).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "license_status": "UNVERIFIED_SOURCE_PROVENANCE",
        **_pe_metadata(destination),
    }


def _runtime_versions(source_root: Path) -> dict[str, Any]:
    """Read the binding/native versions from the selected source in isolation."""

    binding_init = source_root / "pdal" / "__init__.py"
    binding_version = None
    if binding_init.is_file():
        match = _BINDING_VERSION.search(binding_init.read_text(encoding="utf-8"))
        binding_version = match.group(1) if match else None

    versions: dict[str, Any] = {"pdal_python_binding": binding_version}
    runtime = source_root / "pdal_runtime"
    old_path = os.environ.get("PATH", "")
    old_sys_path = list(sys.path)
    handle: Any = None
    try:
        os.environ["PATH"] = os.pathsep.join([str(runtime), old_path]).rstrip(os.pathsep)
        if hasattr(os, "add_dll_directory"):
            handle = os.add_dll_directory(str(runtime))
        sys.path.insert(0, str(source_root))
        import pdal  # type: ignore[import-not-found]

        module_path = Path(pdal.__file__ or "").resolve()
        try:
            module_path.relative_to(source_root / "pdal")
        except ValueError as exc:
            raise RuntimeError(
                f"PDAL binding resolved outside approved source: {module_path}"
            ) from exc
        info = getattr(pdal, "info", None)
        versions["pdal"] = str(getattr(info, "version", "unknown"))
        debug = str(getattr(info, "debug", ""))
        gdal_match = re.search(r"GDAL\s*\(([^)]+)\)", debug)
        versions["gdal"] = gdal_match.group(1) if gdal_match else None
        proj_library = ctypes.CDLL(str(runtime / "proj_9.dll"))

        class _ProjInfo(ctypes.Structure):
            _fields_ = [
                ("major", ctypes.c_int),
                ("minor", ctypes.c_int),
                ("patch", ctypes.c_int),
                ("release", ctypes.c_char_p),
                ("version", ctypes.c_char_p),
                ("searchpath", ctypes.c_char_p),
                ("paths", ctypes.POINTER(ctypes.c_char_p)),
                ("path_count", ctypes.c_size_t),
            ]

        proj_library.proj_info.restype = _ProjInfo
        proj_info = proj_library.proj_info()
        versions["proj"] = proj_info.version.decode("ascii")
    except Exception as exc:
        raise SystemExit(
            "approved source PDAL version probe failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.path[:] = old_sys_path
        os.environ["PATH"] = old_path
        if handle is not None:
            handle.close()
    return versions


def materialize(source: Path, destination: Path) -> Path:
    """Create and verify a new runtime snapshot, returning its manifest path."""

    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise SystemExit(f"refusing to replace existing native runtime snapshot: {destination}")
    if source.name != "_internal" or "TreeLineWarning-rc-20260823f" not in source.as_posix():
        raise SystemExit(
            "approved native runtime source must be the rc-20260823f "
            "TreeLineWarning/_internal directory"
        )
    _ensure_real_tree(source)
    required = (source / "pdal", source / "pdal_runtime")
    missing = [path for path in required if not path.is_dir()]
    if missing:
        raise SystemExit(
            "approved source is missing required PDAL directories: "
            + ", ".join(map(str, missing))
        )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    try:
        pdal_target = temporary / "Lib" / "site-packages" / "pdal"
        runtime_target = temporary / "Library" / "bin"
        plugin_target = temporary / "Lib" / "site-packages" / "bin"
        compat_target = temporary / "compat"
        pdal_target.mkdir(parents=True)
        runtime_target.mkdir(parents=True)
        plugin_target.mkdir(parents=True)
        compat_target.mkdir(parents=True)

        copied = []
        copied.extend(_copy_files(source / "pdal", pdal_target, source, temporary))
        copied.extend(_copy_files(source / "pdal_runtime", runtime_target, source, temporary))

        compatibility = []
        for name in ("openjp2.dll", "zlib-ng2.dll"):
            candidate = source / name
            if candidate.is_file():
                compatibility.append(_copy_one(candidate, compat_target / name, source, temporary))

        excluded_plugins = sorted(
            path.relative_to(source).as_posix()
            for path in (source / "pdal_plugins").glob("*.dll")
        ) if (source / "pdal_plugins").is_dir() else []

        versions = _runtime_versions(source)
        manifest = {
            "schema": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source),
            "source_label": "dist/TreeLineWarning-rc-20260823f/TreeLineWarning/_internal",
            "destination_root": str(destination),
            "snapshot_policy": (
                "new_destination_only; no junctions or symlinks; existing output preserved"
            ),
            "versions": versions,
            "payload": {
                "pdal_binding": "Lib/site-packages/pdal",
                "pdal_native_runtime": "Library/bin",
                "optional_plugin_policy": (
                    "exclude all optional PDAL plugin DLLs; no production stage requires them"
                ),
                "optional_plugins_excluded": excluded_plugins,
                "compatibility_dll_policy": "copied into compat/ with unverified source provenance",
            },
            "files": copied,
            "compatibility_files": compatibility,
        }
        manifest_path = temporary / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        _ensure_real_tree(temporary)
        if destination.exists():
            raise SystemExit(f"refusing to replace existing native runtime snapshot: {destination}")
        os.replace(temporary, destination)
        return destination / MANIFEST_NAME
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = materialize(args.source, args.destination)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
