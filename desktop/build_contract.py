"""Release-build contracts that must hold before PyInstaller runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from importlib.metadata import version
from pathlib import Path

_CPYTHON_TAG = re.compile(r"\.cp(?P<version>\d{2,3})-")
_EXPECTED_LAZRS_VERSION = "0.8.2"
_MIN_NATIVE_PROJ_DATABASE_MINOR = 6


def validate_pdal_binding_abi(
    binding_files: Iterable[Path], *, python_version: tuple[int, int]
) -> None:
    """Reject a PDAL extension built for a different CPython minor version."""

    bindings = tuple(binding_files)
    if not bindings:
        raise SystemExit("TREELINE_PDAL_RUNTIME_ENV contains no PDAL extension module")

    expected_tag = f"{python_version[0]}{python_version[1]}"
    tagged_bindings = []
    incompatible = []
    for binding in bindings:
        match = _CPYTHON_TAG.search(binding.name.lower())
        if match is None:
            continue
        tagged_bindings.append(binding)
        if match.group("version") != expected_tag:
            incompatible.append(binding)

    if tagged_bindings and incompatible:
        names = ", ".join(path.name for path in incompatible)
        raise SystemExit(
            "PDAL binding ABI does not match the PyInstaller interpreter: "
            f"expected CPython {python_version[0]}.{python_version[1]}, found {names}"
        )


def validate_pillow_runtime(*, python_version: tuple[int, int]) -> None:
    """Fail a release build unless Pillow is native-loadable for its interpreter."""

    try:
        import io

        import PIL
        import PIL._imaging
        import PIL.Image

        expected_tag = f"cp{python_version[0]}{python_version[1]}"
        extension_name = Path(PIL._imaging.__file__ or "").name.lower()
        if expected_tag not in extension_name:
            raise RuntimeError(
                "Pillow extension ABI does not match "
                f"CPython {python_version[0]}.{python_version[1]}: "
                f"{extension_name or 'unknown extension'}"
            )
        image = PIL.Image.new("RGB", (1, 1), color=(1, 2, 3))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        PIL.Image.open(io.BytesIO(buffer.getvalue())).load()
    except Exception as exc:
        raise SystemExit(
            f"Pillow native runtime is unusable for the PyInstaller interpreter: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def validate_pyproj_runtime() -> None:
    """Require a local PROJ database and a working network-disabled transform."""

    try:
        import pyproj

        os.environ["PROJ_NETWORK"] = "OFF"
        pyproj.network.set_network_enabled(False)
        data_dir = Path(pyproj.datadir.get_data_dir())
        if not data_dir.is_dir():
            raise RuntimeError(f"PROJ data directory is unavailable: {data_dir}")
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326",
            "EPSG:3857",
            always_xy=True,
            allow_ballpark=False,
            only_best=True,
        )
        x, y = transformer.transform(1.0, 0.0, errcheck=True)
        if not (math.isfinite(float(x)) and math.isfinite(float(y))):
            raise RuntimeError("PROJ smoke transform returned non-finite coordinates")
        if pyproj.network.is_network_enabled():
            raise RuntimeError("PROJ network access must remain disabled")
    except Exception as exc:
        raise SystemExit(
            "pyproj/PROJ offline runtime is unusable for the PyInstaller interpreter: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def validate_lazrs_runtime() -> None:
    """Require the pinned laspy LAZ backend used by desktop workflows."""

    try:
        import laspy
        import lazrs  # noqa: F401

        installed = version("lazrs")
        if installed != _EXPECTED_LAZRS_VERSION:
            raise RuntimeError(
                f"expected lazrs {_EXPECTED_LAZRS_VERSION}, found {installed}"
            )
        available = tuple(laspy.LazBackend.detect_available())
        if not any(backend.name.lower().startswith("lazrs") for backend in available):
            raise RuntimeError(f"laspy detected no lazrs backend: {available!r}")
    except Exception as exc:
        raise SystemExit(
            "laspy/lazrs LAZ backend is unusable for the PyInstaller interpreter: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def validate_native_proj_data(
    root: Path, *, native_runtime_root: Path | None = None
) -> Path:
    """Require the native PROJ data snapshot paired with staged PDAL."""

    root = Path(root).resolve()
    data_dir = root / "Library" / "share" / "proj"
    manifest_path = root / "proj_data_manifest.json"
    database_path = data_dir / "proj.db"
    if not manifest_path.is_file() or not database_path.is_file():
        raise SystemExit(
            "native PROJ data snapshot is incomplete: expected "
            f"{manifest_path} and {database_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"native PROJ data manifest is unreadable: {manifest_path}") from exc
    package = manifest.get("package", {})
    expected_package = {
        "name": "proj",
        "version": "9.8.1",
        "build": "hd30e2cd_0",
        "subdir": "win-64",
    }
    if any(package.get(key) != value for key, value in expected_package.items()):
        raise SystemExit(
            "native PROJ data snapshot has unexpected package metadata: "
            f"{package!r}"
        )
    database_hash = next(
        (
            row.get("sha256")
            for row in manifest.get("files", [])
            if row.get("destination_relative") == "Library/share/proj/proj.db"
        ),
        None,
    )
    if database_hash is None:
        raise SystemExit("native PROJ data manifest does not hash proj.db")
    digest = hashlib.sha256(database_path.read_bytes()).hexdigest()
    if digest.casefold() != str(database_hash).casefold():
        raise SystemExit("native PROJ data proj.db hash does not match its manifest")
    native_sqlite = (
        Path(native_runtime_root).resolve() / "Library" / "bin" / "sqlite3.dll"
        if native_runtime_root is not None
        else None
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import ctypes
import json
import os
import sys
from pathlib import Path

database = Path(sys.argv[1])
sqlite_dll = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
directory_handle = None
if sqlite_dll is not None:
    if hasattr(os, "add_dll_directory"):
        directory_handle = os.add_dll_directory(str(sqlite_dll.parent))
    ctypes.WinDLL(str(sqlite_dll))
import sqlite3

connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
try:
    rows = connection.execute(
        "SELECT key, value FROM metadata "
        "WHERE key IN ('DATABASE.LAYOUT.VERSION.MAJOR', 'DATABASE.LAYOUT.VERSION.MINOR')"
    ).fetchall()
finally:
    connection.close()
print(json.dumps({
    "layout": {str(key): int(value) for key, value in rows},
    "sqlite_version": sqlite3.sqlite_version,
}))
if directory_handle is not None:
    directory_handle.close()
""",
            str(database_path),
            str(native_sqlite) if native_sqlite is not None else "",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if probe.returncode != 0:
        raise SystemExit(
            "native PROJ data database cannot be inspected in an isolated process: "
            f"{probe.stderr.strip()}"
        )
    try:
        probe_payload = json.loads(probe.stdout.strip().splitlines()[-1])
        layout = probe_payload["layout"]
        layout_major = int(layout["DATABASE.LAYOUT.VERSION.MAJOR"])
        layout_minor = int(layout["DATABASE.LAYOUT.VERSION.MINOR"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit("native PROJ data database probe returned invalid metadata") from exc
    if native_sqlite is not None and probe_payload.get("sqlite_version") != "3.53.3":
        raise SystemExit(
            "approved native SQLite version mismatch while reading PROJ data: "
            f"{probe_payload.get('sqlite_version')!r}"
        )
    if layout_major != 1:
        raise SystemExit(f"native PROJ data database major layout is unexpected: {layout_major!r}")
    if layout_minor < _MIN_NATIVE_PROJ_DATABASE_MINOR:
        raise SystemExit(
            "native PROJ data database layout is too old for staged PROJ 9.8.1: "
            f"major={layout_major!r}, minor={layout_minor!r}"
        )
    return data_dir


__all__ = [
    "validate_pdal_binding_abi",
    "validate_lazrs_runtime",
    "validate_native_proj_data",
    "validate_pillow_runtime",
    "validate_pyproj_runtime",
]
