"""PDAL runtime configuration and packaged release dependency probes.

The probes are deliberately separate from the scientific workflow.  They verify
that the release can import the bundled native dependencies, run a tiny validated
SMRF + HAG-NN operation, and perform an explicit network-disabled PROJ transform
without touching a user project or real survey data.
"""

from __future__ import annotations

import ctypes
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import laspy
import numpy as np

_DLL_HANDLES: list[Any] = []
_NATIVE_RUNTIME_PAIR: tuple[Path, Path] | None = None
_NATIVE_PROJ_CONFIGURED = False
_NATIVE_RUNTIME_LOCK = threading.RLock()


def resource_root() -> Path:
    """Return the source root or PyInstaller's internal resource root."""

    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1])).resolve()


def bundled_runtime_dir() -> Path:
    return resource_root() / "pdal_runtime"


def bundled_proj_data_dir() -> Path | None:
    """Return the native PROJ data snapshot when one is bundled/configured."""

    packaged = resource_root() / "pdal_proj_data"
    if (packaged / "proj.db").is_file():
        return packaged
    if getattr(sys, "frozen", False) or getattr(sys, "_MEIPASS", None) is not None:
        return None
    configured = os.environ.get("TREELINE_PDAL_PROJ_DATA_ROOT", "").strip()
    if configured:
        configured_root = Path(configured).resolve()
        candidates = (configured_root / "Library" / "share" / "proj", configured_root)
        for candidate in candidates:
            if (candidate / "proj.db").is_file():
                return candidate
    return None


def _preload_native_sqlite(runtime_dir: Path) -> None:
    """Load the staged SQLite before Python's stdlib extension can load one."""

    sqlite_path = runtime_dir / "sqlite3.dll"
    if not sqlite_path.is_file():
        if getattr(sys, "frozen", False):
            raise RuntimeError(f"bundled native SQLite is unavailable: {sqlite_path}")
        return
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        return
    try:
        _DLL_HANDLES.append(loader(str(sqlite_path)))
    except OSError as exc:
        raise RuntimeError(f"bundled native SQLite failed to load: {sqlite_path}") from exc


def _configure_native_proj_search_path(proj_data_dir: Path, runtime_dir: Path) -> None:
    """Configure UTF-8 resource paths for PROJ and GDAL before importing PDAL."""

    global _NATIVE_RUNTIME_PAIR, _NATIVE_PROJ_CONFIGURED
    proj_data_dir = Path(proj_data_dir).resolve()
    runtime_dir = Path(runtime_dir).resolve()
    pair = (runtime_dir, proj_data_dir)
    with _NATIVE_RUNTIME_LOCK:
        if _NATIVE_RUNTIME_PAIR is not None and _NATIVE_RUNTIME_PAIR != pair:
            raise RuntimeError("native runtime/PROJ data cannot be switched in this process")
        if _NATIVE_PROJ_CONFIGURED:
            return
        database_path = proj_data_dir / "proj.db"
        if not proj_data_dir.is_dir() or not database_path.is_file():
            raise RuntimeError(
                f"bundled native PROJ data is unavailable: {database_path}"
            )
        proj_path = runtime_dir / "proj_9.dll"
        if not proj_path.is_file():
            raise RuntimeError(f"bundled native PROJ library is unavailable: {proj_path}")
        gdal_path = runtime_dir / "gdal.dll"
        if not gdal_path.is_file():
            raise RuntimeError(f"bundled native GDAL library is unavailable: {gdal_path}")
        loader = getattr(ctypes, "CDLL", None)
        if loader is None:
            raise RuntimeError("the runtime cannot load the native PROJ library")
        # Even a partially failed load must not permit switching native runtimes.
        _NATIVE_RUNTIME_PAIR = pair
        try:
            proj = loader(str(proj_path))
            set_search_paths = proj.proj_context_set_search_paths
            set_enable_network = proj.proj_context_set_enable_network
        except (AttributeError, OSError) as exc:
            raise RuntimeError(
                f"bundled native PROJ library does not expose the required API: {proj_path}"
            ) from exc
        set_search_paths.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        set_search_paths.restype = None
        set_enable_network.argtypes = [ctypes.c_void_p, ctypes.c_int]
        set_enable_network.restype = ctypes.c_int
        try:
            gdal = loader(str(gdal_path))
            gdal_set_search_paths = gdal.OSRSetPROJSearchPaths
            gdal_set_enable_network = gdal.OSRSetPROJEnableNetwork
        except (AttributeError, OSError) as exc:
            raise RuntimeError(
                f"bundled native GDAL library does not expose the required API: {gdal_path}"
            ) from exc
        gdal_set_search_paths.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
        gdal_set_search_paths.restype = None
        gdal_set_enable_network.argtypes = [ctypes.c_int]
        gdal_set_enable_network.restype = None
        encoded_path = str(proj_data_dir).encode("utf-8")
        search_paths = (ctypes.c_char_p * 2)(encoded_path, None)
        try:
            # A NULL context updates PROJ's default context.  PROJ documents
            # that subsequently created contexts inherit these search paths.
            set_search_paths(None, 1, search_paths)
            set_enable_network(None, 0)
            # GDAL maintains its own thread-local contexts and otherwise replaces
            # inherited paths with PROJ_DATA read through Windows' narrow getenv.
            # Its API copies a NULL-terminated UTF-8 list for current/future threads.
            gdal_set_search_paths(search_paths)
            gdal_set_enable_network(0)
        except Exception as exc:
            raise RuntimeError(
                f"bundled native PROJ search-path configuration failed: {proj_path}"
            ) from exc
        _DLL_HANDLES.extend((proj, gdal))
        _NATIVE_PROJ_CONFIGURED = True


def _configure_bundled_runtime() -> tuple[Path | None, list[Path]]:
    with _NATIVE_RUNTIME_LOCK:
        return _configure_bundled_runtime_locked()


def _configure_bundled_runtime_locked() -> tuple[Path | None, list[Path]]:
    global _NATIVE_RUNTIME_PAIR
    runtime_dir = bundled_runtime_dir()
    configured_runtime = os.environ.get("TREELINE_PDAL_RUNTIME_ENV", "").strip()
    if (
        not getattr(sys, "frozen", False)
        and getattr(sys, "_MEIPASS", None) is None
        and not runtime_dir.is_dir()
        and configured_runtime
    ):
        runtime_dir = Path(configured_runtime).resolve() / "Library" / "bin"
    proj_data_dir = bundled_proj_data_dir()
    pair = (runtime_dir.resolve(), proj_data_dir.resolve()) if proj_data_dir else None
    if _NATIVE_RUNTIME_PAIR is not None and _NATIVE_RUNTIME_PAIR != pair:
        raise RuntimeError("native runtime/PROJ data cannot be switched in this process")
    os.environ["PROJ_NETWORK"] = "OFF"
    if proj_data_dir is not None:
        # Native PDAL/PROJ consumes this path.  pyproj keeps its wheel data in
        # its own context and reports that path from datadir.get_data_dir().
        os.environ["PROJ_DATA"] = str(proj_data_dir)
    if not runtime_dir.is_dir():
        return None, []

    plugin_dirs = [
        path for path in (resource_root() / "pdal_plugins", runtime_dir) if path.is_dir()
    ]
    if proj_data_dir is None:
        raise RuntimeError("bundled native PROJ data snapshot is unavailable")
    if _NATIVE_PROJ_CONFIGURED:
        return runtime_dir, plugin_dirs
    _NATIVE_RUNTIME_PAIR = pair
    _preload_native_sqlite(runtime_dir)
    os.environ["PATH"] = os.pathsep.join(
        [str(runtime_dir), os.environ.get("PATH", "")]
    ).rstrip(os.pathsep)
    if hasattr(os, "add_dll_directory"):
        _DLL_HANDLES.append(os.add_dll_directory(str(runtime_dir)))
    _configure_native_proj_search_path(proj_data_dir, runtime_dir)
    existing = os.environ.get("PDAL_DRIVER_PATH", "")
    values = [str(path) for path in plugin_dirs]
    if existing:
        values.append(existing)
    os.environ["PDAL_DRIVER_PATH"] = os.pathsep.join(values)
    return runtime_dir, plugin_dirs


def _write_probe_las(path: Path) -> None:
    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 19.0, 20), np.linspace(0.0, 19.0, 20))
    x = grid_x.ravel()
    y = grid_y.ravel()
    z = 1.0 + 0.01 * x + 0.01 * y
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray([0.01, 0.01, 0.01])
    cloud = laspy.LasData(header)
    cloud.x = x
    cloud.y = y
    cloud.z = z
    cloud.classification = np.zeros(len(x), dtype=np.uint8)
    cloud.write(path)


def pillow_health() -> dict[str, Any]:
    """Exercise the bundled Pillow extension without relying on a system install."""

    try:
        import PIL
        import PIL._imaging
        import PIL.Image

        image = PIL.Image.new("RGB", (1, 1), color=(1, 2, 3))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        PIL.Image.open(io.BytesIO(buffer.getvalue())).load()
        module_path = Path(PIL._imaging.__file__).resolve()
        bundled = bool(getattr(sys, "frozen", False)) and resource_root() in module_path.parents
        return {
            "found": True,
            "source": "BUNDLED_LOCAL_RUNTIME" if bundled else "EXTERNAL_RUNTIME",
            "minimal_operation_pass": True,
            "module": str(module_path),
            "pillow_version": PIL.__version__,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - native dependency-specific failure details
        return {
            "found": False,
            "source": "NOT_FOUND",
            "minimal_operation_pass": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _offline_proj_health() -> dict[str, Any]:
    """Return explicit pyproj/PROJ health while enforcing the offline contract."""

    os.environ["PROJ_NETWORK"] = "OFF"
    try:
        import pyproj

        pyproj.network.set_network_enabled(False)
        data_dir = Path(pyproj.datadir.get_data_dir()).resolve()
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326",
            "EPSG:3857",
            always_xy=True,
            allow_ballpark=False,
            only_best=True,
        )
        x, y = transformer.transform(120.0, 30.0, errcheck=True)
        expected_x, expected_y = 13358338.8952, 3503549.8435
        network_enabled = bool(pyproj.network.is_network_enabled())
        operation_pass = (
            data_dir.is_dir()
            and math.isfinite(float(x))
            and math.isfinite(float(y))
            and math.isclose(float(x), expected_x, abs_tol=0.05)
            and math.isclose(float(y), expected_y, abs_tol=0.05)
            and not network_enabled
        )
        return {
            "pyproj": {
                "found": True,
                "version": str(getattr(pyproj, "__version__", "unknown")),
                "minimal_operation_pass": operation_pass,
            },
            "PROJ": {
                "found": data_dir.is_dir(),
                "version": str(getattr(pyproj, "proj_version_str", "unknown")),
                "data_directory": str(data_dir),
                "minimal_operation_pass": operation_pass,
                "transform": {"x": float(x), "y": float(y)},
            },
            "PROJ_NETWORK": "OFF" if not network_enabled else "ON",
            "RUNTIME_NETWORK_REQUIRED": "NO",
            "proj_minimal_operation_pass": operation_pass,
            "proj_error": None if operation_pass else "PROJ offline runtime contract failed",
        }
    except Exception as exc:  # pragma: no cover - dependency-specific failure details
        return {
            "pyproj": {
                "found": False,
                "version": None,
                "minimal_operation_pass": False,
            },
            "PROJ": {
                "found": False,
                "version": None,
                "data_directory": None,
                "minimal_operation_pass": False,
            },
            "PROJ_NETWORK": "OFF",
            "RUNTIME_NETWORK_REQUIRED": "NO",
            "proj_minimal_operation_pass": False,
            "proj_error": f"{type(exc).__name__}: {exc}",
        }


def dependency_health() -> dict[str, Any]:
    """Return an actionable PDAL plus offline pyproj/PROJ release health report."""

    runtime_dir, plugin_dirs = _configure_bundled_runtime()
    proj_health = _offline_proj_health()
    spec = importlib.util.find_spec("pdal")
    if spec is None:
        return {
            "found": False,
            "source": "NOT_FOUND",
            "minimal_operation_pass": False,
            "error": "Python PDAL binding is unavailable",
            **proj_health,
        }

    module_path = Path(spec.origin).resolve() if spec.origin else None
    bundled = (
        bool(getattr(sys, "frozen", False))
        and runtime_dir is not None
        and module_path is not None
        and resource_root() in module_path.parents
    )
    source = "BUNDLED_LOCAL_RUNTIME" if bundled else "EXTERNAL_RUNTIME"
    if not getattr(sys, "frozen", False) and os.environ.get("TREELINE_PDAL_RUNTIME_ENV"):
        source = "EXPLICIT_STAGED_RUNTIME"
    try:
        import pdal

        native_proj_data = bundled_proj_data_dir()
        if getattr(sys, "frozen", False) and (runtime_dir is None or native_proj_data is None):
            raise RuntimeError("frozen native runtime or PROJ data snapshot is unavailable")
        if runtime_dir is not None and native_proj_data is None:
            raise RuntimeError("bundled native PROJ data snapshot is unavailable")
        with tempfile.TemporaryDirectory(prefix="treeline-pdal-probe-") as temporary:
            root = Path(temporary)
            input_path = root / "input.las"
            output_path = root / "output.las"
            epsg_input_path = root / "epsg-input.las"
            epsg_output_path = root / "epsg-output.las"
            laz_path = root / "input.laz"
            laz_output_path = root / "laz-output.las"
            _write_probe_las(input_path)
            source_las = laspy.read(input_path)
            source_point_count = len(source_las.points)
            pipeline = {
                "pipeline": [
                    {"type": "readers.las", "filename": str(input_path)},
                    {
                        "type": "filters.smrf",
                        "cell": 8.0,
                        "slope": 0.2,
                        "threshold": 0.45,
                        "window": 16.0,
                    },
                    {"type": "filters.hag_nn", "count": 8, "allow_extrapolation": True},
                    {
                        "type": "writers.las",
                        "filename": str(output_path),
                        "extra_dims": "all",
                        "forward": "all",
                    },
                ]
            }
            processed = int(pdal.Pipeline(json.dumps(pipeline)).execute())
            with laspy.open(output_path) as reader:
                dimension_names = set(reader.header.point_format.dimension_names)
                output_count = int(reader.header.point_count)
            pdal_pass = processed == output_count > 0 and "HeightAboveGround" in dimension_names

            if native_proj_data is not None:
                epsg_header = laspy.LasHeader(point_format=3, version="1.2")
                epsg_header.scales = np.asarray([0.01, 0.01, 0.01])
                epsg_cloud = laspy.LasData(epsg_header)
                epsg_cloud.x = np.asarray([120.0])
                epsg_cloud.y = np.asarray([30.0])
                epsg_cloud.z = np.asarray([1.0])
                epsg_cloud.classification = np.asarray([0], dtype=np.uint8)
                epsg_cloud.write(epsg_input_path)
                epsg_pipeline = {
                    "pipeline": [
                        {"type": "readers.las", "filename": str(epsg_input_path)},
                        {
                            "type": "filters.reprojection",
                            "in_srs": "EPSG:4326",
                            "out_srs": "EPSG:3857",
                        },
                        {"type": "writers.las", "filename": str(epsg_output_path)},
                    ]
                }
                epsg_processed = int(pdal.Pipeline(json.dumps(epsg_pipeline)).execute())
                with laspy.open(epsg_output_path) as reader:
                    epsg_point = next(reader.chunk_iterator(1))
                    epsg_x = float(epsg_point.x[0])
                    epsg_y = float(epsg_point.y[0])
                epsg_expected = (13358338.8952, 3503549.8435)
                epsg_pass = (
                    epsg_processed == 1
                    and math.isclose(epsg_x, epsg_expected[0], abs_tol=0.05)
                    and math.isclose(epsg_y, epsg_expected[1], abs_tol=0.05)
                )
            else:
                epsg_x = epsg_y = None
                epsg_expected = (13358338.8952, 3503549.8435)
                epsg_pass = None

            source_las.write(laz_path)
            laz_pipeline = {
                "pipeline": [
                    {"type": "readers.las", "filename": str(laz_path)},
                    {
                        "type": "writers.las",
                        "filename": str(laz_output_path),
                        "forward": "all",
                    },
                ]
            }
            laz_processed = int(pdal.Pipeline(json.dumps(laz_pipeline)).execute())
            decoded_laz = laspy.read(laz_path)
            decoded_pdal_laz = laspy.read(laz_output_path)
            with laspy.open(laz_output_path) as reader:
                laz_output_count = int(reader.header.point_count)
            laz_pass = (
                laz_processed == laz_output_count == source_point_count
                and np.array_equal(decoded_laz.X, source_las.X)
                and np.array_equal(decoded_laz.Y, source_las.Y)
                and np.array_equal(decoded_laz.Z, source_las.Z)
                and np.array_equal(decoded_pdal_laz.X, source_las.X)
                and np.array_equal(decoded_pdal_laz.Y, source_las.Y)
                and np.array_equal(decoded_pdal_laz.Z, source_las.Z)
                and np.array_equal(decoded_pdal_laz.intensity, source_las.intensity)
                and np.array_equal(
                    decoded_pdal_laz.classification, source_las.classification
                )
            )
        operation_pass = (
            pdal_pass
            and (epsg_pass is None or epsg_pass)
            and laz_pass
            and bool(proj_health["proj_minimal_operation_pass"])
        )
        return {
            "found": True,
            "source": source,
            "minimal_operation_pass": operation_pass,
            "pdal_minimal_operation_pass": pdal_pass,
            "pdal_epsg_reprojection": {
                "pass": epsg_pass,
                "x": epsg_x,
                "y": epsg_y,
                "expected_x": epsg_expected[0],
                "expected_y": epsg_expected[1],
                "data_directory": str(native_proj_data) if native_proj_data else None,
            },
            "pdal_laz_roundtrip": {
                "pass": laz_pass,
                "processed": laz_processed,
                "output_count": laz_output_count,
            },
            "module": str(module_path) if module_path else None,
            "runtime_directory": str(runtime_dir) if runtime_dir else None,
            "plugin_directories": [str(path) for path in plugin_dirs],
            "error": None if operation_pass else "PDAL dependency health contract failed",
            **proj_health,
        }
    except Exception as exc:  # pragma: no cover - dependency-specific failure details
        return {
            "found": True,
            "source": source,
            "minimal_operation_pass": False,
            "pdal_minimal_operation_pass": False,
            "pdal_epsg_reprojection": {"pass": False},
            "pdal_laz_roundtrip": {"pass": False},
            "module": str(module_path) if module_path else None,
            "runtime_directory": str(runtime_dir) if runtime_dir else None,
            "plugin_directories": [str(path) for path in plugin_dirs],
            "error": f"{type(exc).__name__}: {exc}",
            **proj_health,
        }


__all__ = [
    "bundled_proj_data_dir",
    "bundled_runtime_dir",
    "dependency_health",
    "pillow_health",
    "resource_root",
]
