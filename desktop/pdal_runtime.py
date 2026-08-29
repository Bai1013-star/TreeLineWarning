"""PDAL runtime configuration and packaged release dependency probes.

The probes are deliberately separate from the scientific workflow.  They verify
that the release can import the bundled native dependencies, run a tiny validated
SMRF + HAG-NN operation, and perform an explicit network-disabled PROJ transform
without touching a user project or real survey data.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import laspy
import numpy as np

_DLL_HANDLES: list[Any] = []


def resource_root() -> Path:
    """Return the source root or PyInstaller's internal resource root."""

    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1])).resolve()


def bundled_runtime_dir() -> Path:
    return resource_root() / "pdal_runtime"


def _configure_bundled_runtime() -> tuple[Path | None, list[Path]]:
    runtime_dir = bundled_runtime_dir()
    if not runtime_dir.is_dir():
        return None, []

    plugin_dirs = [
        path for path in (resource_root() / "pdal_plugins", runtime_dir) if path.is_dir()
    ]
    os.environ["PATH"] = os.pathsep.join(
        [str(runtime_dir), os.environ.get("PATH", "")]
    ).rstrip(os.pathsep)
    if hasattr(os, "add_dll_directory"):
        _DLL_HANDLES.append(os.add_dll_directory(str(runtime_dir)))
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
        bundled = resource_root() in module_path.parents
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
        x, y = transformer.transform(1.0, 0.0, errcheck=True)
        network_enabled = bool(pyproj.network.is_network_enabled())
        operation_pass = (
            data_dir.is_dir()
            and math.isfinite(float(x))
            and math.isfinite(float(y))
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
        runtime_dir is not None
        and module_path is not None
        and resource_root() in module_path.parents
    )
    source = "BUNDLED_LOCAL_RUNTIME" if bundled else "EXTERNAL_RUNTIME"
    try:
        import pdal

        with tempfile.TemporaryDirectory(prefix="treeline-pdal-probe-") as temporary:
            root = Path(temporary)
            input_path = root / "input.las"
            output_path = root / "output.las"
            _write_probe_las(input_path)
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
        operation_pass = pdal_pass and bool(proj_health["proj_minimal_operation_pass"])
        return {
            "found": True,
            "source": source,
            "minimal_operation_pass": operation_pass,
            "pdal_minimal_operation_pass": pdal_pass,
            "module": str(module_path) if module_path else None,
            "runtime_directory": str(runtime_dir) if runtime_dir else None,
            "plugin_directories": [str(path) for path in plugin_dirs],
            "error": None if pdal_pass else "PDAL output contract was incomplete",
            **proj_health,
        }
    except Exception as exc:  # pragma: no cover - dependency-specific failure details
        return {
            "found": True,
            "source": source,
            "minimal_operation_pass": False,
            "pdal_minimal_operation_pass": False,
            "module": str(module_path) if module_path else None,
            "runtime_directory": str(runtime_dir) if runtime_dir else None,
            "plugin_directories": [str(path) for path in plugin_dirs],
            "error": f"{type(exc).__name__}: {exc}",
            **proj_health,
        }


__all__ = ["bundled_runtime_dir", "dependency_health", "pillow_health", "resource_root"]
