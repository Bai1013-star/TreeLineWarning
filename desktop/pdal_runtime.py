"""PDAL runtime configuration and a small packaged-dependency probe.

The probe is deliberately separate from the scientific workflow.  It verifies
that the release can import the bundled PDAL binding and invoke the validated
SMRF + HAG-NN operation on a tiny temporary LAS fixture without touching a
user project or real survey data.
"""

from __future__ import annotations

import importlib.util
import json
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


def dependency_health() -> dict[str, Any]:
    """Return an actionable health report for the local PDAL dependency."""

    runtime_dir, plugin_dirs = _configure_bundled_runtime()
    spec = importlib.util.find_spec("pdal")
    if spec is None:
        return {
            "found": False,
            "source": "NOT_FOUND",
            "minimal_operation_pass": False,
            "error": "Python PDAL binding is unavailable",
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
            operation_pass = (
                processed == output_count > 0 and "HeightAboveGround" in dimension_names
            )
        return {
            "found": True,
            "source": source,
            "minimal_operation_pass": operation_pass,
            "module": str(module_path) if module_path else None,
            "runtime_directory": str(runtime_dir) if runtime_dir else None,
            "plugin_directories": [str(path) for path in plugin_dirs],
            "error": None if operation_pass else "PDAL output contract was incomplete",
        }
    except Exception as exc:  # pragma: no cover - dependency-specific failure details
        return {
            "found": True,
            "source": source,
            "minimal_operation_pass": False,
            "module": str(module_path) if module_path else None,
            "runtime_directory": str(runtime_dir) if runtime_dir else None,
            "plugin_directories": [str(path) for path in plugin_dirs],
            "error": f"{type(exc).__name__}: {exc}",
        }


__all__ = ["bundled_runtime_dir", "dependency_health", "resource_root"]
