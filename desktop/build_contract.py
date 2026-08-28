"""Release-build contracts that must hold before PyInstaller runs."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable
from pathlib import Path

_CPYTHON_TAG = re.compile(r"\.cp(?P<version>\d{2,3})-")


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


__all__ = [
    "validate_pdal_binding_abi",
    "validate_pillow_runtime",
    "validate_pyproj_runtime",
]
