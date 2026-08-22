"""Release-build contracts that must hold before PyInstaller runs."""

from __future__ import annotations

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


__all__ = ["validate_pdal_binding_abi"]
