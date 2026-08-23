from pathlib import Path

import pytest
from desktop.build_contract import validate_pdal_binding_abi, validate_pillow_runtime


def test_pdal_binding_abi_accepts_matching_cpython_minor() -> None:
    validate_pdal_binding_abi(
        [Path("pdal/libpdalpython.cp311-win_amd64.pyd")], python_version=(3, 11)
    )


def test_pdal_binding_abi_rejects_mismatched_cpython_minor() -> None:
    with pytest.raises(SystemExit, match=r"expected CPython 3\.12.*cp311"):
        validate_pdal_binding_abi(
            [Path("pdal/libpdalpython.cp311-win_amd64.pyd")], python_version=(3, 12)
        )


def test_pdal_binding_abi_requires_an_extension() -> None:
    with pytest.raises(SystemExit, match="no PDAL extension module"):
        validate_pdal_binding_abi([], python_version=(3, 11))


def test_pillow_runtime_build_contract_accepts_the_active_native_extension() -> None:
    import sys

    validate_pillow_runtime(python_version=sys.version_info[:2])
