from pathlib import Path

import pytest
from desktop.build_contract import (
    validate_lazrs_runtime,
    validate_native_proj_data,
    validate_pdal_binding_abi,
    validate_pillow_runtime,
    validate_pyproj_runtime,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_pyproj_runtime_build_contract_is_offline_and_transformable() -> None:
    validate_pyproj_runtime()


def test_lazrs_runtime_build_contract_detects_the_pinned_backend() -> None:
    validate_lazrs_runtime()


def test_native_proj_data_build_contract_matches_staged_database() -> None:
    import os

    snapshot = Path(os.environ.get(
        "TREELINE_TEST_PDAL_PROJ_DATA_ROOT",
        REPO_ROOT / ".build-pdal-proj-data-final-20260830-02",
    ))
    if not snapshot.is_dir():
        pytest.skip("materialized native PROJ data snapshot is not present")
    data_dir = validate_native_proj_data(snapshot)
    assert (data_dir / "proj.db").is_file()


def test_survey_release_spec_does_not_require_prediction_model_assets() -> None:
    spec = (REPO_ROOT / "desktop_build.spec").read_text(encoding="utf-8")

    assert "models/species_growth.json" not in spec
    assert "models/temporal_attention.npz" not in spec


def test_release_spec_limits_viewer_assets_and_optional_pdal_plugins() -> None:
    spec = (REPO_ROOT / "desktop_build.spec").read_text(encoding="utf-8")

    assert '("viewer_poc/giro3d", "viewer_poc/giro3d")' in spec
    assert '("viewer_poc", "viewer_poc")' not in spec
    assert "_PDAL_OPTIONAL_PLUGIN_ALLOWLIST: frozenset[str] = frozenset()" in spec
    assert "TREELINE_PDAL_PROJ_DATA_ROOT" in spec


def test_release_spec_bundles_pinned_optional_gpu_runtime() -> None:
    spec = (REPO_ROOT / "desktop_build.spec").read_text(encoding="utf-8")
    requirements = (REPO_ROOT / "desktop-release-gpu-requirements.txt").read_text(
        encoding="utf-8"
    )

    assert 'collect_submodules("cupy")' in spec
    assert 'collect_submodules("cupy_backends")' in spec
    assert '"graphlib"' in spec
    assert 'collect_dynamic_libs("nvidia")' in spec
    assert 'collect_data_files("nvidia")' in spec
    assert '"cupy"' not in spec[spec.index("optional_excludes = [") :]
    assert '"cupy.cuda.cutensor"' in spec
    assert '"cupy_backends.cuda.libs.cutensor"' in spec
    assert "cupy-cuda13x==14.2.0" in requirements
    assert "nvidia-cuda-runtime==13.3.29" in requirements


def test_packaged_readme_names_the_user_delivery_workbook() -> None:
    release_builder = (REPO_ROOT / "scripts" / "build_desktop_release.py").read_text(
        encoding="utf-8"
    )
    assert "导出树木风险结果" in release_builder
    assert "树木风险结果.xlsx" in release_builder
    assert "23 列，每棵当前保留树一行" in release_builder
    assert "导出预测输入" not in release_builder
