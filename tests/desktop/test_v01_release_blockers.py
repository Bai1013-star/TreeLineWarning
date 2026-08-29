from __future__ import annotations

from pathlib import Path


def test_desktop_waits_for_background_finalize_before_reloading_workspace() -> None:
    root = Path(__file__).resolve().parents[2]
    page = (root / "viewer_poc" / "giro3d" / "desktop.html").read_text(encoding="utf-8")

    assert "async function waitForFinalize()" in page
    assert "state.step === 'DONE'" in page
    assert "state.step === 'FINALIZE_FAILED'" in page
    assert "pathname === '/api/desktop/batch/finalize'" in page
    assert page.index("window.fetch = async") < page.index('<script src="/desktop.js"></script>')


def test_desktop_capability_surface_explicitly_lists_optional_inputs() -> None:
    root = Path(__file__).resolve().parents[2]
    page = (root / "viewer_poc" / "giro3d" / "desktop.html").read_text(encoding="utf-8")

    for capability in (
        "GPS_TIME",
        "TRAJECTORY",
        "GNSS",
        "CAMERA",
        "CAMERA_POSE",
        "CALIBRATION",
        "GIS",
    ):
        assert f"{capability}:" in page
    assert "NOT_AVAILABLE: ['不可用'" in page
    assert "REQUIRES_CONFIRMATION: ['待确认'" in page


def test_dependency_health_contract_names_offline_proj_and_runtime_requirements() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "desktop" / "pdal_runtime.py").read_text(encoding="utf-8")

    assert '"pyproj"' in source
    assert '"PROJ"' in source
    assert '"PROJ_NETWORK": "OFF"' in source
    assert '"RUNTIME_NETWORK_REQUIRED": "NO"' in source
    assert 'pyproj.network.set_network_enabled(False)' in source
