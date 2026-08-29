from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GIRO = REPO_ROOT / "viewer_poc" / "giro3d"


def test_desktop_loads_corridor_ui_from_local_server_only() -> None:
    page = (GIRO / "desktop.html").read_text(encoding="utf-8")
    ui = (GIRO / "corridor_ui.js").read_text(encoding="utf-8")
    server = (REPO_ROOT / "desktop" / "corridor_server_extensions.py").read_text(
        encoding="utf-8"
    )

    assert '<script src="/corridor-ui.js"></script>' in page
    assert 'path == "/corridor-ui.js"' in server
    assert 'self._static("viewer_poc/giro3d/corridor_ui.js")' in server
    assert "http://" not in ui
    assert "https://" not in ui


def test_corridor_ui_exposes_registration_candidates_and_fail_closed_copy() -> None:
    ui = (GIRO / "corridor_ui.js").read_text(encoding="utf-8")

    for text in (
        "GIS 线路注册与杆塔定位",
        "自动注册坐标",
        "设为锚点",
        "查看候选 A/B/C",
        "人工确认",
        "候选是辅助提示",
        "READY",
        "BLOCKED",
    ):
        assert text in ui
    assert "/api/desktop/corridor/prepare" in ui
    assert "/api/desktop/corridor/anchor" in ui
    assert "/api/desktop/corridor/tower-candidates" in ui
    assert "/api/desktop/corridor/tower-confirm" in ui
    assert "/api/desktop/corridor/batch-ready" in ui


def test_desktop_picker_lists_all_v01_gis_formats() -> None:
    app = (REPO_ROOT / "desktop" / "app.py").read_text(encoding="utf-8")
    assert "GIS (*.kml;*.kmz;*.csv;*.xlsx;*.geojson;*.json;*.shp;*.zip)" in app


def test_corridor_server_exposes_only_loopback_workflow_endpoints() -> None:
    server = (REPO_ROOT / "desktop" / "corridor_server_extensions.py").read_text(
        encoding="utf-8"
    )
    for endpoint in (
        "/api/desktop/corridor/registration",
        "/api/desktop/corridor/topology",
        "/api/desktop/corridor/readiness",
        "/api/desktop/corridor/tower-candidates",
        "/api/desktop/corridor/prepare",
        "/api/desktop/corridor/anchor",
        "/api/desktop/corridor/tower-confirm",
        "/api/desktop/corridor/topology/rebuild",
        "/api/desktop/corridor/batch-ready",
    ):
        assert endpoint in server


def test_corridor_science_gate_requires_registration_and_confirmed_endpoints() -> None:
    corridor = (REPO_ROOT / "desktop" / "corridor_extensions.py").read_text(
        encoding="utf-8"
    )
    topology = (REPO_ROOT / "desktop" / "corridor_topology_extensions.py").read_text(
        encoding="utf-8"
    )
    candidates = (REPO_ROOT / "desktop" / "tower_candidates.py").read_text(
        encoding="utf-8"
    )

    assert "SPAN_ENDPOINT_CONFIRMATION_REQUIRED" in corridor
    assert "CORRIDOR_REGISTRATION_NOT_READY_FOR_SCIENCE" in topology
    assert 'authority: str = "PRESENTATION_ASSISTANCE_ONLY"' in candidates
    assert "GIS_TOPOLOGY_ONLY_NOT_SCIENCE_CONFIRMATION" in (
        REPO_ROOT / "desktop" / "corridor_topology.py"
    ).read_text(encoding="utf-8")


def test_release_spec_collects_offline_pyproj_runtime() -> None:
    spec = (REPO_ROOT / "desktop_build.spec").read_text(encoding="utf-8")
    build_contract = (REPO_ROOT / "desktop" / "build_contract.py").read_text(
        encoding="utf-8"
    )

    assert 'collect_submodules("pyproj")' in spec
    assert 'collect_dynamic_libs("pyproj")' in spec
    assert 'collect_data_files("pyproj")' in spec
    assert "validate_pyproj_runtime()" in spec
    assert 'os.environ["PROJ_NETWORK"] = "OFF"' in build_contract
    assert "pyproj.network.set_network_enabled(False)" in build_contract
