from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest
from core.stage_a import RUNTIME_NETWORK_REQUIRED, Project

REPO_ROOT = Path(__file__).resolve().parents[2]
CDN_PLOTLY = re.compile(r"include_plotlyjs\s*=\s*['\"]cdn['\"]")


def test_stage_a_declares_fully_offline_runtime_and_rejects_online_basemap() -> None:
    assert RUNTIME_NETWORK_REQUIRED is False
    with pytest.raises(ValueError, match="NO_BASEMAP or OFFLINE_BASEMAP"):
        replace(Project.new("Offline project"), basemap_policy="ONLINE_BASEMAP")


def test_runtime_plotly_outputs_are_self_contained() -> None:
    runtime_roots = (REPO_ROOT / "algorithms", REPO_ROOT / "scripts")
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for root in runtime_roots
        for path in root.rglob("*.py")
        if CDN_PLOTLY.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_viewer_and_gis_runtime_assets_are_local_only() -> None:
    # Giro3D is the current viewer surface. The retired viewer_poc/p3a tree is
    # intentionally absent and must not be required by the offline-runtime gate.
    local_surfaces = (
        REPO_ROOT / "viewer_poc" / "giro3d" / "index.html",
        REPO_ROOT / "viewer_poc" / "giro3d" / "viewer.js",
        REPO_ROOT / "viewer_poc" / "giro3d" / "desktop.html",
        REPO_ROOT / "viewer_poc" / "giro3d" / "corridor_ui.js",
        REPO_ROOT / "algorithms" / "folder_pipeline" / "html.py",
        REPO_ROOT / "algorithms" / "folder_pipeline" / "potree_page.py",
        REPO_ROOT / "algorithms" / "manual_span" / "server.py",
        REPO_ROOT / "scripts" / "run_stem_guided_hybrid_comparison.py",
    )
    forbidden = re.compile(
        r"https?://(?:cdn\.|fonts\.|tile\.|maps\.|api\.mapbox|.*arcgis|.*openstreetmap)",
        re.IGNORECASE,
    )
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in local_surfaces
        if forbidden.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []

    corridor = (REPO_ROOT / "viewer_poc" / "giro3d" / "corridor_ui.js").read_text(
        encoding="utf-8"
    )
    assert "http://" not in corridor
    assert "https://" not in corridor
    assert "/api/desktop/corridor/" in corridor

    bundle = (REPO_ROOT / "viewer_poc" / "giro3d" / "vendor" / "giro3d.bundle.js").read_text(
        encoding="utf-8"
    )
    assert "cdn.jsdelivr.net/npm/laz-perf" not in bundle
    assert 'DEFAULT_LAZPERF_PATH = "/laz-perf/"' in bundle


def test_plotly_and_laz_assets_are_served_from_local_runtime_paths() -> None:
    folder_html = (REPO_ROOT / "algorithms" / "folder_pipeline" / "html.py").read_text(
        encoding="utf-8"
    )
    manual_html = (REPO_ROOT / "algorithms" / "manual_span" / "server.py").read_text(
        encoding="utf-8"
    )
    assert 'src="/plotly.js"' in folder_html
    assert 'src="/plotly.js"' in manual_html
    assert '"lazPerfPath": "/laz-perf"' in (
        REPO_ROOT / "viewer_poc" / "giro3d" / "poc_config.json"
    ).read_text(encoding="utf-8")
