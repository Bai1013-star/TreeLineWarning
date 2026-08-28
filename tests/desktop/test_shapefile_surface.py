from __future__ import annotations

from pathlib import Path


def test_desktop_installs_shapefile_capability_surface() -> None:
    root = Path(__file__).resolve().parents[2]
    package = (root / "desktop" / "__init__.py").read_text(encoding="utf-8")
    extension = (root / "desktop" / "shapefile_surface_extensions.py").read_text(
        encoding="utf-8"
    )
    app = (root / "desktop" / "app.py").read_text(encoding="utf-8")

    assert "install_shapefile_surface_extensions()" in package
    assert 'for value in ("SHP", "SHP_ZIP")' in extension
    assert 'projection["gis_supported_formats"] = formats' in extension
    assert "*.shp;*.zip" in app
