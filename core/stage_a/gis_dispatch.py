"""Unified V0.1 GIS dispatcher including lightweight Shapefile support."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from core.stage_a.gis import CanonicalGISImport, import_gis as import_standard_gis
from core.stage_a.gis_shapefile import read_shapefile


def _normalize_kml_asset_ids(imported: CanonicalGISImport) -> CanonicalGISImport:
    """Preserve unique KML Placemark names as external asset/route identities."""

    if imported.source.format not in {"KML", "KMZ"}:
        return imported
    tower_names = [tower.display_name.strip() for tower in imported.towers]
    route_names = [route.display_name.strip() for route in imported.routes]
    towers_unique = (
        bool(tower_names)
        and len(tower_names) == len(set(tower_names))
        and all(tower_names)
    )
    routes_unique = len(route_names) == len(set(route_names)) and all(route_names)

    towers = imported.towers
    if towers_unique:
        towers = tuple(
            replace(
                tower,
                external_asset_id=tower.display_name.strip(),
                attributes={
                    **dict(tower.attributes),
                    "adapter_generated_external_asset_id": tower.external_asset_id,
                },
            )
            for tower in imported.towers
        )
    routes = imported.routes
    if routes_unique:
        routes = tuple(
            replace(route, external_route_id=route.display_name.strip())
            for route in imported.routes
        )
    return replace(imported, towers=towers, routes=routes)


def import_gis(path: Path | str, **kwargs: Any) -> CanonicalGISImport:
    """Read one supported GIS exchange source without guessing its CRS."""

    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix in {".shp", ".zip"}:
        if kwargs:
            raise ValueError("Shapefile input does not accept CSV/XLSX mapping options")
        return read_shapefile(resolved)
    return _normalize_kml_asset_ids(import_standard_gis(resolved, **kwargs))


__all__ = ["import_gis", "read_shapefile"]
