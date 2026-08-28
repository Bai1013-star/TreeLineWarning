"""Unified V0.1 GIS dispatcher including lightweight Shapefile support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.stage_a.gis import CanonicalGISImport, import_gis as import_standard_gis
from core.stage_a.gis_shapefile import read_shapefile


def import_gis(path: Path | str, **kwargs: Any) -> CanonicalGISImport:
    """Read one supported GIS exchange source without guessing its CRS."""

    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix in {".shp", ".zip"}:
        if kwargs:
            raise ValueError("Shapefile input does not accept CSV/XLSX mapping options")
        return read_shapefile(resolved)
    return import_standard_gis(resolved, **kwargs)


__all__ = ["import_gis", "read_shapefile"]
