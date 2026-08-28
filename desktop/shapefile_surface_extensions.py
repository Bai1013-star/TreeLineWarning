"""Small product-surface patch for Shapefile capability reporting."""

from __future__ import annotations

from functools import wraps
from typing import Any

_INSTALLED = False


def install_shapefile_surface_extensions() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from desktop.workflow import DesktopWorkflow

    original_project_projection = DesktopWorkflow.project_projection

    @wraps(original_project_projection)
    def project_projection_with_shapefile(self: Any) -> dict[str, Any]:
        projection = original_project_projection(self)
        formats = list(projection.get("gis_supported_formats") or [])
        for value in ("SHP", "SHP_ZIP"):
            if value not in formats:
                formats.append(value)
        projection["gis_supported_formats"] = formats
        return projection

    DesktopWorkflow.project_projection = project_projection_with_shapefile
    _INSTALLED = True


__all__ = ["install_shapefile_surface_extensions"]
