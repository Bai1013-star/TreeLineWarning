"""Format adapters that emit canonical GIS objects.

Only this module knows how to parse external KML/CSV representations.  The
product model consumes ``TowerSeed`` and ``RouteSeed`` rather than raw rows or
XML nodes.
"""

from __future__ import annotations

import csv
import hashlib
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.stage_a.models import CoordinateDeclaration, LineRole, UnitStatus


@dataclass(frozen=True, slots=True)
class GISSourceIdentity:
    path: str
    format: str
    sha256: str

    def __post_init__(self) -> None:
        if not Path(self.path).is_absolute():
            raise ValueError("GIS source path must be absolute")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("GIS source sha256 must be lowercase SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "format": self.format, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class TowerSeed:
    external_asset_id: str
    display_name: str
    position: tuple[float, float, float]
    position_crs: CoordinateDeclaration
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "external_asset_id": self.external_asset_id,
            "display_name": self.display_name,
            "position": list(self.position),
            "position_crs": self.position_crs.to_dict(),
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class RouteSeed:
    external_route_id: str
    display_name: str
    ordered_external_asset_ids: tuple[str, ...]
    geometry: tuple[tuple[float, float, float], ...] = ()
    position_crs: CoordinateDeclaration | None = None
    route_role: LineRole = LineRole.MAIN
    parent_external_route_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.external_route_id, str) or not self.external_route_id.strip():
            raise ValueError("external_route_id must be a non-empty string")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("display_name must be a non-empty string")
        if type(self.route_role) is not LineRole:
            raise ValueError("route_role must be a LineRole")
        if self.route_role is LineRole.BRANCH and not self.parent_external_route_id:
            raise ValueError("BRANCH routes require a parent_external_route_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "external_route_id": self.external_route_id,
            "display_name": self.display_name,
            "ordered_external_asset_ids": list(self.ordered_external_asset_ids),
            "geometry": [list(item) for item in self.geometry],
            "position_crs": self.position_crs.to_dict() if self.position_crs else None,
            "route_role": self.route_role.value,
            "parent_external_route_id": self.parent_external_route_id,
        }


@dataclass(frozen=True, slots=True)
class GISImportReport:
    source: GISSourceIdentity
    adapter: str
    record_count: int
    route_count: int
    crs_status: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "adapter": self.adapter,
            "record_count": self.record_count,
            "route_count": self.route_count,
            "crs_status": self.crs_status,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class CanonicalGISImport:
    source: GISSourceIdentity
    report: GISImportReport
    towers: tuple[TowerSeed, ...]
    routes: tuple[RouteSeed, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "report": self.report.to_dict(),
            "towers": [tower.to_dict() for tower in self.towers],
            "routes": [route.to_dict() for route in self.routes],
        }


def _source(path: Path, fmt: str) -> GISSourceIdentity:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return GISSourceIdentity(str(path.resolve()), fmt, digest)


def _coords(value: str) -> tuple[float, float, float]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) < 2:
        raise ValueError("GIS coordinate must contain at least x/y or lon/lat")
    return float(parts[0]), float(parts[1]), float(parts[2]) if len(parts) > 2 and parts[2] else 0.0


def _local(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def read_kml(path: Path | str) -> CanonicalGISImport:
    resolved = Path(path).resolve()
    source = _source(resolved, "KML")
    root = ET.fromstring(resolved.read_bytes())
    coordinate = CoordinateDeclaration(
        "EPSG:4326", None, "degree", UnitStatus.REQUIRES_CONFIRMATION, "GEOREFERENCED"
    )
    towers: list[TowerSeed] = []
    routes: list[RouteSeed] = []
    for placemark in root.iter():
        if _local(placemark) != "Placemark":
            continue
        name = next(
            (item.text.strip() for item in placemark if _local(item) == "name" and item.text),
            "GIS object",
        )
        point = next((item for item in placemark.iter() if _local(item) == "Point"), None)
        line = next((item for item in placemark.iter() if _local(item) == "LineString"), None)
        if point is not None:
            coord_node = next(
                (item for item in point.iter() if _local(item) == "coordinates" and item.text), None
            )
            if coord_node is not None:
                position = _coords(coord_node.text.split()[0])
                towers.append(
                    TowerSeed(
                        f"kml:{len(towers) + 1}", name, position, coordinate, {"source_name": name}
                    )
                )
        if line is not None:
            coord_node = next(
                (item for item in line.iter() if _local(item) == "coordinates" and item.text), None
            )
            if coord_node is not None:
                geometry = tuple(_coords(item) for item in coord_node.text.split())
                routes.append(
                    RouteSeed(
                        f"kml-route:{len(routes) + 1}",
                        name,
                        tuple(tower.external_asset_id for tower in towers),
                        geometry,
                        coordinate,
                    )
                )
    report = GISImportReport(
        source,
        "kml",
        len(towers),
        len(routes),
        "KNOWN_GEOGRAPHIC_CRS_UNIT_REQUIRES_CONFIRMATION",
        (
            "KML coordinates are lon/lat degrees; metric registration requires an "
            "explicit CRS transform.",
        ),
    )
    return CanonicalGISImport(source, report, tuple(towers), tuple(routes))


def read_csv(
    path: Path | str,
    *,
    mapping: Mapping[str, str] | None = None,
    crs: str | None = None,
    linear_unit: str | None = None,
) -> CanonicalGISImport:
    """Read tower rows with explicit column/CRS mapping; never infer CRS."""

    resolved = Path(path).resolve()
    source = _source(resolved, "CSV")
    mapping = dict(
        mapping
        or {
            "external_asset_id": "external_asset_id",
            "display_name": "display_name",
            "x": "x",
            "y": "y",
            "z": "z",
        }
    )
    required = {"external_asset_id", "x", "y"}
    if not required.issubset(mapping):
        raise ValueError("CSV mapping must explicitly provide external_asset_id, x, and y")
    if crs and linear_unit in {"m", "metre", "meter"}:
        coordinate = CoordinateDeclaration(crs, None, "m", UnitStatus.METRE, "GEOREFERENCED")
        status = "EXPLICIT_CRS_AND_METRE"
        warnings: tuple[str, ...] = ()
    elif crs and linear_unit in {"ft", "foot", "feet"}:
        coordinate = CoordinateDeclaration(crs, None, "ft", UnitStatus.FOOT, "GEOREFERENCED")
        status = "EXPLICIT_CRS_AND_FOOT"
        warnings = ()
    else:
        coordinate = CoordinateDeclaration.unresolved(horizontal_crs=crs)
        status = "REQUIRES_CONFIRMATION"
        warnings = (
            "CSV CRS and linear unit must be explicitly supplied; numeric ranges "
            "were not interpreted.",
        )
    towers: list[TowerSeed] = []
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 2):
            external = str(row.get(mapping["external_asset_id"], "")).strip()
            if not external:
                raise ValueError(f"CSV row {row_number} has no external_asset_id")
            name = str(row.get(mapping.get("display_name", ""), external)).strip() or external
            try:
                xyz = (
                    float(row[mapping["x"]]),
                    float(row[mapping["y"]]),
                    float(row.get(mapping.get("z", ""), 0.0) or 0.0),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"CSV row {row_number} has invalid coordinates") from exc
            towers.append(TowerSeed(external, name, xyz, coordinate, dict(row)))
    report = GISImportReport(source, "csv", len(towers), 0, status, warnings)
    return CanonicalGISImport(source, report, tuple(towers), ())


def import_gis(path: Path | str, **kwargs: Any) -> CanonicalGISImport:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".kml":
        return read_kml(resolved)
    if suffix == ".csv":
        return read_csv(resolved, **kwargs)
    raise ValueError("unsupported GIS format; Stage A adapters currently support KML and CSV")


__all__ = [
    "CanonicalGISImport",
    "GISImportReport",
    "GISSourceIdentity",
    "RouteSeed",
    "TowerSeed",
    "import_gis",
    "read_csv",
    "read_kml",
]
