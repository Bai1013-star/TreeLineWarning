"""Format adapters that emit canonical GIS objects.

Only this module knows how to parse external GIS representations. Product
logic consumes ``TowerSeed`` and ``RouteSeed`` rather than raw rows, workbooks,
JSON features, ZIP members, or XML nodes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping, Sequence
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
        if len(self.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.sha256
        ):
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

    def __post_init__(self) -> None:
        tower_ids = [tower.external_asset_id for tower in self.towers]
        route_ids = [route.external_route_id for route in self.routes]
        if len(tower_ids) != len(set(tower_ids)):
            raise ValueError("GIS import contains duplicate external tower IDs")
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("GIS import contains duplicate external route IDs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "report": self.report.to_dict(),
            "towers": [tower.to_dict() for tower in self.towers],
            "routes": [route.to_dict() for route in self.routes],
        }


def _source(path: Path, fmt: str, *, payload: bytes | None = None) -> GISSourceIdentity:
    data = path.read_bytes() if payload is None else payload
    return GISSourceIdentity(str(path.resolve()), fmt, hashlib.sha256(data).hexdigest())


def _coords(value: str) -> tuple[float, float, float]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) < 2:
        raise ValueError("GIS coordinate must contain at least x/y or lon/lat")
    return (
        float(parts[0]),
        float(parts[1]),
        float(parts[2]) if len(parts) > 2 and parts[2] else 0.0,
    )


def _coordinate_values(value: Sequence[Any]) -> tuple[float, float, float]:
    if len(value) < 2:
        raise ValueError("GIS coordinate must contain at least two values")
    return (
        float(value[0]),
        float(value[1]),
        float(value[2]) if len(value) > 2 and value[2] is not None else 0.0,
    )


def _local(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _geographic_coordinate() -> CoordinateDeclaration:
    return CoordinateDeclaration(
        "EPSG:4326",
        None,
        "degree",
        UnitStatus.REQUIRES_CONFIRMATION,
        "GEOREFERENCED",
    )


def _declared_coordinate(
    crs: str | None,
    linear_unit: str | None,
) -> tuple[CoordinateDeclaration, str, tuple[str, ...]]:
    unit = str(linear_unit or "").strip().lower()
    if crs and unit in {"m", "metre", "meter"}:
        return (
            CoordinateDeclaration(crs, None, "m", UnitStatus.METRE, "GEOREFERENCED"),
            "EXPLICIT_CRS_AND_METRE",
            (),
        )
    if crs and unit in {"ft", "foot", "feet"}:
        return (
            CoordinateDeclaration(crs, None, "ft", UnitStatus.FOOT, "GEOREFERENCED"),
            "EXPLICIT_CRS_AND_FOOT",
            (),
        )
    return (
        CoordinateDeclaration.unresolved(horizontal_crs=crs),
        "REQUIRES_CONFIRMATION",
        (
            "CRS and linear unit must be explicitly supplied; numeric ranges "
            "were not interpreted.",
        ),
    )


def _safe_attributes(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        name = str(key)
        if value is None or isinstance(value, str | int | float | bool):
            result[name] = value
        else:
            result[name] = str(value)
    return result


def _default_mapping() -> dict[str, str]:
    return {
        "external_asset_id": "external_asset_id",
        "display_name": "display_name",
        "x": "x",
        "y": "y",
        "z": "z",
    }


def _tower_from_row(
    row: Mapping[str, Any],
    row_number: int,
    mapping: Mapping[str, str],
    coordinate: CoordinateDeclaration,
) -> TowerSeed:
    external_column = mapping["external_asset_id"]
    external = str(row.get(external_column, "") or "").strip()
    if not external:
        raise ValueError(f"GIS row {row_number} has no external_asset_id")
    display_column = mapping.get("display_name")
    display_value = row.get(display_column) if display_column else None
    name = str(display_value or external).strip() or external
    z_column = mapping.get("z")
    try:
        xyz = (
            float(row[mapping["x"]]),
            float(row[mapping["y"]]),
            float(row.get(z_column, 0.0) or 0.0) if z_column else 0.0,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"GIS row {row_number} has invalid coordinates") from exc
    return TowerSeed(external, name, xyz, coordinate, _safe_attributes(row))


def _validate_mapping(mapping: Mapping[str, str]) -> dict[str, str]:
    result = {str(key): str(value) for key, value in mapping.items()}
    required = {"external_asset_id", "x", "y"}
    if not required.issubset(result):
        raise ValueError("GIS mapping must explicitly provide external_asset_id, x, and y")
    return result


def _kml_name(placemark: ET.Element, default: str) -> str:
    return next(
        (
            item.text.strip()
            for item in placemark
            if _local(item) == "name" and item.text and item.text.strip()
        ),
        default,
    )


def _parse_kml(
    source: GISSourceIdentity,
    payload: bytes,
    *,
    adapter: str,
) -> CanonicalGISImport:
    root = ET.fromstring(payload)
    coordinate = _geographic_coordinate()
    towers: list[TowerSeed] = []
    routes: list[RouteSeed] = []
    for placemark in root.iter():
        if _local(placemark) != "Placemark":
            continue
        name = _kml_name(placemark, "GIS object")
        point = next((item for item in placemark.iter() if _local(item) == "Point"), None)
        line = next(
            (item for item in placemark.iter() if _local(item) == "LineString"),
            None,
        )
        if point is not None:
            coord_node = next(
                (
                    item
                    for item in point.iter()
                    if _local(item) == "coordinates" and item.text
                ),
                None,
            )
            if coord_node is not None:
                external = f"kml:{len(towers) + 1}"
                towers.append(
                    TowerSeed(
                        external,
                        name,
                        _coords(coord_node.text.split()[0]),
                        coordinate,
                        {"source_name": name},
                    )
                )
        if line is not None:
            coord_node = next(
                (
                    item
                    for item in line.iter()
                    if _local(item) == "coordinates" and item.text
                ),
                None,
            )
            if coord_node is not None:
                geometry = tuple(_coords(item) for item in coord_node.text.split())
                routes.append(
                    RouteSeed(
                        f"kml-route:{len(routes) + 1}",
                        name,
                        (),
                        geometry,
                        coordinate,
                    )
                )
    report = GISImportReport(
        source,
        adapter,
        len(towers),
        len(routes),
        "KNOWN_GEOGRAPHIC_CRS_UNIT_REQUIRES_CONFIRMATION",
        (
            "KML/KMZ coordinates are lon/lat degrees; metric registration requires "
            "an explicit CRS transform.",
            "Route tower membership is intentionally deferred until corridor registration.",
        ),
    )
    return CanonicalGISImport(source, report, tuple(towers), tuple(routes))


def read_kml(path: Path | str) -> CanonicalGISImport:
    resolved = Path(path).resolve()
    return _parse_kml(_source(resolved, "KML"), resolved.read_bytes(), adapter="kml")


def read_kmz(path: Path | str) -> CanonicalGISImport:
    resolved = Path(path).resolve()
    source = _source(resolved, "KMZ")
    try:
        with zipfile.ZipFile(resolved) as archive:
            candidates = sorted(
                name
                for name in archive.namelist()
                if not name.endswith("/") and name.lower().endswith(".kml")
            )
            if not candidates:
                raise ValueError("KMZ archive contains no KML document")
            selected = next(
                (name for name in candidates if Path(name).name.lower() == "doc.kml"),
                candidates[0],
            )
            payload = archive.read(selected)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("KMZ archive is not readable") from exc
    return _parse_kml(source, payload, adapter="kmz")


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
    columns = _validate_mapping(mapping or _default_mapping())
    coordinate, status, warnings = _declared_coordinate(crs, linear_unit)
    towers: list[TowerSeed] = []
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 2):
            towers.append(_tower_from_row(row, row_number, columns, coordinate))
    report = GISImportReport(source, "csv", len(towers), 0, status, warnings)
    return CanonicalGISImport(source, report, tuple(towers), ())


def read_xlsx(
    path: Path | str,
    *,
    mapping: Mapping[str, str] | None = None,
    crs: str | None = None,
    linear_unit: str | None = None,
    sheet_name: str | None = None,
) -> CanonicalGISImport:
    """Read tower rows from XLSX without adding a pandas/GDAL dependency."""

    from openpyxl import load_workbook

    resolved = Path(path).resolve()
    source = _source(resolved, "XLSX")
    columns = _validate_mapping(mapping or _default_mapping())
    coordinate, status, warnings = _declared_coordinate(crs, linear_unit)
    workbook = load_workbook(resolved, read_only=True, data_only=True)
    try:
        if sheet_name is not None:
            if sheet_name not in workbook.sheetnames:
                raise ValueError(f"XLSX worksheet not found: {sheet_name}")
            sheet = workbook[sheet_name]
        else:
            sheet = workbook[workbook.sheetnames[0]]
        rows = sheet.iter_rows(values_only=True)
        try:
            raw_headers = next(rows)
        except StopIteration as exc:
            raise ValueError("XLSX worksheet is empty") from exc
        headers = tuple(str(value or "").strip() for value in raw_headers)
        nonempty = tuple(item for item in headers if item)
        if len(nonempty) != len(set(nonempty)):
            raise ValueError("XLSX header contains duplicate column names")
        towers = []
        for row_number, values in enumerate(rows, 2):
            row = {header: value for header, value in zip(headers, values, strict=False) if header}
            if not any(value is not None and str(value).strip() for value in row.values()):
                continue
            towers.append(_tower_from_row(row, row_number, columns, coordinate))
    finally:
        workbook.close()
    report = GISImportReport(source, "xlsx", len(towers), 0, status, warnings)
    return CanonicalGISImport(source, report, tuple(towers), ())


def _geojson_coordinate(
    payload: Mapping[str, Any],
) -> tuple[CoordinateDeclaration, str, tuple[str, ...]]:
    raw_crs = payload.get("crs")
    if isinstance(raw_crs, Mapping):
        properties = raw_crs.get("properties")
        name = properties.get("name") if isinstance(properties, Mapping) else None
        if isinstance(name, str) and name.strip():
            return (
                CoordinateDeclaration.unresolved(horizontal_crs=name.strip()),
                "EXPLICIT_GEOJSON_CRS_REQUIRES_UNIT_CONFIRMATION",
                ("Legacy GeoJSON CRS member requires explicit linear-unit confirmation.",),
            )
    return (
        _geographic_coordinate(),
        "RFC7946_WGS84_UNIT_REQUIRES_CONFIRMATION",
        (
            "GeoJSON without a legacy CRS member is treated as RFC 7946 WGS84 lon/lat; "
            "metric registration is still required.",
        ),
    )


def _feature_id(
    feature: Mapping[str, Any],
    properties: Mapping[str, Any],
    keys: Sequence[str],
    fallback: str,
) -> str:
    for key in keys:
        value = properties.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    value = feature.get("id")
    return str(value).strip() if value is not None and str(value).strip() else fallback


def _feature_name(properties: Mapping[str, Any], fallback: str) -> str:
    for key in ("display_name", "name", "label"):
        value = properties.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return fallback


def _route_role(properties: Mapping[str, Any]) -> tuple[LineRole, str | None]:
    value = str(properties.get("route_role") or LineRole.MAIN.value).upper()
    try:
        role = LineRole(value)
    except ValueError as exc:
        raise ValueError(f"unsupported GeoJSON route_role: {value}") from exc
    parent_value = properties.get("parent_external_route_id")
    parent = str(parent_value).strip() if parent_value is not None else None
    if parent == "":
        parent = None
    return role, parent


def read_geojson(path: Path | str) -> CanonicalGISImport:
    resolved = Path(path).resolve()
    source = _source(resolved, "GEOJSON")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("GeoJSON document is not readable") from exc
    if not isinstance(payload, Mapping) or payload.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON root must be a FeatureCollection")
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError("GeoJSON FeatureCollection requires a features array")
    coordinate, status, base_warnings = _geojson_coordinate(payload)
    towers: list[TowerSeed] = []
    routes: list[RouteSeed] = []
    unsupported = 0
    for feature in features:
        if not isinstance(feature, Mapping) or feature.get("type") != "Feature":
            unsupported += 1
            continue
        properties = feature.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        geometry = feature.get("geometry")
        if not isinstance(geometry, Mapping):
            unsupported += 1
            continue
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if geometry_type == "Point" and isinstance(coordinates, Sequence):
            external = _feature_id(
                feature,
                properties,
                ("external_asset_id", "tower_id", "asset_id"),
                f"geojson:{len(towers) + 1}",
            )
            towers.append(
                TowerSeed(
                    external,
                    _feature_name(properties, external),
                    _coordinate_values(coordinates),
                    coordinate,
                    _safe_attributes(properties),
                )
            )
        elif geometry_type == "LineString" and isinstance(coordinates, Sequence):
            external = _feature_id(
                feature,
                properties,
                ("external_route_id", "route_id", "line_id"),
                f"geojson-route:{len(routes) + 1}",
            )
            role, parent = _route_role(properties)
            routes.append(
                RouteSeed(
                    external,
                    _feature_name(properties, external),
                    (),
                    tuple(_coordinate_values(item) for item in coordinates),
                    coordinate,
                    role,
                    parent,
                )
            )
        else:
            unsupported += 1
    warnings = list(base_warnings)
    warnings.append("Route tower membership is deferred until corridor registration.")
    if unsupported:
        warnings.append(f"Ignored {unsupported} unsupported/non-feature GeoJSON records.")
    report = GISImportReport(
        source,
        "geojson",
        len(towers),
        len(routes),
        status,
        tuple(warnings),
    )
    return CanonicalGISImport(source, report, tuple(towers), tuple(routes))


def import_gis(path: Path | str, **kwargs: Any) -> CanonicalGISImport:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".kml":
        return read_kml(resolved)
    if suffix == ".kmz":
        return read_kmz(resolved)
    if suffix == ".csv":
        return read_csv(resolved, **kwargs)
    if suffix == ".xlsx":
        return read_xlsx(resolved, **kwargs)
    if suffix in {".geojson", ".json"}:
        return read_geojson(resolved)
    raise ValueError(
        "unsupported GIS format; V0.1 adapters support KML, KMZ, CSV, XLSX, and GeoJSON"
    )


__all__ = [
    "CanonicalGISImport",
    "GISImportReport",
    "GISSourceIdentity",
    "RouteSeed",
    "TowerSeed",
    "import_gis",
    "read_csv",
    "read_geojson",
    "read_kml",
    "read_kmz",
    "read_xlsx",
]
