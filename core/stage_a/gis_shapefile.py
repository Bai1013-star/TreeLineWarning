"""Lightweight ESRI Shapefile adapter for V0.1 corridor exchange data.

The authoritative enterprise representation may be a geodatabase, but field
handoffs commonly arrive as shapefiles.  To avoid a GDAL dependency, this
adapter uses ``pyshp`` and bundled ``pyproj`` only.  A .zip package may contain
both tower-point and route-polyline shapefiles so they share one source
identity and can be associated safely by the corridor topology layer.
"""

from __future__ import annotations

import hashlib
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from core.stage_a.gis import (
    CanonicalGISImport,
    GISImportReport,
    GISSourceIdentity,
    RouteSeed,
    TowerSeed,
)
from core.stage_a.models import CoordinateDeclaration, LineRole, UnitStatus


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _standalone_identity(path: Path) -> GISSourceIdentity:
    digest = hashlib.sha256()
    found = False
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        sidecar = path.with_suffix(suffix)
        if not sidecar.is_file():
            continue
        found = True
        digest.update(suffix.encode("ascii"))
        digest.update(sidecar.read_bytes())
    if not found:
        raise ValueError("Shapefile bundle is not readable")
    return GISSourceIdentity(str(path.resolve()), "SHP", digest.hexdigest())


def _coordinate_from_prj(path: Path) -> tuple[CoordinateDeclaration, str, tuple[str, ...]]:
    prj = path.with_suffix(".prj")
    if not prj.is_file():
        return (
            CoordinateDeclaration.unresolved(),
            "REQUIRES_CONFIRMATION",
            ("Shapefile has no .prj; CRS was not guessed.",),
        )
    try:
        from pyproj import CRS

        crs = CRS.from_wkt(prj.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        raise ValueError("Shapefile .prj cannot be parsed") from exc
    authority = crs.to_authority()
    crs_name = f"{authority[0]}:{authority[1]}" if authority else crs.to_string()
    unit_name = ""
    if crs.axis_info:
        unit_name = str(crs.axis_info[0].unit_name or "").lower()
    if crs.is_geographic:
        return (
            CoordinateDeclaration(
                crs_name,
                None,
                "degree",
                UnitStatus.REQUIRES_CONFIRMATION,
                "GEOREFERENCED",
            ),
            "KNOWN_GEOGRAPHIC_CRS_UNIT_REQUIRES_CONFIRMATION",
            ("Geographic shapefile coordinates require metric registration before analysis.",),
        )
    if "metre" in unit_name or "meter" in unit_name:
        return (
            CoordinateDeclaration(crs_name, None, "m", UnitStatus.METRE, "GEOREFERENCED"),
            "EXPLICIT_CRS_AND_METRE",
            (),
        )
    if "foot" in unit_name or "feet" in unit_name:
        return (
            CoordinateDeclaration(crs_name, None, "ft", UnitStatus.FOOT, "GEOREFERENCED"),
            "EXPLICIT_CRS_AND_FOOT",
            (),
        )
    return (
        CoordinateDeclaration.unresolved(horizontal_crs=crs_name),
        "REQUIRES_CONFIRMATION",
        ("Shapefile CRS unit is not metre/foot and requires confirmation.",),
    )


def _record_mapping(reader: Any, record: Any) -> dict[str, Any]:
    names = [field[0] for field in reader.fields[1:]]
    result: dict[str, Any] = {}
    for key, value in zip(names, record, strict=True):
        if value is None or isinstance(value, str | int | float | bool):
            result[str(key)] = value
        else:
            result[str(key)] = str(value)
    return result


def _lookup(properties: dict[str, Any], names: tuple[str, ...], fallback: str) -> str:
    lower = {str(key).lower(): value for key, value in properties.items()}
    for name in names:
        value = lower.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return fallback


def _point_xyz(shape: Any) -> tuple[float, float, float]:
    if not shape.points:
        raise ValueError("Shapefile point geometry is empty")
    x, y = shape.points[0][:2]
    z_values = getattr(shape, "z", None)
    z = float(z_values[0]) if z_values else 0.0
    return float(x), float(y), z


def _polyline_parts(shape: Any) -> list[tuple[tuple[float, float, float], ...]]:
    points = list(shape.points)
    if len(points) < 2:
        return []
    starts = list(shape.parts) + [len(points)]
    z_values = list(getattr(shape, "z", []) or [])
    result: list[tuple[tuple[float, float, float], ...]] = []
    for index, start in enumerate(starts[:-1]):
        end = starts[index + 1]
        if end - start < 2:
            continue
        part = tuple(
            (
                float(points[item][0]),
                float(points[item][1]),
                float(z_values[item]) if item < len(z_values) else 0.0,
            )
            for item in range(start, end)
        )
        result.append(part)
    return result


def _read_one(
    path: Path,
    source: GISSourceIdentity,
) -> tuple[list[TowerSeed], list[RouteSeed], list[str], set[str]]:
    try:
        import shapefile
    except ImportError as exc:
        raise ValueError("pyshp is required for Shapefile input") from exc

    coordinate, crs_status, warnings = _coordinate_from_prj(path)
    try:
        reader = shapefile.Reader(str(path), encodingErrors="replace")
    except Exception as exc:
        raise ValueError(f"Shapefile is not readable: {path.name}") from exc

    towers: list[TowerSeed] = []
    routes: list[RouteSeed] = []
    statuses = {crs_status}
    point_types = {
        shapefile.POINT,
        getattr(shapefile, "POINTM", -1),
        getattr(shapefile, "POINTZ", -1),
    }
    line_types = {
        shapefile.POLYLINE,
        getattr(shapefile, "POLYLINEM", -1),
        getattr(shapefile, "POLYLINEZ", -1),
    }
    for row_index, item in enumerate(reader.iterShapeRecords(), 1):
        props = _record_mapping(reader, item.record)
        if item.shape.shapeType in point_types:
            fallback = f"{path.stem}:tower:{row_index}"
            external = _lookup(
                props,
                ("external_asset_id", "tower_id", "asset_id", "pole_id", "structure_id", "name"),
                fallback,
            )
            display = _lookup(props, ("display_name", "name", "label", "tower_id"), external)
            towers.append(TowerSeed(external, display, _point_xyz(item.shape), coordinate, props))
            continue
        if item.shape.shapeType in line_types:
            fallback = f"{path.stem}:route:{row_index}"
            external = _lookup(
                props,
                ("external_route_id", "route_id", "line_id", "circuit_id", "name"),
                fallback,
            )
            display = _lookup(props, ("display_name", "name", "label", "line_id"), external)
            role_text = _lookup(props, ("route_role", "line_role"), "MAIN").upper()
            role = LineRole.BRANCH if role_text == "BRANCH" else LineRole.MAIN
            parent = _lookup(props, ("parent_external_route_id", "parent_line_id"), "") or None
            for part_index, geometry in enumerate(_polyline_parts(item.shape), 1):
                route_id = external if part_index == 1 else f"{external}:part:{part_index}"
                routes.append(
                    RouteSeed(
                        route_id,
                        display if part_index == 1 else f"{display} / part {part_index}",
                        (),
                        geometry,
                        coordinate,
                        role,
                        parent,
                    )
                )
            continue
    reader.close()
    return towers, routes, list(warnings), statuses


def _read_paths(paths: list[Path], source: GISSourceIdentity) -> CanonicalGISImport:
    towers: list[TowerSeed] = []
    routes: list[RouteSeed] = []
    warnings: list[str] = []
    statuses: set[str] = set()
    for path in paths:
        item_towers, item_routes, item_warnings, item_statuses = _read_one(path, source)
        towers.extend(item_towers)
        routes.extend(item_routes)
        warnings.extend(item_warnings)
        statuses.update(item_statuses)
    warnings.append("Route tower membership is deferred until corridor registration.")
    status = next(iter(statuses)) if len(statuses) == 1 else "MIXED_CRS_REQUIRES_CONFIRMATION"
    report = GISImportReport(
        source,
        "shapefile",
        len(towers),
        len(routes),
        status,
        tuple(dict.fromkeys(warnings)),
    )
    return CanonicalGISImport(source, report, tuple(towers), tuple(routes))


def read_shapefile(path: Path | str) -> CanonicalGISImport:
    """Read one .shp or a .zip package containing tower/route shapefile layers."""

    resolved = Path(path).resolve()
    if resolved.suffix.lower() == ".shp":
        if not resolved.is_file():
            raise ValueError("Shapefile does not exist")
        return _read_paths([resolved], _standalone_identity(resolved))
    if resolved.suffix.lower() != ".zip" or not resolved.is_file():
        raise ValueError("Shapefile input must be .shp or a .zip package")
    source = GISSourceIdentity(str(resolved), "SHP_ZIP", _sha256_path(resolved))
    try:
        with tempfile.TemporaryDirectory(prefix="treeline-shp-") as temporary:
            root = Path(temporary)
            with zipfile.ZipFile(resolved) as archive:
                safe_members = [
                    name
                    for name in archive.namelist()
                    if not name.endswith("/")
                    and Path(name).suffix.lower() in {".shp", ".shx", ".dbf", ".prj", ".cpg"}
                    and ".." not in Path(name).parts
                ]
                if not any(Path(name).suffix.lower() == ".shp" for name in safe_members):
                    raise ValueError("Shapefile ZIP contains no .shp layer")
                for name in safe_members:
                    target = root / Path(name).name
                    target.write_bytes(archive.read(name))
            shp_paths = sorted(root.glob("*.shp"))
            return _read_paths(shp_paths, source)
    except zipfile.BadZipFile as exc:
        raise ValueError("Shapefile ZIP is not readable") from exc


__all__ = ["read_shapefile"]
