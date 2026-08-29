"""Generate a deterministic synthetic transmission-corridor handoff bundle.

The files mimic common utility/contractor exchange surfaces without pretending
to be data from a real power company.  They are intentionally small so they
can be generated on any development machine and used before real customer data
is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import laspy
import numpy as np
import shapefile
from openpyxl import Workbook
from pyproj import CRS, Transformer

PROJECTED_CRS = "EPSG:32648"
GEOGRAPHIC_CRS = "EPSG:4326"
TOWER_IDS = ("T107", "T108", "T109", "T110")
VOLTAGE_KV = 220


def _projected_towers() -> list[dict[str, object]]:
    offsets = ((0.0, 0.0), (160.0, 12.0), (325.0, 18.0), (490.0, 25.0))
    base_x, base_y = 690_000.0, 3_070_000.0
    return [
        {
            "external_asset_id": tower_id,
            "display_name": tower_id,
            "x": base_x + dx,
            "y": base_y + dy,
            "z": 0.0,
            "voltage_kv": VOLTAGE_KV,
            "circuit_id": "QL-220-A",
            "status": "IN_SERVICE",
        }
        for tower_id, (dx, dy) in zip(TOWER_IDS, offsets, strict=True)
    ]


def _with_lonlat(towers: list[dict[str, object]]) -> list[dict[str, object]]:
    transformer = Transformer.from_crs(PROJECTED_CRS, GEOGRAPHIC_CRS, always_xy=True)
    result = []
    for tower in towers:
        lon, lat = transformer.transform(float(tower["x"]), float(tower["y"]))
        result.append({**tower, "lon": float(lon), "lat": float(lat)})
    return result


def _kml_text(towers: list[dict[str, object]]) -> str:
    placemarks = []
    for tower in towers:
        placemarks.append(
            "<Placemark>"
            f"<name>{tower['display_name']}</name>"
            f"<description>{VOLTAGE_KV} kV · QL-220-A</description>"
            "<Point>"
            f"<coordinates>{tower['lon']:.10f},{tower['lat']:.10f},0</coordinates>"
            "</Point>"
            "</Placemark>"
        )
    route_coords = " ".join(
        f"{tower['lon']:.10f},{tower['lat']:.10f},0" for tower in towers
    )
    placemarks.append(
        "<Placemark><name>QL-220-A</name><LineString><coordinates>"
        f"{route_coords}"
        "</coordinates></LineString></Placemark>"
    )
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<kml xmlns=\"http://www.opengis.net/kml/2.2\"><Document>"
        + "".join(placemarks)
        + "</Document></kml>\n"
    )


def _write_kml_and_kmz(root: Path, towers: list[dict[str, object]]) -> None:
    text = _kml_text(towers)
    (root / "corridor_wgs84.kml").write_text(text, encoding="utf-8")
    with zipfile.ZipFile(root / "corridor_wgs84.kmz", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("doc.kml", text)


def _write_geojson(root: Path, towers: list[dict[str, object]]) -> None:
    features = []
    for tower in towers:
        features.append(
            {
                "type": "Feature",
                "id": tower["external_asset_id"],
                "properties": {
                    "external_asset_id": tower["external_asset_id"],
                    "display_name": tower["display_name"],
                    "voltage_kv": VOLTAGE_KV,
                    "circuit_id": "QL-220-A",
                    "status": "IN_SERVICE",
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [tower["lon"], tower["lat"], 0.0],
                },
            }
        )
    features.append(
        {
            "type": "Feature",
            "id": "QL-220-A",
            "properties": {
                "external_route_id": "QL-220-A",
                "display_name": "QL-220-A 220kV 主线路",
                "voltage_kv": VOLTAGE_KV,
                "route_role": "MAIN",
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[tower["lon"], tower["lat"], 0.0] for tower in towers],
            },
        }
    )
    payload = {"type": "FeatureCollection", "features": features}
    (root / "corridor_wgs84.geojson").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _tabular_rows(towers: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "external_asset_id": tower["external_asset_id"],
            "display_name": tower["display_name"],
            "x": tower["x"],
            "y": tower["y"],
            "z": tower["z"],
            "horizontal_crs": PROJECTED_CRS,
            "linear_unit": "m",
            "voltage_kv": VOLTAGE_KV,
            "circuit_id": "QL-220-A",
            "status": "IN_SERVICE",
        }
        for tower in towers
    ]


def _write_csv_xlsx(root: Path, towers: list[dict[str, object]]) -> None:
    rows = _tabular_rows(towers)
    fields = list(rows[0])
    with (root / "tower_inventory_projected.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Towers"
    sheet.append(fields)
    for row in rows:
        sheet.append([row[field] for field in fields])
    workbook.save(root / "tower_inventory_projected.xlsx")


def _write_shapefile_zip(root: Path, towers: list[dict[str, object]]) -> None:
    staging = root / "_shp_staging"
    staging.mkdir(exist_ok=True)
    tower_base = staging / "towers"
    writer = shapefile.Writer(str(tower_base), shapeType=shapefile.POINTZ)
    writer.field("tower_id", "C", size=32)
    writer.field("name", "C", size=64)
    writer.field("voltage_kv", "N", size=8, decimal=0)
    writer.field("circuit_id", "C", size=32)
    writer.field("status", "C", size=24)
    for tower in towers:
        writer.pointz(float(tower["lon"]), float(tower["lat"]), 0.0)
        writer.record(
            tower["external_asset_id"],
            tower["display_name"],
            VOLTAGE_KV,
            "QL-220-A",
            "IN_SERVICE",
        )
    writer.close()

    route_base = staging / "route"
    route_writer = shapefile.Writer(str(route_base), shapeType=shapefile.POLYLINEZ)
    route_writer.field("line_id", "C", size=32)
    route_writer.field("name", "C", size=64)
    route_writer.field("voltage_kv", "N", size=8, decimal=0)
    route_writer.field("route_role", "C", size=16)
    route_writer.linez(
        [[[float(tower["lon"]), float(tower["lat"]), 0.0, 0.0] for tower in towers]]
    )
    route_writer.record("QL-220-A", "QL-220-A 220kV 主线路", VOLTAGE_KV, "MAIN")
    route_writer.close()

    wkt = CRS.from_epsg(4326).to_wkt()
    for stem in ("towers", "route"):
        (staging / f"{stem}.prj").write_text(wkt, encoding="utf-8")
        (staging / f"{stem}.cpg").write_text("UTF-8", encoding="ascii")

    with zipfile.ZipFile(
        root / "corridor_shapefile_bundle.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(staging.iterdir()):
            archive.write(path, arcname=path.name)
    for path in staging.iterdir():
        path.unlink()
    staging.rmdir()


def _write_synthetic_las(root: Path, towers: list[dict[str, object]]) -> None:
    rng = np.random.default_rng(20260828)
    points: list[tuple[float, float, float, int]] = []

    corridor_x = np.linspace(float(towers[0]["x"]) - 30.0, float(towers[-1]["x"]) + 30.0, 180)
    corridor_y = np.linspace(float(towers[0]["y"]) - 45.0, float(towers[-1]["y"]) + 45.0, 30)
    for x in corridor_x:
        for y in corridor_y:
            z = 0.4 * np.sin((x - corridor_x[0]) / 90.0) + rng.normal(0.0, 0.03)
            points.append((float(x), float(y), float(z), 2))

    for tower in towers:
        cx, cy = float(tower["x"]), float(tower["y"])
        for dx, dy in ((-2.2, -2.2), (-2.2, 2.2), (2.2, -2.2), (2.2, 2.2)):
            for z in np.linspace(0.0, 31.0, 90):
                points.append((cx + dx, cy + dy, float(z), 15))
        for z, half_width in ((18.0, 6.0), (24.0, 8.5), (29.0, 5.0)):
            for x in np.linspace(cx - half_width, cx + half_width, 60):
                points.append((float(x), cy, z, 15))
            for y in np.linspace(cy - half_width * 0.6, cy + half_width * 0.6, 40):
                points.append((cx, float(y), z, 15))

    for fraction in (0.35, 0.62, 0.83):
        cx = float(towers[0]["x"]) + fraction * (
            float(towers[-1]["x"]) - float(towers[0]["x"])
        )
        cy = float(towers[0]["y"]) + fraction * (
            float(towers[-1]["y"]) - float(towers[0]["y"])
        ) + 18.0
        for _ in range(900):
            radius = rng.uniform(0.0, 6.0)
            angle = rng.uniform(0.0, 2.0 * np.pi)
            z = rng.uniform(0.0, 16.0)
            points.append(
                (
                    cx + radius * np.cos(angle),
                    cy + radius * np.sin(angle),
                    float(z),
                    5,
                )
            )

    values = np.asarray(points, dtype=np.float64)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray([0.01, 0.01, 0.01])
    header.offsets = np.floor(values[:, :3].min(axis=0))
    header.add_crs(CRS.from_epsg(32648))
    cloud = laspy.LasData(header)
    cloud.x = values[:, 0]
    cloud.y = values[:, 1]
    cloud.z = values[:, 2]
    cloud.classification = values[:, 3].astype(np.uint8)
    cloud.intensity = np.full(len(values), 1200, dtype=np.uint16)
    cloud.write(root / "synthetic_corridor_epsg32648.las")


def generate_bundle(output: Path) -> Path:
    root = output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    projected = _projected_towers()
    towers = _with_lonlat(projected)
    _write_kml_and_kmz(root, towers)
    _write_geojson(root, towers)
    _write_csv_xlsx(root, towers)
    _write_shapefile_zip(root, towers)
    _write_synthetic_las(root, towers)

    manifest = {
        "schema": "treeline-synthetic-corridor-fixture-v1",
        "synthetic_only": True,
        "projected_crs": PROJECTED_CRS,
        "geographic_crs": GEOGRAPHIC_CRS,
        "line_id": "QL-220-A",
        "voltage_kv": VOLTAGE_KV,
        "expected_tower_order": list(TOWER_IDS),
        "expected_span_count": len(TOWER_IDS) - 1,
        "formats": ["KML", "KMZ", "GEOJSON", "CSV", "XLSX", "SHP_ZIP", "LAS"],
    }
    (root / "fixture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "README.txt").write_text(
        "TreeLineWarning synthetic transmission corridor fixture.\n"
        "This is generated test data, not data from any real utility company.\n"
        "Expected order: T107 -> T108 -> T109 -> T110; expected spans: 3.\n"
        "KML/KMZ/GeoJSON/SHP use WGS84. CSV/XLSX/LAS use EPSG:32648 metres.\n",
        encoding="utf-8",
    )
    return root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic TreeLineWarning corridor inputs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("acceptance_evidence") / "synthetic_corridor_fixture",
    )
    args = parser.parse_args()
    root = generate_bundle(args.output)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
