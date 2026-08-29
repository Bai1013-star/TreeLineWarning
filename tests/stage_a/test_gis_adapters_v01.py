from __future__ import annotations

import json
import zipfile
from pathlib import Path

from core.stage_a import import_gis, read_geojson, read_kmz, read_xlsx
from core.stage_a.models import UnitStatus
from openpyxl import Workbook

KML_TEXT = """<kml><Document>
<Placemark><name>A</name><Point><coordinates>120.0,30.0,8</coordinates></Point></Placemark>
<Placemark><name>B</name><Point><coordinates>120.1,30.1,9</coordinates></Point></Placemark>
<Placemark><name>Main Route</name><LineString><coordinates>
120.0,30.0,0 120.1,30.1,0
</coordinates></LineString></Placemark>
</Document></kml>"""


def test_kml_route_membership_is_deferred_instead_of_binding_all_prior_towers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "route.kml"
    path.write_text(KML_TEXT, encoding="utf-8")
    imported = import_gis(path)
    assert len(imported.towers) == 2
    assert len(imported.routes) == 1
    assert imported.routes[0].ordered_external_asset_ids == ()
    assert "deferred" in " ".join(imported.report.warnings).lower()


def test_kmz_reads_embedded_doc_kml_without_extracting_a_workspace_copy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "route.kmz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("nested/other.kml", "<kml><Document/></kml>")
        archive.writestr("doc.kml", KML_TEXT)
    imported = read_kmz(path)
    assert imported.source.format == "KMZ"
    assert imported.report.adapter == "kmz"
    assert len(imported.towers) == 2
    assert len(imported.routes) == 1


def test_xlsx_adapter_uses_existing_openpyxl_and_explicit_crs(tmp_path: Path) -> None:
    path = tmp_path / "towers.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["external_asset_id", "display_name", "x", "y", "z"])
    sheet.append(["T107", "Tower 107", 500000.0, 3000000.0, 15.0])
    sheet.append(["T108", "Tower 108", 500120.0, 3000010.0, 16.0])
    workbook.save(path)
    workbook.close()

    imported = read_xlsx(path, crs="EPSG:32649", linear_unit="m")
    assert imported.source.format == "XLSX"
    assert len(imported.towers) == 2
    assert imported.report.crs_status == "EXPLICIT_CRS_AND_METRE"
    declaration = imported.towers[0].position_crs
    assert declaration.horizontal_crs == "EPSG:32649"
    assert declaration.unit_status is UnitStatus.METRE


def test_geojson_uses_rfc7946_crs_and_defers_route_tower_membership(
    tmp_path: Path,
) -> None:
    path = tmp_path / "route.geojson"
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "tower-107",
                "properties": {
                    "external_asset_id": "T107",
                    "display_name": "Tower 107",
                },
                "geometry": {"type": "Point", "coordinates": [120.0, 30.0, 8.0]},
            },
            {
                "type": "Feature",
                "id": "route-main",
                "properties": {
                    "external_route_id": "R-MAIN",
                    "display_name": "Main Route",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[120.0, 30.0], [120.1, 30.1]],
                },
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    imported = read_geojson(path)
    assert imported.towers[0].external_asset_id == "T107"
    assert imported.towers[0].position_crs.horizontal_crs == "EPSG:4326"
    assert imported.routes[0].external_route_id == "R-MAIN"
    assert imported.routes[0].ordered_external_asset_ids == ()
    assert imported.report.crs_status == "RFC7946_WGS84_UNIT_REQUIRES_CONFIRMATION"


def test_import_gis_dispatches_all_v01_formats(tmp_path: Path) -> None:
    kml = tmp_path / "route.kml"
    kml.write_text(KML_TEXT, encoding="utf-8")
    kmz = tmp_path / "route.kmz"
    with zipfile.ZipFile(kmz, "w") as archive:
        archive.writestr("doc.kml", KML_TEXT)
    geojson = tmp_path / "route.geojson"
    geojson.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}),
        encoding="utf-8",
    )
    assert import_gis(kml).source.format == "KML"
    assert import_gis(kmz).source.format == "KMZ"
    assert import_gis(geojson).source.format == "GEOJSON"
