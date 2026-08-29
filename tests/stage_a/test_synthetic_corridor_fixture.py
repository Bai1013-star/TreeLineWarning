from __future__ import annotations

import json
from pathlib import Path

import laspy
from core.stage_a import import_gis, read_csv, read_xlsx
from scripts.generate_corridor_fixture_bundle import generate_bundle


def test_synthetic_corridor_bundle_covers_v01_exchange_surfaces(tmp_path: Path) -> None:
    root = generate_bundle(tmp_path / "fixture")

    expected = {
        "corridor_wgs84.kml",
        "corridor_wgs84.kmz",
        "corridor_wgs84.geojson",
        "tower_inventory_projected.csv",
        "tower_inventory_projected.xlsx",
        "corridor_shapefile_bundle.zip",
        "synthetic_corridor_epsg32648.las",
        "fixture_manifest.json",
        "README.txt",
    }
    assert expected.issubset({path.name for path in root.iterdir()})

    expected_towers = ["T107", "T108", "T109", "T110"]
    for name in (
        "corridor_wgs84.kml",
        "corridor_wgs84.kmz",
        "corridor_wgs84.geojson",
        "corridor_shapefile_bundle.zip",
    ):
        imported = import_gis(root / name)
        assert [tower.display_name for tower in imported.towers] == expected_towers
        assert [tower.external_asset_id for tower in imported.towers] == expected_towers
        assert len(imported.routes) == 1
        assert imported.routes[0].external_route_id == "QL-220-A"

    csv_import = read_csv(
        root / "tower_inventory_projected.csv",
        crs="EPSG:32648",
        linear_unit="m",
    )
    xlsx_import = read_xlsx(
        root / "tower_inventory_projected.xlsx",
        crs="EPSG:32648",
        linear_unit="m",
    )
    assert len(csv_import.towers) == 4
    assert len(xlsx_import.towers) == 4
    assert csv_import.towers[0].position_crs.metric_safe
    assert xlsx_import.towers[0].position_crs.metric_safe

    with laspy.open(root / "synthetic_corridor_epsg32648.las") as reader:
        assert reader.header.point_count > 0
        crs = reader.header.parse_crs()
    assert crs is not None
    assert crs.to_epsg() == 32648

    manifest = json.loads((root / "fixture_manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic_only"] is True
    assert manifest["expected_tower_order"] == expected_towers
    assert manifest["expected_span_count"] == 3
