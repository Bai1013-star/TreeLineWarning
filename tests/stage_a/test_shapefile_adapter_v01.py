from __future__ import annotations

import zipfile
from pathlib import Path

import shapefile
from pyproj import CRS

from core.stage_a.gis_shapefile import read_shapefile


def _write_layers(root: Path, *, include_prj: bool = True) -> tuple[Path, Path]:
    tower_base = root / "towers"
    writer = shapefile.Writer(str(tower_base), shapeType=shapefile.POINTZ)
    writer.field("tower_id", "C", size=32)
    writer.field("name", "C", size=64)
    for index, lon in enumerate((106.90, 106.901, 106.902), 107):
        writer.pointz(lon, 27.70, 0.0)
        writer.record(f"T{index}", f"T{index}")
    writer.close()

    route_base = root / "route"
    route = shapefile.Writer(str(route_base), shapeType=shapefile.POLYLINEZ)
    route.field("line_id", "C", size=32)
    route.field("name", "C", size=64)
    route.linez([[[106.8995, 27.70, 0.0, 0.0], [106.9025, 27.70, 0.0, 0.0]]])
    route.record("QL-220-A", "QL-220-A")
    route.close()

    if include_prj:
        wkt = CRS.from_epsg(4326).to_wkt()
        for stem in ("towers", "route"):
            (root / f"{stem}.prj").write_text(wkt, encoding="utf-8")
    return tower_base.with_suffix(".shp"), route_base.with_suffix(".shp")


def test_shapefile_zip_combines_point_and_line_layers_under_one_source(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_layers(staging)
    package = tmp_path / "utility_corridor.zip"
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in staging.iterdir():
            archive.write(path, path.name)

    imported = read_shapefile(package)

    assert imported.source.format == "SHP_ZIP"
    assert len(imported.source.sha256) == 64
    assert [tower.external_asset_id for tower in imported.towers] == ["T107", "T108", "T109"]
    assert len(imported.routes) == 1
    assert imported.routes[0].external_route_id == "QL-220-A"
    assert imported.routes[0].ordered_external_asset_ids == ()
    assert imported.routes[0].position_crs is not None
    assert imported.routes[0].position_crs.horizontal_crs == "EPSG:4326"
    assert imported.report.route_count == 1
    assert "Route tower membership is deferred" in " ".join(imported.report.warnings)


def test_standalone_shapefile_without_prj_never_guesses_crs(tmp_path: Path) -> None:
    tower_path, _route_path = _write_layers(tmp_path, include_prj=False)

    imported = read_shapefile(tower_path)

    assert imported.report.crs_status == "REQUIRES_CONFIRMATION"
    assert imported.towers[0].position_crs.horizontal_crs is None
    assert imported.towers[0].position_crs.unit_status.value == "REQUIRES_CONFIRMATION"
    assert "CRS was not guessed" in " ".join(imported.report.warnings)
