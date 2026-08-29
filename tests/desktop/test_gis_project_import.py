from __future__ import annotations

from pathlib import Path

import pytest
from core.stage_a import load_project, save_project
from core.stage_a.models import CoordinateDeclaration
from core.stage_a.tower_registration import confirm_gis_tower
from desktop.project_management import create_empty_project, import_gis_file


def _write_csv(path: Path, rows: list[tuple[str, str, float, float, float]]) -> None:
    lines = ["external_asset_id,display_name,x,y,z"]
    lines.extend(
        f"{asset},{name},{x},{y},{z}" for asset, name, x, y, z in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_reimporting_identical_gis_source_is_idempotent(tmp_path: Path) -> None:
    descriptor = create_empty_project(tmp_path, "GIS Project")
    root = Path(descriptor["root"])
    source = tmp_path / "towers.csv"
    _write_csv(
        source,
        [
            ("T107", "Tower 107", 500000.0, 3000000.0, 15.0),
            ("T108", "Tower 108", 500120.0, 3000010.0, 16.0),
        ],
    )

    import_gis_file(root, source, crs="EPSG:32649", linear_unit="m")
    first = load_project(root)
    import_gis_file(root, source, crs="EPSG:32649", linear_unit="m")
    second = load_project(root)

    assert len(first.towers) == len(second.towers) == 2
    assert len(first.lines) == len(second.lines) == 1
    assert len(first.routes) == len(second.routes) == 1
    assert len(first.spans) == len(second.spans) == 1
    assert first.to_dict() == second.to_dict()
    assert len(second.metadata["gis_import_history"]) == 1


def test_changed_same_path_replaces_previous_gis_only_revision(tmp_path: Path) -> None:
    descriptor = create_empty_project(tmp_path, "GIS Revision Project")
    root = Path(descriptor["root"])
    source = tmp_path / "towers.csv"
    _write_csv(
        source,
        [
            ("T107", "Tower 107", 500000.0, 3000000.0, 15.0),
            ("T108", "Tower 108", 500120.0, 3000010.0, 16.0),
        ],
    )
    import_gis_file(root, source, crs="EPSG:32649", linear_unit="m")

    _write_csv(
        source,
        [
            ("T107", "Tower 107", 500010.0, 3000000.0, 15.0),
            ("T108", "Tower 108", 500130.0, 3000010.0, 16.0),
            ("T109", "Tower 109", 500250.0, 3000020.0, 17.0),
        ],
    )
    import_gis_file(root, source, crs="EPSG:32649", linear_unit="m")
    project = load_project(root)

    assert len(project.towers) == 3
    assert len(project.lines) == 1
    assert len(project.routes) == 1
    assert len(project.spans) == 2
    by_external = {tower.external_asset_id: tower for tower in project.towers}
    assert by_external["T107"].position[0] == 500010.0
    assert by_external["T109"].position[0] == 500250.0
    assert len(project.metadata["gis_import_history"]) == 2


def test_kml_route_import_does_not_create_spans_before_registration(tmp_path: Path) -> None:
    descriptor = create_empty_project(tmp_path, "KML Project")
    root = Path(descriptor["root"])
    source = tmp_path / "route.kml"
    source.write_text(
        """<kml><Document>
        <Placemark><name>A</name><Point><coordinates>120,30,8</coordinates></Point></Placemark>
        <Placemark><name>B</name><Point><coordinates>120.1,30.1,9</coordinates></Point></Placemark>
        <Placemark><name>Main</name><LineString><coordinates>
        120,30,0 120.1,30.1,0
        </coordinates></LineString></Placemark>
        </Document></kml>""",
        encoding="utf-8",
    )
    import_gis_file(root, source)
    project = load_project(root)

    assert len(project.towers) == 2
    assert len(project.lines) == 1
    assert len(project.routes) == 1
    assert len(project.spans) == 0
    assert project.lines[0].ordered_tower_ids == ()
    assert project.lines[0].source_provenance["tower_membership"] == (
        "DEFERRED_UNTIL_REGISTRATION"
    )


def test_same_external_id_from_conflicting_source_fails_closed(tmp_path: Path) -> None:
    descriptor = create_empty_project(tmp_path, "GIS Conflict Project")
    root = Path(descriptor["root"])
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_csv(first, [("T107", "Tower 107", 500000.0, 3000000.0, 15.0)])
    _write_csv(second, [("T107", "Tower 107", 500050.0, 3000000.0, 15.0)])
    import_gis_file(root, first, crs="EPSG:32649", linear_unit="m")

    with pytest.raises(ValueError, match="GIS_TOWER_ID_CONFLICT_REQUIRES_CONFIRMATION"):
        import_gis_file(root, second, crs="EPSG:32649", linear_unit="m")

    project = load_project(root)
    assert len(project.towers) == 1
    assert project.towers[0].position[0] == 500000.0


def test_changed_gis_revision_cannot_overwrite_confirmed_tower(tmp_path: Path) -> None:
    descriptor = create_empty_project(tmp_path, "GIS Confirmed Revision")
    root = Path(descriptor["root"])
    source = tmp_path / "towers.csv"
    _write_csv(source, [("T107", "Tower 107", 500000.0, 3000000.0, 15.0)])
    import_gis_file(root, source, crs="EPSG:32649", linear_unit="m")
    project = load_project(root)
    confirmed = confirm_gis_tower(
        project,
        project.towers[0].tower_id,
        (500001.0, 3000001.0, 12.0),
        CoordinateDeclaration(
            "EPSG:32649",
            None,
            "m",
            project.towers[0].position_crs.unit_status,
            "GEOREFERENCED",
        ),
    )
    save_project(confirmed, root)

    _write_csv(source, [("T107", "Tower 107", 500010.0, 3000000.0, 15.0)])
    with pytest.raises(ValueError, match="GIS_SOURCE_REVISION_REQUIRES_CONFIRMATION"):
        import_gis_file(root, source, crs="EPSG:32649", linear_unit="m")
