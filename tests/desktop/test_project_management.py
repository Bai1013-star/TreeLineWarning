from pathlib import Path

from desktop.project_management import (
    create_empty_project,
    describe_project,
    import_gis_file,
    import_pointcloud,
)


def test_empty_desktop_project_has_no_analysis_or_prediction_state(tmp_path: Path) -> None:
    project = create_empty_project(tmp_path, "线路巡检 A")

    assert project["project_name"] == "线路巡检 A"
    assert project["project_id"] != project["project_name"]
    assert project["pointcloud"] == "NOT_IMPORTED"
    assert project["towers"] == 0
    assert project["spans"] == 0
    assert project["analysis"] == "NOT_STARTED"
    assert project["prediction"] == "NOT_COMPUTED"
    assert (Path(project["root"]) / "desktop-project.json").is_file()


def test_empty_project_records_las_source_without_starting_science(tmp_path: Path) -> None:
    project = create_empty_project(tmp_path, "空项目")
    source = tmp_path / "source.las"
    source.write_bytes(b"LASF")

    imported = import_pointcloud(Path(project["root"]), source)

    assert imported["pointcloud"] == "IMPORTED_PENDING_TOWER_REGISTRATION"
    assert imported["towers"] == 0
    assert imported["spans"] == 0
    assert imported["analysis"] == "NOT_STARTED"
    assert describe_project(Path(project["root"])) == imported


def test_gis_import_builds_route_topology_and_logical_spans(tmp_path: Path) -> None:
    project = create_empty_project(tmp_path, "拓扑项目")
    gis = tmp_path / "route.kml"
    gis.write_text(
        """<kml><Document>
        <Placemark><name>A</name><Point><coordinates>120,30,8</coordinates></Point></Placemark>
        <Placemark><name>B</name><Point><coordinates>120.1,30.1,9</coordinates></Point></Placemark>
        <Placemark><name>Main</name><LineString><coordinates>
        120,30,8 120.1,30.1,9
        </coordinates></LineString></Placemark>
        </Document></kml>""",
        encoding="utf-8",
    )

    imported = import_gis_file(Path(project["root"]), gis)

    assert imported["lines"] == 1
    assert imported["routes"] == 1
    assert imported["spans"] == 1
    assert imported["topology_workflow"] == "LINE_ROUTE_ORDERED_TOWERS_AUTO_SPANS"
    persisted = Path(project["root"]) / "stage-a-project.json"
    assert '"full_source_copy": false' in persisted.read_text(encoding="utf-8")
