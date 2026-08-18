from pathlib import Path

from desktop.project_management import create_empty_project, describe_project, import_pointcloud


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
