from __future__ import annotations

import time
from pathlib import Path

import laspy
import numpy as np
import pytest
from pyproj import CRS
from core.stage_a import (
    apply_registration_xy,
    confirm_gis_tower,
    load_project,
    registration_result_from_dict,
    save_project,
    scan_registration_record,
)
from desktop.project_management import create_empty_project, import_gis_file, import_pointcloud
from desktop.workflow import DesktopWorkflow, _coordinate_from_scan


def _write_georeferenced_las(path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray([0.01, 0.01, 0.01])
    header.add_crs(CRS.from_epsg(3857))
    points = []
    for x in (0.0, 11.13195, 22.26390):
        for z in np.linspace(0.0, 18.0, 30):
            points.append((x, 0.0, float(z)))
    cloud = laspy.LasData(header)
    values = np.asarray(points)
    cloud.x = values[:, 0]
    cloud.y = values[:, 1]
    cloud.z = values[:, 2]
    cloud.write(path)


def _write_kml(path: Path) -> None:
    path.write_text(
        """<kml><Document>
        <Placemark><name>T107</name><Point><coordinates>0.0000,0.0,0</coordinates></Point></Placemark>
        <Placemark><name>T108</name><Point><coordinates>0.0001,0.0,0</coordinates></Point></Placemark>
        <Placemark><name>T109</name><Point><coordinates>0.0002,0.0,0</coordinates></Point></Placemark>
        <Placemark><name>Main</name><LineString><coordinates>
        -0.00005,0.0,0 0.00025,0.0,0
        </coordinates></LineString></Placemark>
        </Document></kml>""",
        encoding="utf-8",
    )


def _corridor_project(tmp_path: Path) -> tuple[Path, DesktopWorkflow]:
    descriptor = create_empty_project(tmp_path, "Corridor workflow")
    root = Path(descriptor["root"])
    source = tmp_path / "corridor.las"
    gis = tmp_path / "corridor.kml"
    _write_georeferenced_las(source)
    _write_kml(gis)
    import_pointcloud(root, source)
    import_gis_file(root, gis)
    return root, DesktopWorkflow(root)


def _confirm_towers(
    root: Path,
    workflow: DesktopWorkflow,
    registration: object,
    tower_ids: tuple[str, ...],
) -> None:
    project = load_project(root)
    scan = project.scans[-1]
    coordinate = _coordinate_from_scan(scan)
    for tower_id in tower_ids:
        project = load_project(root)
        tower = next(item for item in project.towers if item.tower_id == tower_id)
        gis_xyz = tower.attributes["gis_original_position"]
        xy = apply_registration_xy(
            registration,
            gis_xyz,
            direction="GIS_TO_POINT_CLOUD",
        )
        save_project(
            confirm_gis_tower(
                project,
                tower_id,
                (xy[0], xy[1], 0.0),
                coordinate,
                evidence={"test": "operator-confirmed"},
            ),
            root,
        )


def test_known_crs_registration_builds_topology_but_keeps_gis_spans_blocked(
    tmp_path: Path,
) -> None:
    root, workflow = _corridor_project(tmp_path)

    registration = workflow.prepare_corridor_registration()
    assert registration["state"] == "GEOREFERENCED"
    assert registration["ready_for_science"] is True

    projection = workflow.project_projection()
    assert projection["spans"] == 2
    assert projection["span_readiness_summary"] == {"ready": 0, "blocked": 2, "total": 2}
    assert projection["corridor_topology"]["authority"] == (
        "GIS_TOPOLOGY_ONLY_NOT_SCIENCE_CONFIRMATION"
    )
    assert all(
        tower["confirmation_state"] == "GIS_SEED_ONLY"
        for tower in projection["tower_records"]
    )
    workspace = workflow.workspace_projection()
    assert workspace["ui_state"] == "TOWER_SELECTION"
    assert workspace["corridor_gate"] == "WAITING_FOR_REGISTERED_CONFIRMED_TOWERS"

    restarted = DesktopWorkflow(root)
    assert restarted.corridor_registration_projection()["state"] == "GEOREFERENCED"
    assert restarted.project_projection()["spans"] == 2
    assert restarted.workspace_projection()["ui_state"] == "TOWER_SELECTION"


def test_real_batch_rejects_gis_topology_without_science_ready_registration(
    tmp_path: Path,
) -> None:
    root, workflow = _corridor_project(tmp_path)
    topology = workflow.rebuild_corridor_topology()
    assert topology["generated_span_count"] == 2

    project = load_project(root)
    span_ids = [span.span_id for span in project.spans]
    # Endpoint readiness is not enough to bypass corridor registration. The
    # real analysis gate is deliberately independent of the fixture path.
    with pytest.raises(ValueError, match="CORRIDOR_REGISTRATION_NOT_READY_FOR_SCIENCE"):
        workflow.start_batch(span_ids, mode="REAL")


def test_ready_only_batch_reports_blocked_spans_instead_of_running_them(
    tmp_path: Path,
) -> None:
    root, workflow = _corridor_project(tmp_path)
    workflow.prepare_corridor_registration()

    project = load_project(root)
    scan = project.scans[-1]
    record = scan_registration_record(scan)
    assert record is not None
    registration = registration_result_from_dict(record["result"])
    ordered = project.lines[0].ordered_tower_ids
    _confirm_towers(root, workflow, registration, ordered[:2])

    readiness = workflow.span_readiness_projection()
    assert [item["status"] for item in readiness] == ["READY", "BLOCKED"]
    assert workflow.workspace_projection()["ui_state"] == "TOPOLOGY_READY"

    started = workflow.start_ready_batch(mode="FIXTURE")
    assert len(started["selected_ready_span_ids"]) == 1
    assert len(started["blocked_span_ids"]) == 1
    for _ in range(100):
        batch = workflow.batch_projection()
        if batch["status"] != "RUNNING":
            break
        time.sleep(0.01)
    assert batch["status"] == "COMPLETE"
    assert batch["duplicate_final_original_index"] == 0
