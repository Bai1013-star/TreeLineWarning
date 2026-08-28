from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from core.stage_a import Line, Project, Route, Tower, UnitStatus, load_project, save_project
from core.stage_a.models import CoordinateDeclaration, stable_id
from desktop.corridor_topology import rebuild_deferred_gis_topology
from desktop.project_management import create_empty_project, import_gis_file


def _write_single_route_kml(path: Path) -> None:
    path.write_text(
        """<kml><Document>
        <Placemark><name>T109</name><Point><coordinates>120.002,30.000,0</coordinates></Point></Placemark>
        <Placemark><name>T107</name><Point><coordinates>120.000,30.000,0</coordinates></Point></Placemark>
        <Placemark><name>T108</name><Point><coordinates>120.001,30.000,0</coordinates></Point></Placemark>
        <Placemark><name>Main</name><LineString><coordinates>
        119.9995,30.000,0 120.0025,30.000,0
        </coordinates></LineString></Placemark>
        </Document></kml>""",
        encoding="utf-8",
    )


def test_deferred_kml_route_assigns_towers_by_metric_chainage(tmp_path: Path) -> None:
    descriptor = create_empty_project(tmp_path, "Corridor topology")
    root = Path(descriptor["root"])
    source = tmp_path / "route.kml"
    _write_single_route_kml(source)
    import_gis_file(root, source)

    before = load_project(root)
    assert len(before.spans) == 0
    updated, diagnostics = rebuild_deferred_gis_topology(before)

    assert diagnostics["status"] == "ASSIGNED"
    assert diagnostics["assigned_tower_count"] == 3
    assert diagnostics["generated_span_count"] == 2
    ordered = updated.lines[0].ordered_tower_ids
    by_id = {tower.tower_id: tower.display_name for tower in updated.towers}
    assert [by_id[item] for item in ordered] == ["T107", "T108", "T109"]
    assert [(span.start_tower_id, span.end_tower_id) for span in updated.spans] == [
        (ordered[0], ordered[1]),
        (ordered[1], ordered[2]),
    ]
    assert all(span.source_partition["full_source_copy"] is False for span in updated.spans)


def test_rebuilding_same_route_keeps_existing_span_ids(tmp_path: Path) -> None:
    descriptor = create_empty_project(tmp_path, "Stable spans")
    root = Path(descriptor["root"])
    source = tmp_path / "route.kml"
    _write_single_route_kml(source)
    import_gis_file(root, source)

    first, _ = rebuild_deferred_gis_topology(load_project(root))
    first_ids = tuple(span.span_id for span in first.spans)
    save_project(first, root)

    second, diagnostics = rebuild_deferred_gis_topology(load_project(root))
    assert diagnostics["status"] == "ASSIGNED"
    assert tuple(span.span_id for span in second.spans) == first_ids
    assert second.lines[0].ordered_tower_ids == first.lines[0].ordered_tower_ids


def test_close_parallel_routes_block_affected_routes_instead_of_bridging() -> None:
    coordinate = CoordinateDeclaration(
        "EPSG:32649",
        None,
        "m",
        UnitStatus.METRE,
        "GEOREFERENCED",
    )
    source = {
        "path": "C:/corridor/routes.geojson",
        "format": "GEOJSON",
        "sha256": "a" * 64,
    }
    tower = Tower(
        stable_id("tower"),
        "T107",
        "T107",
        (500000.0, 3000005.0, 0.0),
        coordinate,
        "GIS:GEOJSON",
        {
            "gis_source": source,
            "gis_original_position": [500000.0, 3000005.0, 0.0],
            "gis_original_position_crs": coordinate.to_dict(),
        },
    )
    provenance_a = {
        "gis_source": source,
        "external_route_id": "A",
        "route_position_crs": coordinate.to_dict(),
        "tower_membership": "DEFERRED_UNTIL_REGISTRATION",
    }
    provenance_b = {
        "gis_source": source,
        "external_route_id": "B",
        "route_position_crs": coordinate.to_dict(),
        "tower_membership": "DEFERRED_UNTIL_REGISTRATION",
    }
    line_a = Line(
        stable_id("line"),
        "A",
        (),
        ((499900.0, 3000000.0, 0.0), (500100.0, 3000000.0, 0.0)),
        provenance_a,
    )
    line_b = Line(
        stable_id("line"),
        "B",
        (),
        ((499900.0, 3000010.0, 0.0), (500100.0, 3000010.0, 0.0)),
        provenance_b,
    )
    route_a = Route(
        stable_id("route"),
        line_a.line_id,
        "A",
        (),
        line_a.route_geometry,
        source_provenance=provenance_a,
    )
    route_b = Route(
        stable_id("route"),
        line_b.line_id,
        "B",
        (),
        line_b.route_geometry,
        source_provenance=provenance_b,
    )
    project = replace(
        Project.new("Ambiguous"),
        towers=(tower,),
        lines=(line_a, line_b),
        routes=(route_a, route_b),
    )

    updated, diagnostics = rebuild_deferred_gis_topology(
        project,
        maximum_distance_m=50.0,
        ambiguity_margin_m=2.0,
    )
    assert diagnostics["status"] == "REQUIRES_CONFIRMATION"
    assert diagnostics["ambiguous_tower_ids"] == [tower.tower_id]
    assert set(diagnostics["blocked_route_ids"]) == {line_a.line_id, line_b.line_id}
    assert all(line.ordered_tower_ids == () for line in updated.lines)
    assert all(
        line.source_provenance["tower_membership"] == "SPATIAL_ASSIGNMENT_BLOCKED_V1"
        for line in updated.lines
    )
    assert updated.spans == ()


def test_route_assignment_does_not_touch_manual_line() -> None:
    coordinate = CoordinateDeclaration.local_metric()
    tower_a = Tower(
        stable_id("tower"), None, "A", (0.0, 0.0, 0.0), coordinate, "MANUAL_PICK"
    )
    tower_b = Tower(
        stable_id("tower"), None, "B", (10.0, 0.0, 0.0), coordinate, "MANUAL_PICK"
    )
    line = Line(stable_id("line"), "Manual", (tower_a.tower_id, tower_b.tower_id))
    route = Route(stable_id("route"), line.line_id, "Manual", line.ordered_tower_ids)
    project = replace(
        Project.new("Manual"),
        towers=(tower_a, tower_b),
        lines=(line,),
        routes=(route,),
    )

    updated, diagnostics = rebuild_deferred_gis_topology(project)
    assert diagnostics["status"] == "NO_DEFERRED_ROUTES"
    assert updated == project
