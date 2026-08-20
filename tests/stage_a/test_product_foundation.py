from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import laspy
import numpy as np
import pytest
from core.stage_a import (
    AnalysisRun,
    AnalysisRunStatus,
    CoordinateDeclaration,
    Line,
    LineRole,
    Project,
    RiskPolicy,
    Route,
    Scan,
    SourceDescriptor,
    Span,
    Tower,
    UnitStatus,
    generate_spans_from_ordered_towers,
    generate_spans_from_route,
    import_gis,
    inspect_point_cloud,
    load_project,
    migrate_legacy_project,
    open_project,
    register_one_anchor,
    register_two_anchors,
    save_project,
    to_metric_distance,
)
from core.stage_a.models import stable_id
from core.stage_a.registration import AnchorCorrespondence, RegistrationState
from core.stage_a.shared_source import SharedSourceAccess


def _write_las(path: Path, count: int = 6) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([0.0, 0.0, 0.0])
    data = laspy.LasData(header)
    indexes = np.arange(count, dtype=np.float64)
    data.x = indexes
    data.y = indexes * 0.0
    data.z = indexes * 0.5
    data.write(path)


def test_project_scan_analysis_run_round_trip(tmp_path: Path) -> None:
    source_path = tmp_path / "source.las"
    source_path.write_bytes(b"placeholder")
    source = SourceDescriptor("source-one", str(source_path), "LAS", source_path.stat().st_size)
    scan = Scan(stable_id("scan"), source, str(source_path), {"capabilities": {}})
    tower_a = Tower(
        stable_id("tower"),
        "GIS-A",
        "A",
        (0.0, 0.0, 0.0),
        CoordinateDeclaration.local_metric(),
        "MANUAL",
    )
    tower_b = Tower(
        stable_id("tower"),
        "GIS-B",
        "B",
        (10.0, 0.0, 0.0),
        CoordinateDeclaration.local_metric(),
        "MANUAL",
    )
    line = Line(stable_id("line"), "Line A", (tower_a.tower_id, tower_b.tower_id))
    route = Route(
        stable_id("route"),
        line.line_id,
        "Route A",
        (tower_a.tower_id, tower_b.tower_id),
    )
    span = Span(stable_id("span"), line.line_id, tower_a.tower_id, tower_b.tower_id)
    run = AnalysisRun(
        stable_id("run"),
        scan.scan_id,
        (span.span_id,),
        {"tree": "crown_stem_hybrid"},
        {},
        RiskPolicy().to_dict(),
        AnalysisRunStatus.READY,
    )
    project = replace(
        Project.new("Project A"),
        lines=(line,),
        towers=(tower_a, tower_b),
        spans=(span,),
        routes=(route,),
        scans=(scan,),
        analysis_runs=(run,),
    )
    save_project(project, tmp_path)
    assert load_project(tmp_path).to_dict() == project.to_dict()


def test_capability_inspection_is_header_only_and_unit_safe(tmp_path: Path) -> None:
    path = tmp_path / "scene.las"
    _write_las(path)
    report = inspect_point_cloud(path)
    assert report.has("POINT_CLOUD") and report.has("XYZ")
    assert report.coordinate_declaration.unit_status is UnitStatus.REQUIRES_CONFIRMATION
    assert report.requires_confirmation


def test_shared_source_preparation_is_reused_across_local_span_access(tmp_path: Path) -> None:
    path = tmp_path / "scene.las"
    _write_las(path, 8)
    access = SharedSourceAccess(path, cache_root=tmp_path / "cache", chunk_size=2).prepare()
    first = list(access.iter_span(((0.0, -1.0, -1.0), (2.1, 1.0, 2.0))))
    second = list(access.iter_span(((5.0, -1.0, -1.0), (7.1, 1.0, 5.0))))
    assert sum(len(chunk.xyz) for chunk in first + second) > 0
    assert access.metrics.full_source_read_count == 1
    assert access.metrics.cache_prep_count == 1
    assert access.metrics.span_local_access_count == 2
    assert access.metrics.selected_chunk_read_count >= 2


def test_registration_refuses_one_anchor_as_unique_and_rejects_scale() -> None:
    first = AnchorCorrespondence((0.0, 0.0, 0.0), (10.0, 5.0, 0.0), "A")
    second = AnchorCorrespondence((10.0, 0.0, 0.0), (30.0, 5.0, 0.0), "B")
    assert register_one_anchor(first).state is RegistrationState.ASSISTED_ONE_ANCHOR
    assert not register_one_anchor(first).unique_rigid_transform
    assert register_two_anchors(first, second).state is RegistrationState.REQUIRES_CONFIRMATION


def test_risk_policy_keeps_review_separate_from_risk() -> None:
    policy = RiskPolicy()
    assert policy.classify(clearance_m=0.2, contact=False).risk.value == "WARNING"
    assert policy.classify(clearance_m=2.0, contact=False).risk.value == "SAFE"
    unknown = policy.classify(clearance_m=2.0, contact=None)
    assert unknown.risk.value != "SAFE"
    assert unknown.review.value == "REVIEW_REQUIRED"


def test_csv_adapter_does_not_infer_crs_from_coordinate_magnitude(tmp_path: Path) -> None:
    path = tmp_path / "towers.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("asset", "label", "x", "y"))
        writer.writeheader()
        writer.writerow({"asset": "A", "label": "Tower A", "x": "120", "y": "30"})
    imported = import_gis(
        path, mapping={"external_asset_id": "asset", "display_name": "label", "x": "x", "y": "y"}
    )
    assert imported.report.crs_status == "REQUIRES_CONFIRMATION"
    assert imported.towers[0].position_crs.unit_status is UnitStatus.REQUIRES_CONFIRMATION


def test_kml_adapter_preserves_lon_lat_and_requires_metric_confirmation(tmp_path: Path) -> None:
    path = tmp_path / "route.kml"
    path.write_text(
        """<kml><Document>
        <Placemark><name>A</name><Point><coordinates>120.0,30.0,8.0</coordinates></Point></Placemark>
        <Placemark><name>B</name><Point><coordinates>120.1,30.1,9.0</coordinates></Point></Placemark>
        <Placemark><name>Route</name><LineString><coordinates>
        120.0,30.0,8.0 120.1,30.1,9.0
        </coordinates></LineString></Placemark>
        </Document></kml>""",
        encoding="utf-8",
    )
    imported = import_gis(path)
    assert imported.report.adapter == "kml"
    assert len(imported.towers) == 2 and len(imported.routes) == 1
    assert imported.towers[0].position == (120.0, 30.0, 8.0)
    assert imported.towers[0].position_crs.unit_status is UnitStatus.REQUIRES_CONFIRMATION
    assert imported.routes[0].position_crs is not None


def test_metric_distance_requires_explicit_unit_evidence() -> None:
    with pytest.raises(ValueError, match="unit requires explicit"):
        to_metric_distance(1.0, CoordinateDeclaration.unresolved())
    feet = to_metric_distance(
        1.0, CoordinateDeclaration("EPSG:2263", None, "ft", UnitStatus.FOOT, "GEOREFERENCED")
    )
    assert feet.value_m == pytest.approx(0.3048)
    assessment = RiskPolicy().classify_metric(feet, contact=False)
    assert assessment.clearance_m == pytest.approx(0.3048)


def test_legacy_project_is_not_reinterpreted_without_explicit_migration(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    (legacy_root / "desktop-project.json").write_text(
        json.dumps(
            {"schema_version": "desktop-project-v1", "project_id": "legacy", "project_name": "Old"}
        ),
        encoding="utf-8",
    )
    opened = open_project(legacy_root)
    assert opened.migration_required is True
    assert not (legacy_root / "stage-a-project.json").exists()
    migrated = migrate_legacy_project(legacy_root)
    assert migrated.metadata["compatibility"]["explicit_migration"] is True
    assert load_project(legacy_root).project_id == migrated.project_id


def test_line_route_main_branch_and_ordered_towers_create_logical_spans() -> None:
    tower_ids = ("tower-a", "tower-b", "tower-c")
    main = Line("line-main", "Main line", tower_ids)
    branch = Line(
        "line-branch",
        "Branch line",
        tower_ids[:2],
        line_role=LineRole.BRANCH,
        parent_line_id=main.line_id,
    )
    route = Route(
        "route-main",
        main.line_id,
        "Main route",
        tower_ids,
        route_role=LineRole.MAIN,
    )
    line_spans = generate_spans_from_ordered_towers(
        main, span_id_factory=lambda owner, index: f"span-{index + 1}"
    )
    route_spans = generate_spans_from_route(
        route, span_id_factory=lambda owner, index: f"route-span-{index + 1}"
    )
    assert branch.parent_line_id == main.line_id
    assert [(item.start_tower_id, item.end_tower_id) for item in line_spans] == [
        ("tower-a", "tower-b"),
        ("tower-b", "tower-c"),
    ]
    assert len(route_spans) == 2
    assert all(item.source_partition["kind"] == "LOGICAL" for item in line_spans)
    assert all(
        item.source_partition["access_mode"] == "SHARED_SOURCE_LOCAL" for item in route_spans
    )
    assert all(item.source_partition["full_source_copy"] is False for item in line_spans)
