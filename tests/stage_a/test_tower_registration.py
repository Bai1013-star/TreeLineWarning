from __future__ import annotations

from dataclasses import replace

import pytest
from core.stage_a.models import CoordinateDeclaration, Project, Tower, stable_id
from core.stage_a.tower_registration import (
    GIS_CONFIRMED_POSITION_SOURCE,
    confirm_gis_tower,
    confirmed_tower_ids,
    is_tower_confirmed,
)


def _gis_tower() -> Tower:
    return Tower(
        stable_id("tower"),
        "T107",
        "Tower 107",
        (120.0, 30.0, 8.0),
        CoordinateDeclaration.unresolved(horizontal_crs="EPSG:4326"),
        "GIS:KML",
        {
            "gis_source": {
                "path": "C:/survey/route.kml",
                "format": "KML",
                "sha256": "a" * 64,
            }
        },
    )


def test_confirm_gis_tower_preserves_identity_and_original_gis_position() -> None:
    tower = _gis_tower()
    project = replace(Project.new("Tower confirm"), towers=(tower,))
    point_cloud_crs = CoordinateDeclaration.local_metric()

    updated = confirm_gis_tower(
        project,
        tower.tower_id,
        (1000.0, 2000.0, 12.0),
        point_cloud_crs,
        evidence={"candidate_id": "candidate-1", "raw_local_refined": True},
    )
    confirmed = updated.towers[0]

    assert confirmed.tower_id == tower.tower_id
    assert confirmed.external_asset_id == "T107"
    assert confirmed.display_name == "Tower 107"
    assert confirmed.position == (1000.0, 2000.0, 12.0)
    assert confirmed.position_source == GIS_CONFIRMED_POSITION_SOURCE
    assert confirmed.attributes["gis_original_position"] == [120.0, 30.0, 8.0]
    assert confirmed.attributes["tower_confirmation_v1"]["evidence"]["candidate_id"] == (
        "candidate-1"
    )
    assert is_tower_confirmed(confirmed)
    assert confirmed.tower_id in confirmed_tower_ids(updated)


def test_reconfirm_preserves_first_original_gis_position() -> None:
    tower = _gis_tower()
    project = replace(Project.new("Tower reconfirm"), towers=(tower,))
    crs = CoordinateDeclaration.local_metric()
    first = confirm_gis_tower(project, tower.tower_id, (10.0, 20.0, 3.0), crs)
    second = confirm_gis_tower(first, tower.tower_id, (11.0, 21.0, 3.5), crs)
    confirmed = second.towers[0]
    assert confirmed.attributes["gis_original_position"] == [120.0, 30.0, 8.0]
    assert confirmed.position == (11.0, 21.0, 3.5)


def test_gis_confirmation_rejects_non_gis_tower_and_missing_tower() -> None:
    manual = Tower(
        stable_id("tower"),
        None,
        "Manual",
        (0.0, 0.0, 0.0),
        CoordinateDeclaration.local_metric(),
        "MANUAL_PICK",
    )
    project = replace(Project.new("Manual"), towers=(manual,))
    with pytest.raises(ValueError, match="GIS_TOWER_EXTERNAL_ID_REQUIRED"):
        confirm_gis_tower(
            project,
            manual.tower_id,
            (1.0, 1.0, 1.0),
            CoordinateDeclaration.local_metric(),
        )
    with pytest.raises(ValueError, match="TOWER_NOT_FOUND"):
        confirm_gis_tower(
            project,
            "tower-missing",
            (1.0, 1.0, 1.0),
            CoordinateDeclaration.local_metric(),
        )
