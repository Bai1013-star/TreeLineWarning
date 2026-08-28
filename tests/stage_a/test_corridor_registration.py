from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from core.stage_a import (
    AnchorCorrespondence,
    Project,
    RegistrationState,
    Scan,
    SourceDescriptor,
    apply_registration_xy,
    build_crs_transform,
    georeferenced_registration,
    load_project,
    offline_proj_status,
    register_two_anchors,
    save_project,
    save_scan_registration,
    scan_registration_record,
    transform_xy,
    validate_registration_checkpoint,
)
from core.stage_a.models import stable_id


def test_offline_proj_transform_is_explicit_and_network_disabled() -> None:
    plan = build_crs_transform("EPSG:4326", "EPSG:3857")
    assert plan.source_crs == "EPSG:4326"
    assert plan.target_crs == "EPSG:3857"
    assert not plan.identity
    transformed = transform_xy(plan, ((1.0, 0.0),))[0]
    assert transformed[0] == pytest.approx(111_319.490793, rel=1e-6)
    assert transformed[1] == pytest.approx(0.0, abs=1e-9)
    status = offline_proj_status()
    assert status["network_enabled"] is False
    assert status["proj_network_env"] == "OFF"
    assert status["data_dir"]


def test_georeferenced_registration_transforms_gis_into_point_cloud_crs() -> None:
    registration = georeferenced_registration(
        point_cloud_crs="EPSG:3857",
        gis_crs="EPSG:4326",
    )
    assert registration.state is RegistrationState.GEOREFERENCED
    assert registration.ready_for_science
    assert registration.transform_2d is not None
    assert registration.transform_2d["direction"] == "GIS_TO_POINT_CLOUD"
    transformed = apply_registration_xy(
        registration,
        (1.0, 0.0),
        direction="GIS_TO_POINT_CLOUD",
    )
    assert transformed[0] == pytest.approx(111_319.490793, rel=1e-6)


def test_two_anchor_transform_requires_checkpoint_before_science() -> None:
    first = AnchorCorrespondence((0.0, 0.0, 0.0), (10.0, 20.0, 0.0), "T107")
    second = AnchorCorrespondence((10.0, 0.0, 0.0), (10.0, 30.0, 0.0), "T108")
    registration = register_two_anchors(first, second)
    assert registration.state is RegistrationState.REGISTERED_TWO_ANCHOR
    assert registration.unique_rigid_transform
    assert not registration.ready_for_science

    checkpoint = AnchorCorrespondence((0.0, 5.0, 0.0), (5.0, 20.0, 0.0), "T109")
    verified = validate_registration_checkpoint(
        registration,
        checkpoint,
        maximum_residual_m=0.25,
    )
    assert verified.state is RegistrationState.VERIFIED
    assert verified.ready_for_science
    assert verified.checkpoint_residual_m == pytest.approx(0.0, abs=1e-9)

    recovered = apply_registration_xy(
        verified,
        checkpoint.gis_xyz,
        direction="GIS_TO_POINT_CLOUD",
    )
    assert recovered == pytest.approx(checkpoint.point_cloud_xyz[:2], abs=1e-9)


def test_checkpoint_residual_fails_closed_without_silent_scale() -> None:
    first = AnchorCorrespondence((0.0, 0.0, 0.0), (10.0, 20.0, 0.0), "T107")
    second = AnchorCorrespondence((10.0, 0.0, 0.0), (10.0, 30.0, 0.0), "T108")
    registration = register_two_anchors(first, second)
    checkpoint = AnchorCorrespondence((0.0, 5.0, 0.0), (8.0, 20.0, 0.0), "T109")
    rejected = validate_registration_checkpoint(
        registration,
        checkpoint,
        maximum_residual_m=1.0,
    )
    assert rejected.state is RegistrationState.REQUIRES_CONFIRMATION
    assert not rejected.ready_for_science
    assert rejected.checkpoint_residual_m == pytest.approx(3.0)
    assert rejected.transform_2d is not None
    assert rejected.transform_2d["scale"] == 1.0


def test_invalid_known_crs_fails_closed_to_confirmation() -> None:
    registration = georeferenced_registration(
        point_cloud_crs="EPSG:3857",
        gis_crs="EPSG:999999",
    )
    assert registration.state is RegistrationState.REQUIRES_CONFIRMATION
    assert not registration.ready_for_science
    assert registration.transform_2d is None


def test_registration_persists_on_existing_scan_without_schema_change(tmp_path: Path) -> None:
    source_path = tmp_path / "source.las"
    source_path.write_bytes(b"placeholder")
    source = SourceDescriptor(
        "source-registration",
        str(source_path.resolve()),
        "LAS",
        source_path.stat().st_size,
    )
    scan = Scan(stable_id("scan"), source, str(source_path.resolve()), {"capabilities": {}})
    project = replace(Project.new("Registration project"), scans=(scan,))
    save_project(project, tmp_path)

    first = AnchorCorrespondence((0.0, 0.0, 0.0), (10.0, 20.0, 0.0), "T107")
    second = AnchorCorrespondence((10.0, 0.0, 0.0), (10.0, 30.0, 0.0), "T108")
    checkpoint = AnchorCorrespondence((0.0, 5.0, 0.0), (5.0, 20.0, 0.0), "T109")
    verified = validate_registration_checkpoint(
        register_two_anchors(first, second),
        checkpoint,
        maximum_residual_m=0.25,
    )
    save_scan_registration(
        tmp_path,
        scan.scan_id,
        verified,
        anchors=(first, second),
        checkpoint=checkpoint,
        gis_source_identity={
            "path": str((tmp_path / "route.kml").resolve()),
            "format": "KML",
            "sha256": "a" * 64,
        },
    )

    loaded = load_project(tmp_path)
    assert loaded.schema_version == project.schema_version
    restored_scan = loaded.scans[0]
    assert restored_scan.registration_status == RegistrationState.VERIFIED.value
    record = scan_registration_record(restored_scan)
    assert record is not None
    assert record["result"]["ready_for_science"] is True
    assert record["result"]["checkpoint_residual_m"] == pytest.approx(0.0)
    assert [item["gis_asset_id"] for item in record["anchors"]] == ["T107", "T108"]
    assert record["checkpoint"]["gis_asset_id"] == "T109"
    assert record["gis_source_identity"]["sha256"] == "a" * 64
