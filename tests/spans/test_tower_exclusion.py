"""Registered tower exclusion keeps neighbouring vegetation source identities."""

from __future__ import annotations

import numpy as np
import pytest
from algorithms.spans.tower_exclusion import (
    derive_registered_tower_exclusion,
    load_tower_membership,
    publish_tower_membership,
)
from algorithms.vegetation.coarse_to_fine import _RawRegionWriter


def test_shifted_tower_shaft_excludes_connected_members_but_keeps_near_tree(tmp_path):
    rng = np.random.default_rng(20260929)
    origin = np.asarray((100.0, 200.0, 5.0))
    registered = np.asarray((10.0, 20.0, 1.0))
    shaft_xy = registered[:2] + np.asarray((0.8, 0.75))
    count = 12_000
    z = rng.uniform(1.0, 11.0, count)
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    radius = rng.uniform(0.0, 0.12, count)
    shaft = np.column_stack(
        (shaft_xy[0] + radius * np.cos(angle), shaft_xy[1] + radius * np.sin(angle), z)
    )
    # A real adjacent tree has a separate stem and canopy inside the tower
    # search neighbourhood, but is not connected to the tower's shaft voxels.
    tree_xy = registered[:2] + np.asarray((2.7, 0.8))
    tree = np.column_stack(
        (
            rng.normal(tree_xy[0], 0.18, count),
            rng.normal(tree_xy[1], 0.18, count),
            rng.uniform(1.0, 10.0, count),
        )
    )
    points = np.concatenate((shaft, tree))
    source = np.arange(len(points), dtype=np.int64)
    writer = _RawRegionWriter(tmp_path / "region", origin_xyz=tuple(origin))
    writer.append(points - origin, source, np.zeros(len(points), dtype=np.uint8))
    region = writer.close()

    result = derive_registered_tower_exclusion(
        region,
        (
            {
                "tower_id": "registered-a",
                "base_xyz_absolute_m": registered,
                "top_xyz_absolute_m": registered + np.asarray((0.0, 0.0, 11.0)),
            },
        ),
    )

    assert result.towers[0]["coverage"] == "PARTIAL_CONNECTED_STRUCTURE"
    assert result.towers[0]["shaft_seed_offset_m"] > 0.7
    assert len(result.source_index) > 0.8 * len(shaft)
    assert np.all(result.source_index < len(shaft))


def test_no_vertical_shaft_does_not_delete_nearby_tree(tmp_path):
    rng = np.random.default_rng(29)
    points = np.column_stack(
        (rng.normal(1.0, 0.2, 200), rng.normal(1.0, 0.2, 200), rng.uniform(5.0, 7.0, 200))
    )
    writer = _RawRegionWriter(tmp_path / "region", origin_xyz=(0.0, 0.0, 0.0))
    writer.append(
        points, np.arange(len(points), dtype=np.int64), np.zeros(len(points), dtype=np.uint8)
    )
    region = writer.close()
    result = derive_registered_tower_exclusion(
        region,
        (
            {
                "tower_id": "no-shaft",
                "base_xyz_absolute_m": (0.0, 0.0, 0.0),
                "top_xyz_absolute_m": (0.0, 0.0, 12.0),
            },
        ),
    )
    assert len(result.source_index) == 0
    assert result.towers[0]["coverage"] == "UNAVAILABLE"


def test_denser_nearby_tree_with_crown_bridge_is_ambiguous_not_excluded(tmp_path):
    rng = np.random.default_rng(901)
    registered = np.asarray((900_000_000.0, -700_000_000.0, 340.0))
    pole_xy = registered[:2] + np.asarray((0.75, 0.1))
    tree_xy = registered[:2] + np.asarray((0.95, 1.1))
    pole_z = rng.uniform(340.0, 352.0, 8_000)
    tree_z = rng.uniform(340.0, 352.0, 44_000)
    pole = np.column_stack(
        (rng.normal(pole_xy[0], 0.055, len(pole_z)),
         rng.normal(pole_xy[1], 0.055, len(pole_z)), pole_z)
    )
    tree = np.column_stack(
        (rng.normal(tree_xy[0], 0.07, len(tree_z)),
         rng.normal(tree_xy[1], 0.07, len(tree_z)), tree_z)
    )
    crown_xy = pole_xy + rng.uniform(0.0, 1.0, (5_000, 1)) * (tree_xy - pole_xy)
    crown = np.column_stack((crown_xy, rng.uniform(348.0, 352.0, len(crown_xy))))
    points = np.concatenate((pole, tree, crown))
    # The LAS region has a different origin; exact absolute coordinates must
    # give the same result even far outside int32 grid-cell range.
    origin = registered - np.asarray((100.0, 200.0, 10.0))
    writer = _RawRegionWriter(tmp_path / "region", origin_xyz=tuple(origin))
    writer.append(
        points - origin,
        np.arange(len(points), dtype=np.int64),
        np.zeros(len(points), dtype=np.uint8),
    )
    result = derive_registered_tower_exclusion(
        writer.close(),
        ({
            "tower_id": "ambiguous",
            "base_xyz_absolute_m": registered,
            "top_xyz_absolute_m": registered + np.asarray((0.0, 0.0, 12.0)),
        },),
    )
    assert len(result.source_index) == 0
    assert result.towers[0]["coverage"] == "UNAVAILABLE"
    assert result.towers[0]["reason"] == "AMBIGUOUS_NEARBY_VERTICAL_SHAFTS"


@pytest.mark.parametrize("height_m,lean_m", [(7.0, 0.18), (20.0, 0.55)])
def test_different_heights_and_moderately_leaning_shafts(tmp_path, height_m, lean_m):
    rng = np.random.default_rng(int(height_m * 100))
    origin = np.asarray((-430.0, 85.0, 31.0))
    base = np.asarray((-400.0, 110.0, 32.0))
    z = rng.uniform(base[2], base[2] + height_m, 10_000)
    lean = (z - base[2]) * lean_m / height_m
    shaft = np.column_stack(
        (
            base[0] + 0.7 + lean + rng.normal(0, 0.06, len(z)),
            base[1] + 0.5 + rng.normal(0, 0.06, len(z)),
            z,
        )
    )
    tree = np.column_stack(
        (
            rng.normal(base[0] + 2.9, 0.1, len(z)),
            rng.normal(base[1] + 0.6, 0.1, len(z)),
            z,
        )
    )
    points = np.concatenate((shaft, tree))
    writer = _RawRegionWriter(tmp_path / "region", origin_xyz=tuple(origin))
    writer.append(
        points - origin,
        np.arange(len(points), dtype=np.int64),
        np.zeros(len(points), dtype=np.uint8),
    )
    region = writer.close()
    result = derive_registered_tower_exclusion(
        region,
        (
            {
                "tower_id": "leaning",
                "base_xyz_absolute_m": base,
                "top_xyz_absolute_m": base + np.asarray((0.0, 0.0, height_m)),
            },
        ),
    )
    assert result.towers[0]["coverage"] == "PARTIAL_CONNECTED_STRUCTURE"
    assert 0 < len(result.source_index) <= len(shaft)
    assert np.all(result.source_index < len(shaft))


def test_artifact_binding_rejects_changed_registered_geometry(tmp_path):
    rng = np.random.default_rng(77)
    z = rng.uniform(0.0, 10.0, 5_000)
    points = np.column_stack(
        (rng.normal(0.7, 0.05, len(z)), rng.normal(0.7, 0.05, len(z)), z)
    )
    writer = _RawRegionWriter(tmp_path / "region", origin_xyz=(0.0, 0.0, 0.0))
    writer.append(points, np.arange(len(z), dtype=np.int64), np.zeros(len(z), dtype=np.uint8))
    region = writer.close()
    original_tower = {
        "tower_id": "T001",
        "base_xyz_absolute_m": (0.0, 0.0, 0.0),
        "top_xyz_absolute_m": (0.0, 0.0, 10.0),
    }
    result = derive_registered_tower_exclusion(region, (original_tower,))
    source_identity = {"sha256": "f" * 64}
    publish_tower_membership(
        result, tmp_path, source_identity=source_identity, region_origin_xyz=(0.0, 0.0, 0.0)
    )
    assert load_tower_membership(
        tmp_path,
        source_identity=source_identity,
        registered_towers=(original_tower,),
        region_origin_xyz=(0.0, 0.0, 0.0),
    ) is not None
    changed = {**original_tower, "base_xyz_absolute_m": (1.0, 0.0, 0.0)}
    assert load_tower_membership(
        tmp_path,
        source_identity=source_identity,
        registered_towers=(changed,),
        region_origin_xyz=(0.0, 0.0, 0.0),
    ) is None
    assert load_tower_membership(
        tmp_path,
        source_identity={"sha256": "e" * 64},
        registered_towers=(original_tower,),
        region_origin_xyz=(0.0, 0.0, 0.0),
    ) is None
    assert load_tower_membership(
        tmp_path,
        source_identity=source_identity,
        registered_towers=(original_tower,),
        region_origin_xyz=(1.0, 0.0, 0.0),
    ) is None
    archive = tmp_path / "tower_membership.npz"
    with archive.open("ab") as stream:
        stream.write(b"tampered")
    assert load_tower_membership(
        tmp_path,
        source_identity=source_identity,
        registered_towers=(original_tower,),
        region_origin_xyz=(0.0, 0.0, 0.0),
    ) is None


def test_no_registered_towers_has_no_exclusions(tmp_path):
    writer = _RawRegionWriter(tmp_path / "region", origin_xyz=(0.0, 0.0, 0.0))
    writer.append(
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        np.asarray([42], dtype=np.int64),
        np.zeros(1, dtype=np.uint8),
    )
    result = derive_registered_tower_exclusion(writer.close(), ())
    assert len(result.source_index) == 0 and result.towers == ()
    publish_tower_membership(
        result,
        tmp_path,
        source_identity={"sha256": "a" * 64},
        region_origin_xyz=(0.0, 0.0, 0.0),
    )
    loaded = load_tower_membership(
        tmp_path,
        source_identity={"sha256": "a" * 64},
        registered_towers=(),
        region_origin_xyz=(0.0, 0.0, 0.0),
    )
    assert loaded is not None and len(loaded.source_index) == 0
