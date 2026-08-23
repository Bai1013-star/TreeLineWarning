from __future__ import annotations

import hashlib
from pathlib import Path

import laspy
import numpy as np
import pytest
from algorithms.spans.role_tiles import (
    RoleSpecificTileCache,
    RoleTileCacheError,
    migrate_role_tile_source_identity,
)
from algorithms.vegetation.coarse_to_fine import (
    _RawRegionWriter,
    recover_vegetation_region_raw,
    recover_wire_and_high_recall_regions_raw,
)
from algorithms.vegetation.fast_weighted_preprocess import build_weighted_working_set
from core.stage_b.role_preparation import (
    _publish_global_role_cache,
    _reuse_global_role_cache,
)


def _identity(path: Path, sha256: str) -> dict[str, object]:
    return {
        "source_id": f"generated-{path.parent.name}",
        "path": str(path),
        "format": "LAS",
        "file_size_bytes": 1234,
        "sha256": sha256,
    }


def _region(root: Path):
    root.mkdir(parents=True)
    writer = _RawRegionWriter(root, origin_xyz=(0.0, 0.0, 0.0))
    writer.append(
        np.asarray(((0.0, 0.0, 1.0), (1.0, 1.0, 2.0)), dtype=np.float64),
        np.asarray((4, 9), dtype=np.int64),
        np.asarray((1, 2), dtype=np.uint8),
    )
    return writer.close()


def test_same_content_new_project_reuses_complete_global_role_cache(tmp_path: Path) -> None:
    source_a = tmp_path / "project-a" / "same.las"
    source_b = tmp_path / "project-b" / "renamed.las"
    source_a.parent.mkdir()
    source_b.parent.mkdir()
    source_a.write_bytes(b"same source")
    source_b.write_bytes(b"same source")
    identity_a = _identity(source_a, "a" * 64)
    identity_b = _identity(source_b, "a" * 64)
    region = _region(tmp_path / "region")
    wire_a = tmp_path / "project-a" / "wire"
    vegetation_a = tmp_path / "project-a" / "vegetation"
    RoleSpecificTileCache.prepare_from_raw_region(
        region,
        wire_a,
        role="wire",
        source_identity=identity_a,
        config_digest="wire-v1",
        source_point_count=10,
    )
    RoleSpecificTileCache.prepare_from_raw_region(
        region,
        vegetation_a,
        role="vegetation",
        source_identity=identity_a,
        config_digest="vegetation-v1",
        source_point_count=10,
    )
    global_root = tmp_path / "global"
    cache_key = "b" * 64
    _publish_global_role_cache(
        global_root,
        cache_key=cache_key,
        wire_root=wire_a,
        vegetation_root=vegetation_a,
    )

    wire_b = tmp_path / "project-b" / "wire"
    vegetation_b = tmp_path / "project-b" / "vegetation"
    assert _reuse_global_role_cache(
        global_root,
        cache_key=cache_key,
        wire_root=wire_b,
        vegetation_root=vegetation_b,
    )
    RoleSpecificTileCache.load(
        wire_b,
        role="wire",
        source_identity=identity_b,
        config_digest="wire-v1",
        source_point_count=10,
    )
    RoleSpecificTileCache.load(
        vegetation_b,
        role="vegetation",
        source_identity=identity_b,
        config_digest="vegetation-v1",
        source_point_count=10,
    )


def test_different_content_fails_closed_after_materialization(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    source.write_bytes(b"fixture")
    good = _identity(source, "c" * 64)
    changed = _identity(source, "d" * 64)
    region = _region(tmp_path / "region")
    cache = RoleSpecificTileCache.prepare_from_raw_region(
        region,
        tmp_path / "wire",
        role="wire",
        source_identity=good,
        config_digest="wire-v1",
        source_point_count=10,
    )

    with pytest.raises(RoleTileCacheError, match="SOURCE_MISMATCH"):
        RoleSpecificTileCache.load(
            cache.root,
            role="wire",
            source_identity=changed,
            config_digest="wire-v1",
            source_point_count=10,
        )


def test_relevant_configuration_change_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    source.write_bytes(b"fixture")
    identity = _identity(source, "e" * 64)
    region = _region(tmp_path / "region")
    cache = RoleSpecificTileCache.prepare_from_raw_region(
        region,
        tmp_path / "vegetation",
        role="vegetation",
        source_identity=identity,
        config_digest="vegetation-v1",
        source_point_count=10,
    )

    with pytest.raises(RoleTileCacheError, match="CONFIGURATION_MISMATCH"):
        RoleSpecificTileCache.load(
            cache.root,
            role="vegetation",
            source_identity=identity,
            config_digest="vegetation-v2",
            source_point_count=10,
        )


def test_validated_weak_cache_can_be_rebound_to_proven_strong_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.las"
    source.write_bytes(b"fixture")
    weak = {"source_id": "legacy", "path": str(source)}
    strong = _identity(source, "f" * 64)
    region = _region(tmp_path / "region")
    cache = RoleSpecificTileCache.prepare_from_raw_region(
        region,
        tmp_path / "wire",
        role="wire",
        source_identity=weak,
        config_digest="wire-v1",
        source_point_count=10,
    )

    migrated = migrate_role_tile_source_identity(
        cache.root,
        role="wire",
        previous_source_identity=weak,
        source_identity=strong,
        configuration_digest="wire-v1",
        source_point_count=10,
    )

    assert migrated.metadata["source_identity"] == strong
    with pytest.raises(RoleTileCacheError, match="SOURCE_MISMATCH"):
        RoleSpecificTileCache.load(
            migrated.root,
            role="wire",
            source_identity=weak,
            config_digest="wire-v1",
            source_point_count=10,
        )


def test_fused_wire_vegetation_recovery_is_exact(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    las = laspy.create(point_format=3, file_version="1.2")
    las.x = np.asarray((0.0, 0.1, 1.0, 1.1, 2.0, 2.1))
    las.y = np.asarray((0.0, 0.1, 1.0, 1.1, 2.0, 2.1))
    las.z = np.asarray((1.0, 1.1, 2.0, 2.1, 3.0, 3.1))
    las.classification = np.asarray((1, 2, 3, 4, 5, 6), dtype=np.uint8)
    las.write(source)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    _status, working = build_weighted_working_set(
        source,
        tmp_path / "working",
        voxel_size_m=0.5,
        source_sha256=source_sha,
    )
    retained = np.asarray((0, 2), dtype=np.uint32)
    expected_wire = recover_vegetation_region_raw(
        source, working, retained, tmp_path / "expected-wire"
    )
    expected_vegetation = recover_vegetation_region_raw(
        source,
        working,
        np.arange(working.representative_count, dtype=np.uint32),
        tmp_path / "expected-vegetation",
    )
    wire, vegetation = recover_wire_and_high_recall_regions_raw(
        source,
        working,
        retained,
        tmp_path / "fused-wire",
        tmp_path / "fused-vegetation",
    )

    for expected, actual in (
        (expected_wire, wire),
        (expected_vegetation, vegetation),
    ):
        assert np.array_equal(expected.original_index(), actual.original_index())
        assert np.array_equal(expected.xyz(), actual.xyz())
        assert np.array_equal(expected.classification(), actual.classification())
