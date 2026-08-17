"""High-recall vegetation admission + exact confirmed-conductor exclusion tests.

The systematic vegetation-loss bug was that the source-shared VEGETATION role
index excluded the ENTIRE broad wire-role search-corridor source set (millions
of points), dropping conductor-adjacent vegetation before the tree pipeline.

Corrected semantics (this phase):
- vegetation admission is HIGH RECALL (no broad wire exclusion at role index);
- only the EXACT FULL_RESOLUTION source identity of FINAL CONFIRMED conductor
  strand members is subtracted from the Span-local vegetation tree input;
- broad wire candidate / search points may overlap vegetation candidates;
- stale old-semantics vegetation role caches FAIL CLOSED.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from algorithms.spans.region import subset_raw_region_excluding_source_indices
from algorithms.spans.role_tiles import RoleSpecificTileCache, RoleTileCacheError
from algorithms.spans.wire import _confirmed_strand_member_source_indices
from algorithms.vegetation.coarse_to_fine import _RawRegionWriter, build_fine_representatives

_IDENTITY = {
    "source_content_identity_digest": "a" * 64,
    "source_identity_record_digest": "b" * 64,
}


def _write_region(tmp_path: Path, name: str, xyz, indices, classes):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    writer = _RawRegionWriter(root, origin_xyz=(0.0, 0.0, 0.0))
    writer.append(np.asarray(xyz, dtype=np.float64), np.asarray(indices, dtype=np.int64),
                  np.asarray(classes, dtype=np.uint8))
    return writer.close()


def test_exact_confirmed_member_extraction_only_conductor(tmp_path):
    """Only the exact raw source indices of confirmed strand voxels are extracted."""
    rs = np.random.RandomState(0)
    zs = np.arange(0.0, 20.0, 0.1)
    conductor = np.column_stack(
        (np.full(len(zs), 5.0) + rs.normal(0, 0.02, len(zs)),
         np.full(len(zs), 5.0) + rs.normal(0, 0.02, len(zs)), zs)
    )
    noise = np.column_stack(
        (rs.uniform(0, 10, 300), rs.uniform(0, 10, 300), rs.uniform(0, 20, 300))
    )
    xyz = np.vstack((conductor, noise))
    indices = np.arange(len(xyz), dtype=np.int64)
    region = _write_region(tmp_path, "wire", xyz, indices, np.zeros(len(xyz), dtype=np.uint8))
    arrays = build_fine_representatives(region, tmp_path / "fine.npz", voxel_size_m=0.1)
    conductor_rep_rows = np.flatnonzero(arrays["original_index"] < len(conductor))
    members = _confirmed_strand_member_source_indices(region, [conductor_rep_rows], 0.1)
    expected = np.arange(len(conductor), dtype=np.int64)
    noise_idx = np.arange(len(conductor), len(xyz), dtype=np.int64)
    assert np.isin(expected, members).all()  # every conductor point is a member
    assert not np.isin(noise_idx, members).any()  # nearby noise is NOT removed


def test_exclusion_subsetter_only_removes_exact_indices(tmp_path):
    """The per-Span exclusion subsetter drops only the exact excluded source rows."""
    xyz = np.column_stack((np.arange(8, dtype=float), np.zeros(8), np.ones(8)))
    indices = np.asarray([100, 101, 102, 103, 104, 105, 106, 107], dtype=np.int64)
    region = _write_region(tmp_path, "veg", xyz, indices, np.zeros(8, dtype=np.uint8))
    excluded = np.asarray([101, 104], dtype=np.int64)
    filtered = subset_raw_region_excluding_source_indices(
        region, excluded, tmp_path / "filtered"
    )
    kept = np.asarray(filtered.original_index(), dtype=np.int64)
    assert sorted(kept.tolist()) == [100, 102, 103, 105, 106, 107]
    # A no-op exclusion returns the same region object (no unnecessary copy).
    assert subset_raw_region_excluding_source_indices(
        region, np.empty(0, dtype=np.int64), tmp_path / "noop"
    ) is region


def test_vegetation_role_index_is_high_recall(tmp_path):
    """Vegetation role index keeps every point; wire candidates stay in vegetation."""
    xyz = np.column_stack((np.arange(4, dtype=float), np.zeros(4), np.ones(4)))
    indices = np.asarray([20, 21, 22, 23], dtype=np.int64)
    region = _write_region(tmp_path, "veg", xyz, indices, np.zeros(4, dtype=np.uint8))
    cache = RoleSpecificTileCache.prepare_from_raw_region(
        region, tmp_path / "veg-role", role="vegetation", source_identity=_IDENTITY,
        config_digest="vegetation-config", source_point_count=24, cell_size_m=5.0,
    )
    loaded = RoleSpecificTileCache.load(
        cache.root, role="vegetation", source_identity=_IDENTITY,
        config_digest="vegetation-config", source_point_count=24,
    )
    assert loaded.metadata["admission_semantics"] == "HIGH_RECALL"
    assert loaded.metadata["totals"]["point_count"] == 4


def test_stale_old_exclusion_vegetation_cache_rejected(tmp_path):
    """An old broad-wire-exclusion vegetation role cache must fail closed."""
    xyz = np.column_stack((np.arange(4, dtype=float), np.zeros(4), np.ones(4)))
    indices = np.asarray([20, 21, 22, 23], dtype=np.int64)
    region = _write_region(tmp_path, "veg", xyz, indices, np.zeros(4, dtype=np.uint8))
    # Build with the OLD semantics digest + a wire exclusion to emulate a legacy cache.
    with pytest.raises(ValueError):
        RoleSpecificTileCache.prepare_from_raw_region(
            region, tmp_path / "old-veg", role="vegetation", source_identity=_IDENTITY,
            config_digest="old-config", source_point_count=24, cell_size_m=5.0,
            excluded_source_indices=np.asarray([21, 23], dtype=np.int64),
            exclusion_role="wire", exclusion_source_index_digest="c" * 64,
        )
    # A hand-crafted legacy cache (vegetation role carrying source_index_exclusion)
    # loaded with the new HIGH-RECALL digest must fail closed.
    wire_cache = RoleSpecificTileCache.prepare_from_raw_region(
        region, tmp_path / "wire-role", role="wire", source_identity=_IDENTITY,
        config_digest="wire-config", source_point_count=24, cell_size_m=5.0,
    )
    import json
    import shutil

    legacy = tmp_path / "legacy-veg"
    shutil.copytree(wire_cache.root, legacy)
    meta_path = legacy / "role-tile-cache.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["role"] = "vegetation"
    meta["source_index_exclusion"] = {
        "schema": "treeline-span-role-index-exclusion-v1",
        "rule": "EXCLUDE_EXACT_SOURCE_INDICES",
        "role": "wire",
        "excluded_count": 2,
        "excluded_index_digest": "d" * 64,
        "reference_source_binding_digest": "e" * 64,
        "reference_source_index_digest": "f" * 64,
    }
    meta_path.write_text(json.dumps(meta, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(RoleTileCacheError):
        RoleSpecificTileCache.load(
            legacy, role="vegetation", source_identity=_IDENTITY,
            config_digest="vegetation-config", source_point_count=24,
        )


def test_no_global_workset_and_no_tower_membership_authority():
    """The corrected executor keeps role-tile locality and never uses tower membership."""
    import inspect

    from algorithms.spans import canonical_executor

    source = inspect.getsource(canonical_executor)
    # Span-local role tile caches are the only source path (no global workset).
    assert "subset_raw_tile_cache_oriented" in source
    assert "ROLE_SPECIFIC_SOURCE_TILE_CACHE" in source
    assert "global_workset_used" in source
    # The corrected flow never rebuilds a broad wire-role exclusion into the
    # vegetation role index; exclusion is exact confirmed members, per Span.
    assert "subset_raw_region_excluding_source_indices" in source
    assert "confirmed_member_source_index" in source
    # Tower membership is explicitly not vegetation authority (preparation flag).
    from algorithms.spans.project_adapter import prepare_source_shared

    prep_src = inspect.getsource(prepare_source_shared)
    assert "tower_membership_used_for_vegetation" in prep_src
    assert '"tower_membership_used_for_vegetation": False' in prep_src or \
        '"tower_membership_used_for_vegetation": False' in source


def test_exact_source_overlap_after_exclusion_is_zero(tmp_path):
    """After exact confirmed-conductor exclusion, tree input overlap is zero."""
    rs = np.random.RandomState(1)
    xyz = np.column_stack((rs.uniform(0, 10, 200), rs.uniform(0, 10, 200), rs.uniform(0, 5, 200)))
    indices = np.arange(200, dtype=np.int64)
    region = _write_region(tmp_path, "veg", xyz, indices, np.zeros(200, dtype=np.uint8))
    # A few exact "confirmed conductor" indices to subtract.
    confirmed = np.asarray([3, 7, 11, 42], dtype=np.int64)
    filtered = subset_raw_region_excluding_source_indices(region, confirmed, tmp_path / "f")
    overlap = np.intersect1d(confirmed, np.asarray(filtered.original_index(), dtype=np.int64))
    assert len(overlap) == 0


def test_boundary_accounting_is_exact_for_same_tile_wire_and_tree(tmp_path):
    """Exact subtraction retains a near-wire tree point in the same source tile."""
    xyz = np.asarray(
        [
            [5.00, 5.00, 12.00],  # final confirmed conductor source member
            [5.02, 5.01, 11.98],  # near-wire branch: must remain tree input
            [8.00, 5.00, 4.00],   # far vegetation: must remain tree input
        ],
        dtype=np.float64,
    )
    region = _write_region(
        tmp_path,
        "same-tile",
        xyz,
        np.asarray([71, 72, 73], dtype=np.int64),
        np.zeros(3, dtype=np.uint8),
    )
    confirmed = np.asarray([71], dtype=np.int64)
    filtered = subset_raw_region_excluding_source_indices(
        region, confirmed, tmp_path / "tree-input"
    )
    tree_input = np.asarray(filtered.original_index(), dtype=np.int64)
    accounting = {
        "high_recall_candidate_count": region.count,
        "confirmed_wire_excluded_count": int(np.intersect1d(confirmed, [71, 72, 73]).size),
        "other_excluded_count": 0,
        "final_tree_input_count": filtered.count,
    }
    assert set(tree_input) == {72, 73}
    assert np.intersect1d(confirmed, tree_input).size == 0
    assert accounting["high_recall_candidate_count"] == (
        accounting["confirmed_wire_excluded_count"]
        + accounting["other_excluded_count"]
        + accounting["final_tree_input_count"]
    )
