"""Exact original-index Wire/Tower exclusion for the final tree input.

The final tree input is::

    FINAL_TREE_INPUT = VEGETATION - CONFIRMED_WIRE - CONFIRMED_TOWER

only via exact full-resolution ``original_index`` subtraction — never a
distance buffer, never display points.  Near-wire vegetation is NOT removed
merely because it is close to the conductor; only exact confirmed members are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from algorithms.spans.region import subset_raw_region_excluding_source_indices
from algorithms.vegetation.coarse_to_fine import _RawRegionWriter


def _write_region(root: Path, xyz: np.ndarray, indices: np.ndarray):
    writer = _RawRegionWriter(root, origin_xyz=(0.0, 0.0, 0.0))
    writer.append(
        np.asarray(xyz, dtype=np.float64),
        np.asarray(indices, dtype=np.int64),
        np.zeros(len(xyz), dtype=np.uint8),
    )
    return writer.close()


def _vegetation_fixture(tmp_path: Path):
    """Vegetation region: trees, near-wire vegetation, and (leaked) wire points."""
    rng = np.random.default_rng(11)
    n = 1000
    xyz = rng.uniform([0, 0, 2], [100, 20, 25], (n, 3))
    indices = np.arange(n, dtype=np.int64)
    region = _write_region(tmp_path / "region", xyz, indices)
    # Designate memberships by index ranges.
    # indices 0..49   : confirmed wire members (must be subtracted)
    # indices 50..99  : confirmed tower members (must be subtracted)
    # indices 100..150: near-wire vegetation (must be KEPT — proximity is irrelevant)
    confirmed_wire = indices[0:50]
    confirmed_tower = indices[50:100]
    near_wire_veg = indices[100:151]
    return region, confirmed_wire, confirmed_tower, near_wire_veg


def test_confirmed_wire_exact_exclusion(tmp_path: Path) -> None:
    region, wire, _tower, _near = _vegetation_fixture(tmp_path)
    cleaned = subset_raw_region_excluding_source_indices(
        region, wire, tmp_path / "out_wire"
    )
    kept = np.asarray(cleaned.original_index(), dtype=np.int64)
    assert len(np.intersect1d(kept, wire)) == 0  # no wire member remains
    assert len(kept) == region.count - len(wire)


def test_confirmed_tower_exact_exclusion(tmp_path: Path) -> None:
    region, _wire, tower, _near = _vegetation_fixture(tmp_path)
    cleaned = subset_raw_region_excluding_source_indices(
        region, tower, tmp_path / "out_tower"
    )
    kept = np.asarray(cleaned.original_index(), dtype=np.int64)
    assert len(np.intersect1d(kept, tower)) == 0
    assert len(kept) == region.count - len(tower)


def test_wire_member_cannot_remain_in_final_tree_input(tmp_path: Path) -> None:
    region, wire, tower, _near = _vegetation_fixture(tmp_path)
    excluded = np.unique(np.concatenate((wire, tower)))
    cleaned = subset_raw_region_excluding_source_indices(
        region, excluded, tmp_path / "out_comb"
    )
    kept = np.asarray(cleaned.original_index(), dtype=np.int64)
    assert len(np.intersect1d(kept, wire)) == 0
    assert len(np.intersect1d(kept, tower)) == 0


def test_tower_member_cannot_remain_when_authoritative_membership_exists(
    tmp_path: Path,
) -> None:
    region, _wire, tower, _near = _vegetation_fixture(tmp_path)
    cleaned = subset_raw_region_excluding_source_indices(
        region, tower, tmp_path / "out_tower2"
    )
    kept = np.asarray(cleaned.original_index(), dtype=np.int64)
    assert len(np.intersect1d(kept, tower)) == 0


def test_near_wire_vegetation_is_not_removed_by_distance(tmp_path: Path) -> None:
    # Near-wire vegetation is the RISK target: only exact confirmed members are
    # removed.  Proximity alone must never exclude vegetation.
    region, wire, _tower, near_wire_veg = _vegetation_fixture(tmp_path)
    cleaned = subset_raw_region_excluding_source_indices(
        region, wire, tmp_path / "out_near"
    )
    kept = np.asarray(cleaned.original_index(), dtype=np.int64)
    # The near-wire vegetation rows are NOT confirmed wire members -> kept.
    assert len(np.intersect1d(kept, near_wire_veg)) == len(near_wire_veg)


def test_exclusion_is_exact_original_index_not_display(tmp_path: Path) -> None:
    # Display sampling must never be used as the exclusion authority.  Excluding
    # via a small sample of indices would leave the rest in the tree input.
    region, wire, _tower, _near = _vegetation_fixture(tmp_path)
    sampled_wire = wire[::2]  # a "display" subset (only half)
    cleaned = subset_raw_region_excluding_source_indices(
        region, sampled_wire, tmp_path / "out_display"
    )
    kept = np.asarray(cleaned.original_index(), dtype=np.int64)
    # The unsampled half is still present because it was never excluded.
    remaining = np.setdiff1d(wire, sampled_wire)
    assert len(np.intersect1d(kept, remaining)) == len(remaining)


def test_empty_exclusion_is_identity(tmp_path: Path) -> None:
    region, _wire, _tower, _near = _vegetation_fixture(tmp_path)
    cleaned = subset_raw_region_excluding_source_indices(
        region, np.empty(0, dtype=np.int64), tmp_path / "out_empty"
    )
    assert cleaned is region


def test_excluded_indices_negative_rejected(tmp_path: Path) -> None:
    from algorithms.vegetation.coarse_to_fine import FastResearchError

    region, _wire, _tower, _near = _vegetation_fixture(tmp_path)
    with pytest.raises(FastResearchError):
        subset_raw_region_excluding_source_indices(
            region, np.asarray([-1, 5], dtype=np.int64), tmp_path / "out_neg"
        )


# ---------------------------------------------------------------------------
# Orchestrator audit contract (the counts recorded in run metadata)
# ---------------------------------------------------------------------------

def test_tree_input_audit_counts_and_intersections(tmp_path: Path) -> None:
    from algorithms.folder_pipeline.orchestrator import _tree_input_audit

    region, wire, tower, _near = _vegetation_fixture(tmp_path)
    excluded = np.unique(np.concatenate((wire, tower)))
    cleaned = subset_raw_region_excluding_source_indices(
        region, excluded, tmp_path / "out_audit"
    )
    audit = _tree_input_audit(region, cleaned, excluded, tower)
    assert audit["vegetation_input_count"] == region.count
    assert audit["final_tree_input_count"] == cleaned.count
    assert audit["wire_tree_input_intersection"] == 0
    assert audit["tower_tree_input_intersection"] == 0
