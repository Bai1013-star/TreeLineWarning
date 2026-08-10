"""V4 member_count zero-member (ghost tree) regression tests.

A formal tree hypothesis can survive the final union while the point
assignment leaves it with zero ASSIGNED members.  The ``member_count`` contract
requires a positive built-in int, so such a ghost tree must be skipped (not
crashed on, and never faked to a positive value).  These tests prove:

* the ghost-tree filter compacts tree ids consistently,
* the member_count contract stays strict (zero is rejected),
* the full segmentation route skips a ghost instead of crashing,
* non-ghost scenes are byte-identical (baseline preserved).
"""

from __future__ import annotations

import numpy as np
import pytest
from algorithms.vegetation import v4_tree_segmentation as v4s
from core.contracts.v4_tree_segmentation import (
    V4ApexCandidateRecord,
    V4TreeAssignmentState,
    V4TreeGeometrySummary,
    V4TreeHypothesisState,
    V4TrunkEvidenceAvailability,
)

from tests.vegetation.test_v4_tree_segmentation import (
    _apex_record,
    make_config,
    make_handoff,
    two_trees_scene,
)


def _geometry(member_count: int) -> V4TreeGeometrySummary:
    return V4TreeGeometrySummary(
        member_count=member_count,
        hag_min_m=1.0,
        hag_max_m=5.0,
        robust_hag_m=4.0,
        vertical_extent_m=4.0,
        xy_centroid=(1.0, 1.0, 3.0),
        xy_bounds_min=(0.0, 0.0, 1.0),
        xy_bounds_max=(2.0, 2.0, 5.0),
        crown_area_m2=4.0,
        equivalent_radius_m=1.13,
        responsibility_area_m2=4.0,
        slice_count=3,
        valid_slice_count=3,
        lowest_supported_slice_id="slice-0",
        highest_supported_slice_id="slice-2",
        trunk_evidence_state=V4TrunkEvidenceAvailability.OBSERVED,
        accepted_trunk_count=1,
        primary_trunk_representative_original_index=0,
        multi_stem_flag=False,
        geometry_quality_flags=0,
    )


def test_member_count_zero_rejected_by_strict_contract() -> None:
    """The contract must keep rejecting zero member_count (never max(1, ...))."""
    with pytest.raises(ValueError, match="member_count must be a positive built-in int"):
        _geometry(member_count=0)


def test_member_count_positive_accepts_builtin_int() -> None:
    geometry = _geometry(member_count=7)
    assert geometry.member_count == 7


def _apex(candidate_id: str) -> V4ApexCandidateRecord:
    return _apex_record(candidate_id, (1.0, 1.0, 3.0))


def test_drop_ghost_trees_compacts_ids() -> None:
    # Tree 1 is a ghost: it has no positions and no point carries its id.
    formal_trees = [
        (0, [_apex("apex-0")], V4TreeHypothesisState.ACTIVE),
        (1, [_apex("apex-1")], V4TreeHypothesisState.ACTIVE),
        (2, [_apex("apex-2")], V4TreeHypothesisState.ACTIVE),
    ]
    positions_by_tree = {0: [0, 1], 2: [2, 3]}
    tree_id_by_point = np.asarray([0, 0, 2, 2], dtype="<i4")
    kept, positions, ids = v4s._drop_ghost_trees(
        formal_trees, positions_by_tree, tree_id_by_point
    )
    assert len(kept) == 2
    assert positions == {0: [0, 1], 1: [2, 3]}
    assert ids.tolist() == [0, 0, 1, 1]


def test_drop_ghost_trees_noop_without_ghosts() -> None:
    formal_trees = [
        (0, [_apex("apex-0")], V4TreeHypothesisState.ACTIVE),
        (1, [_apex("apex-1")], V4TreeHypothesisState.ACTIVE),
    ]
    positions_by_tree = {0: [0, 1], 1: [2, 3]}
    tree_id_by_point = np.asarray([0, 0, 1, 1], dtype="<i4")
    kept, positions, ids = v4s._drop_ghost_trees(
        formal_trees, positions_by_tree, tree_id_by_point
    )
    assert len(kept) == 2
    assert positions == positions_by_tree
    assert ids.tolist() == tree_id_by_point.tolist()


def test_segment_v4_trees_skips_ghost_instead_of_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ghost tree (all its points REVIEW) is skipped, never a member_count=0 crash."""
    real = v4s._assign_points

    def patched(*args: object, **kwargs: object):
        state, tree_id, confidence, reason = real(*args, **kwargs)
        state = state.copy()
        tree_id = tree_id.copy()
        mask = tree_id == 1
        state[mask] = int(V4TreeAssignmentState.REVIEW_REQUIRED)
        tree_id[mask] = -1
        return state, tree_id, confidence, reason

    monkeypatch.setattr(v4s, "_assign_points", patched)
    result = v4s.segment_v4_trees(make_handoff(two_trees_scene()), make_config())
    assert len(result.tree_records) == 1
    for record in result.tree_records:
        assert record.member_count > 0
        assert record.geometry_summary.member_count == record.member_count
        assert record.geometry_summary.member_count > 0
