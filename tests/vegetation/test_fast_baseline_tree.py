"""Synthetic tests for the per-span tree pipeline (Parts 12-21, 42)."""

from __future__ import annotations

import numpy as np
from algorithms.spans.tree import (
    SpanChm,
    SpanCrownSegmentation,
    assign_span_tree_membership,
    build_span_chm,
    detect_span_tree_tops,
    segment_span_crowns,
    smooth_span_chm,
)
from core.config.fast_baseline import default_fast_baseline_config

_CFG = default_fast_baseline_config()


def _smooth_crown(cx: float, cy: float, height: float, radius: float, points: int) -> np.ndarray:
    rng = np.random.default_rng(hash((cx, cy, height)) % (2**32))
    rows = []
    for _ in range(points):
        r = radius * np.sqrt(rng.uniform(0.0, 1.0))
        angle = rng.uniform(0.0, 2 * np.pi)
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        z = height * (1.0 - (r / radius) ** 2) * rng.uniform(0.25, 1.0)
        rows.append((x, y, max(1.5, z)))
    return np.asarray(rows, dtype=np.float64)


def _run_tree_pipeline(xyz: np.ndarray) -> tuple[SpanChm, np.ndarray, object, object, np.ndarray]:
    hag = xyz[:, 2].astype(np.float32)
    chm = build_span_chm(xyz, hag, _CFG)
    smoothed = smooth_span_chm(chm, _CFG)
    tops = detect_span_tree_tops(chm, smoothed, _CFG)
    crowns = segment_span_crowns(chm, smoothed, tops, _CFG)
    membership = assign_span_tree_membership(xyz, hag, chm, smoothed, crowns, _CFG)
    return chm, smoothed, tops, crowns, membership


def test_one_crown_yields_one_tree() -> None:
    xyz = _smooth_crown(0.0, 0.0, 12.0, 3.0, 600)
    chm, smoothed, tops, crowns, membership = _run_tree_pipeline(xyz)
    assert tops.accepted_count == 1
    assert np.count_nonzero(membership >= 0) >= 300


def test_multiple_branch_peaks_suppressed_to_one_dominant_top() -> None:
    # A crown with many small branch maxima near the top must still produce one
    # dominant tree top thanks to adaptive suppression.
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(800):
        r = 2.5 * np.sqrt(rng.uniform(0.0, 1.0))
        angle = rng.uniform(0.0, 2 * np.pi)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = 10.0 * (1.0 - (r / 2.5) ** 2) + rng.normal(0.0, 0.25)
        rows.append((x, y, max(1.5, z)))
    xyz = np.asarray(rows, dtype=np.float64)
    chm, smoothed, tops, crowns, membership = _run_tree_pipeline(xyz)
    assert tops.accepted_count == 1


def test_two_separated_crowns_yield_two_trees() -> None:
    first = _smooth_crown(0.0, 0.0, 12.0, 3.0, 500)
    second = _smooth_crown(20.0, 0.0, 9.0, 2.5, 400)
    xyz = np.concatenate((first, second), axis=0)
    chm, smoothed, tops, crowns, membership = _run_tree_pipeline(xyz)
    assert tops.accepted_count == 2
    unique = sorted(int(value) for value in np.unique(membership) if value >= 0)
    assert unique == [0, 1]


def test_two_touching_crowns_produce_two_markers_when_justified() -> None:
    # Two crowns whose edges touch (close centres, distinct peaks) must survive
    # as separate markers and separate watershed regions.
    first = _smooth_crown(0.0, 0.0, 12.0, 2.2, 450)
    second = _smooth_crown(4.5, 0.0, 10.0, 2.2, 450)
    xyz = np.concatenate((first, second), axis=0)
    chm, smoothed, tops, crowns, membership = _run_tree_pipeline(xyz)
    assert tops.accepted_count == 2
    crown_labels = np.unique(crowns.region_label_by_cell)
    assert len(crown_labels) - 1 == 2  # labels 1 and 2 for the two markers


def test_close_shorter_tree_survives_only_with_meaningful_saddle() -> None:
    # The shorter peak is inside the taller tree's adaptive window.  A deep
    # CHM saddle and continuous canopy path are the evidence that it is a real
    # neighbour, not merely a branch shoulder.
    height = np.full((11, 15), 3.0, dtype=np.float32)
    height[5, 4] = 12.0
    height[5, 8] = 8.0
    height[5, 5:8] = np.asarray((8.0, 4.0, 6.0), dtype=np.float32)
    chm = SpanChm(
        height, np.ones_like(height, dtype=bool), np.ones_like(height, dtype=np.int32),
        (0.0, 0.0), 0.5, 1.5,
    )
    tops = detect_span_tree_tops(chm, height, _CFG)
    assert tops.accepted_count == 2
    accepted = [
        item for item in tops.diagnostics["candidate_diagnostics"]
        if item["reason"] == "ACCEPTED"
    ]
    assert any("independent_peak_evidence" in item for item in accepted)


def test_shallow_shoulder_remains_suppressed() -> None:
    height = np.full((11, 15), 3.0, dtype=np.float32)
    height[5, 4] = 12.0
    height[5, 8] = 9.0
    height[5, 5:8] = np.asarray((10.5, 8.7, 8.8), dtype=np.float32)
    chm = SpanChm(
        height, np.ones_like(height, dtype=bool), np.ones_like(height, dtype=np.int32),
        (0.0, 0.0), 0.5, 1.5,
    )
    tops = detect_span_tree_tops(chm, height, _CFG)
    assert tops.accepted_count == 1
    assert "SUPPRESSED_BY_NEIGHBOUR" in {
        item["reason"] for item in tops.diagnostics["candidate_diagnostics"]
    }


def test_suppression_diagnostic_identifies_actual_neighbour() -> None:
    """The diagnostic points to the first accepted marker that suppresses it."""
    height = np.full((15, 25), 3.0, dtype=np.float32)
    height[7, 3] = 15.0
    height[7, 11] = 12.0
    height[7, 15] = 9.0
    height[7, 4:11] = np.linspace(14.0, 11.0, 7, dtype=np.float32)
    height[7, 12:17] = np.asarray((10.0, 9.0, 8.0, 9.0, 7.0), dtype=np.float32)
    chm = SpanChm(
        height, np.ones_like(height, dtype=bool), np.ones_like(height, dtype=np.int32),
        (0.0, 0.0), 0.5, 1.5,
    )
    tops = detect_span_tree_tops(chm, height, _CFG)
    suppressed = next(
        item for item in tops.diagnostics["candidate_diagnostics"]
        if item["reason"] == "SUPPRESSED_BY_NEIGHBOUR"
    )
    assert suppressed["suppressed_by"]["tree_top_id"] == 1
    assert (
        suppressed["suppressed_by"]["suppression_radius_m"]
        == tops.accepted[1].suppression_radius_m
    )


def test_full_tree_segmentation_path_is_deterministic() -> None:
    """Tops, watershed labels, membership, and rejection evidence are repeatable."""
    xyz = np.concatenate((
        _smooth_crown(0.0, 0.0, 12.0, 2.4, 500),
        _smooth_crown(4.5, 0.0, 9.0, 2.0, 420),
    ))
    first = _run_tree_pipeline(xyz)
    second = _run_tree_pipeline(xyz)
    _, _, first_tops, first_crowns, first_membership = first
    _, _, second_tops, second_crowns, second_membership = second
    assert first_tops.accepted_count == second_tops.accepted_count
    assert [item.to_dict() for item in first_tops.accepted] == [
        item.to_dict() for item in second_tops.accepted
    ]
    assert first_tops.diagnostics["candidate_diagnostics"] == second_tops.diagnostics[
        "candidate_diagnostics"
    ]
    np.testing.assert_array_equal(
        first_crowns.region_label_by_cell, second_crowns.region_label_by_cell
    )
    np.testing.assert_array_equal(first_membership, second_membership)
    np.testing.assert_array_equal(
        np.flatnonzero(first_membership >= 0), np.flatnonzero(second_membership >= 0)
    )


def test_chm_rejects_empty_and_low_cells_explicitly() -> None:
    xyz = _smooth_crown(0.0, 0.0, 10.0, 2.0, 300)
    hag = xyz[:, 2].astype(np.float32)
    chm = build_span_chm(xyz, hag, _CFG)
    occupied = np.count_nonzero(chm.occupied)
    assert occupied > 0
    assert np.count_nonzero(np.isnan(chm.height_m)) == chm.shape[0] * chm.shape[1] - occupied


def test_leaf_scale_noise_rejected_as_final_tree() -> None:
    rng = np.random.default_rng(11)
    rows = []
    for _ in range(30):
        rows.append((rng.uniform(0, 10), rng.uniform(0, 10), rng.uniform(1.5, 3.0)))
    xyz = np.asarray(rows, dtype=np.float64)
    chm, smoothed, tops, crowns, membership = _run_tree_pipeline(xyz)
    # No accepted final tree from sparse leaf-scale noise.
    assert membership.max() < 0


def test_membership_requires_hag_consistency_with_surface() -> None:
    crown = _smooth_crown(0.0, 0.0, 10.0, 2.0, 400)
    # Inject a point far below the crown surface: must not be a member.
    xyz = np.vstack((crown, np.asarray([[0.0, 0.0, 0.2]])))
    chm, smoothed, tops, crowns, membership = _run_tree_pipeline(xyz)
    assert membership[-1] == -1


def _apex_rescue_fixture() -> tuple[SpanChm, np.ndarray, SpanCrownSegmentation]:
    raw = np.full((3, 3), 5.0, dtype=np.float32)
    raw[1, 1] = 12.0
    raw[0, 0] = 8.0
    smoothed = np.full((3, 3), 5.0, dtype=np.float32)
    smoothed[1, 1] = 8.0
    labels = np.ones((3, 3), dtype=np.int32)
    seeds = np.zeros((3, 3), dtype=bool)
    seeds[1, 1] = True
    chm = SpanChm(
        raw, np.ones_like(raw, dtype=bool), np.ones_like(raw, dtype=np.int32),
        (0.0, 0.0), 1.0, 1.5,
    )
    crowns = SpanCrownSegmentation(
        labels, np.asarray([9], dtype=np.int64), np.zeros((3, 3), dtype=bool),
        seeds, {},
    )
    return chm, smoothed, crowns


def test_marker_seed_apex_is_preserved_when_smoothing_underestimates_it() -> None:
    chm, smoothed, crowns = _apex_rescue_fixture()
    xyz = np.asarray([[1.5, 1.5, 12.0]], dtype=np.float64)
    membership = assign_span_tree_membership(
        xyz, xyz[:, 2].astype(np.float32), chm, smoothed, crowns,
        default_fast_baseline_config(tree_min_final_members=1),
    )
    assert membership.tolist() == [0]


def test_non_apex_cell_does_not_degenerate_to_raw_chm_upper_bound() -> None:
    chm, smoothed, crowns = _apex_rescue_fixture()
    # Raw CHM in this ordinary cell is 8 m, but a wrong-vertical 8 m point
    # must not become a member just because a raw-cell maximum exists.
    xyz = np.asarray([[0.5, 0.5, 8.0]], dtype=np.float64)
    membership = assign_span_tree_membership(
        xyz, xyz[:, 2].astype(np.float32), chm, smoothed, crowns,
        default_fast_baseline_config(tree_min_final_members=1),
    )
    assert membership.tolist() == [-1]


def test_non_apex_crown_cell_still_uses_smoothed_surface_constraint() -> None:
    chm, smoothed, crowns = _apex_rescue_fixture()
    xyz = np.asarray([[2.5, 2.5, 5.8], [2.5, 2.5, 6.3]], dtype=np.float64)
    membership = assign_span_tree_membership(
        xyz, xyz[:, 2].astype(np.float32), chm, smoothed, crowns,
        default_fast_baseline_config(tree_min_final_members=1),
    )
    assert membership.tolist() == [0, -1]
