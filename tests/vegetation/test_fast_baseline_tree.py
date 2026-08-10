"""Synthetic tests for the per-span tree pipeline (Parts 12-21, 42)."""

from __future__ import annotations

import numpy as np
from algorithms.spans.tree import (
    SpanChm,
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
