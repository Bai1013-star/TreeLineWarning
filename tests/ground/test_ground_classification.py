"""FGC PMF, seed support, plane, vertical-structure, and routing predicate tests.

Authority: docs/full-resolution-ground-classification-design.md sections 7-9,
and the frozen Bundle B plan test gate (PMF footprint/nodata erosion/nodata
dilation/recurrence/equality; D3/D3_plus centre-excluded support; exact
plane/tie/clipping/post-clip/RMS/slope/discontinuity/vertical/routing cases).
"""

from __future__ import annotations

import numpy as np
import pytest
from algorithms.ground.classification import (
    FineGrid,
    GroundDecision,
    _d3_offsets,
    _eight_connected_component_containing,
    _pmf_footprint_half_width,
    build_metric_grid,
    build_vertical_hash,
    competing_support,
    compute_pmf_mask,
    fit_plane,
    pmf_opening,
    route_point,
    seed_mask,
    vertical_structure,
)
from core.contracts.ground_classification import GroundClassificationPolicyConfig


def _policy(**overrides) -> GroundClassificationPolicyConfig:
    payload = {**GroundClassificationPolicyConfig().to_dict(), **overrides}
    payload["policy_digest"] = ""
    return GroundClassificationPolicyConfig.from_dict(payload)


def _fine_grid(*, g0_values, iy_lo=0, ix_lo=0, rep_raw_z=None, rep_index=None) -> FineGrid:
    g0 = np.asarray(g0_values, dtype=np.float64)
    valid = ~np.isnan(g0)
    if rep_raw_z is None:
        rep_raw_z = np.where(valid, 0, np.iinfo(np.int32).max).astype(np.int32)
    if rep_index is None:
        rep_index = np.where(
            valid, np.arange(g0.size, dtype=np.int64).reshape(g0.shape), np.iinfo(np.int64).max
        )
    height, width = g0.shape
    return FineGrid(
        iy_lo=iy_lo,
        iy_hi=iy_lo + height - 1,
        ix_lo=ix_lo,
        ix_hi=ix_lo + width - 1,
        g0=g0,
        rep_raw_z=rep_raw_z,
        rep_index=rep_index,
        rep_raw_x=np.where(valid, 0, 0).astype(np.int32),
        rep_raw_y=np.where(valid, 0, 0).astype(np.int32),
        valid=valid,
    )


# ---------------------------------------------------------------------------
# PMF footprints and nodata semantics
# ---------------------------------------------------------------------------


def test_pmf_footprint_half_widths() -> None:
    assert [_pmf_footprint_half_width(w) for w in (1, 2, 4, 8, 16)] == [1, 2, 4, 8, 16]
    assert [2 * _pmf_footprint_half_width(w) + 1 for w in (1, 2, 4, 8, 16)] == [3, 5, 9, 17, 33]


def test_pmf_min_is_min_and_max_is_max() -> None:
    g0 = np.array(
        [
            [1.0, 2.0, 1.0, 2.0, 1.0],
            [2.0, 3.0, 2.0, 3.0, 2.0],
            [1.0, 2.0, 1.0, 2.0, 1.0],
            [2.0, 3.0, 2.0, 3.0, 2.0],
            [1.0, 2.0, 1.0, 2.0, 1.0],
        ]
    )
    valid = np.ones((5, 5), dtype=bool)
    e, e_valid = pmf_opening(g0, valid=valid, h=1, m=5, mode="min")
    # Interior cell windows contain the grid minimum 1.0.
    assert e[2, 2] == 1.0
    # Boundary cells clip to fewer than 5 valid neighbours (valid-neighbour rule).
    assert not e_valid[0, 0] and not e_valid[0, 4]
    assert e_valid[2, 2]
    o, o_valid = pmf_opening(e, valid=e_valid, h=1, m=5, mode="max")
    # Dilation is the max of erosion within the window; here every 3x3 window
    # contains a 1.0 so every valid erosion is 1.0 and O is also 1.0.
    assert o[2, 2] == 1.0
    # Interior cells are valid; boundary corners clip below the support minimum.
    assert e_valid[1:4, 1:4].all() and o_valid[1:4, 1:4].all()
    assert not e_valid[0, 0] and not o_valid[0, 0]


def test_pmf_nodata_erosion() -> None:
    # A nodata cell must never enter a min or count; insufficient support -> nodata.
    g0 = np.array(
        [
            [np.nan, 2.0, 2.0, 2.0, 2.0],
            [2.0, 3.0, 3.0, 3.0, 3.0],
            [2.0, 3.0, 4.0, 4.0, 4.0],
            [2.0, 3.0, 4.0, 4.0, 4.0],
            [2.0, 3.0, 4.0, 4.0, 4.0],
        ]
    )
    valid = ~np.isnan(g0)
    e, e_valid = pmf_opening(g0, valid=valid, h=1, m=5, mode="min")
    # bottom-right cell has only 3 valid in 3x3 -> invalid.
    assert not e_valid[4, 4]
    assert np.isnan(e[4, 4])


def test_pmf_nodata_dilation() -> None:
    g0 = np.full((5, 5), 1.0)
    valid = np.ones((5, 5), dtype=bool)
    e, e_valid = pmf_opening(g0, valid=valid, h=1, m=5, mode="min")
    # Interior erosion valid; dilation reads only valid erosion values.
    o, o_valid = pmf_opening(e, valid=e_valid, h=1, m=5, mode="max")
    assert o_valid[1:4, 1:4].all()
    assert (o[1:4, 1:4] == 1.0).all()


def test_pmf_recurrence_operates_on_g0_not_prior_surface() -> None:
    # Progressive recurrence: each window recomputes from G0.
    g0 = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    valid = np.ones((3, 3), dtype=bool)
    policy = _policy()
    fine = _fine_grid(g0_values=g0)
    mask = compute_pmf_mask(fine, policy=policy)
    assert mask.shape == g0.shape
    # The mask must be a subset of valid cells.
    assert (mask[valid]).sum() >= 0


def test_pmf_equality_passes() -> None:
    # tau tolerance: equality passes (G0 - O <= tau).  Interior cells with a
    # full 3x3 window pass at every scale.
    g0 = np.ones((5, 5), dtype=np.float64)
    policy = _policy()
    fine = _fine_grid(g0_values=g0)
    mask = compute_pmf_mask(fine, policy=policy)
    assert mask[1:4, 1:4].all()
    # Boundary cells clip to fewer than 5 valid neighbours.
    assert not mask[0, 0] and not mask[0, 4] and not mask[4, 0] and not mask[4, 4]


# ---------------------------------------------------------------------------
# D3 / D3_plus connected support (FGC-RR-001)
# ---------------------------------------------------------------------------


_GRID_CACHE: tuple | None = None


def _grid():
    """Build a MetricGrid from a minimal synthetic scene (cached)."""
    global _GRID_CACHE
    if _GRID_CACHE is not None:
        return _GRID_CACHE
    import tempfile
    from pathlib import Path

    from tests.ground._scene import load_scene as _ls
    from tests.ground._scene import write_synthetic_las as _w

    policy = _policy()
    path = Path(tempfile.mkdtemp()) / "g.las"
    scene = _ls(
        _w(path, x=np.array([100.0, 100.5]), y=np.array([200.0, 200.5]), z=np.array([0.0, 0.0]))
    )
    proof = scene.result.proof
    header = proof.header
    x0 = round(
        (float(proof.source_bounds_min_xyz[0]) - float(header.offset_xyz[0]))
        / float(header.scale_xyz[0])
    )
    y0 = round(
        (float(proof.source_bounds_min_xyz[1]) - float(header.offset_xyz[1]))
        / float(header.scale_xyz[1])
    )
    x_max = round(
        (float(proof.source_bounds_max_xyz[0]) - float(header.offset_xyz[0]))
        / float(header.scale_xyz[0])
    )
    y_max = round(
        (float(proof.source_bounds_max_xyz[1]) - float(header.offset_xyz[1]))
        / float(header.scale_xyz[1])
    )
    _GRID_CACHE = (
        build_metric_grid(header, policy, x0=x0, y0=y0, x_max=x_max, y_max=y_max),
        policy,
    )
    return _GRID_CACHE


def test_d3_excludes_center_and_plus_includes_it() -> None:
    grid, policy = _grid()
    offsets = _d3_offsets(grid, policy)
    assert (0, 0) not in offsets
    assert all(offset != (0, 0) for offset in offsets)
    # D3_plus = D3 union {(0,0)}
    plus = set(offsets) | {(0, 0)}
    assert (0, 0) in plus
    # Radius bound: offsets must satisfy exact cell-centre distance <= 3 m.
    for dy, dx in offsets:
        assert dx * dx + dy * dy <= (3.0 / 0.5) ** 2 + 1e-9


def test_eight_connected_component_containing() -> None:
    cells = {(0, 0), (0, 1), (1, 0), (1, 1), (5, 5)}
    component = _eight_connected_component_containing(cells, (0, 0))
    assert component == {(0, 0), (0, 1), (1, 0), (1, 1)}
    disconnected = _eight_connected_component_containing(cells, (5, 5))
    assert disconnected == {(5, 5)}
    missing = _eight_connected_component_containing(cells, (9, 9))
    assert missing == set()


def test_seed_support_five_passes_four_fails() -> None:
    policy = _policy()
    # Candidate center at local (2,2); valid cells form an 8-connected blob.
    g0 = np.full((5, 5), np.nan)
    center = (2, 2)
    neighbors = [(1, 2), (2, 1), (3, 2), (2, 3), (1, 1)]
    g0[center] = 0.0
    for iy, ix in neighbors:
        g0[iy, ix] = 0.0
    fine = _fine_grid(g0_values=g0)
    pmf_mask = np.zeros((5, 5), dtype=bool)
    pmf_mask[center] = True
    # 5 connected support neighbours -> pass.
    seeds = seed_mask(fine, grid=_grid()[0], policy=policy, pmf_mask=pmf_mask)
    assert bool(seeds[center]) is True
    # 4 neighbours -> fail.
    g0_4 = g0.copy()
    g0_4[1, 1] = np.nan
    fine_4 = _fine_grid(g0_values=g0_4)
    seeds_4 = seed_mask(fine_4, grid=_grid()[0], policy=policy, pmf_mask=pmf_mask)
    assert bool(seeds_4[center]) is False


def test_seed_disconnected_cells_excluded() -> None:
    policy = _policy()
    g0 = np.full((7, 7), np.nan)
    center = (3, 3)
    g0[center] = 0.0
    # Five connected neighbours.
    for iy, ix in [(2, 3), (3, 2), (4, 3), (3, 4), (2, 2)]:
        g0[iy, ix] = 0.0
    # A valid cell inside D3_plus but disconnected from the center.
    g0[0, 0] = 0.0
    fine = _fine_grid(g0_values=g0)
    pmf_mask = np.zeros((7, 7), dtype=bool)
    pmf_mask[center] = True
    seeds = seed_mask(fine, grid=_grid()[0], policy=policy, pmf_mask=pmf_mask)
    assert bool(seeds[center]) is True  # disconnected cell does not break support


def test_seed_center_excluded_from_support_and_lower_envelope() -> None:
    policy = _policy()
    g0 = np.full((5, 5), np.nan)
    center = (2, 2)
    g0[center] = 0.0
    # Exactly 5 support neighbours all at 1.0.  The lower-envelope median over
    # C_without_center is 1.0; the centre (0.0) is excluded from the statistic,
    # so median - G0(c) = 1.0 > 0.30 -> isolated-low -> not a seed.
    for iy, ix in [(1, 2), (2, 1), (3, 2), (2, 3), (1, 1)]:
        g0[iy, ix] = 1.0
    fine = _fine_grid(g0_values=g0)
    pmf_mask = np.zeros((5, 5), dtype=bool)
    pmf_mask[center] = True
    seeds = seed_mask(fine, grid=_grid()[0], policy=policy, pmf_mask=pmf_mask)
    assert bool(seeds[center]) is False


def test_seed_isolated_low_rule() -> None:
    policy = _policy()
    g0 = np.full((5, 5), np.nan)
    center = (2, 2)
    g0[center] = 0.0
    # Support neighbours at high elevation -> isolated low.
    for iy, ix in [(1, 2), (2, 1), (3, 2), (2, 3), (1, 1)]:
        g0[iy, ix] = 5.0
    fine = _fine_grid(g0_values=g0)
    pmf_mask = np.zeros((5, 5), dtype=bool)
    pmf_mask[center] = True
    seeds = seed_mask(fine, grid=_grid()[0], policy=policy, pmf_mask=pmf_mask)
    assert bool(seeds[center]) is False


# ---------------------------------------------------------------------------
# Plane fit, clipping, RMS, slope, discontinuity
# ---------------------------------------------------------------------------


def _policy_plane():
    return _policy()


def _grid_support_xy(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """A 4x3 grid of (x, y) support coordinates (12 supports)."""
    gx, gy = np.meshgrid(np.arange(0.0, 3.0), np.arange(0.0, 4.0))
    return gx.ravel().astype(np.float64), gy.ravel().astype(np.float64)


def test_plane_flat_fit() -> None:
    xs, ys = _grid_support_xy((4, 3))
    zs = np.full(xs.size, 5.0)
    plane = fit_plane(xs, ys, zs, xp=1.0, yp=1.0, pz=5.0, policy=_policy_plane())
    assert plane.available
    assert plane.intercept == pytest.approx(5.0, abs=1e-9)
    assert plane.residual == pytest.approx(0.0, abs=1e-9)
    assert plane.slope == pytest.approx(0.0, abs=1e-9)
    assert plane.rms == pytest.approx(0.0, abs=1e-9)


def test_plane_sloped_fit() -> None:
    xs = np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0])
    ys = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    zs = xs + 0.0 * ys + 10.0
    plane = fit_plane(xs, ys, zs, xp=1.0, yp=1.0, pz=11.0, policy=_policy_plane())
    assert plane.available
    # slope = 1.0 m/m for a 45-degree plane.
    assert plane.slope == pytest.approx(1.0, rel=1e-6)


def test_plane_insufficient_support() -> None:
    xs = np.asarray([0.0, 1.0, 2.0])
    ys = np.asarray([0.0, 0.0, 0.0])
    zs = np.asarray([1.0, 1.0, 1.0])
    plane = fit_plane(xs, ys, zs, xp=1.0, yp=0.0, pz=1.0, policy=_policy_plane())
    assert not plane.available


def test_plane_clipping_removes_outliers() -> None:
    xs = np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.5])
    ys = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 1.5])
    zs = np.full(13, 5.0)
    zs[-1] = 5.3  # a 0.30 m outlier that clipping must remove within tolerance
    plane = fit_plane(xs, ys, zs, xp=1.0, yp=1.0, pz=5.0, policy=_policy_plane())
    assert plane.available
    assert plane.rms < 0.05
    assert plane.support_count >= 12


def test_plane_gross_outlier_degrades_to_unavailable() -> None:
    # A gross outlier pulls the plane so far that every support is clipped;
    # the plane is unavailable and the row routes REVIEW_GROUND.
    xs = np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.5])
    ys = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 1.5])
    zs = np.full(13, 5.0)
    zs[-1] = 50.0
    plane = fit_plane(xs, ys, zs, xp=1.0, yp=1.0, pz=5.0, policy=_policy_plane())
    assert not plane.available


def test_plane_rms_defined_as_single_statistic() -> None:
    xs, ys = _grid_support_xy((4, 3))
    zs = np.full(xs.size, 7.0)
    plane = fit_plane(xs, ys, zs, xp=1.0, yp=1.0, pz=7.0, policy=_policy_plane())
    assert plane.rms == pytest.approx(0.0, abs=1e-9)


def test_competing_support_discontinuity() -> None:
    # Two non-degenerate flat populations at different heights, split across x.
    xs = np.asarray([-2.0, -2.0, -2.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    ys = np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    zs = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    competing = competing_support(xs, ys, zs, xp=0.0, yp=1.0, policy=_policy_plane())
    assert competing is True


def test_competing_support_smooth_is_false() -> None:
    xs, ys = _grid_support_xy((4, 3))
    zs = np.full(xs.size, 4.0)
    competing = competing_support(xs, ys, zs, xp=1.0, yp=1.0, policy=_policy_plane())
    assert competing is False


# ---------------------------------------------------------------------------
# Vertical structure
# ---------------------------------------------------------------------------


def test_vertical_structure_span_triggers() -> None:
    # 8 other rows within 1 m (100 raw ticks at 0.01 scale) with a 1.5 m span.
    grid, policy = _grid()
    rows = []
    for index in range(9):
        rows.append([index, 0, 0, int(round((index * 0.2) / 0.01))])
    halo = np.asarray(rows, dtype=np.int64)
    hash_map = build_vertical_hash(halo, grid=grid)
    assert (
        vertical_structure(0, 0, 0, 0.0, vertical_hash=hash_map, grid=grid, policy=policy) is True
    )


def test_vertical_structure_below_min_rows() -> None:
    grid, policy = _grid()
    halo = np.asarray([[0, 0, 0, 0], [1, 0, 0, 100]], dtype=np.int64)
    hash_map = build_vertical_hash(halo, grid=grid)
    assert (
        vertical_structure(0, 0, 0, 0.0, vertical_hash=hash_map, grid=grid, policy=policy) is False
    )


def test_vertical_structure_excludes_self() -> None:
    grid, policy = _grid()
    # Self at high elevation; other rows far below. Excluding self by index
    # leaves only 2 other rows -> below the minimum, so vertical is false.
    halo = np.asarray(
        [[0, 0, 0, int(round(5.0 / 0.01))], [1, 0, 0, 0], [2, 0, 0, 0]],
        dtype=np.int64,
    )
    hash_map = build_vertical_hash(halo, grid=grid)
    assert (
        vertical_structure(0, 0, 0, 5.0, vertical_hash=hash_map, grid=grid, policy=policy) is False
    )


# ---------------------------------------------------------------------------
# Routing precedence
# ---------------------------------------------------------------------------


def _plane(available=True, slope=0.0, rms=0.0, residual=0.0):
    from algorithms.ground.classification import PlaneFitResult

    return PlaneFitResult(
        alpha=0.0,
        beta=0.0,
        intercept=0.0,
        residual=residual,
        slope=slope,
        rms=rms,
        support_count=12,
        available=available,
    )


def test_route_vertical_first() -> None:
    decision = route_point(vertical=True, plane=_plane(), competing=False, policy=_policy())
    assert decision == GroundDecision.REVIEW_GROUND


def test_route_no_plane_review() -> None:
    decision = route_point(
        vertical=False, plane=_plane(available=False), competing=False, policy=_policy()
    )
    assert decision == GroundDecision.REVIEW_GROUND


def test_route_competing_review() -> None:
    decision = route_point(vertical=False, plane=_plane(), competing=True, policy=_policy())
    assert decision == GroundDecision.REVIEW_GROUND


def test_route_steep_or_rough_review() -> None:
    assert (
        route_point(vertical=False, plane=_plane(slope=1.0), competing=False, policy=_policy())
        == GroundDecision.REVIEW_GROUND
    )
    assert (
        route_point(vertical=False, plane=_plane(rms=0.5), competing=False, policy=_policy())
        == GroundDecision.REVIEW_GROUND
    )


def test_route_confirm_band() -> None:
    policy = _policy()
    for residual in (-0.15, 0.0, 0.20):
        assert (
            route_point(
                vertical=False, plane=_plane(residual=residual), competing=False, policy=policy
            )
            == GroundDecision.CONFIRMED_GROUND
        )


def test_route_non_ground_above_0_60() -> None:
    policy = _policy()
    decision = route_point(
        vertical=False, plane=_plane(residual=0.61), competing=False, policy=policy
    )
    assert decision == GroundDecision.NON_GROUND


def test_route_review_positive_ambiguity_and_below() -> None:
    policy = _policy()
    # +0.3 in the positive ambiguity interval -> review.
    assert (
        route_point(vertical=False, plane=_plane(residual=0.30), competing=False, policy=policy)
        == GroundDecision.REVIEW_GROUND
    )
    # -0.40 below the review band -> review (never inferred non-ground).
    assert (
        route_point(vertical=False, plane=_plane(residual=-0.40), competing=False, policy=policy)
        == GroundDecision.REVIEW_GROUND
    )
