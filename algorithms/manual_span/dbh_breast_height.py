"""Measure the actual raw 1.30 m slab, never a median of other heights.

The caller establishes raw source provenance, original ground and tree identity.
Adjacent sections/axes may locate the stem, but do not supply its diameter.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from algorithms.manual_span.dbh_recovery import DBH_RECOVERY_CONFIG, _section_circle_fit

VERSION = "RAW_LAS_1P30_SECTION_V1"
TARGET_M = 1.30
HALF_THICKNESS_M = 0.05


def _connected_components_xy(points: np.ndarray, maximum_gap_m: float = 0.10):
    """Return deterministic XY components without introducing synthetic points."""
    if not len(points):
        return []
    tree = cKDTree(points)
    neighbours = tree.query_ball_point(points, maximum_gap_m)
    seen = np.zeros(len(points), dtype=bool)
    components = []
    for start in range(len(points)):
        if seen[start]:
            continue
        stack, member = [start], []
        seen[start] = True
        while stack:
            current = stack.pop()
            member.append(current)
            for adjacent in neighbours[current]:
                if not seen[adjacent]:
                    seen[adjacent] = True
                    stack.append(adjacent)
        components.append(np.asarray(sorted(member), dtype=np.int64))
    return components


def _robust_circle_or_spread(points: np.ndarray):
    """Fit a circle to measured XY points, with a measured-spread fallback.

    The fallback is deliberately labelled as such.  It derives the diameter
    only from the original section coordinates and never from crown width,
    allometry, a previous result, or a fabricated default.
    """
    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(points) < 3:
        return None
    active = np.arange(len(points))
    center = np.median(points, axis=0)
    radius = None
    for _ in range(5):
        sample = points[active]
        matrix = np.column_stack((2.0 * sample[:, 0], 2.0 * sample[:, 1], np.ones(len(sample))))
        rhs = np.square(sample[:, 0]) + np.square(sample[:, 1])
        try:
            solution, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
        except np.linalg.LinAlgError:
            break
        if rank < 3:
            break
        center = solution[:2]
        distances = np.linalg.norm(points - center, axis=1)
        radius = float(np.median(distances[active]))
        residual = np.abs(distances - radius)
        median = float(np.median(residual[active]))
        mad = float(np.median(np.abs(residual[active] - median)))
        tolerance = max(0.008, median + 3.5 * max(mad, 0.002))
        updated = np.flatnonzero(residual <= tolerance)
        if len(updated) < 3 or np.array_equal(updated, active):
            break
        active = updated
    if radius is not None and np.isfinite(radius) and 0.015 <= radius <= 0.75:
        residual = np.abs(np.linalg.norm(points[active] - center, axis=1) - radius)
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        return {
            "center": center,
            "radius_m": radius,
            "active": active,
            "rmse_m": rmse,
            "method": "ROBUST_CIRCLE",
        }

    # A partial arc can make the algebraic circle singular.  Its robust XY
    # span is still a direct physical section measurement, so publish it with
    # visibly lower quality and larger uncertainty instead of inventing a DBH.
    lower, upper = np.percentile(points, (5, 95), axis=0)
    diameter = float(max(upper - lower))
    if not np.isfinite(diameter) or not 0.03 <= diameter <= 1.50:
        return None
    return {
        "center": np.median(points, axis=0),
        "radius_m": diameter / 2.0,
        "active": np.arange(len(points)),
        "rmse_m": float(np.std(np.linalg.norm(points - np.median(points, axis=0), axis=1))),
        "method": "ROBUST_MEASURED_SPREAD",
    }


def fit_original_tree_member_breast_height(xyz, original_index=None):
    """Measure DBH from one tree's full-resolution original LAS membership.

    Ground is exactly the lowest Z of this original tree, as requested by the
    desktop workflow.  The target is therefore ``min(original Z) + 1.30 m``.
    Adaptive bands compensate for sparse acquisition, but every selected point
    remains an original LAS point with its original index.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3 or not np.isfinite(xyz).all():
        raise ValueError("ORIGINAL_TREE_MEMBER_DBH_INPUT_INVALID")
    if original_index is None:
        original_index = np.arange(len(xyz), dtype=np.int64)
    original_index = np.asarray(original_index)
    if original_index.shape != (len(xyz),) or original_index.dtype.kind not in "iu":
        raise ValueError("ORIGINAL_TREE_MEMBER_INDEX_INVALID")

    ground_z = float(np.min(xyz[:, 2]))
    hag = xyz[:, 2] - ground_z
    target_z = ground_z + TARGET_M
    low = (hag >= 0.05) & (hag <= 0.80)
    base_center = np.median(xyz[low, :2], axis=0) if np.count_nonzero(low) >= 3 else np.median(xyz[:, :2], axis=0)

    result = None
    selected_positions = None
    selected_half_band = None
    for half_band in (0.05, 0.10, 0.15, 0.25, 0.40):
        positions = np.flatnonzero(np.abs(xyz[:, 2] - target_z) <= half_band + 1.0e-10)
        if len(positions) < 3:
            continue
        # Keep the component search bounded and centred on the low trunk.  A
        # 1.0 m radius still accommodates the full accepted DBH range while
        # excluding remote crown/roadside objects accidentally present in the
        # tree membership.  Deterministic thinning affects speed only; every
        # retained coordinate and index still comes from the original LAS.
        radial = np.linalg.norm(xyz[positions, :2] - base_center, axis=1)
        local_positions = positions[radial <= 1.0]
        if len(local_positions) >= 3:
            positions = local_positions
        if len(positions) > 20000:
            stride = int(np.ceil(len(positions) / 20000.0))
            positions = positions[::stride]
        section_xy = xyz[positions, :2]
        components = _connected_components_xy(section_xy)
        plausible = [component for component in components if len(component) >= 3]
        if plausible:
            component = min(
                plausible,
                key=lambda item: (
                    float(np.linalg.norm(np.median(section_xy[item], axis=0) - base_center)),
                    -len(item),
                    int(item[0]),
                ),
            )
            positions = positions[component]
        else:
            order = np.argsort(np.linalg.norm(section_xy - base_center, axis=1), kind="stable")
            positions = positions[order[: min(64, len(order))]]
        fitted = _robust_circle_or_spread(xyz[positions, :2])
        if fitted is None:
            continue
        result, selected_positions, selected_half_band = fitted, positions, half_band
        if fitted["method"] == "ROBUST_CIRCLE" and len(positions) >= 8:
            break
    if result is None or selected_positions is None:
        return {
            "dbh_cm": None,
            "dbh_method": "UNAVAILABLE",
            "dbh_quality": "UNAVAILABLE",
            "dbh_fit_status": "ORIGINAL_TREE_1P30_SECTION_GEOMETRY_UNAVAILABLE",
            "selected_input_indices": [],
            "selected_original_indices": [],
            "ground_local_z_m": ground_z,
            "measurement_local_z_m": target_z,
            "target_hag_m": TARGET_M,
        }

    active_positions = selected_positions[np.asarray(result["active"], dtype=np.int64)]
    diameter_cm = float(result["radius_m"] * 200.0)
    spread_fallback = result["method"] == "ROBUST_MEASURED_SPREAD"
    return {
        "dbh_cm": diameter_cm,
        "dbh_method": "RAW_GROUND_BH_SECTION",
        "dbh_quality": "LOW_CONFIDENCE_ESTIMATED" if spread_fallback else "ESTIMATED",
        "dbh_point_count": int(len(active_positions)),
        "dbh_fit_rmse": result["rmse_m"],
        "dbh_slice_count": 1,
        "dbh_fit_status": (
            "PASS_ORIGINAL_TREE_LOWEST_PLUS_1P30_MEASURED_SPREAD"
            if spread_fallback
            else "PASS_ORIGINAL_TREE_LOWEST_PLUS_1P30_ADAPTIVE_CIRCLE"
        ),
        "dbh_uncertainty_cm": float(max(1.0 if not spread_fallback else 2.5, 200.0 * result["rmse_m"])),
        "selected_input_indices": active_positions.tolist(),
        "selected_original_indices": original_index[active_positions].astype(np.int64).tolist(),
        "center_xy_local_m": np.asarray(result["center"], dtype=float).tolist(),
        "target_hag_m": TARGET_M,
        "section_bounds_hag_m": [TARGET_M - selected_half_band, TARGET_M + selected_half_band],
        "selected_hag_bounds_m": [float(np.min(hag[active_positions])), float(np.max(hag[active_positions]))],
        "ground_local_z_m": ground_z,
        "measurement_local_z_m": target_z,
        "vertical_half_band_m": selected_half_band,
        "diameter_derived_from": "DIRECT_ORIGINAL_LAS_TREE_MEMBER_SECTION_AT_1P30",
        "fit_kind": result["method"],
    }


def fit_raw_breast_height_section(xyz, hag, center_xy, radius_m):
    xyz = np.asarray(xyz, dtype=np.float64)
    hag = np.asarray(hag, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    result = dict(
        dbh_cm=None, dbh_method="UNAVAILABLE", dbh_quality="UNAVAILABLE",
        dbh_point_count=0, dbh_fit_rmse=None, dbh_slice_count=0,
        dbh_fit_status="RAW_1P30_SECTION_INSUFFICIENT_POINTS",
        dbh_uncertainty_cm=None, dbh_model_version=VERSION,
        selected_input_indices=[], target_hag_m=TARGET_M,
        section_bounds_hag_m=[TARGET_M - HALF_THICKNESS_M, TARGET_M + HALF_THICKNESS_M],
        diameter_derived_from="ACTUAL_RAW_1P30_SLAB_CIRCLE_NOT_INTERPOLATED",
    )
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or hag.shape != (len(xyz),)
            or center.shape != (2,) or not np.isfinite(xyz).all()
            or not np.isfinite(hag).all() or not np.isfinite(center).all()
            or not np.isfinite(radius_m) or not 0.025 <= radius_m <= 0.75):
        raise ValueError("RAW_1P30_SECTION_INPUT_INVALID")
    # Only raw points within the physical 1.30 +/- 0.05 m slab can contribute.
    # The axis is a search hint, not a fabricated cylinder or diameter source.
    near = np.abs(hag - TARGET_M) <= HALF_THICKNESS_M + 1e-10
    near &= np.linalg.norm(xyz[:, :2] - center, axis=1) <= radius_m + 0.08
    positions = np.flatnonzero(near)
    result["raw_section_input_point_count"] = len(positions)
    if len(positions) < DBH_RECOVERY_CONFIG.minimum_points_per_section:
        return result
    points = np.unique(xyz[positions, :2], axis=0)
    diagnostic, fitted_center, radius = _section_circle_fit(points, DBH_RECOVERY_CONFIG)
    result["section_fit"] = diagnostic
    if not diagnostic["accepted"]:
        result["dbh_fit_status"] = "RAW_1P30_SECTION_" + diagnostic["reason"]
        return result
    if np.linalg.norm(fitted_center - center) > 0.10:
        result["dbh_fit_status"] = "RAW_1P30_SECTION_SWITCHED_STEM"
        return result
    tolerance = min(DBH_RECOVERY_CONFIG.maximum_inlier_tolerance_m,
                    max(DBH_RECOVERY_CONFIG.minimum_inlier_tolerance_m,
                        DBH_RECOVERY_CONFIG.radial_tolerance_fraction * radius))
    residual = np.abs(np.linalg.norm(xyz[positions, :2] - fitted_center, axis=1) - radius)
    positions = positions[residual <= tolerance]
    result.update(
        dbh_cm=float(radius * 200), dbh_method="RAW_GROUND_BH_SECTION",
        dbh_quality="ESTIMATED", dbh_point_count=len(positions),
        dbh_fit_rmse=diagnostic["fit_rmse_m"], dbh_slice_count=1,
        dbh_fit_status="PASS_ACTUAL_RAW_GROUND_1P30_SECTION",
        dbh_uncertainty_cm=float(max(0.5, 200 * diagnostic["fit_rmse_m"])),
        selected_input_indices=positions.tolist(), center_xy_local_m=fitted_center.tolist(),
        selected_hag_bounds_m=[float(hag[positions].min()), float(hag[positions].max())],
    )
    return result
