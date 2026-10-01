"""Recover a densely observed one-sided stem only from a coherent height series.

A single incomplete circumference never establishes DBH.  This path needs
independent circles in thirteen adjacent raw-point sections, a straight axis,
and a smooth radius profile before the normal crown/ownership gates run.
"""

from __future__ import annotations

from math import sqrt

import numpy as np

from algorithms.manual_span.dbh_raw_stem import _fit_hag_line
from algorithms.manual_span.dbh_recovery import DBH_RECOVERY_CONFIG, _section_circle_fit

FIRST_HAG_M = 0.85
SECTION_M = 0.15
SECTION_COUNT = 13
MIN_POINTS = 50
MIN_INLIER_FRACTION = 0.80
MIN_ARC_DEG = 120.0
MIN_SECTORS = 5
MAX_SECTION_RMSE_M = 0.008
MAX_AXIS_RESIDUAL_M = 0.025
MAX_RADIUS_PROFILE_RESIDUAL_M = 0.012
MAX_BH_RADIUS_SPREAD_M = 0.010


def _recover_series(
    xyz, hag, original_index, *, first_hag, section_count, min_fraction,
    min_arc, min_sectors, bh_sections, mode,
):
    """Return a proved axis, track and DBH fit, or detailed rejection evidence."""

    xyz = np.asarray(xyz, dtype=np.float64)
    hag = np.asarray(hag, dtype=np.float64)
    original_index = np.asarray(original_index)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or hag.shape != (len(xyz),):
        return None, {"status": "INVALID_INPUT"}
    if original_index.shape != (len(xyz),):
        return None, {"status": "INVALID_INDEX"}
    if not np.isfinite(xyz).all() or not np.isfinite(hag).all():
        return None, {"status": "NONFINITE_INPUT"}

    sections = []
    centers = []
    heights = []
    radii = []
    support_positions = []
    for number in range(section_count):
        low = first_hag + number * SECTION_M
        high = low + SECTION_M
        mask = (hag >= low) & (hag < high if number < section_count - 1 else hag <= high)
        positions = np.flatnonzero(mask)
        xy = np.unique(xyz[positions, :2], axis=0)
        if len(xy) < MIN_POINTS:
            return None, {
                "status": f"{mode}_SECTION_SPARSE", "section": number, "count": len(xy)
            }
        fitted, _, _ = _section_circle_fit(xy, DBH_RECOVERY_CONFIG)
        summary = {
            "section": number,
            "center_hag_m": (low + high) / 2,
            "input_count": len(positions),
            "inlier_fraction": fitted["inlier_fraction"],
            "angular_span_deg": fitted["angular_span_deg"],
            "occupied_sector_count": fitted["occupied_sector_count"],
            "radius_m": fitted["radius_m"],
            "fit_rmse_m": fitted["fit_rmse_m"],
            "fit_status": fitted["status"],
        }
        sections.append(summary)
        if (
            fitted["radius_m"] is None
            or fitted["center_x_m"] is None
            or fitted["center_y_m"] is None
            or fitted["fit_rmse_m"] is None
            or fitted["angular_span_deg"] is None
            or fitted["inlier_fraction"] < min_fraction
            or fitted["angular_span_deg"] < min_arc
            or fitted["occupied_sector_count"] < min_sectors
            or fitted["fit_rmse_m"] > MAX_SECTION_RMSE_M
        ):
            return None, {
                "status": f"{mode}_SECTION_UNPROVEN", "section": number, "sections": sections
            }
        center_xy = np.asarray([fitted["center_x_m"], fitted["center_y_m"]])
        radius = float(fitted["radius_m"])
        tolerance = min(
            DBH_RECOVERY_CONFIG.maximum_inlier_tolerance_m,
            max(
                DBH_RECOVERY_CONFIG.minimum_inlier_tolerance_m,
                DBH_RECOVERY_CONFIG.radial_tolerance_fraction * radius,
            ),
        )
        radial = np.abs(np.linalg.norm(xyz[positions, :2] - center_xy, axis=1) - radius)
        selected = positions[radial <= tolerance]
        if len(selected) < MIN_POINTS:
            return None, {
                "status": f"{mode}_SOURCE_SUPPORT_SPARSE",
                "section": number,
                "sections": sections,
            }
        centers.append([*center_xy, float(np.median(xyz[selected, 2]))])
        heights.append(float(np.median(hag[selected])))
        radii.append(radius)
        if number < bh_sections:
            support_positions.append(selected)

    points = np.asarray(centers)
    hs = np.asarray(heights)
    radii_array = np.asarray(radii)
    line = _fit_hag_line(points, hs, np.arange(section_count))
    if line is None:
        return None, {"status": f"{mode}_AXIS_FIT_FAILED", "sections": sections}
    anchor, unit, slope = line
    delta = points - anchor
    axis_residual = np.linalg.norm(delta - np.outer(delta @ unit, unit), axis=1)
    tilt = float(np.degrees(np.arccos(np.clip(unit[2], -1, 1))))
    if np.max(axis_residual) > MAX_AXIS_RESIDUAL_M or tilt > DBH_RECOVERY_CONFIG.maximum_tilt_deg:
        return None, {
            "status": f"{mode}_AXIS_INCONSISTENT",
            "max_axis_residual_m": float(np.max(axis_residual)),
            "tilt_deg": tilt,
            "sections": sections,
        }
    radius_slope, radius_at_bh = np.polyfit(hs - 1.30, radii_array, 1)
    radius_residual = radii_array - (radius_at_bh + radius_slope * (hs - 1.30))
    if (
        np.max(np.abs(radius_residual)) > MAX_RADIUS_PROFILE_RESIDUAL_M
        or (
            np.max(radii_array[:bh_sections]) - np.min(radii_array[:bh_sections])
            > MAX_BH_RADIUS_SPREAD_M
        )
        or abs(radius_slope) > 0.035
        or not (
            DBH_RECOVERY_CONFIG.diameter_min_m
            <= 2 * radius_at_bh
            <= DBH_RECOVERY_CONFIG.diameter_max_m
        )
    ):
        return None, {
            "status": f"{mode}_RADIUS_PROFILE_INCONSISTENT",
            "max_radius_residual_m": float(np.max(np.abs(radius_residual))),
            "bh_radius_spread_m": float(np.ptp(radii_array[:bh_sections])),
            "radius_slope_m_per_m": float(radius_slope),
            "sections": sections,
        }

    selected_positions = np.unique(np.concatenate(support_positions))
    rmse = sqrt(float(np.mean([s["fit_rmse_m"] ** 2 for s in sections[:bh_sections]])))
    # The unseen back of the stem creates model uncertainty even with dense points.
    uncertainty_cm = max(1.0, 200 * float(np.max(np.abs(radius_residual))), 200 * rmse)
    axis = {
        "bh_reference_xyz": anchor.tolist(),
        "direction_xyz": unit.tolist(),
        "slope_xyz_per_hag": slope.tolist(),
        "radius_m": float(radius_at_bh),
        "tilt_deg": tilt,
        "max_center_residual_m": float(np.max(axis_residual)),
        "anchor_status": f"FITTED_FROM_{section_count}_{mode}_SECTIONS",
        "proposal_source": f"RAW_{mode}_HEIGHT_SERIES",
    }
    track = {
        "status": "PASS",
        "coverage": 1.0,
        "center_consistency": max(0.0, 1 - float(np.mean(axis_residual)) / MAX_AXIS_RESIDUAL_M),
        "cylindrical_support": float(np.mean([s["inlier_fraction"] for s in sections])),
        "refined_axis": axis,
        "partial_arc_sections": sections,
        "original_index": original_index[selected_positions],
    }
    fit = {
        "dbh_cm": float(200 * radius_at_bh),
        "dbh_method": "RECOVERED_STEM",
        "dbh_quality": "ESTIMATE",
        "dbh_point_count": len(selected_positions),
        "dbh_fit_rmse": rmse,
        "dbh_slice_count": bh_sections,
        "dbh_fit_status": "PASS",
        "dbh_uncertainty_cm": uncertainty_cm,
        "dbh_model_version": f"DBH_RECOVERY_V3_{mode}",
        "selected_input_indices": selected_positions.tolist(),
        "diagnostics": {
            "evidence_mode": mode,
            "section_count": section_count,
            "minimum_arc_deg": min_arc,
            "minimum_inlier_fraction": min_fraction,
            "maximum_axis_residual_m": float(np.max(axis_residual)),
            "maximum_radius_profile_residual_m": float(np.max(np.abs(radius_residual))),
            "radius_slope_m_per_m": float(radius_slope),
            "observed_backside": mode == "UPPER_CIRCULAR",
            "sections": sections,
        },
    }
    return {"axis": axis, "track": track, "fit": fit}, {"status": "PASS", "sections": sections}


def recover_partial_arc_identity(xyz, hag, original_index):
    """Require a dense one-sided arc from below BH to the upper stem."""

    return _recover_series(
        xyz, hag, original_index,
        first_hag=FIRST_HAG_M, section_count=SECTION_COUNT,
        min_fraction=MIN_INLIER_FRACTION, min_arc=MIN_ARC_DEG,
        min_sectors=MIN_SECTORS, bh_sections=6, mode="PARTIAL_ARC",
    )


def recover_upper_supported_identity(xyz, hag, original_index):
    """Estimate BH at most 0.075 m below nine complete upper stem circles."""

    return _recover_series(
        xyz, hag, original_index,
        first_hag=1.30, section_count=9,
        min_fraction=DBH_RECOVERY_CONFIG.minimum_inlier_fraction,
        min_arc=DBH_RECOVERY_CONFIG.minimum_angular_span_deg,
        min_sectors=DBH_RECOVERY_CONFIG.minimum_occupied_sectors,
        bh_sections=3, mode="UPPER_CIRCULAR",
    )


def recover_upper_partial_arc_identity(xyz, hag, original_index):
    """Use a continuous one-sided stem beginning immediately below 1.30 m.

    A low branch or ground artifact can spoil 0.85–1.15 m although the
    breast-height section and thirteen successive upper sections remain sound.
    The target height lies inside the observed series, not beyond it.
    """
    return _recover_series(
        xyz, hag, original_index,
        first_hag=1.15, section_count=13,
        min_fraction=MIN_INLIER_FRACTION, min_arc=MIN_ARC_DEG,
        min_sectors=MIN_SECTORS, bh_sections=4, mode="UPPER_PARTIAL_ARC",
    )


def recover_local_bh_series(xyz, hag, original_index, axis):
    """Fit a human-selected stem inside a bounded axis tube near breast height."""
    anchor = np.asarray(axis["bh_reference_xyz"], dtype=float)
    direction = np.asarray(axis["direction_xyz"], dtype=float)
    radius = float(axis["radius_m"])
    if (anchor.shape != (3,) or direction.shape != (3,)
            or not np.isfinite(anchor).all() or not np.isfinite(direction).all()
            or abs(np.linalg.norm(direction) - 1) > 1e-6
            or not 0.025 <= radius <= 0.75):
        return None, {"status": "INVALID_SELECTED_AXIS"}
    delta = np.asarray(xyz) - anchor
    radial = np.linalg.norm(delta - np.outer(delta @ direction, direction), axis=1)
    keep = radial <= max(radius + 0.10, 0.18)
    recovered, diagnostic = _recover_series(
        np.asarray(xyz)[keep], np.asarray(hag)[keep], np.asarray(original_index)[keep],
        first_hag=0.85, section_count=9,
        min_fraction=MIN_INLIER_FRACTION, min_arc=MIN_ARC_DEG,
        min_sectors=MIN_SECTORS, bh_sections=6, mode="LOCAL_BH_ARC",
    )
    if recovered is not None:
        positions = np.flatnonzero(keep)
        recovered["fit"]["selected_input_indices"] = positions[
            np.asarray(recovered["fit"]["selected_input_indices"], dtype=np.int64)
        ].tolist()
    return recovered, diagnostic


def recover_bh_tube_identity(xyz, hag, original_index, *, expected_xy=None):
    """Discover a stem at 1.30 m, then verify nine independent local slices."""
    xyz = np.asarray(xyz)
    hag = np.asarray(hag)
    band = np.abs(hag - 1.30) <= 0.075
    xy = np.unique(xyz[band, :2], axis=0)
    if len(xy) < MIN_POINTS:
        return None, {"status": "BH_TUBE_BREAST_SECTION_SPARSE", "count": len(xy)}
    fitted, _, _ = _section_circle_fit(xy, DBH_RECOVERY_CONFIG)
    if (fitted["radius_m"] is None or fitted["center_x_m"] is None
            or fitted["center_y_m"] is None or fitted["fit_rmse_m"] is None
            or fitted["angular_span_deg"] is None
            or fitted["inlier_fraction"] < MIN_INLIER_FRACTION
            or fitted["angular_span_deg"] < MIN_ARC_DEG
            or fitted["occupied_sector_count"] < MIN_SECTORS
            or fitted["fit_rmse_m"] > MAX_SECTION_RMSE_M):
        return None, {"status": "BH_TUBE_BREAST_SECTION_UNPROVEN", "section": fitted}
    center = np.asarray((fitted["center_x_m"], fitted["center_y_m"]))
    if expected_xy is not None and np.linalg.norm(center - np.asarray(expected_xy)) > 0.60:
        return None, {"status": "BH_TUBE_CENTER_OUTSIDE_SEED"}
    initial = {
        "bh_reference_xyz": [*center, float(np.median(xyz[band, 2]))],
        "direction_xyz": [0.0, 0.0, 1.0],
        "radius_m": float(fitted["radius_m"]),
    }
    recovered, diagnostic = recover_local_bh_series(xyz, hag, original_index, initial)
    if recovered is None:
        return None, {"status": diagnostic["status"], "breast_section": fitted,
                      "series": diagnostic}
    if (abs(recovered["axis"]["radius_m"] - initial["radius_m"]) > 0.03
            or np.linalg.norm(np.asarray(recovered["axis"]["bh_reference_xyz"][:2])
                              - center) > 0.05):
        return None, {"status": "BH_TUBE_BREAST_TO_SERIES_INCONSISTENT",
                      "breast_section": fitted, "series": diagnostic}
    recovered["axis"]["proposal_source"] = "RAW_BH_CIRCLE_LOCAL_NINE_SECTION_SERIES"
    return recovered, {"status": "PASS", "breast_section": fitted,
                       "series": diagnostic}


def recover_ground_anchored_bh(xyz, local_hag, original_index):
    """Measure 1.30 m above a human-chosen local base from agreeing windows."""
    candidates = []
    diagnostics = []
    # Overlapping windows prevent a few millimetres of ground-anchor choice
    # from moving a dense scan line exactly onto a section boundary.  Two
    # independently started, mutually agreeing series are still required.
    for first in (0.70, 0.775, 0.85, 0.925, 1.00, 1.075, 1.15):
        candidate, proof = _recover_series(
            xyz, local_hag, original_index,
            first_hag=first, section_count=9,
            min_fraction=MIN_INLIER_FRACTION, min_arc=MIN_ARC_DEG,
            min_sectors=MIN_SECTORS, bh_sections=4,
            mode="LOCAL_GROUND_BH_ARC",
        )
        diagnostics.append({"first_hag_m": first, "status": proof["status"]})
        if candidate is not None:
            candidates.append((first, candidate))
    if len(candidates) < 2:
        return None, {"status": "LOCAL_GROUND_BH_WINDOWS_INSUFFICIENT",
                      "windows": diagnostics}
    values = np.asarray([c["fit"]["dbh_cm"] for _, c in candidates])
    if np.ptp(values) > 1.0:
        return None, {"status": "LOCAL_GROUND_BH_WINDOWS_DISAGREE",
                      "dbh_values_cm": values.tolist(), "windows": diagnostics}
    center = float(np.median(values))
    first, selected = min(candidates, key=lambda item: (
        abs(item[1]["fit"]["dbh_cm"] - center),
        item[1]["fit"]["dbh_fit_rmse"], item[0],
    ))
    selected["fit"]["diagnostics"]["window_consensus_dbh_cm"] = values.tolist()
    selected["fit"]["diagnostics"]["selected_first_hag_m"] = first
    selected["fit"]["dbh_uncertainty_cm"] = max(
        selected["fit"]["dbh_uncertainty_cm"], float(np.ptp(values))
    )
    return selected, {"status": "PASS", "dbh_values_cm": values.tolist(),
                      "selected_first_hag_m": first, "windows": diagnostics}


def elevated_local_base_probe(xyz, hag, axis):
    """Audit the local root height against all raw low points near a stem.

    The candidate base is discovered from the cloud in 10 cm bins. A broad
    annular surface, a connected stem just above it, and scarce points below
    it are independent evidence of an elevated base. Sparse root/ground data
    remain explicitly unverified rather than silently endorsing coarse HAG.
    """
    xyz = np.asarray(xyz)
    hag = np.asarray(hag)
    if len(xyz) != len(hag) or not len(xyz):
        return {"status": "LOCAL_BASE_HEIGHT_UNVERIFIED", "reason": "EMPTY_RAW_ROOT_ROI"}
    anchor = np.asarray(axis["bh_reference_xyz"], dtype=float)
    direction = np.asarray(axis.get("direction_xyz", [0, 0, 1]), dtype=float)
    stem_radius = float(axis.get("radius_m", 0))
    if (anchor.shape != (3,) or direction.shape != (3,)
            or abs(direction[2]) < 0.5 or not 0.025 <= stem_radius <= 0.75):
        return {"status": "LOCAL_BASE_HEIGHT_UNVERIFIED", "reason": "AXIS_RADIUS_UNPROVEN"}
    center = anchor[:2] + (hag - 1.3)[:, None] * direction[:2] / direction[2]
    radius = np.linalg.norm(xyz[:, :2] - center, axis=1)
    stem_tolerance = max(0.05, 0.25 * stem_radius)
    bark = np.abs(radius - stem_radius) <= stem_tolerance
    ground_inner = stem_radius + max(0.08, 0.2 * stem_radius)
    ground_outer = stem_radius + 0.43
    ground_ring = (radius >= ground_inner) & (radius <= ground_outer)
    local_ground = int(np.count_nonzero(ground_ring & (hag >= -0.20)
                                    & (hag <= 0.20)))
    low_stem = int(np.count_nonzero(bark & (hag >= 0.05)
                                & (hag <= 0.45)))
    # A coherent low stem and nearby ground independently support the coarse
    # root level; an elevated shrub/branch sheet must not override both.
    if local_ground >= 30 and low_stem >= 30:
        return {"status": "NO_ELEVATED_BASE_EVIDENCE",
                "near_coarse_ground_point_count": local_ground,
                "low_stem_point_count": low_stem}
    for low in np.arange(0.20, 2.41, 0.10):
        sheet = (hag >= low) & (hag < low + 0.10) & ground_ring
        below = (hag >= low - 0.30) & (hag < low) & (radius <= ground_outer)
        above = (hag >= low + 0.10) & (hag < low + 0.30) & bark
        count = int(np.count_nonzero(sheet))
        upper_count = int(np.count_nonzero(above))
        if count < 40 or upper_count < max(30, int(0.10 * count)):
            continue
        theta = np.arctan2(xyz[sheet, 1] - center[sheet, 1],
                           xyz[sheet, 0] - center[sheet, 0])
        sectors = np.unique(np.floor((theta + np.pi) / (2 * np.pi) * 12)).size
        radial_span = float(np.ptp(radius[sheet]))
        if (sectors >= 5 and radial_span >= 0.15
                and np.count_nonzero(below) < count * 0.10):
            return {"status": "LOCAL_BASE_HEIGHT_AMBIGUOUS",
                    "sheet_hag_range_m": [float(low), float(low + 0.10)],
                    "horizontal_sheet_point_count": count,
                    "lower_point_count": int(np.count_nonzero(below)),
                    "upper_stem_point_count": upper_count,
                    "angular_sector_count": int(sectors),
                    "radial_extent_m": radial_span,
                    "stem_radius_m": stem_radius}
    return {"status": "LOCAL_BASE_HEIGHT_UNVERIFIED",
            "near_coarse_ground_point_count": local_ground,
            "low_stem_point_count": low_stem}


__all__ = ["recover_partial_arc_identity", "recover_upper_supported_identity",
           "recover_upper_partial_arc_identity", "recover_local_bh_series",
           "recover_bh_tube_identity", "recover_ground_anchored_bh",
           "elevated_local_base_probe"]
