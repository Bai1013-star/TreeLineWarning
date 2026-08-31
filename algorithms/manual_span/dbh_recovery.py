"""Bounded stem DBH recovery and project-local allometric estimates.

This module is deliberately independent from the manual-span metrics and
editor code.  It accepts the exact local XYZ/HAG evidence selected by its
caller, fits robust circles in six fixed breast-height sections, and returns
plain Python values suitable for JSON serialization.

The thresholds in :class:`DbhRecoveryConfig` are the V1 quality guards.  They
are kept immutable so a caller cannot silently change the acceptance policy
for one tree or one run.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import atan, degrees, exp, hypot, isfinite, log, sqrt
from typing import Any

import numpy as np

DBH_RECOVERY_VERSION = "DBH_RECOVERY_V1"
DIRECT_STEM = "DIRECT_STEM"
RECOVERED_STEM = "RECOVERED_STEM"
ALLOMETRIC_ESTIMATE = "ALLOMETRIC_ESTIMATE"
UNAVAILABLE = "UNAVAILABLE"
HIGH = "HIGH"
ESTIMATE = "ESTIMATE"
PASS = "PASS"

# Continuity is a source-evidence prefilter, independent from the circle
# quality guards below.  These values are fixed V1 geometry granularity, not
# values learned from a point count or tuned for a particular dataset.
STEM_CONTINUITY_VOXEL_SIZE_M = 0.02
STEM_CONTINUITY_MIN_SECTIONS = 4
STEM_CONTINUITY_MIDPOINT_HAG_M = 1.30


@dataclass(frozen=True, slots=True)
class DbhRecoveryConfig:
    """Frozen V1 DBH geometry and allometry policy."""

    breast_height_min_m: float = 0.85
    breast_height_max_m: float = 1.75
    section_height_m: float = 0.15
    section_count: int = 6
    stem_search_radius_m: float = 0.60
    minimum_points_per_section: int = 12
    max_fit_points_per_section: int = 4096
    max_circle_hypotheses: int = 256
    diameter_min_m: float = 0.05
    diameter_max_m: float = 1.50
    maximum_inlier_tolerance_m: float = 0.020
    minimum_inlier_tolerance_m: float = 0.008
    radial_tolerance_fraction: float = 0.05
    minimum_inlier_fraction: float = 0.60
    minimum_angular_span_deg: float = 180.0
    minimum_occupied_sectors: int = 6
    angular_sector_count: int = 12
    minimum_accepted_sections: int = 3
    maximum_relative_diameter_spread: float = 0.20
    maximum_tilt_deg: float = 20.0
    maximum_center_track_residual_m: float = 0.03
    minimum_training_trees: int = 10
    maximum_loo_relative_rmse: float = 0.25
    maximum_loo_relative_error: float = 0.50


DBH_RECOVERY_CONFIG = DbhRecoveryConfig()
# A short alias is useful to callers that prefer the generic policy name.
RECOVERY_CONFIG = DBH_RECOVERY_CONFIG


def _json_float(value: float | int | np.floating[Any] | None) -> float | None:
    """Return a finite builtin float, or ``None`` for invalid numbers."""

    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if isfinite(converted) else None


def _json_id(value: Any) -> str:
    """Convert a persistent identity to a stable JSON object key."""

    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and isfinite(float(value)):
        return str(float(value))
    return str(value)


def _json_identity(value: Any) -> str | int | float | bool | None:
    """Return an identity value that the standard JSON encoder accepts."""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        return float(value) if isfinite(float(value)) else str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _safe_input_count(value: Any) -> int:
    try:
        array = np.asarray(value)
    except Exception:  # malformed array-like input is handled by the caller
        return 0
    return int(array.shape[0]) if array.ndim else 0


def _blank_slice_diagnostic(index: int, low: float, high: float, reason: str) -> dict[str, Any]:
    return {
        "index": int(index),
        "lower_hag_m": float(low),
        "upper_hag_m": float(high),
        "input_point_count": 0,
        "unique_point_count": 0,
        "fit_point_count": 0,
        "inlier_point_count": 0,
        "inlier_fraction": None,
        "radius_m": None,
        "diameter_m": None,
        "dbh_cm": None,
        "center_x_m": None,
        "center_y_m": None,
        "center_hag_m": None,
        "angular_span_deg": None,
        "occupied_sector_count": 0,
        "fit_rmse_m": None,
        "accepted": False,
        "status": "SKIPPED",
        "reason": reason,
    }


def _base_result(
    point_count: int,
    slices: list[dict[str, Any]],
    status: str,
    *,
    requested_method: Any = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    accepted = sum(1 for section in slices if bool(section.get("accepted")))
    detail = {
        "algorithm_version": DBH_RECOVERY_VERSION,
        "requested_method": requested_method,
        "config": {key: value for key, value in asdict(DBH_RECOVERY_CONFIG).items()},
    }
    if diagnostics:
        detail.update(diagnostics)
    return {
        "dbh_cm": None,
        "dbh_method": UNAVAILABLE,
        "dbh_quality": UNAVAILABLE,
        "dbh_point_count": int(point_count),
        "dbh_fit_rmse": None,
        "dbh_slice_count": int(accepted),
        "dbh_fit_status": status,
        "dbh_uncertainty_cm": None,
        "dbh_model_version": DBH_RECOVERY_VERSION,
        "slices": slices,
        "diagnostics": detail,
    }


def _sort_xyz_hag(xyz: np.ndarray, hag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort evidence by values so row order cannot affect the result."""

    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0], hag))
    return xyz[order], hag[order]


def _bounded_selection(points: np.ndarray, maximum: int) -> np.ndarray:
    """Select at most ``maximum`` rows while preserving deterministic coverage."""

    if len(points) <= maximum:
        return points
    # Linspace gives each part of the value-sorted evidence a chance to enter
    # the bounded fit and does not depend on NumPy's process-global RNG state.
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def _circle_from_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Fit the unique circle through three non-collinear XY points."""

    p1, p2, p3 = points
    ax = float(p2[0] - p1[0])
    ay = float(p2[1] - p1[1])
    bx = float(p3[0] - p1[0])
    by = float(p3[1] - p1[1])
    determinant = ax * by - ay * bx
    if abs(determinant) <= 1.0e-12:
        return None
    a_sq = float(p2[0] * p2[0] + p2[1] * p2[1] - p1[0] * p1[0] - p1[1] * p1[1])
    b_sq = float(p3[0] * p3[0] + p3[1] * p3[1] - p1[0] * p1[0] - p1[1] * p1[1])
    cx = (a_sq * by - ay * b_sq) / (2.0 * determinant)
    cy = (ax * b_sq - a_sq * bx) / (2.0 * determinant)
    radius = hypot(cx - float(p1[0]), cy - float(p1[1]))
    if not (isfinite(cx) and isfinite(cy) and isfinite(radius)):
        return None
    return np.asarray((cx, cy), dtype=np.float64), float(radius)


def _algebraic_circle(
    points: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, float] | None:
    """Solve a weighted linearized circle fit for already selected inliers."""

    if len(points) < 3:
        return None
    matrix = np.column_stack((2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))))
    rhs = np.sum(points * points, axis=1)
    try:
        if weights is None:
            solution, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
        else:
            safe_weights = np.asarray(weights, dtype=np.float64)
            safe_weights = np.clip(safe_weights, 1.0e-6, None)
            root_weights = np.sqrt(safe_weights)
            solution, _, rank, _ = np.linalg.lstsq(
                matrix * root_weights[:, None], rhs * root_weights, rcond=None
            )
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None
    if int(rank) < 3:
        return None
    cx, cy, constant = (float(value) for value in solution)
    radius_sq = constant + cx * cx + cy * cy
    if not (isfinite(cx) and isfinite(cy) and isfinite(radius_sq)) or radius_sq <= 0.0:
        return None
    return np.asarray((cx, cy), dtype=np.float64), sqrt(radius_sq)


def _radial_residuals(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    return np.hypot(points[:, 0] - center[0], points[:, 1] - center[1]) - radius


def _circle_hypotheses(n: int, count: int) -> np.ndarray:
    """Produce bounded, deterministic three-point hypotheses.

    A local generator is used instead of the global NumPy RNG.  The generated
    indices therefore remain stable when callers fit trees in different order.
    """

    if n < 3 or count <= 0:
        return np.empty((0, 3), dtype=np.int64)
    generator = np.random.default_rng(0xDBA5EED + n * 131)
    hypotheses = generator.integers(0, n, size=(count, 3), dtype=np.int64)
    return hypotheses


def _section_circle_fit(
    points: np.ndarray, config: DbhRecoveryConfig
) -> tuple[dict[str, Any], np.ndarray | None, float | None]:
    """Fit one horizontal section and return diagnostics, center and radius."""

    selected = _bounded_selection(points, config.max_fit_points_per_section)
    # Fit in a section-local frame.  Finalized coordinates can be large
    # projected values; subtracting a robust origin avoids squaring those
    # values in the linearized circle solve and preserves translation accuracy.
    fit_origin = np.median(selected, axis=0)
    fit_points = selected - fit_origin
    diagnostic: dict[str, Any] = {
        "index": None,
        "lower_hag_m": None,
        "upper_hag_m": None,
        "input_point_count": int(len(points)),
        "unique_point_count": int(len(points)),
        "fit_point_count": int(len(selected)),
        "inlier_point_count": 0,
        "inlier_fraction": None,
        "radius_m": None,
        "diameter_m": None,
        "dbh_cm": None,
        "center_x_m": None,
        "center_y_m": None,
        "center_hag_m": None,
        "angular_span_deg": None,
        "occupied_sector_count": 0,
        "fit_rmse_m": None,
        "accepted": False,
        "status": "FIT_FAILED",
        "reason": "FIT_FAILED",
    }
    if len(selected) < config.minimum_points_per_section:
        diagnostic["status"] = "INSUFFICIENT_POINTS"
        diagnostic["reason"] = "INSUFFICIENT_POINTS"
        return diagnostic, None, None

    best_center: np.ndarray | None = None
    best_radius: float | None = None
    best_score: tuple[int, float, float, float] | None = None
    hypotheses = _circle_hypotheses(len(selected), config.max_circle_hypotheses)
    for first, second, third in hypotheses:
        if first == second or first == third or second == third:
            continue
        fit = _circle_from_three(fit_points[[int(first), int(second), int(third)]])
        if fit is None:
            continue
        center, radius = fit
        if not (config.diameter_min_m / 2.0 <= radius <= config.diameter_max_m / 2.0):
            continue
        residuals = np.abs(_radial_residuals(fit_points, center, radius))
        tolerance = min(
            config.maximum_inlier_tolerance_m,
            max(config.minimum_inlier_tolerance_m, config.radial_tolerance_fraction * radius),
        )
        inliers = residuals <= tolerance
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count == 0:
            continue
        inlier_values = residuals[inliers]
        score = (
            inlier_count,
            -float(np.median(inlier_values)),
            -float(np.sqrt(np.mean(inlier_values * inlier_values))),
            -float(radius),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_center = center
            best_radius = radius

    if best_center is None or best_radius is None:
        diagnostic["status"] = "NO_CIRCLE_HYPOTHESIS"
        diagnostic["reason"] = "NO_CIRCLE_HYPOTHESIS"
        return diagnostic, None, None

    # Refit and reclassify a bounded number of times.  The fixed radial guard
    # keeps distant canopy/outlier points from pulling the algebraic estimate.
    center = best_center
    radius = best_radius
    for _ in range(8):
        residuals = _radial_residuals(fit_points, center, radius)
        tolerance = min(
            config.maximum_inlier_tolerance_m,
            max(config.minimum_inlier_tolerance_m, config.radial_tolerance_fraction * radius),
        )
        absolute = np.abs(residuals)
        inliers = absolute <= tolerance
        if int(np.count_nonzero(inliers)) < 3:
            break
        # Refit only the current inliers, then reclassify against the fixed
        # radial guard.  Repeating this bounded step makes the RANSAC result
        # robust to the points that were not part of its winning hypothesis.
        refined = _algebraic_circle(fit_points[inliers])
        if refined is None:
            break
        new_center, new_radius = refined
        if (
            np.linalg.norm(new_center - center) <= 1.0e-10
            and abs(new_radius - radius) <= 1.0e-10
        ):
            center, radius = new_center, new_radius
            break
        center, radius = new_center, new_radius
        if not (config.diameter_min_m / 2.0 <= radius <= config.diameter_max_m / 2.0):
            break

    residuals = _radial_residuals(fit_points, center, radius)
    tolerance = min(
        config.maximum_inlier_tolerance_m,
        max(config.minimum_inlier_tolerance_m, config.radial_tolerance_fraction * radius),
    )
    absolute = np.abs(residuals)
    inliers = absolute <= tolerance
    inlier_count = int(np.count_nonzero(inliers))
    fraction = inlier_count / float(len(selected))
    diameter = 2.0 * radius
    diagnostic["radius_m"] = _json_float(radius)
    diagnostic["diameter_m"] = _json_float(diameter)
    diagnostic["dbh_cm"] = _json_float(diameter * 100.0)
    center_global = center + fit_origin
    diagnostic["center_x_m"] = _json_float(center_global[0])
    diagnostic["center_y_m"] = _json_float(center_global[1])
    diagnostic["inlier_point_count"] = inlier_count
    diagnostic["inlier_fraction"] = _json_float(fraction)
    if inlier_count:
        diagnostic["fit_rmse_m"] = _json_float(sqrt(float(np.mean(residuals[inliers] ** 2))))
    angles = np.mod(
        np.arctan2(
            fit_points[inliers, 1] - center[1], fit_points[inliers, 0] - center[0]
        ),
        2.0 * np.pi,
    )
    if len(angles):
        sorted_angles = np.sort(angles)
        gaps = np.diff(np.concatenate((sorted_angles, sorted_angles[:1] + 2.0 * np.pi)))
        angular_span = degrees(2.0 * np.pi - float(np.max(gaps)))
        sectors = np.floor(angles / (2.0 * np.pi / config.angular_sector_count)).astype(np.int64)
        occupied = int(len(np.unique(sectors)))
        diagnostic["angular_span_deg"] = _json_float(angular_span)
        diagnostic["occupied_sector_count"] = occupied

    if not config.diameter_min_m <= diameter <= config.diameter_max_m:
        reason = "DIAMETER_IMPLAUSIBLE"
    elif inlier_count < config.minimum_points_per_section:
        reason = "INSUFFICIENT_INLIER_POINTS"
    elif fraction < config.minimum_inlier_fraction:
        reason = "INLIER_FRACTION_LOW"
    elif diagnostic["fit_rmse_m"] is None or diagnostic["fit_rmse_m"] > tolerance:
        reason = "RADIAL_RESIDUAL_HIGH"
    elif (
        diagnostic["angular_span_deg"] is None
        or diagnostic["angular_span_deg"] < config.minimum_angular_span_deg
    ):
        reason = "ANGULAR_SPAN_LT_180"
    elif int(diagnostic["occupied_sector_count"]) < config.minimum_occupied_sectors:
        reason = "SECTOR_OCCUPANCY_LT_6"
    else:
        reason = PASS
        diagnostic["accepted"] = True
    diagnostic["status"] = reason
    diagnostic["reason"] = reason
    if not diagnostic["accepted"]:
        return diagnostic, None, None
    return diagnostic, center_global, radius


def _make_slices(config: DbhRecoveryConfig) -> list[dict[str, Any]]:
    return [
        _blank_slice_diagnostic(
            index,
            config.breast_height_min_m + index * config.section_height_m,
            config.breast_height_min_m + (index + 1) * config.section_height_m,
            "NOT_EVALUATED",
        )
        for index in range(config.section_count)
    ]


def _continuity_invalid_result(
    point_count: int, status: str, *, finite_point_count: int = 0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a fail-closed mask and JSON-safe diagnostics for bad input."""

    return np.zeros(max(0, int(point_count)), dtype=bool), {
        "status": status,
        "reason": status,
        "input_point_count": int(point_count),
        "finite_point_count": int(finite_point_count),
        "nonfinite_point_count": int(max(0, point_count - finite_point_count)),
        "breast_height_point_count": 0,
        "selected_point_count": 0,
        "supported_column_count": 0,
        "voxel_size_m": float(STEM_CONTINUITY_VOXEL_SIZE_M),
        "voxel_origin_xy_m": None,
        "section_count": 6,
        "section_point_counts": [0, 0, 0, 0, 0, 0],
        "section_unique_voxel_counts": [0, 0, 0, 0, 0, 0],
        "minimum_supported_sections": int(STEM_CONTINUITY_MIN_SECTIONS),
        "midpoint_hag_m": float(STEM_CONTINUITY_MIDPOINT_HAG_M),
        "supported_columns_sample": [],
    }


def _continuity_voxels(xy: np.ndarray, origin_xy: np.ndarray) -> np.ndarray:
    """Quantize XY after translation to a finite-band local origin."""

    scaled = (xy - origin_xy) / STEM_CONTINUITY_VOXEL_SIZE_M
    # The minimum-origin translation makes ordinary projected coordinates safe
    # to quantize.  A tiny fixed epsilon keeps an exact voxel boundary stable
    # after subtracting large translated coordinates, without changing the
    # 0.02 m geometry granularity.
    return np.floor(scaled + 1.0e-9).astype(np.int64)


def select_continuous_stem_points(
    xyz_local_m: np.ndarray, hag_m: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select rows supported by a vertical stem-like XY voxel track.

    The mask is the same length as the original input and never rewrites or
    reindexes rows.  Six fixed 0.15 m HAG sections cover [0.85, 1.75] m.  An
    XY voxel of fixed 0.02 m size is supported only when it occurs in at least
    four distinct sections, with at least one section below and one above
    1.30 m.  Occupancy is a set of section labels, so a dense single-level
    sheet cannot substitute for vertical continuity.

    Voxel coordinates are formed after subtracting the finite breast-height
    XY minimum.  This keeps quantization stable for large projected offsets;
    no dataset-specific origin, density, or source identity is inferred.
    """

    point_count = _safe_input_count(xyz_local_m)
    try:
        xyz = np.asarray(xyz_local_m, dtype=np.float64)
        hag = np.asarray(hag_m, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return _continuity_invalid_result(point_count, "INVALID_INPUT")
    if xyz.ndim != 2 or xyz.shape[1:] != (3,):
        return _continuity_invalid_result(point_count, "INVALID_XYZ_SHAPE")
    if hag.ndim != 1 or len(hag) != len(xyz):
        return _continuity_invalid_result(len(xyz), "INVALID_HAG_SHAPE")
    finite_rows = np.all(np.isfinite(xyz), axis=1) & np.isfinite(hag)
    finite_count = int(np.count_nonzero(finite_rows))
    if not np.all(finite_rows):
        # The helper is an evidence gate: one malformed row is never silently
        # repaired or allowed to alter the supported-column calculation.
        return _continuity_invalid_result(
            len(xyz), "NONFINITE_INPUT", finite_point_count=finite_count
        )
    if len(xyz) == 0:
        return _continuity_invalid_result(0, "INSUFFICIENT_INPUT_POINTS")

    breast_low = DBH_RECOVERY_CONFIG.breast_height_min_m
    breast_high = DBH_RECOVERY_CONFIG.breast_height_max_m
    breast_mask = (hag >= breast_low) & (hag <= breast_high)
    breast_count = int(np.count_nonzero(breast_mask))
    if breast_count == 0:
        mask, diagnostics = _continuity_invalid_result(len(xyz), "NO_BREAST_HEIGHT_POINTS")
        diagnostics["finite_point_count"] = finite_count
        diagnostics["nonfinite_point_count"] = 0
        return mask, diagnostics

    origin_xy = np.min(xyz[breast_mask, :2], axis=0)
    section_masks: list[np.ndarray] = []
    section_point_counts: list[int] = []
    section_unique_counts: list[int] = []
    section_columns: dict[tuple[int, int], set[int]] = {}
    for section_index in range(DBH_RECOVERY_CONFIG.section_count):
        low = breast_low + section_index * DBH_RECOVERY_CONFIG.section_height_m
        high = breast_low + (section_index + 1) * DBH_RECOVERY_CONFIG.section_height_m
        if section_index == DBH_RECOVERY_CONFIG.section_count - 1:
            section_mask = (hag >= low) & (hag <= high)
        else:
            section_mask = (hag >= low) & (hag < high)
        section_mask &= breast_mask
        section_masks.append(section_mask)
        section_count = int(np.count_nonzero(section_mask))
        section_point_counts.append(section_count)
        if section_count == 0:
            section_unique_counts.append(0)
            continue
        section_voxels = _continuity_voxels(xyz[section_mask, :2], origin_xy)
        unique_voxels = np.unique(section_voxels, axis=0)
        section_unique_counts.append(int(len(unique_voxels)))
        for voxel_x, voxel_y in unique_voxels:
            section_columns.setdefault((int(voxel_x), int(voxel_y)), set()).add(section_index)

    supported_columns = {
        key
        for key, sections in section_columns.items()
        if len(sections) >= STEM_CONTINUITY_MIN_SECTIONS
        and any(section < 3 for section in sections)
        and any(section >= 3 for section in sections)
    }
    selected_mask = np.zeros(len(xyz), dtype=bool)
    if supported_columns:
        for section_mask in section_masks:
            indices = np.flatnonzero(section_mask)
            if len(indices) == 0:
                continue
            voxels = _continuity_voxels(xyz[indices, :2], origin_xy)
            keep = np.fromiter(
                (tuple(voxel) in supported_columns for voxel in voxels),
                dtype=bool,
                count=len(indices),
            )
            selected_mask[indices] = keep

    sorted_supported = sorted(supported_columns)
    sample = [
        {
            "voxel_x": int(voxel_x),
            "voxel_y": int(voxel_y),
            "sections": [int(value) for value in sorted(section_columns[(voxel_x, voxel_y)])],
        }
        for voxel_x, voxel_y in sorted_supported[:32]
    ]
    selected_count = int(np.count_nonzero(selected_mask))
    status = PASS if supported_columns else "NO_CONTINUOUS_STEM_COLUMNS"
    diagnostics = {
        "status": status,
        "reason": status,
        "input_point_count": int(len(xyz)),
        "finite_point_count": finite_count,
        "nonfinite_point_count": 0,
        "breast_height_point_count": breast_count,
        "selected_point_count": selected_count,
        "supported_column_count": int(len(supported_columns)),
        "voxel_size_m": float(STEM_CONTINUITY_VOXEL_SIZE_M),
        "voxel_origin_xy_m": [float(origin_xy[0]), float(origin_xy[1])],
        "section_count": int(DBH_RECOVERY_CONFIG.section_count),
        "section_point_counts": section_point_counts,
        "section_unique_voxel_counts": section_unique_counts,
        "minimum_supported_sections": int(STEM_CONTINUITY_MIN_SECTIONS),
        "midpoint_hag_m": float(STEM_CONTINUITY_MIDPOINT_HAG_M),
        "supported_columns_sample": sample,
    }
    return selected_mask, diagnostics


def _track_diagnostics(
    accepted: list[dict[str, Any]], config: DbhRecoveryConfig
) -> tuple[bool, dict[str, Any]]:
    centers = np.asarray(
        [[section["center_x_m"], section["center_y_m"]] for section in accepted], dtype=np.float64
    )
    heights = np.asarray([section["center_hag_m"] for section in accepted], dtype=np.float64)
    if len(accepted) < 2 or not np.all(np.isfinite(centers)) or not np.all(np.isfinite(heights)):
        return False, {
            "tilt_angle_deg": None,
            "center_track_residual_m": None,
            "center_track_max_residual_m": None,
            "center_track_status": "CENTER_TRACK_UNAVAILABLE",
        }
    design = np.column_stack((np.ones(len(heights)), heights - float(np.mean(heights))))
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(design, centers, rcond=None)
    except np.linalg.LinAlgError:
        return False, {
            "tilt_angle_deg": None,
            "center_track_residual_m": None,
            "center_track_max_residual_m": None,
            "center_track_status": "CENTER_TRACK_FIT_FAILED",
        }
    if int(rank) < 2:
        return False, {
            "tilt_angle_deg": None,
            "center_track_residual_m": None,
            "center_track_max_residual_m": None,
            "center_track_status": "CENTER_TRACK_FIT_FAILED",
        }
    predicted = design @ coefficients
    residuals = np.linalg.norm(centers - predicted, axis=1)
    slope = np.asarray(coefficients[1], dtype=np.float64)
    tilt = degrees(atan(float(np.linalg.norm(slope))))
    max_residual = float(np.max(residuals))
    rms_residual = float(sqrt(np.mean(residuals * residuals)))
    if tilt > config.maximum_tilt_deg + 1.0e-9:
        status = "TILT_EXCEEDS_20_DEG"
        valid = False
    elif max_residual > config.maximum_center_track_residual_m + 1.0e-9:
        status = "CENTER_TRACK_RESIDUAL_HIGH"
        valid = False
    else:
        status = PASS
        valid = True
    return valid, {
        "tilt_angle_deg": _json_float(tilt),
        "center_track_residual_m": _json_float(rms_residual),
        "center_track_max_residual_m": _json_float(max_residual),
        "center_track_status": status,
    }


def fit_stem_dbh(
    xyz_local_m: np.ndarray,
    hag_m: np.ndarray,
    *,
    method: str = RECOVERED_STEM,
) -> dict[str, Any]:
    """Fit stem diameter from exact local XYZ points and HAG values.

    Six non-overlapping 0.15 m sections cover [0.85, 1.75] m.  Every input
    row contributes to ``dbh_point_count``; only finite, section-local unique
    XY rows enter the bounded circle fits.
    """

    point_count = _safe_input_count(xyz_local_m)
    config = DBH_RECOVERY_CONFIG
    slices = _make_slices(config)
    requested_method = method if isinstance(method, str) else str(method)
    if requested_method not in (DIRECT_STEM, RECOVERED_STEM):
        return _base_result(
            point_count,
            slices,
            "INVALID_METHOD",
            requested_method=requested_method,
        )
    try:
        xyz = np.asarray(xyz_local_m, dtype=np.float64)
        hag = np.asarray(hag_m, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return _base_result(
            point_count,
            slices,
            "INVALID_INPUT",
            requested_method=requested_method,
        )
    if xyz.ndim != 2 or xyz.shape[1:] != (3,):
        return _base_result(
            point_count,
            slices,
            "INVALID_XYZ_SHAPE",
            requested_method=requested_method,
        )
    if hag.ndim != 1 or len(hag) != len(xyz):
        return _base_result(
            point_count,
            slices,
            "INVALID_HAG_SHAPE",
            requested_method=requested_method,
        )
    if len(xyz) == 0:
        return _base_result(
            point_count,
            slices,
            "INSUFFICIENT_INPUT_POINTS",
            requested_method=requested_method,
        )
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(hag)):
        return _base_result(
            point_count,
            slices,
            "NONFINITE_INPUT",
            requested_method=requested_method,
        )

    xyz, hag = _sort_xyz_hag(xyz, hag)
    accepted_sections: list[dict[str, Any]] = []
    for index, diagnostic in enumerate(slices):
        low = config.breast_height_min_m + index * config.section_height_m
        high = config.breast_height_min_m + (index + 1) * config.section_height_m
        if index == config.section_count - 1:
            in_section = (hag >= low) & (hag <= high)
        else:
            in_section = (hag >= low) & (hag < high)
        section_xy = xyz[in_section, :2]
        diagnostic["input_point_count"] = int(len(section_xy))
        if len(section_xy):
            section_xy = np.unique(section_xy, axis=0)
        diagnostic["unique_point_count"] = int(len(section_xy))
        if len(section_xy) < config.minimum_points_per_section:
            diagnostic["status"] = "INSUFFICIENT_POINTS"
            diagnostic["reason"] = "INSUFFICIENT_POINTS"
            continue
        fitted, center, radius = _section_circle_fit(section_xy, config)
        for key, value in fitted.items():
            if key not in {"index", "lower_hag_m", "upper_hag_m"}:
                diagnostic[key] = value
        diagnostic["index"] = index
        diagnostic["lower_hag_m"] = float(low)
        diagnostic["upper_hag_m"] = float(high)
        # The section midpoint is the stable track coordinate.  HAG is already
        # the caller's exact ground-relative coordinate, so no XYZ-Z fallback
        # is used here.
        diagnostic["center_hag_m"] = float((low + high) * 0.5) if center is not None else None
        if bool(diagnostic.get("accepted")) and center is not None and radius is not None:
            accepted_sections.append(diagnostic)

    accepted_count = len(accepted_sections)
    details: dict[str, Any] = {
        "accepted_slice_indices": [int(section["index"]) for section in accepted_sections],
        "accepted_slice_count": int(accepted_count),
        "fit_point_count": int(sum(int(section["fit_point_count"]) for section in slices)),
    }
    if accepted_count < config.minimum_accepted_sections:
        return _base_result(
            point_count,
            slices,
            "INSUFFICIENT_ACCEPTED_SLICES",
            requested_method=requested_method,
            diagnostics=details,
        )
    first_index = int(accepted_sections[0]["index"])
    last_index = int(accepted_sections[-1]["index"])
    # Three sections confined to one side of 1.30 m do not establish a
    # breast-height stem track.  Require accepted evidence on both sides of
    # that midpoint while retaining the >=3-section guard above.
    if first_index > 2 or last_index < 3:
        details["accepted_slice_span"] = int(last_index - first_index)
        return _base_result(
            point_count,
            slices,
            "ACCEPTED_SLICES_DO_NOT_SPAN_BREAST_HEIGHT",
            requested_method=requested_method,
            diagnostics=details,
        )

    diameters = np.asarray([float(section["diameter_m"]) for section in accepted_sections])
    median_diameter = float(np.median(diameters))
    relative_spread = float((np.max(diameters) - np.min(diameters)) / median_diameter)
    details["diameter_median_m"] = _json_float(median_diameter)
    details["diameter_relative_spread"] = _json_float(relative_spread)
    if relative_spread > config.maximum_relative_diameter_spread + 1.0e-9:
        return _base_result(
            point_count,
            slices,
            "DIAMETER_SPREAD_HIGH",
            requested_method=requested_method,
            diagnostics=details,
        )

    track_valid, track = _track_diagnostics(accepted_sections, config)
    details.update(track)
    if not track_valid:
        return _base_result(
            point_count,
            slices,
            str(track["center_track_status"]),
            requested_method=requested_method,
            diagnostics=details,
        )

    section_rmse = np.asarray(
        [float(section["fit_rmse_m"]) for section in accepted_sections], dtype=np.float64
    )
    aggregate_rmse = float(sqrt(np.mean(section_rmse * section_rmse)))
    diameter_mad = float(np.median(np.abs(diameters - median_diameter)))
    uncertainty_cm = max(0.01, 100.0 * sqrt(aggregate_rmse**2 + (0.5 * 1.4826 * diameter_mad) ** 2))
    details["diameter_mad_m"] = _json_float(diameter_mad)
    details["aggregate_fit_rmse_m"] = _json_float(aggregate_rmse)
    return {
        "dbh_cm": _json_float(median_diameter * 100.0),
        "dbh_method": requested_method,
        "dbh_quality": HIGH,
        "dbh_point_count": int(point_count),
        "dbh_fit_rmse": _json_float(aggregate_rmse),
        "dbh_slice_count": int(accepted_count),
        "dbh_fit_status": PASS,
        "dbh_uncertainty_cm": _json_float(uncertainty_cm),
        "dbh_model_version": DBH_RECOVERY_VERSION,
        "slices": slices,
        "diagnostics": {
            "algorithm_version": DBH_RECOVERY_VERSION,
            "requested_method": requested_method,
            "config": {key: value for key, value in asdict(config).items()},
            **details,
        },
    }


def _valid_training_row(
    row: Any, config: DbhRecoveryConfig = DBH_RECOVERY_CONFIG
) -> tuple[Any, float, float, str] | None:
    if not isinstance(row, dict):
        return None
    identity = row.get("persistent_tree_id")
    if identity is None or isinstance(identity, (dict, list, set, tuple)):
        return None
    method = row.get("dbh_method")
    quality = row.get("dbh_quality")
    if method not in (DIRECT_STEM, RECOVERED_STEM) or quality != HIGH:
        return None
    try:
        height = float(row.get("height_m"))
        dbh = float(row.get("dbh_cm"))
    except (TypeError, ValueError, OverflowError):
        return None
    if not (
        isfinite(height)
        and isfinite(dbh)
        and height > 0.0
        and config.diameter_min_m * 100.0 <= dbh <= config.diameter_max_m * 100.0
    ):
        return None
    return identity, height, dbh, str(method)


def _ols_log_model(heights: np.ndarray, dbhs: np.ndarray) -> tuple[float, float] | None:
    if len(heights) < 2 or len(dbhs) != len(heights):
        return None
    x = np.log(heights)
    y = np.log(dbhs)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    centered = x - float(np.mean(x))
    denominator = float(np.dot(centered, centered))
    if denominator <= 1.0e-14:
        return None
    slope = float(np.dot(centered, y - float(np.mean(y))) / denominator)
    intercept = float(np.mean(y) - slope * np.mean(x))
    if not (isfinite(intercept) and isfinite(slope)):
        return None
    return intercept, slope


def _model_validation(
    identities: list[Any], heights: np.ndarray, dbhs: np.ndarray, config: DbhRecoveryConfig
) -> tuple[dict[str, Any], tuple[float, float] | None]:
    id_strings = [_json_id(identity) for identity in identities]
    base: dict[str, Any] = {
        "model_version": DBH_RECOVERY_VERSION,
        "status": UNAVAILABLE,
        "reason": None,
        "training_count": int(len(identities)),
        "training_ids": id_strings,
        "height_range_m": [
            _json_float(float(np.min(heights))) if len(heights) else None,
            _json_float(float(np.max(heights))) if len(heights) else None,
        ],
        "coefficients": None,
        "intercept": None,
        "slope": None,
        "loo": [],
        "loo_relative_rmse": None,
        "loo_max_relative_error": None,
        "validation_thresholds": {
            "minimum_training_trees": int(config.minimum_training_trees),
            "maximum_loo_relative_rmse": float(config.maximum_loo_relative_rmse),
            "maximum_loo_relative_error": float(config.maximum_loo_relative_error),
        },
    }
    if len(identities) < config.minimum_training_trees:
        base["reason"] = "INSUFFICIENT_TRAINING_DATA"
        return base, None
    fitted = _ols_log_model(heights, dbhs)
    if fitted is None:
        base["reason"] = "DEGENERATE_TRAINING_HEIGHTS"
        return base, None
    intercept, slope = fitted
    base["intercept"] = _json_float(intercept)
    base["slope"] = _json_float(slope)
    base["coefficients"] = {"intercept": _json_float(intercept), "slope": _json_float(slope)}
    if slope <= 0.0:
        base["reason"] = "NONPOSITIVE_SLOPE"
        return base, None

    relative_errors: list[float] = []
    loo_rows: list[dict[str, Any]] = []
    loo_invalid = False
    for index, identity in enumerate(identities):
        mask = np.ones(len(identities), dtype=bool)
        mask[index] = False
        loo_fit = _ols_log_model(heights[mask], dbhs[mask])
        item: dict[str, Any] = {"persistent_tree_id": _json_id(identity), "status": PASS}
        if loo_fit is None or loo_fit[1] <= 0.0:
            item["status"] = "INVALID_LOO_MODEL"
            item["reason"] = "NONPOSITIVE_SLOPE"
            item["predicted_dbh_cm"] = None
            item["observed_dbh_cm"] = _json_float(dbhs[index])
            item["relative_error"] = None
            loo_invalid = True
            loo_rows.append(item)
            continue
        loo_intercept, loo_slope = loo_fit
        try:
            predicted = exp(loo_intercept + loo_slope * log(float(heights[index])))
        except (OverflowError, ValueError):
            predicted = float("nan")
        relative = abs(predicted - float(dbhs[index])) / float(dbhs[index])
        if not (
            isfinite(predicted)
            and config.diameter_min_m * 100.0 <= predicted <= config.diameter_max_m * 100.0
            and isfinite(relative)
        ):
            item["status"] = "INVALID_LOO_PREDICTION"
            item["reason"] = "PREDICTION_OUT_OF_RANGE"
            item["predicted_dbh_cm"] = None
            item["observed_dbh_cm"] = _json_float(dbhs[index])
            item["relative_error"] = None
            loo_invalid = True
        else:
            relative_errors.append(float(relative))
            item["intercept"] = _json_float(loo_intercept)
            item["slope"] = _json_float(loo_slope)
            item["predicted_dbh_cm"] = _json_float(predicted)
            item["observed_dbh_cm"] = _json_float(dbhs[index])
            item["relative_error"] = _json_float(relative)
        loo_rows.append(item)
    base["loo"] = loo_rows
    if loo_invalid or len(relative_errors) != len(identities):
        base["reason"] = "LOO_VALIDATION_FAILED"
        return base, None
    relative_array = np.asarray(relative_errors, dtype=np.float64)
    relative_rmse = float(sqrt(np.mean(relative_array * relative_array)))
    maximum = float(np.max(relative_array))
    base["loo_relative_rmse"] = _json_float(relative_rmse)
    base["loo_max_relative_error"] = _json_float(maximum)
    if relative_rmse > config.maximum_loo_relative_rmse + 1.0e-12:
        base["reason"] = "LOO_RELATIVE_RMSE_HIGH"
        return base, None
    if maximum > config.maximum_loo_relative_error + 1.0e-12:
        base["reason"] = "LOO_MAX_RELATIVE_ERROR_HIGH"
        return base, None
    base["status"] = PASS
    base["reason"] = PASS
    return base, (intercept, slope)


def _allometry_result(
    identity: Any,
    model: dict[str, Any],
    *,
    reason: str,
    height_m: float | None = None,
    estimate_cm: float | None = None,
    uncertainty_cm: float | None = None,
) -> dict[str, Any]:
    if estimate_cm is None:
        return {
            "persistent_tree_id": _json_identity(identity),
            "height_m": _json_float(height_m),
            "dbh_cm": None,
            "dbh_method": UNAVAILABLE,
            "dbh_quality": UNAVAILABLE,
            "dbh_point_count": 0,
            "dbh_fit_rmse": None,
            "dbh_slice_count": 0,
            "dbh_fit_status": reason,
            "dbh_uncertainty_cm": None,
            "dbh_model_version": DBH_RECOVERY_VERSION,
            "model_validation": model,
        }
    return {
        "persistent_tree_id": _json_identity(identity),
        "height_m": _json_float(height_m),
        "dbh_cm": _json_float(estimate_cm),
        "dbh_method": ALLOMETRIC_ESTIMATE,
        "dbh_quality": ESTIMATE,
        "dbh_point_count": 0,
        "dbh_fit_rmse": None,
        "dbh_slice_count": 0,
        "dbh_fit_status": PASS,
        "dbh_uncertainty_cm": _json_float(uncertainty_cm),
        "dbh_model_version": DBH_RECOVERY_VERSION,
        "model_validation": model,
        "is_estimate": True,
    }


def _target_identity_and_height(row: Any) -> tuple[Any, float | None] | None:
    if not isinstance(row, dict):
        return None
    identity = row.get("persistent_tree_id")
    if identity is None or isinstance(identity, (dict, list, set, tuple)):
        return None
    try:
        height = float(row.get("height_m"))
    except (TypeError, ValueError, OverflowError):
        height = None
    if height is not None and not isfinite(height):
        height = None
    return identity, height


def estimate_local_dbh(
    training_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Estimate target DBH with a validated project-local log allometry.

    Only distinct ``HIGH`` direct/recovered rows train the model.  Every
    target receives an explicit estimate or an ``UNAVAILABLE`` reason, and the
    same model-validation record is embedded for auditability.
    """

    config = DBH_RECOVERY_CONFIG
    training_candidates: dict[str, tuple[Any, float, float, str]] = {}
    if isinstance(training_rows, Iterable) and not isinstance(training_rows, (str, bytes, dict)):
        for row in training_rows:
            candidate = _valid_training_row(row, config)
            if candidate is None:
                continue
            identity, height, dbh, method = candidate
            key = _json_id(identity)
            previous = training_candidates.get(key)
            if previous is None or (height, dbh, method) < (previous[1], previous[2], previous[3]):
                training_candidates[key] = candidate
    ordered_training = sorted(training_candidates.values(), key=lambda item: _json_id(item[0]))
    identities = [item[0] for item in ordered_training]
    heights = np.asarray([item[1] for item in ordered_training], dtype=np.float64)
    dbhs = np.asarray([item[2] for item in ordered_training], dtype=np.float64)
    model, fitted = _model_validation(identities, heights, dbhs, config)

    # Group targets by their persistent identity, using a deterministic row if
    # a caller supplied duplicate target records.
    targets: dict[str, tuple[Any, float | None]] = {}
    if isinstance(target_rows, Iterable) and not isinstance(target_rows, (str, bytes, dict)):
        for row in target_rows:
            candidate = _target_identity_and_height(row)
            if candidate is None:
                continue
            identity, height = candidate
            key = _json_id(identity)
            previous = targets.get(key)
            previous_sort = (
                (float("inf"), "")
                if previous is None or previous[1] is None
                else (previous[1], "")
            )
            current_sort = (float("inf"), "") if height is None else (height, "")
            if previous is None or current_sort < previous_sort:
                targets[key] = (identity, height)

    results: dict[str, dict[str, Any]] = {}
    if fitted is None:
        reason = str(model["reason"] or "MODEL_UNAVAILABLE")
        for key in sorted(targets):
            identity, height = targets[key]
            results[key] = _allometry_result(identity, model, reason=reason, height_m=height)
        return results

    intercept, slope = fitted
    low_height, high_height = float(np.min(heights)), float(np.max(heights))
    loo_rmse = float(model["loo_relative_rmse"])
    # The residual scale is intentionally approximate: it communicates model
    # spread without presenting a calibrated confidence interval.
    residual_scale = max(loo_rmse, 1.0e-4)
    for key in sorted(targets):
        identity, height = targets[key]
        if height is None or height <= 0.0:
            results[key] = _allometry_result(
                identity, model, reason="INVALID_TARGET_HEIGHT", height_m=height
            )
            continue
        if height < low_height - 1.0e-12 or height > high_height + 1.0e-12:
            results[key] = _allometry_result(
                identity,
                model,
                reason="TARGET_HEIGHT_OUT_OF_TRAINING_RANGE",
                height_m=height,
            )
            continue
        try:
            estimate = exp(intercept + slope * log(height))
        except (OverflowError, ValueError):
            estimate = float("nan")
        if not (
            isfinite(estimate)
            and config.diameter_min_m * 100.0 <= estimate <= config.diameter_max_m * 100.0
        ):
            results[key] = _allometry_result(
                identity, model, reason="PREDICTED_DBH_OUT_OF_RANGE", height_m=height
            )
            continue
        results[key] = _allometry_result(
            identity,
            model,
            reason=PASS,
            height_m=height,
            estimate_cm=estimate,
            uncertainty_cm=max(0.01, estimate * residual_scale),
        )
    return results


__all__ = [
    "ALLOMETRIC_ESTIMATE",
    "DBH_RECOVERY_CONFIG",
    "DBH_RECOVERY_VERSION",
    "DIRECT_STEM",
    "ESTIMATE",
    "HIGH",
    "PASS",
    "RECOVERED_STEM",
    "RECOVERY_CONFIG",
    "STEM_CONTINUITY_MIN_SECTIONS",
    "STEM_CONTINUITY_MIDPOINT_HAG_M",
    "STEM_CONTINUITY_VOXEL_SIZE_M",
    "UNAVAILABLE",
    "DbhRecoveryConfig",
    "estimate_local_dbh",
    "fit_stem_dbh",
    "select_continuous_stem_points",
]
