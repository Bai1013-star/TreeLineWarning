"""Axis normal DBH fitting for exact current raw stem evidence.

V2 is a side by side geometry helper.  It does not read a source, assign
ownership, or change V1 policy.  A bounded set of six horizontal section
proposals is used to estimate one continuous stem axis.  The selected axis
then defines a temporary normal plane and an axial HAG coordinate before the
unchanged V1 fitter is called.

The input ``xyz_local_m`` and ``hag_m`` arrays are never modified.  Returned
``selected_input_indices`` are zero based indices into the exact arrays the
caller supplied; they are a JSON safe list, rather than a reindexed point
array.  The transformation uses HAG only to fit/interpolate the axis and to
anchor HAG 1.30 m.  It never substitutes world ``XYZ[:, 2]`` for HAG.
"""

from __future__ import annotations

from dataclasses import asdict
from math import acos, atan, degrees, isfinite, sqrt
from typing import Any

import numpy as np

from algorithms.manual_span.dbh_recovery import (
    DBH_RECOVERY_CONFIG,
    PASS,
    UNAVAILABLE,
    _section_circle_fit,
    fit_stem_dbh,
    select_continuous_stem_points,
)

DBH_RAW_STEM_VERSION = "DBH_RECOVERY_V2"
AXIS_NORMAL_BH_HAG_M = 1.30
AXIS_EXPECTED_XY_RADIUS_M = DBH_RECOVERY_CONFIG.stem_search_radius_m
AXIS_CENTER_RESIDUAL_M = DBH_RECOVERY_CONFIG.maximum_center_track_residual_m
AXIS_EQUIVALENT_ANGLE_DEG = 5.0
# Candidate axes within this fixed physical distance are the same stem.  A
# second axis farther than this is retained as an independent competitor and
# therefore fails closed instead of being ranked by diameter.
AXIS_EQUIVALENT_OFFSET_M = 0.15


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if isfinite(converted) else None


def _json_vector(value: np.ndarray | list[float] | tuple[float, ...]) -> list[float] | None:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        return None
    return [float(item) for item in array]


def _safe_count(value: Any) -> int:
    try:
        array = np.asarray(value)
    except Exception:
        return 0
    return int(array.shape[0]) if array.ndim else 0


def _sort_input(xyz: np.ndarray, hag: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort evidence by values and retain original row indices."""

    original = np.arange(len(xyz), dtype=np.int64)
    order = np.lexsort((original, xyz[:, 2], xyz[:, 1], xyz[:, 0], hag))
    return xyz[order], hag[order], original[order]


def _empty_section(index: int, low: float, high: float, reason: str) -> dict[str, Any]:
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
        "center_z_m": None,
        "center_hag_m": None,
        "angular_span_deg": None,
        "occupied_sector_count": 0,
        "fit_rmse_m": None,
        "proposal_ready": False,
        "accepted": False,
        "status": "SKIPPED",
        "reason": reason,
    }


def _base_unavailable(
    point_count: int,
    status: str,
    *,
    sections: list[dict[str, Any]] | None = None,
    axis: dict[str, Any] | None = None,
    selected_indices: list[int] | None = None,
) -> dict[str, Any]:
    selected = sorted(int(item) for item in (selected_indices or []))
    axis_detail = axis or {
        "status": status,
        "reason": status,
        "preliminary_sections": sections or [],
        "proposals": [],
        "candidates": [],
    }
    return {
        "dbh_cm": None,
        "dbh_method": UNAVAILABLE,
        "dbh_quality": UNAVAILABLE,
        "dbh_point_count": int(point_count),
        "dbh_fit_rmse": None,
        "dbh_slice_count": 0,
        "dbh_fit_status": status,
        "dbh_uncertainty_cm": None,
        "dbh_model_version": DBH_RAW_STEM_VERSION,
        "slices": sections or [],
        "axis": axis_detail,
        "unit_direction_xyz": axis_detail.get("unit_direction_xyz"),
        "breast_height_origin_xyz": axis_detail.get("breast_height_origin_xyz"),
        "selected_input_indices": selected,
        "selected_input_count": len(selected),
        "diagnostics": {
            "algorithm_version": DBH_RAW_STEM_VERSION,
            "status": status,
            "config": {key: value for key, value in asdict(DBH_RECOVERY_CONFIG).items()},
            "axis": axis_detail,
        },
    }


def _line_distance(points: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    offsets = points - origin
    along = offsets @ direction
    residual = offsets - along[:, None] * direction[None, :]
    return np.linalg.norm(residual, axis=1)


def _fit_hag_line(
    points: np.ndarray, heights: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit XYZ as a line parameterized by original HAG values."""

    if len(indices) < 2:
        return None
    selected_points = points[indices]
    selected_heights = heights[indices]
    design = np.column_stack((np.ones(len(indices)), selected_heights - AXIS_NORMAL_BH_HAG_M))
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(design, selected_points, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if int(rank) < 2:
        return None
    bh = np.asarray(coefficients[0], dtype=np.float64)
    slope = np.asarray(coefficients[1], dtype=np.float64)
    slope_norm = float(np.linalg.norm(slope))
    if slope_norm <= 1.0e-10 or not np.all(np.isfinite(bh)) or not np.all(np.isfinite(slope)):
        return None
    direction = slope / slope_norm
    if direction[2] < 0.0:
        direction = -direction
        slope = -slope
    if direction[2] <= 1.0e-8:
        return None
    return bh, direction, slope


def _candidate_from_pair(
    centers: np.ndarray,
    heights: np.ndarray,
    section_indices: np.ndarray,
    first: int,
    second: int,
) -> dict[str, Any]:
    """Build one bounded robust line candidate from a center pair."""

    center_pair = [int(first), int(second)]
    pair = [int(section_indices[first]), int(section_indices[second])]
    diagnostic: dict[str, Any] = {
        "pair": pair,
        "center_pair": center_pair,
        "status": "REJECTED",
        "reason": "REJECTED",
        "inlier_section_indices": [],
        "inlier_count": 0,
        "residuals_m": [],
        "max_perpendicular_residual_m": None,
        "rms_perpendicular_residual_m": None,
        "tilt_angle_deg": None,
        "bh_reference_xyz": None,
        "direction_xyz": None,
        "slope_xyz_per_hag": None,
        "expected_xy_distance_m": None,
        "brackets_true_bh": False,
    }
    if second <= first:
        diagnostic["reason"] = "INVALID_PAIR"
        return diagnostic
    delta = centers[second] - centers[first]
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-10:
        diagnostic["reason"] = "DEGENERATE_PAIR"
        return diagnostic
    initial_direction = delta / norm
    if initial_direction[2] < 0.0:
        initial_direction = -initial_direction
    if initial_direction[2] <= 1.0e-8:
        diagnostic["reason"] = "AXIS_NOT_UPWARD"
        return diagnostic

    inliers = np.asarray((first, second), dtype=np.int64)
    line: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for _ in range(4):
        line = _fit_hag_line(centers, heights, inliers)
        if line is None:
            diagnostic["reason"] = "AXIS_LINE_FIT_FAILED"
            return diagnostic
        bh, direction, slope = line
        residuals = _line_distance(centers, bh, direction)
        nearby = np.flatnonzero(residuals <= AXIS_CENTER_RESIDUAL_M + 1.0e-12)
        # Keep at most one proposal per horizontal section.  Otherwise a
        # primary and secondary circle from one slice could count as two
        # agreeing centers and fabricate a three-section axis.
        best_by_section: dict[int, int] = {}
        for position in nearby:
            label = int(section_indices[position])
            previous = best_by_section.get(label)
            if previous is None or (residuals[position], int(position)) < (
                residuals[previous],
                int(previous),
            ):
                best_by_section[label] = int(position)
        new_inliers = np.asarray(sorted(best_by_section.values()), dtype=np.int64)
        if len(new_inliers) < 2:
            diagnostic["reason"] = "AXIS_CENTER_RESIDUAL_HIGH"
            return diagnostic
        if np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers

    if line is None:
        diagnostic["reason"] = "AXIS_LINE_FIT_FAILED"
        return diagnostic
    bh, direction, slope = line
    residuals = _line_distance(centers, bh, direction)
    nearby = np.flatnonzero(residuals <= AXIS_CENTER_RESIDUAL_M + 1.0e-12)
    best_by_section: dict[int, int] = {}
    for position in nearby:
        label = int(section_indices[position])
        previous = best_by_section.get(label)
        if previous is None or (residuals[position], int(position)) < (
            residuals[previous],
            int(previous),
        ):
            best_by_section[label] = int(position)
    inliers = np.asarray(sorted(best_by_section.values()), dtype=np.int64)
    inlier_count = int(len(inliers))
    diagnostic["inlier_section_indices"] = sorted(
        {int(section_indices[item]) for item in inliers}
    )
    diagnostic["inlier_count"] = inlier_count
    diagnostic["residuals_m"] = [float(item) for item in residuals]
    if inlier_count:
        selected_residuals = residuals[inliers]
        diagnostic["max_perpendicular_residual_m"] = _finite_float(np.max(selected_residuals))
        diagnostic["rms_perpendicular_residual_m"] = _finite_float(
            sqrt(float(np.mean(selected_residuals * selected_residuals)))
        )
    tilt = degrees(atan(float(np.linalg.norm(direction[:2])) / float(direction[2])))
    diagnostic["tilt_angle_deg"] = _finite_float(tilt)
    diagnostic["bh_reference_xyz"] = _json_vector(bh)
    diagnostic["direction_xyz"] = _json_vector(direction)
    diagnostic["slope_xyz_per_hag"] = _json_vector(slope)
    below = (
        bool(np.any(heights[inliers] < AXIS_NORMAL_BH_HAG_M - 1.0e-9))
        if inlier_count
        else False
    )
    above = (
        bool(np.any(heights[inliers] > AXIS_NORMAL_BH_HAG_M + 1.0e-9))
        if inlier_count
        else False
    )
    diagnostic["brackets_true_bh"] = bool(below and above)
    if inlier_count < DBH_RECOVERY_CONFIG.minimum_accepted_sections:
        diagnostic["reason"] = "INSUFFICIENT_AGREEING_CENTERS"
    elif not diagnostic["brackets_true_bh"]:
        diagnostic["reason"] = "AXIS_DOES_NOT_BRACKET_TRUE_BH"
    elif tilt > DBH_RECOVERY_CONFIG.maximum_tilt_deg + 1.0e-9:
        diagnostic["reason"] = "TILT_EXCEEDS_20_DEG"
    elif diagnostic["max_perpendicular_residual_m"] is None or diagnostic[
        "max_perpendicular_residual_m"
    ] > AXIS_CENTER_RESIDUAL_M + 1.0e-12:
        diagnostic["reason"] = "AXIS_CENTER_RESIDUAL_HIGH"
    else:
        diagnostic["status"] = PASS
        diagnostic["reason"] = PASS
    return diagnostic


def _line_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(abs(np.dot(first, second)), -1.0, 1.0))
    return degrees(acos(cosine))


def _line_offset_m(
    first_origin: np.ndarray, first_direction: np.ndarray, second_origin: np.ndarray
) -> float:
    delta = second_origin - first_origin
    perpendicular = delta - float(np.dot(delta, first_direction)) * first_direction
    return float(np.linalg.norm(perpendicular))


def _line_candidates_are_equivalent(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_direction = np.asarray(first["direction_xyz"], dtype=np.float64)
    second_direction = np.asarray(second["direction_xyz"], dtype=np.float64)
    first_origin = np.asarray(first["bh_reference_xyz"], dtype=np.float64)
    second_origin = np.asarray(second["bh_reference_xyz"], dtype=np.float64)
    return (
        _line_angle_deg(first_direction, second_direction) <= AXIS_EQUIVALENT_ANGLE_DEG
        and _line_offset_m(first_origin, first_direction, second_origin) <= AXIS_EQUIVALENT_OFFSET_M
    )


def _axis_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Choose a deterministic orthonormal basis normal to the axis."""

    reference = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    first = np.cross(reference, direction)
    if np.linalg.norm(first) <= 1.0e-10:
        reference = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        first = np.cross(reference, direction)
    first_norm = float(np.linalg.norm(first))
    if first_norm <= 1.0e-10:
        return None
    first /= first_norm
    second = np.cross(direction, first)
    second_norm = float(np.linalg.norm(second))
    if second_norm <= 1.0e-10:
        return None
    second /= second_norm
    return first, second


def _section_proposals(
    xyz: np.ndarray, hag: np.ndarray
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Fit six sections and retain at most two bounded axis proposals each.

    The secondary proposal removes only the primary circle's radial inliers
    before an independent proposal fit.  This is proposal enumeration for
    competing axes; the final V1 DBH fit always receives its full candidate
    evidence (or the separate continuity fallback) and never this residual
    mask.
    """

    config = DBH_RECOVERY_CONFIG
    sections: list[dict[str, Any]] = []
    centers: list[np.ndarray] = []
    center_heights: list[float] = []
    center_section_indices: list[int] = []

    def add_proposal(
        section: dict[str, Any],
        fitted: dict[str, Any],
        section_xyz: np.ndarray,
        section_hag: np.ndarray,
        section_xy: np.ndarray,
        index: int,
        rank: int,
        source: str,
    ) -> tuple[bool, np.ndarray | None, float | None]:
        """Decorate a proposal and append its center if proposal guards pass."""

        proposal = _empty_section(
            index, section["lower_hag_m"], section["upper_hag_m"], "FIT_FAILED"
        )
        proposal["proposal_rank"] = int(rank)
        proposal["proposal_source"] = source
        proposal["input_point_count"] = int(len(section_xyz))
        proposal["unique_point_count"] = int(len(section_xy))
        for key, value in fitted.items():
            if key not in {"index", "lower_hag_m", "upper_hag_m", "center_hag_m"}:
                proposal[key] = value
        proposal["index"] = int(index)
        proposal["lower_hag_m"] = float(section["lower_hag_m"])
        proposal["upper_hag_m"] = float(section["upper_hag_m"])
        radius = _finite_float(proposal.get("radius_m"))
        center_x = _finite_float(proposal.get("center_x_m"))
        center_y = _finite_float(proposal.get("center_y_m"))
        anchor_mask = np.zeros(len(section_xyz), dtype=bool)
        if radius is not None and center_x is not None and center_y is not None:
            residuals = np.hypot(
                section_xy[:, 0] - center_x, section_xy[:, 1] - center_y
            ) - radius
            tolerance = min(
                config.maximum_inlier_tolerance_m,
                max(
                    config.minimum_inlier_tolerance_m,
                    config.radial_tolerance_fraction * radius,
                ),
            )
            radial_inlier_xy = np.abs(residuals) <= tolerance + 1.0e-12
            inlier_xy = section_xy[radial_inlier_xy]
            if len(inlier_xy):
                inlier_xy_set = {tuple(row) for row in inlier_xy}
                anchor_mask = np.fromiter(
                    (tuple(row[:2]) in inlier_xy_set for row in section_xyz),
                    dtype=bool,
                    count=len(section_xyz),
                )
        anchor_count = int(np.count_nonzero(anchor_mask))
        center_hag = (
            float(np.median(section_hag[anchor_mask])) if anchor_count else None
        )
        center_z = (
            float(np.median(section_xyz[anchor_mask, 2])) if anchor_count else None
        )
        proposal["anchor_inlier_point_count"] = anchor_count
        proposal["anchor_inlier_unique_xy_count"] = (
            int(np.count_nonzero(radial_inlier_xy))
            if radius is not None
            and center_x is not None
            and center_y is not None
            and len(section_xy)
            else 0
        )
        proposal["center_hag_m"] = center_hag
        proposal["center_z_m"] = _finite_float(center_z)
        diameter = _finite_float(proposal.get("diameter_m"))
        rmse = _finite_float(proposal.get("fit_rmse_m"))
        tolerance = None
        if radius is not None:
            tolerance = min(
                config.maximum_inlier_tolerance_m,
                max(config.minimum_inlier_tolerance_m, config.radial_tolerance_fraction * radius),
            )
        proposal_ready = bool(
            proposal.get("inlier_point_count", 0) >= config.minimum_points_per_section
            and _finite_float(proposal.get("angular_span_deg")) is not None
            and float(proposal["angular_span_deg"]) >= config.minimum_angular_span_deg
            and int(proposal.get("occupied_sector_count", 0)) >= config.minimum_occupied_sectors
            and diameter is not None
            and config.diameter_min_m <= diameter <= config.diameter_max_m
            and rmse is not None
            and tolerance is not None
            and rmse <= tolerance + 1.0e-12
            and _finite_float(proposal.get("center_x_m")) is not None
            and _finite_float(proposal.get("center_y_m")) is not None
            and anchor_count >= config.minimum_points_per_section
            and center_hag is not None
            and center_z is not None
        )
        proposal["proposal_ready"] = proposal_ready
        proposal["status"] = "PROPOSAL" if proposal_ready else "PROPOSAL_REJECTED"
        proposal["reason"] = "PROPOSAL" if proposal_ready else "AXIS_PROPOSAL_GUARDS_FAILED"
        section.setdefault("proposals", []).append(proposal)
        if not proposal_ready:
            return False, None, None
        center = np.asarray(
            (float(proposal["center_x_m"]), float(proposal["center_y_m"]), float(center_z)),
            dtype=np.float64,
        )
        centers.append(center)
        center_heights.append(float(center_hag))
        center_section_indices.append(int(index))
        return True, center, radius

    for index in range(config.section_count):
        low = config.breast_height_min_m + index * config.section_height_m
        high = config.breast_height_min_m + (index + 1) * config.section_height_m
        if index == config.section_count - 1:
            section_mask = (hag >= low) & (hag <= high)
        else:
            section_mask = (hag >= low) & (hag < high)
        section_xyz = xyz[section_mask]
        section_hag = hag[section_mask]
        diagnostic = _empty_section(index, low, high, "INSUFFICIENT_POINTS")
        diagnostic["input_point_count"] = int(len(section_xyz))
        section_xy = (
            np.unique(section_xyz[:, :2], axis=0)
            if len(section_xyz)
            else np.empty((0, 2), dtype=np.float64)
        )
        diagnostic["unique_point_count"] = int(len(section_xy))
        diagnostic["proposals"] = []
        if len(section_xy) < config.minimum_points_per_section:
            sections.append(diagnostic)
            continue

        primary_fit, _, _ = _section_circle_fit(section_xy, config)
        primary_ready, primary_center, primary_radius = add_proposal(
            diagnostic,
            primary_fit,
            section_xyz,
            section_hag,
            section_xy,
            index,
            0,
            "PRIMARY_HORIZONTAL_SECTION",
        )
        # Copy the primary proposal fields to the stable section-level view so
        # existing diagnostics consumers need not understand the nested list.
        if diagnostic["proposals"]:
            for key, value in diagnostic["proposals"][0].items():
                if key not in {"proposals", "proposal_rank", "proposal_source"}:
                    diagnostic[key] = value
        if primary_center is not None and primary_radius is not None:
            primary_residuals = np.hypot(
                section_xy[:, 0] - primary_center[0], section_xy[:, 1] - primary_center[1]
            ) - primary_radius
            tolerance = min(
                config.maximum_inlier_tolerance_m,
                max(
                    config.minimum_inlier_tolerance_m,
                    config.radial_tolerance_fraction * primary_radius,
                ),
            )
            remaining = section_xy[np.abs(primary_residuals) > tolerance]
            if len(remaining) >= config.minimum_points_per_section:
                remaining_set = {tuple(row) for row in remaining}
                remaining_mask = np.asarray(
                    [tuple(row[:2]) in remaining_set for row in section_xyz], dtype=bool
                )
                secondary_fit, _, _ = _section_circle_fit(remaining, config)
                add_proposal(
                    diagnostic,
                    secondary_fit,
                    section_xyz[remaining_mask],
                    section_hag[remaining_mask],
                    remaining,
                    index,
                    1,
                    "SECONDARY_AFTER_PRIMARY_INLIER_REMOVAL",
                )
        diagnostic["proposal_fit_count"] = int(len(diagnostic["proposals"]))
        diagnostic["primary_proposal_ready"] = bool(primary_ready)
        diagnostic["secondary_proposal_ready"] = bool(
            any(
                item.get("proposal_rank") == 1 and item.get("proposal_ready")
                for item in diagnostic["proposals"]
            )
        )
        diagnostic["status"] = "PROPOSAL" if primary_ready else "PROPOSAL_REJECTED"
        diagnostic["reason"] = "PROPOSAL" if primary_ready else "AXIS_PROPOSAL_GUARDS_FAILED"
        sections.append(diagnostic)
    return (
        sections,
        np.asarray(centers, dtype=np.float64).reshape((-1, 3)),
        np.asarray(center_heights, dtype=np.float64),
        np.asarray(center_section_indices, dtype=np.int64),
    )


def _candidate_axis_detail(
    proposals: list[dict[str, Any]], expected_xy: np.ndarray | None
) -> tuple[dict[str, Any] | None, str]:
    valid = [item for item in proposals if item.get("status") == PASS]
    if expected_xy is not None:
        for item in valid:
            bh = np.asarray(item["bh_reference_xyz"], dtype=np.float64)
            item["expected_xy_distance_m"] = _finite_float(np.linalg.norm(bh[:2] - expected_xy))
            if item["expected_xy_distance_m"] > AXIS_EXPECTED_XY_RADIUS_M + 1.0e-12:
                item["status"] = "REJECTED"
                item["reason"] = "EXPECTED_XY_OUTSIDE_RADIUS"
        valid = [item for item in valid if item.get("status") == PASS]
    if not valid:
        return None, "NO_VALID_AXIS_PROPOSAL"

    clusters: list[list[dict[str, Any]]] = []
    for candidate in valid:
        for cluster in clusters:
            if _line_candidates_are_equivalent(candidate, cluster[0]):
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    if len(clusters) > 1:
        for candidate in valid:
            candidate["status"] = "AMBIGUOUS"
            candidate["reason"] = "AMBIGUOUS_COMPETING_AXIS"
        return None, "AMBIGUOUS_COMPETING_AXES"

    ranked = sorted(
        valid,
        key=lambda item: (
            -int(item["inlier_count"]),
            float(item["rms_perpendicular_residual_m"]),
            float(item["max_perpendicular_residual_m"]),
            float(item["tilt_angle_deg"]),
            tuple(int(value) for value in item["pair"]),
        ),
    )
    selected = ranked[0]
    for candidate in valid:
        candidate["status"] = "SELECTED" if candidate is selected else "COMPETING_SAME_AXIS"
        candidate["reason"] = PASS if candidate is selected else "SAME_AXIS_ALTERNATIVE"
    return selected, PASS


def _axis_transform(
    xyz: np.ndarray, axis: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    bh = np.asarray(axis["bh_reference_xyz"], dtype=np.float64)
    direction = np.asarray(axis["direction_xyz"], dtype=np.float64)
    basis = _axis_basis(direction)
    if basis is None:
        return None
    first, second = basis
    relative = xyz - bh[None, :]
    axial = relative @ direction
    normal_first = relative @ first
    normal_second = relative @ second
    transformed_xyz = np.column_stack((normal_first, normal_second, axial))
    transformed_hag = AXIS_NORMAL_BH_HAG_M + axial
    transform = {
        "basis_u_xyz": _json_vector(first),
        "basis_v_xyz": _json_vector(second),
        "axis_direction_xyz": _json_vector(direction),
        "bh_reference_xyz": _json_vector(bh),
        "bh_hag_m": float(AXIS_NORMAL_BH_HAG_M),
        "scalar_definition": (
            "normal_hag_m = 1.30 + dot(xyz_local_m - bh_reference_xyz, "
            "axis_direction_xyz); source HAG remains unchanged"
        ),
    }
    return transformed_xyz, transformed_hag, transform


def _result_with_axis(
    v1: dict[str, Any],
    *,
    point_count: int,
    sections: list[dict[str, Any]],
    axis: dict[str, Any],
    selected_indices: list[int],
    transform: dict[str, Any],
    fallback: bool,
) -> dict[str, Any]:
    result = {
        "dbh_cm": v1.get("dbh_cm"),
        "dbh_method": v1.get("dbh_method", UNAVAILABLE),
        "dbh_quality": v1.get("dbh_quality", UNAVAILABLE),
        "dbh_point_count": int(point_count),
        "dbh_fit_rmse": v1.get("dbh_fit_rmse"),
        "dbh_slice_count": int(v1.get("dbh_slice_count", 0)),
        "dbh_fit_status": v1.get("dbh_fit_status", UNAVAILABLE),
        "dbh_uncertainty_cm": v1.get("dbh_uncertainty_cm"),
        "dbh_model_version": DBH_RAW_STEM_VERSION,
        "slices": v1.get("slices", []),
        "unit_direction_xyz": axis.get(
            "unit_direction_xyz", axis.get("direction_xyz")
        ),
        "breast_height_origin_xyz": axis.get(
            "breast_height_origin_xyz", axis.get("bh_reference_xyz")
        ),
        "axis": {
            **axis,
            "status": PASS if v1.get("dbh_fit_status") == PASS else "AXIS_PASS_FINAL_FIT_FAILED",
            "preliminary_sections": sections,
            "transform": transform,
            "continuity_fallback_used": bool(fallback),
            "final_fit_status": v1.get("dbh_fit_status", UNAVAILABLE),
        },
        "selected_input_indices": sorted(int(item) for item in selected_indices),
        "selected_input_count": len(selected_indices),
        "diagnostics": {
            "algorithm_version": DBH_RAW_STEM_VERSION,
            "v1_fit": v1.get("diagnostics", {}),
            "axis": {
                **axis,
                "preliminary_sections": sections,
                "transform": transform,
                "continuity_fallback_used": bool(fallback),
                "final_fit_status": v1.get("dbh_fit_status", UNAVAILABLE),
            },
        },
    }
    return result


def fit_axis_normal_dbh(
    xyz: np.ndarray,
    hag: np.ndarray,
    *,
    expected_xy: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit DBH in a robust stem-axis normal plane.

    The axis proposals are derived from at most twelve original horizontal
    section fits (a primary and one secondary proposal per section) and at
    most sixty-six bounded section-pair line candidates.  A winning line is
    anchored at true HAG 1.30 by XYZ interpolation against the original HAG
    values.  Transformed axial distance defines the temporary section
    coordinate; V1's final circle guards are then applied unchanged.
    """

    point_count = _safe_count(xyz)
    try:
        xyz_array = np.asarray(xyz, dtype=np.float64)
        hag_array = np.asarray(hag, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return _base_unavailable(point_count, "INVALID_INPUT")
    if xyz_array.ndim != 2 or xyz_array.shape[1:] != (3,):
        return _base_unavailable(point_count, "INVALID_XYZ_SHAPE")
    if hag_array.ndim != 1 or len(hag_array) != len(xyz_array):
        return _base_unavailable(len(xyz_array), "INVALID_HAG_SHAPE")
    if len(xyz_array) == 0:
        return _base_unavailable(0, "INSUFFICIENT_INPUT_POINTS")
    if not np.all(np.isfinite(xyz_array)) or not np.all(np.isfinite(hag_array)):
        return _base_unavailable(len(xyz_array), "NONFINITE_INPUT")
    expected: np.ndarray | None = None
    if expected_xy is not None:
        try:
            expected = np.asarray(expected_xy, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return _base_unavailable(len(xyz_array), "INVALID_EXPECTED_XY")
        if expected.shape != (2,) or not np.all(np.isfinite(expected)):
            return _base_unavailable(len(xyz_array), "INVALID_EXPECTED_XY")

    sorted_xyz, sorted_hag, original_indices = _sort_input(xyz_array, hag_array)
    sections, centers, center_heights, section_indices = _section_proposals(sorted_xyz, sorted_hag)
    proposals: list[dict[str, Any]] = []
    for left in range(len(centers)):
        for right in range(left + 1, len(centers)):
            candidate = _candidate_from_pair(
                centers, center_heights, section_indices, left, right
            )
            candidate["section_pair"] = [
                int(section_indices[left]),
                int(section_indices[right]),
            ]
            proposals.append(candidate)
    axis, axis_status = _candidate_axis_detail(proposals, expected)
    axis_detail: dict[str, Any] = {
        "status": axis_status,
        "reason": axis_status,
        "preliminary_sections": sections,
        "proposals": proposals,
        "candidates": proposals,
        "plausible_candidates": [
            item
            for item in proposals
            if item.get("status") in {PASS, "SELECTED", "COMPETING_SAME_AXIS", "AMBIGUOUS"}
        ],
        "proposal_section_indices": [int(item) for item in section_indices],
        "proposal_count": len(proposals),
        "stem_candidate_count": int(
            sum(
                item.get("status")
                in {PASS, "SELECTED", "COMPETING_SAME_AXIS", "AMBIGUOUS"}
                for item in proposals
            )
        ),
        "preliminary_fit_count": int(
            sum(len(item.get("proposals", [])) for item in sections)
        ),
        "preliminary_fit_limit": int(2 * DBH_RECOVERY_CONFIG.section_count),
        "bracket_hag_m": float(AXIS_NORMAL_BH_HAG_M),
        "center_residual_limit_m": float(AXIS_CENTER_RESIDUAL_M),
        "tilt_limit_deg": float(DBH_RECOVERY_CONFIG.maximum_tilt_deg),
        "expected_xy": _json_vector(expected) if expected is not None else None,
    }
    if axis is None:
        return _base_unavailable(
            len(xyz_array), axis_status, sections=sections, axis=axis_detail
        )
    axis_detail["unit_direction_xyz"] = axis["direction_xyz"]
    axis_detail["breast_height_origin_xyz"] = axis["bh_reference_xyz"]

    transformed = _axis_transform(sorted_xyz, axis)
    if transformed is None:
        axis_detail["status"] = "AXIS_BASIS_FAILED"
        axis_detail["reason"] = "AXIS_BASIS_FAILED"
        return _base_unavailable(
            len(xyz_array), "AXIS_BASIS_FAILED", sections=sections, axis=axis_detail
        )
    transformed_xyz, transformed_hag, transform = transformed
    band = (transformed_hag >= DBH_RECOVERY_CONFIG.breast_height_min_m) & (
        transformed_hag <= DBH_RECOVERY_CONFIG.breast_height_max_m
    )
    band_sorted_positions = np.flatnonzero(band)
    v1_result = fit_stem_dbh(transformed_xyz, transformed_hag, method="RECOVERED_STEM")
    selected_positions = band_sorted_positions
    fallback = False
    if v1_result.get("dbh_fit_status") != PASS:
        continuity_mask, continuity_diagnostics = select_continuous_stem_points(
            transformed_xyz, transformed_hag
        )
        axis_detail["continuity_diagnostics"] = continuity_diagnostics
        if np.any(continuity_mask):
            fallback_result = fit_stem_dbh(
                transformed_xyz[continuity_mask],
                transformed_hag[continuity_mask],
                method="RECOVERED_STEM",
            )
            axis_detail["continuity_fallback_fit"] = fallback_result
            if fallback_result.get("dbh_fit_status") == PASS:
                v1_result = fallback_result
                selected_positions = np.flatnonzero(continuity_mask)
                fallback = True
        else:
            axis_detail["continuity_fallback_fit"] = {
                "attempted": False,
                "status": continuity_diagnostics.get("status", "NO_CONTINUOUS_STEM_COLUMNS"),
            }
    selected_input = [int(item) for item in original_indices[selected_positions]]
    axis_detail["status"] = PASS
    axis_detail["reason"] = PASS if v1_result.get("dbh_fit_status") == PASS else "FINAL_FIT_FAILED"
    return _result_with_axis(
        v1_result,
        point_count=len(xyz_array),
        sections=sections,
        axis=axis_detail,
        selected_indices=selected_input,
        transform=transform,
        fallback=fallback,
    )


__all__ = [
    "AXIS_CENTER_RESIDUAL_M",
    "AXIS_EXPECTED_XY_RADIUS_M",
    "AXIS_NORMAL_BH_HAG_M",
    "DBH_RAW_STEM_VERSION",
    "fit_axis_normal_dbh",
]
