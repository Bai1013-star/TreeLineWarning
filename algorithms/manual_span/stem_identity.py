"""Crown-to-stem identity evidence; separate from immutable DBH fit policy."""

from __future__ import annotations

import copy

import numpy as np
from scipy.spatial import cKDTree

from algorithms.manual_span.dbh_raw_stem import (
    _axis_basis,
    _axis_transform,
    _candidate_from_pair,
    _fit_hag_line,
    _line_candidates_are_equivalent,
    _section_proposals,
)
from algorithms.manual_span.dbh_recovery import (
    DBH_RECOVERY_CONFIG as CFG,
)
from algorithms.manual_span.dbh_recovery import (
    _section_circle_fit,
    fit_stem_dbh,
    select_continuous_stem_points,
)

VERSION = "DBH_RECOVERY_V3"
LAYERS = np.round(np.arange(1.05, 3.301, 0.25), 3)
UPPER_SHIFT_M = 1.50
MAX_AXIS_CANDIDATES = 6


def crown_anchor(xyz, hag):
    xyz, hag = np.asarray(xyz), np.asarray(hag)
    if xyz.ndim != 2 or xyz.shape != (len(hag), 3) or not len(hag):
        raise ValueError("INVALID_CURRENT_CROWN")
    if not np.isfinite(xyz).all() or not np.isfinite(hag).all():
        raise ValueError("INVALID_CURRENT_CROWN")
    cells = np.unique(np.floor(xyz[:, :2] / 0.35).astype(np.int64), axis=0)
    footprint = (cells + 0.5) * 0.35
    centroid = footprint.mean(axis=0)
    apex = np.median(xyz[hag >= np.percentile(hag, 95), :2], axis=0)
    radius = max(0.60, float(np.percentile(np.linalg.norm(footprint - centroid, axis=1), 95)))
    return {
        "centroid_xy": centroid.tolist(),
        "apex_xy": apex.tolist(),
        "footprint_radius_m": radius,
        "maximum_hag_m": float(hag.max()),
        "member_count": len(xyz),
        "footprint_cells": footprint,
        "footprint_tree": cKDTree(footprint),
    }


def distinct_axes(proposals):
    # Retain deterministic V2 proposal order, but choose the strongest representative
    # of each equivalence group; changing tree iteration order cannot alter this.
    groups = []
    for proposal in proposals:
        for group in groups:
            if _line_candidates_are_equivalent(proposal, group[0]):
                group.append(proposal)
                break
        else:
            groups.append([proposal])
    return [
        copy.deepcopy(
            min(
                group,
                key=lambda p: (
                    -p["inlier_count"],
                    p["rms_perpendicular_residual_m"],
                    tuple(p["bh_reference_xyz"]),
                    tuple(p["direction_xyz"]),
                ),
            )
        )
        for group in groups
    ]


def axis_radius(axis, sections):
    origin, direction = np.asarray(axis["bh_reference_xyz"]), np.asarray(axis["direction_xyz"])
    radii = []
    for section in sections:
        for circle in section.get("proposals", []):
            if not circle.get("proposal_ready"):
                continue
            point = np.array([circle["center_x_m"], circle["center_y_m"], circle["center_z_m"]])
            delta = point - origin
            if np.linalg.norm(delta - (delta @ direction) * direction) <= 0.03 + 1e-12:
                radii.append(circle["radius_m"])
    if not radii:
        raise ValueError("AXIS_RADIUS_UNPROVEN")
    return float(np.median(radii))


def upper_axes(xyz, hag):
    """Upper-section proposals only; lower tracking must replace extrapolated BH."""
    sections, centers, heights, labels = _section_proposals(xyz, hag - UPPER_SHIFT_M)
    candidates = []
    for left in range(len(centers)):
        for right in range(left + 1, len(centers)):
            axis = _candidate_from_pair(centers, heights, labels, left, right)
            if axis["status"] != "PASS":
                continue
            axis["bh_reference_xyz"] = (
                np.asarray(axis["bh_reference_xyz"])
                - UPPER_SHIFT_M * np.asarray(axis["slope_xyz_per_hag"])
            ).tolist()
            axis["proposal_source"] = "CURRENT_UPPER_SECTIONS"
            axis["anchor_status"] = "EXTRAPOLATED_PROPOSAL_ONLY_REQUIRES_DOWNWARD_TRACK"
            axis["radius_m"] = axis_radius(axis, sections)
            candidates.append(axis)
    axes = distinct_axes(candidates)
    proof = {
        "shift_m": UPPER_SHIFT_M,
        "sections": sections,
        "pair_proposal_count": len(candidates),
        "distinct_axis_count": len(axes),
    }
    if len(axes) > MAX_AXIS_CANDIDATES:
        return [], {**proof, "status": "UPPER_CANDIDATE_BUDGET_AMBIGUOUS"}
    return axes, {**proof, "status": "PASS" if axes else "NO_UPPER_AXIS"}


def track_downward(xyz, hag, original_index, axis):
    """Evaluate all layers from crown toward BH, retaining exact support rows."""
    origin, direction = np.asarray(axis["bh_reference_xyz"]), np.asarray(axis["direction_xyz"])
    slope = np.asarray(axis["slope_xyz_per_hag"])
    basis = _axis_basis(direction)
    if basis is None:
        raise ValueError("INVALID_AXIS_BASIS")
    u, v = basis
    relative = xyz - origin
    axial = relative @ direction
    plane = np.column_stack((relative @ u, relative @ v))
    tube = axis["radius_m"] + 0.08
    layers, support, centers, heights = [], [], [], []
    predicted_center = np.zeros(2)
    for h in LAYERS[::-1]:
        position = float((slope * (h - 1.30)) @ direction)
        radial = np.linalg.norm(plane - predicted_center, axis=1)
        mask = (np.abs(axial - position) <= CFG.section_height_m / 2) & (radial <= tube)
        indices = np.flatnonzero(mask)
        if len(indices) < CFG.minimum_points_per_section:
            layers.append(
                {
                    "hag_target_m": float(h),
                    "input_count": len(indices),
                    "status": "REJECTED",
                    "reason": "INSUFFICIENT_POINTS",
                }
            )
            continue
        circle, _, _ = _section_circle_fit(np.unique(plane[indices], axis=0), CFG)
        layer = {
            "hag_target_m": float(h),
            "input_count": len(indices),
            "circle": circle,
            "status": "REJECTED",
        }
        if circle.get("accepted"):
            center = np.array([circle["center_x_m"], circle["center_y_m"]])
            offset = float(np.linalg.norm(center - predicted_center))
            tolerance = min(
                CFG.maximum_inlier_tolerance_m,
                max(
                    CFG.minimum_inlier_tolerance_m,
                    CFG.radial_tolerance_fraction * circle["radius_m"],
                ),
            )
            on_circle = (
                np.abs(np.linalg.norm(plane[indices] - center, axis=1) - circle["radius_m"])
                <= tolerance
            )
            selected = indices[on_circle]
            layer["axis_offset_m"] = offset
            layer["proposal_axis_offset_m"] = float(np.linalg.norm(center))
            layer["predicted_center_normal_xy"] = predicted_center.tolist()
            if offset <= CFG.maximum_center_track_residual_m and len(selected) >= 12:
                world = (
                    origin + direction * np.median(axial[selected]) + u * center[0] + v * center[1]
                )
                actual_hag = float(np.median(hag[selected]))
                layer.update(status="PASS", center_xyz=world.tolist(), center_hag_m=actual_hag)
                centers.append(world)
                heights.append(actual_hag)
                support.append(original_index[selected])
                predicted_center = center
        layers.append(layer)
    required = [r for r in layers if r["hag_target_m"] <= 2.8]
    complete = all(r["status"] == "PASS" for r in required)
    refined = None
    direction_links = []
    if len(centers) >= 3:
        points, hs = np.asarray(centers), np.asarray(heights)
        line = _fit_hag_line(points, hs, np.arange(len(points)))
        if line is not None:
            anchor, unit, refined_slope = line
            delta = points - anchor
            residual = np.linalg.norm(delta - np.outer(delta @ unit, unit), axis=1)
            tilt = float(np.degrees(np.arccos(np.clip(unit[2], -1, 1))))
            for a, b in zip(points[:-1], points[1:], strict=True):
                link = a - b
                direction_links.append(
                    float(np.degrees(np.arctan2(np.linalg.norm(link[:2]), abs(link[2]))))
                )
            if (
                complete
                and np.max(residual) <= 0.03
                and tilt <= CFG.maximum_tilt_deg
                and max(direction_links, default=0) <= CFG.maximum_tilt_deg
                and min(heights) < 1.30 < max(heights)
            ):
                refined = {
                    "bh_reference_xyz": anchor.tolist(),
                    "direction_xyz": unit.tolist(),
                    "slope_xyz_per_hag": refined_slope.tolist(),
                    "tilt_deg": tilt,
                    "max_center_residual_m": float(np.max(residual)),
                    "anchor_status": "INTERPOLATED_FROM_TRACK_SPANNING_TRUE_BH",
                }
    good = [r for r in layers if r["status"] == "PASS"]
    consistency = 1 - float(np.mean([r["axis_offset_m"] for r in good])) / 0.03 if good else 0
    cylindrical = float(np.mean([r["circle"]["inlier_fraction"] for r in good])) if good else 0
    return {
        "status": "PASS" if refined is not None else "CROWN_TO_BH_TRACK_INCOMPLETE",
        "layers": layers,
        "coverage": len(good) / len(layers),
        "center_consistency": max(0, consistency),
        "cylindrical_support": cylindrical,
        "direction_links_deg": direction_links,
        "refined_axis": refined,
        "original_index": np.unique(np.concatenate(support))
        if support
        else np.empty(0, dtype=np.int64),
    }


def score_crown(track, axis, crown, connection, bridge):
    slope = np.asarray(axis["slope_xyz_per_hag"])
    upper = np.asarray(axis["bh_reference_xyz"]) + slope * (2.8 - 1.3)
    distance = float(crown["footprint_tree"].query(upper[:2])[0])
    center = float(np.linalg.norm(upper[:2] - crown["centroid_xy"]))
    apex_point = np.asarray(axis["bh_reference_xyz"]) + slope * (crown["maximum_hag_m"] - 1.3)
    apex = float(np.linalg.norm(apex_point[:2] - crown["apex_xy"]))
    spatial = float(np.exp(-(center + apex) / (2 * crown["footprint_radius_m"])))
    footprint = max(0.0, 1 - distance / 0.60)
    score = (
        0.40 * track["coverage"]
        + 0.25 * track["center_consistency"]
        + 0.20 * track["cylindrical_support"]
        + 0.10 * footprint
        + 0.05 * spatial
    )
    absolute = (
        track["status"] == "PASS"
        and connection["status"] == "PASS"
        and bridge["status"] == "PASS"
        and distance <= 0.60
    )
    return {
        "score": min(1.0, max(0.0, score)),
        "absolute_pass": bool(absolute),
        "crown_center_distance_m": center,
        "crown_apex_distance_m": apex,
        "footprint_distance_m": distance,
        "footprint_intersection": distance <= 0.60,
        "connection": connection,
        "raw_bridge": bridge,
        "components": {
            "coverage": track["coverage"],
            "axis_consistency": track["center_consistency"],
            "cylindrical": track["cylindrical_support"],
            "footprint": footprint,
            "centroid_apex": spatial,
        },
    }


def fit_identified_stem(xyz, hag, axis):
    """Final quality uses the entire ROI, never the identity inlier mask."""
    if axis.get("anchor_status") != "INTERPOLATED_FROM_TRACK_SPANNING_TRUE_BH":
        raise ValueError("IDENTITY_TRACK_MUST_ESTABLISH_TRUE_BH_BEFORE_FITTING")
    transformed = _axis_transform(xyz, axis)
    if transformed is None:
        raise ValueError("INVALID_AXIS_TRANSFORM")
    normal, normal_hag, transform = transformed
    result = fit_stem_dbh(normal, normal_hag, method="RECOVERED_STEM")
    selected = (normal_hag >= 0.85) & (normal_hag <= 1.75)
    attempts = [copy.deepcopy(result)]
    if result["dbh_cm"] is None:
        selected, continuity = select_continuous_stem_points(normal, normal_hag)
        result = fit_stem_dbh(normal[selected], normal_hag[selected], method="RECOVERED_STEM")
        attempts.append({**copy.deepcopy(result), "continuity": continuity})
    offsets = [
        float(np.hypot(s["center_x_m"], s["center_y_m"])) for s in result["slices"] if s["accepted"]
    ]
    if result["dbh_cm"] is not None and (not offsets or max(offsets) > 0.03):
        result.update(
            dbh_cm=None,
            dbh_method="UNAVAILABLE",
            dbh_quality="UNAVAILABLE",
            dbh_fit_status="FINAL_FIT_NOT_ON_IDENTIFIED_AXIS",
        )
    result.update(
        dbh_model_version=VERSION,
        transform=transform,
        fit_attempts=attempts,
        identified_axis=axis,
        selected_input_indices=np.flatnonzero(selected).tolist(),
    )
    return result
