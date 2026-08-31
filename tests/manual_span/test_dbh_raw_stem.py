"""Focused V2 axis normal DBH geometry tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
from algorithms.manual_span.dbh_raw_stem import (
    DBH_RAW_STEM_VERSION,
    fit_axis_normal_dbh,
)


def _tilted_stem(
    *,
    tilt_deg: float = 12.0,
    diameter_m: float = 0.40,
    center_xy: tuple[float, float] = (0.0, 0.0),
    axial_heights: np.ndarray | None = None,
    angles: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a cylinder with world Z as elevation and HAG kept separately."""

    direction = np.asarray(
        (np.sin(np.deg2rad(tilt_deg)), 0.0, np.cos(np.deg2rad(tilt_deg))), dtype=np.float64
    )
    normal_u = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    normal_v = np.cross(direction, normal_u)
    normal_v /= np.linalg.norm(normal_v)
    heights = axial_heights if axial_heights is not None else np.linspace(0.72, 1.88, 14)
    ring_angles = (
        angles
        if angles is not None
        else np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    )
    rows: list[np.ndarray] = []
    hags: list[float] = []
    bh = np.asarray((center_xy[0], center_xy[1], 1.30), dtype=np.float64)
    for height in heights:
        center = bh + direction * (float(height) - 1.30)
        for angle in ring_angles:
            point = center + (diameter_m / 2.0) * (
                normal_u * np.cos(angle) + normal_v * np.sin(angle)
            )
            rows.append(point)
            # This synthetic ground is flat, but the fitter must still use
            # this supplied HAG array rather than treating world Z as HAG.
            hags.append(float(point[2]))
    return np.asarray(rows, dtype=np.float64), np.asarray(hags, dtype=np.float64)


def test_tilted_cylinder_uses_axis_normal_and_true_bh_anchor() -> None:
    xyz, hag = _tilted_stem(tilt_deg=15.0)
    result = fit_axis_normal_dbh(xyz, hag, expected_xy=np.asarray((0.0, 0.0)))
    assert result["dbh_fit_status"] == "PASS"
    assert result["dbh_method"] == "RECOVERED_STEM"
    assert result["dbh_quality"] == "HIGH"
    assert result["dbh_cm"] == pytest.approx(40.0, abs=0.2)
    direction = np.asarray(result["unit_direction_xyz"])
    assert direction[2] > 0.0
    assert np.rad2deg(np.arctan2(np.linalg.norm(direction[:2]), direction[2])) == pytest.approx(
        15.0, abs=1.5
    )
    origin = np.asarray(result["breast_height_origin_xyz"])
    assert origin == pytest.approx((0.0, 0.0, 1.30), abs=0.02)
    assert result["axis"]["status"] == "PASS"
    assert result["axis"]["bracket_hag_m"] == pytest.approx(1.30)
    assert result["axis"]["transform"]["bh_hag_m"] == pytest.approx(1.30)
    assert "source HAG remains unchanged" in result["axis"]["transform"]["scalar_definition"]
    assert result["dbh_model_version"] == DBH_RAW_STEM_VERSION
    json.dumps(result)


def test_missing_upper_sections_cannot_make_axis_without_true_bh_span() -> None:
    xyz, hag = _tilted_stem(axial_heights=np.asarray((0.80, 0.95, 1.10, 1.23)))
    result = fit_axis_normal_dbh(xyz, hag)
    assert result["dbh_method"] == "UNAVAILABLE"
    assert result["dbh_fit_status"] in {"NO_VALID_AXIS_PROPOSAL", "AXIS_DOES_NOT_BRACKET_TRUE_BH"}
    assert result["dbh_cm"] is None
    assert all(
        not section.get("proposal_ready")
        for section in result["axis"]["preliminary_sections"][3:]
    )


def test_partial_arc_remains_rejected_after_axis_proposals() -> None:
    angles = np.linspace(0.0, np.deg2rad(170.0), 48, endpoint=True)
    xyz, hag = _tilted_stem(angles=angles)
    result = fit_axis_normal_dbh(xyz, hag)
    assert result["dbh_method"] == "UNAVAILABLE"
    assert result["dbh_cm"] is None
    assert result["dbh_fit_status"] == "NO_VALID_AXIS_PROPOSAL"
    ready_sections = sum(
        bool(section.get("proposal_ready"))
        for section in result["axis"]["preliminary_sections"]
    )
    assert ready_sections < 3


def test_partial_arc_at_200_degrees_can_pass_when_axis_is_bracketed() -> None:
    angles = np.linspace(0.0, np.deg2rad(200.0), 48, endpoint=True)
    xyz, hag = _tilted_stem(tilt_deg=0.0, angles=angles)
    result = fit_axis_normal_dbh(xyz, hag)
    assert result["dbh_fit_status"] == "PASS"
    assert result["dbh_cm"] == pytest.approx(40.0, abs=0.2)
    assert all(
        section["angular_span_deg"] >= 200.0 - 1.0e-6
        for section in result["axis"]["preliminary_sections"]
    )
    assert all(
        section["occupied_sector_count"] >= 6
        for section in result["axis"]["preliminary_sections"]
    )


def test_proposal_anchor_uses_circle_inliers_over_ground_height_clutter() -> None:
    xyz, hag = _tilted_stem(tilt_deg=0.0, angles=np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False))
    clutter = []
    clutter_hag = []
    for index in range(6):
        lower = 0.85 + index * 0.15
        for point_index in range(80):
            clutter.append((2.0 + 0.01 * point_index, 2.0, 20.0))
            clutter_hag.append(lower + 0.01)
    xyz = np.vstack((xyz, np.asarray(clutter, dtype=np.float64)))
    hag = np.concatenate((hag, np.asarray(clutter_hag, dtype=np.float64)))
    result = fit_axis_normal_dbh(xyz, hag)
    assert result["dbh_fit_status"] == "PASS"
    first_proposal = result["axis"]["preliminary_sections"][0]["proposals"][0]
    assert first_proposal["anchor_inlier_point_count"] >= 12
    assert first_proposal["anchor_inlier_point_count"] < first_proposal["input_point_count"]
    assert first_proposal["center_hag_m"] > 0.90
    assert first_proposal["center_z_m"] < 2.0
    assert first_proposal["center_z_m"] != pytest.approx(20.0)


def test_contaminated_band_can_use_independent_continuity_fallback() -> None:
    xyz, hag = _tilted_stem(tilt_deg=12.0, angles=np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False))
    rng = np.random.default_rng(71)
    clutter = []
    clutter_hag = []
    for height in np.linspace(0.925, 1.675, 6):
        for _ in range(120):
            clutter.append([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), height])
            clutter_hag.append(height)
    xyz = np.vstack((xyz, np.asarray(clutter)))
    hag = np.concatenate((hag, np.asarray(clutter_hag)))
    result = fit_axis_normal_dbh(xyz, hag)
    assert result["dbh_fit_status"] == "PASS"
    assert result["dbh_cm"] == pytest.approx(40.0, abs=0.5)
    assert result["axis"]["continuity_fallback_used"] is True
    assert result["selected_input_count"] < len(xyz)
    assert result["axis"]["continuity_diagnostics"]["selected_point_count"] == result[
        "selected_input_count"
    ]


def test_failed_continuity_fallback_is_retained_for_audit() -> None:
    angles = np.linspace(0.0, np.deg2rad(170.0), 48, endpoint=True)
    xyz, hag = _tilted_stem(tilt_deg=15.0, angles=angles)
    result = fit_axis_normal_dbh(xyz, hag)
    assert result["dbh_method"] == "UNAVAILABLE"
    fallback_fit = result["axis"].get("continuity_fallback_fit")
    assert isinstance(fallback_fit, dict)
    assert fallback_fit.get("dbh_fit_status") != "PASS"


def test_two_spatially_separate_stems_are_ambiguous() -> None:
    first_xyz, first_hag = _tilted_stem(center_xy=(0.0, 0.0), diameter_m=0.30)
    second_xyz, second_hag = _tilted_stem(center_xy=(0.40, 0.0), diameter_m=0.30)
    result = fit_axis_normal_dbh(
        np.vstack((first_xyz, second_xyz)), np.concatenate((first_hag, second_hag))
    )
    assert result["dbh_method"] == "UNAVAILABLE"
    assert result["dbh_fit_status"] == "AMBIGUOUS_COMPETING_AXES"
    assert result["dbh_cm"] is None
    assert len(result["axis"]["preliminary_sections"]) == 6
    assert result["axis"]["preliminary_fit_count"] <= 12
    assert any(item["reason"] == "AMBIGUOUS_COMPETING_AXIS" for item in result["axis"]["proposals"])


def test_determinism_and_translation_preserve_axis_geometry() -> None:
    xyz, hag = _tilted_stem(tilt_deg=10.0, center_xy=(0.25, -0.15))
    first = fit_axis_normal_dbh(xyz, hag, expected_xy=np.asarray((0.25, -0.15)))
    permutation = np.random.default_rng(91).permutation(len(xyz))
    reordered = fit_axis_normal_dbh(
        xyz[permutation], hag[permutation], expected_xy=np.asarray((0.25, -0.15))
    )
    for key in (
        "dbh_cm",
        "dbh_fit_status",
        "dbh_slice_count",
        "unit_direction_xyz",
        "breast_height_origin_xyz",
    ):
        assert reordered[key] == first[key]
    translated = fit_axis_normal_dbh(
        xyz + np.asarray((600_000.0, 4_000_000.0, 0.0)),
        hag,
        expected_xy=np.asarray((600_000.25, 3_999_999.85)),
    )
    assert translated["dbh_fit_status"] == "PASS"
    assert translated["dbh_cm"] == pytest.approx(first["dbh_cm"], abs=0.05)
    translated_direction = np.asarray(translated["unit_direction_xyz"])
    assert translated_direction == pytest.approx(first["unit_direction_xyz"], abs=1.0e-8)
    translated_origin = np.asarray(translated["breast_height_origin_xyz"])
    assert translated_origin == pytest.approx(
        np.asarray(first["breast_height_origin_xyz"]) + (600_000.0, 4_000_000.0, 0.0),
        abs=0.02,
    )


def test_invalid_axis_normal_inputs_fail_closed_with_exact_count() -> None:
    malformed = fit_axis_normal_dbh(np.zeros((4, 2)), np.zeros(4))
    assert malformed["dbh_method"] == "UNAVAILABLE"
    assert malformed["dbh_fit_status"] == "INVALID_XYZ_SHAPE"
    assert malformed["dbh_point_count"] == 4
    xyz, full_hag = _tilted_stem()
    mismatch = fit_axis_normal_dbh(xyz, full_hag[:-1])
    assert mismatch["dbh_method"] == "UNAVAILABLE"
    assert mismatch["dbh_fit_status"] == "INVALID_HAG_SHAPE"
    assert mismatch["dbh_point_count"] == len(xyz)
    nonfinite_xyz = xyz.copy()
    nonfinite_xyz[0, 0] = np.nan
    nonfinite = fit_axis_normal_dbh(nonfinite_xyz, full_hag)
    assert nonfinite["dbh_method"] == "UNAVAILABLE"
    assert nonfinite["dbh_fit_status"] == "NONFINITE_INPUT"
    assert nonfinite["dbh_point_count"] == len(xyz)
