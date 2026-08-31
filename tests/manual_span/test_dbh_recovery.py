"""Focused tests for the independent V1 DBH geometry and allometry helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest
from algorithms.manual_span.dbh_recovery import (
    ALLOMETRIC_ESTIMATE,
    DIRECT_STEM,
    HIGH,
    PASS,
    RECOVERED_STEM,
    UNAVAILABLE,
    estimate_local_dbh,
    fit_stem_dbh,
    select_continuous_stem_points,
)


def _stem(
    *,
    diameter_m: float = 0.40,
    arc_deg: float = 360.0,
    center_offsets: dict[int, tuple[float, float]] | None = None,
    base_center: tuple[float, float] = (0.0, 0.0),
    points_per_slice: int = 48,
    noise_m: float = 0.0,
    outliers_per_slice: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Make deterministic horizontal rings with optional controlled defects."""

    rng = np.random.default_rng(52)
    xyz: list[list[float]] = []
    hag: list[float] = []
    radius = diameter_m / 2.0
    offsets = center_offsets or {}
    for index in range(6):
        height = 0.85 + index * 0.15 + 0.075
        start = 0.0
        if arc_deg >= 359.999:
            angles = np.linspace(start, 2.0 * np.pi, points_per_slice, endpoint=False)
        else:
            angles = np.linspace(start, np.deg2rad(arc_deg), points_per_slice, endpoint=True)
        offset_x, offset_y = offsets.get(index, (0.0, 0.0))
        cx, cy = base_center[0] + offset_x, base_center[1] + offset_y
        for angle in angles:
            x = cx + radius * np.cos(angle) + rng.normal(0.0, noise_m)
            y = cy + radius * np.sin(angle) + rng.normal(0.0, noise_m)
            xyz.append([x, y, height])
            hag.append(height)
        for _ in range(outliers_per_slice):
            xyz.append([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), height])
            hag.append(height)
    return np.asarray(xyz, dtype=np.float64), np.asarray(hag, dtype=np.float64)


def test_known_circle_is_high_quality_and_counts_original_rows() -> None:
    xyz, hag = _stem(diameter_m=0.36)
    # Points outside the fitted interval still count as original evidence.
    xyz = np.vstack((xyz, [[10.0, 10.0, 20.0], [11.0, 11.0, 21.0]]))
    hag = np.concatenate((hag, [20.0, 21.0]))
    result = fit_stem_dbh(xyz, hag, method=RECOVERED_STEM)
    assert result["dbh_fit_status"] == PASS
    assert result["dbh_method"] == RECOVERED_STEM
    assert result["dbh_quality"] == HIGH
    assert result["dbh_cm"] == pytest.approx(36.0, abs=0.1)
    assert result["dbh_point_count"] == len(xyz)
    assert result["dbh_slice_count"] == 6
    assert result["dbh_fit_rmse"] is not None
    assert result["dbh_uncertainty_cm"] is not None
    assert all(section["status"] == PASS for section in result["slices"])
    json.dumps(result)


def test_continuity_prefilter_keeps_vertical_stem_and_drops_spatial_clutter() -> None:
    xyz, hag = _stem(diameter_m=0.36, points_per_slice=48)
    clutter = np.asarray(
        [[3.0 + 0.01 * i, 3.0, 0.925] for i in range(40)], dtype=np.float64
    )
    xyz = np.vstack((xyz, clutter))
    hag = np.concatenate((hag, clutter[:, 2]))
    selected, diagnostics = select_continuous_stem_points(xyz, hag)
    assert selected.dtype == np.bool_
    assert selected.shape == (len(xyz),)
    assert diagnostics["status"] == PASS
    assert diagnostics["supported_column_count"] > 0
    assert diagnostics["selected_point_count"] == int(np.count_nonzero(selected))
    assert not np.any(selected[-len(clutter) :])
    assert np.all(selected[: len(xyz) - len(clutter)])
    assert diagnostics["voxel_size_m"] == pytest.approx(0.02)
    json.dumps(diagnostics)


def test_continuity_voxel_origin_is_translation_stable() -> None:
    local_xyz, local_hag = _stem(points_per_slice=36)
    translated_xyz, translated_hag = _stem(
        points_per_slice=36, base_center=(600_000.0, 4_000_000.0)
    )
    local_mask, local_diagnostics = select_continuous_stem_points(local_xyz, local_hag)
    translated_mask, translated_diagnostics = select_continuous_stem_points(
        translated_xyz, translated_hag
    )
    assert np.array_equal(local_mask, translated_mask)
    assert translated_diagnostics["supported_column_count"] == local_diagnostics[
        "supported_column_count"
    ]


def test_large_projected_coordinates_are_translation_stable() -> None:
    xyz, hag = _stem(diameter_m=0.42, base_center=(600_000.0, 4_000_000.0))
    result = fit_stem_dbh(xyz, hag, method=DIRECT_STEM)
    assert result["dbh_fit_status"] == PASS
    assert result["dbh_cm"] == pytest.approx(42.0, abs=0.1)
    assert result["slices"][0]["center_x_m"] == pytest.approx(600_000.0, abs=0.01)
    assert result["slices"][0]["center_y_m"] == pytest.approx(4_000_000.0, abs=0.01)


def test_outliers_are_rejected_by_bounded_robust_fit() -> None:
    xyz, hag = _stem(diameter_m=0.30, outliers_per_slice=18, noise_m=0.001)
    result = fit_stem_dbh(xyz, hag)
    assert result["dbh_fit_status"] == PASS
    assert result["dbh_cm"] == pytest.approx(30.0, abs=0.4)
    assert all(section["inlier_fraction"] >= 0.60 for section in result["slices"])


def test_nonstem_cloud_fails_without_a_plausible_circle() -> None:
    rng = np.random.default_rng(23)
    hag = np.repeat(np.arange(6) * 0.15 + 0.925, 80)
    xyz = np.column_stack((rng.normal(0.0, 0.32, len(hag)), rng.normal(0.0, 0.32, len(hag)), hag))
    result = fit_stem_dbh(xyz, hag)
    assert result["dbh_method"] == UNAVAILABLE
    assert result["dbh_quality"] == UNAVAILABLE
    assert result["dbh_cm"] is None
    assert result["dbh_point_count"] == len(xyz)


@pytest.mark.parametrize("arc_deg", [90.0, 170.0, 179.0])
def test_partial_arc_below_180_degrees_is_unavailable(arc_deg: float) -> None:
    xyz, hag = _stem(arc_deg=arc_deg, points_per_slice=36)
    result = fit_stem_dbh(xyz, hag)
    assert result["dbh_method"] == UNAVAILABLE
    assert result["dbh_fit_status"] == "INSUFFICIENT_ACCEPTED_SLICES"
    assert all(section["status"] == "ANGULAR_SPAN_LT_180" for section in result["slices"])


def test_partial_arc_at_200_degrees_passes_six_sector_guard() -> None:
    xyz, hag = _stem(arc_deg=200.0, points_per_slice=36)
    result = fit_stem_dbh(xyz, hag)
    assert result["dbh_fit_status"] == PASS
    assert result["dbh_cm"] == pytest.approx(40.0, abs=0.1)
    assert all(section["angular_span_deg"] >= 200.0 - 1.0e-6 for section in result["slices"])
    assert all(section["occupied_sector_count"] >= 6 for section in result["slices"])


def test_continuity_prefilter_does_not_replace_partial_arc_evidence() -> None:
    xyz, hag = _stem(arc_deg=170.0, points_per_slice=36)
    selected, diagnostics = select_continuous_stem_points(xyz, hag)
    assert diagnostics["status"] == PASS
    # Every arc point has a vertical partner, so continuity keeps it.  The
    # angular guard remains the separate circle-fit responsibility.
    assert np.all(selected)
    result = fit_stem_dbh(xyz[selected], hag[selected])
    assert result["dbh_fit_status"] == "INSUFFICIENT_ACCEPTED_SLICES"
    assert result["dbh_cm"] is None


def test_inconsistent_adjacent_centers_fail_center_track_guard() -> None:
    xyz, hag = _stem(center_offsets={3: (0.12, 0.0)})
    result = fit_stem_dbh(xyz, hag)
    assert result["dbh_method"] == UNAVAILABLE
    assert result["dbh_fit_status"] == "CENTER_TRACK_RESIDUAL_HIGH"
    assert result["diagnostics"]["center_track_max_residual_m"] > 0.03


@pytest.mark.parametrize("keep", [slice(0, 3), slice(3, 6)])
def test_accepted_sections_must_cross_130_m_breast_height(keep: slice) -> None:
    xyz, hag = _stem()
    section_index = np.floor((hag - 0.85) / 0.15).astype(int)
    selected = np.isin(section_index, np.arange(6)[keep])
    result = fit_stem_dbh(xyz[selected], hag[selected])
    assert result["dbh_method"] == UNAVAILABLE
    assert result["dbh_fit_status"] == "ACCEPTED_SLICES_DO_NOT_SPAN_BREAST_HEIGHT"


def test_fit_is_deterministic_under_row_permutation_and_is_bounded() -> None:
    xyz, hag = _stem(points_per_slice=5000)
    first = fit_stem_dbh(xyz, hag)
    permutation = np.random.default_rng(12).permutation(len(xyz))
    second = fit_stem_dbh(xyz[permutation], hag[permutation])
    assert first == second
    assert all(section["fit_point_count"] <= 4096 for section in first["slices"])


def test_invalid_and_nonfinite_inputs_fail_closed() -> None:
    malformed = fit_stem_dbh(np.zeros((4, 2)), np.zeros(4))
    assert malformed["dbh_method"] == UNAVAILABLE
    assert malformed["dbh_fit_status"] == "INVALID_XYZ_SHAPE"
    assert malformed["dbh_point_count"] == 4
    xyz, hag = _stem()
    xyz[0, 0] = np.nan
    nonfinite = fit_stem_dbh(xyz, hag)
    assert nonfinite["dbh_method"] == UNAVAILABLE
    assert nonfinite["dbh_fit_status"] == "NONFINITE_INPUT"
    assert nonfinite["dbh_point_count"] == len(xyz)


def test_continuity_rejects_dense_single_level_sheet_and_invalid_shape() -> None:
    rng = np.random.default_rng(91)
    hag = np.full(500, 0.925)
    xyz = np.column_stack((rng.normal(0.0, 0.5, 500), rng.normal(0.0, 0.5, 500), hag))
    selected, diagnostics = select_continuous_stem_points(xyz, hag)
    assert not np.any(selected)
    assert diagnostics["status"] == "NO_CONTINUOUS_STEM_COLUMNS"
    assert diagnostics["section_point_counts"][0] == 500
    assert diagnostics["section_point_counts"][1:] == [0, 0, 0, 0, 0]
    malformed, invalid = select_continuous_stem_points(np.zeros((4, 2)), np.zeros(4))
    assert malformed.dtype == np.bool_
    assert malformed.shape == (4,)
    assert not np.any(malformed)
    assert invalid["status"] == "INVALID_XYZ_SHAPE"


def _training_rows(*, noisy: bool = False) -> list[dict[str, object]]:
    rows = []
    for index in range(12):
        height = 5.0 + index * 0.4
        dbh = 2.5 * height**1.1
        if noisy:
            dbh *= 1.0 + (0.15 if index % 2 else -0.12)
        rows.append(
            {
                "persistent_tree_id": f"tree-{index:02d}",
                "height_m": height,
                "dbh_cm": dbh,
                "dbh_method": DIRECT_STEM if index % 2 else RECOVERED_STEM,
                "dbh_quality": HIGH,
            }
        )
    return rows


def test_allometry_requires_validated_training_and_labels_estimates() -> None:
    training = _training_rows(noisy=True)
    # This duplicate identity must not increase the independent training count.
    training.append(dict(training[0]))
    targets = [
        {"persistent_tree_id": "inside", "height_m": 6.2},
        {"persistent_tree_id": "outside", "height_m": 100.0},
    ]
    result = estimate_local_dbh(training, targets)
    inside = result["inside"]
    assert inside["dbh_method"] == ALLOMETRIC_ESTIMATE
    assert inside["dbh_quality"] == "ESTIMATE"
    assert inside["dbh_fit_status"] == PASS
    assert inside["dbh_cm"] is not None
    assert inside["dbh_model_version"] == "DBH_RECOVERY_V1"
    assert result["outside"]["dbh_method"] == UNAVAILABLE
    assert result["outside"]["dbh_fit_status"] == "TARGET_HEIGHT_OUT_OF_TRAINING_RANGE"
    model = inside["model_validation"]
    assert model["status"] == PASS
    assert model["training_count"] == 12
    assert model["loo_relative_rmse"] <= 0.25
    assert model["loo_max_relative_error"] <= 0.50
    json.dumps(result)


def test_allometry_insufficient_and_loo_rejected_are_explicit() -> None:
    too_few = estimate_local_dbh(
        _training_rows()[:9], [{"persistent_tree_id": "x", "height_m": 6.0}]
    )
    assert too_few["x"]["dbh_method"] == UNAVAILABLE
    assert too_few["x"]["dbh_fit_status"] == "INSUFFICIENT_TRAINING_DATA"
    assert too_few["x"]["dbh_model_version"] == "DBH_RECOVERY_V1"
    bad = _training_rows()
    bad[-1]["dbh_cm"] = 100.0
    rejected = estimate_local_dbh(bad, [{"persistent_tree_id": "x", "height_m": 6.0}])
    assert rejected["x"]["dbh_method"] == UNAVAILABLE
    assert rejected["x"]["dbh_fit_status"] in {
        "LOO_RELATIVE_RMSE_HIGH",
        "LOO_MAX_RELATIVE_ERROR_HIGH",
        "LOO_VALIDATION_FAILED",
    }


def test_allometry_ignores_estimates_as_training_rows() -> None:
    rows = _training_rows()[:9]
    rows.append(
        {
            "persistent_tree_id": "estimate",
            "height_m": 7.0,
            "dbh_cm": 20.0,
            "dbh_method": ALLOMETRIC_ESTIMATE,
            "dbh_quality": "ESTIMATE",
        }
    )
    result = estimate_local_dbh(rows, [{"persistent_tree_id": "target", "height_m": 6.0}])
    assert result["target"]["dbh_method"] == UNAVAILABLE
    assert result["target"]["dbh_fit_status"] == "INSUFFICIENT_TRAINING_DATA"
