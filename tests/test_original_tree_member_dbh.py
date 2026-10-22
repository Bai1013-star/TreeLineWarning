import numpy as np

from algorithms.manual_span.dbh_breast_height import (
    fit_original_tree_member_breast_height,
)


def _cylinder(radius, ground, section_heights, points_per_ring=48):
    angle = np.linspace(0.0, 2.0 * np.pi, points_per_ring, endpoint=False)
    rings = []
    for height in section_heights:
        rings.append(
            np.column_stack(
                (
                    10.0 + radius * np.cos(angle),
                    20.0 + radius * np.sin(angle),
                    np.full(points_per_ring, ground + height),
                )
            )
        )
    return np.vstack(rings)


def test_original_tree_dbh_uses_tree_lowest_plus_1p30():
    ground = 3.75
    points = _cylinder(0.16, ground, [0.0, 0.3, 0.7, 1.27, 1.30, 1.33, 2.0])
    # Wide crown/noise in the same height range must not replace the compact
    # trunk component selected from the original tree membership.
    crown = np.array(
        [[12.0 + i * 0.03, 22.0, ground + 1.30] for i in range(12)],
        dtype=np.float64,
    )
    points = np.vstack((points, crown))
    original_index = np.arange(4000, 4000 + len(points), dtype=np.int64)

    measured = fit_original_tree_member_breast_height(points, original_index)

    assert abs(measured["dbh_cm"] - 32.0) < 0.5
    assert measured["ground_local_z_m"] == ground
    assert measured["measurement_local_z_m"] == ground + 1.30
    assert measured["diameter_derived_from"] == "DIRECT_ORIGINAL_LAS_TREE_MEMBER_SECTION_AT_1P30"
    assert set(measured["selected_original_indices"]).issubset(set(original_index.tolist()))


def test_sparse_original_tree_uses_adaptive_raw_section_not_placeholder():
    ground = -1.2
    # No point lies in the strict +/- 5 cm band.  The adaptive original-LAS
    # section must still measure the trunk instead of publishing None/default.
    points = _cylinder(0.11, ground, [0.0, 0.4, 0.8, 1.18, 1.42, 2.0], 36)
    original_index = np.arange(len(points), dtype=np.int64) * 7 + 11

    measured = fit_original_tree_member_breast_height(points, original_index)

    assert measured["dbh_cm"] is not None
    assert abs(measured["dbh_cm"] - 22.0) < 0.5
    assert measured["vertical_half_band_m"] >= 0.15
    assert measured["target_hag_m"] == 1.30
    assert measured["dbh_fit_status"].startswith("PASS_ORIGINAL_TREE_LOWEST_PLUS_1P30")
