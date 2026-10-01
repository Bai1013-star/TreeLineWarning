import numpy as np
import pytest
from algorithms.manual_span.partial_arc_identity import (
    elevated_local_base_probe,
    recover_ground_anchored_bh,
    recover_upper_partial_arc_identity,
    recover_partial_arc_identity,
    recover_upper_supported_identity,
)


def one_sided_stem(*, clutter=False, missing_section=False):
    rng = np.random.default_rng(123)
    points, heights = [], []
    for index in range(13):
        h = 0.925 + index * 0.15
        if missing_section and index == 5:
            continue
        angle = np.linspace(-75, 75, 120) * np.pi / 180
        radius = 0.081 - 0.006 * (h - 1.30)
        xyz = np.column_stack((
            0.02 * (h - 1.30) + radius * np.cos(angle),
            radius * np.sin(angle),
            np.full(len(angle), h),
        ))
        xyz[:, :2] += rng.normal(0, 0.001, (len(angle), 2))
        points.append(xyz)
        heights.append(np.full(len(xyz), h))
        if clutter:
            noise = rng.uniform(-0.35, 0.35, (160, 3))
            noise[:, 2] = h
            points.append(noise)
            heights.append(np.full(len(noise), h))
    xyz = np.vstack(points)
    hag = np.concatenate(heights)
    return xyz, hag, np.arange(len(xyz), dtype=np.int64)


def test_coherent_one_sided_stem_requires_height_series():
    xyz, hag, source = one_sided_stem()
    candidate, proof = recover_partial_arc_identity(xyz, hag, source)
    assert proof["status"] == "PASS"
    assert candidate["fit"]["dbh_cm"] == pytest.approx(16.2, abs=0.7)
    assert candidate["fit"]["dbh_quality"] == "ESTIMATE"
    assert candidate["track"]["status"] == "PASS"
    assert len(candidate["track"]["original_index"]) > 500
    assert len(candidate["fit"]["diagnostics"]["sections"]) == 13


@pytest.mark.parametrize("clutter,missing", [(True, False), (False, True)])
def test_partial_arc_does_not_convert_clutter_or_one_height_gap_to_dbh(clutter, missing):
    xyz, hag, source = one_sided_stem(clutter=clutter, missing_section=missing)
    candidate, proof = recover_partial_arc_identity(xyz, hag, source)
    assert candidate is None
    assert proof["status"] != "PASS"


def test_upper_full_circles_estimate_only_short_distance_to_breast_height():
    angle = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    xyz = np.vstack([
        np.column_stack((
            (0.123 + 0.004 * (h - 1.30)) * np.cos(angle),
            (0.123 + 0.004 * (h - 1.30)) * np.sin(angle),
            np.full(len(angle), h),
        ))
        for h in 1.375 + 0.15 * np.arange(9)
    ])
    hag = xyz[:, 2].copy()
    candidate, proof = recover_upper_supported_identity(
        xyz, hag, np.arange(len(xyz), dtype=np.int64)
    )
    assert proof["status"] == "PASS"
    assert candidate["fit"]["dbh_cm"] == pytest.approx(24.6, abs=0.3)
    assert candidate["fit"]["dbh_quality"] == "ESTIMATE"
    assert candidate["fit"]["dbh_slice_count"] == 3

    keep = hag < 2.35
    rejected, reason = recover_upper_supported_identity(
        xyz[keep], hag[keep], np.flatnonzero(keep)
    )
    assert rejected is None
    assert reason["status"] == "UPPER_CIRCULAR_SECTION_SPARSE"


def _transformed_stem(*, density, yaw_deg, shift, tilt_x, gap=None,
                      first_h=1.225, section_count=13):
    """Independent synthetic stem with a true root and a sloping centreline."""
    rng = np.random.default_rng(467)
    turn = np.deg2rad(yaw_deg)
    rotation = np.array([[np.cos(turn), -np.sin(turn)],
                         [np.sin(turn), np.cos(turn)]])
    points = []
    for section in range(section_count):
        h = first_h + section * 0.15
        if section == gap:
            continue
        angles = np.deg2rad(np.linspace(-77, 78, density))
        radius = 0.112 - 0.002 * (h - 1.3)
        plane = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
        plane += np.array([tilt_x * (h - 1.3), -0.007 * (h - 1.3)])
        plane = plane @ rotation.T + shift[:2]
        plane += rng.normal(0, 0.0007, size=plane.shape)
        points.append(np.column_stack((plane, np.full(density, h + shift[2]))))
    xyz = np.vstack(points)
    return xyz, xyz[:, 2] - shift[2], np.arange(len(xyz), dtype=np.int64)


@pytest.mark.parametrize("density,yaw,shift,tilt", [
    (65, 0, np.array([0., 0., 0.]), 0.0),
    (180, 63, np.array([183.7, -405.2, 21.4]), 0.018),
])
def test_upper_bh_series_respects_translation_yaw_tilt_and_density(
    density, yaw, shift, tilt,
):
    xyz, hag, source = _transformed_stem(
        density=density, yaw_deg=yaw, shift=shift, tilt_x=tilt,
    )
    result, proof = recover_upper_partial_arc_identity(xyz, hag, source)
    assert proof["status"] == "PASS"
    assert result["fit"]["dbh_cm"] == pytest.approx(22.4, abs=0.5)
    assert result["fit"]["diagnostics"]["sections"][0]["angular_span_deg"] >= 120


def test_upper_bh_series_rejects_missing_breast_height_section():
    xyz, hag, source = _transformed_stem(
        density=150, yaw_deg=22, shift=np.array([4., -9., 2.]),
        tilt_x=0.01, gap=0,
    )
    result, proof = recover_upper_partial_arc_identity(xyz, hag, source)
    assert result is None
    assert proof["status"] == "UPPER_PARTIAL_ARC_SECTION_SPARSE"


def test_root_anchored_windows_agree_across_independent_height_sections():
    xyz, local_hag, source = _transformed_stem(
        density=180, yaw_deg=41, shift=np.array([-34., 18., 4.]),
        tilt_x=0.011, first_h=0.775, section_count=17,
    )
    # The local root coordinate comes from an observed ground point; adding
    # a global coarse-HAG bias must not change a ground-anchored measurement.
    result, proof = recover_ground_anchored_bh(xyz, local_hag, source)
    assert proof["status"] == "PASS"
    assert len(proof["dbh_values_cm"]) >= 2
    assert result["fit"]["dbh_cm"] == pytest.approx(22.4, abs=0.5)


@pytest.mark.parametrize("base_hag", [0.4, 0.7, 1.4])
def test_elevated_root_probe_discovers_height_without_project_specific_window(base_hag):
    radius = np.linspace(0.25, 0.58, 9)
    angle = np.linspace(-np.pi, np.pi, 24, endpoint=False)
    rr, aa = np.meshgrid(radius, angle)
    surface = np.column_stack((rr.ravel() * np.cos(aa.ravel()),
                               rr.ravel() * np.sin(aa.ravel()),
                               np.full(rr.size, base_hag + 0.02)))
    stem_angle = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    stem = np.vstack([
        np.column_stack((0.11 * np.cos(stem_angle), 0.11 * np.sin(stem_angle),
                         np.full(72, base_hag + height)))
        for height in (0.15, 0.23, 0.31)
    ])
    xyz = np.vstack((surface, stem))
    probe = elevated_local_base_probe(
        xyz, xyz[:, 2], {"bh_reference_xyz": [0.0, 0.0, 1.3], "radius_m": 0.11}
    )
    assert probe["status"] == "LOCAL_BASE_HEIGHT_AMBIGUOUS"
    assert probe["sheet_hag_range_m"][0] <= base_hag + 0.02 < probe["sheet_hag_range_m"][1]


def test_missing_root_and_ground_evidence_remains_unverified():
    xyz, hag, _ = _transformed_stem(
        density=100, yaw_deg=15, shift=np.zeros(3), tilt_x=0.005,
    )
    probe = elevated_local_base_probe(
        xyz, hag, {"bh_reference_xyz": [0.0, 0.0, 1.3], "radius_m": 0.112}
    )
    assert probe["status"] == "LOCAL_BASE_HEIGHT_UNVERIFIED"


@pytest.mark.parametrize("stem_radius,base_hag", [(0.30, 0.4), (0.45, 1.4)])
def test_large_stem_terrace_uses_bark_radius_and_wide_ground_ring(stem_radius, base_hag):
    angle = np.linspace(-np.pi, np.pi, 48, endpoint=False)
    radii = np.linspace(stem_radius + .10, stem_radius + .39, 8)
    rr, aa = np.meshgrid(radii, angle)
    sheet = np.column_stack((rr.ravel() * np.cos(aa.ravel()),
                             rr.ravel() * np.sin(aa.ravel()),
                             np.full(rr.size, base_hag + .02)))
    stem = np.vstack([
        np.column_stack((stem_radius * np.cos(angle), stem_radius * np.sin(angle),
                         np.full(len(angle), base_hag + h)))
        for h in (.15, .23, .31)
    ])
    xyz = np.vstack((sheet, stem))
    proof = elevated_local_base_probe(
        xyz, xyz[:, 2], {"bh_reference_xyz": [0, 0, 1.3],
                         "direction_xyz": [0, 0, 1], "radius_m": stem_radius}
    )
    assert proof["status"] == "LOCAL_BASE_HEIGHT_AMBIGUOUS"


@pytest.mark.parametrize("evidence", ["ground_only", "low_stem_only"])
def test_one_root_evidence_class_alone_does_not_certify_ground(evidence):
    angle = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    if evidence == "ground_only":
        parts = [np.column_stack((.35 * np.cos(angle), .35 * np.sin(angle),
                                  np.zeros(len(angle))))]
    else:
        parts = [np.column_stack((.12 * np.cos(angle), .12 * np.sin(angle),
                                  np.full(len(angle), .15)))]
    xyz = np.vstack(parts)
    proof = elevated_local_base_probe(
        xyz, xyz[:, 2], {"bh_reference_xyz": [0, 0, 1.3],
                         "direction_xyz": [0, 0, 1], "radius_m": .12}
    )
    assert proof["status"] == "LOCAL_BASE_HEIGHT_UNVERIFIED"


def test_ground_and_low_bark_vetoes_non_root_horizontal_clutter_on_tilted_stem():
    angle = np.linspace(-np.pi, np.pi, 60, endpoint=False)
    def ring(radius, height):
        cx = .16 * (height - 1.3)
        return np.column_stack((cx + radius * np.cos(angle),
                                radius * np.sin(angle), np.full(len(angle), height)))
    ground = np.vstack([ring(radius, 0) for radius in (.25, .35, .45, .55)])
    stem = np.vstack([ring(.12, h) for h in (.15, .25, .35, 1.2, 1.3)])
    shrub = np.vstack([ring(radius, .72) for radius in (.25, .35, .45, .55)])
    xyz = np.vstack((ground, stem, shrub))
    unit = np.array([.16, 0, 1.0]); unit /= np.linalg.norm(unit)
    proof = elevated_local_base_probe(
        xyz, xyz[:, 2], {"bh_reference_xyz": [0, 0, 1.3],
                         "direction_xyz": unit.tolist(), "radius_m": .12}
    )
    assert proof["status"] == "NO_ELEVATED_BASE_EVIDENCE"
