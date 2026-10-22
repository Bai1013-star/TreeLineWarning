import numpy as np
import pytest

from algorithms.manual_span.dbh_breast_height import fit_raw_breast_height_section
from core.stage_b.dbh_evidence import raw_ground_reference
from core.stage_b.dbh_review import fit_review_stem
from core.stage_b.dbh_evidence import original_raw_query, sha256, ground_reference_at_stem


def cylinder():
    angle = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    heights = np.linspace(0.10, 3.0, 291)
    # Abrupt butt/taper differences deliberately expose pooling other heights.
    rows = []
    for h in heights:
        radius = 0.12 if 1.249 <= h <= 1.351 else 0.14
        rows.extend(np.column_stack((radius * np.cos(angle), radius * np.sin(angle),
                                     np.full(len(angle), 17.4 + h))))
    return np.asarray(rows)


def test_raw_ground_not_processed_tree_minimum_and_actual_target_slab():
    raw = cylinder()
    ground = np.column_stack((np.linspace(-0.4, 0.4, 90),
                             np.zeros(90), np.full(90, 17.4)))
    reference = raw_ground_reference(np.vstack((ground, raw)), 17.42)
    assert reference['ground_z_absolute_m'] == pytest.approx(17.4)
    assert reference['breast_height_z_absolute_m'] == pytest.approx(18.7)
    result = fit_raw_breast_height_section(raw, raw[:, 2] - 17.4, [0, 0], .14)
    assert result['dbh_cm'] == pytest.approx(24.0, abs=.01)
    selected = result['selected_input_indices']
    assert len(selected) >= 12
    assert np.max(np.abs(raw[selected, 2] - 18.7)) <= .050001
    # The separately processed tree has lost its bottom two metres. It must
    # never become the source of DBH or provide a replacement ground zero.
    processed = raw[raw[:, 2] > 19.4]
    assert fit_raw_breast_height_section(processed, processed[:, 2] - 17.4,
                                        [0, 0], .14)['dbh_cm'] is None


def test_no_original_ground_cannot_fall_back_to_processed_ground():
    with pytest.raises(ValueError, match='ORIGINAL_GROUND_EVIDENCE_MISSING'):
        raw_ground_reference(np.array([[0, 0, 21.0], [0, 0, 21.1]]), 17.4)


def test_no_target_points_cannot_interpolate_adjacent_sections():
    raw = cylinder()
    hag = raw[:, 2] - 17.4
    keep = np.abs(hag - 1.3) > .051
    result = fit_raw_breast_height_section(raw[keep], hag[keep], [0, 0], .14)
    assert result['dbh_cm'] is None


def test_human_review_also_measures_actual_slab_not_multisection_median():
    raw = cylinder()
    fit, selected = fit_review_stem(raw, raw[:, 2] - 17.4, seed_position=0)
    assert fit['dbh_cm'] == pytest.approx(24.0, abs=.01)
    assert np.max(np.abs(raw[selected, 2] - 18.7)) <= .050001


def test_unrelated_line_is_not_a_circular_trunk():
    x = np.linspace(-.12, .12, 50)
    raw = np.column_stack((x, np.zeros(50), np.full(50, 1.3)))
    assert fit_raw_breast_height_section(raw, raw[:, 2], [0, 0], .12)['dbh_cm'] is None


def test_original_las_is_reread_instead_of_using_processed_coordinates(tmp_path, monkeypatch):
    import json
    import laspy
    from core.stage_b import dbh_evidence
    las = laspy.create(point_format=3, file_version='1.2')
    las.header.scales = [.0001, .0001, .0001]
    las.x = [3.1234, 3.2234, 3.3234]
    las.y = [5.4321, 5.3321, 5.2321]
    las.z = [17.4, 18.7, 22.5]
    path = tmp_path / 'original.las'
    las.write(path)
    identity = dict(sha256=sha256(path), file_size_bytes=path.stat().st_size,
                    source_point_count=3)
    (tmp_path / 'stage-a-project.json').write_text(json.dumps(dict(project=dict(scans=[
        dict(point_cloud_source=str(path), source_identity=identity)]))), encoding='utf-8')
    # Spatial search returns deliberately wrong/processed coordinates. Only
    # genuine source indices may be used to retrieve measurement coordinates.
    monkeypatch.setattr(dbh_evidence, 'local_raw_query', lambda *a: (
        np.array([[0., 0., 1.3]]), np.array([1], dtype=np.int64), {}))
    raw, idx, proof = original_raw_query(tmp_path, tmp_path, identity, [0, 0], (0, 30))
    assert raw[0] == pytest.approx([3.2234, 5.3321, 18.7], abs=1e-10)
    assert idx.tolist() == [1]
    assert proof['coordinate_source'] == 'DIRECT_ORIGINAL_LAS_INTEGER_COORDINATES'


def test_dense_lower_trunk_is_not_accepted_as_ground():
    raw = cylinder()
    with pytest.raises(ValueError, match='GROUND_SURFACE_UNPROVEN'):
        raw_ground_reference(raw, 17.4)


def test_original_sloped_ground_is_evaluated_at_trunk_not_crown_seed():
    x, y = np.meshgrid(np.linspace(-.5, .5, 25), np.linspace(-.5, .5, 25))
    raw = np.column_stack((x.ravel(), y.ravel(), (17.4 + .10*x + .20*y).ravel()))
    reference = raw_ground_reference(raw, 17.4, [0, 0])
    at_stem = ground_reference_at_stem(reference, [.2, -.1])
    assert at_stem['ground_z_absolute_m'] == pytest.approx(17.4, abs=1e-8)
    assert at_stem['breast_height_z_absolute_m'] == pytest.approx(18.7, abs=1e-8)


def test_human_identity_still_uses_ground_at_axis_not_at_search_seed():
    from core.stage_b.dbh_review import _measure_at_confirmed_stem_ground
    angle,z = np.meshgrid(np.linspace(0,2*np.pi,96,endpoint=False),np.arange(1.20,1.41,.001))
    # Local stem ground rises 0.03 m from the search seed. The physical 1.30 m
    # cross-section is 24 cm, while the old search-height section is 28 cm.
    radius = np.where(np.abs(z-1.33)<=.05000001,.12,.14)
    xyz = np.column_stack((.3+(radius*np.cos(angle)).ravel(),
                           (radius*np.sin(angle)).ravel(),z.ravel()))
    reference = dict(raw_ground_plane_coefficients=[.1,0,10],
        raw_ground_plane_xy_origin_m=[0,0],processed_ground_prior_z_absolute_m=10,
        ground_reference_method='ORIGINAL_RAW_LOCAL_GROUND_MODE',raw_ground_support_point_count=50)
    first = dict(dbh_cm=28,breast_height_origin_xyz=[.3,0,1.3],unit_direction_xyz=[0,0,1])
    fit,selected,hag,ground = _measure_at_confirmed_stem_ground(xyz,first,reference,[0,0,10])
    assert ground['ground_z_absolute_m'] == pytest.approx(10.03)
    assert ground['breast_height_z_absolute_m'] == pytest.approx(11.33)
    assert fit['dbh_cm'] == pytest.approx(24,abs=.01)
    assert np.max(np.abs(hag[selected]-1.30)) <= .050001
