from copy import deepcopy
from desktop import manual_measurements as geometry
from desktop import manual_dbh as dbh


def source():
    return [dict(tree_key='tree-1', tree_code='T-001', height=10., crown_width=3., diameter=20.,
                 horizontal_distance=3., vertical_distance=4., clearance_distance=5.)]


def test_correction_is_separate_from_original_and_clear_restores_original(tmp_path):
    rows = source()
    dbh.apply_measurements(rows, tmp_path, 'run-1')
    geometry.apply_measurements(rows, tmp_path, 'run-1')
    saved = deepcopy(rows)
    result = geometry.save_measurements(tmp_path, 'run-1', rows, 'tree-1',
                                        dict(height=11., vertical_distance=0), correct_existing=True)
    diameter = dbh.save_measurement(tmp_path, 'run-1', rows, 'tree-1', 22., correct_existing=True)
    assert rows == saved
    assert result['effective_fields'] == dict(height=11., vertical_distance=0)
    assert diameter['diameter_source'] == 'MANUAL_FIELD_CORRECTION'
    reopened = source()
    dbh.apply_measurements(reopened, tmp_path, 'run-1')
    geometry.apply_measurements(reopened, tmp_path, 'run-1')
    assert reopened[0]['height'] == 11
    assert reopened[0]['diameter'] == 22
    assert reopened[0]['point_cloud_measurements']['height'] == 10
    assert reopened[0]['point_cloud_diameter'] == 20
    assert reopened[0]['measurement_sources']['vertical_distance'] == 'MANUAL_FIELD_CORRECTION'
    geometry.save_measurements(tmp_path, 'run-1', reopened, 'tree-1',
                               dict(height=None, vertical_distance=None), correct_existing=True)
    restored = dbh.save_measurement(tmp_path, 'run-1', reopened, 'tree-1', 'No', correct_existing=True)
    assert restored['effective_diameter_cm'] == 20
    assert restored['diameter_source'] == 'ORIGINAL_LAS_1P30M'
    final = source()
    dbh.apply_measurements(final, tmp_path, 'run-1')
    geometry.apply_measurements(final, tmp_path, 'run-1')
    assert final[0]['height'] == 10 and final[0]['vertical_distance'] == 4 and final[0]['diameter'] == 20


def test_correction_does_not_leak_to_another_run_or_project(tmp_path):
    rows = source()
    dbh.apply_measurements(rows, tmp_path, 'first')
    geometry.apply_measurements(rows, tmp_path, 'first')
    dbh.save_measurement(tmp_path, 'first', rows, 'tree-1', 22, correct_existing=True)
    geometry.save_measurements(tmp_path, 'first', rows, 'tree-1', dict(height=11), correct_existing=True)
    for root, run in [(tmp_path, 'next'), (tmp_path/'other', 'first')]:
        fresh = source()
        dbh.apply_measurements(fresh, root, run)
        geometry.apply_measurements(fresh, root, run)
        assert fresh[0]['diameter'] == 20 and fresh[0]['height'] == 10
