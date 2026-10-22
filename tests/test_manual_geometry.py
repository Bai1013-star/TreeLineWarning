from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.prediction_api import create_app
from desktop.manual_measurements import apply_measurements, measurement_or_none, save_measurements
from desktop.warning_bridge import register_warning_bridge


def rows():
    return [dict(tree_key='missing', tree_code='T-003', height=None, crown_width=None,
                 horizontal_distance=None, vertical_distance=0, clearance_distance=None),
            dict(tree_key='known', tree_code='T-001', height=10, crown_width=2,
                 horizontal_distance=3, vertical_distance=4, clearance_distance=5)]


@pytest.mark.parametrize('value', [None, '', 'No', '未获取', '未测得'])
def test_missing_is_not_zero(value):
    assert measurement_or_none('height', value) is None


@pytest.mark.parametrize('value', ['No', '未获取', 'invalid', float('nan')])
def test_missing_strings_are_not_prediction_ready(tmp_path, value):
    actual = rows()
    actual[0].update(height=value, clearance_distance=value)
    apply_measurements(actual, tmp_path, 'run-1')
    assert actual[0]['height'] is None
    assert actual[0]['clearance_distance'] is None
    assert not actual[0]['prediction_ready']


@pytest.mark.parametrize('field,value', [('height', 0), ('crown_width', 0), ('clearance_distance', -1),
    ('height', True), ('height', float('nan')), ('height', float('inf')), ('height', 'invalid'), ('unknown', 1)])
def test_invalid_geometry_rejected(field, value):
    with pytest.raises(ValueError):
        measurement_or_none(field, value)


def test_save_reopen_clear_original_and_version_isolation(tmp_path):
    source = rows()
    actual = deepcopy(source)
    apply_measurements(actual, tmp_path, 'run-1')
    assert not actual[0]['prediction_ready']
    assert actual[0]['point_cloud_measurements']['vertical_distance'] == 0
    save_measurements(tmp_path, 'run-1', actual, 'missing', dict(height=11.5, clearance_distance=4.2))
    reopened = rows()
    apply_measurements(reopened, tmp_path, 'run-1')
    assert reopened[0]['height'] == 11.5
    assert reopened[0]['clearance_distance'] == 4.2
    assert reopened[0]['point_cloud_measurements']['height'] is None
    assert reopened[0]['measurement_sources']['height'] == 'MANUAL_FIELD_MEASUREMENT'
    assert reopened[0]['prediction_ready']
    assert source == rows()
    other_run = rows()
    apply_measurements(other_run, tmp_path, 'run-2')
    assert other_run[0]['height'] is None
    other_project = rows()
    apply_measurements(other_project, tmp_path/'another', 'run-1')
    assert other_project[0]['height'] is None
    save_measurements(tmp_path, 'run-1', reopened, 'missing', dict(height='未获取'))
    cleared = rows()
    apply_measurements(cleared, tmp_path, 'run-1')
    assert cleared[0]['height'] is None
    assert cleared[0]['clearance_distance'] == 4.2


def test_originals_and_zero_distance_are_immutable(tmp_path):
    actual = rows()
    apply_measurements(actual, tmp_path, 'run-1')
    for tree, fields in [('missing', dict(vertical_distance=5)), ('known', dict(height=12)),
                         ('unknown', dict(height=12))]:
        with pytest.raises(ValueError):
            save_measurements(tmp_path, 'run-1', actual, tree, fields)
    with pytest.raises(ValueError):
        save_measurements(tmp_path, 'run-1', actual, 'missing', dict(height=12, crown_width=-1))
    assert not (tmp_path/'runtime/warning/manual_measurements.json').exists()


def test_bridge_manual_geometry_post_and_prediction(tmp_path, monkeypatch):
    import desktop.warning_bridge as bridge
    actual = rows()
    actual[0].update(tree_id=3, diameter=None, diameter_source='UNAVAILABLE', point_cloud_diameter=None)
    def context(_):
        result = deepcopy(actual)
        apply_measurements(result, tmp_path, 'run-1')
        return dict(status='READY', project_root=str(tmp_path), analysis_run_id='run-1', trees=result,
                    source='FINALIZED_POINT_CLOUD')
    monkeypatch.setattr(bridge, 'integrated_context', context)
    app = create_app(database_path=tmp_path/'api.db',
                     species_growth_model_path=Path(__file__).parents[1]/'models/species_growth.json')
    register_warning_bridge(app, processing_backend=object(), project_root=tmp_path)
    with TestClient(app) as client:
        for scope in (dict(expected_project_root=str(tmp_path/'another-project')),
                      dict(expected_analysis_run_id='old-run')):
            response = client.post('/integrated/measurements', json=dict(tree_key='missing',
                manual_fields=dict(height=12), **scope))
            assert response.status_code == 409
            assert not (tmp_path/'runtime/warning/manual_measurements.json').exists()
            response = client.post('/integrated/measurements', json=dict(tree_key='missing',
                manual_diameter_cm=25, **scope))
            assert response.status_code == 409
            assert not (tmp_path/'runtime/warning/manual_dbh.json').exists()
        response = client.post('/integrated/measurements', json=dict(tree_key='missing',
            manual_fields=dict(height=12, clearance_distance=5, crown_width=3),
            expected_project_root=str(tmp_path), expected_analysis_run_id='run-1'))
        assert response.status_code == 200, response.text
        body = dict(tree_key='missing', species='杨树', age=20, observation_date='2026-10-07',
                    weather=dict(wind_speed=3, rainfall=2, humidity=50, temperature=12, sunlight_hours=6),
                    prediction_mode='formula', short_horizons_days=[1], long_horizons_months=[1],
                    selected_horizon_months=1)
        prediction = client.post('/integrated/forecast', json=body)
        assert prediction.status_code == 200, prediction.text
        report = prediction.json()
        assert report['input_snapshot']['height'] == 12
        assert report['input_snapshot']['clearance_distance'] == 5
        assert report['input_snapshot']['diameter'] is None
        assert 'height' in report['integrated_source']['manual_fields']
        assert report['integrated_source']['tree_line_distance'] == 'POINT_CLOUD_WITH_EXPLICIT_MANUAL_SUPPLEMENTS'
