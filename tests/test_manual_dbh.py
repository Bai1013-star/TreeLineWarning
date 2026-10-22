from copy import deepcopy
from types import SimpleNamespace
import base64
import sys

import pytest

from desktop.manual_dbh import diameter_or_none, apply_measurements, save_measurement
from desktop.app import DesktopApi
from algorithms.prediction.forecast import ForecastService
from algorithms.prediction.standards import PredictionStandards
from api.prediction_api import create_app, _database_for
from fastapi.testclient import TestClient
from starlette.requests import Request


def test_project_history_isolation_and_request_binding(tmp_path):
    first, second = tmp_path/'first', tmp_path/'second'
    first.mkdir(); second.mkdir()
    workflow = SimpleNamespace(project_root=first)
    app = create_app(database_path=first/'runtime/warning/prediction_history.db')
    app.state.processing_backend = SimpleNamespace(local_app=SimpleNamespace(workflow=workflow))
    app.state.integrated_project_root = first
    with TestClient(app):
        request = Request(dict(type='http', app=app))
        database1 = _database_for(request)
        database1.update_standards({'danger_clearance_distance': 0.31})
        workflow.project_root = second
        assert _database_for(request) is database1
        database2 = _database_for(Request(dict(type='http', app=app)))
        assert database2 is not database1
        assert database2.db_path == second/'runtime/warning/prediction_history.db'
        assert database2.get_standards()['danger_clearance_distance'] != 0.31
        database2.update_standards({'danger_clearance_distance': 0.72})
        workflow.project_root = first
        assert _database_for(Request(dict(type='http', app=app))) is database1
        assert database1.get_standards()['danger_clearance_distance'] == 0.31
        assert database2.get_standards()['danger_clearance_distance'] == 0.72


def test_forecast_api_no_and_manual_source(tmp_path):
    from pathlib import Path
    root = Path(__file__).parents[1]
    app = create_app(database_path=tmp_path/'api.db',
                     species_growth_model_path=root/'models/species_growth.json')
    with TestClient(app) as client:
        data = dict(tree_id=1, species='杨树', height=10, crown_width=3,
                    voltage_level_kv=110, clearance_distance=5,
                    observation_date='2026-10-07', diameter='No')
        response = client.post('/predict/forecast', json=dict(tree_data=data,
            prediction_mode='formula', short_horizons_days=[1], long_horizons_months=[1]))
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['input_snapshot']['diameter'] is None
        assert result['wind_contact_estimate']['status'] == 'INSUFFICIENT_EVIDENCE'
        history = client.get('/history').json()['history']
        assert history[0]['diameter'] is None


def rows():
    return [dict(tree_key='missing', tree_code='T-003', diameter=None),
            dict(tree_key='measured', tree_code='T-001', diameter=19.86)]


@pytest.mark.parametrize('value', [None, '', 'No', ' no '])
def test_no_is_missing(value):
    assert diameter_or_none(value) is None


@pytest.mark.parametrize('value', [0, -1, True, float('nan'), float('inf'), 501, 'abc'])
def test_invalid_manual_rejected(value):
    with pytest.raises(ValueError):
        diameter_or_none(value)


def test_manual_persist_reopen_clear_and_run_isolation(tmp_path):
    original = rows()
    save_measurement(tmp_path, 'run-1', original, 'missing', 26.4)
    reopened = deepcopy(original)
    apply_measurements(reopened, tmp_path, 'run-1')
    assert reopened[0]['diameter'] == 26.4
    assert reopened[0]['point_cloud_diameter'] is None
    assert reopened[0]['diameter_source'] == 'MANUAL_FIELD_MEASUREMENT'
    assert original == rows()
    assert reopened[1]['diameter'] == 19.86
    unrelated = rows()
    apply_measurements(unrelated, tmp_path, 'run-2')
    assert unrelated[0]['diameter'] is None
    save_measurement(tmp_path, 'run-1', reopened, 'missing', 'No')
    cleared = rows()
    apply_measurements(cleared, tmp_path, 'run-1')
    assert cleared[0]['diameter'] is None


def test_manual_cannot_replace_auto_or_other_tree(tmp_path):
    for key in ('measured', 'wrong'):
        with pytest.raises(ValueError):
            save_measurement(tmp_path, 'run-1', rows(), key, 26.4)
    with pytest.raises(ValueError):
        save_measurement(tmp_path, '', rows(), 'missing', 26.4)


@pytest.mark.parametrize('missing', ['No', None])
def test_no_never_published_as_model_dbh(missing):
    from pathlib import Path
    service = ForecastService(None, growth_model_path=Path(__file__).parents[1]/'models/species_growth.json')
    data = dict(species='杨树', height=10, clearance_distance=5, voltage_level_kv=110,
                observation_date='2026-10-07', diameter=missing)
    result = service.forecast(data, PredictionStandards().to_dict(), prediction_mode='formula',
        short_horizons_days=[1], long_horizons_months=[1])
    assert result['input_snapshot']['diameter'] is None
    assert result['dbh_input']['value_cm'] is None
    assert result['selected_target_prediction']['predicted_diameter'] is None
    assert result['short_term_forecast'][0]['predicted_diameter'] is None
    assert result['growth_trend']['diameter_change'] is None
    assert data['diameter'] == missing


def test_manual_forecast_source_preserved():
    from pathlib import Path
    service = ForecastService(None, growth_model_path=Path(__file__).parents[1]/'models/species_growth.json')
    result = service.forecast(dict(species='杨树', height=10, clearance_distance=5,
        voltage_level_kv=110, observation_date='2026-10-07', diameter=26.4,
        diameter_source='MANUAL_FIELD_MEASUREMENT', point_cloud_diameter=None),
        PredictionStandards().to_dict(), prediction_mode='formula',
        short_horizons_days=[1], long_horizons_months=[1])
    assert result['dbh_input']['source'] == 'MANUAL_FIELD_MEASUREMENT'
    assert result['dbh_input']['value_cm'] == 26.4
    assert result['dbh_input']['point_cloud_value_cm'] is None


@pytest.mark.parametrize('cancel', [False, True])
def test_work_order_native_save_as_user_path(tmp_path, monkeypatch, cancel):
    monkeypatch.setitem(sys.modules, 'webview', SimpleNamespace(SAVE_DIALOG='SAVE_DIALOG'))
    target = tmp_path/'operator-choice'/'工单.html'
    calls = []
    def dialog(kind, **options):
        calls.append((kind, options))
        return None if cancel else [str(target)]
    api = DesktopApi(SimpleNamespace())
    api._window = SimpleNamespace(create_file_dialog=dialog)
    body = '<html>人工胸径 26.4 cm / No / 预警工单</html>'.encode('utf-8')
    result = api.save_export_file('TreeLineWarning_工单.html', base64.b64encode(body).decode())
    assert calls[0][0] == 'SAVE_DIALOG'
    assert calls[0][1]['directory'] == ''
    assert calls[0][1]['file_types'][0] == 'Work order HTML (*.html)'
    if cancel:
        assert result == dict(ok=False, cancelled=True)
        assert not target.exists()
    else:
        assert result['path'] == str(target)
        assert target.read_bytes() == body
