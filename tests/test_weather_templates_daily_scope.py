from pathlib import Path
from fastapi.testclient import TestClient
from api.prediction_api import create_app


def test_weather_template_survives_app_restart_and_short_result_is_seven_dated_days(tmp_path):
    database=tmp_path/'weather.db'
    model=Path(__file__).parents[1]/'models/species_growth.json'
    template=dict(name='秋季7天天气',plan=dict(days=7,
        rows=[dict(wind_force=day%5,temperature=10+day,humidity=60,rainfall=day,sunlight_hours=6) for day in range(7)],
        baseline=dict(wind_force=2,temperature=10,selected_fields=['wind_force','temperature'])))
    with TestClient(create_app(database_path=database,species_growth_model_path=model)) as client:
        response=client.post('/weather/templates',json=template)
        assert response.status_code==200,response.text
        identifier=response.json()['id']
    with TestClient(create_app(database_path=database,species_growth_model_path=model)) as client:
        saved=client.get('/weather/templates').json()['templates'][0]
        assert saved['name']==template['name'] and saved['plan']['rows'][6]['temperature']==16
        assert saved['plan']['baseline']['selected_fields']==['wind_force','temperature']
        response=client.post('/predict/forecast',json=dict(prediction_mode='formula',forecast_scope='short',
            tree_data=dict(tree_id=15,species='杨树',height=10,diameter=20,crown_width=3,clearance_distance=8,
                           voltage_level_kv=110,observation_date='2026-12-29',temperature=10,wind_speed=2),
            short_horizons_days=list(range(1,8)),selected_horizon_months=12,
            future_weather=[dict(date=f'2027-01-0{day}',wind_force=day,temperature=day) for day in range(1,5)]))
        assert response.status_code==200,response.text
        report=response.json()
        assert report['forecast_scope']=='short' and report['forecast_days']==7
        assert len(report['short_term_forecast'])==7 and report['long_term_forecast']==[]
        assert report['selected_target_prediction']['target_date']=='2027-01-05'
        assert report['calendar_context']['selected_target_date']=='2027-01-05'
        assert report['risk_timeline_analysis']['peak_risk']['target_date'] <= '2027-01-05'
        assert report['threshold_timing']['scan_days']==7
        assert client.delete(f'/weather/templates/{identifier}').json()['ok']
        assert client.get('/weather/templates').json()['templates']==[]


def test_invalid_template_is_not_persisted(tmp_path):
    with TestClient(create_app(database_path=tmp_path/'invalid.db')) as client:
        response=client.post('/weather/templates',json=dict(name='bad',plan=dict(days=1,rows=[dict(wind_force=14)])))
        assert response.status_code==422
        assert client.get('/weather/templates').json()['templates']==[]


def test_desktop_weather_history_is_shared_when_switching_projects(tmp_path):
    from types import SimpleNamespace
    repository=tmp_path/'程序文件'
    template=dict(name='跨项目天气',plan=dict(days=1,rows=[dict(wind_force=3)],baseline={}))
    for index in range(2):
        app=create_app(database_path=tmp_path/f'project-{index}'/'history.db')
        app.state.processing_backend=SimpleNamespace(repository_root=repository,project_root=tmp_path/f'project-{index}')
        with TestClient(app) as client:
            if index==0:assert client.post('/weather/templates',json=template).status_code==200
            else:assert client.get('/weather/templates').json()['templates'][0]['name']=='跨项目天气'
