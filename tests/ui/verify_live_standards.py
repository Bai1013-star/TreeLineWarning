"""Live standards edits in audit-only projects, restored in finally."""
import copy
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from algorithms.prediction.forecast import ForecastService
from algorithms.prediction.species_growth import SpeciesGrowthModel
from algorithms.prediction.standards import PredictionStandards

out = Path(os.environ['REAL_OUTPUT'])
base = json.loads(Path(os.environ['REAL_SERVERS']).read_text(encoding='utf-8'))['warning'].rstrip('/')
checks = []

def api(path, data=None, method=None):
    request = Request(base + path, data=None if data is None else json.dumps(data).encode(),
                      headers={'Content-Type': 'application/json'}, method=method)
    try:
        with urlopen(request, timeout=120) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)

def forecast(**changes):
    tree = dict(tree_id=772001, species='油松', age=20, height=10, diameter=20,
                crown_width=3, observation_date='2026-10-02', voltage_level_kv=110,
                clearance_distance=10, horizontal_distance=10, vertical_distance=10,
                maximum_sag_winter=0, maximum_sag_summer=0, wind_offset_distance=0,
                wind_speed=0, temperature=20, humidity=60, rainfall=10, sunlight_hours=8,
                contact_target='未碰到')
    tree.update(changes)
    code, result = api('/predict/forecast', dict(tree_data=tree, prediction_mode='formula',
        history_records=[], short_horizons_days=[1, 30], long_horizons_months=[1, 6, 12],
        selected_horizon_months=12), 'POST')
    assert code == 200, (code, result)
    return result

code, original = api('/settings/standards')
assert code == 200
try:
    for threshold in [9.99, 10.0, 10.01]:
        settings = copy.deepcopy(original)
        settings.update(profile_name='双点云独立阈值边界验收', danger_clearance_distance=threshold,
                        use_contact_risk_override=False)
        assert api('/settings/standards', settings, 'PUT')[0] == 200
        result = forecast()
        timing = result['threshold_timing']['danger_clearance']
        assert result['applied_standard']['danger_clearance_distance'] == threshold
        assert timing['current_effective_clearance'] == 10
        assert (timing['days'] == 0) == (10 <= threshold)
        if isinstance(timing['days'], int) and timing['days'] > 0:
            # Independent first-day boundary verification through direct forecasts.
            day = timing['days']
            tree = result['input_snapshot']
            service = ForecastService(None, SpeciesGrowthModel(Path('models/species_growth.json')))
            standard = PredictionStandards.from_mapping(settings)
            current = service._normalize_snapshot(tree, standard)
            values = [service._short_projection(current, d, standard) for d in ([day-1, day] if day>1 else [1])]
            assert values[-1]['predicted_effective_clearance'] <= threshold
            if day>1:
                assert values[-2]['predicted_effective_clearance'] > threshold
        checks.append(dict(name='危险净空临界值', threshold=threshold,
                           days=timing['days'], target_date=timing['target_date'], passed=True))
    settings = copy.deepcopy(original)
    settings.update(profile_name='110kV表与实际标准独立验收', use_contact_risk_override=False)
    settings['voltage_standards']['110'].update(min_horizontal_distance=9.25,
        min_vertical_distance=9.5, min_clearance_distance=9.75,
        critical_clearance_distance=1.25, severe_clearance_distance=2.75)
    assert api('/settings/standards', settings, 'PUT')[0] == 200
    result = forecast()
    measured = result['current_assessment']['standard_analysis']['checks']
    for field, expected in [('horizontal_distance',9.25), ('vertical_distance',9.5), ('clearance_distance',9.75)]:
        assert measured[field]['standard'] == expected
    checks.append(dict(name='电压标准进入实际计算', passed=True, checks=measured))
    settings.update(use_actual_clearance_standard=True, actual_min_clearance_distance=10.5)
    assert api('/settings/standards', settings, 'PUT')[0] == 200
    result = forecast()
    assert result['current_assessment']['standard_analysis']['checks']['clearance_distance']['standard'] == 10.5
    checks.append(dict(name='实际净空标准覆盖生效', passed=True, standard=10.5))
    invalid = copy.deepcopy(settings)
    invalid['danger_clearance_distance'] = -1
    code, response = api('/settings/standards', invalid, 'PUT')
    assert code == 422
    assert api('/settings/standards')[1]['danger_clearance_distance'] == settings['danger_clearance_distance']
    checks.append(dict(name='非法阈值拒绝且不覆盖已保存值', passed=True, status=code))
    january = forecast(observation_date='2026-01-01')
    july = forecast(observation_date='2026-07-01')
    winter = january['short_term_forecast'][-1]['height_growth']
    summer = july['short_term_forecast'][-1]['height_growth']
    assert summer > winter >= 0
    checks.append(dict(name='相同树木30日季节增高对比', winter_m=winter, summer_m=summer, passed=True))
    code, reset = api('/settings/standards/defaults', {}, 'POST')
    assert code == 200 and reset['danger_clearance_distance'] == 5
    checks.append(dict(name='恢复默认标准', passed=True))
finally:
    code, restored = api('/settings/standards', original, 'PUT')
    assert code == 200 and restored == original
    (out/'live_standards_checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(checks, ensure_ascii=False))
