from copy import deepcopy

import pytest

from algorithms.prediction.daily_weather import normalize_daily_weather, normalize_monthly_weather
from algorithms.prediction.forecast import ForecastService
from algorithms.prediction.standards import PredictionStandards


class WeatherSensitiveGrowth:
    def predict(self, species, **weather):
        return max(0, weather['temperature']) / 100, .01


def test_daily_projection_uses_each_date_and_never_carries_weather_into_missing_date():
    service = ForecastService(None, WeatherSensitiveGrowth())
    standard = PredictionStandards()
    original = dict(tree_id=1, species='杨树', age=20, height=10, diameter=20,
                    observation_date='2026-12-30', clearance_distance=5, voltage_level_kv=110,
                    wind_speed=2, temperature=10, rainfall=0, humidity=60, sunlight_hours=5)
    saved = deepcopy(original)
    current = service._normalize_snapshot(original, standard)
    weather = normalize_daily_weather([
        dict(date='2026-12-31', temperature=31, wind_force=2),
        dict(date='2027-01-01', temperature=0, wind_force=8),
    ], current['observation_date'])
    projections = [service._short_projection(current, i, standard, weather) for i in [1, 2, 3]]
    assert [p['target_date'] for p in projections] == ['2026-12-31', '2027-01-01', '2027-01-02']
    assert [p['predicted_wind_speed'] for p in projections] == [2.45, 19, 2]
    assert [p['predicted_temperature'] for p in projections] == [31, 0, 10]
    assert projections[0]['height_growth'] == projections[1]['height_growth'] == .01
    assert projections[2]['height_growth'] > .01
    assert '非实际气象预报' in projections[2]['weather_source']
    assert original == saved


def test_forecast_preserves_daily_plan_separate_from_observed_input():
    service = ForecastService(None, WeatherSensitiveGrowth())
    result = service.forecast(dict(tree_id=4, species='杨树', height=10, diameter=20,
                                   clearance_distance=5, voltage_level_kv=110,
                                   observation_date='2026-10-09', wind_speed=2),
                              PredictionStandards().to_dict(), prediction_mode='formula',
                              short_horizons_days=[1, 2, 3], long_horizons_months=[1],
                              future_weather=[dict(date='2026-10-10', wind_force=10)])
    assert result['input_snapshot']['wind_speed'] == 2
    assert result['future_weather'][0]['wind_force'] == 10
    assert result['short_term_forecast'][0]['predicted_wind_speed'] == 26.5
    assert result['short_term_forecast'][1]['predicted_wind_speed'] == 2
    assert result['daily_weather_enabled'] is True


def test_long_term_forecast_uses_operator_monthly_average_weather():
    service = ForecastService(None, WeatherSensitiveGrowth())
    result = service.forecast(
        dict(
            tree_id=8,
            species='杨树',
            height=10,
            diameter=20,
            clearance_distance=5,
            voltage_level_kv=110,
            observation_date='2026-10-09',
            wind_speed=2,
            temperature=10,
        ),
        PredictionStandards().to_dict(),
        prediction_mode='formula',
        short_horizons_days=[1],
        long_horizons_months=[1, 2],
        future_monthly_weather=[
            dict(month_index=1, wind_force=7, temperature=3, humidity=55),
            dict(month_index=2, wind_force=2, temperature=-4, rainfall=5),
        ],
    )
    first, second = result['long_term_forecast']
    assert result['monthly_weather_enabled'] is True
    assert result['future_monthly_weather'][0]['wind_force'] == 7
    assert first['predicted_wind_speed'] == 15.5
    assert first['predicted_temperature'] == 3
    assert first['weather_provided_fields'] == ['wind_speed', 'humidity', 'temperature']
    assert second['predicted_wind_speed'] == 2.45
    assert second['predicted_temperature'] == -4
    assert second['monthly_height_growth'] == 0.0


@pytest.mark.parametrize('rows', [
    'not a list', [None],
    [dict(month_index=0, wind_force=2)],
    [dict(month_index=3, wind_force=2)],
    [dict(month_index=1, wind_force=2), dict(month_index=1, humidity=60)],
    [dict(month_index=1, wind_force=13)],
    [dict(month_index=1, wind_force=2, wind_speed=2)],
    [dict(month_index=1, humidity=float('nan'))],
])
def test_bad_monthly_weather_is_rejected(rows):
    with pytest.raises((ValueError, TypeError)):
        normalize_monthly_weather(rows, 2)


@pytest.mark.parametrize('rows', [
    'not a list', [None], [dict(date='2026-10-09', wind_force=2)],
    [dict(date='2026-10-25', wind_force=2)], [dict(date='not-date')],
    [dict(date='2026-10-10'), dict(date='2026-10-10')],
    [dict(date='2026-10-10', humidity=101)], [dict(date='2026-10-10', wind_speed=-1)],
    [dict(date='2026-10-10', temperature=float('nan'))],
    [dict(date='2026-10-10', wind_force=2.5)], [dict(date='2026-10-10', wind_force=True)],
    [dict(date='2026-10-10', wind_force=2, wind_speed=2)],
])
def test_bad_daily_weather_is_rejected(rows):
    with pytest.raises((ValueError, TypeError)):
        normalize_daily_weather(rows, '2026-10-09')
