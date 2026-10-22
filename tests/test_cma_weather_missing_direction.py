import pytest
from algorithms.prediction.cma_weather import _parse_station_detail


@pytest.mark.parametrize('fallback, expected', [('', '不确定'), ('9999', '不确定'), ('北风', '北风')])
def test_official_missing_wind_code_is_not_shown_as_direction(fallback, expected):
    result = _parse_station_detail({'data': {'now': dict(temperature=8, humidity=60,
        precipitation=0, windSpeed=0, windDirection='9999'),
        'daily': [dict(dayWindDirection=fallback)],
        'location': dict(latitude=39.89, longitude=116.32)}})
    assert result['weather']['wind_direction'] == expected
