"""Explicit dated weather scenarios; never masquerade as a live forecast."""
from datetime import date
import math

FIELDS = ('wind_speed', 'rainfall', 'humidity', 'temperature', 'sunlight_hours')
BOUNDS = {'wind_speed': (0, 150), 'rainfall': (0, 3000), 'humidity': (0, 100),
          'temperature': (-90, 70), 'sunlight_hours': (0, 24)}
# Representative speeds for a selected force interval, not a measured speed.
FORCE_SPEED = (.15, .95, 2.45, 4.4, 6.75, 9.4, 12.3, 15.5, 19, 22.65, 26.5, 30.6, 34.7)


def normalize_daily_weather(rows, observation_date):
    if rows is None:
        return {}
    if not isinstance(rows, list) or len(rows) > 15:
        raise ValueError('未来逐日天气须为最多 15 行的列表')
    observed = date.fromisoformat(observation_date)
    result = {}
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('每一天的天气必须包含日期及天气字段')
        try:
            day = date.fromisoformat(str(row.get('date', '')))
        except ValueError as error:
            raise ValueError('逐日天气日期须为 YYYY-MM-DD') from error
        if not 1 <= (day - observed).days <= 15 or day.isoformat() in seen:
            raise ValueError('逐日天气日期不能重复，须在观测次日起未来 15 天内')
        seen.add(day.isoformat())
        weather = {}
        for field in FIELDS:
            value = row.get(field)
            if value is None or value == '':
                continue
            if isinstance(value, bool):
                raise ValueError(f'{field} 必须是有效数值')
            number = float(value)
            low, high = BOUNDS[field]
            if not math.isfinite(number) or not low <= number <= high:
                raise ValueError(f'{field} 超出有效范围 {low}–{high}')
            weather[field] = number
        force = row.get('wind_force')
        if force is not None and force != '':
            if isinstance(force, bool) or not math.isfinite(float(force)) or float(force) != int(float(force)) or not 0 <= int(float(force)) <= 12:
                raise ValueError('风级须为 0–12 的整数')
            if 'wind_speed' in weather:
                raise ValueError('一天内请选择风级或实测风速，不能同时输入')
            force = int(float(force))
            weather.update(wind_speed=FORCE_SPEED[force], wind_force=force,
                           wind_input_basis='风级区间代表速度，非精确实测风速')
        if not weather:
            continue
        weather['source'] = str(row.get('source') or '人工逐日天气情景')[:200]
        result[day.isoformat()] = weather
    return result


def normalize_monthly_weather(rows, maximum_months):
    """Validate operator-provided monthly averages for long-term scenarios."""
    if rows is None:
        return {}
    if not isinstance(rows, list) or len(rows) > 120:
        raise ValueError('未来逐月天气须为最多 120 行的列表')
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('每个月的天气必须是一个对象')
        index = row.get('month_index')
        if isinstance(index, bool) or not str(index).strip().isdigit():
            raise ValueError('逐月天气 month_index 须为正整数')
        index = int(index)
        if not 1 <= index <= int(maximum_months) or index in result:
            raise ValueError('逐月天气月份不能重复，且须在本次预测月份范围内')
        weather = {}
        for field in FIELDS:
            value = row.get(field)
            if value is None or value == '':
                continue
            if isinstance(value, bool):
                raise ValueError(f'{field} 必须是有效数值')
            number = float(value)
            low, high = BOUNDS[field]
            if not math.isfinite(number) or not low <= number <= high:
                raise ValueError(f'{field} 超出有效范围 {low}–{high}')
            weather[field] = number
        force = row.get('wind_force')
        if force is not None and force != '':
            if isinstance(force, bool) or not math.isfinite(float(force)) or float(force) != int(float(force)) or not 0 <= int(float(force)) <= 12:
                raise ValueError('月平均风力等级须为 0–12 的整数')
            if 'wind_speed' in weather:
                raise ValueError('一个月内请选择风级或实测风速，不能同时输入')
            force = int(float(force))
            weather.update(
                wind_speed=FORCE_SPEED[force],
                wind_force=force,
                wind_input_basis='月平均风级区间代表速度，非精确实测风速',
            )
        if weather:
            weather['source'] = str(row.get('source') or '人工逐月平均天气情景')[:200]
            result[index] = weather
    return result
