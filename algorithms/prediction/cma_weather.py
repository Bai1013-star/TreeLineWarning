"""China Meteorological Administration weather adapter.

Online mode reads the public JSON files used by ``weather.cma.cn``.  The
nationwide station file is used only to choose the nearest station to the
project coordinates; the station detail file supplies the actual observation.
Downloaded CMA JSON/CSV text can also be parsed locally, so prediction does not
depend on a permanently available network endpoint.
"""

from __future__ import annotations

import csv
import io
import json
import math
import time
from datetime import date, datetime
from threading import Lock
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CMA_MAP_URL = "https://weather.cma.cn/api/map/weather/1"
CMA_STATION_URL = "https://weather.cma.cn/api/weather/view"
CMA_PROVIDER = "中国气象局 weather.cma.cn"
_CACHE_SECONDS = 30 * 60
_LOCK = Lock()
_STATION_CACHE: tuple[float, list[dict[str, Any]]] | None = None
_DETAIL_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": "https://weather.cma.cn/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/140.0 Safari/537.36"
    ),
}


def _json_request(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    target = url
    if params:
        target += "?" + urlencode(params)
    request = Request(target, headers=_HEADERS)
    with urlopen(request, timeout=15) as response:  # noqa: S310 - fixed CMA HTTPS hosts
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("code") != 0:
        raise ValueError("中国气象局返回了无效数据")
    return payload


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or abs(number) >= 9999:
        return None
    return number


def _daylight_hours(latitude: float, on_date: date, condition: str = "") -> float:
    """Estimate effective sunshine from astronomical day length + CMA condition."""

    latitude_radians = math.radians(max(-66.0, min(66.0, latitude)))
    declination = math.radians(23.44) * math.sin(
        2.0 * math.pi * (284 + on_date.timetuple().tm_yday) / 365.0
    )
    argument = -math.tan(latitude_radians) * math.tan(declination)
    day_length = 24.0 / math.pi * math.acos(max(-1.0, min(1.0, argument)))
    text = str(condition or "")
    factor = 0.8 if "晴" in text else 0.55 if "多云" in text else 0.3 if "阴" in text else 0.15
    return round(max(0.0, min(24.0, day_length * factor)), 2)


def _stations_from_map(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data")
    rows = data.get("city") if isinstance(data, dict) else None
    stations: list[dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, list) or len(row) < 6:
            continue
        latitude, longitude = _number(row[4]), _number(row[5])
        if latitude is None or longitude is None:
            continue
        stations.append(
            {
                "station_id": str(row[0]),
                "name": str(row[1]),
                "country": str(row[2]),
                "latitude": latitude,
                "longitude": longitude,
                "administrative_code": str(row[16]) if len(row) > 16 else "",
            }
        )
    if not stations:
        raise ValueError("中国气象局全国站点文件中没有可用站点")
    return stations


def _nearest(stations: list[dict[str, Any]], latitude: float, longitude: float) -> dict[str, Any]:
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("项目经纬度无效")
    cosine = max(0.1, math.cos(math.radians(latitude)))
    return min(
        stations,
        key=lambda item: (item["latitude"] - latitude) ** 2
        + ((item["longitude"] - longitude) * cosine) ** 2,
    )


def _parse_station_detail(payload: dict[str, Any], *, requested: dict[str, Any] | None = None) -> dict[str, Any]:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    location = data.get("location") if isinstance(data.get("location"), dict) else {}
    now = data.get("now") if isinstance(data.get("now"), dict) else {}
    daily = data.get("daily") if isinstance(data.get("daily"), list) else []
    today = daily[0] if daily and isinstance(daily[0], dict) else {}
    latitude = _number(location.get("latitude"))
    longitude = _number(location.get("longitude"))
    if latitude is None and requested:
        latitude = _number(requested.get("latitude"))
    if longitude is None and requested:
        longitude = _number(requested.get("longitude"))
    required = {
        "temperature": _number(now.get("temperature")),
        "humidity": _number(now.get("humidity")),
        "rainfall": _number(now.get("precipitation")),
        "wind_speed": _number(now.get("windSpeed")),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("气象局文件缺少字段：" + "、".join(missing))
    condition = str(today.get("dayText") or "")
    observed_text = str(data.get("lastUpdate") or datetime.now().isoformat(timespec="minutes"))
    try:
        observed_date = datetime.strptime(observed_text[:10], "%Y/%m/%d").date()
    except ValueError:
        observed_date = date.today()
    direction = str(now.get("windDirection") or "").strip()
    if direction in {"", "9999", "999999", "null", "None"}:
        fallback = str(today.get("dayWindDirection") or "").strip()
        direction = fallback if fallback not in {"", "9999", "999999"} else "不确定"
    return {
        "location": {
            "station_id": str(location.get("id") or (requested or {}).get("station_id") or ""),
            "name": str(location.get("name") or (requested or {}).get("name") or ""),
            "path": str(location.get("path") or ""),
            "latitude": latitude,
            "longitude": longitude,
        },
        "weather": {
            **{key: round(float(value), 2) for key, value in required.items()},
            "wind_direction": direction,
            "sunlight_hours": _daylight_hours(float(latitude or 0.0), observed_date, condition),
            "observed_at": observed_text,
            "source": CMA_PROVIDER,
            "source_url": "https://weather.cma.cn/",
            "sunlight_method": "气象局天气现象结合当地理论昼长估算",
            "condition": condition,
        },
        "notice": "已按项目坐标自动匹配最近的中国气象局站点；无需手工填写天气。",
    }


def cma_weather(latitude: float, longitude: float) -> dict[str, Any]:
    """Fetch the nearest official CMA station observation for one coordinate."""

    global _STATION_CACHE
    now = time.monotonic()
    with _LOCK:
        cached = _STATION_CACHE
    if not cached or now - cached[0] >= _CACHE_SECONDS:
        stations = _stations_from_map(_json_request(CMA_MAP_URL, {"t": int(time.time() * 1000)}))
        with _LOCK:
            _STATION_CACHE = (now, stations)
    else:
        stations = cached[1]
    station = _nearest(stations, float(latitude), float(longitude))
    station_id = station["station_id"]
    with _LOCK:
        detail_cached = _DETAIL_CACHE.get(station_id)
    if detail_cached and now - detail_cached[0] < _CACHE_SECONDS:
        return {**detail_cached[1], "cached": True}
    result = _parse_station_detail(
        _json_request(CMA_STATION_URL, {"stationid": station_id}), requested=station
    )
    result["requested_coordinates"] = {"latitude": latitude, "longitude": longitude}
    result["cached"] = False
    with _LOCK:
        _DETAIL_CACHE[station_id] = (now, result)
    return result


_ALIASES = {
    "temperature": ("temperature", "temp", "tem", "气温", "平均气温"),
    "humidity": ("humidity", "rhu", "relative_humidity", "相对湿度", "平均相对湿度"),
    "rainfall": ("rainfall", "precipitation", "pre", "降水量", "降雨量"),
    "wind_speed": ("wind_speed", "windspeed", "win_s_avg_2mi", "平均风速", "风速"),
    "sunlight_hours": ("sunlight_hours", "sunshine", "ssh", "日照时数", "日照"),
    "latitude": ("latitude", "lat", "纬度"),
    "longitude": ("longitude", "lon", "lng", "经度"),
    "observed_at": ("observed_at", "datetime", "date", "time", "观测时间", "日期"),
}


def _lookup(row: dict[str, Any], name: str) -> Any:
    normalized = {str(key).strip().lower(): value for key, value in row.items()}
    for alias in _ALIASES[name]:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None


def parse_cma_file(filename: str, content: str, latitude: float | None = None, longitude: float | None = None) -> dict[str, Any]:
    """Parse a downloaded CMA station JSON or normalized CSV/TXT file."""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("气象局文件为空")
    if len(content.encode("utf-8")) > 20 * 1024 * 1024:
        raise ValueError("气象局文件不能超过 20 MB")
    stripped = content.lstrip("\ufeff \t\r\n")
    if stripped.startswith(("{", "[")):
        payload = json.loads(stripped)
        if isinstance(payload, list):
            if not payload or not isinstance(payload[-1], dict):
                raise ValueError("气象局 JSON 文件没有观测记录")
            rows = payload
        elif isinstance(payload, dict) and (
            isinstance(payload.get("now"), dict)
            or isinstance(payload.get("data"), dict)
            and isinstance(payload["data"].get("now"), dict)
        ):
            result = _parse_station_detail(payload)
            result["weather"]["source"] = f"中国气象局文件：{filename}"
            result["cached"] = False
            return result
        elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
            rows = payload["records"]
        else:
            raise ValueError("暂不识别这个气象局 JSON 结构")
    else:
        sample = stripped[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.DictReader(io.StringIO(stripped), dialect=dialect))
    candidates = [row for row in rows if isinstance(row, dict)]
    if not candidates:
        raise ValueError("气象局文件没有可用观测行")
    if latitude is not None and longitude is not None:
        located = [
            row for row in candidates
            if _number(_lookup(row, "latitude")) is not None
            and _number(_lookup(row, "longitude")) is not None
        ]
        if located:
            candidates = [
                min(
                    located,
                    key=lambda row: (_number(_lookup(row, "latitude")) - float(latitude)) ** 2
                    + (_number(_lookup(row, "longitude")) - float(longitude)) ** 2,
                )
            ]
    row = candidates[-1]
    weather = {name: _number(_lookup(row, name)) for name in ("temperature", "humidity", "rainfall", "wind_speed", "sunlight_hours")}
    if weather["sunlight_hours"] is None:
        weather["sunlight_hours"] = _daylight_hours(float(latitude or 0.0), date.today(), "多云")
    missing = [name for name, value in weather.items() if value is None]
    if missing:
        raise ValueError("气象局文件缺少字段：" + "、".join(missing))
    return {
        "location": {"latitude": latitude, "longitude": longitude},
        "weather": {
            **{key: round(float(value), 2) for key, value in weather.items()},
            "observed_at": str(_lookup(row, "observed_at") or ""),
            "source": f"中国气象局文件：{filename}",
            "sunlight_method": "文件值或当地理论昼长估算",
        },
        "cached": False,
        "notice": "已直接采用气象局文件生成预测天气输入，无需手工填写。",
    }


__all__ = ["cma_weather", "parse_cma_file"]
