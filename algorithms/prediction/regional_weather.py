"""Location-aware weather lookup used by desktop risk reports.

The module deliberately keeps the weather provider behind a small, auditable
adapter: users select a Chinese province/municipality and enter city/district;
the report records the returned coordinates, observation time and provider.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REVERSE_GEOCODING_URL = "https://nominatim.openstreetmap.org/reverse"
PROVIDER_NAME = "Open-Meteo Forecast API"
CACHE_TTL_SECONDS = 20 * 60
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_REVERSE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_LOCK = Lock()


@dataclass(frozen=True)
class RegionalLocation:
    province: str
    city: str
    district: str

    @property
    def query(self) -> str:
        parts = [self.province.strip(), self.city.strip(), self.district.strip(), "中国"]
        return " ".join(part for part in parts if part)

    @property
    def label(self) -> str:
        return "·".join(part for part in (self.province, self.city, self.district) if part.strip())


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    request_url = f"{url}?{urlencode(params, doseq=True)}"
    with urlopen(request_url, timeout=8) as response:  # noqa: S310 - fixed HTTPS provider
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("气象服务返回格式无效")
    return payload


def _get_reverse_json(latitude: float, longitude: float) -> dict[str, Any]:
    """Fetch one user-triggered OSM reverse-geocoding result.

    Nominatim requires identifiable requests and modest traffic. This helper is
    invoked only after a map click and callers cache the result to avoid repeat
    lookups for the same point.
    """
    query = urlencode(
        {
            "lat": latitude,
            "lon": longitude,
            "format": "jsonv2",
            "addressdetails": 1,
            "accept-language": "zh-CN",
            "zoom": 12,
        }
    )
    request = Request(
        f"{REVERSE_GEOCODING_URL}?{query}",
        headers={"User-Agent": "TreeLineWarning/1.0 (desktop map selection)"},
    )
    with urlopen(request, timeout=8) as response:  # noqa: S310 - fixed HTTPS provider
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("地图定位服务返回格式无效")
    return payload


def _number(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key, default)
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return default


def _cache_key(location: RegionalLocation) -> str:
    return "|".join((location.province, location.city, location.district)).strip().lower()


def regional_weather(province: str, city: str, district: str = "") -> dict[str, Any]:
    """Resolve a Chinese administrative location and return current risk inputs.

    The returned fields map directly to the prediction form.  They are a
    time-stamped weather snapshot, not a replacement for local station records
    or official severe-weather warnings.
    """
    location = RegionalLocation(
        str(province or "").strip(),
        str(city or "").strip(),
        str(district or "").strip(),
    )
    if not location.province or not location.city:
        raise ValueError("请选择省/直辖市并填写城市/区县。")
    key = _cache_key(location)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and time.monotonic() - cached[0] < CACHE_TTL_SECONDS:
            return {**cached[1], "cached": True}

    results: list[Any] = []
    # District spellings are not consistently indexed by global geocoders.
    # Try the requested administrative chain first, then relax to city/province
    # while keeping the user's requested district in the report label.
    candidates = [
        location.query,
        f"{location.district} {location.city}",
        location.city,
        location.province,
    ]
    for query in dict.fromkeys(candidate for candidate in candidates if candidate.strip()):
        geocoded = _get_json(
            GEOCODING_URL,
            {"name": query, "count": 1, "language": "zh", "format": "json"},
        )
        value = geocoded.get("results")
        if isinstance(value, list) and value:
            results = value
            break
    if not results:
        raise ValueError(f"未找到“{location.label}”的坐标，请检查城市和区县名称。")
    place = results[0] if isinstance(results[0], dict) else {}
    latitude = _number(place, "latitude")
    longitude = _number(place, "longitude")
    weather = _get_json(
        FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "Asia/Shanghai",
            "current": (
                "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_gusts_10m"
            ),
            "daily": (
                "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                "wind_speed_10m_max,wind_gusts_10m_max,sunshine_duration"
            ),
            "forecast_days": 1,
        },
    )
    current = weather.get("current") if isinstance(weather.get("current"), dict) else {}
    daily = weather.get("daily") if isinstance(weather.get("daily"), dict) else {}

    def first_daily(name: str, default: float = 0.0) -> float:
        values = daily.get(name)
        value = values[0] if isinstance(values, list) and values else default
        return _number({"value": value}, "value", default)

    result = {
        "location": {
            "requested": location.label,
            "resolved_name": str(place.get("name") or location.label),
            "admin1": str(place.get("admin1") or ""),
            "country": str(place.get("country") or "中国"),
            "latitude": latitude,
            "longitude": longitude,
        },
        "weather": {
            "temperature": _number(current, "temperature_2m"),
            "humidity": _number(current, "relative_humidity_2m"),
            "rainfall": first_daily("precipitation_sum"),
            "current_precipitation": _number(current, "precipitation"),
            "wind_speed": _number(current, "wind_speed_10m"),
            "wind_gust": _number(current, "wind_gusts_10m"),
            "daily_max_temperature": first_daily("temperature_2m_max"),
            "daily_min_temperature": first_daily("temperature_2m_min"),
            "daily_max_wind": first_daily("wind_speed_10m_max"),
            "daily_max_gust": first_daily("wind_gusts_10m_max"),
            "sunlight_hours": round(first_daily("sunshine_duration") / 3600, 2),
            "observed_at": str(current.get("time") or ""),
            "forecast_date": str((daily.get("time") or [""])[0]),
            "source": PROVIDER_NAME,
            "source_url": "https://open-meteo.com/en/docs",
        },
        "cached": False,
        "notice": "在线气象快照仅用于本次风险计算；现场实测和属地气象预警优先。",
    }
    with _CACHE_LOCK:
        _CACHE[key] = (time.monotonic(), result)
    return result


def reverse_geocode(latitude: float, longitude: float) -> dict[str, Any]:
    """Return one click-selected Chinese administrative location from OSM."""
    latitude, longitude = round(float(latitude), 6), round(float(longitude), 6)
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("地图坐标无效。")
    key = f"{latitude:.4f}|{longitude:.4f}"
    with _CACHE_LOCK:
        cached = _REVERSE_CACHE.get(key)
        if cached and time.monotonic() - cached[0] < CACHE_TTL_SECONDS:
            return {**cached[1], "cached": True}
    payload = _get_reverse_json(latitude, longitude)
    address = payload.get("address") if isinstance(payload.get("address"), dict) else {}
    province = str(address.get("province") or address.get("state") or address.get("region") or "")
    city = str(
        address.get("city")
        or address.get("municipality")
        or address.get("prefecture")
        or address.get("county")
        or ""
    )
    district = str(
        address.get("city_district")
        or address.get("district")
        or address.get("county")
        or address.get("suburb")
        or ""
    )
    result = {
        "coordinates": {"latitude": latitude, "longitude": longitude},
        "location": {
            "province": province,
            "city": city,
            "district": district,
            "display_name": str(payload.get("display_name") or ""),
        },
        "source": "OpenStreetMap Nominatim Reverse Geocoding",
        "cached": False,
    }
    with _CACHE_LOCK:
        _REVERSE_CACHE[key] = (time.monotonic(), result)
    return result
