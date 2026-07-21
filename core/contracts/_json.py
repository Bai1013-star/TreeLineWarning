"""JSON-safe validation helpers for public contract metadata."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def json_safe(value: Any, field_name: str) -> JsonValue:
    """Return a JSON-safe copy of *value* or raise a field-specific ValueError."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite float values")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} keys must be strings")
            result[key] = json_safe(item, f"{field_name}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [json_safe(item, f"{field_name}[]") for item in value]
    raise ValueError(f"{field_name} must be JSON-serializable, got {type(value).__name__}")


def json_safe_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, JsonValue]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    safe = json_safe(value, field_name)
    if not isinstance(safe, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return safe


def reject_unknown_fields(
    data: Mapping[str, Any],
    allowed_fields: set[str],
    model_name: str,
) -> None:
    unknown = sorted(set(data) - allowed_fields)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"{model_name} has unknown field(s): {joined}")


def require_fields(data: Mapping[str, Any], required_fields: set[str], model_name: str) -> None:
    missing = sorted(required_fields - set(data))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{model_name} is missing required field(s): {joined}")
