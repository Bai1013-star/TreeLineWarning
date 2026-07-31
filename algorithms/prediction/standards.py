"""User-configurable regional and industry prediction standards."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PredictionStandards:
    profile_name: str = "通用输电线路标准"
    region: str = "通用地区"
    industry: str = "输电线路"
    min_vertical_distance: float = 4.0
    min_horizontal_distance: float = 4.0
    min_clearance_distance: float = 5.0
    max_wind_offset_distance: float = 2.0
    wind_warning_speed: float = 10.0
    wind_high_speed: float = 15.0
    rainfall_warning: float = 50.0
    temperature_growth_threshold: float = 30.0
    sunlight_growth_threshold: float = 10.0
    medium_risk_score: float = 40.0
    high_risk_score: float = 70.0
    voltage_standards: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "10": {
                "min_vertical_distance": 2.0,
                "min_horizontal_distance": 1.5,
                "min_clearance_distance": 1.5,
                "max_sag": 2.0,
            },
            "35": {
                "min_vertical_distance": 4.0,
                "min_horizontal_distance": 3.0,
                "min_clearance_distance": 3.0,
                "max_sag": 3.0,
            },
            "66": {
                "min_vertical_distance": 4.0,
                "min_horizontal_distance": 3.5,
                "min_clearance_distance": 3.5,
                "max_sag": 4.0,
            },
            "110": {
                "min_vertical_distance": 4.0,
                "min_horizontal_distance": 3.5,
                "min_clearance_distance": 3.5,
                "max_sag": 5.0,
            },
            "220": {
                "min_vertical_distance": 4.5,
                "min_horizontal_distance": 4.0,
                "min_clearance_distance": 4.0,
                "max_sag": 6.0,
            },
            "330": {
                "min_vertical_distance": 5.5,
                "min_horizontal_distance": 5.0,
                "min_clearance_distance": 5.0,
                "max_sag": 7.0,
            },
            "500": {
                "min_vertical_distance": 7.0,
                "min_horizontal_distance": 7.0,
                "min_clearance_distance": 7.0,
                "max_sag": 9.0,
            },
            "750": {
                "min_vertical_distance": 8.5,
                "min_horizontal_distance": 8.5,
                "min_clearance_distance": 8.5,
                "max_sag": 11.0,
            },
            "1000": {
                "min_vertical_distance": 10.0,
                "min_horizontal_distance": 10.0,
                "min_clearance_distance": 10.0,
                "max_sag": 13.0,
            },
        }
    )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> PredictionStandards:
        if not value:
            return cls()
        allowed = cls.__dataclass_fields__
        cleaned = {key: item for key, item in value.items() if key in allowed}
        for key in allowed:
            if key.startswith(("min_", "max_")) or key.endswith(
                ("_speed", "_warning", "_threshold", "_score")
            ):
                if key in cleaned:
                    cleaned[key] = float(cleaned[key])
        result = cls(**cleaned)
        result.validate()
        return result

    def validate(self) -> None:
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"{key} 不能小于 0")
        if self.wind_high_speed <= self.wind_warning_speed:
            raise ValueError("高风速阈值必须大于预警风速阈值")
        if self.high_risk_score <= self.medium_risk_score:
            raise ValueError("高风险分数阈值必须大于中风险分数阈值")
        if self.high_risk_score > 100:
            raise ValueError("高风险分数阈值不能超过 100")
        for voltage, rules in self.voltage_standards.items():
            try:
                float(voltage)
            except ValueError as exc:
                raise ValueError(f"无效电压等级: {voltage}") from exc
            required = (
                "min_vertical_distance",
                "min_horizontal_distance",
                "min_clearance_distance",
                "max_sag",
            )
            for key in required:
                if key not in rules or float(rules[key]) < 0:
                    raise ValueError(f"{voltage}kV 的 {key} 配置无效")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def rules_for_voltage(self, voltage_level_kv: float) -> dict[str, float]:
        if not self.voltage_standards:
            return {
                "min_vertical_distance": self.min_vertical_distance,
                "min_horizontal_distance": self.min_horizontal_distance,
                "min_clearance_distance": self.min_clearance_distance,
                "max_sag": float("inf"),
            }
        closest = min(
            self.voltage_standards,
            key=lambda value: abs(float(value) - voltage_level_kv),
        )
        rules = {
            key: float(value)
            for key, value in self.voltage_standards[closest].items()
        }
        rules["matched_voltage_level_kv"] = float(closest)
        return rules


def analyze_distances(
    *,
    vertical_distance: float,
    horizontal_distance: float,
    clearance_distance: float,
    wind_offset_distance: float,
    standards: PredictionStandards,
    voltage_level_kv: float | None = None,
) -> dict[str, Any]:
    voltage_rules = (
        standards.rules_for_voltage(voltage_level_kv)
        if voltage_level_kv is not None
        else {
            "min_vertical_distance": standards.min_vertical_distance,
            "min_horizontal_distance": standards.min_horizontal_distance,
            "min_clearance_distance": standards.min_clearance_distance,
        }
    )
    checks = {
        "vertical_distance": {
            "actual": vertical_distance,
            "standard": voltage_rules["min_vertical_distance"],
            "compliant": vertical_distance >= voltage_rules["min_vertical_distance"],
            "label": "垂直距离",
        },
        "horizontal_distance": {
            "actual": horizontal_distance,
            "standard": voltage_rules["min_horizontal_distance"],
            "compliant": horizontal_distance >= voltage_rules["min_horizontal_distance"],
            "label": "水平距离",
        },
        "clearance_distance": {
            "actual": clearance_distance,
            "standard": voltage_rules["min_clearance_distance"],
            "compliant": clearance_distance >= voltage_rules["min_clearance_distance"],
            "label": "净空距离",
        },
        "wind_offset_distance": {
            "actual": wind_offset_distance,
            "standard": standards.max_wind_offset_distance,
            "compliant": wind_offset_distance <= standards.max_wind_offset_distance,
            "label": "风偏距离",
        },
    }
    violations = [
        {
            "field": field,
            "label": check["label"],
            "actual": check["actual"],
            "standard": check["standard"],
        }
        for field, check in checks.items()
        if not check["compliant"]
    ]
    return {
        "compliant": not violations,
        "voltage_rules": voltage_rules,
        "checks": checks,
        "violations": violations,
    }


def analyze_wire(
    *,
    voltage_level_kv: float,
    maximum_sag_winter: float,
    maximum_sag_summer: float,
    observation_month: int,
    standards: PredictionStandards,
) -> dict[str, Any]:
    rules = standards.rules_for_voltage(voltage_level_kv)
    month = min(12, max(1, int(observation_month)))
    summer_factor = (math.cos(2 * math.pi * (month - 7) / 12) + 1) / 2
    current_sag = (
        maximum_sag_winter
        + (maximum_sag_summer - maximum_sag_winter) * summer_factor
    )
    maximum_sag = max(maximum_sag_winter, maximum_sag_summer)
    return {
        "voltage_level_kv": voltage_level_kv,
        "matched_voltage_level_kv": rules["matched_voltage_level_kv"],
        "observation_month": month,
        "season": "夏季工况" if summer_factor >= 0.5 else "冬季工况",
        "current_sag": round(current_sag, 3),
        "maximum_sag": round(maximum_sag, 3),
        "standard_max_sag": rules["max_sag"],
        "compliant": maximum_sag <= rules["max_sag"],
    }
