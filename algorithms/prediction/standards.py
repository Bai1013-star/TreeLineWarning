"""User-configurable regional and industry prediction standards."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_FEATURE_ATTENTION_WEIGHTS = {
    "historical_clearance_distance": 0.25,
    "historical_horizontal_distance": 0.15,
    "historical_vertical_distance": 0.12,
    "monthly_temperature": 0.18,
    "monthly_rainfall": 0.10,
    "monthly_sunlight_hours": 0.08,
    "monthly_wind_speed": 0.04,
    # 十二个月份共同占 0.08；旺长期权重稍高，全年仍保持连续可配置。
    "month_s01": 0.003,
    "month_s02": 0.003,
    "month_s03": 0.005,
    "month_s04": 0.007,
    "month_s05": 0.010,
    "month_s06": 0.012,
    "month_s07": 0.013,
    "month_s08": 0.012,
    "month_s09": 0.007,
    "month_s10": 0.004,
    "month_s11": 0.002,
    "month_s12": 0.002,
}


def _clearance_thresholds() -> dict[str, float]:
    return {
        "critical_clearance_distance": 1.0,
        "severe_clearance_distance": 2.5,
    }

@dataclass(frozen=True)
class PredictionStandards:
    profile_name: str = "通用输电线路标准"
    region: str = "通用地区"
    industry: str = "输电线路"
    standard_reference: str = (
        "工程初始值：含 220V（0.22kV）配电线路、10kV 及以上等级；"
        "风偏列表示最大风偏工况下的最小剩余净距。正式应用前请核对现行地区、"
        "行业、导线绝缘型式及线路运维标准"
    )
    min_vertical_distance: float = 4.0
    min_horizontal_distance: float = 4.0
    min_clearance_distance: float = 5.0
    max_wind_offset_distance: float = 2.0
    wind_warning_speed: float = 10.0
    wind_high_speed: float = 15.0
    rainfall_warning: float = 50.0
    temperature_growth_threshold: float = 30.0
    sunlight_growth_threshold: float = 10.0
    slope_warning_angle: float = 25.0
    medium_risk_score: float = 40.0
    high_risk_score: float = 70.0
    general_hazard_index: float = 0.20
    major_hazard_index: float = 0.40
    emergency_hazard_index: float = 0.70
    risk_weight_distance: float = 0.40
    risk_weight_growth: float = 0.25
    risk_weight_weather: float = 0.20
    risk_weight_history: float = 0.15
    weather_weight_wind: float = 0.40
    weather_weight_rainfall: float = 0.25
    weather_weight_temperature: float = 0.20
    weather_weight_sunlight: float = 0.15
    history_weight_exceedance: float = 0.35
    history_weight_incident: float = 0.30
    history_weight_repeat: float = 0.20
    history_weight_treatment_recurrence: float = 0.15
    feature_attention_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FEATURE_ATTENTION_WEIGHTS)
    )
    voltage_standards: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "0.22": {
                "min_vertical_distance": 1.0,
                "min_horizontal_distance": 1.0,
                "min_wind_offset_distance": 1.0,
                "min_clearance_distance": 1.0,
                "max_sag": 1.0,
                "critical_clearance_distance": 0.2,
                "severe_clearance_distance": 0.5,
            },
            "10": {
                "min_vertical_distance": 2.0,
                "min_horizontal_distance": 1.5,
                "min_wind_offset_distance": 1.5,
                "min_clearance_distance": 3.0,
                "max_sag": 2.0,
                **_clearance_thresholds(),
            },
            "35": {
                "min_vertical_distance": 4.0,
                "min_horizontal_distance": 3.0,
                "min_wind_offset_distance": 3.5,
                "min_clearance_distance": 3.0,
                "max_sag": 3.0,
                **_clearance_thresholds(),
            },
            "66": {
                "min_vertical_distance": 4.0,
                "min_horizontal_distance": 3.5,
                "min_wind_offset_distance": 3.5,
                "min_clearance_distance": 3.5,
                "max_sag": 4.0,
                **_clearance_thresholds(),
            },
            "110": {
                "min_vertical_distance": 4.0,
                "min_horizontal_distance": 3.5,
                "min_wind_offset_distance": 4.0,
                "min_clearance_distance": 3.5,
                "max_sag": 5.0,
                **_clearance_thresholds(),
            },
            "220": {
                "min_vertical_distance": 4.5,
                "min_horizontal_distance": 4.0,
                "min_wind_offset_distance": 5.0,
                "min_clearance_distance": 4.0,
                "max_sag": 6.0,
                **_clearance_thresholds(),
            },
            "330": {
                "min_vertical_distance": 5.5,
                "min_horizontal_distance": 5.0,
                "min_wind_offset_distance": 5.0,
                "min_clearance_distance": 5.0,
                "max_sag": 7.0,
                **_clearance_thresholds(),
            },
            "500": {
                "min_vertical_distance": 7.0,
                "min_horizontal_distance": 7.0,
                "min_wind_offset_distance": 7.0,
                "min_clearance_distance": 7.0,
                "max_sag": 9.0,
                **_clearance_thresholds(),
            },
            "750": {
                "min_vertical_distance": 8.5,
                "min_horizontal_distance": 8.5,
                "min_wind_offset_distance": 8.5,
                "min_clearance_distance": 8.5,
                "max_sag": 11.0,
                **_clearance_thresholds(),
            },
            "1000": {
                "min_vertical_distance": 10.0,
                "min_horizontal_distance": 10.0,
                "min_wind_offset_distance": 10.0,
                "min_clearance_distance": 10.0,
                "max_sag": 13.0,
                **_clearance_thresholds(),
            },
        }
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> PredictionStandards:
        if not value:
            return cls()
        allowed = cls.__dataclass_fields__
        cleaned = {key: item for key, item in value.items() if key in allowed}
        defaults = cls()
        if "feature_attention_weights" in cleaned:
            supplied = cleaned["feature_attention_weights"]
            if not isinstance(supplied, Mapping):
                raise ValueError("特征注意力权重必须是对象")
            merged_attention = {
                key: float(supplied.get(key, default))
                for key, default in defaults.feature_attention_weights.items()
            }
            month_keys = [f"month_s{month:02d}" for month in range(1, 13)]
            supplied_month_keys = [key for key in month_keys if key in supplied]
            if 0 < len(supplied_month_keys) < len(month_keys):
                # 兼容旧版只保存少数月份的配置：保留非月份权重，
                # 将剩余权重按新版全年先验重新分配给 1—12 月。
                non_month_sum = sum(
                    weight
                    for key, weight in merged_attention.items()
                    if key not in month_keys
                )
                month_budget = max(0.0, 1.0 - non_month_sum)
                default_month_sum = sum(
                    defaults.feature_attention_weights[key] for key in month_keys
                )
                for key in month_keys:
                    merged_attention[key] = (
                        defaults.feature_attention_weights[key]
                        / default_month_sum
                        * month_budget
                    )
            cleaned["feature_attention_weights"] = merged_attention
        if "voltage_standards" in cleaned:
            supplied_rules = cleaned["voltage_standards"]
            if not isinstance(supplied_rules, Mapping):
                raise ValueError("电压等级标准必须是对象")
            merged_rules: dict[str, dict[str, float]] = {}
            for voltage, default_rules in defaults.voltage_standards.items():
                supplied_rule = supplied_rules.get(voltage, {})
                if not isinstance(supplied_rule, Mapping):
                    raise ValueError(f"{voltage}kV 标准必须是对象")
                safe_clearance = float(
                    supplied_rule.get(
                        "min_clearance_distance",
                        default_rules["min_clearance_distance"],
                    )
                )
                derived_critical = min(
                    default_rules["critical_clearance_distance"],
                    safe_clearance / 3,
                )
                derived_severe = min(
                    default_rules["severe_clearance_distance"],
                    safe_clearance * 5 / 6,
                )
                merged = {
                    key: float(supplied_rule.get(key, default))
                    for key, default in default_rules.items()
                }
                if "critical_clearance_distance" not in supplied_rule:
                    merged["critical_clearance_distance"] = derived_critical
                if "severe_clearance_distance" not in supplied_rule:
                    merged["severe_clearance_distance"] = derived_severe
                merged_rules[voltage] = merged
            cleaned["voltage_standards"] = merged_rules
        for key in allowed:
            if key in cleaned and isinstance(getattr(defaults, key), (int, float)):
                cleaned[key] = float(cleaned[key])
        result = cls(**cleaned)
        result.validate()
        return result

    def validate(self) -> None:
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, (int, float)):
                if not math.isfinite(float(value)):
                    raise ValueError(f"{key} must be finite")
                if value < 0:
                    raise ValueError(f"{key} 不能小于 0")
        if self.wind_high_speed <= self.wind_warning_speed:
            raise ValueError("高风速阈值必须大于预警风速阈值")
        if self.high_risk_score <= self.medium_risk_score:
            raise ValueError("高风险分数阈值必须大于中风险分数阈值")
        if self.high_risk_score > 100:
            raise ValueError("高风险分数阈值不能超过 100")
        if not (
            0
            < self.general_hazard_index
            < self.major_hazard_index
            < self.emergency_hazard_index
            <= 1
        ):
            raise ValueError("四级风险指数阈值必须满足 0 < 一般 < 重大 < 紧急 <= 1")
        weight_groups = {
            "综合风险 D/G/M/H": (
                self.risk_weight_distance,
                self.risk_weight_growth,
                self.risk_weight_weather,
                self.risk_weight_history,
            ),
            "气象因素": (
                self.weather_weight_wind,
                self.weather_weight_rainfall,
                self.weather_weight_temperature,
                self.weather_weight_sunlight,
            ),
            "历史因素": (
                self.history_weight_exceedance,
                self.history_weight_incident,
                self.history_weight_repeat,
                self.history_weight_treatment_recurrence,
            ),
            "图片示例特征注意力": tuple(self.feature_attention_weights.values()),
        }
        for label, weights in weight_groups.items():
            if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"{label}权重之和必须等于 1")
        for voltage, rules in self.voltage_standards.items():
            try:
                float(voltage)
            except ValueError as exc:
                raise ValueError(f"无效电压等级: {voltage}") from exc
            required = (
                "min_vertical_distance",
                "min_horizontal_distance",
                "min_wind_offset_distance",
                "min_clearance_distance",
                "max_sag",
                "critical_clearance_distance",
                "severe_clearance_distance",
            )
            for key in required:
                if (
                    key not in rules
                    or not math.isfinite(float(rules[key]))
                    or float(rules[key]) < 0
                ):
                    raise ValueError(f"{voltage}kV 的 {key} 配置无效")
            critical = float(rules["critical_clearance_distance"])
            severe = float(rules["severe_clearance_distance"])
            safe = float(rules["min_clearance_distance"])
            if not 0 <= critical < severe < safe:
                raise ValueError(f"{voltage}kV 净空阈值必须满足 0 ≤ 危急 < 严重 < 安全")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def rules_for_voltage(self, voltage_level_kv: float) -> dict[str, float]:
        if not self.voltage_standards:
            return {
                "min_vertical_distance": self.min_vertical_distance,
                "min_horizontal_distance": self.min_horizontal_distance,
                "min_wind_offset_distance": self.min_clearance_distance,
                "min_clearance_distance": self.min_clearance_distance,
                "max_sag": float("inf"),
                "critical_clearance_distance": 1.0,
                "severe_clearance_distance": 2.5,
            }
        closest = min(
            self.voltage_standards,
            key=lambda value: abs(float(value) - voltage_level_kv),
        )
        rules = {key: float(value) for key, value in self.voltage_standards[closest].items()}
        rules["matched_voltage_level_kv"] = float(closest)
        return rules


def classify_clearance_defect(
    *,
    horizontal_distance: float,
    vertical_distance: float,
    clearance_distance: float,
    standards: PredictionStandards,
    voltage_level_kv: float,
) -> dict[str, Any]:
    """Classify clearance defects using configurable voltage-specific thresholds.

    A component distance at or below the critical threshold is an immediate
    override. Otherwise the three-dimensional/effective clearance determines
    the interval, matching the examples supplied with the project.
    """

    rules = standards.rules_for_voltage(voltage_level_kv)
    horizontal = max(0.0, float(horizontal_distance))
    vertical = max(0.0, float(vertical_distance))
    clearance = max(0.0, float(clearance_distance))
    critical = rules["critical_clearance_distance"]
    severe = rules["severe_clearance_distance"]
    safe = rules["min_clearance_distance"]
    governing_component = min(horizontal, vertical, clearance)

    if governing_component <= critical:
        code, level, color = "critical", "危急缺陷", "红色"
        interval = f"任一距离 ≤ {critical:g}m"
        basis = "水平、垂直或净空距离触发危急阈值"
    elif clearance <= severe:
        code, level, color = "severe", "严重缺陷", "橙色"
        interval = f"{critical:g}m < 净空距离 ≤ {severe:g}m"
        basis = "净空距离进入严重缺陷区间"
    elif clearance < safe:
        code, level, color = "general", "一般缺陷", "黄色"
        interval = f"{severe:g}m < 净空距离 < {safe:g}m"
        basis = "净空距离低于安全阈值"
    else:
        code, level, color = "safe", "安全", "绿色"
        interval = f"净空距离 ≥ {safe:g}m"
        basis = "净空距离达到安全阈值且未触发危急覆盖条件"

    return {
        "code": code,
        "level": level,
        "color": color,
        "interval": interval,
        "basis": basis,
        "horizontal_distance": round(horizontal, 3),
        "vertical_distance": round(vertical, 3),
        "clearance_distance": round(clearance, 3),
        "governing_component_distance": round(governing_component, 3),
        "thresholds": {
            "critical_max": critical,
            "severe_max": severe,
            "safe_min": safe,
            "matched_voltage_level_kv": rules["matched_voltage_level_kv"],
        },
        "standard_reference": standards.standard_reference,
    }


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
    if voltage_level_kv is not None:
        wind_offset_clearance = max(
            0.0,
            math.hypot(float(vertical_distance), float(horizontal_distance))
            - float(wind_offset_distance),
        )
        checks["wind_offset_clearance"] = {
            "actual": wind_offset_clearance,
            "standard": voltage_rules["min_wind_offset_distance"],
            "compliant": wind_offset_clearance
            >= voltage_rules["min_wind_offset_distance"],
            "label": "最大风偏后净距",
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
    current_sag = maximum_sag_winter + (maximum_sag_summer - maximum_sag_winter) * summer_factor
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
