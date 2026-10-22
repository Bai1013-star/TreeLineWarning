"""Rule-based single-tree risk predictor with configurable standards."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from algorithms.prediction.species import get_species_factor
from algorithms.prediction.standards import (
    PredictionStandards,
    analyze_distances,
    analyze_wire,
    classify_clearance_defect,
)
from algorithms.prediction.tree_factor import calculate_tree_factor


@dataclass
class TreeInput:
    species: str
    height: float
    age: int
    diameter: float
    wind_speed: float
    rainfall: float
    humidity: float
    slope: float
    temperature: float = 0
    sunlight_hours: float = 0
    distance_to_wire: float = 999
    vertical_distance: float = 999
    horizontal_distance: float = 999
    clearance_distance: float = 999
    wind_offset_distance: float = 0
    voltage_level_kv: float = 110
    maximum_sag_winter: float = 3.5
    maximum_sag_summer: float = 5.0
    observation_month: int = 7
    # Relative conductor displacement from the actually observed point-cloud
    # geometry.  Zero means the supplied distances already include the
    # observation-date conductor position.
    sag_adjustment: float = 0.0
    standards: dict[str, Any] | None = None


@dataclass
class PredictionResult:
    risk_level: str
    risk_score: float
    suggestion: str
    standard_profile: dict[str, Any] | None = None
    standard_analysis: dict[str, Any] | None = None


class TreeRiskPredictor:
    def predict(self, tree: TreeInput) -> PredictionResult:
        self._validate_input(tree)
        standard = PredictionStandards.from_mapping(tree.standards)
        score = 0.0

        # 气象条件
        if tree.wind_speed > standard.wind_high_speed:
            score += 30
        elif tree.wind_speed > standard.wind_warning_speed:
            score += 15
        if tree.rainfall > standard.rainfall_warning:
            score += 20
        if tree.temperature >= standard.temperature_growth_threshold:
            score += 5
        if tree.sunlight_hours >= standard.sunlight_growth_threshold:
            score += 5
        if tree.slope > 25:
            score += 20

        # 树木本身
        score += calculate_tree_factor(tree.height, tree.age, tree.diameter)

        # 树线关系；旧 distance_to_wire 自动映射为净空距离。
        clearance = (
            tree.clearance_distance if tree.clearance_distance < 999 else tree.distance_to_wire
        )
        vertical = tree.vertical_distance if tree.vertical_distance < 999 else clearance
        horizontal = tree.horizontal_distance if tree.horizontal_distance < 999 else clearance
        wire_analysis = analyze_wire(
            voltage_level_kv=tree.voltage_level_kv,
            maximum_sag_winter=tree.maximum_sag_winter,
            maximum_sag_summer=tree.maximum_sag_summer,
            observation_month=tree.observation_month,
            standards=standard,
        )
        # Point-cloud distances already contain the conductor position on the
        # observation date.  Only a forecast-relative displacement may be
        # subtracted; applying (maximum-current) here double-counted sag and
        # incorrectly made winter more dangerous than summer.
        sag_adjustment = max(0.0, float(tree.sag_adjustment))
        adjusted_vertical = max(0.0, vertical - sag_adjustment)
        adjusted_clearance = max(0.0, clearance - sag_adjustment)
        geometric_clearance = max(
            0.0,
            math.hypot(adjusted_vertical, horizontal) - tree.wind_offset_distance,
        )
        effective_clearance = min(adjusted_clearance, geometric_clearance)
        voltage_rules = standard.rules_for_voltage(tree.voltage_level_kv)
        minimum = voltage_rules["min_clearance_distance"]
        if effective_clearance <= minimum:
            score += 40
        elif effective_clearance <= minimum * 2:
            score += 25
        elif effective_clearance <= minimum * 3:
            score += 10

        distance_analysis = analyze_distances(
            vertical_distance=adjusted_vertical,
            horizontal_distance=horizontal,
            clearance_distance=adjusted_clearance,
            wind_offset_distance=tree.wind_offset_distance,
            standards=standard,
            voltage_level_kv=tree.voltage_level_kv,
        )
        clearance_defect = classify_clearance_defect(
            horizontal_distance=horizontal,
            vertical_distance=adjusted_vertical,
            clearance_distance=effective_clearance,
            standards=standard,
            voltage_level_kv=tree.voltage_level_kv,
        )
        # 同样净空下，多项不合规会提高紧迫度，但不重复计满分。
        score += min(15, len(distance_analysis["violations"]) * 5)
        if not wire_analysis["compliant"]:
            score += 15

        score *= get_species_factor(tree.species)
        score = min(100.0, score)

        if score >= standard.high_risk_score:
            level = "高风险"
            suggestion = "建议立即巡检，并按当前地区/行业标准处理树线净空"
        elif score >= standard.medium_risk_score:
            level = "中风险"
            suggestion = "建议加强监测，并缩短树线距离复测周期"
        else:
            level = "低风险"
            suggestion = "符合当前风险阈值，建议按计划维护"

        analysis = {
            **distance_analysis,
            "effective_clearance": round(effective_clearance, 3),
            "wire": wire_analysis,
            "sag_adjustment": round(sag_adjustment, 3),
            "measured_distances": {
                "vertical_distance": vertical,
                "horizontal_distance": horizontal,
                "clearance_distance": clearance,
                "wind_offset_distance": tree.wind_offset_distance,
            },
            "clearance_defect": clearance_defect,
            "weather_checks": {
                "wind_speed": {
                    "actual": tree.wind_speed,
                    "warning": standard.wind_warning_speed,
                    "high": standard.wind_high_speed,
                },
                "rainfall": {
                    "actual": tree.rainfall,
                    "warning": standard.rainfall_warning,
                },
            },
        }
        return PredictionResult(
            risk_level=level,
            risk_score=round(score, 2),
            suggestion=suggestion,
            standard_profile={
                "profile_name": standard.profile_name,
                "region": standard.region,
                "industry": standard.industry,
            },
            standard_analysis=analysis,
        )

    @staticmethod
    def _validate_input(tree: TreeInput) -> None:
        numeric_fields = {
            "height": tree.height,
            "age": tree.age,
            "diameter": tree.diameter,
            "wind_speed": tree.wind_speed,
            "rainfall": tree.rainfall,
            "humidity": tree.humidity,
            "slope": tree.slope,
            "temperature": tree.temperature,
            "sunlight_hours": tree.sunlight_hours,
            "distance_to_wire": tree.distance_to_wire,
            "vertical_distance": tree.vertical_distance,
            "horizontal_distance": tree.horizontal_distance,
            "clearance_distance": tree.clearance_distance,
            "wind_offset_distance": tree.wind_offset_distance,
            "voltage_level_kv": tree.voltage_level_kv,
            "maximum_sag_winter": tree.maximum_sag_winter,
            "maximum_sag_summer": tree.maximum_sag_summer,
            "observation_month": tree.observation_month,
        }
        for name, value in numeric_fields.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} 必须是有限数值")
        if tree.height <= 0 or tree.diameter <= 0:
            raise ValueError("树高和胸径必须大于 0")
        for name in (
            "age",
            "wind_speed",
            "rainfall",
            "slope",
            "sunlight_hours",
            "distance_to_wire",
            "vertical_distance",
            "horizontal_distance",
            "clearance_distance",
            "wind_offset_distance",
            "maximum_sag_winter",
            "maximum_sag_summer",
        ):
            if float(numeric_fields[name]) < 0:
                raise ValueError(f"{name} 不能小于 0")
        if not 0 <= tree.humidity <= 100:
            raise ValueError("湿度必须在 0–100% 之间")
        if not -60 <= tree.temperature <= 60:
            raise ValueError("温度必须在 -60–60℃ 之间")
        if tree.sunlight_hours > 24:
            raise ValueError("日照时长不能超过 24 小时")
        if tree.voltage_level_kv <= 0:
            raise ValueError("电压等级必须大于 0")
        if not 1 <= int(tree.observation_month) <= 12:
            raise ValueError("观测月份必须在 1–12 之间")
