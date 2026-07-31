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
            tree.clearance_distance
            if tree.clearance_distance < 999
            else tree.distance_to_wire
        )
        vertical = (
            tree.vertical_distance if tree.vertical_distance < 999 else clearance
        )
        horizontal = (
            tree.horizontal_distance if tree.horizontal_distance < 999 else clearance
        )
        wire_analysis = analyze_wire(
            voltage_level_kv=tree.voltage_level_kv,
            maximum_sag_winter=tree.maximum_sag_winter,
            maximum_sag_summer=tree.maximum_sag_summer,
            observation_month=tree.observation_month,
            standards=standard,
        )
        sag_adjustment = max(
            0.0, wire_analysis["maximum_sag"] - wire_analysis["current_sag"]
        )
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
