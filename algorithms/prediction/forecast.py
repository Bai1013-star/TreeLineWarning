"""Hybrid short- and long-horizon tree-line risk forecasting."""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import date, datetime, timedelta
from typing import Any

from algorithms.prediction.predictor import TreeInput, TreeRiskPredictor
from algorithms.prediction.species_growth import SpeciesGrowthModel
from algorithms.prediction.standards import PredictionStandards, analyze_wire
from algorithms.prediction.temporal_attention import (
    TemporalPredictor,
    risk_from_score,
)

SPECIES_ANNUAL_GROWTH = {
    "松树": (0.45, 0.75),
    "杉树": (0.50, 0.80),
    "杨树": (1.20, 1.35),
    "柳树": (1.00, 1.20),
    "柏树": (0.32, 0.60),
    "樟树": (0.75, 1.00),
    "其他": (0.55, 0.85),
}


def season_for_month(month: int) -> str:
    if month in (3, 4, 5):
        return "春季"
    if month in (6, 7, 8):
        return "夏季"
    if month in (9, 10, 11):
        return "秋季"
    return "冬季"


def add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(
        value.day,
        (
            date(year + (month == 12), 1 if month == 12 else month + 1, 1)
            - timedelta(days=1)
        ).day,
    )
    return date(year, month, day)


def observation_date_from(value: dict[str, Any]) -> date:
    """Resolve the observation date while keeping older month-only inputs valid."""
    raw_date = value.get("observation_date")
    if raw_date:
        try:
            return date.fromisoformat(str(raw_date)[:10])
        except ValueError as exc:
            raise ValueError("观测日期必须使用 YYYY-MM-DD 格式") from exc

    month = int(value.get("observation_month", 0) or 0)
    year = int(value.get("observation_year", 0) or 0)
    if not 1 <= month <= 12:
        created_at = value.get("created_at")
        if created_at:
            try:
                return date.fromisoformat(str(created_at)[:10])
            except ValueError:
                pass
        month = date.today().month
    if year < 1900 or year > 2200:
        year = date.today().year
    day = int(value.get("observation_day", 1) or 1)
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise ValueError("观测年、月、日组合不是有效日期") from exc


class ForecastService:
    def __init__(
        self,
        temporal_predictor: TemporalPredictor | None = None,
        growth_model: SpeciesGrowthModel | None = None,
        *,
        temporal_model_path: str | None = None,
        growth_model_path: str | None = None,
    ):
        self.rule_predictor = TreeRiskPredictor()
        self.temporal_predictor = temporal_predictor
        self.growth_model = growth_model
        self.temporal_model_path = temporal_model_path
        self.growth_model_path = growth_model_path

    def _growth_model_for_prediction(self) -> SpeciesGrowthModel:
        if self.growth_model is None:
            self.growth_model = SpeciesGrowthModel(self.growth_model_path)
        return self.growth_model

    def _temporal_predictor_for_prediction(self) -> TemporalPredictor:
        if self.temporal_predictor is None:
            self.temporal_predictor = TemporalPredictor(self.temporal_model_path)
        return self.temporal_predictor

    def forecast(
        self,
        snapshot: dict[str, Any],
        standards: dict[str, Any],
        short_horizons_days: list[int] | None = None,
        long_horizons_months: list[int] | None = None,
        historical_records: list[dict[str, Any]] | None = None,
        selected_horizon_months: int | None = None,
    ) -> dict[str, Any]:
        standard = PredictionStandards.from_mapping(standards)
        short_days = sorted(set(short_horizons_days or [1, 7, 14, 30]))
        long_months = sorted(set(long_horizons_months or [1, 2, 3, 6, 12]))
        if any(value <= 0 or value > 90 for value in short_days):
            raise ValueError("短时预测范围必须在 1–90 天")
        if any(value <= 0 or value > 36 for value in long_months):
            raise ValueError("长时预测范围必须在 1–36 个月")
        selected_month = (
            int(selected_horizon_months)
            if selected_horizon_months is not None
            else long_months[-1]
        )
        if selected_month not in long_months:
            raise ValueError("重点分析月份必须包含在长时预测月份中")

        current = self._normalize_snapshot(snapshot)
        self._validate_snapshot(current)
        current_rule = self._rule_result(current, standard)
        short_term = [
            self._short_projection(current, days, standard) for days in short_days
        ]

        usable_history = self._normalize_history(historical_records or [], current)
        monthly_records = [*usable_history[-17:], current]
        long_term = []
        for month_index in range(1, max(long_months) + 1):
            monthly_records.append(
                self._next_month(monthly_records[-1], current, month_index)
            )
            if month_index in long_months:
                long_term.append(
                    self._long_projection(
                        monthly_records,
                        month_index,
                        standard,
                    )
                )

        target = next(
            item for item in long_term if item["horizon_months"] == selected_month
        )
        height_delta = target["predicted_height"] - current["height"]
        clearance_delta = target["predicted_clearance_distance"] - current[
            "clearance_distance"
        ]
        risk_delta = target["risk_score"] - current_rule["risk_score"]
        risk_timeline = self._risk_timeline_analysis(
            current,
            current_rule,
            long_term,
            standard,
        )
        all_assessments = [current_rule, *short_term, *long_term]
        maximum_score = max(float(item["risk_score"]) for item in all_assessments)
        urgency, urgency_icon = self._urgency(maximum_score, standard)
        now = datetime.now()
        quality_warnings = self._data_quality_warnings(current)
        work_order_id = (
            f"TLW-{now:%Y%m%d}-{current['tree_id']:03d}-{now:%H%M%S}"
        )
        return {
            "report_type": "配电线路树线隐患智能预警工单",
            "work_order": {
                "work_order_id": work_order_id,
                "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "urgency": urgency,
                "urgency_icon": urgency_icon,
                "responsible_unit": str(
                    snapshot.get("responsible_unit", "请立即派单至运维班组")
                ),
                "system_signature": "TreeLineWarning 树线隐患智能研判系统",
            },
            "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "calendar_context": {
                "system_current_date": date.today().isoformat(),
                "system_current_month": date.today().month,
                "system_current_season": season_for_month(date.today().month),
                "observation_date": current["observation_date"],
                "observation_year": current["observation_year"],
                "observation_month": current["observation_month"],
                "observation_season": season_for_month(
                    current["observation_month"]
                ),
                # Compatibility aliases: "current" here means current observation.
                "current_date": current["observation_date"],
                "current_month": current["observation_month"],
                "current_season": season_for_month(current["observation_month"]),
                "selected_target_date": target["target_date"],
                "selected_target_year_month": target["target_year_month"],
                "selected_horizon_months": selected_month,
                "selected_target_season": target["season"],
                "risk_scan_months": max(long_months),
            },
            "location_information": {
                "line_name": current["line_name"],
                "tower_section": current["tower_section"],
                "gps_longitude": current["gps_longitude"],
                "gps_latitude": current["gps_latitude"],
                "tree_code": current["tree_code"],
                "species": current["species"],
                "height": current["height"],
                "crown_width": current["crown_width"],
            },
            "input_snapshot": current,
            "applied_standard": standard.to_dict(),
            "current_assessment": current_rule,
            "short_term_forecast": short_term,
            "long_term_forecast": long_term,
            "selected_target_prediction": target,
            "risk_timeline_analysis": risk_timeline,
            "historical_context": {
                "status": "已启用" if len(usable_history) >= 2 else "等待用户提供历史数据",
                "record_count": len(usable_history),
                "effect": (
                    "历史观测已按时间顺序进入 LSTM，近期记录权重高于早期记录。"
                    if len(usable_history) >= 2
                    else "当前以模拟训练、当月状态和季节规律预测；录入至少两期同树历史后自动融合。"
                ),
            },
            "model_context": {
                "risk_model": (
                    "LSTM 三重注意力（近期、季节、树种特征）+ 行业标准规则"
                    if self.temporal_predictor is not None
                    else "季节生长 + 行业标准规则"
                ),
                "growth_model": (
                    "分树种月度气象生长模型（模拟数据训练）"
                    if self.growth_model is not None
                    else "分树种季节规律回退模型"
                ),
            },
            "data_quality": {
                "warnings": quality_warnings,
                "status": "需复核" if quality_warnings else "基础校验通过",
            },
            "growth_trend": {
                "forecast_months": selected_month,
                "height_change": round(height_delta, 3),
                "diameter_change": round(
                    target["predicted_diameter"] - current["diameter"], 3
                ),
                "clearance_change": round(clearance_delta, 3),
                "risk_score_change": round(risk_delta, 2),
                "direction": (
                    "风险上升"
                    if risk_delta > 3
                    else "风险下降"
                    if risk_delta < -3
                    else "风险基本稳定"
                ),
            },
            "treatment_advice": self._treatment_advice(
                current_rule,
                target,
                standard,
                current,
                risk_timeline,
            ),
            "attachments": [
                "三维点云截图（接入现场文件后附加）",
                "树线距离变化曲线（当前状态 + 月度预测）",
                "最佳修剪点位示意图（接入现场坐标后生成）",
            ],
            "report_notes": [
                "短时预测按日/周评估，树木生长影响较小，主要反映气象和弧垂工况。",
                    "长时预测逐月累计树种、温度、日照、降雨、湿度与季节性生长，"
                    "并融合 LSTM 时序评分与标准规则评分。",
                "所有短时日期和未来月份均从观测日期起算，而不是从工单生成日期起算。",
                "重点月份用于展示选定月份结果；风险上升月与修剪时点按完整风险扫描月份自动计算。",
                "未提供未来逐日天气预报时，短时预测沿用观测时气象作为情景假设。",
                "当前树种生长模型使用现实规律约束的模拟数据；收到真实历史数据后应重训校准。",
                "预测值用于预警和巡检排序，真实工程处置仍需现场复测。",
            ],
        }

    def _normalize_snapshot(self, value: dict[str, Any]) -> dict[str, Any]:
        observed_on = observation_date_from(value)
        clearance = float(
            value.get("clearance_distance", value.get("distance_to_wire", 999))
        )
        result = {
            "tree_id": int(value.get("tree_id", 0)),
            "tree_code": str(value.get("tree_code", value.get("tree_id", ""))),
            "species": str(value.get("species", "其他")),
            "height": float(value.get("height", 0)),
            "crown_width": float(value.get("crown_width", 0)),
            "age": float(value.get("age", 0)),
            "diameter": float(value.get("diameter", 0)),
            "slope": float(value.get("slope", 0)),
            "vertical_distance": float(value.get("vertical_distance", clearance)),
            "horizontal_distance": float(value.get("horizontal_distance", clearance)),
            "clearance_distance": clearance,
            "wind_offset_distance": float(value.get("wind_offset_distance", 0)),
            "voltage_level_kv": float(value.get("voltage_level_kv", 110)),
            "maximum_sag_winter": float(value.get("maximum_sag_winter", 3.5)),
            "maximum_sag_summer": float(value.get("maximum_sag_summer", 5)),
            "observation_date": observed_on.isoformat(),
            "observation_year": observed_on.year,
            "observation_month": observed_on.month,
            "observation_day": observed_on.day,
            "wind_speed": float(value.get("wind_speed", 0)),
            "rainfall": float(value.get("rainfall", 0)),
            "humidity": float(value.get("humidity", 0)),
            "temperature": float(value.get("temperature", 20)),
            "sunlight_hours": float(value.get("sunlight_hours", 8)),
            "line_name": str(value.get("line_name", "未填写线路")),
            "tower_section": str(value.get("tower_section", "未填写杆塔区间")),
            "gps_longitude": str(value.get("gps_longitude", "未填写")),
            "gps_latitude": str(value.get("gps_latitude", "未填写")),
        }
        result["distance_to_wire"] = result["clearance_distance"]
        result["month"] = result["observation_month"]
        return result

    @staticmethod
    def _validate_snapshot(value: dict[str, Any]) -> None:
        if value["height"] <= 0:
            raise ValueError("树高必须大于 0")
        if value["diameter"] <= 0:
            raise ValueError("胸径必须大于 0")
        if value["age"] < 0:
            raise ValueError("树龄不能小于 0")
        for field, label in (
            ("vertical_distance", "垂直距离"),
            ("horizontal_distance", "水平距离"),
            ("clearance_distance", "净空距离"),
            ("wind_offset_distance", "风偏距离"),
            ("maximum_sag_winter", "冬季最大弧垂"),
            ("maximum_sag_summer", "夏季最大弧垂"),
            ("wind_speed", "风速"),
            ("rainfall", "降雨量"),
            ("sunlight_hours", "日照时长"),
        ):
            if value[field] < 0:
                raise ValueError(f"{label}不能小于 0")
        if not 0 <= value["humidity"] <= 100:
            raise ValueError("湿度必须在 0–100% 之间")
        if not -60 <= value["temperature"] <= 60:
            raise ValueError("温度必须在 -60–60℃ 之间")
        if value["sunlight_hours"] > 24:
            raise ValueError("日照时长不能超过 24 小时")

    @staticmethod
    def _data_quality_warnings(value: dict[str, Any]) -> list[str]:
        warnings = []
        if date.fromisoformat(value["observation_date"]) > date.today():
            warnings.append("观测日期晚于系统当前日期，请确认是否为情景模拟")
        if value["maximum_sag_summer"] < value["maximum_sag_winter"]:
            warnings.append("夏季最大弧垂小于冬季最大弧垂，请复核测量口径")
        if value["crown_width"] <= 0:
            warnings.append("未填写有效冠幅；当前不影响评分，但工单信息不完整")
        if value["gps_longitude"] == "未填写" or value["gps_latitude"] == "未填写":
            warnings.append("GPS 坐标不完整，工单无法准确定位")
        return warnings

    def _growth_for_month(self, state: dict[str, Any], month: int) -> tuple[float, float]:
        return self._growth_model_for_prediction().predict(
            state["species"],
            month=month,
            temperature=state["temperature"],
            sunlight_hours=state["sunlight_hours"],
            rainfall=state["rainfall"],
            humidity=state["humidity"],
            age=state["age"],
            height=state["height"],
        )

    def _normalize_history(
        self,
        records: list[dict[str, Any]],
        current: dict[str, Any],
    ) -> list[dict[str, Any]]:
        by_date: dict[str, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            if current["tree_id"] and int(record.get("tree_id", 0)) != current["tree_id"]:
                continue
            try:
                merged = {**current, **record}
                if record.get("observation_month") and not record.get(
                    "observation_date"
                ):
                    merged.pop("observation_date", None)
                    merged["observation_year"] = int(
                        record.get(
                            "observation_year", current["observation_year"]
                        )
                        or current["observation_year"]
                    )
                    merged["observation_day"] = int(
                        record.get("observation_day", 1) or 1
                    )
                normalized = self._normalize_snapshot(merged)
            except (TypeError, ValueError):
                continue
            if normalized["height"] <= 0 or normalized["clearance_distance"] <= 0:
                continue
            if normalized["observation_date"] >= current["observation_date"]:
                continue
            by_date[normalized["observation_date"]] = normalized
        return [by_date[key] for key in sorted(by_date)]

    def _short_projection(
        self,
        current: dict[str, Any],
        days: int,
        standard: PredictionStandards,
    ) -> dict[str, Any]:
        state = dict(current)
        observed_on = date.fromisoformat(current["observation_date"])
        target_on = observed_on + timedelta(days=days)
        total_height_growth = 0.0
        total_diameter_growth = 0.0
        for day_index in range(1, days + 1):
            projected_day = observed_on + timedelta(days=day_index)
            state["observation_month"] = projected_day.month
            state["month"] = projected_day.month
            state["age"] = current["age"] + day_index / 365.2425
            monthly_height, monthly_diameter = self._growth_for_month(
                state, projected_day.month
            )
            days_in_month = (
                add_months(projected_day.replace(day=1), 1)
                - projected_day.replace(day=1)
            ).days
            total_height_growth += monthly_height / days_in_month
            total_diameter_growth += monthly_diameter / days_in_month
        state["height"] += total_height_growth
        state["diameter"] += total_diameter_growth
        encroachment = total_height_growth * 0.45
        state["vertical_distance"] = max(0, state["vertical_distance"] - encroachment)
        state["clearance_distance"] = max(0, state["clearance_distance"] - encroachment)
        state["distance_to_wire"] = state["clearance_distance"]
        state["observation_date"] = target_on.isoformat()
        state["observation_year"] = target_on.year
        state["observation_month"] = target_on.month
        state["observation_day"] = target_on.day
        state["month"] = target_on.month
        result = self._rule_result(state, standard)
        wire = result["standard_analysis"]["wire"]
        return {
            "horizon_type": "短时",
            "period": f"{days}天",
            "target_date": target_on.isoformat(),
            "target_year_month": target_on.strftime("%Y年%m月"),
            "prediction_method": "从观测日起算的日尺度规则与弧垂工况",
            "risk_level": result["risk_level"],
            "risk_score": result["risk_score"],
            "predicted_height": round(state["height"], 3),
            "predicted_age": round(state["age"], 3),
            "predicted_diameter": round(state["diameter"], 3),
            "predicted_vertical_distance": round(state["vertical_distance"], 3),
            "predicted_clearance_distance": round(state["clearance_distance"], 3),
            "predicted_effective_clearance": round(
                float(
                    result.get("standard_analysis", {}).get(
                        "effective_clearance",
                        state["clearance_distance"],
                    )
                ),
                3,
            ),
            "predicted_sag": wire["current_sag"],
            "height_growth": round(total_height_growth, 4),
            "predicted_temperature": round(state["temperature"], 2),
            "predicted_sunlight_hours": round(state["sunlight_hours"], 2),
            "predicted_rainfall": round(state["rainfall"], 2),
            "predicted_humidity": round(state["humidity"], 2),
            "predicted_wind_speed": round(state["wind_speed"], 2),
            "season": season_for_month(state["observation_month"]),
            "wire_condition": wire["season"],
            "suggestion": result["suggestion"],
        }

    def _next_month(
        self,
        previous: dict[str, Any],
        baseline: dict[str, Any],
        month_index: int,
    ) -> dict[str, Any]:
        state = dict(previous)
        target_on = add_months(
            date.fromisoformat(baseline["observation_date"]), month_index
        )
        month = target_on.month
        baseline_summer = math.cos(
            2 * math.pi * (baseline["observation_month"] - 7) / 12
        )
        next_summer = math.cos(2 * math.pi * (month - 7) / 12)
        state["temperature"] = baseline["temperature"] + 8 * (
            next_summer - baseline_summer
        )
        state["sunlight_hours"] = max(
            1,
            baseline["sunlight_hours"] + 2.5 * (next_summer - baseline_summer),
        )
        baseline_rain_season = math.cos(
            2 * math.pi * (baseline["observation_month"] - 7.5) / 12
        )
        next_rain_season = math.cos(2 * math.pi * (month - 7.5) / 12)
        previous_rainfall = state["rainfall"]
        state["rainfall"] = max(
            0,
            baseline["rainfall"]
            + 24 * (next_rain_season - baseline_rain_season),
        )
        state["humidity"] = min(
            100,
            max(
                15,
                state["humidity"]
                + 0.16 * (state["rainfall"] - previous_rainfall)
                - 0.20 * (state["temperature"] - baseline["temperature"]),
            ),
        )
        state["observation_month"] = month
        state["observation_year"] = target_on.year
        state["observation_day"] = target_on.day
        state["observation_date"] = target_on.isoformat()
        state["month"] = month
        state["age"] = baseline["age"] + month_index / 12
        height_growth, diameter_growth = self._growth_for_month(state, month)
        state["height"] += height_growth
        state["diameter"] += diameter_growth
        encroachment = height_growth * 0.45
        state["vertical_distance"] = max(0, state["vertical_distance"] - encroachment)
        state["horizontal_distance"] = max(
            0, state["horizontal_distance"] - height_growth * 0.16
        )
        state["clearance_distance"] = max(
            0, state["clearance_distance"] - encroachment
        )
        state["distance_to_wire"] = state["clearance_distance"]
        state["_height_growth"] = height_growth
        state["_diameter_growth"] = diameter_growth
        state["_target_date"] = target_on.isoformat()
        return state

    def _long_projection(
        self,
        records: list[dict[str, Any]],
        months: int,
        standard: PredictionStandards,
    ) -> dict[str, Any]:
        state = records[-1]
        rule = self._rule_result(state, standard)
        lstm = self._temporal_predictor_for_prediction().predict(
            records, state["species"]
        )
        lstm_score = float(lstm["risk_score"])
        combined_score = (
            float(rule["risk_score"])
            if lstm_score is None
            else 0.55 * lstm_score + 0.45 * float(rule["risk_score"])
        )
        level, suggestion = risk_from_score(
            combined_score,
            standard.medium_risk_score,
            standard.high_risk_score,
        )
        wire = analyze_wire(
            voltage_level_kv=state["voltage_level_kv"],
            maximum_sag_winter=state["maximum_sag_winter"],
            maximum_sag_summer=state["maximum_sag_summer"],
            observation_month=state["observation_month"],
            standards=standard,
        )
        target_on = date.fromisoformat(state["_target_date"])
        return {
            "horizon_type": "长时",
            "period": f"{months}个月",
            "horizon_months": months,
            "target_date": target_on.isoformat(),
            "target_year_month": target_on.strftime("%Y年%m月"),
            "prediction_method": (
                "LSTM时序 + 季节生长 + 标准规则"
                if lstm_score is not None
                else "季节生长 + 标准规则"
            ),
            "risk_level": level,
            "risk_score": round(combined_score, 2),
            "rule_risk_score": rule["risk_score"],
            "lstm_risk_score": round(lstm_score, 2) if lstm_score is not None else None,
            "predicted_height": round(state["height"], 3),
            "predicted_age": round(state["age"], 3),
            "predicted_diameter": round(state["diameter"], 3),
            "predicted_vertical_distance": round(state["vertical_distance"], 3),
            "predicted_horizontal_distance": round(state["horizontal_distance"], 3),
            "predicted_clearance_distance": round(state["clearance_distance"], 3),
            "predicted_effective_clearance": round(
                float(
                    rule.get("standard_analysis", {}).get(
                        "effective_clearance",
                        state["clearance_distance"],
                    )
                ),
                3,
            ),
            "predicted_wind_offset_distance": round(state["wind_offset_distance"], 3),
            "predicted_sag": wire["current_sag"],
            "monthly_height_growth": round(float(state.get("_height_growth", 0)), 4),
            "monthly_diameter_growth": round(
                float(state.get("_diameter_growth", 0)), 4
            ),
            "predicted_temperature": round(state["temperature"], 2),
            "predicted_sunlight_hours": round(state["sunlight_hours"], 2),
            "predicted_rainfall": round(state["rainfall"], 2),
            "predicted_humidity": round(state["humidity"], 2),
            "predicted_wind_speed": round(state["wind_speed"], 2),
            "season": season_for_month(target_on.month),
            "wire_condition": wire["season"],
            "suggestion": suggestion,
        }

    @staticmethod
    def _urgency(
        maximum_score: float, standard: PredictionStandards
    ) -> tuple[str, str]:
        if maximum_score >= standard.high_risk_score:
            return "最高级（I级危险）", "⚠⚠⚠"
        if maximum_score >= standard.medium_risk_score:
            return "较高级（II级关注）", "⚠⚠"
        return "常规级（III级巡检）", "⚠"

    @staticmethod
    def _risk_timeline_analysis(
        current_snapshot: dict[str, Any],
        current_assessment: dict[str, Any],
        forecasts: list[dict[str, Any]],
        standard: PredictionStandards,
    ) -> dict[str, Any]:
        severity = {"低风险": 0, "中风险": 1, "高风险": 2}
        current_score = float(current_assessment["risk_score"])
        current_level = str(current_assessment["risk_level"])
        previous_score = current_score
        previous_level = current_level
        timeline = []
        for item in forecasts:
            score = float(item["risk_score"])
            level = str(item["risk_level"])
            change_previous = score - previous_score
            change_current = score - current_score
            category_increase = severity.get(level, 0) > severity.get(
                current_level, 0
            )
            significant = category_increase or change_current >= 3
            timeline.append(
                {
                    "horizon_months": int(item["horizon_months"]),
                    "target_date": item["target_date"],
                    "target_year_month": item["target_year_month"],
                    "risk_level": level,
                    "risk_score": round(score, 2),
                    "change_from_previous": round(change_previous, 2),
                    "change_from_current": round(change_current, 2),
                    "trend": (
                        "上升"
                        if change_previous >= 2
                        or severity.get(level, 0) > severity.get(previous_level, 0)
                        else "下降"
                        if change_previous <= -2
                        else "基本稳定"
                    ),
                    "significant_increase": significant,
                }
            )
            previous_score = score
            previous_level = level

        first_increase = next(
            (item for item in timeline if item["significant_increase"]), None
        )
        peak = max(
            timeline,
            key=lambda item: (
                item["risk_score"],
                -item["horizon_months"],
            ),
        )
        first_medium = next(
            (
                item
                for item in timeline
                if item["risk_score"] >= standard.medium_risk_score
            ),
            None,
        )
        first_high = next(
            (
                item
                for item in timeline
                if item["risk_score"] >= standard.high_risk_score
            ),
            None,
        )
        minimum_clearance = standard.rules_for_voltage(
            current_snapshot["voltage_level_kv"]
        )["min_clearance_distance"]
        first_clearance_violation = next(
            (
                item
                for item, forecast in zip(timeline, forecasts, strict=True)
                if float(
                    forecast.get(
                        "predicted_effective_clearance",
                        forecast["predicted_clearance_distance"],
                    )
                )
                < minimum_clearance
            ),
            None,
        )
        current_violations = current_assessment.get(
            "standard_analysis", {}
        ).get("violations", [])
        current_effective_clearance = float(
            current_assessment.get("standard_analysis", {}).get(
                "effective_clearance",
                current_snapshot["clearance_distance"],
            )
        )
        current_clearance_violation = (
            current_effective_clearance < minimum_clearance
            or any(
                item.get("field") == "clearance_distance"
                for item in current_violations
            )
        )

        observed_on = date.fromisoformat(current_snapshot["observation_date"])
        trigger_candidates = []
        if first_medium is not None and current_score < standard.medium_risk_score:
            trigger_candidates.append(
                (first_medium["horizon_months"], "风险达到中风险阈值", first_medium)
            )
        if first_high is not None and current_score < standard.high_risk_score:
            trigger_candidates.append(
                (first_high["horizon_months"], "风险达到高风险阈值", first_high)
            )
        if first_clearance_violation is not None:
            trigger_candidates.append(
                (
                    first_clearance_violation["horizon_months"],
                    f"预测净空低于 {minimum_clearance:g} m 标准",
                    first_clearance_violation,
                )
            )

        if current_score >= standard.high_risk_score or current_clearance_violation:
            maintenance = {
                "status": "立即处置",
                "requires_trimming": True,
                "recommended_trim_date": observed_on.isoformat(),
                "recommended_trim_month": observed_on.strftime("%Y年%m月"),
                "recommended_deadline": "立即安排，最晚7天内完成现场复测和修剪评估",
                "trigger_month": observed_on.strftime("%Y年%m月"),
                "trigger_reason": (
                    "当前净空已经不符合标准"
                    if current_clearance_violation
                    else "当前已达到高风险"
                ),
                "lead_time_months": 0,
            }
        elif current_score >= standard.medium_risk_score:
            maintenance = {
                "status": "近期修剪",
                "requires_trimming": True,
                "recommended_trim_date": observed_on.isoformat(),
                "recommended_trim_month": observed_on.strftime("%Y年%m月"),
                "recommended_deadline": "建议30天内完成定向修剪和净空复测",
                "trigger_month": observed_on.strftime("%Y年%m月"),
                "trigger_reason": "当前已达到中风险，应在进入高风险前处置",
                "lead_time_months": 0,
            }
        elif trigger_candidates:
            trigger_offset, trigger_reason, trigger = min(
                trigger_candidates, key=lambda value: value[0]
            )
            trim_offset = max(0, trigger_offset - 1)
            trim_on = add_months(observed_on, trim_offset)
            maintenance = {
                "status": "计划修剪",
                "requires_trimming": True,
                "recommended_trim_date": trim_on.isoformat(),
                "recommended_trim_month": trim_on.strftime("%Y年%m月"),
                "recommended_deadline": (
                    f"建议最迟在 {trim_on:%Y年%m月} 完成修剪，"
                    f"不要晚于 {trigger['target_year_month']} 风险触发月"
                ),
                "trigger_month": trigger["target_year_month"],
                "trigger_reason": trigger_reason,
                "lead_time_months": trigger_offset - trim_offset,
            }
        else:
            maintenance = {
                "status": "继续监测",
                "requires_trimming": False,
                "recommended_trim_date": None,
                "recommended_trim_month": None,
                "recommended_deadline": (
                    f"未来 {forecasts[-1]['horizon_months']} 个月未达到修剪触发阈值；"
                    f"建议在风险峰值月 {peak['target_year_month']} 前复测"
                ),
                "trigger_month": None,
                "trigger_reason": "预测期内风险和净空均未达到修剪触发阈值",
                "lead_time_months": None,
            }

        risk_conclusion = (
            (
                f"预计 {first_increase['target_year_month']} 风险首次明显增大，"
                f"较观测时上升 {first_increase['change_from_current']:+.2f} 分，"
                f"达到 {first_increase['risk_level']}"
            )
            if first_increase is not None
            else (
                f"未来 {forecasts[-1]['horizon_months']} 个月未发现明显风险上升；"
                f"峰值出现在 {peak['target_year_month']}，为 {peak['risk_score']:.2f} 分"
            )
        )
        return {
            "risk_increase_detected": first_increase is not None,
            "risk_increase_conclusion": risk_conclusion,
            "first_significant_increase": first_increase,
            "peak_risk": peak,
            "first_medium_risk": first_medium,
            "first_high_risk": first_high,
            "first_clearance_violation": first_clearance_violation,
            "minimum_clearance_standard": minimum_clearance,
            "maintenance_decision": maintenance,
            "timeline": timeline,
        }

    @staticmethod
    def _treatment_advice(
        current: dict[str, Any],
        target: dict[str, Any],
        standard: PredictionStandards,
        input_snapshot: dict[str, Any],
        risk_timeline: dict[str, Any],
    ) -> dict[str, Any]:
        violations = current.get("standard_analysis", {}).get("violations", [])
        violated_names = "、".join(item.get("label", "") for item in violations)
        maintenance = risk_timeline["maintenance_decision"]
        if maintenance["status"] == "立即处置":
            action = "立即现场复测，评估修枝、移栽或线路清障"
        elif maintenance["status"] in {"近期修剪", "计划修剪"}:
            action = "缩短巡检周期，安排定向修剪并复核净空"
        else:
            action = "保持常规巡检，持续采集月度树高与净空"
        voltage_minimum = standard.rules_for_voltage(
            input_snapshot["voltage_level_kv"]
        )["min_clearance_distance"]
        return {
            "action": action,
            "deadline": maintenance["recommended_deadline"],
            "trigger": (
                f"{maintenance['trigger_reason']}；当前不合规项：{violated_names}"
                if violated_names
                else maintenance["trigger_reason"]
            ),
            "risk_increase_conclusion": risk_timeline[
                "risk_increase_conclusion"
            ],
            "recommended_trim_month": maintenance[
                "recommended_trim_month"
            ],
            "trigger_month": maintenance["trigger_month"],
            "target_month_goal": (
                f"{target['target_year_month']}前保持净空距离不低于"
                f" {voltage_minimum:g} m"
            ),
        }

    def _rule_result(
        self, state: dict[str, Any], standard: PredictionStandards
    ) -> dict[str, Any]:
        result = self.rule_predictor.predict(
            TreeInput(
                species=state["species"],
                height=state["height"],
                age=state["age"],
                diameter=state["diameter"],
                wind_speed=state["wind_speed"],
                rainfall=state["rainfall"],
                humidity=state["humidity"],
                slope=state["slope"],
                temperature=state["temperature"],
                sunlight_hours=state["sunlight_hours"],
                distance_to_wire=state["clearance_distance"],
                vertical_distance=state["vertical_distance"],
                horizontal_distance=state["horizontal_distance"],
                clearance_distance=state["clearance_distance"],
                wind_offset_distance=state["wind_offset_distance"],
                voltage_level_kv=state["voltage_level_kv"],
                maximum_sag_winter=state["maximum_sag_winter"],
                maximum_sag_summer=state["maximum_sag_summer"],
                observation_month=state["observation_month"],
                standards=standard.to_dict(),
            )
        )
        return asdict(result)
