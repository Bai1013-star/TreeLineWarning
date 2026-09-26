"""Hybrid short- and long-horizon tree-line risk forecasting."""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import date, datetime, timedelta
from typing import Any

from algorithms.prediction.contact_risk import contact_risk, normalize_contact_target
from algorithms.prediction.feature_attention import (
    FEATURE_CATALOG,
    feature_attention_prior,
)
from algorithms.prediction.predictor import TreeInput, TreeRiskPredictor
from algorithms.prediction.risk_evaluation import (
    calculate_historical_risk,
    calculation_ledger,
    clearance_interval,
    comprehensive_risk,
)
from algorithms.prediction.species_growth import SpeciesGrowthModel
from algorithms.prediction.standards import (
    PredictionStandards,
    analyze_wire,
    classify_clearance_defect,
)
from algorithms.prediction.temporal_attention import (
    TemporalPredictor,
    risk_from_score,
)

SPECIES_ANNUAL_GROWTH = {
    "松树": (0.45, 0.75),
    "杉树": (0.50, 0.80),
    "法国梧桐": (3.00, 3.00),
    "杨树": (3.00, 3.00),
    "榆树": (1.00, 2.00),
    "国槐": (0.60, 1.00),
    "白蜡": (0.80, 1.20),
    "柳树": (1.00, 1.20),
    "柏树": (0.32, 0.60),
    "樟树": (0.75, 1.00),
    "其他": (0.55, 0.85),
}

REQUIRED_INPUT_FIELDS = (
    "observation_year",
    "observation_month",
    "observation_day",
    "species",
    "height",
    "voltage_level_kv",
    "clearance_distance",
)

OPTIONAL_INPUT_FIELDS = (
    "line_name",
    "tower_section",
    "gps_longitude",
    "gps_latitude",
    "tree_id",
    "tree_code",
    "age",
    "diameter",
    "crown_width",
    "growth_direction",
    "slope",
    "historical_patrol_distance",
    "crown_expansion_speed",
    "pruning_record",
    "maximum_sag_winter",
    "maximum_sag_summer",
    "span_length",
    "conductor_height",
    "contact_target",
    "line_direction",
    "vertical_distance",
    "horizontal_distance",
    "wind_offset_distance",
    "wind_speed",
    "wind_direction",
    "rainfall",
    "humidity",
    "temperature",
    "sunlight_hours",
)

ALL_INPUT_FIELDS = (*REQUIRED_INPUT_FIELDS, *OPTIONAL_INPUT_FIELDS)
PREDICTION_MODES = {"formula", "ai"}

INPUT_FIELD_LABELS = {
    "observation_year": "观测年份",
    "observation_month": "观测月份",
    "observation_day": "观测日期",
    "line_name": "线路名称",
    "tower_section": "杆塔区间",
    "gps_longitude": "GPS 经度",
    "gps_latitude": "GPS 纬度",
    "tree_id": "树木编号",
    "tree_code": "树木档案编号",
    "species": "树种",
    "height": "树高",
    "age": "树龄",
    "diameter": "胸径",
    "crown_width": "冠幅",
    "growth_direction": "生长方向",
    "slope": "立地坡度",
    "historical_patrol_distance": "历史巡检距离",
    "crown_expansion_speed": "树冠扩展速度",
    "pruning_record": "修剪记录",
    "voltage_level_kv": "电压等级",
    "maximum_sag_winter": "冬季最大弧垂",
    "maximum_sag_summer": "夏季最大弧垂",
    "span_length": "档距",
    "conductor_height": "导线高度",
    "contact_target": "现场接触对象",
    "line_direction": "线路走向",
    "vertical_distance": "垂直距离",
    "horizontal_distance": "水平距离",
    "clearance_distance": "净空距离",
    "wind_offset_distance": "风偏距离",
    "wind_speed": "风速",
    "wind_direction": "风向",
    "rainfall": "降雨量",
    "humidity": "湿度",
    "temperature": "温度",
    "sunlight_hours": "日照时长",
}


def season_for_month(month: int) -> str:
    if month in (3, 4, 5):
        return "春季"
    if month in (6, 7, 8):
        return "夏季"
    if month in (9, 10, 11):
        return "秋季"
    return "冬季"


def season_stage_for_month(month: int) -> str:
    if month in (3, 4, 5):
        return "春季萌发期"
    if month in (6, 7, 8):
        return "夏季旺长期"
    if month in (9, 10, 11):
        return "秋季缓生/休眠过渡期"
    return "冬季休眠期"


def add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(
        value.day,
        (date(year + (month == 12), 1 if month == 12 else month + 1, 1) - timedelta(days=1)).day,
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
        prediction_mode: str = "ai",
    ) -> dict[str, Any]:
        prediction_mode = str(prediction_mode).strip().lower()
        if prediction_mode not in PREDICTION_MODES:
            raise ValueError("prediction_mode 必须是 formula 或 ai")
        standard = PredictionStandards.from_mapping(standards)
        short_days = sorted(set(short_horizons_days or [1, 7, 14, 30]))
        long_months = sorted(set(long_horizons_months or [1, 2, 3, 6, 12]))
        if any(value <= 0 or value > 90 for value in short_days):
            raise ValueError("短时预测范围必须在 1–90 天")
        if any(value <= 0 or value > 36 for value in long_months):
            raise ValueError("长时预测范围必须在 1–36 个月")
        selected_month = (
            int(selected_horizon_months) if selected_horizon_months is not None else long_months[-1]
        )
        if selected_month not in long_months:
            raise ValueError("重点分析月份必须包含在长时预测月份中")

        current = self._normalize_snapshot(snapshot, standard)
        self._validate_snapshot(current)
        current_rule = self._rule_result(current, standard)
        self._attach_contact_risk(current_rule, current, standard)
        usable_history = self._normalize_history(historical_records or [], current)
        historical_risk = calculate_historical_risk(
            usable_history,
            standard,
            tree_id=current["tree_id"],
        )
        current_attention = feature_attention_prior(current, usable_history, standard)
        current_rule["attention_prior"] = current_attention
        current_rule["clearance_defect"] = current_rule.get("standard_analysis", {}).get(
            "clearance_defect"
        )
        short_term = [self._short_projection(current, days, standard) for days in short_days]

        monthly_records = [*usable_history[-17:], current]
        long_term = []
        for month_index in range(1, max(long_months) + 1):
            next_state = self._next_month(monthly_records[-1], current, month_index)
            if prediction_mode == "ai":
                temporal = self._temporal_predictor_for_prediction().predict(
                    [*monthly_records[-17:], next_state], current["species"]
                )
                self._apply_temporal_tree_adjustment(monthly_records, next_state, temporal)
                next_state["_temporal_prediction"] = temporal
            monthly_records.append(next_state)
            if month_index in long_months:
                long_term.append(
                    self._long_projection(
                        monthly_records,
                        month_index,
                        standard,
                        prediction_mode=prediction_mode,
                    )
                )

        baseline_effective_clearance = float(
            current_rule.get("standard_analysis", {}).get(
                "effective_clearance", current["clearance_distance"]
            )
        )
        self._attach_comprehensive_risk(
            current_rule,
            current,
            standard,
            historical_risk,
            horizon_months=0,
            baseline_effective_clearance=baseline_effective_clearance,
        )
        for item in short_term:
            self._attach_comprehensive_risk(
                item,
                current,
                standard,
                historical_risk,
                horizon_months=max(1, math.ceil(int(item["period"][:-1]) / 30)),
                baseline_effective_clearance=baseline_effective_clearance,
            )
        for item in long_term:
            self._attach_comprehensive_risk(
                item,
                current,
                standard,
                historical_risk,
                horizon_months=int(item["horizon_months"]),
                baseline_effective_clearance=baseline_effective_clearance,
            )

        threshold_timing = self._daily_threshold_timing(
            current,
            current_rule,
            standard,
            historical_risk,
            baseline_effective_clearance,
        )

        target = next(item for item in long_term if item["horizon_months"] == selected_month)
        height_delta = target["predicted_height"] - current["height"]
        crown_width_delta = target["predicted_crown_width"] - current["crown_width"]
        clearance_delta = target["predicted_clearance_distance"] - current["clearance_distance"]
        risk_delta = target["risk_score"] - current_rule["risk_score"]
        comprehensive_risk_delta = (
            target["comprehensive_risk_score"] - current_rule["comprehensive_risk_score"]
        )
        risk_timeline = self._risk_timeline_analysis(
            current,
            current_rule,
            long_term,
            standard,
            threshold_timing=threshold_timing,
        )
        all_assessments = [current_rule, *short_term, *long_term]
        maximum_score = max(float(item["risk_score"]) for item in all_assessments)
        urgency, urgency_icon = self._urgency(maximum_score, standard)
        comprehensive_maximum_score = max(
            float(item["comprehensive_risk_score"]) for item in all_assessments
        )
        comprehensive_urgency, comprehensive_urgency_icon = self._comprehensive_urgency(
            comprehensive_maximum_score, standard
        )
        now = datetime.now()
        quality_warnings = self._data_quality_warnings(current)
        availability = self._input_availability(current)
        observed_snapshot = self._observed_snapshot(current)
        work_order_id = f"TLW-{now:%Y%m%d}-{current['tree_id']:03d}-{now:%H%M%S}"
        ledger = calculation_ledger(
            current,
            current_rule,
            target,
            target["comprehensive_risk"],
            standard,
        )
        return {
            "prediction_mode": prediction_mode,
            "prediction_mode_label": (
                "公式树木生长预测" if prediction_mode == "formula" else "时序模型树木生长预测"
            ),
            "prediction_method": {
                "code": prediction_mode,
                "label": (
                    "分树种生长公式与季节工况引擎"
                    if prediction_mode == "formula"
                    else "LSTM时序注意力树木生长引擎"
                ),
                "description": (
                    "先按树种、月份和气象条件预测树高、胸径与冠幅，再对照线路净空标准。"
                    if prediction_mode == "formula"
                    else "用LSTM近期、季节和树种注意力修正树木生长，再对照线路净空标准。"
                ),
                "uses_temporal_model": prediction_mode == "ai",
                "auditable_formula": "R = wD×D + wG×G + wM×M + wH×H",
            },
            "report_type": "树木生长与树线安全预测报告",
            "work_order": {
                "work_order_id": work_order_id,
                "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "urgency": urgency,
                "urgency_icon": urgency_icon,
                "comprehensive_urgency": comprehensive_urgency,
                "comprehensive_urgency_icon": comprehensive_urgency_icon,
                "responsible_unit": str(snapshot.get("responsible_unit", "请立即派单至运维班组")),
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
                "observation_season": season_for_month(current["observation_month"]),
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
                "line_name": observed_snapshot["line_name"] or "未提供",
                "tower_section": observed_snapshot["tower_section"] or "未提供",
                "gps_longitude": observed_snapshot["gps_longitude"] or "未提供",
                "gps_latitude": observed_snapshot["gps_latitude"] or "未提供",
                "tree_code": observed_snapshot["tree_code"] or "未提供",
                "species": current["species"],
                "height": current["height"],
                "crown_width": observed_snapshot["crown_width"],
            },
            "input_snapshot": current,
            "observed_input_snapshot": observed_snapshot,
            "input_availability": availability,
            "applied_standard": standard.to_dict(),
            "standard_reference": standard.standard_reference,
            "feature_catalog": FEATURE_CATALOG,
            "feature_attention_analysis": current_attention,
            "clearance_defect_classification": current_rule.get("clearance_defect"),
            "current_assessment": current_rule,
            "short_term_forecast": short_term,
            "long_term_forecast": long_term,
            "selected_target_prediction": target,
            "risk_timeline_analysis": risk_timeline,
            "threshold_timing": threshold_timing,
            "historical_context": {
                "status": "已启用" if len(usable_history) >= 2 else "等待用户提供历史数据",
                "record_count": len(usable_history),
                "historical_risk_index": historical_risk["index"],
                "historical_risk_score": round(historical_risk["index"] * 100, 2),
                "calculation": historical_risk,
                "effect": (
                    "历史观测既进入 LSTM，也形成独立历史隐患指数 H 并参与综合风险 R。"
                    if len(usable_history) >= 2
                    else "当前 H 按可用记录计算；录入至少两期同树历史后可计算重复预警和净空变化。"
                ),
            },
            "model_context": {
                "risk_model": (
                    "LSTM 三重学习注意力（近期、季节、树种特征）+ 可配置特征注意力先验"
                    " + D/G/M/H 可解释综合风险"
                    if prediction_mode == "ai"
                    else "D/G/M/H标准公式 + 行业阈值 + 历史风险系数"
                ),
                "long_term_auxiliary_formula": (
                    "辅助风险分 = 0.50×LSTM分 + 0.35×规则分 + 0.15×可配置特征注意力先验分"
                    if prediction_mode == "ai"
                    else "公式风险分 = 标准规则分；四级预警继续以D/G/M/H综合风险R为准"
                ),
                "input_engineering": (
                    "生长方向决定树线侵入比例，树冠扩展速度影响水平净距，近期修剪记录降低短期生长情景；"
                    "档距、导线高度、线路走向与风向作为线路档案和联合研判项保留。"
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
                "diameter_change": round(target["predicted_diameter"] - current["diameter"], 3),
                "crown_width_change": round(crown_width_delta, 3),
                "clearance_change": round(clearance_delta, 3),
                "risk_score_change": round(risk_delta, 2),
                "comprehensive_risk_score_change": round(comprehensive_risk_delta, 2),
                "direction": (
                    "树木持续生长"
                    if height_delta > 0.01 or crown_width_delta > 0.01
                    else "树木变化较小"
                ),
            },
            "calculation_ledger": ledger,
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
                "并融合 LSTM 时序评分、标准规则评分和 D/G/M/H 四因子综合风险。",
                "四级预警以综合风险指数 R 为准；旧版低/中/高风险分作为辅助模型分保留。",
                (
                    "历史隐患 H 由超标率、事故率、重复预警率和处置后复发率加权；"
                    "缺失字段会在报告中标明。"
                ),
                "所有短时日期和未来月份均从观测日期起算，而不是从工单生成日期起算。",
                "重点月份用于展示选定月份结果；风险上升月与修剪时点按完整风险扫描月份自动计算。",
                "图片中的12项特征注意力系数已作为默认先验，用户可在设置中修改，且权重和必须为1。",
                "净空缺陷按危急、严重、一般、安全四档展示；任一距离达到危急阈值时优先覆盖。",
                "生长方向、树冠扩展速度和修剪记录已进入净空推演；未校准的线路几何与方向字段不单独硬加分。",
                standard.standard_reference,
                "未提供未来逐日天气预报时，短时预测沿用观测时气象作为情景假设。",
                "当前树种生长模型使用现实规律约束的模拟数据；收到真实历史数据后应重训校准。",
                "预测值用于预警和巡检排序，真实工程处置仍需现场复测。",
            ],
        }

    def _normalize_snapshot(
        self,
        value: dict[str, Any],
        standard: PredictionStandards | None = None,
    ) -> dict[str, Any]:
        standard = standard or PredictionStandards()
        selection_active = isinstance(value.get("selected_fields"), list)
        if selection_active:
            provided = {
                str(field) for field in value["selected_fields"] if str(field) in ALL_INPUT_FIELDS
            }
            missing = [
                INPUT_FIELD_LABELS[field]
                for field in REQUIRED_INPUT_FIELDS
                if field not in value or value[field] in (None, "")
            ]
            if missing:
                raise ValueError(f"以下核心字段必须提供：{'、'.join(missing)}")
            provided.update(REQUIRED_INPUT_FIELDS)
        else:
            # Legacy callers did not send selection metadata. Preserve their
            # historical default behavior and only activate optional semantics
            # for the new checkbox-enabled client.
            provided = set(ALL_INPUT_FIELDS)

        observed_on = observation_date_from(value)
        clearance = float(value.get("clearance_distance", value.get("distance_to_wire", 999)))
        rules = standard.rules_for_voltage(float(value.get("voltage_level_kv", 110)))
        neutral_temperature = min(20.0, standard.temperature_growth_threshold - 1.0)
        neutral_sunlight = min(6.0, standard.sunlight_growth_threshold - 1.0)
        defaults: dict[str, Any] = {
            "tree_id": 0,
            "tree_code": "",
            "age": 20.0,
            "diameter": max(10.0, float(value.get("height", 0) or 0) * 2.0),
            "crown_width": 0.0,
            "growth_direction": "不确定",
            "slope": 0.0,
            "historical_patrol_distance": clearance,
            "crown_expansion_speed": 0.0,
            "pruning_record": "无记录",
            "maximum_sag_winter": round(float(rules["max_sag"]) * 0.7, 3),
            "maximum_sag_summer": float(rules["max_sag"]),
            "span_length": 0.0,
            "conductor_height": 0.0,
            "contact_target": "none",
            "line_direction": "不确定",
            "vertical_distance": clearance,
            "horizontal_distance": clearance,
            "wind_offset_distance": 0.0,
            "wind_speed": 0.0,
            "wind_direction": "不确定",
            "rainfall": 0.0,
            "humidity": 60.0,
            "temperature": neutral_temperature,
            "sunlight_hours": neutral_sunlight,
            "line_name": "",
            "tower_section": "",
            "gps_longitude": "",
            "gps_latitude": "",
        }

        def selected_value(field: str) -> Any:
            if field in provided and value.get(field) not in (None, ""):
                return value[field]
            return defaults[field]

        result = {
            "tree_id": int(selected_value("tree_id")),
            "tree_code": str(selected_value("tree_code")),
            "species": str(value.get("species", "其他")),
            "height": float(value.get("height", 0)),
            "crown_width": float(selected_value("crown_width")),
            "growth_direction": str(selected_value("growth_direction")),
            "age": float(selected_value("age")),
            "diameter": float(selected_value("diameter")),
            "slope": float(selected_value("slope")),
            "historical_patrol_distance": float(selected_value("historical_patrol_distance")),
            "crown_expansion_speed": float(selected_value("crown_expansion_speed")),
            "pruning_record": str(selected_value("pruning_record")),
            "vertical_distance": float(selected_value("vertical_distance")),
            "horizontal_distance": float(selected_value("horizontal_distance")),
            "clearance_distance": clearance,
            "wind_offset_distance": float(selected_value("wind_offset_distance")),
            "voltage_level_kv": float(value.get("voltage_level_kv", 110)),
            "maximum_sag_winter": float(selected_value("maximum_sag_winter")),
            "maximum_sag_summer": float(selected_value("maximum_sag_summer")),
            "span_length": float(selected_value("span_length")),
            "conductor_height": float(selected_value("conductor_height")),
            "contact_target": normalize_contact_target(selected_value("contact_target")),
            "line_direction": str(selected_value("line_direction")),
            "observation_date": observed_on.isoformat(),
            "observation_year": observed_on.year,
            "observation_month": observed_on.month,
            "observation_day": observed_on.day,
            "wind_speed": float(selected_value("wind_speed")),
            "wind_direction": str(selected_value("wind_direction")),
            "rainfall": float(selected_value("rainfall")),
            "humidity": float(selected_value("humidity")),
            "temperature": float(selected_value("temperature")),
            "sunlight_hours": float(selected_value("sunlight_hours")),
            "line_name": str(selected_value("line_name")),
            "tower_section": str(selected_value("tower_section")),
            "gps_longitude": str(selected_value("gps_longitude")),
            "gps_latitude": str(selected_value("gps_latitude")),
        }
        result["distance_to_wire"] = result["clearance_distance"]
        result["month"] = result["observation_month"]
        result["season_stage"] = season_stage_for_month(result["observation_month"])
        result["_selection_active"] = selection_active
        result["_provided_fields"] = sorted({*provided, "observation_date"})
        result["_omitted_fields"] = sorted(set(OPTIONAL_INPUT_FIELDS) - provided)
        result["_imputations"] = {
            field: {
                "used_value": result[field],
                "reason": self._imputation_reason(field),
            }
            for field in result["_omitted_fields"]
        }
        return result

    @staticmethod
    def _imputation_reason(field: str) -> str:
        if field in {"vertical_distance", "horizontal_distance"}:
            return "未提供该方向距离，模型暂以已提供的净空距离代替"
        if field in {"maximum_sag_winter", "maximum_sag_summer"}:
            return "未提供现场弧垂，模型采用所选电压等级标准的季节情景值"
        if field in {"age", "diameter"}:
            return "未提供生长特征，模型采用透明的通用树木估计值"
        if field in {"wind_speed", "rainfall", "wind_offset_distance", "slope"}:
            return "未提供风险事件数据，当前情景按不额外加险的中性值计算"
        if field in {"humidity", "temperature", "sunlight_hours"}:
            return "未提供生长气象数据，模型采用不触发生长加分的中性情景值"
        if field == "crown_width":
            return "未提供冠幅；当前模型不让该缺失值直接加分"
        if field in {"historical_patrol_distance", "crown_expansion_speed", "pruning_record"}:
            return "未提供历史变化特征；以当前净空或无变化、无记录的中性情景计算"
        if field in {"growth_direction", "wind_direction", "line_direction"}:
            return "未提供方向信息；方向暂记为不确定，不额外加分"
        if field in {"span_length", "conductor_height"}:
            return "未提供线路几何档案；报告标记缺失，不额外加分"
        if field == "contact_target":
            return "未提供现场接触情况；按未碰到处理，不能替代现场巡检确认"
        return "未提供档案数据，仅在报告中标记为缺失"

    @staticmethod
    def _observed_snapshot(value: dict[str, Any]) -> dict[str, Any]:
        omitted = set(value.get("_omitted_fields", []))
        result = {key: field_value for key, field_value in value.items() if not key.startswith("_")}
        for field in omitted:
            result[field] = None
        result["distance_to_wire"] = result.get("clearance_distance")
        return result

    @staticmethod
    def _input_availability(value: dict[str, Any]) -> dict[str, Any]:
        provided = list(value.get("_provided_fields", []))
        omitted = list(value.get("_omitted_fields", []))
        return {
            "selection_enabled": bool(value.get("_selection_active")),
            "required_fields": list(REQUIRED_INPUT_FIELDS),
            "provided_fields": provided,
            "omitted_fields": omitted,
            "provided_count": len([field for field in provided if field in ALL_INPUT_FIELDS]),
            "optional_omitted_count": len(omitted),
            "field_labels": INPUT_FIELD_LABELS,
            "imputations": value.get("_imputations", {}),
            "status": (
                "已采用 "
                f"{len([field for field in provided if field in ALL_INPUT_FIELDS])} 个输入项，"
                f"{len(omitted)} 个可选项未提供"
                if value.get("_selection_active")
                else "兼容旧版请求：按原有字段默认规则计算"
            ),
        }

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
            ("historical_patrol_distance", "历史巡检距离"),
            ("crown_expansion_speed", "树冠扩展速度"),
            ("span_length", "档距"),
            ("conductor_height", "导线高度"),
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
        omitted = set(value.get("_omitted_fields", []))
        if omitted:
            warnings.append(
                f"本次有 {len(omitted)} 个可选字段未启用；模型替代值已在“数据采用情况”中逐项公开"
            )
        if value["crown_width"] <= 0 and "crown_width" not in omitted:
            warnings.append("未填写有效冠幅；当前不影响评分，但工单信息不完整")
        if not omitted and (
            value["gps_longitude"] == "未填写" or value["gps_latitude"] == "未填写"
        ):
            warnings.append("GPS 坐标不完整，工单无法准确定位")
        return warnings

    def _growth_for_month(self, state: dict[str, Any], month: int) -> tuple[float, float]:
        height_growth, diameter_growth = self._growth_model_for_prediction().predict(
            state["species"],
            month=month,
            temperature=state["temperature"],
            sunlight_hours=state["sunlight_hours"],
            rainfall=state["rainfall"],
            humidity=state["humidity"],
            age=state["age"],
            height=state["height"],
        )
        pruning_factor = {
            "近3个月已修剪": 0.65,
            "近6个月已修剪": 0.80,
            "近1年已修剪": 0.90,
        }.get(str(state.get("pruning_record", "无记录")), 1.0)
        return height_growth * pruning_factor, diameter_growth * pruning_factor

    @staticmethod
    def _encroachment_factors(state: dict[str, Any]) -> tuple[float, float]:
        direction = str(state.get("growth_direction", "不确定"))
        vertical_factor = {
            "向导线": 0.75,
            "背离导线": 0.10,
            "平行线路": 0.30,
            "竖直向上": 0.50,
        }.get(direction, 0.45)
        crown_factor = {
            "向导线": 1.00,
            "背离导线": 0.00,
            "平行线路": 0.50,
            "竖直向上": 0.20,
        }.get(direction, 0.35)
        return vertical_factor, crown_factor

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
                if record.get("observation_month") and not record.get("observation_date"):
                    merged.pop("observation_date", None)
                    merged["observation_year"] = int(
                        record.get("observation_year", current["observation_year"])
                        or current["observation_year"]
                    )
                    merged["observation_day"] = int(record.get("observation_day", 1) or 1)
                normalized = self._normalize_snapshot(merged)
            except (TypeError, ValueError):
                continue
            for field in (
                "risk_level",
                "risk_score",
                "standard_analysis",
                "hazard_occurred",
                "incident_occurred",
                "treatment_performed",
                "handled",
                "repeat_alert",
                "created_at",
            ):
                if field in record:
                    normalized[field] = record[field]
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
            monthly_height, monthly_diameter = self._growth_for_month(state, projected_day.month)
            days_in_month = (
                add_months(projected_day.replace(day=1), 1) - projected_day.replace(day=1)
            ).days
            total_height_growth += monthly_height / days_in_month
            total_diameter_growth += monthly_diameter / days_in_month
        state["height"] += total_height_growth
        state["diameter"] += total_diameter_growth
        vertical_factor, crown_factor = self._encroachment_factors(state)
        vertical_encroachment = total_height_growth * vertical_factor
        crown_radial_growth = max(
            total_height_growth * 0.16,
            max(0.0, float(state.get("crown_expansion_speed", 0))) * days / 30.4375,
        )
        crown_expansion = crown_radial_growth * crown_factor
        state["crown_width"] += 2 * crown_radial_growth
        state["vertical_distance"] = max(0, state["vertical_distance"] - vertical_encroachment)
        state["horizontal_distance"] = max(0, state["horizontal_distance"] - crown_expansion)
        state["clearance_distance"] = max(
            0,
            state["clearance_distance"] - max(vertical_encroachment, crown_expansion),
        )
        state["distance_to_wire"] = state["clearance_distance"]
        state["observation_date"] = target_on.isoformat()
        state["observation_year"] = target_on.year
        state["observation_month"] = target_on.month
        state["observation_day"] = target_on.day
        state["month"] = target_on.month
        result = self._rule_result(state, standard)
        self._attach_contact_risk(result, state, standard)
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
            "predicted_crown_width": round(state["crown_width"], 3),
            "predicted_vertical_distance": round(state["vertical_distance"], 3),
            "predicted_horizontal_distance": round(state["horizontal_distance"], 3),
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
            "predicted_wind_offset_distance": round(state["wind_offset_distance"], 3),
            "predicted_sag": wire["current_sag"],
            "height_growth": round(total_height_growth, 4),
            "diameter_growth": round(total_diameter_growth, 4),
            "crown_width_growth": round(2 * crown_radial_growth, 4),
            "predicted_temperature": round(state["temperature"], 2),
            "predicted_sunlight_hours": round(state["sunlight_hours"], 2),
            "predicted_rainfall": round(state["rainfall"], 2),
            "predicted_humidity": round(state["humidity"], 2),
            "predicted_wind_speed": round(state["wind_speed"], 2),
            "season": season_for_month(state["observation_month"]),
            "season_stage": season_stage_for_month(state["observation_month"]),
            "wire_condition": wire["season"],
            "clearance_defect": result.get("standard_analysis", {}).get("clearance_defect"),
            "contact_risk": result["contact_risk"],
            "suggestion": result["suggestion"],
        }

    def _next_month(
        self,
        previous: dict[str, Any],
        baseline: dict[str, Any],
        month_index: int,
    ) -> dict[str, Any]:
        state = dict(previous)
        target_on = add_months(date.fromisoformat(baseline["observation_date"]), month_index)
        month = target_on.month
        baseline_summer = math.cos(2 * math.pi * (baseline["observation_month"] - 7) / 12)
        next_summer = math.cos(2 * math.pi * (month - 7) / 12)
        state["temperature"] = baseline["temperature"] + 8 * (next_summer - baseline_summer)
        state["sunlight_hours"] = max(
            1,
            baseline["sunlight_hours"] + 2.5 * (next_summer - baseline_summer),
        )
        baseline_rain_season = math.cos(2 * math.pi * (baseline["observation_month"] - 7.5) / 12)
        next_rain_season = math.cos(2 * math.pi * (month - 7.5) / 12)
        previous_rainfall = state["rainfall"]
        state["rainfall"] = max(
            0,
            baseline["rainfall"] + 24 * (next_rain_season - baseline_rain_season),
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
        state["season_stage"] = season_stage_for_month(month)
        state["age"] = baseline["age"] + month_index / 12
        height_growth, diameter_growth = self._growth_for_month(state, month)
        state["height"] += height_growth
        state["diameter"] += diameter_growth
        vertical_factor, crown_factor = self._encroachment_factors(state)
        vertical_encroachment = height_growth * vertical_factor
        crown_radial_growth = max(
            height_growth * 0.16,
            max(0.0, float(state.get("crown_expansion_speed", 0))),
        )
        crown_expansion = crown_radial_growth * crown_factor
        state["crown_width"] += 2 * crown_radial_growth
        state["vertical_distance"] = max(0, state["vertical_distance"] - vertical_encroachment)
        state["horizontal_distance"] = max(0, state["horizontal_distance"] - crown_expansion)
        state["clearance_distance"] = max(
            0,
            state["clearance_distance"] - max(vertical_encroachment, crown_expansion),
        )
        state["distance_to_wire"] = state["clearance_distance"]
        state["_height_growth"] = height_growth
        state["_diameter_growth"] = diameter_growth
        state["_crown_width_growth"] = 2 * crown_radial_growth
        state["_target_date"] = target_on.isoformat()
        return state

    def _apply_temporal_tree_adjustment(
        self,
        previous_records: list[dict[str, Any]],
        state: dict[str, Any],
        temporal: dict[str, Any],
    ) -> None:
        """Use learned recency, season and species attention to adjust tree growth.

        The species growth model remains the physical baseline.  AI mode then
        blends recent observed growth and the trained attention gates so its
        tree-height, diameter and crown forecasts are not merely risk-score
        aliases.
        """

        attention = temporal.get("attention") or {}
        species_attention = attention.get("species_features") or {}
        season_attention = attention.get("season_features") or {}

        def focus(field: str) -> float:
            values = [
                float(mapping[field])
                for mapping in (species_attention, season_attention)
                if field in mapping
            ]
            return sum(values) / len(values) if values else 0.5

        time_attention = [float(value) for value in attention.get("time") or []]
        recency_ratio = (
            time_attention[-1] * len(time_attention) if time_attention else 1.0
        )
        recency_factor = max(0.92, min(1.08, 1 + 0.04 * (recency_ratio - 1)))
        height_factor = max(
            0.85,
            min(1.2, (1 + 0.18 * (focus("height") - 0.5)) * recency_factor),
        )
        diameter_factor = max(
            0.85,
            min(1.2, (1 + 0.18 * (focus("diameter") - 0.5)) * recency_factor),
        )
        crown_focus = (focus("height") + focus("horizontal_distance")) / 2
        crown_factor = max(
            0.85,
            min(1.2, (1 + 0.16 * (crown_focus - 0.5)) * recency_factor),
        )

        base_height_growth = float(state.get("_height_growth", 0))
        base_diameter_growth = float(state.get("_diameter_growth", 0))
        base_crown_growth = float(state.get("_crown_width_growth", 0))
        recent_height_growth: float | None = None
        recent_diameter_growth: float | None = None
        if len(previous_records) >= 2:
            recent_height_growth = max(
                0.0,
                float(previous_records[-1]["height"])
                - float(previous_records[-2]["height"]),
            )
            recent_diameter_growth = max(
                0.0,
                float(previous_records[-1]["diameter"])
                - float(previous_records[-2]["diameter"]),
            )

        adjusted_height_growth = base_height_growth
        adjusted_diameter_growth = base_diameter_growth
        if recent_height_growth is not None:
            adjusted_height_growth = 0.75 * base_height_growth + 0.25 * recent_height_growth
        if recent_diameter_growth is not None:
            adjusted_diameter_growth = (
                0.75 * base_diameter_growth + 0.25 * recent_diameter_growth
            )
        adjusted_height_growth *= height_factor
        adjusted_diameter_growth *= diameter_factor
        adjusted_crown_growth = base_crown_growth * crown_factor

        state["height"] += adjusted_height_growth - base_height_growth
        state["diameter"] += adjusted_diameter_growth - base_diameter_growth
        state["crown_width"] += adjusted_crown_growth - base_crown_growth
        vertical_factor, toward_wire_factor = self._encroachment_factors(state)
        vertical_encroachment = adjusted_height_growth * vertical_factor
        crown_encroachment = adjusted_crown_growth / 2 * toward_wire_factor
        previous = previous_records[-1]
        state["vertical_distance"] = max(
            0, float(previous["vertical_distance"]) - vertical_encroachment
        )
        state["horizontal_distance"] = max(
            0, float(previous["horizontal_distance"]) - crown_encroachment
        )
        state["clearance_distance"] = max(
            0,
            float(previous["clearance_distance"])
            - max(vertical_encroachment, crown_encroachment),
        )
        state["distance_to_wire"] = state["clearance_distance"]
        state["_height_growth"] = adjusted_height_growth
        state["_diameter_growth"] = adjusted_diameter_growth
        state["_crown_width_growth"] = adjusted_crown_growth
        state["_tree_growth_attention"] = {
            "height_factor": round(height_factor, 4),
            "diameter_factor": round(diameter_factor, 4),
            "crown_factor": round(crown_factor, 4),
            "recent_growth_blended": recent_height_growth is not None,
        }

    def _long_projection(
        self,
        records: list[dict[str, Any]],
        months: int,
        standard: PredictionStandards,
        *,
        prediction_mode: str = "ai",
    ) -> dict[str, Any]:
        state = records[-1]
        rule = self._rule_result(state, standard)
        self._attach_contact_risk(rule, state, standard)
        attention_prior = feature_attention_prior(state, records[:-1], standard)
        if prediction_mode == "ai":
            temporal_predictor = self._temporal_predictor_for_prediction()
            lstm = state.get("_temporal_prediction") or temporal_predictor.predict(
                records, state["species"]
            )
            lstm_score: float | None = float(lstm["risk_score"])
            combined_score = (
                0.50 * lstm_score
                + 0.35 * float(rule["risk_score"])
                + 0.15 * float(attention_prior["score"])
            )
            method = "LSTM时序注意力修正 + 分树种季节生长"
            score_formula = "0.50×LSTM + 0.35×规则 + 0.15×可配置特征注意力先验"
        else:
            lstm_score = None
            combined_score = float(rule["risk_score"])
            method = "分树种生长公式 + 季节与气象修正"
            score_formula = "公式风险分 = 标准规则分；四级预警以D/G/M/H综合风险R为准"
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
            "prediction_method": method,
            "risk_level": level,
            "risk_score": round(combined_score, 2),
            "rule_risk_score": rule["risk_score"],
            "lstm_risk_score": round(lstm_score, 2) if lstm_score is not None else None,
            "feature_attention_prior_score": attention_prior["score"],
            "attention_prior": attention_prior,
            "auxiliary_score_formula": score_formula,
            "predicted_height": round(state["height"], 3),
            "predicted_age": round(state["age"], 3),
            "predicted_diameter": round(state["diameter"], 3),
            "predicted_crown_width": round(state["crown_width"], 3),
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
            "monthly_diameter_growth": round(float(state.get("_diameter_growth", 0)), 4),
            "monthly_crown_width_growth": round(
                float(state.get("_crown_width_growth", 0)), 4
            ),
            "tree_growth_attention": state.get("_tree_growth_attention"),
            "predicted_temperature": round(state["temperature"], 2),
            "predicted_sunlight_hours": round(state["sunlight_hours"], 2),
            "predicted_rainfall": round(state["rainfall"], 2),
            "predicted_humidity": round(state["humidity"], 2),
            "predicted_wind_speed": round(state["wind_speed"], 2),
            "season": season_for_month(target_on.month),
            "season_stage": season_stage_for_month(target_on.month),
            "wire_condition": wire["season"],
            "clearance_defect": rule.get("standard_analysis", {}).get("clearance_defect"),
            "contact_risk": rule["contact_risk"],
            "suggestion": suggestion,
        }

    @staticmethod
    def _attach_comprehensive_risk(
        assessment: dict[str, Any],
        baseline: dict[str, Any],
        standard: PredictionStandards,
        historical_risk: dict[str, Any],
        *,
        horizon_months: int,
        baseline_effective_clearance: float,
    ) -> None:
        """Attach the authoritative four-level D/G/M/H assessment in-place."""
        if horizon_months == 0:
            effective = float(
                assessment.get("standard_analysis", {}).get(
                    "effective_clearance", baseline["clearance_distance"]
                )
            )
            forecast_clearance = effective
            weather_snapshot = baseline
        else:
            effective = float(
                assessment.get(
                    "predicted_effective_clearance",
                    assessment.get(
                        "predicted_clearance_distance",
                        baseline["clearance_distance"],
                    ),
                )
            )
            forecast_clearance = effective
            weather_snapshot = {
                "wind_speed": assessment.get("predicted_wind_speed", baseline["wind_speed"]),
                "rainfall": assessment.get("predicted_rainfall", baseline["rainfall"]),
                "temperature": assessment.get("predicted_temperature", baseline["temperature"]),
                "sunlight_hours": assessment.get(
                    "predicted_sunlight_hours", baseline["sunlight_hours"]
                ),
            }
        safety_clearance = standard.rules_for_voltage(baseline["voltage_level_kv"])[
            "min_clearance_distance"
        ]
        result = comprehensive_risk(
            effective_clearance=effective,
            safety_clearance=safety_clearance,
            baseline_clearance=baseline_effective_clearance,
            forecast_clearance=forecast_clearance,
            weather_snapshot=weather_snapshot,
            historical=historical_risk,
            standard=standard,
        )
        assessment["comprehensive_risk"] = result
        assessment["comprehensive_risk_index"] = result["index"]
        assessment["comprehensive_risk_score"] = result["score"]
        assessment["hazard_level"] = result["hazard_level"]
        assessment["hazard_code"] = result["hazard_code"]
        assessment["warning_color"] = result["warning_color"]
        assessment["risk_escalation_indicator"] = result["risk_escalation_indicator"]
        assessment["hazard_suggestion"] = result["action"]
        if horizon_months == 0:
            distance_checks = assessment.get("standard_analysis", {}).get("checks", {})
            horizontal = distance_checks.get("horizontal_distance", {}).get(
                "actual", baseline["horizontal_distance"]
            )
            vertical = distance_checks.get("vertical_distance", {}).get(
                "actual", baseline["vertical_distance"]
            )
        else:
            horizontal = assessment.get(
                "predicted_horizontal_distance", baseline["horizontal_distance"]
            )
            vertical = assessment.get("predicted_vertical_distance", baseline["vertical_distance"])
        assessment["clearance_defect"] = classify_clearance_defect(
            horizontal_distance=float(horizontal),
            vertical_distance=float(vertical),
            clearance_distance=effective,
            standards=standard,
            voltage_level_kv=float(baseline["voltage_level_kv"]),
        )
        if horizon_months > 0:
            assessment["clearance_interval"] = clearance_interval(
                forecast_clearance,
                horizon_months,
                historical_risk,
            )

    @staticmethod
    def _attach_contact_risk(
        assessment: dict[str, Any], state: dict[str, Any], standard: PredictionStandards
    ) -> None:
        """Attach field contact and optionally apply it as a hard risk override."""
        result = contact_risk(
            contact_target=state.get("contact_target", "none"),
            tree_height=float(state["height"]),
            conductor_height=float(state.get("conductor_height", 0.0)),
            clearance_distance=float(state.get("clearance_distance", 0.0)),
            general_risk_distance=standard.general_risk_clearance_distance,
            safe_distance=standard.safe_clearance_distance,
        )
        if not standard.use_contact_risk_override:
            result = {
                **result,
                "applied": False,
                "requires_immediate_pruning": False,
                "action": "已记录现场接触情况，未参与本次风险升级",
                "basis": f"{result['basis']}；设置中未启用接触情况风险升级。",
            }
        else:
            result["applied"] = True
        assessment["contact_risk"] = result

    @staticmethod
    def _urgency(maximum_score: float, standard: PredictionStandards) -> tuple[str, str]:
        if maximum_score >= standard.high_risk_score:
            return "最高级（I级危险）", "⚠⚠⚠"
        if maximum_score >= standard.medium_risk_score:
            return "较高级（II级关注）", "⚠⚠"
        return "常规级（III级巡检）", "⚠"

    @staticmethod
    def _comprehensive_urgency(
        maximum_score: float, standard: PredictionStandards
    ) -> tuple[str, str]:
        if maximum_score >= standard.emergency_hazard_index * 100:
            return "最高级（I级危险）", "⚠⚠⚠"
        if maximum_score >= standard.major_hazard_index * 100:
            return "较高级（II级关注）", "⚠⚠"
        if maximum_score >= standard.general_hazard_index * 100:
            return "关注级（III级监测）", "⚠"
        return "安全级（IV级巡检）", "✓"

    def _daily_threshold_timing(
        self,
        current: dict[str, Any],
        current_assessment: dict[str, Any],
        standard: PredictionStandards,
        historical_risk: dict[str, Any],
        baseline_effective_clearance: float,
    ) -> dict[str, Any]:
        """Estimate the first day a configured risk threshold is reached.

        The scan deliberately uses daily projections instead of rounding to a
        future month, so the report can state how many days remain before the
        next warning level or the user-defined pruning threshold is reached.
        """
        observed_on = date.fromisoformat(current["observation_date"])
        current_index = float(current_assessment.get("comprehensive_risk_index", 0.0))
        levels = (
            ("一般隐患", "general", standard.general_hazard_index),
            ("重大隐患", "major", standard.major_hazard_index),
            ("紧急隐患", "emergency", standard.emergency_hazard_index),
        )
        next_level = next((item for item in levels if item[2] > current_index), None)
        safety_clearance = standard.rules_for_voltage(current["voltage_level_kv"])[
            "min_clearance_distance"
        ]
        current_effective_clearance = float(
            current_assessment.get("standard_analysis", {}).get(
                "effective_clearance", current["clearance_distance"]
            )
        )
        immediate_contact = bool(
            current_assessment.get("contact_risk", {}).get("requires_immediate_pruning")
        )
        current_clearance_violation = current_effective_clearance < safety_clearance

        def timing_record(
            *,
            threshold: float | None,
            label: str,
            code: str,
            days: int | None = None,
            reason: str = "",
        ) -> dict[str, Any]:
            return {
                "label": label,
                "code": code,
                "threshold_index": round(threshold, 4) if threshold is not None else None,
                "days": days,
                "target_date": (observed_on + timedelta(days=days)).isoformat()
                if days is not None
                else None,
                "within_scan_window": days is not None,
                "reason": reason,
            }

        next_timing = (
            timing_record(threshold=next_level[2], label=next_level[0], code=next_level[1])
            if next_level
            else timing_record(
                threshold=None,
                label="已处于最高风险等级",
                code="emergency",
                days=0,
                reason="当前综合风险指数已达到紧急隐患等级",
            )
        )
        pruning_timing = timing_record(
            threshold=standard.pruning_trigger_hazard_index,
            label="设置的修剪/预警阈值",
            code="configured_pruning_threshold",
        )

        danger_clearance = max(0.0, float(standard.danger_clearance_distance))

        def clearance_timing_record(
            *,
            days: int | None = None,
            reason: str = "",
        ) -> dict[str, Any]:
            return {
                "label": "危险区域净空距离阈值",
                "code": "configured_danger_clearance",
                "threshold_clearance_distance": round(danger_clearance, 4),
                "current_effective_clearance": round(current_effective_clearance, 4),
                "days": days,
                "target_date": (observed_on + timedelta(days=days)).isoformat()
                if days is not None
                else None,
                "within_scan_window": days is not None,
                "reason": reason,
            }

        danger_clearance_timing = clearance_timing_record()
        if current_effective_clearance <= danger_clearance:
            danger_clearance_timing = clearance_timing_record(
                days=0,
                reason=(
                    f"当前有效净空 {current_effective_clearance:g} m 已达到或低于"
                    f"设置的危险区域净空 {danger_clearance:g} m"
                ),
            )

        if immediate_contact:
            pruning_timing = timing_record(
                threshold=standard.pruning_trigger_hazard_index,
                label="设置的修剪/预警阈值",
                code="configured_pruning_threshold",
                days=0,
                reason=current_assessment["contact_risk"].get("basis", "现场接触高风险"),
            )
        elif current_clearance_violation:
            pruning_timing = timing_record(
                threshold=standard.pruning_trigger_hazard_index,
                label="设置的修剪/预警阈值",
                code="configured_pruning_threshold",
                days=0,
                reason=f"当前有效净空低于 {safety_clearance:g} m 标准",
            )
        elif current_index >= standard.pruning_trigger_hazard_index:
            pruning_timing = timing_record(
                threshold=standard.pruning_trigger_hazard_index,
                label="设置的修剪/预警阈值",
                code="configured_pruning_threshold",
                days=0,
                reason="当前综合风险指数已达到设置的修剪/预警阈值",
            )

        pending_next = next_timing["days"] is None
        pending_pruning = pruning_timing["days"] is None
        pending_danger_clearance = danger_clearance_timing["days"] is None
        for days in range(1, 366):
            if not (pending_next or pending_pruning or pending_danger_clearance):
                break
            projection = self._short_projection(current, days, standard)
            self._attach_comprehensive_risk(
                projection,
                current,
                standard,
                historical_risk,
                horizon_months=max(1, math.ceil(days / 30)),
                baseline_effective_clearance=baseline_effective_clearance,
            )
            index = float(projection["comprehensive_risk_index"])
            effective_clearance = float(
                projection.get(
                    "predicted_effective_clearance",
                    projection["predicted_clearance_distance"],
                )
            )
            if pending_next and next_level and index >= next_level[2]:
                next_timing = timing_record(
                    threshold=next_level[2],
                    label=next_level[0],
                    code=next_level[1],
                    days=days,
                    reason=f"预测综合风险指数达到 {next_level[2]:.2f}",
                )
                pending_next = False
            if pending_pruning and (
                index >= standard.pruning_trigger_hazard_index
                or effective_clearance < safety_clearance
                or projection.get("contact_risk", {}).get("requires_immediate_pruning")
            ):
                reason = (
                    f"预测综合风险指数达到 {standard.pruning_trigger_hazard_index:.2f}"
                    if index >= standard.pruning_trigger_hazard_index
                    else f"预测有效净空低于 {safety_clearance:g} m 标准"
                    if effective_clearance < safety_clearance
                    else projection["contact_risk"].get("basis", "预测现场接触高风险")
                )
                pruning_timing = timing_record(
                    threshold=standard.pruning_trigger_hazard_index,
                    label="设置的修剪/预警阈值",
                    code="configured_pruning_threshold",
                    days=days,
                    reason=reason,
                )
                pending_pruning = False
            if pending_danger_clearance and effective_clearance <= danger_clearance:
                danger_clearance_timing = clearance_timing_record(
                    days=days,
                    reason=(
                        f"预测有效净空降至 {effective_clearance:g} m，达到设置的"
                        f"危险区域净空 {danger_clearance:g} m"
                    ),
                )
                pending_danger_clearance = False

        if next_timing["days"] is None:
            next_timing["reason"] = "未来 365 天内未达到下一风险等级阈值"
        if pruning_timing["days"] is None:
            pruning_timing["reason"] = "未来 365 天内未达到设置的修剪/预警阈值"
        if danger_clearance_timing["days"] is None:
            danger_clearance_timing["reason"] = (
                f"未来 365 天内预计不会达到设置的危险区域净空 {danger_clearance:g} m"
            )
        return {
            "scan_days": 365,
            "current_index": round(current_index, 4),
            "current_level": current_assessment.get(
                "hazard_level", current_assessment.get("risk_level")
            ),
            "next_level": next_timing,
            "pruning_trigger": pruning_timing,
            "minimum_clearance_standard": safety_clearance,
            "danger_clearance": danger_clearance_timing,
        }

    @staticmethod
    def _risk_timeline_analysis(
        current_snapshot: dict[str, Any],
        current_assessment: dict[str, Any],
        forecasts: list[dict[str, Any]],
        standard: PredictionStandards,
        *,
        threshold_timing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        severity = {
            "安全": 0,
            "低风险": 0,
            "一般隐患": 1,
            "中风险": 1,
            "重大隐患": 2,
            "高风险": 2,
            "紧急隐患": 3,
        }
        current_score = float(current_assessment["risk_score"])
        current_hazard_index = float(
            current_assessment.get("comprehensive_risk_index", current_score / 100)
        )
        current_level = str(current_assessment["risk_level"])
        previous_score = current_score
        previous_level = current_level
        timeline = []
        for item in forecasts:
            score = float(item["risk_score"])
            hazard_index = float(item.get("comprehensive_risk_index", score / 100))
            level = str(item["risk_level"])
            change_previous = score - previous_score
            change_current = score - current_score
            category_increase = severity.get(level, 0) > severity.get(current_level, 0)
            significant = category_increase or change_current >= 3
            timeline.append(
                {
                    "horizon_months": int(item["horizon_months"]),
                    "target_date": item["target_date"],
                    "target_year_month": item["target_year_month"],
                    "risk_level": level,
                    "risk_score": round(score, 2),
                    "comprehensive_risk_index": round(hazard_index, 4),
                    "hazard_level": item.get("hazard_level"),
                    "comprehensive_risk_score": item.get("comprehensive_risk_score"),
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

        first_increase = next((item for item in timeline if item["significant_increase"]), None)
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
                if item["comprehensive_risk_index"] >= standard.pruning_trigger_hazard_index
            ),
            None,
        )
        first_high = next(
            (
                item
                for item in timeline
                if item["comprehensive_risk_index"] >= standard.emergency_hazard_index
            ),
            None,
        )
        minimum_clearance = standard.rules_for_voltage(current_snapshot["voltage_level_kv"])[
            "min_clearance_distance"
        ]
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
        current_violations = current_assessment.get("standard_analysis", {}).get("violations", [])
        current_effective_clearance = float(
            current_assessment.get("standard_analysis", {}).get(
                "effective_clearance",
                current_snapshot["clearance_distance"],
            )
        )
        current_clearance_violation = current_effective_clearance < minimum_clearance or any(
            item.get("field") == "clearance_distance" for item in current_violations
        )
        current_contact_risk = current_assessment.get("contact_risk", {})
        first_contact_high = next(
            (
                forecast
                for forecast in forecasts
                if forecast.get("contact_risk", {}).get("code") in {"danger", "high"}
            ),
            None,
        )

        observed_on = date.fromisoformat(current_snapshot["observation_date"])
        trigger_candidates = []
        if (
            first_medium is not None
            and current_hazard_index < standard.pruning_trigger_hazard_index
        ):
            trigger_candidates.append(
                (
                    first_medium["horizon_months"],
                    "风险达到设置的修剪/预警阈值",
                    first_medium,
                )
            )
        if first_high is not None and current_hazard_index < standard.emergency_hazard_index:
            trigger_candidates.append(
                (first_high["horizon_months"], "风险达到紧急隐患阈值", first_high)
            )
        if first_clearance_violation is not None:
            trigger_candidates.append(
                (
                    first_clearance_violation["horizon_months"],
                    f"预测净空低于 {minimum_clearance:g} m 标准",
                    first_clearance_violation,
                )
            )

        daily_pruning = (threshold_timing or {}).get("pruning_trigger", {})
        if daily_pruning.get("days") == 0:
            maintenance = {
                "status": "立即修剪",
                "requires_trimming": True,
                "recommended_trim_date": observed_on.isoformat(),
                "recommended_trim_month": observed_on.strftime("%Y年%m月"),
                "recommended_deadline": "当前已达到设置的修剪/预警阈值；立即现场复测并安排修剪",
                "trigger_month": observed_on.strftime("%Y年%m月"),
                "trigger_reason": daily_pruning.get("reason", "当前达到修剪阈值"),
                "lead_time_months": 0,
                "days_to_trigger": 0,
            }
        elif daily_pruning.get("days") is not None:
            trigger_days = int(daily_pruning["days"])
            trigger_date = date.fromisoformat(str(daily_pruning["target_date"]))
            trim_date = observed_on + timedelta(days=max(0, trigger_days - 1))
            maintenance = {
                "status": "计划修剪",
                "requires_trimming": True,
                "recommended_trim_date": trim_date.isoformat(),
                "recommended_trim_month": trim_date.strftime("%Y年%m月"),
                "recommended_deadline": (
                    f"预计 {trigger_days} 天后（{trigger_date:%Y-%m-%d}）达到设置的修剪/预警阈值；"
                    f"建议最迟在 {trim_date:%Y-%m-%d} 完成修剪。"
                ),
                "trigger_month": trigger_date.strftime("%Y年%m月"),
                "trigger_reason": daily_pruning.get("reason", "预测达到修剪阈值"),
                "lead_time_months": round(trigger_days / 30.4375, 2),
                "days_to_trigger": trigger_days,
            }
        elif current_contact_risk.get("requires_immediate_pruning"):
            maintenance = {
                "status": "立即修剪",
                "requires_trimming": True,
                "recommended_trim_date": observed_on.isoformat(),
                "recommended_trim_month": observed_on.strftime("%Y年%m月"),
                "recommended_deadline": "立即修剪；作业前必须完成现场安全复核",
                "trigger_month": observed_on.strftime("%Y年%m月"),
                "trigger_reason": current_contact_risk.get("basis", "现场接触高风险"),
                "lead_time_months": 0,
            }
        elif (
            current_hazard_index >= standard.pruning_trigger_hazard_index
            or current_clearance_violation
        ):
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
                    else "当前已达到设置的修剪/预警阈值"
                ),
                "lead_time_months": 0,
            }
        elif first_contact_high is not None:
            trim_offset = max(0, int(first_contact_high["horizon_months"]) - 1)
            trim_on = add_months(observed_on, trim_offset)
            maintenance = {
                "status": "计划修剪",
                "requires_trimming": True,
                "recommended_trim_date": trim_on.isoformat(),
                "recommended_trim_month": trim_on.strftime("%Y年%m月"),
                "recommended_deadline": (
                    f"建议最迟在 {trim_on:%Y年%m月} 完成修剪，"
                    f"避免树木超过导线高度的预测触发月 {first_contact_high['target_year_month']}"
                ),
                "trigger_month": first_contact_high["target_year_month"],
                "trigger_reason": first_contact_high["contact_risk"]["basis"],
                "lead_time_months": int(first_contact_high["horizon_months"]) - trim_offset,
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
        planning_level = (
            "高风险"
            if maintenance["status"] in {"立即处置", "立即修剪"}
            else "中风险"
            if maintenance["requires_trimming"]
            else "低风险"
        )
        planning_risk = {
            "risk_level": planning_level,
            "decision": maintenance["status"],
            "decision_month": maintenance["recommended_trim_month"],
            "trigger_month": maintenance["trigger_month"],
            "peak_month": peak["target_year_month"],
            "conclusion": (
                f"{planning_level}：{maintenance['status']}；"
                f"{maintenance['recommended_deadline']}"
            ),
        }
        return {
            "risk_increase_detected": first_increase is not None,
            "risk_increase_conclusion": risk_conclusion,
            "first_significant_increase": first_increase,
            "peak_risk": peak,
            "first_medium_risk": first_medium,
            "first_high_risk": first_high,
            "first_clearance_violation": first_clearance_violation,
            "threshold_timing": threshold_timing,
            "minimum_clearance_standard": minimum_clearance,
            "maintenance_decision": maintenance,
            "planning_risk": planning_risk,
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
        if maintenance["status"] in {"立即处置", "立即修剪"}:
            action = "立即现场复测，评估修枝、移栽或线路清障"
        elif maintenance["status"] in {"近期修剪", "计划修剪"}:
            action = "缩短巡检周期，安排定向修剪并复核净空"
        else:
            action = "保持常规巡检，持续采集月度树高与净空"
        voltage_minimum = standard.rules_for_voltage(input_snapshot["voltage_level_kv"])[
            "min_clearance_distance"
        ]
        return {
            "action": action,
            "deadline": maintenance["recommended_deadline"],
            "trigger": (
                f"{maintenance['trigger_reason']}；当前不合规项：{violated_names}"
                if violated_names
                else maintenance["trigger_reason"]
            ),
            "risk_increase_conclusion": risk_timeline["risk_increase_conclusion"],
            "recommended_trim_month": maintenance["recommended_trim_month"],
            "trigger_month": maintenance["trigger_month"],
            "target_month_goal": (
                f"{target['target_year_month']}前保持净空距离不低于 {voltage_minimum:g} m"
            ),
        }

    def _rule_result(self, state: dict[str, Any], standard: PredictionStandards) -> dict[str, Any]:
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
