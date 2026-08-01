"""Explainable D/G/M/H risk evaluation for tree-line forecasts.

The four indices follow the point-cloud prediction disclosure used by this
project: distance risk (D), growth trend (G), meteorology (M), and historical
hazard (H).  All coefficients come from ``PredictionStandards`` so a user can
replace the engineering defaults with a regional or industry profile.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from algorithms.prediction.standards import PredictionStandards

MODEL_STATUS = "experimental"
PARAMETER_SOURCE = "project_default_unverified"


def _finite_number(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
) -> float:
    """Convert an external numeric input without silently repairing it."""
    if value is None:
        raise ValueError(f"{name} must not be None")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {number}")
    return number


def _validated_standard(standard: PredictionStandards) -> PredictionStandards:
    if not isinstance(standard, PredictionStandards):
        raise ValueError("standard must be a PredictionStandards instance")
    standard.validate()
    return standard


def _snapshot_number(
    snapshot: dict[str, Any],
    field: str,
    *,
    default: float | None = None,
    minimum: float | None = None,
) -> float:
    if field not in snapshot:
        if default is None:
            raise ValueError(f"{field} is required")
        value = default
    else:
        value = snapshot[field]
    return _finite_number(field, value, minimum=minimum)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, _finite_number("risk value", value)))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "是",
        "有",
        "已发生",
        "已处置",
    }


def hazard_level(index: float, standard: PredictionStandards) -> dict[str, Any]:
    """Map the 0-1 comprehensive index to the four disclosed warning levels."""
    _validated_standard(standard)
    value = _finite_number("risk index", index, minimum=0)
    if value > 1:
        raise ValueError(f"risk index must be <= 1, got {value}")
    if value >= standard.emergency_hazard_index:
        return {
            "code": "emergency",
            "level": "紧急隐患",
            "color": "红色",
            "action": "立即现场复测并启动应急处置或修剪评估",
        }
    if value >= standard.major_hazard_index:
        return {
            "code": "major",
            "level": "重大隐患",
            "color": "黄色",
            "action": "纳入优先处置计划，缩短巡检周期并安排定向修剪",
        }
    if value >= standard.general_hazard_index:
        return {
            "code": "general",
            "level": "一般隐患",
            "color": "蓝色",
            "action": "加密监测，重点复核树线距离和生长趋势",
        }
    return {
        "code": "safe",
        "level": "安全",
        "color": "绿色",
        "action": "保持计划巡检和月度数据积累",
    }


def _record_clearance(record: dict[str, Any]) -> float | None:
    analysis = record.get("standard_analysis")
    if isinstance(analysis, dict) and "effective_clearance" in analysis:
        return _finite_number(
            "record.standard_analysis.effective_clearance",
            analysis["effective_clearance"],
            minimum=0,
        )
    for field in ("effective_clearance", "clearance_distance", "distance_to_wire"):
        if field in record:
            return _finite_number(f"record.{field}", record[field], minimum=0)
    return None


def _unavailable_history(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "index": 0.0,
        "record_count": 0,
        "distance_exceedance_rate": 0.0,
        "incident_rate": 0.0,
        "repeat_alert_rate": 0.0,
        "treatment_recurrence_rate": 0.0,
        "clearance_change_std": None,
        "counts": {
            "distance_exceedances": 0,
            "incidents": 0,
            "repeated_alerts": 0,
            "treatments": 0,
            "treatment_recurrences": 0,
        },
        "weights": {},
        "data_completeness": "Historical risk is unavailable.",
        "formula": "H = unavailable",
        "substitution": "H = 0.0000 (unavailable)",
        "model_status": MODEL_STATUS,
        "parameter_source": PARAMETER_SOURCE,
    }


def calculate_historical_risk(
    records: list[dict[str, Any]],
    standard: PredictionStandards,
    *,
    tree_id: int | None = None,
) -> dict[str, Any]:
    """Calculate H from exceedance, incidents, repeated alerts and recurrence.

    Missing incident/treatment fields stay missing and contribute zero.  The
    returned completeness notes make that limitation visible in the report.
    """
    _validated_standard(standard)
    if not isinstance(records, list):
        raise ValueError("records must be a list of history records")
    if tree_id in (None, "", 0):
        return _unavailable_history("missing_tree_id")
    tree_number = _finite_number("tree_id", tree_id, minimum=0)
    if not tree_number.is_integer() or tree_number == 0:
        raise ValueError(f"tree_id must be a non-zero integer, got {tree_id!r}")
    requested_tree_id = int(tree_number)

    usable: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("tree_id") in (None, ""):
            continue
        record_tree_number = _finite_number(
            "record.tree_id", record["tree_id"], minimum=0
        )
        if not record_tree_number.is_integer():
            raise ValueError(
                f"record.tree_id must be an integer, got {record['tree_id']!r}"
            )
        if int(record_tree_number) != requested_tree_id:
            continue
        usable.append(record)

    usable.sort(key=lambda item: str(item.get("observation_date") or item.get("created_at") or ""))
    if not usable:
        return _unavailable_history("no_matching_records")

    exceedance_count = 0
    incident_count = 0
    alert_flags: list[bool] = []
    treatment_positions: list[int] = []
    clearances: list[float] = []
    has_incident_field = False
    has_treatment_field = False

    for position, record in enumerate(usable):
        voltage = _finite_number(
            "record.voltage_level_kv",
            record.get("voltage_level_kv", 110),
            minimum=0,
        )
        minimum = standard.rules_for_voltage(voltage)["min_clearance_distance"]
        clearance = _record_clearance(record)
        if clearance is not None:
            clearances.append(clearance)
        analysis = record.get("standard_analysis")
        analysis_violation = bool(
            isinstance(analysis, dict)
            and any(
                item.get("field")
                in {
                    "vertical_distance",
                    "horizontal_distance",
                    "clearance_distance",
                    "wind_offset_distance",
                }
                for item in analysis.get("violations", [])
                if isinstance(item, dict)
            )
        )
        exceeded = analysis_violation or (clearance is not None and clearance < minimum)
        exceedance_count += int(exceeded)

        explicit_incident = None
        for field in ("hazard_occurred", "incident_occurred", "事故发生"):
            if field in record:
                has_incident_field = True
                explicit_incident = _truthy(record[field])
                break
        level = str(record.get("risk_level", ""))
        score = (
            _finite_number("record.risk_score", record["risk_score"], minimum=0)
            if "risk_score" in record
            else 0.0
        )
        derived_incident = level in {"高风险", "重大隐患", "紧急隐患"} or (
            score >= standard.high_risk_score
        )
        incident = explicit_incident if explicit_incident is not None else derived_incident
        incident_count += int(incident)
        alert_flags.append(
            exceeded
            or incident
            or level in {"中风险", "一般隐患"}
            or score >= standard.medium_risk_score
        )

        for field in ("treatment_performed", "handled", "已处置"):
            if field in record:
                has_treatment_field = True
                if _truthy(record[field]):
                    treatment_positions.append(position)
                break

    repeated_alerts = sum(
        1
        for previous, current in zip(alert_flags, alert_flags[1:], strict=False)
        if previous and current
    )
    treatment_recurrences = sum(
        1 for position in treatment_positions if any(alert_flags[position + 1 :])
    )
    count = len(usable)
    exceedance_rate = exceedance_count / count
    incident_rate = incident_count / count
    repeat_rate = repeated_alerts / max(1, count - 1)
    treatment_rate = treatment_recurrences / max(1, len(treatment_positions))
    weights = {
        "exceedance": standard.history_weight_exceedance,
        "incident": standard.history_weight_incident,
        "repeat": standard.history_weight_repeat,
        "treatment_recurrence": standard.history_weight_treatment_recurrence,
    }
    index = _clamp(
        weights["exceedance"] * exceedance_rate
        + weights["incident"] * incident_rate
        + weights["repeat"] * repeat_rate
        + weights["treatment_recurrence"] * treatment_rate
    )
    changes = [
        current - previous
        for previous, current in zip(clearances, clearances[1:], strict=False)
    ]
    clearance_change_std = statistics.stdev(changes) if len(changes) >= 2 else None
    completeness = []
    if not has_incident_field:
        completeness.append("未提供事故标记，事故率由高风险记录代估")
    if not has_treatment_field:
        completeness.append("未提供处置记录，处置后复发率按 0 计")
    if len(clearances) < 3:
        completeness.append("净空序列少于3期，区间使用模拟先验误差")
    return {
        "available": True,
        "reason": None,
        "index": round(index, 4),
        "record_count": count,
        "distance_exceedance_rate": round(exceedance_rate, 4),
        "incident_rate": round(incident_rate, 4),
        "repeat_alert_rate": round(repeat_rate, 4),
        "treatment_recurrence_rate": round(treatment_rate, 4),
        "clearance_change_std": (
            round(clearance_change_std, 4) if clearance_change_std is not None else None
        ),
        "counts": {
            "distance_exceedances": exceedance_count,
            "incidents": incident_count,
            "repeated_alerts": repeated_alerts,
            "treatments": len(treatment_positions),
            "treatment_recurrences": treatment_recurrences,
        },
        "weights": weights,
        "data_completeness": "；".join(completeness) or "历史字段完整",
        "formula": ("H = h₁×超标率 + h₂×事故率 + h₃×重复预警率 + h₄×处置后复发率"),
        "substitution": (
            f"H = {weights['exceedance']:.2f}×{exceedance_rate:.4f} + "
            f"{weights['incident']:.2f}×{incident_rate:.4f} + "
            f"{weights['repeat']:.2f}×{repeat_rate:.4f} + "
            f"{weights['treatment_recurrence']:.2f}×{treatment_rate:.4f} "
            f"= {index:.4f}"
        ),
        "model_status": MODEL_STATUS,
        "parameter_source": PARAMETER_SOURCE,
    }


def weather_risk(snapshot: dict[str, Any], standard: PredictionStandards) -> dict[str, Any]:
    _validated_standard(standard)
    if not isinstance(snapshot, dict):
        raise ValueError("weather snapshot must be a mapping")
    wind_speed = _finite_number("wind_speed", snapshot.get("wind_speed"), minimum=0)
    rainfall = _finite_number("rainfall", snapshot.get("rainfall"), minimum=0)
    temperature = _finite_number("temperature", snapshot.get("temperature"))
    sunlight_hours = _finite_number(
        "sunlight_hours", snapshot.get("sunlight_hours"), minimum=0
    )
    components = {
        "wind": _clamp(wind_speed / max(standard.wind_high_speed, 1e-9)),
        "rainfall": _clamp(rainfall / max(standard.rainfall_warning, 1e-9)),
        "temperature": _clamp(
            max(0.0, temperature)
            / max(standard.temperature_growth_threshold, 1e-9)
        ),
        "sunlight": _clamp(sunlight_hours / max(standard.sunlight_growth_threshold, 1e-9)),
    }
    weights = {
        "wind": standard.weather_weight_wind,
        "rainfall": standard.weather_weight_rainfall,
        "temperature": standard.weather_weight_temperature,
        "sunlight": standard.weather_weight_sunlight,
    }
    index = sum(weights[key] * components[key] for key in components)
    return {
        "index": round(_clamp(index), 4),
        "components": {key: round(value, 4) for key, value in components.items()},
        "weights": weights,
        "formula": "M = a₁×风速归一值 + a₂×降雨归一值 + a₃×温度归一值 + a₄×日照归一值",
        "substitution": (
            f"M = {weights['wind']:.2f}×{components['wind']:.4f} + "
            f"{weights['rainfall']:.2f}×{components['rainfall']:.4f} + "
            f"{weights['temperature']:.2f}×{components['temperature']:.4f} + "
            f"{weights['sunlight']:.2f}×{components['sunlight']:.4f} = {index:.4f}"
        ),
        "model_status": MODEL_STATUS,
        "parameter_source": PARAMETER_SOURCE,
    }


def comprehensive_risk(
    *,
    effective_clearance: float,
    safety_clearance: float,
    baseline_clearance: float,
    forecast_clearance: float,
    weather_snapshot: dict[str, Any],
    historical: dict[str, Any],
    standard: PredictionStandards,
) -> dict[str, Any]:
    """Return D/G/M/H, weighted R, four-level warning and full substitution."""
    _validated_standard(standard)
    effective_clearance = _finite_number(
        "effective_clearance", effective_clearance, minimum=0
    )
    safety_clearance = _finite_number(
        "safety_clearance", safety_clearance, minimum=0
    )
    baseline_clearance = _finite_number(
        "baseline_clearance", baseline_clearance, minimum=0
    )
    forecast_clearance = _finite_number(
        "forecast_clearance", forecast_clearance, minimum=0
    )
    if not isinstance(historical, dict):
        raise ValueError("historical must be a mapping")
    distance = _clamp(
        (safety_clearance - effective_clearance) / max(safety_clearance, 1e-9)
    )
    growth = _clamp(
        (baseline_clearance - forecast_clearance) / max(baseline_clearance, 1e-9)
    )
    weather = weather_risk(weather_snapshot, standard)
    history_index = _finite_number("historical.index", historical.get("index"), minimum=0)
    if history_index > 1:
        raise ValueError(f"historical.index must be <= 1, got {history_index}")
    factors = {
        "D": distance,
        "G": growth,
        "M": float(weather["index"]),
        "H": history_index,
    }
    weights = {
        "D": standard.risk_weight_distance,
        "G": standard.risk_weight_growth,
        "M": standard.risk_weight_weather,
        "H": standard.risk_weight_history,
    }
    contributions = {key: weights[key] * factors[key] for key in factors}
    index = _clamp(sum(contributions.values()))
    classification = hazard_level(index, standard)
    thresholds = [
        standard.general_hazard_index,
        standard.major_hazard_index,
        standard.emergency_hazard_index,
    ]
    if index >= standard.emergency_hazard_index:
        next_threshold = standard.emergency_hazard_index
        escalation_indicator = 1.0
    else:
        next_threshold = next(value for value in thresholds if value > index)
        escalation_indicator = 1 / (
            1 + math.exp(-(index - next_threshold) / 0.08)
        )
    return {
        "index": round(index, 4),
        "score": round(index * 100, 2),
        "hazard_level": classification["level"],
        "hazard_code": classification["code"],
        "warning_color": classification["color"],
        "action": classification["action"],
        "risk_escalation_indicator": round(escalation_indicator * 100, 1),
        "next_level_threshold": next_threshold,
        "factors": {key: round(value, 4) for key, value in factors.items()},
        "weights": weights,
        "contributions": {key: round(value, 4) for key, value in contributions.items()},
        "distance_formula": "D = max(0, (dₛ-dₜ)/dₛ)",
        "distance_substitution": (
            f"D = max(0, ({safety_clearance:.3f}-{effective_clearance:.3f})/"
            f"{safety_clearance:.3f}) = {distance:.4f}"
        ),
        "growth_formula": "G = max(0, (d当前-d预测)/d当前)",
        "growth_substitution": (
            f"G = max(0, ({baseline_clearance:.3f}-{forecast_clearance:.3f})/"
            f"{baseline_clearance:.3f}) = {growth:.4f}"
        ),
        "weather": weather,
        "history": historical,
        "formula": "R = w₁D + w₂G + w₃M + w₄H",
        "substitution": (
            f"R = {weights['D']:.2f}×{distance:.4f} + "
            f"{weights['G']:.2f}×{growth:.4f} + "
            f"{weights['M']:.2f}×{weather['index']:.4f} + "
            f"{weights['H']:.2f}×{history_index:.4f} = {index:.4f}"
        ),
        "escalation_formula": "升级情景指标 = sigmoid((R-下一等级阈值)/0.08)",
        "model_status": MODEL_STATUS,
        "parameter_source": PARAMETER_SOURCE,
    }


def clearance_interval(
    predicted_clearance: float,
    months: int,
    historical: dict[str, Any],
) -> dict[str, Any]:
    """Create a transparent 95% scenario interval around a distance forecast."""
    if not isinstance(historical, dict):
        raise ValueError("historical must be a mapping")
    predicted_clearance = max(
        0.0, _finite_number("predicted_clearance", predicted_clearance)
    )
    months = _finite_number("months", months, minimum=0)
    history_std = historical.get("clearance_change_std")
    if history_std is None:
        monthly_std = 0.08
        source = "模拟先验月误差 0.08m（历史期数不足）"
    else:
        monthly_std = max(
            0.03,
            _finite_number("historical.clearance_change_std", history_std, minimum=0),
        )
        source = f"历史月度净空变化标准差 {monthly_std:.4f}m"
    margin = 1.96 * monthly_std * math.sqrt(max(1, months))
    lower = max(0.0, predicted_clearance - margin)
    upper = max(lower, predicted_clearance + margin)
    return {
        "confidence": 0.95,
        "lower": round(lower, 3),
        "upper": round(upper, 3),
        "margin": round(margin, 3),
        "source": source,
        "formula": "95%区间 = 预测净空 ± 1.96×月变化标准差×√预测月数",
        "substitution": (
            f"{predicted_clearance:.3f} ± 1.96×{monthly_std:.4f}×√{months} "
            f"= [{lower:.3f}, {upper:.3f}] m"
        ),
        "model_status": MODEL_STATUS,
        "parameter_source": PARAMETER_SOURCE,
    }


def standard_comparisons(
    snapshot: dict[str, Any],
    standard: PredictionStandards,
    assessment: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build report rows pairing each input value with its selected standard."""
    _validated_standard(standard)
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be a mapping")
    rules = standard.rules_for_voltage(
        _snapshot_number(snapshot, "voltage_level_kv", default=110, minimum=0)
    )
    rows: list[dict[str, Any]] = []
    selection_active = bool(snapshot.get("_selection_active"))
    provided_fields = set(snapshot.get("_provided_fields", []))
    imputations = snapshot.get("_imputations", {})

    def add(
        category: str,
        field: str,
        label: str,
        actual: Any,
        standard_text: str,
        exceeded: bool = False,
    ) -> None:
        if selection_active and field not in provided_fields and field != "effective_clearance":
            imputation = imputations.get(field, {})
            used_value = imputation.get("used_value", snapshot.get(field))
            rows.append(
                {
                    "category": category,
                    "field": field,
                    "label": label,
                    "actual": f"未提供（模型采用 {used_value}）",
                    "standard": standard_text,
                    "status": "未提供/模型估计",
                    "exceeded": False,
                }
            )
            return
        rows.append(
            {
                "category": category,
                "field": field,
                "label": label,
                "actual": actual,
                "standard": standard_text,
                "status": "超出/不满足" if exceeded else "正常/仅记录",
                "exceeded": bool(exceeded),
            }
        )

    for field, label in (
        ("observation_date", "观测日期"),
        ("line_name", "线路名称"),
        ("tower_section", "杆塔区间"),
        ("gps_longitude", "GPS经度"),
        ("gps_latitude", "GPS纬度"),
    ):
        add("观测批次与位置", field, label, snapshot.get(field), "档案项，无统一数值阈值")
    for field, label, unit in (
        ("tree_id", "树木编号", ""),
        ("tree_code", "树木档案编号", ""),
        ("species", "树种", ""),
        ("height", "树高", "m"),
        ("age", "树龄", "年"),
        ("diameter", "胸径", "cm"),
        ("crown_width", "冠幅", "m"),
    ):
        actual = snapshot.get(field)
        add(
            "树木本身",
            field,
            label,
            f"{actual}{unit}" if unit else actual,
            "测量/档案项，无统一风险阈值",
        )
    slope = _snapshot_number(snapshot, "slope", default=0, minimum=0)
    add(
        "树木本身",
        "slope",
        "立地坡度",
        f"{slope:g}°",
        f"关注值 ≤ {standard.slope_warning_angle:g}°",
        slope > standard.slope_warning_angle,
    )
    voltage = _snapshot_number(snapshot, "voltage_level_kv", default=110, minimum=0)
    add(
        "电线本身",
        "voltage_level_kv",
        "电压等级",
        f"{voltage:g}kV",
        f"匹配 {rules['matched_voltage_level_kv']:g}kV 参数表",
    )
    for field, label in (
        ("maximum_sag_winter", "冬季最大弧垂"),
        ("maximum_sag_summer", "夏季最大弧垂"),
    ):
        actual = _snapshot_number(snapshot, field, default=0, minimum=0)
        add(
            "电线本身",
            field,
            label,
            f"{actual:g}m",
            f"≤ {rules['max_sag']:g}m",
            actual > rules["max_sag"],
        )
    distance_rules = {
        "vertical_distance": ("垂直距离", rules["min_vertical_distance"], ">="),
        "horizontal_distance": ("水平距离", rules["min_horizontal_distance"], ">="),
        "clearance_distance": ("净空距离", rules["min_clearance_distance"], ">="),
        "wind_offset_distance": ("风偏距离", standard.max_wind_offset_distance, "<="),
    }
    for field, (label, threshold, operator) in distance_rules.items():
        actual = _snapshot_number(
            snapshot,
            field,
            default=0,
            minimum=None if field == "temperature" else 0,
        )
        exceeded = actual < threshold if operator == ">=" else actual > threshold
        add("树线之间", field, label, f"{actual:g}m", f"{operator} {threshold:g}m", exceeded)
    weather_rules = (
        ("wind_speed", "风速", "m/s", standard.wind_warning_speed, ">="),
        ("rainfall", "降雨量", "mm", standard.rainfall_warning, ">="),
        ("humidity", "湿度", "%", 100.0, ">"),
        ("temperature", "温度", "℃", standard.temperature_growth_threshold, ">="),
        ("sunlight_hours", "日照时长", "h", standard.sunlight_growth_threshold, ">="),
    )
    for field, label, unit, threshold, operator in weather_rules:
        actual = float(snapshot.get(field, 0))
        exceeded = actual >= threshold if operator == ">=" else actual > threshold
        meaning = "风险/生长关注值" if field not in {"humidity"} else "有效范围上限"
        add(
            "气象条件",
            field,
            label,
            f"{actual:g}{unit}",
            f"{meaning} {operator} {threshold:g}{unit}",
            exceeded,
        )
    if assessment:
        effective = assessment.get("standard_analysis", {}).get("effective_clearance")
        if effective is not None:
            effective = _finite_number("effective_clearance", effective, minimum=0)
            add(
                "树线之间",
                "effective_clearance",
                "弧垂/风偏修正后有效净空",
                f"{effective:g}m",
                f">= {rules['min_clearance_distance']:g}m",
                effective < rules["min_clearance_distance"],
            )
    return rows


def calculation_ledger(
    snapshot: dict[str, Any],
    assessment: dict[str, Any],
    target: dict[str, Any],
    risk: dict[str, Any],
    standard: PredictionStandards,
) -> dict[str, Any]:
    """Collect formulas and exact substitutions shown below the report."""
    _validated_standard(standard)
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be a mapping")
    if (
        not isinstance(assessment, dict)
        or not isinstance(target, dict)
        or not isinstance(risk, dict)
    ):
        raise ValueError("assessment, target, and risk must be mappings")
    analysis = assessment.get("standard_analysis", {})
    measured = analysis.get("measured_distances", {})
    if not isinstance(analysis, dict) or not isinstance(measured, dict):
        raise ValueError("assessment.standard_analysis and measured_distances must be mappings")
    vertical = _finite_number(
        "vertical_distance",
        measured.get("vertical_distance", snapshot.get("vertical_distance")),
        minimum=0,
    )
    horizontal = _finite_number(
        "horizontal_distance",
        measured.get("horizontal_distance", snapshot.get("horizontal_distance")),
        minimum=0,
    )
    clearance = _finite_number(
        "clearance_distance",
        measured.get("clearance_distance", snapshot.get("clearance_distance")),
        minimum=0,
    )
    wind_offset = _finite_number(
        "wind_offset_distance",
        measured.get("wind_offset_distance", snapshot.get("wind_offset_distance")),
        minimum=0,
    )
    sag_adjustment = _finite_number(
        "sag_adjustment", analysis.get("sag_adjustment", 0), minimum=0
    )
    adjusted_vertical = max(0.0, vertical - sag_adjustment)
    adjusted_clearance = max(0.0, clearance - sag_adjustment)
    geometry = max(0.0, math.hypot(adjusted_vertical, horizontal) - wind_offset)
    effective = min(adjusted_clearance, geometry)
    rules = standard.rules_for_voltage(
        _snapshot_number(snapshot, "voltage_level_kv", minimum=0)
    )
    return {
        "point_cloud_distance_formulas": [
            {
                "name": "三维欧氏距离",
                "formula": "d₃D = √((x树-x线)²+(y树-y线)²+(z树-z线)²)",
                "data": f"点云最邻近结果（表单净空）= {clearance:.3f}m",
            },
            {
                "name": "水平距离",
                "formula": "dH = √((x树-x线)²+(y树-y线)²)",
                "data": f"dH = {horizontal:.3f}m；标准 ≥ {rules['min_horizontal_distance']:.3f}m",
            },
            {
                "name": "垂直距离",
                "formula": "dV = |z树-z线|",
                "data": f"dV = {vertical:.3f}m；标准 ≥ {rules['min_vertical_distance']:.3f}m",
            },
            {
                "name": "有效净空",
                "formula": "d有效=min(max(0,d净空-Δ弧垂), max(0,√((dV-Δ弧垂)²+dH²)-d风偏))",
                "data": (
                    f"min(max(0,{clearance:.3f}-{sag_adjustment:.3f}), "
                    f"max(0,√(({vertical:.3f}-{sag_adjustment:.3f})²+"
                    f"{horizontal:.3f}²)-{wind_offset:.3f})) = {effective:.3f}m"
                ),
            },
        ],
        "risk_formulas": [
            {
                "name": "距离风险 D",
                "formula": risk["distance_formula"],
                "data": risk["distance_substitution"],
            },
            {
                "name": "生长趋势 G",
                "formula": risk["growth_formula"],
                "data": risk["growth_substitution"],
            },
            {
                "name": "气象指数 M",
                "formula": risk["weather"]["formula"],
                "data": risk["weather"]["substitution"],
            },
            {
                "name": "历史隐患 H",
                "formula": risk["history"]["formula"],
                "data": risk["history"]["substitution"],
            },
            {"name": "综合风险 R", "formula": risk["formula"], "data": risk["substitution"]},
            {
                "name": "LSTM与规则融合分",
                "formula": "S融合 = 0.55×S_LSTM + 0.45×S规则",
                "data": (
                    f"S融合 = 0.55×{float(target.get('lstm_risk_score') or 0):.2f} + "
                    f"0.45×{float(target.get('rule_risk_score') or 0):.2f} = "
                    f"{float(target.get('risk_score') or 0):.2f}（辅助模型分；四级预警以 R 为准）"
                ),
            },
            {
                "name": "风险升级情景概率",
                "formula": risk["escalation_formula"],
                "data": f"升级情景指标 = {risk['risk_escalation_indicator']:.1f}",
            },
        ],
        "risk_level_intervals": [
            {"level": "安全", "range": f"R < {standard.general_hazard_index:g}", "color": "绿色"},
            {
                "level": "一般隐患",
                "range": f"{standard.general_hazard_index:g} ≤ R < {standard.major_hazard_index:g}",
                "color": "蓝色",
            },
            {
                "level": "重大隐患",
                "range": (
                    f"{standard.major_hazard_index:g} ≤ R < "
                    f"{standard.emergency_hazard_index:g}"
                ),
                "color": "黄色",
            },
            {
                "level": "紧急隐患",
                "range": f"R ≥ {standard.emergency_hazard_index:g}",
                "color": "红色",
            },
        ],
        "standard_comparisons": standard_comparisons(snapshot, standard, assessment),
        "selected_target_interval": target.get("clearance_interval"),
        "model_metadata": {
            "model_status": MODEL_STATUS,
            "parameter_source": PARAMETER_SOURCE,
        },
    }
