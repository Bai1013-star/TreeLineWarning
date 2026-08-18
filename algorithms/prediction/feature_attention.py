"""Configurable feature-attention prior and reportable input catalogue."""

from __future__ import annotations

from typing import Any

from algorithms.prediction.standards import PredictionStandards

FEATURE_CATALOG = [
    {
        "category": "当前树线空间特征",
        "features": [
            {"field": "horizontal_distance", "label": "最小水平距离", "source": "实测/点云"},
            {"field": "vertical_distance", "label": "最小垂直距离", "source": "实测/点云"},
            {
                "field": "clearance_distance",
                "label": "最小空间净距（三维净空）",
                "source": "实测/点云或几何计算",
            },
        ],
    },
    {
        "category": "树木属性特征",
        "features": [
            {"field": "species", "label": "树种", "source": "档案/选择"},
            {"field": "height", "label": "树高", "source": "实测"},
            {"field": "crown_width", "label": "冠幅", "source": "实测"},
            {"field": "age", "label": "树龄", "source": "档案/估测"},
            {"field": "growth_direction", "label": "生长方向", "source": "巡检选择"},
        ],
    },
    {
        "category": "历史变化特征",
        "features": [
            {"field": "historical_patrol_distance", "label": "历史巡检距离", "source": "历史记录"},
            {"field": "crown_expansion_speed", "label": "树冠扩展速度", "source": "历史序列/填写"},
            {"field": "pruning_record", "label": "修剪记录", "source": "历史记录/选择"},
        ],
    },
    {
        "category": "季节特征",
        "features": [
            {"field": "spring_stage", "label": "春季萌发期", "source": "观测月份派生"},
            {"field": "summer_stage", "label": "夏季旺长期", "source": "观测月份派生"},
            {"field": "dormancy_stage", "label": "秋冬休眠期", "source": "观测月份派生"},
        ],
    },
    {
        "category": "气象特征",
        "features": [
            {"field": "temperature", "label": "温度", "source": "月均值/填写"},
            {"field": "rainfall", "label": "降雨", "source": "月累计值/填写"},
            {"field": "humidity", "label": "湿度", "source": "月均值/填写"},
            {"field": "wind_speed", "label": "风速", "source": "月均值/填写"},
            {"field": "wind_direction", "label": "风向", "source": "气象记录/选择"},
        ],
    },
    {
        "category": "线路环境特征",
        "features": [
            {"field": "span_length", "label": "档距", "source": "线路档案/填写"},
            {"field": "conductor_height", "label": "导线高度", "source": "线路档案/实测"},
            {"field": "line_direction", "label": "线路走向", "source": "线路档案/选择"},
            {"field": "voltage_level_kv", "label": "电压等级", "source": "线路档案/选择"},
        ],
    },
]


ATTENTION_FEATURES = {
    "historical_clearance_distance": (
        "三维最短净距（历史值）",
        "最重要：历史距离是预测未来距离最直接的依据",
    ),
    "historical_horizontal_distance": (
        "水平净距（历史值）",
        "次重要：辅助判断横向侵入趋势",
    ),
    "historical_vertical_distance": (
        "垂直净距（历史值）",
        "次重要：辅助判断树高与弧垂方向的接近趋势",
    ),
    "monthly_temperature": ("月均温度", "重要：温度直接影响树木生长速率"),
    "monthly_rainfall": ("月降水量", "较重要：降水影响树木生长"),
    "monthly_sunlight_hours": ("月均日照时数", "较重要：日照影响光合作用"),
    "monthly_wind_speed": ("月均风速", "一般：影响树线放电风险但非生长主因"),
    **{
        f"month_s{month:02d}": (
            f"S_{month:02d}（{month}月独热编码）",
            "月份季节先验：旺长期较高、休眠期较低，并与其他月份共同作用",
        )
        for month in range(1, 13)
    },
}


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _distance_pressure(value: float, *, critical: float, safe: float) -> float:
    if value <= critical:
        return 1.0
    if value >= safe:
        return 0.0
    return _clamp((safe - value) / max(0.001, safe - critical))


def feature_attention_prior(
    snapshot: dict[str, Any],
    historical_records: list[dict[str, Any]],
    standards: PredictionStandards,
) -> dict[str, Any]:
    """Calculate a transparent configurable prior without replacing learned LSTM attention."""

    latest_history = historical_records[-1] if historical_records else {}
    rules = standards.rules_for_voltage(float(snapshot["voltage_level_kv"]))

    def historical_value(field: str, fallback: float) -> tuple[float, str]:
        value = latest_history.get(field)
        if value is None or value == "":
            return float(fallback), "当前观测回退值（尚无对应历史值）"
        return float(value), "最近一条历史记录"

    clearance, clearance_source = historical_value(
        "clearance_distance",
        snapshot.get("historical_patrol_distance", snapshot["clearance_distance"]),
    )
    horizontal, horizontal_source = historical_value(
        "horizontal_distance", snapshot["horizontal_distance"]
    )
    vertical, vertical_source = historical_value("vertical_distance", snapshot["vertical_distance"])
    month = int(snapshot["observation_month"])
    temperature = float(snapshot["temperature"])
    rainfall = float(snapshot["rainfall"])
    sunlight = float(snapshot["sunlight_hours"])
    wind = float(snapshot["wind_speed"])

    values: dict[str, tuple[float, float, str]] = {
        "historical_clearance_distance": (
            clearance,
            _distance_pressure(
                clearance,
                critical=rules["critical_clearance_distance"],
                safe=rules["min_clearance_distance"],
            ),
            clearance_source,
        ),
        "historical_horizontal_distance": (
            horizontal,
            _distance_pressure(
                horizontal,
                critical=rules["critical_clearance_distance"],
                safe=rules["min_horizontal_distance"],
            ),
            horizontal_source,
        ),
        "historical_vertical_distance": (
            vertical,
            _distance_pressure(
                vertical,
                critical=rules["critical_clearance_distance"],
                safe=rules["min_vertical_distance"],
            ),
            vertical_source,
        ),
        "monthly_temperature": (
            temperature,
            _clamp((temperature - 5.0) / max(1.0, standards.temperature_growth_threshold - 5.0)),
            "当前月份气象输入",
        ),
        "monthly_rainfall": (
            rainfall,
            _clamp(rainfall / max(1.0, standards.rainfall_warning)),
            "当前月份气象输入",
        ),
        "monthly_sunlight_hours": (
            sunlight,
            _clamp(sunlight / max(1.0, standards.sunlight_growth_threshold)),
            "当前月份气象输入",
        ),
        "monthly_wind_speed": (
            wind,
            _clamp(wind / max(1.0, standards.wind_warning_speed)),
            "当前月份气象输入",
        ),
    }
    for encoded_month in range(1, 13):
        active = 1.0 if month == encoded_month else 0.0
        values[f"month_s{encoded_month:02d}"] = (active, active, "月份独热编码")

    rows = []
    weighted_score = 0.0
    for field, (label, rationale) in ATTENTION_FEATURES.items():
        raw_value, signal, source = values[field]
        weight = float(standards.feature_attention_weights[field])
        contribution = weight * signal * 100.0
        weighted_score += contribution
        rows.append(
            {
                "field": field,
                "label": label,
                "weight": round(weight, 4),
                "raw_value": round(raw_value, 3),
                "normalized_signal": round(signal, 4),
                "contribution_points": round(contribution, 3),
                "source": source,
                "rationale": rationale,
            }
        )

    return {
        "score": round(weighted_score, 2),
        "weight_sum": round(sum(standards.feature_attention_weights.values()), 6),
        "formula": "P = 100 × Σ(可配置权重_i × 归一化风险信号_i)",
        "rows": rows,
        "configuration_source": "用户设置；初始值采用图片中的注意力权重示例",
        "learned_attention_note": (
            "该表是可解释的先验权重；LSTM 内部的近期生长、季节和树种注意力"
            "仍由训练数据学习，两者在长时辅助分中融合。"
        ),
    }
