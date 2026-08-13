import math

import pytest
from algorithms.prediction.predictor import TreeInput, TreeRiskPredictor
from algorithms.prediction.risk_evaluation import (
    calculate_historical_risk,
    clearance_interval,
    comprehensive_risk,
    weather_risk,
)
from algorithms.prediction.standards import (
    PredictionStandards,
    analyze_distances,
    analyze_wire,
    classify_clearance_defect,
)


def test_distance_analysis_checks_all_four_distances():
    standard = PredictionStandards()
    analysis = analyze_distances(
        vertical_distance=3,
        horizontal_distance=6,
        clearance_distance=4,
        wind_offset_distance=3,
        standards=standard,
    )
    assert analysis["compliant"] is False
    assert {item["field"] for item in analysis["violations"]} == {
        "vertical_distance",
        "clearance_distance",
        "wind_offset_distance",
    }


def test_defaults_cover_220v_all_months_and_voltage_wind_clearance():
    standard = PredictionStandards()
    month_keys = {f"month_s{month:02d}" for month in range(1, 13)}

    assert month_keys <= standard.feature_attention_weights.keys()
    assert math.isclose(sum(standard.feature_attention_weights.values()), 1.0)
    assert standard.rules_for_voltage(0.22)["matched_voltage_level_kv"] == 0.22
    assert standard.rules_for_voltage(0.22)["min_wind_offset_distance"] == 1.0
    assert "min_wind_offset_distance" in standard.rules_for_voltage(220)


def test_voltage_wind_clearance_is_checked_after_offset():
    analysis = analyze_distances(
        vertical_distance=2,
        horizontal_distance=2,
        clearance_distance=3,
        wind_offset_distance=2,
        standards=PredictionStandards(),
        voltage_level_kv=10,
    )

    check = analysis["checks"]["wind_offset_clearance"]
    assert check["actual"] == pytest.approx(math.hypot(2, 2) - 2)
    assert check["standard"] == 1.5
    assert check["compliant"] is False


def test_old_partial_month_attention_is_migrated_to_all_twelve_months():
    old = PredictionStandards().to_dict()
    old["feature_attention_weights"] = {
        key: value
        for key, value in old["feature_attention_weights"].items()
        if key in {
            "historical_clearance_distance",
            "historical_horizontal_distance",
            "historical_vertical_distance",
            "monthly_temperature",
            "monthly_rainfall",
            "monthly_sunlight_hours",
            "monthly_wind_speed",
            "month_s01",
            "month_s02",
            "month_s05",
            "month_s06",
            "month_s07",
        }
    }

    migrated = PredictionStandards.from_mapping(old)
    assert all(
        f"month_s{month:02d}" in migrated.feature_attention_weights
        for month in range(1, 13)
    )
    assert math.isclose(sum(migrated.feature_attention_weights.values()), 1.0)


def test_custom_standard_changes_prediction():
    common = dict(
        species="杉树",
        height=18,
        age=20,
        diameter=45,
        wind_speed=8,
        rainfall=20,
        humidity=60,
        slope=10,
        vertical_distance=4.5,
        horizontal_distance=4.5,
        clearance_distance=4.5,
        wind_offset_distance=1.5,
    )
    strict = PredictionStandards(
        min_vertical_distance=6,
        min_horizontal_distance=6,
        min_clearance_distance=6,
        max_wind_offset_distance=1,
        voltage_standards={
            "110": {
                "min_vertical_distance": 6,
                "min_horizontal_distance": 6,
                "min_clearance_distance": 6,
                "max_sag": 4,
            }
        },
    )
    relaxed = PredictionStandards(
        min_vertical_distance=2,
        min_horizontal_distance=2,
        min_clearance_distance=2,
        max_wind_offset_distance=3,
        voltage_standards={
            "110": {
                "min_vertical_distance": 2,
                "min_horizontal_distance": 2,
                "min_clearance_distance": 2,
                "max_sag": 8,
            }
        },
    )
    predictor = TreeRiskPredictor()
    strict_result = predictor.predict(TreeInput(**common, standards=strict.to_dict()))
    relaxed_result = predictor.predict(TreeInput(**common, standards=relaxed.to_dict()))
    assert strict_result.risk_score > relaxed_result.risk_score
    assert len(strict_result.standard_analysis["violations"]) == 4
    assert relaxed_result.standard_analysis["compliant"] is True


def test_standard_validation_rejects_invalid_threshold_order():
    with pytest.raises(ValueError, match="高风速"):
        PredictionStandards.from_mapping({"wind_warning_speed": 20, "wind_high_speed": 10})


def test_voltage_rules_and_seasonal_sag():
    standard = PredictionStandards()
    summer = analyze_wire(
        voltage_level_kv=220,
        maximum_sag_winter=4,
        maximum_sag_summer=6,
        observation_month=7,
        standards=standard,
    )
    winter = analyze_wire(
        voltage_level_kv=220,
        maximum_sag_winter=4,
        maximum_sag_summer=6,
        observation_month=1,
        standards=standard,
    )
    assert summer["matched_voltage_level_kv"] == 220
    assert summer["current_sag"] == 6
    assert winter["current_sag"] == 4
    assert summer["compliant"] is True


def test_historical_risk_is_weighted_from_real_record_indicators():
    standard = PredictionStandards()
    history = calculate_historical_risk(
        [
            {
                "tree_id": 1,
                "observation_date": "2026-01-01",
                "clearance_distance": 3.0,
                "risk_level": "高风险",
            },
            {
                "tree_id": 1,
                "observation_date": "2026-02-01",
                "clearance_distance": 3.1,
                "hazard_occurred": True,
                "treatment_performed": True,
            },
            {
                "tree_id": 1,
                "observation_date": "2026-03-01",
                "clearance_distance": 3.2,
                "risk_score": 80,
            },
        ],
        standard,
        tree_id=1,
    )

    assert history["index"] == 1
    assert history["counts"] == {
        "distance_exceedances": 3,
        "incidents": 3,
        "repeated_alerts": 2,
        "treatments": 1,
        "treatment_recurrences": 1,
    }
    assert "H =" in history["substitution"]


def test_comprehensive_risk_exposes_all_factors_and_four_level_result():
    standard = PredictionStandards()
    history = {"index": 0.8}
    result = comprehensive_risk(
        effective_clearance=1.0,
        safety_clearance=3.5,
        baseline_clearance=5.0,
        forecast_clearance=2.0,
        weather_snapshot={
            "wind_speed": 15,
            "rainfall": 50,
            "temperature": 30,
            "sunlight_hours": 10,
        },
        historical=history,
        standard=standard,
    )

    assert set(result["factors"]) == {"D", "G", "M", "H"}
    assert result["factors"]["M"] == 1
    assert result["hazard_level"] in {
        "安全",
        "一般隐患",
        "重大隐患",
        "紧急隐患",
    }
    assert result["score"] == round(result["index"] * 100, 2)
    assert "R =" in result["substitution"]


def test_standard_validation_rejects_weight_sum_not_equal_to_one():
    with pytest.raises(ValueError, match="综合风险"):
        PredictionStandards.from_mapping({"risk_weight_history": 0.25})

    with pytest.raises(ValueError, match="finite"):
        PredictionStandards(risk_weight_distance=math.nan).validate()

    with pytest.raises(ValueError, match="图片示例特征注意力"):
        PredictionStandards.from_mapping(
            {"feature_attention_weights": {"historical_clearance_distance": 0.24}}
        )


@pytest.mark.parametrize(
    ("horizontal", "vertical", "clearance", "expected"),
    [
        (1.52, 0.93, 1.71, "危急缺陷"),
        (2.85, 1.67, 3.30, "安全"),
        (1.98, 1.12, 2.28, "严重缺陷"),
        (2.43, 1.38, 2.80, "一般缺陷"),
        (3.12, 2.05, 3.73, "安全"),
    ],
)
def test_10kv_clearance_defect_examples(horizontal, vertical, clearance, expected):
    result = classify_clearance_defect(
        horizontal_distance=horizontal,
        vertical_distance=vertical,
        clearance_distance=clearance,
        standards=PredictionStandards(),
        voltage_level_kv=10,
    )

    assert result["level"] == expected
    assert result["thresholds"] == {
        "critical_max": 1.0,
        "severe_max": 2.5,
        "safe_min": 3.0,
        "matched_voltage_level_kv": 10.0,
    }


def test_historical_risk_never_mixes_tree_records_or_missing_tree_id():
    standard = PredictionStandards()
    records = [
        {"tree_id": 1, "observation_date": "2026-03-01", "clearance_distance": 3.0},
        {"tree_id": 2, "observation_date": "2026-01-01", "clearance_distance": 99.0},
    ]

    isolated = calculate_historical_risk(records, standard, tree_id=1)
    assert isolated["available"] is True
    assert isolated["record_count"] == 1
    assert isolated["counts"]["distance_exceedances"] == 1

    unavailable = calculate_historical_risk(records, standard, tree_id=0)
    assert unavailable["available"] is False
    assert unavailable["reason"] == "missing_tree_id"
    assert unavailable["index"] == 0

    no_match = calculate_historical_risk(records, standard, tree_id=99)
    assert no_match["available"] is False
    assert no_match["reason"] == "no_matching_records"


def test_historical_risk_handles_empty_single_and_unordered_records():
    standard = PredictionStandards()
    empty = calculate_historical_risk([], standard, tree_id=1)
    assert empty["available"] is False
    assert empty["reason"] == "no_matching_records"

    single = calculate_historical_risk(
        [{"tree_id": 1, "observation_date": "2026-03-01", "clearance_distance": 4}],
        standard,
        tree_id=1,
    )
    assert single["available"] is True
    assert single["clearance_change_std"] is None

    unordered = calculate_historical_risk(
        [
            {"tree_id": 1, "observation_date": "2026-03-01", "clearance_distance": 4.0},
            {"tree_id": 1, "observation_date": "2026-01-01", "clearance_distance": 3.0},
            {"tree_id": 1, "observation_date": "2026-02-01", "clearance_distance": 3.5},
        ],
        standard,
        tree_id=1,
    )
    assert unordered["available"] is True
    assert unordered["clearance_change_std"] == 0


def test_comprehensive_risk_has_exact_weights_and_threshold_boundaries():
    standard = PredictionStandards()
    result = comprehensive_risk(
        effective_clearance=0,
        safety_clearance=4,
        baseline_clearance=4,
        forecast_clearance=2,
        weather_snapshot={"wind_speed": 0, "rainfall": 0, "temperature": 0, "sunlight_hours": 0},
        historical={"index": 0.5},
        standard=standard,
    )
    assert result["contributions"] == {"D": 0.4, "G": 0.125, "M": 0.0, "H": 0.075}
    assert result["index"] == 0.6
    assert result["model_status"] == "experimental"
    assert result["parameter_source"] == "project_default_unverified"

    boundary_standard = PredictionStandards(
        risk_weight_distance=0,
        risk_weight_growth=0,
        risk_weight_weather=0,
        risk_weight_history=1,
    )
    for index, code in ((0.20, "general"), (0.40, "major"), (0.70, "emergency")):
        boundary = comprehensive_risk(
            effective_clearance=4,
            safety_clearance=4,
            baseline_clearance=4,
            forecast_clearance=4,
            weather_snapshot={
                "wind_speed": 0,
                "rainfall": 0,
                "temperature": 0,
                "sunlight_hours": 0,
            },
            historical={"index": index},
            standard=boundary_standard,
        )
        assert boundary["hazard_code"] == code


@pytest.mark.parametrize(
    ("snapshot", "field"),
    [
        (
            {"wind_speed": math.nan, "rainfall": 0, "temperature": 0, "sunlight_hours": 0},
            "wind_speed",
        ),
        (
            {"wind_speed": math.inf, "rainfall": 0, "temperature": 0, "sunlight_hours": 0},
            "wind_speed",
        ),
        ({"wind_speed": None, "rainfall": 0, "temperature": 0, "sunlight_hours": 0}, "wind_speed"),
        ({"wind_speed": "bad", "rainfall": 0, "temperature": 0, "sunlight_hours": 0}, "wind_speed"),
        ({"wind_speed": -1, "rainfall": 0, "temperature": 0, "sunlight_hours": 0}, "wind_speed"),
    ],
)
def test_weather_risk_rejects_invalid_external_values(snapshot, field):
    with pytest.raises(ValueError, match=field):
        weather_risk(snapshot, PredictionStandards())


def test_clearance_interval_is_ordered_for_zero_negative_and_sparse_history():
    for predicted, months, history in (
        (0, 0, {}),
        (-2, 0, {}),
        (1, 3, {"clearance_change_std": None}),
        (1, 3, {"clearance_change_std": 0.1}),
    ):
        interval = clearance_interval(predicted, months, history)
        assert 0 <= interval["lower"] <= interval["upper"]
        assert interval["model_status"] == "experimental"

    with pytest.raises(ValueError, match="months"):
        clearance_interval(1, -1, {})
    with pytest.raises(ValueError, match="months"):
        clearance_interval(1, math.inf, {})
