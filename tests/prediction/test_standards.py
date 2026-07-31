import pytest
from algorithms.prediction.predictor import TreeInput, TreeRiskPredictor
from algorithms.prediction.standards import (
    PredictionStandards,
    analyze_distances,
    analyze_wire,
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
        PredictionStandards.from_mapping(
            {"wind_warning_speed": 20, "wind_high_speed": 10}
        )


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
