import json

import pytest
from algorithms.prediction.forecast import ForecastService
from algorithms.prediction.species_growth import SpeciesGrowthModel
from algorithms.prediction.standards import PredictionStandards


class FakeTemporalPredictor:
    def predict(self, records, species):
        return {"risk_score": 52.0}


@pytest.fixture
def growth_model_path(tmp_path):
    model_path = tmp_path / "species-growth.json"
    model_path.write_text(
        json.dumps(
            {
                "species_models": {
                    "其他": {
                        "height_coefficients": [0.02] + [0.0] * 11,
                        "diameter_coefficients": [0.01] + [0.0] * 11,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return model_path


@pytest.fixture
def growth_model(growth_model_path):
    return SpeciesGrowthModel(growth_model_path)


def test_species_growth_model_requires_explicit_path_or_environment(monkeypatch):
    monkeypatch.delenv("TREELINE_WARNING_SPECIES_GROWTH_MODEL_PATH", raising=False)
    with pytest.raises(
        RuntimeError, match="TREELINE_WARNING_SPECIES_GROWTH_MODEL_PATH"
    ):
        SpeciesGrowthModel()


def test_species_growth_model_path_priority_and_bad_model(growth_model_path, monkeypatch, tmp_path):
    environment_path = tmp_path / "environment-growth.json"
    environment_path.write_text(
        json.dumps({"species_models": {"其他": {}}}), encoding="utf-8"
    )
    monkeypatch.setenv("TREELINE_WARNING_SPECIES_GROWTH_MODEL_PATH", str(environment_path))

    assert SpeciesGrowthModel(growth_model_path).model_path == growth_model_path
    assert SpeciesGrowthModel().model_path == environment_path

    missing_path = tmp_path / "missing-growth.json"
    with pytest.raises(RuntimeError) as error:
        SpeciesGrowthModel(missing_path)
    assert str(missing_path) in str(error.value)
    assert error.value.__cause__ is not None

    broken_path = tmp_path / "broken-growth.json"
    broken_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(RuntimeError) as error:
        SpeciesGrowthModel(broken_path)
    assert str(broken_path) in str(error.value)
    assert error.value.__cause__ is not None

    incompatible_path = tmp_path / "incompatible-growth.json"
    incompatible_path.write_text(
        json.dumps({"species_models": []}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="incompatible-growth"):
        SpeciesGrowthModel(incompatible_path)


def test_forecast_accepts_growth_model_path(growth_model_path):
    service = ForecastService(FakeTemporalPredictor(), growth_model_path=growth_model_path)
    assert service._growth_model_for_prediction().model_path == growth_model_path


def test_short_and_long_forecast_contains_growth_and_wire_metrics(growth_model):
    service = ForecastService(FakeTemporalPredictor(), growth_model)
    snapshot = {
        "tree_id": 1,
        "species": "杨树",
        "height": 20,
        "age": 20,
        "diameter": 40,
        "slope": 10,
        "voltage_level_kv": 220,
        "maximum_sag_winter": 4,
        "maximum_sag_summer": 6,
        "observation_month": 5,
        "vertical_distance": 6,
        "horizontal_distance": 6,
        "clearance_distance": 5,
        "wind_offset_distance": 1,
        "wind_speed": 8,
        "rainfall": 20,
        "humidity": 60,
        "temperature": 24,
        "sunlight_hours": 9,
    }
    result = service.forecast(
        snapshot,
        PredictionStandards().to_dict(),
        short_horizons_days=[1, 7],
        long_horizons_months=[1, 2],
    )
    assert [item["period"] for item in result["short_term_forecast"]] == [
        "1天",
        "7天",
    ]
    assert [item["period"] for item in result["long_term_forecast"]] == [
        "1个月",
        "2个月",
    ]
    assert result["long_term_forecast"][-1]["predicted_height"] > snapshot["height"]
    assert (
        result["long_term_forecast"][-1]["predicted_clearance_distance"]
        < snapshot["clearance_distance"]
    )
    assert result["long_term_forecast"][0]["lstm_risk_score"] == 52
    assert "growth_trend" in result
    assert "risk_timeline_analysis" in result
    assert "maintenance_decision" in result["risk_timeline_analysis"]


def test_forecast_calendar_starts_from_observation_date_and_crosses_year(growth_model):
    service = ForecastService(FakeTemporalPredictor(), growth_model)
    snapshot = {
        "tree_id": 18,
        "species": "松树",
        "height": 16,
        "age": 25,
        "diameter": 36,
        "crown_width": 4.8,
        "slope": 6,
        "voltage_level_kv": 220,
        "maximum_sag_winter": 4,
        "maximum_sag_summer": 6,
        "observation_date": "2026-11-30",
        # A conflicting legacy month must not override the full date.
        "observation_month": 5,
        "vertical_distance": 7,
        "horizontal_distance": 7,
        "clearance_distance": 6,
        "wind_offset_distance": 1,
        "wind_speed": 7,
        "rainfall": 25,
        "humidity": 62,
        "temperature": 12,
        "sunlight_hours": 6,
        "gps_longitude": "117.1",
        "gps_latitude": "36.5",
    }
    result = service.forecast(
        snapshot,
        PredictionStandards().to_dict(),
        short_horizons_days=[30],
        long_horizons_months=[1, 2],
    )

    assert result["calendar_context"]["observation_date"] == "2026-11-30"
    assert result["calendar_context"]["observation_month"] == 11
    assert result["short_term_forecast"][0]["target_date"] == "2026-12-30"
    assert [
        item["target_date"] for item in result["long_term_forecast"]
    ] == ["2026-12-30", "2027-01-30"]
    assert [
        item["target_year_month"] for item in result["long_term_forecast"]
    ] == ["2026年12月", "2027年01月"]
    assert all(
        item["season"] == "冬季" for item in result["long_term_forecast"]
    )
    assert all(
        item["wire_condition"] == "冬季工况"
        for item in result["long_term_forecast"]
    )
    assert result["long_term_forecast"][-1]["predicted_age"] > snapshot["age"]
    assert "4 m" in result["treatment_advice"]["target_month_goal"]


def test_month_end_target_is_clamped_instead_of_skipping_february(growth_model):
    service = ForecastService(FakeTemporalPredictor(), growth_model)
    snapshot = {
        "tree_id": 19,
        "species": "柏树",
        "height": 12,
        "age": 30,
        "diameter": 28,
        "crown_width": 3.5,
        "slope": 4,
        "observation_date": "2026-01-31",
        "vertical_distance": 8,
        "horizontal_distance": 8,
        "clearance_distance": 7,
        "wind_offset_distance": 0.5,
        "wind_speed": 5,
        "rainfall": 10,
        "humidity": 55,
        "temperature": 5,
        "sunlight_hours": 5,
        "gps_longitude": "117.1",
        "gps_latitude": "36.5",
    }
    result = service.forecast(
        snapshot,
        PredictionStandards().to_dict(),
        short_horizons_days=[1],
        long_horizons_months=[1],
    )
    assert result["long_term_forecast"][0]["target_date"] == "2026-02-28"
    assert result["long_term_forecast"][0]["target_year_month"] == "2026年02月"


def test_selected_month_is_separate_from_full_risk_scan(growth_model):
    service = ForecastService(FakeTemporalPredictor(), growth_model)
    snapshot = {
        "tree_id": 20,
        "species": "杨树",
        "height": 15,
        "age": 18,
        "diameter": 30,
        "crown_width": 4,
        "slope": 5,
        "observation_date": "2026-01-15",
        "voltage_level_kv": 110,
        "maximum_sag_winter": 3.5,
        "maximum_sag_summer": 5,
        "vertical_distance": 8,
        "horizontal_distance": 8,
        "clearance_distance": 7,
        "wind_offset_distance": 0.5,
        "wind_speed": 5,
        "rainfall": 15,
        "humidity": 55,
        "temperature": 5,
        "sunlight_hours": 5,
    }
    result = service.forecast(
        snapshot,
        PredictionStandards().to_dict(),
        short_horizons_days=[1],
        long_horizons_months=list(range(1, 13)),
        selected_horizon_months=3,
    )

    assert result["calendar_context"]["selected_horizon_months"] == 3
    assert result["selected_target_prediction"]["horizon_months"] == 3
    assert result["growth_trend"]["forecast_months"] == 3
    assert result["calendar_context"]["risk_scan_months"] == 12
    assert len(result["long_term_forecast"]) == 12
    assert result["risk_timeline_analysis"]["peak_risk"]["horizon_months"] <= 12


def test_risk_timeline_gives_risk_month_and_trim_deadline():
    standard = PredictionStandards()
    result = ForecastService._risk_timeline_analysis(
        {
            "observation_date": "2026-01-15",
            "voltage_level_kv": 110,
            "clearance_distance": 4.2,
        },
        {
            "risk_level": "低风险",
            "risk_score": 20,
            "standard_analysis": {
                "effective_clearance": 4.2,
                "violations": [],
            },
        },
        [
            {
                "horizon_months": 1,
                "target_date": "2026-02-15",
                "target_year_month": "2026年02月",
                "risk_level": "低风险",
                "risk_score": 21,
                "predicted_clearance_distance": 3.8,
                "predicted_effective_clearance": 3.8,
            },
            {
                "horizon_months": 2,
                "target_date": "2026-03-15",
                "target_year_month": "2026年03月",
                "risk_level": "中风险",
                "risk_score": 43,
                "predicted_clearance_distance": 3.4,
                "predicted_effective_clearance": 3.4,
            },
            {
                "horizon_months": 3,
                "target_date": "2026-04-15",
                "target_year_month": "2026年04月",
                "risk_level": "高风险",
                "risk_score": 72,
                "predicted_clearance_distance": 3.2,
                "predicted_effective_clearance": 3.2,
            },
        ],
        standard,
    )

    assert result["first_significant_increase"]["target_year_month"] == "2026年03月"
    assert result["first_clearance_violation"]["target_year_month"] == "2026年03月"
    assert result["peak_risk"]["target_year_month"] == "2026年04月"
    decision = result["maintenance_decision"]
    assert decision["status"] == "计划修剪"
    assert decision["recommended_trim_month"] == "2026年02月"
    assert decision["trigger_month"] == "2026年03月"
    assert "最迟" in decision["recommended_deadline"]


def test_current_clearance_violation_requires_immediate_action():
    standard = PredictionStandards()
    result = ForecastService._risk_timeline_analysis(
        {
            "observation_date": "2026-07-20",
            "voltage_level_kv": 110,
            "clearance_distance": 3,
        },
        {
            "risk_level": "高风险",
            "risk_score": 80,
            "standard_analysis": {
                "effective_clearance": 3,
                "violations": [
                    {"field": "clearance_distance", "label": "净空距离"}
                ],
            },
        },
        [
            {
                "horizon_months": 1,
                "target_date": "2026-08-20",
                "target_year_month": "2026年08月",
                "risk_level": "高风险",
                "risk_score": 85,
                "predicted_clearance_distance": 2.9,
                "predicted_effective_clearance": 2.9,
            }
        ],
        standard,
    )
    decision = result["maintenance_decision"]
    assert decision["status"] == "立即处置"
    assert decision["recommended_trim_month"] == "2026年07月"
    assert "7天内" in decision["recommended_deadline"]
