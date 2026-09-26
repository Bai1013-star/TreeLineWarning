import json

import pytest
from algorithms.prediction.contact_risk import contact_risk
from algorithms.prediction.forecast import ForecastService
from algorithms.prediction.species_growth import SpeciesGrowthModel
from algorithms.prediction.standards import PredictionStandards
from api.prediction_api import predict_contact_risk


@pytest.mark.parametrize(
    ("target", "height", "line_height", "expected"),
    [
        ("未碰到", 12, 16, "安全"),
        ("碰到电线", 12, 16, "高风险"),
        ("树枝接触线路", 12, 16, "高风险"),
        ("树枝穿过线路", 12, 16, "危险"),
        ("碰到螺旋避雷针", 12, 16, "高风险"),
        ("碰到接地线", 12, 16, "高风险"),
        ("未碰到", 16.1, 16, "高风险"),
    ],
)
def test_contact_risk_levels(target, height, line_height, expected):
    result = contact_risk(
        contact_target=target,
        tree_height=height,
        conductor_height=line_height,
    )

    assert result["level"] == expected
    assert result["requires_immediate_pruning"] is (expected in {"危险", "高风险"})


@pytest.mark.parametrize(
    ("clearance", "expected"),
    [(0.8, "一般风险"), (3.0, "一般风险"), (5.0, "安全"), (8.0, "安全")],
)
def test_no_contact_uses_user_configurable_distances(clearance, expected):
    result = contact_risk(
        contact_target="未接触（按距离判断）",
        tree_height=12,
        conductor_height=16,
        clearance_distance=clearance,
        general_risk_distance=1.0,
        safe_distance=5.0,
    )

    assert result["level"] == expected
    assert result["thresholds"] == {"general_risk_max": 1.0, "safe_min": 5.0}


def test_physical_relation_has_priority_over_clearance():
    result = contact_risk(
        contact_target="树枝穿过线路",
        tree_height=12,
        conductor_height=16,
        clearance_distance=20,
        general_risk_distance=1,
        safe_distance=5,
    )

    assert result["code"] == "danger"
    assert "物理关系优先" in result["basis"]


def test_safe_distance_must_exceed_general_risk_distance():
    with pytest.raises(ValueError, match="安全距离必须大于一般风险距离"):
        contact_risk(
            contact_target="未碰到",
            tree_height=12,
            conductor_height=16,
            clearance_distance=2,
            general_risk_distance=5,
            safe_distance=5,
        )


def test_high_contact_risk_forces_immediate_pruning(tmp_path):
    growth_path = tmp_path / "growth.json"
    growth_path.write_text(
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
    service = ForecastService(
        temporal_predictor=None,
        growth_model=SpeciesGrowthModel(growth_path),
    )
    result = service.forecast(
        {
            "tree_id": 8,
            "species": "其他",
            "height": 12,
            "conductor_height": 16,
            "contact_target": "碰到接地线",
            "voltage_level_kv": 10,
            "clearance_distance": 6,
        },
        PredictionStandards().to_dict(),
        short_horizons_days=[1],
        long_horizons_months=[1],
        prediction_mode="formula",
    )

    assert result["current_assessment"]["contact_risk"]["level"] == "高风险"
    assert result["risk_timeline_analysis"]["maintenance_decision"]["status"] == "立即修剪"
    assert result["risk_timeline_analysis"]["maintenance_decision"]["requires_trimming"]


def test_contact_risk_can_be_recorded_without_forcing_an_upgrade(tmp_path):
    growth_path = tmp_path / "growth.json"
    growth_path.write_text(
        json.dumps(
            {
                "species_models": {
                    "其他": {
                        "height_coefficients": [0.0] * 12,
                        "diameter_coefficients": [0.0] * 12,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    service = ForecastService(None, SpeciesGrowthModel(growth_path))
    standard = PredictionStandards(
        use_actual_clearance_standard=True,
        actual_min_clearance_distance=0,
        use_contact_risk_override=False,
    )
    result = service.forecast(
        {
            "tree_id": 9,
            "species": "其他",
            "height": 12,
            "conductor_height": 16,
            "contact_target": "碰到接地线",
            "voltage_level_kv": 10,
            "clearance_distance": 0,
        },
        standard.to_dict(),
        short_horizons_days=[1],
        long_horizons_months=[1],
        prediction_mode="formula",
    )

    contact = result["current_assessment"]["contact_risk"]
    assert contact["level"] == "高风险"
    assert contact["applied"] is False
    assert contact["requires_immediate_pruning"] is False


def test_standalone_contact_risk_allows_omitting_annual_growth():
    result = predict_contact_risk(
        {
            "species": "榆树",
            "height": 14,
            "conductor_height": 16,
            "contact_target": "未碰到",
        }
    )

    assert result["risk_level"] == "安全"
    assert result["annual_height_growth_m"] is None
    assert result["projected_height_one_year_m"] is None
    assert result["one_year_over_line"] is False
