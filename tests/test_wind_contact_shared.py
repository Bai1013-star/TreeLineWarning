"""No fabricated DBH; all three forecast entry points share wind screening."""
import pytest
from desktop.wind_contact import estimate_wind_contact, attach_wind_contact_estimates
from fastapi.testclient import TestClient
from api.prediction_api import create_app
from core.database import Database
from tests.test_prediction_api import _write_growth_model


def test_missing_measured_dbh_is_not_replaced_with_default():
    result = estimate_wind_contact(species="油松", clearance_m=2, height_m=10,
                                  crown_width_m=3, diameter_cm=None)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert "beaufort_force" not in result


def test_known_chinese_pine_name_uses_pine_group():
    result = estimate_wind_contact(species="油松", clearance_m=.2, height_m=10,
                                  crown_width_m=3, diameter_cm=20)
    assert result["matched_species_group"] == "松树"
    assert result["beaufort_force"] >= 0
    assert result['beaufort_force_basis'] == 'GUST_SPEED_EQUIVALENT_ONLY'
    assert '不等于' in result['basis']
    assert '未逐树实测' in result['parameter_basis']
    assert len(result['references']) == 2


def test_branch_cannot_bend_beyond_its_length():
    result = estimate_wind_contact(species="油松", clearance_m=8, height_m=4,
                                  crown_width_m=1, diameter_cm=11)
    assert result['status'] == 'OUTSIDE_MODEL_RANGE'
    assert 'beaufort_force' not in result


def test_target_imputation_is_not_a_measurement():
    forecast = {"selected_target_prediction": dict(predicted_clearance_distance=1,
                predicted_height=11, predicted_crown_width=4, predicted_diameter=21)}
    attach_wind_contact_estimates(forecast, dict(species="油松", height=10,
                                 clearance_distance=2, crown_width=3, diameter=None))
    assert forecast["target_wind_contact_estimate"]["status"] == "INSUFFICIENT_EVIDENCE"


def test_deselected_diameter_must_not_be_used_for_wind():
    forecast = {}
    attach_wind_contact_estimates(forecast, dict(species="油松", height=10,
        clearance_distance=2, crown_width=3, diameter=20,
        selected_fields=["species", "height", "clearance_distance", "crown_width"]))
    assert forecast["wind_contact_estimate"]["status"] == "INSUFFICIENT_EVIDENCE"


def test_legacy_clearance_alias_remains_supported():
    forecast = {}
    attach_wind_contact_estimates(forecast, dict(species="油松", height=10,
        distance_to_wire=2, crown_width=3, diameter=20))
    assert forecast["wind_contact_estimate"]["required_crown_deflection_m"] == 2


@pytest.mark.parametrize("diameter", [20, None])
def test_manual_and_external_api_publish_same_wind_fields(tmp_path, diameter):
    client = TestClient(create_app(database=Database(tmp_path/"test.db"),
        species_growth_model_path=_write_growth_model(tmp_path/"growth.json")))
    response = client.post("/predict/forecast", json=dict(prediction_mode="formula",
        short_horizons_days=[1], long_horizons_months=[1], selected_horizon_months=1,
        tree_data=dict(tree_id=1, species="油松", height=10, crown_width=3, diameter=diameter,
                       observation_date="2026-10-02", voltage_level_kv=110, clearance_distance=.2)))
    assert response.status_code == 200, response.text
    result = response.json()
    expected = "SCREENING_ESTIMATE" if diameter else "INSUFFICIENT_EVIDENCE"
    assert result["wind_contact_estimate"]["status"] == expected
    assert "target_wind_contact_estimate" in result
