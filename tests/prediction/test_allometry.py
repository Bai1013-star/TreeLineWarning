import pytest
from algorithms.prediction.allometry import (
    calculate_species_parameters,
    estimate_tree_age,
    formula_catalog_summary,
)
from api.prediction_api import create_app
from fastapi.testclient import TestClient


def _result_by_id(result, model_id):
    return next(item for item in result["results"] if item["model_id"] == model_id)


def test_french_plane_uses_beijing_direct_biomass_formula_and_alias():
    result = calculate_species_parameters(
        species="法国梧桐", diameter_cm=30, height_m=15, region="北京丰台"
    )

    model = _result_by_id(result, "BJ2025_XUANLINGMU_AGB_D2H")
    assert result["canonical_species"] == "悬铃木"
    assert model["estimate"] == pytest.approx(421.316, abs=0.001)
    assert model["proxy"] is False
    assert model["extrapolated"] is False


def test_white_ash_keeps_direct_and_proxy_models_distinct():
    result = calculate_species_parameters(species="白蜡", diameter_cm=30, height_m=12)

    direct = _result_by_id(result, "SHZ2026_白蜡_TOTAL")
    proxy = _result_by_id(result, "BJ2025_BAILASHU_PROXY_AGB_D2H")
    assert direct["target"] == "total_biomass"
    assert direct["estimate"] == pytest.approx(719.487, abs=0.001)
    assert proxy["proxy"] is True
    assert any("代理" in warning for warning in proxy["warnings"])


def test_jilin_richards_age_model_is_traceable_and_warns_for_extrapolation():
    result = calculate_species_parameters(species="榆树", age_years=30)

    age_model = _result_by_id(result, "JL2025_榆树_AGE_DBH")
    assert age_model["target"] == "dbh_from_age"
    assert age_model["estimate"] > 0
    assert age_model["input_complete"] is True

    extrapolated = calculate_species_parameters(species="榆树", age_years=2)
    model = _result_by_id(extrapolated, "JL2025_榆树_AGE_DBH")
    assert model["extrapolated"] is True


def test_tree_age_estimator_fuses_tree_height_and_dbh_for_direct_entry():
    result = estimate_tree_age(species="杨树", diameter_cm=30, height_m=15)

    assert result["status"] == "estimated"
    assert result["estimate_age_years"] == pytest.approx(11.0, abs=0.1)
    assert result["results"][0]["formula"].startswith("t = 0.60")
    assert "g_D" in result["results"][0]["formula_explanation"]
    assert "树高" in result["height_note"]

    plane = estimate_tree_age(species="法国梧桐", diameter_cm=30, height_m=15)
    assert plane["status"] == "estimated"
    assert plane["estimate_age_years"] == pytest.approx(10.2, abs=0.1)


@pytest.mark.parametrize("species", ["法国梧桐", "杨树", "榆树", "国槐", "白蜡"])
def test_main_fengtai_species_have_direct_height_and_dbh_age_estimates(species):
    result = estimate_tree_age(species=species, diameter_cm=30, height_m=15)

    assert result["status"] == "estimated"
    assert result["estimate_age_years"] is not None
    assert result["results"][0]["model_id"].startswith("EMPIRICAL_")


def test_formula_catalog_and_api_expose_sources_and_validation():
    summary = formula_catalog_summary()
    assert summary["formula_count"] >= 69
    assert "法国梧桐" in summary["species"]

    client = TestClient(create_app())
    assert client.get("/species/formulas").status_code == 200
    response = client.post(
        "/predict/species-parameters",
        json={"species": "国槐", "diameter_cm": 30, "height_m": 12, "region": "北京丰台"},
    )
    assert response.status_code == 200
    assert any(item["model_id"] == "BJ2025_GUOHUAI_AGB_D2H" for item in response.json()["results"])
    age_response = client.post(
        "/predict/tree-age", json={"species": "榆树", "diameter_cm": 30, "height_m": 15}
    )
    assert age_response.status_code == 200
    assert age_response.json()["status"] == "estimated"
    page = client.get("/").text
    assert "树高 + 胸径经验融合估算" in page
    assert 'id="estimateTreeAgeFromInputs"' in page
    assert 'id="treeAgeEstimator"' not in page
    assert "scheduleTreeAgeEstimate" in page
    assert "/predict/tree-age" in page
    assert "05 · 树线风险预测与定位" not in page
    assert 'id="inputLocationMap"' in page
    assert "calculateFormula" not in page
    assert (
        client.post(
            "/predict/species-parameters", json={"species": "国槐", "diameter_cm": 0}
        ).status_code
        == 422
    )
