import json
import threading
from pathlib import Path

import api.prediction_api as prediction_api
import pytest
from algorithms.prediction.species_growth import SpeciesGrowthModel
from algorithms.prediction.temporal_attention import (
    FEATURE_NAMES,
    NumpyAttentionLSTM,
    TemporalModelConfig,
    TemporalPredictor,
)
from api.prediction_api import normalize_prediction
from core.database import Database
from fastapi.testclient import TestClient


def _write_growth_model(path: Path) -> Path:
    species = ("松树", "杉树", "杨树", "柳树", "柏树", "樟树", "其他")
    models = {
        name: {
            "height_coefficients": [0.02] + [0.0] * 11,
            "diameter_coefficients": [0.01] + [0.0] * 11,
            "monthly_mean_height_growth": {"1": 0.01, "7": 0.03},
        }
        for name in species
    }
    path.write_text(
        json.dumps({"sample_count": 12, "species_models": models}), encoding="utf-8"
    )
    return path


def _write_temporal_model(path: Path) -> Path:
    model = NumpyAttentionLSTM(
        TemporalModelConfig(hidden_size=8, species_count=1, seed=9)
    )
    model.save(
        path,
        {
            "config": {
                "feature_count": len(FEATURE_NAMES),
                "hidden_size": 8,
                "species_count": 1,
                "seed": 9,
            },
            "normalization": {
                "mean": [0.0] * len(FEATURE_NAMES),
                "std": [1.0] * len(FEATURE_NAMES),
            },
            "species_map": {"其他": 0, "杨树": 0},
            "sequence_length": 2,
        },
    )
    return path


def _growth_model(tmp_path: Path) -> SpeciesGrowthModel:
    return SpeciesGrowthModel(_write_growth_model(tmp_path / "growth.json"))


def _temporal_records() -> list[dict[str, float]]:
    base = {
        "height": 12.0,
        "diameter": 20.0,
        "vertical_distance": 7.0,
        "horizontal_distance": 7.0,
        "clearance_distance": 6.0,
        "wind_offset_distance": 0.5,
        "voltage_level_kv": 110.0,
        "maximum_sag_winter": 3.0,
        "maximum_sag_summer": 4.0,
        "wind_speed": 5.0,
        "rainfall": 30.0,
        "humidity": 60.0,
        "temperature": 20.0,
        "sunlight_hours": 8.0,
    }
    return [base, {**base, "height": 12.1}]


def test_normalize_pipeline_list_result():
    raw = [
        {
            "tree_id": 1,
            "result": {
                "risk_level": "中风险",
                "risk_score": 60.0,
                "suggestion": "建议加强监测",
            },
        }
    ]
    assert normalize_prediction(raw) == {
        "risk_level": "中风险",
        "risk_score": 60,
        "suggestion": "建议加强监测",
    }


def test_normalize_pipeline_dict_formats():
    result = {"risk_level": "中风险", "risk_score": 60, "suggestion": "建议加强监测"}
    assert normalize_prediction(result) == result
    assert normalize_prediction({"results": [{"result": result}]}) == result


def test_homepage_disables_cache_and_links_observation_month():
    client = TestClient(prediction_api.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "no-store" in response.headers["cache-control"]
    html = response.text
    assert "id=\"observation_year\"" not in html  # Fields are rendered by JavaScript.
    assert "observation_month" in html
    assert "field-standard" in html
    assert "s_risk_weight_history" in html
    assert "逐项标准对照" in html
    assert "公式、系数与逐步代入" in html
    assert "risk_escalation_indicator" in html
    assert "risk_escalation_probability" not in html
    assert "专利点云预测技术对照" not in html
    assert (
        "$('observation_month').addEventListener('change'"
        in html
    )
    assert "renderTargetMonths()" in html
    assert "历史数据与 LSTM 增量训练" in html
    assert "accept=\".xlsx,.xlsm,.csv,.tsv,.txt,.json\"" in html
    assert "id=\"historyPaste\"" in html
    assert "id=\"trainHistory\"" in html
    assert "id=\"selectAllOptional\"" in html
    assert "id=\"selectCoreOnly\"" in html
    assert "data-use-field" in html
    assert "未提供项不会作为实测值保存" in html


def test_create_app_defers_database_creation(tmp_path: Path):
    database_path = tmp_path / "deferred.db"
    app = prediction_api.create_app(database_path=database_path)

    assert not database_path.exists()
    assert TestClient(app).get("/health").status_code == 200
    assert not database_path.exists()


def test_create_app_uses_injected_database(tmp_path: Path):
    database_path = tmp_path / "injected.db"
    db = Database(database_path)
    app = prediction_api.create_app(database=db)

    response = TestClient(app).get("/settings/standards")

    assert response.status_code == 200
    assert database_path.exists()
    db.close()


def test_create_app_keeps_database_paths_isolated(tmp_path: Path):
    first_path = tmp_path / "first.db"
    second_path = tmp_path / "second.db"
    first_app = prediction_api.create_app(database_path=first_path)
    second_app = prediction_api.create_app(database_path=second_path)

    with TestClient(first_app) as first_client, TestClient(second_app) as second_client:
        first_settings = first_client.get("/settings/standards").json()
        first_settings["profile_name"] = "first"
        assert first_client.put("/settings/standards", json=first_settings).status_code == 200

        assert second_client.get("/settings/standards").json()["profile_name"] != "first"
        assert first_app.state.database is not second_app.state.database


def test_injected_database_has_path_and_environment_priority(tmp_path: Path, monkeypatch):
    injected = Database(tmp_path / "injected.db")
    configured_path = tmp_path / "configured.db"
    environment_path = tmp_path / "environment.db"
    monkeypatch.setenv("TREELINE_WARNING_DATABASE_PATH", str(environment_path))
    app = prediction_api.create_app(database=injected, database_path=configured_path)

    with TestClient(app) as client:
        assert client.get("/settings/standards").status_code == 200

    assert app.state.database is injected
    assert app.state.database_owned is False
    assert not configured_path.exists()
    assert not environment_path.exists()
    assert injected.get_standards()["profile_name"]
    injected.close()


def test_database_path_has_environment_priority(tmp_path: Path, monkeypatch):
    configured_path = tmp_path / "configured.db"
    environment_path = tmp_path / "environment.db"
    monkeypatch.setenv("TREELINE_WARNING_DATABASE_PATH", str(environment_path))
    app = prediction_api.create_app(database_path=configured_path)

    with TestClient(app) as client:
        assert client.get("/settings/standards").status_code == 200
        owned_database = app.state.database

    assert configured_path.exists()
    assert not environment_path.exists()
    assert app.state.database_owned is True
    with pytest.raises(RuntimeError, match="Database is closed"):
        owned_database.get_standards()


def test_unaccessed_app_shutdown_does_not_create_database(tmp_path: Path):
    database_path = tmp_path / "unaccessed.db"
    app = prediction_api.create_app(database_path=database_path)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert not database_path.exists()
    assert app.state.database is None


def test_injected_database_remains_open_after_app_shutdown(tmp_path: Path):
    db = Database(tmp_path / "injected.db")
    app = prediction_api.create_app(database=db)

    with TestClient(app) as client:
        assert client.get("/settings/standards").status_code == 200

    assert db.get_standards()["profile_name"]
    db.close()
    db.close()
    with pytest.raises(RuntimeError, match="Database is closed"):
        db.get_standards()


def test_database_stores_only_result_strings(tmp_path: Path):
    db = Database(tmp_path / "history.db")
    db.save_prediction(
        {
            "tree_id": 1,
            "risk_level": "中风险",
            "suggestion": "建议加强监测",
        }
    )
    row = db.get_predictions()[0]
    assert row["risk_level"] == "中风险"
    assert row["prediction"] == "建议加强监测"
    db.close()


def test_database_reads_legacy_dict_without_exposing_dict(tmp_path: Path):
    db = Database(tmp_path / "legacy.db")
    db.conn.execute(
        """
        INSERT INTO prediction_history (risk_level, prediction)
        VALUES (?, ?)
        """,
        (
            "{'risk_level': '中风险', 'suggestion': '建议加强监测'}",
            "{'risk_level': '中风险', 'suggestion': '建议加强监测'}",
        ),
    )
    db.conn.commit()
    row = db.get_predictions()[0]
    assert row["risk_level"] == "中风险"
    assert row["prediction"] == "建议加强监测"
    db.close()


def test_training_dataset_and_run_are_persisted(tmp_path: Path):
    db = Database(tmp_path / "training.db")
    parsed = {
        "source_name": "history.xlsx",
        "source_format": "xlsx",
        "sheet_name": "巡检",
        "row_count": 3,
        "valid_row_count": 3,
        "invalid_row_count": 0,
        "tree_count": 1,
        "eligible_tree_count": 1,
        "trainable_sequence_count": 2,
        "records": [{"tree_id": "1", "risk_score": 40}],
        "preview": [{"tree_id": "1", "risk_score": 40}],
        "errors": [],
        "trainable": True,
        "requirements": [],
    }
    dataset_id = db.save_training_dataset(parsed)
    run_id = db.save_training_run(
        dataset_id,
        {
            "status": "not_deployed",
            "deployed": False,
            "before_validation_mae": 6.2,
            "after_validation_mae": 6.5,
        },
    )

    assert db.get_training_dataset(dataset_id)["records"][0]["tree_id"] == "1"
    assert db.list_training_datasets()[0]["source_name"] == "history.xlsx"
    assert db.list_training_runs()[0]["id"] == run_id
    assert db.list_training_runs()[0]["deployed"] is False
    db.close()


def test_history_upload_and_training_api(tmp_path: Path, monkeypatch):
    db = Database(tmp_path / "training-api.db")
    source_model = _write_temporal_model(tmp_path / "source.npz")
    output_dir = tmp_path / "candidates"
    client = TestClient(
        prediction_api.create_app(
            database=db,
            temporal_model_path=source_model,
            training_output_dir=output_dir,
        )
    )
    csv_text = (
        "tree_id,species,observation_date,height,diameter,vertical_distance,"
        "horizontal_distance,clearance_distance,wind_offset_distance,"
        "voltage_level_kv,maximum_sag_winter,maximum_sag_summer,wind_speed,"
        "rainfall,humidity,temperature,sunlight_hours,risk_score\n"
        "1,杨树,2026-01-15,17.2,33.2,6.4,6.2,5.4,0.8,110,3.5,5,6,20,60,5,5,32\n"
        "1,杨树,2026-02-15,17.22,33.24,6.39,6.19,5.39,0.8,110,3.5,5,7,28,62,8,6,35\n"
        "1,杨树,2026-03-15,17.26,33.31,6.37,6.18,5.37,0.9,110,3.5,5,9,42,66,14,7,43"
    )
    imported = client.post(
        "/training/import/file",
        files={"file": ("history.csv", csv_text.encode("utf-8"), "text/csv")},
    )
    assert imported.status_code == 200
    dataset_id = imported.json()["dataset_id"]
    assert imported.json()["trainable"] is True
    assert "records" not in imported.json()

    def fake_fine_tune(
        records, *, model_path, output_path, epochs, learning_rate, allow_replace
    ):
        assert len(records) == 3
        assert model_path == source_model
        assert output_path.parent == output_dir
        assert epochs == 5
        assert learning_rate == 0.001
        assert allow_replace is False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"candidate")
        return {
            "status": "deployed",
            "deployed": True,
            "before_validation_mae": 3.2,
            "after_validation_mae": 3.1,
            "sequence_samples": 2,
        }

    monkeypatch.setattr(prediction_api, "fine_tune_with_history", fake_fine_tune)
    trained = client.post(
        f"/training/train/{dataset_id}",
        json={"epochs": 5, "learning_rate": 0.001},
    )
    assert trained.status_code == 200
    assert trained.json()["status"] == "deployed"
    assert trained.json()["candidate_model_name"].endswith(".npz")
    assert not source_model.with_suffix(".report.json").exists()
    history = client.get("/training/history").json()
    assert history["datasets"][0]["id"] == dataset_id
    assert history["runs"][0]["metrics"]["after_validation_mae"] == 3.1
    db.close()


def test_training_template_download():
    client = TestClient(prediction_api.app)
    response = client.get("/training/template.csv")
    assert response.status_code == 200
    assert response.content.startswith(b"\xef\xbb\xbf")
    assert "attachment" in response.headers["content-disposition"]


def test_predict_and_history_contract(tmp_path: Path):
    db = Database(tmp_path / "api.db")
    raw = [
        {
            "tree_id": 1,
            "result": {
                "risk_level": "中风险",
                "risk_score": 60.0,
                "suggestion": "建议加强监测",
            },
        }
    ]
    class FakePipeline:
        def predict(self, _):
            return raw

    client = TestClient(
        prediction_api.create_app(database=db, prediction_pipeline=FakePipeline())
    )

    response = client.post("/predict", json={"tree_id": 1, "species": "松树"})
    assert response.status_code == 200
    assert response.json() == {
        "risk_level": "中风险",
        "risk_score": 60,
        "suggestion": "建议加强监测",
    }
    saved = client.get("/history").json()["history"][0]
    assert saved["risk_level"] == "中风险"
    assert saved["prediction"] == "建议加强监测"
    db.close()


def test_temporal_prediction_endpoint_contract(tmp_path: Path):
    db = Database(tmp_path / "temporal-api.db")

    class FakeTemporalPredictor:
        def predict(self, records, species):
            assert len(records) == 2
            assert species == "杨树"
            return {
                "risk_level": "高风险",
                "risk_score": 82.5,
                "suggestion": "建议立即巡检并评估修剪或清障",
                "model": "numpy_lstm_triple_attention",
                "attention": {"time": [0.2, 0.8]},
            }

    client = TestClient(
        prediction_api.create_app(database=db, temporal_predictor=FakeTemporalPredictor())
    )
    response = client.post(
        "/predict/temporal",
        json={
            "tree_id": 8,
            "species": "杨树",
            "records": _temporal_records(),
        },
    )
    assert response.status_code == 200
    assert response.json()["risk_level"] == "高风险"
    saved = db.get_predictions()[0]
    assert saved["risk_level"] == "高风险"
    assert saved["prediction"] == "建议立即巡检并评估修剪或清障"
    db.close()


def test_standard_settings_round_trip_and_pipeline_injection(tmp_path: Path):
    db = Database(tmp_path / "standards-api.db")
    received = {}

    def fake_predict(payload):
        received.update(payload)
        return [
            {
                "result": {
                    "risk_level": "中风险",
                    "risk_score": 55,
                    "suggestion": "建议加强监测",
                }
            }
        ]

    class FakePipeline:
        def predict(self, payload):
            return fake_predict(payload)

    client = TestClient(
        prediction_api.create_app(database=db, prediction_pipeline=FakePipeline())
    )
    settings = client.get("/settings/standards").json()
    settings.update(
        {
            "profile_name": "华东园区标准",
            "region": "华东",
            "industry": "工业园区",
            "min_clearance_distance": 7,
        }
    )
    response = client.put("/settings/standards", json=settings)
    assert response.status_code == 200
    assert response.json()["min_clearance_distance"] == 7

    response = client.post(
        "/predict",
        json={
            "tree_id": 1,
            "vertical_distance": 8,
            "horizontal_distance": 8,
            "clearance_distance": 8,
            "wind_offset_distance": 1,
        },
    )
    assert response.status_code == 200
    assert received["standards"]["profile_name"] == "华东园区标准"
    assert received["standards"]["min_clearance_distance"] == 7
    db.close()


def test_database_persists_distance_and_standard_analysis(tmp_path: Path):
    db = Database(tmp_path / "distance-history.db")
    db.save_prediction(
        {
            "tree_id": 9,
            "vertical_distance": 3,
            "horizontal_distance": 7,
            "clearance_distance": 4,
            "wind_offset_distance": 2.5,
            "voltage_level_kv": 220,
            "maximum_sag_winter": 4,
            "maximum_sag_summer": 6,
            "observation_date": "2026-07-15",
            "observation_year": 2026,
            "observation_month": 7,
            "risk_level": "高风险",
            "suggestion": "立即巡检",
            "standard_profile": {
                "profile_name": "测试标准",
                "region": "测试地区",
                "industry": "测试行业",
            },
            "standard_analysis": {
                "compliant": False,
                "violations": [{"field": "clearance_distance"}],
            },
        }
    )
    row = db.get_predictions()[0]
    assert row["vertical_distance"] == 3
    assert row["horizontal_distance"] == 7
    assert row["clearance_distance"] == 4
    assert row["wind_offset_distance"] == 2.5
    assert row["voltage_level_kv"] == 220
    assert row["maximum_sag_winter"] == 4
    assert row["maximum_sag_summer"] == 6
    assert row["observation_date"] == "2026-07-15"
    assert row["observation_year"] == 2026
    assert row["standard_profile"]["profile_name"] == "测试标准"
    assert row["standard_analysis"]["compliant"] is False
    db.close()


def test_forecast_endpoint_returns_selectable_horizons(tmp_path: Path):
    db = Database(tmp_path / "forecast-api.db")

    class FakeTemporalPredictor:
        def predict(self, records, species):
            return {"risk_score": 50}

    client = TestClient(
        prediction_api.create_app(
            database=db,
            temporal_predictor=FakeTemporalPredictor(),
            species_growth_model=_growth_model(tmp_path),
        )
    )
    response = client.post(
        "/predict/forecast",
        json={
            "tree_data": {
                "tree_id": 2,
                "species": "杨树",
                "height": 18,
                "age": 15,
                "diameter": 35,
                "slope": 8,
                "voltage_level_kv": 110,
                "maximum_sag_winter": 3.5,
                "maximum_sag_summer": 5,
                "observation_month": 6,
                "vertical_distance": 6,
                "horizontal_distance": 6,
                "clearance_distance": 5,
                "wind_offset_distance": 1,
                "wind_speed": 8,
                "rainfall": 20,
                "humidity": 60,
                "temperature": 24,
                "sunlight_hours": 9,
            },
            "short_horizons_days": [1, 14],
            "long_horizons_months": [1, 2],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert [item["period"] for item in body["short_term_forecast"]] == ["1天", "14天"]
    assert [item["period"] for item in body["long_term_forecast"]] == [
        "1个月",
        "2个月",
    ]
    assert body["growth_trend"]["height_change"] > 0
    assert body["report_type"] == "配电线路树线隐患智能预警工单"
    assert body["calendar_context"]["selected_horizon_months"] == 2
    assert body["selected_target_prediction"]["period"] == "2个月"
    assert body["work_order"]["work_order_id"].startswith("TLW-")
    assert body["historical_context"]["status"] == "等待用户提供历史数据"
    assert body["model_context"]["growth_model"].startswith("分树种月度")
    assert "risk_timeline_analysis" in body
    snapshot = db.get_predictions()[0]["risk_snapshot"]
    assert snapshot["model_status"] == "experimental"
    assert snapshot["parameter_source"] == "project_default_unverified"
    db.close()


def test_forecast_endpoint_persists_unchecked_fields_as_null(tmp_path: Path):
    db = Database(tmp_path / "optional-inputs.db")

    class FakeTemporalPredictor:
        def predict(self, records, species):
            return {"risk_score": 45}

    selected_fields = [
        "observation_year",
        "observation_month",
        "observation_day",
        "species",
        "height",
        "voltage_level_kv",
        "clearance_distance",
    ]
    response = TestClient(
        prediction_api.create_app(
            database=db,
            temporal_predictor=FakeTemporalPredictor(),
            species_growth_model=_growth_model(tmp_path),
        )
    ).post(
        "/predict/forecast",
        json={
            "tree_data": {
                "selected_fields": selected_fields,
                "observation_year": 2026,
                "observation_month": 1,
                "observation_day": 15,
                "species": "杨树",
                "height": 18,
                "voltage_level_kv": 110,
                "clearance_distance": 5,
            },
            "short_horizons_days": [1],
            "long_horizons_months": [1],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["observed_input_snapshot"]["gps_longitude"] is None
    assert body["input_availability"]["optional_omitted_count"] > 0
    history_row = db.get_predictions()[0]
    assert history_row["diameter"] is None
    assert history_row["clearance_distance"] == 5
    db.close()


def test_database_returns_tree_history_in_chronological_order(tmp_path: Path):
    db = Database(tmp_path / "tree-history.db")
    db.save_prediction(
        {
            "tree_id": 3,
            "height": 10,
            "clearance_distance": 8,
            "risk_level": "低风险",
            "suggestion": "首次",
        }
    )
    db.save_prediction(
        {
            "tree_id": 99,
            "height": 20,
            "clearance_distance": 2,
            "risk_level": "高风险",
            "suggestion": "其他树",
        }
    )
    db.save_prediction(
        {
            "tree_id": 3,
            "height": 10.2,
            "clearance_distance": 7.8,
            "risk_level": "低风险",
            "suggestion": "第二次",
        }
    )
    rows = db.get_predictions_by_tree(3)
    assert [row["prediction"] for row in rows] == ["首次", "第二次"]
    assert [row["height"] for row in rows] == [10, 10.2]
    db.close()


def test_species_growth_model_is_trained_for_all_species(tmp_path: Path):
    model = _growth_model(tmp_path)
    assert set(model.models) == {"松树", "杉树", "杨树", "柳树", "柏树", "樟树", "其他"}
    report = model.metadata
    assert report["sample_count"] == 12
    poplar = report["species_models"]["杨树"]["monthly_mean_height_growth"]
    assert poplar["7"] > poplar["1"]
    height_growth, diameter_growth = model.predict(
        "杨树",
        month=7,
        temperature=28,
        sunlight_hours=10,
        rainfall=70,
        humidity=72,
        age=20,
        height=18,
    )
    assert height_growth > 0
    assert diameter_growth > 0


def test_forecast_explicit_history_is_fused(tmp_path: Path):
    db = Database(tmp_path / "history-fusion.db")

    class RecordingTemporalPredictor:
        def __init__(self):
            self.record_counts = []

        def predict(self, records, species):
            self.record_counts.append(len(records))
            return {"risk_score": 48}

    temporal = RecordingTemporalPredictor()
    client = TestClient(
        prediction_api.create_app(
            database=db,
            temporal_predictor=temporal,
            species_growth_model=_growth_model(tmp_path),
        )
    )
    base = {
        "tree_id": 12,
        "species": "杨树",
        "height": 15,
        "age": 18,
        "diameter": 30,
        "slope": 5,
        "vertical_distance": 8,
        "horizontal_distance": 8,
        "clearance_distance": 7,
        "wind_offset_distance": 1,
        "voltage_level_kv": 110,
        "maximum_sag_winter": 3,
        "maximum_sag_summer": 4.5,
        "observation_month": 7,
        "wind_speed": 6,
        "rainfall": 45,
        "humidity": 65,
        "temperature": 27,
        "sunlight_hours": 9,
    }
    history = [
        {**base, "height": 14.6, "clearance_distance": 7.4, "observation_month": 5},
        {**base, "height": 14.8, "clearance_distance": 7.2, "observation_month": 6},
    ]
    response = client.post(
        "/predict/forecast",
        json={
            "tree_data": base,
            "history_records": history,
            "short_horizons_days": [7],
            "long_horizons_months": [1, 2, 3],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["historical_context"]["status"] == "已启用"
    assert body["historical_context"]["record_count"] == 2
    assert temporal.record_counts[0] == 4
    assert temporal.record_counts[-1] == 6
    db.close()


def _training_dataset(db: Database) -> int:
    return db.save_training_dataset(
        {
            "source_name": "history.csv",
            "source_format": "csv",
            "row_count": 2,
            "valid_row_count": 2,
            "invalid_row_count": 0,
            "tree_count": 1,
            "trainable_sequence_count": 1,
            "records": [{"tree_id": "1"}, {"tree_id": "1"}],
        }
    )


def test_app_model_configuration_is_isolated_and_injected_models_win(
    tmp_path: Path, monkeypatch
):
    explicit_temporal = _write_temporal_model(tmp_path / "explicit-temporal.npz")
    environment_temporal = _write_temporal_model(tmp_path / "environment-temporal.npz")
    explicit_growth = _write_growth_model(tmp_path / "explicit-growth.json")
    environment_growth = _write_growth_model(tmp_path / "environment-growth.json")
    monkeypatch.setenv("TREELINE_WARNING_TEMPORAL_MODEL_PATH", str(environment_temporal))
    monkeypatch.setenv(
        "TREELINE_WARNING_SPECIES_GROWTH_MODEL_PATH", str(environment_growth)
    )
    injected_temporal = TemporalPredictor(explicit_temporal)
    injected_growth = SpeciesGrowthModel(explicit_growth)
    first = prediction_api.create_app(
        temporal_predictor=injected_temporal,
        temporal_model_path=environment_temporal,
        species_growth_model=injected_growth,
        species_growth_model_path=environment_growth,
    )
    second = prediction_api.create_app(
        temporal_model_path=explicit_temporal,
        species_growth_model_path=explicit_growth,
    )

    assert first.state.temporal_predictor is injected_temporal
    assert first.state.species_growth_model is injected_growth
    assert second.state.temporal_predictor is None
    assert second.state.species_growth_model is None
    status = TestClient(first).get("/model/status").json()
    assert status["temporal_model"]["source"] == "injected"
    assert status["species_growth_model"]["source"] == "injected"
    assert str(tmp_path) not in str(status)


def test_unconfigured_models_and_invalid_temporal_requests_have_safe_errors(tmp_path: Path):
    app = prediction_api.create_app(database=Database(tmp_path / "state.db"))
    client = TestClient(app)
    assert client.post("/predict/temporal", json={"records": []}).status_code == 422
    assert client.post("/predict/temporal", json={"records": [{}]}).status_code == 422
    temporal_response = client.post(
        "/predict/temporal", json={"records": _temporal_records()}
    )
    assert temporal_response.status_code == 503
    assert client.post("/predict/forecast", json={"tree_data": {}}).status_code == 503
    app.state.database.close()


def test_predict_rejects_invalid_pipeline_result(tmp_path: Path):
    class InvalidPipeline:
        def predict(self, _):
            return {"risk_level": "低风险"}

    db = Database(tmp_path / "invalid-predict.db")
    response = TestClient(
        prediction_api.create_app(database=db, prediction_pipeline=InvalidPipeline())
    ).post("/predict", json={"tree_id": 1})
    assert response.status_code == 422
    db.close()


def test_training_requires_output_directory_and_prevents_second_task(tmp_path: Path, monkeypatch):
    db = Database(tmp_path / "training-lock.db")
    dataset_id = _training_dataset(db)
    source_model = _write_temporal_model(tmp_path / "training-source.npz")
    missing_output_app = prediction_api.create_app(
        database=db, temporal_model_path=source_model
    )
    assert (
        TestClient(missing_output_app).post(f"/training/train/{dataset_id}").status_code
        == 503
    )

    started = threading.Event()
    release = threading.Event()

    def blocking_fine_tune(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return {"status": "not_deployed", "deployed": False}

    monkeypatch.setattr(prediction_api, "fine_tune_with_history", blocking_fine_tune)
    app = prediction_api.create_app(
        database=db,
        temporal_model_path=source_model,
        training_output_dir=tmp_path / "candidates",
    )
    first_result: dict[str, object] = {}

    def submit_first() -> None:
        first_result["response"] = TestClient(app).post(
            f"/training/train/{dataset_id}"
        )

    thread = threading.Thread(target=submit_first)
    thread.start()
    assert started.wait(timeout=5)
    assert TestClient(app).post(f"/training/train/{dataset_id}").status_code == 409
    release.set()
    thread.join(timeout=5)
    assert first_result["response"].status_code == 200
    db.close()


def test_risk_snapshot_is_auditable_and_old_or_broken_rows_are_safe(tmp_path: Path):
    db = Database(tmp_path / "snapshots.db")
    db.save_prediction(
        {
            "tree_id": 1,
            "risk_level": "高风险",
            "suggestion": "巡检",
            "risk_snapshot": {
                "comprehensive_risk": {"index": 0.7},
                "comprehensive_risk_score": 70,
                "hazard_level": "紧急隐患",
                "risk_escalation_indicator": 44,
                "calculation_ledger": {"model_metadata": {"model_status": "experimental"}},
                "model_status": "experimental",
                "parameter_source": "project_default_unverified",
                "unapproved_raw_history": [{"tree_id": 999}],
            },
        }
    )
    db.save_prediction({"tree_id": 2, "risk_level": "低风险", "suggestion": "观察"})
    db.conn.execute(
        "UPDATE prediction_history SET risk_snapshot_json = ? WHERE tree_id = ?",
        ("{broken", 2),
    )
    db.conn.commit()
    client = TestClient(prediction_api.create_app(database=db))
    rows = client.get("/history").json()["history"]
    assert rows[0]["risk_snapshot"] == {
        "available": False,
        "reason": "invalid_risk_snapshot",
    }
    assert rows[1]["risk_snapshot"]["hazard_level"] == "紧急隐患"
    assert "unapproved_raw_history" not in rows[1]["risk_snapshot"]
    summary = client.get("/report").json()["comprehensive_risk_summary"]
    assert summary["available_records"] == 1
    assert summary["hazard_levels"] == {"紧急隐患": 1}
    db.close()


def test_root_uses_explicit_template_and_missing_template_is_safe(tmp_path: Path):
    template = tmp_path / "index.html"
    template.write_text("<html>temporary template</html>", encoding="utf-8")
    assert (
        TestClient(prediction_api.create_app(web_index_path=template)).get("/").text
        == "<html>temporary template</html>"
    )
    missing = prediction_api.create_app(web_index_path=tmp_path / "missing.html")
    assert TestClient(missing).get("/").status_code == 503
