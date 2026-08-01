"""FastAPI application for TreeLineWarning predictions."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from algorithms.prediction.forecast import ALL_INPUT_FIELDS, ForecastService
from algorithms.prediction.historical_training import (
    MAX_UPLOAD_BYTES,
    fine_tune_with_history,
    parse_history_file,
    parse_history_text,
    training_template_csv,
)
from algorithms.prediction.history_report import generate_history_report
from algorithms.prediction.i1_pipeline import I1PredictionPipeline
from algorithms.prediction.species_growth import SpeciesGrowthModel
from algorithms.prediction.standards import (
    PredictionStandards,
    analyze_distances,
    analyze_wire,
)
from algorithms.prediction.temporal_attention import (
    TemporalPredictor,
    risk_from_score,
    validate_sequence,
)
from core.database import DATABASE_PATH_ENV, Database, create_database
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool

TEMPORAL_MODEL_PATH_ENV = "TREELINE_WARNING_TEMPORAL_MODEL_PATH"
SPECIES_GROWTH_MODEL_PATH_ENV = "TREELINE_WARNING_SPECIES_GROWTH_MODEL_PATH"
TRAINING_OUTPUT_DIR_ENV = "TREELINE_WARNING_TRAINING_OUTPUT_DIR"
WEB_INDEX_PATH_ENV = "TREELINE_WARNING_WEB_INDEX_PATH"
EXPERIMENTAL_MODEL_STATUS = "experimental"
EXPERIMENTAL_PARAMETER_SOURCE = "project_default_unverified"
UPLOAD_FILE = File(...)


@asynccontextmanager
async def _lifespan(application: FastAPI):
    try:
        yield
    finally:
        if application.state.database_owned and application.state.database is not None:
            application.state.database.close()


def _configured_path(
    explicit_path: str | Path | None, environment_variable: str
) -> tuple[Path | None, str]:
    """Resolve a path without reading it or exposing it outside application state."""
    if explicit_path is not None and str(explicit_path).strip():
        return Path(explicit_path).expanduser(), "explicit_path"
    environment_path = os.getenv(environment_variable)
    if environment_path and environment_path.strip():
        return Path(environment_path).expanduser(), "environment"
    return None, "unconfigured"


def _source_layout_template_path() -> Path | None:
    candidate = Path(__file__).resolve().parents[1] / "web" / "templates" / "index.html"
    return candidate if candidate.is_file() else None


def _database_for(request: Request) -> Database:
    configured_database = request.app.state.database
    if configured_database is not None:
        return configured_database
    configured_path = request.app.state.database_path
    if configured_path is None:
        configured_path = os.getenv(DATABASE_PATH_ENV)
    configured_database = create_database(configured_path)
    request.app.state.database = configured_database
    request.app.state.database_owned = True
    return configured_database


def _pipeline_for(request: Request) -> I1PredictionPipeline:
    return request.app.state.pipeline


def _model_unavailable(model_name: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=f"{model_name}未配置或无法加载；请由部署人员配置有效模型。",
    )


def _temporal_predictor_for(request: Request) -> TemporalPredictor:
    predictor = request.app.state.temporal_predictor
    if predictor is not None:
        return predictor
    model_path = request.app.state.temporal_model_path
    if model_path is None:
        raise _model_unavailable("时序模型")
    try:
        predictor = TemporalPredictor(model_path)
    except RuntimeError as exc:
        raise _model_unavailable("时序模型") from exc
    request.app.state.temporal_predictor = predictor
    request.app.state.temporal_model_loaded = True
    return predictor


def _species_growth_model_for(request: Request) -> SpeciesGrowthModel:
    model = request.app.state.species_growth_model
    if model is not None:
        return model
    model_path = request.app.state.species_growth_model_path
    if model_path is None:
        raise _model_unavailable("树种生长模型")
    try:
        model = SpeciesGrowthModel(model_path)
    except RuntimeError as exc:
        raise _model_unavailable("树种生长模型") from exc
    request.app.state.species_growth_model = model
    request.app.state.species_growth_model_loaded = True
    return model


def _model_status_entry(
    request: Request, *, attribute: str, path_attribute: str, source_attribute: str
) -> dict[str, Any]:
    model = getattr(request.app.state, attribute)
    path = getattr(request.app.state, path_attribute)
    source = getattr(request.app.state, source_attribute)
    entry: dict[str, Any] = {
        "configured": model is not None or path is not None,
        "loaded": bool(getattr(request.app.state, f"{attribute}_loaded")),
        "source": source,
    }
    if path is not None:
        entry["file_name"] = Path(path).name
    return entry


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if all(hasattr(value, name) for name in ("risk_level", "risk_score", "suggestion")):
        return {
            "risk_level": value.risk_level,
            "risk_score": value.risk_score,
            "suggestion": value.suggestion,
        }
    return {}


def normalize_prediction(raw: Any) -> dict[str, Any]:
    """Normalize supported pipeline responses to the public API contract."""
    candidate = raw
    if isinstance(candidate, dict) and "results" in candidate:
        candidate = candidate["results"]
    if isinstance(candidate, list):
        if not candidate:
            raise ValueError("预测流水线返回了空结果")
        candidate = candidate[0]
    candidate = _as_mapping(candidate)
    if "result" in candidate:
        candidate = _as_mapping(candidate["result"])
    required = ("risk_level", "risk_score", "suggestion")
    if not all(key in candidate for key in required):
        raise ValueError("预测流水线返回结构缺少 risk_level/risk_score/suggestion")
    score = float(candidate["risk_score"])
    if score.is_integer():
        score = int(score)
    result = {
        "risk_level": str(candidate["risk_level"]),
        "risk_score": score,
        "suggestion": str(candidate["suggestion"]),
    }
    for key in ("standard_profile", "standard_analysis"):
        if isinstance(candidate.get(key), dict):
            result[key] = candidate[key]
    return result


def _observed_storage_input(tree_data: dict[str, Any]) -> dict[str, Any]:
    """Persist unchecked optional fields as NULL instead of model defaults."""
    selected = tree_data.get("selected_fields")
    if not isinstance(selected, list):
        return dict(tree_data)
    provided = {str(field) for field in selected}
    result = {**tree_data}
    result.pop("selected_fields", None)
    for field in ALL_INPUT_FIELDS:
        if field not in provided:
            result[field] = None
    if result.get("clearance_distance") is not None:
        result["distance_to_wire"] = result["clearance_distance"]
    return result


def _risk_snapshot(forecast_result: dict[str, Any]) -> dict[str, Any]:
    assessment = forecast_result["current_assessment"]
    comprehensive = assessment.get("comprehensive_risk", {})
    return {
        "comprehensive_risk": comprehensive,
        "comprehensive_risk_score": assessment.get("comprehensive_risk_score"),
        "hazard_level": assessment.get("hazard_level"),
        "risk_escalation_indicator": assessment.get("risk_escalation_indicator"),
        "calculation_ledger": forecast_result.get("calculation_ledger", {}),
        "model_status": comprehensive.get("model_status", EXPERIMENTAL_MODEL_STATUS),
        "parameter_source": comprehensive.get(
            "parameter_source", EXPERIMENTAL_PARAMETER_SOURCE
        ),
    }


def _web_index_for(request: Request) -> Path:
    template_path = request.app.state.web_index_path
    if template_path is None or not Path(template_path).is_file():
        raise HTTPException(
            status_code=503,
            detail="Web 模板未配置或不可用；请由部署人员配置模板路径。",
        )
    return Path(template_path)


def root(request: Request) -> FileResponse:
    return FileResponse(
        _web_index_for(request),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def health() -> dict[str, str]:
    return {"message": "TreeLineWarning API running"}


def predict(request: Request, tree_data: dict[str, Any]) -> dict[str, Any]:
    database = _database_for(request)
    prediction_input = {**tree_data, "standards": database.get_standards()}
    try:
        final_result = normalize_prediction(_pipeline_for(request).predict(prediction_input))
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail="预测输入或返回结构无效。") from exc
    database.save_prediction({**_observed_storage_input(prediction_input), **final_result})
    return final_result


def model_status(request: Request) -> dict[str, Any]:
    return {
        "temporal_model": _model_status_entry(
            request,
            attribute="temporal_predictor",
            path_attribute="temporal_model_path",
            source_attribute="temporal_model_source",
        ),
        "species_growth_model": _model_status_entry(
            request,
            attribute="species_growth_model",
            path_attribute="species_growth_model_path",
            source_attribute="species_growth_model_source",
        ),
        "experimental_risk": {
            "model_status": EXPERIMENTAL_MODEL_STATUS,
            "parameter_source": EXPERIMENTAL_PARAMETER_SOURCE,
        },
    }


def _public_import_result(parsed: dict[str, Any], dataset_id: int) -> dict[str, Any]:
    return {key: value for key, value in parsed.items() if key != "records"} | {
        "dataset_id": dataset_id
    }


async def import_training_file(
    request: Request,
    file: UploadFile = UPLOAD_FILE,
) -> dict[str, Any]:
    database = _database_for(request)
    filename = file.filename or "未命名文件"
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    try:
        parsed = parse_history_file(filename, content)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    dataset_id = database.save_training_dataset(parsed)
    return _public_import_result(parsed, dataset_id)


def import_training_text(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    database = _database_for(request)
    try:
        parsed = parse_history_text(
            str(payload.get("text", "")), str(payload.get("format", "auto"))
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    dataset_id = database.save_training_dataset(parsed)
    return _public_import_result(parsed, dataset_id)


def _training_model_path(request: Request) -> Path:
    model_path = request.app.state.temporal_model_path
    if model_path is None and request.app.state.temporal_predictor is not None:
        model_path = getattr(request.app.state.temporal_predictor, "model_path", None)
    if model_path is None or not Path(model_path).is_file():
        raise _model_unavailable("时序模型")
    return Path(model_path)


async def train_history_dataset(
    request: Request, dataset_id: int, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    database = _database_for(request)
    dataset = database.get_training_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="训练数据集不存在")
    payload = payload or {}
    try:
        epochs = int(payload.get("epochs", 15))
        learning_rate = float(payload.get("learning_rate", 0.0005))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="训练参数无效。") from exc
    model_path = _training_model_path(request)
    output_dir = request.app.state.training_output_dir
    if output_dir is None:
        raise HTTPException(status_code=503, detail="训练输出目录未配置；请联系部署人员。")
    training_lock = request.app.state.training_lock
    if not training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="已有训练任务正在运行。")
    candidate_name = f"candidate-dataset-{dataset_id}-{uuid4().hex}.npz"
    output_path = Path(output_dir) / candidate_name
    try:
        metrics = await run_in_threadpool(
            fine_tune_with_history,
            dataset["records"],
            model_path=model_path,
            output_path=output_path,
            epochs=epochs,
            learning_rate=learning_rate,
            allow_replace=False,
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail="训练数据或参数无效。") from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=503, detail="训练模型不可用或训练未完成。") from exc
    finally:
        training_lock.release()
    run_id = database.save_training_run(dataset_id, metrics)
    metric_summary = {
        key: metrics.get(key)
        for key in (
            "sequence_samples",
            "training_samples",
            "validation_samples",
            "epochs",
            "learning_rate",
            "before_validation_mae",
            "after_validation_mae",
            "improvement",
            "deployed",
        )
    }
    return {
        "training_run_id": run_id,
        "dataset_id": dataset_id,
        "candidate_model_name": candidate_name if metrics.get("deployed") else None,
        "status": metrics.get("status", "not_deployed"),
        "metrics": metric_summary,
    }


def training_history(request: Request) -> dict[str, Any]:
    database = _database_for(request)
    return {
        "datasets": database.list_training_datasets(limit=20),
        "runs": database.list_training_runs(limit=20),
    }


def training_template() -> Response:
    return Response(
        content=training_template_csv().encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="treeline_history_training_template.csv"'
        },
    )


def predict_temporal(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise HTTPException(status_code=422, detail="records 必须是非空时间序列列表")
    if not all(isinstance(record, dict) for record in records):
        raise HTTPException(status_code=422, detail="records 中的每项必须是对象")
    try:
        validate_sequence(records)
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail="时序预测输入无效。") from exc
    predictor = _temporal_predictor_for(request)
    species = str(payload.get("species", "其他"))
    try:
        result = predictor.predict(records, species)
        latest = records[-1]
        database = _database_for(request)
        standard = PredictionStandards.from_mapping(database.get_standards())
        result["risk_level"], result["suggestion"] = risk_from_score(
            float(result["risk_score"]),
            standard.medium_risk_score,
            standard.high_risk_score,
        )
        clearance = float(
            latest.get("clearance_distance", latest.get("distance_to_wire", 999))
        )
        result["standard_profile"] = {
            "profile_name": standard.profile_name,
            "region": standard.region,
            "industry": standard.industry,
        }
        result["standard_analysis"] = analyze_distances(
            vertical_distance=float(latest.get("vertical_distance", clearance)),
            horizontal_distance=float(latest.get("horizontal_distance", clearance)),
            clearance_distance=clearance,
            wind_offset_distance=float(latest.get("wind_offset_distance", 0)),
            standards=standard,
            voltage_level_kv=float(latest.get("voltage_level_kv", 110)),
        )
        result["standard_analysis"]["wire"] = analyze_wire(
            voltage_level_kv=float(latest.get("voltage_level_kv", 110)),
            maximum_sag_winter=float(latest.get("maximum_sag_winter", 3.5)),
            maximum_sag_summer=float(latest.get("maximum_sag_summer", 5)),
            observation_month=int(latest.get("observation_month", latest.get("month", 7))),
            standards=standard,
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail="时序预测输入无效。") from exc
    database.save_prediction(
        {
            **latest,
            "tree_id": payload.get("tree_id", latest.get("tree_id", 0)),
            "species": species,
            **result,
        }
    )
    return result


def predict_forecast(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    tree_data = payload.get("tree_data", payload)
    if not isinstance(tree_data, dict):
        raise HTTPException(status_code=422, detail="tree_data 必须是对象")
    temporal_predictor = _temporal_predictor_for(request)
    species_growth_model = _species_growth_model_for(request)
    database = _database_for(request)
    try:
        historical_records = payload.get("history_records")
        if historical_records is None:
            historical_records = database.get_predictions_by_tree(
                int(tree_data.get("tree_id", 0)), limit=36
            )
        elif not isinstance(historical_records, list):
            raise ValueError("history_records 必须是按时间顺序排列的数据列表")
        service = ForecastService(temporal_predictor, species_growth_model)
        forecast_result = service.forecast(
            tree_data,
            database.get_standards(),
            short_horizons_days=payload.get("short_horizons_days"),
            long_horizons_months=payload.get("long_horizons_months"),
            historical_records=historical_records,
            selected_horizon_months=payload.get("selected_horizon_months"),
        )
    except (TypeError, ValueError, KeyError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail="预测输入无效。") from exc
    current = forecast_result["current_assessment"]
    database.save_prediction(
        {
            **forecast_result["observed_input_snapshot"],
            **current,
            "risk_snapshot": _risk_snapshot(forecast_result),
        }
    )
    return forecast_result


def history(request: Request) -> dict[str, Any]:
    rows = _database_for(request).get_predictions()
    return {"count": len(rows), "history": rows}


def get_standards(request: Request) -> dict[str, Any]:
    return _database_for(request).get_standards()


def update_standards(request: Request, settings: dict[str, Any]) -> dict[str, Any]:
    try:
        return _database_for(request).update_standards(settings)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def report(request: Request) -> dict[str, Any]:
    """Generate a report and summarize auditable experimental risk snapshots."""
    database = _database_for(request)
    rows = database.get_predictions()
    result = generate_history_report(rows, standards=database.get_standards())
    snapshots = [row.get("risk_snapshot", {}) for row in rows]
    available = [snapshot for snapshot in snapshots if snapshot.get("available", True)]
    scores = [
        float(snapshot["comprehensive_risk_score"])
        for snapshot in available
        if isinstance(snapshot.get("comprehensive_risk_score"), (int, float))
    ]
    hazard_levels: dict[str, int] = {}
    for snapshot in available:
        level = snapshot.get("hazard_level")
        if isinstance(level, str):
            hazard_levels[level] = hazard_levels.get(level, 0) + 1
    result["comprehensive_risk_summary"] = {
        "available_records": len(available),
        "average_score": round(sum(scores) / len(scores), 2) if scores else None,
        "hazard_levels": hazard_levels,
    }
    return result


def create_app(
    database: Database | None = None,
    *,
    database_path: str | Path | None = None,
    prediction_pipeline: I1PredictionPipeline | None = None,
    temporal_predictor: TemporalPredictor | None = None,
    temporal_model_path: str | Path | None = None,
    species_growth_model: SpeciesGrowthModel | None = None,
    species_growth_model_path: str | Path | None = None,
    training_output_dir: str | Path | None = None,
    web_index_path: str | Path | None = None,
) -> FastAPI:
    """Create an isolated application without opening databases or models."""
    temporal_path, temporal_source = _configured_path(
        temporal_model_path, TEMPORAL_MODEL_PATH_ENV
    )
    growth_path, growth_source = _configured_path(
        species_growth_model_path, SPECIES_GROWTH_MODEL_PATH_ENV
    )
    output_dir, output_source = _configured_path(
        training_output_dir, TRAINING_OUTPUT_DIR_ENV
    )
    template_path, template_source = _configured_path(web_index_path, WEB_INDEX_PATH_ENV)
    if template_path is None:
        template_path = _source_layout_template_path()
        template_source = "source_layout" if template_path is not None else "unconfigured"
    application = FastAPI(title="TreeLineWarning Prediction API", lifespan=_lifespan)
    application.state.database = database
    application.state.database_path = database_path
    application.state.database_owned = False
    application.state.pipeline = prediction_pipeline or I1PredictionPipeline()
    application.state.temporal_predictor = temporal_predictor
    application.state.temporal_model_path = temporal_path
    application.state.temporal_model_source = "injected" if temporal_predictor else temporal_source
    application.state.temporal_predictor_loaded = temporal_predictor is not None
    application.state.species_growth_model = species_growth_model
    application.state.species_growth_model_path = growth_path
    application.state.species_growth_model_source = (
        "injected" if species_growth_model else growth_source
    )
    application.state.species_growth_model_loaded = species_growth_model is not None
    application.state.training_output_dir = output_dir
    application.state.training_output_source = output_source
    application.state.training_lock = Lock()
    application.state.web_index_path = template_path
    application.state.web_index_source = template_source
    application.add_api_route("/", root, methods=["GET"], response_class=FileResponse)
    application.add_api_route("/health", health, methods=["GET"])
    application.add_api_route("/predict", predict, methods=["POST"])
    application.add_api_route("/model/status", model_status, methods=["GET"])
    application.add_api_route("/training/import/file", import_training_file, methods=["POST"])
    application.add_api_route("/training/import/text", import_training_text, methods=["POST"])
    application.add_api_route(
        "/training/train/{dataset_id}", train_history_dataset, methods=["POST"]
    )
    application.add_api_route("/training/history", training_history, methods=["GET"])
    application.add_api_route("/training/template.csv", training_template, methods=["GET"])
    application.add_api_route("/predict/temporal", predict_temporal, methods=["POST"])
    application.add_api_route("/predict/forecast", predict_forecast, methods=["POST"])
    application.add_api_route("/history", history, methods=["GET"])
    application.add_api_route("/settings/standards", get_standards, methods=["GET"])
    application.add_api_route("/settings/standards", update_standards, methods=["PUT"])
    application.add_api_route("/report", report, methods=["GET"])
    return application


app = create_app()
