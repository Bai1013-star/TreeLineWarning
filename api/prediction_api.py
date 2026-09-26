"""FastAPI application for TreeLineWarning predictions."""

from __future__ import annotations

import os
import secrets
from base64 import b64decode
from binascii import Error as Base64Error
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from algorithms.prediction.allometry import (
    calculate_species_parameters,
    estimate_tree_age,
    formula_catalog_summary,
)
from algorithms.prediction.beijing_species import catalog_payload
from algorithms.prediction.contact_risk import contact_risk
from algorithms.prediction.forecast import ALL_INPUT_FIELDS, PREDICTION_MODES, ForecastService
from algorithms.prediction.historical_training import (
    MAX_UPLOAD_BYTES,
    fine_tune_with_history,
    normalize_history_rows,
    parse_history_file,
    parse_history_text,
    parse_tabular_file,
    training_template_csv,
)
from algorithms.prediction.history_report import generate_history_report
from algorithms.prediction.i1_pipeline import I1PredictionPipeline
from algorithms.prediction.regional_weather import regional_weather, reverse_geocode
from algorithms.prediction.species import annual_height_growth
from algorithms.prediction.species_growth import SpeciesGrowthModel
from algorithms.prediction.species_recognition import (
    MAX_IMAGE_BYTES,
    MAX_POINT_CLOUD_BYTES,
)
from algorithms.prediction.species_recognition import (
    delete_sample as delete_species_sample,
)
from algorithms.prediction.species_recognition import (
    predict as predict_species_from_files,
)
from algorithms.prediction.species_recognition import (
    save_sample as save_species_sample,
)
from algorithms.prediction.species_recognition import (
    status as species_recognition_status,
)
from algorithms.prediction.species_recognition import (
    train as train_species_recognition,
)
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
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

TEMPORAL_MODEL_PATH_ENV = "TREELINE_WARNING_TEMPORAL_MODEL_PATH"
SPECIES_GROWTH_MODEL_PATH_ENV = "TREELINE_WARNING_SPECIES_GROWTH_MODEL_PATH"
TRAINING_OUTPUT_DIR_ENV = "TREELINE_WARNING_TRAINING_OUTPUT_DIR"
WEB_INDEX_PATH_ENV = "TREELINE_WARNING_WEB_INDEX_PATH"
ACCESS_USERNAME_ENV = "TREELINE_WARNING_ACCESS_USERNAME"
ACCESS_PASSWORD_ENV = "TREELINE_WARNING_ACCESS_PASSWORD"
DEFAULT_ACCESS_USERNAME = "treeline"
EXPERIMENTAL_MODEL_STATUS = "experimental"
EXPERIMENTAL_PARAMETER_SOURCE = "project_default_unverified"
UPLOAD_FILE = File(...)
OPTIONAL_UPLOAD_FILES = File(default=[])
OPTIONAL_POINT_CLOUD_FILE = File(default=None)
MAX_BATCH_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH_ROWS = 2000


def _access_credentials(username: str | None, password: str | None) -> tuple[str, str | None]:
    configured_username = username
    if configured_username is None:
        configured_username = os.getenv(ACCESS_USERNAME_ENV, DEFAULT_ACCESS_USERNAME)
    configured_username = configured_username.strip() or DEFAULT_ACCESS_USERNAME

    configured_password = password
    if configured_password is None:
        configured_password = os.getenv(ACCESS_PASSWORD_ENV)
    if configured_password is not None and not configured_password:
        configured_password = None
    return configured_username, configured_password


def _valid_basic_credentials(
    authorization: str | None, expected_username: str, expected_password: str
) -> bool:
    if not authorization:
        return False
    scheme, separator, encoded = authorization.partition(" ")
    if not separator or scheme.lower() != "basic" or not encoded:
        return False
    try:
        decoded = b64decode(encoded, validate=True).decode("utf-8")
    except (Base64Error, UnicodeDecodeError, ValueError):
        return False
    username, separator, password = decoded.partition(":")
    if not separator:
        return False
    username_matches = secrets.compare_digest(username, expected_username)
    password_matches = secrets.compare_digest(password, expected_password)
    return username_matches and password_matches


def _authentication_required() -> Response:
    return Response(
        content="Authentication required",
        status_code=401,
        media_type="text/plain; charset=utf-8",
        headers={
            "WWW-Authenticate": 'Basic realm="TreeLineWarning", charset="UTF-8"',
            "Cache-Control": "no-store",
        },
    )


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
        "prediction_mode": forecast_result.get("prediction_mode", "ai"),
        "prediction_mode_label": forecast_result.get("prediction_mode_label"),
        "comprehensive_risk": comprehensive,
        "comprehensive_risk_score": assessment.get("comprehensive_risk_score"),
        "hazard_level": assessment.get("hazard_level"),
        "risk_escalation_indicator": assessment.get("risk_escalation_indicator"),
        "calculation_ledger": forecast_result.get("calculation_ledger", {}),
        "clearance_defect": assessment.get("clearance_defect"),
        "attention_prior": assessment.get("attention_prior"),
        "model_status": comprehensive.get("model_status", EXPERIMENTAL_MODEL_STATUS),
        "parameter_source": comprehensive.get("parameter_source", EXPERIMENTAL_PARAMETER_SOURCE),
    }


def _web_index_for(request: Request) -> Path:
    template_path = request.app.state.web_index_path
    if template_path is None or not Path(template_path).is_file():
        raise HTTPException(
            status_code=503,
            detail="Web 模板未配置或不可用；请由部署人员配置模板路径。",
        )
    return Path(template_path)


def _species_recognition_root(request: Request) -> Path:
    output_dir = request.app.state.training_output_dir
    if output_dir is None:
        raise HTTPException(status_code=503, detail="树种识别数据目录未配置。")
    return Path(output_dir) / "species_recognition"


async def _read_species_uploads(
    photos: list[UploadFile],
    point_cloud: UploadFile | None,
    video_count: int = 0,
    video_frame_count: int = 0,
) -> tuple[list[tuple[str, bytes]], tuple[str, bytes] | None]:
    if not 0 <= video_count <= 2:
        raise HTTPException(status_code=422, detail="一次最多选择 2 个视频。")
    if not 0 <= video_frame_count <= video_count * 5:
        raise HTTPException(status_code=422, detail="视频代表画面数量与所选视频不匹配。")
    photo_count = len(photos) - video_frame_count
    if photo_count < 0:
        raise HTTPException(status_code=422, detail="视频代表画面数量与上传文件不匹配。")
    photo_payloads: list[tuple[str, bytes]] = []
    if photo_count > 3:
        raise HTTPException(status_code=422, detail="一次最多选择 3 张照片。")
    for photo in photos:
        name = photo.filename or "未命名照片"
        content = await photo.read(MAX_IMAGE_BYTES + 1)
        await photo.close()
        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=422, detail=f"照片 {name} 超过 10 MB。")
        photo_payloads.append((name, content))
    cloud_payload: tuple[str, bytes] | None = None
    if point_cloud is not None:
        name = point_cloud.filename or "未命名点云"
        content = await point_cloud.read(MAX_POINT_CLOUD_BYTES + 1)
        await point_cloud.close()
        if len(content) > MAX_POINT_CLOUD_BYTES:
            raise HTTPException(status_code=422, detail=f"点云 {name} 超过 80 MB。")
        cloud_payload = (name, content)
    return photo_payloads, cloud_payload


def root(request: Request) -> FileResponse:
    return FileResponse(
        _web_index_for(request),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def help_page(request: Request) -> FileResponse:
    help_path = _web_index_for(request).with_name("help.html")
    if not help_path.is_file():
        raise HTTPException(status_code=503, detail="帮助页面未配置或不可用。")
    return FileResponse(
        help_path,
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


def _empty_training_pool() -> dict[str, Any]:
    return {
        "source_name": "训练数据池",
        "source_format": "database",
        "sheet_name": None,
        "row_count": 0,
        "valid_row_count": 0,
        "invalid_row_count": 0,
        "tree_count": 0,
        "eligible_tree_count": 0,
        "trainable_sequence_count": 0,
        "detected_columns": [],
        "records": [],
        "preview": [],
        "errors": [],
        "trainable": False,
    }


def _training_pool(database: Database) -> dict[str, Any]:
    records = database.list_training_records()
    if not records:
        return _empty_training_pool()
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("tree_id", "")), str(record.get("observation_date", "")))
        deduplicated[key] = record
    return normalize_history_rows(
        deduplicated.values(),
        source_name="训练数据池",
        source_format="database",
        sheet_name=None,
    )


def training_pool(request: Request) -> dict[str, Any]:
    return {
        key: value
        for key, value in _training_pool(_database_for(request)).items()
        if key != "records"
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


async def import_batch_file(file: UploadFile = UPLOAD_FILE) -> dict[str, Any]:
    """Parse Excel or text rows for browser preview without persisting them."""

    filename = file.filename or "未命名文件"
    content = await file.read(MAX_BATCH_UPLOAD_BYTES + 1)
    await file.close()
    try:
        return parse_tabular_file(
            filename,
            content,
            max_upload_bytes=MAX_BATCH_UPLOAD_BYTES,
            max_rows=MAX_BATCH_ROWS,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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


async def train_history_pool(
    request: Request, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    database = _database_for(request)
    parsed = _training_pool(database)
    if not parsed["trainable"]:
        raise HTTPException(
            status_code=422,
            detail="训练数据池尚未形成至少 2 个时序样本，请继续加入同一树木不同日期的观测。",
        )
    snapshot = {
        **parsed,
        "source_name": "训练数据池快照",
        "source_format": "pool_snapshot",
    }
    dataset_id = database.save_training_dataset(snapshot)
    return await train_history_dataset(request, dataset_id, payload)


def training_history(request: Request) -> dict[str, Any]:
    database = _database_for(request)
    datasets = database.list_training_datasets(limit=50)
    for dataset in datasets:
        dataset["deletable"] = True
    return {
        "datasets": datasets,
        "runs": database.list_training_runs(limit=20),
    }


def delete_training_dataset(request: Request, dataset_id: int) -> dict[str, Any]:
    """Permanently remove one incremental dataset and its linked audit/model files."""

    database = _database_for(request)
    dataset = database.get_training_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="训练数据集不存在")
    deleted = database.delete_training_dataset(dataset_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail="训练数据集不存在")
    removed_files = _delete_training_artifacts(request, deleted.get("runs", []))
    pool = training_pool(request)
    return {
        "deleted_dataset": deleted,
        "training_pool": pool,
        "removed_model_files": removed_files,
        "message": "历史增量数据、关联训练记录及候选模型已删除；后续训练不会再使用这些观测。",
    }


def _delete_training_artifacts(request: Request, runs: list[dict[str, Any]]) -> list[str]:
    """Erase candidate model files only when they are inside the configured output directory."""
    output_dir = request.app.state.training_output_dir
    if output_dir is None:
        return []
    root = Path(output_dir).resolve()
    removed: list[str] = []
    for run in runs:
        metrics = run.get("metrics") if isinstance(run, dict) else None
        raw_path = metrics.get("output_path") if isinstance(metrics, dict) else None
        if not raw_path:
            continue
        candidate = Path(str(raw_path)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            candidate.unlink()
            removed.append(candidate.name)
    return removed


def delete_training_run(request: Request, run_id: int) -> dict[str, Any]:
    """Permanently delete one incremental-training history entry and its candidate model."""
    deleted = _database_for(request).delete_training_run(run_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail="训练历史记录不存在")
    removed_files = _delete_training_artifacts(request, [deleted])
    return {
        "deleted_run": deleted,
        "removed_model_files": removed_files,
        "message": "训练历史记录及关联候选模型已删除。",
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
        clearance = float(latest.get("clearance_distance", latest.get("distance_to_wire", 999)))
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
    prediction_mode = str(payload.get("prediction_mode", "ai")).strip().lower()
    if prediction_mode not in PREDICTION_MODES:
        raise HTTPException(status_code=422, detail="prediction_mode 必须是 formula 或 ai")
    temporal_predictor = None if prediction_mode == "formula" else _temporal_predictor_for(request)
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
            prediction_mode=prediction_mode,
        )
    except (TypeError, ValueError, KeyError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail="预测输入无效。") from exc
    current = forecast_result["current_assessment"]
    database.save_prediction(
        {
            **forecast_result["observed_input_snapshot"],
            **current,
            "risk_snapshot": _risk_snapshot(forecast_result),
            "full_report": forecast_result,
        }
    )
    return forecast_result


def predict_contact_risk(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the standalone field-contact decision without running the LSTM."""
    try:
        species = str(payload.get("species", "")).strip()
        if not species:
            raise ValueError("树种不能为空")
        height = float(payload["height"])
        conductor_height = float(payload.get("conductor_height", 0))
        if height <= 0:
            raise ValueError("树高必须大于 0")
        if conductor_height < 0:
            raise ValueError("导线高度不能小于 0")
        growth = annual_height_growth(species)
        raw_growth = payload.get("annual_height_growth_m")
        annual_growth = None if raw_growth in (None, "") else float(raw_growth)
        if annual_growth is not None and not 0 <= annual_growth <= 6:
            raise ValueError("年生长量必须在 0—6 m 之间")
        result = contact_risk(
            contact_target=payload.get("contact_target", "未碰到"),
            tree_height=height,
            conductor_height=conductor_height,
            clearance_distance=(
                None
                if payload.get("clearance_distance") in (None, "")
                else float(payload["clearance_distance"])
            ),
            general_risk_distance=float(payload.get("general_risk_distance", 1.0)),
            safe_distance=float(payload.get("safe_distance", 5.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    one_year_height = round(height + annual_growth, 2) if annual_growth is not None else None
    one_year_over_line = bool(
        annual_growth is not None and conductor_height > 0 and one_year_height > conductor_height
    )
    future_action = None
    if result["requires_immediate_pruning"]:
        future_action = result["action"]
    elif one_year_over_line:
        future_action = "预计一年内可能超过导线高度，建议在生长季前安排预防性修剪"
    return {
        "species": species,
        "tree_height_m": height,
        "conductor_height_m": conductor_height,
        "annual_height_growth_m": annual_growth,
        "growth_reference": growth,
        "projected_height_one_year_m": one_year_height,
        "one_year_over_line": one_year_over_line,
        "project_location": str(payload.get("project_location", "")).strip(),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "risk_level": result["level"],
        "risk_code": result["code"],
        "clearance_distance_m": result["clearance_distance"],
        "distance_thresholds": result["thresholds"],
        "requires_immediate_pruning": result["requires_immediate_pruning"],
        "basis": result["basis"],
        "action": future_action or result["action"],
    }


def species_formula_catalog() -> dict[str, Any]:
    """Expose the source-traceable domestic species formula catalog."""
    return formula_catalog_summary()


def beijing_species_catalog() -> dict[str, Any]:
    """Return the shared Beijing warning/recognition tree catalog."""
    return catalog_payload()


def predict_species_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    """Calculate applicable age, biomass and allometric formula results."""
    try:

        def optional_number(field: str) -> float | None:
            value = payload.get(field)
            if value in (None, ""):
                return None
            return float(value)

        return calculate_species_parameters(
            species=str(payload.get("species", "")).strip(),
            diameter_cm=optional_number("diameter_cm"),
            height_m=optional_number("height_m"),
            age_years=optional_number("age_years"),
            region=str(payload.get("region", "北京丰台")).strip() or "北京丰台",
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def predict_tree_age(payload: dict[str, Any]) -> dict[str, Any]:
    """Estimate age from a supported species' traceable DBH-age curve."""
    try:
        diameter = payload.get("diameter_cm")
        if diameter in (None, ""):
            raise ValueError("胸径不能为空")
        height = payload.get("height_m")
        return estimate_tree_age(
            species=str(payload.get("species", "")).strip(),
            diameter_cm=float(diameter),
            height_m=None if height in (None, "") else float(height),
            region=str(payload.get("region", "北京丰台")).strip() or "北京丰台",
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def species_recognition_overview(request: Request) -> dict[str, Any]:
    """Return local labelled-sample and model availability information."""
    return species_recognition_status(_species_recognition_root(request))


async def save_species_recognition_sample(
    request: Request,
    species: str = Form(...),
    photos: list[UploadFile] = OPTIONAL_UPLOAD_FILES,
    video_count: int = Form(0),
    video_frame_count: int = Form(0),
    point_cloud: UploadFile | None = OPTIONAL_POINT_CLOUD_FILE,
) -> dict[str, Any]:
    try:
        photo_payloads, cloud_payload = await _read_species_uploads(
            photos, point_cloud, video_count, video_frame_count
        )
        return save_species_sample(
            _species_recognition_root(request),
            species,
            photo_payloads,
            video_count,
            video_frame_count,
            cloud_payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def train_species_recognition_model(request: Request) -> dict[str, Any]:
    try:
        return train_species_recognition(_species_recognition_root(request))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def delete_species_recognition_sample(request: Request, sample_id: str) -> dict[str, Any]:
    try:
        return delete_species_sample(_species_recognition_root(request), sample_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def predict_species_recognition(
    request: Request,
    photos: list[UploadFile] = OPTIONAL_UPLOAD_FILES,
    video_count: int = Form(0),
    video_frame_count: int = Form(0),
    point_cloud: UploadFile | None = OPTIONAL_POINT_CLOUD_FILE,
) -> dict[str, Any]:
    try:
        photo_payloads, cloud_payload = await _read_species_uploads(
            photos, point_cloud, video_count, video_frame_count
        )
        return predict_species_from_files(
            _species_recognition_root(request),
            photo_payloads,
            video_count,
            video_frame_count,
            cloud_payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def history(request: Request) -> dict[str, Any]:
    rows = _database_for(request).get_predictions(limit=100)
    return {"count": len(rows), "history": rows}


def history_detail(request: Request, prediction_id: int) -> dict[str, Any]:
    row = _database_for(request).get_prediction(prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="未找到这条历史预测记录。")
    return row


def delete_history(request: Request, prediction_id: int) -> dict[str, Any]:
    deleted = _database_for(request).delete_prediction(prediction_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail="未找到这条历史预测记录。")
    return {"deleted": deleted, "message": "该历史预测及其完整报告已永久删除。"}


def regional_weather_lookup(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return regional_weather(
            province=str(payload.get("province", "")),
            city=str(payload.get("city", "")),
            district=str(payload.get("district", "")),
        )
    except (OSError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"在线气象查询失败：{exc}") from exc


def map_reverse_geocode(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return reverse_geocode(
            latitude=float(payload.get("latitude")),
            longitude=float(payload.get("longitude")),
        )
    except (OSError, TimeoutError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"地图定位失败：{exc}") from exc


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
    access_username: str | None = None,
    access_password: str | None = None,
) -> FastAPI:
    """Create an isolated application without opening databases or models."""
    temporal_path, temporal_source = _configured_path(temporal_model_path, TEMPORAL_MODEL_PATH_ENV)
    growth_path, growth_source = _configured_path(
        species_growth_model_path, SPECIES_GROWTH_MODEL_PATH_ENV
    )
    output_dir, output_source = _configured_path(training_output_dir, TRAINING_OUTPUT_DIR_ENV)
    template_path, template_source = _configured_path(web_index_path, WEB_INDEX_PATH_ENV)
    if template_path is None:
        template_path = _source_layout_template_path()
        template_source = "source_layout" if template_path is not None else "unconfigured"
    configured_username, configured_password = _access_credentials(access_username, access_password)
    application = FastAPI(title="TreeLineWarning Prediction API", lifespan=_lifespan)
    application.state.access_auth_enabled = configured_password is not None
    application.state.access_username = configured_username

    if configured_password is not None:

        @application.middleware("http")
        async def require_access_password(request: Request, call_next):
            if not _valid_basic_credentials(
                request.headers.get("Authorization"),
                configured_username,
                configured_password,
            ):
                return _authentication_required()
            response = await call_next(request)
            response.headers.setdefault("Cache-Control", "no-store")
            return response

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
    static_path = template_path.parent.parent / "static"
    if static_path.is_dir():
        application.mount("/static", StaticFiles(directory=static_path), name="static")
    application.add_api_route("/", root, methods=["GET"], response_class=FileResponse)
    application.add_api_route("/help", help_page, methods=["GET"], response_class=FileResponse)
    application.add_api_route("/health", health, methods=["GET"])
    application.add_api_route("/predict", predict, methods=["POST"])
    application.add_api_route("/model/status", model_status, methods=["GET"])
    application.add_api_route("/batch/import/file", import_batch_file, methods=["POST"])
    application.add_api_route("/training/import/file", import_training_file, methods=["POST"])
    application.add_api_route("/training/import/text", import_training_text, methods=["POST"])
    application.add_api_route("/training/pool", training_pool, methods=["GET"])
    application.add_api_route(
        "/training/train/{dataset_id}", train_history_dataset, methods=["POST"]
    )
    application.add_api_route("/training/train-pool", train_history_pool, methods=["POST"])
    application.add_api_route("/training/history", training_history, methods=["GET"])
    application.add_api_route(
        "/training/datasets/{dataset_id}",
        delete_training_dataset,
        methods=["DELETE"],
    )
    application.add_api_route("/training/runs/{run_id}", delete_training_run, methods=["DELETE"])
    application.add_api_route("/training/template.csv", training_template, methods=["GET"])
    application.add_api_route("/predict/temporal", predict_temporal, methods=["POST"])
    application.add_api_route("/predict/forecast", predict_forecast, methods=["POST"])
    application.add_api_route("/predict/contact-risk", predict_contact_risk, methods=["POST"])
    application.add_api_route("/species/formulas", species_formula_catalog, methods=["GET"])
    application.add_api_route("/species/catalog", beijing_species_catalog, methods=["GET"])
    application.add_api_route(
        "/predict/species-parameters", predict_species_parameters, methods=["POST"]
    )
    application.add_api_route("/predict/tree-age", predict_tree_age, methods=["POST"])
    application.add_api_route(
        "/species-recognition/status", species_recognition_overview, methods=["GET"]
    )
    application.add_api_route(
        "/species-recognition/samples", save_species_recognition_sample, methods=["POST"]
    )
    application.add_api_route(
        "/species-recognition/train", train_species_recognition_model, methods=["POST"]
    )
    application.add_api_route(
        "/species-recognition/samples/{sample_id}",
        delete_species_recognition_sample,
        methods=["DELETE"],
    )
    application.add_api_route(
        "/species-recognition/predict", predict_species_recognition, methods=["POST"]
    )
    application.add_api_route("/history", history, methods=["GET"])
    application.add_api_route("/history/{prediction_id}", history_detail, methods=["GET"])
    application.add_api_route("/history/{prediction_id}", delete_history, methods=["DELETE"])
    application.add_api_route("/weather/regional", regional_weather_lookup, methods=["POST"])
    application.add_api_route("/location/reverse", map_reverse_geocode, methods=["POST"])
    application.add_api_route("/settings/standards", get_standards, methods=["GET"])
    application.add_api_route("/settings/standards", update_standards, methods=["PUT"])
    application.add_api_route("/report", report, methods=["GET"])
    return application


app = create_app()
