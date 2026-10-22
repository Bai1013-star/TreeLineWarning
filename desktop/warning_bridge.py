"""Direct bridge from current point-cloud results to warning forecasts.

Survey geometry and tree-to-wire distances come from the current workflow.
Missing DBH can be supplied as an explicitly labelled operator measurement;
it is saved separately and never replaces original LAS evidence.
"""

from __future__ import annotations

import json
import math
import os
import zlib
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request


WEATHER_FIELDS = ("wind_speed", "rainfall", "humidity", "temperature", "sunlight_hours")


def _seasonal_neutral_weather(month: int) -> dict[str, float]:
    """Conservative neutral values used only for manually deselected fields."""

    if month in (3, 4, 5):
        return {"wind_speed": 3.0, "rainfall": 10.0, "humidity": 55.0, "temperature": 16.0, "sunlight_hours": 7.0}
    if month in (6, 7, 8):
        return {"wind_speed": 3.0, "rainfall": 20.0, "humidity": 70.0, "temperature": 27.0, "sunlight_hours": 8.0}
    if month in (9, 10, 11):
        return {"wind_speed": 3.0, "rainfall": 5.0, "humidity": 60.0, "temperature": 15.0, "sunlight_hours": 6.0}
    return {"wind_speed": 3.0, "rainfall": 1.0, "humidity": 55.0, "temperature": 2.0, "sunlight_hours": 5.0}


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}必须是有效数值") from exc
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        suffix = f"且不能小于 {minimum:g}" if minimum is not None else ""
        raise ValueError(f"{label}必须是有限数值{suffix}")
    return number


def _workflow(processing_backend: Any) -> Any:
    local_app = getattr(processing_backend, "local_app", None)
    workflow = getattr(local_app, "workflow", None)
    if workflow is None:
        raise HTTPException(status_code=409, detail="当前项目尚未进入可直连的处理工作流。")
    return workflow


def _current_project_root(processing_backend: Any, fallback: Path) -> Path:
    """Resolve the project selected in the processing workspace right now.

    The desktop shell can switch projects without restarting the warning web
    application. Reports must therefore follow the active workflow instead of
    continuing to read and write the project that happened to be open at boot.
    """

    try:
        root = Path(_workflow(processing_backend).project_root).resolve()
        if root.is_dir():
            return root
    except (AttributeError, HTTPException, OSError, TypeError, ValueError):
        pass
    return Path(fallback).resolve()


def _project_location(workflow: Any) -> dict[str, Any] | None:
    """Return a WGS84 point-cloud centroid when the project CRS is known."""

    try:
        capability = workflow.capability_projection()
        details = capability.get("details") or {}
        lower = capability.get("bounds_min_xyz") or details.get("bounds_min_xyz")
        upper = capability.get("bounds_max_xyz") or details.get("bounds_max_xyz")
        declaration = capability.get("coordinate_declaration") or {}
        if declaration.get("coordinate_kind") != "GEOREFERENCED":
            return None
        if (capability.get("is_georeferenced_crs", details.get("is_georeferenced_crs"))) is False:
            return None
        source_crs = (
            capability.get("crs_string")
            or details.get("crs_string")
            or declaration.get("horizontal_crs")
        )
        if not source_crs or not isinstance(lower, (list, tuple)) or not isinstance(upper, (list, tuple)):
            return None
        x = (float(lower[0]) + float(upper[0])) / 2.0
        y = (float(lower[1]) + float(upper[1])) / 2.0
        from pyproj import Transformer

        longitude, latitude = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True).transform(x, y)
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            return None
        return {
            "latitude": round(float(latitude), 6),
            "longitude": round(float(longitude), 6),
            "source": "POINT_CLOUD_CRS_CENTROID",
            "source_crs": str(source_crs),
        }
    except (AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError):
        return None


def integrated_context(processing_backend: Any) -> dict[str, Any]:
    """Return every current tree measurement, including review-ready rows.

    Final publication remains explicitly identified, but the data viewer must
    not pretend that an already computed editor result is empty merely because
    the operator has not pressed the final publication button yet.
    """

    workflow = _workflow(processing_backend)
    try:
        from desktop.integrated_cache import (
            load_current_report_cache,
            save_current_report_cache,
        )

        report = load_current_report_cache(workflow)
        if report is None:
            report = workflow.current_report()
            if isinstance(report, dict) and report.get("status") == "FINALIZED":
                save_current_report_cache(workflow, report)
        if not isinstance(report, dict):
            report = {}
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        report = {}
    finalized = report.get("status") == "FINALIZED"
    rows = report.get("trees") or []
    if not finalized:
        try:
            from desktop.review_measurement_cache import review_rows
            rows = review_rows(workflow)
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            rows = []
    if not rows:
        return {
            "status": "LOCKED",
            "message": report.get("message") or "当前项目还没有可用的单木处理结果。",
            "trees": [],
            "source": "FINALIZED_POINT_CLOUD" if finalized else "REVIEW_READY_POINT_CLOUD",
            "finalized": finalized,
            "publication_state": "FINALIZED" if finalized else "REVIEW_REQUIRED",
        }
    def span_key(row: dict[str, Any]) -> str:
        value = row.get("span_id")
        if value not in (None, ""):
            return str(value)
        code = str(row.get("display_tree_id") or "")
        parts = code.split("-")
        return parts[1] if len(parts) >= 3 and parts[1].isdigit() else "1"

    ordered_spans = sorted(
        {span_key(row) for row in rows},
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    span_labels = {value: f"档距{index}" for index, value in enumerate(ordered_spans, start=1)}
    trees: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        persistent_id = str(row.get("persistent_tree_id") or "").strip()
        height = row.get("height_m")
        clearance = row.get("minimum_clearance_m")
        if not persistent_id:
            continue
        prediction_ready = height is not None and clearance is not None
        current_span = span_key(row)
        trees.append(
            {
                "tree_key": persistent_id,
                "tree_id": zlib.crc32(persistent_id.encode("utf-8")) & 0x7FFFFFFF,
                "tree_code": row.get("display_tree_id") or f"TREE-{index:03d}",
                "persistent_tree_id": persistent_id,
                "height": height,
                "diameter": row.get("diameter_cm"),
                "crown_width": row.get("crown_width_m"),
                "horizontal_distance": row.get("horizontal_clearance_m"),
                "vertical_distance": row.get("vertical_clearance_m"),
                "clearance_distance": clearance,
                "quality_state": row.get("quality_state"),
                "current_risk": row.get("current_risk"),
                "measurement_source": (
                    "定稿点云与最近树线三维点对"
                    if finalized
                    else "当前复核结果与最近树线三维点对"
                ),
                "prediction_ready": prediction_ready,
                "publication_state": "FINALIZED" if finalized else "REVIEW_REQUIRED",
                "line_id": row.get("line_id"),
                "span_id": current_span,
                "span_label": span_labels[current_span],
                "start_tower_id": row.get("start_tower_id"),
                "end_tower_id": row.get("end_tower_id"),
            }
        )
    from desktop.manual_dbh import apply_measurements
    run_id = str(getattr(workflow, '_workflow', {}).get('last_batch_id') or '')
    apply_measurements(trees, workflow.project_root, run_id)
    from desktop.manual_measurements import apply_measurements as apply_geometry
    apply_geometry(trees, workflow.project_root, run_id)
    ready_count = sum(bool(tree["prediction_ready"]) for tree in trees)
    return {
        "status": "READY" if trees else "EMPTY",
        "message": (
            f"已读取全部 {len(trees)} 棵单木，其中 {ready_count} 棵可直接预测。"
            + ("结果已定稿。" if finalized else "当前为复核中结果，完成复核后将转为定稿数据。")
        ),
        "source": "FINALIZED_POINT_CLOUD" if finalized else "REVIEW_READY_POINT_CLOUD",
        "tree_count": len(trees),
        "prediction_ready_count": ready_count,
        "trees": trees,
        "analysis_run_id": run_id,
        "project_root": str(workflow.project_root),
        "report_gate": report.get("gate"),
        "finalized": finalized,
        "publication_state": "FINALIZED" if finalized else "REVIEW_REQUIRED",
        "project_location": _project_location(workflow),
        "map": {"mode": "OFFLINE_WORLD_VECTOR", "network_required": False},
    }


def _tree_payload(context: dict[str, Any], payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    tree_key = str(payload.get("tree_key") or "").strip()
    row = next((item for item in context["trees"] if item["tree_key"] == tree_key), None)
    if row is None:
        raise ValueError("请选择当前单木")
    if not row.get("prediction_ready", True):
        raise ValueError("该单木尚缺树高或净空距离，暂时不能用于预测")
    species = str(payload.get("species") or "").strip()
    if not species:
        raise ValueError("请选择树种")
    age = _finite(payload.get("age"), "树龄", minimum=0)
    weather = payload.get("weather")
    if not isinstance(weather, dict):
        raise ValueError("天气状况必须完整提供")
    labels = {
        "wind_speed": "风速",
        "rainfall": "降雨量",
        "humidity": "湿度",
        "temperature": "温度",
        "sunlight_hours": "日照时长",
    }
    supplied_weather_values = {
        field: _finite(weather.get(field), labels[field], minimum=None if field == "temperature" else 0)
        for field in WEATHER_FIELDS
    }
    if supplied_weather_values["humidity"] > 100:
        raise ValueError("湿度必须在 0—100% 之间")
    if not -60 <= supplied_weather_values["temperature"] <= 60:
        raise ValueError("温度必须在 -60—60℃ 之间")
    if supplied_weather_values["sunlight_hours"] > 24:
        raise ValueError("日照时长不能超过 24 小时")

    raw_observed = str(payload.get("observation_date") or date.today().isoformat()).strip()
    try:
        observed = date.fromisoformat(raw_observed[:10])
    except ValueError as exc:
        raise ValueError("观测日期必须是有效的 YYYY-MM-DD 日期") from exc
    if observed.year < 1900 or observed.year > 2200:
        raise ValueError("观测日期年份必须在 1900—2200 之间")
    raw_weather_selection = weather.get("selected_fields")
    if isinstance(raw_weather_selection, list):
        weather_selected = [
            field for field in WEATHER_FIELDS if field in {str(item) for item in raw_weather_selection}
        ]
    else:
        weather_selected = list(WEATHER_FIELDS)
    weather_source = str(weather.get("source") or "手动填写").strip()
    if weather_source.startswith("中国气象局"):
        weather_selected = list(WEATHER_FIELDS)
    neutral = _seasonal_neutral_weather(observed.month)
    weather_values = {
        field: supplied_weather_values[field] if field in weather_selected else neutral[field]
        for field in WEATHER_FIELDS
    }
    selected = [
        "observation_year", "observation_month", "observation_day", "species", "height",
        "voltage_level_kv", "clearance_distance", "tree_id", "tree_code", "age",
        "line_name", "tower_section",
        *weather_selected,
    ]
    tree_data: dict[str, Any] = {
        "selected_fields": selected,
        "observation_date": observed.isoformat(),
        "observation_year": observed.year,
        "observation_month": observed.month,
        "observation_day": observed.day,
        "tree_id": row["tree_id"],
        "tree_code": row["tree_code"],
        "line_name": f"线路{row.get('line_id') or 1}",
        "tower_section": row.get("span_label") or "档距1",
        "species": species,
        "age": age,
        "height": _finite(row["height"], "点云树高", minimum=0),
        # The active engineering scenario uses the configured 110 kV rule set.
        # Distances themselves are never defaulted: they come from the exact
        # finalized nearest tree/wire pair below.
        "voltage_level_kv": 110.0,
        "clearance_distance": _finite(row["clearance_distance"], "点云净空", minimum=0),
        **weather_values,
    }
    weather_observed_at = str(weather.get("observed_at") or "").strip()
    tree_data["weather_source"] = weather_source
    tree_data["weather_observed_at"] = weather_observed_at
    tree_data["weather_selected_fields"] = weather_selected
    tree_data["weather_defaulted_fields"] = [
        field for field in WEATHER_FIELDS if field not in weather_selected
    ]
    selected.extend(["weather_source", "weather_observed_at"])
    tree_data['diameter_source'] = row.get('diameter_source', 'ORIGINAL_LAS_1P30M' if row.get('diameter') is not None else 'UNAVAILABLE')
    tree_data['point_cloud_diameter'] = row.get('point_cloud_diameter', row.get('diameter'))
    tree_data['measurement_sources'] = row.get('measurement_sources', {})
    tree_data['point_cloud_measurements'] = row.get('point_cloud_measurements', {})
    for source, target in (
        ("diameter", "diameter"),
        ("crown_width", "crown_width"),
        ("horizontal_distance", "horizontal_distance"),
        ("vertical_distance", "vertical_distance"),
    ):
        value = row.get(source)
        if value is not None:
            tree_data[target] = _finite(value, target, minimum=0)
            selected.append(target)
    return tree_data, tree_key


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _save_twin_profile(
    project_root: Path,
    tree_key: str,
    tree_data: dict[str, Any],
    forecast: dict[str, Any],
    point_cloud_source: str,
) -> None:
    path = project_root / "runtime" / "warning" / "twin_profiles.json"
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        existing = {}
    profiles = existing.get("profiles") if isinstance(existing, dict) else None
    if not isinstance(profiles, dict):
        profiles = {}
    profiles[tree_key] = {
        "species": tree_data["species"],
        "age": tree_data["age"],
        "observation_date": tree_data["observation_date"],
        "weather": {field: tree_data[field] for field in WEATHER_FIELDS},
        "current": {
            "height": tree_data["height"],
            "diameter": tree_data.get("diameter"),
            "crown_width": tree_data.get("crown_width"),
        },
        "short_term_forecast": forecast.get("short_term_forecast", []),
        "long_term_forecast": forecast.get("long_term_forecast", []),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(
        path,
        {
            "schema": "treeline-twin-profiles-v1",
            "source": f"operator_species_age_weather_plus_{point_cloud_source.lower()}",
            "profiles": profiles,
        },
    )


def _save_latest_report(
    project_root: Path,
    forecast: dict[str, Any],
    *,
    batch_id: str | None = None,
    batch_index: int | None = None,
    batch_count: int | None = None,
) -> None:
    """Persist one report or an ordered multi-tree, span-grouped report set."""

    path = project_root / "runtime" / "warning" / "latest_report.json"
    saved_at = datetime.now(timezone.utc).isoformat()
    reports = [forecast]
    normalized_batch = str(batch_id or "").strip()
    if normalized_batch and type(batch_index) is int and type(batch_count) is int:
        if not (1 <= batch_count <= 500 and 0 <= batch_index < batch_count):
            raise ValueError("批量报告序号或总数无效")
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, ValueError, json.JSONDecodeError):
            existing = {}
        if existing.get("batch_id") == normalized_batch and isinstance(existing.get("reports"), list):
            reports = list(existing["reports"])
        if batch_index == 0:
            reports = []
        while len(reports) <= batch_index:
            reports.append(None)
        reports[batch_index] = forecast
        reports = [item for item in reports if isinstance(item, dict)]
    _atomic_json(
        path,
        {
            "schema": "treeline-latest-report-v2",
            "saved_at": saved_at,
            "batch_id": normalized_batch or None,
            "expected_report_count": batch_count if normalized_batch else 1,
            "reports": reports,
            # Backward compatibility for callers that only restore one tree.
            "report": reports[0] if reports else forecast,
        },
    )


def register_warning_bridge(
    application: FastAPI,
    *,
    processing_backend: Any,
    project_root: Path,
) -> None:
    """Register same-origin integrated routes on the warning application."""

    application.state.processing_backend = processing_backend
    application.state.integrated_project_root = Path(project_root).resolve()

    def context_route() -> dict[str, Any]:
        return integrated_context(processing_backend)

    def measurement_route(payload: dict[str, Any]) -> dict[str, Any]:
        from desktop.manual_dbh import save_measurement
        context = integrated_context(processing_backend)
        root = Path(context.get('project_root') or
            _current_project_root(processing_backend, application.state.integrated_project_root))
        # Bind UI submissions to the project/version observed before awaiting.
        # A project switch must never save a former tree into the new project.
        expected_root = payload.get('expected_project_root')
        expected_run = payload.get('expected_analysis_run_id')
        if ((expected_root is not None and Path(str(expected_root)).resolve() != root.resolve())
                or (expected_run is not None and str(expected_run) != str(context.get('analysis_run_id') or ''))):
            raise HTTPException(status_code=409, detail='项目或分析版本已改变，请刷新当前数据后重新补填。')
        try:
            if 'manual_fields' in payload:
                from desktop.manual_measurements import save_measurements
                result = save_measurements(root, context.get('analysis_run_id'), context.get('trees', []),
                    str(payload.get('tree_key') or ''), payload['manual_fields'],
                    correct_existing=payload.get('correct_existing_measurements') is True)
                return {'ok': True, **result}
            result = save_measurement(root, context.get('analysis_run_id'), context.get('trees', []),
                str(payload.get('tree_key') or ''), payload.get('manual_diameter_cm'),
                correct_existing=payload.get('correct_existing_measurements') is True)
            return {'ok': True, **result}
        except (ValueError, TypeError, OSError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    def forecast_route(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        from api.prediction_api import predict_forecast

        context = integrated_context(processing_backend)
        active_project_root = Path(context.get('project_root') or
            _current_project_root(processing_backend, application.state.integrated_project_root))
        request.state.integrated_project_root = active_project_root
        if context.get("status") != "READY":
            raise HTTPException(status_code=409, detail=context.get("message"))
        try:
            tree_data, tree_key = _tree_payload(context, payload)
            forecast = predict_forecast(
                request,
                {
                    "prediction_mode": str(payload.get("prediction_mode") or "formula"),
                    "forecast_scope": payload.get('forecast_scope'),
                    "tree_data": tree_data,
                    "short_horizons_days": payload.get("short_horizons_days") or [1, 7, 14, 30],
                    "long_horizons_months": payload.get("long_horizons_months")
                    or list(range(1, 37)),
                    "selected_horizon_months": int(payload.get("selected_horizon_months") or 12),
                    "future_weather": payload.get('future_weather'),
                    "future_monthly_weather": payload.get('future_monthly_weather'),
                },
            )
            target_date = str(payload.get("target_date") or "").strip()
            if target_date:
                try:
                    parsed_target = date.fromisoformat(target_date[:10])
                except ValueError as exc:
                    raise ValueError("预测目标日期必须是有效的 YYYY-MM-DD 日期") from exc
                if parsed_target <= date.fromisoformat(tree_data["observation_date"]):
                    raise ValueError("预测目标日期必须晚于观测日期")
                forecast["requested_target_date"] = parsed_target.isoformat()
            # Shared prediction API attaches identical wind/evidence estimates
            # for project, manual and imported-table workflows.
            point_cloud_source = str(context.get("source") or "REVIEW_READY_POINT_CLOUD")
            _save_twin_profile(
                active_project_root,
                tree_key,
                tree_data,
                forecast,
                point_cloud_source,
            )
            forecast["integrated_source"] = {
                "tree_key": tree_key,
                "geometry": point_cloud_source,
                "tree_line_distance": "EXACT_3D_NEAREST_PAIR_COMPONENTS",
                "manual_fields": ["species", "age"]
                if str(tree_data.get("weather_source") or "").startswith("中国气象局")
                else ["species", "age", *WEATHER_FIELDS],
                "weather_source": tree_data.get("weather_source"),
                "weather_selected_fields": tree_data.get("weather_selected_fields", []),
                "weather_defaulted_fields": tree_data.get("weather_defaulted_fields", []),
                "excel_used": False,
            }
            if tree_data.get('diameter_source') in ('MANUAL_FIELD_MEASUREMENT', 'MANUAL_FIELD_CORRECTION'):
                forecast['integrated_source']['manual_fields'].append('diameter')
            manual_geometry = [key for key, source in tree_data.get('measurement_sources', {}).items()
                               if source in ('MANUAL_FIELD_MEASUREMENT', 'MANUAL_FIELD_CORRECTION')]
            forecast['integrated_source']['manual_fields'].extend(manual_geometry)
            if manual_geometry:
                forecast['integrated_source']['tree_line_distance'] = 'POINT_CLOUD_WITH_EXPLICIT_MANUAL_SUPPLEMENTS'
            _save_latest_report(
                active_project_root,
                forecast,
                batch_id=str(payload.get("forecast_batch_id") or "") or None,
                batch_index=(
                    int(payload["forecast_batch_index"])
                    if payload.get("forecast_batch_index") is not None
                    else None
                ),
                batch_count=(
                    int(payload["forecast_batch_count"])
                    if payload.get("forecast_batch_count") is not None
                    else None
                ),
            )
            return forecast
        except HTTPException:
            raise
        except (TypeError, ValueError, KeyError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    application.add_api_route("/integrated/context", context_route, methods=["GET"])
    application.add_api_route('/integrated/measurements', measurement_route, methods=['POST'])
    application.add_api_route("/integrated/forecast", forecast_route, methods=["POST"])

    def latest_report_route() -> dict[str, Any]:
        active_project_root = _current_project_root(
            processing_backend, application.state.integrated_project_root
        )
        path = active_project_root / "runtime" / "warning" / "latest_report.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return {"status": "EMPTY", "report": None}
        report = payload.get("report") if isinstance(payload, dict) else None
        reports = payload.get("reports") if isinstance(payload, dict) else None
        if not isinstance(reports, list):
            reports = [report] if isinstance(report, dict) else []
        reports = [item for item in reports if isinstance(item, dict)]
        return {
            "status": "READY" if reports else "EMPTY",
            "saved_at": payload.get("saved_at") if isinstance(payload, dict) else None,
            "report": reports[0] if reports else None,
            "reports": reports,
        }

    application.add_api_route("/integrated/latest-report", latest_report_route, methods=["GET"])


__all__ = ["integrated_context", "register_warning_bridge"]
