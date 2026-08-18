"""Prediction boundary for the desktop's baseline and long-horizon modes.

Science outputs are read-only inputs. The fallback behaviour below belongs to
the recovered historical ``ForecastService`` and is disclosed in results.
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PREDICTION_INTERFACE_VERSION = "treeline-local-prediction-v1"
BASELINE_MODEL_ID = "tree-risk-rule-baseline"
BASELINE_MODEL_VERSION = "origin-feature-chinese-ui-a502ec9"
BASELINE_HORIZON = "current-baseline"
FORMULA_MODEL_ID = "forecast-service-formula"
AI_MODEL_ID = "forecast-service-temporal-attention"
SPECIES_MODEL_SHA256 = "125A7F91C5257379B4967B542208901A6EFE2FA371939BC0F12A9FBF7591EEDB"
TEMPORAL_MODEL_SHA256 = "3EF27CFA77E2051C5236F87A4B7145FDEE402A230787C9AF7BB5C7C96628748C"
SHORT_HORIZONS_DAYS = (1, 7, 14, 30)
LONG_HORIZONS_MONTHS = (1, 2, 3, 6, 12)


@dataclass(frozen=True, slots=True)
class PredictionInputPackage:
    schema_version: str
    project_id: str
    group_id: str
    span_id: str
    source_science_run_id: str
    observation_timestamp: str
    observations: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PredictionResultPackage:
    schema_version: str
    project_id: str
    group_id: str
    span_id: str
    source_science_run_id: str
    status: str
    entries: tuple[dict[str, Any], ...]
    horizons: tuple[str, ...]
    missing_fields: tuple[str, ...]
    diagnostic: str | None
    runtime_seconds: float
    failed_tree_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_prediction_input(
    *, project_id: str, group_id: str, span_id: str, run_id: str,
    trees: list[dict[str, Any]], span_context: dict[str, Any] | None = None,
) -> PredictionInputPackage:
    """Build a portable package from terminal-science projections only."""
    observations = []
    for tree in trees:
        identity = tree.get("prediction_identity") or {}
        observations.append(
            {
                "persistent_tree_id": tree.get("persistent_tree_id"),
                "global_tree_id": tree.get("global_tree_id"),
                "quality": tree.get("category"),
                "quality_flags": {
                    "needs_review": bool(identity.get("needs_review")),
                    "eligible_for_long_term_history": bool(
                        identity.get("eligible_for_long_term_history")
                    ),
                },
                "tree_geometry": {
                    "apex_height_m": tree.get("apex_height_m"),
                    "apex_xyz_absolute_m": tree.get("apex_xyz_absolute_m"),
                    "full_resolution_point_count": tree.get("full_resolution_point_count"),
                },
                "current_safety": {
                    "minimum_clearance_m": tree.get("minimum_clearance_m"),
                    "dangerous": tree.get("dangerous"),
                    "danger_point_count": tree.get("danger_point_count"),
                    "nearest_wire_id": (tree.get("clearance") or {}).get("nearest_wire_id"),
                },
                "span_context": dict(span_context or {}),
            }
        )
    return PredictionInputPackage(
        schema_version=PREDICTION_INTERFACE_VERSION,
        project_id=project_id, group_id=group_id, span_id=span_id,
        source_science_run_id=run_id,
        observation_timestamp=datetime.now(timezone.utc).isoformat(),
        observations=tuple(observations),
    )


def _resource_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))


def model_asset_paths() -> tuple[Path, Path]:
    models = _resource_root() / "models"
    return models / "species_growth.json", models / "temporal_attention.npz"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def verify_model_assets() -> dict[str, Any]:
    species_path, temporal_path = model_asset_paths()
    species_hash = _sha256(species_path)
    temporal_hash = _sha256(temporal_path)
    if species_hash != SPECIES_MODEL_SHA256:
        raise RuntimeError("species_growth.json SHA256 does not match deployed asset")
    if temporal_hash != TEMPORAL_MODEL_SHA256:
        raise RuntimeError("temporal_attention.npz SHA256 does not match deployed asset")
    return {
        "species_model_path": str(species_path),
        "temporal_model_path": str(temporal_path),
        "species_model_sha256": species_hash,
        "temporal_model_sha256": temporal_hash,
        "provenance": "SYNTHETIC_BASELINE",
    }


class LocalPredictionProvider:
    """Run recovered baseline, formula, or temporal forecasting by stable ID."""
    interface_version = PREDICTION_INTERFACE_VERSION

    def run(
        self, package: PredictionInputPackage, *, prediction_mode: str = "baseline"
    ) -> PredictionResultPackage:
        mode = str(prediction_mode).strip().lower()
        if mode not in {"baseline", "formula", "ai"}:
            raise ValueError("prediction_mode must be baseline, formula, or ai")
        started = time.perf_counter()
        try:
            model_info = verify_model_assets() if mode in {"formula", "ai"} else None
            entries: list[dict[str, Any]] = []
            failed: list[str] = []
            for observation in package.observations:
                persistent_id = observation.get("persistent_tree_id")
                if not isinstance(persistent_id, str) or not persistent_id:
                    failed.append(str(persistent_id or "UNBOUND_TREE"))
                    continue
                try:
                    entry = (
                        _run_baseline(observation, package.source_science_run_id)
                        if mode == "baseline"
                        else _run_forecast(observation, package, mode, model_info or {})
                    )
                    entries.append(entry)
                except (TypeError, ValueError, KeyError, RuntimeError):
                    failed.append(persistent_id)
            status = "COMPLETE" if not failed else ("PARTIAL" if entries else "FAILED")
            horizons = ((BASELINE_HORIZON,) if mode == "baseline" else tuple(
                [f"{days}天" for days in SHORT_HORIZONS_DAYS]
                + [f"{months}个月" for months in LONG_HORIZONS_MONTHS]
            ))
            return PredictionResultPackage(
                schema_version=PREDICTION_INTERFACE_VERSION,
                project_id=package.project_id, group_id=package.group_id, span_id=package.span_id,
                source_science_run_id=package.source_science_run_id, status=status,
                entries=tuple(entries), horizons=horizons, missing_fields=(),
                diagnostic=_mode_diagnostic(mode), runtime_seconds=time.perf_counter() - started,
                failed_tree_ids=tuple(failed),
            )
        except (OSError, TypeError, ValueError, KeyError, RuntimeError) as exc:
            return PredictionResultPackage(
                schema_version=PREDICTION_INTERFACE_VERSION,
                project_id=package.project_id, group_id=package.group_id, span_id=package.span_id,
                source_science_run_id=package.source_science_run_id, status="FAILED",
                entries=(), horizons=(), missing_fields=(),
                diagnostic=f"PREDICTION_CORE_REJECTED_OUTPUT: {exc}",
                runtime_seconds=time.perf_counter() - started,
                failed_tree_ids=tuple(
                    str(item.get("persistent_tree_id") or "UNBOUND_TREE")
                    for item in package.observations
                ),
            )


def _mode_diagnostic(mode: str) -> str:
    if mode == "baseline":
        return "当前规则风险；使用历史规则核心的透明默认项。"
    return "时序注意力预测（合成数据基线）" if mode == "ai" else "机理/规则生长预测"


def _run_baseline(observation: dict[str, Any], science_run_id: str) -> dict[str, Any]:
    from algorithms.prediction.predictor import TreeInput, TreeRiskPredictor
    geometry, safety = observation["tree_geometry"], observation["current_safety"]
    clearance = safety.get("minimum_clearance_m")
    result = TreeRiskPredictor().predict(TreeInput(
        species="未知", height=float(geometry.get("apex_height_m") or 0.0), age=0,
        diameter=0.0, wind_speed=0.0, rainfall=0.0, humidity=0.0, slope=0.0,
        distance_to_wire=float(clearance) if clearance is not None else 999.0,
    ))
    return {
        "persistent_tree_id": observation["persistent_tree_id"], "science_run_id": science_run_id,
        "prediction_mode": "baseline", "prediction_model_id": BASELINE_MODEL_ID,
        "prediction_model_version": BASELINE_MODEL_VERSION, "model_label": "当前规则风险",
        "current_science": dict(safety),
        "horizons": [{"horizon": BASELINE_HORIZON, "future_clearance_m": None,
                      "future_growth_m": None, "predicted_risk_level": result.risk_level,
                      "predicted_risk_score": result.risk_score, "target_date": None,
                      "confidence": None}],
        "input_quality": "PARTIAL_WITH_CORE_DEFAULTS",
        "missing_feature_flags": [
            "species",
            "age",
            "diameter",
            "wind_speed",
            "rainfall",
            "humidity",
            "slope",
            "voltage_level_kv",
        ],
        "core_default_flags": ["species:未知", "age:0", "diameter:0", "weather:zero",
                               "slope:0", "voltage_level_kv:110"],
        "model_provenance": "SYNTHETIC_BASELINE",
    }


def _forecast_snapshot(
    observation: dict[str, Any], package: PredictionInputPackage
) -> dict[str, Any]:
    """Map only science values; omitted values use historic ForecastService fallbacks."""
    geometry, safety = observation["tree_geometry"], observation["current_safety"]
    context = observation.get("span_context") or {}
    return {
        "tree_id": int(observation.get("global_tree_id") or 0) + 1,
        "tree_code": observation["persistent_tree_id"], "species": "其他",
        "height": geometry.get("apex_height_m"),
        "clearance_distance": safety.get("minimum_clearance_m"),
        "distance_to_wire": safety.get("minimum_clearance_m"),
        "observation_date": package.observation_timestamp[:10],
        "tower_section": context.get("tower_pair"), "span_length": context.get("span_length_m"),
    }


def _run_forecast(
    observation: dict[str, Any],
    package: PredictionInputPackage,
    mode: str,
    model_info: dict[str, Any],
) -> dict[str, Any]:
    from algorithms.prediction.forecast import ForecastService
    from algorithms.prediction.species_growth import SpeciesGrowthModel
    from algorithms.prediction.standards import PredictionStandards
    from algorithms.prediction.temporal_attention import TemporalPredictor
    growth = SpeciesGrowthModel(model_info["species_model_path"])
    temporal = TemporalPredictor(model_info["temporal_model_path"]) if mode == "ai" else None
    forecast = ForecastService(temporal_predictor=temporal, growth_model=growth).forecast(
        _forecast_snapshot(observation, package), PredictionStandards().to_dict(),
        short_horizons_days=list(SHORT_HORIZONS_DAYS),
        long_horizons_months=list(LONG_HORIZONS_MONTHS), prediction_mode=mode,
    )
    horizons = [_short_horizon(item) for item in forecast.get("short_term_forecast", [])]
    horizons.extend(_long_horizon(item) for item in forecast.get("long_term_forecast", []))
    method = forecast.get("prediction_method") or {}
    return {
        "persistent_tree_id": observation["persistent_tree_id"],
        "science_run_id": package.source_science_run_id, "prediction_mode": mode,
        "prediction_model_id": AI_MODEL_ID if mode == "ai" else FORMULA_MODEL_ID,
        "prediction_model_version": BASELINE_MODEL_VERSION,
        "model_label": "时序注意力预测（合成数据基线）" if mode == "ai" else "机理/规则生长预测",
        "model_provenance": model_info["provenance"],
        "model_hashes": {"species_growth.json": model_info["species_model_sha256"],
                         "temporal_attention.npz": model_info["temporal_model_sha256"]},
        "uses_temporal_model": bool(method.get("uses_temporal_model")),
        "current_science": dict(observation["current_safety"]),
        "current_assessment": forecast.get("current_assessment"), "horizons": horizons,
        "input_quality": (forecast.get("data_quality") or {}).get("status", "未提供"),
        "input_availability": forecast.get("input_availability"),
        "data_quality": forecast.get("data_quality"),
        "species_fallback": "未识别 / 其他（模型回退）",
    }


def _short_horizon(item: dict[str, Any]) -> dict[str, Any]:
    return {"horizon": item.get("period"), "target_date": item.get("target_date"),
            "future_clearance_m": item.get("predicted_clearance_distance"),
            "future_growth_m": item.get("predicted_height_change"),
            "predicted_risk_level": item.get("risk_level"),
            "predicted_risk_score": item.get("risk_score"), "confidence": None}


def _long_horizon(item: dict[str, Any]) -> dict[str, Any]:
    return {"horizon": f"{item.get('horizon_months')}个月", "target_date": item.get("target_date"),
            "future_height_m": item.get("predicted_height"),
            "future_crown_width_m": item.get("predicted_crown_width"),
            "future_clearance_m": item.get("predicted_clearance_distance"),
            "future_growth_m": item.get("predicted_height_change"),
            "predicted_risk_level": item.get("risk_level"),
            "predicted_risk_score": item.get("risk_score"), "confidence": None}
