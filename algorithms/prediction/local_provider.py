"""Strict local prediction-provider boundary.

The desktop's shipped provider invokes the locally tracked rule-based core
from ``origin/feature/chinese-ui``. It uses only defaults that the core's
own I1 adapter formally defines; every such default remains visible in output.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

PREDICTION_INTERFACE_VERSION = "treeline-local-prediction-v1"
BASELINE_MODEL_ID = "tree-risk-rule-baseline"
BASELINE_MODEL_VERSION = "origin-feature-chinese-ui-a502ec9"
BASELINE_HORIZON = "current-baseline"


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_prediction_input(
    *, project_id: str, group_id: str, span_id: str, run_id: str, trees: list[dict[str, Any]]
) -> PredictionInputPackage:
    """Create the portable prediction input from terminal-science projections.

    Only fields known to be present in the science result are sent across this
    boundary.  ``None`` is preserved, never converted to model defaults.
    """

    observations = []
    for tree in trees:
        persistent_id = tree.get("persistent_tree_id")
        prediction_identity = tree.get("prediction_identity") or {}
        observations.append(
            {
                "persistent_tree_id": persistent_id,
                "global_tree_id": tree.get("global_tree_id"),
                "quality": tree.get("category"),
                "quality_flags": {
                    "needs_review": bool(prediction_identity.get("needs_review")),
                    "eligible_for_long_term_history": bool(
                        prediction_identity.get("eligible_for_long_term_history")
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
            }
        )
    return PredictionInputPackage(
        schema_version=PREDICTION_INTERFACE_VERSION,
        project_id=project_id,
        group_id=group_id,
        span_id=span_id,
        source_science_run_id=run_id,
        observation_timestamp=datetime.now(timezone.utc).isoformat(),
        observations=tuple(observations),
    )


class LocalPredictionProvider:
    """Run the shipped local baseline through the stable desktop contract."""

    interface_version = PREDICTION_INTERFACE_VERSION

    def run(self, package: PredictionInputPackage) -> PredictionResultPackage:
        started = time.perf_counter()
        try:
            entries = tuple(_run_baseline(item, package.source_science_run_id) for item in package.observations)
            return PredictionResultPackage(
                schema_version=PREDICTION_INTERFACE_VERSION,
                project_id=package.project_id,
                group_id=package.group_id,
                span_id=package.span_id,
                source_science_run_id=package.source_science_run_id,
                status="COMPLETE",
                entries=entries,
                horizons=(BASELINE_HORIZON,),
                missing_fields=(),
                diagnostic=(
                    "Baseline / Rule-based: no forward growth model is packaged; "
                    "current-baseline risk uses documented core defaults."
                ),
                runtime_seconds=time.perf_counter() - started,
            )
        except (TypeError, ValueError, KeyError, ImportError) as exc:
            return PredictionResultPackage(
                schema_version=PREDICTION_INTERFACE_VERSION,
                project_id=package.project_id,
                group_id=package.group_id,
                span_id=package.span_id,
                source_science_run_id=package.source_science_run_id,
                status="FAILED",
                entries=(),
                horizons=(),
                missing_fields=(),
                diagnostic=f"PREDICTION_CORE_REJECTED_OUTPUT: {exc}",
                runtime_seconds=time.perf_counter() - started,
            )


def _run_baseline(observation: dict[str, Any], science_run_id: str) -> dict[str, Any]:
    """Adapt one terminal-science record using only core-defined defaults."""

    from algorithms.prediction.predictor import TreeInput, TreeRiskPredictor

    geometry = observation.get("tree_geometry") or {}
    safety = observation.get("current_safety") or {}
    clearance = safety.get("minimum_clearance_m")
    missing_features = [
        "species",
        "age",
        "diameter",
        "wind_speed",
        "rainfall",
        "humidity",
        "slope",
        "voltage_level_kv",
    ]
    if geometry.get("apex_height_m") is None:
        missing_features.append("apex_height_m")
    if clearance is None:
        missing_features.append("minimum_clearance_m")
    # These are the explicit defaults used by the source core's I1 adapter.
    defaulted_features = (
        "species:未知",
        "age:0",
        "diameter:0",
        "weather:zero",
        "slope:0",
        "voltage_level_kv:110",
    )
    source_input = TreeInput(
        species="未知",
        height=float(geometry.get("apex_height_m") or 0.0),
        age=0,
        diameter=0.0,
        wind_speed=0.0,
        rainfall=0.0,
        humidity=0.0,
        slope=0.0,
        distance_to_wire=float(clearance) if clearance is not None else 999.0,
    )
    result = TreeRiskPredictor().predict(source_input)
    return {
        "persistent_tree_id": observation["persistent_tree_id"],
        "science_run_id": science_run_id,
        "prediction_run_id": f"{BASELINE_MODEL_ID}:{science_run_id}",
        "prediction_model_id": BASELINE_MODEL_ID,
        "prediction_model_version": BASELINE_MODEL_VERSION,
        "model_label": "Baseline / Rule-based",
        "current_science": {
            "minimum_clearance_m": clearance,
            "dangerous": safety.get("dangerous"),
        },
        "horizons": [{
            "horizon": BASELINE_HORIZON,
            "future_clearance_m": None,
            "future_growth_m": None,
            "predicted_risk_level": result.risk_level,
            "predicted_risk_score": result.risk_score,
            "confidence": None,
        }],
        "input_quality": "PARTIAL_WITH_CORE_DEFAULTS"
        if len(missing_features) == 8
        else "INCOMPLETE",
        "missing_feature_flags": missing_features,
        "core_default_flags": list(defaulted_features),
    }
