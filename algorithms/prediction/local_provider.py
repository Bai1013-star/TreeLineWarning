"""Strict local prediction-provider boundary.

This module intentionally does *not* estimate missing biological, weather, or
electrical inputs.  A configured local core must declare the fields it needs;
otherwise it is reported as unavailable instead of receiving invented values.
"""

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

PREDICTION_INTERFACE_VERSION = "treeline-local-prediction-v1"
PREDICTION_ENTRYPOINT_ENV = "TREELINE_WARNING_PREDICTION_ENTRYPOINT"


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


def _load_configured_core() -> tuple[Callable[[dict[str, Any]], Any] | None, str | None]:
    target = os.environ.get(PREDICTION_ENTRYPOINT_ENV, "").strip()
    if not target:
        return None, None
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        return None, "PREDICTION_ENTRYPOINT_INVALID"
    try:
        entrypoint = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        return None, f"PREDICTION_ENTRYPOINT_UNAVAILABLE: {type(exc).__name__}"
    return entrypoint, None


def _required_fields(core: Callable[[dict[str, Any]], Any]) -> tuple[str, ...] | None:
    required = getattr(core, "PREDICTION_REQUIRED_FIELDS", None)
    if not isinstance(required, (list, tuple)) or not all(
        isinstance(item, str) for item in required
    ):
        return None
    return tuple(required)


def adapt_prediction_core_output(
    raw: Any, package: PredictionInputPackage, elapsed: float
) -> PredictionResultPackage:
    """Validate a core result without assigning results by array position."""

    if not isinstance(raw, dict):
        raise ValueError("prediction core output must be an object")
    entries = raw.get("entries")
    if not isinstance(entries, list):
        raise ValueError("prediction core output must contain entries[]")
    known_ids = {item.get("persistent_tree_id") for item in package.observations}
    normalized = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("persistent_tree_id") not in known_ids:
            raise ValueError("prediction core returned an unknown persistent_tree_id")
        normalized.append(dict(entry))
    horizons = raw.get("horizons", [])
    if not isinstance(horizons, list) or not all(isinstance(item, str) for item in horizons):
        raise ValueError("prediction core horizons must be string labels")
    return PredictionResultPackage(
        schema_version=PREDICTION_INTERFACE_VERSION,
        project_id=package.project_id,
        group_id=package.group_id,
        span_id=package.span_id,
        source_science_run_id=package.source_science_run_id,
        status="COMPLETE",
        entries=tuple(normalized),
        horizons=tuple(horizons),
        missing_fields=(),
        diagnostic=None,
        runtime_seconds=elapsed,
    )


class LocalPredictionProvider:
    """Calls an explicitly configured, contract-declaring local prediction core."""

    interface_version = PREDICTION_INTERFACE_VERSION

    def run(self, package: PredictionInputPackage) -> PredictionResultPackage:
        started = time.perf_counter()
        core, load_error = _load_configured_core()
        if core is None:
            return PredictionResultPackage(
                schema_version=PREDICTION_INTERFACE_VERSION,
                project_id=package.project_id,
                group_id=package.group_id,
                span_id=package.span_id,
                source_science_run_id=package.source_science_run_id,
                status="BLOCKED",
                entries=(),
                horizons=(),
                missing_fields=(),
                diagnostic=load_error or "PREDICTION_CORE_NOT_CONFIGURED",
                runtime_seconds=time.perf_counter() - started,
            )
        required = _required_fields(core)
        if required is None:
            return PredictionResultPackage(
                schema_version=PREDICTION_INTERFACE_VERSION,
                project_id=package.project_id,
                group_id=package.group_id,
                span_id=package.span_id,
                source_science_run_id=package.source_science_run_id,
                status="BLOCKED",
                entries=(), horizons=(), missing_fields=(),
                diagnostic="PREDICTION_CORE_HAS_NO_DECLARED_INTERFACE",
                runtime_seconds=time.perf_counter() - started,
            )
        available = set(package.observations[0]) if package.observations else set()
        missing = tuple(sorted(set(required) - available))
        if missing:
            return PredictionResultPackage(
                schema_version=PREDICTION_INTERFACE_VERSION,
                project_id=package.project_id,
                group_id=package.group_id,
                span_id=package.span_id,
                source_science_run_id=package.source_science_run_id,
                status="BLOCKED",
                entries=(),
                horizons=(),
                missing_fields=missing,
                diagnostic="PREDICTION_INPUT_FIELDS_UNAVAILABLE",
                runtime_seconds=time.perf_counter() - started,
            )
        try:
            raw = core(package.to_dict())
            return adapt_prediction_core_output(raw, package, time.perf_counter() - started)
        except (TypeError, ValueError, KeyError) as exc:
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
