"""Species-specific monthly tree growth model used by long-term forecasts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.prediction.species import SPECIES_ANNUAL_HEIGHT_GROWTH

_UNCONFIGURED_MODEL_PATH = Path("__treeline_species_growth_model_not_configured__")
# Deprecated compatibility sentinel for callers that imported this name.  It is
# deliberately not a repository-relative default and is treated as unconfigured.
DEFAULT_SPECIES_GROWTH_MODEL_PATH = _UNCONFIGURED_MODEL_PATH
MODEL_PATH_ENVIRONMENT_VARIABLE = "TREELINE_WARNING_SPECIES_GROWTH_MODEL_PATH"

# Fraction of one year's biological growth assigned to each calendar month.
# Values sum to 1.0.  This phenology curve keeps winter dormant, concentrates
# most extension in spring/summer, and prevents the old near-uniform monthly
# scaling from reporting implausible autumn/winter growth.
MONTHLY_PHENOLOGY_WEIGHTS = (
    0.005,
    0.010,
    0.080,
    0.145,
    0.180,
    0.205,
    0.175,
    0.100,
    0.050,
    0.025,
    0.015,
    0.010,
)
CONIFER_PHENOLOGY_WEIGHTS = (0.002, 0.006, 0.080, 0.200, 0.240, 0.220, 0.140, 0.065, 0.025, 0.010, 0.006, 0.006)
RING_POROUS_PHENOLOGY_WEIGHTS = (0.002, 0.005, 0.120, 0.210, 0.230, 0.180, 0.110, 0.060, 0.035, 0.020, 0.015, 0.013)
EVERGREEN_BROADLEAF_PHENOLOGY_WEIGHTS = (0.010, 0.015, 0.080, 0.130, 0.160, 0.180, 0.160, 0.110, 0.070, 0.040, 0.025, 0.020)


def phenology_weights_for_species(species: str) -> tuple[float, ...]:
    name = str(species or "其他")
    if name in {"松树", "杉树", "柏树"}:
        return CONIFER_PHENOLOGY_WEIGHTS
    if name in {"榆树", "国槐", "白蜡"}:
        return RING_POROUS_PHENOLOGY_WEIGHTS
    if name in {"樟树"}:
        return EVERGREEN_BROADLEAF_PHENOLOGY_WEIGHTS
    return MONTHLY_PHENOLOGY_WEIGHTS


def monthly_phenology_weight(month: int, species: str = "其他") -> float:
    if not 1 <= int(month) <= 12:
        raise ValueError("month 必须在 1—12 之间")
    return phenology_weights_for_species(species)[int(month) - 1]


def resolve_species_growth_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve an explicit model path or the deployment environment setting."""
    candidate = model_path
    if candidate is None or Path(candidate) == _UNCONFIGURED_MODEL_PATH:
        candidate = os.environ.get(MODEL_PATH_ENVIRONMENT_VARIABLE)
    if candidate is None or not str(candidate).strip():
        raise RuntimeError(
            "未配置树种生长模型；请传入 model_path 或设置 "
            f"{MODEL_PATH_ENVIRONMENT_VARIABLE}。"
        )
    return Path(candidate).expanduser()


def growth_features(
    month: int,
    temperature: float,
    sunlight_hours: float,
    rainfall: float,
    humidity: float,
    age: float,
    height: float,
) -> np.ndarray:
    """Create the same nonlinear climate/season features used during training."""
    angle = 2 * math.pi * (int(month) - 1) / 12
    temperature = float(temperature)
    sunlight_hours = float(sunlight_hours)
    rainfall = float(rainfall)
    humidity = float(humidity)
    return np.asarray(
        [
            1.0,
            temperature,
            temperature**2,
            sunlight_hours,
            rainfall,
            humidity,
            math.sin(angle),
            math.cos(angle),
            temperature * sunlight_hours,
            min(rainfall, 120.0) * sunlight_hours,
            float(age),
            float(height),
        ],
        dtype=np.float64,
    )


class SpeciesGrowthModel:
    """Load trained per-species regressors and predict one month of growth."""

    def __init__(
        self,
        model_path: str | Path | None = None,
    ):
        self.model_path = resolve_species_growth_model_path(model_path)
        try:
            if not self.model_path.is_file():
                raise FileNotFoundError(self.model_path)
            self.metadata: dict[str, Any] = json.loads(
                self.model_path.read_text(encoding="utf-8")
            )
            species_models = self.metadata["species_models"]
            if not isinstance(species_models, dict):
                raise TypeError("species_models 必须是对象")
            self.models = dict(species_models)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise RuntimeError(
                f"无法加载树种生长模型：{self.model_path}；请提供有效的模型 JSON 文件。"
            ) from exc

    def predict(
        self,
        species: str,
        *,
        month: int,
        temperature: float,
        sunlight_hours: float,
        rainfall: float,
        humidity: float,
        age: float,
        height: float,
    ) -> tuple[float, float]:
        model = self.models.get(species, self.models.get("其他"))
        if not model:
            raise RuntimeError("树种生长模型中缺少“其他”回退模型")
        features = growth_features(
            month,
            temperature,
            sunlight_hours,
            rainfall,
            humidity,
            age,
            height,
        )
        raw_height = max(0.0, float(features @ np.asarray(model["height_coefficients"])))
        raw_diameter = max(0.0, float(features @ np.asarray(model["diameter_coefficients"])))
        phenology = monthly_phenology_weight(month, species)
        # Trained coefficients historically represented a typical month. Map
        # that annualized amount onto the actual calendar month.
        height = raw_height * 12.0 * phenology
        diameter = raw_diameter * 12.0 * phenology
        local_reference = SPECIES_ANNUAL_HEIGHT_GROWTH.get(species)
        if local_reference is not None:
            # Keep the field-calibrated annual reference authoritative while
            # distributing it by daily/monthly phenology rather than dividing
            # equally into twelve months.
            height = float(local_reference["reference"]) * phenology
        # Simulated training data cannot justify extrapolating extreme values.
        return (
            float(np.clip(height, 0.0, 0.30)),
            float(np.clip(diameter, 0.0, 0.40)),
        )
