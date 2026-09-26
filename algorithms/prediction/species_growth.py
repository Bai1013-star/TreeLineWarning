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
        height = float(features @ np.asarray(model["height_coefficients"]))
        diameter = float(features @ np.asarray(model["diameter_coefficients"]))
        local_reference = SPECIES_ANNUAL_HEIGHT_GROWTH.get(species)
        if local_reference is not None:
            # Keep the user-provided FengTai annual growth rate authoritative
            # for these species, while retaining a smooth dormant/growing
            # season pattern in monthly forecasts.
            angle = 2 * math.pi * (int(month) - 4) / 12
            seasonal_factor = 1.0 + 0.35 * math.sin(angle)
            height = float(local_reference["reference"]) / 12 * seasonal_factor
        # Simulated training data cannot justify extrapolating extreme values.
        return (
            float(np.clip(height, 0.0, 0.30)),
            float(np.clip(diameter, 0.0, 0.40)),
        )
