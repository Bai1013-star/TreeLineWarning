"""Train per-species monthly growth regressors from mechanism-based simulations."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.prediction.species_growth import (
    growth_features,
)
from algorithms.prediction.synthetic_training import (
    SPECIES_MAP,
    SPECIES_PROFILES,
    generate_synthetic_dataset,
)
from algorithms.prediction.temporal_attention import FEATURE_NAMES

LOGGER = logging.getLogger(__name__)


def _fit_ridge(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    penalty = np.eye(features.shape[1], dtype=np.float64) * 1e-5
    penalty[0, 0] = 0
    return np.linalg.solve(features.T @ features + penalty, features.T @ targets)


def train_species_growth(
    output_path: str | Path | None = None,
    sample_count: int = 10000,
    sequence_length: int = 18,
    seed: int = 20260728,
    *,
    allow_replace: bool = False,
) -> dict[str, Any]:
    """Train and save a distinct climate/season growth model for each species."""
    if output_path is None or not str(output_path).strip():
        raise ValueError("训练必须显式提供 output_path。")
    output_path = Path(output_path)
    if output_path.exists() and not allow_replace:
        raise FileExistsError(f"训练输出已存在：{output_path}")
    data = generate_synthetic_dataset(sample_count, sequence_length, seed)
    sequences = data["features"].astype(np.float64)
    rng = np.random.default_rng(seed + 41)
    height_index = FEATURE_NAMES.index("height")
    diameter_index = FEATURE_NAMES.index("diameter")
    temp_index = FEATURE_NAMES.index("temperature")
    sunlight_index = FEATURE_NAMES.index("sunlight_hours")
    rainfall_index = FEATURE_NAMES.index("rainfall")
    humidity_index = FEATURE_NAMES.index("humidity")
    id_to_species = {value: key for key, value in SPECIES_MAP.items()}
    species_models: dict[str, Any] = {}

    for species_id, species in id_to_species.items():
        rows = []
        height_targets = []
        diameter_targets = []
        sample_ids = np.flatnonzero(data["species_ids"] == species_id)
        for sample_id in sample_ids:
            final_month = int(data["months"][sample_id])
            for step in range(sequence_length - 1):
                current = sequences[sample_id, step]
                next_row = sequences[sample_id, step + 1]
                next_month = (
                    (final_month - (sequence_length - 1 - (step + 1)) - 1) % 12
                ) + 1
                rows.append(
                    growth_features(
                        next_month,
                        next_row[temp_index],
                        next_row[sunlight_index],
                        next_row[rainfall_index],
                        next_row[humidity_index],
                        data["ages"][sample_id] + (step + 1) / 12,
                        next_row[height_index],
                    )
                )
                height_targets.append(
                    max(0.0, next_row[height_index] - current[height_index])
                )
                diameter_targets.append(
                    max(0.0, next_row[diameter_index] - current[diameter_index])
                )

        x = np.asarray(rows, dtype=np.float64)
        y_height = np.asarray(height_targets, dtype=np.float64)
        y_diameter = np.asarray(diameter_targets, dtype=np.float64)
        order = rng.permutation(len(x))
        split = int(len(order) * 0.8)
        train_idx, test_idx = order[:split], order[split:]
        height_coef = _fit_ridge(x[train_idx], y_height[train_idx])
        diameter_coef = _fit_ridge(x[train_idx], y_diameter[train_idx])
        predicted_height = np.clip(x[test_idx] @ height_coef, 0, 0.30)
        predicted_diameter = np.clip(x[test_idx] @ diameter_coef, 0, 0.12)

        monthly_height = {}
        monthly_diameter = {}
        final_months = data["months"][sample_ids].astype(int)
        for month in range(1, 13):
            # The final step growth is the delta between the last two observations.
            mask = final_months == month
            selected = sequences[sample_ids[mask]]
            if len(selected):
                monthly_height[str(month)] = round(
                    float(
                        np.mean(
                            selected[:, -1, height_index]
                            - selected[:, -2, height_index]
                        )
                    ),
                    5,
                )
                monthly_diameter[str(month)] = round(
                    float(
                        np.mean(
                            selected[:, -1, diameter_index]
                            - selected[:, -2, diameter_index]
                        )
                    ),
                    5,
                )

        species_models[species] = {
            "profile": SPECIES_PROFILES[species],
            "training_rows": int(len(x)),
            "height_coefficients": [round(float(v), 12) for v in height_coef],
            "diameter_coefficients": [round(float(v), 12) for v in diameter_coef],
            "test_height_mae_m_per_month": round(
                float(np.mean(np.abs(predicted_height - y_height[test_idx]))), 6
            ),
            "test_diameter_mae_cm_per_month": round(
                float(np.mean(np.abs(predicted_diameter - y_diameter[test_idx]))),
                6,
            ),
            "monthly_mean_height_growth": monthly_height,
            "monthly_mean_diameter_growth": monthly_diameter,
        }

    report: dict[str, Any] = {
        "model": "per_species_monthly_growth_ridge",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "training_source": "现实生长机理约束的模拟数据",
        "sample_count": sample_count,
        "sequence_length_months": sequence_length,
        "feature_names": [
            "constant",
            "temperature",
            "temperature_squared",
            "sunlight_hours",
            "rainfall",
            "humidity",
            "month_sine",
            "month_cosine",
            "temperature_x_sunlight",
            "rainfall_x_sunlight",
            "tree_age_years",
            "tree_height_m",
        ],
        "species_models": species_models,
        "limitations": [
            "当前参数由现实规律约束的模拟样本训练，适合系统联调、趋势预警与演示。",
            "用户提供真实月度巡检数据后，应重新训练并完成地区与树种校准。",
            "预测不能替代现场测距、树木健康检查和电力行业安全处置规程。",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        dir=output_path.parent,
        encoding="utf-8",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        json.dump(report, stream, ensure_ascii=False, indent=2)
    try:
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    LOGGER.info(
        "Species growth model trained: samples=%d, species=%d",
        sample_count,
        len(species_models),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="训练多树种月度生长模拟模型")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--sequence-length", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    train_species_growth(args.output, args.samples, args.sequence_length, args.seed)


if __name__ == "__main__":
    main()
