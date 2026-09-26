"""Generate mechanism-constrained tree histories and train the NumPy LSTM."""

from __future__ import annotations

import argparse
import json
import logging
import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.prediction.temporal_attention import (
    FEATURE_NAMES,
    TemporalModelConfig,
    build_model,
)

LOGGER = logging.getLogger(__name__)


SPECIES_PROFILES = {
    "松树": {"growth": 0.90, "opt_temp": 20.0, "light": 0.75, "risk": 1.08},
    "杉树": {"growth": 0.82, "opt_temp": 19.0, "light": 0.68, "risk": 0.96},
    "法国梧桐": {"growth": 2.90, "opt_temp": 23.0, "light": 0.96, "risk": 1.20},
    "杨树": {"growth": 2.90, "opt_temp": 23.0, "light": 1.00, "risk": 1.18},
    "榆树": {"growth": 1.50, "opt_temp": 21.0, "light": 0.82, "risk": 1.05},
    "国槐": {"growth": 0.80, "opt_temp": 24.0, "light": 0.78, "risk": 0.96},
    "白蜡": {"growth": 1.00, "opt_temp": 22.0, "light": 0.84, "risk": 1.04},
    "柳树": {"growth": 1.18, "opt_temp": 22.0, "light": 0.93, "risk": 1.12},
    "柏树": {"growth": 0.66, "opt_temp": 21.0, "light": 0.62, "risk": 0.90},
    "樟树": {"growth": 1.02, "opt_temp": 25.0, "light": 0.86, "risk": 1.02},
    "其他": {"growth": 0.90, "opt_temp": 22.0, "light": 0.78, "risk": 1.00},
}
SPECIES_MAP = {name: index for index, name in enumerate(SPECIES_PROFILES)}


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _risk_target(sequence: np.ndarray, profile: dict[str, float], rng) -> float:
    height = sequence[:, FEATURE_NAMES.index("height")]
    diameter = sequence[:, FEATURE_NAMES.index("diameter")]
    vertical = sequence[:, FEATURE_NAMES.index("vertical_distance")]
    horizontal = sequence[:, FEATURE_NAMES.index("horizontal_distance")]
    clearance = sequence[:, FEATURE_NAMES.index("clearance_distance")]
    wind_offset = sequence[:, FEATURE_NAMES.index("wind_offset_distance")]
    wind = sequence[:, FEATURE_NAMES.index("wind_speed")]
    rainfall = sequence[:, FEATURE_NAMES.index("rainfall")]
    temperature = sequence[:, FEATURE_NAMES.index("temperature")]
    voltage = sequence[:, FEATURE_NAMES.index("voltage_level_kv")]
    maximum_sag = np.maximum(
        sequence[:, FEATURE_NAMES.index("maximum_sag_winter")],
        sequence[:, FEATURE_NAMES.index("maximum_sag_summer")],
    )
    geometric = np.maximum(0, np.hypot(vertical, horizontal) - wind_offset)
    distance = np.minimum(clearance, geometric)
    weights = np.exp(np.linspace(-2.2, 0.0, len(sequence)))
    weights /= weights.sum()
    recent_growth = np.diff(height, prepend=height[0])

    sag_allowance = np.interp(
        voltage,
        (10, 35, 66, 110, 220, 330, 500, 750, 1000),
        (2, 3, 4, 5, 6, 7, 9, 11, 13),
    )
    score = (
        _clip((8 - distance[-1]) / 8, 0, 1) * 38
        + _clip(float(np.sum(wind * weights)) / 25, 0, 1) * 20
        + _clip(float(np.sum(rainfall * weights)) / 120, 0, 1) * 9
        + _clip(height[-1] / 35, 0, 1) * 10
        + _clip(height[-1] * 100 / max(diameter[-1], 1) / 120, 0, 1) * 8
        + _clip(float(np.sum(recent_growth * weights)) / 0.12, 0, 1) * 12
        + _clip(float(np.sum(np.maximum(temperature - 32, 0) * weights)) / 8, 0, 1) * 5
        + _clip(float(np.max((maximum_sag - sag_allowance) / sag_allowance)), 0, 1)
        * 10
    ) * profile["risk"]
    return _clip(score + rng.normal(0, 2), 0, 100)


def generate_synthetic_dataset(
    sample_count: int = 2000,
    sequence_length: int = 18,
    seed: int = 20260728,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = np.zeros(
        (sample_count, sequence_length, len(FEATURE_NAMES)), dtype=np.float32
    )
    species_ids = np.zeros(sample_count, dtype=np.int64)
    months = np.zeros(sample_count, dtype=np.float32)
    ages = np.zeros(sample_count, dtype=np.float32)
    targets = np.zeros(sample_count, dtype=np.float32)
    names = list(SPECIES_PROFILES)

    for sample in range(sample_count):
        species = names[int(rng.integers(0, len(names)))]
        profile = SPECIES_PROFILES[species]
        species_ids[sample] = SPECIES_MAP[species]
        age = float(rng.integers(4, 81))
        ages[sample] = age
        start_month = int(rng.integers(1, 13))
        climate_offset = rng.normal(0, 3)
        scenario = rng.choice(("normal", "watch", "critical"), p=(0.50, 0.32, 0.18))
        if scenario == "critical":
            height, diameter, distance, wind_bias = (
                rng.uniform(22, 36),
                rng.uniform(12, 52),
                rng.uniform(0.1, 3.8),
                rng.uniform(5, 10),
            )
        elif scenario == "watch":
            height, diameter, distance, wind_bias = (
                rng.uniform(14, 32),
                rng.uniform(15, 65),
                rng.uniform(2.5, 9),
                rng.uniform(1, 5),
            )
        else:
            height, diameter, distance, wind_bias = (
                rng.uniform(4, 26),
                rng.uniform(18, 75),
                rng.uniform(7, 20),
                0,
            )
        lean_factor = rng.uniform(0.2, 0.8)
        voltage = float(
            rng.choice((10, 35, 66, 110, 220, 330, 500, 750, 1000))
        )
        sag_reference = float(
            np.interp(
                voltage,
                (10, 35, 66, 110, 220, 330, 500, 750, 1000),
                (2, 3, 4, 5, 6, 7, 9, 11, 13),
            )
        )
        winter_sag = sag_reference * rng.uniform(0.55, 0.95)
        summer_sag = winter_sag * rng.uniform(1.08, 1.35)
        for step in range(sequence_length):
            month = ((start_month + step - 1) % 12) + 1
            summer = math.cos(2 * math.pi * (month - 7) / 12)
            temperature = 17 + 11 * summer + climate_offset + rng.normal(0, 2.3)
            sunlight = _clip(8.2 + 3.2 * summer + rng.normal(0, 0.8), 1.5, 14.5)
            rain_base = 43 + 24 * math.cos(2 * math.pi * (month - 7.5) / 12)
            rainfall = _clip(rng.gamma(2, max(rain_base, 5) / 2), 0, 180)
            humidity = _clip(
                48 + rainfall * 0.24 - temperature * 0.25 + rng.normal(0, 7),
                20,
                100,
            )
            wind = _clip(
                rng.weibull(2) * 8
                + wind_bias
                + (rng.random() < 0.05) * rng.uniform(8, 20),
                0,
                38,
            )
            thermal = math.exp(-((temperature - profile["opt_temp"]) / 10) ** 2)
            light_response = min(sunlight / 10, 1.2) * profile["light"]
            moisture = _clip((rainfall + 20) / 85, 0.35, 1.15)
            summer_activity = max(0.08, (summer + 1) / 2)
            growth = max(
                0,
                0.095
                * profile["growth"]
                * _clip(1.24 - (age + step / 12) / 105, 0.38, 1.18)
                * thermal
                * light_response
                * moisture
                * summer_activity
                + rng.normal(0, 0.006),
            )
            height += growth
            # Height is in metres and DBH is in centimetres; this produces
            # roughly 0.4–1.8 cm/year depending on species and climate.
            diameter += max(0, growth * rng.uniform(1.2, 2.4))
            distance = max(0.1, distance - growth * lean_factor)
            wind_offset = _clip(
                wind * height / 260 + rng.normal(0, 0.08), 0, 5
            )
            vertical_distance = max(0.1, distance + rng.uniform(-0.5, 2.5))
            horizontal_distance = max(0.1, distance + rng.uniform(-0.5, 3.5))
            features[sample, step] = (
                height,
                diameter,
                vertical_distance,
                horizontal_distance,
                distance,
                wind_offset,
                voltage,
                winter_sag,
                summer_sag,
                wind,
                rainfall,
                humidity,
                temperature,
                sunlight,
            )
            months[sample] = month
        targets[sample] = _risk_target(features[sample], profile, rng)
    return {
        "features": features,
        "species_ids": species_ids,
        "months": months,
        "ages": ages,
        "targets": targets,
    }


def _risk_class(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 70, 2, np.where(values >= 40, 1, 0))


def train(
    output_path: str | Path | None = None,
    sample_count: int = 2000,
    sequence_length: int = 18,
    epochs: int = 30,
    seed: int = 20260728,
    *,
    report_path: str | Path | None = None,
    allow_replace: bool = False,
) -> dict[str, Any]:
    if output_path is None or not str(output_path).strip():
        raise ValueError("训练必须显式提供 output_path。")
    output_path = Path(output_path)
    if output_path.exists() and not allow_replace:
        raise FileExistsError(f"训练输出已存在：{output_path}")
    if report_path is not None and not str(report_path).strip():
        raise ValueError("report_path 不能为空字符串。")
    report_path = Path(report_path) if report_path is not None else None
    if report_path is not None and report_path.exists() and not allow_replace:
        raise FileExistsError(f"训练报告已存在：{report_path}")
    rng = np.random.default_rng(seed)
    data = generate_synthetic_dataset(sample_count, sequence_length, seed)
    indices = rng.permutation(sample_count)
    train_end, valid_end = int(sample_count * 0.75), int(sample_count * 0.875)
    train_idx, valid_idx, test_idx = (
        indices[:train_end],
        indices[train_end:valid_end],
        indices[valid_end:],
    )
    train_values = data["features"][train_idx]
    mean = train_values.reshape(-1, len(FEATURE_NAMES)).mean(axis=0)
    std = train_values.reshape(-1, len(FEATURE_NAMES)).std(axis=0)
    std = np.where(std < 1e-6, 1, std).astype(np.float32)
    normalized = ((data["features"] - mean) / std).astype(np.float32)

    config = TemporalModelConfig(species_count=len(SPECIES_MAP), seed=seed)
    model = build_model(config)
    best_params = None
    best_valid = float("inf")
    stale = 0
    history = []
    batch_size = 128
    for epoch in range(1, epochs + 1):
        shuffled = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(shuffled), batch_size):
            index = shuffled[start : start + batch_size]
            losses.append(
                model.train_batch(
                    normalized[index],
                    data["species_ids"][index],
                    data["months"][index],
                    data["targets"][index],
                    learning_rate=0.002,
                )
            )
        valid_prediction = model.forward(
            normalized[valid_idx],
            data["species_ids"][valid_idx],
            data["months"][valid_idx],
        )["risk_score"]
        valid_mae = float(np.mean(np.abs(valid_prediction - data["targets"][valid_idx])))
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_mae": valid_mae}
        )
        LOGGER.info(
            "Temporal model epoch=%d train_loss=%.5f validation_mae=%.3f",
            epoch,
            np.mean(losses),
            valid_mae,
        )
        if valid_mae < best_valid - 0.02:
            best_valid = valid_mae
            best_params = {name: value.copy() for name, value in model.params.items()}
            stale = 0
        else:
            stale += 1
            if stale >= 6:
                break
    model.params = best_params

    output = model.forward(
        normalized[test_idx],
        data["species_ids"][test_idx],
        data["months"][test_idx],
    )
    predicted, expected = output["risk_score"], data["targets"][test_idx]
    mae = float(np.mean(np.abs(predicted - expected)))
    rmse = float(np.sqrt(np.mean((predicted - expected) ** 2)))
    predicted_class = _risk_class(predicted)
    expected_class = _risk_class(expected)
    accuracy = float(np.mean(predicted_class == expected_class))
    class_names = ("低风险", "中风险", "高风险")
    class_metrics = {}
    for class_id, class_name in enumerate(class_names):
        true_positive = int(
            np.sum((predicted_class == class_id) & (expected_class == class_id))
        )
        expected_count = int(np.sum(expected_class == class_id))
        predicted_count = int(np.sum(predicted_class == class_id))
        class_metrics[class_name] = {
            "test_count": expected_count,
            "precision": round(true_positive / max(predicted_count, 1), 4),
            "recall": round(true_positive / max(expected_count, 1), 4),
        }
    time_attention = output["time_attention"]
    feature_attention = output["species_feature_attention"]
    height_growth = np.diff(data["features"][:, :, 0], axis=1)
    step_months = np.empty_like(height_growth)
    for step in range(1, sequence_length):
        months_back = sequence_length - 1 - step
        step_months[:, step - 1] = ((data["months"] - months_back - 1) % 12) + 1
    summer_growth = float(height_growth[np.isin(step_months, (6, 7, 8))].mean())
    winter_growth = float(height_growth[np.isin(step_months, (12, 1, 2))].mean())
    all_classes = _risk_class(data["targets"])
    species_attention_by_name = _sigmoid_for_report(model.params["species_logits"])

    metadata = {
        "config": asdict(config),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "species_map": SPECIES_MAP,
        "feature_names": FEATURE_NAMES,
        "sequence_length": sequence_length,
        "training_source": "mechanism_constrained_synthetic_v1",
        "seed": seed,
        "sample_count": sample_count,
    }
    model.save(output_path, metadata, allow_replace=allow_replace)
    quarter = max(1, sequence_length // 4)
    report = {
        "model_path": str(output_path),
        "data_source": "机理约束合成数据；尚未使用真实巡检标签",
        "sample_count": sample_count,
        "sequence_length_months": sequence_length,
        "epochs_completed": len(history),
        "test_samples": len(test_idx),
        "test_mae": round(mae, 4),
        "test_rmse": round(rmse, 4),
        "risk_class_accuracy": round(accuracy, 4),
        "risk_class_metrics": class_metrics,
        "training_class_distribution": {
            class_name: int(np.sum(all_classes == class_id))
            for class_id, class_name in enumerate(class_names)
        },
        "seasonal_growth_check": {
            "summer_mean_height_growth": round(summer_growth, 6),
            "winter_mean_height_growth": round(winter_growth, 6),
            "summer_to_winter_ratio": round(summer_growth / max(winter_growth, 1e-8), 4),
        },
        "recency_attention": {
            "oldest_quarter_mean": round(float(time_attention[:, :quarter].mean()), 6),
            "latest_quarter_mean": round(float(time_attention[:, -quarter:].mean()), 6),
        },
        "mean_species_feature_attention": {
            name: round(float(value), 6)
            for name, value in zip(
                FEATURE_NAMES, feature_attention.mean(axis=0), strict=True
            )
        },
        "species_feature_attention_by_species": {
            species_name: {
                feature_name: round(float(value), 6)
                for feature_name, value in zip(
                    FEATURE_NAMES, species_attention_by_name[index], strict=True
                )
            }
            for species_name, index in SPECIES_MAP.items()
        },
        "last_epoch": history[-1],
        "limitations": [
            "指标只反映合成数据内部泛化能力，不代表真实线路现场精度",
            "投入业务前必须用真实历史巡检与风险标签做迁移学习和校准",
        ],
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            dir=report_path.parent,
            encoding="utf-8",
            delete=False,
        ) as stream:
            temporary_report_path = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2)
        try:
            temporary_report_path.replace(report_path)
        except Exception:
            temporary_report_path.unlink(missing_ok=True)
            raise
    LOGGER.info(
        "Temporal model trained: samples=%d, epochs=%d, test_mae=%.4f",
        sample_count,
        len(history),
        mae,
    )
    return report


def _sigmoid_for_report(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -30, 30)
    return 1 / (1 + np.exp(-value))


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 TreeLineWarning LSTM 注意力模型")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--sequence-length", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    train(
        args.output,
        args.samples,
        args.sequence_length,
        args.epochs,
        args.seed,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
