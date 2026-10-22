"""Pure NumPy LSTM with recency, season, and species-feature attention."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_NAMES = (
    "height",
    "diameter",
    "vertical_distance",
    "horizontal_distance",
    "clearance_distance",
    "wind_offset_distance",
    "voltage_level_kv",
    "maximum_sag_winter",
    "maximum_sag_summer",
    "wind_speed",
    "rainfall",
    "humidity",
    "temperature",
    "sunlight_hours",
)
_UNCONFIGURED_MODEL_PATH = Path("__treeline_temporal_model_not_configured__")
# Deprecated compatibility sentinel.  It is never used as a repository model path.
DEFAULT_MODEL_PATH = _UNCONFIGURED_MODEL_PATH
MODEL_PATH_ENVIRONMENT_VARIABLE = "TREELINE_WARNING_TEMPORAL_MODEL_PATH"


def resolve_temporal_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve an explicit model path or the deployment environment setting."""
    candidate = model_path
    if candidate is None or Path(candidate) == _UNCONFIGURED_MODEL_PATH:
        candidate = os.environ.get(MODEL_PATH_ENVIRONMENT_VARIABLE)
    if candidate is None or not str(candidate).strip():
        raise RuntimeError(
            "未配置时序模型；请传入 model_path 或设置 "
            f"{MODEL_PATH_ENVIRONMENT_VARIABLE}。"
        )
    return Path(candidate).expanduser()


@dataclass(frozen=True)
class TemporalModelConfig:
    feature_count: int = len(FEATURE_NAMES)
    hidden_size: int = 24
    species_count: int = 7
    seed: int = 20260728


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -30, 30)
    return 1 / (1 + np.exp(-value))


class NumpyAttentionLSTM:
    """Small trainable LSTM; gradients are calculated with BPTT."""

    def __init__(self, config: TemporalModelConfig | None = None):
        self.config = config or TemporalModelConfig()
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        h, f = cfg.hidden_size, cfg.feature_count
        self.params: dict[str, np.ndarray] = {
            "species_logits": rng.normal(0, 0.15, (cfg.species_count, f)).astype(np.float32),
            "season_w": rng.normal(0, 0.18, (3, f)).astype(np.float32),
            "season_b": np.zeros(f, dtype=np.float32),
            "wx": rng.normal(0, np.sqrt(1 / f), (f, 4 * h)).astype(np.float32),
            "wh": rng.normal(0, np.sqrt(1 / h), (h, 4 * h)).astype(np.float32),
            "b": np.concatenate(
                (np.zeros(h), np.ones(h), np.zeros(2 * h))
            ).astype(np.float32),
            "w_time": rng.normal(0, 0.12, h).astype(np.float32),
            "recency_raw": np.asarray([1.5], dtype=np.float32),
            "w_out": rng.normal(0, 0.12, h).astype(np.float32),
            "b_out": np.zeros(1, dtype=np.float32),
        }
        self._adam_m = {name: np.zeros_like(value) for name, value in self.params.items()}
        self._adam_v = {name: np.zeros_like(value) for name, value in self.params.items()}
        self._step = 0

    def forward(
        self,
        features: np.ndarray,
        species_ids: np.ndarray,
        month: np.ndarray,
        *,
        training: bool = False,
    ) -> dict[str, Any]:
        p = self.params
        batch, steps, _ = features.shape
        hidden_size = self.config.hidden_size
        species_weights = _sigmoid(p["species_logits"][species_ids])
        season_input = np.stack(
            (
                np.cos(2 * np.pi * (month - 7) / 12),
                features[:, :, FEATURE_NAMES.index("temperature")].mean(axis=1),
                features[:, :, FEATURE_NAMES.index("sunlight_hours")].mean(axis=1),
            ),
            axis=1,
        ).astype(np.float32)
        season_pre = season_input @ p["season_w"] + p["season_b"]
        season_weights = _sigmoid(season_pre)
        attended = features * species_weights[:, None, :] * season_weights[:, None, :]

        hidden = np.zeros((batch, hidden_size), dtype=np.float32)
        cell = np.zeros_like(hidden)
        hidden_steps = []
        lstm_cache = []
        for index in range(steps):
            previous_hidden = hidden
            previous_cell = cell
            z = attended[:, index] @ p["wx"] + previous_hidden @ p["wh"] + p["b"]
            i = _sigmoid(z[:, :hidden_size])
            f = _sigmoid(z[:, hidden_size : 2 * hidden_size])
            g = np.tanh(z[:, 2 * hidden_size : 3 * hidden_size])
            o = _sigmoid(z[:, 3 * hidden_size :])
            cell = f * previous_cell + i * g
            hidden = o * np.tanh(cell)
            hidden_steps.append(hidden)
            if training:
                lstm_cache.append(
                    (attended[:, index], previous_hidden, previous_cell, i, f, g, o, cell)
                )
        hidden_sequence = np.stack(hidden_steps, axis=1)

        recency = np.linspace(-1, 0, steps, dtype=np.float32)
        recency_strength = np.log1p(np.exp(p["recency_raw"][0]))
        time_logits = hidden_sequence @ p["w_time"] + recency_strength * recency
        time_logits -= time_logits.max(axis=1, keepdims=True)
        time_weights = np.exp(time_logits)
        time_weights /= time_weights.sum(axis=1, keepdims=True)
        context = np.sum(hidden_sequence * time_weights[:, :, None], axis=1)
        prediction = _sigmoid(context @ p["w_out"] + p["b_out"][0])
        result: dict[str, Any] = {
            "risk_score": prediction * 100,
            "time_attention": time_weights,
            "species_feature_attention": species_weights,
            "season_feature_attention": season_weights,
        }
        if training:
            result["_cache"] = {
                "features": features,
                "species_ids": species_ids,
                "species_weights": species_weights,
                "season_input": season_input,
                "season_weights": season_weights,
                "attended": attended,
                "hidden_sequence": hidden_sequence,
                "lstm": lstm_cache,
                "time_weights": time_weights,
                "context": context,
                "prediction": prediction,
                "recency": recency,
            }
        return result

    def train_batch(
        self,
        features: np.ndarray,
        species_ids: np.ndarray,
        month: np.ndarray,
        targets: np.ndarray,
        learning_rate: float = 0.002,
    ) -> float:
        output = self.forward(features, species_ids, month, training=True)
        cache = output["_cache"]
        prediction = cache["prediction"]
        expected = targets / 100
        batch = len(features)
        loss = float(np.mean((prediction - expected) ** 2))
        gradients = {name: np.zeros_like(value) for name, value in self.params.items()}

        d_prediction = 2 * (prediction - expected) / batch
        d_output_logit = d_prediction * prediction * (1 - prediction)
        gradients["w_out"] = cache["context"].T @ d_output_logit
        gradients["b_out"][0] = d_output_logit.sum()
        d_context = d_output_logit[:, None] * self.params["w_out"][None, :]

        hidden_sequence = cache["hidden_sequence"]
        time_weights = cache["time_weights"]
        d_attention = np.sum(d_context[:, None, :] * hidden_sequence, axis=2)
        d_time_logits = time_weights * (
            d_attention - np.sum(d_attention * time_weights, axis=1, keepdims=True)
        )
        gradients["w_time"] = np.sum(hidden_sequence * d_time_logits[:, :, None], axis=(0, 1))
        raw = self.params["recency_raw"][0]
        softplus_derivative = float(_sigmoid(np.asarray([raw]))[0])
        gradients["recency_raw"][0] = (
            np.sum(d_time_logits * cache["recency"][None, :]) * softplus_derivative
        )
        d_hidden = (
            time_weights[:, :, None] * d_context[:, None, :]
            + d_time_logits[:, :, None] * self.params["w_time"][None, None, :]
        )

        d_attended = np.zeros_like(cache["attended"])
        d_hidden_next = np.zeros((batch, self.config.hidden_size), dtype=np.float32)
        d_cell_next = np.zeros_like(d_hidden_next)
        for index in range(features.shape[1] - 1, -1, -1):
            x, previous_hidden, previous_cell, i, f, g, o, cell = cache["lstm"][index]
            dh = d_hidden[:, index] + d_hidden_next
            tanh_cell = np.tanh(cell)
            do = dh * tanh_cell
            dc = dh * o * (1 - tanh_cell**2) + d_cell_next
            df = dc * previous_cell
            d_cell_next = dc * f
            di = dc * g
            dg = dc * i
            dz = np.concatenate(
                (
                    di * i * (1 - i),
                    df * f * (1 - f),
                    dg * (1 - g**2),
                    do * o * (1 - o),
                ),
                axis=1,
            )
            gradients["wx"] += x.T @ dz
            gradients["wh"] += previous_hidden.T @ dz
            gradients["b"] += dz.sum(axis=0)
            d_attended[:, index] = dz @ self.params["wx"].T
            d_hidden_next = dz @ self.params["wh"].T

        species_weights = cache["species_weights"]
        season_weights = cache["season_weights"]
        d_species = np.sum(
            d_attended * features * season_weights[:, None, :], axis=1
        )
        d_species_logits = d_species * species_weights * (1 - species_weights)
        np.add.at(gradients["species_logits"], species_ids, d_species_logits)

        d_season = np.sum(
            d_attended * features * species_weights[:, None, :], axis=1
        )
        d_season_pre = d_season * season_weights * (1 - season_weights)
        gradients["season_w"] = cache["season_input"].T @ d_season_pre
        gradients["season_b"] = d_season_pre.sum(axis=0)

        self._step += 1
        for name, gradient in gradients.items():
            gradient = np.clip(gradient, -5, 5)
            self._adam_m[name] = 0.9 * self._adam_m[name] + 0.1 * gradient
            self._adam_v[name] = 0.999 * self._adam_v[name] + 0.001 * gradient**2
            corrected_m = self._adam_m[name] / (1 - 0.9**self._step)
            corrected_v = self._adam_v[name] / (1 - 0.999**self._step)
            self.params[name] -= learning_rate * corrected_m / (
                np.sqrt(corrected_v) + 1e-8
            )
        return loss

    def save(
        self,
        path: str | Path,
        metadata: dict[str, Any],
        *,
        allow_replace: bool = False,
    ) -> None:
        path = Path(path)
        if path.exists() and not allow_replace:
            raise FileExistsError(f"输出模型已存在：{path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {f"param_{name}": value for name, value in self.params.items()}
        payload["metadata"] = np.asarray(
            json.dumps(metadata, ensure_ascii=False), dtype=np.str_
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=path.suffix or ".npz", dir=path.parent, delete=False
            ) as stream:
                temporary_path = Path(stream.name)
            np.savez_compressed(temporary_path, **payload)
            temporary_path.replace(path)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | Path) -> tuple[NumpyAttentionLSTM, dict[str, Any]]:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            model = cls(TemporalModelConfig(**metadata["config"]))
            for name in model.params:
                value = archive[f"param_{name}"].copy()
                if value.shape != model.params[name].shape:
                    raise ValueError(f"参数 {name} 的形状与模型配置不匹配")
                model.params[name] = value
        return model, metadata


def load_temporal_model(path: str | Path) -> tuple[NumpyAttentionLSTM, dict[str, Any]]:
    """Load a checkpoint while exposing deployment-friendly errors."""
    resolved_path = Path(path)
    try:
        if not resolved_path.is_file():
            raise FileNotFoundError(resolved_path)
        return NumpyAttentionLSTM.load(resolved_path)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(
            f"无法加载时序模型：{resolved_path}；请提供有效的 .npz 模型文件。"
        ) from exc


def build_model(config: TemporalModelConfig | None = None) -> NumpyAttentionLSTM:
    return NumpyAttentionLSTM(config)


def validate_sequence(records: Sequence[dict[str, Any]]) -> None:
    if len(records) < 2:
        raise ValueError("LSTM 时序预测至少需要同一棵树的两个时间点")
    missing = []
    for name in FEATURE_NAMES:
        if name in {
            "wind_offset_distance",
            "voltage_level_kv",
            "maximum_sag_winter",
            "maximum_sag_summer",
        }:
            continue
        if name in {
            "vertical_distance",
            "horizontal_distance",
            "clearance_distance",
        }:
            if any(
                name not in record
                and not (
                    name != "wind_offset_distance" and "distance_to_wire" in record
                )
                for record in records
            ):
                missing.append(name)
        elif any(name not in record for record in records):
            missing.append(name)
    if missing:
        raise ValueError(f"时序数据缺少特征: {', '.join(missing)}")


def risk_from_score(
    score: float, medium_threshold: float = 40, high_threshold: float = 70
) -> tuple[str, str]:
    if score >= high_threshold:
        return "高风险", "建议立即巡检并评估修剪或清障"
    if score >= medium_threshold:
        return "中风险", "建议加强监测并缩短巡检周期"
    return "低风险", "建议按计划正常维护"


class TemporalPredictor:
    """Load a trained checkpoint and serve normalized sequence predictions."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = resolve_temporal_model_path(model_path)
        try:
            self.model, metadata = load_temporal_model(self.model_path)
            self.metadata = dict(metadata)
            self.mean = np.asarray(metadata["normalization"]["mean"], dtype=np.float32)
            self.std = np.asarray(metadata["normalization"]["std"], dtype=np.float32)
            self.species_map = dict(metadata["species_map"])
            self.sequence_length = int(metadata["sequence_length"])
            if self.mean.shape != (len(FEATURE_NAMES),) or self.std.shape != self.mean.shape:
                raise ValueError("归一化参数与特征数量不匹配")
        except RuntimeError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise RuntimeError(
                f"无法加载时序模型：{self.model_path}；请提供结构兼容的 .npz 模型文件。"
            ) from exc

    def predict(self, records: Sequence[dict[str, Any]], species: str) -> dict[str, Any]:
        validate_sequence(records)
        values = np.asarray(
            [
                [_temporal_value(record, name) for name in FEATURE_NAMES]
                for record in records
            ],
            dtype=np.float32,
        )
        if len(values) > self.sequence_length:
            values = values[-self.sequence_length :]
        elif len(values) < self.sequence_length:
            pad = np.repeat(values[:1], self.sequence_length - len(values), axis=0)
            values = np.concatenate((pad, values), axis=0)
        normalized = (values - self.mean) / self.std
        month = float(records[-1].get("month", 7))
        species_id = self.species_map.get(species, self.species_map.get("其他", 0))
        output = self.model.forward(
            normalized[None, :, :],
            np.asarray([species_id], dtype=np.int64),
            np.asarray([month], dtype=np.float32),
        )
        score = round(float(output["risk_score"][0]), 2)
        level, suggestion = risk_from_score(score)
        return {
            "risk_level": level,
            "risk_score": score,
            "suggestion": suggestion,
            "model": "numpy_lstm_triple_attention",
            "attention": {
                "time": [
                    round(float(value), 5) for value in output["time_attention"][0]
                ],
                "species_features": {
                    name: round(float(value), 5)
                    for name, value in zip(
                        FEATURE_NAMES,
                        output["species_feature_attention"][0],
                        strict=True,
                    )
                },
                "season_features": {
                    name: round(float(value), 5)
                    for name, value in zip(
                        FEATURE_NAMES,
                        output["season_feature_attention"][0],
                        strict=True,
                    )
                },
            },
        }


def _temporal_value(record: dict[str, Any], name: str) -> float:
    if name in record:
        return float(record[name])
    if name == "wind_offset_distance":
        return 0.0
    if name == "voltage_level_kv":
        return 110.0
    if name == "maximum_sag_winter":
        return 3.5
    if name == "maximum_sag_summer":
        return 5.0
    if name in {"vertical_distance", "horizontal_distance", "clearance_distance"}:
        return float(record["distance_to_wire"])
    raise KeyError(name)
