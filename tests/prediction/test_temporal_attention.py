import numpy as np
import pytest
from algorithms.prediction.species_growth_training import train_species_growth
from algorithms.prediction.synthetic_training import train as train_synthetic
from algorithms.prediction.temporal_attention import (
    FEATURE_NAMES,
    NumpyAttentionLSTM,
    TemporalModelConfig,
    TemporalPredictor,
    risk_from_score,
    validate_sequence,
)


def _write_predictor_model(path):
    model = NumpyAttentionLSTM(
        TemporalModelConfig(hidden_size=8, species_count=1, seed=9)
    )
    model.save(
        path,
        {
            "config": {
                "feature_count": len(FEATURE_NAMES),
                "hidden_size": 8,
                "species_count": 1,
                "seed": 9,
            },
            "normalization": {
                "mean": [0.0] * len(FEATURE_NAMES),
                "std": [1.0] * len(FEATURE_NAMES),
            },
            "species_map": {"其他": 0},
            "sequence_length": 2,
        },
    )
    return path


def test_temporal_sequence_validation():
    row = {name: 1.0 for name in FEATURE_NAMES}
    validate_sequence([row, row])


def test_temporal_sequence_requires_history():
    row = {name: 1.0 for name in FEATURE_NAMES}
    with pytest.raises(ValueError, match="至少需要"):
        validate_sequence([row])


@pytest.mark.parametrize(
    ("score", "level"),
    [(10, "低风险"), (40, "中风险"), (69.9, "中风险"), (70, "高风险")],
)
def test_risk_thresholds(score, level):
    assert risk_from_score(score)[0] == level


def test_numpy_lstm_forward_attention_and_training():
    model = NumpyAttentionLSTM(
        TemporalModelConfig(hidden_size=8, species_count=2, seed=3)
    )
    features = np.zeros((6, 5, len(FEATURE_NAMES)), dtype=np.float32)
    species = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    months = np.asarray([1, 4, 7, 10, 7, 7], dtype=np.float32)
    targets = np.asarray([10, 25, 45, 60, 75, 90], dtype=np.float32)
    before = model.forward(features, species, months)
    loss = model.train_batch(features, species, months, targets)
    after = model.forward(features, species, months)

    assert np.isfinite(loss)
    assert before["risk_score"].shape == (6,)
    assert np.allclose(after["time_attention"].sum(axis=1), 1)
    assert after["time_attention"][0, -1] > after["time_attention"][0, 0]


def test_numpy_lstm_round_trip(tmp_path):
    config = TemporalModelConfig(hidden_size=8, species_count=2, seed=9)
    model = NumpyAttentionLSTM(config)
    path = tmp_path / "model.npz"
    metadata = {
        "config": {
            "feature_count": len(FEATURE_NAMES),
            "hidden_size": 8,
            "species_count": 2,
            "seed": 9,
        }
    }
    model.save(path, metadata)
    loaded, loaded_metadata = NumpyAttentionLSTM.load(path)
    assert loaded_metadata == metadata
    assert np.array_equal(loaded.params["wx"], model.params["wx"])


def test_temporal_predictor_requires_explicit_path_or_environment(monkeypatch):
    monkeypatch.delenv("TREELINE_WARNING_TEMPORAL_MODEL_PATH", raising=False)
    with pytest.raises(RuntimeError, match="TREELINE_WARNING_TEMPORAL_MODEL_PATH"):
        TemporalPredictor()


def test_temporal_predictor_prefers_explicit_path_over_environment(tmp_path, monkeypatch):
    explicit_path = _write_predictor_model(tmp_path / "explicit.npz")
    environment_path = _write_predictor_model(tmp_path / "environment.npz")
    monkeypatch.setenv("TREELINE_WARNING_TEMPORAL_MODEL_PATH", str(environment_path))

    predictor = TemporalPredictor(explicit_path)

    assert predictor.model_path == explicit_path


def test_temporal_predictor_uses_environment_path_and_wraps_bad_models(tmp_path, monkeypatch):
    model_path = _write_predictor_model(tmp_path / "environment.npz")
    monkeypatch.setenv("TREELINE_WARNING_TEMPORAL_MODEL_PATH", str(model_path))
    assert TemporalPredictor().model_path == model_path

    broken_path = tmp_path / "broken.npz"
    broken_path.write_bytes(b"not a numpy model")
    with pytest.raises(RuntimeError) as error:
        TemporalPredictor(broken_path)
    assert str(broken_path) in str(error.value)
    assert error.value.__cause__ is not None

    incompatible_path = tmp_path / "incompatible.npz"
    model = NumpyAttentionLSTM(
        TemporalModelConfig(hidden_size=8, species_count=1, seed=7)
    )
    model.save(
        incompatible_path,
        {"config": {"feature_count": len(FEATURE_NAMES), "hidden_size": 8}},
    )
    with pytest.raises(RuntimeError, match="incompatible.npz"):
        TemporalPredictor(incompatible_path)


def test_training_requires_explicit_output_paths():
    with pytest.raises(ValueError, match="output_path"):
        train_species_growth()
    with pytest.raises(ValueError, match="output_path"):
        train_synthetic()


def test_synthetic_training_writes_only_the_explicit_tmp_model_path(tmp_path):
    output_path = tmp_path / "trained.npz"
    report = train_synthetic(
        output_path,
        sample_count=16,
        sequence_length=2,
        epochs=1,
        seed=4,
    )

    assert output_path.exists()
    assert not output_path.with_suffix(".report.json").exists()
    assert report["model_path"] == str(output_path)
