"""Prediction pipeline tests."""

import json
from pathlib import Path

import pytest
from algorithms.prediction.i1_pipeline import I1PredictionPipeline
from algorithms.prediction.i1_prediction import I1TreeData
from algorithms.prediction.pipeline import PredictionPipeline
from algorithms.prediction.storage import PredictionStorage


def test_pipeline_high_risk():

    trees = [

        I1TreeData(
            tree_id=1,
            species="松树",
            height=25,
            age=40,
            diameter=50,
            distance_to_wire=3,
            wind_speed=20,
            rainfall=60,
            humidity=80,
            slope=30
        )

    ]


    result = PredictionPipeline().run(trees)


    report = result["report"]


    assert report["total_tree"] == 1

    assert report["high_risk_count"] == 1

    assert 1 in report["high_risk_tree_ids"]



def test_pipeline_low_risk():

    trees = [

        I1TreeData(
            tree_id=2,
            species="杉树",
            height=5,
            age=5,
            diameter=10,
            distance_to_wire=30,
            wind_speed=2,
            rainfall=0,
            humidity=40,
            slope=5
        )

    ]


    result = PredictionPipeline().run(trees)


    report = result["report"]


    assert report["total_tree"] == 1

    assert report["low_risk_count"] == 1


def test_pipeline_does_not_write_results_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tree = I1TreeData(
        tree_id=3,
        species="松树",
        height=10,
        age=10,
        diameter=20,
        distance_to_wire=20,
    )

    PredictionPipeline().run([tree])

    assert not (tmp_path / "prediction_results.json").exists()


def test_prediction_storage_requires_explicit_nonempty_path():
    with pytest.raises(TypeError):
        PredictionStorage()
    with pytest.raises(ValueError, match="explicit output path"):
        PredictionStorage(None)
    with pytest.raises(ValueError, match="explicit output path"):
        PredictionStorage("")


def test_pipeline_persists_only_to_explicit_storage_path(tmp_path: Path, monkeypatch):
    working_directory = tmp_path / "working"
    output_directory = tmp_path / "output"
    working_directory.mkdir()
    output_directory.mkdir()
    monkeypatch.chdir(working_directory)
    output_path = output_directory / "results.json"
    tree = I1TreeData(
        tree_id=4,
        species="松树",
        height=10,
        age=10,
        diameter=20,
        distance_to_wire=20,
    )

    PredictionPipeline(
        persist=True, storage=PredictionStorage(output_path)
    ).run([tree])

    assert output_path.exists()
    assert not (working_directory / "prediction_results.json").exists()


def test_i1_run_remains_backward_compatible(tmp_path: Path):
    (tmp_path / "trees.json").write_text(
        json.dumps(
            [
                {
                    "tree_id": 5,
                    "species": "松树",
                    "height": 10,
                    "age": 10,
                    "diameter": 20,
                    "distance_to_wire": 20,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = I1PredictionPipeline().run(tmp_path)

    assert set(result) == {"results", "report"}
    assert result["report"]["total_tree"] == 1


def test_i1_pipeline_exposes_legacy_pipeline_alias():
    pipeline = I1PredictionPipeline()

    assert pipeline.predict_pipeline is pipeline.pipeline
