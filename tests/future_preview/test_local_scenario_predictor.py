from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from core.future_preview import (
    ALLOWED_HORIZONS,
    EXPERIMENTAL,
    FORMAL_EXTERNAL,
    LOCAL_MODEL,
    LOCAL_SCENARIO,
    NOT_CURRENT_SURVEY_SCIENCE,
    NOT_VALIDATED_LONG_TERM_BIOLOGICAL_FORECAST,
    SYNTHETIC_FUTURE_GEOMETRY,
    FinalizedTreeGeometry,
    FutureTreeGeometryPrediction,
    FutureTreePredictor,
    LocalScenarioConfig,
    LocalScenarioPredictor,
)


def _points() -> np.ndarray:
    return np.array(
        [
            [-2.0, -0.75, 0.0],
            [-1.0, 0.60, 2.0],
            [0.0, -0.40, 5.0],
            [1.0, 0.80, 7.0],
            [2.0, -0.25, 10.0],
        ],
        dtype=np.float64,
    )


def _tree(points: np.ndarray | None = None, **metrics: float | None) -> FinalizedTreeGeometry:
    return FinalizedTreeGeometry(
        persistent_tree_id="tree-17",
        span_id="span-A",
        xyz_local_m=_points() if points is None else points,
        height_m=metrics.get("height_m", 10.0),
        crown_major_m=metrics.get("crown_major_m", 4.0),
        crown_minor_m=metrics.get("crown_minor_m", 2.0),
        dbh_cm=metrics.get("dbh_cm", 42.0),
        anchor_xyz_local_m=metrics.get("anchor_xyz_local_m", (0.0, 0.0, 0.0)),
    )


def test_public_predictor_protocol_and_all_allowed_horizons() -> None:
    predictor = LocalScenarioPredictor()
    assert isinstance(predictor, FutureTreePredictor)
    assert ALLOWED_HORIZONS == (1, 3, 5)
    for horizon in ALLOWED_HORIZONS:
        result = predictor.predict(_tree(), horizon)
        assert result.persistent_tree_id == "tree-17"
        assert result.span_id == "span-A"
        assert result.horizon_years == horizon
        assert result.forecast_horizon_years == horizon
        assert result.prediction_mode == LOCAL_SCENARIO
        assert result.experimental is True
        assert {
            SYNTHETIC_FUTURE_GEOMETRY,
            EXPERIMENTAL,
            NOT_CURRENT_SURVEY_SCIENCE,
            NOT_VALIDATED_LONG_TERM_BIOLOGICAL_FORECAST,
        }.issubset(result.tags)
        assert "original_index" not in result.to_dict()
        assert result.predicted_height_m is not None
        assert result.predicted_height_m > 10.0
        assert result.predicted_dbh_cm == 42.0


def test_anisotropic_tree_centered_scaling_preserves_order() -> None:
    points = np.array(
        [
            [-2.0, -1.0, 0.0],
            [2.0, -1.0, 0.0],
            [-2.0, 1.0, 4.0],
            [2.0, 1.0, 4.0],
        ],
        dtype=np.float64,
    )
    tree = _tree(points, height_m=4.0, crown_major_m=4.0, crown_minor_m=2.0)
    result = LocalScenarioPredictor().predict(tree, 1)

    # PCA picks X as the major axis for this sample and the output keeps the
    # caller's point order.  Horizontal anisotropy and vertical scaling are
    # independently inspectable from the unchanged tree-centered anchor.
    np.testing.assert_allclose(result.xyz_local_m[:, 0], points[:, 0] * (4.15 / 4.0))
    np.testing.assert_allclose(result.xyz_local_m[:, 1], points[:, 1] * (2.10 / 2.0))
    np.testing.assert_allclose(result.xyz_local_m[:, 2], points[:, 2] * (4.20 / 4.0))
    assert result.predicted_crown_major_m == pytest.approx(4.15)
    assert result.predicted_crown_minor_m == pytest.approx(2.10)


def test_anchor_makes_translation_invariance_explicit() -> None:
    points = _points()
    offset = np.array([100.0, -25.0, 8.0])
    base = _tree(points)
    translated = FinalizedTreeGeometry(
        persistent_tree_id=base.persistent_tree_id,
        span_id=base.span_id,
        xyz_local_m=points + offset,
        height_m=base.height_m,
        crown_major_m=base.crown_major_m,
        crown_minor_m=base.crown_minor_m,
        dbh_cm=base.dbh_cm,
        anchor_xyz_local_m=np.asarray(base.anchor_xyz_local_m) + offset,
    )
    predictor = LocalScenarioPredictor()
    base_result = predictor.predict(base, 3)
    translated_result = predictor.predict(translated, 3)
    np.testing.assert_allclose(
        translated_result.xyz_local_m,
        base_result.xyz_local_m + offset,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_arrays_are_detached_owned_and_read_only() -> None:
    source = _points()
    tree = _tree(source)
    source[0, 0] = 999.0
    assert tree.xyz_local_m[0, 0] == -2.0
    assert not tree.xyz_local_m.flags.writeable

    result = LocalScenarioPredictor().predict(tree, 1)
    assert not result.xyz_local_m.flags.writeable
    assert not result.growth_direction_xy.flags.writeable
    assert not np.shares_memory(result.xyz_local_m, tree.xyz_local_m)
    with pytest.raises(ValueError):
        result.xyz_local_m[0, 0] = 0.0
    with pytest.raises(ValueError):
        result.growth_direction_xy[0] = 1.0
    with pytest.raises(ValueError):
        result.xyz_local_m.setflags(write=True)
    with pytest.raises(ValueError):
        result.growth_direction_xy.setflags(write=True)


def test_missing_metrics_are_honest_and_dbh_is_not_fabricated() -> None:
    tree = _tree(height_m=None, crown_major_m=None, crown_minor_m=None, dbh_cm=None)
    result = LocalScenarioPredictor().predict(tree, 5)
    assert result.predicted_height_m is None
    assert result.predicted_crown_major_m is None
    assert result.predicted_crown_minor_m is None
    assert result.predicted_dbh_cm is None
    assert any("canonical" in assumption.lower() for assumption in result.assumptions)
    assert any("DBH" in assumption for assumption in result.assumptions)
    assert np.all(np.isfinite(result.xyz_local_m))


def test_flat_sample_is_accepted_with_explicit_axis_assumption() -> None:
    points = np.array(
        [[3.0, 4.0, 0.0], [3.0, 4.0, 2.0], [3.0, 4.0, 4.0]], dtype=np.float64
    )
    result = LocalScenarioPredictor().predict(_tree(points), 1)
    assert np.all(np.isfinite(result.xyz_local_m))
    assert any("insufficient planar spread" in assumption for assumption in result.assumptions)


def test_deterministic_for_repeated_calls_and_permuted_samples() -> None:
    tree = _tree()
    predictor = LocalScenarioPredictor()
    first = predictor.predict(tree, 5)
    second = predictor.predict(tree, 5)
    np.testing.assert_array_equal(first.xyz_local_m, second.xyz_local_m)
    assert first.to_dict() == second.to_dict()

    order = np.array([4, 1, 3, 0, 2])
    permuted = _tree(_points()[order])
    permuted_result = predictor.predict(permuted, 5)
    restored = np.empty_like(permuted_result.xyz_local_m)
    restored[order] = permuted_result.xyz_local_m
    np.testing.assert_allclose(restored, first.xyz_local_m, rtol=0.0, atol=1.0e-12)


def test_config_changes_are_finite_and_bounded() -> None:
    config = LocalScenarioConfig(
        annual_height_change_m=10.0,
        annual_crown_major_change_m=10.0,
        annual_crown_minor_change_m=10.0,
        max_dimension_scale=1.1,
        growth_direction_xy=(0.5, -0.25),
    )
    result = LocalScenarioPredictor(config).predict(_tree(), 5)
    assert result.predicted_height_m == pytest.approx(11.0)
    assert result.predicted_crown_major_m == pytest.approx(4.4)
    assert result.predicted_crown_minor_m == pytest.approx(2.2)
    expected_shift = np.array([2.5, -1.25])
    without_shift = LocalScenarioPredictor(
        LocalScenarioConfig(
            annual_height_change_m=10.0,
            annual_crown_major_change_m=10.0,
            annual_crown_minor_change_m=10.0,
            max_dimension_scale=1.1,
        )
    ).predict(_tree(), 5)
    np.testing.assert_allclose(
        result.xyz_local_m[:, :2] - expected_shift,
        without_shift.xyz_local_m[:, :2],
    )


@pytest.mark.parametrize("horizon", [0, 2, 4, 6, -1, True, 1.0, "1"])
def test_invalid_horizon_is_rejected(horizon: object) -> None:
    with pytest.raises(ValueError, match="horizon_years"):
        LocalScenarioPredictor().predict(_tree(), horizon)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"persistent_tree_id": ""},
        {"span_id": "   "},
        {"xyz_local_m": [[0.0, 0.0, np.nan]]},
        {"xyz_local_m": np.array([[1.0 + 2.0j, 0.0, 0.0]])},
        {"xyz_local_m": [0.0, 0.0]},
        {"xyz_local_m": []},
        {"height_m": 0.0},
        {"crown_major_m": 1.0, "crown_minor_m": 2.0},
        {"dbh_cm": -1.0},
        {"anchor_xyz_local_m": [[0.0, 0.0, 0.0]]},
    ],
)
def test_invalid_finalized_geometry_is_rejected(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "persistent_tree_id": "tree-17",
        "span_id": "span-A",
        "xyz_local_m": _points(),
        "height_m": 10.0,
        "crown_major_m": 4.0,
        "crown_minor_m": 2.0,
        "dbh_cm": 42.0,
        "anchor_xyz_local_m": (0.0, 0.0, 0.0),
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        FinalizedTreeGeometry(**values)  # type: ignore[arg-type]


def test_input_type_and_config_failures_are_explicit() -> None:
    with pytest.raises(TypeError, match="FinalizedTreeGeometry"):
        LocalScenarioPredictor().predict(object(), 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="LocalScenarioConfig"):
        LocalScenarioPredictor(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        LocalScenarioConfig(annual_height_change_m=np.nan)


def test_output_contract_remains_frozen_and_validates_safety_labels() -> None:
    result = LocalScenarioPredictor().predict(_tree(), 1)
    assert dataclasses.is_dataclass(result)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.prediction_mode = "OTHER"  # type: ignore[misc]
    with pytest.raises(ValueError, match="required"):
        FutureTreeGeometryPrediction(
            persistent_tree_id="tree",
            span_id="span",
            horizon_years=1,
            predicted_xyz_local_m=np.zeros((1, 3)),
            predicted_height_m=None,
            predicted_crown_major_m=None,
            predicted_crown_minor_m=None,
            predicted_dbh_cm=None,
            growth_direction_xy=np.zeros(2),
            prediction_mode=LOCAL_SCENARIO,
            model="test",
            version="1",
            experimental=True,
            assumptions=("safe",),
            tags=(SYNTHETIC_FUTURE_GEOMETRY,),
        )


def test_output_contract_allows_declared_extension_modes() -> None:
    tags = (
        SYNTHETIC_FUTURE_GEOMETRY,
        EXPERIMENTAL,
        NOT_CURRENT_SURVEY_SCIENCE,
        NOT_VALIDATED_LONG_TERM_BIOLOGICAL_FORECAST,
    )
    for mode in (LOCAL_MODEL, FORMAL_EXTERNAL):
        result = FutureTreeGeometryPrediction(
            persistent_tree_id="tree",
            span_id="span",
            horizon_years=1,
            predicted_xyz_local_m=np.zeros((1, 3)),
            predicted_height_m=None,
            predicted_crown_major_m=None,
            predicted_crown_minor_m=None,
            predicted_dbh_cm=None,
            growth_direction_xy=np.zeros(2),
            prediction_mode=mode,
            model="external-contract-test",
            version="1",
            experimental=True,
            assumptions=("extension mode is explicitly labelled",),
            tags=tags,
        )
        assert result.prediction_mode == mode
