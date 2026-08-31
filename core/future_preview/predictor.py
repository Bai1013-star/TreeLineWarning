"""A bounded, deterministic local scenario geometry predictor.

This module is deliberately a geometry preview, not a tree-growth model.  It
uses only NumPy and a caller-provided finalized geometry sample.  The output
is always labelled as synthetic and experimental so that a UI can keep it
separate from the current survey result.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

ALLOWED_HORIZONS: tuple[int, int, int] = (1, 3, 5)
LOCAL_SCENARIO = "LOCAL_SCENARIO"
LOCAL_MODEL = "LOCAL_MODEL"
FORMAL_EXTERNAL = "FORMAL_EXTERNAL"
SYNTHETIC_FUTURE_GEOMETRY = "SYNTHETIC_FUTURE_GEOMETRY"
EXPERIMENTAL = "EXPERIMENTAL"
NOT_CURRENT_SURVEY_SCIENCE = "NOT_CURRENT_SURVEY_SCIENCE"
NOT_VALIDATED_LONG_TERM_BIOLOGICAL_FORECAST = "NOT_VALIDATED_LONG_TERM_BIOLOGICAL_FORECAST"

_MODEL = "local-scenario-geometry"
_VERSION = "1.0.0"
_PCA_EPSILON = 1.0e-12
_REQUIRED_TAGS = (
    SYNTHETIC_FUTURE_GEOMETRY,
    EXPERIMENTAL,
    NOT_CURRENT_SURVEY_SCIENCE,
    NOT_VALIDATED_LONG_TERM_BIOLOGICAL_FORECAST,
)
def _non_empty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _positive_number(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _numeric_input_array(value: Any, name: str) -> np.ndarray:
    """Copy an array-like value while rejecting booleans and object values."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric XYZ array") from exc
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain numeric values")
    if raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain numeric values")
    try:
        result = np.array(value, dtype=np.float64, order="C", copy=True, subok=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric XYZ array") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _readonly_xyz(value: Any, name: str) -> tuple[np.ndarray, bool]:
    result = _numeric_input_array(value, name)
    if result.ndim == 1:
        if result.shape != (3,):
            raise ValueError(f"{name} must have shape (3,) or (N, 3)")
        was_single = True
    elif result.ndim == 2:
        if result.shape[1] != 3 or result.shape[0] < 1:
            raise ValueError(f"{name} must have shape (3,) or (N, 3) with N >= 1")
        was_single = False
    else:
        raise ValueError(f"{name} must have shape (3,) or (N, 3)")
    return _immutable_array(result), was_single


def _readonly_vector(value: Any, name: str, length: int) -> np.ndarray:
    result = _numeric_input_array(value, name)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    return _immutable_array(result)


def _optional_metric(value: Any, name: str, *, allow_zero: bool = False) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, name)
    if allow_zero:
        if result < 0.0:
            raise ValueError(f"{name} must be greater than or equal to zero")
    elif result <= 0.0:
        raise ValueError(f"{name} must be greater than zero when supplied")
    return result


def _validate_horizon(horizon_years: Any) -> int:
    if isinstance(horizon_years, bool) or not isinstance(horizon_years, int | np.integer):
        raise ValueError(f"horizon_years must be one of {ALLOWED_HORIZONS}")
    horizon = int(horizon_years)
    if horizon not in ALLOWED_HORIZONS:
        raise ValueError(f"horizon_years must be one of {ALLOWED_HORIZONS}")
    return horizon


def _immutable_array(value: np.ndarray) -> np.ndarray:
    """Return a detached NumPy view backed by immutable bytes."""

    contiguous = np.array(value, dtype=np.float64, order="C", copy=True, subok=False)
    raw = contiguous.tobytes(order="C")
    result = np.frombuffer(raw, dtype=np.float64).reshape(contiguous.shape)
    result.setflags(write=False)
    return result


def _as_tuple_of_text(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str | bytes | bytearray):
        raise ValueError(f"{name} must be a sequence of non-empty strings")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of non-empty strings") from exc
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain only non-empty strings")
    return result


@dataclass(frozen=True, slots=True)
class FinalizedTreeGeometry:
    """Read-only geometry and canonical metrics for one finalized tree.

    ``xyz_local_m`` may be one XYZ point or an ``(N, 3)`` sample.  The sample
    supplies geometry only; dimensions are taken from the nullable canonical
    metric fields and are never inferred from the sample.  ``dbh_cm`` is in
    centimetres because that is the unit used by the survey-facing adapter.
    """

    persistent_tree_id: str
    span_id: str
    xyz_local_m: np.ndarray
    height_m: float | None = None
    crown_major_m: float | None = None
    crown_minor_m: float | None = None
    dbh_cm: float | None = None
    anchor_xyz_local_m: np.ndarray | None = None
    _xyz_was_single: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "persistent_tree_id",
            _non_empty_text(self.persistent_tree_id, "persistent_tree_id"),
        )
        object.__setattr__(self, "span_id", _non_empty_text(self.span_id, "span_id"))

        xyz, was_single = _readonly_xyz(self.xyz_local_m, "xyz_local_m")
        object.__setattr__(self, "xyz_local_m", xyz)
        object.__setattr__(self, "_xyz_was_single", was_single)

        height = _optional_metric(self.height_m, "height_m")
        major = _optional_metric(self.crown_major_m, "crown_major_m")
        minor = _optional_metric(self.crown_minor_m, "crown_minor_m")
        dbh = _optional_metric(self.dbh_cm, "dbh_cm")
        if major is not None and minor is not None and major < minor:
            raise ValueError("crown_major_m must be greater than or equal to crown_minor_m")
        object.__setattr__(self, "height_m", height)
        object.__setattr__(self, "crown_major_m", major)
        object.__setattr__(self, "crown_minor_m", minor)
        object.__setattr__(self, "dbh_cm", dbh)

        if self.anchor_xyz_local_m is None:
            anchor = None
        else:
            anchor, anchor_single = _readonly_xyz(self.anchor_xyz_local_m, "anchor_xyz_local_m")
            if not anchor_single:
                raise ValueError("anchor_xyz_local_m must have shape (3,)")
        object.__setattr__(self, "anchor_xyz_local_m", anchor)

    @property
    def current_height_m(self) -> float | None:
        """Compatibility alias for callers that name the metrics explicitly current."""

        return self.height_m

    @property
    def current_crown_major_m(self) -> float | None:
        return self.crown_major_m

    @property
    def current_crown_minor_m(self) -> float | None:
        return self.crown_minor_m

    @property
    def current_dbh_cm(self) -> float | None:
        return self.dbh_cm

    @property
    def dbh_m(self) -> float | None:
        """Metric-unit compatibility view; the stored survey value is centimetres."""

        return None if self.dbh_cm is None else self.dbh_cm / 100.0

    @property
    def known_anchor_xyz_local_m(self) -> np.ndarray | None:
        return self.anchor_xyz_local_m

    def to_dict(self) -> dict[str, Any]:
        return {
            "persistent_tree_id": self.persistent_tree_id,
            "span_id": self.span_id,
            "xyz_local_m": self.xyz_local_m.tolist(),
            "height_m": self.height_m,
            "crown_major_m": self.crown_major_m,
            "crown_minor_m": self.crown_minor_m,
            "dbh_cm": self.dbh_cm,
            "anchor_xyz_local_m": (
                None if self.anchor_xyz_local_m is None else self.anchor_xyz_local_m.tolist()
            ),
        }


@dataclass(frozen=True, slots=True)
class LocalScenarioConfig:
    """Finite, bounded illustrative changes used by ``LocalScenarioPredictor``."""

    annual_height_change_m: float = 0.20
    annual_crown_major_change_m: float = 0.15
    annual_crown_minor_change_m: float = 0.10
    canonical_height_m: float = 10.0
    canonical_crown_major_m: float = 4.0
    canonical_crown_minor_m: float = 3.0
    max_dimension_scale: float = 1.50
    growth_direction_xy: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        for name in (
            "annual_height_change_m",
            "annual_crown_major_change_m",
            "annual_crown_minor_change_m",
        ):
            value = _finite_number(getattr(self, name), name, minimum=0.0)
            if value > 10.0:
                raise ValueError(f"{name} must be no greater than 10.0")
            object.__setattr__(self, name, value)
        for name in ("canonical_height_m", "canonical_crown_major_m", "canonical_crown_minor_m"):
            object.__setattr__(self, name, _positive_number(getattr(self, name), name))
        max_scale = _finite_number(self.max_dimension_scale, "max_dimension_scale", minimum=1.0)
        if max_scale > 10.0:
            raise ValueError("max_dimension_scale must be no greater than 10.0")
        object.__setattr__(self, "max_dimension_scale", max_scale)
        if self.canonical_crown_major_m < self.canonical_crown_minor_m:
            raise ValueError(
                "canonical_crown_major_m must be greater than or equal to canonical_crown_minor_m"
            )
        direction = _readonly_vector(self.growth_direction_xy, "growth_direction_xy", 2)
        object.__setattr__(self, "growth_direction_xy", (float(direction[0]), float(direction[1])))


@dataclass(frozen=True, slots=True)
class FutureTreeGeometryPrediction:
    """Immutable synthetic overlay for one tree and one allowed horizon."""

    persistent_tree_id: str
    span_id: str
    horizon_years: int
    predicted_xyz_local_m: np.ndarray
    predicted_height_m: float | None
    predicted_crown_major_m: float | None
    predicted_crown_minor_m: float | None
    predicted_dbh_cm: float | None
    growth_direction_xy: np.ndarray
    prediction_mode: str
    model: str
    version: str
    experimental: bool
    assumptions: tuple[str, ...]
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "persistent_tree_id",
            _non_empty_text(self.persistent_tree_id, "persistent_tree_id"),
        )
        object.__setattr__(self, "span_id", _non_empty_text(self.span_id, "span_id"))
        object.__setattr__(self, "horizon_years", _validate_horizon(self.horizon_years))
        xyz, _ = _readonly_xyz(self.predicted_xyz_local_m, "predicted_xyz_local_m")
        object.__setattr__(self, "predicted_xyz_local_m", xyz)
        object.__setattr__(
            self,
            "predicted_height_m",
            _optional_metric(self.predicted_height_m, "predicted_height_m"),
        )
        major = _optional_metric(self.predicted_crown_major_m, "predicted_crown_major_m")
        minor = _optional_metric(self.predicted_crown_minor_m, "predicted_crown_minor_m")
        if major is not None and minor is not None and major < minor:
            raise ValueError(
                "predicted_crown_major_m must be greater than or equal to predicted_crown_minor_m"
            )
        object.__setattr__(self, "predicted_crown_major_m", major)
        object.__setattr__(self, "predicted_crown_minor_m", minor)
        object.__setattr__(
            self,
            "predicted_dbh_cm",
            _optional_metric(self.predicted_dbh_cm, "predicted_dbh_cm"),
        )
        object.__setattr__(
            self,
            "growth_direction_xy",
            _readonly_vector(self.growth_direction_xy, "growth_direction_xy", 2),
        )
        object.__setattr__(
            self,
            "prediction_mode",
            _non_empty_text(self.prediction_mode, "prediction_mode"),
        )
        object.__setattr__(self, "model", _non_empty_text(self.model, "model"))
        object.__setattr__(self, "version", _non_empty_text(self.version, "version"))
        if type(self.experimental) is not bool or not self.experimental:
            raise ValueError("experimental must be True for a local scenario prediction")
        assumptions = _as_tuple_of_text(self.assumptions, "assumptions")
        tags = _as_tuple_of_text(self.tags, "tags")
        if not set(_REQUIRED_TAGS).issubset(tags):
            raise ValueError("tags must include all required synthetic/experimental safety labels")
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "tags", tags)

    @property
    def parent_persistent_tree_id(self) -> str:
        return self.persistent_tree_id

    @property
    def forecast_horizon_years(self) -> int:
        """Adapter-facing name for the immutable requested horizon."""

        return self.horizon_years

    @property
    def synthetic_xyz_local_m(self) -> np.ndarray:
        """Explicit safety-name alias for the predicted geometry array."""

        return self.predicted_xyz_local_m

    @property
    def xyz_local_m(self) -> np.ndarray:
        return self.predicted_xyz_local_m

    @property
    def predicted_dbh_m(self) -> float | None:
        return None if self.predicted_dbh_cm is None else self.predicted_dbh_cm / 100.0

    @property
    def model_id(self) -> str:
        return self.model

    @property
    def model_name(self) -> str:
        """Adapter-facing name for the deterministic local scenario implementation."""

        return self.model

    @property
    def model_version(self) -> str:
        return self.version

    def to_dict(self) -> dict[str, Any]:
        return {
            "persistent_tree_id": self.persistent_tree_id,
            "span_id": self.span_id,
            "horizon_years": self.horizon_years,
            "predicted_xyz_local_m": self.predicted_xyz_local_m.tolist(),
            "predicted_height_m": self.predicted_height_m,
            "predicted_crown_major_m": self.predicted_crown_major_m,
            "predicted_crown_minor_m": self.predicted_crown_minor_m,
            "predicted_dbh_cm": self.predicted_dbh_cm,
            "growth_direction_xy": self.growth_direction_xy.tolist(),
            "prediction_mode": self.prediction_mode,
            "model": self.model,
            "version": self.version,
            "experimental": self.experimental,
            "assumptions": list(self.assumptions),
            "tags": list(self.tags),
        }


@runtime_checkable
class FutureTreePredictor(Protocol):
    """Extensibility boundary for predictors that consume finalized geometry."""

    def predict(
        self, tree: FinalizedTreeGeometry, horizon_years: int
    ) -> FutureTreeGeometryPrediction:
        """Return one synthetic prediction for an allowed horizon."""


class LocalScenarioPredictor:
    """Predict bounded local geometry scenarios with CPU-only NumPy operations."""

    def __init__(self, config: LocalScenarioConfig | None = None) -> None:
        if config is not None and not isinstance(config, LocalScenarioConfig):
            raise TypeError("config must be a LocalScenarioConfig or None")
        self.config = LocalScenarioConfig() if config is None else config

    def predict(
        self, tree: FinalizedTreeGeometry, horizon_years: int
    ) -> FutureTreeGeometryPrediction:
        if not isinstance(tree, FinalizedTreeGeometry):
            raise TypeError("tree must be a FinalizedTreeGeometry")
        horizon = _validate_horizon(horizon_years)
        config = self.config
        points = tree.xyz_local_m.reshape(-1, 3)
        anchor = (
            tree.anchor_xyz_local_m
            if tree.anchor_xyz_local_m is not None
            else np.array(
                [np.mean(points[:, 0]), np.mean(points[:, 1]), np.min(points[:, 2])],
                dtype=np.float64,
            )
        )
        center_xy = np.array(anchor[:2], dtype=np.float64, copy=True)
        z_base = float(anchor[2])
        major_axis, minor_axis, flat_sample = _pca_axes(points[:, :2], center_xy)

        current_height = (
            tree.height_m if tree.height_m is not None else config.canonical_height_m
        )
        current_major = (
            tree.crown_major_m if tree.crown_major_m is not None else config.canonical_crown_major_m
        )
        current_minor = (
            tree.crown_minor_m if tree.crown_minor_m is not None else config.canonical_crown_minor_m
        )
        target_height = _bounded_target(
            current_height, config.annual_height_change_m, horizon, config.max_dimension_scale
        )
        target_major = _bounded_target(
            current_major, config.annual_crown_major_change_m, horizon, config.max_dimension_scale
        )
        target_minor = _bounded_target(
            current_minor, config.annual_crown_minor_change_m, horizon, config.max_dimension_scale
        )

        relative_xy = points[:, :2] - center_xy
        major_coordinate = relative_xy @ major_axis
        minor_coordinate = relative_xy @ minor_axis
        future_xy = center_xy + (
            major_coordinate[:, None] * major_axis[None, :] * (target_major / current_major)
            + minor_coordinate[:, None] * minor_axis[None, :] * (target_minor / current_minor)
        )
        future_xy += np.asarray(config.growth_direction_xy, dtype=np.float64) * horizon
        future_z = z_base + (points[:, 2] - z_base) * (target_height / current_height)
        future = np.column_stack((future_xy, future_z))
        if tree._xyz_was_single:
            future = future[0]

        assumptions = [
            "Canonical dimensions are used only as an explicit geometry-scaling assumption "
            "when a metric is missing.",
            "Canonical metrics and the supplied geometry sample are separate; metrics are "
            "never inferred from the sample.",
            "Height and crown changes are finite illustrative annual increments bounded by "
            "max_dimension_scale.",
            "The XY transformation uses deterministic PCA axes of the supplied local sample.",
            "The Z transformation is tree-centered at the supplied anchor or the sample Z minimum.",
            "DBH is carried through unchanged when supplied and remains None when missing; "
            "no DBH growth is modelled.",
            "This output is a synthetic preview overlay and must remain separate from current "
            "survey geometry.",
            SYNTHETIC_FUTURE_GEOMETRY,
            EXPERIMENTAL,
            NOT_CURRENT_SURVEY_SCIENCE,
            NOT_VALIDATED_LONG_TERM_BIOLOGICAL_FORECAST,
        ]
        if tree.height_m is None or tree.crown_major_m is None or tree.crown_minor_m is None:
            assumptions.append(
                "One or more missing dimensions use the configured canonical geometry "
                "assumption."
            )
        if flat_sample:
            assumptions.append(
                "The sample has insufficient planar spread; canonical XY axes were used for "
                "this geometry preview."
            )
        if tree.dbh_cm is None:
            assumptions.append(
                "DBH is unknown in the finalized input and is intentionally left unknown."
            )

        return FutureTreeGeometryPrediction(
            persistent_tree_id=tree.persistent_tree_id,
            span_id=tree.span_id,
            horizon_years=horizon,
            predicted_xyz_local_m=future,
            predicted_height_m=(None if tree.height_m is None else target_height),
            predicted_crown_major_m=(None if tree.crown_major_m is None else target_major),
            predicted_crown_minor_m=(None if tree.crown_minor_m is None else target_minor),
            predicted_dbh_cm=tree.dbh_cm,
            growth_direction_xy=np.asarray(config.growth_direction_xy, dtype=np.float64),
            prediction_mode=LOCAL_SCENARIO,
            model=_MODEL,
            version=_VERSION,
            experimental=True,
            assumptions=tuple(assumptions),
            tags=_REQUIRED_TAGS,
        )


def _bounded_target(current: float, annual_change: float, horizon: int, max_scale: float) -> float:
    return min(current + annual_change * horizon, current * max_scale)


def _pca_axes(xy: np.ndarray, center_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    centered = np.asarray(xy, dtype=np.float64) - center_xy
    if centered.shape[0] < 2:
        return (
            np.array([1.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            True,
        )
    spread = np.ptp(centered, axis=0)
    if float(np.max(spread)) <= _PCA_EPSILON:
        return (
            np.array([1.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            True,
        )
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = float(eigenvalues[-1])
    second = float(eigenvalues[-2])
    tolerance = max(abs(largest), 1.0) * 1.0e-12
    if largest <= _PCA_EPSILON or largest - second <= tolerance:
        major_axis = np.array([1.0, 0.0], dtype=np.float64)
        return major_axis, np.array([0.0, 1.0], dtype=np.float64), True
    major_axis = np.asarray(eigenvectors[:, -1], dtype=np.float64)
    norm = float(np.linalg.norm(major_axis))
    if not math.isfinite(norm) or norm <= _PCA_EPSILON:
        return (
            np.array([1.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            True,
        )
    major_axis = major_axis / norm
    if major_axis[0] < -_PCA_EPSILON or (
        abs(major_axis[0]) <= _PCA_EPSILON and major_axis[1] < 0.0
    ):
        major_axis = -major_axis
    minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=np.float64)
    return major_axis, minor_axis, False
