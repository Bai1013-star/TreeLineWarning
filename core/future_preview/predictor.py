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
from datetime import date, timedelta
from functools import lru_cache
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

SPECIES_TWIN_MORPHOLOGY = {
    "松树": (0.20, 0.62, 0.65, "针叶树顶端优势型"),
    "杉树": (0.22, 0.65, 0.68, "针叶树顶端优势型"),
    "柏树": (0.18, 0.60, 0.62, "常绿针叶紧凑冠型"),
    "法国梧桐": (0.42, 0.86, 0.78, "速生阔叶扩冠型"),
    "杨树": (0.34, 0.78, 0.72, "速生阔叶高生长型"),
    "柳树": (0.46, 0.90, 0.70, "速生阔叶宽冠型"),
    "榆树": (0.34, 0.82, 0.80, "环孔材春季前置型"),
    "国槐": (0.36, 0.84, 0.82, "环孔材扩冠型"),
    "白蜡": (0.33, 0.80, 0.78, "环孔材春季前置型"),
    "樟树": (0.38, 0.86, 0.76, "常绿阔叶延长生长期型"),
    "其他": (0.32, 0.75, 0.80, "通用阔叶保守型"),
}


def species_twin_morphology(species: str) -> dict[str, float | str]:
    crown_ratio, minor_ratio, dbh_ratio, label = SPECIES_TWIN_MORPHOLOGY.get(
        str(species or "其他"), SPECIES_TWIN_MORPHOLOGY["其他"]
    )
    return {
        "crown_width_per_height": crown_ratio,
        "minor_to_major_increment": minor_ratio,
        "dbh_cm_per_height_m": dbh_ratio,
        "architecture": label,
    }
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


def _smoothstep(value: np.ndarray | float, edge0: float, edge1: float) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    scaled = np.clip((raw - edge0) / (edge1 - edge0), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


@lru_cache(maxsize=512)
def seasonal_growth_years(start_iso: str, horizon_days: int, species: str = "其他") -> float:
    """Integrate a seasonal annual growth fraction at daily resolution."""

    if not 1 <= int(horizon_days) <= 3660:
        raise ValueError("horizon_days must be in 1..3660")
    from algorithms.prediction.species_growth import phenology_weights_for_species

    current = date.fromisoformat(start_iso)
    weights = phenology_weights_for_species(species)
    total = 0.0
    remaining = int(horizon_days)
    day = current
    # Integrate exact calendar-day fractions one month at a time. This is
    # numerically identical to a per-day loop but keeps the five-year slider
    # responsive when trajectories request hundreds of horizons.
    while remaining:
        next_month = date(
            day.year + (day.month == 12),
            1 if day.month == 12 else day.month + 1,
            1,
        )
        month_start = date(day.year, day.month, 1)
        days_in_month = (next_month - month_start).days
        available = (next_month - day).days
        take = min(remaining, available)
        total += weights[day.month - 1] * take / days_in_month
        remaining -= take
        day += timedelta(days=take)
    return total


def _weather_multiplier(weather: dict[str, Any]) -> float:
    """Bounded scenario modifier, not a fitted biological causal model."""

    temperature = float(weather.get("temperature", 20.0))
    sunlight = float(weather.get("sunlight_hours", 6.0))
    rainfall = float(weather.get("rainfall", 0.0))
    humidity = float(weather.get("humidity", 60.0))
    wind = float(weather.get("wind_speed", 0.0))
    temperature_factor = max(0.25, 1.0 - abs(temperature - 23.0) / 45.0)
    light_factor = min(1.12, max(0.75, 0.78 + sunlight / 30.0))
    moisture_factor = min(1.12, max(0.72, 0.78 + rainfall / 250.0 + humidity / 500.0))
    wind_factor = min(1.0, max(0.72, 1.0 - max(0.0, wind - 8.0) / 55.0))
    return min(1.25, max(0.35, temperature_factor * light_factor * moisture_factor * wind_factor))


def seasonal_growth_metrics(
    *,
    horizon_days: int,
    start_date: str,
    species: str,
    age: float,
    weather: dict[str, Any],
) -> dict[str, float | str]:
    """Return the same phenology-driven increments used by the 3-D twin.

    Keeping the scalar trajectory separate lets the UI draw a five-year
    trend without repeatedly transforming every display point.  Spring and
    summer receive most of the annual allocation; autumn tapers and winter is
    close to dormant through :data:`_MONTH_GROWTH_WEIGHTS`.
    """

    if not math.isfinite(float(age)) or float(age) < 0:
        raise ValueError("age must be a non-negative finite number")
    from algorithms.prediction.species import annual_height_growth

    growth_reference = annual_height_growth(str(species or "其他"))
    annual_height = float(growth_reference.get("reference", 0.7))
    age_factor = 1.0 if age <= 20 else (0.88 if age <= 50 else (0.72 if age <= 100 else 0.55))
    seasonal_years = seasonal_growth_years(
        str(start_date)[:10], int(horizon_days), str(species or "其他")
    )
    weather_factor = _weather_multiplier(weather)
    height_increment = max(0.0, annual_height * seasonal_years * age_factor * weather_factor)
    morphology = species_twin_morphology(species)
    return {
        "height_increment_m": height_increment,
        "crown_width_increment_m": height_increment
        * float(morphology["crown_width_per_height"]),
        "dbh_increment_cm": height_increment * float(morphology["dbh_cm_per_height_m"]),
        "seasonal_growth_years": seasonal_years,
        "weather_multiplier": weather_factor,
        "species": str(species or "其他"),
        "age": float(age),
        "architecture": str(morphology["architecture"]),
        "minor_to_major_increment": float(morphology["minor_to_major_increment"]),
    }


def seasonal_nonuniform_prediction(
    tree: FinalizedTreeGeometry,
    *,
    horizon_days: int,
    start_date: str,
    species: str,
    age: float,
    weather: dict[str, Any],
) -> dict[str, Any]:
    """Create a seasonal, non-rigid tree point-cloud twin overlay.

    The base is fixed.  Height increment is concentrated toward the top,
    crown expansion is limited to upper-canopy points along separate PCA axes,
    and DBH increment affects lower-stem radial coordinates.  Consequently the
    transformation is never a whole-tree uniform scale.
    """

    if not isinstance(tree, FinalizedTreeGeometry):
        raise TypeError("tree must be a FinalizedTreeGeometry")
    if not math.isfinite(float(age)) or float(age) < 0:
        raise ValueError("age must be a non-negative finite number")
    points = tree.xyz_local_m.reshape(-1, 3)
    anchor = (
        tree.anchor_xyz_local_m
        if tree.anchor_xyz_local_m is not None
        else np.asarray([np.mean(points[:, 0]), np.mean(points[:, 1]), np.min(points[:, 2])])
    )
    center_xy = np.asarray(anchor[:2], dtype=np.float64)
    base_z = float(anchor[2])
    current_height = tree.height_m or max(float(np.ptp(points[:, 2])), 0.1)
    major_axis, minor_axis, flat_sample = _pca_axes(points[:, :2], center_xy)
    metrics = seasonal_growth_metrics(
        horizon_days=horizon_days,
        start_date=start_date,
        species=species,
        age=age,
        weather=weather,
    )
    seasonal_years = float(metrics["seasonal_growth_years"])
    weather_factor = float(metrics["weather_multiplier"])
    height_increment = float(metrics["height_increment_m"])
    crown_width_increment = float(metrics["crown_width_increment_m"])
    dbh_increment_cm = float(metrics["dbh_increment_cm"])

    relative_z = points[:, 2] - base_z
    height_fraction = np.clip(relative_z / max(current_height, 1.0e-6), 0.0, 1.0)
    top_weight = _smoothstep(height_fraction, 0.12, 1.0)
    crown_weight = _smoothstep(height_fraction, 0.38, 0.82)
    stem_weight = 1.0 - _smoothstep(height_fraction, 0.08, 0.48)

    relative_xy = points[:, :2] - center_xy
    major_coordinate = relative_xy @ major_axis
    minor_coordinate = relative_xy @ minor_axis
    # Model several branch sectors instead of inflating the whole crown.  The
    # deterministic phase keeps the same tree stable while the horizon changes;
    # low-activity sectors remain almost fixed and active tips receive most of
    # the current period's increment.  This is a visual growth envelope, not a
    # claim that every source point is a botanical branch measurement.
    angle = np.arctan2(minor_coordinate, major_coordinate)
    species_phase = (sum(ord(char) for char in str(species or "其他")) % 360) * math.pi / 180
    time_phase = 2 * math.pi * min(max(seasonal_years, 0.0), 5.0) / 5.0
    branch_signal = np.maximum(
        np.cos(3 * angle + species_phase + time_phase),
        0.45 * np.cos(5 * angle - species_phase * 0.5 + time_phase * 0.7),
    )
    branch_activity = np.where(branch_signal > 0.18, 0.35 + 0.65 * np.clip(branch_signal, 0, 1), 0.06)
    crown_weight = crown_weight * branch_activity
    current_major = tree.crown_major_m or max(float(np.ptp(major_coordinate)), 0.5)
    current_minor = tree.crown_minor_m or max(float(np.ptp(minor_coordinate)), 0.5)
    crown_major_factor = 1.0 + crown_width_increment / max(current_major, 0.25)
    minor_ratio = float(metrics["minor_to_major_increment"])
    crown_minor_factor = 1.0 + crown_width_increment * minor_ratio / max(current_minor, 0.25)
    trunk_factor = 1.0
    if tree.dbh_cm is not None and tree.dbh_cm > 0:
        trunk_factor += dbh_increment_cm / tree.dbh_cm
    major_factor = 1.0 + crown_weight * (crown_major_factor - 1.0) + stem_weight * (trunk_factor - 1.0)
    minor_factor = 1.0 + crown_weight * (crown_minor_factor - 1.0) + stem_weight * (trunk_factor - 1.0)
    future_xy = center_xy + major_coordinate[:, None] * major_axis * major_factor[:, None] + minor_coordinate[:, None] * minor_axis * minor_factor[:, None]
    future_z = points[:, 2] + height_increment * top_weight
    future = np.column_stack((future_xy, future_z))
    if tree._xyz_was_single:
        future = future[0]

    return {
        "xyz_local_m": future,
        "predicted_height_m": None if tree.height_m is None else tree.height_m + height_increment,
        "predicted_crown_major_m": None if tree.crown_major_m is None else tree.crown_major_m + crown_width_increment,
        "predicted_crown_minor_m": None if tree.crown_minor_m is None else tree.crown_minor_m + crown_width_increment * minor_ratio,
        "predicted_dbh_cm": None if tree.dbh_cm is None else tree.dbh_cm + dbh_increment_cm,
        "height_increment_m": height_increment,
        "crown_width_increment_m": crown_width_increment,
        "dbh_increment_cm": dbh_increment_cm,
        "seasonal_growth_years": seasonal_years,
        "weather_multiplier": weather_factor,
        "species": str(species or "其他"),
        "species_architecture": metrics["architecture"],
        "age": float(age),
        "start_date": str(start_date)[:10],
        "target_date": (date.fromisoformat(str(start_date)[:10]) + timedelta(days=int(horizon_days))).isoformat(),
        "flat_sample_axes_fallback": flat_sample,
        "transformation": "NON_RIGID_STEM_CROWN_PHENOLOGY",
        "branch_growth_mode": "SELECTIVE_ANGULAR_BRANCH_SECTORS",
        "assumptions": [
            "树干基部中心与地面高程固定；下部可随胸径增长发生径向增粗，顶梢增高按相对高度非线性分配。",
            "树冠仅在上部点集扩张，长短轴分别变化；树干胸径在下部径向变化。",
            "每个时间步只激活部分树冠方向的枝条扇区，其他枝条保持近静止，避免整冠同步等比例放大。",
            "逐日积分春季萌发、夏季旺长、秋季缓生、冬季近休眠的月度生长权重。",
            f"树种采用“{metrics['architecture']}”形态参数；人工选择的树种优先于自动识别建议。",
            "天气为用户输入的情景修正并设有上下限，不代表经长期样地校准的因果模型。",
            "预测孪生点云与当前实测点云分层显示，不替代现场复测。",
        ],
    }
