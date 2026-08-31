"""Explicitly synthetic, local geometry previews for finalized tree records.

The public API is intentionally small.  It accepts an already-finalized tree
geometry and returns an immutable, clearly labelled overlay that callers may
render separately from current survey geometry.
"""

from .predictor import (
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

__all__ = [
    "ALLOWED_HORIZONS",
    "EXPERIMENTAL",
    "FORMAL_EXTERNAL",
    "LOCAL_SCENARIO",
    "LOCAL_MODEL",
    "NOT_CURRENT_SURVEY_SCIENCE",
    "NOT_VALIDATED_LONG_TERM_BIOLOGICAL_FORECAST",
    "SYNTHETIC_FUTURE_GEOMETRY",
    "FinalizedTreeGeometry",
    "FutureTreeGeometryPrediction",
    "FutureTreePredictor",
    "LocalScenarioConfig",
    "LocalScenarioPredictor",
]
