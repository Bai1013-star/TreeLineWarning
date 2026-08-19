"""Breast-height diameter (DBH) evidence helpers.

The actual DBH computation happens inside ``metrics.py`` from the single
shared full-resolution HAG pass; this module owns the frozen quality taxonomy
and the small deterministic classifier so the rule is easy to review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from algorithms.manual_span.config import ManualSpanConfig

DBH_MEASURED = "MEASURED"
DBH_ESTIMATED = "ESTIMATED"
DBH_REVIEW = "REVIEW"
DBH_UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class DbhEvidence:
    """Stem cross-section evidence isolated for one tree."""

    dbh_m: float | None
    dbh_quality: str
    stem_point_count: int
    band_point_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dbh_m": self.dbh_m,
            "dbh_quality": self.dbh_quality,
            "stem_point_count": self.stem_point_count,
            "band_point_count": self.band_point_count,
        }


def classify_dbh(
    stem_points_xy: Any,
    band_point_count: int,
    manual: ManualSpanConfig,
) -> DbhEvidence:
    """Classify DBH from the breast-height stem XY spread.

    *stem_points_xy* is the (N, 2) XY of full-resolution points at breast
    height inside the stem radius, or ``None``.  DBH is the mean of the robust
    X and Y widths (p90 - p10) — a horizontal diameter, never crown-derived.
    """

    stem_count = 0 if stem_points_xy is None else int(len(stem_points_xy))
    if stem_count == 0:
        quality = DBH_UNAVAILABLE if band_point_count == 0 else DBH_REVIEW
        return DbhEvidence(None, quality, 0, band_point_count)
    points = np.asarray(stem_points_xy, dtype=np.float64)
    width_x = float(np.quantile(points[:, 0], 0.90) - np.quantile(points[:, 0], 0.10))
    width_y = float(np.quantile(points[:, 1], 0.90) - np.quantile(points[:, 1], 0.10))
    dbh = 0.5 * (width_x + width_y)
    plausible = manual.dbh_min_m <= dbh <= manual.dbh_max_m
    if stem_count >= manual.dbh_measured_min_points and plausible:
        quality = DBH_MEASURED
    elif stem_count >= manual.dbh_estimated_min_points and plausible:
        quality = DBH_ESTIMATED
    else:
        quality = DBH_REVIEW
    return DbhEvidence(float(dbh) if plausible else None, quality, stem_count, band_point_count)


__all__ = [
    "DBH_MEASURED",
    "DBH_ESTIMATED",
    "DBH_REVIEW",
    "DBH_UNAVAILABLE",
    "DbhEvidence",
    "classify_dbh",
]
