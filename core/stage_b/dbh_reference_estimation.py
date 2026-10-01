"""Retire project-reference diameters without losing their audit trail.

A mean of other trees is not a measurement of the target stem. Earlier
reference values remain historical provenance while the target stays
unavailable until its own point-cloud evidence supports a diameter.
"""

from __future__ import annotations

from typing import Any

VERSION = "DBH_REFERENCE_RETIRED_V2"
SOURCE = "DBH_RECOVERY_V3_REFERENCE"


def apply_reference_estimates(
    rows: list[dict[str, Any]], *, reference_dbh_cm: float | None = None,
) -> dict[str, Any]:
    """Mark unresolved trees unavailable; never synthesize a target DBH."""
    if reference_dbh_cm is not None:
        raise ValueError("DBH_CROSS_TREE_REFERENCE_DISABLED")
    missing = [
        row for row in rows
        if row.get("dbh_cm") is None or row.get("dbh_method") == "REFERENCE_ESTIMATE"
    ]
    for row in missing:
        if row.get("dbh_method") == "REFERENCE_ESTIMATE":
            row["dbh_stale_reference_estimate"] = {
                "value_cm": row.get("dbh_cm"),
                "source": row.get("dbh_source"),
                "reference": row.get("dbh_reference_estimate"),
                "status": "RETIRED_NOT_TREE_MEASUREMENT",
            }
        reason = row.get("dbh_pre_reference_failure_reason") or row.get("dbh_fit_status")
        if reason in {"REFERENCE_ESTIMATE", "REFERENCE_REQUIRED", None}:
            reason = "PER_TREE_STEM_EVIDENCE_REQUIRED"
        row.update(
            dbh_cm=None, dbh_method="UNAVAILABLE", dbh_quality="UNAVAILABLE",
            dbh_point_count=0, dbh_evidence_point_count=0,
            dbh_band_point_count=0, dbh_fit_rmse=None, dbh_slice_count=0,
            dbh_source=VERSION, dbh_model_version=None, dbh_uncertainty_cm=None,
            dbh_fit_status=reason,
        )
        row.pop("dbh_reference_estimate", None)
        row["dbh_pre_reference_failure_reason"] = reason
        row.pop("selected_candidate_id", None)
    return {
        "status": "PER_TREE_STEM_EVIDENCE_REQUIRED" if missing else "NOT_NEEDED",
        "estimated_tree_count": 0,
        "remaining_tree_count": len(missing),
    }


__all__ = ["apply_reference_estimates", "SOURCE", "VERSION"]
