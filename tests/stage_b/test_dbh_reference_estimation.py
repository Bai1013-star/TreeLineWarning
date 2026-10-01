from __future__ import annotations

import pytest
from core.stage_b.dbh_reference_estimation import apply_reference_estimates


def row(tree_id, value, method, reason="PASS"):
    return {
        "global_tree_id": tree_id,
        "persistent_tree_id": f"tree-{tree_id}",
        "dbh_cm": value,
        "dbh_method": method,
        "dbh_quality": "HIGH" if value is not None else "UNAVAILABLE",
        "dbh_fit_status": reason,
        "dbh_point_count": 90 if value is not None else 0,
        "dbh_evidence_point_count": 90 if value is not None else 0,
    }


def test_other_trees_never_supply_missing_tree_diameter():
    rows = [
        row(2, 27.3, "RECOVERED_STEM"),
        row(7, 18.3, "RECOVERED_STEM"),
        row(4, None, "UNAVAILABLE", "NO_CROWN_TO_STEM_IDENTITY"),
    ]
    summary = apply_reference_estimates(rows)
    assert summary == {
        "status": "PER_TREE_STEM_EVIDENCE_REQUIRED",
        "estimated_tree_count": 0,
        "remaining_tree_count": 1,
    }
    assert [r["dbh_cm"] for r in rows] == [27.3, 18.3, None]
    assert rows[2]["dbh_fit_status"] == "NO_CROWN_TO_STEM_IDENTITY"


def test_prior_reference_value_is_retired_with_provenance():
    old = row(4, 22.8, "REFERENCE_ESTIMATE", "REFERENCE_ESTIMATE")
    old["dbh_reference_estimate"] = {
        "basis": "PROJECT_TRUSTED_MEASUREMENTS", "value_cm": 22.8,
        "reference_tree_ids": [2, 7],
    }
    old["dbh_pre_reference_failure_reason"] = "TOP_CANDIDATE_CONFLICT"
    apply_reference_estimates([old])
    assert old["dbh_cm"] is None
    assert old["dbh_method"] == old["dbh_quality"] == "UNAVAILABLE"
    assert old["dbh_fit_status"] == "TOP_CANDIDATE_CONFLICT"
    assert old["dbh_point_count"] == old["dbh_evidence_point_count"] == 0
    assert "dbh_reference_estimate" not in old
    assert old["dbh_stale_reference_estimate"] == {
        "value_cm": 22.8,
        "source": None,
        "reference": {"basis": "PROJECT_TRUSTED_MEASUREMENTS", "value_cm": 22.8,
                      "reference_tree_ids": [2, 7]},
        "status": "RETIRED_NOT_TREE_MEASUREMENT",
    }


def test_user_or_project_reference_cannot_enter_measurement():
    rows = [row(1, None, "UNAVAILABLE", "NO_ABSOLUTE_CANDIDATE")]
    with pytest.raises(ValueError, match="DBH_CROSS_TREE_REFERENCE_DISABLED"):
        apply_reference_estimates(rows, reference_dbh_cm=25.0)
    assert rows[0]["dbh_cm"] is None


def test_existing_stem_measurement_unchanged():
    measured = row(1, 24.5, "REVIEW_CONFIRMED_STEM")
    assert apply_reference_estimates([measured])["status"] == "NOT_NEEDED"
    assert measured["dbh_cm"] == 24.5
