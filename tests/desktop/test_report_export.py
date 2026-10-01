from __future__ import annotations

from desktop.report_export import render_current_report_html


def test_html_export_contains_only_current_final_rows() -> None:
    report = {
        "status": "FINALIZED",
        "kpis": {"tree_count": 1, "minimum_clearance_m": 0.9, "warning_threshold_m": 1.5},
        "trees": [
            {
                "display_tree_id": "T-002",
                "persistent_tree_id": "survivor",
                "height_m": 5.0,
                "minimum_clearance_m": 0.9,
                "current_risk": "WARNING",
            }
        ],
    }

    rendered = render_current_report_html(report)

    assert "survivor" in rendered
    assert "deleted-tree" not in rendered
    assert "EDITOR_FINAL_PROJECTION" in rendered


def test_html_export_shows_incomplete_tower_exclusion() -> None:
    report = {"status": "FINALIZED", "trees": [], "kpis": {},
              "tower_exclusion_warnings": [{"message": "杆塔 T001：未能几何确认，不能视为已剔除"}]}
    rendered = render_current_report_html(report)
    assert "杆塔 T001：未能几何确认，不能视为已剔除" in rendered
    assert 'class="warning"' in rendered
