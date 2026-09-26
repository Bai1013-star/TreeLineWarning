"""Generate a complete, JSON-serializable operational report."""

from __future__ import annotations

from collections import Counter
from typing import Any


def generate_history_report(
    rows: list[dict[str, Any]], standards: dict[str, Any] | None = None
) -> dict[str, Any]:
    levels = Counter(str(row.get("risk_level", "未知")) for row in rows)
    species = Counter(str(row.get("species", "未知")) for row in rows)
    urgent = [
        {
            "tree_id": row.get("tree_id"),
            "risk_level": row.get("risk_level"),
            "prediction": row.get("prediction"),
            "created_at": row.get("created_at"),
        }
        for row in rows
        if row.get("risk_level") == "高风险"
    ]
    non_compliant = [
        {
            "tree_id": row.get("tree_id"),
            "violations": row.get("standard_analysis", {}).get("violations", []),
            "standard_profile": row.get("standard_profile", {}),
            "created_at": row.get("created_at"),
        }
        for row in rows
        if row.get("standard_analysis")
        and not row.get("standard_analysis", {}).get("compliant", True)
    ]
    def average(field: str) -> float | None:
        values = [
            float(row[field])
            for row in rows
            if row.get(field) not in (None, "")
        ]
        return round(sum(values) / len(values), 3) if values else None

    return {
        "report_title": "TreeLineWarning 树线风险综合数据报告",
        "applied_standard": standards or {},
        "summary": {
            "total_predictions": len(rows),
            "risk_distribution": dict(levels),
            "species_distribution": dict(species),
            "high_risk_count": len(urgent),
            "standard_non_compliant_count": len(non_compliant),
        },
        "environment_factors": {
            "temperature_available": any(row.get("temperature") for row in rows),
            "sunlight_available": any(row.get("sunlight_hours") for row in rows),
        },
        "tree_metrics": {
            "average_height": average("height"),
            "average_diameter": average("diameter"),
            "species_distribution": dict(species),
        },
        "wire_metrics": {
            "voltage_levels_kv": sorted(
                {
                    float(row["voltage_level_kv"])
                    for row in rows
                    if row.get("voltage_level_kv")
                }
            ),
            "average_winter_maximum_sag": average("maximum_sag_winter"),
            "average_summer_maximum_sag": average("maximum_sag_summer"),
        },
        "tree_line_metrics": {
            "average_vertical_distance": average("vertical_distance"),
            "average_horizontal_distance": average("horizontal_distance"),
            "average_clearance_distance": average("clearance_distance"),
            "average_wind_offset_distance": average("wind_offset_distance"),
        },
        "weather_metrics": {
            "average_wind_speed": average("wind_speed"),
            "average_rainfall": average("rainfall"),
            "average_humidity": average("humidity"),
            "average_temperature": average("temperature"),
            "average_sunlight_hours": average("sunlight_hours"),
        },
        "priority_actions": urgent,
        "standard_non_compliance": non_compliant,
        "history": rows,
    }
