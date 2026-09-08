"""Shared measured values for desktop reports and spreadsheet publication."""

from __future__ import annotations

import math
from typing import Any

from core.stage_b.dbh_recovery import publication_fields, unavailable

CLEARANCE_FIELDS = (
    "horizontal_clearance_m",
    "vertical_clearance_m",
    "clearance_component_source",
)


def clearance_components(record: Any) -> dict[str, Any]:
    """Project the authoritative 3D nearest pair; never claim independent minima."""
    empty = dict.fromkeys(CLEARANCE_FIELDS)
    if record is None or getattr(record, "status", None) != "VALID":
        return empty
    try:
        tree = tuple(record.nearest_tree_xyz_local_m)
        wire = tuple(record.nearest_wire_xyz_local_m)
        if len(tree) != 3 or len(wire) != 3:
            return empty
        values = tuple(float(v) for v in (*tree, *wire))
        if not all(math.isfinite(v) for v in values):
            return empty
        dx, dy, dz = (values[i] - values[i + 3] for i in range(3))
        minimum = float(record.minimum_clearance_m)
        if not math.isfinite(minimum) or not math.isclose(
            math.hypot(dx, dy, dz), minimum, rel_tol=1e-7, abs_tol=1e-6
        ):
            return empty
    except (AttributeError, TypeError, ValueError, OverflowError):
        return empty
    return {
        "horizontal_clearance_m": math.hypot(dx, dy),
        "vertical_clearance_m": abs(dz),
        "clearance_component_source": "EXACT_3D_NEAREST_PAIR_COMPONENTS",
    }


def dbh_fields(metric: Any, recovered: dict[str, Any] | None) -> dict[str, Any]:
    """Publish bound recovery even when ordinary ground-based metrics are absent."""
    if recovered is not None:
        return publication_fields(recovered)
    if metric is None:
        return publication_fields(unavailable("GROUND_CONTEXT_UNAVAILABLE"))
    evidence = metric.dbh
    row = unavailable("BREAST_HEIGHT_" + evidence.dbh_quality)
    row.update(
        dbh_cm=None if evidence.dbh_m is None else float(evidence.dbh_m) * 100.0,
        dbh_quality=evidence.dbh_quality,
        dbh_point_count=evidence.stem_point_count,
        dbh_source="BREAST_HEIGHT_1P30M_ROBUST_XY_WIDTH",
    )
    if evidence.dbh_m is not None:
        row["dbh_method"] = "DIRECT_STEM"
    fields = publication_fields(row)
    fields["diameter_source"] = "BREAST_HEIGHT_1P30M_" + evidence.dbh_quality
    return fields
