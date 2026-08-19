"""Final persistent tree identity for the Folder pipeline.

Each final tree (after human DELETE / MERGE) receives exactly ONE unique
deterministic ``persistent_tree_id``.  The allocation order is the span
longitudinal position (lowest ``s`` first) with the final workflow tree id as
the tie-break, so a refresh or re-run of the same final membership always
produces the same ids.  Deleted trees never appear; a MERGE produces exactly
one final entity.

Format: ``<group_id>-S<span_seq:03d>-T<seq:03d>`` — e.g. ``G001-S001-T001``,
compatible with the existing project T### / S### identity family.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PERSISTENT_ID_SCHEMA = "folder-to-excel-persistent-tree-id-v1"


class IdentityError(RuntimeError):
    """Raised when a unique persistent id cannot be assigned."""


def assign_persistent_tree_ids(
    rows: list[dict[str, Any]],
    workspace: Any,
    *,
    group_id: str = "G001",
    span_id: str = "S001",
    start_tower_id: str = "T001",
    end_tower_id: str = "T002",
) -> list[dict[str, Any]]:
    """Assign a unique deterministic persistent_tree_id to every final row."""

    selection = _read_selection(workspace)
    group_id = str(selection.get("group_id") or group_id)
    span_id = str(selection.get("span_id") or span_id)
    start_tower_id = str(selection.get("start_tower_id") or start_tower_id)
    end_tower_id = str(selection.get("end_tower_id") or end_tower_id)
    project_id = selection.get("project_id")
    span_seq = _span_sequence(span_id)

    def _order_key(index: int) -> tuple[float, int]:
        row = rows[index]
        s_value = row.get("longitudinal_s_m")
        if s_value is None:
            return (float("inf"), int(row.get("tree_id", index)))
        return (float(s_value), int(row.get("tree_id", index)))

    ordered_indices = sorted(range(len(rows)), key=_order_key)
    for order, row_index in enumerate(ordered_indices, start=1):
        rows[row_index]["persistent_tree_id"] = f"{group_id}-S{span_seq:03d}-T{order:03d}"
        rows[row_index]["project_id"] = project_id
        rows[row_index]["group_id"] = group_id
        rows[row_index]["span_id"] = span_id
        rows[row_index]["start_tower_id"] = start_tower_id
        rows[row_index]["end_tower_id"] = end_tower_id

    ids = [row["persistent_tree_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise IdentityError("PERSISTENT_TREE_ID_DUPLICATES")
    if len(ids) != len(rows):
        raise IdentityError("PERSISTENT_TREE_ID_COUNT_MISMATCH")
    return rows


def _read_selection(workspace: Any) -> dict[str, Any]:
    path = getattr(workspace, "span_selection_path", None)
    if path is None or not Path(path).is_file():
        return {}
    return json.loads(Path(path).read_text("utf-8"))


def _span_sequence(span_id: str) -> int:
    digits = "".join(ch for ch in str(span_id) if ch.isdigit())
    if not digits:
        return 1
    return int(digits)


__all__ = [
    "PERSISTENT_ID_SCHEMA",
    "IdentityError",
    "assign_persistent_tree_ids",
]
