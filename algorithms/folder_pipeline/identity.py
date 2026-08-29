"""Final persistent and human-facing tree identity for the Folder pipeline.

Each final tree (after human DELETE / MERGE) receives exactly ONE unique
deterministic ``persistent_tree_id``.  The durable machine identity remains
span-independent and keeps the reviewed ``<group_id>-T<seq:03d>`` contract.

A separate ``display_tree_id`` is also assigned for operators.  It follows
``T-<span>-<tree>`` (for example ``T-1-001``).  The per-span sequence starts at
the endpoint with the smaller human tower number and proceeds longitudinally
toward the larger-numbered endpoint.  This presentation identity never replaces
or mutates the durable ``persistent_tree_id``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PERSISTENT_ID_SCHEMA = "folder-to-excel-persistent-tree-id-v1"
DISPLAY_ID_SCHEMA = "treeline-tree-display-id-v1"
_NUMERIC_SUFFIX_RE = re.compile(r"(\d+)(?!.*\d)")


class IdentityError(RuntimeError):
    """Raised when a unique persistent id cannot be assigned."""


def _numeric_suffix(value: str) -> int | None:
    match = _NUMERIC_SUFFIX_RE.search(str(value).strip())
    return None if match is None else int(match.group(1))


def _display_span_number(span_id: str) -> int:
    value = _numeric_suffix(span_id)
    return value if value is not None and value > 0 else 1


def _display_order_key(
    row: dict[str, Any],
    index: int,
    *,
    reverse: bool,
) -> tuple[float, int]:
    s_value = row.get("longitudinal_s_m")
    tree_id = int(row.get("tree_id", index))
    if s_value is None:
        return (float("inf"), tree_id)
    longitudinal = float(s_value)
    return (-longitudinal if reverse else longitudinal, tree_id)


def assign_persistent_tree_ids(
    rows: list[dict[str, Any]],
    workspace: Any,
    *,
    group_id: str = "G001",
    span_id: str = "S001",
    start_tower_id: str = "T001",
    end_tower_id: str = "T002",
) -> list[dict[str, Any]]:
    """Assign durable machine IDs plus separate operator display IDs.

    ``persistent_tree_id = <group_id>-T<seq:03d>`` remains span-independent.
    ``display_tree_id = T-<span>-<seq:03d>`` is presentation-only and is ordered
    from the smaller-numbered endpoint toward the larger-numbered endpoint.
    """

    selection = _read_selection(workspace)
    group_id = str(selection.get("group_id") or group_id)
    span_id = str(selection.get("span_id") or span_id)
    start_tower_id = str(selection.get("start_tower_id") or start_tower_id)
    end_tower_id = str(selection.get("end_tower_id") or end_tower_id)
    project_id = selection.get("project_id")

    def _persistent_order_key(index: int) -> tuple[float, int]:
        row = rows[index]
        s_value = row.get("longitudinal_s_m")
        if s_value is None:
            return (float("inf"), int(row.get("tree_id", index)))
        return (float(s_value), int(row.get("tree_id", index)))

    # Preserve the already-reviewed permanent ID semantics exactly.
    ordered_indices = sorted(range(len(rows)), key=_persistent_order_key)
    for order, row_index in enumerate(ordered_indices, start=1):
        rows[row_index]["persistent_tree_id"] = f"{group_id}-T{order:03d}"
        rows[row_index]["project_id"] = project_id
        rows[row_index]["group_id"] = group_id
        rows[row_index]["span_id"] = span_id
        rows[row_index]["start_tower_id"] = start_tower_id
        rows[row_index]["end_tower_id"] = end_tower_id

    # Human-facing numbering is independent of the permanent identity.  When
    # both endpoint numbers are known, 001 is always nearest the smaller one.
    start_number = _numeric_suffix(start_tower_id)
    end_number = _numeric_suffix(end_tower_id)
    reverse_display = (
        start_number is not None
        and end_number is not None
        and start_number != end_number
        and start_number > end_number
    )
    direction_source = (
        "LOWER_TOWER_NUMBER"
        if start_number is not None and end_number is not None and start_number != end_number
        else "SPAN_START_FALLBACK"
    )
    display_indices = sorted(
        range(len(rows)),
        key=lambda index: _display_order_key(rows[index], index, reverse=reverse_display),
    )
    display_span_number = _display_span_number(span_id)
    for display_order, row_index in enumerate(display_indices, start=1):
        rows[row_index]["display_tree_id"] = (
            f"T-{display_span_number}-{display_order:03d}"
        )
        rows[row_index]["display_tree_id_schema"] = DISPLAY_ID_SCHEMA
        rows[row_index]["display_tree_order"] = display_order
        rows[row_index]["display_span_number"] = display_span_number
        rows[row_index]["numbering_direction_source"] = direction_source
        rows[row_index]["numbering_origin_tower_id"] = (
            end_tower_id if reverse_display else start_tower_id
        )
        rows[row_index]["numbering_destination_tower_id"] = (
            start_tower_id if reverse_display else end_tower_id
        )

    ids = [row["persistent_tree_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise IdentityError("PERSISTENT_TREE_ID_DUPLICATES")
    if len(ids) != len(rows):
        raise IdentityError("PERSISTENT_TREE_ID_COUNT_MISMATCH")
    display_ids = [row["display_tree_id"] for row in rows]
    if len(display_ids) != len(set(display_ids)):
        raise IdentityError("DISPLAY_TREE_ID_DUPLICATES")
    return rows


def _read_selection(workspace: Any) -> dict[str, Any]:
    path = getattr(workspace, "span_selection_path", None)
    if path is None or not Path(path).is_file():
        return {}
    return json.loads(Path(path).read_text("utf-8"))


__all__ = [
    "PERSISTENT_ID_SCHEMA",
    "DISPLAY_ID_SCHEMA",
    "IdentityError",
    "assign_persistent_tree_ids",
]
