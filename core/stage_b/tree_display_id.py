"""Human-facing final tree numbering for Survey V0.1.

``persistent_tree_id`` remains the durable machine identity.  This module adds
an independent presentation identity of the form ``T-<span>-<tree>`` where the
per-span tree sequence always starts from the endpoint with the smaller human
tower number and proceeds longitudinally toward the larger-numbered endpoint.

The numbering is presentation/export metadata only.  It never changes tree
membership, global ownership, persistent identity, clearance, or risk science.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from core.stage_a.models import Project, Span, Tower

TREE_DISPLAY_ID_SCHEMA = "treeline-tree-display-id-v1"
_TOWER_NUMBER_RE = re.compile(r"(\d+)(?!.*\d)")


def _tower_number(tower: Tower) -> int | None:
    """Return the human tower number without trusting opaque stable IDs first."""

    for value in (tower.external_asset_id, tower.display_name):
        if not isinstance(value, str):
            continue
        match = _TOWER_NUMBER_RE.search(value.strip())
        if match is not None:
            return int(match.group(1))
    return None


def ordered_product_spans(project: Project) -> tuple[Span, ...]:
    """Reproduce the Stage-B bridge's deterministic product-span ordering."""

    spans_by_id = {span.span_id: span for span in project.spans}
    ordered: list[Span] = []
    seen: set[str] = set()
    for line_id in sorted(line.line_id for line in project.lines):
        line = next(line for line in project.lines if line.line_id == line_id)
        for start_tower_id, end_tower_id in zip(
            line.ordered_tower_ids,
            line.ordered_tower_ids[1:],
            strict=False,
        ):
            span = next(
                (
                    item
                    for item in project.spans
                    if item.line_id == line_id
                    and item.start_tower_id == start_tower_id
                    and item.end_tower_id == end_tower_id
                ),
                None,
            )
            if span is not None and span.span_id not in seen:
                ordered.append(span)
                seen.add(span.span_id)
    for span_id in sorted(spans_by_id):
        if span_id not in seen:
            ordered.append(spans_by_id[span_id])
            seen.add(span_id)
    return tuple(ordered)


def _record_value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _longitudinal_sort_key(record: Any, *, reverse: bool) -> tuple[float, int]:
    value = _record_value(record, "longitudinal_s_m")
    tree_id = int(_record_value(record, "global_tree_id", 0))
    if value is None:
        return (float("inf"), tree_id)
    distance = float(value)
    return (-distance if reverse else distance, tree_id)


def assign_tree_display_ids(
    project: Project,
    records: Sequence[Any],
) -> dict[int, dict[str, Any]]:
    """Assign deterministic human-facing IDs to final tree records.

    The science ``span_id`` is zero-based and follows the deterministic bridge
    ordering.  For each such span, trees are ordered by ``longitudinal_s_m``.
    When the product start tower has the larger human tower number, the order is
    reversed so ``001`` is still nearest the smaller-numbered endpoint.
    """

    if type(project) is not Project:
        raise TypeError("project must be a Stage-A Project")
    ordered_spans = ordered_product_spans(project)
    towers = {tower.tower_id: tower for tower in project.towers}
    grouped: dict[int, list[Any]] = {}
    for record in records:
        span_index = int(_record_value(record, "span_id"))
        grouped.setdefault(span_index, []).append(record)

    result: dict[int, dict[str, Any]] = {}
    for span_index, group in sorted(grouped.items()):
        span_number = span_index + 1
        product_span = (
            ordered_spans[span_index]
            if 0 <= span_index < len(ordered_spans)
            else None
        )
        reverse = False
        direction_source = "SPAN_START_FALLBACK"
        origin_tower: Tower | None = None
        destination_tower: Tower | None = None
        if product_span is not None:
            start = towers.get(product_span.start_tower_id)
            end = towers.get(product_span.end_tower_id)
            if start is not None and end is not None:
                start_number = _tower_number(start)
                end_number = _tower_number(end)
                if (
                    start_number is not None
                    and end_number is not None
                    and start_number != end_number
                ):
                    reverse = start_number > end_number
                    direction_source = "LOWER_TOWER_NUMBER"
                    origin_tower, destination_tower = (
                        (end, start) if reverse else (start, end)
                    )
                else:
                    origin_tower, destination_tower = start, end

        ordered_group = sorted(
            group,
            key=lambda record: _longitudinal_sort_key(record, reverse=reverse),
        )
        for tree_order, record in enumerate(ordered_group, start=1):
            global_tree_id = int(_record_value(record, "global_tree_id"))
            result[global_tree_id] = {
                "display_tree_id": f"T-{span_number}-{tree_order:03d}",
                "display_tree_id_schema": TREE_DISPLAY_ID_SCHEMA,
                "display_span_number": span_number,
                "display_tree_order": tree_order,
                "numbering_direction_source": direction_source,
                "numbering_origin_tower_id": (
                    None if origin_tower is None else origin_tower.tower_id
                ),
                "numbering_origin_tower_label": (
                    None if origin_tower is None else origin_tower.display_name
                ),
                "numbering_destination_tower_id": (
                    None if destination_tower is None else destination_tower.tower_id
                ),
                "numbering_destination_tower_label": (
                    None if destination_tower is None else destination_tower.display_name
                ),
            }
    return result


# Keep the old private spelling available to downstream compatibility callers
# while exposing the authoritative topology walk to export metadata code.
_ordered_product_spans = ordered_product_spans


__all__ = ["TREE_DISPLAY_ID_SCHEMA", "assign_tree_display_ids", "ordered_product_spans"]
