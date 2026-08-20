"""Pure Line/Route topology helpers for future multi-Span execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from core.stage_a.models import Line, Route, Span, stable_id

SpanIdFactory = Callable[[str, int], str]


def _make_spans(
    *,
    owner_id: str,
    line_id: str,
    ordered_tower_ids: Sequence[str],
    owner_kind: str,
    owner_metadata: Mapping[str, Any] | None,
    span_id_factory: SpanIdFactory | None,
) -> tuple[Span, ...]:
    if len(ordered_tower_ids) < 2:
        return ()
    metadata = dict(owner_metadata or {})
    result: list[Span] = []
    for sequence_index, (start_tower_id, end_tower_id) in enumerate(
        zip(ordered_tower_ids, ordered_tower_ids[1:], strict=False)
    ):
        span_id = (
            span_id_factory(owner_id, sequence_index)
            if span_id_factory is not None
            else stable_id("span")
        )
        result.append(
            Span(
                span_id=span_id,
                line_id=line_id,
                start_tower_id=start_tower_id,
                end_tower_id=end_tower_id,
                attributes={
                    "topology_origin": "ORDERED_TOWER_SEQUENCE",
                    "topology_owner_kind": owner_kind,
                    "topology_owner_id": owner_id,
                    "sequence_index": sequence_index,
                    **metadata,
                },
                source_partition={
                    "kind": "LOGICAL",
                    "access_mode": "SHARED_SOURCE_LOCAL",
                    "full_source_copy": False,
                    "sequence_index": sequence_index,
                    "owner_kind": owner_kind,
                    "owner_id": owner_id,
                },
            )
        )
    return tuple(result)


def generate_spans_from_ordered_towers(
    line: Line, *, span_id_factory: SpanIdFactory | None = None
) -> tuple[Span, ...]:
    """Generate adjacent logical Spans from a Line's ordered Tower sequence.

    This creates topology records only. It never copies, opens, or scans a
    point-cloud file; later science stages receive the shared source/cache and
    the logical partition metadata instead.
    """

    return _make_spans(
        owner_id=line.line_id,
        line_id=line.line_id,
        ordered_tower_ids=line.ordered_tower_ids,
        owner_kind="LINE",
        owner_metadata={"line_role": line.line_role.value},
        span_id_factory=span_id_factory,
    )


def generate_spans_from_route(
    route: Route, *, span_id_factory: SpanIdFactory | None = None
) -> tuple[Span, ...]:
    """Generate adjacent logical Spans from a Route's ordered Tower sequence."""

    return _make_spans(
        owner_id=route.route_id,
        line_id=route.line_id,
        ordered_tower_ids=route.ordered_tower_ids,
        owner_kind="ROUTE",
        owner_metadata={"route_role": route.route_role.value},
        span_id_factory=span_id_factory,
    )


__all__ = [
    "SpanIdFactory",
    "generate_spans_from_ordered_towers",
    "generate_spans_from_route",
]
