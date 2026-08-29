from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from core.stage_a.models import CoordinateDeclaration, Line, Project, Span, Tower, UnitStatus
from core.stage_b.tree_display_id import TREE_DISPLAY_ID_SCHEMA, assign_tree_display_ids


def _project(*, reverse_first_span: bool = False) -> Project:
    coordinate = CoordinateDeclaration(
        "EPSG:32648",
        None,
        "m",
        UnitStatus.METRE,
        "GEOREFERENCED",
    )
    towers = (
        Tower("tower-107", "T107", "T107", (0.0, 0.0, 0.0), coordinate, "TEST"),
        Tower("tower-108", "T108", "T108", (100.0, 0.0, 0.0), coordinate, "TEST"),
        Tower("tower-109", "T109", "T109", (200.0, 0.0, 0.0), coordinate, "TEST"),
    )
    first_pair = ("tower-108", "tower-107") if reverse_first_span else ("tower-107", "tower-108")
    line_order = (
        ("tower-108", "tower-107", "tower-109")
        if reverse_first_span
        else ("tower-107", "tower-108", "tower-109")
    )
    line = Line("line-1", "QL-220-A", line_order)
    spans = (
        Span("span-1", line.line_id, first_pair[0], first_pair[1]),
        Span("span-2", line.line_id, line_order[1], line_order[2]),
    )
    return replace(Project.new("numbering"), lines=(line,), towers=towers, spans=spans)


def _record(tree_id: int, span_id: int, longitudinal_s_m: float):
    return SimpleNamespace(
        global_tree_id=tree_id,
        span_id=span_id,
        longitudinal_s_m=longitudinal_s_m,
    )


def test_tree_display_id_uses_span_number_and_longitudinal_order() -> None:
    records = (
        _record(10, 0, 70.0),
        _record(11, 0, 10.0),
        _record(12, 0, 40.0),
        _record(20, 1, 15.0),
    )
    values = assign_tree_display_ids(_project(), records)

    assert values[11]["display_tree_id"] == "T-1-001"
    assert values[12]["display_tree_id"] == "T-1-002"
    assert values[10]["display_tree_id"] == "T-1-003"
    assert values[20]["display_tree_id"] == "T-2-001"
    assert values[11]["display_tree_id_schema"] == TREE_DISPLAY_ID_SCHEMA
    assert values[11]["numbering_origin_tower_label"] == "T107"
    assert values[11]["numbering_direction_source"] == "LOWER_TOWER_NUMBER"


def test_reverse_span_still_starts_from_smaller_tower_number() -> None:
    # Product orientation is deliberately T108 -> T107.  Longitudinal s=90 is
    # therefore nearest T107 and must receive 001 despite the reversed span.
    records = (
        _record(1, 0, 10.0),
        _record(2, 0, 50.0),
        _record(3, 0, 90.0),
    )
    values = assign_tree_display_ids(_project(reverse_first_span=True), records)

    assert values[3]["display_tree_id"] == "T-1-001"
    assert values[2]["display_tree_id"] == "T-1-002"
    assert values[1]["display_tree_id"] == "T-1-003"
    assert values[3]["numbering_origin_tower_label"] == "T107"
    assert values[3]["numbering_destination_tower_label"] == "T108"


def test_presentation_numbering_does_not_require_or_change_persistent_id() -> None:
    record = SimpleNamespace(
        global_tree_id=7,
        span_id=0,
        longitudinal_s_m=22.0,
        persistent_tree_id="tree-stable-machine-id",
    )
    values = assign_tree_display_ids(_project(), (record,))

    assert values[7]["display_tree_id"] == "T-1-001"
    assert record.persistent_tree_id == "tree-stable-machine-id"
