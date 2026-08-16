"""Focused pure-contract tests for the multi-Span fast-track boundary."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from algorithms.spans.project_adapter import SpanAnalysisInput, span_region_configuration
from algorithms.spans.reconcile import reconcile_trees
from algorithms.spans.tree_runner import _row_in_span_core, _span_bounds_in_scene_local_frame
from core.config.fast_baseline import FastBaselineConfig
from core.project.span_sync import (
    MissingTowerEndpointError,
    TopologyDriftError,
    desired_group_spans,
)


class _TowerRegistry:
    def __init__(self, tower_ids):
        self._ids = set(tower_ids)

    def get(self, tower_id):
        if tower_id not in self._ids:
            raise KeyError(tower_id)
        return object()


def _session(groups, tower_ids):
    loaded = SimpleNamespace(
        tower_group_registry=SimpleNamespace(groups=tuple(groups)),
        tower_registry=_TowerRegistry(tower_ids),
    )
    return SimpleNamespace(loaded_project=loaded)


def test_group_adjacency_never_crosses_group_boundary():
    session = _session(
        (
            SimpleNamespace(group_id="G001", ordered_tower_ids=("T001", "T002", "T005")),
            SimpleNamespace(group_id="G002", ordered_tower_ids=("T003", "T004")),
        ),
        {"T001", "T002", "T003", "T004", "T005"},
    )

    pairs = [
        (item.group_id, item.from_tower_id, item.to_tower_id)
        for item in desired_group_spans(session)
    ]

    assert pairs == [
        ("G001", "T001", "T002"),
        ("G001", "T002", "T005"),
        ("G002", "T003", "T004"),
    ]
    assert ("G001", "T002", "T003") not in pairs


def test_missing_group_endpoint_fails_closed():
    session = _session(
        (SimpleNamespace(group_id="G001", ordered_tower_ids=("T001", "T002")),),
        {"T001"},
    )

    with pytest.raises(MissingTowerEndpointError, match="MISSING_TOWER_ENDPOINT"):
        desired_group_spans(session)


def test_duplicate_tower_membership_fails_closed_even_if_numbers_look_adjacent():
    session = _session(
        (
            SimpleNamespace(group_id="G001", ordered_tower_ids=("T001", "T002")),
            SimpleNamespace(group_id="G002", ordered_tower_ids=("T002", "T003")),
        ),
        {"T001", "T002", "T003"},
    )

    with pytest.raises(TopologyDriftError, match="TOPOLOGY_DRIFT"):
        desired_group_spans(session)


def test_analysis_input_serializes_explicit_endpoint_and_source_authority():
    digest = "a" * 64
    value = SpanAnalysisInput(
        project_id="project",
        source_identity={"source_id": "raw"},
        group_id="G001",
        span_id="S001",
        from_tower_id="T001",
        to_tower_id="T002",
        from_tower_revision=1,
        to_tower_revision=1,
        from_endpoint_geometry={"base_xyz_absolute_m": [0, 0, 0]},
        to_endpoint_geometry={"base_xyz_absolute_m": [10, 0, 0]},
        span_geometry={"length_m": 10},
        science_configuration_digest=digest,
        risk_configuration_digest=digest,
        validated_source_preparation_references=("artifacts/project-runs/source-shared/preparation.json",),
    )

    dumped = value.to_dict()
    assert dumped["group_id"] == "G001"
    assert dumped["from_tower_id"] == "T001"
    assert dumped["validated_source_preparation_references"]


def test_span_region_configuration_preserves_project_and_tracked_provenance():
    configuration = span_region_configuration()
    assert configuration["vegetation_search_region"]["half_width_m"] == 20.0
    assert configuration["vegetation_search_region"]["longitudinal_halo_m"] == 30.0
    assert configuration["vegetation_search_region"]["configuration_digest"] == (
        "79d6e9a201e0f195e7e90ef138f26e57289198f04fe6712dd65c90f41469c0c2"
    )
    assert configuration["wire_region"]["half_width_m"] == 9.0
    assert configuration["wire_region"]["longitudinal_halo_m"] == 6.0
    assert configuration["risk"]["threshold_m"] == 2.0
    assert configuration["risk"]["configuration_digest"] == (
        "063797f25c810cf857231ab7026eb030e735759182ded5f45365035110f919de"
    )


def test_authoritative_absolute_span_bounds_are_converted_to_region_local_frame():
    origin = (-100.0, 20.0, -2.0)
    minimum, maximum = _span_bounds_in_scene_local_frame(
        (-90.0, 30.0, 0.0), (-70.0, 50.0, 12.0), origin
    )

    np.testing.assert_allclose(minimum, (10.0, 10.0, 2.0))
    np.testing.assert_allclose(maximum, (30.0, 30.0, 14.0))


def test_span_core_membership_uses_the_same_local_frame_conversion():
    span = SimpleNamespace(
        core_bounds_min_xyz=(-90.0, 30.0, 0.0),
        core_bounds_max_xyz=(-70.0, 50.0, 12.0),
    )
    points = np.asarray(
        ((10.0, 10.0, 2.0), (30.0, 30.0, 14.0), (31.0, 30.0, 14.0)),
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        _row_in_span_core(points, span, (-100.0, 20.0, -2.0)),
        (True, True, False),
    )


def test_reconciliation_uses_span_list_position_not_authoritative_span_id():
    outcome = SimpleNamespace(
        span=SimpleNamespace(line_id=1, span_id=1, span_name="S002", line_name="G001"),
        raw_tree_id=np.asarray((0, 0), dtype=np.int32),
        raw_original_index=np.asarray((10, 11), dtype=np.int64),
        raw_xyz_local_m=np.asarray(((1.0, 2.0, 3.0), (1.1, 2.1, 3.1)), dtype=np.float64),
        in_core=np.asarray((True, True), dtype=bool),
        tree_records=(
            {
                "tree_id": 0,
                "apex_xy_local_m": [1.0, 2.0],
                "apex_height_m": 3.0,
            },
        ),
    )

    result = reconcile_trees(
        [outcome], FastBaselineConfig(), origin_xyz=(0.0, 0.0, 0.0)
    )

    assert result.final_tree_count == 1
    assert result.global_tree_records[0].span_id == 1
    assert result.diagnostics["single_span_fast_path"] is True
