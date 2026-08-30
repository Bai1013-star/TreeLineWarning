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
        validated_source_preparation_references=(
            "artifacts/project-runs/source-shared/preparation.json",
        ),
    )

    dumped = value.to_dict()
    assert dumped["group_id"] == "G001"
    assert dumped["from_tower_id"] == "T001"
    assert dumped["validated_source_preparation_references"]


def test_span_region_configuration_uses_one_authoritative_span_corridor():
    configuration = span_region_configuration()
    assert configuration["vegetation_search_region"]["half_width_m"] == 9.0
    assert configuration["vegetation_search_region"]["longitudinal_halo_m"] == 6.0
    assert (
        configuration["vegetation_search_region"]["source"]
        == "tracked FastBaselineConfig span corridor"
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

    result = reconcile_trees([outcome], FastBaselineConfig(), origin_xyz=(0.0, 0.0, 0.0))

    assert result.final_tree_count == 1
    assert result.global_tree_records[0].span_id == 1
    assert result.diagnostics["single_span_fast_path"] is True


def _cross_span_outcome(span, tree_records, original_index, xyz, tree_id, in_core):
    return SimpleNamespace(
        span=span,
        tree_records=tuple(tree_records),
        raw_original_index=np.asarray(original_index, dtype=np.int64),
        raw_xyz_local_m=np.asarray(xyz, dtype=np.float64),
        raw_tree_id=np.asarray(tree_id, dtype=np.int32),
        in_core=np.asarray(in_core, dtype=bool),
    )


def _cross_span_record(span_id, start_x, end_x):
    return SimpleNamespace(
        line_id=0,
        line_name="TowerLine",
        span_id=span_id,
        span_name=f"S{span_id + 1:03d}",
        start_xyz_absolute_m=(start_x, 0.0, 0.0),
        end_xyz_absolute_m=(end_x, 0.0, 0.0),
        length_m=end_x - start_x,
        half_width_m=9.0,
    )


def test_cross_span_tree_has_one_stable_owner_and_halo_is_context_only():
    """Tower A--S005--B--S006--C keeps one physical crown as one entity."""
    s005 = _cross_span_record(4, 0.0, 10.0)
    s006 = _cross_span_record(5, 10.0, 20.0)
    s005_outcome = _cross_span_outcome(
        s005,
        (
            {"tree_id": 0, "apex_xy_local_m": [3.0, 0.0, 12.0], "apex_height_m": 12.0},
            {"tree_id": 1, "apex_xy_local_m": [9.5, 0.0, 13.0], "apex_height_m": 13.0},
        ),
        (100, 101, 102, 200, 201, 202),
        (
            (3.0, 0.0, 9.0),
            (3.1, 0.0, 10.0),
            (2.9, 0.0, 12.0),
            (9.5, 0.0, 10.0),
            (9.7, 0.0, 11.0),
            (9.4, 0.0, 13.0),
        ),
        (0, 0, 0, 1, 1, 1),
        (True, True, True, True, True, True),
    )
    s006_outcome = _cross_span_outcome(
        s006,
        (
            # This is the S005 crown seen only through S006's backward halo.
            {"tree_id": 0, "apex_xy_local_m": [9.5, 0.0, 13.0], "apex_height_m": 13.0},
            {"tree_id": 1, "apex_xy_local_m": [15.0, 0.0, 11.0], "apex_height_m": 11.0},
        ),
        (200, 201, 202, 300, 301, 302),
        (
            (9.5, 0.0, 10.0),
            (9.7, 0.0, 11.0),
            (9.4, 0.0, 13.0),
            (15.0, 0.0, 8.0),
            (15.2, 0.0, 9.0),
            (14.9, 0.0, 11.0),
        ),
        (0, 0, 0, 1, 1, 1),
        (False, False, False, True, True, True),
    )

    forward = reconcile_trees(
        [s005_outcome, s006_outcome], FastBaselineConfig(), origin_xyz=(0.0, 0.0, 0.0)
    )
    reverse = reconcile_trees(
        [s006_outcome, s005_outcome], FastBaselineConfig(), origin_xyz=(0.0, 0.0, 0.0)
    )

    assert forward.final_tree_count == 3
    shared = next(
        record for record in forward.global_tree_records if record.apex_xy_local_m[0] == 9.5
    )
    assert shared.span_id == 4
    assert shared.full_resolution_point_count == 3
    assert shared.core_point_count == 3
    assert len(shared.merged_span_tree_keys) == 2
    assert shared.merged_span_tree_keys == ((4, 1), (5, 0))
    assert set(forward.span_tree_to_global) == {(4, 0), (4, 1), (5, 0), (5, 1)}
    assert set(forward.span_tree_to_global).isdisjoint({(0, 0), (0, 1), (1, 0), (1, 1)})
    # A selected S006 run observes the old crown as context but cannot allocate it.
    assert (
        reconcile_trees(
            [s006_outcome], FastBaselineConfig(), origin_xyz=(0.0, 0.0, 0.0)
        ).final_tree_count
        == 1
    )
    assert [
        (record.span_id, record.apex_xy_local_m, record.full_resolution_point_count)
        for record in forward.global_tree_records
    ] == [
        (record.span_id, record.apex_xy_local_m, record.full_resolution_point_count)
        for record in reverse.global_tree_records
    ]
    assert np.array_equal(forward.final_original_index, reverse.final_original_index)
    assert np.array_equal(forward.final_xyz_local_m, reverse.final_xyz_local_m)
    assert np.array_equal(forward.final_tree_id, reverse.final_tree_id)
    assert [record.merged_span_tree_keys for record in forward.global_tree_records] == [
        record.merged_span_tree_keys for record in reverse.global_tree_records
    ]
    assert forward.span_tree_to_global == reverse.span_tree_to_global
    assert forward.diagnostics["duplicate_final_original_index"] == 0
    assert reverse.diagnostics["duplicate_final_original_index"] == 0
