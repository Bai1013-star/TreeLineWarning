from __future__ import annotations

import numpy as np
import pytest
from core.contracts import (
    WireAssignmentQualityFlag,
    WireAssignmentReasonFlag,
    WireAssignmentSet,
    WireConfirmationChannel,
    WireInstanceSet,
    WireSeedConfig,
    WireSeparationConfig,
    WireSeparationStatistics,
    WireTrackQualityFlag,
    WireTrackSource,
    WireTrackStatus,
    WireTrackSummary,
)


def _summary(
    track_id: int,
    *,
    status: WireTrackStatus,
    wire_id: int,
) -> WireTrackSummary:
    return WireTrackSummary(
        track_id=track_id,
        wire_id=wire_id,
        status=status,
        source=WireTrackSource.AUTOMATIC,
        point_count=2,
        longitudinal_span_m=1.0,
        occupied_slice_count=2,
        coverage_ratio=1.0,
        mean_candidate_score=0.85,
        high_candidate_fraction=0.5,
        medium_candidate_fraction=0.5,
        low_candidate_fraction=0.0,
        direction_consistency=1.0,
        transverse_residual=0.0,
        curvature_proxy=0.0,
        maximum_gap_m=0.0,
        branch_conflict_count=0,
        median_u=0.0,
        median_v=0.0,
        first_s=0.0,
        actual_main_direction=(1.0, 0.0, 0.0),
        axis_angle_deg=0.0,
        confirmation_channel=(
            WireConfirmationChannel.OBVIOUS
            if status == WireTrackStatus.CONFIRMED
            else WireConfirmationChannel.NONE
        ),
        start_xyz=(0.0, 0.0, 0.0),
        end_xyz=(1.0, 0.0, 0.0),
        quality_flags=0,
    )


def test_a4_enum_values_are_fixed() -> None:
    assert [int(value) for value in WireTrackStatus] == [0, 1, 2]
    assert [int(value) for value in WireConfirmationChannel] == [0, 1, 2]
    assert [int(value) for value in WireTrackSource] == [0, 1, 2]
    assert int(WireTrackQualityFlag.MANUAL_SEEDED) == 2048
    assert int(WireAssignmentReasonFlag.UNASSIGNED_OUTLIER) == 512
    assert int(WireAssignmentQualityFlag.INVALID_INPUT) == 128


def test_config_seed_and_assignment_contracts() -> None:
    config = WireSeparationConfig(expected_input_point_count=4)
    assert config.slice_step_m == pytest.approx(0.1)
    assert config.fine_min_span_m == pytest.approx(6.0)
    assert WireSeparationConfig.from_dict(config.to_dict()) == config
    seed = WireSeedConfig(seed_original_indices=(10,), seed_group_ids=(2,))
    assert WireSeedConfig.from_dict(seed.to_dict()) == seed
    with pytest.raises(ValueError, match="equal length"):
        WireSeedConfig(seed_original_indices=(1,), seed_group_ids=())
    assignment = WireAssignmentSet(
        "scene_simple",
        np.arange(10, 14, dtype=np.int64),
        np.arange(4, dtype=np.int64),
        np.zeros((4, 3), dtype=np.float32),
        np.array([-1, 0, 1, -1], dtype=np.int32),
        np.array([-1, 0, -1, -1], dtype=np.int32),
        np.array([0, 0.8, 0.9, 0], dtype=np.float32),
        np.zeros(4, dtype=np.uint16),
        np.zeros(4, dtype=np.uint16),
    )
    assert assignment.point_count == 4
    with pytest.raises(ValueError, match="also have a track"):
        WireAssignmentSet(
            "scene_simple",
            np.arange(10, 14, dtype=np.int64),
            np.arange(4, dtype=np.int64),
            np.zeros((4, 3), dtype=np.float32),
            np.full(4, -1, dtype=np.int32),
            np.array([-1, 0, -1, -1], dtype=np.int32),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.uint16),
            np.zeros(4, dtype=np.uint16),
        )


def test_instance_offsets_and_zero_wire_serialization() -> None:
    confirmed = _summary(0, status=WireTrackStatus.CONFIRMED, wire_id=0)
    review = _summary(1, status=WireTrackStatus.REVIEW_REQUIRED, wire_id=-1)
    instances = WireInstanceSet(
        np.array([1, 2, 3, 4], dtype=np.int64),
        np.array([0, 2, 4], dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
        np.array([0, 2], dtype=np.int64),
        (confirmed, review),
    )
    assert instances.track_count == 2
    assert instances.wire_count == 1
    empty = WireInstanceSet(
        np.empty(0, dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.array([0], dtype=np.int64),
        (),
    )
    assert empty.track_count == 0
    assert empty.wire_count == 0
    statistics = WireSeparationStatistics(
        input_point_count=4,
        candidate_point_count=2,
        primary_slice_count=1,
        slice_cluster_count=1,
        initial_provisional_track_count=2,
        total_track_count=2,
        confirmed_wire_count=1,
        provisional_track_count=0,
        review_required_track_count=1,
        assigned_track_point_count=4,
        assigned_wire_point_count=2,
        unassigned_candidate_count=0,
        duplicate_assignment_count=0,
        overlap_duplicate_assignment_count=0,
        elapsed_seconds=0.0,
    )
    assert statistics.confirmed_wire_count == 1
