"""V4 single-tree segmentation runtime tests (PLAN-FROZEN consolidated plan).

The tests use small deterministic synthetic full-resolution measured scenes
constructed directly as ``V4InputHandoffResult`` objects so the frozen
Algorithm-V1 scoring and scene semantics are exercised without the heavy FRV
chain.  Every scene is deterministic (no RNG), and every point carries a unique
``original_index``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from algorithms.vegetation.v4_tree_segmentation import (
    assignment_base,
    assignment_confidence,
    assignment_decision,
    assignment_support_score,
    association_exclusive01,
    association_strong01,
    merge_support_score,
    near,
    profile_confirmed_stop,
    relation_decision,
    responsibility_pressure,
    responsibility_r0,
    responsibility_score,
    safe_ratio,
    segment_v4_trees,
    separate_support_score,
    signed,
    similar,
    trunk_apex_score,
    trunk_apex_state,
    trunk_candidate_score,
    trunk_candidate_state,
)
from core.contracts.frv import frv_digest
from core.contracts.v4_inputs import (
    V4InputHandoffResult,
    V4RecoveredVegetationSet,
    V4RecoveryCoverageReport,
    V4ValidatedFrvMembershipView,
    frv_v4_recovered_index_digest,
)
from core.contracts.v4_tree_segmentation import (
    RESPONSIBILITY_MIN,
    RESPONSIBILITY_REVIEW_MARGIN,
    TRUNK_APEX_REVIEW_MIN,
    TRUNK_APEX_STRONG_MIN,
    V4ApexCandidateRecord,
    V4ApexQualityFlag,
    V4AssociationLookupKind,
    V4ConfigStatus,
    V4ProfileSampleState,
    V4TreeAssignmentState,
    V4TreeHypothesisState,
    V4TreeRelation,
    V4TreeSegmentationConfig,
    V4TrunkApexAssociationEvidence,
    V4TrunkApexAssociationState,
    V4TrunkCandidateRecord,
    V4TrunkCandidateState,
)

from tests.io.test_source_provider import _source_identity

# ---------------------------------------------------------------------------
# Shared synthetic-handoff builders (imported by artifact and CLI tests)
# ---------------------------------------------------------------------------


def make_config(**overrides) -> V4TreeSegmentationConfig:
    fields = {
        "schema_version": "v4-tree-segmentation-config-v1",
        "status": V4ConfigStatus.TEST_ONLY,
        "xy_cell_size_m": 0.5,
        "apex_neighborhood_radius_m": 1.0,
        "minimum_apex_prominence_m": 0.15,
        "density_trend_window_m": 1.0,
        "slice_thickness_m": 1.0,
        "slice_overlap_fraction": 0.25,
        "relation_neighborhood_radius_m": 4.0,
        "maximum_points": 200000,
        "maximum_grid_cells": 400000,
        "maximum_fragments": 5000,
        "maximum_local_edges": 50000,
        "maximum_path_evaluations": 500000,
        "maximum_profile_samples": 500000,
    }
    fields.update(overrides)
    digest_fields = {
        key: value.value if isinstance(value, V4ConfigStatus) else value
        for key, value in fields.items()
    }
    return V4TreeSegmentationConfig(**fields, config_digest=frv_digest(digest_fields))


def make_tree_points(cx: float, cy: float) -> list[tuple[float, float, float]]:
    """A deterministic dome crown on a thick trunk column."""

    points: list[tuple[float, float, float]] = []
    for ring in range(10):
        radius = 0.2 + 1.5 * ring / 10
        for slot in range(14):
            theta = 2.0 * math.pi * slot / 14
            x = cx + radius * math.cos(theta)
            y = cy + radius * math.sin(theta)
            hag = 3.0 + 3.2 * (1.0 - (radius / 1.6) ** 2)
            points.append((x, y, hag))
    for offset in (-0.2, 0.0, 0.2):
        for _second in range(3):
            points.append((cx + offset, cy + offset, 6.2))
    # Thick trunk: 3 points per bin over 0.4-3.0 m.
    for k in range(14):
        hag = 0.4 + 2.6 * k / 14
        for _dup in range(3):
            points.append((cx + 0.05 * math.sin(k), cy + 0.05 * math.cos(k * 0.7), hag))
    return points


def make_handoff(
    scene_points: list[tuple[float, float, float]],
    *,
    row_permutation: list[int] | None = None,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, 0.0, 0.0),
        (14.0, 6.0, 7.0),
    ),
    membership_artifact: Any = None,
) -> V4InputHandoffResult:
    n = len(scene_points)
    if n == 0:
        xyz = np.zeros((0, 3), dtype=np.float64)
    else:
        xyz = np.asarray(scene_points, dtype=np.float64)
    hag = xyz[:, 2].astype(np.float32)
    original = np.arange(n, dtype=np.int64)
    if row_permutation is not None:
        order = np.asarray(row_permutation, dtype=np.int64)
        # Row j carries the point whose original_index is order[j]; the
        # original_index -> geometry mapping is preserved.
        xyz = xyz[order]
        hag = hag[order]
        original = original[order]
    source = _source_identity(
        "C:/synthetic.las", "a" * 64, n, bounds
    )
    recovered = V4RecoveredVegetationSet.__new__(V4RecoveredVegetationSet)
    object.__setattr__(recovered, "source_identity", source)
    object.__setattr__(
        recovered, "original_index", _ro(original.astype("<i8"))
    )
    object.__setattr__(recovered, "xyz_local_m", _ro(xyz.astype("<f8")))
    object.__setattr__(recovered, "hag_m", _ro(hag.astype("<f4")))
    object.__setattr__(
        recovered, "membership_state", _ro(np.ones(n, dtype="|u1"))
    )
    object.__setattr__(
        recovered, "hag_evidence_state", _ro(np.ones(n, dtype="|u1"))
    )
    object.__setattr__(
        recovered, "segmentation_hag_readiness", _ro(np.ones(n, dtype="|u1"))
    )
    object.__setattr__(
        recovered, "candidate_reason_flags", _ro(np.zeros(n, dtype="<u4"))
    )
    object.__setattr__(recovered, "candidate_confidence", None)
    object.__setattr__(
        recovered, "frv_membership_artifact_digest", "b" * 64
    )
    object.__setattr__(
        recovered,
        "recovered_index_digest",
        frv_v4_recovered_index_digest(
            source_content_identity_digest=source.source_content_identity_digest,
            original_index=original,
        ),
    )
    view = V4ValidatedFrvMembershipView.__new__(V4ValidatedFrvMembershipView)
    object.__setattr__(view, "source_identity", source)
    if membership_artifact is not None:
        object.__setattr__(view, "membership_artifact", membership_artifact)
    object.__setattr__(view, "view_digest", "d" * 64)
    report = V4RecoveryCoverageReport.__new__(V4RecoveryCoverageReport)
    object.__setattr__(report, "report_digest", "e" * 64)
    handoff = V4InputHandoffResult.__new__(V4InputHandoffResult)
    object.__setattr__(handoff, "validated_membership", view)
    object.__setattr__(handoff, "recovered_vegetation", recovered)
    object.__setattr__(handoff, "unresolved_measured_points", None)
    object.__setattr__(handoff, "unrecovered_regions", ())
    object.__setattr__(handoff, "coverage_report", report)
    object.__setattr__(handoff, "output_reference", None)
    object.__setattr__(handoff, "handoff_digest", "c" * 64)
    return handoff


def _ro(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def two_trees_scene() -> list[tuple[float, float, float]]:
    return make_tree_points(2.0, 2.0) + make_tree_points(9.0, 2.0)


# ---------------------------------------------------------------------------
# Frozen Algorithm-V1 scoring vectors
# ---------------------------------------------------------------------------


def test_responsibility_first_seven_one_no_rival() -> None:
    r0 = responsibility_r0(
        apex_support=1.0,
        path_bottleneck=1.0,
        density_trend=1.0,
        hag_continuity=1.0,
        distance_decay=1.0,
        connectivity=1.0,
        empty_gap=1.0,
    )
    assert abs(r0 - 1.0) < 1e-12
    pressure = responsibility_pressure(r0, None)
    assert pressure == 1.0
    score = responsibility_score(
        apex_support=1.0,
        path_bottleneck=1.0,
        density_trend=1.0,
        hag_continuity=1.0,
        distance_decay=1.0,
        connectivity=1.0,
        empty_gap=1.0,
        pressure=pressure,
    )
    assert abs(score - 1.0) < 1e-12


def test_responsibility_first_seven_half_equal_rival() -> None:
    r0 = responsibility_r0(
        apex_support=0.5,
        path_bottleneck=0.5,
        density_trend=0.5,
        hag_continuity=0.5,
        distance_decay=0.5,
        connectivity=0.5,
        empty_gap=0.5,
    )
    assert abs(r0 - 0.5) < 1e-12
    pressure = responsibility_pressure(r0, r0)
    assert abs(pressure - 0.5) < 1e-12
    score = responsibility_score(
        apex_support=0.5,
        path_bottleneck=0.5,
        density_trend=0.5,
        hag_continuity=0.5,
        distance_decay=0.5,
        connectivity=0.5,
        empty_gap=0.5,
        pressure=pressure,
    )
    assert abs(score - 0.5) < 1e-12
    assert score < RESPONSIBILITY_MIN


def test_responsibility_rival_margin() -> None:
    r0_best = 0.70
    r0_second = 0.64
    pressure = responsibility_pressure(r0_best, r0_second)
    assert pressure < 1.0
    # The 0.06 margin keeps the secondary within the review margin.
    assert r0_best - r0_second <= RESPONSIBILITY_REVIEW_MARGIN + 1e-12


def test_relation_all_plus_merges() -> None:
    channels = (1.0,) * 8
    merge = merge_support_score(channels)
    separate = separate_support_score(channels)
    assert abs(merge - 1.0) < 1e-12
    assert abs(separate - 0.0) < 1e-12
    assert relation_decision(merge, separate) is V4TreeRelation.MERGE


def test_relation_negative_with_neutral_trunk_keeps_separate() -> None:
    channels = (-1.0,) * 7 + (0.0,)
    merge = merge_support_score(channels)
    separate = separate_support_score(channels)
    assert abs(merge - 0.0) < 1e-12
    assert abs(separate - 0.94) < 1e-12
    assert relation_decision(merge, separate) is V4TreeRelation.KEEP_SEPARATE


def test_relation_all_neutral_reviews() -> None:
    assert relation_decision(0.0, 0.0) is V4TreeRelation.REVIEW


def test_trunk_candidate_fixed_vectors() -> None:
    assert abs(trunk_candidate_score(1.0, 1.0, 1.0, 1.0) - 1.0) < 1e-12
    assert trunk_candidate_state(1.0) is V4TrunkCandidateState.ACCEPTED
    assert abs(trunk_candidate_score(0.5, 0.5, 0.5, 0.5) - 0.5) < 1e-12
    assert trunk_candidate_state(0.5) is V4TrunkCandidateState.SUPPRESSED


def test_trunk_apex_fixed_vectors() -> None:
    assert abs(trunk_apex_score(1.0, 1.0, 1.0, 1.0, 1.0) - 1.0) < 1e-12
    assert trunk_apex_state(0.55) is V4TrunkApexAssociationState.REVIEW_ASSOCIATION
    assert trunk_apex_state(0.65) is V4TrunkApexAssociationState.STRONG_ASSOCIATION
    assert trunk_apex_state(0.50) is V4TrunkApexAssociationState.UNSUPPORTED_ASSOCIATION


def test_strong01_boundaries() -> None:
    review = association_strong01(0.55, V4TrunkApexAssociationState.REVIEW_ASSOCIATION)
    exclusive = association_exclusive01(0.55, V4TrunkApexAssociationState.REVIEW_ASSOCIATION)
    assert review == 0.0 and exclusive == 0.0
    strong = association_strong01(0.65, V4TrunkApexAssociationState.STRONG_ASSOCIATION)
    assert abs(strong - 0.5) < 1e-12
    next_strong = association_strong01(
        math.nextafter(TRUNK_APEX_STRONG_MIN, math.inf),
        V4TrunkApexAssociationState.STRONG_ASSOCIATION,
    )
    assert next_strong > 0.5
    one = association_strong01(1.0, V4TrunkApexAssociationState.STRONG_ASSOCIATION)
    assert abs(one - 1.0) < 1e-12
    unsupported = association_exclusive01(
        0.0, V4TrunkApexAssociationState.UNSUPPORTED_ASSOCIATION
    )
    assert abs(unsupported - 1.0) < 1e-12
    assert (
        abs(
            association_exclusive01(
                0.0, V4TrunkApexAssociationState.UNSUPPORTED_ASSOCIATION
            )
            - 1.0
        )
        < 1e-12
    )
    assert (
        abs(
            association_exclusive01(
                TRUNK_APEX_REVIEW_MIN, V4TrunkApexAssociationState.UNSUPPORTED_ASSOCIATION
            )
        )
        < 1e-9
    )


def test_assignment_fixed_vectors() -> None:
    all_plus = (1.0,) * 9
    all_minus = (-1.0,) * 9
    first8_minus_plus_quality = (-1.0,) * 8 + (1.0,)
    all_zero = (0.0,) * 9
    assert abs(assignment_support_score(all_plus) - 1.0) < 1e-12
    assert abs(assignment_support_score(all_minus) - 0.0) < 1e-12
    assert abs(assignment_support_score(first8_minus_plus_quality) - 0.06) < 1e-12
    assert abs(assignment_support_score(all_zero) - 0.5) < 1e-12
    # Base excludes competitor (weight 0.10) and includes segmentation quality
    # (weight 0.06); its eight weights therefore sum to exactly 0.90.
    assert abs(assignment_base((1.0,) * 8) - 1.0) < 1e-12
    assert abs(assignment_base((-1.0,) * 7 + (1.0,)) - (1.0 / 15.0)) < 1e-12
    assert abs(assignment_base((-1.0,) * 8) - 0.0) < 1e-12


def test_assignment_decisions() -> None:
    # first eight -1, quality +1 -> UNASSIGNED with exact confidence.
    support = 0.06
    decision = assignment_decision(
        support=support,
        second=None,
        mandatory_review=False,
        comparison_complete=True,
    )
    assert decision is V4TreeAssignmentState.UNASSIGNED
    confidence = assignment_confidence(V4TreeAssignmentState.UNASSIGNED, support, None)
    assert abs(confidence - 0.9032258064516129) < 1e-9
    # all nine -1 -> REVIEW_REQUIRED via mandatory review.
    support_minus = 0.0
    decision_minus = assignment_decision(
        support=support_minus,
        second=None,
        mandatory_review=True,
        comparison_complete=True,
    )
    assert decision_minus is V4TreeAssignmentState.REVIEW_REQUIRED
    assert assignment_confidence(V4TreeAssignmentState.REVIEW_REQUIRED, 0.0, None) == 0.0
    # best final support 0.70 second 0.65 -> REVIEW_REQUIRED (margin below 0.10).
    decision_margin = assignment_decision(
        support=0.70,
        second=0.65,
        mandatory_review=False,
        comparison_complete=True,
    )
    assert decision_margin is V4TreeAssignmentState.REVIEW_REQUIRED
    assert assignment_confidence(V4TreeAssignmentState.REVIEW_REQUIRED, 0.0, None) == 0.0
    # best final support 0.70 second None -> ASSIGNED (clear best).
    decision_clear = assignment_decision(
        support=0.70,
        second=None,
        mandatory_review=False,
        comparison_complete=True,
    )
    assert decision_clear is V4TreeAssignmentState.ASSIGNED
    # all zero -> UNASSIGNED confidence 0.1935483870967742.
    conf_zero = assignment_confidence(V4TreeAssignmentState.UNASSIGNED, 0.5, None)
    assert abs(conf_zero - 0.1935483870967742) < 1e-9


def test_normalizer_boundaries() -> None:
    assert abs(near(0.0, 1.0) - 1.0) < 1e-12
    assert abs(near(0.5, 1.0) - 0.5) < 1e-12
    assert abs(near(1.0, 1.0) - 0.0) < 1e-12
    assert abs(near(2.0, 1.0) - 0.0) < 1e-12
    assert abs(similar(0.0, 1.0) - 1.0) < 1e-12
    assert abs(signed(1.0) - 1.0) < 1e-12
    assert abs(signed(0.5) - 0.0) < 1e-12
    assert abs(signed(0.0) - (-1.0)) < 1e-12
    assert abs(safe_ratio(0.0, 1.0, 1.0) - 0.0) < 1e-12
    assert abs(safe_ratio(1.0, 1.0, 1.0) - 1.0) < 1e-12


def test_profile_confirmed_stop_rules() -> None:
    def stop(measured_gap, confirmed_valley, density_declined, hag_declined):
        return profile_confirmed_stop(
            measured_gap=measured_gap,
            confirmed_valley=confirmed_valley,
            local_density_declined=density_declined,
            local_hag_declined=hag_declined,
        )

    assert stop(True, False, False, False) is True
    assert stop(False, True, True, False) is True
    assert stop(False, True, False, True) is True
    assert stop(False, True, False, False) is False
    assert stop(False, False, False, False) is False


# ---------------------------------------------------------------------------
# Scene integration tests
# ---------------------------------------------------------------------------


def _run(scene, config=None):
    handoff = make_handoff(scene)
    return segment_v4_trees(handoff, config or make_config())


def test_single_tree_scene() -> None:
    result = _run(make_tree_points(2.0, 2.0))
    assert len(result.tree_records) == 1
    tree = result.tree_records[0]
    assert tree.member_count > 50
    states = result.assignment_state_by_point
    count = int(result.original_index.shape[0])
    assigned = int((states == 1).sum())
    review = int((states == 2).sum())
    unassigned = int((states == 0).sum())
    assert assigned + review + unassigned == count
    assert assigned > 0
    assert result.audit.tree_count == 1


def test_two_separated_trees_scene() -> None:
    result = _run(two_trees_scene())
    assert len(result.tree_records) == 2
    assigned = int((result.assignment_state_by_point == 1).sum())
    assert assigned > 100
    # Tree IDs are contiguous and unique per assigned point.
    ids = [t.tree_id for t in result.tree_records]
    assert ids == [0, 1]
    assert result.audit.tree_count == 2


def test_empty_input_scene() -> None:
    result = _run([])
    assert len(result.tree_records) == 0
    assert result.audit.tree_count == 0
    assert int(result.original_index.shape[0]) == 0
    assert int((result.assignment_state_by_point == 1).sum()) == 0


def test_no_valid_apex_all_review() -> None:
    # A flat scene with no discernible peaks: no accepted apex -> zero trees,
    # every point REVIEW_REQUIRED.
    flat: list[tuple[float, float, float]] = []
    for i in range(60):
        for j in range(60):
            if (i * 60 + j) % 3 == 0:
                flat.append((i * 0.2, j * 0.2, 1.0 + 0.01 * ((i + j) % 5)))
    result = _run(flat, make_config(minimum_apex_prominence_m=0.5))
    assert len(result.tree_records) == 0
    states = result.assignment_state_by_point
    assert int((states == int(V4TreeAssignmentState.REVIEW_REQUIRED)).sum()) == int(
        result.original_index.shape[0]
    )


def test_conservation_exact_original_index() -> None:
    scene = two_trees_scene()
    result = _run(scene)
    assert list(result.original_index) == sorted(result.original_index)
    count = int(result.original_index.shape[0])
    assigned = int((result.assignment_state_by_point == 1).sum())
    review = int((result.assignment_state_by_point == 2).sum())
    unassigned = int((result.assignment_state_by_point == 0).sum())
    assert assigned + review + unassigned == count
    members = list(result.ordered_tree_member_original_index)
    assert len(members) == assigned
    assert members == sorted(members)
    # No duplicate tree ownership.
    assert len(set(members)) == len(members)


def test_determinism_shuffled_row_order() -> None:
    scene = two_trees_scene()
    n = len(scene)
    permutation = list(range(n))
    rng = np.random.default_rng(42)
    rng.shuffle(permutation)
    handoff_a = make_handoff(scene)
    handoff_b = make_handoff(scene, row_permutation=permutation)
    result_a = segment_v4_trees(handoff_a, make_config())
    result_b = segment_v4_trees(handoff_b, make_config())
    assert result_a.result_digest == result_b.result_digest
    assert [t.tree_id for t in result_a.tree_records] == [t.tree_id for t in result_b.tree_records]


def test_determinism_repeat_run() -> None:
    result_a = _run(two_trees_scene())
    result_b = _run(two_trees_scene())
    assert result_a.result_digest == result_b.result_digest


def test_trunk_candidate_detected() -> None:
    result = _run(make_tree_points(2.0, 2.0))
    accepted = [
        record
        for record in result.trunk_records
        if record.candidate_state is V4TrunkCandidateState.ACCEPTED
    ]
    assert len(accepted) >= 1
    trunk = accepted[0]
    assert trunk.observed_vertical_span_m > 1.0
    assert trunk.representative_original_index >= 0


def test_apex_candidates_recorded() -> None:
    result = _run(two_trees_scene())
    assert len(result.apex_records) >= 2
    for apex in result.apex_records:
        assert apex.diagnostic_digest == apex.computed_digest()


def test_rotating_profile_24_rays() -> None:
    result = _run(make_tree_points(2.0, 2.0))
    assert len(result.rotating_profile_records) >= 1
    for profile in result.rotating_profile_records:
        assert len(profile.rays) == 24


def test_association_and_path_records_available() -> None:
    result = _run(make_tree_points(2.0, 2.0))
    accepted = [
        record
        for record in result.trunk_records
        if record.candidate_state is V4TrunkCandidateState.ACCEPTED
    ]
    if accepted:
        assert len(result.trunk_apex_association_records) >= 1
        assert len(result.structural_path_records) >= 1


def test_tree_records_geometry_consistent() -> None:
    result = _run(two_trees_scene())
    for tree in result.tree_records:
        geometry = tree.geometry_summary
        assert geometry.member_count == tree.member_count
        assert geometry.vertical_extent_m >= 0.0
        assert geometry.hag_max_m >= geometry.hag_min_m
        assert tree.member_original_index_digest
        assert tree.confidence >= 0.0


def test_cpu_resource_gate_fails_closed() -> None:
    """Exhausting the path-evaluation resource gate hard-fails, never truncates."""

    from core.contracts.v4_tree_segmentation import V4TreeSegmentationError

    handoff = make_handoff(two_trees_scene())
    config = make_config(maximum_path_evaluations=4)
    with pytest.raises(V4TreeSegmentationError):
        segment_v4_trees(handoff, config)


def test_cpu_resource_gate_bounded_success() -> None:
    """A generous but bounded resource ceiling still produces a valid result."""

    handoff = make_handoff(two_trees_scene())
    config = make_config(
        maximum_points=1000,
        maximum_grid_cells=20000,
        maximum_fragments=500,
        maximum_local_edges=5000,
        maximum_path_evaluations=20000,
        maximum_profile_samples=20000,
    )
    result = segment_v4_trees(handoff, config)
    assert result.audit.total_points == int(result.original_index.shape[0])


# ---------------------------------------------------------------------------
# V4-BUNDLE-A review-fix regression tests (V4A-IR-015)
# ---------------------------------------------------------------------------


def test_apex_support_uses_prominence_not_absolute_hag() -> None:
    """V4A-IR-003/015 #4: apex support must use prominence, never CHM height."""
    from algorithms.vegetation.v4_tree_segmentation import (
        _build_grid,
        _canonical_population,
        _detect_apex_candidates,
        derive_algorithm_scales,
        safe_ratio,
    )

    handoff = make_handoff(two_trees_scene())
    config = make_config()
    population = _canonical_population(handoff)
    scales = derive_algorithm_scales(config)
    grid = _build_grid(population, scales, config)
    records = _detect_apex_candidates(population, grid, scales, config)
    assert records
    for record in records:
        expected_density = safe_ratio(
            record.density_points_per_m2,
            record.neighborhood_median_density_points_per_m2,
            scales.rho_floor,
        )
        if record.prominence_available:
            expected = 0.60 * min(1.0, max(0.0, record.prominence_m / scales.la)) + (
                0.40 * expected_density
            )
        else:
            expected = 0.0 + 0.40 * expected_density
        assert abs(record.apex_support_score - expected) < 1e-9


def test_apex_plateau_density_uses_supporting_point_cell(monkeypatch) -> None:
    """V4A-IR-003: rho_apex follows the canonical supporting point, not the
    first cell ID of a collapsed multi-cell plateau."""

    from algorithms.vegetation import v4_tree_segmentation as v4s

    population = v4s._canonical_population(
        make_handoff(
            [
                (1.1, 0.1, 4.0),
                (0.1, 0.1, 4.0),
                (0.2, 0.1, 3.8),
                (0.3, 0.1, 3.7),
                (0.4, 0.1, 3.6),
                (0.45, 0.1, 3.5),
            ]
        )
    )
    point_cell_ids = np.asarray([1, 0, 0, 0, 0, 0], dtype="<i8")
    point_cell_ids.setflags(write=False)
    summaries = {
        0: v4s._CellSummary(
            0,
            0,
            0,
            5,
            (3.5, 3.6, 3.7, 3.8, 4.0),
            3.8,
            4.0,
            4.0,
            (0.25, 0.25),
            20.0,
            1,
        ),
        1: v4s._CellSummary(1, 1, 0, 1, (4.0,), 4.0, 4.0, 4.0, (0.75, 0.25), 4.0, 0),
    }
    grid = v4s._Grid(
        0.5,
        (0.0, 0.0),
        2,
        1,
        summaries,
        {0: [0], 1: [1]},
        point_cell_ids,
        (0, 1),
        {0: [1, 2, 3, 4, 5], 1: [0]},
    )
    monkeypatch.setattr(v4s, "_smoothed_chm_map", lambda _grid: {0: 4.0, 1: 4.0})
    monkeypatch.setattr(v4s, "_local_max_plateaus", lambda _grid, _height: [(0, 1)])
    monkeypatch.setattr(v4s, "_annulus_median_height", lambda *_args: 2.0)
    monkeypatch.setattr(v4s, "_apex_neighborhood_median_density", lambda *_args: 10.0)
    records = v4s._detect_apex_candidates(
        population,
        grid,
        v4s.derive_algorithm_scales(make_config()),
        make_config(),
    )
    assert records[0].supporting_original_index == 0
    assert records[0].density_points_per_m2 == 4.0
    assert records[0].apex_support_score == pytest.approx(0.76)


def test_missing_prominence_apex_not_accepted() -> None:
    """V4A-IR-003: an edge-limited apex with missing prominence is suppressed."""
    from algorithms.vegetation.v4_tree_segmentation import (
        _build_grid,
        _canonical_population,
        _detect_apex_candidates,
        derive_algorithm_scales,
    )

    # A small isolated dome: the La..2La annulus contains no occupied cells, so
    # prominence is NOT_AVAILABLE and the apex cannot be formally accepted.
    points: list[tuple[float, float, float]] = []
    for radius in (0.1, 0.25, 0.4):
        for slot in range(8):
            theta = 2.0 * math.pi * slot / 8
            points.append(
                (2.0 + radius * math.cos(theta), 2.0 + radius * math.sin(theta), 4.0 - radius)
            )
    points.append((2.0, 2.0, 4.0))
    handoff = make_handoff(points, bounds=((0.0, 0.0, 0.0), (4.0, 4.0, 5.0)))
    config = make_config()
    population = _canonical_population(handoff)
    scales = derive_algorithm_scales(config)
    grid = _build_grid(population, scales, config)
    records = _detect_apex_candidates(population, grid, scales, config)
    assert records
    for record in records:
        assert record.quality_flags & int(V4ApexQualityFlag.SUPPRESSED)
        assert record.quality_flags & int(V4ApexQualityFlag.PROMINENCE_MISSING)


def test_final_core_occupancy_counts_only_full_bins() -> None:
    """V4A-IR-004/015 #5: only core bins with >=3 points count toward TC."""
    from algorithms.vegetation.v4_tree_segmentation import (
        _build_grid,
        _canonical_population,
        _detect_trunk_candidates,
        derive_algorithm_scales,
    )

    # A vertical trunk column at x=2 (bins 0,1,3 with 6 points each) plus a
    # bin-2 voxel offset ~0.9 m from the fitted axis.  The offset voxel keeps
    # bin 2 inside the raw component, but its points fall outside the core
    # radius, so bin 2 must NOT count toward TC.
    points: list[tuple[float, float, float]] = []
    for _k, hag in enumerate((0.3, 0.8, 1.8)):
        for dup in range(6):
            points.append((2.0 + 0.02 * dup, 2.0 - 0.02 * dup, hag))
    for _dup in range(3):
        points.append((2.9, 2.0, 1.3))
    handoff = make_handoff(points)
    config = make_config()
    population = _canonical_population(handoff)
    scales = derive_algorithm_scales(config)
    grid = _build_grid(population, scales, config)
    records, _lookup = _detect_trunk_candidates(population, grid, scales, config)
    accepted = [
        record
        for record in records
        if record.candidate_state is V4TrunkCandidateState.ACCEPTED
    ]
    assert accepted, "expected an accepted trunk candidate"
    trunk = accepted[0]
    # Bin 2 (sparse core) does not count toward TC; evaluated spans all 4 bins.
    assert trunk.occupied_required_bins == 3
    assert trunk.evaluated_bins == 4
    assert abs(trunk.tc - 0.75) < 1e-9


def test_tvc_per_bin_connected_chain() -> None:
    """V4A-IR-005/015 #6: TVc_bins counts only chain-connected intermediate bins."""
    from types import SimpleNamespace

    from algorithms.vegetation.v4_tree_segmentation import (
        _vertical_continuity,
        _Voxel,
        derive_algorithm_scales,
    )
    from core.contracts.v4_tree_segmentation import V4ApexCandidateRecord, V4TrunkCandidateRecord

    grid = SimpleNamespace(cols=10, rows=10)
    scales = derive_algorithm_scales(make_config())

    def voxel(cell_id: int, bin_id: int, xyz: tuple[float, float, float]) -> _Voxel:
        return _Voxel(
            cell_id=cell_id,
            hag_bin_id=bin_id,
            positions=(),
            centroid_xyz=xyz,
            density=10.0,
            min_original_index=cell_id,
        )

    corridor = {
        (11, 0): voxel(11, 0, (2.0, 2.0, 0.25)),
        (11, 1): voxel(11, 1, (2.0, 2.0, 0.75)),
        (33, 2): voxel(33, 2, (3.0, 3.0, 1.25)),
        (33, 3): voxel(33, 3, (3.0, 3.0, 1.75)),
    }
    trunk = V4TrunkCandidateRecord.__new__(V4TrunkCandidateRecord)
    object.__setattr__(trunk, "core_hag_max_m", 0.3)
    apex = V4ApexCandidateRecord.__new__(V4ApexCandidateRecord)
    object.__setattr__(apex, "hag_m", 2.1)
    tvc_bins, tg, tvc_available, tg_available, _bins = _vertical_continuity(
        trunk, apex, corridor, grid, scales  # type: ignore[arg-type]
    )
    assert tvc_available is True
    # Bin 1 is chain-connected from bin 0; bin 2 is not connected (far XY);
    # bin 3 cannot bridge the broken chain.
    assert abs(tvc_bins - 1.0 / 3.0) < 1e-12
    assert abs(tg - near(2.0 * scales.btr, scales.ld)) < 1e-12


def test_trunk_relation_total_lookup_out_of_radius() -> None:
    """V4A-IR-005/015 #7: OBSERVED_OUT_OF_RADIUS is observed exclusivity."""
    from algorithms.vegetation.v4_tree_segmentation import (
        _trunk_relation_evidence,
        derive_algorithm_scales,
    )

    # Two apices 6 m apart (Lr=4) with one distinct trunk each, associated
    # strongly; the cross pairs are geometrically out of radius.
    trunk_a = _trunk_record("trunk-00000000", (2.0, 2.0, 1.0))
    trunk_b = _trunk_record("trunk-00000001", (8.0, 2.0, 1.0))
    apex_a = _apex_record("apex-00000000", (2.0, 2.0, 5.0))
    apex_b = _apex_record("apex-00000001", (8.0, 2.0, 5.0))
    assoc_a = _association_record("ta-0", trunk_a, apex_a, 1.0)
    assoc_b = _association_record("ta-1", trunk_b, apex_b, 1.0)
    evidence = _trunk_relation_evidence(
        [apex_a, apex_b],
        [assoc_a, assoc_b],
        [trunk_a, trunk_b],
        derive_algorithm_scales(make_config()),
    )
    pair = tuple(sorted(("hyp-apex-00000000", "hyp-apex-00000001")))
    shared, distinct = evidence[pair]
    assert shared == 0.0
    assert abs(distinct - 1.0) < 1e-12


def test_trunk_relation_lookup_not_available_in_radius() -> None:
    """V4A-IR-005/015 #7: in-radius NOT_AVAILABLE never creates support."""
    from algorithms.vegetation.v4_tree_segmentation import (
        _trunk_relation_evidence,
        derive_algorithm_scales,
    )

    trunk = _trunk_record("trunk-00000000", (2.0, 2.0, 1.0))
    apex_a = _apex_record("apex-00000000", (2.0, 2.0, 5.0))
    apex_b = _apex_record("apex-00000001", (3.0, 2.0, 5.0))  # in radius
    assoc_a = _association_record("ta-0", trunk, apex_a, 1.0)
    assoc_b = _association_record("ta-1", trunk, apex_b, 1.0, score_available=False)
    evidence = _trunk_relation_evidence(
        [apex_a, apex_b],
        [assoc_a, assoc_b],
        [trunk],
        derive_algorithm_scales(make_config()),
    )
    pair = tuple(sorted(("hyp-apex-00000000", "hyp-apex-00000001")))
    shared, distinct = evidence[pair]
    # NOT_AVAILABLE is not 0-score and not exclusivity: shared and distinct
    # both receive no contribution.
    assert shared == 0.0
    assert distinct == 0.0


def test_responsibility_traversal_independence() -> None:
    """V4A-IR-008/015 #13: forward/reverse/shuffled orders give identical claims."""

    scene = two_trees_scene()
    handoff = make_handoff(scene)
    config = make_config()
    result = segment_v4_trees(handoff, config)
    apices = list(result.apex_records)
    # Run the solver directly with forward, reversed, and shuffled apex order.
    forward = _solver_claims(apices, handoff, config)
    reversed_claims = _solver_claims(list(reversed(apices)), handoff, config)
    shuffled = _solver_claims(apices[1:] + apices[:1], handoff, config)
    for cell_id, claims in forward.items():
        rev = reversed_claims.get(cell_id, [])
        shuf = shuffled.get(cell_id, [])
        assert [c.score for c in claims] == [c.score for c in rev]
        assert [c.hypothesis_id for c in claims] == [c.hypothesis_id for c in rev]
        assert [c.score for c in claims] == [c.score for c in shuf]


def test_early_merge_final_keep_separate() -> None:
    """V4A-IR-015 #14: early PROVISIONAL_MERGE + final KEEP_SEPARATE -> two trees."""
    from algorithms.vegetation.v4_tree_segmentation import (
        _early_relation,
        _final_relation,
        derive_algorithm_scales,
    )

    scales = derive_algorithm_scales(make_config())
    # Two close co-crown apices: an early MERGE seed with a measured saddle.
    apex_a = _apex_record("apex-00000000", (2.0, 2.0, 5.0))
    apex_b = _apex_record("apex-00000001", (2.05, 2.0, 4.95))
    early = _early_relation(apex_a, apex_b, scales, 1.0, 0.0, True, 0.0)
    # Later strong distinct-trunk evidence flips the final decision.
    final = _final_relation(
        apex_a, apex_b, scales, 0.0, 1.0, False, 0.0, True, 5.0, 0.0, True, None
    )
    assert "PROVISIONAL_MERGE" in early.stage
    assert final.relation is V4TreeRelation.KEEP_SEPARATE


def test_profile_not_available_and_observed_empty() -> None:
    """V4A-IR-007/015 #10/#11: real NOT_AVAILABLE and OBSERVED_EMPTY samples."""
    from algorithms.vegetation.v4_tree_segmentation import (
        _build_grid,
        _canonical_population,
        _ray_band_inventory,
        _rotating_profile,
        derive_algorithm_scales,
    )

    handoff = make_handoff(two_trees_scene())
    config = make_config()
    population = _canonical_population(handoff)
    scales = derive_algorithm_scales(config)
    grid = _build_grid(population, scales, config)
    apices = _accepted_apices(handoff, config)
    assert apices
    states_seen = set()
    for apex in apices:
        profile = _rotating_profile(apex, grid, scales, config)
        for ray in profile.rays:
            for sample in ray.samples:
                states_seen.add(sample.state)
    # The pipeline produces both measured sample states.
    assert V4ProfileSampleState.OCCUPIED in states_seen
    assert V4ProfileSampleState.OBSERVED_EMPTY in states_seen
    # A radial band whose local cell inventory extends beyond the grid is typed
    # NOT_AVAILABLE (unknown), never empty.
    apex = apices[0]
    occupied, _densities, _support, incomplete = _ray_band_inventory(
        grid,
        apex,
        (-1.0, 0.0),
        scales.lr * 2.0,
        scales.lr * 2.0 + scales.lxy,
        scales,
    )
    assert incomplete is True
    assert occupied == []


def test_resource_ceilings_fail_closed() -> None:
    """V4A-IR-014/015 #22: tiny fragment/edge/profile ceilings hard-fail."""
    from core.contracts.v4_tree_segmentation import V4TreeSegmentationError

    scene = two_trees_scene()
    with pytest.raises(V4TreeSegmentationError):
        segment_v4_trees(make_handoff(scene), make_config(maximum_fragments=1))
    with pytest.raises(V4TreeSegmentationError):
        segment_v4_trees(make_handoff(scene), make_config(maximum_local_edges=1))
    with pytest.raises(V4TreeSegmentationError):
        segment_v4_trees(make_handoff(scene), make_config(maximum_profile_samples=1))


def test_assignment_channels_affect_final_support() -> None:
    """V4A-IR-011/015 #17: slice/gap/structure channels change final support."""
    from algorithms.vegetation.v4_tree_segmentation import assignment_support_score

    base = (-1.0,) * 8 + (1.0,)  # slice=-1, continuity=-1, ..., quality=+1
    base_support = assignment_support_score(base)
    # Flip slice measured support to +1.
    plus_slice = (1.0,) + base[1:]
    assert assignment_support_score(plus_slice) > base_support
    # Flip structure direction to +1.
    plus_structure = list(base)
    plus_structure[5] = 1.0
    assert assignment_support_score(tuple(plus_structure)) > base_support
    # Flip gap to a measured positive (+1 = no gap).
    plus_gap = list(base)
    plus_gap[6] = 1.0
    assert assignment_support_score(tuple(plus_gap)) > base_support


def test_structural_path_upper_core_source_and_tie() -> None:
    """V4A-IR-006/015 #8/#9: upper-core source + apex-neighbourhood target and
    equal-cost insertion-order determinism."""
    from types import SimpleNamespace

    from algorithms.vegetation.v4_tree_segmentation import (
        V4StructuralPathAvailability,
        _canonical_population,
        _structural_path,
        _Voxel,
        derive_algorithm_scales,
    )

    grid = SimpleNamespace(cols=10, rows=10)
    scales = derive_algorithm_scales(make_config())

    def voxel(cell_id: int, bin_id: int, xyz: tuple[float, float, float], index: int) -> _Voxel:
        return _Voxel(
            cell_id=cell_id,
            hag_bin_id=bin_id,
            positions=(index,),
            centroid_xyz=xyz,
            density=10.0,
            min_original_index=index,
        )

    source = voxel(11, 1, (2.0, 2.0, 0.75), 0)
    node_x = voxel(12, 2, (2.5, 2.0, 1.25), 1)
    node_y = voxel(21, 2, (2.0, 2.5, 1.25), 2)
    target = voxel(22, 3, (2.5, 2.5, 1.75), 3)
    # Same-bin corridor support that is not a trunk-core member must never be
    # admitted as a structural-path source, even when it is already inside the
    # apex target neighbourhood.
    non_core_decoy = voxel(33, 1, (3.0, 3.0, 1.0), 1)
    corridor_forward = {
        (11, 1): source,
        (12, 2): node_x,
        (21, 2): node_y,
        (22, 3): target,
        (33, 1): non_core_decoy,
    }
    corridor_reversed = {
        (33, 1): non_core_decoy,
        (22, 3): target,
        (21, 2): node_y,
        (12, 2): node_x,
        (11, 1): source,
    }
    handoff = make_handoff(
        [
            (2.0, 2.0, 0.75),
            (2.5, 2.0, 1.25),
            (2.0, 2.5, 1.25),
            (2.5, 2.5, 1.75),
            (2.0, 2.0, 0.25),
        ]
    )
    population = _canonical_population(handoff)
    trunk = _trunk_record("trunk-00000000", (2.0, 2.0, 1.0))
    apex = _apex_record("apex-00000000", (3.0, 3.0, 1.0))
    # Core bins {0,1} -> upper-core half = {1}; only the source is eligible.
    trunk_core_positions = (0, 4)
    evidence_fwd, _ = _structural_path(
        trunk, apex, trunk_core_positions, corridor_forward,
        {}, population, grid, scales, 50000, 50000,
    )
    evidence_rev, _ = _structural_path(
        trunk, apex, trunk_core_positions, corridor_reversed,
        {}, population, grid, scales, 50000, 50000,
    )
    assert evidence_fwd.availability is V4StructuralPathAvailability.OBSERVED
    assert evidence_fwd.path_node_key_sequence == evidence_rev.path_node_key_sequence
    # Only the upper-core source starts; only the apex-neighbourhood target is
    # reached, and the equal-cost tie is resolved canonically (X before Y).
    assert evidence_fwd.source_node_key == "11:1:0"
    assert evidence_fwd.target_node_key == "22:3:3"
    assert evidence_fwd.path_node_key_sequence == ("11:1:0", "12:2:1", "22:3:3")


def test_structural_path_equal_cost_uses_predecessor_key(monkeypatch) -> None:
    """V4A-IR-006: an equal total-cost/hop path replaces a previously found
    path when its predecessor node key is canonically smaller."""

    from types import SimpleNamespace

    from algorithms.vegetation import v4_tree_segmentation as v4s

    def voxel(cell_id: int, bin_id: int, xyz: tuple[float, float, float], index: int):
        return v4s._Voxel(
            cell_id=cell_id,
            hag_bin_id=bin_id,
            positions=(index,),
            centroid_xyz=xyz,
            density=10.0,
            min_original_index=index,
        )

    source = voxel(11, 1, (0.0, 0.0, 0.0), 0)
    pred_a = voxel(12, 2, (1.0, 0.0, 0.0), 1)
    pred_b = voxel(21, 2, (0.0, 1.0, 0.0), 2)
    target = voxel(22, 3, (1.0, 1.0, 0.0), 3)
    corridor = {
        (11, 1): source,
        (12, 2): pred_a,
        (21, 2): pred_b,
        (22, 3): target,
    }
    adjacency = {
        (11, 1): [(12, 2), (21, 2)],
        (12, 2): [(11, 1), (22, 3)],
        (21, 2): [(11, 1), (22, 3)],
        (22, 3): [(12, 2), (21, 2)],
    }
    costs = {
        frozenset(((11, 1), (12, 2))): 2.0,
        frozenset(((12, 2), (22, 3))): 1.0,
        frozenset(((11, 1), (21, 2))): 1.0,
        frozenset(((21, 2), (22, 3))): 2.0,
    }
    key_by_identity = {id(value): key for key, value in corridor.items()}
    monkeypatch.setattr(v4s, "_local_adjacency", lambda *_args: adjacency)
    monkeypatch.setattr(
        v4s,
        "_voxel_edge_cost",
        lambda left, right, _scales: costs[
            frozenset((key_by_identity[id(left)], key_by_identity[id(right)]))
        ],
    )
    handoff = make_handoff(
        [(0.0, 0.0, 0.5), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (2.0, 2.0, 1.5)]
    )
    population = v4s._canonical_population(handoff)
    trunk = _trunk_record("trunk-00000000", (0.0, 0.0, 1.0))
    apex = _apex_record("apex-00000000", (1.0, 1.0, 1.5))
    evidence, _complete = v4s._structural_path(
        trunk,
        apex,
        (0,),
        corridor,
        corridor,
        population,
        SimpleNamespace(cols=10, rows=10),
        v4s.derive_algorithm_scales(make_config(apex_neighborhood_radius_m=0.4)),
        100,
        100,
    )
    assert evidence.path_node_key_sequence == (
        "11:1:0",
        "12:2:1",
        "22:3:3",
    )


def test_structural_path_near_connected_component_has_full_efficiency() -> None:
    """V4A-IR-006: a measured reachable anchor within Ls uses the frozen
    direct-component efficiency of exactly one."""

    from types import SimpleNamespace

    from algorithms.vegetation import v4_tree_segmentation as v4s

    handoff = make_handoff([(0.0, 0.0, 0.5)])
    population = v4s._canonical_population(handoff)
    voxel = v4s._Voxel(
        cell_id=0,
        hag_bin_id=1,
        positions=(0,),
        centroid_xyz=(0.0, 0.0, 0.5),
        density=4.0,
        min_original_index=0,
    )
    trunk = _trunk_record("trunk-00000000", (0.0, 0.0, 0.5))
    apex = _apex_record("apex-00000000", (0.0, 0.0, 1.0))
    evidence, _complete = v4s._structural_path(
        trunk,
        apex,
        (0,),
        {(0, 1): voxel},
        {(0, 1): voxel},
        population,
        SimpleNamespace(cols=1, rows=1),
        v4s.derive_algorithm_scales(make_config()),
        10,
        10,
    )
    assert evidence.structural_path_efficiency == 1.0


def test_adjacent_fragment_graph_revalidates_early_path() -> None:
    """V4A-IR-009: a measured break in the adjacent-slice fragment graph
    downgrades an otherwise strong early CSP association to unsupported."""

    from algorithms.vegetation import v4_tree_segmentation as v4s
    from core.contracts.v4_tree_segmentation import (
        V4StructuralPathAvailability,
        V4StructuralPathEvidence,
    )

    handoff = make_handoff(
        [
            (0.0, 0.0, 0.5),
            (0.2, 0.0, 0.5),
            (0.1, 0.0, 1.0),
            (0.3, 0.0, 1.0),
            (10.0, 0.0, 1.5),
            (10.2, 0.0, 1.5),
        ]
    )
    population = v4s._canonical_population(handoff)
    hypothesis_id = "hyp-apex-00000004"
    fragments = [
        v4s._Fragment("frag-0", 0, hypothesis_id, (0, 1), "0" * 64),
        v4s._Fragment("frag-1", 1, hypothesis_id, (2, 3), "1" * 64),
        v4s._Fragment("frag-2", 2, hypothesis_id, (4, 5), "2" * 64),
    ]
    config = make_config()
    scales = v4s.derive_algorithm_scales(config)
    adjacency, continuity, available = v4s._build_adjacent_fragment_graph(
        fragments, population, scales, config
    )
    assert adjacency["frag-0"] == ("frag-1",)
    assert adjacency["frag-2"] == ()
    assert available[hypothesis_id] is True
    assert continuity[hypothesis_id] == 0.5
    _single_adjacency, single_continuity, single_available = (
        v4s._build_adjacent_fragment_graph(fragments[:1], population, scales, config)
    )
    assert single_available[hypothesis_id] is True
    assert single_continuity[hypothesis_id] == 0.5

    trunk = _trunk_record("trunk-00000000", (0.0, 0.0, 0.5))
    apex = v4s._finalize_record(
        V4ApexCandidateRecord,
        "diagnostic_digest",
        candidate_id="apex-00000004",
        supporting_original_index=4,
        xyz_local_m=(10.0, 0.0, 1.5),
        hag_m=1.5,
        prominence_m=1.0,
        prominence_available=True,
        plateau_support_count=3,
        cell_point_count=3,
        density_points_per_m2=10.0,
        neighborhood_median_density_points_per_m2=10.0,
        apex_support_score=0.8,
        quality_flags=0,
        reason_flags=0,
    )
    path = v4s._finalize_record(
        V4StructuralPathEvidence,
        "evidence_digest",
        path_id="path-0-4",
        trunk_candidate_id=trunk.candidate_id,
        apex_candidate_id=apex.candidate_id,
        domain_bounds_min_xyz=(0.0, 0.0, 0.0),
        domain_bounds_max_xyz=(10.2, 0.0, 1.5),
        node_count=3,
        edge_count=2,
        source_node_key="source",
        target_node_key="target",
        path_node_key_sequence=("source", "middle", "target"),
        straight_length_m=10.0,
        weighted_path_length_m=10.5,
        structural_path_efficiency=0.95,
        availability=V4StructuralPathAvailability.OBSERVED,
        resource_audit="test",
        quality_flags=0,
        reason_flags=0,
    )
    association = _association_record("ta-0-4", trunk, apex, 0.9)
    association_fields = {
        name: getattr(association, name)
        for name in V4TrunkApexAssociationEvidence.__dataclass_fields__
        if name != "diagnostic_digest"
    }
    association_fields["structural_path_evidence_reference"] = path.path_id
    association = v4s._finalize_record(
        V4TrunkApexAssociationEvidence,
        "diagnostic_digest",
        **association_fields,
    )
    revalidated = v4s._revalidate_associations_with_fragment_graph(
        [association],
        {association.association_id: path},
        fragments,
        adjacency,
        {trunk.candidate_id: (0, (0, 1))},
        {apex.candidate_id: apex},
        population,
        scales,
    )
    assert (
        revalidated[0].association_state
        is V4TrunkApexAssociationState.UNSUPPORTED_ASSOCIATION
    )
    assert revalidated[0].association_score == 0.0


def test_profile_driven_responsibility_stop() -> None:
    """V4A-IR-007/008/015 #12: profile_confirmed_stop is consumed by claims."""
    from algorithms.vegetation.v4_tree_segmentation import (
        _build_grid,
        _canonical_population,
        _hypothesis_claims,
        _ProfileStopInfo,
        derive_algorithm_scales,
    )

    handoff = make_handoff(two_trees_scene())
    config = make_config()
    population = _canonical_population(handoff)
    scales = derive_algorithm_scales(config)
    grid = _build_grid(population, scales, config)
    apices = _accepted_apices(handoff, config)
    assert apices
    apex = apices[0]
    cell_summaries = grid.cell_id_to_summary
    full = _hypothesis_claims(apex, "hyp-test", grid, cell_summaries, scales, config)
    assert full
    # A measured gap at radius 0 stops every claim beyond the apex cell.
    stop = _ProfileStopInfo(
        boundary_radius_by_ray={},
        gap_radius_by_ray={ray: 0.0 for ray in range(24)},
    )
    stopped = _hypothesis_claims(
        apex, "hyp-test", grid, cell_summaries, scales, config, profile_stop=stop
    )
    assert not stopped


def test_final_union_preserves_review_edge() -> None:
    """V4A-IR-010/015 #15: a final REVIEW edge is preserved as a conflict."""
    from algorithms.vegetation.v4_tree_segmentation import (
        _final_relation,
        _final_union,
        _hypothesis_key,
        derive_algorithm_scales,
    )
    from core.contracts.v4_tree_segmentation import V4TreeRelation

    scales = derive_algorithm_scales(make_config())
    apex_a = _apex_record("apex-00000000", (2.0, 2.0, 5.0))
    apex_b = _apex_record("apex-00000001", (4.0, 2.0, 5.5))
    # Neutral-ish channels keep both supports below threshold -> REVIEW.
    relation = _final_relation(
        apex_a, apex_b, scales, 0.0, 0.0, True, 0.0, True, 0.5, 0.0, True, 0.5
    )
    assert relation.relation is V4TreeRelation.REVIEW
    components, states, conflict_edges = _final_union([apex_a, apex_b], [relation])
    key_a = _hypothesis_key(apex_a)
    key_b = _hypothesis_key(apex_b)
    # Both stay ACTIVE but the REVIEW edge is preserved for assignment.
    assert states[key_a] is V4TreeHypothesisState.ACTIVE
    assert states[key_b] is V4TreeHypothesisState.ACTIVE
    assert components[key_a] != components[key_b]
    assert tuple(sorted((key_a, key_b))) in conflict_edges


def test_final_review_point_propagation() -> None:
    """V4A-IR-010/015 #16: REVIEW conflict points become REVIEW_REQUIRED."""
    from algorithms.vegetation.v4_tree_segmentation import (
        _assign_points,
        _build_grid,
        _canonical_population,
        _hypothesis_key,
        _responsibility_domains,
        _responsibility_solver,
        derive_algorithm_scales,
    )
    from core.contracts.v4_tree_segmentation import V4TreeHypothesisState

    handoff = make_handoff(make_tree_points(2.0, 2.0) + make_tree_points(4.0, 2.0))
    config = make_config()
    population = _canonical_population(handoff)
    scales = derive_algorithm_scales(config)
    grid = _build_grid(population, scales, config)
    apices = _accepted_apices(handoff, config)
    assert len(apices) >= 2
    claims = _responsibility_solver(apices, grid, scales, config)
    domains = _responsibility_domains(claims)
    trees = [(i, [apex], V4TreeHypothesisState.ACTIVE) for i, apex in enumerate(apices)]
    key_a = _hypothesis_key(apices[0])
    key_b = _hypothesis_key(apices[1])
    conflict = {(key_a, key_b)}
    both_cells = [
        cell_id
        for cell_id, cell_claims in claims.items()
        if {claim.hypothesis_id for claim in cell_claims} >= {key_a, key_b}
    ]
    assert both_cells, "overlapping crowns must share claimed cells"
    state, _tid, _conf, _reason = _assign_points(
        population,
        grid,
        trees,
        domains,
        claims,
        (),
        scales,
        config,
        conflict_edges=conflict,
    )
    for position in range(int(population.original_index.shape[0])):
        cell_id = int(grid.point_cell_ids[position])
        if cell_id in both_cells:
            assert state[position] == int(V4TreeAssignmentState.REVIEW_REQUIRED)
    exclusive_any_normal = any(
        state[position] != int(V4TreeAssignmentState.REVIEW_REQUIRED)
        for position in range(int(population.original_index.shape[0]))
        if int(grid.point_cell_ids[position]) not in both_cells
    )
    assert exclusive_any_normal


# Internal helpers shared by the review-fix tests.


def _trunk_record(candidate_id: str, xyz: tuple[float, float, float]) -> V4TrunkCandidateRecord:
    from algorithms.vegetation.v4_tree_segmentation import _finalize_record
    from core.contracts.v4_tree_segmentation import V4TrunkCandidateState

    return _finalize_record(
        V4TrunkCandidateRecord,
        "diagnostic_digest",
        candidate_id=candidate_id,
        representative_original_index=0,
        supporting_original_index_digest="0" * 64,
        centroid_xyz=xyz,
        bounds_min_xyz=(xyz[0] - 0.5, xyz[1] - 0.5, 0.0),
        bounds_max_xyz=(xyz[0] + 0.5, xyz[1] + 0.5, 3.0),
        refit_principal_axis=(0.0, 0.0, 1.0),
        core_hag_min_m=0.5,
        core_hag_max_m=3.0,
        observed_vertical_span_m=2.5,
        k_min=1,
        k_max=6,
        evaluated_bins=6,
        occupied_required_bins=6,
        max_bin_centroid_xy_drift_m=0.1,
        tv=1.0,
        ta=1.0,
        ts=0.9,
        tc=1.0,
        trunk_candidate_score=0.98,
        candidate_state=V4TrunkCandidateState.ACCEPTED,
        quality_flags=0,
        reason_flags=0,
    )


def _apex_record(candidate_id: str, xyz: tuple[float, float, float]) -> V4ApexCandidateRecord:
    from algorithms.vegetation.v4_tree_segmentation import _finalize_record

    return _finalize_record(
        V4ApexCandidateRecord,
        "diagnostic_digest",
        candidate_id=candidate_id,
        supporting_original_index=0,
        xyz_local_m=xyz,
        hag_m=xyz[2],
        prominence_m=2.0,
        prominence_available=True,
        plateau_support_count=5,
        cell_point_count=5,
        density_points_per_m2=20.0,
        neighborhood_median_density_points_per_m2=10.0,
        apex_support_score=0.8,
        quality_flags=0,
        reason_flags=0,
    )


def _association_record(
    association_id: str,
    trunk: V4TrunkCandidateRecord,
    apex: V4ApexCandidateRecord,
    score: float,
    score_available: bool = True,
) -> V4TrunkApexAssociationEvidence:
    from algorithms.vegetation.v4_tree_segmentation import _finalize_record

    return _finalize_record(
        V4TrunkApexAssociationEvidence,
        "diagnostic_digest",
        association_id=association_id,
        trunk_candidate_id=trunk.candidate_id,
        apex_candidate_id=apex.candidate_id,
        tx=1.0,
        tx_available=score_available,
        td=1.0,
        td_available=score_available,
        tvc=1.0,
        tvc_available=score_available,
        tg=1.0,
        tg_available=score_available,
        tr=1.0,
        tr_available=score_available,
        association_score_available=score_available,
        association_score=score if score_available else 0.0,
        association_state=V4TrunkApexAssociationState.STRONG_ASSOCIATION,
        structural_path_evidence_reference=None,
        lookup_kind=V4AssociationLookupKind.EVALUATED,
        quality_flags=0,
        reason_flags=0,
    )


def _accepted_apices(handoff, config):
    from algorithms.vegetation.v4_tree_segmentation import (
        _build_grid,
        _canonical_population,
        _detect_apex_candidates,
        derive_algorithm_scales,
    )

    population = _canonical_population(handoff)
    scales = derive_algorithm_scales(config)
    grid = _build_grid(population, scales, config)
    records = _detect_apex_candidates(population, grid, scales, config)
    return [
        record
        for record in records
        if not (record.quality_flags & int(V4ApexQualityFlag.SUPPRESSED))
    ]


def _solver_claims(apices, handoff, config):
    from algorithms.vegetation.v4_tree_segmentation import (
        _build_grid,
        _canonical_population,
        _responsibility_solver,
        derive_algorithm_scales,
    )

    population = _canonical_population(handoff)
    scales = derive_algorithm_scales(config)
    grid = _build_grid(population, scales, config)
    return _responsibility_solver(apices, grid, scales, config)


# ---------------------------------------------------------------------------
# REVIEWED_FIXED_POLICY pre-algorithm authority guard
# ---------------------------------------------------------------------------


def _artifact_with_purpose(purpose: str):
    from core.contracts.frv import (
        FRVArtifactBlockDescriptor,
        FRVArtifactPayloadSchema,
        FullResolutionVegetationMembershipArtifact,
    )

    from tests.contracts.test_frv_contracts import artifact_dict, block_dict, schema_dict

    payload_schema = FRVArtifactPayloadSchema.from_dict(schema_dict())
    block = FRVArtifactBlockDescriptor.from_dict(block_dict(payload_schema))
    return FullResolutionVegetationMembershipArtifact.from_dict(
        artifact_dict([block], payload_schema, configuration_purpose=purpose)
    )


def test_authority_guard_reviewed_upstream_reviewed_status_allowed() -> None:
    from core.contracts.frv import FrvConfigurationPurpose

    artifact = _artifact_with_purpose(FrvConfigurationPurpose.REVIEWED_FIXED_POLICY.value)
    handoff = make_handoff(two_trees_scene(), membership_artifact=artifact)
    config = make_config(status=V4ConfigStatus.REVIEWED_FIXED_POLICY)
    result = segment_v4_trees(handoff, config)
    assert result.tree_records
    assert result.audit.total_points > 0


def test_authority_guard_reviewed_upstream_production_status_rejected() -> None:
    from core.contracts.frv import FrvConfigurationPurpose
    from core.contracts.v4_tree_segmentation import V4TreeSegmentationError

    artifact = _artifact_with_purpose(FrvConfigurationPurpose.REVIEWED_FIXED_POLICY.value)
    handoff = make_handoff(two_trees_scene(), membership_artifact=artifact)
    config = make_config(status=V4ConfigStatus.PRODUCTION_APPROVED)
    with pytest.raises(V4TreeSegmentationError):
        segment_v4_trees(handoff, config)


def test_authority_guard_production_upstream_production_status_allowed() -> None:
    from core.contracts.frv import FrvConfigurationPurpose

    artifact = _artifact_with_purpose(FrvConfigurationPurpose.PRODUCTION_APPROVED.value)
    handoff = make_handoff(two_trees_scene(), membership_artifact=artifact)
    config = make_config(status=V4ConfigStatus.PRODUCTION_APPROVED)
    result = segment_v4_trees(handoff, config)
    assert result.tree_records


def test_authority_guard_missing_upstream_rejected_for_formal_v4() -> None:
    from core.contracts.v4_tree_segmentation import V4TreeSegmentationError

    # The synthetic handoff carries no embedded membership artifact: the
    # upstream authority is missing, so any formal V4 status fails closed.
    handoff = make_handoff(two_trees_scene())
    for status in (
        V4ConfigStatus.REVIEWED_FIXED_POLICY,
        V4ConfigStatus.PRODUCTION_APPROVED,
    ):
        with pytest.raises(V4TreeSegmentationError):
            segment_v4_trees(handoff, make_config(status=status))


def test_authority_guard_nonformal_upstream_rejected_for_formal_v4() -> None:
    from core.contracts.frv import FrvConfigurationPurpose
    from core.contracts.v4_tree_segmentation import V4TreeSegmentationError

    for purpose in (
        FrvConfigurationPurpose.TEST_ONLY,
        FrvConfigurationPurpose.CALIBRATION_CANDIDATE,
    ):
        artifact = _artifact_with_purpose(purpose.value)
        handoff = make_handoff(two_trees_scene(), membership_artifact=artifact)
        for status in (
            V4ConfigStatus.REVIEWED_FIXED_POLICY,
            V4ConfigStatus.PRODUCTION_APPROVED,
        ):
            with pytest.raises(V4TreeSegmentationError):
                segment_v4_trees(handoff, make_config(status=status))


def test_authority_guard_does_not_change_numerical_output() -> None:
    """The authority gate leaves Algorithm-V1 numerical output unchanged.

    Two otherwise equivalent valid executions that reach the computational
    core (reviewed and production) produce the same segmentation result.
    """
    from core.contracts.frv import FrvConfigurationPurpose

    scene = two_trees_scene()
    reviewed_artifact = _artifact_with_purpose(
        FrvConfigurationPurpose.REVIEWED_FIXED_POLICY.value
    )
    production_artifact = _artifact_with_purpose(
        FrvConfigurationPurpose.PRODUCTION_APPROVED.value
    )
    reviewed = segment_v4_trees(
        make_handoff(scene, membership_artifact=reviewed_artifact),
        make_config(status=V4ConfigStatus.REVIEWED_FIXED_POLICY),
    )
    production = segment_v4_trees(
        make_handoff(scene, membership_artifact=production_artifact),
        make_config(status=V4ConfigStatus.PRODUCTION_APPROVED),
    )
    # The numerical segmentation output is identical: same tree IDs, point
    # assignments, members, and conservation.  (The result digest itself binds
    # the input handoff/config digests, which differ by authority.)
    assert np.array_equal(
        reviewed.tree_id_by_point, production.tree_id_by_point
    )
    assert np.array_equal(
        reviewed.assignment_state_by_point, production.assignment_state_by_point
    )
    assert np.array_equal(
        reviewed.ordered_tree_member_original_index,
        production.ordered_tree_member_original_index,
    )
    assert reviewed.audit.total_points == production.audit.total_points
    assert len(reviewed.tree_records) == len(production.tree_records)
    for reviewed_record, production_record in zip(
        reviewed.tree_records, production.tree_records, strict=True
    ):
        assert reviewed_record.tree_id == production_record.tree_id
        assert reviewed_record.member_count == production_record.member_count


def test_authority_guard_production_upstream_reviewed_status_rejected() -> None:
    from core.contracts.frv import FrvConfigurationPurpose
    from core.contracts.v4_tree_segmentation import V4TreeSegmentationError

    # Matrix row E: contradictory upstream/downstream authority fails closed.
    artifact = _artifact_with_purpose(FrvConfigurationPurpose.PRODUCTION_APPROVED.value)
    handoff = make_handoff(two_trees_scene(), membership_artifact=artifact)
    with pytest.raises(V4TreeSegmentationError):
        segment_v4_trees(handoff, make_config(status=V4ConfigStatus.REVIEWED_FIXED_POLICY))
