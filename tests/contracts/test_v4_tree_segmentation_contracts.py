"""Focused V4 tree-segmentation contract tests (PLAN-FROZEN consolidated plan)."""

from __future__ import annotations

import numpy as np
import pytest
from core.contracts.frv import SourceIdentity, frv_digest
from core.contracts.v4_tree_segmentation import (
    ASSIGNMENT_MIN,
    ASSIGNMENT_REVIEW_MARGIN,
    RELATION_MIN,
    RELATION_REVIEW_MARGIN,
    RESPONSIBILITY_MIN,
    RESPONSIBILITY_REVIEW_MARGIN,
    ROTATING_PROFILE_RAY_COUNT,
    TRUNK_APEX_REVIEW_MIN,
    TRUNK_APEX_STRONG_MIN,
    TRUNK_CANDIDATE_MIN,
    TRUNK_WEAK_MIN,
    V4_TREE_SEGMENTATION_ALGORITHM_VERSION,
    V4_TREE_SEGMENTATION_CONFIG_SCHEMA_VERSION,
    V4_TREE_SEGMENTATION_SCHEMA_VERSION,
    V4ApexCandidateRecord,
    V4ApexQualityFlag,
    V4AssociationLookupKind,
    V4ConfigStatus,
    V4ProfileSampleState,
    V4RotatingProfileEvidence,
    V4RotatingProfileRayRecord,
    V4RotatingProfileRaySample,
    V4RotatingProfileState,
    V4SegmentationQualityFlag,
    V4StructuralPathAvailability,
    V4StructuralPathEvidence,
    V4TreeCrownSummary,
    V4TreeGeometrySummary,
    V4TreeHypothesisRecord,
    V4TreeHypothesisState,
    V4TreeInstanceRecord,
    V4TreeRelation,
    V4TreeRelationEvidence,
    V4TreeSegmentationAudit,
    V4TreeSegmentationConfig,
    V4TreeSegmentationErrorCode,
    V4TreeSegmentationResult,
    V4TreeSliceSummary,
    V4TrunkApexAssociationEvidence,
    V4TrunkApexAssociationState,
    V4TrunkCandidateRecord,
    V4TrunkCandidateState,
    V4TrunkEvidenceAvailability,
    v4_member_index_digest,
    validate_v4_tree_segmentation_result,
)

from tests.io.test_source_provider import _source_identity

_SHA = "a" * 64
_SHA_B = "b" * 64


def _ro(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _config(status: V4ConfigStatus = V4ConfigStatus.TEST_ONLY) -> V4TreeSegmentationConfig:
    fields = {
        "schema_version": V4_TREE_SEGMENTATION_CONFIG_SCHEMA_VERSION,
        "status": status,
        "xy_cell_size_m": 0.5,
        "apex_neighborhood_radius_m": 1.0,
        "minimum_apex_prominence_m": 0.5,
        "density_trend_window_m": 1.0,
        "slice_thickness_m": 1.0,
        "slice_overlap_fraction": 0.25,
        "relation_neighborhood_radius_m": 3.0,
        "maximum_points": 100000,
        "maximum_grid_cells": 100000,
        "maximum_fragments": 1000,
        "maximum_local_edges": 10000,
        "maximum_path_evaluations": 100000,
        "maximum_profile_samples": 100000,
    }
    digest_fields = {
        key: value.value if isinstance(value, V4ConfigStatus) else value
        for key, value in fields.items()
    }
    return V4TreeSegmentationConfig(**fields, config_digest=frv_digest(digest_fields))


def _config_from_dict() -> V4TreeSegmentationConfig:
    config = _config()
    payload = config.to_dict()
    return V4TreeSegmentationConfig.from_dict(payload)


def _source() -> SourceIdentity:
    return _source_identity("C:/s.las", _SHA, 4, ([0.0, 0.0, 0.0], [3.0, 3.0, 10.0]))


def _finalize(record_type, digest_field, **fields):
    placeholder = record_type.__new__(record_type)
    for name, value in fields.items():
        object.__setattr__(placeholder, name, value)
    object.__setattr__(placeholder, digest_field, "0" * 64)
    digest = placeholder.computed_digest()
    return record_type(**fields, **{digest_field: digest})


def _apex(candidate_id: str = "apex-00000001", index: int = 1) -> V4ApexCandidateRecord:
    return _finalize(
        V4ApexCandidateRecord,
        "diagnostic_digest",
        candidate_id=candidate_id,
        supporting_original_index=index,
        xyz_local_m=(1.0, 1.0, 6.0),
        hag_m=6.0,
        prominence_m=1.5,
        prominence_available=True,
        plateau_support_count=5,
        cell_point_count=5,
        density_points_per_m2=20.0,
        neighborhood_median_density_points_per_m2=10.0,
        apex_support_score=0.8,
        quality_flags=0,
        reason_flags=0,
    )


def _trunk(candidate_id: str = "trunk-00000002", index: int = 2) -> V4TrunkCandidateRecord:
    return _finalize(
        V4TrunkCandidateRecord,
        "diagnostic_digest",
        candidate_id=candidate_id,
        representative_original_index=index,
        supporting_original_index_digest=_SHA,
        centroid_xyz=(1.0, 1.0, 2.0),
        bounds_min_xyz=(0.9, 0.9, 0.5),
        bounds_max_xyz=(1.1, 1.1, 3.5),
        refit_principal_axis=(0.0, 0.0, 1.0),
        core_hag_min_m=0.5,
        core_hag_max_m=3.5,
        observed_vertical_span_m=3.0,
        k_min=1,
        k_max=7,
        evaluated_bins=7,
        occupied_required_bins=7,
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


def _association() -> V4TrunkApexAssociationEvidence:
    return _finalize(
        V4TrunkApexAssociationEvidence,
        "diagnostic_digest",
        association_id="ta-1",
        trunk_candidate_id="trunk-00000002",
        apex_candidate_id="apex-00000001",
        tx=0.9,
        tx_available=True,
        td=0.9,
        td_available=True,
        tvc=0.9,
        tvc_available=True,
        tg=1.0,
        tg_available=True,
        tr=0.9,
        tr_available=True,
        association_score_available=True,
        association_score=0.92,
        association_state=V4TrunkApexAssociationState.STRONG_ASSOCIATION,
        structural_path_evidence_reference=None,
        lookup_kind=V4AssociationLookupKind.EVALUATED,
        quality_flags=0,
        reason_flags=0,
    )


def _path() -> V4StructuralPathEvidence:
    return _finalize(
        V4StructuralPathEvidence,
        "evidence_digest",
        path_id="path-1",
        trunk_candidate_id="trunk-00000002",
        apex_candidate_id="apex-00000001",
        domain_bounds_min_xyz=(0.5, 0.5, 0.0),
        domain_bounds_max_xyz=(1.5, 1.5, 6.5),
        node_count=12,
        edge_count=14,
        source_node_key="0:1:2",
        target_node_key="1:11:2",
        path_node_key_sequence=("0:1:2", "1:11:2"),
        straight_length_m=4.0,
        weighted_path_length_m=4.2,
        structural_path_efficiency=0.95,
        availability=V4StructuralPathAvailability.OBSERVED,
        resource_audit="corridor=12 edges=14",
        quality_flags=0,
        reason_flags=0,
    )


def _profile() -> V4RotatingProfileEvidence:
    rays = []
    for ray_index in range(ROTATING_PROFILE_RAY_COUNT):
        rays.append(
            V4RotatingProfileRayRecord(
                ray_index=ray_index,
                theta_rad=2.0 * np.pi * ray_index / ROTATING_PROFILE_RAY_COUNT,
                samples=(
                    V4RotatingProfileRaySample(
                        sample_index=0,
                        radial_distance_m=0.0,
                        state=V4ProfileSampleState.OCCUPIED,
                        hag_q75_m=6.0,
                        median_density_points_per_m2=20.0,
                        supporting_original_index_digest=_SHA,
                    ),
                ),
                smoothed_hag_median_m=6.0,
                smoothed_density_median_points_per_m2=20.0,
                valley_state=V4RotatingProfileState.PROFILE_NO_VALLEY,
                measured_gap=False,
                boundary_sample_index=None,
            )
        )
    return _finalize(
        V4RotatingProfileEvidence,
        "evidence_digest",
        profile_id="profile-apex-00000001",
        apex_candidate_id="apex-00000001",
        hypothesis_id=None,
        rays=tuple(rays),
        profile_state=V4RotatingProfileState.PROFILE_NO_VALLEY,
        measured_gap=False,
        profile_valley_support=0.0,
        quality_flags=0,
        reason_flags=0,
    )


def _result(assign_all: bool = True) -> V4TreeSegmentationResult:
    config = _config()
    count = 3
    original_index = _ro(np.asarray([0, 1, 2], dtype="<i8"))
    xyz = _ro(np.zeros((count, 3), dtype="<f8"))
    hag = _ro(np.asarray([6.0, 5.0, 4.0], dtype="<f4"))
    if assign_all:
        states = _ro(np.asarray([1, 1, 1], dtype="|u1"))
        tree_ids = _ro(np.asarray([0, 0, 0], dtype="<i4"))
        offsets = _ro(np.asarray([0, 3], dtype="<i8"))
        members = _ro(np.asarray([0, 1, 2], dtype="<i8"))
    else:
        states = _ro(np.asarray([0, 1, 2], dtype="|u1"))
        tree_ids = _ro(np.asarray([-1, 0, -1], dtype="<i4"))
        offsets = _ro(np.asarray([0, 1], dtype="<i8"))
        members = _ro(np.asarray([1], dtype="<i8"))
    confidences = _ro(np.asarray([1.0, 1.0, 0.0], dtype="<f4"))
    reasons = _ro(np.zeros(count, dtype="<u4"))
    apex = _apex()
    crown = V4TreeCrownSummary(
        crown_area_m2=1.0,
        equivalent_radius_m=0.56,
        responsibility_area_m2=1.0,
        boundary_vertex_count=0,
        boundary_digest=_SHA,
    )
    geometry = V4TreeGeometrySummary(
        member_count=3 if assign_all else 1,
        hag_min_m=4.0,
        hag_max_m=6.0,
        robust_hag_m=5.8,
        vertical_extent_m=2.0,
        xy_centroid=(1.0, 1.0, 5.0),
        xy_bounds_min=(0.0, 0.0, 4.0),
        xy_bounds_max=(2.0, 2.0, 6.0),
        crown_area_m2=1.0,
        equivalent_radius_m=0.56,
        responsibility_area_m2=1.0,
        slice_count=1,
        valid_slice_count=1,
        lowest_supported_slice_id="slice-1",
        highest_supported_slice_id="slice-1",
        trunk_evidence_state=V4TrunkEvidenceAvailability.OBSERVED,
        accepted_trunk_count=1,
        primary_trunk_representative_original_index=2,
        multi_stem_flag=False,
        geometry_quality_flags=0,
    )
    tree = V4TreeInstanceRecord(
        tree_id=0,
        primary_apex_candidate_id=apex.candidate_id,
        secondary_apex_candidate_ids=(),
        primary_trunk_candidate_id="trunk-00000002",
        secondary_trunk_candidate_ids=(),
        trunk_apex_association_ids=("ta-1",),
        member_count=3 if assign_all else 1,
        member_original_index_digest=v4_member_index_digest(members),
        crown_summary=crown,
        geometry_summary=geometry,
        slice_ids=("slice-1",),
        relation_ids=(),
        confidence=0.9,
        quality_flags=0,
        reason_flags=0,
        future_visual_evidence_reference=None,
    )
    hypothesis = V4TreeHypothesisRecord(
        hypothesis_id="hyp-apex-00000001",
        primary_apex_candidate_id=apex.candidate_id,
        secondary_apex_candidate_ids=(),
        trunk_candidate_ids=(),
        fragment_refs=(),
        profile_refs=("profile-apex-00000001",),
        provisional_group_id=None,
        state=V4TreeHypothesisState.CONFIRMED,
        quality_flags=0,
        reason_flags=0,
    )
    audit = V4TreeSegmentationAudit(
        schema_version=V4_TREE_SEGMENTATION_SCHEMA_VERSION,
        total_points=count,
        apex_candidate_count=1,
        accepted_trunk_count=1,
        weak_trunk_count=0,
        suppressed_trunk_count=0,
        unmatched_trunk_count=0,
        strong_association_count=1,
        review_association_count=0,
        unsupported_association_count=0,
        path_observed_count=1,
        path_not_available_count=0,
        profile_confirmed_count=0,
        profile_weak_count=0,
        profile_no_valley_count=1,
        profile_not_available_count=0,
        hypothesis_count=1,
        slice_count=1,
        relation_provisional_count=0,
        relation_final_count=0,
        tree_count=1,
        phase_resource_audit="ok",
        determinism_evidence="canonical",
        warnings=(),
        limitations=(),
    )
    result = V4TreeSegmentationResult(
        schema_version=V4_TREE_SEGMENTATION_SCHEMA_VERSION,
        algorithm_version=V4_TREE_SEGMENTATION_ALGORITHM_VERSION,
        source_identity=_source(),
        config=config,
        handoff_digest=_SHA,
        validated_view_digest=_SHA_B,
        coverage_report_digest=_SHA,
        original_index=original_index,
        xyz_local_m=xyz,
        hag_m=hag,
        assignment_state_by_point=states,
        tree_id_by_point=tree_ids,
        assignment_confidence_by_point=confidences,
        assignment_reason_flags_by_point=reasons,
        tree_member_offsets=offsets,
        ordered_tree_member_original_index=members,
        apex_records=(apex,),
        trunk_records=(_trunk(),),
        trunk_apex_association_records=(_association(),),
        structural_path_records=(_path(),),
        rotating_profile_records=(_profile(),),
        hypothesis_records=(hypothesis,),
        slice_records=(
            V4TreeSliceSummary(
                slice_id="slice-1",
                hypothesis_id="hyp-apex-00000001",
                hag_min_m=4.0,
                hag_max_m=6.0,
                point_count=3,
                centroid_xyz=(1.0, 1.0, 5.0),
                bounds_min_xyz=(0.0, 0.0, 4.0),
                bounds_max_xyz=(2.0, 2.0, 6.0),
                area_m2=1.0,
                density_points_per_m2=3.0,
                boundary_digest=_SHA,
                member_original_index_digest=_SHA,
                quality_flags=0,
                reason_flags=0,
            ),
        ),
        relation_records=(),
        tree_records=(tree,),
        crown_boundary_vertices=_ro(np.zeros((0, 2), dtype="<f8")),
        crown_boundary_offsets=_ro(np.zeros(2, dtype="<i8")),
        audit=audit,
        result_digest="0" * 64,
    )
    result = _with_result_digest(result)
    return result


def _with_result_digest(result: V4TreeSegmentationResult) -> V4TreeSegmentationResult:
    clone = V4TreeSegmentationResult.__new__(V4TreeSegmentationResult)
    for name in V4TreeSegmentationResult.__dataclass_fields__:
        if name == "result_digest":
            object.__setattr__(clone, name, result.computed_result_digest())
        else:
            object.__setattr__(clone, name, getattr(result, name))
    return clone


def test_frozen_version_constants() -> None:
    assert V4_TREE_SEGMENTATION_SCHEMA_VERSION == "v4-tree-segmentation-result-v1"
    assert V4_TREE_SEGMENTATION_ALGORITHM_VERSION == "v4-algorithm-v1"
    assert V4_TREE_SEGMENTATION_CONFIG_SCHEMA_VERSION == "v4-tree-segmentation-config-v1"


def test_frozen_algorithm_constants() -> None:
    assert TRUNK_APEX_STRONG_MIN == 0.65
    assert TRUNK_APEX_REVIEW_MIN == 0.55
    assert TRUNK_CANDIDATE_MIN == 0.62
    assert TRUNK_WEAK_MIN == 0.52
    assert RESPONSIBILITY_MIN == 0.55
    assert RESPONSIBILITY_REVIEW_MARGIN == 0.08
    assert RELATION_MIN == 0.45
    assert RELATION_REVIEW_MARGIN == 0.12
    assert ASSIGNMENT_MIN == 0.62
    assert ASSIGNMENT_REVIEW_MARGIN == 0.10
    assert ROTATING_PROFILE_RAY_COUNT == 24


def test_config_round_trip_and_digest() -> None:
    config = _config()
    parsed = _config_from_dict()
    assert parsed == config
    assert config.config_digest == config.computed_digest()
    assert config.status is V4ConfigStatus.TEST_ONLY


def test_config_rejects_bad_status() -> None:
    with pytest.raises(ValueError):
        _config(status="INVALID")  # type: ignore[arg-type]


def test_config_rejects_digest_mismatch() -> None:
    fields = _config().to_dict()
    fields["config_digest"] = "0" * 64
    with pytest.raises(ValueError):
        V4TreeSegmentationConfig.from_dict(fields)


def test_apex_record_digest_self_check() -> None:
    apex = _apex()
    assert apex.diagnostic_digest == apex.computed_digest()
    assert apex.supporting_original_index == 1


def test_apex_record_rejects_writable_digest_placeholder() -> None:
    with pytest.raises(ValueError):
        V4ApexCandidateRecord(
            candidate_id="apex-1",
            supporting_original_index=1,
            xyz_local_m=(1.0, 1.0, 6.0),
            hag_m=6.0,
            prominence_m=None,
            prominence_available=False,
            plateau_support_count=1,
            cell_point_count=1,
            density_points_per_m2=1.0,
            neighborhood_median_density_points_per_m2=1.0,
            apex_support_score=0.5,
            quality_flags=0,
            reason_flags=0,
            diagnostic_digest="0" * 64,
        )


def test_apex_round_trip_from_dict() -> None:
    apex = _apex()
    parsed = V4ApexCandidateRecord.from_dict(apex.to_dict())
    assert parsed == apex


def test_trunk_round_trip() -> None:
    trunk = _trunk()
    assert trunk.diagnostic_digest == trunk.computed_digest()
    parsed = V4TrunkCandidateRecord.from_dict(trunk.to_dict())
    assert parsed == trunk


def test_association_round_trip() -> None:
    association = _association()
    parsed = V4TrunkApexAssociationEvidence.from_dict(association.to_dict())
    assert parsed == association
    assert parsed.association_state is V4TrunkApexAssociationState.STRONG_ASSOCIATION


def test_path_round_trip() -> None:
    path = _path()
    parsed = V4StructuralPathEvidence.from_dict(path.to_dict())
    assert parsed == path
    assert parsed.availability is V4StructuralPathAvailability.OBSERVED


def test_profile_has_exactly_24_rays() -> None:
    profile = _profile()
    assert len(profile.rays) == ROTATING_PROFILE_RAY_COUNT
    parsed = V4RotatingProfileEvidence.from_dict(profile.to_dict())
    assert parsed == profile


def test_profile_rejects_ray_count() -> None:
    rays = _profile().rays[:10]
    with pytest.raises(ValueError):
        _finalize(
            V4RotatingProfileEvidence,
            "evidence_digest",
            profile_id="profile-1",
            apex_candidate_id="apex-1",
            hypothesis_id=None,
            rays=rays,
            profile_state=V4RotatingProfileState.PROFILE_NO_VALLEY,
            measured_gap=False,
            profile_valley_support=0.0,
            quality_flags=0,
            reason_flags=0,
        )


def test_relation_round_trip() -> None:
    evidence = _finalize(
        V4TreeRelationEvidence,
        "diagnostic_digest",
        relation_id="rel-1",
        hypothesis_a_id="hyp-a",
        hypothesis_b_id="hyp-b",
        horizontal_distance_e=0.5,
        height_difference_e=0.5,
        saddle_drop_e=0.5,
        xy_crown_connectivity_e=0.5,
        empty_gap_evidence_e=0.0,
        slice_continuity_e=0.5,
        local_point_geometry_e=0.0,
        optional_trunk_evidence_e=0.0,
        merge_support=0.6,
        separate_support=0.2,
        relation=V4TreeRelation.MERGE,
        stage="FINAL",
        quality_flags=0,
        reason_flags=0,
    )
    parsed = V4TreeRelationEvidence.from_dict(evidence.to_dict())
    assert parsed == evidence


def _with_digest(record, digest_field):
    placeholder = type(record).__new__(type(record))
    for name in type(record).__dataclass_fields__:
        value = getattr(record, name)
        object.__setattr__(placeholder, name, value)
    digest = placeholder.computed_digest()
    clone = type(record).__new__(type(record))
    for name in type(record).__dataclass_fields__:
        if name == digest_field:
            value = digest
        else:
            value = getattr(record, name)
        object.__setattr__(clone, name, value)
    return clone


def test_relation_rejects_out_of_range_channel() -> None:
    with pytest.raises(ValueError):
        V4TreeRelationEvidence(
            relation_id="rel-1",
            hypothesis_a_id="hyp-a",
            hypothesis_b_id="hyp-b",
            horizontal_distance_e=2.0,
            height_difference_e=0.0,
            saddle_drop_e=0.0,
            xy_crown_connectivity_e=0.0,
            empty_gap_evidence_e=0.0,
            slice_continuity_e=0.0,
            local_point_geometry_e=0.0,
            optional_trunk_evidence_e=0.0,
            merge_support=0.0,
            separate_support=0.0,
            relation=V4TreeRelation.REVIEW,
            stage="FINAL",
            quality_flags=0,
            reason_flags=0,
            diagnostic_digest="0" * 64,
        )


def test_result_all_assigned_conserves() -> None:
    result = _result(assign_all=True)
    validate_v4_tree_segmentation_result(result)
    assert result.audit.tree_count == 1
    assert int(result.original_index.shape[0]) == 3
    assert result.result_digest == result.computed_result_digest()


def test_result_mixed_states_conserves() -> None:
    result = _result(assign_all=False)
    validate_v4_tree_segmentation_result(result)
    states = result.assignment_state_by_point
    assert int((states == 1).sum()) == 1


def test_result_rejects_duplicate_members() -> None:
    base = _result(assign_all=True)
    payload = base.to_dict()
    payload["ordered_tree_member_original_index"] = _ro(np.asarray([0, 0, 1], dtype="<i8"))
    parsed = V4TreeSegmentationResult.from_dict(payload)
    with pytest.raises(ValueError):
        validate_v4_tree_segmentation_result(parsed)


def test_error_codes_frozen() -> None:
    assert V4TreeSegmentationErrorCode.INPUT_V1_REJECTED.value == "INPUT_V1_REJECTED"
    assert (
        V4TreeSegmentationErrorCode.CONSERVATION_FAILED.value == "CONSERVATION_FAILED"
    )
    assert (
        V4TreeSegmentationErrorCode.PUBLICATION_FAILED.value == "PUBLICATION_FAILED"
    )


def test_quality_flags_known_mask() -> None:
    known = 0
    for member in V4ApexQualityFlag.__members__.values():
        known |= int(member.value)
    assert int(V4ApexQualityFlag.REVIEW_REQUIRED) & known
    assert int(V4ApexQualityFlag.SUPPRESSED) & known


def test_tree_instance_requires_member_count_match() -> None:
    base = _result().tree_records[0]
    with pytest.raises(ValueError):
        V4TreeInstanceRecord(
            tree_id=base.tree_id,
            primary_apex_candidate_id=base.primary_apex_candidate_id,
            secondary_apex_candidate_ids=base.secondary_apex_candidate_ids,
            primary_trunk_candidate_id=base.primary_trunk_candidate_id,
            secondary_trunk_candidate_ids=base.secondary_trunk_candidate_ids,
            trunk_apex_association_ids=base.trunk_apex_association_ids,
            member_count=99,
            member_original_index_digest=base.member_original_index_digest,
            crown_summary=base.crown_summary,
            geometry_summary=base.geometry_summary,
            slice_ids=base.slice_ids,
            relation_ids=base.relation_ids,
            confidence=base.confidence,
            quality_flags=base.quality_flags,
            reason_flags=base.reason_flags,
            future_visual_evidence_reference=None,
        )


def test_segmentation_quality_flag_review_required() -> None:
    assert int(V4SegmentationQualityFlag.REVIEW_REQUIRED) == 1
