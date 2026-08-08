"""FGCPOL1 policy, routing states, and exact-schema contract tests.

Authority: docs/full-resolution-ground-classification-design.md sections 2,
11, and 12, plus the frozen Bundle A plan test gate.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from core.contracts.frv import frv_digest
from core.contracts.ground_classification import (
    CONFIRMED_GROUND,
    FGC_ALGORITHM_ID,
    FGC_POLICY_DIGEST_DOMAIN,
    FGC_POLICY_SCHEMA_VERSION,
    NON_GROUND,
    REVIEW_GROUND,
    UNROUTED,
    FgcDecisionChunkRecord,
    FgcErrorCode,
    FgcSpoolTileSummary,
    GroundClassificationPolicyConfig,
    GroundDecision,
    fgc_free_space_requirement_bytes,
    fgc_tile_layout_digest,
    fgc_working_set_bytes,
    map_derivative_class,
    validate_ground_classification_policy,
    validate_ordered_chunk_ranges,
)


def test_terminal_decision_values() -> None:
    assert GroundDecision.NON_GROUND == 0
    assert GroundDecision.CONFIRMED_GROUND == 1
    assert GroundDecision.REVIEW_GROUND == 2
    assert GroundDecision.UNROUTED == 255
    assert NON_GROUND == 0
    assert CONFIRMED_GROUND == 1
    assert REVIEW_GROUND == 2
    assert UNROUTED == 255


def test_default_policy_has_frozen_values() -> None:
    policy = GroundClassificationPolicyConfig()
    assert policy.schema_version == FGC_POLICY_SCHEMA_VERSION
    assert policy.algorithm_id == FGC_ALGORITHM_ID
    assert policy.tile_side_m == 256.0
    assert policy.halo_m == 20.0
    assert policy.fine_cell_m == 0.5
    assert policy.coverage_cell_m == 1.0
    assert policy.pmf_windows_m == (1.0, 2.0, 4.0, 8.0, 16.0)
    assert policy.pmf_minimum_valid_cells == 5
    assert policy.seed_radius_m == 3.0
    assert policy.seed_connected_support_cells == 5
    assert policy.seed_lower_envelope_cells == 5
    assert policy.seed_isolated_low_m == 0.30
    assert policy.plane_radius_m == 3.0
    assert policy.plane_min_supports == 12
    assert policy.plane_max_supports == 32
    assert policy.clip_tolerance_m == 0.20
    assert policy.clip_rounds == 2
    assert policy.singular_ratio == 1e-10
    assert policy.slope_max == 0.75
    assert policy.rms_max_m == 0.12
    assert policy.competing_half_min_supports == 6
    assert policy.competing_disagreement_m == 0.35
    assert policy.vertical_min_other_rows == 8
    assert policy.vertical_radius_m == 1.0
    assert policy.vertical_span_m == 1.50
    assert policy.residual_confirm_min_m == -0.15
    assert policy.residual_confirm_max_m == 0.20
    assert policy.residual_review_min_m == -0.35
    assert policy.residual_review_max_m == 0.60
    assert policy.residual_non_ground_above_m == 0.60
    assert policy.coverage_occupancy_fraction == 0.20
    assert policy.coverage_hole_width_m == 25.0
    assert policy.max_source_points == 50_000_000
    assert policy.max_core_tile_points == 8_000_000
    assert policy.max_halo_tile_points == 8_000_000
    assert policy.max_halo_grid_cells == 500_000
    assert policy.max_plane_batch == 65_536
    assert policy.decision_memmap_bytes == 64 * 1024 * 1024
    assert policy.max_coverage_cells == 50_000_000


def test_policy_digest_uses_fgcpol1_domain() -> None:
    policy = GroundClassificationPolicyConfig()
    payload = policy.to_digest_dict()
    payload["digest_domain"] = FGC_POLICY_DIGEST_DOMAIN
    expected = frv_digest(payload)
    assert policy.policy_digest == expected
    validate_ground_classification_policy(policy)


def test_policy_rejects_tampered_digest() -> None:
    payload = GroundClassificationPolicyConfig().to_dict()
    forged = dict(payload)
    forged["policy_digest"] = "0" * 64
    with pytest.raises(ValueError, match="policy_digest"):
        GroundClassificationPolicyConfig.from_dict(forged)


def test_policy_rejects_unknown_field() -> None:
    payload = GroundClassificationPolicyConfig().to_dict()
    payload["p_hag_config_digest"] = "0" * 64  # P-HAG must never enter the policy
    with pytest.raises(ValueError, match="unknown field"):
        GroundClassificationPolicyConfig.from_dict(payload)


def test_policy_rejects_nonfinite_value() -> None:
    payload = GroundClassificationPolicyConfig().to_dict()
    payload["tile_side_m"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        GroundClassificationPolicyConfig.from_dict(payload)


def test_policy_rejects_unordered_pmf_windows() -> None:
    payload = GroundClassificationPolicyConfig().to_dict()
    payload["pmf_windows_m"] = [16.0, 8.0, 4.0, 2.0, 1.0]
    with pytest.raises(ValueError, match="ascending"):
        GroundClassificationPolicyConfig.from_dict(payload)


def test_policy_rejects_path_like_value() -> None:
    payload = GroundClassificationPolicyConfig().to_dict()
    payload["tile_side_m"] = Path("C:/tmp")
    with pytest.raises(ValueError):
        GroundClassificationPolicyConfig.from_dict(payload)


def test_policy_rejects_bad_schema() -> None:
    payload = GroundClassificationPolicyConfig().to_dict()
    payload["schema_version"] = "other-schema"
    with pytest.raises(ValueError, match="schema version"):
        GroundClassificationPolicyConfig.from_dict(payload)


def test_map_derivative_class_semantics() -> None:
    # CONFIRMED_GROUND always becomes 2.
    assert map_derivative_class(2, GroundDecision.CONFIRMED_GROUND) == 2
    assert map_derivative_class(5, GroundDecision.CONFIRMED_GROUND) == 2
    # REVIEW and NON preserve input except stale class 2 -> 1.
    assert map_derivative_class(2, GroundDecision.REVIEW_GROUND) == 1
    assert map_derivative_class(2, GroundDecision.NON_GROUND) == 1
    assert map_derivative_class(0, GroundDecision.REVIEW_GROUND) == 0
    assert map_derivative_class(1, GroundDecision.REVIEW_GROUND) == 1
    assert map_derivative_class(5, GroundDecision.NON_GROUND) == 5
    # REVIEW_GROUND must never become class 2; UNROUTED is not mappable.
    assert map_derivative_class(1, GroundDecision.REVIEW_GROUND) != 2
    with pytest.raises(ValueError):
        map_derivative_class(300, GroundDecision.NON_GROUND)


def test_working_set_is_within_16_gib() -> None:
    # With the reviewed per-tile point caps the frozen working-set bound stays
    # far inside the 16 GB ceiling (design section 11).
    policy = GroundClassificationPolicyConfig()
    working_set = fgc_working_set_bytes(policy)
    assert working_set < 16 * 1024 * 1024 * 1024
    assert policy.max_halo_grid_cells * 128 + policy.max_plane_batch * 128 > 0


def test_free_space_formula() -> None:
    policy = GroundClassificationPolicyConfig()
    n = 1000
    input_bytes = 1_000_000
    required = fgc_free_space_requirement_bytes(input_bytes, n, policy)
    base = 2 * input_bytes + 20 * n + 2 * n + 51 * n + 128 * 1024 * 1024
    assert required == math.ceil(1.15 * base)


def test_tile_layout_digest_is_deterministic_and_ordered() -> None:
    tiles_a = [(3, 1), (1, 0), (2, 2)]
    tiles_b = [(1, 0), (2, 2), (3, 1)]
    digest_a = fgc_tile_layout_digest(tiles_a, 100)
    digest_b = fgc_tile_layout_digest(tiles_b, 100)
    assert digest_a == digest_b
    digest_c = fgc_tile_layout_digest(tiles_b, 101)
    assert digest_a != digest_c


def test_fgc_error_codes_are_stable_strings() -> None:
    assert FgcErrorCode.UNSUPPORTED_COORDINATE_GRID == "UNSUPPORTED_COORDINATE_GRID"
    assert FgcErrorCode.GROUND_COVERAGE_GAP == "GROUND_COVERAGE_GAP"
    assert FgcErrorCode.PUBLICATION_COLLISION == "PUBLICATION_COLLISION"
    assert FgcErrorCode.COMPLETION_MANIFEST_FINALIZATION_FAILED == (
        "COMPLETION_MANIFEST_FINALIZATION_FAILED"
    )


def test_spool_tile_summary_roundtrip() -> None:
    tile = FgcSpoolTileSummary(tile_id_ty=0, tile_id_tx=1, row_count=7, sha256="a" * 64)
    assert tile.tile_id == (0, 1)
    restored = FgcSpoolTileSummary.from_dict(tile.to_dict())
    assert restored == tile
    with pytest.raises(ValueError):
        FgcSpoolTileSummary(tile_id_ty=0, tile_id_tx=1, row_count=-1, sha256="a" * 64)


def test_ordered_chunk_ranges_reject_gap_overlap_and_misorder() -> None:
    records = [
        FgcDecisionChunkRecord(
            start_index=0,
            end_index=5,
            relative_path="chunk_0.bin",
            sha256="a" * 64,
            size_bytes=5,
            state_counts={"non_ground": 5, "confirmed_ground": 0, "review_ground": 0},
        ),
        FgcDecisionChunkRecord(
            start_index=5,
            end_index=10,
            relative_path="chunk_1.bin",
            sha256="a" * 64,
            size_bytes=5,
            state_counts={"non_ground": 0, "confirmed_ground": 5, "review_ground": 0},
        ),
    ]
    validate_ordered_chunk_ranges(records, source_point_count=10)

    with pytest.raises(ValueError):
        validate_ordered_chunk_ranges(records, source_point_count=11)

    overlapping = [
        FgcDecisionChunkRecord(
            start_index=0,
            end_index=6,
            relative_path="chunk_0.bin",
            sha256="a" * 64,
            size_bytes=6,
            state_counts={"non_ground": 6, "confirmed_ground": 0, "review_ground": 0},
        ),
        FgcDecisionChunkRecord(
            start_index=5,
            end_index=10,
            relative_path="chunk_1.bin",
            sha256="a" * 64,
            size_bytes=5,
            state_counts={"non_ground": 0, "confirmed_ground": 5, "review_ground": 0},
        ),
    ]
    with pytest.raises(ValueError, match="gap or overlap"):
        validate_ordered_chunk_ranges(overlapping, source_point_count=10)

    gapped = [
        FgcDecisionChunkRecord(
            start_index=0,
            end_index=4,
            relative_path="chunk_0.bin",
            sha256="a" * 64,
            size_bytes=4,
            state_counts={"non_ground": 4, "confirmed_ground": 0, "review_ground": 0},
        ),
        FgcDecisionChunkRecord(
            start_index=6,
            end_index=10,
            relative_path="chunk_1.bin",
            sha256="a" * 64,
            size_bytes=4,
            state_counts={"non_ground": 0, "confirmed_ground": 4, "review_ground": 0},
        ),
    ]
    with pytest.raises(ValueError, match="gap or overlap"):
        validate_ordered_chunk_ranges(gapped, source_point_count=10)


def test_decision_chunk_state_counts_must_sum() -> None:
    with pytest.raises(ValueError, match="state counts"):
        FgcDecisionChunkRecord(
            start_index=0,
            end_index=5,
            relative_path="chunk.bin",
            sha256="a" * 64,
            size_bytes=5,
            state_counts={"non_ground": 3, "confirmed_ground": 0, "review_ground": 0},
        )
