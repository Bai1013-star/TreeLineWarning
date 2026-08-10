"""V4 single-tree segmentation runtime (Algorithm-V1, PLAN-FROZEN).

``segment_v4_trees`` consumes a formal digest-bound v2 V4 input handoff and a
``V4TreeSegmentationConfig`` and returns a fully validated
``V4TreeSegmentationResult``.  The pipeline is a single pass with internal
deterministic stages: canonical normal population, robust CHM and apex
detection, trunk-candidate detection, bounded trunk-apex association with
CSP-style local structural paths, rotating-profile crown-boundary priors,
synchronous 2D responsibility, 3D slices/fragments, final relation, one
constraint-aware FINAL UNION, deterministic Tree IDs and exact conservation.

Every scoring function in section 4.2A of the consolidated plan is implemented
as a pure deterministic function so the frozen fixed vectors are unit-testable.
No global all-point graph, no global trunk×apex dense matrix, no
processing-order dependence and no ``dict.get(..., 0)`` fallbacks.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from core.contracts.frv import FrvConfigurationPurpose, SourceIdentity
from core.contracts.v4_inputs import V4InputHandoffResult
from core.contracts.v4_tree_segmentation import (
    ASSIGNMENT_MIN,
    ASSIGNMENT_REVIEW_MARGIN,
    LPATH_XY_MIN_LA_FACTOR,
    LPATH_XY_MIN_LTR_XY_FACTOR,
    LTA_MIN_LA_FACTOR,
    LTA_MIN_XY_FACTOR,
    LTR_XY_MIN_LA_FACTOR,
    LTR_XY_MIN_XY_FACTOR,
    LTR_Z_MIN_SLICE_FACTOR,
    PCA_ABS_EPS,
    PCA_REL_GAP_MIN,
    RELATION_MIN,
    RELATION_REVIEW_MARGIN,
    RESPONSIBILITY_MIN,
    RESPONSIBILITY_REVIEW_MARGIN,
    ROTATING_PROFILE_DENSITY_RATIO,
    ROTATING_PROFILE_DROP_MIN,
    ROTATING_PROFILE_RAY_COUNT,
    ROTATING_PROFILE_REBOUND_MIN,
    ROTATING_PROFILE_VALLEY_MIN,
    STRUCTURAL_PATH_DENSITY_WEIGHT,
    STRUCTURAL_PATH_DOWNWARD_WEIGHT,
    TRUNK_APEX_REVIEW_MIN,
    TRUNK_APEX_STRONG_MIN,
    TRUNK_BIN_MIN_POINTS,
    TRUNK_CANDIDATE_MIN,
    TRUNK_COMPONENT_MIN_BINS,
    TRUNK_COMPONENT_MIN_POINTS,
    TRUNK_CORE_RADIUS_FACTOR,
    TRUNK_WEAK_MIN,
    V4_TREE_SEGMENTATION_ALGORITHM_VERSION,
    V4_TREE_SEGMENTATION_SCHEMA_VERSION,
    V4ApexCandidateRecord,
    V4ApexQualityFlag,
    V4AssignmentReasonFlag,
    V4AssociationLookupKind,
    V4AssociationQualityFlag,
    V4ConfigStatus,
    V4ProfileQualityFlag,
    V4ProfileSampleState,
    V4RotatingProfileEvidence,
    V4RotatingProfileRayRecord,
    V4RotatingProfileRaySample,
    V4RotatingProfileState,
    V4StructuralPathAvailability,
    V4StructuralPathEvidence,
    V4TreeAssignmentState,
    V4TreeCrownSummary,
    V4TreeGeometrySummary,
    V4TreeHypothesisRecord,
    V4TreeHypothesisState,
    V4TreeInstanceRecord,
    V4TreeRelation,
    V4TreeRelationEvidence,
    V4TreeSegmentationAudit,
    V4TreeSegmentationConfig,
    V4TreeSegmentationError,
    V4TreeSegmentationErrorCode,
    V4TreeSegmentationResult,
    V4TreeSliceSummary,
    V4TrunkApexAssociationEvidence,
    V4TrunkApexAssociationState,
    V4TrunkCandidateRecord,
    V4TrunkCandidateState,
    V4TrunkEvidenceAvailability,
    V4TrunkQualityFlag,
    v4_member_index_digest,
    validate_v4_tree_segmentation_result,
)

# ---------------------------------------------------------------------------
# Algorithm-V1 pure scoring primitives
# ---------------------------------------------------------------------------


def c01(value: float) -> float:
    """Clamp a binary64 value into ``[0, 1]``."""

    return min(1.0, max(0.0, value))


def near(distance: float, scale: float) -> float:
    """Physical ``near`` normalizer: ``1 - clamp01(|d|/L)``."""

    return 1.0 - c01(abs(distance) / scale)


def similar(delta: float, scale: float) -> float:
    """Physical ``similar`` normalizer (identical shape to ``near``)."""

    return 1.0 - c01(abs(delta) / scale)


def signed(quantity: float) -> float:
    """Map ``[0,1]`` support into signed ``[-1,1]`` evidence."""

    return 2.0 * c01(quantity) - 1.0


def clamp11(value: float) -> float:
    """Clamp a binary64 value into ``[-1, 1]``."""

    return min(1.0, max(-1.0, value))


def rho_floor_for(lxy: float) -> float:
    """Floor density (points/m^2) for safe ratios."""

    return 1.0 / (lxy * lxy)


def safe_ratio(a: float, b: float, rho_floor: float) -> float:
    """Deterministic safe ratio ``clamp01(a / max(b, rho_floor))``."""

    return c01(a / max(b, rho_floor))


@dataclass(frozen=True, slots=True)
class V4AlgorithmScales:
    """Derived Algorithm-V1 physical scales from the config."""

    lxy: float
    la: float
    ld: float
    ls: float
    lr: float
    ltr_xy: float
    ltr_z: float
    lta: float
    lpath_xy: float
    btr: float
    lprofile: float
    rho_floor: float


def derive_algorithm_scales(config: V4TreeSegmentationConfig) -> V4AlgorithmScales:
    """Derive all Algorithm-V1 derived scales from the config's core physics."""

    lxy = config.xy_cell_size_m
    la = config.apex_neighborhood_radius_m
    ld = config.density_trend_window_m
    ls = config.slice_thickness_m
    lr = config.relation_neighborhood_radius_m
    ltr_xy = max(LTR_XY_MIN_XY_FACTOR * lxy, LTR_XY_MIN_LA_FACTOR * la)
    ltr_z = max(LTR_Z_MIN_SLICE_FACTOR * ls, la)
    lta = max(LTA_MIN_LA_FACTOR * la, LTA_MIN_XY_FACTOR * lxy)
    lpath_xy = max(LPATH_XY_MIN_LTR_XY_FACTOR * ltr_xy, LPATH_XY_MIN_LA_FACTOR * la)
    btr = max(lxy, 0.5 * ls)
    return V4AlgorithmScales(
        lxy=lxy,
        la=la,
        ld=ld,
        ls=ls,
        lr=lr,
        ltr_xy=ltr_xy,
        ltr_z=ltr_z,
        lta=lta,
        lpath_xy=lpath_xy,
        btr=btr,
        lprofile=lr,
        rho_floor=rho_floor_for(lxy),
    )


_RESPONSIBILITY_WEIGHTS = (0.16, 0.16, 0.14, 0.14, 0.10, 0.10, 0.10, 0.10)
_RELATION_WEIGHTS = (0.14, 0.10, 0.16, 0.14, 0.14, 0.16, 0.10, 0.06)
_ASSIGNMENT_WEIGHTS = (0.18, 0.14, 0.12, 0.10, 0.12, 0.10, 0.08, 0.10, 0.06)
_TRUNK_CHANNEL_WEIGHTS = (0.35, 0.30, 0.20, 0.15)


# ---------------------------------------------------------------------------
# Local edge resource model
# ---------------------------------------------------------------------------
# ``maximum_local_edges`` guards are implementation safety/resource bounds, not
# semantic validity bounds.  Under the 0.10 m coarse-to-fine input the number
# of local voxel / fragment / pair structures legitimately grows while the
# computation remains safe, so every guard projects the dominant working
# memory / work of its *local* computation instead of an arbitrary fixed count.
# The same deterministic local structure that OLD evaluated cheaply must never
# be dropped; the projection only rejects genuinely unsafe local problems.
#
# Documented per-entry projections (conservative byte estimates):
#   * a Python dict entry (tuple key + int value + dict slot)  ~160 bytes
#   * one compact int64 src/dst edge pair                        ~16 bytes
#   * one candidate-pair evaluation work-equivalent              ~128 bytes
_MAXIMUM_LOCAL_WORKING_BYTES = 4 * (1 << 30)  # 4 GiB per local V4 sub-computation
_PROJECTED_DICT_ENTRY_BYTES = 160
_PROJECTED_EDGE_PAIR_BYTES = 16
_PROJECTED_PAIR_EVALUATION_BYTES = 128


def _projected_bytes(entry_count: int, bytes_per_entry: int) -> int:
    """Return the conservative projected working size of a local structure."""
    return int(entry_count) * int(bytes_per_entry)


def _raise_local_edge_resource_limit() -> None:
    """Fail closed on a genuinely unsafe local edge resource projection."""
    raise V4TreeSegmentationError(
        code=V4TreeSegmentationErrorCode.RESOURCE_LIMIT,
        field_name="maximum_local_edges",
    )
_ASSOCIATION_WEIGHTS = (0.28, 0.20, 0.22, 0.18, 0.12)


def responsibility_r0(
    *,
    apex_support: float,
    path_bottleneck: float,
    density_trend: float,
    hag_continuity: float,
    distance_decay: float,
    connectivity: float,
    empty_gap: float,
) -> float:
    """First-pass responsibility score without the neighbour-pressure channel."""

    return (
        math.fsum(
            [
                _RESPONSIBILITY_WEIGHTS[0] * apex_support,
                _RESPONSIBILITY_WEIGHTS[1] * path_bottleneck,
                _RESPONSIBILITY_WEIGHTS[2] * density_trend,
                _RESPONSIBILITY_WEIGHTS[3] * hag_continuity,
                _RESPONSIBILITY_WEIGHTS[4] * distance_decay,
                _RESPONSIBILITY_WEIGHTS[5] * connectivity,
                _RESPONSIBILITY_WEIGHTS[6] * empty_gap,
            ]
        )
        / 0.90
    )


def responsibility_pressure(r0_self: float, best_rival_r0: float | None) -> float:
    """Neighbour-pressure channel (``1`` when no legal rival is observed)."""

    if best_rival_r0 is None:
        return 1.0
    return c01(0.5 + (r0_self - best_rival_r0) / (2.0 * RESPONSIBILITY_REVIEW_MARGIN))


def responsibility_score(
    *,
    apex_support: float,
    path_bottleneck: float,
    density_trend: float,
    hag_continuity: float,
    distance_decay: float,
    connectivity: float,
    empty_gap: float,
    pressure: float,
) -> float:
    """Final responsibility claim score over all eight channels."""

    return math.fsum(
        [
            _RESPONSIBILITY_WEIGHTS[0] * apex_support,
            _RESPONSIBILITY_WEIGHTS[1] * path_bottleneck,
            _RESPONSIBILITY_WEIGHTS[2] * density_trend,
            _RESPONSIBILITY_WEIGHTS[3] * hag_continuity,
            _RESPONSIBILITY_WEIGHTS[4] * distance_decay,
            _RESPONSIBILITY_WEIGHTS[5] * connectivity,
            _RESPONSIBILITY_WEIGHTS[6] * empty_gap,
            _RESPONSIBILITY_WEIGHTS[7] * pressure,
        ]
    )


def merge_support_score(
    eight_channels: tuple[float, float, float, float, float, float, float, float],
) -> float:
    """Positive signed-evidence support for MERGE."""

    return math.fsum(
        [
            weight * max(channel, 0.0)
            for weight, channel in zip(_RELATION_WEIGHTS, eight_channels, strict=True)
        ]
    )


def separate_support_score(
    eight_channels: tuple[float, float, float, float, float, float, float, float],
) -> float:
    """Positive signed-evidence support for KEEP_SEPARATE."""

    return math.fsum(
        [
            weight * max(-channel, 0.0)
            for weight, channel in zip(_RELATION_WEIGHTS, eight_channels, strict=True)
        ]
    )


def relation_decision(
    merge_support: float, separate_support: float
) -> V4TreeRelation:
    """MERGE / KEEP_SEPARATE / REVIEW from the two support scores."""

    if (
        merge_support >= RELATION_MIN
        and merge_support - separate_support >= RELATION_REVIEW_MARGIN
    ):
        return V4TreeRelation.MERGE
    if (
        separate_support >= RELATION_MIN
        and separate_support - merge_support >= RELATION_REVIEW_MARGIN
    ):
        return V4TreeRelation.KEEP_SEPARATE
    return V4TreeRelation.REVIEW


def trunk_candidate_score(tv: float, ta: float, ts: float, tc: float) -> float:
    """Four-channel trunk-candidate score."""

    return math.fsum(
        [
            _TRUNK_CHANNEL_WEIGHTS[0] * tv,
            _TRUNK_CHANNEL_WEIGHTS[1] * ta,
            _TRUNK_CHANNEL_WEIGHTS[2] * ts,
            _TRUNK_CHANNEL_WEIGHTS[3] * tc,
        ]
    )


def trunk_candidate_state(score: float) -> V4TrunkCandidateState:
    if score >= TRUNK_CANDIDATE_MIN:
        return V4TrunkCandidateState.ACCEPTED
    if score >= TRUNK_WEAK_MIN:
        return V4TrunkCandidateState.WEAK
    return V4TrunkCandidateState.SUPPRESSED


def trunk_apex_score(tx: float, td: float, tvc: float, tg: float, tr: float) -> float:
    """Five-channel trunk-apex association score."""

    return math.fsum(
        [
            _ASSOCIATION_WEIGHTS[0] * tx,
            _ASSOCIATION_WEIGHTS[1] * td,
            _ASSOCIATION_WEIGHTS[2] * tvc,
            _ASSOCIATION_WEIGHTS[3] * tg,
            _ASSOCIATION_WEIGHTS[4] * tr,
        ]
    )


def trunk_apex_state(score: float) -> V4TrunkApexAssociationState:
    if score >= TRUNK_APEX_STRONG_MIN:
        return V4TrunkApexAssociationState.STRONG_ASSOCIATION
    if score >= TRUNK_APEX_REVIEW_MIN:
        return V4TrunkApexAssociationState.REVIEW_ASSOCIATION
    return V4TrunkApexAssociationState.UNSUPPORTED_ASSOCIATION


def association_strong01(score: float, state: V4TrunkApexAssociationState) -> float:
    """Signed strong-trunk support with an inclusive 0.65 boundary."""

    if state is V4TrunkApexAssociationState.STRONG_ASSOCIATION:
        return 0.5 + 0.5 * c01((score - TRUNK_APEX_STRONG_MIN) / 0.35)
    return 0.0


def association_exclusive01(score: float, state: V4TrunkApexAssociationState) -> float:
    """Signed observed-exclusive support for an unsupported/out-of-radius edge."""

    if state is V4TrunkApexAssociationState.UNSUPPORTED_ASSOCIATION:
        return 1.0 - c01(score / TRUNK_APEX_REVIEW_MIN)
    return 0.0


def assignment_base(eight_channels_excluding_competitor: tuple[float, ...]) -> float:
    """Base assignment support from the eight non-competitor channels."""

    non_competitor_weights = _ASSIGNMENT_WEIGHTS[:7] + _ASSIGNMENT_WEIGHTS[8:]
    weighted = math.fsum(
        [
            weight * channel
            for weight, channel in zip(
                non_competitor_weights,
                eight_channels_excluding_competitor,
                strict=True,
            )
        ]
    )
    return 0.5 + 0.5 * weighted / 0.90


def assignment_support_score(nine_channels: tuple[float, ...]) -> float:
    """Final assignment support over all nine channels."""

    weighted = math.fsum(
        [
            weight * channel
            for weight, channel in zip(_ASSIGNMENT_WEIGHTS, nine_channels, strict=True)
        ]
    )
    return 0.5 + 0.5 * weighted


def assignment_competitor_channel(base: float, best_other_base: float | None) -> float:
    if best_other_base is None:
        return 1.0
    return clamp11((base - best_other_base) / 0.10)


def assignment_confidence(
    state: V4TreeAssignmentState,
    best: float,
    second: float | None,
) -> float:
    """Exact assignment confidence per the frozen formula."""

    if state is V4TreeAssignmentState.ASSIGNED:
        margin = 1.0 if second is None else c01((best - second) / 0.10)
        return c01(0.5 * best + 0.5 * margin)
    if state is V4TreeAssignmentState.UNASSIGNED:
        return c01(1.0 - best / ASSIGNMENT_MIN)
    return 0.0


def assignment_decision(
    *,
    support: float,
    second: float | None,
    mandatory_review: bool,
    comparison_complete: bool,
) -> V4TreeAssignmentState:
    """Assignment decision over the FINAL assignment support.

    The ``0.62 / 0.10`` thresholds operate on the final nine-channel
    ``assignment_support`` of the best candidate (``support``) and the second
    candidate, never on an intermediate eight-channel base.
    """

    if not comparison_complete:
        return V4TreeAssignmentState.REVIEW_REQUIRED
    if mandatory_review:
        return V4TreeAssignmentState.REVIEW_REQUIRED
    if support >= ASSIGNMENT_MIN and (
        second is None or support - second >= ASSIGNMENT_REVIEW_MARGIN
    ):
        return V4TreeAssignmentState.ASSIGNED
    if (
        support >= ASSIGNMENT_MIN
        and second is not None
        and support - second < ASSIGNMENT_REVIEW_MARGIN
    ):
        return V4TreeAssignmentState.REVIEW_REQUIRED
    return V4TreeAssignmentState.UNASSIGNED


def profile_confirmed_stop(
    *,
    measured_gap: bool,
    confirmed_valley: bool,
    local_density_declined: bool,
    local_hag_declined: bool,
) -> bool:
    """Rotating-profile transition stop authority (never a sole valley)."""

    if measured_gap:
        return True
    return confirmed_valley and (local_density_declined or local_hag_declined)


def _member_digest(values: Any) -> str:
    return v4_member_index_digest(values)


# ---------------------------------------------------------------------------
# Internal grid / cell index
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CellSummary:
    """One occupied XY cell summary used by CHM, responsibility and profile."""

    cell_id: int
    col: int
    row: int
    count: int
    hag_values: tuple[float, ...]
    hag_q75: float
    hag_max: float
    chm_height: float
    cell_center: tuple[float, float]
    density: float
    min_original_index: int


@dataclass(frozen=True, slots=True)
class _Grid:
    """Deterministic XY uniform grid over the normal population."""

    cell_size: float
    origin_xy: tuple[float, float]
    cols: int
    rows: int
    cell_id_to_summary: dict[int, _CellSummary]
    col_to_cell_ids: dict[int, list[int]]
    point_cell_ids: np.ndarray
    occupied_cell_ids: tuple[int, ...]
    cell_positions: dict[int, list[int]]

    def cell_id(self, col: int, row: int) -> int:
        return row * self.cols + col

    def neighbors_8(self, cell_id: int) -> tuple[int, ...]:
        row = cell_id // self.cols
        col = cell_id % self.cols
        result: list[int] = []
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                if drow == 0 and dcol == 0:
                    continue
                nrow = row + drow
                ncol = col + dcol
                if 0 <= nrow < self.rows and 0 <= ncol < self.cols:
                    result.append(nrow * self.cols + ncol)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class _NormalPopulation:
    """Canonical read-only normal population sorted by original_index."""

    source_identity: SourceIdentity
    original_index: np.ndarray
    xyz_local_m: np.ndarray
    hag_m: np.ndarray
    membership_state: np.ndarray
    hag_evidence_state: np.ndarray
    segmentation_hag_readiness: np.ndarray
    candidate_reason_flags: np.ndarray
    raw_weight: np.ndarray
    handoff_digest: str
    validated_view_digest: str
    coverage_report_digest: str
    point_index_by_original_index: dict[int, int]


def _canonical_population(handoff: V4InputHandoffResult) -> _NormalPopulation:
    recovered = handoff.recovered_vegetation
    if recovered is None:
        raise V4TreeSegmentationError(
            code=V4TreeSegmentationErrorCode.INPUT_ROOT_INVALID,
            field_name="recovered_vegetation",
        )
    view = handoff.validated_membership
    report = handoff.coverage_report
    if view is None or report is None:
        raise V4TreeSegmentationError(
            code=V4TreeSegmentationErrorCode.INPUT_ROOT_INVALID,
            field_name="validated_membership",
        )
    order = np.argsort(np.asarray(recovered.original_index, dtype=np.int64), kind="stable")
    original_index = np.asarray(recovered.original_index, dtype="<i8")[order]
    xyz_local_m = np.asarray(recovered.xyz_local_m, dtype=np.float64)[order]
    hag_m = np.asarray(recovered.hag_m, dtype=np.float32)[order]
    membership_state = np.asarray(recovered.membership_state, dtype="|u1")[order]
    hag_evidence_state = np.asarray(recovered.hag_evidence_state, dtype="|u1")[order]
    segmentation_hag_readiness = np.asarray(
        recovered.segmentation_hag_readiness, dtype="|u1"
    )[order]
    candidate_reason_flags = np.asarray(recovered.candidate_reason_flags, dtype="<u4")[order]
    recovered_weight = getattr(recovered, "raw_weight", None)
    if recovered_weight is None:
        raw_weight = np.ones(len(original_index), dtype="<u4")
    else:
        raw_weight = np.asarray(recovered_weight, dtype="<u4")[order]
        if raw_weight.shape != original_index.shape or np.any(raw_weight == 0):
            raise V4TreeSegmentationError(
                code=V4TreeSegmentationErrorCode.INPUT_ROOT_INVALID,
                field_name="raw_weight",
            )
    for array in (
        original_index,
        xyz_local_m,
        hag_m,
        membership_state,
        hag_evidence_state,
        segmentation_hag_readiness,
        candidate_reason_flags,
        raw_weight,
    ):
        array.setflags(write=False)
    count = int(original_index.shape[0])
    index_lookup = {int(original_index[i]): i for i in range(count)}
    return _NormalPopulation(
        source_identity=view.source_identity,
        original_index=original_index,
        xyz_local_m=xyz_local_m,
        hag_m=hag_m,
        membership_state=membership_state,
        hag_evidence_state=hag_evidence_state,
        segmentation_hag_readiness=segmentation_hag_readiness,
        candidate_reason_flags=candidate_reason_flags,
        raw_weight=raw_weight,
        handoff_digest=handoff.handoff_digest,
        validated_view_digest=view.view_digest,
        coverage_report_digest=report.report_digest,
        point_index_by_original_index=index_lookup,
    )


def _raw_weight_sum(
    population: _NormalPopulation,
    positions: list[int] | tuple[int, ...] | np.ndarray,
) -> int:
    """Return the source-point population represented by selected rows.

    ``fragment.positions`` is intentionally a tuple. Coercing every selection
    container to a one-dimensional integer array prevents NumPy from treating
    a tuple as a multidimensional index.
    """

    index = np.asarray(positions, dtype=np.int64)
    return int(np.sum(population.raw_weight[index], dtype=np.int64))


def _build_grid(
    population: _NormalPopulation,
    scales: V4AlgorithmScales,
    config: V4TreeSegmentationConfig,
) -> _Grid:
    xyz = population.xyz_local_m
    count = int(xyz.shape[0])
    if count > config.maximum_points:
        raise V4TreeSegmentationError(
            code=V4TreeSegmentationErrorCode.RESOURCE_LIMIT, field_name="maximum_points"
        )
    if count == 0:
        return _Grid(
            cell_size=scales.lxy,
            origin_xy=(0.0, 0.0),
            cols=1,
            rows=1,
            cell_id_to_summary={},
            col_to_cell_ids={},
            point_cell_ids=np.empty(0, dtype=np.int64),
            occupied_cell_ids=(),
            cell_positions={},
        )
    xy = xyz[:, :2]
    if not np.isfinite(xy).all():
        raise V4TreeSegmentationError(
            code=V4TreeSegmentationErrorCode.INPUT_NONFINITE, field_name="xyz_local_m"
        )
    margin = scales.lr + scales.lxy
    min_x = float(np.min(xy[:, 0])) - margin
    max_x = float(np.max(xy[:, 0])) + margin
    min_y = float(np.min(xy[:, 1])) - margin
    max_y = float(np.max(xy[:, 1])) + margin
    origin_x = math.floor(min_x / scales.lxy) * scales.lxy
    origin_y = math.floor(min_y / scales.lxy) * scales.lxy
    cols = int(math.floor(max_x / scales.lxy)) - int(math.floor(min_x / scales.lxy)) + 1
    rows = int(math.floor(max_y / scales.lxy)) - int(math.floor(min_y / scales.lxy)) + 1
    total_cells = cols * rows
    if total_cells > config.maximum_grid_cells:
        raise V4TreeSegmentationError(
            code=V4TreeSegmentationErrorCode.RESOURCE_LIMIT, field_name="maximum_grid_cells"
        )
    cols = max(1, cols)
    rows = max(1, rows)
    total_cells = cols * rows
    if total_cells > config.maximum_grid_cells:
        raise V4TreeSegmentationError(
            code=V4TreeSegmentationErrorCode.RESOURCE_LIMIT, field_name="maximum_grid_cells"
        )
    col = np.floor((xy[:, 0] - origin_x) / scales.lxy).astype(np.int64)
    row = np.floor((xy[:, 1] - origin_y) / scales.lxy).astype(np.int64)
    cell_ids = row * cols + col
    cell_ids.flags.writeable = False
    occupancy: dict[int, list[int]] = {}
    for position in range(count):
        occupancy.setdefault(int(cell_ids[position]), []).append(position)
    summaries: dict[int, _CellSummary] = {}
    col_index: dict[int, list[int]] = {}
    cell_positions: dict[int, list[int]] = {}
    for cell_id in sorted(occupancy):
        positions = sorted(occupancy[cell_id])
        cell_positions[cell_id] = positions
        cid = cell_id
        ccol = cid % cols
        c_row = cid // cols
        hag_vals = tuple(float(population.hag_m[pos]) for pos in positions)
        q75 = _quantile(sorted(hag_vals), 0.75)
        chm = max(hag_vals) if len(hag_vals) >= TRUNK_BIN_MIN_POINTS else q75
        center = (
            origin_x + (ccol + 0.5) * scales.lxy,
            origin_y + (c_row + 0.5) * scales.lxy,
        )
        summaries[cid] = _CellSummary(
            cell_id=cid,
            col=ccol,
            row=c_row,
            count=len(positions),
            hag_values=hag_vals,
            hag_q75=q75,
            hag_max=max(hag_vals),
            chm_height=chm,
            cell_center=center,
            density=_raw_weight_sum(population, positions) / (scales.lxy * scales.lxy),
            min_original_index=int(population.original_index[positions[0]]),
        )
        col_index.setdefault(ccol, []).append(cid)
    for ccol in col_index:
        col_index[ccol] = sorted(col_index[ccol])
    return _Grid(
        cell_size=scales.lxy,
        origin_xy=(origin_x, origin_y),
        cols=cols,
        rows=rows,
        cell_id_to_summary=summaries,
        col_to_cell_ids=col_index,
        point_cell_ids=cell_ids,
        occupied_cell_ids=tuple(sorted(summaries)),
        cell_positions=cell_positions,
    )


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _cell_density(grid: _Grid, cell_id: int) -> float:
    summary = grid.cell_id_to_summary.get(cell_id)
    if summary is None:
        return 0.0
    return summary.density


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("median of an empty list is undefined")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


# ---------------------------------------------------------------------------
# Apex detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ApexProposal:
    """Internal accepted/rejected apex proposal before record construction."""

    candidate_id: str
    supporting_original_index: int
    position: int
    xyz_local: tuple[float, float, float]
    hag: float
    prominence_m: float | None
    plateau_cells: tuple[int, ...]
    plateau_point_count: int
    cell_point_count: int
    density: float
    neighborhood_median_density: float
    apex_support_score: float
    quality_flags: int
    reason_flags: int
    accepted: bool


def _smoothed_chm_height(
    grid: _Grid, cell_id: int
) -> float:
    """Deterministic 3x3 mean-smoothing of the support-filtered CHM."""

    summary = grid.cell_id_to_summary.get(cell_id)
    if summary is None:
        return 0.0
    heights = [summary.chm_height]
    for neighbor in grid.neighbors_8(cell_id):
        neighbor_summary = grid.cell_id_to_summary.get(neighbor)
        if neighbor_summary is not None:
            heights.append(neighbor_summary.chm_height)
    return sum(heights) / len(heights)


def _smoothed_chm_map(grid: _Grid) -> dict[int, float]:
    return {
        cell_id: _smoothed_chm_height(grid, cell_id)
        for cell_id in grid.occupied_cell_ids
    }


def _local_max_plateaus(grid: _Grid, height_by_cell: dict[int, float]) -> list[tuple[int, ...]]:
    """Return equal-height local-maximum plateau cell groups (deterministic)."""

    is_local_max: dict[int, bool] = {}
    for cell_id in grid.occupied_cell_ids:
        height = height_by_cell[cell_id]
        local = True
        for neighbor in grid.neighbors_8(cell_id):
            neighbor_height = height_by_cell.get(neighbor, -math.inf)
            if neighbor_height > height:
                local = False
                break
        is_local_max[cell_id] = local
    # Union equal-height adjacent local maxima into plateaus.
    parent: dict[int, int] = {}

    def find(cell_id: int) -> int:
        while parent[cell_id] != cell_id:
            parent[cell_id] = parent[parent[cell_id]]
            cell_id = parent[cell_id]
        return cell_id

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    for cell_id in grid.occupied_cell_ids:
        parent[cell_id] = cell_id
    for cell_id in grid.occupied_cell_ids:
        if not is_local_max[cell_id]:
            continue
        height = height_by_cell[cell_id]
        for neighbor in grid.neighbors_8(cell_id):
            if neighbor in height_by_cell and is_local_max.get(neighbor, False):
                if height_by_cell[neighbor] == height:
                    union(cell_id, neighbor)
    plateau_groups: dict[int, list[int]] = {}
    for cell_id in grid.occupied_cell_ids:
        if is_local_max[cell_id]:
            root = find(cell_id)
            plateau_groups.setdefault(root, []).append(cell_id)
    result = []
    for _root in sorted(plateau_groups):
        cells = sorted(plateau_groups[_root])
        result.append(tuple(cells))
    return result


def _annulus_median_height(
    grid: _Grid,
    cells: tuple[int, ...],
    scales: V4AlgorithmScales,
    height_by_cell: dict[int, float],
) -> float | None:
    center_x = sum(grid.cell_id_to_summary[c].cell_center[0] for c in cells) / len(cells)
    center_y = sum(grid.cell_id_to_summary[c].cell_center[1] for c in cells) / len(cells)
    heights: list[float] = []
    for cell_id in grid.occupied_cell_ids:
        if cell_id in cells:
            continue
        summary = grid.cell_id_to_summary[cell_id]
        dx = summary.cell_center[0] - center_x
        dy = summary.cell_center[1] - center_y
        distance = math.hypot(dx, dy)
        if scales.la <= distance <= 2.0 * scales.la:
            heights.append(height_by_cell.get(cell_id, summary.chm_height))
    if not heights:
        return None
    return _median(heights)


def _apex_neighborhood_median_density(
    grid: _Grid, cell_id: int, scales: V4AlgorithmScales
) -> float:
    center = grid.cell_id_to_summary[cell_id].cell_center
    densities: list[float] = []
    for other_id in grid.occupied_cell_ids:
        other = grid.cell_id_to_summary[other_id]
        distance = math.hypot(
            other.cell_center[0] - center[0], other.cell_center[1] - center[1]
        )
        if distance <= scales.la:
            densities.append(other.density)
    return _median(densities)


def _detect_apex_candidates(
    population: _NormalPopulation,
    grid: _Grid,
    scales: V4AlgorithmScales,
    config: V4TreeSegmentationConfig,
) -> list[V4ApexCandidateRecord]:
    smoothed = _smoothed_chm_map(grid)
    plateaus = _local_max_plateaus(grid, smoothed)
    proposals: list[_ApexProposal] = []
    for plateau in plateaus:
        positions: list[int] = []
        plateau_point_count = 0
        for cell_id in plateau:
            summary = grid.cell_id_to_summary[cell_id]
            plateau_point_count += summary.count
            for _pos, _oidx in enumerate(_positions_in_cell(grid, cell_id)):
                positions.append(_pos)
        # Select the supporting point: HAG desc, original_index asc.
        best_position = None
        best_hag = -math.inf
        best_index = None
        for cell_id in plateau:
            for position in _positions_in_cell(grid, cell_id):
                hag = float(population.hag_m[position])
                original = int(population.original_index[position])
                if hag > best_hag or (hag == best_hag and original < best_index):
                    best_hag = hag
                    best_position = position
                    best_index = original
        assert best_position is not None
        # ``rho_apex`` is the density of the supporting apex point's measured
        # cell.  A multi-cell plateau's lexicographically first cell is not
        # necessarily the cell selected by HAG/original-index tie authority.
        main_cell = int(grid.point_cell_ids[best_position])
        summary = grid.cell_id_to_summary[main_cell]
        prominence = _annulus_median_height(grid, plateau, scales, smoothed)
        plateau_height = max(smoothed[cell_id] for cell_id in plateau)
        prominence_m = None if prominence is None else plateau_height - prominence
        neighborhood_density = _apex_neighborhood_median_density(grid, main_cell, scales)
        # Apex support uses prominence, never the absolute CHM height.  The
        # prominence term is only meaningful when prominence was actually
        # measured; a missing-prominence candidate cannot be formally accepted.
        prominence_support = c01(
            (prominence_m if prominence_m is not None else 0.0) / scales.la
        )
        apex_support = (
            0.60 * prominence_support
            + 0.40 * safe_ratio(summary.density, neighborhood_density, scales.rho_floor)
        )
        quality = 0
        reason = 0
        low_support = plateau_point_count < TRUNK_BIN_MIN_POINTS
        if low_support:
            quality |= int(V4ApexQualityFlag.SPARSE_SUPPORT)
        if prominence_m is None:
            # Required apex-support channel missing: the candidate cannot be
            # formally accepted.  Suppression here is NOT a scene override; the
            # affected population follows the frozen REVIEW path downstream.
            quality |= int(V4ApexQualityFlag.PROMINENCE_MISSING)
            quality |= int(V4ApexQualityFlag.EDGE_LIMITED)
            quality |= int(V4ApexQualityFlag.SUPPRESSED)
            accepted = False
        else:
            accepted = prominence_m >= config.minimum_apex_prominence_m
            if not accepted:
                quality |= int(V4ApexQualityFlag.SUPPRESSED)
        if low_support:
            # The support rule rejects unreliable low-support extremes.
            accepted = False
            quality |= int(V4ApexQualityFlag.SUPPRESSED)
        if len(plateau) > 1:
            reason |= int(V4ApexQualityFlag.PLATEAU_COLLAPSE)
        candidate_id = f"apex-{best_index:08d}"
        proposals.append(
            _ApexProposal(
                candidate_id=candidate_id,
                supporting_original_index=best_index,
                position=best_position,
                xyz_local=tuple(float(v) for v in population.xyz_local_m[best_position]),
                hag=best_hag,
                prominence_m=prominence_m,
                plateau_cells=plateau,
                plateau_point_count=plateau_point_count,
                cell_point_count=summary.count,
                density=summary.density,
                neighborhood_median_density=neighborhood_density,
                apex_support_score=apex_support,
                quality_flags=quality,
                reason_flags=reason,
                accepted=accepted,
            )
        )
    # Deterministic ordering: HAG desc, prominence desc, original_index asc.
    proposals.sort(
        key=lambda proposal: (
            -proposal.hag,
            -(
                proposal.prominence_m
                if proposal.prominence_m is not None
                else -1.0
            ),
            proposal.supporting_original_index,
        )
    )
    records: list[V4ApexCandidateRecord] = []
    for proposal in proposals:
        records.append(
            _finalize_record(
                V4ApexCandidateRecord,
                "diagnostic_digest",
                candidate_id=proposal.candidate_id,
                supporting_original_index=proposal.supporting_original_index,
                xyz_local_m=proposal.xyz_local,
                hag_m=proposal.hag,
                prominence_m=proposal.prominence_m,
                prominence_available=proposal.prominence_m is not None,
                plateau_support_count=proposal.plateau_point_count,
                cell_point_count=proposal.cell_point_count,
                density_points_per_m2=proposal.density,
                neighborhood_median_density_points_per_m2=proposal.neighborhood_median_density,
                apex_support_score=proposal.apex_support_score,
                quality_flags=proposal.quality_flags,
                reason_flags=proposal.reason_flags,
            )
        )
    return records


def _with_record_digest(record: Any, digest: str) -> Any:
    clone = type(record).__new__(type(record))
    for name in type(record).__dataclass_fields__:
        if name == "diagnostic_digest" or name == "evidence_digest":
            object.__setattr__(clone, name, digest)
        else:
            object.__setattr__(clone, name, getattr(record, name))
    return clone


def _finalize_record(record_type: type, digest_field: str, **fields: Any) -> Any:
    """Construct a validated digest-bound record with its self-computed digest."""

    placeholder = record_type.__new__(record_type)
    for name, value in fields.items():
        object.__setattr__(placeholder, name, value)
    object.__setattr__(placeholder, digest_field, "0" * 64)
    digest = placeholder.computed_digest()
    return record_type(**fields, **{digest_field: digest})


def _positions_in_cell(grid: _Grid, cell_id: int) -> list[int]:
    # Direct lookup from the persistent cell -> ordered positions mapping built
    # during grid construction (ascending point position order).
    return grid.cell_positions.get(cell_id, [])


# ---------------------------------------------------------------------------
# Trunk candidate detection (B2.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Voxel:
    cell_id: int
    hag_bin_id: int
    positions: tuple[int, ...]
    centroid_xyz: tuple[float, float, float]
    density: float
    min_original_index: int


def _build_voxels(
    population: _NormalPopulation,
    grid: _Grid,
    scales: V4AlgorithmScales,
) -> dict[tuple[int, int], _Voxel]:
    voxels: dict[tuple[int, int], list[int]] = {}
    count = int(population.original_index.shape[0])
    for position in range(count):
        hag = float(population.hag_m[position])
        bin_id = int(math.floor(hag / scales.btr))
        cell_id = int(grid.point_cell_ids[position])
        voxels.setdefault((cell_id, bin_id), []).append(position)
    result: dict[tuple[int, int], _Voxel] = {}
    for key in sorted(voxels):
        positions = sorted(voxels[key])
        if len(positions) < TRUNK_BIN_MIN_POINTS:
            continue
        centroid = tuple(
            float(np.mean(population.xyz_local_m[positions, axis]))
            for axis in range(3)
        )
        result[key] = _Voxel(
            cell_id=key[0],
            hag_bin_id=key[1],
            positions=tuple(positions),
            centroid_xyz=centroid,
            density=_raw_weight_sum(population, positions) / (scales.lxy * scales.lxy),
            min_original_index=int(population.original_index[positions[0]]),
        )
    return result


def _voxel_adjacent(
    voxel_a: _Voxel,
    voxel_b: _Voxel,
    grid: _Grid,
    scales: V4AlgorithmScales,
) -> bool:
    row_a = voxel_a.cell_id // grid.cols
    col_a = voxel_a.cell_id % grid.cols
    row_b = voxel_b.cell_id // grid.cols
    col_b = voxel_b.cell_id % grid.cols
    if (
        abs(row_a - row_b) > 1
        or abs(col_a - col_b) > 1
        or abs(voxel_a.hag_bin_id - voxel_b.hag_bin_id) > 1
    ):
        return False
    if row_a == row_b and col_a == col_b and voxel_a.hag_bin_id == voxel_b.hag_bin_id:
        return False
    dx = voxel_a.centroid_xyz[0] - voxel_b.centroid_xyz[0]
    dy = voxel_a.centroid_xyz[1] - voxel_b.centroid_xyz[1]
    xy_distance = math.hypot(dx, dy)
    return xy_distance <= scales.ltr_xy


def _voxel_adjacent_neighbor_keys(
    voxels: dict[tuple[int, int], _Voxel],
    grid: _Grid,
    scales: V4AlgorithmScales,
    key: tuple[int, int],
    voxel: _Voxel,
) -> list[tuple[int, int]]:
    """Return ``key``'s adjacent neighbour keys in the frozen enumeration order.

    ``(drow, dcol, dbin)`` over ``{-1, 0, 1}``, skipping the self cell, with
    ``_voxel_adjacent`` as the exact acceptance predicate.  It lets callers
    stream the voxel graph instead of materializing every edge at once.
    """
    cell_id = voxel.cell_id
    col = cell_id % grid.cols
    row = cell_id // grid.cols
    bin_id = voxel.hag_bin_id
    neighbors: list[tuple[int, int]] = []
    for drow in (-1, 0, 1):
        nrow = row + drow
        if nrow < 0 or nrow >= grid.rows:
            continue
        for dcol in (-1, 0, 1):
            ncol = col + dcol
            if ncol < 0 or ncol >= grid.cols:
                continue
            for dbin in (-1, 0, 1):
                if drow == 0 and dcol == 0 and dbin == 0:
                    continue
                neighbor_key = (nrow * grid.cols + ncol, bin_id + dbin)
                neighbor = voxels.get(neighbor_key)
                if neighbor is None:
                    continue
                if _voxel_adjacent(voxel, neighbor, grid, scales):
                    neighbors.append(neighbor_key)
    return neighbors


def _connected_voxel_components(
    voxels: dict[tuple[int, int], _Voxel],
    grid: _Grid,
    scales: V4AlgorithmScales,
    maximum_local_edges: int | None = None,
) -> list[list[_Voxel]]:
    keys = sorted(voxels)
    # Resource-aware guard: with streamed edges the peak retained working
    # memory is the union-find parent dictionary (O(occupied voxels)); the
    # edges themselves are never materialized.
    if maximum_local_edges is not None:
        if (
            _projected_bytes(len(voxels), _PROJECTED_DICT_ENTRY_BYTES)
            > _MAXIMUM_LOCAL_WORKING_BYTES
        ):
            _raise_local_edge_resource_limit()
    parent = {key: key for key in keys}

    def find(key: tuple[int, int]) -> tuple[int, int]:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    for key_a in keys:
        for key_b in _voxel_adjacent_neighbor_keys(voxels, grid, scales, key_a, voxels[key_a]):
            union(key_a, key_b)
    groups: dict[tuple[int, int], list[_Voxel]] = {}
    for key in keys:
        root = find(key)
        groups.setdefault(root, []).append(voxels[key])
    result = []
    for _root in sorted(groups):
        group = sorted(groups[_root], key=lambda voxel: voxel.min_original_index)
        result.append(group)
    return result


def _pca_axis(points: np.ndarray) -> tuple[tuple[float, float, float], np.ndarray, np.ndarray]:
    """Return (principal_axis, centroid, eigen_lambda) for measured points.

    The symmetric 3x3 covariance is accumulated with ``math.fsum`` over rows in
    canonical original-index order (callers pass points sorted that way), so
    the centroid/covariance do not change under row ordering.  The
    eigendecomposition uses ``np.linalg.eigh``.
    """
    count = points.shape[0]
    centroid = np.asarray(
        [
            math.fsum(float(points[position, axis]) for position in range(count))
            / max(count, 1)
            for axis in range(3)
        ],
        dtype=np.float64,
    )
    covariance = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(i, 3):
            covariance[i, j] = (
                math.fsum(
                    float(points[position, i] - centroid[i])
                    * float(points[position, j] - centroid[j])
                    for position in range(count)
                )
                / max(count, 1)
            )
    for i in range(3):
        for j in range(i + 1, 3):
            covariance[j, i] = covariance[i, j]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    axis = np.asarray(eigenvectors[:, 0], dtype=np.float64)
    axis = _canonical_axis_sign(axis)
    return (tuple(float(v) for v in axis), centroid, eigenvalues)


def _canonical_axis_sign(axis: np.ndarray) -> np.ndarray:
    value = np.asarray(axis, dtype=np.float64).copy()
    if value[2] < 0.0:
        value = -value
    elif value[2] == 0.0:
        for index in (0, 1):
            if value[index] < 0.0:
                value = -value
                break
    return value


def _distance_to_axis(point: np.ndarray, centroid: np.ndarray, axis: Any) -> float:
    axis_array = np.asarray(axis, dtype=np.float64)
    delta = point - centroid
    projection = float(np.dot(delta, axis_array))
    closest = centroid + projection * axis_array
    return float(np.linalg.norm(point - closest))


def _trunk_axis_at_hag(
    centroid: np.ndarray, axis: np.ndarray, hag: float
) -> tuple[float, float]:
    """XY intersection of the fitted infinite axis at a given HAG."""

    direction_z = axis[2]
    if abs(direction_z) < PCA_ABS_EPS:
        return (float(centroid[0]), float(centroid[1]))
    t = (hag - centroid[2]) / direction_z
    return (float(centroid[0] + t * axis[0]), float(centroid[1] + t * axis[1]))


def _detect_trunk_candidates(
    population: _NormalPopulation,
    grid: _Grid,
    scales: V4AlgorithmScales,
    config: V4TreeSegmentationConfig,
) -> tuple[list[V4TrunkCandidateRecord], dict[str, tuple[int, int]]]:
    voxels = _build_voxels(population, grid, scales)
    components = _connected_voxel_components(
        voxels, grid, scales, config.maximum_local_edges
    )
    records: list[V4TrunkCandidateRecord] = []
    id_lookup: dict[str, tuple[int, int]] = {}
    for component in components:
        positions = sorted(
            {position for voxel in component for position in voxel.positions}
        )
        distinct_bins = {voxel.hag_bin_id for voxel in component}
        if (
            len(positions) < TRUNK_COMPONENT_MIN_POINTS
            or len(distinct_bins) < TRUNK_COMPONENT_MIN_BINS
        ):
            quality = int(V4TrunkQualityFlag.INSUFFICIENT_SUPPORT)
            if len(positions) < TRUNK_COMPONENT_MIN_POINTS:
                quality |= int(V4TrunkQualityFlag.WEAK_CORE_SUPPORT)
            records.append(
                _suppressed_trunk_record(component, positions, quality)
            )
            continue
        points = np.asarray([population.xyz_local_m[pos] for pos in positions], dtype=np.float64)
        axis, centroid, lambdas = _pca_axis(points)
        if lambdas[0] <= PCA_ABS_EPS or (
            lambdas[0] - lambdas[1]
        ) / max(lambdas[0], PCA_ABS_EPS) < PCA_REL_GAP_MIN:
            records.append(
                _suppressed_trunk_record(
                    component,
                    positions,
                    int(V4TrunkQualityFlag.PCA_DEGENERATE)
                    | int(V4TrunkQualityFlag.INSUFFICIENT_SUPPORT),
                )
            )
            continue
        core_radius = TRUNK_CORE_RADIUS_FACTOR * scales.ltr_xy
        core_positions = [
            position
            for position in positions
            if _distance_to_axis(population.xyz_local_m[position], centroid, axis) <= core_radius
        ]
        # Per-bin core occupancy.  A bin counts as OCCUPIED for the core support
        # gate and the TC numerator only when its core point_count meets
        # TRUNK_BIN_MIN_POINTS (>=3).  One or two core points in a bin do not
        # count toward TC.
        core_by_bin: dict[int, list[int]] = {}
        for position in core_positions:
            bin_id = int(math.floor(float(population.hag_m[position]) / scales.btr))
            core_by_bin.setdefault(bin_id, []).append(position)
        occupied_core_bins = {
            bin_id
            for bin_id, bin_positions in core_by_bin.items()
            if len(bin_positions) >= TRUNK_BIN_MIN_POINTS
        }
        if (
            len(core_positions) < TRUNK_COMPONENT_MIN_POINTS
            or len(occupied_core_bins) < TRUNK_COMPONENT_MIN_BINS
        ):
            records.append(
                _suppressed_trunk_record(
                    component,
                    positions,
                    int(V4TrunkQualityFlag.WEAK_CORE_SUPPORT)
                    | int(V4TrunkQualityFlag.INSUFFICIENT_SUPPORT),
                )
            )
            continue
        core_points = np.asarray(
            [population.xyz_local_m[pos] for pos in core_positions], dtype=np.float64
        )
        refit_axis, refit_centroid, refit_lambdas = _pca_axis(core_points)
        if refit_lambdas[0] <= PCA_ABS_EPS or (
            refit_lambdas[0] - refit_lambdas[1]
        ) / max(refit_lambdas[0], PCA_ABS_EPS) < PCA_REL_GAP_MIN:
            records.append(
                _suppressed_trunk_record(
                    component,
                    positions,
                    int(V4TrunkQualityFlag.PCA_DEGENERATE)
                    | int(V4TrunkQualityFlag.WEAK_CORE_SUPPORT),
                )
            )
            continue
        # Per-bin summaries over the core set.  evaluated_bins spans the full
        # [k_min, k_max] core range; only bins meeting TRUNK_BIN_MIN_POINTS
        # count toward the TC numerator.
        k_min = min(core_by_bin)
        k_max = max(core_by_bin)
        evaluated_bins = k_max - k_min + 1
        occupied_required_bins = len(occupied_core_bins)
        if evaluated_bins <= 0:
            raise V4TreeSegmentationError(
                code=V4TreeSegmentationErrorCode.INTERNAL_INVARIANT,
                field_name="evaluated_bins",
            )
        max_drift = 0.0
        for bin_id in sorted(core_by_bin):
            bin_positions = sorted(core_by_bin[bin_id])
            bin_centroid = tuple(
                float(np.mean(population.xyz_local_m[bin_positions, axis]))
                for axis in range(3)
            )
            hag_values = sorted(float(population.hag_m[pos]) for pos in bin_positions)
            median_hag = _median(hag_values)
            axis_xy = _trunk_axis_at_hag(refit_centroid, refit_axis, median_hag)
            drift = math.hypot(bin_centroid[0] - axis_xy[0], bin_centroid[1] - axis_xy[1])
            max_drift = max(max_drift, drift)
        core_hag_values = [float(population.hag_m[pos]) for pos in core_positions]
        core_hag_min = min(core_hag_values)
        core_hag_max = max(core_hag_values)
        vertical_span = core_hag_max - core_hag_min
        tv = c01(vertical_span / scales.ltr_z)
        vz = abs(float(refit_axis[2]))
        ta = c01((vz - 0.50) / 0.50)
        ts = near(max_drift, scales.ltr_xy)
        tc = occupied_required_bins / evaluated_bins
        score = trunk_candidate_score(tv, ta, ts, tc)
        state = trunk_candidate_state(score)
        representative = sorted(
            core_positions,
            key=lambda position: (
                _distance_to_axis(population.xyz_local_m[position], refit_centroid, refit_axis),
                int(population.original_index[position]),
            ),
        )[0]
        representative_index = int(population.original_index[representative])
        core_member_digest = _member_digest(
            [population.original_index[pos] for pos in core_positions]
        )
        candidate_id = f"trunk-{representative_index:08d}"
        quality = 0
        reason = 0
        if state is V4TrunkCandidateState.WEAK:
            quality |= int(V4TrunkQualityFlag.LOW_SCORE)
        if state is V4TrunkCandidateState.SUPPRESSED:
            quality |= int(V4TrunkQualityFlag.LOW_SCORE)
        record = _finalize_record(
            V4TrunkCandidateRecord,
            "diagnostic_digest",
            candidate_id=candidate_id,
            representative_original_index=representative_index,
            supporting_original_index_digest=core_member_digest,
            centroid_xyz=tuple(float(v) for v in refit_centroid),
            bounds_min_xyz=tuple(
                float(np.min(population.xyz_local_m[core_positions, axis]))
                for axis in range(3)
            ),
            bounds_max_xyz=tuple(
                float(np.max(population.xyz_local_m[core_positions, axis]))
                for axis in range(3)
            ),
            refit_principal_axis=tuple(float(v) for v in refit_axis),
            core_hag_min_m=core_hag_min,
            core_hag_max_m=core_hag_max,
            observed_vertical_span_m=vertical_span,
            k_min=k_min,
            k_max=k_max,
            evaluated_bins=evaluated_bins,
            occupied_required_bins=occupied_required_bins,
            max_bin_centroid_xy_drift_m=max_drift,
            tv=tv,
            ta=ta,
            ts=ts,
            tc=tc,
            trunk_candidate_score=score,
            candidate_state=state,
            quality_flags=quality,
            reason_flags=reason,
        )
        records.append(record)
        id_lookup[record.candidate_id] = (representative, tuple(core_positions))
    records.sort(key=lambda record: record.candidate_id)
    return records, id_lookup


def _suppressed_trunk_record(
    component: list[_Voxel],
    positions: list[int],
    quality: int,
) -> V4TrunkCandidateRecord:
    core_hag_values = [float(voxel.centroid_xyz[2]) for voxel in component]
    centroid = tuple(
        sum(
            voxel.centroid_xyz[i] for voxel in component
        )
        / max(len(component), 1)
        for i in range(3)
    )
    representative_index = min(
        (int(voxel.min_original_index) for voxel in component), default=0
    )
    return _finalize_record(
        V4TrunkCandidateRecord,
        "diagnostic_digest",
        candidate_id=f"trunk-{representative_index:08d}",
        representative_original_index=representative_index,
        supporting_original_index_digest=_member_digest([]),
        centroid_xyz=centroid,
        bounds_min_xyz=centroid,
        bounds_max_xyz=centroid,
        refit_principal_axis=(0.0, 0.0, 1.0),
        core_hag_min_m=min(core_hag_values, default=0.0),
        core_hag_max_m=max(core_hag_values, default=0.0),
        observed_vertical_span_m=max(core_hag_values, default=0.0)
        - min(core_hag_values, default=0.0),
        k_min=0,
        k_max=0,
        evaluated_bins=1,
        occupied_required_bins=1,
        max_bin_centroid_xy_drift_m=0.0,
        tv=0.0,
        ta=0.0,
        ts=0.0,
        tc=1.0,
        trunk_candidate_score=0.0,
        candidate_state=V4TrunkCandidateState.SUPPRESSED,
        quality_flags=quality,
        reason_flags=quality,
    )


# ---------------------------------------------------------------------------
# Structural path (B2.2A) and trunk-apex association (B2.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _AssociationResult:
    evidence: V4TrunkApexAssociationEvidence
    path: V4StructuralPathEvidence | None
    tvc_bins: float | None


def _trunk_apex_xy_distance(
    trunk: V4TrunkCandidateRecord,
    apex: V4ApexCandidateRecord,
) -> float:
    axis_xy = _trunk_axis_at_hag(
        np.asarray(trunk.centroid_xyz),
        np.asarray(trunk.refit_principal_axis),
        apex.hag_m,
    )
    return math.hypot(axis_xy[0] - apex.xyz_local_m[0], axis_xy[1] - apex.xyz_local_m[1])


def _corridor_voxels(
    trunk_representative_xyz: tuple[float, float, float],
    trunk_hag_min: float,
    trunk_hag_max: float,
    apex: V4ApexCandidateRecord,
    voxels: dict[tuple[int, int], _Voxel],
    scales: V4AlgorithmScales,
) -> dict[tuple[int, int], _Voxel]:
    # The structural corridor starts from the trunk representative -> apex
    # segment (not the trunk centroid), per the frozen B2.2A local domain.
    trunk_x = float(trunk_representative_xyz[0])
    trunk_y = float(trunk_representative_xyz[1])
    apex_x = float(apex.xyz_local_m[0])
    apex_y = float(apex.xyz_local_m[1])
    segment_dx = apex_x - trunk_x
    segment_dy = apex_y - trunk_y
    segment_length = math.hypot(segment_dx, segment_dy)
    hag_min = min(trunk_hag_min, apex.hag_m) - scales.btr
    hag_max = max(trunk_hag_max, apex.hag_m) + scales.btr
    corridor: dict[tuple[int, int], _Voxel] = {}
    for key, voxel in voxels.items():
        voxel_center = voxel.centroid_xyz
        if not (hag_min <= voxel_center[2] <= hag_max):
            continue
        if segment_length <= PCA_ABS_EPS:
            distance = math.hypot(
                voxel_center[0] - trunk_x, voxel_center[1] - trunk_y
            )
        else:
            t = (
                (voxel_center[0] - trunk_x) * segment_dx
                + (voxel_center[1] - trunk_y) * segment_dy
            ) / (segment_length * segment_length)
            t = min(1.0, max(0.0, t))
            closest_x = trunk_x + t * segment_dx
            closest_y = trunk_y + t * segment_dy
            distance = math.hypot(voxel_center[0] - closest_x, voxel_center[1] - closest_y)
        if distance <= scales.lpath_xy:
            corridor[key] = voxel
    return corridor


def _voxel_edge_cost(
    voxel_u: _Voxel,
    voxel_v: _Voxel,
    scales: V4AlgorithmScales,
) -> float:
    center_u = np.asarray(voxel_u.centroid_xyz)
    center_v = np.asarray(voxel_v.centroid_xyz)
    step = float(np.linalg.norm(center_u - center_v))
    downward_fraction = c01(
        max(0.0, voxel_u.centroid_xyz[2] - voxel_v.centroid_xyz[2])
        / max(scales.btr, scales.ls)
    )
    density_similarity = min(
        safe_ratio(voxel_u.density, voxel_v.density, scales.rho_floor),
        safe_ratio(voxel_v.density, voxel_u.density, scales.rho_floor),
    )
    return step * (
        1.0
        + STRUCTURAL_PATH_DOWNWARD_WEIGHT * downward_fraction
        + STRUCTURAL_PATH_DENSITY_WEIGHT * (1.0 - density_similarity)
    )


def _node_key(voxel: _Voxel) -> str:
    return f"{voxel.cell_id}:{voxel.hag_bin_id}:{voxel.min_original_index}"


def _structural_path(
    trunk: V4TrunkCandidateRecord,
    apex: V4ApexCandidateRecord,
    trunk_core_positions: tuple[int, ...],
    corridor: dict[tuple[int, int], _Voxel],
    voxels: dict[tuple[int, int], _Voxel],
    population: _NormalPopulation,
    grid: _Grid,
    scales: V4AlgorithmScales,
    resource_budget: int,
    maximum_local_edges: int,
) -> tuple[V4StructuralPathEvidence, bool]:
    """Run a deterministic sparse Dijkstra in the local corridor.

    Returns ``(evidence, inventory_complete)``.  Exceeding the frozen resource
    ceilings hard-fails; it never returns a truncated partial path.
    """
    if len(corridor) == 0:
        return (
            _path_evidence_not_available(trunk, apex, scales),
            True,
        )
    # Frozen source set: trunk upper-core support nodes only (the upper half
    # of the occupied core bins), never every core bin.
    trunk_core_bins = sorted(
        {
            int(math.floor(float(population.hag_m[pos]) / scales.btr))
            for pos in trunk_core_positions
        }
    )
    if not trunk_core_bins:
        return (
            _path_evidence_not_available(trunk, apex, scales),
            True,
        )
    k_upper = trunk_core_bins[len(trunk_core_bins) // 2]
    trunk_upper_bins = {bin_id for bin_id in trunk_core_bins if bin_id >= k_upper}
    # Frozen target set: apex-neighbourhood support nodes (apex La XY
    # neighbourhood), not merely ``HAG within +/- 1 bin``.
    apex_xy = np.asarray(apex.xyz_local_m[:2])
    source_keys: list[tuple[int, int]] = []
    target_keys: list[tuple[int, int]] = []
    trunk_core_position_set = set(trunk_core_positions)
    for key, voxel in corridor.items():
        if voxel.hag_bin_id in trunk_upper_bins and any(
            position in trunk_core_position_set for position in voxel.positions
        ):
            source_keys.append(key)
        if (
            math.hypot(
                voxel.centroid_xyz[0] - apex_xy[0],
                voxel.centroid_xyz[1] - apex_xy[1],
            )
            <= scales.la
        ):
            target_keys.append(key)
    source_keys.sort()
    target_keys.sort()
    if not source_keys:
        return (
            _path_evidence_not_available(trunk, apex, scales),
            True,
        )
    node_keys: dict[tuple[int, int], str] = {key: _node_key(corridor[key]) for key in corridor}
    # Edges are streamed from the frozen 26-neighbour enumeration and never
    # materialized as a scene-wide (or corridor-wide) adjacency dict.  The peak
    # retained working memory is the Dijkstra ``best``/``heap`` dictionaries,
    # both O(corridor voxels).
    if maximum_local_edges is not None:
        if (
            _projected_bytes(len(corridor), _PROJECTED_DICT_ENTRY_BYTES)
            > _MAXIMUM_LOCAL_WORKING_BYTES
        ):
            _raise_local_edge_resource_limit()
    edge_count = 0
    for key_a in corridor:
        voxel_a = corridor[key_a]
        for _neighbor_key in _voxel_adjacent_neighbor_keys(
            corridor, grid, scales, key_a, voxel_a
        ):
            edge_count += 1
    edge_count //= 2  # every undirected corridor edge was counted from both ends
    if edge_count > resource_budget:
        raise V4TreeSegmentationError(
            code=V4TreeSegmentationErrorCode.RESOURCE_LIMIT,
            field_name="maximum_path_evaluations",
        )
    # Every edge must be a finite, strictly positive step whose edge_cost is
    # finite and >= step; otherwise the local inventory fails closed.
    for key_a in corridor:
        voxel_a = corridor[key_a]
        for key_b in _voxel_adjacent_neighbor_keys(
            corridor, grid, scales, key_a, voxel_a
        ):
            edge_cost = _voxel_edge_cost(voxel_a, corridor[key_b], scales)
            step = float(
                np.linalg.norm(
                    np.asarray(voxel_a.centroid_xyz)
                    - np.asarray(corridor[key_b].centroid_xyz)
                )
            )
            if (
                not math.isfinite(step)
                or step <= 0.0
                or not math.isfinite(edge_cost)
                or edge_cost < step
            ):
                return (
                    _path_evidence_not_available(trunk, apex, scales),
                    False,
                )
    # Dijkstra with the frozen deterministic heap key
    # (total_cost, hop_count, predecessor_node_key, current_node_key).
    best: dict[tuple[int, int], tuple[float, int, tuple[int, int] | None]] = {}
    heap: list[tuple[float, int, str, str, tuple[int, int]]] = []
    for source in source_keys:
        heap.append((0.0, 0, "", node_keys[source], source))
        best[source] = (0.0, 0, None)
    heapq.heapify(heap)
    popped = 0
    while heap:
        total_cost, hop_count, pred_key, _current_key, current = heapq.heappop(heap)
        current_best = best.get(current)
        if current_best is None:
            continue
        best_pred_key = (
            "" if current_best[2] is None else node_keys[current_best[2]]
        )
        if (total_cost, hop_count, pred_key) != (
            current_best[0],
            current_best[1],
            best_pred_key,
        ):
            continue
        popped += 1
        if popped > resource_budget:
            raise V4TreeSegmentationError(
                code=V4TreeSegmentationErrorCode.RESOURCE_LIMIT,
                field_name="maximum_path_evaluations",
            )
        if current in target_keys:
            break
        for neighbor in _voxel_adjacent_neighbor_keys(
            corridor, grid, scales, current, corridor[current]
        ):
            step = _voxel_edge_cost(corridor[current], corridor[neighbor], scales)
            new_cost = total_cost + step
            new_hops = hop_count + 1
            previous_entry = best.get(neighbor)
            previous_key = (
                (math.inf, math.inf, "")
                if previous_entry is None
                else (
                    previous_entry[0],
                    previous_entry[1],
                    ""
                    if previous_entry[2] is None
                    else node_keys[previous_entry[2]],
                )
            )
            if (new_cost, new_hops, node_keys[current]) < previous_key:
                best[neighbor] = (new_cost, new_hops, current)
                heapq.heappush(
                    heap,
                    (
                        new_cost,
                        new_hops,
                        node_keys[current],
                        node_keys[neighbor],
                        neighbor,
                    ),
                )
    reached_targets = [key for key in target_keys if key in best]
    if not reached_targets:
        return (
            _finalize_record(
                V4StructuralPathEvidence,
                "evidence_digest",
                path_id=f"path-{trunk.candidate_id}-{apex.candidate_id}",
                trunk_candidate_id=trunk.candidate_id,
                apex_candidate_id=apex.candidate_id,
                domain_bounds_min_xyz=(0.0, 0.0, 0.0),
                domain_bounds_max_xyz=(0.0, 0.0, 0.0),
                node_count=len(corridor),
                edge_count=edge_count,
                source_node_key=node_keys[source_keys[0]] if source_keys else None,
                target_node_key=None,
                path_node_key_sequence=None,
                straight_length_m=0.0,
                weighted_path_length_m=0.0,
                structural_path_efficiency=0.0,
                availability=V4StructuralPathAvailability.OBSERVED,
                resource_audit=f"corridor={len(corridor)}",
                quality_flags=int(V4TrunkQualityFlag.OCCLUDED_INVENTORY),
                reason_flags=int(V4TrunkQualityFlag.OCCLUDED_INVENTORY),
            ),
            True,
        )
    # Choose the canonical target (min cost, min hops, min key).
    reached_targets.sort(
        key=lambda key: (best[key][0], best[key][1], key)
    )
    target = reached_targets[0]
    total_cost, _hops, _pred = best[target]
    path_keys: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = target
    while cursor is not None:
        path_keys.append(cursor)
        entry = best[cursor]
        cursor = entry[2]
    path_keys.reverse()
    selected_source = path_keys[0]
    trunk_upper_anchor = np.asarray(corridor[selected_source].centroid_xyz)
    apex_xyz = np.asarray(apex.xyz_local_m)
    straight_length = float(np.linalg.norm(trunk_upper_anchor - apex_xyz))
    if straight_length <= scales.ls:
        # Frozen direct/near component rule: once measured connectivity exists,
        # a trunk upper anchor already within Ls of the apex has full path
        # efficiency.  Dividing by Ls here would incorrectly penalize it.
        efficiency = 1.0
    else:
        efficiency = c01(
            straight_length / max(total_cost, straight_length, scales.ls)
        )
    path_id = f"path-{trunk.candidate_id}-{apex.candidate_id}"
    evidence = _finalize_record(
        V4StructuralPathEvidence,
        "evidence_digest",
        path_id=path_id,
        trunk_candidate_id=trunk.candidate_id,
        apex_candidate_id=apex.candidate_id,
        domain_bounds_min_xyz=_corridor_bounds(corridor)[0],
        domain_bounds_max_xyz=_corridor_bounds(corridor)[1],
        node_count=len(corridor),
        edge_count=edge_count,
        source_node_key=node_keys[selected_source],
        target_node_key=node_keys[target],
        path_node_key_sequence=tuple(node_keys[key] for key in path_keys),
        straight_length_m=straight_length,
        weighted_path_length_m=total_cost,
        structural_path_efficiency=efficiency,
        availability=V4StructuralPathAvailability.OBSERVED,
        resource_audit=f"corridor={len(corridor)} edges={edge_count}",
        quality_flags=0,
        reason_flags=0,
    )
    return evidence, True


def _corridor_bounds(
    corridor: dict[tuple[int, int], _Voxel],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not corridor:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xs = [voxel.centroid_xyz[0] for voxel in corridor.values()]
    ys = [voxel.centroid_xyz[1] for voxel in corridor.values()]
    zs = [voxel.centroid_xyz[2] for voxel in corridor.values()]
    return (
        (min(xs), min(ys), min(zs)),
        (max(xs), max(ys), max(zs)),
    )


def _path_evidence_not_available(
    trunk: V4TrunkCandidateRecord,
    apex: V4ApexCandidateRecord,
    scales: V4AlgorithmScales,
) -> V4StructuralPathEvidence:
    return _finalize_record(
        V4StructuralPathEvidence,
        "evidence_digest",
        path_id=f"path-{trunk.candidate_id}-{apex.candidate_id}",
        trunk_candidate_id=trunk.candidate_id,
        apex_candidate_id=apex.candidate_id,
        domain_bounds_min_xyz=(0.0, 0.0, 0.0),
        domain_bounds_max_xyz=(0.0, 0.0, 0.0),
        node_count=0,
        edge_count=0,
        source_node_key=None,
        target_node_key=None,
        path_node_key_sequence=None,
        straight_length_m=0.0,
        weighted_path_length_m=None,
        structural_path_efficiency=None,
        availability=V4StructuralPathAvailability.NOT_AVAILABLE,
        resource_audit="incomplete",
        quality_flags=int(V4TrunkQualityFlag.OCCLUDED_INVENTORY),
        reason_flags=int(V4TrunkQualityFlag.OCCLUDED_INVENTORY),
    )


def _compute_association(
    trunk: V4TrunkCandidateRecord,
    trunk_core_positions: tuple[int, ...],
    trunk_representative_xyz: tuple[float, float, float],
    trunk_representative_hag: float,
    apex: V4ApexCandidateRecord,
    voxels: dict[tuple[int, int], _Voxel],
    population: _NormalPopulation,
    grid: _Grid,
    scales: V4AlgorithmScales,
    config: V4TreeSegmentationConfig,
) -> _AssociationResult:
    d_xy = _trunk_apex_xy_distance(trunk, apex)
    # OBSERVED_UNSUPPORTED_GEOMETRY: an apex at or below the trunk
    # representative HAG (both finite) is complete geometry, not missing.
    if apex.hag_m <= trunk_representative_hag:
        evidence = _finalize_record(
            V4TrunkApexAssociationEvidence,
            "diagnostic_digest",
            association_id=f"ta-{trunk.candidate_id}-{apex.candidate_id}",
            trunk_candidate_id=trunk.candidate_id,
            apex_candidate_id=apex.candidate_id,
            tx=near(d_xy, scales.lta),
            tx_available=True,
            td=0.0,
            td_available=False,
            tvc=0.0,
            tvc_available=False,
            tg=0.0,
            tg_available=False,
            tr=0.0,
            tr_available=False,
            association_score_available=True,
            association_score=0.0,
            association_state=V4TrunkApexAssociationState.UNSUPPORTED_ASSOCIATION,
            structural_path_evidence_reference=None,
            lookup_kind=V4AssociationLookupKind.EVALUATED,
            quality_flags=0,
            reason_flags=0,
        )
        return _AssociationResult(evidence, None, None)
    if d_xy > scales.lr:
        evidence = _finalize_record(
            V4TrunkApexAssociationEvidence,
            "diagnostic_digest",
            association_id=f"ta-{trunk.candidate_id}-{apex.candidate_id}",
            trunk_candidate_id=trunk.candidate_id,
            apex_candidate_id=apex.candidate_id,
            tx=0.0,
            tx_available=False,
            td=0.0,
            td_available=False,
            tvc=0.0,
            tvc_available=False,
            tg=0.0,
            tg_available=False,
            tr=0.0,
            tr_available=False,
            association_score_available=False,
            association_score=0.0,
            association_state=V4TrunkApexAssociationState.UNSUPPORTED_ASSOCIATION,
            structural_path_evidence_reference=None,
            lookup_kind=V4AssociationLookupKind.OBSERVED_OUT_OF_RADIUS,
            quality_flags=0,
            reason_flags=0,
        )
        return _AssociationResult(evidence, None, None)
    # TX channel.
    tx = near(d_xy, scales.lta)
    tx_available = True
    # TD channel: trunk representative -> apex vector against the canonical
    # upward trunk axis.
    trunk_axis = np.asarray(trunk.refit_principal_axis)
    to_apex = np.asarray(apex.xyz_local_m) - np.asarray(trunk_representative_xyz)
    vector_norm = float(np.linalg.norm(to_apex))
    if vector_norm <= PCA_ABS_EPS:
        td_available = False
        td = 0.0
    else:
        cosine = float(np.dot(to_apex / vector_norm, trunk_axis))
        td = c01(max(0.0, cosine))
        td_available = True
    # CSP structural path over the trunk-representative-rooted corridor; the
    # same local corridor supplies the TVc vertical-continuity bins.
    corridor = _corridor_voxels(
        trunk_representative_xyz,
        trunk.core_hag_min_m,
        trunk.core_hag_max_m,
        apex,
        voxels,
        scales,
    )
    # TVc_bins over the intermediate corridor.
    tvc_bins, tg, tvcb_available, tga_available, intermediate_bins = _vertical_continuity(
        trunk, apex, corridor, grid, scales
    )
    path_evidence, inventory_complete = _structural_path(
        trunk,
        apex,
        trunk_core_positions,
        corridor,
        voxels,
        population,
        grid,
        scales,
        config.maximum_path_evaluations,
        config.maximum_local_edges,
    )
    if path_evidence.availability is V4StructuralPathAvailability.OBSERVED:
        tvc = 0.5 * tvc_bins + 0.5 * path_evidence.structural_path_efficiency
        tvc_available = tvcb_available
    else:
        tvc = 0.0
        tvc_available = False
    tg_available = tga_available and inventory_complete
    # TR channel: trunk upper support voxels to apex neighborhood boundary links.
    tr, tr_available = _crown_endpoint_connectivity(
        trunk, apex, corridor, grid, scales, config.maximum_local_edges
    )
    all_observed = (
        tx_available and td_available and tvc_available and tg_available and tr_available
    )
    if all_observed:
        score = trunk_apex_score(tx, td, tvc, tg, tr)
        state = trunk_apex_state(score)
        score_available = True
        quality = 0
    else:
        score = 0.0
        state = V4TrunkApexAssociationState.REVIEW_ASSOCIATION
        score_available = False
        quality = int(V4AssociationQualityFlag.EVIDENCE_INCOMPLETE)
        if not tvc_available:
            quality |= int(V4AssociationQualityFlag.PATH_NOT_AVAILABLE)
    evidence = _finalize_record(
        V4TrunkApexAssociationEvidence,
        "diagnostic_digest",
        association_id=f"ta-{trunk.candidate_id}-{apex.candidate_id}",
        trunk_candidate_id=trunk.candidate_id,
        apex_candidate_id=apex.candidate_id,
        tx=tx,
        tx_available=tx_available,
        td=td,
        td_available=td_available,
        tvc=tvc,
        tvc_available=tvc_available,
        tg=tg,
        tg_available=tg_available,
        tr=tr,
        tr_available=tr_available,
        association_score_available=score_available,
        association_score=score,
        association_state=state,
        structural_path_evidence_reference=(
            path_evidence.path_id
            if path_evidence.availability is V4StructuralPathAvailability.OBSERVED
            else None
        ),
        lookup_kind=V4AssociationLookupKind.EVALUATED,
        quality_flags=quality,
        reason_flags=quality,
    )
    return _AssociationResult(evidence, path_evidence, tvc_bins)


def _vertical_continuity(
    trunk: V4TrunkCandidateRecord,
    apex: V4ApexCandidateRecord,
    corridor: dict[tuple[int, int], _Voxel],
    grid: _Grid,
    scales: V4AlgorithmScales,
) -> tuple[float, float, bool, bool, tuple[int, int]]:
    """Return (tvc_bins, tg, tvc_available, tg_available, (k_start, k_stop)).

    ``TVc_bins = supported_bins / evaluated_bins`` over the intermediate HAG
    corridor.  A bin is supported only when it contains an OCCUPIED support
    voxel connected to the preceding supported bin through a canonical
    26-neighbour voxel edge; an arbitrary voxel anywhere in the bin is NOT
    sufficient.  The chain is contiguous: an unsupported bin breaks it.
    """

    trunk_upper_bin = int(math.floor(trunk.core_hag_max_m / scales.btr))
    apex_bin = int(math.floor(apex.hag_m / scales.btr))
    k_start = trunk_upper_bin + 1
    k_stop = apex_bin - 1
    if k_stop < k_start:
        directly_adjacent = apex_bin <= trunk_upper_bin + 1
        tvc_bins = 1.0 if directly_adjacent else 0.0
        tg = 1.0
        return tvc_bins, tg, True, True, (k_start, k_stop)
    voxels_by_bin: dict[int, list[_Voxel]] = {}
    for voxel in corridor.values():
        voxels_by_bin.setdefault(voxel.hag_bin_id, []).append(voxel)
    prev_supported: list[_Voxel] = voxels_by_bin.get(trunk_upper_bin, [])
    evaluated_bins = k_stop - k_start + 1
    supported_bins = 0
    longest_gap = 0
    gap_run = 0
    for bin_id in range(k_start, k_stop + 1):
        current_voxels = voxels_by_bin.get(bin_id, [])
        connected: list[_Voxel] = []
        for voxel in current_voxels:
            for prev in prev_supported:
                if _voxel_adjacent(prev, voxel, grid, scales):
                    connected.append(voxel)
                    break
        if connected:
            supported_bins += 1
            prev_supported = connected
            gap_run = 0
        else:
            prev_supported = []
            gap_run += 1
            longest_gap = max(longest_gap, gap_run)
    tvc_bins = supported_bins / evaluated_bins
    tg = near(longest_gap * scales.btr, scales.ld)
    return tvc_bins, tg, True, True, (k_start, k_stop)


def _crown_endpoint_connectivity(
    trunk: V4TrunkCandidateRecord,
    apex: V4ApexCandidateRecord,
    corridor: dict[tuple[int, int], _Voxel],
    grid: _Grid,
    scales: V4AlgorithmScales,
    maximum_local_edges: int,
) -> tuple[float, bool]:
    apex_xy = np.asarray(apex.xyz_local_m[:2])
    apex_bin = int(math.floor(apex.hag_m / scales.btr))
    trunk_upper_bin = int(math.floor(trunk.core_hag_max_m / scales.btr))
    # The local pair domain (the corridor) supplies both support sets; never
    # scan arbitrary global upper voxels.
    upper_voxels = [
        voxel for key, voxel in corridor.items()
        if voxel.hag_bin_id >= trunk_upper_bin
    ]
    apex_voxels = [
        voxel for key, voxel in corridor.items()
        if abs(voxel.hag_bin_id - apex_bin) <= 1
        and math.hypot(voxel.centroid_xyz[0] - apex_xy[0], voxel.centroid_xyz[1] - apex_xy[1])
        <= scales.la
    ]
    if not upper_voxels or not apex_voxels:
        return 0.0, False
    # The ratio denominator is the full upper x apex pair count (frozen
    # semantics).  Resource-aware guard on the projected pair-evaluation work;
    # the ``linked`` numerator is enumerated exactly via each apex voxel's
    # 26-neighbourhood instead of an O(U x A) all-pairs loop.
    evaluated = len(upper_voxels) * len(apex_voxels)
    if (
        _projected_bytes(evaluated, _PROJECTED_PAIR_EVALUATION_BYTES)
        > _MAXIMUM_LOCAL_WORKING_BYTES
    ):
        _raise_local_edge_resource_limit()
    upper_by_key: dict[tuple[int, int], _Voxel] = {
        (voxel.cell_id, voxel.hag_bin_id): voxel for voxel in upper_voxels
    }
    linked = 0
    for apex_voxel in apex_voxels:
        for _neighbor_key in _voxel_adjacent_neighbor_keys(
            upper_by_key, grid, scales, (apex_voxel.cell_id, apex_voxel.hag_bin_id), apex_voxel
        ):
            linked += 1
    if evaluated == 0:
        return 0.0, True
    return linked / evaluated, True


# ---------------------------------------------------------------------------
# Rotating profile (4.4A)
# ---------------------------------------------------------------------------


def _ray_band_inventory(
    grid: _Grid,
    apex: V4ApexCandidateRecord,
    direction: tuple[float, float],
    band_start: float,
    band_stop: float,
    scales: V4AlgorithmScales,
) -> tuple[list[float], list[float], list[int], bool]:
    """Scan one radial band of the profile ray.

    Returns ``(occupied_hag_q75, densities, supporting_indexes, incomplete)``.
    A sample is ``OBSERVED_EMPTY`` ONLY when the local cell inventory for the
    radial band is known complete; any part of the band falling outside the
    grid marks the inventory incomplete (``NOT_AVAILABLE``), never empty.
    """
    apex_x = float(apex.xyz_local_m[0])
    apex_y = float(apex.xyz_local_m[1])
    perpendicular = (-direction[1], direction[0])
    half = 0.5 * scales.lxy
    corners: list[tuple[float, float]] = []
    for radial in (band_start, band_stop):
        px = apex_x + direction[0] * radial
        py = apex_y + direction[1] * radial
        corners.append((px + perpendicular[0] * half, py + perpendicular[1] * half))
        corners.append((px - perpendicular[0] * half, py - perpendicular[1] * half))
    min_x = min(x for x, _y in corners)
    max_x = max(x for x, _y in corners)
    min_y = min(y for _x, y in corners)
    max_y = max(y for _x, y in corners)
    origin_x, origin_y = grid.origin_xy
    col_floor = int(math.floor((min_x - origin_x) / scales.lxy))
    col_ceil = int(math.floor((max_x - origin_x) / scales.lxy))
    row_floor = int(math.floor((min_y - origin_y) / scales.lxy))
    row_ceil = int(math.floor((max_y - origin_y) / scales.lxy))
    occupied: list[float] = []
    densities: list[float] = []
    supporting_indexes: list[int] = []
    for row in range(row_floor, row_ceil + 1):
        center_y = origin_y + (row + 0.5) * scales.lxy
        for col in range(col_floor, col_ceil + 1):
            center_x = origin_x + (col + 0.5) * scales.lxy
            dx = center_x - apex_x
            dy = center_y - apex_y
            projection = dx * direction[0] + dy * direction[1]
            perp = abs(dx * direction[1] - dy * direction[0])
            if perp > half:
                continue
            if not (band_start <= projection < band_stop):
                continue
            if row < 0 or row >= grid.rows or col < 0 or col >= grid.cols:
                return occupied, densities, supporting_indexes, True
            summary = grid.cell_id_to_summary.get(row * grid.cols + col)
            if summary is None:
                continue
            if summary.count >= TRUNK_BIN_MIN_POINTS:
                occupied.append(summary.hag_q75)
                densities.append(summary.density)
                supporting_indexes.append(summary.min_original_index)
    return occupied, densities, supporting_indexes, False


def _rotating_profile(
    apex: V4ApexCandidateRecord,
    grid: _Grid,
    scales: V4AlgorithmScales,
    config: V4TreeSegmentationConfig,
) -> V4RotatingProfileEvidence:
    rays: list[V4RotatingProfileRayRecord] = []
    max_samples = int(math.ceil(scales.lr / scales.lxy)) + 1
    # Pre-flight resource gate: the full 24-ray sample inventory is known before
    # any sample is constructed.
    if ROTATING_PROFILE_RAY_COUNT * max_samples > config.maximum_profile_samples:
        raise V4TreeSegmentationError(
            code=V4TreeSegmentationErrorCode.RESOURCE_LIMIT,
            field_name="maximum_profile_samples",
        )
    for ray_index in range(ROTATING_PROFILE_RAY_COUNT):
        theta = 2.0 * math.pi * ray_index / ROTATING_PROFILE_RAY_COUNT
        direction = (math.cos(theta), math.sin(theta))
        samples: list[V4RotatingProfileRaySample] = []
        for sample_index in range(max_samples):
            if len(samples) >= config.maximum_profile_samples:
                raise V4TreeSegmentationError(
                    code=V4TreeSegmentationErrorCode.RESOURCE_LIMIT,
                    field_name="maximum_profile_samples",
                )
            radial = sample_index * scales.lxy
            band_start = radial
            band_stop = band_start + scales.lxy
            occupied, densities, supporting_indexes, incomplete = _ray_band_inventory(
                grid, apex, direction, band_start, band_stop, scales
            )
            if incomplete:
                samples.append(
                    V4RotatingProfileRaySample(
                        sample_index=sample_index,
                        radial_distance_m=radial,
                        state=V4ProfileSampleState.NOT_AVAILABLE,
                        hag_q75_m=None,
                        median_density_points_per_m2=None,
                        supporting_original_index_digest=_member_digest(()),
                    )
                )
            elif occupied:
                samples.append(
                    V4RotatingProfileRaySample(
                        sample_index=sample_index,
                        radial_distance_m=radial,
                        state=V4ProfileSampleState.OCCUPIED,
                        hag_q75_m=_median(sorted(occupied)),
                        median_density_points_per_m2=_median(sorted(densities)),
                        supporting_original_index_digest=_member_digest(supporting_indexes),
                    )
                )
            else:
                samples.append(
                    V4RotatingProfileRaySample(
                        sample_index=sample_index,
                        radial_distance_m=radial,
                        state=V4ProfileSampleState.OBSERVED_EMPTY,
                        hag_q75_m=None,
                        median_density_points_per_m2=None,
                        supporting_original_index_digest=_member_digest(()),
                    )
                )
        rays.append(_analyze_ray(ray_index, theta, samples, scales))
    profile_state, measured_gap, support = _profile_summary(rays)
    quality = 0
    reason = 0
    if profile_state is V4RotatingProfileState.PROFILE_NOT_AVAILABLE:
        quality |= int(V4ProfileQualityFlag.NOT_AVAILABLE)
    if measured_gap:
        quality |= int(V4ProfileQualityFlag.MEASURED_GAP)
    if profile_state is V4RotatingProfileState.PROFILE_WEAK_VALLEY:
        quality |= int(V4ProfileQualityFlag.WEAK_VALLEY)
    if profile_state is V4RotatingProfileState.PROFILE_NO_VALLEY:
        quality |= int(V4ProfileQualityFlag.NO_VALLEY)
    return _finalize_record(
        V4RotatingProfileEvidence,
        "evidence_digest",
        profile_id=f"profile-{apex.candidate_id}",
        apex_candidate_id=apex.candidate_id,
        hypothesis_id=None,
        rays=tuple(rays),
        profile_state=profile_state,
        measured_gap=measured_gap,
        profile_valley_support=support,
        quality_flags=quality,
        reason_flags=reason,
    )


def _median_smoothing(values: list[float]) -> list[float]:
    if not values:
        return []
    smoothed: list[float] = []
    for index in range(len(values)):
        window = values[max(0, index - 1) : min(len(values), index + 2)]
        smoothed.append(_median(window))
    return smoothed


def _smooth_contiguous_observed(
    samples: list[V4RotatingProfileRaySample],
) -> tuple[dict[int, float], dict[int, float]]:
    """3-sample median smoothing over contiguous OCCUPIED radial runs.

    OBSERVED_EMPTY and NOT_AVAILABLE samples remain spatially present and break
    contiguity: occupied samples separated by a gap are never smoothed as
    adjacent.
    """
    occupied = [
        sample
        for sample in samples
        if sample.state is V4ProfileSampleState.OCCUPIED
    ]
    hag_smoothed: dict[int, float] = {}
    density_smoothed: dict[int, float] = {}
    runs: list[list[V4RotatingProfileRaySample]] = []
    current: list[V4RotatingProfileRaySample] = []
    previous_index: int | None = None
    for sample in occupied:
        if previous_index is not None and sample.sample_index != previous_index + 1:
            runs.append(current)
            current = []
        current.append(sample)
        previous_index = sample.sample_index
    if current:
        runs.append(current)
    for run in runs:
        hag_values = [sample.hag_q75_m for sample in run]
        density_values = [sample.median_density_points_per_m2 for sample in run]
        smoothed_hag = _median_smoothing(hag_values)
        smoothed_density = _median_smoothing(density_values)
        for sample, hag, density in zip(run, smoothed_hag, smoothed_density, strict=True):
            hag_smoothed[sample.sample_index] = hag
            density_smoothed[sample.sample_index] = density
    return hag_smoothed, density_smoothed


def _analyze_ray(
    ray_index: int,
    theta: float,
    samples: list[V4RotatingProfileRaySample],
    scales: V4AlgorithmScales,
) -> V4RotatingProfileRayRecord:
    observed = [
        sample for sample in samples if sample.state is V4ProfileSampleState.OCCUPIED
    ]
    has_not_available = any(
        sample.state is V4ProfileSampleState.NOT_AVAILABLE for sample in samples
    )
    measured_gap_run = 0
    longest_gap_run = 0
    for sample in samples:
        if sample.state is V4ProfileSampleState.OBSERVED_EMPTY:
            measured_gap_run += 1
        else:
            measured_gap_run = 0
        longest_gap_run = max(longest_gap_run, measured_gap_run)
    measured_gap = longest_gap_run * scales.lxy >= scales.ld
    if has_not_available or len(observed) < 3:
        return V4RotatingProfileRayRecord(
            ray_index=ray_index,
            theta_rad=theta,
            samples=tuple(samples),
            smoothed_hag_median_m=None,
            smoothed_density_median_points_per_m2=None,
            valley_state=V4RotatingProfileState.PROFILE_NOT_AVAILABLE,
            measured_gap=measured_gap,
            boundary_sample_index=None,
        )
    hag_smoothed, density_smoothed = _smooth_contiguous_observed(samples)
    valley_state = V4RotatingProfileState.PROFILE_NO_VALLEY
    boundary = None
    for sample in observed:
        radial = sample.radial_distance_m
        inner_start = max(0.0, radial - scales.ld)
        inner_indices = [
            other.sample_index
            for other in observed
            if inner_start <= other.radial_distance_m < radial
        ]
        outer_indices = [
            other.sample_index
            for other in observed
            if radial < other.radial_distance_m <= radial + scales.ld
        ]
        if not inner_indices or not outer_indices:
            continue
        inner_h = _median([hag_smoothed[index] for index in inner_indices])
        valley_h = hag_smoothed[sample.sample_index]
        outer_h = _median([hag_smoothed[index] for index in outer_indices])
        inner_rho = _median([density_smoothed[index] for index in inner_indices])
        valley_rho = density_smoothed[sample.sample_index]
        outer_rho = _median([density_smoothed[index] for index in outer_indices])
        drop = c01((inner_h - valley_h) / max(scales.la, scales.ls))
        rebound = c01((outer_h - valley_h) / max(scales.la, scales.ls))
        density_valley = c01(
            1.0
            - valley_rho / max(min(inner_rho, outer_rho), scales.rho_floor)
        )
        support = min(drop, rebound, density_valley)
        candidate = (
            drop >= ROTATING_PROFILE_DROP_MIN
            and rebound >= ROTATING_PROFILE_REBOUND_MIN
            and valley_rho
            <= ROTATING_PROFILE_DENSITY_RATIO * min(inner_rho, outer_rho)
        )
        if candidate and support >= ROTATING_PROFILE_VALLEY_MIN:
            valley_state = V4RotatingProfileState.PROFILE_CONFIRMED_VALLEY
            boundary = sample.sample_index
            break
        if candidate:
            valley_state = V4RotatingProfileState.PROFILE_WEAK_VALLEY
            boundary = sample.sample_index
    return V4RotatingProfileRayRecord(
        ray_index=ray_index,
        theta_rad=theta,
        samples=tuple(samples),
        smoothed_hag_median_m=(
            _median(list(hag_smoothed.values())) if hag_smoothed else None
        ),
        smoothed_density_median_points_per_m2=(
            _median(list(density_smoothed.values())) if density_smoothed else None
        ),
        valley_state=valley_state,
        measured_gap=measured_gap,
        boundary_sample_index=boundary,
    )


def _profile_summary(
    rays: list[V4RotatingProfileRayRecord],
) -> tuple[V4RotatingProfileState, bool, float | None]:
    any_not_available = any(
        ray.valley_state is V4RotatingProfileState.PROFILE_NOT_AVAILABLE
        for ray in rays
    )
    any_gap = any(ray.measured_gap for ray in rays)
    confirmed = sum(
        ray.valley_state is V4RotatingProfileState.PROFILE_CONFIRMED_VALLEY
        for ray in rays
    )
    weak = sum(
        ray.valley_state is V4RotatingProfileState.PROFILE_WEAK_VALLEY for ray in rays
    )
    if any_not_available:
        return V4RotatingProfileState.PROFILE_NOT_AVAILABLE, any_gap, None
    if confirmed > 0:
        return V4RotatingProfileState.PROFILE_CONFIRMED_VALLEY, any_gap, (
            confirmed / ROTATING_PROFILE_RAY_COUNT
        )
    if weak > 0:
        return V4RotatingProfileState.PROFILE_WEAK_VALLEY, any_gap, (
            weak / ROTATING_PROFILE_RAY_COUNT
        )
    return V4RotatingProfileState.PROFILE_NO_VALLEY, any_gap, 0.0


def _first_gap_radius(
    samples: list[V4RotatingProfileRaySample], scales: V4AlgorithmScales
) -> float | None:
    """Radial distance where the first measured observed-empty run >= Ld begins."""
    run = 0
    for sample in samples:
        if sample.state is V4ProfileSampleState.OBSERVED_EMPTY:
            run += 1
            if run * scales.lxy >= scales.ld:
                return max(0.0, (sample.sample_index - run + 1) * scales.lxy)
        else:
            run = 0
    return None


@dataclass(frozen=True, slots=True)
class _ProfileStopInfo:
    """Per-ray profile transition authority for responsibility propagation."""

    boundary_radius_by_ray: dict[int, float]
    gap_radius_by_ray: dict[int, float]


def _profile_stop_info(
    profile: V4RotatingProfileEvidence, scales: V4AlgorithmScales
) -> _ProfileStopInfo | None:
    boundary: dict[int, float] = {}
    gap: dict[int, float] = {}
    for ray in profile.rays:
        if (
            ray.valley_state is V4RotatingProfileState.PROFILE_CONFIRMED_VALLEY
            and ray.boundary_sample_index is not None
        ):
            boundary[ray.ray_index] = ray.boundary_sample_index * scales.lxy
        if ray.measured_gap:
            gap_radius = _first_gap_radius(ray.samples, scales)
            if gap_radius is not None:
                gap[ray.ray_index] = gap_radius
    if not boundary and not gap:
        return None
    return _ProfileStopInfo(boundary, gap)


# ---------------------------------------------------------------------------
# Responsibility solver (4.2A-A)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Claim:
    """One hypothesis claim over an occupied cell."""

    hypothesis_id: str
    apex_candidate_id: str
    cell_id: int
    score: float
    r0: float
    channels: tuple[float, float, float, float, float, float, float]
    path_length_m: float


def _hypothesis_key(apex: V4ApexCandidateRecord) -> str:
    return f"hyp-{apex.candidate_id}"


def _responsibility_solver(
    apices: list[V4ApexCandidateRecord],
    grid: _Grid,
    scales: V4AlgorithmScales,
    config: V4TreeSegmentationConfig,
    profile_stop_by_hypothesis: dict[str, _ProfileStopInfo] | None = None,
) -> dict[int, list[_Claim]]:
    claims_by_cell: dict[int, list[_Claim]] = {}
    if not apices or not grid.occupied_cell_ids:
        return claims_by_cell
    cell_summaries = grid.cell_id_to_summary
    profile_stops = profile_stop_by_hypothesis or {}
    for apex in apices:
        hypothesis_id = _hypothesis_key(apex)
        result = _hypothesis_claims(
            apex,
            hypothesis_id,
            grid,
            cell_summaries,
            scales,
            config,
            profile_stop=profile_stops.get(hypothesis_id),
        )
        for cell_id, claim in result.items():
            claims_by_cell.setdefault(cell_id, []).append(claim)
    # Two-pass pressure against the immutable first-round state.  Rival
    # pressure for claim i is the best OTHER legal claim (j != i); the claim
    # itself is never its own rival.
    for cell_id, claims in claims_by_cell.items():
        updated: list[_Claim] = []
        for claim in claims:
            rival_r0: float | None = None
            for other in claims:
                if other.hypothesis_id != claim.hypothesis_id:
                    if rival_r0 is None or other.r0 > rival_r0:
                        rival_r0 = other.r0
            pressure = responsibility_pressure(claim.r0, rival_r0)
            score = responsibility_score(
                apex_support=claim.channels[0],
                path_bottleneck=claim.channels[1],
                density_trend=claim.channels[2],
                hag_continuity=claim.channels[3],
                distance_decay=claim.channels[4],
                connectivity=claim.channels[5],
                empty_gap=claim.channels[6],
                pressure=pressure,
            )
            if score < RESPONSIBILITY_MIN:
                continue
            updated.append(
                _Claim(
                    hypothesis_id=claim.hypothesis_id,
                    apex_candidate_id=claim.apex_candidate_id,
                    cell_id=cell_id,
                    score=score,
                    r0=claim.r0,
                    channels=claim.channels,
                    path_length_m=claim.path_length_m,
                )
            )
        claims_by_cell[cell_id] = sorted(
            updated, key=lambda claim: (-claim.score, claim.hypothesis_id)
        )
    # Keep best + secondary within margin.
    result: dict[int, list[_Claim]] = {}
    for cell_id, claims in claims_by_cell.items():
        if not claims:
            continue
        best = claims[0]
        kept = [best]
        for claim in claims[1:]:
            if (
                claim.score >= RESPONSIBILITY_MIN
                and best.score - claim.score <= RESPONSIBILITY_REVIEW_MARGIN
            ):
                kept.append(claim)
        result[cell_id] = kept
    return result


def _persistent_decline_stop(
    cell_id: int,
    predecessor: dict[int, int | None],
    channels_by_cell: dict[int, tuple[float, float, float, float, float, float, float]],
    scales: V4AlgorithmScales,
) -> bool:
    """Hard stop when two consecutive complete ``Ld`` windows both show
    ``D < 0.35 AND H < 0.35`` along the canonical path.

    A window containing a cell with no measured channel (a gap in measured
    coverage) is not a complete window and resets the consecutive counter; it
    is never treated as evidence of decline.
    """
    path: list[int] = []
    cursor: int | None = cell_id
    while cursor is not None:
        path.append(cursor)
        cursor = predecessor.get(cursor)
    path.reverse()
    if len(path) < 2:
        return False
    window_cells: list[int] = []
    window_length = 0.0
    consecutive_declining = 0
    for index in range(1, len(path)):
        cell = path[index]
        window_length += scales.lxy
        window_cells.append(cell)
        if window_length >= scales.ld:
            d_values: list[float] = []
            h_values: list[float] = []
            complete = True
            for window_cell in window_cells:
                entry = channels_by_cell.get(window_cell)
                if entry is None:
                    complete = False
                    break
                d_values.append(entry[2])
                h_values.append(entry[3])
            if not complete:
                consecutive_declining = 0
            else:
                d_median = _median(d_values)
                h_median = _median(h_values)
                if d_median < 0.35 and h_median < 0.35:
                    consecutive_declining += 1
                else:
                    consecutive_declining = 0
            window_length = 0.0
            window_cells = []
            if consecutive_declining >= 2:
                return True
    return False


def _hypothesis_claims(
    apex: V4ApexCandidateRecord,
    hypothesis_id: str,
    grid: _Grid,
    cell_summaries: dict[int, Any],
    scales: V4AlgorithmScales,
    config: V4TreeSegmentationConfig,
    profile_stop: _ProfileStopInfo | None = None,
) -> dict[int, _Claim]:
    apex_cell = _nearest_cell(apex, grid)
    if apex_cell is None:
        return {}
    apex_q75 = cell_summaries[apex_cell].hag_q75
    # Dijkstra over all cells with uniform step cost, tracking path aggregates.
    distance: dict[int, float] = {}
    predecessor: dict[int, int | None] = {}
    hop_count: dict[int, int] = {}
    min_bottleneck: dict[int, float] = {}
    min_hag_continuity: dict[int, float] = {}
    supported_edges: dict[int, int] = {}
    evaluated_edges: dict[int, int] = {}
    longest_empty_run: dict[int, float] = {}
    current_empty_run: dict[int, int] = {}
    heap: list[tuple[float, int, int, int]] = []
    distance[apex_cell] = 0.0
    hop_count[apex_cell] = 0
    predecessor[apex_cell] = None
    min_bottleneck[apex_cell] = 1.0
    min_hag_continuity[apex_cell] = 1.0
    supported_edges[apex_cell] = 0
    evaluated_edges[apex_cell] = 0
    longest_empty_run[apex_cell] = 0.0
    current_empty_run[apex_cell] = 0
    heapq.heappush(heap, (0.0, 0, apex_cell, 0))
    popped = 0
    while heap:
        current_distance, current_hops, cell_id, _tie = heapq.heappop(heap)
        if distance.get(cell_id) != current_distance:
            continue
        popped += 1
        if popped > config.maximum_path_evaluations:
            raise V4TreeSegmentationError(
                code=V4TreeSegmentationErrorCode.RESOURCE_LIMIT,
                field_name="maximum_path_evaluations",
            )
        summary = cell_summaries.get(cell_id)
        for neighbor in grid.neighbors_8(cell_id):
            neighbor_summary = cell_summaries.get(neighbor)
            new_distance = current_distance + scales.lxy
            new_hops = current_hops + 1
            previous = distance.get(neighbor)
            if previous is not None and (
                new_distance,
                new_hops,
            ) >= (previous, hop_count.get(neighbor, 0)):
                continue
            new_empty_run = 0 if neighbor_summary is not None else (current_empty_run[cell_id] + 1)
            new_longest = max(longest_empty_run[cell_id], new_empty_run * scales.lxy)
            if new_longest >= scales.ld:
                continue
            # Bottleneck and HAG continuity along the path.
            if neighbor_summary is not None:
                bottleneck = min(
                    min_bottleneck[cell_id],
                    c01(
                        neighbor_summary.hag_q75 / max(apex_q75, scales.ls)
                    ),
                )
                neighbor_hag = neighbor_summary.hag_q75
                current_hag = summary.hag_q75 if summary is not None else current_hag_fallback(
                    cell_id, cell_summaries
                )
                continuity = min(
                    min_hag_continuity[cell_id],
                    similar(abs(neighbor_hag - current_hag), max(scales.ls, scales.lxy)),
                )
                new_supported = supported_edges[cell_id] + 1
            else:
                bottleneck = min_bottleneck[cell_id]
                continuity = min_hag_continuity[cell_id]
                new_supported = supported_edges[cell_id]
            distance[neighbor] = new_distance
            hop_count[neighbor] = new_hops
            predecessor[neighbor] = cell_id
            min_bottleneck[neighbor] = bottleneck
            min_hag_continuity[neighbor] = continuity
            supported_edges[neighbor] = new_supported
            evaluated_edges[neighbor] = evaluated_edges[cell_id] + 1
            longest_empty_run[neighbor] = new_longest
            current_empty_run[neighbor] = new_empty_run
            heapq.heappush(
                heap, (new_distance, new_hops, neighbor, neighbor)
            )
    # Compute channels for every reached occupied cell.
    channels_by_cell: dict[int, tuple[float, float, float, float, float, float, float]] = {}
    for cell_id in sorted(distance):
        summary = cell_summaries.get(cell_id)
        if summary is None:
            continue
        path_length = distance[cell_id]
        r_channel = near(path_length, scales.lr)
        b_channel = min_bottleneck[cell_id]
        h_channel = min_hag_continuity[cell_id]
        evaluated = evaluated_edges[cell_id]
        c_channel = (
            supported_edges[cell_id] / evaluated if evaluated > 0 else 1.0
        )
        g_channel = near(longest_empty_run[cell_id], scales.ld)
        d_channel = _density_trend_channel(
            apex_cell, cell_id, predecessor, cell_summaries, scales
        )
        channels_by_cell[cell_id] = (
            apex.apex_support_score,
            b_channel,
            d_channel,
            h_channel,
            r_channel,
            c_channel,
            g_channel,
        )
    result: dict[int, _Claim] = {}
    for cell_id in sorted(channels_by_cell):
        summary = cell_summaries[cell_id]
        path_length = distance[cell_id]
        apex_support, b_channel, d_channel, h_channel, r_channel, c_channel, g_channel = (
            channels_by_cell[cell_id]
        )
        # Rotating-profile hard transition stop (frozen profile_confirmed_stop):
        # a measured gap >= Ld stops unconditionally; a confirmed valley stops
        # only when the existing local density or HAG decline is present.  The
        # stop applies before score eligibility, so a high score cannot override
        # it.
        if profile_stop is not None:
            apex_x = float(apex.xyz_local_m[0])
            apex_y = float(apex.xyz_local_m[1])
            center = summary.cell_center
            angle = math.atan2(center[1] - apex_y, center[0] - apex_x)
            if angle < 0.0:
                angle += 2.0 * math.pi
            ray_index = int(
                angle / (2.0 * math.pi / ROTATING_PROFILE_RAY_COUNT)
            )
            distance_from_apex = math.hypot(center[0] - apex_x, center[1] - apex_y)
            gap_radius = profile_stop.gap_radius_by_ray.get(ray_index)
            if gap_radius is not None and distance_from_apex >= gap_radius:
                continue
            boundary_radius = profile_stop.boundary_radius_by_ray.get(ray_index)
            if (
                boundary_radius is not None
                and distance_from_apex >= boundary_radius
                and (d_channel < 0.35 or h_channel < 0.35)
            ):
                continue
        # Persistent hard stop: two consecutive complete Ld windows with both
        # D < 0.35 and H < 0.35 along the canonical path.
        if _persistent_decline_stop(
            cell_id, predecessor, channels_by_cell, scales
        ):
            continue
        r0 = responsibility_r0(
            apex_support=apex_support,
            path_bottleneck=b_channel,
            density_trend=d_channel,
            hag_continuity=h_channel,
            distance_decay=r_channel,
            connectivity=c_channel,
            empty_gap=g_channel,
        )
        result[cell_id] = _Claim(
            hypothesis_id=hypothesis_id,
            apex_candidate_id=apex.candidate_id,
            cell_id=cell_id,
            score=0.0,
            r0=r0,
            channels=(
                apex_support,
                b_channel,
                d_channel,
                h_channel,
                r_channel,
                c_channel,
                g_channel,
            ),
            path_length_m=path_length,
        )
    return result


def current_hag_fallback(cell_id: int, cell_summaries: dict[int, Any]) -> float:
    summary = cell_summaries.get(cell_id)
    return summary.hag_q75 if summary is not None else 0.0


def _nearest_cell(apex: V4ApexCandidateRecord, grid: _Grid) -> int | None:
    best_cell = None
    best_distance = math.inf
    for cell_id in grid.occupied_cell_ids:
        summary = grid.cell_id_to_summary[cell_id]
        distance = math.hypot(
            summary.cell_center[0] - apex.xyz_local_m[0],
            summary.cell_center[1] - apex.xyz_local_m[1],
        )
        if distance < best_distance or (
            distance == best_distance and (best_cell is None or cell_id < best_cell)
        ):
            best_cell = cell_id
            best_distance = distance
    return best_cell


def _density_trend_channel(
    apex_cell: int,
    cell_id: int,
    predecessor: dict[int, int | None],
    cell_summaries: dict[int, Any],
    scales: V4AlgorithmScales,
) -> float:
    path: list[int] = []
    cursor: int | None = cell_id
    while cursor is not None:
        path.append(cursor)
        cursor = predecessor.get(cursor)
    path.reverse()
    if len(path) <= 1:
        return 1.0
    target_distance = min(scales.ld, len(path) * scales.lxy)
    inner_cells: list[int] = []
    forward_cells: list[int] = []
    accumulated = 0.0
    split_index = len(path)
    for index in range(len(path)):
        if index == 0:
            continue
        accumulated += scales.lxy
        if accumulated <= target_distance:
            inner_cells.append(path[index])
        else:
            split_index = index
            break
    forward_cells = path[split_index:]
    inner_densities = [
        cell_summaries[cell].density for cell in inner_cells if cell in cell_summaries
    ]
    forward_densities = [
        cell_summaries[cell].density for cell in forward_cells if cell in cell_summaries
    ]
    if not inner_densities or not forward_densities:
        # A missing/insufficient density window is typed missing, NOT a measured
        # decline; a neutral 1.0 (no negative evidence) is used instead of 0.0.
        return 1.0
    return safe_ratio(
        _median(forward_densities), _median(inner_densities), scales.rho_floor
    )


# ---------------------------------------------------------------------------
# 3D slices and fragments
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Fragment:
    fragment_id: str
    slice_index: int
    hypothesis_id: str
    positions: tuple[int, ...]
    member_digest: str


@dataclass(frozen=True, slots=True)
class _HypothesisDomain:
    hypothesis_id: str
    apex_candidate_id: str
    cell_ids: tuple[int, ...]


def _responsibility_domains(
    claims_by_cell: dict[int, list[_Claim]],
) -> dict[str, _HypothesisDomain]:
    domains: dict[str, list[int]] = {}
    for cell_id, claims in claims_by_cell.items():
        for claim in claims:
            domains.setdefault(claim.hypothesis_id, []).append(cell_id)
    result: dict[str, _HypothesisDomain] = {}
    for hypothesis_id in sorted(domains):
        result[hypothesis_id] = _HypothesisDomain(
            hypothesis_id=hypothesis_id,
            apex_candidate_id=hypothesis_id,
            cell_ids=tuple(sorted(domains[hypothesis_id])),
        )
    return result


def _slice_schedule(
    hag_min: float, hag_max: float, config: V4TreeSegmentationConfig
) -> list[tuple[float, float]]:
    if hag_max < hag_min:
        return []
    stride = config.slice_thickness_m * (1.0 - config.slice_overlap_fraction)
    if stride <= 0.0:
        stride = config.slice_thickness_m
    slices: list[tuple[float, float]] = []
    current = hag_min
    while current <= hag_max:
        if len(slices) >= config.maximum_fragments:
            raise V4TreeSegmentationError(
                code=V4TreeSegmentationErrorCode.RESOURCE_LIMIT,
                field_name="maximum_fragments",
            )
        slices.append((current, current + config.slice_thickness_m))
        current += stride
    return slices


def _build_slices_and_fragments(
    domains: dict[str, _HypothesisDomain],
    claims_by_cell: dict[int, list[_Claim]],
    population: _NormalPopulation,
    grid: _Grid,
    scales: V4AlgorithmScales,
    config: V4TreeSegmentationConfig,
) -> tuple[list[V4TreeSliceSummary], list[_Fragment]]:
    """Materialize the 3D evidence inventory with per-hypothesis bounds.

    ``maximum_fragments`` is a local workset ceiling: each responsibility
    hypothesis is independently bounded, while the full scene retains only the
    fragment inventory required for later cross-hypothesis relation and
    assignment evidence.  A global count is not a valid resource proxy for a
    multi-tree scene: many small, independently bounded crowns would otherwise
    be rejected before any V4 decision is made.
    """

    slices: list[V4TreeSliceSummary] = []
    fragments: list[_Fragment] = []
    for hypothesis_id in sorted(domains):
        hypothesis_fragment_count = 0
        domain = domains[hypothesis_id]
        domain_cells = set(domain.cell_ids)
        cell_positions: dict[int, list[int]] = {}
        for cell_id in domain_cells:
            cell_positions[cell_id] = _positions_in_cell(grid, cell_id)
        hag_values = [
            float(population.hag_m[pos])
            for positions in cell_positions.values()
            for pos in positions
        ]
        if not hag_values:
            continue
        hag_min = min(hag_values)
        hag_max = max(hag_values)
        schedule = _slice_schedule(hag_min, hag_max, config)
        for slice_index, (slice_min, slice_max) in enumerate(schedule):
            member_positions = sorted(
                pos
                for positions in cell_positions.values()
                for pos in positions
                if slice_min <= float(population.hag_m[pos]) < slice_max
            )
            if not member_positions:
                continue
            xs = [float(population.xyz_local_m[pos, 0]) for pos in member_positions]
            ys = [float(population.xyz_local_m[pos, 1]) for pos in member_positions]
            zs = [float(population.xyz_local_m[pos, 2]) for pos in member_positions]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            density = _raw_weight_sum(population, member_positions) / (
                area if area > 0 else scales.lxy * scales.lxy
            )
            slice_id = f"slice-{hypothesis_id}-{slice_index:03d}"
            slices.append(
                V4TreeSliceSummary(
                    slice_id=slice_id,
                    hypothesis_id=hypothesis_id,
                    hag_min_m=slice_min,
                    hag_max_m=slice_max,
                    point_count=len(member_positions),
                    centroid_xyz=(
                        sum(xs) / len(xs),
                        sum(ys) / len(ys),
                        sum(zs) / len(zs),
                    ),
                    bounds_min_xyz=(min(xs), min(ys), min(zs)),
                    bounds_max_xyz=(max(xs), max(ys), max(zs)),
                    area_m2=area,
                    density_points_per_m2=density,
                    boundary_digest=_member_digest(member_positions),
                    member_original_index_digest=_member_digest(
                        [population.original_index[pos] for pos in member_positions]
                    ),
                    quality_flags=0,
                    reason_flags=0,
                )
            )
            fragment_positions = _connected_positions_in_slice(
                member_positions, grid, scales
            )
            for fragment_index, component in enumerate(fragment_positions):
                if hypothesis_fragment_count >= config.maximum_fragments:
                    raise V4TreeSegmentationError(
                        code=V4TreeSegmentationErrorCode.RESOURCE_LIMIT,
                        field_name="maximum_fragments",
                    )
                fragments.append(
                    _Fragment(
                        fragment_id=f"frag-{slice_id}-{fragment_index:03d}",
                        slice_index=slice_index,
                        hypothesis_id=hypothesis_id,
                        positions=tuple(sorted(component)),
                        member_digest=_member_digest(
                            [population.original_index[pos] for pos in component]
                        ),
                    )
                )
                hypothesis_fragment_count += 1
    return slices, fragments


def _connected_positions_in_slice(
    positions: list[int],
    grid: _Grid,
    scales: V4AlgorithmScales,
) -> list[list[int]]:
    cell_to_positions: dict[int, list[int]] = {}
    for position in positions:
        cell_to_positions.setdefault(int(grid.point_cell_ids[position]), []).append(position)
    cell_ids = sorted(cell_to_positions)
    parent = {cell_id: cell_id for cell_id in cell_ids}

    def find(cell_id: int) -> int:
        while parent[cell_id] != cell_id:
            parent[cell_id] = parent[parent[cell_id]]
            cell_id = parent[cell_id]
        return cell_id

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    cell_set = set(cell_ids)
    for cell_id in cell_ids:
        for neighbor in grid.neighbors_8(cell_id):
            if neighbor in cell_set:
                union(cell_id, neighbor)
    groups: dict[int, list[int]] = {}
    for cell_id in cell_ids:
        groups.setdefault(find(cell_id), []).append(cell_id)
    result = []
    for _root in sorted(groups):
        component_cells = sorted(groups[_root])
        component_positions = sorted(
            pos for cell_id in component_cells for pos in cell_to_positions[cell_id]
        )
        result.append(component_positions)
    return result


def _fragment_metrics(
    fragment: _Fragment,
    population: _NormalPopulation,
    scales: V4AlgorithmScales,
) -> tuple[tuple[float, float], tuple[float, float], float, float, float, float]:
    positions = fragment.positions
    xs = [float(population.xyz_local_m[position, 0]) for position in positions]
    ys = [float(population.xyz_local_m[position, 1]) for position in positions]
    hags = [float(population.hag_m[position]) for position in positions]
    centroid = (math.fsum(xs) / len(xs), math.fsum(ys) / len(ys))
    extent = (
        max(max(xs) - min(xs), scales.lxy),
        max(max(ys) - min(ys), scales.lxy),
    )
    density = _raw_weight_sum(population, positions) / (
        extent[0] * extent[1]
    )
    return centroid, extent, density, min(hags), max(hags), max(extent)


def _fragment_link_quality(
    fragment_a: _Fragment,
    fragment_b: _Fragment,
    population: _NormalPopulation,
    scales: V4AlgorithmScales,
) -> float:
    """Measured adjacent-slice fragment continuity over frozen evidence kinds."""

    centroid_a, extent_a, density_a, hag_min_a, hag_max_a, max_extent_a = (
        _fragment_metrics(fragment_a, population, scales)
    )
    centroid_b, extent_b, density_b, hag_min_b, hag_max_b, max_extent_b = (
        _fragment_metrics(fragment_b, population, scales)
    )
    bounds_a = _fragment_xy_bounds(population, fragment_a)
    bounds_b = _fragment_xy_bounds(population, fragment_b)
    overlap_x = max(0.0, min(bounds_a[2], bounds_b[2]) - max(bounds_a[0], bounds_b[0]))
    overlap_y = max(0.0, min(bounds_a[3], bounds_b[3]) - max(bounds_a[1], bounds_b[1]))
    overlap_area = overlap_x * overlap_y
    area_a = extent_a[0] * extent_a[1]
    area_b = extent_b[0] * extent_b[1]
    boundary_overlap = c01(overlap_area / max(min(area_a, area_b), scales.lxy**2))
    centroid_shift = near(
        math.hypot(centroid_a[0] - centroid_b[0], centroid_a[1] - centroid_b[1]),
        scales.lr,
    )
    extent_ratio = min(max_extent_a, max_extent_b) / max(max_extent_a, max_extent_b)
    density_continuity = min(
        safe_ratio(density_a, density_b, scales.rho_floor),
        safe_ratio(density_b, density_a, scales.rho_floor),
    )
    measured_gap = max(0.0, max(hag_min_a, hag_min_b) - min(hag_max_a, hag_max_b))
    gap_support = near(measured_gap, scales.ls)
    # Extent ratio is also the measured taper/branch compatibility term; same
    # hypothesis identity supplies the retained responsibility-claim channel.
    return math.fsum(
        (
            0.25 * boundary_overlap,
            0.20 * centroid_shift,
            0.15 * extent_ratio,
            0.15 * density_continuity,
            0.15 * gap_support,
            0.10,
        )
    )


def _build_adjacent_fragment_graph(
    fragments: list[_Fragment],
    population: _NormalPopulation,
    scales: V4AlgorithmScales,
    config: V4TreeSegmentationConfig,
) -> tuple[dict[str, tuple[str, ...]], dict[str, float], dict[str, bool]]:
    """Build the bounded adjacent-slice 3D fragment graph and continuity audit.

    Fragment comparisons are bounded per responsibility hypothesis.  Adjacent
    fragments from separate crowns never interact, so making their independent
    local work share one scene-wide counter incorrectly turns tree count into a
    semantic resource failure.
    """

    by_hypothesis_slice: dict[tuple[str, int], list[_Fragment]] = {}
    for fragment in fragments:
        by_hypothesis_slice.setdefault(
            (fragment.hypothesis_id, fragment.slice_index), []
        ).append(fragment)
    adjacency: dict[str, list[str]] = {fragment.fragment_id: [] for fragment in fragments}
    evaluated_by_hypothesis: dict[str, int] = {}
    linked_by_hypothesis: dict[str, int] = {}
    hypotheses = sorted({fragment.hypothesis_id for fragment in fragments})
    for hypothesis_id in hypotheses:
        evaluated_local = 0
        slice_indexes = sorted(
            index
            for candidate, index in by_hypothesis_slice
            if candidate == hypothesis_id
        )
        for slice_index in slice_indexes:
            next_index = slice_index + 1
            if (hypothesis_id, next_index) not in by_hypothesis_slice:
                continue
            left = sorted(
                by_hypothesis_slice[(hypothesis_id, slice_index)],
                key=lambda fragment: fragment.fragment_id,
            )
            right = sorted(
                by_hypothesis_slice[(hypothesis_id, next_index)],
                key=lambda fragment: fragment.fragment_id,
            )
            for fragment_a in left:
                for fragment_b in right:
                    evaluated_local += 1
                    # Resource-aware guard: the adjacent-slice fragment link
                    # predicate has no single necessary geometric condition, so
                    # candidate enumeration stays exact; the guard projects the
                    # pair-evaluation work for this hypothesis instead of an
                    # arbitrary fixed count.
                    if (
                        _projected_bytes(
                            evaluated_local, _PROJECTED_PAIR_EVALUATION_BYTES
                        )
                        > _MAXIMUM_LOCAL_WORKING_BYTES
                    ):
                        _raise_local_edge_resource_limit()
                    evaluated_by_hypothesis[hypothesis_id] = (
                        evaluated_by_hypothesis.get(hypothesis_id, 0) + 1
                    )
                    if (
                        _fragment_link_quality(fragment_a, fragment_b, population, scales)
                        >= 0.50
                    ):
                        adjacency[fragment_a.fragment_id].append(fragment_b.fragment_id)
                        adjacency[fragment_b.fragment_id].append(fragment_a.fragment_id)
                        linked_by_hypothesis[hypothesis_id] = (
                            linked_by_hypothesis.get(hypothesis_id, 0) + 1
                        )
        evaluated_by_hypothesis[hypothesis_id] = evaluated_local
    continuity: dict[str, float] = {}
    available: dict[str, bool] = {}
    for hypothesis_id in hypotheses:
        slice_count = len(
            {
                fragment.slice_index
                for fragment in fragments
                if fragment.hypothesis_id == hypothesis_id
            }
        )
        evaluated = evaluated_by_hypothesis.get(hypothesis_id, 0)
        if slice_count <= 1:
            # A legal single-slice tree has no observed adjacent edge.  The
            # signed assignment channel is therefore neutral (e=0), which is
            # represented by the fraction-domain midpoint.
            continuity[hypothesis_id] = 0.5
            available[hypothesis_id] = True
        elif evaluated > 0:
            continuity[hypothesis_id] = linked_by_hypothesis.get(hypothesis_id, 0) / evaluated
            available[hypothesis_id] = True
        else:
            continuity[hypothesis_id] = 0.0
            available[hypothesis_id] = False
    return (
        {key: tuple(sorted(neighbors)) for key, neighbors in adjacency.items()},
        continuity,
        available,
    )


def _revalidate_associations_with_fragment_graph(
    associations: list[V4TrunkApexAssociationEvidence],
    path_by_association: dict[str, V4StructuralPathEvidence],
    fragments: list[_Fragment],
    fragment_adjacency: dict[str, tuple[str, ...]],
    trunk_id_lookup: dict[str, tuple[int, tuple[int, ...]]],
    apex_by_id: dict[str, V4ApexCandidateRecord],
    population: _NormalPopulation,
    scales: V4AlgorithmScales,
) -> list[V4TrunkApexAssociationEvidence]:
    """Downgrade early CSP associations that fail measured 3D revalidation."""

    fragments_by_hypothesis: dict[str, list[_Fragment]] = {}
    for fragment in fragments:
        fragments_by_hypothesis.setdefault(fragment.hypothesis_id, []).append(fragment)
    position_by_original = {
        int(index): position for position, index in enumerate(population.original_index)
    }
    result: list[V4TrunkApexAssociationEvidence] = []
    for association in associations:
        path = path_by_association.get(association.association_id)
        if path is None or path.availability is not V4StructuralPathAvailability.OBSERVED:
            result.append(association)
            continue
        apex = apex_by_id[association.apex_candidate_id]
        hypothesis_id = _hypothesis_key(apex)
        candidate_fragments = fragments_by_hypothesis.get(hypothesis_id, [])
        trunk_lookup = trunk_id_lookup.get(association.trunk_candidate_id)
        core_positions = set(trunk_lookup[1]) if trunk_lookup is not None else set()
        apex_position = position_by_original.get(apex.supporting_original_index)
        start_ids = {
            fragment.fragment_id
            for fragment in candidate_fragments
            if core_positions.intersection(fragment.positions)
        }
        target_ids = {
            fragment.fragment_id
            for fragment in candidate_fragments
            if (
                apex_position is not None
                and apex_position in fragment.positions
            )
            or (
                max(float(population.hag_m[position]) for position in fragment.positions)
                >= apex.hag_m - scales.ls
                and min(
                    math.hypot(
                        float(population.xyz_local_m[position, 0]) - apex.xyz_local_m[0],
                        float(population.xyz_local_m[position, 1]) - apex.xyz_local_m[1],
                    )
                    for position in fragment.positions
                )
                <= scales.la
            )
        }
        reachable = False
        if start_ids and target_ids:
            pending = list(sorted(start_ids, reverse=True))
            seen = set(start_ids)
            while pending:
                current = pending.pop()
                if current in target_ids:
                    reachable = True
                    break
                for neighbor in fragment_adjacency.get(current, ()):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        pending.append(neighbor)
        fields = {
            name: getattr(association, name)
            for name in V4TrunkApexAssociationEvidence.__dataclass_fields__
            if name != "diagnostic_digest"
        }
        if not start_ids or not target_ids:
            quality = int(association.quality_flags) | int(
                V4AssociationQualityFlag.EVIDENCE_INCOMPLETE
            )
            fields.update(
                association_score_available=False,
                association_score=0.0,
                association_state=V4TrunkApexAssociationState.REVIEW_ASSOCIATION,
                quality_flags=quality,
                reason_flags=quality,
            )
        elif not reachable:
            fields.update(
                tvc=0.0,
                tr=0.0,
                association_score_available=True,
                association_score=0.0,
                association_state=V4TrunkApexAssociationState.UNSUPPORTED_ASSOCIATION,
            )
        result.append(
            _finalize_record(
                V4TrunkApexAssociationEvidence,
                "diagnostic_digest",
                **fields,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Relation, FINAL UNION and assignment
# ---------------------------------------------------------------------------


def _early_relation(
    hypothesis_a: V4ApexCandidateRecord,
    hypothesis_b: V4ApexCandidateRecord,
    scales: V4AlgorithmScales,
    shared_trunk: float,
    distinct_trunk: float,
    connected_xy: bool,
    saddle_gap: float | None,
) -> V4TreeRelationEvidence:
    dx = math.hypot(
        hypothesis_a.xyz_local_m[0] - hypothesis_b.xyz_local_m[0],
        hypothesis_a.xyz_local_m[1] - hypothesis_b.xyz_local_m[1],
    )
    dh = abs(hypothesis_a.hag_m - hypothesis_b.hag_m)
    horizontal_distance_e = 1.0 - 2.0 * c01(dx / scales.lr)
    height_difference_e = 1.0 - 2.0 * c01(dh / scales.la)
    if saddle_gap is None:
        saddle_drop_e = 0.0
    else:
        saddle_drop_e = 1.0 - 2.0 * c01(max(0.0, saddle_gap) / scales.la)
    xy_connectivity_e = signed(1.0 if connected_xy else 0.0)
    empty_gap_e = 0.0
    slice_continuity_e = 0.0
    local_geometry_e = 0.0
    trunk_e = clamp11(shared_trunk - distinct_trunk)
    channels = (
        horizontal_distance_e,
        height_difference_e,
        saddle_drop_e,
        xy_connectivity_e,
        empty_gap_e,
        slice_continuity_e,
        local_geometry_e,
        trunk_e,
    )
    merge = merge_support_score(channels)
    separate = separate_support_score(channels)
    decision = relation_decision(merge, separate)
    provisional = {
        V4TreeRelation.MERGE: "PROVISIONAL_MERGE",
        V4TreeRelation.KEEP_SEPARATE: "PROVISIONAL_KEEP_SEPARATE",
        V4TreeRelation.REVIEW: "PROVISIONAL_REVIEW",
    }[decision]
    return _finalize_record(
        V4TreeRelationEvidence,
        "diagnostic_digest",
        relation_id=f"rel-early-{hypothesis_a.candidate_id}-{hypothesis_b.candidate_id}",
        hypothesis_a_id=_hypothesis_key(hypothesis_a),
        hypothesis_b_id=_hypothesis_key(hypothesis_b),
        horizontal_distance_e=horizontal_distance_e,
        height_difference_e=height_difference_e,
        saddle_drop_e=saddle_drop_e,
        xy_crown_connectivity_e=xy_connectivity_e,
        empty_gap_evidence_e=empty_gap_e,
        slice_continuity_e=slice_continuity_e,
        local_point_geometry_e=local_geometry_e,
        optional_trunk_evidence_e=trunk_e,
        merge_support=merge,
        separate_support=separate,
        relation=decision,
        stage=provisional,
        quality_flags=0,
        reason_flags=0,
    )


def _final_relation(
    hypothesis_a: V4ApexCandidateRecord,
    hypothesis_b: V4ApexCandidateRecord,
    scales: V4AlgorithmScales,
    shared_trunk: float,
    distinct_trunk: float,
    connected_xy: bool,
    slice_continuity_fraction: float,
    slice_continuity_available: bool,
    gap_m: float,
    local_geometry_q: float,
    local_geometry_available: bool,
    saddle_gap: float | None,
) -> V4TreeRelationEvidence:
    dx = math.hypot(
        hypothesis_a.xyz_local_m[0] - hypothesis_b.xyz_local_m[0],
        hypothesis_a.xyz_local_m[1] - hypothesis_b.xyz_local_m[1],
    )
    dh = abs(hypothesis_a.hag_m - hypothesis_b.hag_m)
    horizontal_distance_e = 1.0 - 2.0 * c01(dx / scales.lr)
    height_difference_e = 1.0 - 2.0 * c01(dh / scales.la)
    if saddle_gap is None:
        saddle_drop_e = 0.0
    else:
        saddle_drop_e = 1.0 - 2.0 * c01(max(0.0, saddle_gap) / scales.la)
    xy_connectivity_e = signed(1.0 if connected_xy else 0.0)
    # empty-gap evidence from the actual measured unsupported run.
    empty_gap_e = 1.0 - 2.0 * c01(gap_m / scales.ld)
    # slice continuity from the real adjacent-slice fragment graph.
    slice_continuity_e = signed(slice_continuity_fraction)
    # local point geometry from the real region principal directions/densities.
    if local_geometry_available:
        local_geometry_e = signed(local_geometry_q)
    else:
        local_geometry_e = 0.0
    trunk_e = clamp11(shared_trunk - distinct_trunk)
    channels = (
        horizontal_distance_e,
        height_difference_e,
        saddle_drop_e,
        xy_connectivity_e,
        empty_gap_e,
        slice_continuity_e,
        local_geometry_e,
        trunk_e,
    )
    merge = merge_support_score(channels)
    separate = separate_support_score(channels)
    decision = relation_decision(merge, separate)
    # Final required slice/local evidence missing -> localized REVIEW; it is
    # never converted into an observed -1 or neutral 0.
    if not slice_continuity_available or not local_geometry_available:
        decision = V4TreeRelation.REVIEW
    return _finalize_record(
        V4TreeRelationEvidence,
        "diagnostic_digest",
        relation_id=f"rel-final-{hypothesis_a.candidate_id}-{hypothesis_b.candidate_id}",
        hypothesis_a_id=_hypothesis_key(hypothesis_a),
        hypothesis_b_id=_hypothesis_key(hypothesis_b),
        horizontal_distance_e=horizontal_distance_e,
        height_difference_e=height_difference_e,
        saddle_drop_e=saddle_drop_e,
        xy_crown_connectivity_e=xy_connectivity_e,
        empty_gap_evidence_e=empty_gap_e,
        slice_continuity_e=slice_continuity_e,
        local_point_geometry_e=local_geometry_e,
        optional_trunk_evidence_e=trunk_e,
        merge_support=merge,
        separate_support=separate,
        relation=decision,
        stage="FINAL",
        quality_flags=0,
        reason_flags=0,
    )


def _fragment_xy_bounds(
    population: _NormalPopulation, fragment: _Fragment
) -> tuple[float, float, float, float]:
    xs = [float(population.xyz_local_m[position, 0]) for position in fragment.positions]
    ys = [float(population.xyz_local_m[position, 1]) for position in fragment.positions]
    return (min(xs), min(ys), max(xs), max(ys))


def _fragments_xy_linked(
    fragments_a: list[_Fragment],
    fragments_b: list[_Fragment],
    population: _NormalPopulation,
) -> bool:
    for fragment_a in fragments_a:
        bounds_a = _fragment_xy_bounds(population, fragment_a)
        for fragment_b in fragments_b:
            bounds_b = _fragment_xy_bounds(population, fragment_b)
            if (
                bounds_a[2] >= bounds_b[0]
                and bounds_b[2] >= bounds_a[0]
                and bounds_a[3] >= bounds_b[1]
                and bounds_b[3] >= bounds_a[1]
            ):
                return True
    return False


def _slice_continuity_fraction(
    hypothesis_a: str,
    hypothesis_b: str,
    slices: list[V4TreeSliceSummary],
    fragments: list[_Fragment],
    population: _NormalPopulation,
    adjacent_continuity: dict[str, float] | None = None,
    adjacent_available: dict[str, bool] | None = None,
) -> tuple[float, bool]:
    """Real adjacent-slice fragment-graph continuity.

    Adjacent/overlapping slice pairs are linked only when their measured
    fragments share XY boundary overlap.  Returns ``(fraction, available)``;
    when no adjacent slice pair can be evaluated the evidence is typed missing
    (``available=False``), never a hardcoded zero.
    """
    fragments_by_slice: dict[tuple[str, int], list[_Fragment]] = {}
    for fragment in fragments:
        fragments_by_slice.setdefault(
            (fragment.hypothesis_id, fragment.slice_index), []
        ).append(fragment)
    # Slice records carry the canonical schedule ordinal in the final ``-NNN``
    # suffix.  Empty scheduled slices are intentionally omitted, so deriving an
    # ordinal from record position would silently shift every later fragment.
    slice_index_by_id: dict[str, int] = {
        record.slice_id: int(record.slice_id.rsplit("-", 1)[1]) for record in slices
    }
    slices_a = [s for s in slices if s.hypothesis_id == hypothesis_a]
    slices_b = [s for s in slices if s.hypothesis_id == hypothesis_b]
    if not slices_a or not slices_b:
        return 0.0, False
    evaluated = 0
    linked = 0
    for slice_a in slices_a:
        fragments_a = fragments_by_slice.get(
            (hypothesis_a, slice_index_by_id[slice_a.slice_id]), []
        )
        for slice_b in slices_b:
            overlap = min(slice_a.hag_max_m, slice_b.hag_max_m) - max(
                slice_a.hag_min_m, slice_b.hag_min_m
            )
            if overlap <= 0.0:
                continue
            evaluated += 1
            fragments_b = fragments_by_slice.get(
                (hypothesis_b, slice_index_by_id[slice_b.slice_id]), []
            )
            if _fragments_xy_linked(fragments_a, fragments_b, population):
                linked += 1
    if evaluated == 0:
        return 0.0, False
    cross_fraction = linked / evaluated
    if adjacent_continuity is None or adjacent_available is None:
        return cross_fraction, True
    if not adjacent_available.get(hypothesis_a, False) or not adjacent_available.get(
        hypothesis_b, False
    ):
        return 0.0, False
    return (
        0.50 * cross_fraction
        + 0.25 * adjacent_continuity[hypothesis_a]
        + 0.25 * adjacent_continuity[hypothesis_b],
        True,
    )


def _region_positions(
    apex: V4ApexCandidateRecord,
    grid: _Grid,
    population: _NormalPopulation,
    scales: V4AlgorithmScales,
) -> list[int]:
    apex_x = float(apex.xyz_local_m[0])
    apex_y = float(apex.xyz_local_m[1])
    positions: list[int] = []
    for cell_id in grid.occupied_cell_ids:
        summary = grid.cell_id_to_summary[cell_id]
        if (
            math.hypot(
                summary.cell_center[0] - apex_x, summary.cell_center[1] - apex_y
            )
            <= scales.la
        ):
            positions.extend(_positions_in_cell(grid, cell_id))
    return positions


def _region_principal_axis(
    apex: V4ApexCandidateRecord,
    grid: _Grid,
    population: _NormalPopulation,
    scales: V4AlgorithmScales,
) -> tuple[float, float, float] | None:
    positions = _region_positions(apex, grid, population, scales)
    if len(positions) < TRUNK_COMPONENT_MIN_POINTS:
        return None
    points = np.asarray(
        [population.xyz_local_m[position] for position in positions], dtype=np.float64
    )
    axis, _centroid, lambdas = _pca_axis(points)
    if lambdas[0] <= PCA_ABS_EPS or (
        lambdas[0] - lambdas[1]
    ) / max(lambdas[0], PCA_ABS_EPS) < PCA_REL_GAP_MIN:
        return None
    return tuple(float(value) for value in axis)


def _region_density(
    apex: V4ApexCandidateRecord,
    grid: _Grid,
    scales: V4AlgorithmScales,
) -> float | None:
    apex_x = float(apex.xyz_local_m[0])
    apex_y = float(apex.xyz_local_m[1])
    densities = [
        summary.density
        for cell_id in grid.occupied_cell_ids
        if (
            math.hypot(
                grid.cell_id_to_summary[cell_id].cell_center[0] - apex_x,
                grid.cell_id_to_summary[cell_id].cell_center[1] - apex_y,
            )
            <= scales.la
        )
        for summary in (grid.cell_id_to_summary[cell_id],)
    ]
    return _median(densities) if densities else None


def _region_gap_m(
    apex_a: V4ApexCandidateRecord,
    apex_b: V4ApexCandidateRecord,
    grid: _Grid,
    scales: V4AlgorithmScales,
) -> float:
    """Longest measured unsupported run along the straight connecting corridor."""
    ax = float(apex_a.xyz_local_m[0])
    ay = float(apex_a.xyz_local_m[1])
    bx = float(apex_b.xyz_local_m[0])
    by = float(apex_b.xyz_local_m[1])
    total = math.hypot(bx - ax, by - ay)
    steps = max(1, int(math.ceil(total / scales.lxy)))
    longest = 0.0
    run = 0.0
    summaries = list(grid.cell_id_to_summary.values())
    for step in range(steps + 1):
        x = ax + (bx - ax) * step / steps
        y = ay + (by - ay) * step / steps
        supported = any(
            math.hypot(summary.cell_center[0] - x, summary.cell_center[1] - y)
            <= 0.75 * scales.lxy
            and summary.count >= TRUNK_BIN_MIN_POINTS
            for summary in summaries
        )
        if supported:
            run = 0.0
        else:
            run += scales.lxy
            longest = max(longest, run)
    return longest


def _final_union(
    hypotheses: list[V4ApexCandidateRecord],
    final_relations: list[V4TreeRelationEvidence],
) -> tuple[
    dict[str, int],
    dict[str, V4TreeHypothesisState],
    set[tuple[str, str]],
]:
    """Solve MERGE/KEEP_SEPARATE constraints in one pass; conflicts go REVIEW.

    Returns ``(component_by_key, states, conflict_edges)`` where
    ``conflict_edges`` preserves every final REVIEW edge and every
    KEEP_SEPARATE-inside-a-MERGE conflict so assignment can push the precisely
    affected shared/competing points into REVIEW_REQUIRED.
    """

    parent = {_hypothesis_key(apex): _hypothesis_key(apex) for apex in hypotheses}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    keep_constraints: list[tuple[str, str]] = []
    for relation in final_relations:
        if relation.relation is V4TreeRelation.MERGE:
            union(relation.hypothesis_a_id, relation.hypothesis_b_id)
        elif relation.relation is V4TreeRelation.KEEP_SEPARATE:
            key_a, key_b = sorted((relation.hypothesis_a_id, relation.hypothesis_b_id))
            keep_constraints.append((key_a, key_b))
    # Conflict detection: a KEEP_SEPARATE edge inside one MERGE component.
    conflict_pairs: set[tuple[str, str]] = set()
    for key_a, key_b in keep_constraints:
        if find(key_a) == find(key_b):
            conflict_pairs.add((key_a, key_b))
    # Every final REVIEW edge is also a preserved conflict edge: both
    # hypotheses may stay ACTIVE, but their shared/competing points cannot be
    # ordinarily assignable.
    review_edges: set[tuple[str, str]] = set()
    for relation in final_relations:
        if relation.relation is V4TreeRelation.REVIEW:
            review_edges.add(
                tuple(sorted((relation.hypothesis_a_id, relation.hypothesis_b_id)))
            )
    conflict_edges = conflict_pairs | review_edges
    states: dict[str, V4TreeHypothesisState] = {}
    components: dict[str, int] = {}
    next_component = 0
    for apex in hypotheses:
        key = _hypothesis_key(apex)
        if find(key) == key:
            components[key] = next_component
            next_component += 1
    for apex in hypotheses:
        key = _hypothesis_key(apex)
        root = find(key)
        state = V4TreeHypothesisState.ACTIVE
        for key_a, key_b in conflict_pairs:
            if find(key_a) == find(key_b) and (root == find(key_a)):
                state = V4TreeHypothesisState.REVIEW
                break
        states[key] = state
    # Re-key components through the root so all members share the component id.
    component_by_key: dict[str, int] = {}
    for apex in hypotheses:
        key = _hypothesis_key(apex)
        component_by_key[key] = components[find(key)]
    return component_by_key, states, conflict_edges


def _final_trees_from_union(
    hypotheses: list[V4ApexCandidateRecord],
    component_by_key: dict[str, int],
    states: dict[str, V4TreeHypothesisState],
) -> list[tuple[int, list[V4ApexCandidateRecord], V4TreeHypothesisState]]:
    groups: dict[int, list[V4ApexCandidateRecord]] = {}
    for apex in hypotheses:
        key = _hypothesis_key(apex)
        groups.setdefault(component_by_key[key], []).append(apex)
    result = []
    for component_id in sorted(groups):
        members = sorted(groups[component_id], key=lambda apex: apex.supporting_original_index)
        state = states[_hypothesis_key(members[0])]
        result.append((component_id, members, state))
    return result


def _points_in_domain(
    domain: _HypothesisDomain,
    grid: _Grid,
    population: _NormalPopulation,
) -> list[int]:
    result: list[int] = []
    domain_cells = set(domain.cell_ids)
    for position in range(int(population.original_index.shape[0])):
        if int(grid.point_cell_ids[position]) in domain_cells:
            result.append(position)
    return result


def _fragment_xy_axis(
    population: _NormalPopulation, positions: list[int]
) -> tuple[float, float] | None:
    """Principal XY axis of a fragment (canonical sign), or None when degenerate."""
    if len(positions) < 2:
        return None
    xs = [float(population.xyz_local_m[position, 0]) for position in positions]
    ys = [float(population.xyz_local_m[position, 1]) for position in positions]
    center_x = math.fsum(xs) / len(xs)
    center_y = math.fsum(ys) / len(ys)
    sxx = math.fsum((value - center_x) ** 2 for value in xs)
    syy = math.fsum((value - center_y) ** 2 for value in ys)
    sxy = math.fsum(
        (x_value - center_x) * (y_value - center_y)
        for x_value, y_value in zip(xs, ys, strict=True)
    )
    trace = sxx + syy
    determinant = sxx * syy - sxy * sxy
    discriminant = math.sqrt(max(0.0, trace * trace - 4.0 * determinant))
    lambda_max = 0.5 * (trace + discriminant)
    if lambda_max <= PCA_ABS_EPS:
        return None
    if abs(sxy) <= PCA_ABS_EPS:
        axis = (1.0, 0.0) if sxx >= syy else (0.0, 1.0)
    else:
        axis = (sxy, lambda_max - sxx)
    norm = math.hypot(axis[0], axis[1])
    if norm <= PCA_ABS_EPS:
        return None
    axis = (axis[0] / norm, axis[1] / norm)
    if axis[0] < 0.0 or (axis[0] == 0.0 and axis[1] < 0.0):
        axis = (-axis[0], -axis[1])
    return axis


def _slice_pair_continuity(
    tree_slices: list[V4TreeSliceSummary],
) -> float:
    """Boundary-overlap continuity across adjacent (HAG-overlapping) slices."""
    if len(tree_slices) < 2:
        return 0.5
    ordered = sorted(tree_slices, key=lambda record: record.hag_min_m)
    evaluated = 0
    linked = 0
    for index_a in range(len(ordered)):
        for index_b in range(index_a + 1, len(ordered)):
            overlap = min(ordered[index_a].hag_max_m, ordered[index_b].hag_max_m) - max(
                ordered[index_a].hag_min_m, ordered[index_b].hag_min_m
            )
            if overlap <= 0.0:
                continue
            evaluated += 1
            box_a_min = ordered[index_a].bounds_min_xyz
            box_a_max = ordered[index_a].bounds_max_xyz
            box_b_min = ordered[index_b].bounds_min_xyz
            box_b_max = ordered[index_b].bounds_max_xyz
            if (
                box_a_max[0] >= box_b_min[0]
                and box_b_max[0] >= box_a_min[0]
                and box_a_max[1] >= box_b_min[1]
                and box_b_max[1] >= box_a_min[1]
            ):
                linked += 1
    if evaluated == 0:
        return 0.5
    return linked / evaluated


def _gap_from_channel(empty_gap_channel: float, scales: V4AlgorithmScales) -> float:
    """Recover the measured unsupported-run length from the near() channel."""
    return (1.0 - empty_gap_channel) * scales.ld


def _assign_points(
    population: _NormalPopulation,
    grid: _Grid,
    trees: list[tuple[int, list[V4ApexCandidateRecord], V4TreeHypothesisState]],
    domains: dict[str, _HypothesisDomain],
    claims_by_cell: dict[int, list[_Claim]],
    review_hypothesis_domains: tuple[str, ...],
    scales: V4AlgorithmScales,
    config: V4TreeSegmentationConfig,
    extra_review_cells: set[int] | None = None,
    slices: list[V4TreeSliceSummary] | None = None,
    fragments: list[_Fragment] | None = None,
    conflict_edges: set[tuple[str, str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = int(population.original_index.shape[0])
    state_by_point = np.full(count, int(V4TreeAssignmentState.UNASSIGNED), dtype="|u1")
    tree_id_by_point = np.full(count, -1, dtype="<i4")
    confidence_by_point = np.full(count, 0.0, dtype="<f4")
    reason_by_point = np.full(count, 0, dtype="<u4")
    review_cells: set[int] = set()
    for hypothesis_id in review_hypothesis_domains:
        domain = domains.get(hypothesis_id)
        if domain is not None:
            review_cells.update(domain.cell_ids)
    if extra_review_cells:
        review_cells.update(extra_review_cells)
    slices = slices or []
    fragments = fragments or []
    # Per-hypothesis fragment support inventory: actual measured fragment
    # membership, median HAG/density and principal XY structure axis.
    fragment_positions: dict[str, set[int]] = {}
    for fragment in fragments:
        fragment_positions.setdefault(fragment.hypothesis_id, set()).update(
            fragment.positions
        )
    fragment_hag_median: dict[str, float] = {}
    fragment_density: dict[str, float] = {}
    fragment_centroid_xy: dict[str, tuple[float, float]] = {}
    fragment_axis_xy: dict[str, tuple[float, float]] = {}
    for hypothesis_id, positions_set in fragment_positions.items():
        ordered = sorted(positions_set)
        if not ordered:
            continue
        hag_values = [float(population.hag_m[pos]) for pos in ordered]
        fragment_hag_median[hypothesis_id] = _quantile(sorted(hag_values), 0.5)
        distinct_cells = len(set(grid.point_cell_ids[pos] for pos in ordered))
        area = (scales.lxy * scales.lxy) * max(distinct_cells, 1)
        fragment_density[hypothesis_id] = _raw_weight_sum(population, ordered) / area
        axis = _fragment_xy_axis(population, ordered)
        if axis is not None:
            center_x = (
                math.fsum(float(population.xyz_local_m[pos, 0]) for pos in ordered)
                / len(ordered)
            )
            center_y = (
                math.fsum(float(population.xyz_local_m[pos, 1]) for pos in ordered)
                / len(ordered)
            )
            fragment_centroid_xy[hypothesis_id] = (center_x, center_y)
            fragment_axis_xy[hypothesis_id] = axis
    # Per-tree adjacent-slice continuity from the actual slice records.
    tree_continuity: dict[int, float] = {}
    for tree_id, (_component_id, members, state) in enumerate(trees):
        if state is not V4TreeHypothesisState.ACTIVE:
            continue
        member_hypotheses = [_hypothesis_key(member) for member in members]
        tree_slices = [
            record for record in slices if record.hypothesis_id in member_hypotheses
        ]
        tree_continuity[tree_id] = _slice_pair_continuity(tree_slices)
    # Precompute per-tree hypothesis membership and the apex/hypothesis lookup
    # tables once, so point assignment never rescans the whole tree list per
    # point (no O(N x T) membership work).
    apex_by_id: dict[str, V4ApexCandidateRecord] = {}
    hypothesis_to_tree: dict[str, int] = {}
    for tree_id, (_component_id, members, state) in enumerate(trees):
        for member in members:
            apex_by_id[member.candidate_id] = member
        if state is V4TreeHypothesisState.ACTIVE:
            for member in members:
                hypothesis_to_tree.setdefault(_hypothesis_key(member), tree_id)
    # A cell is conflict-affected when any preserved conflict edge is claimed by
    # both endpoints; this depends only on the cell, never on the point.
    conflict_affected_cells: set[int] = set()
    if conflict_edges:
        for cell_id, cell_claims in claims_by_cell.items():
            claimed_hypotheses = {claim.hypothesis_id for claim in cell_claims}
            for key_a, key_b in conflict_edges:
                if key_a in claimed_hypotheses and key_b in claimed_hypotheses:
                    conflict_affected_cells.add(cell_id)
                    break
    for position in range(count):
        cell_id = int(grid.point_cell_ids[position])
        if cell_id in review_cells:
            # Points in a REVIEW-hypothesis domain carry mandatory review.
            state_by_point[position] = int(V4TreeAssignmentState.REVIEW_REQUIRED)
            tree_id_by_point[position] = -1
            confidence_by_point[position] = 0.0
            reason_by_point[position] = int(V4AssignmentReasonFlag.REVIEW_MANDATORY)
            continue
        # Final REVIEW edge/component conflict domain: a point claimed by both
        # endpoints of a preserved conflict edge cannot be ordinarily
        # assignable, even when both hypotheses stay ACTIVE.
        if cell_id in conflict_affected_cells:
            state_by_point[position] = int(V4TreeAssignmentState.REVIEW_REQUIRED)
            tree_id_by_point[position] = -1
            confidence_by_point[position] = 0.0
            reason_by_point[position] = int(V4AssignmentReasonFlag.REVIEW_MANDATORY)
            continue
        cell_claims = claims_by_cell.get(cell_id, [])
        candidate_tree_ids: list[int] = []
        candidate_channels: list[tuple[float, ...]] = []
        candidate_hypothesis_keys: list[str] = []
        point_xyz = population.xyz_local_m[position]
        point_hag = float(population.hag_m[position])
        # Only trees that actually claim this cell are candidates.  For each
        # tree the first claim in cell_claims order is used, and candidate trees
        # are visited in enumerate(trees) order, exactly matching the frozen
        # candidate ordering.
        first_claim_by_tree: dict[int, _Claim] = {}
        for claim in cell_claims:
            tree_id = hypothesis_to_tree.get(claim.hypothesis_id)
            if tree_id is not None and tree_id not in first_claim_by_tree:
                first_claim_by_tree[tree_id] = claim
        for tree_id in sorted(first_claim_by_tree):
            claim = first_claim_by_tree[tree_id]
            member = claim.apex_candidate_id
            apex = apex_by_id[member]
            hypothesis_id = claim.hypothesis_id
            # slice measured support: actual fragment membership, never the 2D
            # responsibility domain membership.
            is_fragment_member = position in fragment_positions.get(hypothesis_id, set())
            # structure direction from the actual fragment principal axis.
            structure_direction: float | None = None
            if is_fragment_member and hypothesis_id in fragment_axis_xy:
                centroid = fragment_centroid_xy[hypothesis_id]
                ux = float(point_xyz[0]) - centroid[0]
                uy = float(point_xyz[1]) - centroid[1]
                norm = math.hypot(ux, uy)
                if norm <= PCA_ABS_EPS:
                    structure_direction = 1.0
                else:
                    axis = fragment_axis_xy[hypothesis_id]
                    structure_direction = abs((ux * axis[0] + uy * axis[1]) / norm)
            gap_m = _gap_from_channel(claim.channels[6], scales)
            channels = _assignment_channels(
                population,
                grid,
                point_xyz,
                point_hag,
                cell_id,
                claim,
                apex,
                is_fragment_member,
                tree_continuity.get(tree_id, 0.0),
                structure_direction,
                gap_m,
                hypothesis_id,
                fragment_hag_median,
                fragment_density,
                scales,
                config,
            )
            candidate_tree_ids.append(tree_id)
            candidate_channels.append(channels)
            candidate_hypothesis_keys.append(hypothesis_id)
        mandatory_review = False
        if not candidate_tree_ids:
            state_by_point[position] = int(V4TreeAssignmentState.UNASSIGNED)
            confidence_by_point[position] = float(
                assignment_confidence(
                    V4TreeAssignmentState.UNASSIGNED, 0.0, None
                )
            )
            reason_by_point[position] = int(V4AssignmentReasonFlag.UNASSIGNED_LOW_SUPPORT)
            continue
        # Candidate ranking uses the FINAL nine-channel assignment support,
        # never an intermediate eight-channel base.
        bases = [
            assignment_base(channels[:7] + channels[8:])
            for channels in candidate_channels
        ]
        candidate_supports: list[float] = []
        for index, channels in enumerate(candidate_channels):
            best_other_base = max(
                (base for j, base in enumerate(bases) if j != index), default=None
            )
            competitor = assignment_competitor_channel(bases[index], best_other_base)
            nine = list(channels)
            nine[7] = competitor
            candidate_supports.append(assignment_support_score(tuple(nine)))
        ordered_indices = sorted(
            range(len(candidate_supports)),
            key=lambda index: (
                -candidate_supports[index],
                candidate_hypothesis_keys[index],
            ),
        )
        best_index = ordered_indices[0]
        best_support = candidate_supports[best_index]
        second_support = (
            candidate_supports[ordered_indices[1]] if len(ordered_indices) > 1 else None
        )
        # Mandatory review if the segmentation-quality channel of the best
        # candidate is -1.
        if candidate_channels[best_index][8] < 0.0:
            mandatory_review = True
        decision = assignment_decision(
            support=best_support,
            second=second_support,
            mandatory_review=mandatory_review,
            comparison_complete=True,
        )
        if decision is V4TreeAssignmentState.ASSIGNED:
            state_by_point[position] = int(V4TreeAssignmentState.ASSIGNED)
            tree_id_by_point[position] = candidate_tree_ids[best_index]
            confidence_by_point[position] = float(
                assignment_confidence(
                    V4TreeAssignmentState.ASSIGNED, best_support, second_support
                )
            )
            reason_by_point[position] = int(
                V4AssignmentReasonFlag.ASSIGNED_CLEAR_BEST
            )
        elif decision is V4TreeAssignmentState.REVIEW_REQUIRED:
            state_by_point[position] = int(V4TreeAssignmentState.REVIEW_REQUIRED)
            tree_id_by_point[position] = -1
            confidence_by_point[position] = 0.0
            reason_by_point[position] = int(
                V4AssignmentReasonFlag.REVIEW_MANDATORY
                if mandatory_review
                else V4AssignmentReasonFlag.REVIEW_MARGIN
            )
        else:
            state_by_point[position] = int(V4TreeAssignmentState.UNASSIGNED)
            tree_id_by_point[position] = -1
            confidence_by_point[position] = float(
                assignment_confidence(
                    V4TreeAssignmentState.UNASSIGNED, best_support, second_support
                )
            )
            reason_by_point[position] = int(V4AssignmentReasonFlag.UNASSIGNED_LOW_SUPPORT)
    for array in (
        state_by_point,
        tree_id_by_point,
        confidence_by_point,
        reason_by_point,
    ):
        array.setflags(write=False)
    return state_by_point, tree_id_by_point, confidence_by_point, reason_by_point


def _assignment_channels(
    population: _NormalPopulation,
    grid: _Grid,
    point_xyz: np.ndarray,
    point_hag: float,
    cell_id: int,
    claim: _Claim,
    apex: V4ApexCandidateRecord,
    is_fragment_member: bool,
    tree_continuity: float,
    structure_direction: float | None,
    gap_m: float,
    hypothesis_id: str,
    fragment_hag_median: dict[str, float],
    fragment_density: dict[str, float],
    scales: V4AlgorithmScales,
    config: V4TreeSegmentationConfig,
) -> tuple[float, ...]:
    # 1. slice measured support: actual measured fragment membership.
    slice_support_e = 1.0 if is_fragment_member else -1.0
    # 2. adjacent-slice continuity from the actual slice evidence.
    continuity_e = signed(tree_continuity)
    # 3. XY prior
    xy_prior_e = signed(claim.score)
    # 4. apex support
    d_xy = math.hypot(
        point_xyz[0] - apex.xyz_local_m[0], point_xyz[1] - apex.xyz_local_m[1]
    )
    apex_support_e = signed(0.5 * apex.apex_support_score + 0.5 * near(d_xy, scales.lr))
    # 5. local HAG/density: point vs the measured fragment median HAG and the
    #    bidirectional local density ratio.
    summary = grid.cell_id_to_summary.get(cell_id)
    point_rho = summary.density if summary is not None else apex.density_points_per_m2
    if hypothesis_id in fragment_hag_median:
        delta_hag = point_hag - fragment_hag_median[hypothesis_id]
    else:
        delta_hag = point_hag - apex.hag_m
    if hypothesis_id in fragment_density:
        fragment_rho = fragment_density[hypothesis_id]
        density_ratio = min(
            safe_ratio(point_rho, fragment_rho, scales.rho_floor),
            safe_ratio(fragment_rho, point_rho, scales.rho_floor),
        )
    else:
        density_ratio = min(
            safe_ratio(point_rho, apex.density_points_per_m2, scales.rho_floor),
            safe_ratio(apex.density_points_per_m2, point_rho, scales.rho_floor),
        )
    local_e = signed(
        0.5 * similar(delta_hag, scales.ls)
        + 0.5 * density_ratio
    )
    # 6. structure direction: actual fragment principal-axis alignment, or the
    #    frozen optional NOT_AVAILABLE -> e=0 when PCA support is insufficient.
    if structure_direction is not None:
        structure_e = signed(structure_direction)
    else:
        structure_e = 0.0
    # 7. gap/occlusion: measured unsupported run from the claim path.
    gap_e = 1.0 - 2.0 * c01(gap_m / scales.ld)
    # 8. competitor margin (filled later).
    competitor_e = 0.0
    # 9. segmentation quality.
    quality_e = 1.0
    return (
        slice_support_e,
        continuity_e,
        xy_prior_e,
        apex_support_e,
        local_e,
        structure_e,
        gap_e,
        competitor_e,
        quality_e,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _xy_bucket_key(x_value: float, y_value: float, radius: float) -> tuple[int, int]:
    return (int(math.floor(x_value / radius)), int(math.floor(y_value / radius)))


def _local_trunk_apex_pairs(
    trunks: list[V4TrunkCandidateRecord],
    apices: list[V4ApexCandidateRecord],
    radius: float,
    maximum_local_edges: int,
) -> list[tuple[V4TrunkCandidateRecord, V4ApexCandidateRecord]]:
    apex_buckets: dict[tuple[int, int], list[V4ApexCandidateRecord]] = {}
    for apex in apices:
        apex_buckets.setdefault(
            _xy_bucket_key(apex.xyz_local_m[0], apex.xyz_local_m[1], radius), []
        ).append(apex)
    pairs: list[tuple[V4TrunkCandidateRecord, V4ApexCandidateRecord]] = []
    evaluated = 0
    for trunk in trunks:
        trunk_key = _xy_bucket_key(trunk.centroid_xyz[0], trunk.centroid_xyz[1], radius)
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for apex in apex_buckets.get(
                    (trunk_key[0] + offset_x, trunk_key[1] + offset_y), []
                ):
                    evaluated += 1
                    if (
                        _projected_bytes(
                            evaluated, _PROJECTED_PAIR_EVALUATION_BYTES
                        )
                        > _MAXIMUM_LOCAL_WORKING_BYTES
                    ):
                        _raise_local_edge_resource_limit()
                    if math.hypot(
                        trunk.centroid_xyz[0] - apex.xyz_local_m[0],
                        trunk.centroid_xyz[1] - apex.xyz_local_m[1],
                    ) <= radius:
                        pairs.append((trunk, apex))
                        if (
                            _projected_bytes(len(pairs), _PROJECTED_EDGE_PAIR_BYTES)
                            > _MAXIMUM_LOCAL_WORKING_BYTES
                        ):
                            _raise_local_edge_resource_limit()
    return pairs


def _local_apex_pairs(
    apices: list[V4ApexCandidateRecord],
    radius: float,
    maximum_local_edges: int,
) -> list[tuple[V4ApexCandidateRecord, V4ApexCandidateRecord]]:
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, apex in enumerate(apices):
        buckets.setdefault(
            _xy_bucket_key(apex.xyz_local_m[0], apex.xyz_local_m[1], radius), []
        ).append(index)
    pairs: list[tuple[V4ApexCandidateRecord, V4ApexCandidateRecord]] = []
    evaluated = 0
    for index_a, apex_a in enumerate(apices):
        bucket = _xy_bucket_key(apex_a.xyz_local_m[0], apex_a.xyz_local_m[1], radius)
        candidate_indexes: set[int] = set()
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                candidate_indexes.update(
                    buckets.get((bucket[0] + offset_x, bucket[1] + offset_y), [])
                )
        for index_b in sorted(index for index in candidate_indexes if index > index_a):
            evaluated += 1
            if (
                _projected_bytes(evaluated, _PROJECTED_PAIR_EVALUATION_BYTES)
                > _MAXIMUM_LOCAL_WORKING_BYTES
            ):
                _raise_local_edge_resource_limit()
            apex_b = apices[index_b]
            if math.hypot(
                apex_a.xyz_local_m[0] - apex_b.xyz_local_m[0],
                apex_a.xyz_local_m[1] - apex_b.xyz_local_m[1],
            ) <= radius:
                pairs.append((apex_a, apex_b))
                if (
                    _projected_bytes(len(pairs), _PROJECTED_EDGE_PAIR_BYTES)
                    > _MAXIMUM_LOCAL_WORKING_BYTES
                ):
                    _raise_local_edge_resource_limit()
    return pairs


def _check_authority_guard(
    handoff: V4InputHandoffResult, config: V4TreeSegmentationConfig
) -> None:
    """Enforce the frozen upstream/downstream authority matrix.

    Formal V4 execution requires matching formal authority.  The upstream FRV
    authority is recovered only from the validated, digest-bound embedded
    membership artifact (never from a caller assertion).  TEST_ONLY and
    CALIBRATION_CANDIDATE V4 statuses retain their existing non-formal
    behavior; every formal status requires a matching formal upstream purpose.
    This guard is provenance/authority-only: it changes neither Algorithm-V1
    nor point meaning.
    """

    if config.status in (
        V4ConfigStatus.TEST_ONLY,
        V4ConfigStatus.CALIBRATION_CANDIDATE,
    ):
        return
    view = handoff.validated_membership
    artifact = getattr(view, "membership_artifact", None)
    upstream = (
        getattr(artifact, "configuration_purpose", None)
        if artifact is not None
        else None
    )
    if upstream not in (
        FrvConfigurationPurpose.REVIEWED_FIXED_POLICY,
        FrvConfigurationPurpose.PRODUCTION_APPROVED,
    ):
        raise V4TreeSegmentationError(
            code=V4TreeSegmentationErrorCode.CONFIG_STATUS_UNSUPPORTED,
            field_name="configuration_purpose",
        )
    if upstream is FrvConfigurationPurpose.REVIEWED_FIXED_POLICY:
        if config.status is V4ConfigStatus.REVIEWED_FIXED_POLICY:
            return
        raise V4TreeSegmentationError(
            code=V4TreeSegmentationErrorCode.CONFIG_STATUS_UNSUPPORTED,
            field_name="status",
        )
    if config.status is V4ConfigStatus.PRODUCTION_APPROVED:
        return
    raise V4TreeSegmentationError(
        code=V4TreeSegmentationErrorCode.CONFIG_STATUS_UNSUPPORTED,
        field_name="status",
    )


def segment_v4_trees(
    handoff: V4InputHandoffResult,
    config: V4TreeSegmentationConfig,
) -> V4TreeSegmentationResult:
    """Run the full V4 single-tree segmentation pipeline."""

    _check_authority_guard(handoff, config)
    population = _canonical_population(handoff)
    scales = derive_algorithm_scales(config)
    grid = _build_grid(population, scales, config)
    apex_records = _detect_apex_candidates(population, grid, scales, config)
    accepted_apices = [
        record
        for record in apex_records
        if not (record.quality_flags & int(V4ApexQualityFlag.SUPPRESSED))
    ]
    # Apices suppressed because the required prominence channel is NOT_AVAILABLE
    # cannot be formally accepted; their measured population follows REVIEW.
    review_prominence_apices = [
        record
        for record in apex_records
        if (record.quality_flags & int(V4ApexQualityFlag.SUPPRESSED))
        and (record.quality_flags & int(V4ApexQualityFlag.PROMINENCE_MISSING))
    ]
    trunk_records, trunk_id_lookup = _detect_trunk_candidates(
        population, grid, scales, config
    )
    accepted_trunks = [
        record
        for record in trunk_records
        if record.candidate_state is V4TrunkCandidateState.ACCEPTED
    ]
    # Structural paths and associations.
    voxels = _build_voxels(population, grid, scales)
    associations: list[V4TrunkApexAssociationEvidence] = []
    structural_paths: list[V4StructuralPathEvidence] = []
    path_by_association: dict[str, V4StructuralPathEvidence] = {}
    for trunk, apex in _local_trunk_apex_pairs(
        accepted_trunks, accepted_apices, scales.lr, config.maximum_local_edges
    ):
        representative_position, core_positions = trunk_id_lookup[trunk.candidate_id]
        trunk_representative_xyz = tuple(
            float(value) for value in population.xyz_local_m[representative_position]
        )
        trunk_representative_hag = float(population.hag_m[representative_position])
        result = _compute_association(
            trunk,
            core_positions,
            trunk_representative_xyz,
            trunk_representative_hag,
            apex,
            voxels,
            population,
            grid,
            scales,
            config,
        )
        associations.append(result.evidence)
        if result.path is not None:
            structural_paths.append(result.path)
            path_by_association[result.evidence.association_id] = result.path
    # Rotating profiles for accepted apices.
    profiles = [
        _rotating_profile(apex, grid, scales, config) for apex in accepted_apices
    ]
    # Frozen profile_confirmed_stop authority: a NOT_AVAILABLE profile never
    # hard-stops; only confirmed valleys / measured gaps feed responsibility.
    profile_stop_by_hypothesis: dict[str, _ProfileStopInfo] = {}
    for apex, profile in zip(accepted_apices, profiles, strict=True):
        if profile.profile_state is V4RotatingProfileState.PROFILE_NOT_AVAILABLE:
            continue
        stop = _profile_stop_info(profile, scales)
        if stop is not None:
            profile_stop_by_hypothesis[_hypothesis_key(apex)] = stop
    # Responsibility.
    claims_by_cell = _responsibility_solver(
        accepted_apices,
        grid,
        scales,
        config,
        profile_stop_by_hypothesis=profile_stop_by_hypothesis,
    )
    # Claims for apices suppressed by a missing required prominence channel:
    # their measured population follows the frozen REVIEW path.
    prominence_review_cells: set[int] = set()
    if review_prominence_apices:
        cell_summaries = grid.cell_id_to_summary
        for apex in review_prominence_apices:
            review_claims = _hypothesis_claims(
                apex,
                _hypothesis_key(apex),
                grid,
                cell_summaries,
                scales,
                config,
            )
            prominence_review_cells.update(review_claims.keys())
    domains = _responsibility_domains(claims_by_cell)
    slices, fragments = _build_slices_and_fragments(
        domains, claims_by_cell, population, grid, scales, config
    )
    fragment_adjacency, adjacent_continuity, adjacent_available = (
        _build_adjacent_fragment_graph(fragments, population, scales, config)
    )
    # Early CSP paths are evidence only.  Revalidate each materialized
    # association against the measured adjacent-slice fragment graph before it
    # can contribute final trunk relation evidence.
    associations = _revalidate_associations_with_fragment_graph(
        associations,
        path_by_association,
        fragments,
        fragment_adjacency,
        trunk_id_lookup,
        {apex.candidate_id: apex for apex in accepted_apices},
        population,
        scales,
    )
    # Trunk relation evidence is consumed only for locally related apex pairs;
    # computing the global all-pairs matrix would be unnecessary O(A^2 x T^2)
    # work, so evidence is evaluated exactly for the local pairs below.
    local_apex_pairs = _local_apex_pairs(
        accepted_apices, scales.lr, config.maximum_local_edges
    )
    trunk_relation_by_pair = _trunk_relation_evidence(
        accepted_apices,
        associations,
        accepted_trunks,
        scales,
        pairs=local_apex_pairs,
    )
    # Per-apex local region evidence for the final relation channels.
    region_axis_by_apex: dict[str, tuple[float, float, float] | None] = {}
    region_density_by_apex: dict[str, float | None] = {}
    for apex in accepted_apices:
        region_axis_by_apex[apex.candidate_id] = _region_principal_axis(
            apex, grid, population, scales
        )
        region_density_by_apex[apex.candidate_id] = _region_density(apex, grid, scales)
    early_relations: list[V4TreeRelationEvidence] = []
    final_relations: list[V4TreeRelationEvidence] = []
    for apex_a, apex_b in local_apex_pairs:
        key_a = _hypothesis_key(apex_a)
        key_b = _hypothesis_key(apex_b)
        shared_trunk, distinct_trunk = trunk_relation_by_pair.get(
            tuple(sorted((key_a, key_b))), (0.0, 0.0)
        )
        connected_xy = _cells_xy_connected(apex_a, apex_b, claims_by_cell, grid)
        saddle_gap = _saddle_gap(apex_a, apex_b, grid, scales)
        early_relations.append(
            _early_relation(
                apex_a,
                apex_b,
                scales,
                shared_trunk,
                distinct_trunk,
                connected_xy,
                saddle_gap,
            )
        )
        continuity, continuity_available = _slice_continuity_fraction(
            key_a,
            key_b,
            slices,
            fragments,
            population,
            adjacent_continuity,
            adjacent_available,
        )
        axis_a = region_axis_by_apex.get(apex_a.candidate_id)
        axis_b = region_axis_by_apex.get(apex_b.candidate_id)
        rho_a = region_density_by_apex.get(apex_a.candidate_id)
        rho_b = region_density_by_apex.get(apex_b.candidate_id)
        local_geometry_available = (
            axis_a is not None
            and axis_b is not None
            and rho_a is not None
            and rho_b is not None
        )
        if local_geometry_available:
            abs_dot = abs(sum(a * b for a, b in zip(axis_a, axis_b, strict=True)))
            density_sim = min(
                safe_ratio(rho_a, rho_b, scales.rho_floor),
                safe_ratio(rho_b, rho_a, scales.rho_floor),
            )
            local_geometry_q = 0.5 * abs_dot + 0.5 * density_sim
        else:
            local_geometry_q = 0.0
        gap_m = _region_gap_m(apex_a, apex_b, grid, scales)
        final_relations.append(
            _final_relation(
                apex_a,
                apex_b,
                scales,
                shared_trunk,
                distinct_trunk,
                connected_xy,
                continuity,
                continuity_available,
                gap_m,
                local_geometry_q,
                local_geometry_available,
                saddle_gap,
            )
        )
    component_by_key, states, conflict_edges = _final_union(
        accepted_apices, final_relations
    )
    trees = _final_trees_from_union(accepted_apices, component_by_key, states)
    # Keep only ACTIVE trees for formal Tree IDs; REVIEW hypotheses produce no tree.
    formal_trees = [
        (component_id, members, state)
        for component_id, members, state in trees
        if state is V4TreeHypothesisState.ACTIVE
    ]
    if not accepted_apices or not formal_trees:
        # N>0 without a valid apex, or union conflict without an active tree:
        # every normal point is REVIEW_REQUIRED and tree_count is zero.
        state_by_point = np.full(
            int(population.original_index.shape[0]),
            int(V4TreeAssignmentState.REVIEW_REQUIRED),
            dtype="|u1",
        )
        tree_id_by_point = np.full(
            int(population.original_index.shape[0]), -1, dtype="<i4"
        )
        confidence_by_point = np.zeros(
            int(population.original_index.shape[0]), dtype="<f4"
        )
        reason_by_point = np.full(
            int(population.original_index.shape[0]),
            int(V4AssignmentReasonFlag.REVIEW_MANDATORY),
            dtype="<u4",
        )
        state_by_point.setflags(write=False)
        tree_id_by_point.setflags(write=False)
        confidence_by_point.setflags(write=False)
        reason_by_point.setflags(write=False)
        tree_records: list[V4TreeInstanceRecord] = []
        positions_by_tree: dict[int, list[int]] = {}
        accepted_trunk_records = [
            record
            for record in trunk_records
            if record.candidate_state is V4TrunkCandidateState.ACCEPTED
        ]
    else:
        review_hypothesis_domains = tuple(
            key
            for key, state in states.items()
            if state is V4TreeHypothesisState.REVIEW
        )
        state_by_point, tree_id_by_point, confidence_by_point, reason_by_point = _assign_points(
            population,
            grid,
            formal_trees,
            domains,
            claims_by_cell,
            review_hypothesis_domains,
            scales,
            config,
            extra_review_cells=prominence_review_cells,
            slices=slices,
            fragments=fragments,
            conflict_edges=conflict_edges,
        )
        accepted_trunk_records = [
            record
            for record in trunk_records
            if record.candidate_state is V4TrunkCandidateState.ACCEPTED
        ]
        # Single O(N) grouped membership representation, built once and reused
        # by tree ordering, tree-record construction and ragged members.
        positions_by_tree = _group_assigned_positions(
            population, tree_id_by_point, state_by_point
        )
        # Deterministic Tree ID ordering: primary apex original index then the
        # minimum assigned member original index, so ragged members are globally
        # strictly increasing across trees.
        formal_trees, tree_id_remap = _order_final_trees(
            formal_trees, population, positions_by_tree
        )
        remapped_ids = np.full(tree_id_by_point.shape, -1, dtype="<i4")
        for position in range(tree_id_by_point.shape[0]):
            value = int(tree_id_by_point[position])
            if value >= 0:
                remapped_ids[position] = tree_id_remap[value]
        remapped_ids.setflags(write=False)
        tree_id_by_point = remapped_ids
        positions_by_tree = {
            tree_id_remap[old_id]: positions
            for old_id, positions in positions_by_tree.items()
        }
        tree_records = _build_tree_records(
            population,
            grid,
            formal_trees,
            domains,
            claims_by_cell,
            slices,
            fragments,
            associations,
            accepted_trunk_records,
            positions_by_tree,
            confidence_by_point,
            scales,
        )
    ordered_members, offsets = _ragged_members(
        tree_records, population, positions_by_tree
    )
    audit = _build_audit(
        population,
        apex_records,
        trunk_records,
        associations,
        structural_paths,
        profiles,
        accepted_apices,
        slices,
        early_relations,
        final_relations,
        tree_records,
        states,
    )
    result = V4TreeSegmentationResult(
        schema_version=V4_TREE_SEGMENTATION_SCHEMA_VERSION,
        algorithm_version=V4_TREE_SEGMENTATION_ALGORITHM_VERSION,
        source_identity=population.source_identity,
        config=config,
        handoff_digest=population.handoff_digest,
        validated_view_digest=population.validated_view_digest,
        coverage_report_digest=population.coverage_report_digest,
        original_index=_ro_array(np.asarray(population.original_index, dtype="<i8")),
        xyz_local_m=_ro_array(np.asarray(population.xyz_local_m, dtype="<f8").copy()),
        hag_m=_ro_array(np.asarray(population.hag_m, dtype="<f4").copy()),
        assignment_state_by_point=_ro_array(state_by_point),
        tree_id_by_point=_ro_array(np.asarray(tree_id_by_point, dtype="<i4")),
        assignment_confidence_by_point=_ro_array(
            np.asarray(confidence_by_point, dtype="<f4")
        ),
        assignment_reason_flags_by_point=_ro_array(
            np.asarray(reason_by_point, dtype="<u4")
        ),
        tree_member_offsets=_ro_array(np.asarray(offsets, dtype="<i8")),
        ordered_tree_member_original_index=_ro_array(
            np.asarray(ordered_members, dtype="<i8")
        ),
        apex_records=tuple(apex_records),
        trunk_records=tuple(trunk_records),
        trunk_apex_association_records=tuple(associations),
        structural_path_records=tuple(structural_paths),
        rotating_profile_records=tuple(profiles),
        hypothesis_records=_build_hypothesis_records(accepted_apices, states),
        slice_records=tuple(slices),
        relation_records=tuple(early_relations + final_relations),
        tree_records=tuple(tree_records),
        crown_boundary_vertices=_ro_array(
            np.zeros((0, 2), dtype="<f8")
        ),
        crown_boundary_offsets=_ro_array(np.zeros(len(tree_records) + 1, dtype="<i8")),
        audit=audit,
        result_digest="0" * 64,
    )
    result = _with_result_digest(result)
    validate_v4_tree_segmentation_result(result)
    return result


def _with_result_digest(result: V4TreeSegmentationResult) -> V4TreeSegmentationResult:
    clone = V4TreeSegmentationResult.__new__(V4TreeSegmentationResult)
    for name in V4TreeSegmentationResult.__dataclass_fields__:
        if name == "result_digest":
            object.__setattr__(clone, name, result.computed_result_digest())
        else:
            object.__setattr__(clone, name, getattr(result, name))
    return clone


def _ro_array(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _cells_xy_connected(
    apex_a: V4ApexCandidateRecord,
    apex_b: V4ApexCandidateRecord,
    claims_by_cell: dict[int, list[_Claim]],
    grid: _Grid,
) -> bool:
    key_a = _hypothesis_key(apex_a)
    key_b = _hypothesis_key(apex_b)
    a_cells = {
        cell_id
        for cell_id, claims in claims_by_cell.items()
        if any(claim.hypothesis_id == key_a for claim in claims)
    }
    b_cells = {
        cell_id
        for cell_id, claims in claims_by_cell.items()
        if any(claim.hypothesis_id == key_b for claim in claims)
    }
    for cell_id in a_cells:
        for neighbor in grid.neighbors_8(cell_id):
            if neighbor in b_cells:
                return True
    return False


def _saddle_gap(
    apex_a: V4ApexCandidateRecord,
    apex_b: V4ApexCandidateRecord,
    grid: _Grid,
    scales: V4AlgorithmScales,
) -> float | None:
    midpoint = (
        0.5 * (apex_a.xyz_local_m[0] + apex_b.xyz_local_m[0]),
        0.5 * (apex_a.xyz_local_m[1] + apex_b.xyz_local_m[1]),
    )
    best = None
    for cell_id in grid.occupied_cell_ids:
        summary = grid.cell_id_to_summary[cell_id]
        distance = math.hypot(
            summary.cell_center[0] - midpoint[0],
            summary.cell_center[1] - midpoint[1],
        )
        if distance <= scales.la and (best is None or summary.chm_height < best):
            best = summary.chm_height
    if best is None:
        return None
    lower_apex = min(apex_a.hag_m, apex_b.hag_m)
    return max(0.0, lower_apex - best)


def _trunk_relation_evidence(
    apices: list[V4ApexCandidateRecord],
    associations: list[V4TrunkApexAssociationEvidence],
    accepted_trunks: list[V4TrunkCandidateRecord],
    scales: V4AlgorithmScales,
    *,
    pairs: list[tuple[V4ApexCandidateRecord, V4ApexCandidateRecord]] | None = None,
) -> dict[tuple[str, str], tuple[float, float]]:
    result: dict[tuple[str, str], tuple[float, float]] = {}
    if pairs is None:
        pairs = [
            (apices[index_a], apices[index_b])
            for index_a in range(len(apices))
            for index_b in range(index_a + 1, len(apices))
        ]
    trunk_by_id = {trunk.candidate_id: trunk for trunk in accepted_trunks}
    apex_by_id = {apex.candidate_id: apex for apex in apices}
    trunk_scores: dict[str, dict[str, float]] = {}
    trunk_states: dict[str, dict[str, V4TrunkApexAssociationState]] = {}
    trunk_available: dict[str, dict[str, bool]] = {}
    trunk_lookup: dict[str, dict[str, V4AssociationLookupKind]] = {}
    for association in associations:
        trunk_id = association.trunk_candidate_id
        apex_id = association.apex_candidate_id
        trunk_scores.setdefault(trunk_id, {})[apex_id] = association.association_score
        trunk_states.setdefault(trunk_id, {})[apex_id] = association.association_state
        trunk_available.setdefault(trunk_id, {})[apex_id] = (
            association.association_score_available
        )
        trunk_lookup.setdefault(trunk_id, {})[apex_id] = association.lookup_kind
    trunks = sorted(trunk_by_id)

    def _total_lookup(
        trunk_id: str, apex_id: str
    ) -> tuple[V4AssociationLookupKind, float, V4TrunkApexAssociationState | None, bool]:
        """Total typed lookup independent of sparse materialization.

        ``OBSERVED_OUT_OF_RADIUS`` is derived from the frozen trunk-apex pair
        geometry whenever a pair is not materialized, so the total semantics
        never depend on whether a sparse dict key happens to exist.
        """
        trunk = trunk_by_id.get(trunk_id)
        apex = apex_by_id.get(apex_id)
        if trunk is None or apex is None:
            return V4AssociationLookupKind.NOT_AVAILABLE, 0.0, None, False
        d_xy = _trunk_apex_xy_distance(trunk, apex)
        if d_xy > scales.lr:
            return V4AssociationLookupKind.OBSERVED_OUT_OF_RADIUS, 0.0, None, False
        if trunk_id not in trunk_available or apex_id not in trunk_available[trunk_id]:
            return V4AssociationLookupKind.NOT_AVAILABLE, 0.0, None, False
        lookup = trunk_lookup[trunk_id][apex_id]
        if lookup is V4AssociationLookupKind.OBSERVED_OUT_OF_RADIUS:
            return V4AssociationLookupKind.OBSERVED_OUT_OF_RADIUS, 0.0, None, False
        if not trunk_available[trunk_id][apex_id]:
            return (
                V4AssociationLookupKind.NOT_AVAILABLE,
                0.0,
                trunk_states[trunk_id][apex_id],
                False,
            )
        return (
            V4AssociationLookupKind.EVALUATED,
            trunk_scores[trunk_id][apex_id],
            trunk_states[trunk_id][apex_id],
            True,
        )

    def strong_for(trunk_id: str, apex_id: str) -> float | None:
        kind, score, state, _available = _total_lookup(trunk_id, apex_id)
        if kind is V4AssociationLookupKind.OBSERVED_OUT_OF_RADIUS:
            return 0.0
        if kind is V4AssociationLookupKind.NOT_AVAILABLE:
            return None
        return association_strong01(score, state)

    def exclusive_for(trunk_id: str, apex_id: str) -> float | None:
        kind, score, state, _available = _total_lookup(trunk_id, apex_id)
        if kind is V4AssociationLookupKind.OBSERVED_OUT_OF_RADIUS:
            return 1.0
        if kind is V4AssociationLookupKind.NOT_AVAILABLE:
            return None
        return association_exclusive01(score, state)

    for apex_a, apex_b in pairs:
        key_pair = tuple(sorted((_hypothesis_key(apex_a), _hypothesis_key(apex_b))))
        shared = 0.0
        distinct = 0.0
        for trunk_id in trunks:
            strong_a = strong_for(trunk_id, apex_a.candidate_id)
            strong_b = strong_for(trunk_id, apex_b.candidate_id)
            if strong_a is not None and strong_b is not None:
                shared = max(shared, min(strong_a, strong_b))
        for trunk_a in trunks:
            for trunk_b in trunks:
                if trunk_a == trunk_b:
                    continue
                values = (
                    strong_for(trunk_a, apex_a.candidate_id),
                    strong_for(trunk_b, apex_b.candidate_id),
                    exclusive_for(trunk_a, apex_b.candidate_id),
                    exclusive_for(trunk_b, apex_a.candidate_id),
                )
                if any(value is None for value in values):
                    continue
                distinct = max(
                    distinct,
                    min(values[0], values[1], values[2], values[3]),
                )
        result[key_pair] = (shared, distinct)
    return result


def _build_hypothesis_records(
    apices: list[V4ApexCandidateRecord],
    states: dict[str, V4TreeHypothesisState],
) -> tuple[V4TreeHypothesisRecord, ...]:
    records: list[V4TreeHypothesisRecord] = []
    for apex in apices:
        key = _hypothesis_key(apex)
        records.append(
            V4TreeHypothesisRecord(
                hypothesis_id=key,
                primary_apex_candidate_id=apex.candidate_id,
                secondary_apex_candidate_ids=(),
                trunk_candidate_ids=(),
                fragment_refs=(),
                profile_refs=(f"profile-{apex.candidate_id}",),
                provisional_group_id=None,
                state=states.get(key, V4TreeHypothesisState.ACTIVE),
                quality_flags=0,
                reason_flags=0,
            )
        )
    return tuple(records)


def _build_tree_records(
    population: _NormalPopulation,
    grid: _Grid,
    trees: list[tuple[int, list[V4ApexCandidateRecord], V4TreeHypothesisState]],
    domains: dict[str, _HypothesisDomain],
    claims_by_cell: dict[int, list[_Claim]],
    slices: list[V4TreeSliceSummary],
    fragments: list[_Fragment],
    associations: list[V4TrunkApexAssociationEvidence],
    accepted_trunk_records: list[V4TrunkCandidateRecord],
    positions_by_tree: dict[int, list[int]],
    confidence_by_point: np.ndarray,
    scales: V4AlgorithmScales,
) -> list[V4TreeInstanceRecord]:
    accepted_trunk_by_id = {
        record.candidate_id: record for record in accepted_trunk_records
    }
    records: list[V4TreeInstanceRecord] = []
    for tree_id, (_component_id, members, _state) in enumerate(trees):
        member_positions = positions_by_tree.get(tree_id, [])
        member_original_indexes = [
            int(population.original_index[position]) for position in member_positions
        ]
        member_digest = _member_digest(member_original_indexes)
        primary_apex = sorted(
            members,
            key=lambda apex: (
                -apex.hag_m,
                -(
                    apex.prominence_m
                    if apex.prominence_m is not None
                    else -1.0
                ),
                apex.supporting_original_index,
            ),
        )[0]
        secondary_apices = [
            apex.candidate_id
            for apex in sorted(
                members,
                key=lambda apex: (
                    -apex.hag_m,
                    -(
                        apex.prominence_m
                        if apex.prominence_m is not None
                        else -1.0
                    ),
                    apex.supporting_original_index,
                ),
            )
            if apex.candidate_id != primary_apex.candidate_id
        ]
        member_hypotheses = [_hypothesis_key(apex) for apex in members]
        tree_slices = [
            slice_record
            for slice_record in slices
            if slice_record.hypothesis_id in member_hypotheses
        ]
        hag_values = [float(population.hag_m[pos]) for pos in member_positions]
        if hag_values:
            hag_min = min(hag_values)
            hag_max = max(hag_values)
            robust_hag = _quantile(sorted(hag_values), 0.9)
        else:
            hag_min = primary_apex.hag_m
            hag_max = primary_apex.hag_m
            robust_hag = primary_apex.hag_m
        xs = [float(population.xyz_local_m[pos, 0]) for pos in member_positions]
        ys = [float(population.xyz_local_m[pos, 1]) for pos in member_positions]
        zs = [float(population.xyz_local_m[pos, 2]) for pos in member_positions]
        if xs:
            xy_centroid = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
            bounds_min = (min(xs), min(ys), min(zs))
            bounds_max = (max(xs), max(ys), max(zs))
            crown_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        else:
            xy_centroid = primary_apex.xyz_local_m
            bounds_min = primary_apex.xyz_local_m
            bounds_max = primary_apex.xyz_local_m
            crown_area = 0.0
        equivalent_radius = math.sqrt(max(0.0, crown_area) / math.pi)
        accepted_trunk_ids = [
            association.trunk_candidate_id
            for association in associations
            if association.apex_candidate_id in {apex.candidate_id for apex in members}
            and association.lookup_kind is V4AssociationLookupKind.EVALUATED
            and association.association_state is V4TrunkApexAssociationState.STRONG_ASSOCIATION
        ]
        accepted_trunk_ids = sorted(set(accepted_trunk_ids))
        primary_trunk = None
        if accepted_trunk_ids:
            primary_trunk = accepted_trunk_ids[0]
        geometry = V4TreeGeometrySummary(
            member_count=len(member_positions),
            hag_min_m=hag_min,
            hag_max_m=hag_max,
            robust_hag_m=robust_hag,
            vertical_extent_m=hag_max - hag_min,
            xy_centroid=xy_centroid,
            xy_bounds_min=bounds_min,
            xy_bounds_max=bounds_max,
            crown_area_m2=crown_area,
            equivalent_radius_m=equivalent_radius,
            responsibility_area_m2=sum(
                len(domains[hyp].cell_ids) * scales.lxy * scales.lxy
                for hyp in member_hypotheses
                if hyp in domains
            ),
            slice_count=len(tree_slices),
            valid_slice_count=len(tree_slices),
            lowest_supported_slice_id=(
                tree_slices[0].slice_id if tree_slices else None
            ),
            highest_supported_slice_id=(
                tree_slices[-1].slice_id if tree_slices else None
            ),
            trunk_evidence_state=(
                V4TrunkEvidenceAvailability.OBSERVED
                if accepted_trunk_ids
                else V4TrunkEvidenceAvailability.NOT_AVAILABLE
            ),
            accepted_trunk_count=len(accepted_trunk_ids),
            primary_trunk_representative_original_index=None,
            multi_stem_flag=len(accepted_trunk_ids) > 1,
            geometry_quality_flags=0,
        )
        if primary_trunk is not None:
            trunk_record = accepted_trunk_by_id[primary_trunk]
            geometry = _with_trunk_representative(geometry, trunk_record)
        crown = V4TreeCrownSummary(
            crown_area_m2=crown_area,
            equivalent_radius_m=equivalent_radius,
            responsibility_area_m2=geometry.responsibility_area_m2,
            boundary_vertex_count=0,
            boundary_digest=_member_digest(()),
        )
        member_confidences = [
            float(confidence_by_point[position]) for position in member_positions
        ]
        confidence = float(np.mean(member_confidences)) if member_confidences else 0.0
        records.append(
            V4TreeInstanceRecord(
                tree_id=tree_id,
                primary_apex_candidate_id=primary_apex.candidate_id,
                secondary_apex_candidate_ids=tuple(secondary_apices),
                primary_trunk_candidate_id=primary_trunk,
                secondary_trunk_candidate_ids=tuple(
                    sorted(set(accepted_trunk_ids) - {primary_trunk})
                )
                if primary_trunk is not None
                else tuple(sorted(set(accepted_trunk_ids))),
                trunk_apex_association_ids=tuple(
                    sorted(
                        association.association_id
                        for association in associations
                        if association.apex_candidate_id
                        in {apex.candidate_id for apex in members}
                        and association.lookup_kind is V4AssociationLookupKind.EVALUATED
                    )
                ),
                member_count=len(member_positions),
                member_original_index_digest=member_digest,
                crown_summary=crown,
                geometry_summary=geometry,
                slice_ids=tuple(sorted(slice_record.slice_id for slice_record in tree_slices)),
                relation_ids=(),
                confidence=confidence,
                quality_flags=0,
                reason_flags=0,
                future_visual_evidence_reference=None,
            )
        )
    return records


def _with_trunk_representative(
    geometry: V4TreeGeometrySummary, trunk: V4TrunkCandidateRecord
) -> V4TreeGeometrySummary:
    clone = V4TreeGeometrySummary.__new__(V4TreeGeometrySummary)
    for name in V4TreeGeometrySummary.__dataclass_fields__:
        value = getattr(geometry, name)
        if name == "primary_trunk_representative_original_index":
            value = trunk.representative_original_index
        object.__setattr__(clone, name, value)
    return clone


def _remap_tree_ids(
    tree_id_by_point: np.ndarray, population: _NormalPopulation
) -> np.ndarray:
    """Re-key per-tree indices into contiguous deterministic component IDs."""

    mapping: dict[int, int] = {}
    next_id = 0
    for tree_id in sorted(set(int(value) for value in tree_id_by_point if value >= 0)):
        mapping[tree_id] = next_id
        next_id += 1
    result = np.full(tree_id_by_point.shape, -1, dtype="<i4")
    for position in range(tree_id_by_point.shape[0]):
        value = int(tree_id_by_point[position])
        if value >= 0:
            result[position] = mapping[value]
    result.setflags(write=False)
    return result


def _group_assigned_positions(
    population: _NormalPopulation,
    tree_id_by_point: np.ndarray,
    state_by_point: np.ndarray,
) -> dict[int, list[int]]:
    """Build ``tree_id -> member positions`` once in O(N).

    Positions are appended in ascending point position order, which preserves
    the exact deterministic ordering of the previous per-tree full-N scans.
    """

    count = int(population.original_index.shape[0])
    positions_by_tree: dict[int, list[int]] = {}
    assigned = int(V4TreeAssignmentState.ASSIGNED)
    for position in range(count):
        tree_id = int(tree_id_by_point[position])
        if tree_id >= 0 and int(state_by_point[position]) == assigned:
            positions_by_tree.setdefault(tree_id, []).append(position)
    return positions_by_tree


def _order_final_trees(
    formal_trees: list[tuple[int, list[V4ApexCandidateRecord], V4TreeHypothesisState]],
    population: _NormalPopulation,
    positions_by_tree: dict[int, list[int]],
) -> tuple[list[tuple[int, list[V4ApexCandidateRecord], V4TreeHypothesisState]], dict[int, int]]:
    """Order formal trees by the frozen Tree ID key and return the re-key map."""

    def primary_apex(members: list[V4ApexCandidateRecord]) -> V4ApexCandidateRecord:
        return sorted(
            members,
            key=lambda apex: (
                -apex.hag_m,
                -(apex.prominence_m if apex.prominence_m is not None else -1.0),
                apex.supporting_original_index,
            ),
        )[0]

    keys: dict[int, tuple[int, int]] = {}
    for tree_index, (_component_id, members, _state) in enumerate(formal_trees):
        apex = primary_apex(members)
        member_positions = positions_by_tree.get(tree_index, [])
        member_indexes = [
            int(population.original_index[position]) for position in member_positions
        ]
        minimum_member = min(member_indexes) if member_indexes else apex.supporting_original_index
        keys[tree_index] = (apex.supporting_original_index, minimum_member)
    order = sorted(range(len(formal_trees)), key=lambda index: keys[index])
    remap = {old_index: new_index for new_index, old_index in enumerate(order)}
    return [formal_trees[index] for index in order], remap


def _ragged_members(
    tree_records: list[V4TreeInstanceRecord],
    population: _NormalPopulation,
    positions_by_tree: dict[int, list[int]],
) -> tuple[list[int], list[int]]:
    ordered: list[int] = []
    offsets: list[int] = [0]
    for tree_id in range(len(tree_records)):
        member_positions = positions_by_tree.get(tree_id, [])
        # Ascending positions yield ascending original indexes (the population
        # is sorted by original_index); the explicit sort is preserved for the
        # frozen ordering guarantee.
        indexes = [
            int(population.original_index[position]) for position in member_positions
        ]
        indexes.sort()
        ordered.extend(indexes)
        offsets.append(len(ordered))
    return ordered, offsets


def _build_audit(
    population: _NormalPopulation,
    apex_records: list[V4ApexCandidateRecord],
    trunk_records: list[V4TrunkCandidateRecord],
    associations: list[V4TrunkApexAssociationEvidence],
    structural_paths: list[V4StructuralPathEvidence],
    profiles: list[V4RotatingProfileEvidence],
    accepted_apices: list[V4ApexCandidateRecord],
    slices: list[V4TreeSliceSummary],
    early_relations: list[V4TreeRelationEvidence],
    final_relations: list[V4TreeRelationEvidence],
    tree_records: list[V4TreeInstanceRecord],
    states: dict[str, V4TreeHypothesisState],
) -> V4TreeSegmentationAudit:
    accepted_trunks = sum(
        record.candidate_state is V4TrunkCandidateState.ACCEPTED
        for record in trunk_records
    )
    weak_trunks = sum(
        record.candidate_state is V4TrunkCandidateState.WEAK for record in trunk_records
    )
    suppressed_trunks = sum(
        record.candidate_state is V4TrunkCandidateState.SUPPRESSED
        for record in trunk_records
    )
    accepted_trunk_ids = {
        record.candidate_id
        for record in trunk_records
        if record.candidate_state is V4TrunkCandidateState.ACCEPTED
    }
    matched_trunk_ids = {
        association.trunk_candidate_id for association in associations
    }
    strong = sum(
        association.association_state is V4TrunkApexAssociationState.STRONG_ASSOCIATION
        and association.association_score_available
        for association in associations
    )
    review_assoc = sum(
        association.association_state is V4TrunkApexAssociationState.REVIEW_ASSOCIATION
        for association in associations
    )
    unsupported = sum(
        association.association_state is V4TrunkApexAssociationState.UNSUPPORTED_ASSOCIATION
        for association in associations
    )
    path_observed = sum(
        path.availability is V4StructuralPathAvailability.OBSERVED
        for path in structural_paths
    )
    path_na = sum(
        path.availability is V4StructuralPathAvailability.NOT_AVAILABLE
        for path in structural_paths
    )
    profile_confirmed = sum(
        profile.profile_state is V4RotatingProfileState.PROFILE_CONFIRMED_VALLEY
        for profile in profiles
    )
    profile_weak = sum(
        profile.profile_state is V4RotatingProfileState.PROFILE_WEAK_VALLEY
        for profile in profiles
    )
    profile_no = sum(
        profile.profile_state is V4RotatingProfileState.PROFILE_NO_VALLEY
        for profile in profiles
    )
    profile_na = sum(
        profile.profile_state is V4RotatingProfileState.PROFILE_NOT_AVAILABLE
        for profile in profiles
    )
    warnings = []
    limitations = []
    if not trunk_records:
        warnings.append("no trunk candidates observed")
    if not accepted_apices:
        limitations.append("no accepted apex candidates")
    if any(
        state is V4TreeHypothesisState.REVIEW for state in states.values()
    ):
        warnings.append("union constraint conflict produced REVIEW hypotheses")
    return V4TreeSegmentationAudit(
        schema_version=V4_TREE_SEGMENTATION_SCHEMA_VERSION,
        total_points=int(population.original_index.shape[0]),
        apex_candidate_count=len(apex_records),
        accepted_trunk_count=accepted_trunks,
        weak_trunk_count=weak_trunks,
        suppressed_trunk_count=suppressed_trunks,
        unmatched_trunk_count=len(accepted_trunk_ids - matched_trunk_ids),
        strong_association_count=strong,
        review_association_count=review_assoc,
        unsupported_association_count=unsupported,
        path_observed_count=path_observed,
        path_not_available_count=path_na,
        profile_confirmed_count=profile_confirmed,
        profile_weak_count=profile_weak,
        profile_no_valley_count=profile_no,
        profile_not_available_count=profile_na,
        hypothesis_count=len(accepted_apices),
        slice_count=len(slices),
        relation_provisional_count=len(early_relations),
        relation_final_count=len(final_relations),
        tree_count=len(tree_records),
        phase_resource_audit="ok",
        determinism_evidence="canonical-original-index",
        warnings=tuple(warnings),
        limitations=tuple(limitations),
    )


def _all_normal_ready_guard(population: _NormalPopulation) -> None:
    if not np.all(population.segmentation_hag_readiness == 1):
        raise V4TreeSegmentationError(
            code=V4TreeSegmentationErrorCode.INPUT_INDEX_INVALID,
            field_name="segmentation_hag_readiness",
        )


def _final_tree_ids_by_component(
    trees: list[tuple[int, list[V4ApexCandidateRecord], V4TreeHypothesisState]],
) -> dict[int, int]:
    return {component_id: index for index, (component_id, _m, _s) in enumerate(trees)}


__all__ = [
    "segment_v4_trees",
    "c01",
    "near",
    "similar",
    "signed",
    "clamp11",
    "safe_ratio",
    "derive_algorithm_scales",
    "V4AlgorithmScales",
    "responsibility_r0",
    "responsibility_pressure",
    "responsibility_score",
    "merge_support_score",
    "separate_support_score",
    "relation_decision",
    "trunk_candidate_score",
    "trunk_candidate_state",
    "trunk_apex_score",
    "trunk_apex_state",
    "association_strong01",
    "association_exclusive01",
    "assignment_base",
    "assignment_support_score",
    "assignment_competitor_channel",
    "assignment_confidence",
    "assignment_decision",
    "profile_confirmed_stop",
]
