"""Deterministic source-point mapping into formal V3 tree-segmentation results."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

import numpy as np
from core.contracts.features import PointFeatureSet
from core.contracts.ground import GroundQualityFlag, HeightAboveGroundSet
from core.contracts.tree_inputs import (
    TreeSegmentationInputClusterDisposition,
    TreeSegmentationInputDisposition,
    TreeSegmentationInputIdentity,
    TreeSegmentationInputSet,
)
from core.contracts.vegetation_instances import (
    VegetationClusterStatus,
    VegetationInstancePointSet,
)
from core.contracts.vegetation_trees import (
    TREE_SEGMENTATION_ALGORITHM_VERSION,
    TREE_SEGMENTATION_SCHEMA_VERSION,
    TreeInstanceRecord,
    TreeSegmentationConfig,
    TreeSegmentationQualityFlag,
    TreeSegmentationResult,
    TreeSegmentationStatistics,
    TreeSegmentationStatus,
)

from algorithms.vegetation.canopy_height_model import (
    CanopyHeightModelConfig,
    CanopyHeightModelResult,
    CanopySupportFlag,
)
from algorithms.vegetation.canopy_segmentation import (
    CanopySegmentationConfig,
    CanopySegmentationResult,
)
from algorithms.vegetation.treetop_candidates import (
    TreetopCandidateResult,
    TreetopDetectionConfig,
)

_FIXED_METADATA_BYTES = 1_048_576
_INT32_MAX = int(np.iinfo(np.int32).max)
_INT64_MAX = int(np.iinfo(np.int64).max)
_INTP_MAX = int(np.iinfo(np.intp).max)
_ALLOWED_SUPPORT_BITS = int(
    CanopySupportFlag.CONFIRMED | CanopySupportFlag.REVIEW_REQUIRED | CanopySupportFlag.PROVISIONAL
)
_LIMITATIONS = (
    "segments are provisional canopy-region hypotheses and are not confirmed botanical trees",
    "formal tree IDs are unavailable and remain -1",
    "tree geometry and crown summaries are not constructed",
    "excluded and eligible-unassigned source points share segment_id -1",
    "point-level ambiguity is represented only by aggregate audit metadata",
    "tree-to-wire distance and risk outputs are not computed",
)
_STATUS_DISPOSITIONS = {
    VegetationClusterStatus.CONFIRMED: (
        TreeSegmentationInputDisposition.ELIGIBLE,
        "confirmed_distance_ready",
    ),
    VegetationClusterStatus.REVIEW_REQUIRED: (
        TreeSegmentationInputDisposition.REVIEW_ONLY,
        "review_required",
    ),
    VegetationClusterStatus.PROVISIONAL: (
        TreeSegmentationInputDisposition.REVIEW_ONLY,
        "provisional",
    ),
    VegetationClusterStatus.REJECTED_AS_INSTANCE: (
        TreeSegmentationInputDisposition.EXCLUDED,
        "rejected_as_instance",
    ),
}


def _positive_integer(value: Any, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")
    return normalized


@dataclass(frozen=True)
class TreeResultConstructionConfig:
    """Resource-only controls for deterministic formal-result construction."""

    maximum_source_points: int = 10_000_000
    maximum_grid_cells: int = 10_000_000
    maximum_candidates: int = 100_000
    maximum_source_clusters: int = 1_000_000
    maximum_grouped_member_rows: int = 10_000_000
    maximum_additional_bytes: int = 2_147_483_648
    chunk_point_count: int = 250_000

    def __post_init__(self) -> None:
        for field in fields(self):
            maximum = None
            if field.name == "maximum_candidates":
                maximum = 2_147_483_646
            elif field.name == "maximum_source_clusters":
                maximum = 2_147_483_647
            object.__setattr__(
                self,
                field.name,
                _positive_integer(getattr(self, field.name), field.name, maximum),
            )


class TreeResultConstructionErrorCode(str, Enum):
    RESOURCE_LIMIT = "resource_limit"
    ALLOCATION_FAILED = "allocation_failed"
    INPUT_BUNDLE_MISMATCH = "input_bundle_mismatch"
    POINT_MAPPING_MISMATCH = "point_mapping_mismatch"
    INTERNAL_INVARIANT = "internal_invariant"


class TreeResultConstructionError(RuntimeError):
    """A compact deterministic V3B4B failure."""

    def __init__(
        self,
        code: TreeResultConstructionErrorCode,
        message: str,
        *,
        source_point_count: int | None = None,
        eligible_point_count: int | None = None,
        assigned_point_count: int | None = None,
        candidate_count: int | None = None,
        source_cluster_count: int | None = None,
        grid_cell_count: int | None = None,
        chunk_start: int | None = None,
        chunk_end: int | None = None,
        estimated_additional_bytes: int | None = None,
        mismatched_cell_count: int | None = None,
    ) -> None:
        if type(code) is not TreeResultConstructionErrorCode:
            raise ValueError("code must be a TreeResultConstructionErrorCode")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        scalar_values = (
            ("source_point_count", source_point_count),
            ("eligible_point_count", eligible_point_count),
            ("assigned_point_count", assigned_point_count),
            ("candidate_count", candidate_count),
            ("source_cluster_count", source_cluster_count),
            ("grid_cell_count", grid_cell_count),
            ("chunk_start", chunk_start),
            ("chunk_end", chunk_end),
            ("estimated_additional_bytes", estimated_additional_bytes),
            ("mismatched_cell_count", mismatched_cell_count),
        )
        for name, value in scalar_values:
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be an exact non-negative Python int or None")
            setattr(self, name, value)
        if chunk_start is not None and chunk_end is not None and chunk_end < chunk_start:
            raise ValueError("chunk_end must be greater than or equal to chunk_start")
        self.code = code
        super().__init__(message)


def _array_metadata(
    value: Any,
    dtype: Any,
    shape: tuple[int | None, ...],
    name: str,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise ValueError(f"{name} must be an exact numpy ndarray")
    if value.dtype != np.dtype(dtype) or value.ndim != len(shape):
        raise ValueError(f"{name} has invalid dtype or rank")
    if any(
        expected is not None and expected != actual
        for expected, actual in zip(shape, value.shape, strict=True)
    ):
        raise ValueError(f"{name} has invalid shape")
    if not value.flags.owndata or not value.flags.c_contiguous or value.flags.writeable:
        raise ValueError(f"{name} has invalid ownership, contiguity, or mutability")
    return value


def _python_nonnegative(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative Python integer")
    return value


def _validate_scalar_shells(
    tree_input: TreeSegmentationInputSet,
    canopy_model: CanopyHeightModelResult,
    treetop_candidates: TreetopCandidateResult,
    canopy_segmentation: CanopySegmentationResult,
) -> None:
    if type(tree_input.identity) is not TreeSegmentationInputIdentity:
        raise ValueError("tree_input.identity has invalid type")
    if type(tree_input.vegetation_points) is not VegetationInstancePointSet:
        raise ValueError("tree_input.vegetation_points has invalid type")
    if type(tree_input.geometry_features) is not PointFeatureSet:
        raise ValueError("tree_input.geometry_features has invalid type")
    if type(tree_input.heights_above_ground) is not HeightAboveGroundSet:
        raise ValueError("tree_input.heights_above_ground has invalid type")
    if type(canopy_model.source_identity) is not TreeSegmentationInputIdentity:
        raise ValueError("canopy_model.source_identity has invalid type")
    if type(canopy_model.config) is not CanopyHeightModelConfig:
        raise ValueError("canopy_model.config has invalid type")
    if type(treetop_candidates.source_identity) is not TreeSegmentationInputIdentity:
        raise ValueError("treetop_candidates.source_identity has invalid type")
    if type(treetop_candidates.source_chm_config) is not CanopyHeightModelConfig:
        raise ValueError("treetop_candidates.source_chm_config has invalid type")
    if type(treetop_candidates.config) is not TreetopDetectionConfig:
        raise ValueError("treetop_candidates.config has invalid type")
    if type(canopy_segmentation.source_identity) is not TreeSegmentationInputIdentity:
        raise ValueError("canopy_segmentation.source_identity has invalid type")
    if type(canopy_segmentation.source_chm_config) is not CanopyHeightModelConfig:
        raise ValueError("canopy_segmentation.source_chm_config has invalid type")
    if type(canopy_segmentation.source_treetop_config) is not TreetopDetectionConfig:
        raise ValueError("canopy_segmentation.source_treetop_config has invalid type")
    if type(canopy_segmentation.config) is not CanopySegmentationConfig:
        raise ValueError("canopy_segmentation.config has invalid type")

    for owner, names in (
        (canopy_model, ("included_point_count", "excluded_point_count")),
        (
            treetop_candidates,
            (
                "local_maximum_plateau_count",
                "evidence_rejected_count",
                "suppressed_candidate_count",
            ),
        ),
        (
            canopy_segmentation,
            (
                "seeded_component_count",
                "unseeded_component_count",
                "assigned_cell_count",
                "unassigned_domain_cell_count",
                "ambiguous_cell_count",
            ),
        ),
    ):
        for name in names:
            _python_nonnegative(getattr(owner, name), name)

    for name in (
        "eligible_cluster_ids",
        "review_only_cluster_ids",
        "excluded_cluster_ids",
        "cluster_dispositions",
    ):
        if type(getattr(tree_input, name)) is not tuple:
            raise ValueError(f"{name} must be a tuple")


def _validate_structural_arrays(
    tree_input: TreeSegmentationInputSet,
    canopy_model: CanopyHeightModelResult,
    treetop_candidates: TreetopCandidateResult,
    canopy_segmentation: CanopySegmentationResult,
) -> tuple[int, int, int, int, int, int]:
    points = tree_input.vegetation_points
    _array_metadata(
        points.original_index,
        np.int64,
        (None,),
        "vegetation_points.original_index",
    )
    p = len(points.original_index)
    for identity, name in (
        (tree_input.identity, "tree_input.identity"),
        (canopy_model.source_identity, "canopy_model.source_identity"),
        (treetop_candidates.source_identity, "treetop_candidates.source_identity"),
        (canopy_segmentation.source_identity, "canopy_segmentation.source_identity"),
    ):
        if identity.point_count != p:
            raise ValueError(f"{name}.point_count must equal the actual source point count")
    point_specs = {
        "original_index": (np.int64, (p,)),
        "sample_index": (np.int64, (p,)),
        "xyz_local": (np.float32, (p, 3)),
        "wire_id_by_point": (np.int32, (p,)),
        "vegetation_rule_score": (np.float32, (p,)),
        "vegetation_confidence": (np.float32, (p,)),
        "automatic_vegetation_level": (np.uint8, (p,)),
        "vegetation_level": (np.uint8, (p,)),
        "hag_state": (np.uint8, (p,)),
        "local_scale_raw_m": (np.float32, (p,)),
        "local_scale_m": (np.float32, (p,)),
        "local_scale_state": (np.uint8, (p,)),
        "cluster_id_by_point": (np.int32, (p,)),
        "vegetation_id_by_point": (np.int32, (p,)),
        "cluster_status_by_point": (np.uint8, (p,)),
        "distance_ready_by_point": (np.bool_, (p,)),
        "assignment_confidence": (np.float32, (p,)),
        "assignment_reason_flags": (np.uint16, (p,)),
        "assignment_quality_flags": (np.uint16, (p,)),
        "manual_provenance_flags": (np.uint16, (p,)),
    }
    for name, (dtype, shape) in point_specs.items():
        _array_metadata(getattr(points, name), dtype, shape, f"vegetation_points.{name}")

    geometry = tree_input.geometry_features
    geometry_specs = {
        "xyz_local": (np.float32, (p, 3)),
        "origin_xyz": (np.float64, (3,)),
        "original_index": (np.int64, (p,)),
        "sample_index": (np.int64, (p,)),
        "linearity": (np.float32, (p,)),
        "planarity": (np.float32, (p,)),
        "scattering": (np.float32, (p,)),
        "verticality": (np.float32, (p,)),
        "local_density": (np.float32, (p,)),
        "height_m": (np.float32, (p,)),
        "neighbor_count": (np.uint16, (p,)),
        "eigenvalues": (np.float32, (p, 3)),
        "principal_direction": (np.float32, (p, 3)),
        "normal": (np.float32, (p, 3)),
        "quality_flags": (np.uint16, (p,)),
    }
    for name, (dtype, shape) in geometry_specs.items():
        _array_metadata(getattr(geometry, name), dtype, shape, f"geometry_features.{name}")

    heights = tree_input.heights_above_ground
    height_specs = {
        "original_index": (np.int64, (p,)),
        "sample_index": (np.int64, (p,)),
        "ground_z_m": (np.float32, (p,)),
        "raw_height_agl_m": (np.float32, (p,)),
        "height_agl_m": (np.float32, (p,)),
        "ground_cell_id": (np.int32, (p,)),
        "ground_distance_m": (np.float32, (p,)),
        "quality_flags": (np.uint16, (p,)),
    }
    for name, (dtype, shape) in height_specs.items():
        _array_metadata(getattr(heights, name), dtype, shape, f"heights_above_ground.{name}")

    cluster_id_shell = _array_metadata(
        tree_input.cluster_id,
        np.int32,
        (None,),
        "cluster_id",
    )
    c = len(cluster_id_shell)
    _array_metadata(tree_input.cluster_id, np.int32, (c,), "cluster_id")
    _array_metadata(tree_input.cluster_status, np.uint8, (c,), "cluster_status")
    _array_metadata(
        tree_input.cluster_vegetation_id,
        np.int32,
        (c,),
        "cluster_vegetation_id",
    )
    _array_metadata(
        tree_input.cluster_member_offsets,
        np.int64,
        (c + 1,),
        "cluster_member_offsets",
    )
    member_shell = _array_metadata(
        tree_input.ordered_cluster_member_row_index,
        np.int64,
        (None,),
        "ordered_cluster_member_row_index",
    )
    m = len(member_shell)
    _array_metadata(
        tree_input.ordered_cluster_member_row_index,
        np.int64,
        (m,),
        "ordered_cluster_member_row_index",
    )
    _array_metadata(
        tree_input.ground_usable_by_point,
        np.bool_,
        (p,),
        "ground_usable_by_point",
    )
    _array_metadata(
        tree_input.usable_ground_member_count_by_cluster,
        np.int64,
        (c,),
        "usable_ground_member_count_by_cluster",
    )

    height_shell = _array_metadata(
        canopy_model.canopy_height_m,
        np.float32,
        (None, None),
        "canopy_height_m",
    )
    shape = height_shell.shape
    n = int(shape[0]) * int(shape[1])
    for name, dtype in (
        ("canopy_height_m", np.float32),
        ("occupied_mask", np.bool_),
        ("point_count_by_cell", np.int32),
        ("support_flags_by_cell", np.uint8),
        ("peak_support_flags_by_cell", np.uint8),
    ):
        _array_metadata(getattr(canopy_model, name), dtype, shape, name)

    candidate_id_shell = _array_metadata(
        treetop_candidates.candidate_id,
        np.int32,
        (None,),
        "candidate_id",
    )
    k = len(candidate_id_shell)
    candidate_specs = {
        "candidate_id": (np.int32, (k,)),
        "row_index": (np.int64, (k,)),
        "column_index": (np.int64, (k,)),
        "local_xy": (np.float64, (k, 2)),
        "peak_height_m": (np.float32, (k,)),
        "prominence_m": (np.float32, (k,)),
        "prominence_available": (np.bool_, (k,)),
        "plateau_cell_count": (np.int32, (k,)),
        "source_cell_point_count": (np.int32, (k,)),
        "source_support_flags": (np.uint8, (k,)),
        "source_peak_support_flags": (np.uint8, (k,)),
        "neighbour_occupied_cell_count": (np.int32, (k,)),
        "neighbourhood_occupied_fraction": (np.float32, (k,)),
        "neighbourhood_support_flags": (np.uint8, (k,)),
        "neighbourhood_support_fraction": (np.float32, (k, 3)),
        "edge_limited": (np.bool_, (k,)),
    }
    for name, (dtype, expected_shape) in candidate_specs.items():
        _array_metadata(getattr(treetop_candidates, name), dtype, expected_shape, name)

    label_shell = _array_metadata(
        canopy_segmentation.region_label_by_cell,
        np.int32,
        (None, None),
        "region_label_by_cell",
    )
    segmentation_shape = label_shell.shape
    for name, dtype in (
        ("region_label_by_cell", np.int32),
        ("segmentation_domain_mask", np.bool_),
        ("ambiguity_mask", np.bool_),
    ):
        _array_metadata(getattr(canopy_segmentation, name), dtype, segmentation_shape, name)
    segmentation_specs = {
        "marker_flat_cell_index_by_candidate": (np.int64, (k,)),
        "region_cell_count_by_candidate": (np.int64, (k,)),
        "ambiguous_cell_count_by_candidate": (np.int64, (k,)),
        "region_touches_grid_edge_by_candidate": (np.bool_, (k,)),
    }
    for name, (dtype, expected_shape) in segmentation_specs.items():
        _array_metadata(getattr(canopy_segmentation, name), dtype, expected_shape, name)

    allowed_count = len(tree_input.eligible_cluster_ids) + len(tree_input.review_only_cluster_ids)
    return p, n, k, c, m, allowed_count


def _record_reserve(k: int, incidence_count: int) -> int:
    return 2_048 * k + 96 * incidence_count


def _phase_peaks(
    p: int,
    n: int,
    k: int,
    c: int,
    allowed_count: int,
    b: int,
    assigned: int,
    incidence: int,
) -> dict[str, int]:
    f = _FIXED_METADATA_BYTES
    q = min(65_536, max(p, assigned, k + 1))
    r = _record_reserve(k, incidence)
    return {
        "membership": f
        + max(
            96 * c,
            p + 9 * c + 105 * b,
            p + 9 * c + 4 * allowed_count + 40 * b,
        ),
        "chm_audit_pass_1": f + 4 * allowed_count + 9 * c + max(13 * n + 64 * b, 14 * n),
        "chm_audit_pass_2": f + 4 * allowed_count + 9 * c + 15 * n + 4 * p + 16 * k + 64 * b,
        "point_grouping": f + 9 * c + 9 * p + 8 * assigned + 32 * k + 8 + 138 * b,
        "incidence_pass_1": f + 9 * c + 9 * p + 8 * assigned + 32 * k + 8 + 135 * b,
        "incidence_pass_2": f
        + 9 * c
        + 9 * p
        + 8 * assigned
        + 40 * k
        + 16
        + 6 * incidence
        + 143 * b,
        "record_construction": f + 9 * p + 8 * assigned + 16 * (k + 1) + 6 * incidence + r,
        "formal_constructor": f
        + 9 * p
        + 8 * assigned
        + 9 * c
        + 16 * (k + 1)
        + 6 * incidence
        + r
        + 40 * k
        + 32 * q,
    }


def _resource_error(
    message: str,
    *,
    p: int,
    n: int,
    k: int,
    c: int,
    estimated: int | None = None,
    assigned: int | None = None,
) -> TreeResultConstructionError:
    return TreeResultConstructionError(
        TreeResultConstructionErrorCode.RESOURCE_LIMIT,
        message,
        source_point_count=int(p),
        assigned_point_count=None if assigned is None else int(assigned),
        candidate_count=int(k),
        source_cluster_count=int(c),
        grid_cell_count=int(n),
        estimated_additional_bytes=None if estimated is None else int(estimated),
    )


def _preflight_counts(
    p: int,
    n: int,
    k: int,
    c: int,
    m: int,
    allowed_count: int,
    config: TreeResultConstructionConfig,
) -> tuple[int, dict[str, int]]:
    limits = (
        (p, config.maximum_source_points, "source point count exceeds maximum_source_points"),
        (n, config.maximum_grid_cells, "grid cell count exceeds maximum_grid_cells"),
        (k, config.maximum_candidates, "candidate count exceeds maximum_candidates"),
        (c, config.maximum_source_clusters, "source cluster count exceeds maximum_source_clusters"),
        (
            m,
            config.maximum_grouped_member_rows,
            "source membership row count exceeds maximum_grouped_member_rows",
        ),
    )
    for value, maximum, message in limits:
        if value > maximum:
            raise _resource_error(message, p=p, n=n, k=k, c=c)
    for value, name in (
        (p, "source point count"),
        (n, "grid cell count"),
        (k, "candidate count"),
        (c, "source cluster count"),
        (m, "source membership row count"),
        (allowed_count, "allowed cluster ID count"),
        (c + 1, "cluster offset count"),
        (k + 1, "segment offset count"),
    ):
        if value > _INT64_MAX or value > _INTP_MAX:
            raise _resource_error(f"{name} is not representable", p=p, n=n, k=k, c=c)
    if k > _INT32_MAX - 1:
        raise _resource_error("candidate labels cannot be represented as int32", p=p, n=n, k=k, c=c)
    if p and p - 1 > _INT64_MAX:
        raise _resource_error(
            "source row indexes cannot be represented as int64", p=p, n=n, k=k, c=c
        )
    b = min(p, config.chunk_point_count)
    peaks = _phase_peaks(p, n, k, c, allowed_count, b, p, p)
    phase, maximum = max(peaks.items(), key=lambda item: item[1])
    if maximum > config.maximum_additional_bytes:
        raise _resource_error(
            f"{phase} conservative peak exceeds maximum_additional_bytes",
            p=p,
            n=n,
            k=k,
            c=c,
            estimated=maximum,
        )
    return b, peaks


def _check_phase(
    phase: str,
    peaks: dict[str, int],
    config: TreeResultConstructionConfig,
    *,
    p: int,
    n: int,
    k: int,
    c: int,
    assigned: int | None = None,
) -> None:
    estimated = peaks[phase]
    if estimated > config.maximum_additional_bytes:
        raise _resource_error(
            f"{phase} peak exceeds maximum_additional_bytes",
            p=p,
            n=n,
            k=k,
            c=c,
            estimated=estimated,
            assigned=assigned,
        )


def _canonical_dataclass(value: Any, expected_type: type[Any], name: str) -> None:
    if type(value) is not expected_type:
        raise ValueError(f"{name} has invalid type")
    canonical = expected_type(**{field.name: getattr(value, field.name) for field in fields(value)})
    if value != canonical or any(
        type(getattr(value, field.name)) is not type(getattr(canonical, field.name))
        for field in fields(value)
    ):
        raise ValueError(f"{name} is not canonical")


def _validate_canonical_scalars(
    tree_input: TreeSegmentationInputSet,
    canopy_model: CanopyHeightModelResult,
    treetop_candidates: TreetopCandidateResult,
    canopy_segmentation: CanopySegmentationResult,
) -> None:
    for identity, name in (
        (tree_input.identity, "tree_input.identity"),
        (canopy_model.source_identity, "canopy_model.source_identity"),
        (treetop_candidates.source_identity, "treetop_candidates.source_identity"),
        (canopy_segmentation.source_identity, "canopy_segmentation.source_identity"),
    ):
        _canonical_dataclass(identity, TreeSegmentationInputIdentity, name)
        if type(identity.origin_xyz) is not tuple or any(
            type(value) is not float for value in identity.origin_xyz
        ):
            raise ValueError(f"{name}.origin_xyz is not canonical")
    for value, expected_type, name in (
        (canopy_model.config, CanopyHeightModelConfig, "canopy_model.config"),
        (
            treetop_candidates.source_chm_config,
            CanopyHeightModelConfig,
            "treetop_candidates.source_chm_config",
        ),
        (treetop_candidates.config, TreetopDetectionConfig, "treetop_candidates.config"),
        (
            canopy_segmentation.source_chm_config,
            CanopyHeightModelConfig,
            "canopy_segmentation.source_chm_config",
        ),
        (
            canopy_segmentation.source_treetop_config,
            TreetopDetectionConfig,
            "canopy_segmentation.source_treetop_config",
        ),
        (
            canopy_segmentation.config,
            CanopySegmentationConfig,
            "canopy_segmentation.config",
        ),
    ):
        _canonical_dataclass(value, expected_type, name)


def _bundle_error(message: str, *, p: int, n: int, k: int, c: int) -> TreeResultConstructionError:
    return TreeResultConstructionError(
        TreeResultConstructionErrorCode.INPUT_BUNDLE_MISMATCH,
        message,
        source_point_count=int(p),
        candidate_count=int(k),
        source_cluster_count=int(c),
        grid_cell_count=int(n),
    )


def _mapping_error(
    message: str,
    *,
    p: int,
    n: int,
    k: int,
    c: int,
    eligible: int | None = None,
    assigned: int | None = None,
    mismatched: int | None = None,
) -> TreeResultConstructionError:
    return TreeResultConstructionError(
        TreeResultConstructionErrorCode.POINT_MAPPING_MISMATCH,
        message,
        source_point_count=int(p),
        eligible_point_count=None if eligible is None else int(eligible),
        assigned_point_count=None if assigned is None else int(assigned),
        candidate_count=int(k),
        source_cluster_count=int(c),
        grid_cell_count=int(n),
        mismatched_cell_count=None if mismatched is None else int(mismatched),
    )


def _validate_identity_and_alignment(
    tree_input: TreeSegmentationInputSet,
    canopy_model: CanopyHeightModelResult,
    treetop_candidates: TreetopCandidateResult,
    canopy_segmentation: CanopySegmentationResult,
    chunk_size: int,
    *,
    p: int,
    n: int,
    k: int,
    c: int,
) -> None:
    identity = tree_input.identity
    if any(
        candidate != identity
        for candidate in (
            canopy_model.source_identity,
            treetop_candidates.source_identity,
            canopy_segmentation.source_identity,
        )
    ):
        raise _bundle_error("source identities do not match", p=p, n=n, k=k, c=c)
    geometry = tree_input.geometry_features
    heights = tree_input.heights_above_ground
    if (
        geometry.source_id != identity.source_id
        or heights.source_id != identity.source_id
        or geometry.manifest_source_id != identity.manifest_source_id
        or heights.manifest_source_id != identity.manifest_source_id
        or not np.array_equal(
            geometry.origin_xyz, np.asarray(identity.origin_xyz, dtype=np.float64)
        )
    ):
        raise _bundle_error("source identifiers or local origins do not match", p=p, n=n, k=k, c=c)
    points = tree_input.vegetation_points
    step = max(1, chunk_size)
    for start in range(0, p, step):
        end = min(p, start + step)
        if not np.array_equal(
            points.original_index[start:end], geometry.original_index[start:end]
        ) or not np.array_equal(
            points.original_index[start:end], heights.original_index[start:end]
        ):
            raise _bundle_error("original_index arrays are not aligned", p=p, n=n, k=k, c=c)
        if not np.array_equal(
            points.sample_index[start:end], geometry.sample_index[start:end]
        ) or not np.array_equal(points.sample_index[start:end], heights.sample_index[start:end]):
            raise _bundle_error("sample_index arrays are not aligned", p=p, n=n, k=k, c=c)
        if not np.array_equal(points.xyz_local[start:end], geometry.xyz_local[start:end]):
            raise _bundle_error("xyz_local arrays are not aligned", p=p, n=n, k=k, c=c)


def _validate_partitions_and_membership(
    tree_input: TreeSegmentationInputSet,
    chunk_size: int,
    *,
    p: int,
    n: int,
    k: int,
    c: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = tree_input.cluster_member_offsets
    member_rows = tree_input.ordered_cluster_member_row_index
    m = len(member_rows)
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != m
        or np.any(offsets < 0)
        or np.any(offsets > m)
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("cluster_member_offsets must be nonnegative, monotonic, and spanning")
    if c == 0 and (len(offsets) != 1 or int(offsets[0]) != 0):
        raise ValueError("zero clusters require offsets [0]")
    if c and np.any(offsets[1:] == offsets[:-1]):
        raise ValueError("every formal source cluster must be nonempty")

    if any(
        type(value) is not int or value < 0
        for name in (
            "eligible_cluster_ids",
            "review_only_cluster_ids",
            "excluded_cluster_ids",
        )
        for value in getattr(tree_input, name)
    ):
        raise ValueError("cluster partition IDs must be non-negative Python integers")
    if any(
        type(value) is not TreeSegmentationInputClusterDisposition
        for value in tree_input.cluster_dispositions
    ):
        raise ValueError("cluster_dispositions contains an invalid record")
    partition_rows = (
        *(
            (value, TreeSegmentationInputDisposition.ELIGIBLE)
            for value in tree_input.eligible_cluster_ids
        ),
        *(
            (value, TreeSegmentationInputDisposition.REVIEW_ONLY)
            for value in tree_input.review_only_cluster_ids
        ),
        *(
            (value, TreeSegmentationInputDisposition.EXCLUDED)
            for value in tree_input.excluded_cluster_ids
        ),
    )
    if any(
        tuple(values) != tuple(sorted(values)) or len(values) != len(set(values))
        for values in (
            tree_input.eligible_cluster_ids,
            tree_input.review_only_cluster_ids,
            tree_input.excluded_cluster_ids,
        )
    ):
        raise ValueError("cluster partitions must be strictly increasing and unique")
    if len(partition_rows) != c or len({row[0] for row in partition_rows}) != c:
        raise ValueError("cluster partitions must contain every cluster exactly once")

    try:
        cluster_order = np.argsort(tree_input.cluster_id, kind="stable")
        seen_member_row = np.zeros(p, dtype=np.bool_)
        cluster_readiness = np.empty(c, dtype=np.bool_)
    except MemoryError:
        raise
    ordered_ids = tree_input.cluster_id[cluster_order]
    if c and (np.any(ordered_ids < 0) or np.any(ordered_ids[1:] <= ordered_ids[:-1])):
        raise ValueError("cluster IDs must be unique nonnegative values")
    expected_partition = {cluster_id: disposition for cluster_id, disposition in partition_rows}
    dispositions = tuple(tree_input.cluster_dispositions)
    if tuple(record.cluster_id for record in dispositions) != tuple(
        int(value) for value in ordered_ids
    ):
        raise ValueError("cluster_dispositions must follow canonical cluster ID order")

    points = tree_input.vegetation_points
    allowed_statuses = {int(value) for value in VegetationClusterStatus}
    if np.any(~np.isin(tree_input.cluster_status, tuple(allowed_statuses))):
        raise ValueError("cluster_status contains an unsupported value")
    if np.any(tree_input.cluster_status == int(VegetationClusterStatus.UNASSIGNED)):
        raise ValueError("formal clusters cannot have UNASSIGNED status")
    step = max(1, chunk_size)
    usable_counts = np.zeros(c, dtype=np.int64)
    for position in range(c):
        cluster_id = int(tree_input.cluster_id[position])
        status = VegetationClusterStatus(int(tree_input.cluster_status[position]))
        expected_disposition, expected_reason = _STATUS_DISPOSITIONS[status]
        disposition = dispositions[int(np.searchsorted(ordered_ids, cluster_id))]
        if (
            expected_partition.get(cluster_id) is not expected_disposition
            or disposition.disposition is not expected_disposition
            or disposition.reason != expected_reason
        ):
            raise ValueError("cluster disposition, status, or reason is inconsistent")
        start = int(offsets[position])
        end = int(offsets[position + 1])
        readiness = True
        for member_start in range(start, end, step):
            member_end = min(end, member_start + step)
            rows = member_rows[member_start:member_end]
            if np.any(rows < 0) or np.any(rows >= p):
                raise ValueError("source cluster member row is out of range")
            ordered_chunk = np.sort(rows, kind="stable")
            if len(ordered_chunk) > 1 and np.any(ordered_chunk[1:] == ordered_chunk[:-1]):
                raise ValueError("duplicate source cluster member row")
            if np.any(seen_member_row[rows]):
                raise ValueError("source cluster member row appears more than once")
            seen_member_row[rows] = True
            if np.any(points.cluster_id_by_point[rows] != cluster_id) or np.any(
                points.cluster_status_by_point[rows] != int(status)
            ):
                raise ValueError("source cluster member metadata is inconsistent")
            usable_counts[position] += int(
                np.count_nonzero(tree_input.ground_usable_by_point[rows])
            )
            readiness = readiness and bool(np.all(points.distance_ready_by_point[rows]))
            if status is VegetationClusterStatus.CONFIRMED:
                vegetation_id = int(tree_input.cluster_vegetation_id[position])
                if vegetation_id < 0 or np.any(
                    points.vegetation_id_by_point[rows] != vegetation_id
                ):
                    raise ValueError("confirmed source cluster vegetation IDs are inconsistent")
                if not np.all(points.distance_ready_by_point[rows]):
                    raise ValueError("confirmed source cluster must be fully distance-ready")
            elif (
                int(tree_input.cluster_vegetation_id[position]) != -1
                or np.any(points.vegetation_id_by_point[rows] != -1)
                or np.any(points.distance_ready_by_point[rows])
            ):
                raise ValueError("nonconfirmed source cluster has formal vegetation assignment")
        cluster_readiness[position] = readiness

    for start in range(0, p, step):
        end = min(p, start + step)
        assigned = points.cluster_id_by_point[start:end] >= 0
        if not np.array_equal(seen_member_row[start:end], assigned):
            raise ValueError("source membership does not exactly cover assigned rows")
        unassigned = ~assigned
        if (
            np.any(
                points.cluster_status_by_point[start:end][unassigned]
                != int(VegetationClusterStatus.UNASSIGNED)
            )
            or np.any(points.vegetation_id_by_point[start:end][unassigned] != -1)
            or np.any(points.distance_ready_by_point[start:end][unassigned])
        ):
            raise ValueError("unassigned source rows have inconsistent metadata")
    if not np.array_equal(usable_counts, tree_input.usable_ground_member_count_by_cluster):
        raise ValueError("usable ground member counts are inconsistent")
    del seen_member_row, usable_counts
    return cluster_order, cluster_readiness


def _allowed_cluster_ids(tree_input: TreeSegmentationInputSet) -> np.ndarray:
    values = np.asarray(
        (*tree_input.eligible_cluster_ids, *tree_input.review_only_cluster_ids),
        dtype=np.int32,
    )
    if len(values):
        values.sort(kind="stable")
        if np.any(values[1:] == values[:-1]):
            raise ValueError("allowed cluster IDs must be unique")
    return values


def _included_mask(
    tree_input: TreeSegmentationInputSet,
    canopy_model: CanopyHeightModelResult,
    allowed_cluster_ids: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    points = tree_input.vegetation_points
    cluster_ids = points.cluster_id_by_point[start:end]
    allowed = np.zeros(end - start, dtype=np.bool_)
    if len(allowed_cluster_ids):
        positions = np.searchsorted(allowed_cluster_ids, cluster_ids)
        valid = positions < len(allowed_cluster_ids)
        allowed[valid] = allowed_cluster_ids[positions[valid]] == cluster_ids[valid]
    statuses = points.cluster_status_by_point[start:end]
    status_allowed = (
        (statuses == int(VegetationClusterStatus.CONFIRMED))
        | (statuses == int(VegetationClusterStatus.REVIEW_REQUIRED))
        | (statuses == int(VegetationClusterStatus.PROVISIONAL))
    )
    hag = tree_input.heights_above_ground.height_agl_m[start:end]
    return (
        (cluster_ids >= 0)
        & allowed
        & status_allowed
        & (points.wire_id_by_point[start:end] < 0)
        & tree_input.ground_usable_by_point[start:end]
        & (hag.astype(np.float64, copy=False) >= np.float64(canopy_model.config.minimum_hag_m))
    )


def _validate_source_content_and_geometry(
    tree_input: TreeSegmentationInputSet,
    canopy_model: CanopyHeightModelResult,
    treetop_candidates: TreetopCandidateResult,
    canopy_segmentation: CanopySegmentationResult,
    allowed_cluster_ids: np.ndarray,
    chunk_size: int,
    *,
    p: int,
    n: int,
    k: int,
    c: int,
) -> tuple[int, tuple[float, float] | None, tuple[int, int]]:
    points = tree_input.vegetation_points
    hag = tree_input.heights_above_ground.height_agl_m
    expected_ground_usable = (
        tree_input.heights_above_ground.quality_flags
        & int(GroundQualityFlag.INVALID_GROUND_ESTIMATE)
    ) == 0
    if not np.array_equal(expected_ground_usable, tree_input.ground_usable_by_point):
        raise ValueError("ground_usable_by_point is inconsistent with A2B quality flags")

    eligible_count = 0
    minimum_x = math.inf
    maximum_x = -math.inf
    minimum_y = math.inf
    maximum_y = -math.inf
    known_statuses = np.asarray([int(value) for value in VegetationClusterStatus], dtype=np.uint8)
    step = max(1, chunk_size)
    for start in range(0, p, step):
        end = min(p, start + step)
        xyz = points.xyz_local[start:end]
        if not np.isfinite(xyz).all():
            raise ValueError("xyz_local contains nonfinite values")
        usable = tree_input.ground_usable_by_point[start:end]
        if np.any(~np.isfinite(hag[start:end][usable])) or np.any(hag[start:end][usable] < 0.0):
            raise ValueError("ground-usable HAG must be finite and nonnegative")
        if np.any(~np.isin(points.cluster_status_by_point[start:end], known_statuses)):
            raise ValueError("cluster_status_by_point contains an unsupported value")
        if np.any(points.cluster_id_by_point[start:end] < -1):
            raise ValueError("cluster_id_by_point contains an invalid negative value")
        if np.any(points.wire_id_by_point[start:end] < -1):
            raise ValueError("wire_id_by_point contains an invalid negative value")
        included = _included_mask(tree_input, canopy_model, allowed_cluster_ids, start, end)
        count = int(np.count_nonzero(included))
        eligible_count += count
        if count:
            xy = xyz[included, :2].astype(np.float64, copy=False)
            minimum_x = min(minimum_x, float(np.min(xy[:, 0])))
            maximum_x = max(maximum_x, float(np.max(xy[:, 0])))
            minimum_y = min(minimum_y, float(np.min(xy[:, 1])))
            maximum_y = max(maximum_y, float(np.max(xy[:, 1])))

    if eligible_count == 0:
        reconstructed_origin = None
        reconstructed_shape = (0, 0)
    else:
        cell_size = np.float64(canopy_model.config.cell_size_m)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            origin_x = np.floor(np.float64(minimum_x) / cell_size) * cell_size
            origin_y = np.floor(np.float64(minimum_y) / cell_size) * cell_size
            origin_x = np.float64(0.0) if origin_x == 0.0 else origin_x
            origin_y = np.float64(0.0) if origin_y == 0.0 else origin_y
            relative_x = np.maximum((np.float64(maximum_x) - origin_x) / cell_size, 0.0)
            relative_y = np.maximum((np.float64(maximum_y) - origin_y) / cell_size, 0.0)
        if (
            not np.isfinite(origin_x)
            or not np.isfinite(origin_y)
            or not np.isfinite(relative_x)
            or not np.isfinite(relative_y)
            or relative_x < 0.0
            or relative_y < 0.0
            or relative_x >= float(2**63)
            or relative_y >= float(2**63)
        ):
            raise _mapping_error(
                "source geometry cannot be represented on an int64 grid",
                p=p,
                n=n,
                k=k,
                c=c,
                eligible=eligible_count,
            )
        column_count = int(np.floor(relative_x).astype(np.int64)) + 1
        row_count = int(np.floor(relative_y).astype(np.int64)) + 1
        grid_cells = row_count * column_count
        if grid_cells > _INT64_MAX or grid_cells > _INTP_MAX:
            raise _mapping_error(
                "reconstructed grid shape is not representable",
                p=p,
                n=n,
                k=k,
                c=c,
                eligible=eligible_count,
            )
        reconstructed_origin = (float(origin_x), float(origin_y))
        reconstructed_shape = (row_count, column_count)

    candidate_shape = treetop_candidates.source_grid_shape
    segmentation_shape = canopy_segmentation.source_grid_shape
    if (
        canopy_model.canopy_height_m.shape != candidate_shape
        or candidate_shape != segmentation_shape
        or canopy_segmentation.region_label_by_cell.shape != segmentation_shape
        or canopy_model.grid_origin_xy != treetop_candidates.source_grid_origin_xy
        or treetop_candidates.source_grid_origin_xy != canopy_segmentation.source_grid_origin_xy
    ):
        raise _bundle_error("derived grid origins or shapes do not agree", p=p, n=n, k=k, c=c)
    if (
        reconstructed_shape != canopy_model.canopy_height_m.shape
        or reconstructed_origin != canopy_model.grid_origin_xy
    ):
        raise _mapping_error(
            "reconstructed source geometry disagrees with the supplied grid",
            p=p,
            n=n,
            k=k,
            c=c,
            eligible=eligible_count,
        )
    return eligible_count, reconstructed_origin, reconstructed_shape


def _config_values(value: Any) -> dict[str, Any]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _validate_bundle_content(
    canopy_model: CanopyHeightModelResult,
    treetop_candidates: TreetopCandidateResult,
    canopy_segmentation: CanopySegmentationResult,
    *,
    p: int,
    n: int,
    k: int,
    c: int,
) -> None:
    if (
        canopy_model.config != treetop_candidates.source_chm_config
        or canopy_model.config != canopy_segmentation.source_chm_config
        or treetop_candidates.config != canopy_segmentation.source_treetop_config
    ):
        raise _bundle_error("derived stage configurations do not agree", p=p, n=n, k=k, c=c)
    if not np.array_equal(treetop_candidates.candidate_id, np.arange(k, dtype=np.int32)):
        raise ValueError("candidate_id must equal the contiguous range 0..K-1")
    shape = canopy_model.canopy_height_m.shape
    rows = treetop_candidates.row_index
    columns = treetop_candidates.column_index
    if k and (
        np.any(rows < 0)
        or np.any(rows >= shape[0])
        or np.any(columns < 0)
        or np.any(columns >= shape[1])
    ):
        raise ValueError("candidate cell is outside the source grid")
    marker = rows * np.int64(shape[1]) + columns
    if not np.array_equal(marker, canopy_segmentation.marker_flat_cell_index_by_candidate):
        raise _bundle_error(
            "candidate marker cells do not agree with segmentation", p=p, n=n, k=k, c=c
        )
    if k and len(np.unique(marker)) != k:
        raise ValueError("candidate cells must be unique")
    if k:
        origin = canopy_model.grid_origin_xy
        assert origin is not None
        expected_xy = np.empty((k, 2), dtype=np.float64)
        expected_xy[:, 0] = (
            origin[0] + (columns.astype(np.float64) + 0.5) * canopy_model.config.cell_size_m
        )
        expected_xy[:, 1] = (
            origin[1] + (rows.astype(np.float64) + 0.5) * canopy_model.config.cell_size_m
        )
        if not np.array_equal(treetop_candidates.local_xy, expected_xy):
            raise ValueError("candidate local_xy is inconsistent with its grid cell")
        for actual, source, name in (
            (
                treetop_candidates.peak_height_m,
                canopy_model.canopy_height_m[rows, columns],
                "peak_height_m",
            ),
            (
                treetop_candidates.source_cell_point_count,
                canopy_model.point_count_by_cell[rows, columns],
                "source_cell_point_count",
            ),
            (
                treetop_candidates.source_support_flags,
                canopy_model.support_flags_by_cell[rows, columns],
                "source_support_flags",
            ),
            (
                treetop_candidates.source_peak_support_flags,
                canopy_model.peak_support_flags_by_cell[rows, columns],
                "source_peak_support_flags",
            ),
        ):
            if not np.array_equal(actual, source):
                raise _bundle_error(
                    f"candidate {name} disagrees with CHM",
                    p=p,
                    n=n,
                    k=k,
                    c=c,
                )
    if (
        not np.isfinite(treetop_candidates.peak_height_m).all()
        or not np.isfinite(treetop_candidates.prominence_m).all()
        or np.any(treetop_candidates.prominence_m < 0.0)
        or np.any(treetop_candidates.plateau_cell_count <= 0)
        or np.any(treetop_candidates.source_cell_point_count <= 0)
        or np.any(treetop_candidates.neighbour_occupied_cell_count < 0)
        or not np.isfinite(treetop_candidates.neighbourhood_occupied_fraction).all()
        or not np.isfinite(treetop_candidates.neighbourhood_support_fraction).all()
    ):
        raise ValueError("candidate evidence contains invalid values")

    labels = canopy_segmentation.region_label_by_cell
    domain = canopy_segmentation.segmentation_domain_mask
    ambiguity = canopy_segmentation.ambiguity_mask
    if np.any(labels < 0) or np.any(labels > k):
        raise ValueError("segmentation labels must be in [0, K]")
    if np.any(labels[~domain] != 0):
        raise ValueError("outside-domain segmentation labels must be zero")
    if np.any(ambiguity & (~domain | (labels == 0))):
        raise ValueError("ambiguity must be a subset of assigned domain cells")
    threshold = canopy_segmentation.config.minimum_segmentation_hag_m
    if threshold is None:
        threshold = canopy_model.config.minimum_hag_m
    expected_domain = canopy_model.occupied_mask & (
        canopy_model.canopy_height_m.astype(np.float64, copy=False) >= np.float64(threshold)
    )
    if not np.array_equal(domain, expected_domain):
        raise _bundle_error("segmentation domain disagrees with CHM", p=p, n=n, k=k, c=c)
    if k:
        if not np.array_equal(labels.ravel()[marker], np.arange(1, k + 1, dtype=np.int32)):
            raise _bundle_error("marker labels do not equal candidate_id + 1", p=p, n=n, k=k, c=c)
        if not np.all(domain.ravel()[marker]):
            raise _bundle_error(
                "candidate marker is outside the segmentation domain", p=p, n=n, k=k, c=c
            )

    region_counts = np.zeros(k, dtype=np.int64)
    ambiguous_counts = np.zeros(k, dtype=np.int64)
    flat_labels = labels.ravel()
    positive = flat_labels > 0
    if np.any(positive):
        np.add.at(region_counts, flat_labels[positive] - 1, 1)
    ambiguous_labels = flat_labels[ambiguity.ravel()]
    if len(ambiguous_labels):
        np.add.at(ambiguous_counts, ambiguous_labels - 1, 1)
    edge_flags = np.zeros(k, dtype=np.bool_)
    if k and shape != (0, 0):
        for boundary in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
            positive_boundary = boundary[boundary > 0]
            if len(positive_boundary):
                edge_flags[np.unique(positive_boundary) - 1] = True
    if (
        not np.array_equal(region_counts, canopy_segmentation.region_cell_count_by_candidate)
        or not np.array_equal(
            ambiguous_counts,
            canopy_segmentation.ambiguous_cell_count_by_candidate,
        )
        or not np.array_equal(
            edge_flags,
            canopy_segmentation.region_touches_grid_edge_by_candidate,
        )
    ):
        raise _bundle_error("segmentation candidate audit arrays disagree", p=p, n=n, k=k, c=c)
    assigned_cells = int(np.count_nonzero(domain & (labels > 0)))
    unassigned_cells = int(np.count_nonzero(domain & (labels == 0)))
    ambiguous_cells = int(np.count_nonzero(ambiguity))
    if (
        assigned_cells != canopy_segmentation.assigned_cell_count
        or unassigned_cells != canopy_segmentation.unassigned_domain_cell_count
        or ambiguous_cells != canopy_segmentation.ambiguous_cell_count
        or (k and np.any(region_counts <= 0))
    ):
        raise _bundle_error("segmentation scalar audit counts disagree", p=p, n=n, k=k, c=c)


def _map_rows(
    tree_input: TreeSegmentationInputSet,
    canopy_model: CanopyHeightModelResult,
    canopy_segmentation: CanopySegmentationResult,
    allowed_cluster_ids: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    included = _included_mask(tree_input, canopy_model, allowed_cluster_ids, start, end)
    local_positions = np.flatnonzero(included)
    if not len(local_positions):
        empty = np.empty(0, dtype=np.int64)
        return included, empty, empty
    xy = tree_input.vegetation_points.xyz_local[start:end][local_positions, :2].astype(
        np.float64,
        copy=True,
    )
    origin = canopy_model.grid_origin_xy
    assert origin is not None
    cell_size = np.float64(canopy_model.config.cell_size_m)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative_x = np.maximum((xy[:, 0] - np.float64(origin[0])) / cell_size, 0.0)
        relative_y = np.maximum((xy[:, 1] - np.float64(origin[1])) / cell_size, 0.0)
    if (
        not np.isfinite(relative_x).all()
        or not np.isfinite(relative_y).all()
        or np.any(relative_x < 0.0)
        or np.any(relative_y < 0.0)
        or np.any(relative_x >= float(2**63))
        or np.any(relative_y >= float(2**63))
    ):
        raise ValueError("eligible source coordinates cannot be represented as grid indexes")
    columns = np.floor(relative_x).astype(np.int64)
    rows = np.floor(relative_y).astype(np.int64)
    shape = canopy_segmentation.source_grid_shape
    if (
        np.any(rows < 0)
        or np.any(rows >= shape[0])
        or np.any(columns < 0)
        or np.any(columns >= shape[1])
    ):
        raise ValueError("eligible source row or column is outside the reconstructed grid")
    return included, rows, columns


def _support_bits(statuses: np.ndarray) -> np.ndarray:
    result = np.zeros(len(statuses), dtype=np.uint8)
    result[statuses == int(VegetationClusterStatus.CONFIRMED)] = int(CanopySupportFlag.CONFIRMED)
    result[statuses == int(VegetationClusterStatus.REVIEW_REQUIRED)] = int(
        CanopySupportFlag.REVIEW_REQUIRED
    )
    result[statuses == int(VegetationClusterStatus.PROVISIONAL)] = int(
        CanopySupportFlag.PROVISIONAL
    )
    return result


def _audit_chm_and_map_points(
    tree_input: TreeSegmentationInputSet,
    canopy_model: CanopyHeightModelResult,
    canopy_segmentation: CanopySegmentationResult,
    allowed_cluster_ids: np.ndarray,
    chunk_size: int,
    eligible_count: int,
    *,
    p: int,
    n: int,
    k: int,
    c: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    shape = canopy_model.canopy_height_m.shape
    point_count_audit = np.zeros(n, dtype=np.int64)
    canopy_height_audit = np.full(n, -np.inf, dtype=np.float32)
    support_audit = np.zeros(n, dtype=np.uint8)
    occupied_audit = np.empty(n, dtype=np.bool_)
    step = max(1, chunk_size)
    included_seen = 0
    for start in range(0, p, step):
        end = min(p, start + step)
        included, rows, columns = _map_rows(
            tree_input,
            canopy_model,
            canopy_segmentation,
            allowed_cluster_ids,
            start,
            end,
        )
        positions = np.flatnonzero(included)
        included_seen += len(positions)
        if not len(positions):
            continue
        flat = rows * np.int64(shape[1]) + columns
        hag = tree_input.heights_above_ground.height_agl_m[start:end][positions]
        support = _support_bits(
            tree_input.vegetation_points.cluster_status_by_point[start:end][positions]
        )
        np.add.at(point_count_audit, flat, 1)
        np.maximum.at(canopy_height_audit, flat, hag)
        np.bitwise_or.at(support_audit, flat, support)
    if included_seen != eligible_count:
        raise _mapping_error(
            "eligibility count changed between deterministic passes",
            p=p,
            n=n,
            k=k,
            c=c,
            eligible=eligible_count,
        )
    if n and int(np.max(point_count_audit, initial=0)) > _INT32_MAX:
        raise _mapping_error(
            "per-cell point count cannot be represented as int32",
            p=p,
            n=n,
            k=k,
            c=c,
            eligible=eligible_count,
        )
    occupied_audit[:] = point_count_audit > 0
    canopy_height_audit[~occupied_audit] = np.nan
    zero = occupied_audit & (canopy_height_audit == 0.0)
    canopy_height_audit[zero] = np.float32(0.0)
    mismatched = (
        int(np.count_nonzero(point_count_audit != canopy_model.point_count_by_cell.ravel()))
        + int(np.count_nonzero(occupied_audit != canopy_model.occupied_mask.ravel()))
        + int(
            np.count_nonzero(
                ~(
                    (canopy_height_audit == canopy_model.canopy_height_m.ravel())
                    | (
                        np.isnan(canopy_height_audit)
                        & np.isnan(canopy_model.canopy_height_m.ravel())
                    )
                )
            )
        )
        + int(np.count_nonzero(support_audit != canopy_model.support_flags_by_cell.ravel()))
    )
    if mismatched:
        raise _mapping_error(
            "CHM audit pass 1 disagrees with supplied rasters",
            p=p,
            n=n,
            k=k,
            c=c,
            eligible=eligible_count,
            mismatched=mismatched,
        )
    if (
        canopy_model.included_point_count != eligible_count
        or canopy_model.excluded_point_count != p - eligible_count
    ):
        raise _bundle_error(
            "CHM included/excluded counts disagree with source eligibility", p=p, n=n, k=k, c=c
        )

    peak_support_audit = np.zeros(n, dtype=np.uint8)
    segment_id_by_point = np.full(p, -1, dtype=np.int32)
    member_count_by_segment = np.zeros(k, dtype=np.int64)
    assigned_count = 0
    unseeded_count = 0
    outside_count = 0
    ambiguous_assigned_count = 0
    for start in range(0, p, step):
        end = min(p, start + step)
        included, rows, columns = _map_rows(
            tree_input,
            canopy_model,
            canopy_segmentation,
            allowed_cluster_ids,
            start,
            end,
        )
        positions = np.flatnonzero(included)
        if not len(positions):
            continue
        flat = rows * np.int64(shape[1]) + columns
        hag = tree_input.heights_above_ground.height_agl_m[start:end][positions]
        status_support = _support_bits(
            tree_input.vegetation_points.cluster_status_by_point[start:end][positions]
        )
        at_peak = hag == canopy_height_audit[flat]
        if np.any(at_peak):
            np.bitwise_or.at(peak_support_audit, flat[at_peak], status_support[at_peak])
        domain = canopy_segmentation.segmentation_domain_mask.ravel()[flat]
        labels = canopy_segmentation.region_label_by_cell.ravel()[flat]
        ambiguity = canopy_segmentation.ambiguity_mask.ravel()[flat]
        if np.any(labels < 0) or np.any(labels > k):
            raise ValueError("mapped segmentation label is outside [0, K]")
        positive = labels > 0
        global_rows = positions.astype(np.int64, copy=True) + np.int64(start)
        segment_values = labels[positive] - 1
        segment_id_by_point[global_rows[positive]] = segment_values
        assigned_count += int(np.count_nonzero(positive))
        unseeded_count += int(np.count_nonzero(domain & ~positive))
        outside_count += int(np.count_nonzero(~domain))
        ambiguous_assigned_count += int(np.count_nonzero(ambiguity & positive))
        if np.any(positive):
            np.add.at(member_count_by_segment, segment_values, 1)

    peak_mismatch = int(
        np.count_nonzero(peak_support_audit != canopy_model.peak_support_flags_by_cell.ravel())
    )
    if peak_mismatch:
        raise _mapping_error(
            "CHM peak-support audit disagrees with supplied raster",
            p=p,
            n=n,
            k=k,
            c=c,
            eligible=eligible_count,
            assigned=assigned_count,
            mismatched=peak_mismatch,
        )
    excluded_count = p - eligible_count
    eligible_unassigned = unseeded_count + outside_count
    if (
        assigned_count + eligible_unassigned + excluded_count != p
        or assigned_count + eligible_unassigned != eligible_count
        or int(np.sum(member_count_by_segment, dtype=np.int64)) != assigned_count
    ):
        raise _mapping_error(
            "point mapping reconciliation failed",
            p=p,
            n=n,
            k=k,
            c=c,
            eligible=eligible_count,
            assigned=assigned_count,
        )
    if k and np.any(member_count_by_segment <= 0):
        raise _mapping_error(
            "one or more candidate-linked segments have no source member point",
            p=p,
            n=n,
            k=k,
            c=c,
            eligible=eligible_count,
            assigned=assigned_count,
        )
    audit = {
        "source_point_count": p,
        "eligible_point_count": eligible_count,
        "assigned_point_count": assigned_count,
        "unseeded_domain_point_count": unseeded_count,
        "outside_domain_eligible_point_count": outside_count,
        "eligible_unassigned_point_count": eligible_unassigned,
        "excluded_point_count": excluded_count,
        "ambiguous_assigned_point_count": ambiguous_assigned_count,
    }
    del (
        point_count_audit,
        canopy_height_audit,
        support_audit,
        occupied_audit,
        peak_support_audit,
        allowed_cluster_ids,
    )
    return segment_id_by_point, member_count_by_segment, audit


def _group_members(
    segment_id_by_point: np.ndarray,
    member_count_by_segment: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = len(member_count_by_segment)
    offsets = np.empty(k + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = member_count_by_segment
    np.cumsum(offsets[1:], dtype=np.int64, out=offsets[1:])
    assigned = int(offsets[-1])
    members = np.empty(assigned, dtype=np.int64)
    cursors = offsets[:-1].copy()
    p = len(segment_id_by_point)
    step = max(1, chunk_size)
    for start in range(0, p, step):
        end = min(p, start + step)
        segment_chunk = segment_id_by_point[start:end]
        assigned_positions = np.flatnonzero(segment_chunk >= 0)
        if not len(assigned_positions):
            continue
        source_rows = assigned_positions.astype(np.int64, copy=True) + np.int64(start)
        segment_labels = np.array(
            segment_chunk[assigned_positions],
            dtype=np.int32,
            order="C",
            copy=True,
        )
        order = np.argsort(segment_labels, kind="stable")
        sorted_labels = segment_labels[order]
        sorted_rows = source_rows[order]
        run_start = 0
        while run_start < len(sorted_labels):
            segment = int(sorted_labels[run_start])
            run_end = run_start + 1
            while run_end < len(sorted_labels) and int(sorted_labels[run_end]) == segment:
                run_end += 1
            destination = int(cursors[segment])
            count = run_end - run_start
            members[destination : destination + count] = sorted_rows[run_start:run_end]
            cursors[segment] += count
            run_start = run_end
    if not np.array_equal(cursors, offsets[1:]):
        raise RuntimeError("member grouping cursors do not equal final offsets")
    for segment in range(k):
        rows = members[int(offsets[segment]) : int(offsets[segment + 1])]
        if len(rows) > 1 and np.any(rows[1:] <= rows[:-1]):
            raise RuntimeError("grouped source rows are not strictly increasing")
    return offsets, members


def _cluster_incidence(
    tree_input: TreeSegmentationInputSet,
    segment_id_by_point: np.ndarray,
    cluster_order: np.ndarray,
    cluster_readiness: np.ndarray,
    chunk_size: int,
    k: int,
    config: TreeResultConstructionConfig,
    *,
    p: int,
    n: int,
    c: int,
    allowed_count: int,
    b: int,
    assigned_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stamp = np.full(k, -1, dtype=np.int64)
    incidence_counts = np.zeros(k, dtype=np.int64)
    offsets = tree_input.cluster_member_offsets
    rows_by_cluster = tree_input.ordered_cluster_member_row_index
    step = max(1, chunk_size)
    for ordinal, source_position_raw in enumerate(cluster_order):
        source_position = int(source_position_raw)
        start = int(offsets[source_position])
        end = int(offsets[source_position + 1])
        for member_start in range(start, end, step):
            member_end = min(end, member_start + step)
            member_rows = rows_by_cluster[member_start:member_end]
            values = segment_id_by_point[member_rows]
            assigned = values[values >= 0]
            if not len(assigned):
                continue
            order = np.argsort(assigned, kind="stable")
            sorted_segments = assigned[order]
            previous = -1
            for raw_segment in sorted_segments:
                segment = int(raw_segment)
                if segment == previous:
                    continue
                previous = segment
                if int(stamp[segment]) != ordinal:
                    stamp[segment] = ordinal
                    incidence_counts[segment] += 1

    incidence_offsets = np.empty(k + 1, dtype=np.int64)
    incidence_offsets[0] = 0
    incidence_offsets[1:] = incidence_counts
    np.cumsum(
        incidence_offsets[1:],
        dtype=np.int64,
        out=incidence_offsets[1:],
    )
    incidence_count = int(incidence_offsets[-1])
    if incidence_count > config.maximum_grouped_member_rows:
        raise _resource_error(
            "source-cluster incidence count exceeds maximum_grouped_member_rows",
            p=p,
            n=n,
            k=k,
            c=c,
            assigned=assigned_count,
        )
    exact_peaks = _phase_peaks(
        p,
        n,
        k,
        c,
        allowed_count,
        b,
        assigned_count,
        incidence_count,
    )
    for phase in (
        "incidence_pass_2",
        "record_construction",
        "formal_constructor",
    ):
        _check_phase(
            phase,
            exact_peaks,
            config,
            p=p,
            n=n,
            k=k,
            c=c,
            assigned=assigned_count,
        )
    cluster_ids = np.empty(incidence_count, dtype=np.int32)
    cluster_statuses = np.empty(incidence_count, dtype=np.uint8)
    readiness = np.empty(incidence_count, dtype=np.bool_)
    cursors = incidence_offsets[:-1].copy()
    stamp.fill(-1)

    for ordinal, source_position_raw in enumerate(cluster_order):
        source_position = int(source_position_raw)
        start = int(offsets[source_position])
        end = int(offsets[source_position + 1])
        for member_start in range(start, end, step):
            member_end = min(end, member_start + step)
            member_rows = rows_by_cluster[member_start:member_end]
            values = segment_id_by_point[member_rows]
            assigned = values[values >= 0]
            if not len(assigned):
                continue
            order = np.argsort(assigned, kind="stable")
            sorted_segments = assigned[order]
            previous = -1
            for raw_segment in sorted_segments:
                segment = int(raw_segment)
                if segment == previous:
                    continue
                previous = segment
                if int(stamp[segment]) != ordinal:
                    stamp[segment] = ordinal
                    destination = int(cursors[segment])
                    cluster_ids[destination] = tree_input.cluster_id[source_position]
                    cluster_statuses[destination] = tree_input.cluster_status[source_position]
                    readiness[destination] = cluster_readiness[source_position]
                    cursors[segment] += 1
    if not np.array_equal(cursors, incidence_offsets[1:]):
        raise RuntimeError("source-cluster incidence cursors do not equal final offsets")
    for segment in range(k):
        ids = cluster_ids[int(incidence_offsets[segment]) : int(incidence_offsets[segment + 1])]
        if not len(ids) or (len(ids) > 1 and np.any(ids[1:] <= ids[:-1])):
            raise RuntimeError("source-cluster incidence slices must be nonempty and sorted")
    return incidence_offsets, cluster_ids, cluster_statuses, readiness


def _warnings(audit: dict[str, int]) -> tuple[str, ...]:
    values: list[str] = []
    if audit["ambiguous_assigned_point_count"] > 0:
        values.append("ambiguous assigned source points are retained in their V3B3 regions")
    if audit["eligible_unassigned_point_count"] > 0:
        values.append("eligible source points remain unassigned to a candidate-linked region")
    if audit["cross_cluster_record_count"] > 0:
        values.append("one or more canopy segments contain points from multiple V2 source clusters")
    if audit["nonconfirmed_source_record_count"] > 0:
        values.append(
            "one or more canopy segments include a V2 source cluster that is not confirmed"
        )
    return tuple(values)


def _provenance(
    tree_input: TreeSegmentationInputSet,
    canopy_model: CanopyHeightModelResult,
    treetop_candidates: TreetopCandidateResult,
    canopy_segmentation: CanopySegmentationResult,
    config: TreeResultConstructionConfig,
    audit: dict[str, int],
) -> dict[str, Any]:
    shape = canopy_model.canopy_height_m.shape
    origin = canopy_model.grid_origin_xy
    return {
        "source_identity": tree_input.identity.to_dict(),
        "canopy_height_model": {
            "config": _config_values(canopy_model.config),
            "included_point_count": canopy_model.included_point_count,
            "excluded_point_count": canopy_model.excluded_point_count,
            "grid_shape": list(shape),
            "grid_origin_xy": None if origin is None else list(origin),
        },
        "treetop_candidates": {
            "config": _config_values(treetop_candidates.config),
            "candidate_count": len(treetop_candidates.candidate_id),
            "local_maximum_plateau_count": treetop_candidates.local_maximum_plateau_count,
            "evidence_rejected_count": treetop_candidates.evidence_rejected_count,
            "suppressed_candidate_count": treetop_candidates.suppressed_candidate_count,
            "grid_shape": list(treetop_candidates.source_grid_shape),
            "grid_origin_xy": (
                None
                if treetop_candidates.source_grid_origin_xy is None
                else list(treetop_candidates.source_grid_origin_xy)
            ),
        },
        "canopy_segmentation": {
            "config": _config_values(canopy_segmentation.config),
            "candidate_count": canopy_segmentation.candidate_count,
            "seeded_component_count": canopy_segmentation.seeded_component_count,
            "unseeded_component_count": canopy_segmentation.unseeded_component_count,
            "assigned_cell_count": canopy_segmentation.assigned_cell_count,
            "unassigned_domain_cell_count": canopy_segmentation.unassigned_domain_cell_count,
            "ambiguous_cell_count": canopy_segmentation.ambiguous_cell_count,
            "grid_shape": list(canopy_segmentation.source_grid_shape),
            "grid_origin_xy": (
                None
                if canopy_segmentation.source_grid_origin_xy is None
                else list(canopy_segmentation.source_grid_origin_xy)
            ),
        },
        "result_construction": _config_values(config),
        "mapping_audit": {**audit, "chm_audit_success": True},
        "id_mapping": {
            "candidate_to_segment_rule": "candidate_id equals segment_id",
            "raster_label_rule": "positive region_label minus one equals segment_id",
            "point_unassigned_sentinel": -1,
            "tree_id_sentinel": -1,
            "segment_scope_description": "candidate-linked provisional canopy region",
        },
    }


def _coordinate_context(
    tree_input: TreeSegmentationInputSet,
    canopy_model: CanopyHeightModelResult,
) -> dict[str, Any]:
    origin = canopy_model.grid_origin_xy
    return {
        "frame": "LOCAL_CARTESIAN_XYZ",
        "unit": "m",
        "origin_xyz": list(tree_input.identity.origin_xyz),
        "grid": {
            "origin_xy": None if origin is None else list(origin),
            "shape": list(canopy_model.canopy_height_m.shape),
            "cell_size_m": canopy_model.config.cell_size_m,
            "row_axis": "local_y",
            "column_axis": "local_x",
            "index_rule": "floor((coordinate - grid_origin) / cell_size)",
            "flat_index_rule": "row * column_count + column",
        },
        "height": {"field": "height_agl_m", "unit": "m"},
    }


def _construct_result(
    tree_input: TreeSegmentationInputSet,
    canopy_model: CanopyHeightModelResult,
    treetop_candidates: TreetopCandidateResult,
    canopy_segmentation: CanopySegmentationResult,
    config: TreeResultConstructionConfig,
    segment_id_by_point: np.ndarray,
    member_offsets: np.ndarray,
    members: np.ndarray,
    incidence_offsets: np.ndarray,
    incidence_ids: np.ndarray,
    incidence_statuses: np.ndarray,
    incidence_readiness: np.ndarray,
    audit: dict[str, int],
) -> TreeSegmentationResult:
    p = len(segment_id_by_point)
    k = len(treetop_candidates.candidate_id)
    records: list[TreeInstanceRecord] = []
    cross_cluster_count = 0
    nonconfirmed_count = 0
    for segment_id in range(k):
        start = int(incidence_offsets[segment_id])
        end = int(incidence_offsets[segment_id + 1])
        source_ids = tuple(int(value) for value in incidence_ids[start:end])
        source_statuses = tuple(
            VegetationClusterStatus(int(value)) for value in incidence_statuses[start:end]
        )
        source_readiness = tuple(bool(value) for value in incidence_readiness[start:end])
        if len(source_ids) > 1:
            cross_cluster_count += 1
        if any(status is not VegetationClusterStatus.CONFIRMED for status in source_statuses):
            nonconfirmed_count += 1
        records.append(
            TreeInstanceRecord(
                tree_id=-1,
                segment_id=segment_id,
                status=TreeSegmentationStatus.PROVISIONAL,
                source_cluster_ids=source_ids,
                source_cluster_statuses=source_statuses,
                source_cluster_distance_ready=source_readiness,
                tree_distance_ready=False,
                confidence=0.0,
                geometry_summary=None,
                crown_summary=None,
                quality_flags=TreeSegmentationQualityFlag.NONE,
            )
        )
    audit["candidate_count"] = k
    audit["record_count"] = k
    audit["cross_cluster_record_count"] = cross_cluster_count
    audit["nonconfirmed_source_record_count"] = nonconfirmed_count

    tree_id_by_point = np.full(p, -1, dtype=np.int32)
    tree_ready_by_point = np.zeros(p, dtype=np.bool_)
    arrays = (
        segment_id_by_point,
        tree_id_by_point,
        tree_ready_by_point,
        member_offsets,
        members,
    )
    for value in arrays:
        if type(value) is not np.ndarray or not value.flags.owndata or not value.flags.c_contiguous:
            raise RuntimeError("formal result array is not canonical before freezing")
        value.setflags(write=False)
    expected_ids = tuple(id(value) for value in arrays)
    statistics = TreeSegmentationStatistics(
        input_point_count=p,
        record_count=k,
        confirmed_tree_count=0,
        review_required_count=0,
        provisional_count=k,
        rejected_count=0,
        unresolved_count=0,
        elapsed_seconds=0.0,
        phase_timings_seconds={},
    )
    try:
        result = TreeSegmentationResult(
            schema_version=TREE_SEGMENTATION_SCHEMA_VERSION,
            algorithm_version=TREE_SEGMENTATION_ALGORITHM_VERSION,
            source_id=tree_input.identity.source_id,
            manifest_source_id=tree_input.identity.manifest_source_id,
            origin_xyz=tree_input.identity.origin_xyz,
            config=TreeSegmentationConfig(),
            statistics=statistics,
            records=tuple(records),
            segment_id_by_point=segment_id_by_point,
            tree_id_by_point=tree_id_by_point,
            tree_distance_ready_by_point=tree_ready_by_point,
            tree_member_offsets=member_offsets,
            ordered_tree_member_row_index=members,
            input_provenance=_provenance(
                tree_input,
                canopy_model,
                treetop_candidates,
                canopy_segmentation,
                config,
                audit,
            ),
            coordinate_context=_coordinate_context(tree_input, canopy_model),
            warnings=_warnings(audit),
            limitations=_LIMITATIONS,
        )
    except MemoryError:
        raise
    except Exception as exc:
        raise TreeResultConstructionError(
            TreeResultConstructionErrorCode.INTERNAL_INVARIANT,
            "formal TreeSegmentationResult construction rejected prevalidated data",
            source_point_count=int(p),
            assigned_point_count=int(audit["assigned_point_count"]),
            candidate_count=int(k),
            source_cluster_count=int(len(tree_input.cluster_id)),
            grid_cell_count=int(canopy_model.canopy_height_m.size),
        ) from exc
    actual_ids = (
        id(result.segment_id_by_point),
        id(result.tree_id_by_point),
        id(result.tree_distance_ready_by_point),
        id(result.tree_member_offsets),
        id(result.ordered_tree_member_row_index),
    )
    if actual_ids != expected_ids:
        raise TreeResultConstructionError(
            TreeResultConstructionErrorCode.INTERNAL_INVARIANT,
            "formal TreeSegmentationResult constructor did not retain all result arrays",
            source_point_count=int(p),
            assigned_point_count=int(audit["assigned_point_count"]),
            candidate_count=int(k),
            source_cluster_count=int(len(tree_input.cluster_id)),
            grid_cell_count=int(canopy_model.canopy_height_m.size),
        )
    return result


def _build_tree_segmentation_result(
    tree_input: TreeSegmentationInputSet,
    canopy_model: CanopyHeightModelResult,
    treetop_candidates: TreetopCandidateResult,
    canopy_segmentation: CanopySegmentationResult,
    config: TreeResultConstructionConfig,
) -> TreeSegmentationResult:
    _validate_scalar_shells(tree_input, canopy_model, treetop_candidates, canopy_segmentation)
    p, n, k, c, m, allowed_count = _validate_structural_arrays(
        tree_input,
        canopy_model,
        treetop_candidates,
        canopy_segmentation,
    )
    b, conservative_peaks = _preflight_counts(
        p,
        n,
        k,
        c,
        m,
        allowed_count,
        config,
    )

    _validate_canonical_scalars(
        tree_input,
        canopy_model,
        treetop_candidates,
        canopy_segmentation,
    )
    _validate_identity_and_alignment(
        tree_input,
        canopy_model,
        treetop_candidates,
        canopy_segmentation,
        config.chunk_point_count,
        p=p,
        n=n,
        k=k,
        c=c,
    )
    _check_phase(
        "membership",
        conservative_peaks,
        config,
        p=p,
        n=n,
        k=k,
        c=c,
    )
    cluster_order, cluster_readiness = _validate_partitions_and_membership(
        tree_input,
        config.chunk_point_count,
        p=p,
        n=n,
        k=k,
        c=c,
    )
    allowed_ids = _allowed_cluster_ids(tree_input)
    eligible_count, _origin, reconstructed_shape = _validate_source_content_and_geometry(
        tree_input,
        canopy_model,
        treetop_candidates,
        canopy_segmentation,
        allowed_ids,
        config.chunk_point_count,
        p=p,
        n=n,
        k=k,
        c=c,
    )
    reconstructed_n = reconstructed_shape[0] * reconstructed_shape[1]
    if reconstructed_n > config.maximum_grid_cells:
        raise _resource_error(
            "reconstructed grid exceeds maximum_grid_cells",
            p=p,
            n=reconstructed_n,
            k=k,
            c=c,
        )
    _validate_bundle_content(
        canopy_model,
        treetop_candidates,
        canopy_segmentation,
        p=p,
        n=n,
        k=k,
        c=c,
    )
    _check_phase(
        "chm_audit_pass_1",
        conservative_peaks,
        config,
        p=p,
        n=n,
        k=k,
        c=c,
    )
    _check_phase(
        "chm_audit_pass_2",
        conservative_peaks,
        config,
        p=p,
        n=n,
        k=k,
        c=c,
    )
    segment_ids, member_counts, audit = _audit_chm_and_map_points(
        tree_input,
        canopy_model,
        canopy_segmentation,
        allowed_ids,
        config.chunk_point_count,
        eligible_count,
        p=p,
        n=n,
        k=k,
        c=c,
    )
    assigned = audit["assigned_point_count"]
    if assigned > config.maximum_grouped_member_rows:
        raise _resource_error(
            "assigned member count exceeds maximum_grouped_member_rows",
            p=p,
            n=n,
            k=k,
            c=c,
            assigned=assigned,
        )
    exact_peaks = _phase_peaks(
        p,
        n,
        k,
        c,
        allowed_count,
        b,
        assigned,
        assigned,
    )
    _check_phase(
        "point_grouping",
        exact_peaks,
        config,
        p=p,
        n=n,
        k=k,
        c=c,
        assigned=assigned,
    )
    member_offsets, members = _group_members(
        segment_ids,
        member_counts,
        config.chunk_point_count,
    )
    _check_phase(
        "incidence_pass_1",
        exact_peaks,
        config,
        p=p,
        n=n,
        k=k,
        c=c,
        assigned=assigned,
    )
    incidence_offsets, incidence_ids, incidence_statuses, incidence_readiness = _cluster_incidence(
        tree_input,
        segment_ids,
        cluster_order,
        cluster_readiness,
        config.chunk_point_count,
        k,
        config,
        p=p,
        n=n,
        c=c,
        allowed_count=allowed_count,
        b=b,
        assigned_count=assigned,
    )
    return _construct_result(
        tree_input,
        canopy_model,
        treetop_candidates,
        canopy_segmentation,
        config,
        segment_ids,
        member_offsets,
        members,
        incidence_offsets,
        incidence_ids,
        incidence_statuses,
        incidence_readiness,
        audit,
    )


def build_tree_segmentation_result(
    tree_input: TreeSegmentationInputSet,
    canopy_model: CanopyHeightModelResult,
    treetop_candidates: TreetopCandidateResult,
    canopy_segmentation: CanopySegmentationResult,
    config: TreeResultConstructionConfig | None = None,
) -> TreeSegmentationResult:
    """Map source rows through the validated V3B1/V3B3 grid and labels."""

    if type(tree_input) is not TreeSegmentationInputSet:
        raise TypeError("tree_input must be a TreeSegmentationInputSet")
    if type(canopy_model) is not CanopyHeightModelResult:
        raise TypeError("canopy_model must be a CanopyHeightModelResult")
    if type(treetop_candidates) is not TreetopCandidateResult:
        raise TypeError("treetop_candidates must be a TreetopCandidateResult")
    if type(canopy_segmentation) is not CanopySegmentationResult:
        raise TypeError("canopy_segmentation must be a CanopySegmentationResult")
    if config is None:
        effective_config = TreeResultConstructionConfig()
    elif type(config) is TreeResultConstructionConfig:
        effective_config = config
        canonical_config = TreeResultConstructionConfig(
            **{field.name: getattr(config, field.name) for field in fields(config)}
        )
        if effective_config != canonical_config or any(
            type(getattr(effective_config, field.name))
            is not type(getattr(canonical_config, field.name))
            for field in fields(config)
        ):
            raise ValueError("config is not canonical")
    else:
        raise TypeError("config must be a TreeResultConstructionConfig or None")
    try:
        return _build_tree_segmentation_result(
            tree_input,
            canopy_model,
            treetop_candidates,
            canopy_segmentation,
            effective_config,
        )
    except MemoryError as exc:
        raise TreeResultConstructionError(
            TreeResultConstructionErrorCode.ALLOCATION_FAILED,
            "tree result construction allocation failed",
            source_point_count=int(tree_input.identity.point_count),
            candidate_count=int(len(treetop_candidates.candidate_id)),
            source_cluster_count=int(len(tree_input.cluster_id)),
            grid_cell_count=int(canopy_model.canopy_height_m.size),
        ) from exc


__all__ = [
    "TreeResultConstructionConfig",
    "TreeResultConstructionError",
    "TreeResultConstructionErrorCode",
    "build_tree_segmentation_result",
]
