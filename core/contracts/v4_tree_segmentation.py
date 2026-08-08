"""V4 single-tree segmentation contracts (PLAN-FROZEN consolidated plan).

This module freezes the V4-BUNDLE-A public contracts: the segmentation config,
the closed error/state enums, the reason/quality flags, the apex/trunk/trunk-
apex/structural-path/rotating-profile/hypothesis/slice/relation/tree records,
the final point-assignment representation and ``V4TreeSegmentationResult``.
It performs no file access and no source replay; every record is a frozen,
digest-bound dataclass validated in ``__post_init__``.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum, IntFlag
from pathlib import Path
from typing import Any

import numpy as np

from core.contracts.frv import SourceIdentity, frv_digest

# ---------------------------------------------------------------------------
# Public version constants
# ---------------------------------------------------------------------------

V4_TREE_SEGMENTATION_SCHEMA_VERSION = "v4-tree-segmentation-result-v1"
V4_TREE_SEGMENTATION_ALGORITHM_VERSION = "v4-algorithm-v1"
V4_TREE_SEGMENTATION_CONFIG_SCHEMA_VERSION = "v4-tree-segmentation-config-v1"
V4_TREE_SEGMENTATION_RESULT_DIGEST_DOMAIN = "V4TREESEG1"

# Algorithm-V1 immutable physical scale multipliers and gates.  These are not
# part of the config schema; they are frozen scoring authority shared with the
# runtime and tests.
LTR_XY_MIN_XY_FACTOR = 2.0
LTR_XY_MIN_LA_FACTOR = 0.35
LTR_Z_MIN_SLICE_FACTOR = 2.0
LTA_MIN_LA_FACTOR = 1.0
LTA_MIN_XY_FACTOR = 2.0
LPATH_XY_MIN_LTR_XY_FACTOR = 1.0
LPATH_XY_MIN_LA_FACTOR = 0.5
STRUCTURAL_PATH_DOWNWARD_WEIGHT = 0.25
STRUCTURAL_PATH_DENSITY_WEIGHT = 0.25
ROTATING_PROFILE_RAY_COUNT = 24
ROTATING_PROFILE_STEP_FACTOR = 1.0
ROTATING_PROFILE_HALF_WIDTH_FACTOR = 0.5
ROTATING_PROFILE_VALLEY_MIN = 0.65
ROTATING_PROFILE_DROP_MIN = 0.25
ROTATING_PROFILE_REBOUND_MIN = 0.25
ROTATING_PROFILE_DENSITY_RATIO = 0.65
TRUNK_BIN_MIN_POINTS = 3
TRUNK_COMPONENT_MIN_POINTS = 6
TRUNK_COMPONENT_MIN_BINS = 3
TRUNK_CORE_RADIUS_FACTOR = 0.5
PCA_REL_GAP_MIN = 1e-6
PCA_ABS_EPS = 1e-12
RESPONSIBILITY_MIN = 0.55
RESPONSIBILITY_REVIEW_MARGIN = 0.08
RELATION_MIN = 0.45
RELATION_REVIEW_MARGIN = 0.12
ASSIGNMENT_MIN = 0.62
ASSIGNMENT_REVIEW_MARGIN = 0.10
TRUNK_CANDIDATE_MIN = 0.62
TRUNK_WEAK_MIN = 0.52
TRUNK_APEX_STRONG_MIN = 0.65
TRUNK_APEX_REVIEW_MIN = 0.55

_RESPONSIBILITY_WEIGHTS = (0.16, 0.16, 0.14, 0.14, 0.10, 0.10, 0.10, 0.10)
_RELATION_WEIGHTS = (0.14, 0.10, 0.16, 0.14, 0.14, 0.16, 0.10, 0.06)
_ASSIGNMENT_WEIGHTS = (0.18, 0.14, 0.12, 0.10, 0.12, 0.10, 0.08, 0.10, 0.06)
_TRUNK_CHANNEL_WEIGHTS = (0.35, 0.30, 0.20, 0.15)
_ASSOCIATION_WEIGHTS = (0.28, 0.20, 0.22, 0.18, 0.12)
_RELATION_CHANNEL_NAMES = (
    "horizontal_distance",
    "height_difference",
    "saddle_drop",
    "xy_crown_connectivity",
    "empty_gap_evidence",
    "slice_continuity",
    "local_point_geometry",
    "optional_trunk_evidence",
)
_ASSIGNMENT_CHANNEL_NAMES = (
    "slice_measured_support",
    "adjacent_slice_continuity",
    "xy_prior",
    "apex_support",
    "local_hag_density",
    "structure_direction",
    "gap_occlusion",
    "competitor_margin",
    "segmentation_quality",
)


# ---------------------------------------------------------------------------
# Closed enums
# ---------------------------------------------------------------------------


class V4TreeAssignmentState(IntEnum):
    UNASSIGNED = 0
    ASSIGNED = 1
    REVIEW_REQUIRED = 2


class V4TreeRelation(str, Enum):
    MERGE = "MERGE"
    KEEP_SEPARATE = "KEEP_SEPARATE"
    REVIEW = "REVIEW"


class V4ProvisionalRelation(str, Enum):
    PROVISIONAL_MERGE = "PROVISIONAL_MERGE"
    PROVISIONAL_KEEP_SEPARATE = "PROVISIONAL_KEEP_SEPARATE"
    PROVISIONAL_REVIEW = "PROVISIONAL_REVIEW"


class V4TrunkCandidateState(str, Enum):
    ACCEPTED = "ACCEPTED"
    WEAK = "WEAK"
    SUPPRESSED = "SUPPRESSED"


class V4TrunkApexAssociationState(str, Enum):
    STRONG_ASSOCIATION = "STRONG_ASSOCIATION"
    REVIEW_ASSOCIATION = "REVIEW_ASSOCIATION"
    UNSUPPORTED_ASSOCIATION = "UNSUPPORTED_ASSOCIATION"


class V4TrunkEvidenceAvailability(str, Enum):
    OBSERVED = "OBSERVED"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class V4AssociationLookupKind(str, Enum):
    EVALUATED = "EVALUATED"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    OBSERVED_OUT_OF_RADIUS = "OBSERVED_OUT_OF_RADIUS"


class V4StructuralPathAvailability(str, Enum):
    OBSERVED = "OBSERVED"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class V4RotatingProfileState(str, Enum):
    PROFILE_CONFIRMED_VALLEY = "PROFILE_CONFIRMED_VALLEY"
    PROFILE_WEAK_VALLEY = "PROFILE_WEAK_VALLEY"
    PROFILE_NO_VALLEY = "PROFILE_NO_VALLEY"
    PROFILE_NOT_AVAILABLE = "PROFILE_NOT_AVAILABLE"


class V4ProfileSampleState(str, Enum):
    OCCUPIED = "OCCUPIED"
    OBSERVED_EMPTY = "OBSERVED_EMPTY"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class V4TreeHypothesisState(str, Enum):
    ACTIVE = "ACTIVE"
    MERGED = "MERGED"
    CONFIRMED = "CONFIRMED"
    REVIEW = "REVIEW"
    SUPPRESSED_EMPTY = "SUPPRESSED_EMPTY"


class V4ConfigStatus(str, Enum):
    TEST_ONLY = "TEST_ONLY"
    CALIBRATION_CANDIDATE = "CALIBRATION_CANDIDATE"
    REVIEWED_FIXED_POLICY = "REVIEWED_FIXED_POLICY"
    PRODUCTION_APPROVED = "PRODUCTION_APPROVED"


class V4TreeSegmentationErrorCode(str, Enum):
    INPUT_ROOT_INVALID = "INPUT_ROOT_INVALID"
    INPUT_V1_REJECTED = "INPUT_V1_REJECTED"
    INPUT_MANIFEST_INVALID = "INPUT_MANIFEST_INVALID"
    INPUT_DIGEST_MISMATCH = "INPUT_DIGEST_MISMATCH"
    INPUT_IDENTITY_MISMATCH = "INPUT_IDENTITY_MISMATCH"
    INPUT_INDEX_INVALID = "INPUT_INDEX_INVALID"
    INPUT_NONFINITE = "INPUT_NONFINITE"
    CONFIG_INVALID = "CONFIG_INVALID"
    CONFIG_DIGEST_MISMATCH = "CONFIG_DIGEST_MISMATCH"
    CONFIG_STATUS_UNSUPPORTED = "CONFIG_STATUS_UNSUPPORTED"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    SPATIAL_INDEX_INCOMPLETE = "SPATIAL_INDEX_INCOMPLETE"
    CANDIDATE_INVENTORY_INCOMPLETE = "CANDIDATE_INVENTORY_INCOMPLETE"
    ASSOCIATION_INVENTORY_INCOMPLETE = "ASSOCIATION_INVENTORY_INCOMPLETE"
    PATH_EVALUATION_INCOMPLETE = "PATH_EVALUATION_INCOMPLETE"
    PROFILE_EVALUATION_INCOMPLETE = "PROFILE_EVALUATION_INCOMPLETE"
    RESPONSIBILITY_INCOMPLETE = "RESPONSIBILITY_INCOMPLETE"
    SLICE_EVALUATION_INCOMPLETE = "SLICE_EVALUATION_INCOMPLETE"
    RELATION_EVALUATION_INCOMPLETE = "RELATION_EVALUATION_INCOMPLETE"
    UNION_CONSTRAINT_CONFLICT = "UNION_CONSTRAINT_CONFLICT"
    ASSIGNMENT_INCOMPLETE = "ASSIGNMENT_INCOMPLETE"
    CONSERVATION_FAILED = "CONSERVATION_FAILED"
    DETERMINISM_FAILED = "DETERMINISM_FAILED"
    ARTIFACT_SERIALIZATION_FAILED = "ARTIFACT_SERIALIZATION_FAILED"
    ARTIFACT_HASH_FAILED = "ARTIFACT_HASH_FAILED"
    PUBLICATION_FAILED = "PUBLICATION_FAILED"
    INTERNAL_INVARIANT = "INTERNAL_INVARIANT"


class V4ApexQualityFlag(IntFlag):
    NONE = 0
    LOW_SUPPORT = 1
    EDGE_LIMITED = 2
    SPARSE_SUPPORT = 4
    PROMINENCE_MISSING = 8
    PLATEAU_COLLAPSE = 16
    SUPPRESSED = 32
    REVIEW_REQUIRED = 64


class V4TrunkQualityFlag(IntFlag):
    NONE = 0
    INSUFFICIENT_SUPPORT = 1
    PCA_DEGENERATE = 2
    WEAK_CORE_SUPPORT = 4
    LOW_SCORE = 8
    UNMATCHED = 16
    OCCLUDED_INVENTORY = 32
    REVIEW_REQUIRED = 64


class V4AssociationQualityFlag(IntFlag):
    NONE = 0
    EVIDENCE_INCOMPLETE = 1
    UNSUPPORTED_GEOMETRY = 2
    OCCLUSION = 4
    PATH_NOT_AVAILABLE = 8
    REVIEW_BAND = 16
    REVIEW_REQUIRED = 32


class V4ProfileQualityFlag(IntFlag):
    NONE = 0
    NOT_AVAILABLE = 1
    OCCLUDED_SAMPLE = 2
    NO_VALLEY = 4
    WEAK_VALLEY = 8
    MEASURED_GAP = 16
    REVIEW_REQUIRED = 32


class V4SegmentationQualityFlag(IntFlag):
    NONE = 0
    REVIEW_REQUIRED = 1
    OPTIONAL_EVIDENCE_INSUFFICIENT = 2
    INPUT_LOCALIZED_REVIEW = 4
    DETERMINISM_NOTE = 8
    TRUNK_ABSENT = 16


class V4AssignmentReasonFlag(IntFlag):
    NONE = 0
    ASSIGNED_CLEAR_BEST = 1
    ASSIGNED_NO_RIVAL = 2
    REVIEW_MANDATORY = 4
    REVIEW_MARGIN = 8
    REVIEW_CONFLICT = 16
    UNASSIGNED_LOW_SUPPORT = 32
    HARD_FAIL = 64


# ---------------------------------------------------------------------------
# Private strict validation helpers
# ---------------------------------------------------------------------------


def _exact_str(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a built-in str")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if value != value.strip():
        raise ValueError(f"{name} must not have surrounding whitespace")
    return value


def _exact_enum(value: Any, enum_type: type, name: str) -> Any:
    if type(value) is not enum_type:
        raise ValueError(f"{name} must be a {enum_type.__name__}")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative built-in int")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive built-in int")
    return value


def _finite_float(value: Any, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite built-in float")
    return value


def _finite_vec3(value: Any, name: str) -> tuple[float, float, float]:
    if type(value) not in (list, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a three-value sequence")
    return (
        _finite_float(value[0], f"{name}[0]"),
        _finite_float(value[1], f"{name}[1]"),
        _finite_float(value[2], f"{name}[2]"),
    )


def _sha(value: Any, name: str) -> str:
    text = _exact_str(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact_fields(data: Any, fields: set[str], name: str) -> dict[str, Any]:
    payload = _mapping(data, name)
    actual = set(payload)
    missing = fields - actual
    unknown = actual - fields
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    return dict(payload)


def _enum_from_value(value: Any, enum_type: type, name: str) -> Any:
    if type(value) is not str:
        raise ValueError(f"{name} must be a string enum value")
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is unsupported") from exc


def _str_tuple(value: Any, name: str) -> tuple[str, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be a JSON array")
    for item in value:
        _exact_str(item, f"{name}[]")
    return tuple(value)


def _record_tuple(value: Any, record_type: type, name: str) -> tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be a JSON array")
    records = tuple(value)
    for item in records:
        if not isinstance(item, record_type):
            raise ValueError(f"{name} must contain {record_type.__name__} values")
    return records


def _flag_mask(value: Any, flag_type: type, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative built-in int flag mask")
    known = 0
    for member in flag_type.__members__.values():
        known |= int(member.value)
    if value & ~known:
        raise ValueError(f"{name} contains unknown flag bit(s)")
    return value


def _json_safe_records(value: Any, name: str) -> dict[str, Any]:
    return dict(value)


def _require_ndarray(
    value: Any,
    name: str,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    finite: bool,
) -> None:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a numpy ndarray")
    if value.dtype != dtype:
        raise ValueError(f"{name} must use dtype {dtype.str}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not value.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be C-contiguous")
    if value.flags.writeable:
        raise ValueError(f"{name} must be read-only")
    if finite and not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


# ---------------------------------------------------------------------------
# V4TreeSegmentationConfig
# ---------------------------------------------------------------------------


def _derive_v4_config_digest(
    *,
    schema_version: str,
    status: V4ConfigStatus,
    xy_cell_size_m: float,
    apex_neighborhood_radius_m: float,
    minimum_apex_prominence_m: float,
    density_trend_window_m: float,
    slice_thickness_m: float,
    slice_overlap_fraction: float,
    relation_neighborhood_radius_m: float,
    maximum_points: int,
    maximum_grid_cells: int,
    maximum_fragments: int,
    maximum_local_edges: int,
    maximum_path_evaluations: int,
    maximum_profile_samples: int,
) -> str:
    return frv_digest(
        {
            "digest_domain": "V4SEGCFG1",
            "schema_version": schema_version,
            "status": status.value,
            "xy_cell_size_m": xy_cell_size_m,
            "apex_neighborhood_radius_m": apex_neighborhood_radius_m,
            "minimum_apex_prominence_m": minimum_apex_prominence_m,
            "density_trend_window_m": density_trend_window_m,
            "slice_thickness_m": slice_thickness_m,
            "slice_overlap_fraction": slice_overlap_fraction,
            "relation_neighborhood_radius_m": relation_neighborhood_radius_m,
            "maximum_points": maximum_points,
            "maximum_grid_cells": maximum_grid_cells,
            "maximum_fragments": maximum_fragments,
            "maximum_local_edges": maximum_local_edges,
            "maximum_path_evaluations": maximum_path_evaluations,
            "maximum_profile_samples": maximum_profile_samples,
        }
    )


@dataclass(frozen=True, slots=True)
class V4TreeSegmentationConfig:
    """Frozen V4 segmentation configuration.

    Only the seven core physical scales and the operational resource ceilings
    are configurable.  All Algorithm-V1 channel weights and decision constants
    are immutable and never appear in this schema.
    """

    schema_version: str
    status: V4ConfigStatus
    xy_cell_size_m: float
    apex_neighborhood_radius_m: float
    minimum_apex_prominence_m: float
    density_trend_window_m: float
    slice_thickness_m: float
    slice_overlap_fraction: float
    relation_neighborhood_radius_m: float
    maximum_points: int
    maximum_grid_cells: int
    maximum_fragments: int
    maximum_local_edges: int
    maximum_path_evaluations: int
    maximum_profile_samples: int
    config_digest: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str:
            raise TypeError("schema_version must be a built-in str")
        if self.schema_version != V4_TREE_SEGMENTATION_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported V4 segmentation config schema version")
        _exact_enum(self.status, V4ConfigStatus, "status")
        for name in (
            "xy_cell_size_m",
            "apex_neighborhood_radius_m",
            "minimum_apex_prominence_m",
            "density_trend_window_m",
            "slice_thickness_m",
            "relation_neighborhood_radius_m",
        ):
            _finite_float(getattr(self, name), name)
        if self.xy_cell_size_m <= 0.0:
            raise ValueError("xy_cell_size_m must be strictly positive")
        if self.apex_neighborhood_radius_m <= 0.0:
            raise ValueError("apex_neighborhood_radius_m must be strictly positive")
        if self.minimum_apex_prominence_m < 0.0:
            raise ValueError("minimum_apex_prominence_m must be nonnegative")
        if self.density_trend_window_m <= 0.0:
            raise ValueError("density_trend_window_m must be strictly positive")
        if self.slice_thickness_m <= 0.0:
            raise ValueError("slice_thickness_m must be strictly positive")
        if self.relation_neighborhood_radius_m <= 0.0:
            raise ValueError("relation_neighborhood_radius_m must be strictly positive")
        if type(self.slice_overlap_fraction) is not float:
            raise ValueError("slice_overlap_fraction must be a built-in float")
        if not (0.0 <= self.slice_overlap_fraction < 1.0):
            raise ValueError("slice_overlap_fraction must be in [0, 1)")
        for name in (
            "maximum_points",
            "maximum_grid_cells",
            "maximum_fragments",
            "maximum_local_edges",
            "maximum_path_evaluations",
            "maximum_profile_samples",
        ):
            _positive_int(getattr(self, name), name)
        _sha(self.config_digest, "config_digest")
        if self.config_digest != self.computed_digest():
            raise ValueError("config_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": "V4SEGCFG1",
            "schema_version": self.schema_version,
            "status": self.status.value,
            "xy_cell_size_m": self.xy_cell_size_m,
            "apex_neighborhood_radius_m": self.apex_neighborhood_radius_m,
            "minimum_apex_prominence_m": self.minimum_apex_prominence_m,
            "density_trend_window_m": self.density_trend_window_m,
            "slice_thickness_m": self.slice_thickness_m,
            "slice_overlap_fraction": self.slice_overlap_fraction,
            "relation_neighborhood_radius_m": self.relation_neighborhood_radius_m,
            "maximum_points": self.maximum_points,
            "maximum_grid_cells": self.maximum_grid_cells,
            "maximum_fragments": self.maximum_fragments,
            "maximum_local_edges": self.maximum_local_edges,
            "maximum_path_evaluations": self.maximum_path_evaluations,
            "maximum_profile_samples": self.maximum_profile_samples,
        }

    def computed_digest(self) -> str:
        payload = dict(self.to_digest_dict())
        payload.pop("digest_domain")
        return frv_digest(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.to_digest_dict())
        payload.pop("digest_domain")
        payload["config_digest"] = self.config_digest
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4TreeSegmentationConfig:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["status"] = _enum_from_value(values["status"], V4ConfigStatus, "status")
        return cls(**values)


# ---------------------------------------------------------------------------
# V4ApexCandidateRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4ApexCandidateRecord:
    """One measured apex proposal (never a tree)."""

    candidate_id: str
    supporting_original_index: int
    xyz_local_m: tuple[float, float, float]
    hag_m: float
    prominence_m: float | None
    prominence_available: bool
    plateau_support_count: int
    cell_point_count: int
    density_points_per_m2: float
    neighborhood_median_density_points_per_m2: float
    apex_support_score: float
    quality_flags: int
    reason_flags: int
    diagnostic_digest: str

    def __post_init__(self) -> None:
        _exact_str(self.candidate_id, "candidate_id")
        if type(self.supporting_original_index) is not int:
            raise ValueError("supporting_original_index must be a built-in int")
        if self.supporting_original_index < 0:
            raise ValueError("supporting_original_index must be nonnegative")
        object.__setattr__(self, "xyz_local_m", _finite_vec3(self.xyz_local_m, "xyz_local_m"))
        _finite_float(self.hag_m, "hag_m")
        if self.hag_m < 0.0:
            raise ValueError("hag_m must be nonnegative")
        if self.prominence_m is not None:
            _finite_float(self.prominence_m, "prominence_m")
        if type(self.prominence_available) is not bool:
            raise ValueError("prominence_available must be a built-in bool")
        _non_negative_int(self.plateau_support_count, "plateau_support_count")
        _positive_int(self.cell_point_count, "cell_point_count")
        _finite_float(self.density_points_per_m2, "density_points_per_m2")
        _finite_float(
            self.neighborhood_median_density_points_per_m2,
            "neighborhood_median_density_points_per_m2",
        )
        if self.density_points_per_m2 < 0.0:
            raise ValueError("density_points_per_m2 must be nonnegative")
        if self.neighborhood_median_density_points_per_m2 < 0.0:
            raise ValueError(
                "neighborhood_median_density_points_per_m2 must be nonnegative"
            )
        _finite_float(self.apex_support_score, "apex_support_score")
        if not (0.0 <= self.apex_support_score <= 1.0):
            raise ValueError("apex_support_score must be in [0, 1]")
        _flag_mask(self.quality_flags, V4ApexQualityFlag, "quality_flags")
        _flag_mask(self.reason_flags, V4ApexQualityFlag, "reason_flags")
        _sha(self.diagnostic_digest, "diagnostic_digest")
        if self.diagnostic_digest != self.computed_digest():
            raise ValueError("diagnostic_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": "V4APX1",
            "candidate_id": self.candidate_id,
            "supporting_original_index": self.supporting_original_index,
            "xyz_local_m": list(self.xyz_local_m),
            "hag_m": self.hag_m,
            "prominence_m": self.prominence_m,
            "prominence_available": self.prominence_available,
            "plateau_support_count": self.plateau_support_count,
            "cell_point_count": self.cell_point_count,
            "density_points_per_m2": self.density_points_per_m2,
            "neighborhood_median_density_points_per_m2": (
                self.neighborhood_median_density_points_per_m2
            ),
            "apex_support_score": self.apex_support_score,
            "quality_flags": self.quality_flags,
            "reason_flags": self.reason_flags,
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.to_digest_dict())
        payload.pop("digest_domain")
        payload["diagnostic_digest"] = self.diagnostic_digest
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4ApexCandidateRecord:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["xyz_local_m"] = _finite_vec3(values["xyz_local_m"], "xyz_local_m")
        if values["prominence_m"] is not None:
            values["prominence_m"] = _finite_float(values["prominence_m"], "prominence_m")
        return cls(**values)


# ---------------------------------------------------------------------------
# V4TrunkCandidateRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4TrunkCandidateRecord:
    """One measured trunk candidate with fitted axis and four support channels."""

    candidate_id: str
    representative_original_index: int
    supporting_original_index_digest: str
    centroid_xyz: tuple[float, float, float]
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]
    refit_principal_axis: tuple[float, float, float]
    core_hag_min_m: float
    core_hag_max_m: float
    observed_vertical_span_m: float
    k_min: int
    k_max: int
    evaluated_bins: int
    occupied_required_bins: int
    max_bin_centroid_xy_drift_m: float
    tv: float
    ta: float
    ts: float
    tc: float
    trunk_candidate_score: float
    candidate_state: V4TrunkCandidateState
    quality_flags: int
    reason_flags: int
    diagnostic_digest: str

    def __post_init__(self) -> None:
        _exact_str(self.candidate_id, "candidate_id")
        if type(self.representative_original_index) is not int:
            raise ValueError("representative_original_index must be a built-in int")
        if self.representative_original_index < 0:
            raise ValueError("representative_original_index must be nonnegative")
        _sha(self.supporting_original_index_digest, "supporting_original_index_digest")
        for name in ("centroid_xyz", "bounds_min_xyz", "bounds_max_xyz"):
            object.__setattr__(self, name, _finite_vec3(getattr(self, name), name))
        object.__setattr__(
            self,
            "refit_principal_axis",
            _finite_vec3(self.refit_principal_axis, "refit_principal_axis"),
        )
        for name in ("core_hag_min_m", "core_hag_max_m", "observed_vertical_span_m"):
            _finite_float(getattr(self, name), name)
        if self.observed_vertical_span_m < 0.0:
            raise ValueError("observed_vertical_span_m must be nonnegative")
        _non_negative_int(self.k_min, "k_min")
        _non_negative_int(self.k_max, "k_max")
        _positive_int(self.evaluated_bins, "evaluated_bins")
        _non_negative_int(self.occupied_required_bins, "occupied_required_bins")
        if self.occupied_required_bins > self.evaluated_bins:
            raise ValueError("occupied_required_bins must not exceed evaluated_bins")
        _finite_float(self.max_bin_centroid_xy_drift_m, "max_bin_centroid_xy_drift_m")
        if self.max_bin_centroid_xy_drift_m < 0.0:
            raise ValueError("max_bin_centroid_xy_drift_m must be nonnegative")
        for name in ("tv", "ta", "ts", "tc", "trunk_candidate_score"):
            _finite_float(getattr(self, name), name)
        for name in ("tv", "ta", "ts", "tc"):
            if not (0.0 <= getattr(self, name) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        _exact_enum(self.candidate_state, V4TrunkCandidateState, "candidate_state")
        _flag_mask(self.quality_flags, V4TrunkQualityFlag, "quality_flags")
        _flag_mask(self.reason_flags, V4TrunkQualityFlag, "reason_flags")
        _sha(self.diagnostic_digest, "diagnostic_digest")
        if self.diagnostic_digest != self.computed_digest():
            raise ValueError("diagnostic_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": "V4TRK1",
            "candidate_id": self.candidate_id,
            "representative_original_index": self.representative_original_index,
            "supporting_original_index_digest": self.supporting_original_index_digest,
            "centroid_xyz": list(self.centroid_xyz),
            "bounds_min_xyz": list(self.bounds_min_xyz),
            "bounds_max_xyz": list(self.bounds_max_xyz),
            "refit_principal_axis": list(self.refit_principal_axis),
            "core_hag_min_m": self.core_hag_min_m,
            "core_hag_max_m": self.core_hag_max_m,
            "observed_vertical_span_m": self.observed_vertical_span_m,
            "k_min": self.k_min,
            "k_max": self.k_max,
            "evaluated_bins": self.evaluated_bins,
            "occupied_required_bins": self.occupied_required_bins,
            "max_bin_centroid_xy_drift_m": self.max_bin_centroid_xy_drift_m,
            "tv": self.tv,
            "ta": self.ta,
            "ts": self.ts,
            "tc": self.tc,
            "trunk_candidate_score": self.trunk_candidate_score,
            "candidate_state": self.candidate_state.value,
            "quality_flags": self.quality_flags,
            "reason_flags": self.reason_flags,
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.to_digest_dict())
        payload.pop("digest_domain")
        payload["diagnostic_digest"] = self.diagnostic_digest
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4TrunkCandidateRecord:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        for name in ("centroid_xyz", "bounds_min_xyz", "bounds_max_xyz", "refit_principal_axis"):
            values[name] = _finite_vec3(values[name], name)
        values["candidate_state"] = _enum_from_value(
            values["candidate_state"], V4TrunkCandidateState, "candidate_state"
        )
        return cls(**values)


# ---------------------------------------------------------------------------
# V4StructuralPathEvidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4StructuralPathEvidence:
    """One local CSP-style trunk-rooted sparse structural path."""

    path_id: str
    trunk_candidate_id: str
    apex_candidate_id: str
    domain_bounds_min_xyz: tuple[float, float, float]
    domain_bounds_max_xyz: tuple[float, float, float]
    node_count: int
    edge_count: int
    source_node_key: str | None
    target_node_key: str | None
    path_node_key_sequence: tuple[str, ...] | None
    straight_length_m: float
    weighted_path_length_m: float | None
    structural_path_efficiency: float | None
    availability: V4StructuralPathAvailability
    resource_audit: str
    quality_flags: int
    reason_flags: int
    evidence_digest: str

    def __post_init__(self) -> None:
        _exact_str(self.path_id, "path_id")
        _exact_str(self.trunk_candidate_id, "trunk_candidate_id")
        _exact_str(self.apex_candidate_id, "apex_candidate_id")
        object.__setattr__(
            self,
            "domain_bounds_min_xyz",
            _finite_vec3(self.domain_bounds_min_xyz, "domain_bounds_min_xyz"),
        )
        object.__setattr__(
            self,
            "domain_bounds_max_xyz",
            _finite_vec3(self.domain_bounds_max_xyz, "domain_bounds_max_xyz"),
        )
        _non_negative_int(self.node_count, "node_count")
        _non_negative_int(self.edge_count, "edge_count")
        if self.source_node_key is not None:
            _exact_str(self.source_node_key, "source_node_key")
        if self.target_node_key is not None:
            _exact_str(self.target_node_key, "target_node_key")
        if self.path_node_key_sequence is not None:
            object.__setattr__(
                self,
                "path_node_key_sequence",
                _str_tuple(self.path_node_key_sequence, "path_node_key_sequence"),
            )
        _finite_float(self.straight_length_m, "straight_length_m")
        if self.straight_length_m < 0.0:
            raise ValueError("straight_length_m must be nonnegative")
        if self.weighted_path_length_m is not None:
            _finite_float(self.weighted_path_length_m, "weighted_path_length_m")
            if self.weighted_path_length_m < 0.0:
                raise ValueError("weighted_path_length_m must be nonnegative")
        if self.structural_path_efficiency is not None:
            _finite_float(
                self.structural_path_efficiency, "structural_path_efficiency"
            )
            if not (0.0 <= self.structural_path_efficiency <= 1.0):
                raise ValueError("structural_path_efficiency must be in [0, 1]")
        _exact_enum(
            self.availability, V4StructuralPathAvailability, "availability"
        )
        _exact_str(self.resource_audit, "resource_audit")
        _flag_mask(self.quality_flags, V4TrunkQualityFlag, "quality_flags")
        _flag_mask(self.reason_flags, V4TrunkQualityFlag, "reason_flags")
        _sha(self.evidence_digest, "evidence_digest")
        if self.evidence_digest != self.computed_digest():
            raise ValueError("evidence_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": "V4PATH1",
            "path_id": self.path_id,
            "trunk_candidate_id": self.trunk_candidate_id,
            "apex_candidate_id": self.apex_candidate_id,
            "domain_bounds_min_xyz": list(self.domain_bounds_min_xyz),
            "domain_bounds_max_xyz": list(self.domain_bounds_max_xyz),
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "source_node_key": self.source_node_key,
            "target_node_key": self.target_node_key,
            "path_node_key_sequence": (
                list(self.path_node_key_sequence)
                if self.path_node_key_sequence is not None
                else None
            ),
            "straight_length_m": self.straight_length_m,
            "weighted_path_length_m": self.weighted_path_length_m,
            "structural_path_efficiency": self.structural_path_efficiency,
            "availability": self.availability.value,
            "resource_audit": self.resource_audit,
            "quality_flags": self.quality_flags,
            "reason_flags": self.reason_flags,
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.to_digest_dict())
        payload.pop("digest_domain")
        payload["evidence_digest"] = self.evidence_digest
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4StructuralPathEvidence:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        for name in ("domain_bounds_min_xyz", "domain_bounds_max_xyz"):
            values[name] = _finite_vec3(values[name], name)
        values["availability"] = _enum_from_value(
            values["availability"], V4StructuralPathAvailability, "availability"
        )
        return cls(**values)


# ---------------------------------------------------------------------------
# V4TrunkApexAssociationEvidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4TrunkApexAssociationEvidence:
    """One materialized trunk-apex association edge with five evidence channels."""

    association_id: str
    trunk_candidate_id: str
    apex_candidate_id: str
    tx: float
    tx_available: bool
    td: float
    td_available: bool
    tvc: float
    tvc_available: bool
    tg: float
    tg_available: bool
    tr: float
    tr_available: bool
    association_score_available: bool
    association_score: float
    association_state: V4TrunkApexAssociationState
    structural_path_evidence_reference: str | None
    lookup_kind: V4AssociationLookupKind
    quality_flags: int
    reason_flags: int
    diagnostic_digest: str

    def __post_init__(self) -> None:
        _exact_str(self.association_id, "association_id")
        _exact_str(self.trunk_candidate_id, "trunk_candidate_id")
        _exact_str(self.apex_candidate_id, "apex_candidate_id")
        for name in ("tx", "td", "tvc", "tg", "tr", "association_score"):
            _finite_float(getattr(self, name), name)
        for name in ("tx", "td", "tvc", "tg", "tr"):
            if not (0.0 <= getattr(self, name) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "tx_available",
            "td_available",
            "tvc_available",
            "tg_available",
            "tr_available",
            "association_score_available",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a built-in bool")
        _exact_enum(
            self.association_state, V4TrunkApexAssociationState, "association_state"
        )
        if self.structural_path_evidence_reference is not None:
            _exact_str(
                self.structural_path_evidence_reference,
                "structural_path_evidence_reference",
            )
        _exact_enum(self.lookup_kind, V4AssociationLookupKind, "lookup_kind")
        _flag_mask(self.quality_flags, V4AssociationQualityFlag, "quality_flags")
        _flag_mask(self.reason_flags, V4AssociationQualityFlag, "reason_flags")
        _sha(self.diagnostic_digest, "diagnostic_digest")
        if self.diagnostic_digest != self.computed_digest():
            raise ValueError("diagnostic_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": "V4TABLK1",
            "association_id": self.association_id,
            "trunk_candidate_id": self.trunk_candidate_id,
            "apex_candidate_id": self.apex_candidate_id,
            "tx": self.tx,
            "tx_available": self.tx_available,
            "td": self.td,
            "td_available": self.td_available,
            "tvc": self.tvc,
            "tvc_available": self.tvc_available,
            "tg": self.tg,
            "tg_available": self.tg_available,
            "tr": self.tr,
            "tr_available": self.tr_available,
            "association_score_available": self.association_score_available,
            "association_score": self.association_score,
            "association_state": self.association_state.value,
            "structural_path_evidence_reference": (
                self.structural_path_evidence_reference
            ),
            "lookup_kind": self.lookup_kind.value,
            "quality_flags": self.quality_flags,
            "reason_flags": self.reason_flags,
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.to_digest_dict())
        payload.pop("digest_domain")
        payload["diagnostic_digest"] = self.diagnostic_digest
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4TrunkApexAssociationEvidence:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["association_state"] = _enum_from_value(
            values["association_state"],
            V4TrunkApexAssociationState,
            "association_state",
        )
        values["lookup_kind"] = _enum_from_value(
            values["lookup_kind"], V4AssociationLookupKind, "lookup_kind"
        )
        return cls(**values)


# ---------------------------------------------------------------------------
# Rotating-profile records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4RotatingProfileRaySample:
    """One radial sample on one rotating-profile ray."""

    sample_index: int
    radial_distance_m: float
    state: V4ProfileSampleState
    hag_q75_m: float | None
    median_density_points_per_m2: float | None
    supporting_original_index_digest: str

    def __post_init__(self) -> None:
        _non_negative_int(self.sample_index, "sample_index")
        _finite_float(self.radial_distance_m, "radial_distance_m")
        if self.radial_distance_m < 0.0:
            raise ValueError("radial_distance_m must be nonnegative")
        _exact_enum(self.state, V4ProfileSampleState, "state")
        for name in ("hag_q75_m", "median_density_points_per_m2"):
            value = getattr(self, name)
            if value is not None:
                _finite_float(value, name)
        _sha(self.supporting_original_index_digest, "supporting_original_index_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "radial_distance_m": self.radial_distance_m,
            "state": self.state.value,
            "hag_q75_m": self.hag_q75_m,
            "median_density_points_per_m2": self.median_density_points_per_m2,
            "supporting_original_index_digest": self.supporting_original_index_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4RotatingProfileRaySample:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["state"] = _enum_from_value(values["state"], V4ProfileSampleState, "state")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class V4RotatingProfileRayRecord:
    """One deterministic rotating-profile ray."""

    ray_index: int
    theta_rad: float
    samples: tuple[V4RotatingProfileRaySample, ...]
    smoothed_hag_median_m: float | None
    smoothed_density_median_points_per_m2: float | None
    valley_state: V4RotatingProfileState
    measured_gap: bool
    boundary_sample_index: int | None

    def __post_init__(self) -> None:
        _non_negative_int(self.ray_index, "ray_index")
        _finite_float(self.theta_rad, "theta_rad")
        object.__setattr__(
            self,
            "samples",
            _record_tuple(self.samples, V4RotatingProfileRaySample, "samples"),
        )
        for name in ("smoothed_hag_median_m", "smoothed_density_median_points_per_m2"):
            value = getattr(self, name)
            if value is not None:
                _finite_float(value, name)
        _exact_enum(self.valley_state, V4RotatingProfileState, "valley_state")
        if type(self.measured_gap) is not bool:
            raise ValueError("measured_gap must be a built-in bool")
        if self.boundary_sample_index is not None:
            _non_negative_int(self.boundary_sample_index, "boundary_sample_index")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ray_index": self.ray_index,
            "theta_rad": self.theta_rad,
            "samples": [sample.to_dict() for sample in self.samples],
            "smoothed_hag_median_m": self.smoothed_hag_median_m,
            "smoothed_density_median_points_per_m2": (
                self.smoothed_density_median_points_per_m2
            ),
            "valley_state": self.valley_state.value,
            "measured_gap": self.measured_gap,
            "boundary_sample_index": self.boundary_sample_index,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4RotatingProfileRayRecord:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["samples"] = tuple(
            V4RotatingProfileRaySample.from_dict(_mapping(item, "sample"))
            for item in values["samples"]
        )
        values["valley_state"] = _enum_from_value(
            values["valley_state"], V4RotatingProfileState, "valley_state"
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class V4RotatingProfileEvidence:
    """24-ray deterministic crown-boundary evidence for one apex/hypothesis."""

    profile_id: str
    apex_candidate_id: str
    hypothesis_id: str | None
    rays: tuple[V4RotatingProfileRayRecord, ...]
    profile_state: V4RotatingProfileState
    measured_gap: bool
    profile_valley_support: float | None
    quality_flags: int
    reason_flags: int
    evidence_digest: str

    def __post_init__(self) -> None:
        _exact_str(self.profile_id, "profile_id")
        _exact_str(self.apex_candidate_id, "apex_candidate_id")
        if self.hypothesis_id is not None:
            _exact_str(self.hypothesis_id, "hypothesis_id")
        object.__setattr__(
            self,
            "rays",
            _record_tuple(self.rays, V4RotatingProfileRayRecord, "rays"),
        )
        if len(self.rays) != ROTATING_PROFILE_RAY_COUNT:
            raise ValueError("rays must contain exactly 24 ray records")
        _exact_enum(self.profile_state, V4RotatingProfileState, "profile_state")
        if type(self.measured_gap) is not bool:
            raise ValueError("measured_gap must be a built-in bool")
        if self.profile_valley_support is not None:
            _finite_float(self.profile_valley_support, "profile_valley_support")
            if not (0.0 <= self.profile_valley_support <= 1.0):
                raise ValueError("profile_valley_support must be in [0, 1]")
        _flag_mask(self.quality_flags, V4ProfileQualityFlag, "quality_flags")
        _flag_mask(self.reason_flags, V4ProfileQualityFlag, "reason_flags")
        _sha(self.evidence_digest, "evidence_digest")
        if self.evidence_digest != self.computed_digest():
            raise ValueError("evidence_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": "V4PROF1",
            "profile_id": self.profile_id,
            "apex_candidate_id": self.apex_candidate_id,
            "hypothesis_id": self.hypothesis_id,
            "rays": [ray.to_dict() for ray in self.rays],
            "profile_state": self.profile_state.value,
            "measured_gap": self.measured_gap,
            "profile_valley_support": self.profile_valley_support,
            "quality_flags": self.quality_flags,
            "reason_flags": self.reason_flags,
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.to_digest_dict())
        payload.pop("digest_domain")
        payload["evidence_digest"] = self.evidence_digest
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4RotatingProfileEvidence:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["rays"] = tuple(
            V4RotatingProfileRayRecord.from_dict(_mapping(item, "ray"))
            for item in values["rays"]
        )
        values["profile_state"] = _enum_from_value(
            values["profile_state"], V4RotatingProfileState, "profile_state"
        )
        return cls(**values)


# ---------------------------------------------------------------------------
# V4TreeHypothesisRecord and V4TreeSliceSummary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4TreeHypothesisRecord:
    """A reversible provisional/final tree hypothesis."""

    hypothesis_id: str
    primary_apex_candidate_id: str
    secondary_apex_candidate_ids: tuple[str, ...]
    trunk_candidate_ids: tuple[str, ...]
    fragment_refs: tuple[str, ...]
    profile_refs: tuple[str, ...]
    provisional_group_id: str | None
    state: V4TreeHypothesisState
    quality_flags: int
    reason_flags: int

    def __post_init__(self) -> None:
        _exact_str(self.hypothesis_id, "hypothesis_id")
        _exact_str(self.primary_apex_candidate_id, "primary_apex_candidate_id")
        object.__setattr__(
            self,
            "secondary_apex_candidate_ids",
            _str_tuple(self.secondary_apex_candidate_ids, "secondary_apex_candidate_ids"),
        )
        object.__setattr__(
            self,
            "trunk_candidate_ids",
            _str_tuple(self.trunk_candidate_ids, "trunk_candidate_ids"),
        )
        object.__setattr__(
            self, "fragment_refs", _str_tuple(self.fragment_refs, "fragment_refs")
        )
        object.__setattr__(
            self, "profile_refs", _str_tuple(self.profile_refs, "profile_refs")
        )
        if self.provisional_group_id is not None:
            _exact_str(self.provisional_group_id, "provisional_group_id")
        _exact_enum(self.state, V4TreeHypothesisState, "state")
        _flag_mask(self.quality_flags, V4SegmentationQualityFlag, "quality_flags")
        _flag_mask(self.reason_flags, V4SegmentationQualityFlag, "reason_flags")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "primary_apex_candidate_id": self.primary_apex_candidate_id,
            "secondary_apex_candidate_ids": list(self.secondary_apex_candidate_ids),
            "trunk_candidate_ids": list(self.trunk_candidate_ids),
            "fragment_refs": list(self.fragment_refs),
            "profile_refs": list(self.profile_refs),
            "provisional_group_id": self.provisional_group_id,
            "state": self.state.value,
            "quality_flags": self.quality_flags,
            "reason_flags": self.reason_flags,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4TreeHypothesisRecord:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["secondary_apex_candidate_ids"] = _str_tuple(
            values["secondary_apex_candidate_ids"], "secondary_apex_candidate_ids"
        )
        values["trunk_candidate_ids"] = _str_tuple(
            values["trunk_candidate_ids"], "trunk_candidate_ids"
        )
        values["fragment_refs"] = _str_tuple(values["fragment_refs"], "fragment_refs")
        values["profile_refs"] = _str_tuple(values["profile_refs"], "profile_refs")
        values["state"] = _enum_from_value(
            values["state"], V4TreeHypothesisState, "state"
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class V4TreeSliceSummary:
    """One height slice summary for a hypothesis."""

    slice_id: str
    hypothesis_id: str
    hag_min_m: float
    hag_max_m: float
    point_count: int
    centroid_xyz: tuple[float, float, float]
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]
    area_m2: float
    density_points_per_m2: float
    boundary_digest: str
    member_original_index_digest: str
    quality_flags: int
    reason_flags: int

    def __post_init__(self) -> None:
        _exact_str(self.slice_id, "slice_id")
        _exact_str(self.hypothesis_id, "hypothesis_id")
        _finite_float(self.hag_min_m, "hag_min_m")
        _finite_float(self.hag_max_m, "hag_max_m")
        if self.hag_min_m < 0.0 or self.hag_max_m < self.hag_min_m:
            raise ValueError("invalid slice HAG interval")
        _positive_int(self.point_count, "point_count")
        for name in ("centroid_xyz", "bounds_min_xyz", "bounds_max_xyz"):
            object.__setattr__(self, name, _finite_vec3(getattr(self, name), name))
        _finite_float(self.area_m2, "area_m2")
        if self.area_m2 < 0.0:
            raise ValueError("area_m2 must be nonnegative")
        _finite_float(self.density_points_per_m2, "density_points_per_m2")
        if self.density_points_per_m2 < 0.0:
            raise ValueError("density_points_per_m2 must be nonnegative")
        _sha(self.boundary_digest, "boundary_digest")
        _sha(self.member_original_index_digest, "member_original_index_digest")
        _flag_mask(self.quality_flags, V4SegmentationQualityFlag, "quality_flags")
        _flag_mask(self.reason_flags, V4SegmentationQualityFlag, "reason_flags")

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "hypothesis_id": self.hypothesis_id,
            "hag_min_m": self.hag_min_m,
            "hag_max_m": self.hag_max_m,
            "point_count": self.point_count,
            "centroid_xyz": list(self.centroid_xyz),
            "bounds_min_xyz": list(self.bounds_min_xyz),
            "bounds_max_xyz": list(self.bounds_max_xyz),
            "area_m2": self.area_m2,
            "density_points_per_m2": self.density_points_per_m2,
            "boundary_digest": self.boundary_digest,
            "member_original_index_digest": self.member_original_index_digest,
            "quality_flags": self.quality_flags,
            "reason_flags": self.reason_flags,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4TreeSliceSummary:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        for name in ("centroid_xyz", "bounds_min_xyz", "bounds_max_xyz"):
            values[name] = _finite_vec3(values[name], name)
        return cls(**values)


# ---------------------------------------------------------------------------
# V4TreeRelationEvidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4TreeRelationEvidence:
    """Eight-channel relation evidence between two hypothesis endpoints."""

    relation_id: str
    hypothesis_a_id: str
    hypothesis_b_id: str
    horizontal_distance_e: float
    height_difference_e: float
    saddle_drop_e: float
    xy_crown_connectivity_e: float
    empty_gap_evidence_e: float
    slice_continuity_e: float
    local_point_geometry_e: float
    optional_trunk_evidence_e: float
    merge_support: float
    separate_support: float
    relation: V4TreeRelation
    stage: str
    quality_flags: int
    reason_flags: int
    diagnostic_digest: str

    def __post_init__(self) -> None:
        _exact_str(self.relation_id, "relation_id")
        _exact_str(self.hypothesis_a_id, "hypothesis_a_id")
        _exact_str(self.hypothesis_b_id, "hypothesis_b_id")
        for name in (
            "horizontal_distance_e",
            "height_difference_e",
            "saddle_drop_e",
            "xy_crown_connectivity_e",
            "empty_gap_evidence_e",
            "slice_continuity_e",
            "local_point_geometry_e",
            "optional_trunk_evidence_e",
            "merge_support",
            "separate_support",
        ):
            _finite_float(getattr(self, name), name)
        for name in (
            "horizontal_distance_e",
            "height_difference_e",
            "saddle_drop_e",
            "xy_crown_connectivity_e",
            "empty_gap_evidence_e",
            "slice_continuity_e",
            "local_point_geometry_e",
            "optional_trunk_evidence_e",
        ):
            if not (-1.0 <= getattr(self, name) <= 1.0):
                raise ValueError(f"{name} must be in [-1, 1]")
        for name in ("merge_support", "separate_support"):
            if not (0.0 <= getattr(self, name) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        _exact_enum(self.relation, V4TreeRelation, "relation")
        _exact_str(self.stage, "stage")
        _flag_mask(self.quality_flags, V4SegmentationQualityFlag, "quality_flags")
        _flag_mask(self.reason_flags, V4SegmentationQualityFlag, "reason_flags")
        _sha(self.diagnostic_digest, "diagnostic_digest")
        if self.diagnostic_digest != self.computed_digest():
            raise ValueError("diagnostic_digest does not match canonical fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": "V4REL1",
            "relation_id": self.relation_id,
            "hypothesis_a_id": self.hypothesis_a_id,
            "hypothesis_b_id": self.hypothesis_b_id,
            "horizontal_distance_e": self.horizontal_distance_e,
            "height_difference_e": self.height_difference_e,
            "saddle_drop_e": self.saddle_drop_e,
            "xy_crown_connectivity_e": self.xy_crown_connectivity_e,
            "empty_gap_evidence_e": self.empty_gap_evidence_e,
            "slice_continuity_e": self.slice_continuity_e,
            "local_point_geometry_e": self.local_point_geometry_e,
            "optional_trunk_evidence_e": self.optional_trunk_evidence_e,
            "merge_support": self.merge_support,
            "separate_support": self.separate_support,
            "relation": self.relation.value,
            "stage": self.stage,
            "quality_flags": self.quality_flags,
            "reason_flags": self.reason_flags,
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.to_digest_dict())
        payload.pop("digest_domain")
        payload["diagnostic_digest"] = self.diagnostic_digest
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4TreeRelationEvidence:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["relation"] = _enum_from_value(values["relation"], V4TreeRelation, "relation")
        return cls(**values)


# ---------------------------------------------------------------------------
# V4TreeCrownSummary and V4TreeGeometrySummary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4TreeCrownSummary:
    """V4 crown geometry derived from assigned measured points and slices."""

    crown_area_m2: float
    equivalent_radius_m: float
    responsibility_area_m2: float
    boundary_vertex_count: int
    boundary_digest: str

    def __post_init__(self) -> None:
        for name in ("crown_area_m2", "equivalent_radius_m", "responsibility_area_m2"):
            _finite_float(getattr(self, name), name)
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        _non_negative_int(self.boundary_vertex_count, "boundary_vertex_count")
        _sha(self.boundary_digest, "boundary_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "crown_area_m2": self.crown_area_m2,
            "equivalent_radius_m": self.equivalent_radius_m,
            "responsibility_area_m2": self.responsibility_area_m2,
            "boundary_vertex_count": self.boundary_vertex_count,
            "boundary_digest": self.boundary_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4TreeCrownSummary:
        return cls(**_exact_fields(data, set(cls.__dataclass_fields__), cls.__name__))


@dataclass(frozen=True, slots=True)
class V4TreeGeometrySummary:
    """Measured tree geometry summary from assigned points and slices."""

    member_count: int
    hag_min_m: float
    hag_max_m: float
    robust_hag_m: float
    vertical_extent_m: float
    xy_centroid: tuple[float, float, float]
    xy_bounds_min: tuple[float, float, float]
    xy_bounds_max: tuple[float, float, float]
    crown_area_m2: float
    equivalent_radius_m: float
    responsibility_area_m2: float
    slice_count: int
    valid_slice_count: int
    lowest_supported_slice_id: str | None
    highest_supported_slice_id: str | None
    trunk_evidence_state: V4TrunkEvidenceAvailability
    accepted_trunk_count: int
    primary_trunk_representative_original_index: int | None
    multi_stem_flag: bool
    geometry_quality_flags: int

    def __post_init__(self) -> None:
        _positive_int(self.member_count, "member_count")
        for name in ("hag_min_m", "hag_max_m", "robust_hag_m", "vertical_extent_m"):
            _finite_float(getattr(self, name), name)
        if self.hag_max_m < self.hag_min_m:
            raise ValueError("hag_max_m must not be below hag_min_m")
        for name in ("xy_centroid", "xy_bounds_min", "xy_bounds_max"):
            object.__setattr__(self, name, _finite_vec3(getattr(self, name), name))
        for name in ("crown_area_m2", "equivalent_radius_m", "responsibility_area_m2"):
            _finite_float(getattr(self, name), name)
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        _non_negative_int(self.slice_count, "slice_count")
        _non_negative_int(self.valid_slice_count, "valid_slice_count")
        if self.valid_slice_count > self.slice_count:
            raise ValueError("valid_slice_count must not exceed slice_count")
        for name in ("lowest_supported_slice_id", "highest_supported_slice_id"):
            value = getattr(self, name)
            if value is not None:
                _exact_str(value, name)
        _exact_enum(
            self.trunk_evidence_state, V4TrunkEvidenceAvailability, "trunk_evidence_state"
        )
        _non_negative_int(self.accepted_trunk_count, "accepted_trunk_count")
        if self.primary_trunk_representative_original_index is not None:
            _non_negative_int(
                self.primary_trunk_representative_original_index,
                "primary_trunk_representative_original_index",
            )
        if type(self.multi_stem_flag) is not bool:
            raise ValueError("multi_stem_flag must be a built-in bool")
        _flag_mask(
            self.geometry_quality_flags, V4SegmentationQualityFlag, "geometry_quality_flags"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_count": self.member_count,
            "hag_min_m": self.hag_min_m,
            "hag_max_m": self.hag_max_m,
            "robust_hag_m": self.robust_hag_m,
            "vertical_extent_m": self.vertical_extent_m,
            "xy_centroid": list(self.xy_centroid),
            "xy_bounds_min": list(self.xy_bounds_min),
            "xy_bounds_max": list(self.xy_bounds_max),
            "crown_area_m2": self.crown_area_m2,
            "equivalent_radius_m": self.equivalent_radius_m,
            "responsibility_area_m2": self.responsibility_area_m2,
            "slice_count": self.slice_count,
            "valid_slice_count": self.valid_slice_count,
            "lowest_supported_slice_id": self.lowest_supported_slice_id,
            "highest_supported_slice_id": self.highest_supported_slice_id,
            "trunk_evidence_state": self.trunk_evidence_state.value,
            "accepted_trunk_count": self.accepted_trunk_count,
            "primary_trunk_representative_original_index": (
                self.primary_trunk_representative_original_index
            ),
            "multi_stem_flag": self.multi_stem_flag,
            "geometry_quality_flags": self.geometry_quality_flags,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4TreeGeometrySummary:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        for name in ("xy_centroid", "xy_bounds_min", "xy_bounds_max"):
            values[name] = _finite_vec3(values[name], name)
        values["trunk_evidence_state"] = _enum_from_value(
            values["trunk_evidence_state"],
            V4TrunkEvidenceAvailability,
            "trunk_evidence_state",
        )
        return cls(**values)


# ---------------------------------------------------------------------------
# V4TreeInstanceRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4TreeInstanceRecord:
    """One final confirmed tree record."""

    tree_id: int
    primary_apex_candidate_id: str
    secondary_apex_candidate_ids: tuple[str, ...]
    primary_trunk_candidate_id: str | None
    secondary_trunk_candidate_ids: tuple[str, ...]
    trunk_apex_association_ids: tuple[str, ...]
    member_count: int
    member_original_index_digest: str
    crown_summary: V4TreeCrownSummary
    geometry_summary: V4TreeGeometrySummary
    slice_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    confidence: float
    quality_flags: int
    reason_flags: int
    future_visual_evidence_reference: None

    def __post_init__(self) -> None:
        _non_negative_int(self.tree_id, "tree_id")
        _exact_str(self.primary_apex_candidate_id, "primary_apex_candidate_id")
        object.__setattr__(
            self,
            "secondary_apex_candidate_ids",
            _str_tuple(self.secondary_apex_candidate_ids, "secondary_apex_candidate_ids"),
        )
        if self.primary_trunk_candidate_id is not None:
            _exact_str(self.primary_trunk_candidate_id, "primary_trunk_candidate_id")
        object.__setattr__(
            self,
            "secondary_trunk_candidate_ids",
            _str_tuple(self.secondary_trunk_candidate_ids, "secondary_trunk_candidate_ids"),
        )
        object.__setattr__(
            self,
            "trunk_apex_association_ids",
            _str_tuple(self.trunk_apex_association_ids, "trunk_apex_association_ids"),
        )
        _positive_int(self.member_count, "member_count")
        _sha(self.member_original_index_digest, "member_original_index_digest")
        if type(self.crown_summary) is not V4TreeCrownSummary:
            raise ValueError("crown_summary must be a V4TreeCrownSummary")
        if type(self.geometry_summary) is not V4TreeGeometrySummary:
            raise ValueError("geometry_summary must be a V4TreeGeometrySummary")
        object.__setattr__(self, "slice_ids", _str_tuple(self.slice_ids, "slice_ids"))
        object.__setattr__(
            self, "relation_ids", _str_tuple(self.relation_ids, "relation_ids")
        )
        _finite_float(self.confidence, "confidence")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")
        _flag_mask(self.quality_flags, V4SegmentationQualityFlag, "quality_flags")
        _flag_mask(self.reason_flags, V4SegmentationQualityFlag, "reason_flags")
        if self.future_visual_evidence_reference is not None:
            raise ValueError("future_visual_evidence_reference must be null in V1")
        if self.member_count != self.geometry_summary.member_count:
            raise ValueError("member_count must match geometry_summary.member_count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "primary_apex_candidate_id": self.primary_apex_candidate_id,
            "secondary_apex_candidate_ids": list(self.secondary_apex_candidate_ids),
            "primary_trunk_candidate_id": self.primary_trunk_candidate_id,
            "secondary_trunk_candidate_ids": list(self.secondary_trunk_candidate_ids),
            "trunk_apex_association_ids": list(self.trunk_apex_association_ids),
            "member_count": self.member_count,
            "member_original_index_digest": self.member_original_index_digest,
            "crown_summary": self.crown_summary.to_dict(),
            "geometry_summary": self.geometry_summary.to_dict(),
            "slice_ids": list(self.slice_ids),
            "relation_ids": list(self.relation_ids),
            "confidence": self.confidence,
            "quality_flags": self.quality_flags,
            "reason_flags": self.reason_flags,
            "future_visual_evidence_reference": self.future_visual_evidence_reference,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4TreeInstanceRecord:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["secondary_apex_candidate_ids"] = _str_tuple(
            values["secondary_apex_candidate_ids"], "secondary_apex_candidate_ids"
        )
        values["secondary_trunk_candidate_ids"] = _str_tuple(
            values["secondary_trunk_candidate_ids"], "secondary_trunk_candidate_ids"
        )
        values["trunk_apex_association_ids"] = _str_tuple(
            values["trunk_apex_association_ids"], "trunk_apex_association_ids"
        )
        values["crown_summary"] = V4TreeCrownSummary.from_dict(
            _mapping(values["crown_summary"], "crown_summary")
        )
        values["geometry_summary"] = V4TreeGeometrySummary.from_dict(
            _mapping(values["geometry_summary"], "geometry_summary")
        )
        values["slice_ids"] = _str_tuple(values["slice_ids"], "slice_ids")
        values["relation_ids"] = _str_tuple(values["relation_ids"], "relation_ids")
        return cls(**values)


# ---------------------------------------------------------------------------
# V4TreeSegmentationAudit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4TreeSegmentationAudit:
    """Counts, phase/resource evidence, warnings and limitations."""

    schema_version: str
    total_points: int
    apex_candidate_count: int
    accepted_trunk_count: int
    weak_trunk_count: int
    suppressed_trunk_count: int
    unmatched_trunk_count: int
    strong_association_count: int
    review_association_count: int
    unsupported_association_count: int
    path_observed_count: int
    path_not_available_count: int
    profile_confirmed_count: int
    profile_weak_count: int
    profile_no_valley_count: int
    profile_not_available_count: int
    hypothesis_count: int
    slice_count: int
    relation_provisional_count: int
    relation_final_count: int
    tree_count: int
    phase_resource_audit: str
    determinism_evidence: str
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != V4_TREE_SEGMENTATION_SCHEMA_VERSION:
            raise ValueError("unsupported audit schema version")
        for name in (
            "total_points",
            "apex_candidate_count",
            "accepted_trunk_count",
            "weak_trunk_count",
            "suppressed_trunk_count",
            "unmatched_trunk_count",
            "strong_association_count",
            "review_association_count",
            "unsupported_association_count",
            "path_observed_count",
            "path_not_available_count",
            "profile_confirmed_count",
            "profile_weak_count",
            "profile_no_valley_count",
            "profile_not_available_count",
            "hypothesis_count",
            "slice_count",
            "relation_provisional_count",
            "relation_final_count",
            "tree_count",
        ):
            _non_negative_int(getattr(self, name), name)
        _exact_str(self.phase_resource_audit, "phase_resource_audit")
        _exact_str(self.determinism_evidence, "determinism_evidence")
        object.__setattr__(self, "warnings", _str_tuple(self.warnings, "warnings"))
        object.__setattr__(
            self, "limitations", _str_tuple(self.limitations, "limitations")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "total_points": self.total_points,
            "apex_candidate_count": self.apex_candidate_count,
            "accepted_trunk_count": self.accepted_trunk_count,
            "weak_trunk_count": self.weak_trunk_count,
            "suppressed_trunk_count": self.suppressed_trunk_count,
            "unmatched_trunk_count": self.unmatched_trunk_count,
            "strong_association_count": self.strong_association_count,
            "review_association_count": self.review_association_count,
            "unsupported_association_count": self.unsupported_association_count,
            "path_observed_count": self.path_observed_count,
            "path_not_available_count": self.path_not_available_count,
            "profile_confirmed_count": self.profile_confirmed_count,
            "profile_weak_count": self.profile_weak_count,
            "profile_no_valley_count": self.profile_no_valley_count,
            "profile_not_available_count": self.profile_not_available_count,
            "hypothesis_count": self.hypothesis_count,
            "slice_count": self.slice_count,
            "relation_provisional_count": self.relation_provisional_count,
            "relation_final_count": self.relation_final_count,
            "tree_count": self.tree_count,
            "phase_resource_audit": self.phase_resource_audit,
            "determinism_evidence": self.determinism_evidence,
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4TreeSegmentationAudit:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["warnings"] = _str_tuple(values["warnings"], "warnings")
        values["limitations"] = _str_tuple(values["limitations"], "limitations")
        return cls(**values)


# ---------------------------------------------------------------------------
# V4TreeSegmentationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4TreeSegmentationResult:
    """The final original-index-keyed V4 segmentation result.

    Dense arrays are canonical and read-only; ragged members are strictly
    increasing original indexes; every normal point has exactly one assignment
    state and at most one tree ID.
    """

    schema_version: str
    algorithm_version: str
    source_identity: SourceIdentity
    config: V4TreeSegmentationConfig
    handoff_digest: str
    validated_view_digest: str
    coverage_report_digest: str
    original_index: np.ndarray
    xyz_local_m: np.ndarray
    hag_m: np.ndarray
    assignment_state_by_point: np.ndarray
    tree_id_by_point: np.ndarray
    assignment_confidence_by_point: np.ndarray
    assignment_reason_flags_by_point: np.ndarray
    tree_member_offsets: np.ndarray
    ordered_tree_member_original_index: np.ndarray
    apex_records: tuple[V4ApexCandidateRecord, ...]
    trunk_records: tuple[V4TrunkCandidateRecord, ...]
    trunk_apex_association_records: tuple[V4TrunkApexAssociationEvidence, ...]
    structural_path_records: tuple[V4StructuralPathEvidence, ...]
    rotating_profile_records: tuple[V4RotatingProfileEvidence, ...]
    hypothesis_records: tuple[V4TreeHypothesisRecord, ...]
    slice_records: tuple[V4TreeSliceSummary, ...]
    relation_records: tuple[V4TreeRelationEvidence, ...]
    tree_records: tuple[V4TreeInstanceRecord, ...]
    crown_boundary_vertices: np.ndarray
    crown_boundary_offsets: np.ndarray
    audit: V4TreeSegmentationAudit
    result_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != V4_TREE_SEGMENTATION_SCHEMA_VERSION:
            raise ValueError("unsupported V4 tree segmentation schema version")
        if self.algorithm_version != V4_TREE_SEGMENTATION_ALGORITHM_VERSION:
            raise ValueError("unsupported V4 tree segmentation algorithm version")
        if type(self.source_identity) is not SourceIdentity:
            raise ValueError("source_identity must be a SourceIdentity")
        if type(self.config) is not V4TreeSegmentationConfig:
            raise ValueError("config must be a V4TreeSegmentationConfig")
        for name in ("handoff_digest", "validated_view_digest", "coverage_report_digest"):
            _sha(getattr(self, name), name)
        count = int(self.original_index.shape[0])
        _require_ndarray(
            self.original_index,
            "original_index",
            dtype=np.dtype("<i8"),
            shape=(count,),
            finite=False,
        )
        _require_ndarray(
            self.xyz_local_m,
            "xyz_local_m",
            dtype=np.dtype("<f8"),
            shape=(count, 3),
            finite=True,
        )
        _require_ndarray(
            self.hag_m, "hag_m", dtype=np.dtype("<f4"), shape=(count,), finite=True
        )
        _require_ndarray(
            self.assignment_state_by_point,
            "assignment_state_by_point",
            dtype=np.dtype("|u1"),
            shape=(count,),
            finite=False,
        )
        _require_ndarray(
            self.tree_id_by_point,
            "tree_id_by_point",
            dtype=np.dtype("<i4"),
            shape=(count,),
            finite=False,
        )
        _require_ndarray(
            self.assignment_confidence_by_point,
            "assignment_confidence_by_point",
            dtype=np.dtype("<f4"),
            shape=(count,),
            finite=True,
        )
        _require_ndarray(
            self.assignment_reason_flags_by_point,
            "assignment_reason_flags_by_point",
            dtype=np.dtype("<u4"),
            shape=(count,),
            finite=False,
        )
        _require_ndarray(
            self.tree_member_offsets,
            "tree_member_offsets",
            dtype=np.dtype("<i8"),
            shape=(len(self.tree_records) + 1,),
            finite=False,
        )
        member_count = int(self.ordered_tree_member_original_index.shape[0])
        _require_ndarray(
            self.ordered_tree_member_original_index,
            "ordered_tree_member_original_index",
            dtype=np.dtype("<i8"),
            shape=(member_count,),
            finite=False,
        )
        boundary_vertices = int(self.crown_boundary_vertices.shape[0])
        _require_ndarray(
            self.crown_boundary_vertices,
            "crown_boundary_vertices",
            dtype=np.dtype("<f8"),
            shape=(boundary_vertices, 2),
            finite=True,
        )
        _require_ndarray(
            self.crown_boundary_offsets,
            "crown_boundary_offsets",
            dtype=np.dtype("<i8"),
            shape=(len(self.tree_records) + 1,),
            finite=False,
        )
        for name, record_type in (
            ("apex_records", V4ApexCandidateRecord),
            ("trunk_records", V4TrunkCandidateRecord),
            ("trunk_apex_association_records", V4TrunkApexAssociationEvidence),
            ("structural_path_records", V4StructuralPathEvidence),
            ("rotating_profile_records", V4RotatingProfileEvidence),
            ("hypothesis_records", V4TreeHypothesisRecord),
            ("slice_records", V4TreeSliceSummary),
            ("relation_records", V4TreeRelationEvidence),
            ("tree_records", V4TreeInstanceRecord),
        ):
            object.__setattr__(
                self, name, _record_tuple(getattr(self, name), record_type, name)
            )
        if type(self.audit) is not V4TreeSegmentationAudit:
            raise ValueError("audit must be a V4TreeSegmentationAudit")
        _sha(self.result_digest, "result_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "source_identity": self.source_identity.to_dict(),
            "config": self.config.to_dict(),
            "handoff_digest": self.handoff_digest,
            "validated_view_digest": self.validated_view_digest,
            "coverage_report_digest": self.coverage_report_digest,
            "original_index": self.original_index,
            "xyz_local_m": self.xyz_local_m,
            "hag_m": self.hag_m,
            "assignment_state_by_point": self.assignment_state_by_point,
            "tree_id_by_point": self.tree_id_by_point,
            "assignment_confidence_by_point": self.assignment_confidence_by_point,
            "assignment_reason_flags_by_point": self.assignment_reason_flags_by_point,
            "tree_member_offsets": self.tree_member_offsets,
            "ordered_tree_member_original_index": self.ordered_tree_member_original_index,
            "apex_records": [record.to_dict() for record in self.apex_records],
            "trunk_records": [record.to_dict() for record in self.trunk_records],
            "trunk_apex_association_records": [
                record.to_dict() for record in self.trunk_apex_association_records
            ],
            "structural_path_records": [
                record.to_dict() for record in self.structural_path_records
            ],
            "rotating_profile_records": [
                record.to_dict() for record in self.rotating_profile_records
            ],
            "hypothesis_records": [
                record.to_dict() for record in self.hypothesis_records
            ],
            "slice_records": [record.to_dict() for record in self.slice_records],
            "relation_records": [
                record.to_dict() for record in self.relation_records
            ],
            "tree_records": [record.to_dict() for record in self.tree_records],
            "crown_boundary_vertices": self.crown_boundary_vertices,
            "crown_boundary_offsets": self.crown_boundary_offsets,
            "audit": self.audit.to_dict(),
            "result_digest": self.result_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4TreeSegmentationResult:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["source_identity"] = SourceIdentity.from_dict(
            _mapping(values["source_identity"], "source_identity")
        )
        values["config"] = V4TreeSegmentationConfig.from_dict(
            _mapping(values["config"], "config")
        )
        for name in (
            "original_index",
            "xyz_local_m",
            "hag_m",
            "assignment_state_by_point",
            "tree_id_by_point",
            "assignment_confidence_by_point",
            "assignment_reason_flags_by_point",
            "tree_member_offsets",
            "ordered_tree_member_original_index",
            "crown_boundary_vertices",
            "crown_boundary_offsets",
        ):
            array = np.asarray(values[name])
            array.setflags(write=False)
            values[name] = array
        values["apex_records"] = tuple(
            V4ApexCandidateRecord.from_dict(_mapping(item, "apex_record"))
            for item in values["apex_records"]
        )
        values["trunk_records"] = tuple(
            V4TrunkCandidateRecord.from_dict(_mapping(item, "trunk_record"))
            for item in values["trunk_records"]
        )
        values["trunk_apex_association_records"] = tuple(
            V4TrunkApexAssociationEvidence.from_dict(_mapping(item, "association_record"))
            for item in values["trunk_apex_association_records"]
        )
        values["structural_path_records"] = tuple(
            V4StructuralPathEvidence.from_dict(_mapping(item, "path_record"))
            for item in values["structural_path_records"]
        )
        values["rotating_profile_records"] = tuple(
            V4RotatingProfileEvidence.from_dict(_mapping(item, "profile_record"))
            for item in values["rotating_profile_records"]
        )
        values["hypothesis_records"] = tuple(
            V4TreeHypothesisRecord.from_dict(_mapping(item, "hypothesis_record"))
            for item in values["hypothesis_records"]
        )
        values["slice_records"] = tuple(
            V4TreeSliceSummary.from_dict(_mapping(item, "slice_record"))
            for item in values["slice_records"]
        )
        values["relation_records"] = tuple(
            V4TreeRelationEvidence.from_dict(_mapping(item, "relation_record"))
            for item in values["relation_records"]
        )
        values["tree_records"] = tuple(
            V4TreeInstanceRecord.from_dict(_mapping(item, "tree_record"))
            for item in values["tree_records"]
        )
        values["audit"] = V4TreeSegmentationAudit.from_dict(
            _mapping(values["audit"], "audit")
        )
        return cls(**values)

    def computed_result_digest(self) -> str:
        return frv_digest(
            {
                "digest_domain": V4_TREE_SEGMENTATION_RESULT_DIGEST_DOMAIN,
                "schema_version": self.schema_version,
                "algorithm_version": self.algorithm_version,
                "source_identity": self.source_identity.to_dict(),
                "config": self.config.to_dict(),
                "handoff_digest": self.handoff_digest,
                "validated_view_digest": self.validated_view_digest,
                "coverage_report_digest": self.coverage_report_digest,
                "original_index": list(np.asarray(self.original_index, dtype=np.int64)),
                "xyz_local_m": list(np.asarray(self.xyz_local_m, dtype=np.float64).tolist()),
                "hag_m": list(np.asarray(self.hag_m, dtype=np.float32).tolist()),
                "assignment_state_by_point": list(
                    np.asarray(self.assignment_state_by_point, dtype=np.uint8)
                ),
                "tree_id_by_point": list(np.asarray(self.tree_id_by_point, dtype=np.int32)),
                "assignment_confidence_by_point": list(
                    np.asarray(self.assignment_confidence_by_point, dtype=np.float32)
                ),
                "assignment_reason_flags_by_point": list(
                    np.asarray(self.assignment_reason_flags_by_point, dtype=np.uint32)
                ),
                "tree_member_offsets": list(np.asarray(self.tree_member_offsets, dtype=np.int64)),
                "ordered_tree_member_original_index": list(
                    np.asarray(self.ordered_tree_member_original_index, dtype=np.int64)
                ),
                "apex_records": [record.to_dict() for record in self.apex_records],
                "trunk_records": [record.to_dict() for record in self.trunk_records],
                "trunk_apex_association_records": [
                    record.to_dict() for record in self.trunk_apex_association_records
                ],
                "structural_path_records": [
                    record.to_dict() for record in self.structural_path_records
                ],
                "rotating_profile_records": [
                    record.to_dict() for record in self.rotating_profile_records
                ],
                "hypothesis_records": [
                    record.to_dict() for record in self.hypothesis_records
                ],
                "slice_records": [record.to_dict() for record in self.slice_records],
                "relation_records": [
                    record.to_dict() for record in self.relation_records
                ],
                "tree_records": [record.to_dict() for record in self.tree_records],
                "crown_boundary_vertices": list(
                    np.asarray(self.crown_boundary_vertices, dtype=np.float64).tolist()
                ),
                "crown_boundary_offsets": list(
                    np.asarray(self.crown_boundary_offsets, dtype=np.int64)
                ),
                "audit": self.audit.to_dict(),
            }
        )


# ---------------------------------------------------------------------------
# Conservation and digest validation
# ---------------------------------------------------------------------------


def v4_member_index_digest(indexes: Any) -> str:
    """Return the canonical V4MEM1 member original-index digest.

    The digest binds the exact ordered original-index sequence that the formal
    ragged member block (and every tree record member digest) must reproduce.
    """

    values = np.asarray(indexes, dtype=np.int64).ravel()
    hasher = hashlib.sha256()
    hasher.update(b"V4MEM1\0")
    hasher.update(struct.pack("<Q", int(values.shape[0])))
    for value in values:
        hasher.update(struct.pack("<Q", int(value)))
    return hasher.hexdigest()


def validate_v4_tree_segmentation_result(result: V4TreeSegmentationResult) -> None:
    """Run the full result conservation and semantic validator (fail closed).

    Artifact/checksum self-consistency alone is not enough: the validator
    re-derives every identity/digest/reference from the actual measured data so
    a semantically corrupt but rehashed artifact cannot pass.
    """

    count = int(result.original_index.shape[0])
    indexes = np.asarray(result.original_index, dtype=np.int64)
    if indexes.size == 0:
        if count != 0:
            raise ValueError("original_index must be empty when the population is empty")
    else:
        if indexes[0] < 0:
            raise ValueError("original_index values must be nonnegative")
        if not (np.diff(indexes) > 0).all():
            raise ValueError("original_index must be strictly increasing and unique")
        if indexes[-1] >= result.source_identity.source_point_count:
            raise ValueError("original_index values must be inside the source domain")
    states = np.asarray(result.assignment_state_by_point, dtype=np.uint8)
    known = {int(state.value) for state in V4TreeAssignmentState}
    if not bool(np.isin(states, np.fromiter(known, dtype=np.uint8)).all()):
        raise ValueError("assignment_state_by_point contains an unknown state")
    tree_ids = np.asarray(result.tree_id_by_point, dtype=np.int32)
    if not ((tree_ids == -1) | (tree_ids >= 0)).all():
        raise ValueError("tree_id_by_point must be -1 or nonnegative")
    if not ((states == int(V4TreeAssignmentState.ASSIGNED)) == (tree_ids >= 0)).all():
        raise ValueError("tree_id_by_point >= 0 must match ASSIGNED state exactly")
    if not ((states != int(V4TreeAssignmentState.ASSIGNED)) == (tree_ids == -1)).all():
        raise ValueError("non-assigned points must carry tree_id -1")
    confidences = np.asarray(result.assignment_confidence_by_point, dtype=np.float32)
    if np.any((confidences < 0.0) | (confidences > 1.0)):
        raise ValueError("assignment_confidence_by_point must be in [0, 1]")
    if not np.all(np.isfinite(confidences)):
        raise ValueError("assignment_confidence_by_point must be finite")

    assigned_count = int(np.count_nonzero(states == int(V4TreeAssignmentState.ASSIGNED)))
    review_count = int(np.count_nonzero(states == int(V4TreeAssignmentState.REVIEW_REQUIRED)))
    unassigned_count = int(
        np.count_nonzero(states == int(V4TreeAssignmentState.UNASSIGNED))
    )
    if assigned_count + review_count + unassigned_count != count:
        raise ValueError("assignment states must conserve the normal population")

    tree_count = len(result.tree_records)
    # Every ASSIGNED point must reference a valid, existing tree.
    assigned_tree_ids = tree_ids[states == int(V4TreeAssignmentState.ASSIGNED)]
    if tree_count == 0:
        if assigned_count != 0:
            raise ValueError("ASSIGNED points must not exist when tree_count is zero")
    else:
        if not ((assigned_tree_ids >= 0) & (assigned_tree_ids < tree_count)).all():
            raise ValueError("every ASSIGNED tree_id must be a valid tree index")

    offsets = np.asarray(result.tree_member_offsets, dtype=np.int64)
    if offsets.shape != (tree_count + 1,):
        raise ValueError("tree_member_offsets must have one entry per tree plus one")
    if offsets[0] != 0 or int(offsets[-1]) != assigned_count:
        raise ValueError("tree_member_offsets must span exactly the assigned count")
    if not (np.diff(offsets) > 0).all():
        raise ValueError("every formal tree must have at least one member")
    members = np.asarray(result.ordered_tree_member_original_index, dtype=np.int64)
    if int(members.shape[0]) != assigned_count:
        raise ValueError("ragged members must equal the assigned count")
    if members.size:
        # Each offset-delimited tree member block is strictly increasing;
        # across trees the member sets are unique.  The concatenation of all
        # tree blocks is NOT required to be globally increasing.
        for tree_id in range(tree_count):
            block = members[int(offsets[tree_id]) : int(offsets[tree_id + 1])]
            if not (np.diff(block) > 0).all():
                raise ValueError(
                    "each tree member block must be strictly increasing by original index"
                )
            expected_block = indexes[
                (states == int(V4TreeAssignmentState.ASSIGNED))
                & (tree_ids == tree_id)
            ]
            if not np.array_equal(block, expected_block):
                raise ValueError(
                    "each ragged tree member block must match its dense tree_id members"
                )
        if len(set(int(value) for value in members)) != int(members.shape[0]):
            raise ValueError("no original_index may occur in more than one tree")
        expected_set = set(
            int(index) for index in indexes[states == int(V4TreeAssignmentState.ASSIGNED)]
        )
        if set(int(value) for value in members) != expected_set:
            raise ValueError(
                "ragged member union must equal the dense ASSIGNED original-index set"
            )

    boundary_offsets = np.asarray(result.crown_boundary_offsets, dtype=np.int64)
    if boundary_offsets.shape != (tree_count + 1,):
        raise ValueError("crown_boundary_offsets must have one entry per tree plus one")
    if boundary_offsets[0] != 0 or int(boundary_offsets[-1]) != int(
        result.crown_boundary_vertices.shape[0]
    ):
        raise ValueError("crown_boundary_offsets must span the boundary vertices")

    # Identity inventories for cross-reference validation.
    apex_ids = {record.candidate_id for record in result.apex_records}
    trunk_ids = {record.candidate_id for record in result.trunk_records}
    association_ids = {
        record.association_id for record in result.trunk_apex_association_records
    }
    path_ids = {record.path_id for record in result.structural_path_records}
    profile_ids = {record.profile_id for record in result.rotating_profile_records}
    slice_ids = {record.slice_id for record in result.slice_records}
    relation_ids = {record.relation_id for record in result.relation_records}
    hypothesis_ids = {record.hypothesis_id for record in result.hypothesis_records}
    identity_inventories = (
        ("apex", apex_ids, len(result.apex_records)),
        ("trunk", trunk_ids, len(result.trunk_records)),
        ("association", association_ids, len(result.trunk_apex_association_records)),
        ("structural path", path_ids, len(result.structural_path_records)),
        ("rotating profile", profile_ids, len(result.rotating_profile_records)),
        ("slice", slice_ids, len(result.slice_records)),
        ("relation", relation_ids, len(result.relation_records)),
        ("hypothesis", hypothesis_ids, len(result.hypothesis_records)),
    )
    for label, identities, record_count in identity_inventories:
        if len(identities) != record_count:
            raise ValueError(f"{label} identities must be unique")

    # Hypothesis records: every ref must resolve.
    for record in result.hypothesis_records:
        if record.primary_apex_candidate_id not in apex_ids:
            raise ValueError("hypothesis primary apex reference is dangling")
        for apex_id in record.secondary_apex_candidate_ids:
            if apex_id not in apex_ids:
                raise ValueError("hypothesis secondary apex reference is dangling")
        for trunk_id in record.trunk_candidate_ids:
            if trunk_id not in trunk_ids:
                raise ValueError("hypothesis trunk reference is dangling")
        # Bundle A currently has no persisted fragment-record inventory.  A
        # non-empty reference therefore cannot be resolved and must fail
        # closed; accepting an arbitrary string here would admit a rehashed
        # dangling fragment reference.
        if record.fragment_refs:
            raise ValueError("hypothesis fragment reference is dangling")
        for profile_id in record.profile_refs:
            if profile_id not in profile_ids:
                raise ValueError("hypothesis profile reference is dangling")

    # Association records: trunk/apex/path refs must resolve.
    for record in result.trunk_apex_association_records:
        if record.trunk_candidate_id not in trunk_ids:
            raise ValueError("association trunk reference is dangling")
        if record.apex_candidate_id not in apex_ids:
            raise ValueError("association apex reference is dangling")
        if (
            record.structural_path_evidence_reference is not None
            and record.structural_path_evidence_reference not in path_ids
        ):
            raise ValueError("association structural-path reference is dangling")
        if record.structural_path_evidence_reference is not None:
            path_record = next(
                path
                for path in result.structural_path_records
                if path.path_id == record.structural_path_evidence_reference
            )
            if (
                path_record.trunk_candidate_id != record.trunk_candidate_id
                or path_record.apex_candidate_id != record.apex_candidate_id
            ):
                raise ValueError(
                    "association structural-path reference has incompatible endpoints"
                )

    # Structural path records: trunk/apex refs must resolve.
    for record in result.structural_path_records:
        if record.trunk_candidate_id not in trunk_ids:
            raise ValueError("structural path trunk reference is dangling")
        if record.apex_candidate_id not in apex_ids:
            raise ValueError("structural path apex reference is dangling")

    # Rotating-profile records: apex/hypothesis refs must resolve.
    for record in result.rotating_profile_records:
        if record.apex_candidate_id not in apex_ids:
            raise ValueError("rotating-profile apex reference is dangling")
        if record.hypothesis_id is not None and record.hypothesis_id not in hypothesis_ids:
            raise ValueError("rotating-profile hypothesis reference is dangling")
        if record.hypothesis_id is not None:
            hypothesis = next(
                item
                for item in result.hypothesis_records
                if item.hypothesis_id == record.hypothesis_id
            )
            if record.apex_candidate_id not in (
                hypothesis.primary_apex_candidate_id,
                *hypothesis.secondary_apex_candidate_ids,
            ):
                raise ValueError(
                    "rotating-profile hypothesis reference has an incompatible apex"
                )

    # Slice records: hypothesis ref must resolve.
    for record in result.slice_records:
        if record.hypothesis_id not in hypothesis_ids:
            raise ValueError("slice hypothesis reference is dangling")

    # Relation records: endpoint hypothesis refs must resolve.
    for record in result.relation_records:
        if record.hypothesis_a_id not in hypothesis_ids:
            raise ValueError("relation endpoint A reference is dangling")
        if record.hypothesis_b_id not in hypothesis_ids:
            raise ValueError("relation endpoint B reference is dangling")

    for tree_id, record in enumerate(result.tree_records):
        if record.tree_id != tree_id:
            raise ValueError("tree records must be in contiguous canonical tree_id order")
        start = int(offsets[tree_id])
        stop = int(offsets[tree_id + 1])
        if record.member_count != stop - start:
            raise ValueError("tree record member_count must match the member block")
        # Tree member digest must equal the actual ragged block members.
        block_members = members[start:stop]
        if record.member_original_index_digest != v4_member_index_digest(block_members):
            raise ValueError("tree record member digest does not match ragged members")
        # Tree identity references must resolve.
        if record.primary_apex_candidate_id not in apex_ids:
            raise ValueError("tree primary apex reference is dangling")
        for apex_id in record.secondary_apex_candidate_ids:
            if apex_id not in apex_ids:
                raise ValueError("tree secondary apex reference is dangling")
        if record.primary_trunk_candidate_id is not None:
            if record.primary_trunk_candidate_id not in trunk_ids:
                raise ValueError("tree primary trunk reference is dangling")
        for trunk_id in record.secondary_trunk_candidate_ids:
            if trunk_id not in trunk_ids:
                raise ValueError("tree secondary trunk reference is dangling")
        for association_id in record.trunk_apex_association_ids:
            if association_id not in association_ids:
                raise ValueError("tree trunk-apex association reference is dangling")
        for slice_id in record.slice_ids:
            if slice_id not in slice_ids:
                raise ValueError("tree slice reference is dangling")
        for relation_id in record.relation_ids:
            if relation_id not in relation_ids:
                raise ValueError("tree relation reference is dangling")
        # No object may reference a record from another incompatible identity.
        tree_apexes = [record.primary_apex_candidate_id, *record.secondary_apex_candidate_ids]
        if len(set(tree_apexes)) != len(tree_apexes):
            raise ValueError("tree apex references must be unique")
        try:
            tree_hypotheses = {
                _hypothesis_id_for_apex(apex_id, result.hypothesis_records)
                for apex_id in tree_apexes
            }
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        for slice_id in record.slice_ids:
            slice_record = _slice_by_id(result.slice_records, slice_id)
            if slice_record.hypothesis_id not in tree_hypotheses:
                raise ValueError("tree slice references an unrelated hypothesis")
        for relation_id in record.relation_ids:
            relation_record = _relation_by_id(result.relation_records, relation_id)
            if (
                relation_record.hypothesis_a_id not in tree_hypotheses
                or relation_record.hypothesis_b_id not in tree_hypotheses
            ):
                raise ValueError("tree relation references an unrelated hypothesis")
        geometry = record.geometry_summary
        for name in ("lowest_supported_slice_id", "highest_supported_slice_id"):
            slice_id = getattr(geometry, name)
            if slice_id is not None and slice_id not in slice_ids:
                raise ValueError(f"geometry {name} reference is dangling")
    accepted_trunk_ids = {
        record.candidate_id
        for record in result.trunk_records
        if record.candidate_state is V4TrunkCandidateState.ACCEPTED
    }
    matched_trunk_ids = {
        record.trunk_candidate_id for record in result.trunk_apex_association_records
    }
    expected_audit_counts = {
        "total_points": count,
        "apex_candidate_count": len(result.apex_records),
        "accepted_trunk_count": len(accepted_trunk_ids),
        "weak_trunk_count": sum(
            record.candidate_state is V4TrunkCandidateState.WEAK
            for record in result.trunk_records
        ),
        "suppressed_trunk_count": sum(
            record.candidate_state is V4TrunkCandidateState.SUPPRESSED
            for record in result.trunk_records
        ),
        "unmatched_trunk_count": len(accepted_trunk_ids - matched_trunk_ids),
        "strong_association_count": sum(
            record.association_score_available
            and record.association_state
            is V4TrunkApexAssociationState.STRONG_ASSOCIATION
            for record in result.trunk_apex_association_records
        ),
        "review_association_count": sum(
            record.association_state
            is V4TrunkApexAssociationState.REVIEW_ASSOCIATION
            for record in result.trunk_apex_association_records
        ),
        "unsupported_association_count": sum(
            record.association_state
            is V4TrunkApexAssociationState.UNSUPPORTED_ASSOCIATION
            for record in result.trunk_apex_association_records
        ),
        "path_observed_count": sum(
            record.availability is V4StructuralPathAvailability.OBSERVED
            for record in result.structural_path_records
        ),
        "path_not_available_count": sum(
            record.availability is V4StructuralPathAvailability.NOT_AVAILABLE
            for record in result.structural_path_records
        ),
        "profile_confirmed_count": sum(
            record.profile_state is V4RotatingProfileState.PROFILE_CONFIRMED_VALLEY
            for record in result.rotating_profile_records
        ),
        "profile_weak_count": sum(
            record.profile_state is V4RotatingProfileState.PROFILE_WEAK_VALLEY
            for record in result.rotating_profile_records
        ),
        "profile_no_valley_count": sum(
            record.profile_state is V4RotatingProfileState.PROFILE_NO_VALLEY
            for record in result.rotating_profile_records
        ),
        "profile_not_available_count": sum(
            record.profile_state is V4RotatingProfileState.PROFILE_NOT_AVAILABLE
            for record in result.rotating_profile_records
        ),
        "hypothesis_count": len(result.hypothesis_records),
        "slice_count": len(result.slice_records),
        "relation_provisional_count": sum(
            record.stage != "FINAL" for record in result.relation_records
        ),
        "relation_final_count": sum(
            record.stage == "FINAL" for record in result.relation_records
        ),
        "tree_count": tree_count,
    }
    for name, expected in expected_audit_counts.items():
        if getattr(result.audit, name) != expected:
            raise ValueError(f"audit.{name} must match the semantic record inventory")
    if result.result_digest != result.computed_result_digest():
        raise ValueError("result_digest does not match canonical result fields")


def _hypothesis_id_for_apex(
    apex_id: str, hypothesis_records: tuple[Any, ...]
) -> str:
    for record in hypothesis_records:
        if record.primary_apex_candidate_id == apex_id:
            return record.hypothesis_id
    raise ValueError(
        f"tree apex {apex_id} has no matching hypothesis record"
    )


def _slice_by_id(records: tuple[Any, ...], slice_id: str) -> Any:
    for record in records:
        if record.slice_id == slice_id:
            return record
    raise ValueError(f"slice reference {slice_id} is dangling")


def _relation_by_id(records: tuple[Any, ...], relation_id: str) -> Any:
    for record in records:
        if record.relation_id == relation_id:
            return record
    raise ValueError(f"relation reference {relation_id} is dangling")


def v4_tree_result_digest(result: V4TreeSegmentationResult) -> str:
    """Return the canonical semantic V4 result digest (self-computed)."""

    return result.computed_result_digest()


# ---------------------------------------------------------------------------
# Artifact contract types
# ---------------------------------------------------------------------------

V4_TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION = "v4-tree-segmentation-artifact-v1"
V4_TREE_SEGMENTATION_JSON_FILENAME = "v4_tree_segmentation.json"
V4_TREE_SEGMENTATION_NPZ_FILENAME = "v4_tree_segmentation.npz"
V4_TREE_SEGMENTATION_MANIFEST_FILENAME = "v4_tree_segmentation_manifest.json"

_RESULT_ARRAY_FIELDS = (
    "original_index",
    "xyz_local_m",
    "hag_m",
    "assignment_state_by_point",
    "tree_id_by_point",
    "assignment_confidence_by_point",
    "assignment_reason_flags_by_point",
    "tree_member_offsets",
    "ordered_tree_member_original_index",
    "crown_boundary_vertices",
    "crown_boundary_offsets",
)


def v4_tree_segmentation_result_json(result: V4TreeSegmentationResult) -> dict[str, Any]:
    """Return the JSON-safe result metadata (arrays excluded for the NPZ)."""

    payload = dict(result.to_dict())
    for name in _RESULT_ARRAY_FIELDS:
        payload.pop(name)
    payload["array_fields"] = list(_RESULT_ARRAY_FIELDS)
    return payload


@dataclass(frozen=True, slots=True)
class V4TreeSegmentationOutputReference:
    """Runtime access information for a published V4 result artifact."""

    root_path: Path
    metadata_json_path: Path
    npz_path: Path
    manifest_path: Path

    def __post_init__(self) -> None:
        for name in ("root_path", "metadata_json_path", "npz_path", "manifest_path"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a pathlib.Path instance")


@dataclass(frozen=True, slots=True)
class V4TreeSegmentationArtifactManifest:
    """Checksums, byte counts and array inventory for the published artifact."""

    artifact_schema_version: str
    result_digest: str
    metadata_sha256: str
    metadata_bytes: int
    npz_sha256: str
    npz_bytes: int
    array_inventory: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        if self.artifact_schema_version != V4_TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported V4 tree-segmentation artifact schema")
        _sha(self.result_digest, "result_digest")
        _sha(self.metadata_sha256, "metadata_sha256")
        _sha(self.npz_sha256, "npz_sha256")
        object.__setattr__(
            self,
            "metadata_bytes",
            _non_negative_int(self.metadata_bytes, "metadata_bytes"),
        )
        object.__setattr__(self, "npz_bytes", _non_negative_int(self.npz_bytes, "npz_bytes"))
        if not isinstance(self.array_inventory, Mapping):
            raise ValueError("array_inventory must be a mapping")
        expected = set(_RESULT_ARRAY_FIELDS)
        if set(self.array_inventory) != expected:
            raise ValueError("array_inventory must cover exactly the result arrays")
        for name, spec in self.array_inventory.items():
            if not isinstance(spec, Mapping):
                raise ValueError(f"array_inventory[{name}] must be a mapping")
            for key in ("dtype", "shape"):
                if key not in spec:
                    raise ValueError(f"array_inventory[{name}] is missing {key}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "result_digest": self.result_digest,
            "metadata_sha256": self.metadata_sha256,
            "metadata_bytes": self.metadata_bytes,
            "npz_sha256": self.npz_sha256,
            "npz_bytes": self.npz_bytes,
            "array_inventory": dict(self.array_inventory),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V4TreeSegmentationArtifactManifest:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        inventory = dict(values["array_inventory"])
        parsed: dict[str, dict[str, Any]] = {}
        for name, spec in inventory.items():
            if not isinstance(spec, Mapping):
                raise ValueError("array_inventory values must be mappings")
            parsed[name] = {
                "dtype": spec["dtype"],
                "shape": tuple(spec["shape"]),
            }
        values["array_inventory"] = parsed
        return cls(**values)


@dataclass(frozen=True, slots=True)
class LoadedV4TreeSegmentationArtifact:
    """A re-loaded, fully validated V4 result artifact."""

    result: V4TreeSegmentationResult
    output_reference: V4TreeSegmentationOutputReference
    manifest: V4TreeSegmentationArtifactManifest


# ---------------------------------------------------------------------------
# Public errors
# ---------------------------------------------------------------------------


class V4TreeSegmentationError(Exception):
    """Closed V4 segmentation failure with a stable code and field context."""

    def __init__(self, *, code: V4TreeSegmentationErrorCode, field_name: str) -> None:
        if type(code) is not V4TreeSegmentationErrorCode:
            raise TypeError("code must be a V4TreeSegmentationErrorCode")
        _exact_str(field_name, "field_name")
        self.code = code
        self.field_name = field_name
        super().__init__(f"{code.value}:{field_name}")


__all__ = [
    "V4_TREE_SEGMENTATION_SCHEMA_VERSION",
    "V4_TREE_SEGMENTATION_ALGORITHM_VERSION",
    "V4_TREE_SEGMENTATION_CONFIG_SCHEMA_VERSION",
    "V4_TREE_SEGMENTATION_RESULT_DIGEST_DOMAIN",
    "V4_TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION",
    "V4_TREE_SEGMENTATION_JSON_FILENAME",
    "V4_TREE_SEGMENTATION_NPZ_FILENAME",
    "V4_TREE_SEGMENTATION_MANIFEST_FILENAME",
    "v4_tree_segmentation_result_json",
    "V4TreeSegmentationOutputReference",
    "V4TreeSegmentationArtifactManifest",
    "LoadedV4TreeSegmentationArtifact",
    "LTR_XY_MIN_XY_FACTOR",
    "LTR_XY_MIN_LA_FACTOR",
    "LTR_Z_MIN_SLICE_FACTOR",
    "LTA_MIN_LA_FACTOR",
    "LTA_MIN_XY_FACTOR",
    "LPATH_XY_MIN_LTR_XY_FACTOR",
    "LPATH_XY_MIN_LA_FACTOR",
    "STRUCTURAL_PATH_DOWNWARD_WEIGHT",
    "STRUCTURAL_PATH_DENSITY_WEIGHT",
    "ROTATING_PROFILE_RAY_COUNT",
    "ROTATING_PROFILE_VALLEY_MIN",
    "ROTATING_PROFILE_DROP_MIN",
    "ROTATING_PROFILE_REBOUND_MIN",
    "ROTATING_PROFILE_DENSITY_RATIO",
    "TRUNK_BIN_MIN_POINTS",
    "TRUNK_COMPONENT_MIN_POINTS",
    "TRUNK_COMPONENT_MIN_BINS",
    "TRUNK_CORE_RADIUS_FACTOR",
    "PCA_REL_GAP_MIN",
    "PCA_ABS_EPS",
    "RESPONSIBILITY_MIN",
    "RESPONSIBILITY_REVIEW_MARGIN",
    "RELATION_MIN",
    "RELATION_REVIEW_MARGIN",
    "ASSIGNMENT_MIN",
    "ASSIGNMENT_REVIEW_MARGIN",
    "TRUNK_CANDIDATE_MIN",
    "TRUNK_WEAK_MIN",
    "TRUNK_APEX_STRONG_MIN",
    "TRUNK_APEX_REVIEW_MIN",
    "V4TreeAssignmentState",
    "V4TreeRelation",
    "V4ProvisionalRelation",
    "V4TrunkCandidateState",
    "V4TrunkApexAssociationState",
    "V4TrunkEvidenceAvailability",
    "V4AssociationLookupKind",
    "V4StructuralPathAvailability",
    "V4RotatingProfileState",
    "V4ProfileSampleState",
    "V4TreeHypothesisState",
    "V4ConfigStatus",
    "V4TreeSegmentationErrorCode",
    "V4ApexQualityFlag",
    "V4TrunkQualityFlag",
    "V4AssociationQualityFlag",
    "V4ProfileQualityFlag",
    "V4SegmentationQualityFlag",
    "V4AssignmentReasonFlag",
    "V4TreeSegmentationConfig",
    "V4ApexCandidateRecord",
    "V4TrunkCandidateRecord",
    "V4StructuralPathEvidence",
    "V4TrunkApexAssociationEvidence",
    "V4RotatingProfileRaySample",
    "V4RotatingProfileRayRecord",
    "V4RotatingProfileEvidence",
    "V4TreeHypothesisRecord",
    "V4TreeSliceSummary",
    "V4TreeRelationEvidence",
    "V4TreeCrownSummary",
    "V4TreeGeometrySummary",
    "V4TreeInstanceRecord",
    "V4TreeSegmentationAudit",
    "V4TreeSegmentationResult",
    "V4TreeSegmentationError",
    "validate_v4_tree_segmentation_result",
    "v4_tree_result_digest",
    "v4_member_index_digest",
]
