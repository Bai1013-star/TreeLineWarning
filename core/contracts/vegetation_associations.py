"""Portable, pair-only contracts for V2A review associations.

V2A is deliberately not an identity or merge contract: every published row is
only an explainable request for joint review of two existing V2 clusters.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import Any

import numpy as np

from core.contracts._json import json_safe_mapping, reject_unknown_fields, require_fields


class VegetationAssociationStatus(IntEnum):
    ABSTAINED = 0
    POSSIBLE_REVIEW_ASSOCIATION = 1
    STRONG_REVIEW_ASSOCIATION = 2
    BLOCKED_BY_CONFLICT = 3


class VegetationAssociationReasonFlag(IntFlag):
    NONE = 0
    INPUTS_VERIFIED = 1
    ONE_INVALID_SCALE = 2
    BOTH_INVALID_SCALES = 4
    CLAMPED_HIGH_HEAVY = 8
    EXTREME_SCALE_RATIO = 16
    ANISOTROPIC_ISOTROPIC_CONFLICT = 32
    LOW_MULTI_CLUSTER_AMBIGUITY = 64
    LINEAGE_UNAVAILABLE = 128
    DECISION_INPUT_UNAVAILABLE = 256
    DECISIVE_BLOCKER = 512
    POSSIBLE_REVIEW = 1024
    STRONG_REVIEW = 2048
    INVALID_SCALE_AABB_DIAGNOSTIC = 4096
    DIRECTIONAL_EVIDENCE_UNAVAILABLE = 8192


class VegetationAssociationProvenanceFlag(IntFlag):
    NONE = 0
    V2_VERIFIED = 1
    V1_VERIFIED = 2
    A2A_VERIFIED = 4
    A2B_VERIFIED = 8
    SCALE_A_VALID = 16
    SCALE_B_VALID = 32
    HAG_RELIABLE_A = 64
    HAG_RELIABLE_B = 128
    LOW_AMBIGUOUS = 256
    SPLIT_LINEAGE_UNAVAILABLE = 512
    DIRECTION_A_STABLE = 1024
    DIRECTION_B_STABLE = 2048


POSITIVE_EVIDENCE_NAMES = (
    "spatial_continuation",
    "density_aware_continuity",
    "directional_branch_crown_continuation",
    "local_scale_compatibility",
    "geometry_distribution_compatibility",
    "hag_continuity",
    "crown_profile_compatibility",
    "low_supported_boundary_evidence",
    "shared_preliminary_parent_context",
    "topology_split_provenance_context",
)
BLOCKING_EVIDENCE_NAMES = (
    "physical_anisotropic_gap",
    "strong_vertical_separation",
    "touching_but_distinct_crown",
    "planar_artificial_ambiguity",
    "pole_or_isolated_slender_ambiguity",
    "compact_layered_ambiguity",
    "mixed_vegetation_artificial_evidence",
    "incompatible_reliable_hag",
    "incompatible_scale_density",
    "explicit_contradictory_evidence",
)


def _finite(value: float, name: str, minimum: float | None = None) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _sha(value: str, name: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class VegetationAssociationConfig:
    source_id: str = "scene_simple"
    search_multiplier: float = 4.0
    minimum_search_radius_m: float = 0.06
    maximum_search_radius_m: float = 0.60
    per_cluster_candidate_cap: int = 12
    boundary_sample_cap: int = 32
    scale_ratio_width: float = 2.0
    density_ratio_width: float = 3.0
    directional_width: float = 0.55
    profile_width: float = 0.45
    spatial_possible_threshold: float = 0.42
    possible_confidence_threshold: float = 0.48
    strong_confidence_threshold: float = 0.78
    decisive_blocker_threshold: float = 0.72
    required_independent_positive_channels: int = 3
    low_support_requirement: int = 2
    record_memory: bool = True

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        for name in self.__dataclass_fields__:
            if name in {"source_id", "record_memory"}:
                continue
            value = getattr(self, name)
            if name in {
                "per_cluster_candidate_cap",
                "boundary_sample_cap",
                "required_independent_positive_channels",
                "low_support_requirement",
            }:
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            else:
                _finite(value, name, 0.0)
        if not isinstance(self.record_memory, bool):
            raise ValueError("record_memory must be bool")
        if not 0 < self.minimum_search_radius_m <= self.maximum_search_radius_m:
            raise ValueError("search radius bounds are invalid")
        if not 0 < self.possible_confidence_threshold < self.strong_confidence_threshold <= 1:
            raise ValueError("confidence thresholds are invalid")
        if not 0 < self.decisive_blocker_threshold <= 1:
            raise ValueError("decisive blocker threshold is invalid")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VegetationAssociationConfig:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class VegetationAssociationStatistics:
    input_cluster_count: int
    candidate_pair_count: int
    abstained_count: int
    possible_review_count: int
    strong_review_count: int
    blocked_count: int
    elapsed_seconds: float
    rss_start_bytes: int | None = None
    rss_peak_bytes: int | None = None
    rss_end_bytes: int | None = None
    phase_timings_seconds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "phase_timings_seconds":
                if not isinstance(value, Mapping) or any(
                    not isinstance(k, str)
                    or not isinstance(v, int | float)
                    or not math.isfinite(v)
                    or v < 0
                    for k, v in value.items()
                ):
                    raise ValueError("phase timings must be finite non-negative values")
                object.__setattr__(self, name, dict(value))
            elif name == "elapsed_seconds":
                _finite(value, name, 0.0)
            elif value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be non-negative")
        if self.candidate_pair_count != sum(
            (
                self.abstained_count,
                self.possible_review_count,
                self.strong_review_count,
                self.blocked_count,
            )
        ):
            raise ValueError("status counts must sum to candidate_pair_count")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VegetationAssociationStatistics:
        require_fields(
            data,
            set(cls.__dataclass_fields__)
            - {"rss_start_bytes", "rss_peak_bytes", "rss_end_bytes", "phase_timings_seconds"},
            cls.__name__,
        )
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


_ARRAY_SPEC = {
    "pair_cluster_ids": (np.int32, 2),
    "pair_status": (np.uint8, 1),
    "pair_confidence": (np.float32, 1),
    "pair_positive_evidence": (np.float32, 10),
    "pair_blocking_evidence": (np.float32, 10),
    "pair_reason_flags": (np.uint32, 1),
    "pair_distance_summary_m": (np.float32, 5),
    "pair_scale_density_summary": (np.float32, 4),
    "pair_hag_summary": (np.float32, 4),
    "pair_low_support_counts": (np.int32, 6),
    "pair_provenance_flags": (np.uint32, 1),
    "pair_decision_input_completeness": (np.float32, 1),
}


@dataclass(frozen=True)
class VegetationAssociationResult:
    schema_version: str
    algorithm_version: str
    source_id: str
    manifest_source_id: str
    origin_xyz: tuple[float, float, float]
    output_npz_filename: str
    output_npz_sha256: str
    output_npz_bytes: int
    config: VegetationAssociationConfig
    input_provenance: Mapping[str, Any]
    arrays: Mapping[str, Any]
    statistics: VegetationAssociationStatistics
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    created_at: str = ""

    def __post_init__(self) -> None:
        if (
            self.schema_version != "vegetation-associations-v1"
            or self.algorithm_version != "v2a-review-level-vegetation-association-v1"
        ):
            raise ValueError("unsupported V2A schema or algorithm")
        if not self.source_id or not self.manifest_source_id or not self.created_at:
            raise ValueError("source identity and creation time are required")
        if any(ch in self.output_npz_filename for ch in "/\\:"):
            raise ValueError("output NPZ must be a sibling filename")
        _sha(self.output_npz_sha256, "output_npz_sha256")
        if not isinstance(self.output_npz_bytes, int) or self.output_npz_bytes < 0:
            raise ValueError("output bytes must be non-negative")
        origin = tuple(float(v) for v in self.origin_xyz)
        if len(origin) != 3 or not all(math.isfinite(v) for v in origin):
            raise ValueError("origin must be finite XYZ")
        object.__setattr__(self, "origin_xyz", origin)
        if not isinstance(self.config, VegetationAssociationConfig) or not isinstance(
            self.statistics, VegetationAssociationStatistics
        ):
            raise ValueError("invalid V2A result contract")
        object.__setattr__(
            self, "input_provenance", json_safe_mapping(self.input_provenance, "input_provenance")
        )
        object.__setattr__(self, "arrays", json_safe_mapping(self.arrays, "arrays"))
        object.__setattr__(self, "warnings", tuple(map(str, self.warnings)))
        object.__setattr__(self, "limitations", tuple(map(str, self.limitations)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "source_id": self.source_id,
            "manifest_source_id": self.manifest_source_id,
            "origin_xyz": list(self.origin_xyz),
            "output_npz_filename": self.output_npz_filename,
            "output_npz_sha256": self.output_npz_sha256,
            "output_npz_bytes": self.output_npz_bytes,
            "config": self.config.to_dict(),
            "input_provenance": dict(self.input_provenance),
            "arrays": dict(self.arrays),
            "statistics": self.statistics.to_dict(),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VegetationAssociationResult:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values = dict(data)
        values["config"] = VegetationAssociationConfig.from_dict(values["config"])
        values["statistics"] = VegetationAssociationStatistics.from_dict(values["statistics"])
        return cls(**values)


def validate_association_arrays(data: Mapping[str, np.ndarray]) -> int:
    if set(data) != set(_ARRAY_SPEC):
        raise ValueError("V2A NPZ must contain exactly the pair-level arrays")
    count: int | None = None
    for name, (dtype, columns) in _ARRAY_SPEC.items():
        value = data[name]
        expected = (len(value),) if columns == 1 else (len(value), columns)
        if value.dtype != np.dtype(dtype) or value.shape != expected or value.dtype == object:
            raise ValueError(f"{name} dtype or shape is invalid")
        if count is None:
            count = len(value)
        if len(value) != count:
            raise ValueError("pair arrays have inconsistent length")
        if np.issubdtype(value.dtype, np.floating) and (
            not np.isfinite(value).all()
            or np.any((value < 0) | (value > 1))
            and name
            not in {"pair_distance_summary_m", "pair_scale_density_summary", "pair_hag_summary"}
        ):
            raise ValueError(f"{name} has invalid floating values")
    pairs = data["pair_cluster_ids"]
    if (
        np.any(pairs[:, 0] < 0)
        or np.any(pairs[:, 0] >= pairs[:, 1])
        or len(np.unique(pairs, axis=0)) != len(pairs)
        or (
            len(pairs) > 1
            and np.any(np.lexsort((pairs[:, 1], pairs[:, 0])) != np.arange(len(pairs)))
        )
    ):
        raise ValueError("pair rows must be sorted unique ordered IDs")
    if not np.isin(data["pair_status"], [int(v) for v in VegetationAssociationStatus]).all():
        raise ValueError("invalid V2A status")
    if np.any(data["pair_decision_input_completeness"] > 1):
        raise ValueError("decision input completeness must be bounded")
    return int(count or 0)
