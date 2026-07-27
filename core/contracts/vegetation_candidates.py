# ruff: noqa: E501
"""Contracts for V1 explainable woody-vegetation candidate selection.

V1 keeps every A1 algorithm point aligned.  Point arrays are persisted in NPZ;
the paired JSON manifest contains only portable metadata and summaries.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Any

import numpy as np

from core.contracts._json import json_safe_mapping, reject_unknown_fields, require_fields


class VegetationCandidateLevel(IntEnum):
    REJECTED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class VegetationHagState(IntEnum):
    RELIABLE = 0
    UNCERTAIN = 1
    INVALID_OR_UNAVAILABLE = 2
    STRONGLY_NEGATIVE = 3


class VegetationEvidenceFlag(IntFlag):
    NONE = 0
    HEIGHT_EVIDENCE = 1
    HEIGHT_FALLBACK_USED = 2
    VOLUMETRIC_EVIDENCE = 4
    DENSITY_SUPPORT = 8
    A4_NONCONFIRMED_TRACK_ASSOCIATION = 16


class VegetationReasonFlag(IntFlag):
    NONE = 0
    CONFIRMED_WIRE_HARD_EXCLUSION = 1
    STRUCTURAL_INVALID_HARD_EXCLUSION = 2
    RELIABLE_NEAR_GROUND_HARD_EXCLUSION = 4
    STRONGLY_NEGATIVE_RAW_HAG = 8
    REVIEW_REQUIRED_LOW_CONFIDENCE = 16
    MANUAL_EXCLUSION = 32


class VegetationQualityFlag(IntFlag):
    NONE = 0
    GEOMETRY_DEGRADED = 1
    HAG_UNCERTAIN = 2
    HAG_INVALID_OR_UNAVAILABLE = 4
    DENSITY_UNUSABLE = 8
    HEIGHT_FALLBACK_DEGENERATE = 16
    RGB_UNAVAILABLE_OR_UNUSABLE = 32
    INTENSITY_UNAVAILABLE_OR_UNUSABLE = 64


class VegetationManualProvenanceFlag(IntFlag):
    NONE = 0
    MANUAL_INCLUDE_MATCHED = 1
    MANUAL_EXCLUDE_MATCHED = 2
    MANUAL_CONFLICT = 4
    MANUAL_WIRE_BLOCKED = 8


class VegetationManualAction(IntEnum):
    NONE = 0
    INCLUDE = 1
    EXCLUDE = 2
    WIRE_BLOCKED = 3


def _shape(value: Any, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(item) for item in shape)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return (len(value),)
    raise ValueError(f"{name} must expose shape")


def _array(value: Any, shape: tuple[int, ...], dtype: np.dtype[Any], name: str) -> None:
    if _shape(value, name) != shape or getattr(value, "dtype", None) is None:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    if np.dtype(value.dtype) != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")


def _finite(value: float, name: str, minimum: float | None = None) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _identifier(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class VegetationManualSelectionConfig:
    include_original_index: tuple[int, ...] = ()
    exclude_original_index: tuple[int, ...] = ()
    include_boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = ()
    exclude_boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = ()

    def __post_init__(self) -> None:
        for name in ("include_original_index", "exclude_original_index"):
            values = tuple(int(value) for value in getattr(self, name))
            if any(value < 0 for value in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique non-negative indexes")
            object.__setattr__(self, name, values)
        for name in ("include_boxes", "exclude_boxes"):
            boxes = []
            for lower, upper in getattr(self, name):
                lower = tuple(float(value) for value in lower)
                upper = tuple(float(value) for value in upper)
                if (
                    len(lower) != 3
                    or len(upper) != 3
                    or not all(math.isfinite(value) for value in (*lower, *upper))
                    or any(a > b for a, b in zip(lower, upper, strict=True))
                ):
                    raise ValueError(f"{name} must contain finite inclusive XYZ boxes")
                boxes.append((lower, upper))
            object.__setattr__(self, name, tuple(boxes))

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (
                list(getattr(self, name))
                if "index" in name
                else [[list(a), list(b)] for a, b in getattr(self, name)]
            )
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VegetationManualSelectionConfig:
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class VegetationCandidateConfig:
    source_id: str = "scene_simple"
    near_ground_hag_m: float = 0.30
    height_full_hag_m: float = 3.00
    strong_negative_raw_hag_m: float = 0.30
    ground_support_distance_m: float = 1.50
    scattering_low: float = 0.02
    scattering_high: float = 0.20
    planarity_low: float = 0.35
    planarity_high: float = 0.70
    linearity_low: float = 0.75
    linearity_high: float = 0.95
    height_weight: float = 0.35
    volumetric_weight: float = 0.45
    density_weight: float = 0.20
    auxiliary_weight: float = 0.00
    planarity_penalty_weight: float = 0.25
    linearity_penalty_weight: float = 0.20
    maximum_penalty: float = 0.45
    low_score: float = 0.35
    medium_score: float = 0.55
    high_score: float = 0.75
    medium_confidence: float = 0.50
    high_confidence: float = 0.75
    geometry_confidence_weight: float = 0.35
    hag_confidence_weight: float = 0.35
    ground_support_confidence_weight: float = 0.15
    density_confidence_weight: float = 0.15
    record_memory: bool = True

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        for name in self.__dataclass_fields__:
            if name in {"source_id", "record_memory"}:
                continue
            value = _finite(getattr(self, name), name, 0.0)
            object.__setattr__(self, name, value)
        if not isinstance(self.record_memory, bool):
            raise ValueError("record_memory must be bool")
        if not (
            self.height_full_hag_m > self.near_ground_hag_m and self.ground_support_distance_m > 0.0
        ):
            raise ValueError("metre thresholds are unordered")
        for low, high, name in (
            (self.scattering_low, self.scattering_high, "scattering"),
            (self.planarity_low, self.planarity_high, "planarity"),
            (self.linearity_low, self.linearity_high, "linearity"),
        ):
            if not (0.0 <= low < high <= 1.0):
                raise ValueError(f"{name} normalization bounds are invalid")
        if not (0.0 < self.low_score < self.medium_score < self.high_score <= 1.0):
            raise ValueError("score thresholds are invalid")
        if not (0.0 < self.medium_confidence < self.high_confidence <= 1.0):
            raise ValueError("confidence thresholds are invalid")
        if (
            sum(
                (
                    self.height_weight,
                    self.volumetric_weight,
                    self.density_weight,
                    self.auxiliary_weight,
                )
            )
            <= 0.0
        ):
            raise ValueError("positive evidence weights must sum above zero")
        if self.maximum_penalty > 1.0:
            raise ValueError("maximum_penalty must not exceed one")
        if not math.isclose(
            sum(
                (
                    self.geometry_confidence_weight,
                    self.hag_confidence_weight,
                    self.ground_support_confidence_weight,
                    self.density_confidence_weight,
                )
            ),
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("confidence weights must sum to one")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VegetationCandidateConfig:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class VegetationCandidatePointSet:
    source_id: str
    manifest_source_id: str
    xyz_local: Any
    origin_xyz: Any
    original_index: Any
    sample_index: Any
    wire_id_by_point: Any
    hag_state: Any
    vegetation_rule_score: Any
    vegetation_confidence: Any
    automatic_vegetation_level: Any
    vegetation_level: Any
    vegetation_mask: Any
    vegetation_evidence_flags: Any
    vegetation_reason_flags: Any
    vegetation_quality_flags: Any
    manual_provenance_flags: Any
    manual_action: Any
    geometry_quality_flags: Any
    ground_quality_flags: Any

    def __post_init__(self) -> None:
        if not self.source_id or not self.manifest_source_id:
            raise ValueError("source identifiers must be non-empty")
        count, dims = _shape(self.xyz_local, "xyz_local")
        if dims != 3:
            raise ValueError("xyz_local must have shape (M, 3)")
        _array(self.xyz_local, (count, 3), np.dtype(np.float32), "xyz_local")
        _array(self.origin_xyz, (3,), np.dtype(np.float64), "origin_xyz")
        for name, dtype in (
            ("original_index", np.int64),
            ("sample_index", np.int64),
            ("wire_id_by_point", np.int32),
            ("hag_state", np.uint8),
            ("automatic_vegetation_level", np.uint8),
            ("vegetation_level", np.uint8),
            ("vegetation_mask", np.bool_),
            ("vegetation_evidence_flags", np.uint16),
            ("vegetation_reason_flags", np.uint16),
            ("vegetation_quality_flags", np.uint16),
            ("manual_provenance_flags", np.uint8),
            ("manual_action", np.uint8),
            ("geometry_quality_flags", np.uint16),
            ("ground_quality_flags", np.uint16),
        ):
            _array(getattr(self, name), (count,), np.dtype(dtype), name)
        for name in ("vegetation_rule_score", "vegetation_confidence"):
            _array(getattr(self, name), (count,), np.dtype(np.float32), name)
            if not np.isfinite(np.asarray(getattr(self, name))).all() or not np.all(
                (np.asarray(getattr(self, name)) >= 0) & (np.asarray(getattr(self, name)) <= 1)
            ):
                raise ValueError(f"{name} must be finite in [0, 1]")
        for name in ("original_index", "sample_index"):
            value = np.asarray(getattr(self, name))
            if np.any(value < 0) or len(np.unique(value)) != count:
                raise ValueError(f"{name} must be unique non-negative")
        if (
            not np.isfinite(np.asarray(self.xyz_local)).all()
            or not np.isfinite(np.asarray(self.origin_xyz)).all()
        ):
            raise ValueError("coordinates must be finite")
        levels = [int(value) for value in VegetationCandidateLevel]
        if (
            not np.isin(self.hag_state, [int(value) for value in VegetationHagState]).all()
            or not np.isin(self.automatic_vegetation_level, levels).all()
            or not np.isin(self.vegetation_level, levels).all()
            or not np.isin(
                self.manual_action, [int(value) for value in VegetationManualAction]
            ).all()
        ):
            raise ValueError("invalid V1 enum array value")
        if np.any(
            np.asarray(self.vegetation_mask)
            != (np.asarray(self.vegetation_level) != int(VegetationCandidateLevel.REJECTED))
        ):
            raise ValueError("vegetation_mask must equal final non-rejected level")

    @property
    def point_count(self) -> int:
        return _shape(self.xyz_local, "xyz_local")[0]

    def array_metadata(self) -> dict[str, Any]:
        values = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"source_id", "manifest_source_id"}
        }
        return {
            name: {
                "shape": list(_shape(value, name)),
                "dtype": str(value.dtype),
                "unit": "m"
                if name in {"xyz_local", "origin_xyz"}
                else ("bool" if name == "vegetation_mask" else "unitless"),
            }
            for name, value in values.items()
        }


@dataclass(frozen=True)
class VegetationCandidateStatistics:
    input_point_count: int
    candidate_count: int
    rejected_count: int
    low_count: int
    medium_count: int
    high_count: int
    confirmed_wire_exclusion_count: int
    reliable_near_ground_exclusion_count: int
    structural_invalid_count: int
    elapsed_seconds: float
    rss_start_bytes: int | None = None
    rss_peak_bytes: int | None = None
    rss_end_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name not in {
                "elapsed_seconds",
                "rss_start_bytes",
                "rss_peak_bytes",
                "rss_end_bytes",
            }:
                _identifier(getattr(self, name), name)
        for name in ("rss_start_bytes", "rss_peak_bytes", "rss_end_bytes"):
            if getattr(self, name) is not None:
                _identifier(getattr(self, name), name)
        _finite(self.elapsed_seconds, "elapsed_seconds", 0.0)
        if (
            self.input_point_count != self.rejected_count + self.candidate_count
            or self.candidate_count != self.low_count + self.medium_count + self.high_count
        ):
            raise ValueError("level counts are inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VegetationCandidateStatistics:
        required = set(cls.__dataclass_fields__) - {
            "rss_start_bytes",
            "rss_peak_bytes",
            "rss_end_bytes",
        }
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class VegetationCandidateResult:
    schema_version: str
    source_id: str
    manifest_source_id: str
    output_npz_filename: str
    output_npz_sha256: str
    output_npz_bytes: int
    config: VegetationCandidateConfig
    manual_selection: VegetationManualSelectionConfig
    input_provenance: Mapping[str, Any]
    statistics: VegetationCandidateStatistics
    arrays: Mapping[str, Any]
    counts: Mapping[str, Any]
    formulas: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    algorithm_version: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if (
            not self.schema_version
            or not self.source_id
            or not self.manifest_source_id
            or not self.output_npz_filename
        ):
            raise ValueError("schema, source IDs, and NPZ filename must be non-empty")
        if (
            "/" in self.output_npz_filename
            or "\\" in self.output_npz_filename
            or ":" in self.output_npz_filename
        ):
            raise ValueError("output_npz_filename must be a sibling filename")
        if len(self.output_npz_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.output_npz_sha256
        ):
            raise ValueError("output_npz_sha256 must be lowercase SHA-256")
        _identifier(self.output_npz_bytes, "output_npz_bytes")
        if (
            not isinstance(self.config, VegetationCandidateConfig)
            or not isinstance(self.manual_selection, VegetationManualSelectionConfig)
            or not isinstance(self.statistics, VegetationCandidateStatistics)
        ):
            raise ValueError("V1 result contains invalid contracts")
        for name in ("input_provenance", "arrays", "counts", "formulas"):
            object.__setattr__(self, name, json_safe_mapping(getattr(self, name), name))
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))
        if not self.algorithm_version or not self.created_at:
            raise ValueError("algorithm_version and created_at must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "manifest_source_id": self.manifest_source_id,
            "output_npz_filename": self.output_npz_filename,
            "output_npz_sha256": self.output_npz_sha256,
            "output_npz_bytes": self.output_npz_bytes,
            "config": self.config.to_dict(),
            "manual_selection": self.manual_selection.to_dict(),
            "input_provenance": dict(self.input_provenance),
            "statistics": self.statistics.to_dict(),
            "arrays": dict(self.arrays),
            "counts": dict(self.counts),
            "formulas": dict(self.formulas),
            "warnings": list(self.warnings),
            "algorithm_version": self.algorithm_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VegetationCandidateResult:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values = dict(data)
        values["config"] = VegetationCandidateConfig.from_dict(values["config"])
        values["manual_selection"] = VegetationManualSelectionConfig.from_dict(
            values["manual_selection"]
        )
        values["statistics"] = VegetationCandidateStatistics.from_dict(values["statistics"])
        return cls(**values)
