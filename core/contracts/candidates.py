"""Contracts for A3 explainable wire-candidate point selection.

Coordinates are local metres.  Per-point arrays remain in NPZ artifacts; the
JSON result contains only metadata, provenance, configuration, and summaries.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import Any

import numpy as np

from core.contracts._json import json_safe_mapping, reject_unknown_fields, require_fields


class CandidateReasonFlag(IntFlag):
    """Stable uint16 evidence bits for each A3 point."""

    NONE = 0
    HIGH_LINEARITY = 1
    LOW_SCATTERING = 2
    ACCEPTABLE_PLANARITY = 4
    SUFFICIENT_HEIGHT = 8
    ACCEPTABLE_DENSITY = 16
    MANUAL_INCLUDE = 32
    MANUAL_EXCLUDE = 64
    HEIGHT_FALLBACK_USED = 128


class WireCandidateQualityFlag(IntFlag):
    """Stable uint16 A3 quality bits; these are distinct from A2 flags."""

    NONE = 0
    GEOMETRY_QUALITY_DEGRADED = 1
    GROUND_QUALITY_DEGRADED = 2
    HAG_UNAVAILABLE = 4
    BELOW_GROUND_INPUT = 8
    EXTREME_DENSITY = 16
    AMBIGUOUS_PRINCIPAL_DIRECTION = 32
    LOW_CONFIDENCE = 64
    INVALID_INPUT_FEATURE = 128


class CandidateLevel(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class CandidateSource(IntEnum):
    NONE = 0
    RULE = 1
    MANUAL = 2
    HYBRID = 3


def _array_shape(value: Any, field_name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(item) for item in shape)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return (len(value),)
    raise ValueError(f"{field_name} must expose a shape")


def _require_array(value: Any, shape: tuple[int, ...], dtype: np.dtype[Any], name: str) -> None:
    if _array_shape(value, name) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    actual = getattr(value, "dtype", None)
    if actual is None or np.dtype(actual) != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")


def _non_negative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _finite_unit_interval(value: float, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return value


def _positive_float(value: float, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _shape_dtype(value: Any, unit: str) -> dict[str, Any]:
    return {"shape": list(_array_shape(value, "array")), "dtype": str(value.dtype), "unit": unit}


@dataclass(frozen=True)
class ManualSelectionConfig:
    """Explicit original-index and local-coordinate manual candidate selections."""

    include_original_index: tuple[int, ...] = ()
    exclude_original_index: tuple[int, ...] = ()
    include_boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = ()
    manual_include_score: float = 1.0
    allow_numeric_invalid_override: bool = False

    def __post_init__(self) -> None:
        for name in ("include_original_index", "exclude_original_index"):
            values = tuple(int(value) for value in getattr(self, name))
            if any(value < 0 for value in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique non-negative indexes")
            object.__setattr__(self, name, values)
        boxes = []
        for lower, upper in self.include_boxes:
            lower = tuple(float(value) for value in lower)
            upper = tuple(float(value) for value in upper)
            if (
                len(lower) != 3
                or len(upper) != 3
                or not all(math.isfinite(value) for value in (*lower, *upper))
                or any(low > high for low, high in zip(lower, upper, strict=True))
            ):
                raise ValueError("include_boxes must contain finite inclusive XYZ boxes")
            boxes.append((lower, upper))
        object.__setattr__(self, "include_boxes", tuple(boxes))
        object.__setattr__(
            self,
            "manual_include_score",
            _finite_unit_interval(self.manual_include_score, "manual_include_score"),
        )
        if not isinstance(self.allow_numeric_invalid_override, bool):
            raise ValueError("allow_numeric_invalid_override must be bool")

    @property
    def has_include(self) -> bool:
        return bool(self.include_original_index or self.include_boxes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "include_original_index": list(self.include_original_index),
            "exclude_original_index": list(self.exclude_original_index),
            "include_boxes": [[list(lower), list(upper)] for lower, upper in self.include_boxes],
            "manual_include_score": self.manual_include_score,
            "allow_numeric_invalid_override": self.allow_numeric_invalid_override,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ManualSelectionConfig:
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class WireCandidateConfig:
    """Fixed, explainable A3 rule-scoring thresholds and penalty weights."""

    source_id: str = "scene_simple"
    linearity_guard: float = 0.60
    scattering_guard: float = 0.10
    linearity_full: float = 0.90
    scattering_good: float = 0.01
    planarity_good: float = 0.20
    planarity_bad: float = 0.60
    height_low_m: float = 1.0
    height_full_m: float = 6.0
    density_zero_low: float = 50.0
    density_full_low: float = 150.0
    density_full_high: float = 1500.0
    density_zero_high: float = 3000.0
    weight_linearity: float = 0.35
    weight_scattering: float = 0.25
    weight_planarity: float = 0.10
    weight_height: float = 0.20
    weight_density: float = 0.10
    penalty_geometry: float = 0.15
    penalty_ground_gradient: float = 0.10
    penalty_hag_unavailable: float = 0.15
    penalty_below_ground: float = 0.15
    penalty_extreme_hag: float = 0.15
    penalty_principal_ambiguity: float = 0.05
    penalty_extreme_density: float = 0.05
    penalty_cap: float = 0.40
    high_gradient_hag_weight: float = 0.35
    rule_min_score: float = 0.45
    medium_score: float = 0.65
    high_score: float = 0.80
    evidence_linearity: float = 0.80
    evidence_scattering: float = 0.05
    evidence_planarity: float = 0.40
    evidence_height_score: float = 0.50
    evidence_density_score: float = 0.50
    record_memory: bool = True

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("WireCandidateConfig.source_id must be non-empty")
        for name in self.__dataclass_fields__:
            if name not in {"source_id", "record_memory"}:
                object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        if not isinstance(self.record_memory, bool):
            raise ValueError("record_memory must be bool")
        if not (
            self.linearity_guard < self.linearity_full <= 1.0
            and 0.0 < self.scattering_good < self.scattering_guard <= 1.0
            and 0.0 < self.planarity_good < self.planarity_bad <= 1.0
            and self.height_low_m < self.height_full_m
            and 0.0
            < self.density_zero_low
            < self.density_full_low
            <= self.density_full_high
            < self.density_zero_high
            and 0.0 < self.rule_min_score < self.medium_score < self.high_score <= 1.0
            and 0.0 < self.penalty_cap <= 1.0
            and 0.0 < self.high_gradient_hag_weight < 1.0
        ):
            raise ValueError("WireCandidateConfig threshold ordering is invalid")
        if (
            sum(
                (
                    self.weight_linearity,
                    self.weight_scattering,
                    self.weight_planarity,
                    self.weight_height,
                    self.weight_density,
                )
            )
            <= 0.0
        ):
            raise ValueError("candidate score weights must sum above zero")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WireCandidateConfig:
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class CandidateEvidenceSet:
    """Float32 component scores and the hard geometry guard, aligned to A1."""

    linearity_score: Any
    scattering_score: Any
    planarity_score: Any
    height_score: Any
    density_score: Any
    quality_penalty: Any
    geometry_guard_mask: Any

    def __post_init__(self) -> None:
        point_count = _array_shape(self.linearity_score, "linearity_score")[0]
        for name in (
            "linearity_score",
            "scattering_score",
            "planarity_score",
            "height_score",
            "density_score",
            "quality_penalty",
        ):
            value = getattr(self, name)
            _require_array(value, (point_count,), np.dtype(np.float32), name)
            if not np.isfinite(np.asarray(value)).all() or not np.all(
                (np.asarray(value) >= 0.0) & (np.asarray(value) <= 1.0)
            ):
                raise ValueError(f"{name} must be finite in [0, 1]")
        _require_array(
            self.geometry_guard_mask,
            (point_count,),
            np.dtype(np.bool_),
            "geometry_guard_mask",
        )

    @property
    def point_count(self) -> int:
        return _array_shape(self.linearity_score, "linearity_score")[0]

    def array_metadata(self) -> dict[str, Any]:
        values = {
            "linearity_score": self.linearity_score,
            "scattering_score": self.scattering_score,
            "planarity_score": self.planarity_score,
            "height_score": self.height_score,
            "density_score": self.density_score,
            "quality_penalty": self.quality_penalty,
            "geometry_guard_mask": self.geometry_guard_mask,
        }
        return {
            name: _shape_dtype(value, "bool" if name.endswith("mask") else "unitless")
            for name, value in values.items()
        }


@dataclass(frozen=True)
class CandidatePointSet:
    """All A3 point-wise outputs, preserving exact A1/A2 input ordering."""

    source_id: str
    manifest_source_id: str
    xyz_local: Any
    origin_xyz: Any
    original_index: Any
    sample_index: Any
    principal_direction: Any
    rule_score: Any
    candidate_score: Any
    candidate_mask: Any
    candidate_level: Any
    candidate_source: Any
    evidence_flags: Any
    quality_flags: Any
    geometry_quality_flags: Any
    ground_quality_flags: Any

    def __post_init__(self) -> None:
        if not self.source_id or not self.manifest_source_id:
            raise ValueError("CandidatePointSet source identifiers must be non-empty")
        point_count, dimensions = _array_shape(self.xyz_local, "xyz_local")
        if dimensions != 3:
            raise ValueError("xyz_local must have shape (M, 3)")
        _require_array(self.xyz_local, (point_count, 3), np.dtype(np.float32), "xyz_local")
        _require_array(self.origin_xyz, (3,), np.dtype(np.float64), "origin_xyz")
        _require_array(
            self.principal_direction,
            (point_count, 3),
            np.dtype(np.float32),
            "principal_direction",
        )
        for name in ("original_index", "sample_index"):
            value = getattr(self, name)
            _require_array(value, (point_count,), np.dtype(np.int64), name)
            if np.any(np.asarray(value) < 0) or len(np.unique(value)) != point_count:
                raise ValueError(f"{name} must be unique non-negative values")
        for name in ("rule_score", "candidate_score"):
            value = getattr(self, name)
            _require_array(value, (point_count,), np.dtype(np.float32), name)
            if not np.isfinite(np.asarray(value)).all() or not np.all(
                (np.asarray(value) >= 0.0) & (np.asarray(value) <= 1.0)
            ):
                raise ValueError(f"{name} must be finite in [0, 1]")
        _require_array(self.candidate_mask, (point_count,), np.dtype(np.bool_), "candidate_mask")
        for name in ("candidate_level", "candidate_source"):
            _require_array(getattr(self, name), (point_count,), np.dtype(np.uint8), name)
        for name in (
            "evidence_flags",
            "quality_flags",
            "geometry_quality_flags",
            "ground_quality_flags",
        ):
            _require_array(getattr(self, name), (point_count,), np.dtype(np.uint16), name)
        if (
            not np.isfinite(np.asarray(self.xyz_local)).all()
            or not np.isfinite(np.asarray(self.origin_xyz)).all()
            or not np.isfinite(np.asarray(self.principal_direction)).all()
        ):
            raise ValueError("candidate coordinates and directions must be finite")
        level = np.asarray(self.candidate_level)
        source = np.asarray(self.candidate_source)
        mask = np.asarray(self.candidate_mask)
        if not np.isin(level, [int(member) for member in CandidateLevel]).all():
            raise ValueError("candidate_level contains an invalid enum value")
        if not np.isin(source, [int(member) for member in CandidateSource]).all():
            raise ValueError("candidate_source contains an invalid enum value")
        if np.any(mask != (level != int(CandidateLevel.NONE))):
            raise ValueError("candidate_mask and candidate_level must agree")
        if np.any(mask != (source != int(CandidateSource.NONE))):
            raise ValueError("candidate_mask and candidate_source must agree")

    @property
    def point_count(self) -> int:
        return _array_shape(self.xyz_local, "xyz_local")[0]

    def array_metadata(self) -> dict[str, Any]:
        values = {
            "xyz_local": self.xyz_local,
            "origin_xyz": self.origin_xyz,
            "original_index": self.original_index,
            "sample_index": self.sample_index,
            "principal_direction": self.principal_direction,
            "rule_score": self.rule_score,
            "candidate_score": self.candidate_score,
            "candidate_mask": self.candidate_mask,
            "candidate_level": self.candidate_level,
            "candidate_source": self.candidate_source,
            "evidence_flags": self.evidence_flags,
            "quality_flags": self.quality_flags,
            "geometry_quality_flags": self.geometry_quality_flags,
            "ground_quality_flags": self.ground_quality_flags,
        }
        units = {
            "xyz_local": "m",
            "origin_xyz": "m",
            "original_index": "index",
            "sample_index": "index",
            "principal_direction": "unitless",
            "rule_score": "unitless",
            "candidate_score": "unitless",
            "candidate_mask": "bool",
            "candidate_level": "enum",
            "candidate_source": "enum",
            "evidence_flags": "bitmask",
            "quality_flags": "bitmask",
            "geometry_quality_flags": "bitmask",
            "ground_quality_flags": "bitmask",
        }
        return {name: _shape_dtype(value, units[name]) for name, value in values.items()}


@dataclass(frozen=True)
class WireCandidateStatistics:
    """A3 counts, height handling diagnostics, runtime, and process RSS."""

    input_point_count: int
    candidate_count: int
    low_count: int
    medium_count: int
    high_count: int
    rule_count: int
    manual_count: int
    hybrid_count: int
    hag_normal_count: int
    hag_mixed_count: int
    hag_fallback_count: int
    elapsed_seconds: float
    rss_start_bytes: int | None = None
    rss_peak_bytes: int | None = None
    rss_end_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name == "elapsed_seconds":
                continue
            value = getattr(self, name)
            if value is not None:
                _non_negative_int(value, name)
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if self.candidate_count != self.low_count + self.medium_count + self.high_count:
            raise ValueError("candidate level counts must equal candidate_count")
        if self.candidate_count != self.rule_count + self.manual_count + self.hybrid_count:
            raise ValueError("candidate source counts must equal candidate_count")
        if self.input_point_count != (
            self.hag_normal_count + self.hag_mixed_count + self.hag_fallback_count
        ):
            raise ValueError("HAG handling counts must equal input_point_count")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WireCandidateStatistics:
        required = {
            name
            for name in cls.__dataclass_fields__
            if name not in {"rss_start_bytes", "rss_peak_bytes", "rss_end_bytes"}
        }
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**dict(data))


@dataclass(frozen=True)
class WireCandidateResult:
    """JSON-safe A3 provenance and metadata for the candidate NPZ artifact."""

    source_id: str
    manifest_source_id: str
    input_paths: Mapping[str, Any]
    input_sha256: Mapping[str, Any]
    output_npz_path: Path
    output_json_path: Path
    output_npz_sha256: str
    output_npz_bytes: int
    origin_xyz: tuple[float, float, float]
    config: WireCandidateConfig
    manual_selection: ManualSelectionConfig
    mode: str
    statistics: WireCandidateStatistics
    arrays: Mapping[str, Any]
    evidence_counts: Mapping[str, Any]
    quality_counts: Mapping[str, Any]
    formulas: Mapping[str, Any]
    warnings: tuple[str, ...]
    algorithm_version: str
    created_at: str

    def __post_init__(self) -> None:
        if not self.source_id or not self.manifest_source_id:
            raise ValueError("WireCandidateResult source identifiers must be non-empty")
        if self.mode not in {"rule", "union", "intersection", "manual_override"}:
            raise ValueError("WireCandidateResult.mode is invalid")
        for name in ("output_npz_path", "output_json_path"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if len(self.output_npz_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.output_npz_sha256
        ):
            raise ValueError("output_npz_sha256 must be a SHA-256 lowercase hex digest")
        _non_negative_int(self.output_npz_bytes, "output_npz_bytes")
        origin = tuple(float(value) for value in self.origin_xyz)
        if len(origin) != 3 or not all(math.isfinite(value) for value in origin):
            raise ValueError("origin_xyz must contain three finite coordinates")
        object.__setattr__(self, "origin_xyz", origin)
        if not isinstance(self.config, WireCandidateConfig):
            raise ValueError("config must be WireCandidateConfig")
        if not isinstance(self.manual_selection, ManualSelectionConfig):
            raise ValueError("manual_selection must be ManualSelectionConfig")
        if not isinstance(self.statistics, WireCandidateStatistics):
            raise ValueError("statistics must be WireCandidateStatistics")
        if not self.algorithm_version or not self.created_at:
            raise ValueError("algorithm_version and created_at must be non-empty")
        object.__setattr__(
            self,
            "input_paths",
            json_safe_mapping(
                {name: str(path) for name, path in self.input_paths.items()}, "input_paths"
            ),
        )
        object.__setattr__(
            self, "input_sha256", json_safe_mapping(self.input_sha256, "input_sha256")
        )
        object.__setattr__(self, "arrays", json_safe_mapping(self.arrays, "arrays"))
        object.__setattr__(
            self, "evidence_counts", json_safe_mapping(self.evidence_counts, "evidence_counts")
        )
        object.__setattr__(
            self, "quality_counts", json_safe_mapping(self.quality_counts, "quality_counts")
        )
        object.__setattr__(self, "formulas", json_safe_mapping(self.formulas, "formulas"))
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "manifest_source_id": self.manifest_source_id,
            "input_paths": {name: str(path) for name, path in self.input_paths.items()},
            "input_sha256": dict(self.input_sha256),
            "output_npz_path": str(self.output_npz_path),
            "output_json_path": str(self.output_json_path),
            "output_npz_sha256": self.output_npz_sha256,
            "output_npz_bytes": self.output_npz_bytes,
            "origin_xyz": list(self.origin_xyz),
            "config": self.config.to_dict(),
            "manual_selection": self.manual_selection.to_dict(),
            "mode": self.mode,
            "statistics": self.statistics.to_dict(),
            "arrays": dict(self.arrays),
            "evidence_counts": dict(self.evidence_counts),
            "quality_counts": dict(self.quality_counts),
            "formulas": dict(self.formulas),
            "warnings": list(self.warnings),
            "algorithm_version": self.algorithm_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WireCandidateResult:
        required = set(cls.__dataclass_fields__)
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, required, cls.__name__)
        return cls(
            source_id=data["source_id"],
            manifest_source_id=data["manifest_source_id"],
            input_paths=data["input_paths"],
            input_sha256=data["input_sha256"],
            output_npz_path=Path(data["output_npz_path"]),
            output_json_path=Path(data["output_json_path"]),
            output_npz_sha256=data["output_npz_sha256"],
            output_npz_bytes=data["output_npz_bytes"],
            origin_xyz=tuple(data["origin_xyz"]),
            config=WireCandidateConfig.from_dict(data["config"]),
            manual_selection=ManualSelectionConfig.from_dict(data["manual_selection"]),
            mode=data["mode"],
            statistics=WireCandidateStatistics.from_dict(data["statistics"]),
            arrays=data["arrays"],
            evidence_counts=data["evidence_counts"],
            quality_counts=data["quality_counts"],
            formulas=data["formulas"],
            warnings=tuple(data["warnings"]),
            algorithm_version=data["algorithm_version"],
            created_at=data["created_at"],
        )
